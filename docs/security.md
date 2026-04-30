# Security Model

OpenLegion is designed assuming agents will be compromised. Defense-in-depth across the layers below prevents a compromised agent from accessing credentials, escaping isolation, exhausting paid services, or affecting other agents.

## Security Layers

| Layer | Mechanism | What It Prevents |
|-------|-----------|-----------------|
| 1. Runtime isolation | Docker containers or Sandbox microVMs | Agent escape, kernel exploits |
| 2. Container hardening | Non-root user, no-new-privileges, memory/CPU limits, cap_drop=ALL | Privilege escalation, resource abuse |
| 3. Network egress controls | iptables egress filter inside the browser container; SSRF early-reject in mesh and `http_tool` | Outbound access to private/internal networks |
| 4. Credential separation | Vault holds keys, agents call via proxy; URL/body redaction on the way back | Key leakage, unauthorized API use |
| 5. Permission enforcement | Per-agent ACLs for messaging, blackboard, pub/sub, APIs, **and per-action browser gating** | Unauthorized data access, scope creep |
| 6. Input validation | Path traversal prevention, safe condition eval, token budgets, iteration limits | Injection, runaway loops |
| 7. Unicode sanitization | Invisible character stripping at multiple choke points | Prompt injection via invisible Unicode |
| 8. Solver spend controls | CAPTCHA kill switch, provider circuit breaker, per-agent + per-tenant monthly cost caps | Runaway third-party charges, billing DoS |
| 9. Operator-only surfaces | Stricter auth tier (`_require_operator_or_internal`) on metrics, cookie import, fingerprint reset | Cross-tenant data access, agent-side privilege escalation |

## Runtime Isolation

### Agent Containers (Default)

Agents run as non-root (UID 1000) with:
- `no-new-privileges` security option
- 384MB memory limit (agents are slim — no browser)
- 0.15 CPU quota (agents are I/O-bound, waiting on LLM APIs)
- PID limit: 256 processes (`pids_limit: 256`)
- `cap_drop: ALL` (no capabilities re-added)
- Read-only root filesystem (`read_only: True`)
- Tmpfs at `/tmp` (100MB, noexec, nosuid)
- No host filesystem access (only `/data` volume)
- Regular Docker bridge network — agents have internet egress. SSRF protection at the application layer in `src/agent/builtins/http_tool.py` blocks private/CGNAT/IPv4-mapped/6to4/Teredo ranges, pins DNS, allows max 5 redirects with re-validation at each hop, and strips `Authorization` on cross-origin redirects.

```bash
openlegion start  # Default container isolation
```

### Browser Service Container

Browser operations run in a separate, longer-lived container with a different posture (writable `/home/browser` for Firefox state) and plan-tier-scaled resources: 2–8GB RAM, 0.5–2.0 CPU, 512MB–2GB SHM, 1–10 concurrent browsers (see Architecture). Capability set: `cap_drop=["ALL"]` plus `cap_add=["NET_ADMIN","SETUID","SETGID"]` — the minimum needed to install the egress filter and drop privileges via `gosu`.

The container's authoritative SSRF control is an **iptables egress filter** installed by `docker/browser-entrypoint.sh` before the browser process starts. The entrypoint runs as root, uses `NET_ADMIN` to REJECT outbound traffic to RFC1918, loopback, link-local, CGNAT, and IANA-reserved IPv4 ranges plus IPv6 equivalents, then drops to UID 1000 via `tini -- gosu browser:browser python -m src.browser`. The long-running Firefox/FastAPI process holds no effective capabilities (`no-new-privileges` blocks re-acquisition).

The mesh-side `_resolve_and_pin()` check on `navigate`/`open_tab` (`src/host/server.py`) is a friendly early-reject only — the iptables filter is the boundary that must hold.

Egress operator knobs:
- `BROWSER_EGRESS_ALLOWLIST=cidr,...` — punch holes for specific destinations.
- `BROWSER_EGRESS_DISABLE=1` — disable the filter entirely (operator opt-out, not recommended).
- `OPENLEGION_BROWSER_ALLOW_HOST_NETWORK=1|true|yes` — required to launch the browser service in host-network mode. Without this flag, the runtime raises `RuntimeError` at boot rather than silently regressing the SSRF control (host-network shares the host's iptables namespace, so the filter cannot install).
- Private-IP proxies (`BROWSER_PROXY_URL` whose host is a literal RFC1918 / loopback / link-local / reserved IP) hard-fail at startup unless covered by `BROWSER_EGRESS_ALLOWLIST`. Hostname-based proxies are unaffected.

### Docker Sandbox MicroVMs

Each agent gets its own Linux kernel via hypervisor isolation:
- Apple Virtualization.framework (macOS) or Hyper-V (Windows)
- Full kernel boundary between agents
- Communication only via `docker sandbox exec` transport
- Even code execution inside the agent cannot see other agents or the host

```bash
openlegion start --sandbox  # MicroVM isolation (Docker Desktop 4.58+)
```

## Credential Vault

Agents **never** hold API keys. The credential vault (`src/host/credentials.py`) works as follows:

1. Credentials are loaded from environment variables on the host using two prefixes:
   - `OPENLEGION_SYSTEM_*` — system tier (LLM provider keys, never agent-accessible)
   - `OPENLEGION_CRED_*` — agent tier (tool/service keys, access controlled by `allowed_credentials`)
2. Agents make API calls by POSTing to `/mesh/api` on the mesh host
3. The vault injects the appropriate credentials server-side
4. The response is relayed back to the agent
5. Budget limits are enforced before dispatching, token usage recorded after

### Two-Tier Credential Scoping

Credentials are split into two tiers to prevent agents from accessing LLM provider keys:

| Tier | Examples | Who Can Access |
|------|----------|---------------|
| **System** | `anthropic_api_key`, `openai_api_key`, `gemini_api_base` | Mesh proxy only (internal). Agents can **never** resolve these. |
| **Agent** | `brave_search_api_key`, `myservice_password`, user-created credentials | Only agents in the `allowed_credentials` allowlist |

System credentials are identified by matching known provider names (`anthropic`, `openai`, `gemini`, `deepseek`, `moonshot`, `minimax`, `xai`, `groq`, `zai`) with key suffixes (`_api_key`, `_api_base`). Everything else is an agent credential.

Per-agent access is controlled by `allowed_credentials` glob patterns in `config/permissions.json`:

- `["*"]` -- grants access to all agent-tier credentials
- `["brave_search_*", "myapp_*"]` -- access only matching names
- `[]` -- no vault access (Pydantic default — deny all unless explicitly configured)

Even with `allowed_credentials: ["*"]`, system credentials are **always** blocked. Agents also cannot store or overwrite system credential names via `vault_store`.

> **Footnote — CAPTCHA solver credentials bypass the vault.** Four keys live as env vars only and are stripped from `config/settings.json` at load time with a one-time warning (`flags._ENV_ONLY_FLAGS`): `CAPTCHA_SOLVER_KEY`, `CAPTCHA_SOLVER_KEY_SECONDARY`, `CAPTCHA_SOLVER_PROXY_LOGIN`, `CAPTCHA_SOLVER_PROXY_PASSWORD`. The dashboard writes them via `os.environ[...]` directly. They are *not* visible in the credentials UI and do *not* flow through `OPENLEGION_CRED_*` ACLs. Agents never see them — solver provider calls happen entirely inside the browser service.

### Credential Redaction

`src/shared/redaction.py` is the single source of truth for stripping secrets from any text that might reach LLM context, logs, or agent-visible responses. It exposes three layers:

1. **Pattern-based string redaction (`SECRET_PATTERNS`)** — 9 compiled regexes covering known secret shapes:
   - OpenAI / Anthropic short form (`sk-…`)
   - Anthropic full form (`sk-ant-api…`)
   - GitHub OAuth access tokens (`gho_…`)
   - GitHub fine-grained PATs (`github_pat_…`)
   - Slack bot / user tokens (`xoxb-…` / `xoxp-…`)
   - AWS access key IDs (`AKIA…`)
   - Generic hex blobs (≥40 chars, with boundary lookarounds)
   - Generic base64 blobs (≥40 chars, with boundary lookarounds)
   - JWTs anchored on the JOSE header prefix `eyJ` (kills the false-positive class on benign three-dot version strings)

2. **URL-component-aware redaction (`redact_url`)** — drops userinfo and fragments, replaces JWT-shaped path segments with `[REDACTED]`, and strips values for sensitive query-parameter names (keeping the keys for debuggability). Sensitive parameter set covers generic auth (`api_key`, `token`, `secret`, `password`, `signature`, `clientKey` for 2Captcha / CapSolver, …), OAuth flow params (`code`, `state`), AWS SigV4 (`x-amz-*`), GCS signed URLs (`x-goog-*`), Azure SAS (`sv`, `sig`, `st`, `se`, `sp`, `sr`, `spr`), and magic-link tokens. Operators can opt specific param names out via `OPENLEGION_REDACTION_URL_QUERY_ALLOW=name1,name2,...`.

3. **Recursive (`deep_redact`)** — walks dicts / lists / tuples, runs strings through `redact_string`, and routes URL-shaped strings through `redact_url` first.

Call sites:
- **HTTP responses** — `http_request` strips resolved `$CRED{name}` values from response headers and body before returning results to the agent.
- **Browser snapshots / element queries** — `browser_get_elements` and `browser_navigate` deep-redact accessibility tree text and resolved `$CRED{name}` values from form fills.
- **Captured network metadata** — `inspect_requests` returns URLs only (no bodies / headers) and the URLs flow through `redact_url`.
- **Solver logs** — CAPTCHA solver request URLs are redacted before logging (the `clientKey` query param is in the sensitive set).

### Adding New Service Integrations

New external services are added as vault handlers, not as agent-side code:

```python
# In src/host/credentials.py
# 1. Add provider detection in _detect_provider()
# 2. Add credential injection in _inject_credentials()
# 3. The agent calls it like any other API through the mesh proxy
```

## Permission Matrix

Every inter-agent operation checks per-agent ACLs defined in `config/permissions.json`:

```json
{
  "researcher": {
    "can_message": [],
    "can_publish": ["research_complete"],
    "can_subscribe": ["new_lead"],
    "blackboard_read": ["projects/sales/*"],
    "blackboard_write": ["projects/sales/*"],
    "allowed_apis": ["llm", "brave_search"],
    "allowed_credentials": ["brave_search_*"],
    "can_use_browser": true,
    "browser_actions": ["navigate", "snapshot", "find_text", "screenshot"]
  }
}
```

- **Project-scoped blackboard** -- agents can only access keys under their project's namespace (`projects/{name}/*`). The `MeshClient` auto-prefixes all blackboard keys with the project namespace, so agents use natural keys while isolation is enforced transparently. Standalone agents get empty blackboard permissions.
- **Glob patterns** for blackboard paths and credential access
- **Explicit allowlists** for messaging, pub/sub, API access, and credential access
- **Default deny** -- if not listed, it's blocked
- Enforced at the mesh host before every operation

### Per-action Browser Gating

Agents with `can_use_browser=true` can be further restricted to a subset of browser actions via `browser_actions: list[str] | None`. The mesh validates the requested action name against `KNOWN_BROWSER_ACTIONS` (in `src/host/permissions.py`, currently 26 names: `navigate`, `snapshot`, `click`, `type`, `hover`, `screenshot`, `reset`, `focus`, `status`, `detect_captcha`, `scroll`, `wait_for`, `press_key`, `go_back`, `go_forward`, `switch_tab`, `upload_file`, `download`, `find_text`, `open_tab`, `fill_form`, `click_xy`, `inspect_requests`, `solve_captcha`, `request_captcha_help`, `request_browser_login`) and rejects unknowns with HTTP 400. Permission is then checked via `PermissionMatrix.can_browser_action`.

Three states for `browser_actions`:

| Value | Meaning |
|-------|---------|
| `None` (default; field omitted) | All current and future actions. Default-allow UX — turning the browser on grants full surface. |
| `["*"]` | All actions (explicit form). |
| Specific list (e.g. `["navigate", "snapshot"]`) | Only the listed actions; everything else denied. |
| `[]` | No actions. Equivalent to `can_use_browser=false`. |

The asymmetry vs. `allowed_credentials` (where `[]` is the safe deny-all default) is intentional: browser permissions default-allow because turning the browser on without granting actions is rarely what an operator wants. Use `[]` only when you mean "deny all browser actions for this agent". The default-permission inheritance (when an agent has no own entry but a `default` key exists in the permissions file) propagates `browser_actions` along with the other fields.

### Reserved Agent IDs

`RESERVED_AGENT_IDS = {"mesh", "operator", "canary-probe"}` (`src/shared/types.py`). Agent creation rejects these names; `canary-probe` is reserved for the stealth-canary subsystem so a real agent cannot collide with its profile. The CLI also explicitly rejects the literal `operator` from project membership (`src/cli/config.py`) — operator is a system trust zone, not a project member.

## Input Validation

### SSRF Protection

Two distinct controls — different traffic, different posture:

**Agent HTTP traffic** (`src/agent/builtins/http_tool.py`, application-layer):
- Resolves hostnames and rejects private, loopback, link-local, and reserved IP ranges
- Checks both initial URLs and redirect targets (via httpx event hook)
- IPv4-mapped IPv6 addresses (e.g., `::ffff:127.0.0.1`) are also blocked
- CGNAT range (100.64.0.0/10, RFC 6598), 6to4 (`2002::/16`), and Teredo (`2001::/32`) are blocked
- Max 5 redirects with re-validation at each hop; `Authorization` stripped on cross-origin redirect
- Prevents agents from using the HTTP tool to scan internal networks or access host services

**Browser-initiated traffic** is filtered by the **iptables egress rules in the browser service container** (see Browser Service Container above). That filter is the authoritative SSRF boundary for everything the browser does — page loads, embedded subresources, `inspect_requests` activity, downloads. The mesh-side `_resolve_and_pin()` check on `navigate` and `open_tab` is a friendly early-reject only; it returns HTTP 400 with a clear error before the browser ever opens a connection, but it is not the security boundary.

**Proxy configuration:**
- The dashboard accepts only HTTP / HTTPS proxies for system or per-agent egress; SOCKS4 / SOCKS5 are rejected with HTTP 400.
- The browser service rejects a proxy whose host is a literal RFC1918 / loopback / link-local / reserved IP at startup unless `BROWSER_EGRESS_ALLOWLIST` covers it. Hostname-based proxies are unaffected.
- Independently, CAPTCHA solver provider tasks (e.g. proxied reCAPTCHA) accept SOCKS4 / SOCKS5 via `CAPTCHA_SOLVER_PROXY_TYPE` because the provider performs the request server-side, outside the browser container — that proxy never touches engine-internal networks.

### Path Traversal Prevention

Agent file tools (`src/agent/builtins/file_tool.py`) validate all paths through four stages:
1. **Stage 0 — Absolute path rejection**: strips the `/data/` prefix from the candidate path; any remaining absolute path is rejected outright.
2. **Stage 1 — Pre-resolution `..` check**: walks every path component and rejects any `..` segment *before* filesystem resolution — catches traversal attempts that rely on resolution order.
3. **Stage 2 — Symlink-safe walk**: resolves each path component individually using `lstat()` to detect symlinks at every step, preventing symlink chains that point outside `/data`.
4. **Stage 3 — Final `is_relative_to()` check**: confirms the fully resolved path is still under `/data`.
All file operations are scoped to the container's `/data` volume.

### Skill Self-Authoring

Agents can write and register new tools (`skill_tool`). All submitted code is validated through AST analysis before execution:
- Forbidden imports (23 modules including `os`, `subprocess`, `socket`, `importlib`, etc.)
- Forbidden calls (16 functions including `eval`, `exec`, `open`, `compile`, etc.)
- Forbidden attribute accesses (11 attributes including `__dict__`, `__subclasses__`, `__globals__`, etc.)
- Skills are capped at 10,000 characters.

### Bounded Execution

- Task mode: 20 iterations maximum (`AgentLoop.MAX_ITERATIONS`)
- Chat mode: 30 tool rounds maximum per turn (`CHAT_MAX_TOOL_ROUNDS`), auto-compaction every 200 rounds (`CHAT_MAX_TOTAL_ROUNDS`) with session continuation (up to 5 auto-continues)
- Per-agent token budgets enforced at the vault layer
- CAPTCHA solver activity is bounded by a per-agent rate limit (default 20/hr), per-agent and per-tenant monthly cost caps, and pacing jitter (3000–12000 ms between solves) — see CAPTCHA Solver Controls below
- Prevents runaway loops and unbounded spend

### Rate Limiting

Per-agent rate limits on mesh endpoints prevent abuse and resource exhaustion:

| Endpoint | Limit | Window |
|----------|-------|--------|
| `api_proxy` | 30 requests | 60 seconds |
| `vault_resolve` | 5 requests | 60 seconds |
| `vault_store` | 10 requests | 3600 seconds |
| `blackboard_read` | 200 requests | 60 seconds |
| `blackboard_write` | 100 requests | 60 seconds |
| `publish` | 200 requests | 60 seconds |
| `notify` | 10 requests | 60 seconds |
| `cron_create` | 10 requests | 3600 seconds |
| `spawn` | 5 requests | 3600 seconds |
| `wallet_read` | 120 requests | 60 seconds |
| `wallet_transfer` | 10 requests | 3600 seconds |
| `wallet_execute` | 10 requests | 3600 seconds |
| `image_gen` | 10 requests | 60 seconds |
| `agent_profile` | 30 requests | 60 seconds |
| `upload_stage` | 30 requests | 60 seconds |
| `upload_apply` | 30 requests | 60 seconds |
| `ext_credentials` | 30 requests | 60 seconds |
| `ext_status` | 60 requests | 60 seconds |

`_RATE_LIMITS` currently has 18 entries: 16 declared statically in `src/host/server.py` plus `ext_credentials` and `ext_status` registered when external-API support initializes. All other endpoints default to 100 requests per 60 seconds.

Exceeding a rate limit returns HTTP 429. Rate-limit buckets are automatically cleaned up when agents are deregistered.

## Unicode Sanitization (Prompt Injection Defense)

Agents process untrusted text from user messages, web pages, HTTP responses, tool outputs, blackboard data, and MCP servers. Attackers can embed invisible instructions using tag characters (U+E0001-E007F), RTL overrides (U+202A-202E), zero-width spaces, variation selectors, and other invisible codepoints that LLM tokenizers decode while being invisible to humans.

`sanitize_for_prompt()` in `src/shared/utils.py` is called wherever untrusted text crosses into LLM context. Key choke points:

| Choke Point | File | What It Covers |
|-------------|------|----------------|
| User input | `src/agent/server.py` | All user messages from all channels/CLI |
| Tool results | `src/agent/loop.py` | All tool outputs (browser, web search, HTTP, file, run_command, memory, MCP) |
| System prompt context | `src/agent/loop.py` | Workspace bootstrap, blackboard goals, memory facts, learnings, tool history |

### What Gets Stripped

- **Dangerous categories** (Cc, Cf, Co, Cs, Cn) except TAB/LF/CR, ZWNJ/ZWJ, VS15/VS16
- **Data smuggling vectors**: VS1-14, VS17-256, Combining Grapheme Joiner, Hangul fillers, Object Replacement
- **Normalization**: U+2028/U+2029 (line/paragraph separator) to `\n`

### What Is Preserved

Normal text in all scripts (Arabic, Hebrew, CJK, Devanagari, etc.), emoji with ZWJ sequences, ZWNJ for Persian/Arabic, tabs, newlines, and VS15/VS16 for emoji presentation.

### Adding New Paths to LLM Context

If you add a new path where untrusted text reaches LLM context (new tool, new system prompt section, new message source), wrap it with `sanitize_for_prompt()`. See `tests/test_sanitize.py` for the full test suite.

## Webhooks

Named webhook endpoints support inbound HTTP payloads from external services. Security controls:

- **Body size limit** — 1 MB (1,048,576 bytes). Enforced in two stages: a Content-Length pre-check rejects oversized requests immediately, followed by an authoritative check on the fully read body.
- **Optional HMAC-SHA256 signature verification** — If a webhook is configured with a secret, every request must include an `X-Webhook-Signature` header. The signature is verified with `hmac.compare_digest` (constant-time comparison) against `HMAC-SHA256(secret, body)`. Requests with invalid or missing signatures are rejected with HTTP 401.

## System Introspection

The `/mesh/introspect` endpoint lets agents query their own runtime state (permissions, budget, fleet, cron, health). Security controls:

- **Auth enforced** — requires valid `MESH_AUTH_TOKEN` like all mesh endpoints
- **No sensitive data** — returns permission patterns, budget numbers, and fleet roster; never credentials, host paths, or container config
- **Fleet filtering** — agents only see teammates they have `can_message` permission for, plus themselves
- **Cron scoping** — agents only see their own scheduled jobs
- **Input sanitization** — all introspect data (agent IDs, roles, cron schedules) is sanitized via `sanitize_for_prompt()` before reaching LLM context, with agent IDs truncated to 60 chars and roles to 80 chars

The introspect data flows into agents through three layers, each with its own sanitization:
1. `SYSTEM.md` — generated at startup, refreshed on cache miss (5-min TTL)
2. Runtime Context block — injected into the system prompt each turn
3. `get_system_status` tool — on-demand fresh data

## Mesh Authentication

Each agent receives a unique auth token at startup (`MESH_AUTH_TOKEN`). All requests from agents to the mesh include this token for verification. This prevents:
- Spoofed agent requests
- Container-to-container communication bypassing the mesh
- Unauthorized access to mesh endpoints

### Auth Tiers

The mesh distinguishes three caller tiers via header / token shape:

| Tier | How identified | Used by |
|------|----------------|---------|
| **Agent** | Bearer token issued at agent startup | Normal `/mesh/*` calls from agent containers |
| **Operator** | Operator session / token | Dashboard-originated mutations, fleet-management routes |
| **Internal** | Loopback request with `x-mesh-internal: 1` (validated as loopback by the server) | Same-host startup glue, browser-service ↔ mesh polling |

Most mesh endpoints accept any of the three. Sensitive surfaces use `_require_operator_or_internal`, which returns **HTTP 403 to agent-tier callers** even with a valid Bearer token. Endpoints demoted to operator-only:
- `GET /mesh/system/metrics` — fleet-wide health, cost, and budget
- `GET /mesh/agents/{id}/metrics` — per-agent cost / budget detail

This is a behavior change from previous versions where any authenticated agent could read these endpoints. Custom agents that polled them now receive 403; switch to dashboard-side reads instead.

### Operator-only Browser Surfaces

Several browser-related dashboard endpoints are operator-only because they let a human inject state into agent sessions:

- `POST /api/agents/{agent_id}/browser/import_cookies` — operator imports cookie / session data into the named agent's browser. 256 KB body cap, 10 imports / hour rate limit, audit-logged. Agent-tier callers cannot use this — only operator session.
- `POST /api/agents/{agent_id}/fingerprint-health/reset` — clears the rolling rejection window after the operator rotates the agent's stealth profile (see Fingerprint Burn Detection below). CSRF-guarded by router.
- `POST /api/browser-captcha-help/complete` and `/cancel` — operator-side resolution of an agent's CAPTCHA help request.

## CAPTCHA Solver Controls

Third-party CAPTCHA solvers (2Captcha, CapSolver) are paid per call. The browser service enforces a layered set of controls before any provider HTTP fires; each layer can short-circuit later layers.

### Kill Switch (`CAPTCHA_DISABLED`)

Fleet-wide kill switch. Set `CAPTCHA_DISABLED=true` to disable all solver activity. Read via the central flags layer (`src/browser/flags.py`), evaluated as **Gate 0** of `_metered_solve` — fires before health, breaker, rate limit, and cost-cap checks. When active:
- Records an audit event with reason `kill_switch_active`
- Returns a `solver_outcome="no_solver"` envelope with `next_action="request_captcha_help"` so the agent escalates to a human

Per-agent override is supported via `flags.set_agent_override(agent, "CAPTCHA_DISABLED", "true")` — re-evaluated on the next solve attempt with no process restart needed. Reset by clearing the env var or override.

### Provider Circuit Breaker

Provider-wide breaker on each solver instance (`src/browser/captcha.py`):
- 3 failures inside any 5-minute sliding window opens the breaker
- Open duration: 10 minutes; auto-resets on read after expiry
- The breaker is shared across agents (one solver instance per provider), so a problem with the upstream provider trips the breaker for the whole fleet — not per-agent

A separate one-shot **health check** runs against the provider's status endpoint (`/balance` for 2Captcha, `/getBalance` for CapSolver) before the first solve; failure leaves the breaker open until a successful re-check.

### Per-agent and Per-tenant Cost Caps

Both caps are opt-in (no default). Configure with USD amounts; enforcement uses millicents (1/100,000 USD) internally for accumulator precision.

- `CAPTCHA_COST_LIMIT_USD_PER_AGENT_MONTH` — short-circuits with `solver_outcome="cost_cap"` before provider HTTP when the agent's running monthly spend would exceed the cap.
- `CAPTCHA_COST_LIMIT_USD_PER_TENANT_MONTH` — same pattern at tenant (project) granularity. The metrics tick emits **threshold alerts at 50 % / 80 % / 100 %** of the cap (once per crossing per month) on the EventBus as `tenant_spend_threshold` events.

Tenant lookup is by `_tenant_for(agent_id)` reverse-mapping `config/projects/`; agents not enrolled in a project have no tenant and do not appear in tenant rollups. State resets at month rollover.

When a cap is configured but the solver provider name or the (provider, kind) price is unknown, the solve **fails closed** with `solver_outcome="provider_missing"` or `"price_missing"` rather than letting an untrackable charge slip past. Reset by configuring the provider, waiting for next month, or explicitly disabling the cap.

The on-disk counter file (`data/captcha_costs.json` by default; override via `CAPTCHA_COST_COUNTER_PATH`) is in-memory + JSON snapshot — restart loses at most one tick's worth of recorded spend.

### Fingerprint Burn Detection

`src/browser/service.py` runs a per-agent rolling window of size 10 over post-solve page-state probes (vendor-specific selectors for Cloudflare 1xxx, DataDome, PerimeterX, Imperva, Akamai BMP, plus branded rejection text). When ≥50 % of the window indicates rejection, the next CAPTCHA envelope surfaces:
- `fingerprint_burn=True`
- `next_action="retry_with_fresh_profile"`

The window is **not** cleared automatically. The operator must rotate the agent's stealth profile and call `POST /api/agents/{agent_id}/fingerprint-health/reset` to clear the rejection state. This avoids ping-ponging between burn / not-burn while the agent is still fingerprinted.

## File-transfer Endpoints

Two-stage upload prevents large bodies from sitting in agent address space and supports idempotent retries:

- `POST /mesh/browser/upload-stage` — Phase A: streams raw bytes into a tmpfs-backed staging directory keyed by an opaque handle. Body-size cap from `OPENLEGION_UPLOAD_STAGE_MAX_MB` (default 50 MB, min 1). Resolved staging paths are validated with `is_relative_to` against the staging dir to block traversal. `Idempotency-Key` header supported. Abandoned `.partial` files reaped at 5× TTL.
- `POST /mesh/browser/upload_file` — Phase B: resolves staged handles to bytes, streams them to the browser container, and drives the `upload_file` action. Cap of 5 files per call. Idempotency is per `(caller, key)` and a cached envelope is returned on replay without re-driving the browser. Cross-replay (same call, different key per handle) returns 404 on the inconsistent handle.
- `POST /mesh/browser/download` — triggers a download in the browser, then streams it to the target agent's `/artifacts/ingest` via a nonce-guarded `_download_stream?nonce=...` and cleans up via `_download_cleanup`. The browser-suggested filename is sanitized with `_sanitize_artifact_name` before reaching the agent. Operator kill switch: `BROWSER_DOWNLOADS_DISABLED=true` returns a 403 forbidden envelope.

All three are gated by `permissions.can_browser_action(target, "upload_file" | "download")` plus the standard rate limits (`upload_stage`, `upload_apply` — see Rate Limiting).
