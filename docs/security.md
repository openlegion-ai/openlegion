# Security Model

This is OpenLegion's approach to **AI agent security**: how a secure, self-hosted AI agent runtime keeps autonomous agents isolated, least-privileged, cost-bounded, and auditable.

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
- Memory / CPU split by role (`src/host/runtime.py:304-311`): **workers get 384 MB / 0.15 CPU**, **operator gets 128 MB / 0.05 CPU** (operator is detected via `env_overrides.ALLOWED_TOOLS` — it's a coordination agent, not a tool-runner, so it gets the lighter ceiling).
- PID limit: 256 processes (`pids_limit: 256`)
- `cap_drop: ALL` (no capabilities re-added)
- Read-only root filesystem (`read_only: True`)
- Tmpfs at `/tmp` (100MB, noexec, nosuid)
- No host filesystem access (only `/data` volume)
- Regular Docker bridge network — agents have internet egress. SSRF protection at the application layer in `src/agent/builtins/http_tool.py` blocks private / loopback / link-local / reserved / unspecified (`0.0.0.0`) / CGNAT / IPv4-mapped / 6to4 / Teredo / multicast ranges, pins DNS per hop (fail-closed on resolution error), allows max 5 redirects with re-validation at each hop, and strips `Authorization` on cross-origin redirects. **This application-layer block is the only SSRF control for agent traffic — there is no kernel-enforced fallback** (unlike the browser container; see §SSRF Protection for the asymmetry).

```bash
openlegion start  # Default container isolation
```

### Browser Service Container

Browser operations run in a separate, longer-lived container with a different posture (writable `/home/browser` for Firefox state) and plan-tier-scaled resources: 2–8GB RAM, 0.5–2.0 CPU, 512MB–2GB SHM, 1–10 concurrent browsers (see Architecture). Capability set: `cap_drop=["ALL"]` plus `cap_add=["NET_ADMIN","SETUID","SETGID"]` — the minimum needed to install the egress filter and drop privileges via `gosu`.

The container's authoritative SSRF control is an **iptables egress filter** installed by `docker/browser-entrypoint.sh` before the browser process starts. The entrypoint runs as root, uses `NET_ADMIN` to REJECT outbound traffic to RFC1918, loopback, link-local, CGNAT, and IANA-reserved IPv4 ranges plus IPv6 equivalents, then drops to UID 1000 via `exec tini -- gosu browser:browser python -m src.browser`. The long-running Firefox/FastAPI process holds no effective capabilities (`no-new-privileges` blocks re-acquisition). **tini (PID 1, root) retains** `NET_ADMIN/SETUID/SETGID` in its effective set — no code in this repo asks tini to exercise those caps (tini just forks/execs its child and reaps zombies), but this is a defense-in-depth note worth being explicit about.

The filter is **fail-closed**:
- Missing `iptables-restore` → exit 1
- Missing `NET_ADMIN` cap (i.e. `iptables -L OUTPUT -n` fails) → exit 1
- IPv6 rules fail to install while IPv6 kernel support exists → exit 1
- Loopback (`-o lo`) is allowed so the FastAPI service can reach in-container peers (per-agent KasmVNCs, the upload-staging surface)
- Bypassing the filter requires explicit `BROWSER_EGRESS_DISABLE=1`

The mesh-side `_resolve_and_pin()` check on `navigate`/`open_tab` (`src/host/server.py`) is a friendly early-reject only — the iptables filter is the boundary that must hold. When `BROWSER_EGRESS_DISABLE=1` is set or the service is launched in host-network mode (requires `OPENLEGION_BROWSER_ALLOW_HOST_NETWORK=1`), **the iptables filter is bypassed** and the only remaining SSRF defense is the mesh-side early-reject — which covers only `navigate`/`open_tab`. Be deliberate before flipping either knob.

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

System credentials are identified by matching known provider names with key suffixes (`_api_key`, `_api_base`). Everything else is an agent credential. The known set is `_LITELLM_NATIVE_PROVIDERS` (`src/host/credentials.py:212-216`) — **15 providers** as of this writing: `anthropic`, `openai`, `openrouter`, `gemini`, `mistral`, `deepseek`, `groq`, `together_ai`, `fireworks_ai`, `perplexity`, `minimax`, `moonshot`, `xai`, `zai`, `ollama`. A credential named e.g. `openrouter_api_key` or `mistral_api_base` is therefore system-tier and unreachable to agents regardless of `allowed_credentials` configuration.

Per-agent access is controlled by `allowed_credentials` glob patterns in `config/permissions.json`:

- `["*"]` -- grants access to all agent-tier credentials
- `["brave_search_*", "myapp_*"]` -- access only matching names
- `[]` -- no vault access (Pydantic default — deny all unless explicitly configured)

Even with `allowed_credentials: ["*"]`, system credentials are **always** blocked. Agents also cannot store or overwrite system credential names via `vault_store`.

> **Be explicit: CRED-tier credentials are agent-readable plaintext by design.** The two tiers are asymmetric. `OPENLEGION_SYSTEM_*` keys are **never** returned to an agent — they are injected server-side by the mesh proxy and the agent only ever sees the API response. `OPENLEGION_CRED_*` credentials are different: an agent whose `allowed_credentials` glob matches a CRED name can resolve its **plaintext** via `vault_resolve` / a `$CRED{name}` handle. That is the intended contract (it is how an agent authenticates a tool call it makes itself), but it means a CRED is only as confined as the agents you grant it to. Scope `allowed_credentials` to the narrowest glob that works; assume any agent that can match a CRED can read it. See `docs/security-remediation-review-2026-05-29.md` (L3, L14) — and note the MCP asymmetry below.

> **`$CRED{}` http_tool handles vs. MCP env secrets.** When an agent uses `$CRED{name}` through `http_tool`, the plaintext is resolved **server-side in the mesh** and the agent process never holds it (responses are redacted on the way back — see Credential Redaction). MCP is the asymmetric case: `$CRED{}` handles referenced in an MCP server's `env` / `args` are resolved into the agent container's `MCP_SERVERS` environment variable (`src/host/runtime.py:_build_mcp_servers_env`), so an MCP-using agent's own process **can** read those secrets from its environment. This is unavoidable for stdio MCP — the subprocess needs the secret in-container to authenticate — but it is a real difference from the never-plaintext http_tool path. See `docs/mcp.md` and `docs/security-remediation-review-2026-05-29.md` (L14).

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

### `.env` Persistence

`config/.env` is the persistent backing store for credentials. Writes go through `_persist_to_env()` in `src/host/credentials.py:91-186` with the following protections:

- Reject env keys/values containing `\r\n` — env-injection prevention (`credentials.py:100-101`).
- Validate env key format against `^[A-Za-z_][A-Za-z0-9_]*$` (`credentials.py:103`).
- Atomic write — temp file + `chmod(0o600)` + `fsync` + `rename` (`credentials.py:140-153`). The `.env` file is created/replaced with `0o600` permissions on every persist.
- Values single-quoted to disable python-dotenv interpolation; production also passes `interpolate=False` as a second layer.

`SandboxBackend` writes the agent's auth token to `.agent.env` with `chmod(0o600)` (`runtime.py:875`) so the microVM root filesystem can't be read by another user on the host.

### Wallet Seed Protection

The wallet master seed is a 256-bit BIP-39 mnemonic stored as `OPENLEGION_SYSTEM_WALLET_MASTER_SEED` and never resolvable by agents (system-tier credential). Reveal handling (`src/dashboard/server.py:3907-3963`):

- **`POST /dashboard/api/wallet/init`** generates the seed and returns it **once** in the response body, with `Cache-Control: no-store, Pragma: no-cache` to defeat caches and intermediate logging. If a seed is already configured, returns HTTP 409.
- **`GET /dashboard/api/wallet/seed`** returns **HTTP 410 Gone** with detail "Seed reveal disabled. The seed was shown at wallet init time." There is no second-chance reveal endpoint — if the operator missed it, rotation is the only path forward.

All signing happens server-side in `src/host/wallet.py`; per-agent EVM keys are derived via BIP-44 `m/44'/60'/{agent_index}'/0/0` and per-agent Solana keys via HMAC-SHA512 over a PBKDF2 of the seed. Private keys never leave the mesh process.

### Named API Keys for External Integrations

`src/host/api_keys.py` issues named API keys for outside callers (webhooks, scripts, etc.):

- **Key ID**: `"ak_" + secrets.token_hex(6)`. **Raw key**: `secrets.token_urlsafe(32)`.
- **Storage**: only a salted hash, `sha256(key_id + raw_key)`, persists to `config/api_keys.json`. The raw key is returned exactly once at creation and is unrecoverable thereafter.
- **`list_keys()`** never returns hashes — only id / name / created / last_used metadata.
- **Comparison** uses `_hmac.compare_digest` for constant-time matching.
- Legacy fallback: the `OPENLEGION_API_KEY` env var is still accepted, also via `_hmac.compare_digest`.

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

- **Team-scoped blackboard** -- agents can only access keys under their team's namespace (`projects/{name}/*`, on-disk prefix retained through PR 2 of the project→team rename). The `MeshClient` auto-prefixes all blackboard keys with the team namespace, so agents use natural keys while isolation is enforced transparently. Solo agents get empty blackboard permissions.
- **Glob patterns** for blackboard paths and credential access
- **Explicit allowlists** for messaging, pub/sub, API access, and credential access
- **Default deny** -- if not listed, it's blocked
- Enforced at the mesh host before every operation

### Per-action Browser Gating

Agents with `can_use_browser=true` can be further restricted to a subset of browser actions via `browser_actions: list[str] | None`. Two distinct checks run on every browser call:

1. **Input validation** — the requested action name must appear in `KNOWN_BROWSER_ACTIONS` (`src/host/permissions.py:31-67`, currently 26 names: `navigate`, `snapshot`, `click`, `type`, `hover`, `screenshot`, `reset`, `focus`, `status`, `detect_captcha`, `scroll`, `wait_for`, `press_key`, `go_back`, `go_forward`, `switch_tab`, `upload_file`, `download`, `find_text`, `open_tab`, `fill_form`, `click_xy`, `inspect_requests`, `solve_captcha`, `request_captcha_help`, `request_browser_login`). Typos and unknown action names are **rejected with HTTP 400** before any permission check fires. This is an input validator, not a permission gate — its job is to reject malformed requests, not to enumerate the allowed surface.
2. **Permission check** — `PermissionMatrix.can_browser_action(agent, action)` (`permissions.py:196-228`) consults `browser_actions`:

| `browser_actions` value | Meaning |
|-------------------------|---------|
| `None` (default; field omitted) | All current and future actions. Default-allow UX — turning the browser on grants full surface. Notably, future actions added to `KNOWN_BROWSER_ACTIONS` will pass through without a permissions-file change. |
| `["*"]` | All actions (explicit form). Functionally identical to `None`, but signals operator intent. |
| Specific list (e.g. `["navigate", "snapshot"]`) | Only the listed actions; everything else denied. Does **not** re-validate the list against `KNOWN_BROWSER_ACTIONS`, so a grant for a future action name will work the day that action ships. |
| `[]` | Deny all browser actions. Equivalent to `can_use_browser=false`. |

The asymmetry vs. `allowed_credentials` (where `[]` is the safe deny-all default) is intentional: browser permissions default-allow because turning the browser on without granting actions is rarely what an operator wants. Use `[]` only when you mean "deny all browser actions for this agent". The default-permission inheritance (when an agent has no own entry but a `default` key exists in the permissions file) propagates `browser_actions` along with the other fields.

### Reserved Agent IDs

`RESERVED_AGENT_IDS = {"mesh", "operator", "canary-probe"}` (`src/shared/types.py`). Agent creation rejects these names; `canary-probe` is reserved for the stealth-canary subsystem so a real agent cannot collide with its profile. The CLI also explicitly rejects the literal `operator` from team membership (`src/cli/config.py`) — operator is a system trust zone, not a team member.

### `MessageOrigin` Propagation

Not a security primitive in its own right, but the propagation pattern matters for delivery correctness on inter-agent handoffs. `MessageOrigin` (channel + user + kind) is the routing handle that lets a completion notification reach the originating channel/user when work has been handed off across agents. `wake_agent` and `create_task` (`src/agent/mesh_client.py`) both accept an optional `origin: MessageOrigin` and merge `origin_header(origin)` (from `src/shared/trace.py`) into the outbound request. New cross-agent paths that produce work for another agent should read `current_origin` once and forward it to both calls — otherwise the receiving agent's lane worker has no way to auto-notify the originating channel when the handoff completes, and the user sees silence.

## Input Validation

### SSRF Protection

Two distinct controls cover two distinct traffic paths, with **deliberately asymmetric layering**:

**Agent HTTP traffic** (`src/agent/builtins/http_tool.py`, application-layer — **this is the only SSRF control for agent containers; no kernel-level fallback exists**):
- Resolves hostnames via `socket.getaddrinfo` and rejects every IP in the resolution set if any is blocked
- Blocks `is_private`, `is_loopback`, `is_link_local`, `is_reserved`, `is_unspecified` (`0.0.0.0`), `is_multicast`, plus CGNAT (`100.64.0.0/10`, RFC 6598)
- IPv4-mapped IPv6 (`::ffff:.../96`), 6to4 (`2002::/16`), and Teredo (`2001::/32`) are recursively decoded and their embedded IPv4 re-checked
- Pins DNS by replacing the hostname with the resolved IP in the request URL, preserving the original Host/SNI for TLS validation
- **Fail-closed on DNS error** — `ValueError("SSRF protection: DNS resolution failed (fail-closed)")` rather than allowing the request through
- Max 5 redirects with re-validation at each hop (re-resolves DNS, re-checks IP); `Authorization` stripped on cross-origin redirects
- A compromised agent container with arbitrary code execution (e.g. a `tool_authoring` bypass or `exec_tool` escape) **could in principle reach private IPs via raw syscalls** — the agent container has a regular bridge network, not iptables egress filtering. The application-layer block stops the supported tool surface; it is not a kernel boundary.

**Browser-initiated traffic** is filtered by the **iptables egress rules in the browser service container** (see Browser Service Container above). That filter is the authoritative SSRF boundary for everything the browser does — page loads, embedded subresources, `inspect_requests` activity, downloads, redirects, XHR, fetch, WebSockets. The mesh-side `_resolve_and_pin()` check on `navigate` and `open_tab` is a **friendly early-reject only**; it returns HTTP 400 with a clear error before the browser ever opens a connection, but it covers only those two action paths — everything else relies on iptables.

The asymmetry is intentional. The browser container is rich-by-necessity (Firefox + Playwright + Xvnc) and needs kernel-level confinement; the agent container is slim and locked down (read-only fs, no caps, no browser) so the kernel boundary it offers is the container itself. Both designs are deliberate, but **don't assume a reader who sees one layer can infer the other**.

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

### Workspace File Surface

Workspace files (the agent's own `SOUL.md` / `INSTRUCTIONS.md` / `MEMORY.md` / etc.) are managed through a dedicated `/workspace/{filename}` endpoint on the **agent's own** FastAPI server, not via the general `/data` file tools. Three protections:

- **`_WORKSPACE_ALLOWLIST` frozenset** (`src/agent/server.py`) — 9 entries: `SOUL.md`, `HEARTBEAT.md`, `USER.md`, `INSTRUCTIONS.md`, `AGENTS.md`, `MEMORY.md`, `INTERFACE.md`, `GOALS.md`, `GOALS.json`. Reads / writes to anything outside this list return HTTP 400.
- **`_FILE_CAPS`** (`src/agent/server.py:332-340`) — char-count caps per file, enforced on `PUT` with HTTP 413 on overflow: `SOUL.md=4000`, `INSTRUCTIONS.md=12000`, `AGENTS.md=12000`, `USER.md=4000`, `MEMORY.md=16000`, `INTERFACE.md=4000`, `HEARTBEAT.md=None` (uncapped).
- **`x-mesh-internal` gate on `PUT /workspace/{filename}`** (`src/agent/server.py:397-402`) — the agent cannot call its own workspace endpoint via `http_tool` or `exec+curl`. Writes must originate from the mesh on loopback with the internal header set. Reads from the agent's own loop are allowed without the gate.

### Tool Self-Authoring

Agents can write and register new tools (`tool_authoring`). Submitted code passes an AST analysis before being saved:
- Forbidden imports (23 modules including `os`, `subprocess`, `socket`, `importlib`, etc.)
- Forbidden calls (16 functions including `eval`, `exec`, `open`, `compile`, etc.)
- Forbidden attribute accesses (11 attributes including `__dict__`, `__subclasses__`, `__globals__`, etc.)
- A forgotten-`await` check (sync functions that call `mesh_client`/`memory_store` coroutines are rejected).
- Tools are capped at 10,000 characters.

**This AST validation is authoring HYGIENE, not a security boundary.** It catches obvious footguns and keeps self-authored tools well-formed — it does **not** contain a malicious agent. Agents already have `run_command`, so in-container code execution is part of the design; a determined agent never needs to smuggle anything past this validator. The **container hardening is the real boundary** (non-root UID 1000, `cap_drop=ALL`, `no-new-privileges`, read-only root fs, memory/CPU/PID limits — see Runtime Isolation above). Do not treat the forbidden-imports / forbidden-calls lists as a sandbox.

**Marketplace tools are loaded without load-time AST validation.** `ToolRegistry.MARKETPLACE_TOOLS_DIR` (`/app/marketplace_tools`) is discovered by importing each module — arbitrary code runs at import time, with no AST gate. This is acceptable today only because that directory is **operator-populated and mounted read-only**. If a remote or agent-reachable marketplace-install path is ever added, pin installs to a verified commit SHA so the audited code is the code that runs.

See `docs/security-remediation-review-2026-05-29.md` (M1, H15) for the full finding.

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
| `api_proxy` | 6000 requests | 60 seconds |
| `vault_resolve` | 10000 requests | 60 seconds |
| `vault_store` | 600 requests | 60 seconds |
| `blackboard_read` | 20000 requests | 60 seconds |
| `blackboard_write` | 10000 requests | 60 seconds |
| `publish` | 20000 requests | 60 seconds |
| `notify` | 3000 requests | 60 seconds |
| `cron_create` | 1000 requests | 60 seconds |
| `spawn` | 600 requests | 60 seconds |
| `wallet_read` | 6000 requests | 60 seconds |
| `wallet_transfer` | 600 requests | 60 seconds |
| `wallet_execute` | 600 requests | 60 seconds |
| `image_gen` | 600 requests | 60 seconds |
| `agent_profile` | 6000 requests | 60 seconds |
| `upload_stage` | 3000 requests | 60 seconds |
| `upload_apply` | 3000 requests | 60 seconds |
| `auth_failure` | 60 requests | 60 seconds |

`_RATE_LIMITS` declares 17 entries statically in `src/host/server.py:788-815`; `ext_credentials` and `ext_status` are registered dynamically when external-API support initializes. All other endpoints fall through to the default `(10000, 60)` — i.e. 10000 requests per 60 seconds. These ceilings exist to catch a genuinely runaway loop in a single-tenant deployment; cost budgets (`costs.py`) and per-tx wallet caps are the real spend guardrails.

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

### `BROWSER_AUTH_TOKEN` is a fleet-wide superuser credential

`BROWSER_AUTH_TOKEN` is a **single, fleet-wide** bearer token. The browser service has **no per-agent identity**: `_verify_auth` (`src/browser/server.py`) compares the request's `Authorization: Bearer <token>` against the one shared token via `hmac.compare_digest` and nothing else. The service trusts the `agent_id` in the URL path (`/browser/{agent_id}/...`) entirely — it never cross-checks that the caller "is" that agent. **Any holder of `BROWSER_AUTH_TOKEN` can therefore drive ANY agent's browser** (navigate, screenshot, fill forms, read the accessibility tree, import session state, etc.).

Because of this, the token must be tightly held:

- **It lives only in the mesh process.** The mesh sets it on the browser container (`BROWSER_AUTH_TOKEN` in `DockerBackend`'s browser environment, `src/host/runtime.py`) and attaches it as the upstream bearer when proxying browser calls.
- **It is NEVER injected into an agent container's environment.** Agent containers receive only their own per-agent `MESH_AUTH_TOKEN`; they reach the browser through the mesh, which holds the browser token on their behalf. Do not add `BROWSER_AUTH_TOKEN` to `env_overrides`, `extra_env`, or any agent-facing config.
- **It must never be written to logs.** Treat it like a root password for the browser fleet.

See `docs/security-remediation-review-2026-05-29.md` (M5).

### Auth Tiers

The mesh distinguishes three caller tiers, but only **two** identification mechanisms exist — Bearer token and the internal-header-plus-loopback pair. Operator is not a separate auth header; it is just the Bearer token whose `agent_id` resolves to `"operator"`.

| Tier | How identified | Used by |
|------|----------------|---------|
| **Agent** | `Authorization: Bearer <token>` matched via `hmac.compare_digest` against the per-agent token issued at startup (`_extract_verified_agent_id`, `src/host/server.py:830-862`). | Normal `/mesh/*` calls from agent containers |
| **Operator** | Same Bearer mechanism as Agent, but the token matches `_auth_tokens["operator"]` (`server.py:864-873`). There is no separate operator-session header on the mesh; the dashboard's `ol_session` cookie is a distinct surface that protects the dashboard UI itself (and the VNC proxy), not `/mesh/*` calls. | Dashboard-originated mutations, fleet-management routes, operator-as-agent calls |
| **Internal** | `x-mesh-internal: 1` header **AND** loopback `request.client.host` (`_is_internal_caller`, `server.py:314-337`). Either alone is insufficient. | Same-host startup glue, browser-service ↔ mesh polling |

When `_auth_tokens` is empty (dev/unconfigured), `_extract_verified_agent_id` returns `"unknown"` and authentication is effectively off — operators running in that mode should know auth is fully disabled.

Most mesh endpoints accept any of the three. Sensitive surfaces use `_require_operator_or_internal`, which returns **HTTP 403 to agent-tier callers** even with a valid Bearer token (recorded as `_record_denial("role")`); unknown tokens get HTTP 401 (`_record_denial("auth")`). Endpoints demoted to operator-only include:
- `GET /mesh/system/metrics` — fleet-wide health, cost, and budget
- `GET /mesh/agents/{id}/metrics` — per-agent cost / budget detail
- `GET /mesh/agents/{id}/stale-tasks` — oldest non-terminal tasks for an agent

This is a behavior change from previous versions where any authenticated agent could read these endpoints. Custom agents that polled them now receive 403; switch to dashboard-side reads instead.

Denial counters (`auth`, `scope`, `role`, `permission`, `rate`) are surfaced on `/mesh/system/metrics` as `tool_denials_24h` for the operator heartbeat. Observability only — no enforcement effect.

### Dashboard Session Cookie

The dashboard surface (UI, `/dashboard/*`, `/ws/events`, `/agent-vnc/*`) is gated by an `ol_session` cookie verified in `src/dashboard/auth.py`:

- **Cookie format**: `"{expiry}.{signature}"` where signature is `HMAC-SHA256(cookie-signing-key, expiry_str)`.
- **Cookie signing key** is derived once via `HMAC-SHA256(access_token, "ol-cookie-signing")` from `/opt/openlegion/.access_token` and cached.
- **24-hour hard cap** (`COOKIE_MAX_AGE = 24*3600`, `auth.py:30,83-84`): cookies whose claimed expiry exceeds `now + 24h + 5min skew` are **rejected even if the signature verifies**. Defense in depth against a misbehaving or compromised issuer setting long-lived cookies.
- **Comparison** uses `hmac.compare_digest` for constant-time signature checks.

### SSO Trust Boundary

SSO is split between two components, only one of which is engine code:

- **Engine implements** `verify_session_cookie()` (above) and reads the `ol_session` cookie on every dashboard request.
- **Engine does NOT implement** `/__auth/callback`, HMAC token issuance, or one-time-use replay protection. Those live in an external **Caddy auth gate** sidecar deployed via cloud-init (per CLAUDE.md / Architecture). The engine receives only the cookie the gate sets after consuming the SSO token.

In other words: any guarantee about token replay / single-use lives outside engine code. Don't attribute SSO-tier guarantees to engine; they belong to the auth gate.

### Dashboard XSS / CSRF Posture

- **Primary XSS defense is Jinja `autoescape=True`** on dashboard templates. CSP is set on the dashboard index (`script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.tailwindcss.com https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline'; …`) but the `'unsafe-inline'` + `'unsafe-eval'` allowances are required for Alpine.js's `Function()` constructor and the Tailwind CDN. Be honest: those allowances materially weaken CSP's XSS-mitigation value — treat CSP here as defense-in-depth around the source-origin guarantees (`default-src 'self'`, `object-src 'none'`), not as XSS protection in its own right.
- **CSRF**: state-changing dashboard endpoints require the `X-Requested-With` header (`_csrf_check`, `src/dashboard/server.py:716-726`), wired as a global router dependency on `/dashboard/*`. Relies on browsers' CORS preflight blocking custom headers from cross-origin scripts.

### VNC Reverse Proxy

`/agent-vnc/{agent_id}/{path}` (HTTP + WebSocket) is a cross-zone surface — it terminates browser traffic at the dashboard and forwards into the browser service. Two distinct checks before the proxy reaches across:

- **Reject any caller bearing a known agent Bearer token** (HTTP 403, WS close code 1008). Token compared via `hmac.compare_digest` against every entry in `_auth_tokens`. WS path also checks the `?token=` query param.
- **Require `ol_session`** dashboard cookie (HTTP and WebSocket). A logged-in human is the only valid caller.

The proxy then attaches a service-tier Bearer for the upstream browser-service call. (A new `httpx.AsyncClient(timeout=10)` per request is a known cost — see CLAUDE.md note #5.)

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
- `CAPTCHA_COST_LIMIT_USD_PER_TENANT_MONTH` — same pattern at tenant (team) granularity. The metrics tick emits **threshold alerts at 50 % / 80 % / 100 %** of the cap (once per crossing per month) on the EventBus as `tenant_spend_threshold` events.

Tenant lookup is by `_tenant_for(agent_id)` reverse-mapping `config/projects/`; agents not enrolled on a team have no tenant and do not appear in tenant rollups. State resets at month rollover.

When a cap is configured but the solver provider name or the (provider, kind) price is unknown, the solve **fails closed** with `solver_outcome="provider_missing"` or `"price_missing"` rather than letting an untrackable charge slip past. Reset by configuring the provider, waiting for next month, or explicitly disabling the cap.

The on-disk counter file (`data/captcha_costs.json` by default; override via `CAPTCHA_COST_COUNTER_PATH`) is in-memory + JSON snapshot — restart loses at most one tick's worth of recorded spend.

### Fingerprint Burn Detection

`src/browser/service.py` runs a per-agent rolling window of size 10 over post-solve page-state probes (vendor-specific selectors for Cloudflare 1xxx, DataDome, PerimeterX, Imperva, Akamai BMP, plus branded rejection text). When ≥50 % of the window indicates rejection, the next CAPTCHA envelope surfaces:
- `fingerprint_burn=True`
- `next_action="retry_with_fresh_profile"`

The window is **not** cleared automatically. The operator must rotate the agent's stealth profile and call `POST /api/agents/{agent_id}/fingerprint-health/reset` to clear the rejection state. This avoids ping-ponging between burn / not-burn while the agent is still fingerprinted.

## Residual Risks & Honest Limitations

These are known, accepted limitations — documented so no one mistakes a best-effort control for a guarantee. Full context in `docs/security-remediation-review-2026-05-29.md`.

- **(M20) The browser's mesh-side SSRF URL check is best-effort, not the boundary.** The `_resolve_and_pin()` early-reject on `navigate` / `open_tab` (`src/host/server.py`) is a friendly pre-check that races against DNS — it does not defend against DNS rebinding, and it covers only those two action paths. In the default Docker **bridge** deployment, the **container iptables egress filter** (installed by `docker/browser-entrypoint.sh`) is the **authoritative** anti-rebinding / anti-SSRF layer for everything the browser does. If that filter is bypassed (`BROWSER_EGRESS_DISABLE=1` or host-network mode), only the partial mesh-side check remains. Do not rely on the mesh-side URL check as a security boundary.

- **(M23) Stored agent memory can carry plaintext prompt injection that re-injects on future tasks.** Untrusted text an agent commits to its own memory (via the memory store) is replayed into that agent's LLM context on later tasks. `sanitize_for_prompt()` runs on that path, but it is **Unicode hygiene only** (strips invisible / smuggling codepoints — see Unicode Sanitization above); it is **not** a prompt-injection defense and does not detect or neutralize natural-language injection. A poisoned memory will re-inject every time it is loaded. The blast radius is **per-agent** — an agent's memory is private to that agent and does not cross into other agents — but a single agent can persistently re-prompt itself. Treat agent memory as attacker-influenceable if the agent ever processes untrusted input.

- **(L19) The `global/` blackboard namespace is fleet-shared, and `wallet_allowed_contracts == []` means allow-all.** Two by-design quirks worth stating plainly: (a) `global/` is a **fleet-wide** namespace, not a per-team or per-agent one. The operator-handoff sub-prefixes have explicit carve-outs (`global/tasks/operator/*` is operator-read-only but any agent may write; `global/output/{agent}/*` is per-sender — see `can_read_blackboard` / `can_write_blackboard` in `src/host/permissions.py`), but any *other* `global/*` key is governed only by an agent's normal `blackboard_read` / `blackboard_write` globs — so a `global/*` glob grants cross-cutting fleet-wide access, unlike the team-scoped `projects/{name}/*` keys. Treat `global/` as shared state, not an isolation boundary, and avoid broad `global/*` globs. (b) For the wallet contract dimension, `wallet_allowed_contracts == []` means **allow-all** (`can_access_wallet_contract` returns `True` on the empty list — `permissions.py:382-383`), not deny-all — the empty list is the "no contract restriction" sentinel. This is still gated by `can_use_wallet*`, so it is not an open door, but the `[]`-means-allow-all polarity is the **opposite** of `allowed_credentials` (`[]` = deny-all) and must not be confused.

## File-transfer Endpoints

Two-stage upload prevents large bodies from sitting in agent address space and supports idempotent retries:

- `POST /mesh/browser/upload-stage` — Phase A: streams raw bytes into a tmpfs-backed staging directory keyed by an opaque handle. Body-size cap from `OPENLEGION_UPLOAD_STAGE_MAX_MB` (default **50 MB**, min 1). Resolved staging paths are validated with `is_relative_to` against the staging dir to block traversal. `Idempotency-Key` header supported. Abandoned `.partial` files reaped at **5× TTL**.
- `POST /mesh/browser/upload_file` — Phase B: resolves staged handles to bytes, streams them to the browser container, and drives the `upload_file` action. Cap of **5 files per call** (`_UPLOAD_MAX_FILES`). Idempotency is per `(caller, key)` and a cached envelope is returned on replay without re-driving the browser. Cross-replay (same call, different key per handle) returns 404 on the inconsistent handle.
- `POST /mesh/browser/download` — triggers a download in the browser, then streams it to the target agent's `/artifacts/ingest` via a nonce-guarded `_download_stream?nonce=...` and cleans up via `_download_cleanup`. The browser-suggested filename is sanitized with `_sanitize_artifact_name` before reaching the agent. Operator kill switch: `BROWSER_DOWNLOADS_DISABLED=true` returns a 403 forbidden envelope.

All three are gated by `permissions.can_browser_action(target, "upload_file" | "download")` plus the standard rate limits (`upload_stage`, `upload_apply` — see Rate Limiting).
