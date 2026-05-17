# CLAUDE.md — engine

## Overview

OpenLegion Engine is a container-isolated multi-agent runtime. LLM-powered agents run in Docker containers, coordinated through a central mesh host. Users interact via CLI REPL, messaging channels (Telegram/Discord/Slack/WhatsApp/Webhook), or a web dashboard.

## Architecture

### System Layout

```
User (CLI REPL / Telegram / Discord / Slack / WhatsApp / Webhook)
  → Mesh Host (FastAPI :8420) — routes messages, enforces permissions, proxies APIs, VNC proxy
    → Agent Containers (FastAPI :8400 each) — isolated execution with private memory
    → Browser Service Container (FastAPI :8500) — per-agent Camoufox + per-agent Xvnc/Openbox/unclutter on displays :100..:163 paired with KasmVNC ports 6100..6163
```

Three trust zones: **User** (full trust), **Mesh** (trusted coordinator), **Agents** (untrusted, sandboxed). All cross-zone communication is HTTP + JSON with Pydantic contracts from `src/shared/types.py`.

### Entry Points

| Entry Point | File | What It Starts |
|---|---|---|
| `openlegion` CLI | `src/cli/__main__.py` → `src/cli/main.py` | CLI commands: start, stop, status, chat, wallet, version |
| Agent container | `src/agent/__main__.py` → `src/agent/server.py` | Agent FastAPI server on :8400 |
| Browser service | `src/browser/__main__.py` → `src/browser/server.py` | FastAPI on :8500 (per-agent Xvnc/Openbox/unclutter spawned lazily per browser) |
| Mesh host | `src/host/server.py` | FastAPI app on :8420 (started by CLI) |
| Dashboard | `src/dashboard/server.py` | Mounted as router on mesh host |

### Module Responsibilities

| Path | Responsibility |
|---|---|
| **`src/shared/`** | |
| `types.py` | Pydantic models — THE cross-component contract. `_generate_id()` helper. `AGENT_ID_RE_PATTERN` unified regex. `RESERVED_AGENT_IDS = {"mesh", "operator", "canary-probe"}`. `DashboardEvent.type` Literal enumerates 50 WebSocket event names. `HARD_EDIT_FIELDS = {"model","permissions","budget","thinking"}` / `SOFT_EDIT_FIELDS = {"instructions","soul","heartbeat","heartbeat_schedule","interface","role"}` — source of truth for the **undo-receipt TTL** split (5 min soft / 30 min hard); all edits apply immediately via the unified `/edit-soft` endpoint, the TTL is just how long the user has to click Undo. `VALID_OUTCOMES = {"accepted","acknowledged","rework","rejected"}` for task ratings (`acknowledged` = neutral ➖). |
| `utils.py` | `sanitize_for_prompt()`, `setup_logging()`, misc helpers |
| `trace.py` | Distributed trace-ID generation and propagation |
| `models.py` | Model cost/context window registry backed by LiteLLM, `estimate_cost()` |
| `redaction.py` | Central credential/URL redactor. `SECRET_PATTERNS`, `SENSITIVE_QUERY_PARAMS`, `redact_url()`, `deep_redact()`. |
| **`src/agent/`** | |
| `loop.py` | Agent execution loop (task + chat mode). `MAX_ITERATIONS=20`, `CHAT_MAX_TOOL_ROUNDS=30`, `CHAT_MAX_TOTAL_ROUNDS=200`, `_MAX_SESSION_CONTINUES=5`, `HEARTBEAT_MAX_ITERATIONS=12`. `chat()` is `task_id`-aware: when a `x-task-id` rides the wake chain, the loop auto-calls `set_task_status(done\|failed)` on completion / exception so handed-off tasks don't dangle. |
| `server.py` | Agent FastAPI server. `_FILE_CAPS` enforced on workspace writes (HTTP 413). `_WORKSPACE_ALLOWLIST` (SOUL/HEARTBEAT/USER/INSTRUCTIONS/AGENTS/MEMORY/INTERFACE/OBSERVATIONS) gates reads/writes. |
| `llm.py` | LLM client — routes through mesh proxy, never holds keys |
| `context.py` | Context window management (write-then-compact, `_SUMMARIZATION_INPUT_LIMIT=20_000`). Empty summary falls back to hard prune. |
| `skills.py` | Skill registry and tool discovery |
| `memory.py` | Per-agent SQLite + sqlite-vec + FTS5 memory store |
| `workspace.py` | Persistent markdown workspace. Bootstrap files: TEAM.md (legacy `PROJECT.md` resolves as a read-only fallback), SYSTEM.md, INSTRUCTIONS.md, SOUL.md, USER.md, MEMORY.md, INTERFACE.md. Also manages HEARTBEAT.md, daily logs, learnings. |
| `mesh_client.py` | Agent-side HTTP client for mesh. `wake_agent` and `create_task` accept optional `origin: MessageOrigin` and merge `origin_header(origin)` into request headers. |
| `loop_detector.py` | Tool loop detection with escalating responses (warn/block/terminate) |
| `mcp_client.py` | MCP tool server client and lifecycle |
| `attachments.py` | Multimodal attachment enrichment (images → base64 vision blocks, PDFs → text extraction) |
| **`src/agent/builtins/`** | |
| `browser_tool.py` | Browser automation via shared Camoufox service (24 `@skill` tools): navigation/DOM, interaction, inspection, file transfer, CAPTCHA + handoff. Screenshot capture emits multimodal image blocks. |
| `exec_tool.py` | Shell execution scoped to `/data` (`_MAX_TIMEOUT=300`) |
| `file_tool.py` | File I/O with two-stage path traversal protection (`lstat()` for symlink safety) |
| `http_tool.py` | HTTP requests with CRED handles, SSRF protection (DNS pinning, IP blocking incl. CGNAT, 6to4, Teredo), cross-origin auth header stripping |
| `memory_tool.py` | Memory search with hierarchical fallback, memory save |
| `mesh_tool.py` | Blackboard (with `sanitize_for_prompt()`), pub/sub, notify_user, list_agents, artifacts, cron, spawn |
| `coordination_tool.py` | Structured multi-agent coordination — `hand_off`, `check_inbox`, `update_status`, `complete_task`. `update_status` takes optional `task_id`; with 2+ active tasks and no `task_id` it returns `{error: "ambiguous_task", active: [...], hint}`. With an empty inbox or no active tasks it returns `{updated: False, state, reason: "no active tasks"}` (no-op, not an error). `hand_off` propagates `MessageOrigin` so completion notifications reach the originating channel, and forwards `task_id` on the wake so the recipient's loop auto-closes the task. `check_inbox` also surfaces `events[]` from the back-edge inbox prefix `inbox/{agent}/task_event/` so an originating agent learns when a handed-off task reached terminal status. |
| `vault_tool.py` | Credential generation without returning actual values |
| `web_search_tool.py` | DuckDuckGo search (no API key needed) |
| `image_gen_tool.py` | Image generation via Gemini or OpenAI DALL-E 3, saves output as artifacts |
| `skill_tool.py` | Self-authoring with AST validation. `_FORBIDDEN_IMPORTS` (23 modules), `_FORBIDDEN_CALLS` (16 functions), `_FORBIDDEN_ATTRS` (11 attributes). |
| `subagent_tool.py` | In-process subagents (MAX_DEPTH=2, MAX_CONCURRENT=3, MAX_TTL=600s, DEFAULT_MAX_ITERATIONS=10) |
| `introspect_tool.py` | Runtime state query (permissions, budget, fleet, cron, health) |
| `wallet_tool.py` | Wallet operations — get address, get balance, read contract, transfer, execute (Ethereum + Solana) |
| `fleet_tool.py` | Operator-only fleet management (`list_templates`, `apply_template`). `apply_template` accepts `agent_overrides: dict[str, dict]` per-slot — allowed fields `{model, instructions, soul, heartbeat, interface}`; `role` is template-fixed. Mesh validates upfront before any agent is created; the create loop itself is per-slot, not atomic (see Known Constraints #12). |
| `operator_tools.py` | Operator-only fleet/team orchestration. `read_agent_config` (canonical inverse of `edit_agent` — returns the same field surface so the operator can review current values before mutating; optional `fields=[...]` subset), `list_peer_artifacts` / `read_peer_artifact` (operator can read peer-written artifacts via `GET /mesh/agents/{id}/artifacts[/{name}]`; mirrors the existing dashboard peer-artifact read path, 5 MB cap, path-traversal rejected mesh-side), `edit_agent` (one tool — ALL fields apply immediately and emit an undo receipt: 5-min for soft fields, 30-min for hard fields; no pre-execution confirm step; `heartbeat_schedule` retargets the live cron job), `undo_change`, deprecated stub `confirm_edit` (no-op, retained for in-flight LLM conversations that may still emit it), `cancel_pending_action`, `archive_audit_before`, `save_observations`, `inspect_agents` (optional `stale_threshold_hours: int`), `inspect_teams`, `create_agent`, `create_team` (provenance gate dropped — operator can spawn autonomously), `add_agents_to_team`, `remove_agents_from_team`, `update_team_context`, `set_team_goal`, `manage_agent`, `manage_team`, `manage_task`, `summarize_team_progress`. The 8 legacy `*_project` tools are retained as recoverable error stubs that return `{"error": "renamed", "new_tool": "*_team", ...}` so stale LLM prompts retry on the canonical name. `_OPERATOR_PERMISSION_CEILING` still blocks `can_spawn=True` and `can_use_wallet=True` grants. |
| **`src/host/`** | |
| `server.py` | Mesh FastAPI app — 100 endpoints (`@app.*` decorators), all permission-checked. `_RATE_LIMITS` (19 entries — incl. `auth_failure: (60, 60)` on the agent-self-report path; internal `x-mesh-internal` callers bypass since they are the load-bearing quarantine trigger). `/mesh/agents/{id}/edit-soft` accepts every editable field (soft + hard); receipt TTL is field-aware via `_ttl_for_field()`: `_CHANGE_TTL_SECONDS=300` for soft fields, `_HARD_CHANGE_TTL_SECONDS=1800` for hard fields. `pending_actions` table is now only for deletion confirmations (5-min TTL, mesh-side). The `/mesh/agents/{id}/propose` endpoint was retired in PR #927; `/mesh/config/confirm` remains as the consume endpoint for the delete-confirmation flow (mesh-side pending_actions rows, `action_kind="delete"`). VNC reverse proxy with agent token rejection. Localhost validation for `x-mesh-internal`. `_require_operator_or_internal` permission tier between "any authenticated agent" and loopback-only. `system_metrics` (operator-or-internal) surfaces `blackboard_cross_project_total` (process-lifetime read/write counter, observability-only) and `tool_denials_24h` (24h rolling counter, categories `auth`/`scope`/`role`/`permission`/`rate`). Per-agent fields on metrics: `per_agent_cost_today_usd`, `per_agent_cost_vs_yesterday_ratio`, `outcome_rejected_24h_count`, `execution_failures_24h_count`, `stale_tasks_24h_count`. `GET /mesh/agents/{agent_id}/stale-tasks?threshold_hours=N` (operator-only) returns up to 5 oldest non-terminal tasks. |
| `mesh.py` | Blackboard (SQLite WAL), PubSub, MessageRouter, `audit_log` table with `undoable` + `archived` columns. Composite index `idx_audit_log_active(archived, id DESC)`. |
| `runtime.py` | RuntimeBackend ABC → DockerBackend / SandboxBackend. Container security: non-root UID 1000, `cap_drop=[ALL]`, `no-new-privileges`, `read_only=True`, `tmpfs=/tmp` (100m, noexec, nosuid), `mem_limit=384m`, `cpu_quota=15000` (0.15 CPU), `pids_limit=256`. |
| `transport.py` | Transport ABC → HttpTransport / SandboxTransport |
| `credentials.py` | Two-tier credential vault (SYSTEM_*/CRED_*) + LLM API proxy. OpenAI OAuth support. |
| `permissions.py` | Per-agent ACL enforcement (glob patterns, deny-all default). `can_spawn`, `can_manage_cron`, `can_browser_action`. `KNOWN_BROWSER_ACTIONS` frozenset (mesh-side input validator; rejects typos with HTTP 400). `browser_actions: list[str] \| None` on `AgentPermissions`: `None` = all known actions allowed (default), `["*"]` = explicit allow-all, `[]` = deny-all, specific list = opt-out narrowing. |
| `lanes.py` | Per-agent FIFO task queues (followup/steer/collect modes) |
| `health.py` | Health monitor with auto-restart and rate limiting |
| `costs.py` | Per-agent cost tracking + budget enforcement (SQLite) |
| `cron.py` | Persistent cron scheduler with heartbeat probes. `_UPDATABLE_FIELDS` frozenset allowlist. |
| `failover.py` | Model health tracking + failover chains |
| `traces.py` | Request tracing + grouped summaries |
| `transcript.py` | Provider-specific transcript sanitization |
| `webhooks.py` | Named webhook endpoints (payloads sanitized, 1MB body size limit) |
| `wallet.py` | WalletService — Ethereum and Solana wallet operations |
| `api_keys.py` | Named API key management — salted SHA-256 hashes in `config/api_keys.json`. Raw keys returned once at creation. |
| **`src/browser/`** | |
| `__main__.py` | Starts the FastAPI command server. Per-agent Xvnc + Openbox + unclutter stacks spawned lazily by `BrowserManager._spawn_per_agent_x_stack`. |
| `server.py` | Browser service FastAPI app. Raises `RuntimeError` on startup when `MESH_AUTH_TOKEN` is set but `BROWSER_AUTH_TOKEN` is missing (warns in dev). |
| `service.py` | BrowserManager with per-agent Camoufox instances. `_MAX_WALK_DEPTH=50` for DOM snapshot. Per-agent X11 WID tracking for targeted VNC focus. Fingerprint health monitor: rolling per-agent rejection window (`_FINGERPRINT_WINDOW_SIZE=10`, burn threshold 50%); post-solve probe checks vendor selectors (Cloudflare 1xxx, DataDome, PerimeterX, Imperva, Akamai BMP) + branded rejection text; burn surfaces `fingerprint_burn=True` + `next_action="retry_with_fresh_profile"`; operator clears manually after profile rotation (no auto-rotate). Per-tenant cost telemetry: per-minute threshold pass on the metrics tick (50/80/100% via `CAPTCHA_COST_LIMIT_USD_PER_TENANT_MONTH`), once per crossing per month; payload dispatched as a `browser_metrics` WebSocket event with `data.type == "tenant_spend_threshold"`. |
| `captcha.py` | CAPTCHA solver core. 2captcha + capsolver providers, vendor classifier, `_VALID_CAPTCHA_KINDS` allowlist (behavioral kinds rejected via `request_captcha_help` handoff), millicent (1/100,000 USD) cost accounting, fleet-wide `CAPTCHA_DISABLED` kill switch, per-agent + per-tenant cost caps, circuit breaker. |
| `captcha_cost_counter.py` | Tenant rollup helpers — `_tenant_for(agent_id)` (LRU(256), reverse map from `config/projects/`), `get_tenant_total`, `get_tenant_breakdown`, `record_tenant_threshold_alerts` (single-fire per crossing per month). In-memory state is current-month only. |
| `captcha_policy.py` | Per-site CAPTCHA policy classifier. Maps domain → behavior recommendation (auto-solve / handoff / skip). |
| `js_challenge.py` | JS-challenge vendor detection. Identifies Cloudflare Under Attack, DataDome, PerimeterX, Imperva, Akamai BMP behavioral challenges that must route to `request_captcha_help`. |
| `session_persistence.py` | Session continuity sidecar. Per-agent storage_state JSON; **opt-in** via `BROWSER_SESSION_PERSISTENCE_ENABLED` (default false). |
| `flags.py` | Centralized flag loader. `KNOWN_FLAGS` (53 entries) with override precedence per-agent → settings.json → env → default. `_ENV_ONLY_FLAGS` blacklist for sensitive solver creds (`CAPTCHA_SOLVER_KEY`, `CAPTCHA_SOLVER_KEY_SECONDARY`, `CAPTCHA_SOLVER_PROXY_LOGIN`, `CAPTCHA_SOLVER_PROXY_PASSWORD`) — STRIPPED from `config/settings.json` at load, env-only by design. |
| `profile_schema.py` | Browser profile schema versioning + uBO migration. |
| `ref_handle.py` | RefHandle / ShadowHop reference resolution — stable element handles across snapshots. |
| `canary.py` | Stealth canary. Background `canary-probe` agent sweeps test surfaces for fingerprint detectability. |
| `recorder.py` | Behavior recorder for replay/debugging. |
| `redaction.py` | Credential redaction for browser output (delegates to `src/shared/redaction.py`). |
| `stealth.py` | Anti-bot fingerprint building (Windows fingerprint, WebRTC kill, `BROWSER_UA_VERSION` override). Mobile emulation via `BROWSER_DEVICE_PROFILE` — caveat: UA strings change, but underlying Camoufox Firefox engine doesn't, so server-side TLS/JA3 fingerprint may still be desktop. |
| `timing.py` | Timing jitter for human-like behavior. |
| **`src/channels/`** | |
| `__init__.py` | `AT_MENTION_RE` regex for mention parsing |
| `base.py` | Abstract Channel with PairingManager. All messages sanitized. |
| `telegram.py` | Telegram bot adapter (sanitized streaming) |
| `discord.py` | Discord bot adapter (sanitized streaming) |
| `slack.py` | Slack adapter (Socket Mode, sanitized streaming) |
| `whatsapp.py` | WhatsApp Cloud API adapter (`X-Hub-Signature-256` verification, warns when disabled) |
| **`src/dashboard/`** | |
| `server.py` | Dashboard FastAPI router + 143 API endpoints + VNC URL injection. Alpine.js SPA + Tailwind via CDN (no build step), Jinja `autoescape=True`, CSP headers, CSRF via `X-Requested-With` on state-changing endpoints. `_notifications_producer` fires `kind="delivered"` on `task_status_changed` with `new_status="done"` (skipping operator-self and pre-rated transitions) — replaces the legacy outcome=="delivered" check which never fired. Four top-nav tabs (`chat`/`workplace`/`fleet`/`system` → labels Chat/Work/Team/Settings — IDs frozen for URL stability, see Constraint #14). Work tab carries a sticky "Needs you" panel (pending deletion confirmations / credential / browser-login / captcha handoffs / blockers / unrated deliveries synthetic item) and an activity feed with pinned blockers + "Recently delivered" cards that carry inline 👍/➖/👎 rating buttons (POSTs to `/api/workplace/tasks/{id}/outcome`, 👎 opens an inline rework brief that auto-spawns the follow-up task). Per-section skeleton loaders + error banners with retry per `workplaceSectionLoading.<section>` / `workplaceErrors.<section>` buckets. System tab has 11 sub-tabs (`activity`, `costs`, `automation`, `integrations`, `apikeys`, `wallet`, `network`, `storage`, `operator`, `browser`, `settings`); operator is the first card in the standalone fleet view (excluded from quota/cost/broadcast math) and is rejected by `_create_project` / `_add_agent_to_project`. |
| `events.py` | EventBus for real-time WebSocket streaming. `threading.Lock` on `emit()`. |
| `auth.py` | Session cookie verification for dashboard access |
| `notifications.py` | Persistent notifications store (SQLite, WAL). Schema: `dashboard_notifications(id, agent_id, ts, kind, title, body, read_at, payload_json)`. Frozen `_KNOWN_KINDS = {delivered, approval, alert, info, blocker, credential}` (unknown accepted with warning). `list_recent()` orders unread-first then `ts DESC`. Backs the top-nav bell. |
| `telemetry.py` | Telemetry sink (`dashboard_telemetry` table). `_MAX_EVENTS=100_000` retention cap, per-session rate limit `RATE_LIMIT_EVENTS_PER_MIN=60`. |
| `platform_success.py` | Per-tenant success scoring. |
| `static/` | JS (app.js, websocket.js), CSS, avatars (50 SVGs), favicons |
| `templates/` | Alpine.js SPA template (index.html) |
| **`src/cli/`** | |
| `main.py` | CLI entry point: start, stop, status, chat, wallet, version |
| `config.py` | Config loading, Docker helpers, fleet template system (`_load_templates()`, `_create_agent_from_template()`) |
| `runtime.py` | RuntimeContext — lifecycle management. `_RESERVED_AGENT_IDS` validation. |
| `repl.py` | REPLSession — interactive command dispatch |
| `channels.py` | ChannelManager — messaging channel lifecycle |
| `formatting.py` | Tool display, styled output, response rendering |
| **`src/templates/`** | 13 YAML fleet templates: starter, content, deep-research, devteam, monitor, sales, competitive-intel, lead-enrichment, price-intelligence, review-ops, social-listening, research, opportunity-finder |
| **Other** | |
| `src/setup_wizard.py` | Interactive setup wizard with validation |
| `src/marketplace.py` | Git-based skill marketplace (install/remove, git hooks disabled via `core.hooksPath=/dev/null`) |

## Cross-Repo Integration

### Engine is standalone

The engine has NO direct dependencies on app/ or provisioner/. No imports, no calls, no shared code. Integration is handled by provisioner and app externally.

### Provisioner → Engine

Provisioner manages engine instances via Docker/systemd on Hetzner VPS:
- Deploys code via `git clone` in cloud-init. The live update path in `provisioner/app/services/ssh.py:run_update()` runs git pull + Docker rebuild + `systemctl restart openlegion` over SSH.
- Writes `.env` with API keys and config via SSH (base64 encoded to prevent injection).
- Health checks by SSH-ing to localhost and hitting `GET /mesh/agents` with `x-mesh-internal: 1`.
- Starts/stops via `systemctl restart openlegion`.

### App → Engine (SSO)

1. App generates HMAC token: `HMAC-SHA256(access_token, "{subdomain}:{expiry}")` → `{expiry}.{signature}`
2. Redirects user to `https://{subdomain}.engine.openlegion.ai/__auth/callback?token=...`
3. Auth gate (deployed via cloud-init) verifies HMAC, sets `ol_session` cookie (one-time-use token replay protection)
4. Caddy reverse proxy uses `forward_auth` to check cookie on every request

### Exposed Endpoints

- `/mesh/agents` — health check target (provisioner hits via SSH + localhost)
- `/__auth/callback` — SSO callback (handled by auth gate behind Caddy, not engine code)
- Dashboard UI on :8420 — user-facing after SSO
- `/agent-vnc/{agent_id}/{path}` — per-agent reverse proxy to that agent's KasmVNC port (resolved via the display allocator inside the browser service)
- `/ws/events` — WebSocket for real-time dashboard updates

## Patterns & Conventions

### Naming
- snake_case for files, functions, variables
- PascalCase for classes and Pydantic models
- `setup_logging("component.module")` for loggers
- `@skill` decorator for agent capabilities
- `_UPPER_CASE` for module-level constants

### Error Handling
- Domain-specific exceptions propagated with context
- Overly broad catches avoided — transient vs permanent distinguished
- `sanitize_for_prompt()` strips invisible Unicode across all input boundaries
- Security errors return generic messages (no leaking internals)

### Async Patterns
- Async by default (FastAPI + asyncio). Blocking calls wrapped in `run_in_executor`.
- `TYPE_CHECKING` imports for circular dependency prevention

### Config & Environment
- `.env` file loaded via python-dotenv at CLI startup
- `OPENLEGION_SYSTEM_<PROVIDER>_API_KEY` env vars for LLM provider keys (mesh-only). **Agent creation validates that the chosen model's provider has credentials configured** — `create_agent` / `apply_template` reject with HTTP 400 (or `ValueError` on the CLI path) if e.g. you ask for `openai/gpt-4o-mini` but only `OPENLEGION_SYSTEM_ANTHROPIC_API_KEY` is set. `available_providers` is surfaced on `/mesh/introspect?section=llm` and `/mesh/system/metrics` so the operator can pick a reachable model up front. Helpers `resolve_provider_for_model()` and `get_available_providers()` live in `src/shared/models.py`.
- `OPENLEGION_CRED_<NAME>` env vars for agent-tier credentials
- `OPENLEGION_MAX_AGENTS`, `OPENLEGION_MAX_PROJECTS` for plan limits
- `OPENLEGION_LOG_FORMAT=json` for production
- `OPENLEGION_BROWSER_MAX_CONCURRENT` (legacy alias `MAX_BROWSERS` still honored) — per-service cap on simultaneous Camoufox instances. **Startup-only**; restart the browser service to apply changes.

### Logging
- `logger = setup_logging("component.module")` — every module
- JSON format in production, human-readable in dev

### State
- All SQLite with WAL mode — blackboard, memory, costs, cron, traces. `busy_timeout=30000` for mesh/costs/memory/wallet; `busy_timeout=5000` for traces.
- No Redis, no external databases

### Security Boundaries

- **Agents never hold API keys.** All LLM/API calls go through mesh credential vault.
- **No `eval()`/`exec()` on untrusted input.** Skill self-authoring uses AST validation.
- **Permission checks on all mesh endpoints.** Default deny.
- **Rate limits on state-mutating mesh endpoints.** 18 categories in `server.py:_RATE_LIMITS`.
- **File path traversal protection.** Two-stage validation in `file_tool.py` (reject `..` before resolution, then walk with symlink resolution via `lstat()`). Workspace `_read_file()` uses `resolve` + `is_relative_to`.
- **Agent container hardening.** Non-root (UID 1000), `no-new-privileges`, `cap_drop=[ALL]`, `read_only=True`, `tmpfs=/tmp` (100m, noexec, nosuid), 384MB memory, 0.15 CPU, `pids_limit=256`. Browser service container has a different posture (writable /home/browser for Firefox state) — see **Browser container egress filter** below.
- **All untrusted text sanitized** via `sanitize_for_prompt()` before reaching LLM context.
- **VNC proxy blocks agent Bearer tokens.** Dashboard auth required (`ol_session` cookie on HTTP and WebSocket).
- **AST validation for skill self-authoring.** `_FORBIDDEN_IMPORTS` (23 modules), `_FORBIDDEN_CALLS` (16 functions incl. eval, exec, open), `_FORBIDDEN_ATTRS` (11 attributes incl. `__dict__`, `__subclasses__`).
- **SSRF protection.** DNS pinning + IP blocking including `0.0.0.0`, CGNAT (`100.64.0.0/10`), IPv4-mapped IPv6, 6to4 (`2002::/16`), Teredo (`2001::/32`). Max 5 redirects with re-validation at each hop.
- **Browser container egress filter.** Browser container runs with `cap_drop=["ALL"]` + `cap_add=["NET_ADMIN","SETUID","SETGID"]`. The entrypoint (`docker/browser-entrypoint.sh`) installs an iptables egress filter as root that REJECTs outbound traffic to RFC1918 / loopback / link-local / CGNAT / IANA-reserved IPv4 and IPv6 equivalents, then drops to UID 1000 via `gosu`. This is the authoritative SSRF control for browser-initiated traffic; mesh-side `_resolve_and_pin()` is the friendly early-reject. Loopback (`-o lo`) is allowed for in-container services. Host network mode is refused unless `OPENLEGION_BROWSER_ALLOW_HOST_NETWORK=1`. Operator allowlist via `BROWSER_EGRESS_ALLOWLIST=cidr,...`; full disable via `BROWSER_EGRESS_DISABLE=1`.
- **Credential isolation.** Two-tier vault (SYSTEM_*/CRED_*), opaque handles. Dashboard shows masked values (last 4 chars).
- **Bounded execution.** 20 iterations for tasks, 30 tool rounds for chat, 200 total chat rounds, token budgets per task.
- **Write-then-compact.** Before discarding context, important facts flush to MEMORY.md. Empty summary falls back to hard prune.
- **CSRF protection.** Dashboard state-changing endpoints require `X-Requested-With` header.
- **Workspace file caps.** `_FILE_CAPS` enforced on workspace writes (HTTP 413). 7 entries: `SOUL.md: 4000`, `INSTRUCTIONS.md: 12000`, `AGENTS.md: 12000`, `USER.md: 4000`, `MEMORY.md: 16000`, `HEARTBEAT.md: None` (uncapped), `INTERFACE.md: 4000`.
- **Webhook body size limit.** 1MB with Content-Length pre-check.
- **Wallet seed protection.** Seed reveal endpoint returns HTTP 410 (seed shown once at init). Init response has `Cache-Control: no-store`.
- **Env file permissions.** `.agent.env` written with `chmod(0o600)`.
- **Per-action browser permission gate.** `AgentPermissions.browser_actions: list[str] | None` narrows the per-action surface. Semantics: `None` = all known actions allowed (back-compat), `["*"]` = explicit allow-all, `[]` = deny all, specific list = opt-out narrowing. `KNOWN_BROWSER_ACTIONS` rejects unknown action names with HTTP 400.
- **Operator-or-internal permission tier.** `_require_operator_or_internal` on `host/server.py` is a third tier between "any authenticated agent" and loopback-only `x-mesh-internal`. Gates endpoints like `/mesh/system/metrics` and `/mesh/agents/{id}/metrics`.
- **Reserved agent IDs.** `RESERVED_AGENT_IDS = {"mesh", "operator", "canary-probe"}` — `canary-probe` is the stable agent ID owned by the stealth canary sweeper.
- **Two-stage upload protocol.** `/mesh/browser/upload-stage` stages bytes to a tmpfs scratch dir with idempotency-key support; `/mesh/browser/upload_file` resolves staged handles and forwards to the browser service. A 5×TTL `.partial` reaper sweeps abandoned stages. Per-file cap 50 MB (`OPENLEGION_UPLOAD_STAGE_MAX_MB`), max 5 files (`_UPLOAD_MAX_FILES=5`).
- **CAPTCHA solver credentials bypass the vault.** `CAPTCHA_SOLVER_KEY`, `CAPTCHA_SOLVER_KEY_SECONDARY`, `CAPTCHA_SOLVER_PROXY_LOGIN`, `CAPTCHA_SOLVER_PROXY_PASSWORD` are listed in `flags._ENV_ONLY_FLAGS` and stripped from `config/settings.json` at load. Env-only by design; pushed to the browser service container via `/restart-agents`.
- **Denial counter (observability).** `tool_denials_24h` on `/mesh/system/metrics` counts denials in five frozen categories — `auth`, `scope`, `role`, `permission`, `rate`. 24h rolling window, lazy day rollover. Operator-or-internal gated. No enforcement effect.
- **Blackboard cross-project counter (observability).** `_blackboard_xproject_count` (process-lifetime, `read`/`write` keys) increments only when a worker touches a key authored by a worker in a disjoint project set. Operator and `x-mesh-internal` callers bypass; standalone agents (no project membership) skipped. Surfaced as `blackboard_cross_project_total` on `system_metrics`. NO enforcement effect.

## Dependencies & Infrastructure

### Key Dependencies
- `fastapi` + `uvicorn` — HTTP servers
- `httpx` — async HTTP client
- `pydantic` — data validation and cross-component contracts
- `litellm` (pinned `>=1.83.0,<1.84.0`) — model routing (100+ LLM providers)
- `sqlite-vec` — vector similarity search for agent memory
- `docker` — container management
- `pyyaml` — template/config parsing
- `click` — CLI framework
- `websockets` — dashboard real-time updates
- `pypdf` — PDF text extraction for attachments

### Optional Dependency Groups
- `channels` — python-telegram-bot, discord.py, slack-bolt
- `wallet` — web3, eth-account, mnemonic, solders, solana
- `mcp` — mcp (Model Context Protocol)
- `dev` — pytest, pytest-asyncio, pytest-cov, ruff

### Runtime Infrastructure
- **Runtime**: Python 3.10+, FastAPI, asyncio
- **Isolation**: Docker containers per agent, bridge network (`openlegion_agents`). `Dockerfile.agent` and `Dockerfile.browser` in repo root.
- **Browser**: Shared container running one Camoufox + Xvnc + Openbox + unclutter stack per agent (display 100..163, paired KasmVNC ports 6100..6163). Per-agent X11 WID targeting for focus.
- **Dashboard**: Alpine.js SPA — no React, no build step (Tailwind via CDN at index.html)
- **CI**: GitHub Actions — lint (ruff) + tests (pytest) on Python 3.11 and 3.12
- **Dependencies**: `pyproject.toml` uses minimum version bounds (`>=`), no lock file

## Known Constraints & Decisions

1. **Fleet model, not hierarchy.** No CEO agent. Users talk to agents directly. Agents coordinate through blackboard.
2. **Skills over features.** New agent capabilities added as `@skill` decorated functions, not loop changes.
3. **Module-level globals.** `_skill_staging` in skills.py (threading lock protected), `_client` in http_tool.py (connection pooling). Avoid adding more.
4. **Subagent browser concurrency.** Module-level state means subagents shouldn't use browser concurrently.
5. **VNC proxy creates httpx client per request** — acceptable at current usage levels.
6. **`src/shared/types.py` is the contract.** Every cross-component message is a Pydantic model here. Distinct from `DashboardEvent.type` (50 WebSocket event-name literals) — the two are easy to conflate.
7. **LLM tool-calling message roles must alternate.** `user → assistant(tool_calls) → tool(result) → assistant`. `_trim_context` merges summary into first user message to preserve this invariant.
8. **busy_timeout variance.** Traces uses 5000ms while other SQLite connections use 30000ms.
9. **Monolithic server files.** `dashboard/server.py` (145 endpoints) and `host/server.py` (100 endpoints) are single function-scoped definitions.
10. **Shared SQLite + JSON helpers.** `src/shared/sqlite_helpers.py::open_db(path, *, busy_timeout_ms=30000, check_same_thread=False)` captures the standard sqlite3.connect + busy_timeout pattern (9 sites use it; 11 sites with compound pragmas — `isolation_level=None` autocommit, URI mode — remain inline). `src/shared/utils.py::dumps_safe(obj, **kwargs)` wraps `json.dumps(default=str)` and passes through `indent`/`sort_keys`/`separators`/`ensure_ascii`. New code should reach for both helpers first before duplicating the patterns.
11. **Browser surface caveats.** CAPTCHA cost is tracked in **millicents** (1/100,000 USD), persisted to `data/captcha_costs.json`. Per-agent + per-tenant monthly caps with 50/80/100% threshold alerts. Fleet-wide `CAPTCHA_DISABLED` kill switch + per-provider circuit breaker. **Session continuity is opt-in** via `BROWSER_SESSION_PERSISTENCE_ENABLED` (default false). `BROWSER_DEVICE_PROFILE` rewrites UA strings but the underlying Camoufox Firefox engine doesn't change — server-side TLS/JA3 fingerprint may still be desktop. **Fingerprint burn does not auto-rotate** — operators must clear the burn flag manually after rotating the profile.
12. **`apply_template` is per-slot, not atomic.** Mesh validates upfront (unknown agent names, unknown override fields, oversized string fields rejected before any agent is created), but the create loop is not transactional — a mid-loop failure leaves earlier-created agents in place. Operator/LLM should verify the returned `created` list matches the requested slot set.
13. **`MessageOrigin` propagation pattern.** `wake_agent` and `create_task` both accept an optional `origin: MessageOrigin` and merge `origin_header(origin)` into the request. New cross-agent paths that produce work for another agent should read `current_origin` once and forward it to both calls — otherwise the receiving agent's lane worker has no way to auto-notify the originating channel/user when the handoff completes.
14. **Tab IDs are frozen for URL stability.** The four top-nav tab IDs (`chat`, `fleet`, `workplace`, `system`) are load-bearing — they appear in URL paths, JS state vars, and dashboard endpoints. User-facing labels diverge: `fleet` renders as "Teams" (post-rename), `workplace` renders as "Work", `system` renders as "Settings". Renaming the IDs would break deep-links and persisted preferences.
15. **Wizard state machine is `idle | ask | confirming | building | first-output | build_failed`.** Persisted to `localStorage.ol_wizard`; resets to `idle` on unknown values. The wizard mounts only when `step !== 'idle'`. Mutually exclusive with the "What's new" tour (existing-fleet users only); tour gates on `localStorage.olSeenWhatsNew !== 'true'` AND `fleetAgents.length > 0` AND `wizard.step === 'idle'`. Tour state lives in memory only — a mid-flight reload aborts.
16. **Auto-close requires task_id plumbing.** Handed-off tasks auto-transition to terminal status only when the wake chain carries `x-task-id` (via `wake_agent(task_id=...)`, threaded through `LaneManager.QueuedTask.task_id` and the `_direct_dispatch` `x-task-id` header). Wake calls without a task_id (legacy callers, heartbeats, manual chats) won't auto-close — that's intentional. Back-edge events go to the originating agent's blackboard at `inbox/{agent}/task_event/{id}` with a 7-day TTL; surfaced via `check_inbox`. Humans still get the lane-worker auto-notify path (no back-edge); self-handoffs (`origin_user == assignee`) skip the back-edge to keep an originator's inbox clean.
17. **Project → team rename — completed (3 PRs).** The domain term flipped from "project" to "team" across every surface: workspace shared-context file is `TEAM.md`, fleet-management dir is `config/teams/`, HTTP routes are `/mesh/teams/*` + `/api/teams/*`, operator tools are `*_team`, DB column is `tasks.team_id`, WebSocket events emit `team_*` (the legacy `project_*` literals stay in `DashboardEvent.type` for type-safety on historical-record code but are no longer emitted). Permanent zero-cost shims retained: `ProjectMetadata = TeamMetadata` type alias, `MeshClient.*_project` methods proxy to `*_team`, `MeshClient.project_name` property aliases `team_name`, `AgentPermissions.can_manage_projects` mirrors `can_manage_teams` via model-validator, CLI `--project` flag (hidden), `OPENLEGION_PROJECT_SCOPE_MODE` / `OPENLEGION_MAX_PROJECTS` env-var fallbacks, `PROJECT.md` workspace bootstrap read fallback (write path dropped). Operator `*_project` tools (`create_project`, `inspect_projects`, etc.) are recoverable error stubs returning `{"error": "renamed", "new_tool": "*_team", ...}` so stale LLM prompts retry on the canonical name. The `tasks.project_id` → `tasks.team_id` column rename is default-on at startup; operators can opt out via `OPENLEGION_TEAM_MIGRATION_RENAME_DB=0` for emergency rollback (orchestration introspects the live column via PRAGMA so either schema shape works without source changes). Public dict keys on task records still include `project_id` for back-compat with consumer code; same for `target_kind="project"` on `pending_actions` rows (backend schema value, not a domain term). The blackboard storage prefix (`projects/{name}/`) is unchanged — that's a backend on-disk namespace, not a domain rename.

## Git Workflow

- **MANDATORY: Use worktrees for ALL code changes.** Every subagent that touches code MUST use `isolation: "worktree"`. Multiple agents work concurrently — without worktrees they overwrite each other's changes, cause conflicts, and break in-progress work. No exceptions.
- **Never `pip install` from a worktree.** Hijacks the global `openlegion` entry point.
- **Never merge directly to main.** Always create a PR via `gh pr create` and merge through GitHub.
- **Wait for CI before merging.** Run `gh pr checks <number> --watch`.
- **Branch naming:** `feat/`, `fix/`, `refactor/`, `docs/`, etc.
- **Commit style:** descriptive subject line, body explains "why". No Co-Authored-By trailers.

## Work Patterns

- **Use subagents to parallelize independent work.** When a task decomposes into independent subtasks (e.g., editing unrelated files, researching separate questions), dispatch them as concurrent subagents rather than working sequentially.
- **Always run tests in a subagent.** Keeps test output out of the main context window.
- **Use Explore subagents for broad codebase research.** Reserve direct Glob/Grep for targeted, known-location searches.

## Testing

```bash
# Unit + integration (fast, no Docker needed)
pytest tests/ --ignore=tests/test_e2e.py --ignore=tests/test_e2e_chat.py --ignore=tests/test_e2e_memory.py --ignore=tests/test_e2e_triggering.py -x

# Single test file
pytest tests/test_loop.py -x -v
```

- Mock LLM responses, not the loop. See `tests/test_loop.py:_make_loop()`.
- `AsyncMock` for async methods, SQLite in-memory or `tmp_path` for DB paths.
- E2E tests skip gracefully without Docker + API key.
- Tests follow `tests/test_<module>.py` naming; exceptions worth knowing: `test_embedding_fallback.py` (memory fallback path), `test_integration.py` (cross-component), `test_e2e*.py` (require Docker + API key).

## Code Patterns

### Adding a new built-in tool

1. Create `src/agent/builtins/your_tool.py`
2. Use the `@skill` decorator with `name`, `description`, and `parameters`
3. Parameters dict defines the JSON schema for LLM function calling
4. Accept `mesh_client` and/or `workspace_manager` as keyword-only args if needed (auto-injected)
5. Return a dict (serialized to JSON for the LLM)
6. Add tests in `tests/test_builtins.py`

### Adding a new mesh endpoint

1. Add the route in `src/host/server.py` inside `create_mesh_app()`
2. Enforce permissions — check `permissions.can_*()` before acting
3. Use Pydantic models from `src/shared/types.py` for request/response
4. If agents need to call it, add a method to `src/agent/mesh_client.py`

### Adding a new channel

1. Subclass `Channel` from `src/channels/base.py`
2. Implement `start()`, `stop()`, `send_notification()`
3. Message handling provided by base class (`handle_message`)
4. Add startup logic in `src/cli/channels.py:ChannelManager`

### Adding a new fleet template

1. Create `src/templates/your_template.yaml`
2. Define agents with `role`, `model`, `instructions`, `soul`, `resources`
3. Optionally include `heartbeat_rules`, `permissions`, `budget`
4. Templates auto-discovered by `_load_templates()` in `src/cli/config.py`

## Review State

2026-05-16 — Added /mesh/teams/* and /api/teams/* route aliases as scaffold for the upcoming project→team rename. No behavior change; pure addition.

2026-05-16 — Project → team rename, PR 2 of 3. Internals flipped end-to-end; every back-compat surface in the PR-2 spec is preserved through PR 3. Highlights: `TeamMetadata` / `AgentPermissions.can_manage_teams` / `DashboardEvent.type` team-named literals; `_emit_team_event` helper dual-emits every project lifecycle event under both `project_*` and `team_*` names with both `project_id` and `team_id` keys in the payload; `MeshClient` gains 13 canonical `team_*` methods, legacy `*_project` methods preserved as proxies; `WorkspaceManager` reads `TEAM.md` then falls back to `PROJECT.md`; `DockerBackend` mounts both names; agent server adds `PUT /team` mirror; operator gains 8 canonical `*_team` tools alongside legacy entries; `OPENLEGION_TEAM_SCOPE_MODE` / `OPENLEGION_MAX_TEAMS` env vars with legacy fallback; new `src/host/team_migration.py` idempotent startup migrator (`config/projects/` → `config/teams/` with downgrade symlink, workspace `PROJECT.md` → `TEAM.md` copy, gated DB column rename); 20 migration tests. Deferred to a follow-up PR (out of PR-2 scope cut): full Alpine state rename in `app.js`/`index.html`, source-level `tasks.project_id` → `tasks.team_id` switch in `orchestration.py` (the migration code is wired and tested; only the env-var unwiring remains), CLI command renames, the 13 fleet-template `PROJECT.md` references, and per-doc copy passes outside CLAUDE.md.

2026-05-17 — Project → team rename, PR 3 of 3 (complete). Lands the deferred frontend rename (`activeTeam`, `teams`, `soloAgents`, etc. — 20+ Alpine state vars + computed props + HTML bindings; `/api/teams/*` HTTP calls; WS subscribers flipped to `team_*`; Standalone → Solo in user-facing copy + the `broadcast-solo-input` element id, with the legacy `standalone`/`unassigned` keywords kept in the command-palette fuzzy match for muscle-memory back-compat). Sunsets: 16 `/mesh/projects/*` HTTP routes + 8 `/api/projects/*` dashboard routes (plus `/mesh/tasks/project/{id}`, `/mesh/costs/project/{id}`, `/api/workplace/projects`) deleted in favor of the canonical `/teams/*` siblings; 8 operator `*_project` tools converted to recoverable `{"error": "renamed", ...}` stubs that keep the `@skill` registration so stale LLM prompts retry cleanly; WebSocket dual-emit dropped (only `team_*` fires now); `PROJECT.md` workspace write/mount dropped (read-only fallback kept for emergency recovery); agent `PUT /project` endpoint dropped (PUT /team only). DB column rename `tasks.project_id` → `tasks.team_id` is now default-on at startup with an `OPENLEGION_TEAM_MIGRATION_RENAME_DB=0` opt-out; orchestration introspects the live column via PRAGMA so either schema works without source changes. Permanent zero-cost shims explicitly retained: `ProjectMetadata` type alias, `MeshClient.*_project` proxies, CLI `--project` flag, `OPENLEGION_PROJECT_SCOPE_MODE` / `OPENLEGION_MAX_PROJECTS` env fallback, `can_manage_projects` field + validator. Public dict keys still emit `project_id` alongside `team_id` for consumer back-compat; `target_kind="project"` on `pending_actions` rows is a backend schema value, not touched. Tests: 1 file deleted (`test_team_routes_alias.py`), 270 lines of per-tool legacy stubs collapsed into one parametrized assertion, every `/mesh/projects/*` test URL flipped to `/mesh/teams/*`.
