# Architecture

OpenLegion is a container-isolated multi-agent runtime. LLM-powered agents run in Docker containers, coordinated through a central mesh host. A `SandboxBackend` exists for Docker Desktop microVMs and is used opportunistically — when sandbox init fails it falls back to `DockerBackend`, so Docker is the practical default.

## System Overview

```
User (CLI REPL / Telegram / Discord / Slack / WhatsApp / Webhook / Dashboard)
  -> Mesh Host (FastAPI :8420) -- routes inter-agent messages, enforces permissions,
                                   proxies LLM/API calls, reverse-proxies VNC,
                                   serves the dashboard SPA + /ws/events
    -> Agent Containers (FastAPI :8400 each) -- isolated execution, private memory
                                                 (CLI REPL and dashboard chat hit
                                                  agent endpoints directly)
    -> Browser Service Container (FastAPI :8500) -- one Camoufox per agent,
                                                     per-agent Xvnc + Openbox + unclutter
                                                     on displays :100..:163 paired with
                                                     KasmVNC ports 6100..6163
```

VNC is reached by the user via the mesh proxy at `/agent-vnc/{agent_id}/{path}`, not by hitting a KasmVNC port directly. The `:6080` port from earlier shared-browser designs is legacy and no longer the user-facing surface.

## Trust Zones

Inter-component communication crosses three primary trust zones plus an operator-or-internal tier:

| Level | Zone | Description |
|-------|------|-------------|
| 0 | **Untrusted** | External input (webhooks, user prompts). Sanitized via `sanitize_for_prompt()`; webhook bodies capped at 1 MB; WhatsApp + per-hook HMAC-SHA256 verification when configured. |
| 1 | **Sandboxed** | Agent containers. Non-root UID 1000, `cap_drop=ALL`, `no-new-privileges`, `read_only=True`, `tmpfs=/tmp`, 384MB / 0.15 CPU (worker) or 128MB / 0.05 CPU (operator). Agents hold no LLM API keys — all provider calls proxy through the mesh. |
| 2 | **Trusted** | Mesh host. Holds credentials, manages containers, routes messages. Most mesh endpoints require any-auth (per-agent Bearer token). |
| 2.5 | **Operator-or-internal** | `_require_operator_or_internal` (a third tier between any-auth and loopback-only) gates fleet-wide metrics, per-agent metrics, stale-tasks, and `audit/archive`. |
| 3 | **Loopback-only** | `x-mesh-internal: 1` header **and** loopback source IP — both required. |

SSRF protection is split across two distinct layers, not one:
- **Agent HTTP egress** is policed by `_resolve_and_pin()` in `src/agent/builtins/http_tool.py` (per-hop DNS pinning, blocks private/CGNAT/6to4/Teredo/IPv4-mapped IPv6, cross-origin auth-header stripping, max 5 redirects).
- **Browser container egress** is policed by an iptables filter installed by `docker/browser-entrypoint.sh` — the authoritative kernel-enforced control for Firefox/Playwright/XHR/fetch/WebSocket traffic. Operators can adjust via `BROWSER_EGRESS_ALLOWLIST` or disable with `BROWSER_EGRESS_DISABLE=1`.

Agent containers run on a standard Docker bridge (`openlegion_agents`), not an `internal=True` network — bridge mode is required for port publishing on Docker Desktop. Sufficient isolation comes from agents having no credentials and no path out except the mesh proxy.

Everything between zones is HTTP + JSON with Pydantic contracts defined in `src/shared/types.py`.

## Fleet Model

OpenLegion uses a **fleet model, not hierarchy**. There is no CEO agent that routes or delegates. Users talk to agents directly. Agents coordinate through three mechanisms:

- **Blackboard** — shared SQLite-backed key-value state (WAL, CAS via `write_if_version`, audit-log with undo/archive). Lives in `src/host/mesh.py`.
- **PubSub** — event-driven notifications between agents (`src/host/mesh.py:PubSub`).
- **Coordination protocol** — `hand_off`, `check_inbox`, `update_status`, `complete_task` in `src/agent/builtins/coordination_tool.py`. Hand-offs are always durable SQLite task records routed via `LaneManager`; the legacy blackboard `tasks/{agent}/{handoff_id}` path was removed in PR #835.

### Operator agent

A reserved agent named `operator` is auto-created at startup (`_ensure_operator_agent` is defined in `src/cli/config.py` and called from `src/cli/runtime.py`). It runs at lower resource caps (128MB RAM, 0.05 CPU) and carries fleet-management tools (`fleet_tool.py`, `operator_tools.py`). The operator is the only agent that can apply fleet templates, archive teams, or confirm hard config edits.

### Reserved agent IDs

`RESERVED_AGENT_IDS = {"mesh", "operator", "canary-probe"}` (frozenset in `src/shared/types.py`). User-created agents are rejected against this set. `canary-probe` is the stable ID used by the stealth canary in `src/browser/canary.py`, which sweeps test surfaces in the background for fingerprint detectability.

### MessageOrigin propagation

`MessageOrigin` (`src/shared/types.py`) tags every cross-component message with its originating `kind` / `channel` / `user`. `wake_agent` and `create_task` both accept an optional `origin` and merge `origin_header(origin)` into the request. New cross-agent paths that produce work for another agent should read `current_origin` once and forward it to both calls — otherwise the receiving lane worker has no way to auto-notify the originating channel/user when the handoff completes.

## Component Map

### Mesh Host (`src/host/`)

The mesh host runs on the user's machine as a single FastAPI process. It is the trusted coordinator.

| Module | Responsibility |
|--------|---------------|
| `mesh.py` | Blackboard (SQLite WAL, atomic CAS, audit log with undo/archive), PubSub, MessageRouter (with cross-team block + capability-based addressing) |
| `server.py` | FastAPI app factory — 117 `@app.*` endpoints, all permission-checked. Three auth tiers (any-auth / operator-or-internal / loopback). |
| `runtime.py` | `RuntimeBackend` ABC → `DockerBackend` / `SandboxBackend`; the browser-service container lives here too. SandboxBackend falls back to DockerBackend on init failure. |
| `transport.py` | `Transport` ABC → `HttpTransport` / `SandboxTransport` |
| `credentials.py` | Two-tier credential vault (`OPENLEGION_SYSTEM_*` / `OPENLEGION_CRED_*`) + LLM API proxy. OpenAI / Anthropic OAuth support. |
| `permissions.py` | Per-agent ACL enforcement (glob patterns, deny-all default). `can_*()` checks, `KNOWN_BROWSER_ACTIONS` input validator. |
| `health.py` | Health monitor with auto-restart (`MAX_FAILURES=3`, `RESTART_LIMIT=3` per `RESTART_WINDOW=3600s`) and exponential backoff. |
| `costs.py` | Per-agent LLM cost tracking and post-hoc budget warnings (`data/costs.db`, SQLite WAL). |
| `cron.py` | Persistent cron scheduler with heartbeat probes. Accepts 5-field cron or `every N[s/m/h/d]`. |
| `lanes.py` | Per-agent FIFO task queues with two modes: `followup` (default), `steer` (inject into running chat). Auto-notify forwards results back via `MessageOrigin`. |
| `failover.py` | Model health tracking and failover chains (in-memory). |
| `webhooks.py` | Named webhook endpoints (sanitized payloads, 1MB body size limit, optional HMAC). |
| `traces.py` | Request tracing + grouped summaries. Uses `busy_timeout=5000` (lower than the 30000 used elsewhere). |
| `transcript.py` | Provider-specific transcript sanitization (Gemini/Mistral tool-id formats, tool-call pairing). |
| `wallet.py` | Wallet signing service for Ethereum + Solana. Master seed only on the mesh. |
| `api_keys.py` | Named API key management — salted SHA-256 hashes (`sha256(key_id + raw_key)`) stored in `config/api_keys.json`, `X-API-Key` header auth with legacy `OPENLEGION_API_KEY` env fallback. Raw key shown once at creation. |
| `pending_actions.py`, `change_history.py` | Soft-edit / hard-edit pending-action store with TTL split (5 min soft, 30 min hard) and 5-min undo receipts. |
| `orchestration.py`, `orchestration_migration.py` | Durable task records (V2 coordination path) and v1→v2 migration helpers. |

### Agent (`src/agent/`)

Each agent runs in an isolated Docker container with its own FastAPI server (29 endpoints).

| Module | Responsibility |
|--------|---------------|
| `__main__.py` | Container entry point; reads env config, wires components, starts server. Requires `AGENT_ID` / `AGENT_ROLE` / `MESH_URL`. |
| `loop.py` | Bounded execution loop. Task: 20 iterations. Chat: 30 tool rounds per turn, 200 total rounds. Heartbeat: 12 iterations. Session continues cap at 5 (`_MAX_SESSION_CONTINUES`). Env-var bounds clamped via `_clamp_env()`. |
| `tools.py` | Tool discovery and registry; `@tool` decorator system. |
| `mcp_client.py` | MCP server lifecycle management and tool routing. |
| `memory.py` | SQLite + sqlite-vec + FTS5 hierarchical memory (`EMBEDDING_DIM=1536`, weighted 0.7 vector / 0.3 keyword with salience decay). |
| `workspace.py` | Persistent markdown workspace. Scaffold files (`_SCAFFOLD_FILES`): `INSTRUCTIONS.md`, `SOUL.md`, `USER.md`, `MEMORY.md`, `INTERFACE.md`. `HEARTBEAT.md` is intentionally NOT bootstrapped — it is created lazily on first write. `TEAM.md` / `SYSTEM.md` are read-only bootstrap inclusions, not writable scaffolds. (Pre-rename `PROJECT.md` files are migrated to `TEAM.md` once at startup by `team_migration.py`; the bootstrap loader itself reads only `TEAM.md`/`team.md`.) |
| `context.py` | Context window management. Flush facts at 60%, summarize-and-replace at 70%, warn at 80%. Empty summary falls back to hard prune. Write-then-compact flushes facts to `MEMORY.md` before discarding context. |
| `llm.py` | LLM client (`chat_stream()` and `chat()`) — routes through mesh proxy, never holds keys. |
| `mesh_client.py` | HTTP client for agent-to-mesh communication; merges `origin_header()` into `wake_agent` / `create_task`. |
| `loop_detector.py` | Stuck tool-call detection. Warn at 2 repeats, block at 4, terminate at 9. `memory_search` exempt from warn/block. |
| `attachments.py` | Multimodal attachment enrichment (images → base64 vision blocks, PDFs → text extraction). |
| `server.py` | Agent-side FastAPI server: `/task`, `/chat`, `/chat/stream`, `/chat/steer`, `/heartbeat`, `/status`, `/cancel`, `/result`, `/capabilities`, `/invoke`, `/workspace`, `/workspace/{filename}`, `/history`, `/activity`, `/heartbeat-context`, `/artifacts`, `/files`, `/team` (the legacy `/project` alias was removed in PR 3 of the project→team rename), `/config`, and others. |

### Built-in Tools (`src/agent/builtins/`)

| Module | Tools Provided |
|--------|---------------|
| `exec_tool.py` | Shell execution scoped to `/data` (`_MAX_TIMEOUT=300`). |
| `file_tool.py` | File I/O with two-stage path traversal protection (`..` rejected lexically, then walked with `is_symlink` checks). |
| `http_tool.py` | HTTP requests with `$CRED{}` handle substitution and SSRF protection (DNS pinning, IP blocking incl. CGNAT/6to4/Teredo). |
| `browser_tool.py` | Browser automation via per-agent Camoufox (25 `@tool` tools — navigation, interaction, inspection, file transfer, CAPTCHA + handoff). |
| `mesh_tool.py` | Blackboard (with `sanitize_for_prompt()`), PubSub, `notify_user`, `list_agents`, artifacts, cron, spawn. |
| `memory_tool.py` | Memory search with hierarchical fallback, memory save. |
| `coordination_tool.py` | `hand_off`, `check_inbox`, `update_status`, `complete_task`. Origin-propagating; always writes durable SQLite task records (the legacy blackboard path was removed in PR #835). |
| `vault_tool.py` | Credential generation without returning actual values. |
| `web_search_tool.py` | DuckDuckGo search (no API key needed). |
| `image_gen_tool.py` | Image generation via Gemini or OpenAI DALL-E 3 (routed through the mesh proxy with fixed-price cost tracking). |
| `subagent_tool.py` | In-process subagents (`MAX_DEPTH=2`, `MAX_CONCURRENT=3`, `MAX_TTL=600s`). |
| `introspect_tool.py` | Runtime state query — permissions, budget, fleet, cron, health. |
| `tool_authoring.py` | Self-authoring with AST validation. Forbidden imports/calls/attrs prevent `eval`, `exec`, `open`, `__subclasses__`, etc. |
| `fleet_tool.py` | Operator-only fleet management — `list_templates`, `apply_template` (per-slot creation, not atomic across slots). |
| `operator_tools.py` | Operator-only orchestration — `edit_agent` (single tool; all fields apply immediately and emit an undo receipt: 5-min for soft fields, 30-min for hard fields), `undo_change`, `inspect_agents`, `inspect_teams`, team/agent CRUD via `create_team` / `add_agents_to_team` / … (legacy `*_project` names still callable). |
| `wallet_tool.py` | Wallet operations — get address, get balance, read contract, transfer, execute (Ethereum + Solana). |

### Browser Service (`src/browser/`)

The browser service runs in a separate container with `cap_drop=ALL` plus `cap_add=["NET_ADMIN","SETUID","SETGID"]`. The entrypoint runs as root, uses `NET_ADMIN` to install an iptables egress filter, then `exec tini -- gosu browser:browser python -m src.browser` so the long-running Firefox process runs as UID 1000 with no effective capabilities (`no-new-privileges` prevents re-acquisition).

| Module | Responsibility |
|--------|---------------|
| `__main__.py` | Starts the FastAPI command server. Per-agent Xvnc + Openbox + unclutter stacks are spawned lazily inside `BrowserManager._spawn_per_agent_x_stack`. |
| `server.py` | FastAPI app. Raises `RuntimeError` on startup if `MESH_AUTH_TOKEN` set but `BROWSER_AUTH_TOKEN` absent in production. |
| `service.py` | `BrowserManager` with one Camoufox per agent. Per-agent X11 WID tracking for targeted VNC focus. Fingerprint health monitor + per-tenant cost telemetry. |
| `display_allocator.py` | 64-slot allocator pairing display `:100..:163` with KasmVNC port `6100..6163`. The legacy shared display lives on `:99/:6080` but is not the active user-facing surface. |
| `captcha.py` / `captcha_cost_counter.py` / `captcha_policy.py` | CAPTCHA solver core (2captcha + capsolver), per-tenant millicent-precision cost rollups, per-site policy classifier. |
| `js_challenge.py` | JS-challenge vendor detection (Cloudflare UAM, DataDome, PerimeterX, Imperva, Akamai BMP). |
| `session_persistence.py` | Opt-in storage_state sidecar (`BROWSER_SESSION_PERSISTENCE_ENABLED`). |
| `profile_schema.py` | Browser profile schema versioning + uBO migration. |
| `ref_handle.py` | Stable element handles across snapshots (RefHandle / ShadowHop). |
| `canary.py` | Stealth canary — runs as the reserved `canary-probe` agent to sweep fingerprint test surfaces. |
| `recorder.py` | Behavior recorder for replay/debugging. |
| `redaction.py` | Backward-compat shim that delegates to `src/shared/redaction.py`. |
| `stealth.py` | Anti-detection configuration. Mobile profile (`BROWSER_DEVICE_PROFILE`) rewrites UA strings but leaves the underlying Firefox engine fingerprint desktop. |
| `timing.py` | Human-like timing jitter. |
| `flags.py` | Centralized flag loader. `_ENV_ONLY_FLAGS` strips CAPTCHA solver creds from `config/settings.json` at load (env-only by design, bypasses the credential vault). |

### Browser Container Resources

Resources scale with the `OPENLEGION_MAX_AGENTS` environment variable (`src/host/runtime.py:578-599`). **The number of concurrent browser sessions is gated separately by `OPENLEGION_BROWSER_MAX_CONCURRENT` (legacy alias `MAX_BROWSERS`) resolved in `src/browser/__main__.py` — default = memory-derived autodetect (`_autodetect_default_max_browsers` / `_max_from_memory`), clamped to `[1, 64]`, startup-only.**

| Tier | `OPENLEGION_MAX_AGENTS` | RAM | SHM | CPU | Max Browsers |
|------|------------------------|-----|-----|-----|-------------|
| Basic | ≤ 1 | 2GB | 512MB | 1.0 core | 1 |
| Growth | 2–5 | 4GB | 1GB | 1.5 cores | `max_agents` |
| Pro | 6–15 | 8GB | 2GB | 2.0 cores | `min(max_agents, 10)` |
| Pro Max | > 15 | 16GB | 4GB | 4.0 cores | `min(max_agents, 30)` |

SHM (shared memory) is critical for Firefox compositor IPC — too small causes VNC rendering freezes. Each per-agent X stack shares this allocation.

### Channels (`src/channels/`)

| Module | Platform |
|--------|----------|
| `base.py` | Abstract `Channel` + `PairingManager`. Provides @mention routing, /commands, message chunking. All messages sanitized via `sanitize_for_prompt()`. |
| `telegram.py` | Telegram Bot API adapter (sanitized streaming). |
| `discord.py` | Discord Bot adapter (sanitized streaming). |
| `slack.py` | Slack adapter (Socket Mode via slack-bolt, sanitized streaming). |
| `whatsapp.py` | WhatsApp Cloud API adapter. `WHATSAPP_APP_SECRET` is mandatory (decoupled from `MESH_AUTH_TOKEN`): `start()` raises without it and the webhook returns 503 fail-closed. Mandatory `X-Hub-Signature-256` HMAC verification (401 on mismatch). Inbound dedup by `message["id"]` via bounded TTL/LRU set. |

### Dashboard (`src/dashboard/`)

Alpine.js SPA + Tailwind via CDN (no build step) with Jinja `autoescape=True` and CSP headers. CSRF protection via the `X-Requested-With` header on state-changing endpoints. The user reaches all endpoints under the `/dashboard` prefix.

| Module | Purpose |
|--------|---------|
| `server.py` | Dashboard FastAPI router with 160 API endpoints + VNC URL injection. Top-nav tabs: Chat / Teams / Work / Settings (internal IDs `chat` / `fleet` / `workplace` / `system` — frozen for URL stability). |
| `events.py` | `EventBus` for real-time WebSocket streaming on `/ws/events` (mounted on the mesh app, not the dashboard router). |
| `auth.py` | Session cookie verification — HMAC-SHA256 over `{expiry}.{signature}`, 24h max lifetime, 5-min skew tolerance. |
| `notifications.py` | Persistent notifications store (SQLite WAL) — backs the top-nav bell. |
| `telemetry.py` | Telemetry event sink for the SPA (`dashboard_telemetry` table). |
| `platform_success.py` | Per-tenant success scoring. |

The SSO callback (`/__auth/callback`) is **not implemented in this repo** — it lives in an upstream Caddy auth-gate sidecar deployed via cloud-init. Engine only verifies the `ol_session` cookie that gate sets.

### CLI (`src/cli/`)

| Module | Purpose |
|--------|---------|
| `main.py` | Click commands and entry point (`start`, `stop`, `status`, `chat`, `version`, `wallet`, `teams` (alias `projects`), `team` (alias `project`), `tasks`, `pending`, `confirm`, `cancel`, `reset`). |
| `config.py` | Config loading, Docker helpers, fleet template system (`_load_templates()`, `_create_agent_from_template()`, `_apply_template()`). |
| `runtime.py` | `RuntimeContext` — full lifecycle management. Auto-creates the `operator` agent, enforces `RESERVED_AGENT_IDS`, falls back from `SandboxBackend` to `DockerBackend` on init failure. |
| `repl.py` | `REPLSession` — interactive command dispatch. **Talks directly to agent endpoints** (`/chat`, `/chat/stream`), not through a mesh router hop. |
| `channels.py` | `ChannelManager` — messaging channel lifecycle. |
| `formatting.py` | Tool display, styled output, response rendering. |

### Fleet templates (`src/templates/`)

13 YAML templates: `starter`, `content`, `deep-research`, `devteam`, `monitor`, `sales`, `competitive-intel`, `lead-enrichment`, `price-intelligence`, `review-ops`, `social-listening`, `research`, `opportunity-finder`. Each defines one or more agents with `role`, `model` (the placeholder `{default_model}` is substituted at apply time), `instructions`, `soul`, `heartbeat`, `interface`, `budget`, and `permissions`. Templates are auto-discovered by `_load_templates()` in `src/cli/config.py`. The operator's `apply_template` is per-slot — a mid-loop failure leaves earlier-created agents in place, so callers should verify the returned `created` list.

### Shared (`src/shared/`)

| Module | Purpose |
|--------|---------|
| `types.py` | **THE contract** — all Pydantic models shared between host and agents (853 lines, 27 models). Holds `RESERVED_AGENT_IDS`, `HARD_EDIT_FIELDS` / `SOFT_EDIT_FIELDS`, and the `DashboardEvent.type` Literal enumerating 53 WebSocket event names. |
| `utils.py` | ID generation, structured logging, prompt injection sanitization (`sanitize_for_prompt()`). |
| `trace.py` | Distributed trace context propagation, `origin_header()` helper. |
| `models.py` | Model cost and context window registry (backed by LiteLLM). |
| `redaction.py` | Central credential/URL redactor — `SECRET_PATTERNS` (9 regexes), `SENSITIVE_QUERY_PARAMS`, `redact_url()`, `deep_redact()`. |
| `operator_playbooks.py` | Operator system prompts and the bootstrap greeting copy. |

## Storage Layer

All persistent state is SQLite (WAL mode, `busy_timeout=30000` except `traces` which uses 5000). No Redis, no external databases.

| Location | What | Owner |
|---|---|---|
| `data/costs.db` | Per-agent + per-team LLM cost ledger | `src/host/costs.py` |
| `data/wallet.db` | Agent wallet index, addresses, derivation metadata | `src/host/wallet.py` |
| `data/captcha_costs.json` | CAPTCHA cost ledger in millicents (1/100,000 USD) | `src/browser/captcha_cost_counter.py` |
| Mesh-side blackboard / pubsub / audit-log SQLite | Inter-agent state, atomic CAS, undo/archive | `src/host/mesh.py` |
| `data/traces.db` | Request traces (lower `busy_timeout=5000`) | `src/host/traces.py` |
| `data/dashboard.db` | Notifications + telemetry | `src/dashboard/notifications.py`, `src/dashboard/telemetry.py` |
| Per-agent Docker volume `openlegion_data_{name}` → `/data` | Agent workspace markdown, memory DB, daily logs | `src/agent/workspace.py`, `src/agent/memory.py` |
| `config/agents.yaml` | Per-agent config (model, role, instructions, budget, resources) | `src/cli/config.py` |
| `config/permissions.json` | Per-agent ACL matrix | `src/host/permissions.py` |
| `config/cron.json` | Cron + heartbeat job definitions (atomic temp+rename) | `src/host/cron.py` |
| `config/api_keys.json` | Named API key hashes — `sha256(key_id + raw_key)` | `src/host/api_keys.py` |
| `config/mesh.yaml` | Mesh + LLM default config | `src/cli/config.py` |
| `config/teams/{name}/` | Team metadata + `team.md` (read-only mount into member containers). The startup migrator creates a `config/projects` symlink at the legacy path so pre-rename code paths still resolve. | `src/cli/config.py` |
| `config/settings.json` | Dashboard / browser flag overrides (CAPTCHA solver creds STRIPPED here at load) | `src/browser/flags.py` |
| `.env` | API keys and config — written atomically with `chmod(0o600)` | `src/host/credentials.py` |

## Team Isolation

Agents can be organized into **teams** — isolated namespaces that scope
blackboard access, agent visibility, and shared context. (Internally
the storage prefix and config directory are still `projects/...`
through PR 2 of the project→team rename.)

### How It Works

- **Blackboard scoping**: The `MeshClient` auto-prefixes all blackboard keys with `projects/{name}/`. An agent writing to `tasks/research_01` on the "sales" team actually writes to `projects/sales/tasks/research_01`. Agents see natural keys — the prefix is stripped on read. This is enforced at both the client (auto-namespacing) and server (permission matrix) layers.
- **Agent visibility**: `list_agents` returns only team peers for team agents, or only the agent itself for solo agents.
- **TEAM.md**: Only team members receive a `TEAM.md` mounted read-only into their container at `/app/TEAM.md`. Solo agents get none. (Pre-rename `PROJECT.md` files are migrated to `TEAM.md` once at startup by `team_migration.py`; the bootstrap loader reads only `TEAM.md`/`team.md`.)
- **Permission management on agent create**: `POST /mesh/agents/create` writes defaults of `blackboard_read=["*"]`, `blackboard_write=["tasks/*","context/*","status/*","output/*","artifacts/*"]`, `can_publish=["*"]`, `can_subscribe=["*"]` (`src/host/server.py:4020-4021`). The effective per-team scope comes from the MeshClient's auto-prefix at runtime, not from the on-disk permission file.
- **Remove from team**: When an agent is removed from a team, `blackboard_read` and `blackboard_write` are cleared to `[]`.
- **Solo agents**: Agents not assigned to any team have no scoped blackboard prefix, see only themselves in `list_agents`, and receive no `TEAM.md`.
- **Cross-team blackboard counter**: `_blackboard_xproject_count` is a process-lifetime observability counter (no enforcement) surfaced on `/mesh/system/metrics` as `blackboard_cross_project_total`.

### Team Data

Team configuration is stored in `config/projects/{name}/`. Each team
directory contains `metadata.yaml` (name, description, created_at,
members list) and `team.md` / legacy `project.md` (shared context
mounted read-only into member containers).

## Data Flow

### Task Execution

```
User -> CLI/Channel/Dashboard -> Agent /task endpoint
  (CLI talks directly; channels and dashboard funnel through the mesh
   for routing/permission/origin stamping, then dispatch via LaneManager)
  LaneManager -> followup/steer mode -> agent /task or /chat/steer
  Agent: load context -> LLM call (via mesh proxy) -> tool execution -> iterate
  Agent -> result via MessageOrigin -> auto-notify back to originating channel
```

### Chat Mode

```
User -> CLI/Dashboard -> Agent /chat/stream endpoint (direct)
  Agent: load workspace bootstrap + memory search -> LLM streaming call -> tool rounds
  Agent -> SSE stream (text_delta, tool_start, tool_result, done) -> User
```

The CLI REPL and dashboard chat both call agent endpoints directly. The mesh router (`MessageRouter` in `src/host/mesh.py`) handles **inter-agent** routing (`wake_agent`, `message`) and capability-based addressing, not the user-to-agent hot path.

### WebSocket events

The dashboard receives real-time state via `/ws/events` (mounted on the mesh app at `src/host/server.py:9571`). `DashboardEvent.type` is a `Literal` of 53 event names in `src/shared/types.py` — the regex-sweep guard `test_every_emit_string_in_src_matches_a_dashboard_event_literal` prevents silent drops.

## Runtime Backends

| Backend | Isolation Level | Requirements |
|---------|----------------|-------------|
| `DockerBackend` | Container (shared kernel) | Any Docker install |
| `SandboxBackend` | MicroVM (own kernel) | Docker Desktop 4.58+ |

Both implement `RuntimeBackend` ABC so the rest of the system is isolation-agnostic. The backend is selected at startup via `--sandbox`; if `SandboxBackend` init raises `subprocess.TimeoutExpired` or `RuntimeError`, `RuntimeContext.start()` falls back to `DockerBackend` (`src/cli/runtime.py:544-560`).

Dockerfiles live at the repo root: `Dockerfile.agent` (python:3.12-slim, non-root UID 1000, `ENTRYPOINT ["tini","--"]`) and `Dockerfile.browser` (Firefox + Camoufox + Xvnc + Openbox + unclutter stack; root entrypoint that installs iptables then drops to UID 1000 via gosu).

## Key Invariants

1. **Agents never hold API keys** — all calls route through the mesh credential vault.
2. **No `eval()` / `exec()` on untrusted input** — tool self-authoring uses AST validation (forbidden imports/calls/attrs).
3. **Permission checks before every cross-boundary operation** — default deny.
4. **Path traversal protection** — agent file operations confined to `/data` (two-stage: lexical `..` reject, then walk with `is_symlink` check).
5. **Bounded execution** — 20 task iterations, 30 tool rounds per chat turn, 200 total chat rounds, **5 session continues max** (`_MAX_SESSION_CONTINUES`). Context compaction triggers at 0.70 fullness, not on a fixed round count; empty summary falls back to hard prune.
6. **Write-then-compact** — facts are flushed to `MEMORY.md` before discarding context.
7. **Tool-call message grouping** — `assistant(tool_calls)` and `tool(results)` are never separated in context trimming.
8. **Unicode sanitization** — all untrusted text passes through `sanitize_for_prompt()` before reaching LLM context (user input, tool results, system prompt context).
9. **Team isolation** — blackboard keys are auto-namespaced under `projects/{name}/`, agent visibility is scoped to team peers, and solo agents have no scoped blackboard prefix.
10. **MessageOrigin propagation** — every cross-agent path that produces work reads `current_origin` once and forwards it to both `wake_agent` and `create_task` so completion notifications reach the originating channel/user.
