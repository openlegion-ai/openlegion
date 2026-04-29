# CLAUDE.md — engine

## Overview

OpenLegion Engine is a container-isolated multi-agent runtime. LLM-powered agents run in Docker containers, coordinated through a central mesh host. Users interact via CLI REPL, messaging channels (Telegram/Discord/Slack/WhatsApp/Webhook), or a web dashboard.

## Architecture

### System Layout

```
User (CLI REPL / Telegram / Discord / Slack / WhatsApp / Webhook)
  → Mesh Host (FastAPI :8420) — routes messages, enforces permissions, proxies APIs, VNC proxy
    → Agent Containers (FastAPI :8400 each) — isolated execution with private memory
    → Browser Service Container (FastAPI :8500 + KasmVNC :6080) — shared Camoufox browser
```

Three trust zones: **User** (full trust), **Mesh** (trusted coordinator), **Agents** (untrusted, sandboxed). All cross-zone communication is HTTP + JSON with Pydantic contracts from `src/shared/types.py`.

### Entry Points

| Entry Point | File | What It Starts |
|---|---|---|
| `openlegion` CLI | `src/cli/__main__.py` → `src/cli/main.py` | CLI commands: start, stop, status, chat, wallet, version |
| Agent container | `src/agent/__main__.py` → `src/agent/server.py` | Agent FastAPI server on :8400 |
| Browser service | `src/browser/__main__.py` → `src/browser/server.py` | KasmVNC + Openbox + FastAPI on :8500 |
| Mesh host | `src/host/server.py` | FastAPI app on :8420 (started by CLI) |
| Dashboard | `src/dashboard/server.py` | Mounted as router on mesh host |

### Module Responsibilities

| Path | Responsibility |
|---|---|
| **`src/shared/`** | |
| `types.py` | Pydantic models — THE cross-component contract (335 lines, 24 models). `_generate_id()` helper. `AGENT_ID_RE_PATTERN` unified regex. |
| `utils.py` | `sanitize_for_prompt()`, `setup_logging()`, misc helpers |
| `trace.py` | Distributed trace-ID generation and propagation |
| `models.py` | Model cost/context window registry backed by LiteLLM, `estimate_cost()` |
| **`src/agent/`** | |
| `loop.py` | Agent execution loop (task + chat mode). `MAX_ITERATIONS=20`, `CHAT_MAX_TOOL_ROUNDS=30`, `CHAT_MAX_TOTAL_ROUNDS=200`, `_MAX_SESSION_CONTINUES=5`, `HEARTBEAT_MAX_ITERATIONS=10`. Env-var bounds clamped via `_clamp_env()`. |
| `server.py` | Agent FastAPI server (27 endpoints). `_FILE_CAPS` (7 entries) enforced on workspace writes (HTTP 413). `_WORKSPACE_ALLOWLIST` frozenset gates reads/writes. |
| `llm.py` | LLM client — routes through mesh proxy, never holds keys |
| `context.py` | Context window management (write-then-compact, `_SUMMARIZATION_INPUT_LIMIT=20_000`). Empty summary falls back to hard prune. |
| `skills.py` | Skill registry and tool discovery |
| `memory.py` | Per-agent SQLite + sqlite-vec + FTS5 memory store |
| `workspace.py` | Persistent markdown workspace. Bootstrap files: PROJECT.md, SYSTEM.md, INSTRUCTIONS.md, SOUL.md, USER.md, MEMORY.md, INTERFACE.md. Also manages HEARTBEAT.md (loaded separately), daily logs, learnings. |
| `mesh_client.py` | Agent-side HTTP client for mesh communication |
| `loop_detector.py` | Tool loop detection with escalating responses (warn/block/terminate) |
| `mcp_client.py` | MCP tool server client and lifecycle |
| `attachments.py` | Multimodal attachment enrichment (images → base64 vision blocks, PDFs → text extraction) |
| **`src/agent/builtins/`** | |
| `browser_tool.py` | Browser automation via shared Camoufox service. CAPTCHA detection/solving, screenshot capture with multimodal image blocks. |
| `exec_tool.py` | Shell execution scoped to `/data` (`_MAX_TIMEOUT=300`) |
| `file_tool.py` | File I/O with two-stage path traversal protection (`lstat()` for symlink safety) |
| `http_tool.py` | HTTP requests with CRED handles, SSRF protection (DNS pinning, IP blocking incl. CGNAT, 6to4, Teredo), cross-origin auth header stripping |
| `memory_tool.py` | Memory search with hierarchical fallback, memory save |
| `mesh_tool.py` | Blackboard (with `sanitize_for_prompt()`), pub/sub, notify_user, list_agents, artifacts, cron, spawn |
| `coordination_tool.py` | Structured multi-agent coordination protocol — hand_off, check_inbox, update_status. Higher-level wrappers over blackboard for inter-agent work handoffs. |
| `vault_tool.py` | Credential generation without returning actual values |
| `web_search_tool.py` | DuckDuckGo search (no API key needed) |
| `image_gen_tool.py` | Image generation via Gemini or OpenAI DALL-E 3, saves output as artifacts |
| `skill_tool.py` | Self-authoring with AST validation. `_FORBIDDEN_IMPORTS` (23 modules), `_FORBIDDEN_CALLS` (16 functions), `_FORBIDDEN_ATTRS` (11 attributes). |
| `subagent_tool.py` | In-process subagents (MAX_DEPTH=2, MAX_CONCURRENT=3, MAX_TTL=600s, DEFAULT_MAX_ITERATIONS=10) |
| `introspect_tool.py` | Runtime state query (permissions, budget, fleet, cron, health) |
| `wallet_tool.py` | Wallet operations — get address, get balance, read contract, transfer, execute (Ethereum + Solana) |
| `fleet_tool.py` | Operator-only fleet management tools (`list_templates`, `apply_template`) |
| `operator_tools.py` | Operator-only tools for fleet/project orchestration (`propose_edit`, `confirm_edit`, `save_observations`, `read_agent_history`, `create_agent`, `list_projects`, `get_project`, `create_project`, `add_agents_to_project`, `remove_agents_from_project`, `update_project_context`) |
| **`src/host/`** | |
| `server.py` | Mesh FastAPI app factory — ~60 endpoints, all permission-checked. `_RATE_LIMITS` dict (16 entries: 14 static + `ext_credentials`/`ext_status` added at external-API init). VNC reverse proxy with agent token rejection. Localhost validation for `x-mesh-internal`. |
| `mesh.py` | Blackboard (SQLite WAL), PubSub, MessageRouter |
| `runtime.py` | RuntimeBackend ABC → DockerBackend / SandboxBackend. Container security: non-root UID 1000, `cap_drop=[ALL]`, `no-new-privileges`, `read_only=True`, `tmpfs=/tmp` (100m, noexec, nosuid), `mem_limit=384m`, `cpu_quota=15000` (0.15 CPU), `pids_limit=256`. |
| `transport.py` | Transport ABC → HttpTransport / SandboxTransport |
| `credentials.py` | Two-tier credential vault (SYSTEM_*/CRED_*) + LLM API proxy. OpenAI OAuth support. |
| `permissions.py` | Per-agent ACL enforcement (glob patterns, deny-all default). `can_spawn`, `can_manage_cron`. |
| `lanes.py` | Per-agent FIFO task queues (followup/steer/collect modes) |
| `health.py` | Health monitor with auto-restart and rate limiting |
| `costs.py` | Per-agent cost tracking + budget enforcement (SQLite) |
| `cron.py` | Persistent cron scheduler with heartbeat probes. `_UPDATABLE_FIELDS` frozenset allowlist. |
| `failover.py` | Model health tracking + failover chains |
| `traces.py` | Request tracing + grouped summaries |
| `transcript.py` | Provider-specific transcript sanitization |
| `webhooks.py` | Named webhook endpoints (payloads sanitized, 1MB body size limit) |
| `watchers.py` | File watcher with polling (messages sanitized) |
| `wallet.py` | WalletService — Ethereum and Solana wallet operations |
| `api_keys.py` | Named API key management — salted SHA-256 hashes stored in `config/api_keys.json`. Raw keys returned once at creation. |
| `containers.py` | Backward-compat alias for `DockerBackend` (used by E2E tests) |
| **`src/browser/`** | |
| `__main__.py` | Starts KasmVNC (Xvnc), Openbox WM, FastAPI command server |
| `server.py` | Browser service FastAPI app. Raises `RuntimeError` on startup when auth token missing in production (MESH_AUTH_TOKEN set but BROWSER_AUTH_TOKEN absent); warns only in dev. |
| `service.py` | BrowserManager with per-agent Camoufox instances. `_MAX_WALK_DEPTH=50` for DOM snapshot. Per-agent X11 WID tracking for targeted VNC focus. §22 fingerprint health monitor: rolling per-agent rejection window (`_FINGERPRINT_WINDOW_SIZE=10`, burn threshold 50%); post-solve page-state monitor probes vendor-specific selectors (Cloudflare 1xxx, DataDome, PerimeterX, Imperva, Akamai BMP) + branded rejection text; burn surfaces `fingerprint_burn=True` + `next_action="retry_with_fresh_profile"` on subsequent captcha envelopes; operator clears manually after profile rotation. |
| `redaction.py` | Credential redaction for browser output |
| `stealth.py` | Anti-bot fingerprint building (Windows fingerprint, WebRTC kill, `BROWSER_UA_VERSION` override) |
| `timing.py` | Timing jitter for human-like behavior |
| **`src/channels/`** | |
| `__init__.py` | `AT_MENTION_RE` regex for mention parsing |
| `base.py` | Abstract Channel with PairingManager. All messages sanitized. |
| `telegram.py` | Telegram bot adapter (sanitized streaming) |
| `discord.py` | Discord bot adapter (sanitized streaming) |
| `slack.py` | Slack adapter (Socket Mode, sanitized streaming) |
| `whatsapp.py` | WhatsApp Cloud API adapter (`X-Hub-Signature-256` verification, warns when signature verification disabled) |
| **`src/dashboard/`** | |
| `server.py` | Dashboard FastAPI router + 106 API endpoints + VNC URL injection. Alpine.js SPA with `autoescape=True`, CSP headers, CSRF via `X-Requested-With` requirement on state-changing endpoints. |
| `events.py` | EventBus for real-time WebSocket streaming. `threading.Lock` on `emit()`. |
| `auth.py` | Session cookie verification for dashboard access |
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
- Deploys code via `git clone` in cloud-init. An `update.sh` script is shipped alongside, but the live update path in `provisioner/app/services/ssh.py:run_update()` runs the equivalent commands inline over SSH (git pull + Docker rebuild + `systemctl restart openlegion`).
- Writes `.env` with API keys and config via SSH (base64 encoded to prevent injection)
- Health checks by SSH-ing to localhost and hitting `GET /mesh/agents` with `x-mesh-internal: 1`
- Starts/stops via `systemctl restart openlegion`

### App → Engine (SSO)

1. App generates HMAC token: `HMAC-SHA256(access_token, "{subdomain}:{expiry}")` → `{expiry}.{signature}`
2. Redirects user to `https://{subdomain}.engine.openlegion.ai/__auth/callback?token=...`
3. Auth gate (deployed via cloud-init) verifies HMAC, sets `ol_session` cookie (one-time-use token replay protection)
4. Caddy reverse proxy uses `forward_auth` to check cookie on every request

### Exposed Endpoints

- `/mesh/agents` — health check target (provisioner hits via SSH + localhost)
- `/__auth/callback` — SSO callback (handled by auth gate behind Caddy, not engine code)
- Dashboard UI on :8420 — user-facing after SSO
- `/vnc/{path}` — reverse proxy to browser container's KasmVNC
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
- `sanitize_for_prompt()` strips invisible Unicode across all input boundaries (88 call sites across 16 source files)
- Security errors return generic messages (no leaking internals)

### Async Patterns
- Async by default (FastAPI + asyncio). Blocking calls wrapped in `run_in_executor`.
- `TYPE_CHECKING` imports for circular dependency prevention

### Config & Environment
- `.env` file loaded via python-dotenv at CLI startup
- `OPENLEGION_SYSTEM_<PROVIDER>_API_KEY` env vars for LLM provider keys (mesh-only)
- `OPENLEGION_CRED_<NAME>` env vars for agent-tier credentials
- `OPENLEGION_MAX_AGENTS`, `OPENLEGION_MAX_PROJECTS` for plan limits
- `OPENLEGION_LOG_FORMAT=json` for production
- `OPENLEGION_BROWSER_MAX_CONCURRENT` (default `5`, legacy name `MAX_BROWSERS` still honored) — per-service cap on simultaneous Camoufox instances. **Startup-only**; restart the browser service to apply changes (see `src/browser/__main__.py:_resolve_max_browsers`). Runtime reconfig was deferred per plan §10.2 — bounding the acquire semaphore mid-flight is non-trivial.

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
- **Rate limits on state-mutating mesh endpoints.** 16 rate-limited categories defined in `server.py:_RATE_LIMITS` (14 static + 2 added at external-API init).
- **File path traversal protection.** Two-stage validation in `file_tool.py` (reject `..` before resolution, then walk with symlink resolution via `lstat()`). Workspace `_read_file()` uses `resolve` + `is_relative_to`.
- **Agent container hardening.** Non-root (UID 1000), `no-new-privileges`, `cap_drop=[ALL]`, `read_only=True`, `tmpfs=/tmp` (100m, noexec, nosuid), 384MB memory, 0.15 CPU, `pids_limit=256`. Browser service container has a different posture (writable /home/browser for Firefox state) — see **Browser container network egress filter** below for its privilege model.
- **All untrusted text sanitized** via `sanitize_for_prompt()` before reaching LLM context.
- **VNC proxy blocks agent Bearer tokens.** Dashboard auth required (`ol_session` cookie on HTTP and WebSocket).
- **AST validation for skill self-authoring.** `_FORBIDDEN_IMPORTS` (23 modules), `_FORBIDDEN_CALLS` (16 functions incl. eval, exec, open), `_FORBIDDEN_ATTRS` (11 attributes incl. `__dict__`, `__subclasses__`).
- **SSRF protection.** DNS pinning + IP blocking including `0.0.0.0` (unspecified), CGNAT (`100.64.0.0/10`), IPv4-mapped IPv6, 6to4 (`2002::/16`), Teredo (`2001::/32`). Max 5 redirects with re-validation at each hop.
- **Browser container network egress filter.** The shared browser service container runs with `cap_drop=["ALL"]` + `cap_add=["NET_ADMIN","SETUID","SETGID"]` — the minimum capability set needed. The entrypoint (`docker/browser-entrypoint.sh`) runs as root, uses `NET_ADMIN` to install an iptables egress filter that REJECTs outbound traffic to RFC1918 / loopback / link-local / CGNAT / IANA-reserved IPv4 ranges and IPv6 equivalents, then execs `tini -- gosu browser:browser python -m src.browser`. tini (PID 1, root) retains the three caps for the container lifetime; no code in this repo asks tini to exercise them (tini's job is to fork/exec its child and reap zombies). The long-running Firefox/FastAPI process runs as UID 1000 via gosu with no effective capabilities (non-root users do not inherit caps, and `no-new-privileges` prevents re-acquisition). This is the authoritative SSRF control for browser-initiated traffic — the mesh-side `_resolve_and_pin()` check is kept as a friendly early-reject. Loopback (`-o lo`) is allowed so the browser can reach in-container services (KasmVNC :6080, FastAPI :8500 — `/browser/*` requires bearer auth, `/uploads/*` intentionally unauthenticated for navigating to user-uploaded files). Host network mode is refused unless `OPENLEGION_BROWSER_ALLOW_HOST_NETWORK=1` is set. Operator allowlist via `BROWSER_EGRESS_ALLOWLIST=cidr,...`; full disable via `BROWSER_EGRESS_DISABLE=1`.
- **Credential isolation.** Two-tier vault (SYSTEM_*/CRED_*), opaque handles. Dashboard shows masked values (last 4 chars).
- **Bounded execution.** 20 iterations for tasks, 30 tool rounds for chat, 200 total chat rounds, token budgets per task. Env-var bounds clamped with validation.
- **Write-then-compact.** Before discarding context, important facts flush to MEMORY.md. Empty summary falls back to hard prune.
- **CSRF protection.** Dashboard state-changing endpoints require `X-Requested-With` header.
- **Workspace file caps.** `_FILE_CAPS` enforced on workspace writes (HTTP 413). 7 entries: `SOUL.md: 4000`, `INSTRUCTIONS.md: 12000`, `AGENTS.md: 12000`, `USER.md: 4000`, `MEMORY.md: 16000`, `HEARTBEAT.md: None` (uncapped), `INTERFACE.md: 4000`.
- **Webhook body size limit.** 1MB with Content-Length pre-check.
- **Wallet seed protection.** Seed reveal endpoint returns HTTP 410 (seed shown once at init). Init response has `Cache-Control: no-store`.
- **Env file permissions.** `.agent.env` written with `chmod(0o600)`.

## Dependencies & Infrastructure

### Key Dependencies
- `fastapi` + `uvicorn` — HTTP servers
- `httpx` — async HTTP client
- `pydantic` — data validation and cross-component contracts
- `litellm` (pinned `>=1.83.0,<1.84.0`) — model routing (100+ LLM providers). The `<1.82.7` bound was added when 1.82.7/1.82.8 were flagged malicious; those versions are now yanked from PyPI so the safe floor moved to 1.83.0.
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
- **Browser**: Shared container with KasmVNC (Xvnc + Openbox), Camoufox, per-agent X11 WID targeting
- **Dashboard**: Alpine.js SPA — no React, no build step
- **CI**: GitHub Actions — lint (ruff) + tests (pytest) on Python 3.11 and 3.12
- **Dependencies**: `pyproject.toml` uses minimum version bounds (`>=`), no lock file

## Known Constraints & Decisions

1. **Fleet model, not hierarchy.** No CEO agent. Users talk to agents directly. Agents coordinate through blackboard.
2. **Skills over features.** New agent capabilities added as `@skill` decorated functions, not loop changes.
3. **Module-level globals.** `_skill_staging` in skills.py (threading lock protected), `_client` in http_tool.py (connection pooling). Avoid adding more.
4. **Subagent browser concurrency.** Module-level state means subagents shouldn't use browser concurrently.
5. **VNC proxy creates httpx client per request** — acceptable at current usage levels.
6. **`src/shared/types.py` is the contract.** Every cross-component message is a Pydantic model here (335 lines, 24 models).
7. **LLM tool-calling message roles must alternate.** `user → assistant(tool_calls) → tool(result) → assistant`. `_trim_context` merges summary into first user message to preserve this invariant.
8. **busy_timeout variance.** Traces uses 5000ms while other SQLite connections use 30000ms.
9. **Monolithic server files.** `dashboard/server.py` (~3045 lines, 106 endpoints) and `host/server.py` (~1566 lines, ~60 endpoints) are single function-scoped definitions.
10. **`containers.py` backward-compat alias.** Only consumed by E2E tests.

## Git Workflow

- **MANDATORY: Use worktrees for ALL code changes.** Every subagent that touches code MUST use `isolation: "worktree"`. Multiple agents work concurrently — without worktrees they overwrite each other's changes, cause conflicts, and break in-progress work. No exceptions. This applies to the main conversation dispatching subagents too — always set `isolation: "worktree"` on the Agent tool.
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
- 62 test files covering all modules.

### Test File Mapping

| Source | Test file |
|---|---|
| `src/agent/loop.py` | `tests/test_loop.py`, `tests/test_chat.py` |
| `src/agent/memory.py` | `tests/test_memory.py`, `tests/test_memory_integration.py` |
| `src/agent/workspace.py` | `tests/test_workspace.py`, `tests/test_chat_workspace.py` |
| `src/agent/context.py` | `tests/test_context.py` |
| `src/agent/attachments.py` | `tests/test_attachments.py` |
| `src/agent/skills.py` + builtins | `tests/test_skills.py`, `tests/test_builtins.py`, `tests/test_memory_tools.py` |
| `src/agent/builtins/vault_tool.py` | `tests/test_vault.py` |
| `src/agent/builtins/wallet_tool.py` | `tests/test_wallet.py`, `tests/test_wallet_tool.py` |
| `src/agent/builtins/image_gen_tool.py` | `tests/test_image_gen.py` |
| `src/agent/builtins/subagent_tool.py` | `tests/test_subagent.py` |
| `src/agent/builtins/web_search_tool.py` | `tests/test_web_search_tool.py` |
| `src/agent/builtins/coordination_tool.py` | `tests/test_coordination.py` |
| `src/agent/mcp_client.py` | `tests/test_mcp_client.py`, `tests/test_mcp_e2e.py` |
| `src/agent/loop_detector.py` | `tests/test_loop_detector.py` |
| `src/agent/server.py` | `tests/test_agent_server.py` |
| `src/agent/llm.py` | `tests/test_llm_param_allowlist.py` |
| `src/host/mesh.py` | `tests/test_mesh.py` |
| `src/host/credentials.py` | `tests/test_credentials.py` |
| `src/host/runtime.py` | `tests/test_runtime.py` |
| `src/host/transport.py` | `tests/test_transport.py` |
| `src/host/permissions.py` | `tests/test_permissions.py` |
| `src/host/costs.py` | `tests/test_costs.py` |
| `src/host/cron.py` | `tests/test_cron.py` |
| `src/host/health.py` | `tests/test_health.py` |
| `src/host/lanes.py` | `tests/test_lanes.py` |
| `src/host/traces.py` | `tests/test_traces.py` |
| `src/host/transcript.py` | `tests/test_transcript.py` |
| `src/host/failover.py` | `tests/test_failover.py` |
| `src/host/webhooks.py` | `tests/test_webhooks.py` |
| `src/host/watchers.py` | `tests/test_watchers.py` |
| `src/host/wallet.py` | `tests/test_wallet_endpoints.py` |
| `src/host/api_keys.py` | `tests/test_api_keys.py` |
| `src/host/server.py` | `tests/test_dashboard.py` |
| `src/dashboard/server.py` | `tests/test_dashboard.py`, `tests/test_dashboard_workspace.py` |
| `src/dashboard/auth.py` | `tests/test_dashboard_auth.py` |
| `src/dashboard/events.py` | `tests/test_events.py` |
| `src/browser/service.py` | `tests/test_browser_service.py` |
| `src/shared/models.py` | `tests/test_models.py` |
| `src/shared/types.py` | `tests/test_types.py` |
| `src/shared/utils.py` | `tests/test_sanitize.py` |
| `src/templates/` | `tests/test_templates.py` |
| `src/marketplace.py` | `tests/test_marketplace.py` |
| `src/channels/base.py` | `tests/test_channels.py` |
| `src/channels/discord.py` | `tests/test_discord.py` |
| `src/channels/slack.py` | `tests/test_slack.py` |
| `src/channels/whatsapp.py` | `tests/test_whatsapp.py` |
| `src/agent/memory.py` (fallback) | `tests/test_embedding_fallback.py` |
| `src/cli/` | `tests/test_cli_commands.py`, `tests/test_setup_wizard.py` |
| `src/cli/config.py` | `tests/test_projects.py` |
| Cross-component | `tests/test_integration.py` |
| E2E (Docker + API key) | `tests/test_e2e.py`, `tests/test_e2e_chat.py`, `tests/test_e2e_memory.py`, `tests/test_e2e_triggering.py` |

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

### Documentation Alignment Audit — 2026-04-13 (✅ complete)

**Worktree:** `docs/alignment-audit` (off `main`). **Scope:** `docs/*` + `README.md` + `QUICKSTART.md` (13 files, ~3900 lines).

**Method.** Phase 1 spawned 5 parallel subagents to extract factual inventories from the codebase (commands, config/env, lifecycle, security, tools/integrations) → merged as Code Truth at `/tmp/engine-docs-audit/code-truth.md`. Phase 2 spawned 5 more subagents to cross-reference every claim in every doc file against Code Truth. Full discrepancy report at `/tmp/engine-docs-audit/discrepancy-report.md`; per-file findings at `/tmp/engine-docs-audit/phase2/`.

**Totals.** 22 🔴 lies · 25 🟡 stale · 53 🟢 missing · 31 🔵 vague · 7 ❌ broken examples.

**Highest-impact corrections (🔴):**
- **"No external network" for agents is false** (README.md §Trust Zones line 222, docs/architecture.md line 20, docs/security.md line 30). Agents run on a standard Docker bridge with NAT egress; SSRF is enforced at the application layer (`src/host/runtime.py:165-174`).
- **Blackboard tool names wrong** in README.md lines 396–398 and docs/agent-tools.md lines 62–64 (`read_shared_state`/`write_shared_state`/`list_shared_state` do not exist; actual names are `read_blackboard`/`write_blackboard`/`list_blackboard`).
- **All six MCP `@anthropic/mcp-server-*` package names in docs/mcp.md:124–130 are wrong**; real scope is `@modelcontextprotocol/server-*`. Users installing them get package-not-found errors.
- **`TaskAssignment` Pydantic example raises `ValidationError`** (docs/development.md:169–178) — missing required `workflow_id` and `step_id` fields.
- **`allowed_credentials: ["*"]` is NOT the default for new agents** (docs/security.md:73); the Pydantic default is `[]` (deny all).
- **`set_cron` parameters incomplete** — entire tool-mode (direct tool call without LLM) is invisible (docs/agent-tools.md:85 missing `tool_name`/`tool_params`).
- **Fleet template table in README lines 656–666 has five wrong agent rosters** (`deep-research`, `competitive-intel`, `monitor`, `social-listening`, `lead-enrichment`) and two templates missing (`opportunity-finder`, `research`).
- **Memory size caps wrong** — INSTRUCTIONS.md is 12K not 8K (docs/memory.md:81, docs/dashboard.md:102); bootstrap total is 48K not 40K (docs/memory.md:89); FTS5 uses unicode61 not trigram tokenizer (docs/memory.md:143); 90% "emergency hard-prune" threshold doesn't exist as a separate code path.
- **QUICKSTART.md:278** says browser is Chromium; it's Camoufox (stealth Firefox). Build is two images (~1 + ~3 min), not a single "~2 min" build.
- **`OPENLEGION_MAX_PROJECTS=0` disables projects, not "unlimited"** (docs/configuration.md:277). Unlimited is when the var is absent.
- **Heartbeat "starts with scaffold prefix"** (docs/triggering.md:195) — actual check is exact equality `rules.strip() == "# Heartbeat Rules"` (`src/agent/server.py:449`).

**Most serious gaps (🟢):**
- `docs/channels.md` omits `WHATSAPP_APP_SECRET` (production requirement for X-Hub-Signature-256 webhook verification).
- `docs/dashboard.md` has zero coverage of auth/CSRF/SSO flow (`ol_session` cookie, `X-Requested-With` requirement, VNC bearer-token rejection, hosted vs dev mode).
- `coordination_tool.py` (4 tools), `operator_tools.py` (10 tools), and `fleet_tool.py` (2 tools) are entirely undocumented in `docs/agent-tools.md`.
- `docs/configuration.md` missing: `OPENLEGION_SYSTEM_PROXY`, `HTTP_PROXY`/`HTTPS_PROXY`, `OPENLEGION_TOOL_TIMEOUT`, execution-limit env vars, `INTERFACE.md` bootstrap file, `config/settings.json`, `config/network.yaml`.
- Tool-mode cron (direct tool invocation without LLM) entirely absent from `docs/triggering.md`.
- `_UPDATABLE_FIELDS` for cron PUT is `{"schedule", "message", "enabled", "suppress_empty", "tool_name", "tool_params"}` — docs only mention schedule/message.
- Webhook HMAC signature verification (`require_signature=True`, `X-Webhook-Signature` header) absent from `docs/security.md`.
- `docs/development.md`: 7 test files missing from test-file map; `src/host/wallet.py`, `src/host/api_keys.py`, `src/shared/models.py`, `src/cli/proxy.py`, and several builtins missing from project tree; test count "2240" should be ~3121; dev dependency table missing `pytest-cov`, `pytest-xdist`, `websockets`, `pypdf`, `anthropic`, `python-multipart`.
- `docs/dashboard.md` API endpoint table missing ~40+ real endpoints (wallet, network/proxy, external-api-keys, audit, storage, uploads, system-settings, etc.).

**CLAUDE.md staleness found while extracting Code Truth (informational — technically outside scope):**
- Tool modules listed: 14 → 16 exist (`fleet_tool.py`, `operator_tools.py` missing).
- `_RATE_LIMITS`: "13 entries" → 16 (14 static + 2 runtime-added).
- Fleet templates: "11" → 13 (`research.yaml`, `opportunity-finder.yaml` missing).
- Mesh endpoints: "~43" → ~60.
- `sanitize_for_prompt()` call sites: "~60" → 88.
- `_FILE_CAPS` described as 2 entries → actually 7.

**Phase 3 — fixes applied.** 6 parallel subagents edited 13 files: `README.md`, `QUICKSTART.md`, `CLAUDE.md`, and `docs/{agent-tools,architecture,channels,configuration,dashboard,development,mcp,memory,security,triggering}.md`. Priority order applied: 🔴 lies → 🟡 stale → ❌ broken → 🟢 missing → 🔵 vague. Scope expansions: (a) `CLAUDE.md` module table / counts / endpoint stats were refreshed alongside the main scope; (b) operator-only tools (`fleet_tool.py`, `operator_tools.py`) documented in a clearly-labeled section; (c) `skills/README.md` blackboard-tool-name fix applied as collateral cleanup when grep'd up during verification.

**Verification.** Two final-pass subagents ran against the post-fix tree:
1. Spot-check of all 14 highest-impact fixes against current doc text + source code — PASS on every check, zero regressions.
2. Cross-repo sanity check against `app/` and `provisioner/` for SSO and deployment claims — all PASS except one minor `update.sh` mechanism note (engine docs said "updates via `update.sh`" but provisioner's `ssh.py:run_update()` actually runs the equivalent commands inline). Fixed in the §Cross-Repo Integration line above.

**Artifacts (retained in `/tmp/engine-docs-audit/` for reference):**
- `code-truth.md` + `phase1/01..05.md` — factual inventories from reading the code
- `phase2/A..E.md` — per-file discrepancy reports
- `discrepancy-report.md` — consolidated findings

**Worktree state.** `docs/alignment-audit` branch off `main`, 13 tracked files modified + 1 collateral (`skills/README.md`). Ready for commit and PR.
