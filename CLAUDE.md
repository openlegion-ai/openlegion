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
| `openlegion` CLI | `src/cli/__main__.py` → `src/cli/main.py` | CLI commands: start, stop, status, chat, version |
| Agent container | `src/agent/__main__.py` → `src/agent/server.py` | Agent FastAPI server on :8400 |
| Browser service | `src/browser/__main__.py` → `src/browser/server.py` | KasmVNC + Openbox + FastAPI on :8500 |
| Mesh host | `src/host/server.py` | FastAPI app on :8420 (started by CLI) |
| Dashboard | `src/dashboard/server.py` | Mounted as router on mesh host |

### Module Map

| Path | Responsibility |
|---|---|
| **`src/shared/`** | |
| `types.py` | Pydantic models — THE cross-component contract. `_generate_id()` helper. |
| `utils.py` | `sanitize_for_prompt()`, `setup_logging()`, misc helpers |
| `trace.py` | Distributed trace-ID generation and propagation |
| `models.py` | Model cost/context window registry backed by LiteLLM |
| **`src/agent/`** | |
| `loop.py` | Agent execution loop (task mode + chat mode). `MAX_ITERATIONS=20`, `CHAT_MAX_TOOL_ROUNDS=30`, `CHAT_MAX_TOTAL_ROUNDS=200`, `_MAX_SESSION_CONTINUES=5`, `HEARTBEAT_MAX_ITERATIONS=10`. |
| `server.py` | Agent FastAPI server (27 endpoints: `/task`, `/cancel`, `/status`, `/result`, `/capabilities`, `/invoke`, `/chat`, `/chat/steer`, `/chat/stream`, `/chat/reset`, `/chat/history`, `/history`, `/message`, `/workspace`, `/workspace/{filename}`, `/project`, `/workspace-logs`, `/workspace-learnings`, `/heartbeat-context`, `/artifacts`, `/files`, etc.) |
| `llm.py` | LLM client — routes through mesh proxy, never holds keys |
| `context.py` | Context window management (write-then-compact, `_SUMMARIZATION_INPUT_LIMIT=20_000`) |
| `skills.py` | Skill registry and tool discovery |
| `memory.py` | Per-agent SQLite + sqlite-vec + FTS5 memory store |
| `workspace.py` | Persistent markdown workspace. Bootstrap files: PROJECT.md, SYSTEM.md, INSTRUCTIONS.md, SOUL.md, USER.md, MEMORY.md. Also manages HEARTBEAT.md (loaded separately), daily logs, learnings. |
| `mesh_client.py` | Agent-side HTTP client for mesh communication |
| `loop_detector.py` | Tool loop detection with escalating responses |
| `mcp_client.py` | MCP tool server client and lifecycle |
| `attachments.py` | Multimodal attachment enrichment (images → base64 vision blocks, PDFs → text extraction) |
| **`src/agent/builtins/`** | |
| `browser_tool.py` | Browser automation via shared Camoufox service. Includes CAPTCHA detection/solving, screenshot capture with multimodal image blocks. |
| `exec_tool.py` | Shell execution scoped to `/data` (`_MAX_TIMEOUT=300`) |
| `file_tool.py` | File I/O with two-stage path traversal protection (`lstat()` for symlink safety) |
| `http_tool.py` | HTTP requests with CRED handles, SSRF protection, cross-origin auth header stripping |
| `memory_tool.py` | Memory search with hierarchical fallback, memory save |
| `mesh_tool.py` | Blackboard (with `sanitize_for_prompt()`), pub/sub, notify_user, list_agents, artifacts, cron, spawn |
| `vault_tool.py` | Credential generation without returning actual values |
| `web_search_tool.py` | DuckDuckGo search (no API key needed) |
| `skill_tool.py` | Self-authoring with AST validation (`_FORBIDDEN_ATTRS` denylist) |
| `subagent_tool.py` | In-process subagents (MAX_DEPTH=2, MAX_CONCURRENT=3, MAX_TTL=600s, DEFAULT_MAX_ITERATIONS=10) |
| `introspect_tool.py` | Runtime state query (permissions, budget, fleet, cron, health) |
| `wallet_tool.py` | Wallet operations — get address, get balance, read contract, transfer, execute (Ethereum + Solana) |
| **`src/host/`** | |
| `server.py` | Mesh FastAPI app factory — 37 endpoints, all permission-checked. VNC reverse proxy with agent token rejection. Localhost validation for `x-mesh-internal`. |
| `mesh.py` | Blackboard (SQLite WAL), PubSub, MessageRouter |
| `runtime.py` | RuntimeBackend ABC → DockerBackend / SandboxBackend. Agent network, browser container, VNC URL tracking. |
| `transport.py` | Transport ABC → HttpTransport / SandboxTransport. `_AGENT_ID_RE` validation. |
| `credentials.py` | Two-tier credential vault (SYSTEM_*/CRED_*) + LLM API proxy. `_convert_messages_to_anthropic()` for OAuth path. |
| `permissions.py` | Per-agent ACL enforcement (glob patterns, deny-all default). `can_spawn`, `can_manage_cron`. |
| `lanes.py` | Per-agent FIFO task queues (followup/steer/collect modes) |
| `health.py` | Health monitor with auto-restart and rate limiting |
| `costs.py` | Per-agent cost tracking + budget enforcement (SQLite) |
| `cron.py` | Persistent cron scheduler with heartbeat probes. `_UPDATABLE_FIELDS` frozenset allowlist. |
| `failover.py` | Model health tracking + failover chains |
| `traces.py` | Request tracing + grouped summaries |
| `transcript.py` | Provider-specific transcript sanitization |
| `webhooks.py` | Named webhook endpoints (payloads sanitized) |
| `watchers.py` | File watcher with polling (messages sanitized) |
| `wallet.py` | WalletService — Ethereum and Solana wallet operations |
| `containers.py` | Backward-compat alias for `DockerBackend` |
| **`src/browser/`** | |
| `__main__.py` | Starts KasmVNC (Xvnc), Openbox WM, FastAPI command server |
| `server.py` | Browser service FastAPI app. Auth token warning on startup. |
| `service.py` | BrowserManager with per-agent Camoufox instances. `_MAX_WALK_DEPTH=50` for DOM snapshot. |
| `redaction.py` | Credential redaction for browser output |
| `stealth.py` | Anti-bot fingerprint building (Windows fingerprint, WebRTC kill, `BROWSER_UA_VERSION` override) |
| `timing.py` | Timing jitter for human-like behavior |
| **`src/channels/`** | |
| `base.py` | Abstract Channel with PairingManager. All messages sanitized. |
| `telegram.py` | Telegram bot adapter (sanitized streaming) |
| `discord.py` | Discord bot adapter (sanitized streaming) |
| `slack.py` | Slack adapter (Socket Mode, sanitized streaming) |
| `whatsapp.py` | WhatsApp Cloud API adapter (`X-Hub-Signature-256` verification) |
| **`src/dashboard/`** | |
| `server.py` | Dashboard FastAPI router + API endpoints + VNC URL injection. Alpine.js SPA with `autoescape=True`, CSP headers, `_verify_dashboard_auth` dependency. |
| `events.py` | EventBus for real-time WebSocket streaming. `threading.Lock` on `emit()`. |
| `auth.py` | Session cookie verification for dashboard access |
| `static/` | JS (app.js, websocket.js), CSS, avatars, favicons |
| `templates/` | Alpine.js SPA template (index.html) |
| **`src/cli/`** | |
| `main.py` | CLI entry point: start, stop, status, chat, version |
| `config.py` | Config loading, Docker helpers, fleet template system (`_load_templates()`, `_create_agent_from_template()`) |
| `runtime.py` | RuntimeContext — lifecycle management. `_RESERVED_AGENT_IDS` validation. |
| `repl.py` | REPLSession — interactive command dispatch |
| `channels.py` | ChannelManager — messaging channel lifecycle |
| `formatting.py` | Tool display, styled output, response rendering |
| **`src/templates/`** | 11 YAML fleet templates: starter, content, deep-research, devteam, monitor, sales, competitive-intel, lead-enrichment, price-intelligence, review-ops, social-listening |
| **Other** | |
| `src/setup_wizard.py` | Interactive setup wizard with validation |
| `src/marketplace.py` | Git-based skill marketplace (install/remove) |

## Cross-Repo Integration

### Engine is standalone

The engine has NO direct dependencies on app/ or provisioner/. No imports, no calls, no shared code. Integration is handled by provisioner and app externally:

### Provisioner → Engine

Provisioner manages engine instances via Docker/systemd on Hetzner VPS:
- Deploys code via `git clone` in cloud-init, updates via `update.sh` (git pull + Docker rebuild)
- Writes `.env` with API keys and config via SSH
- Health checks by SSH-ing to localhost and hitting `GET /mesh/agents` with `x-mesh-internal: 1`
- Starts/stops via `systemctl restart openlegion`

### App → Engine (SSO)

1. App generates HMAC token: `HMAC-SHA256(access_token, "{subdomain}:{expiry}")` → `{expiry}.{signature}`
2. Redirects user to `https://{subdomain}.engine.openlegion.ai/__auth/callback?token=...`
3. Auth gate (deployed via cloud-init) verifies HMAC, sets `ol_session` cookie
4. Caddy reverse proxy uses `forward_auth` to check cookie on every request

### Exposed Endpoints

- `/mesh/agents` — health check target (provisioner hits via SSH + localhost)
- `/__auth/callback` — SSO callback (app redirects users here, handled by auth gate behind Caddy)
- Dashboard UI on :8420 — user-facing after SSO
- `/vnc/{path}` — reverse proxy to browser container's KasmVNC

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
- `OPENLEGION_SYSTEM_<PROVIDER>_API_KEY` env vars for LLM provider keys (mesh-only)
- `OPENLEGION_CRED_<NAME>` env vars for agent-tier credentials
- `OPENLEGION_MAX_AGENTS`, `OPENLEGION_MAX_PROJECTS` for plan limits
- `OPENLEGION_LOG_FORMAT=json` for production

### Logging
- `logger = setup_logging("component.module")` — every module
- JSON format in production, human-readable in dev

### State
- All SQLite with WAL mode — blackboard, memory, costs, cron, traces. `busy_timeout=30000` for mesh/costs/memory; `busy_timeout=5000` for traces.
- No Redis, no external databases

## Security Boundaries

- **Agents never hold API keys.** All LLM/API calls go through mesh credential vault.
- **No `eval()`/`exec()` on untrusted input.**
- **Permission checks on all mesh endpoints.** Default deny.
- **File path traversal protection.** Two-stage validation (reject `..` before resolution, then walk with symlink resolution via `lstat()`).
- **Container hardening.** Non-root (UID 1000), `no-new-privileges`, `cap_drop=[ALL]`, `read_only=True`, `tmpfs=/tmp`, 384MB memory, 0.15 CPU, `pids_limit=256`.
- **All untrusted text sanitized** via `sanitize_for_prompt()` before reaching LLM context.
- **VNC proxy blocks agent Bearer tokens.** Dashboard auth required (`ol_session` cookie on HTTP and WebSocket).
- **AST validation for skill self-authoring.** `_FORBIDDEN_ATTRS` denylist + forbidden imports/calls.
- **SSRF protection.** DNS pinning + IP blocking including `0.0.0.0` (`ip.is_unspecified`) and CGNAT (`100.64.0.0/10`).
- **Credential isolation.** Two-tier vault (SYSTEM_*/CRED_*), opaque handles.
- **Bounded execution.** 20 iterations for tasks, 30 tool rounds for chat, 200 total chat rounds, token budgets per task.
- **Write-then-compact.** Before discarding context, important facts flush to MEMORY.md.

## Dependencies & Infrastructure

### Key Dependencies
- `fastapi` + `uvicorn` — HTTP servers
- `httpx` — async HTTP client
- `pydantic` — data validation and cross-component contracts
- `litellm` — model routing (100+ LLM providers)
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

### Infrastructure
- **Runtime**: Python 3.10+, FastAPI, asyncio
- **Isolation**: Docker containers per agent, bridge network (`openlegion_agents`)
- **Browser**: Shared container with KasmVNC (Xvnc + Openbox), Camoufox
- **Dashboard**: Alpine.js SPA — no React, no build step
- **CI**: GitHub Actions — lint (ruff) + tests (pytest) on Python 3.11 and 3.12

## Known Constraints & Decisions

1. **Fleet model, not hierarchy.** No CEO agent. Users talk to agents directly. Agents coordinate through blackboard.
2. **Skills over features.** New agent capabilities added as `@skill` decorated functions, not loop changes.
3. **Module-level globals.** `_skill_staging` in skills.py (threading lock protected), `_client` in http_tool.py (connection pooling). Avoid adding more.
4. **Subagent browser concurrency.** Module-level state means subagents shouldn't use browser concurrently.
5. **VNC proxy creates httpx client per request** — acceptable at current usage levels.
6. **`src/shared/types.py` is the contract.** Every cross-component message is a Pydantic model here.
7. **LLM tool-calling message roles must alternate.** `user → assistant(tool_calls) → tool(result) → assistant`. The `_trim_context` method preserves this invariant.
8. **busy_timeout variance.** Traces uses 5000 while other SQLite connections use 30000.

## Git Workflow

- **Always use worktrees.** Multiple agents work concurrently. Use `isolation: "worktree"` for subagents.
- **Never `pip install` from a worktree.** Hijacks the global `openlegion` entry point.
- **Never merge directly to main.** Always create a PR via `gh pr create` and merge through GitHub.
- **Wait for CI before merging.** Run `gh pr checks <number> --watch`.
- **Branch naming:** `feat/`, `fix/`, `refactor/`, `docs/`, etc.
- **Commit style:** descriptive subject line, body explains "why". No Co-Authored-By trailers.

## Testing

```bash
# Unit + integration (fast, no Docker needed)
pytest tests/ --ignore=tests/test_e2e.py --ignore=tests/test_e2e_chat.py --ignore=tests/test_e2e_memory.py --ignore=tests/test_e2e_triggering.py -x

# Single test file
pytest tests/test_loop.py -x -v
```

- **Always run tests in a subagent.** Use the Agent tool so the main context window isn't polluted.
- Mock LLM responses, not the loop. See `tests/test_loop.py:_make_loop()`.
- `AsyncMock` for async methods, SQLite in-memory or `tmp_path` for DB paths.
- E2E tests skip gracefully without Docker + API key.
- 59 test files covering all modules.

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
| `src/agent/builtins/subagent_tool.py` | `tests/test_subagent.py` |
| `src/agent/builtins/web_search_tool.py` | `tests/test_web_search_tool.py` |
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

## Target State (Production Readiness Gaps)

### Architecture Decisions Needed

1. **Monolithic server functions (E1).** `dashboard/server.py` (2862 lines) and `host/server.py` (1406 lines) are single function-scoped definitions. Splitting into sub-routers by domain (agents, credentials, wallet, browser, cron) would improve maintainability. This is the highest-impact structural change remaining.
2. **containers.py (A1).** Backward-compat alias for DockerBackend — only consumed by E2E tests. Needs E2E test update to remove.

### Pattern Standards

- **Logger formatting (D1).** 130 f-string logger calls across 26 files mixed with 162 %-style calls. %-style is idiomatic (deferred evaluation), but mechanical migration is high-risk for low value. Decision: standardize in new code, migrate opportunistically.
- **Cross-module shared code.** Several deferred consolidations (B1 redact patterns, B5 provider detection, B8 ID generators, B9 API key validation) require extracting shared code into `src/shared/`. Each is low-risk individually but touching import graphs across trust zones.

### Security Requirements for Production

- **Formal security audit.** SSRF protection, path traversal, credential isolation, and AST validation are implemented but have not been externally audited.
- **Rate limiting completeness.** Mesh endpoints have rate limits but coverage should be verified against all state-mutating endpoints.
- **Dependency pinning.** `pyproject.toml` uses minimum version bounds (`>=`). Production should pin exact versions or use lock file.

### Current → Target Gaps

| Gap | Priority | Effort |
|---|---|---|
| E1 — Split monolithic servers into sub-routers | High | Large |
| Dependency version pinning / lock file | High | Small |
| Deferred consolidations (B1, B5, B8, B9) | Medium | Medium |
| A1 — Remove containers.py alias (needs E2E verification) | Low | Small |
| D1 — Logger format standardization | Low | Medium |
| External security audit | High | External |

## Review State

### Review 1 — Cleanup (completed 2025)

### Phase 1 Fixes (applied)

| ID | Category | File(s) | Fix |
|---|---|---|---|
| E1 | Error Handling | loop.py | Tool failure error-fill: when a tool raises, fill remaining tool_call_ids with error results so LLM role alternation is never broken |
| E4 | Error Handling | loop.py, context.py | Wrap compaction in try/except with trim fallback; wrap _flush_to_memory so flush failure doesn't abort compaction |
| P1 | Performance | workspace.py, loop.py | Mtime-cached + pre-sanitized bootstrap and learnings; callers skip redundant sanitize_for_prompt() |
| P2 | Performance | skills.py | Cache inspect.signature() at decoration time; memoize get_tool_definitions() and get_descriptions() |
| P5 | Performance | loop.py, cron.py | TTL-cached goals fetch (300s); cached heartbeat probe entries to eliminate duplicate blackboard queries |

## Cleanup Inventory (Phase 1 — Read-Only Findings)

**Test Baseline:** 2240 collected, 2212 passed, 0 failed, 28 skipped, 0 errors (105s)
- Skips: 15 Telegram (optional dep), 6 Discord (optional dep), 7 MCP E2E (optional dep)
- Zero test failures. Zero lint errors (ruff clean).

### Category A — Dead Weight

| ID | File & Location | What's Wrong | Proposed Action | Risk |
|---|---|---|---|---|
| A1 | `src/host/containers.py` (entire file, 16 lines) | Backward-compat alias `ContainerManager = DockerBackend`. Only referenced in 4 E2E test files. | Update E2E tests to import `DockerBackend` directly, remove file | Low |
| A2 | `src/agent/loop.py:1480` | `_execute_chat_tool_call()` method defined but never called. Superseded by `_execute_chat_tools_parallel()` (line 1440). | Remove dead method | None |
| A3 | `src/agent/workspace.py:96` | `PROMPT_FILES` class attribute never referenced anywhere. Actual bootstrap uses `_BOOTSTRAP_FILES` (line 186). | Remove dead attribute | None |
| A4 | `src/agent/mesh_client.py:506` | `api_call()` method defined but never called from anywhere (not even tests). Leftover from earlier design. | Remove dead method | None |
| A5 | `src/agent/builtins/http_tool.py:47` | `close_client()` defined but never called — the shared httpx client is never cleaned up on shutdown. | Wire into agent lifespan in `__main__.py`, or remove if not needed | Low |
| A6 | `src/host/costs.py:21` | `_DEFAULT_COST = (0.003, 0.015)` defined but never referenced. The actual fallback is `_DEFAULT_COST` in `shared/models.py:69`. | Remove dead constant | None |
| A7 | `src/host/costs.py:63` | `self._event_bus = event_bus` stored but never read by any method. The constructor accepts `event_bus` param that is never consumed. | Remove parameter and attribute | None |
| A8 | `src/host/transport.py:332` | `resolve_url()` function only used in tests, never in production code. | Remove function, update test | Low |
| A9 | `src/cli/formatting.py:106` | `echo_json()` function defined but never called from anywhere. | Remove dead function | None |

**Notes:** No unused imports found (AST analysis of all 78 files). No commented-out code blocks. No orphan files. No TODO/FIXME/HACK/XXX comments.

### Category B — Redundancy

| ID | File & Location | What's Wrong | Proposed Action | Risk |
|---|---|---|---|---|
| B1 | `src/agent/builtins/browser_tool.py:20-30` + `src/browser/redaction.py:12-22` | `_REDACT_PATTERNS` list (9 compiled regexes) duplicated identically in both files | Move patterns to `src/shared/`, import from both | Low |
| B2 | `src/agent/context.py:196-230` + `:232-276` | `maybe_compact()` and `force_compact()` share identical flush→summarize→log pattern (~30 lines duplicated) | Extract shared `_do_compact()` helper | Low |
| B3 | `src/dashboard/server.py:1755-1820` | Event preview parsing logic (JSON→preview→error handling) duplicated for blackboard and pubsub events | Extract `_parse_event_preview(data)` helper | None |
| B4 | `src/dashboard/server.py:2450-2465` | Credential rollback loop duplicated between `ValueError` and `Exception` handlers | Use `finally` or extract helper | None |
| B5 | `src/host/credentials.py:437` + `src/host/transcript.py:37` | Duplicate provider-detection logic: both map model prefixes to provider names with overlapping tables. transcript.py comment says "mirrors credentials.py" | Extract shared `resolve_provider()` into `src/shared/models.py` | Low |
| B6 | `src/host/mesh.py:339` | `remove_agent_watches()` duplicates the `pattern=None` branch of `remove_watch()` (same 3 lines) | Have `remove_agent_watches` delegate to `remove_watch(agent_id, None)` | None |
| B7 | `src/channels/base.py:203` | Inline `re.match(r"^@(\w+)\s+(.+)$", text, re.DOTALL)` when the same pattern is already compiled as `AT_MENTION_RE` in `channels/__init__.py` | Import and use `AT_MENTION_RE` | None |
| B8 | `src/shared/utils.py:15` + `src/shared/types.py:16` | Near-duplicate ID generators: `generate_id()` and `_generate_id()` with trivially different interfaces | Have `_generate_id` delegate to `generate_id`, or unify | Low |
| B9 | `src/setup_wizard.py:576` + `src/dashboard/server.py:1193` | Near-identical API key validation logic (same litellm call, same error classification, same auth keyword list) | Extract shared `validate_api_key()` function | Low |

### Category C — AI Slop

**Overview:** 113 broad `except Exception` handlers found. 24 silent `pass`, 89 log-only. Most are intentional infrastructure resilience.

| ID | File & Location | What's Wrong | Proposed Action | Risk |
|---|---|---|---|---|
| C1 | `src/dashboard/server.py:1356` | WalletService init failure silently swallowed — error info lost | Log at warning level before `pass` | None |
| C2 | `src/host/credentials.py:548` | `discover_ollama_models()` catches all exceptions silently | Catch `(httpx.HTTPError, OSError, KeyError, ValueError)` specifically | Low |
| C3 | `src/host/wallet.py:894` | CoinGecko price fetch catches all exceptions silently | Catch `(httpx.HTTPError, OSError, KeyError, ValueError)` specifically | Low |
| C4 | `src/agent/skills.py:130,166` | Defensive `getattr(self, "_mcp_client", None)` on attribute that is always set in `__init__` | Replace with `self._mcp_client` | None |
| C5 | `src/agent/skills.py:260,287` | Defensive `getattr(self, "_descriptions_cache", None)` on caches initialized in `__init__` | Replace with direct attribute access | None |

**Not flagged (acceptable patterns):** Cleanup/teardown `pass` handlers, keepalive `pass`, subagent `task.result()` fallback, browser xdotool, health/cron/channel log-only handlers.

### Category D — Inconsistency

| ID | File & Location | What's Wrong | Proposed Action | Risk |
|---|---|---|---|---|
| D1 | 19 files across codebase | Mixed f-string and %-style logger formatting. 130 f-string vs 162 %-style. 19 files use both. | Standardize to %-style (idiomatic, deferred evaluation) | Low |
| D2 | 8 files: `types.py`, `loop.py`, `memory.py`, `mesh_client.py`, `cron.py`, `mesh.py`, `watchers.py`, `webhooks.py` | 42 `Optional[X]` usages where `X \| None` is dominant (278 usages). All have `from __future__ import annotations`. | Replace with `X \| None`, remove `Optional` import | None |
| D3 | `src/agent/attachments.py:26` | Uses `logging.getLogger(__name__)` while every other `src/agent/` file uses `setup_logging("agent.module")`. Won't get JSON formatting in production. | Replace with `setup_logging("agent.attachments")` | None |
| D4 | `src/host/runtime.py:466` | `import json as _json` inside function body when `json` is already imported at module level (line 15) | Remove redundant import, use `json.loads` | None |
| D5 | `src/channels/telegram.py:520,526,552,558` | Uses literal `4096` (API limit) while `MAX_TG_LEN = 4000` constant exists. Two different limits for same platform. | Unify to single constant | None |
| D6 | `src/channels/slack.py:91,121-131` | Logs successful auth/connection at WARNING level while other channels use INFO | Change to `logger.info` | None |

### Category E — Code Quality

| ID | File & Location | What's Wrong | Proposed Action | Risk |
|---|---|---|---|---|
| E1 | `src/dashboard/server.py` (2862 lines) + `src/host/server.py` (1406 lines) | Two monolithic function-scoped API definitions. Dashboard router is ~2400 lines, mesh app is ~1340 lines. | **Deferred** — too large/risky for cleanup scope | Medium |
| E2 | `src/agent/builtins/file_tool.py:205`, `src/agent/server.py:254,393,404,428` | Magic numbers (500, 5000, 8000, 16000) for truncation limits without named constants | Extract to named constants | None |
| E3 | `src/host/cron.py:167` | Magic number 43200 (30 days in minutes) for cron scan range | Define `_MAX_CRON_SCAN_MINUTES = 43200` | None |
| E4 | `src/host/server.py:100-103` | `_MAX_SYSTEM_PROMPT`, `_MAX_BB_KEY_LEN`, `_MAX_BB_VALUE_BYTES` defined inside function body instead of module level | Move to module-level constants | None |

### Summary Statistics

- **78 Python source files**, ~32K lines of code (src/ only)
- **Zero unused imports**, zero orphan files, zero commented-out code
- **Ruff clean** (rules E, F, W, I all passing)
- **33 actionable findings** (9 dead weight, 9 redundancy, 5 AI slop, 6 inconsistency, 4 code quality)
- **1 deferred finding** (E1 — monolithic server functions)

### Cleanup Status

**Applied (30 files modified, ~230 lines removed):**
- **Batch 1 — Dead Weight:** A2-A9 applied. A1 (containers.py) skipped — only used in E2E tests we can't run to verify.
- **Batch 2 — Standardization:** D2-D6 applied. D1 (logger f-string→%-style) skipped — 130 calls across 26 files, high risk of typos for minimal functional benefit.
- **Batch 3 — Consolidation:** B2 (compact helper), B3 (event preview), B4 (rollback), B6 (mesh watch), B7 (AT_MENTION_RE) applied. B1 (redact patterns), B5 (provider detection), B8 (ID generators), B9 (API validation) deferred — cross-module shared code extraction, moderate risk.
- **Batch 4 — AI Slop:** C1-C5 all applied.
- **Batch 5 — Code Quality:** E2-E4 all applied.

**Final test results:** 2236 passed, 28 skipped, 0 failures. Ruff clean.

**Deferred / Needs Human Review:**
- **A1** — `containers.py` alias removal requires E2E test updates we can't verify
- **D1** — Logger f-string→%-style (130 calls, 26 files) — large mechanical change, low value
- **B1** — `_REDACT_PATTERNS` duplication across trust zones (agent vs browser container) — may be intentional defense-in-depth
- **B5** — Provider detection duplication (credentials.py vs transcript.py) — needs shared module extraction
- **B8** — `generate_id()` vs `_generate_id()` — trivially different interfaces, used in Pydantic defaults
- **B9** — API key validation duplication (setup_wizard vs dashboard) — moderate refactor
- **E1** — Monolithic server functions (2862 + 1406 lines) — architectural refactor, out of scope

### Review 2 — Production Readiness (2026-03-17)

**Status:** Stage 3 complete (surgical fixes applied). Tests passing. Ruff clean.

**Stage 3 fixes applied (engine):**
- **U15-1/U14-2/U15-2/U15-4/U11-1**: Added rate limits to `/mesh/api`, `/mesh/api/stream`, `/mesh/notify`, `/mesh/spawn`, `/mesh/vault/store`
- **U14-1**: Webhook body size limit (1MB) with Content-Length pre-check
- **U14-4**: Webhook response no longer includes full agent reply
- **U16-2**: Browser service fails startup when auth token missing in production
- **U17-1**: WhatsApp webhook logs warning when signature verification disabled
- **U19-2**: CSRF protection via `X-Requested-With` header requirement on all state-changing dashboard endpoints
- **U19-3**: Wallet seed reveal endpoint disabled (seed shown once at init)
- **U8-1+U8-2**: Skill self-authoring forbidden imports expanded (pathlib, io, pty, gc, asyncio, etc.) and `type()` blocked
- **U18-1**: Git clone hooks disabled via `core.hooksPath=/dev/null`
- **U18-3**: Marketplace AST validation now includes `__init__.py` files
- **U5-1**: `_trim_context` merges summary into first user message (fixes role alternation)
- **U2-7**: Empty LLM summary now falls back to hard prune
- **U9-7**: Inter-agent messages sanitized via `sanitize_for_prompt()` before daily log
- **U9-5**: `_FILE_CAPS` enforced on workspace writes (HTTP 413)
- **U5-2**: Env-var iteration bounds clamped with validation
- **U12-1**: Agent ID regex unified to `AGENT_ID_RE_PATTERN` in `src/shared/types.py`

**Stage 3 fixes applied (app):**
- **U20-1 (CRITICAL)**: `getInstance()` now uses explicit column selection excluding `sshPrivateKey`, `hostKey`, `accessToken`, `proxyConfig`

**Findings summary (engine only):**
- 0 critical, 6 high, ~26 medium, 60+ low
- **High**: skill_tool `type()` bypass (U8-1), webhook no body size limit (U14-1), wallet stale prices (U14-8), LLM proxy no rate limit (U15-1), browser auth silently disabled (U16-2), WhatsApp webhook spoofable (U17-1)
- **Key medium**: no auth on agent server (U9-1), no CSRF on dashboard (U19-2), `_trim_context` role alternation violation (U5-1), unsynchronized SQLite writes in memory.py (U3-1), empty summary causes context loss (U2-7), wallet approve bypass (U8-9)

**Cross-repo findings:**
- Auth SSO chain verified solid across all 3 repos
- Provisioner configure/resize claims lack `user_id` in WHERE clause (U26-2/U26-3)
- UFW rule `172.16.0.0/12` broader than needed, could allow private network access to mesh (U27-6)
