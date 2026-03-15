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
| `server.py` | Agent FastAPI server (25+ endpoints: `/task`, `/cancel`, `/status`, `/result`, `/capabilities`, `/invoke`, `/chat`, `/chat/steer`, `/chat/stream`, `/chat/reset`, `/chat/history`, `/history`, `/message`, `/workspace`, `/workspace/{filename}`, `/project`, `/workspace-logs`, `/workspace-learnings`, `/heartbeat-context`, `/artifacts`, `/files`, etc.) |
| `llm.py` | LLM client — routes through mesh proxy, never holds keys |
| `context.py` | Context window management (write-then-compact, `_SUMMARIZATION_INPUT_LIMIT=20_000`) |
| `skills.py` | Skill registry and tool discovery |
| `memory.py` | Per-agent SQLite + sqlite-vec + FTS5 memory store |
| `workspace.py` | Persistent markdown workspace (SOUL.md, INSTRUCTIONS.md, MEMORY.md, daily logs, learnings) |
| `mesh_client.py` | Agent-side HTTP client for mesh communication |
| `loop_detector.py` | Tool loop detection with escalating responses |
| `mcp_client.py` | MCP tool server client and lifecycle |
| `attachments.py` | Multimodal attachment enrichment (images → base64 vision blocks, PDFs → text extraction) |
| **`src/agent/builtins/`** | |
| `browser_tool.py` | Browser automation via shared Camoufox service. Includes CAPTCHA detection/solving, screenshot capture with multimodal image blocks. |
| `exec_tool.py` | Shell execution scoped to `/data` (`_MAX_TIMEOUT=300`) |
| `file_tool.py` | File I/O with two-stage path traversal protection (`lstat()` for symlink safety) |
| `http_tool.py` | HTTP requests with CRED handles, SSRF protection, cross-origin auth header stripping |
| `memory_tool.py` | Memory search with hierarchical fallback |
| `mesh_tool.py` | Blackboard (with `sanitize_for_prompt()`), pub/sub, notify_user, list_agents |
| `vault_tool.py` | Credential generation without returning actual values |
| `web_search_tool.py` | DuckDuckGo search (no API key needed) |
| `skill_tool.py` | Self-authoring with AST validation (`_FORBIDDEN_ATTRS` denylist) |
| `subagent_tool.py` | In-process subagents (MAX_DEPTH=2, MAX_CONCURRENT=3, MAX_TTL=600s, DEFAULT_MAX_ITERATIONS=10) |
| `introspect_tool.py` | Runtime state query (permissions, budget, fleet, cron, health) |
| **`src/host/`** | |
| `server.py` | Mesh FastAPI app factory — 31+ endpoints, all permission-checked. VNC reverse proxy with agent token rejection. Localhost validation for `x-mesh-internal`. |
| `mesh.py` | Blackboard (SQLite WAL), PubSub, MessageRouter |
| `orchestrator.py` | DAG workflow executor with safe condition eval (regex, no eval()) |
| `runtime.py` | RuntimeBackend ABC → DockerBackend / SandboxBackend. Agent network, browser container, VNC URL tracking. `_quote_env_value()` for .env escaping. |
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
| `containers.py` | Backward-compat alias for `DockerBackend` |
| **`src/browser/`** | |
| `__main__.py` | Starts KasmVNC (Xvnc), Openbox WM, FastAPI command server |
| `server.py` | Browser service FastAPI app. Auth token warning on startup. |
| `service.py` | BrowserManager with per-agent Camoufox instances. `_MAX_WALK_DEPTH=50` for DOM snapshot. |
| `redaction.py` | Credential redaction for browser output |
| `stealth.py` | Anti-bot fingerprint building (Windows fingerprint, WebRTC kill) |
| `timing.py` | Timing jitter for human-like behavior |
| **`src/channels/`** | |
| `base.py` | Abstract Channel with PairingManager. All messages sanitized. |
| `telegram.py` | Telegram bot adapter (sanitized streaming) |
| `discord.py` | Discord bot adapter (sanitized streaming) |
| `slack.py` | Slack adapter (Socket Mode, sanitized streaming) |
| `whatsapp.py` | WhatsApp Cloud API adapter (`X-Hub-Signature-256` verification) |
| `webhook.py` | HTTP webhook adapter (Bearer token auth with `hmac.compare_digest`) |
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
- **No `eval()`/`exec()` on untrusted input.** Workflow conditions use regex-based safe parser.
- **Permission checks on all mesh endpoints.** Default deny.
- **File path traversal protection.** Two-stage validation (reject `..` before resolution, then walk with symlink resolution via `lstat()`).
- **Container hardening.** Non-root (UID 1000), `no-new-privileges`, 384MB memory, 0.15 CPU, pids_limit.
- **All untrusted text sanitized** via `sanitize_for_prompt()` before reaching LLM context.
- **VNC proxy blocks agent Bearer tokens.** Dashboard auth required (`ol_session` cookie on HTTP and WebSocket).
- **AST validation for skill self-authoring.** `_FORBIDDEN_ATTRS` denylist + forbidden imports/calls.
- **SSRF protection.** DNS pinning + IP blocking including `0.0.0.0` (`ip.is_unspecified`).
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

### Infrastructure
- **Runtime**: Python 3.10+, FastAPI, asyncio
- **Isolation**: Docker containers per agent, bridge network (`openlegion_agents`)
- **Browser**: Shared container with KasmVNC (Xvnc + Openbox), Camoufox
- **Dashboard**: Alpine.js SPA — no React, no build step
- **CI**: GitHub Actions — lint (ruff) + tests (pytest) on Python 3.11 and 3.12

## Known Constraints & Decisions

1. **Fleet model, not hierarchy.** No CEO agent. Users talk to agents directly. Agents coordinate through blackboard and YAML workflows.
2. **Skills over features.** New agent capabilities added as `@skill` decorated functions, not loop changes.
3. **Polling in orchestrator.** `_wait_for_task_result` uses polling instead of push-based notification (known tech debt).
4. **Module-level globals.** `_skill_staging` in skills.py (threading lock protected), `_client` in browser tool (connection pooling). Avoid adding more.
5. **Subagent browser concurrency.** Module-level state means subagents shouldn't use browser concurrently.
6. **VNC proxy creates httpx client per request** — acceptable at current usage levels.
7. **`src/shared/types.py` is the contract.** Every cross-component message is a Pydantic model here.
8. **LLM tool-calling message roles must alternate.** `user → assistant(tool_calls) → tool(result) → assistant`. The `_trim_context` method preserves this invariant.
9. **busy_timeout variance.** Traces uses 5000 while other SQLite connections use 30000.

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
- 56 test files covering all modules.

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
| `src/agent/builtins/subagent_tool.py` | `tests/test_subagent.py` |
| `src/agent/builtins/web_search_tool.py` | `tests/test_web_search_tool.py` |
| `src/agent/mcp_client.py` | `tests/test_mcp_client.py`, `tests/test_mcp_e2e.py` |
| `src/agent/loop_detector.py` | `tests/test_loop_detector.py` |
| `src/agent/server.py` | `tests/test_agent_server.py` |
| `src/agent/llm.py` | `tests/test_llm_param_allowlist.py` |
| `src/host/mesh.py` | `tests/test_mesh.py` |
| `src/host/orchestrator.py` | `tests/test_orchestrator.py` |
| `src/host/credentials.py` | `tests/test_credentials.py` |
| `src/host/runtime.py` | `tests/test_runtime.py` |
| `src/host/transport.py` | `tests/test_transport.py` |
| `src/host/costs.py` | `tests/test_costs.py` |
| `src/host/cron.py` | `tests/test_cron.py` |
| `src/host/health.py` | `tests/test_health.py` |
| `src/host/lanes.py` | `tests/test_lanes.py` |
| `src/host/traces.py` | `tests/test_traces.py` |
| `src/host/transcript.py` | `tests/test_transcript.py` |
| `src/host/failover.py` | `tests/test_failover.py` |
| `src/host/webhooks.py` | `tests/test_webhooks.py` |
| `src/host/watchers.py` | `tests/test_watchers.py` |
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
3. Optionally include `heartbeat_rules`, `permissions`, `budget`, `workflows`
4. Templates auto-discovered by `_load_templates()` in `src/cli/config.py`

## Review State

### Categories & Findings

| Category | Files | Issues Found |
|---|---|---|
| **Error Handling** | loop.py, context.py | 10 issues: uncaught tool failures break LLM role alternation, compaction failures crash sessions, json.dumps on exotic types, missing state reset in streaming |
| **Performance** | workspace.py, skills.py, loop.py, cron.py | 7 issues: bootstrap/learnings re-read from disk every turn, inspect.signature() on every tool call, tool definitions rebuilt per LLM call, goals fetched every chat turn, duplicate blackboard queries in heartbeat |
| **Resilience** | memory.py, context.py | 3 issues: unbounded log table growth, _hard_prune misses assistant-assistant consecutive roles, flush failure aborts compaction |

### Phases

- **Phase 1 (applied):** Top 5 highest-impact fixes across all categories
- **Phase 2 (planned):** Remaining error-handling hardening
- **Phase 3 (planned):** Remaining performance optimizations

### Phase 1 Fixes (applied)

| ID | Category | File(s) | Fix |
|---|---|---|---|
| E1 | Error Handling | loop.py | Tool failure error-fill: when a tool raises, fill remaining tool_call_ids with error results so LLM role alternation is never broken |
| E4 | Error Handling | loop.py, context.py | Wrap compaction in try/except with trim fallback; wrap _flush_to_memory so flush failure doesn't abort compaction |
| P1 | Performance | workspace.py, loop.py | Mtime-cached + pre-sanitized bootstrap and learnings; callers skip redundant sanitize_for_prompt() |
| P2 | Performance | skills.py | Cache inspect.signature() at decoration time; memoize get_tool_definitions() and get_descriptions() |
| P5 | Performance | loop.py, cron.py | TTL-cached goals fetch (300s); cached heartbeat probe entries to eliminate duplicate blackboard queries |
