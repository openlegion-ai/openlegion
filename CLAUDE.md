# CLAUDE.md — engine

## Overview

OpenLegion Engine is a container-isolated multi-agent runtime. LLM-powered agents run in Docker containers, coordinated through a central mesh host. Users interact via CLI REPL, messaging channels (Telegram/Discord/Slack/WhatsApp/Webhook), or a web dashboard.

## Current State

### Architecture (verified 2026-03-24)

#### System Layout

```
User (CLI REPL / Telegram / Discord / Slack / WhatsApp / Webhook)
  → Mesh Host (FastAPI :8420) — routes messages, enforces permissions, proxies APIs, VNC proxy
    → Agent Containers (FastAPI :8400 each) — isolated execution with private memory
    → Browser Service Container (FastAPI :8500 + KasmVNC :6080) — shared Camoufox browser
```

Three trust zones: **User** (full trust), **Mesh** (trusted coordinator), **Agents** (untrusted, sandboxed). All cross-zone communication is HTTP + JSON with Pydantic contracts from `src/shared/types.py`.

#### Entry Points

| Entry Point | File | What It Starts |
|---|---|---|
| `openlegion` CLI | `src/cli/__main__.py` → `src/cli/main.py` | CLI commands: start, stop, status, chat, wallet, version |
| Agent container | `src/agent/__main__.py` → `src/agent/server.py` | Agent FastAPI server on :8400 |
| Browser service | `src/browser/__main__.py` → `src/browser/server.py` | KasmVNC + Openbox + FastAPI on :8500 |
| Mesh host | `src/host/server.py` | FastAPI app on :8420 (started by CLI) |
| Dashboard | `src/dashboard/server.py` | Mounted as router on mesh host |

### Module Map

| Path | Responsibility |
|---|---|
| **`src/shared/`** | |
| `types.py` | Pydantic models — THE cross-component contract (~335 lines, 24 models). `_generate_id()` helper. `AGENT_ID_RE_PATTERN` unified regex. |
| `utils.py` | `sanitize_for_prompt()`, `setup_logging()`, misc helpers |
| `trace.py` | Distributed trace-ID generation and propagation |
| `models.py` | Model cost/context window registry backed by LiteLLM |
| **`src/agent/`** | |
| `loop.py` | Agent execution loop (task mode + chat mode). `MAX_ITERATIONS=20`, `CHAT_MAX_TOOL_ROUNDS=30`, `CHAT_MAX_TOTAL_ROUNDS=200`, `_MAX_SESSION_CONTINUES=5`, `HEARTBEAT_MAX_ITERATIONS=10`. Env-var bounds clamped via `_clamp_env()`. |
| `server.py` | Agent FastAPI server (27 endpoints). `_FILE_CAPS` enforced on workspace writes (HTTP 413). |
| `llm.py` | LLM client — routes through mesh proxy, never holds keys |
| `context.py` | Context window management (write-then-compact, `_SUMMARIZATION_INPUT_LIMIT=20_000`). Empty summary falls back to hard prune. |
| `skills.py` | Skill registry and tool discovery |
| `memory.py` | Per-agent SQLite + sqlite-vec + FTS5 memory store |
| `workspace.py` | Persistent markdown workspace. Bootstrap files: PROJECT.md, SYSTEM.md, INSTRUCTIONS.md, SOUL.md, USER.md, MEMORY.md. Also manages HEARTBEAT.md (loaded separately), daily logs, learnings. |
| `mesh_client.py` | Agent-side HTTP client for mesh communication |
| `loop_detector.py` | Tool loop detection with escalating responses (warn/block/terminate) |
| `mcp_client.py` | MCP tool server client and lifecycle |
| `attachments.py` | Multimodal attachment enrichment (images → base64 vision blocks, PDFs → text extraction) |
| **`src/agent/builtins/`** | |
| `browser_tool.py` | Browser automation via shared Camoufox service. CAPTCHA detection/solving, screenshot capture with multimodal image blocks. |
| `exec_tool.py` | Shell execution scoped to `/data` (`_MAX_TIMEOUT=300`) |
| `file_tool.py` | File I/O with two-stage path traversal protection (`lstat()` for symlink safety) |
| `http_tool.py` | HTTP requests with CRED handles, SSRF protection (DNS pinning, IP blocking incl. CGNAT), cross-origin auth header stripping |
| `memory_tool.py` | Memory search with hierarchical fallback, memory save |
| `mesh_tool.py` | Blackboard (with `sanitize_for_prompt()`), pub/sub, notify_user, list_agents, artifacts, cron, spawn |
| `coordination_tool.py` | Structured multi-agent coordination protocol — hand_off, check_inbox, update_status. Higher-level wrappers over blackboard for inter-agent work handoffs. |
| `vault_tool.py` | Credential generation without returning actual values |
| `web_search_tool.py` | DuckDuckGo search (no API key needed) |
| `image_gen_tool.py` | Image generation via Gemini or OpenAI DALL-E 3, saves output as artifacts |
| `skill_tool.py` | Self-authoring with AST validation (`_FORBIDDEN_ATTRS` denylist, `_FORBIDDEN_IMPORTS` (22 modules) and `_FORBIDDEN_CALLS` (16 functions including type, eval, exec, open)) |
| `subagent_tool.py` | In-process subagents (MAX_DEPTH=2, MAX_CONCURRENT=3, MAX_TTL=600s, DEFAULT_MAX_ITERATIONS=10) |
| `introspect_tool.py` | Runtime state query (permissions, budget, fleet, cron, health) |
| `wallet_tool.py` | Wallet operations — get address, get balance, read contract, transfer, execute (Ethereum + Solana) |
| **`src/host/`** | |
| `server.py` | Mesh FastAPI app factory — 42 endpoints, all permission-checked. Rate limits on state-mutating endpoints (`_RATE_LIMITS` dict at top of file). VNC reverse proxy with agent token rejection. Localhost validation for `x-mesh-internal`. |
| `mesh.py` | Blackboard (SQLite WAL), PubSub, MessageRouter |
| `runtime.py` | RuntimeBackend ABC → DockerBackend / SandboxBackend. Agent network, browser container, VNC URL tracking. Container security config (non-root, cap_drop, memory/CPU limits). |
| `transport.py` | Transport ABC → HttpTransport / SandboxTransport. |
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
| `api_keys.py` | Named API key management — salted SHA-256 hashes stored in `config/api_keys.json`. Raw keys returned once at creation. Falls back to `OPENLEGION_API_KEY` env var. |
| `containers.py` | Backward-compat alias for `DockerBackend` (used by E2E tests) |
| **`src/browser/`** | |
| `__main__.py` | Starts KasmVNC (Xvnc), Openbox WM, FastAPI command server |
| `server.py` | Browser service FastAPI app. Raises `RuntimeError` on startup when auth token missing in production (MESH_AUTH_TOKEN set but BROWSER_AUTH_TOKEN absent); warns only in dev. |
| `service.py` | BrowserManager with per-agent Camoufox instances. `_MAX_WALK_DEPTH=50` for DOM snapshot. Per-agent X11 WID tracking for targeted VNC focus. |
| `redaction.py` | Credential redaction for browser output |
| `stealth.py` | Anti-bot fingerprint building (Windows fingerprint, WebRTC kill, `BROWSER_UA_VERSION` override) |
| `timing.py` | Timing jitter for human-like behavior |
| **`src/channels/`** | |
| `__init__.py` | `AT_MENTION_RE` regex for mention parsing |
| `base.py` | Abstract Channel with PairingManager. All messages sanitized. |
| `telegram.py` | Telegram bot adapter (sanitized streaming) |
| `discord.py` | Discord bot adapter (sanitized streaming) |
| `slack.py` | Slack adapter (Socket Mode, sanitized streaming) |
| `whatsapp.py` | WhatsApp Cloud API adapter (`X-Hub-Signature-256` verification, logs warning when signature verification disabled) |
| **`src/dashboard/`** | |
| `server.py` | Dashboard FastAPI router + API endpoints + VNC URL injection. Alpine.js SPA with `autoescape=True`, CSP headers, CSRF via `X-Requested-With` requirement on state-changing endpoints. |
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
| **`src/templates/`** | 11 YAML fleet templates: starter, content, deep-research, devteam, monitor, sales, competitive-intel, lead-enrichment, price-intelligence, review-ops, social-listening |
| **Other** | |
| `src/setup_wizard.py` | Interactive setup wizard with validation |
| `src/marketplace.py` | Git-based skill marketplace (install/remove, git hooks disabled via `core.hooksPath=/dev/null`) |

### Cross-Repo Integration Points

#### Engine is standalone

The engine has NO direct dependencies on app/ or provisioner/. No imports, no calls, no shared code. Integration is handled by provisioner and app externally:

#### Provisioner → Engine

Provisioner manages engine instances via Docker/systemd on Hetzner VPS:
- Deploys code via `git clone` in cloud-init, updates via `update.sh` (git pull + Docker rebuild)
- Writes `.env` with API keys and config via SSH (base64 encoded to prevent injection)
- Health checks by SSH-ing to localhost and hitting `GET /mesh/agents` with `x-mesh-internal: 1`
- Starts/stops via `systemctl restart openlegion`

#### App → Engine (SSO)

1. App generates HMAC token: `HMAC-SHA256(access_token, "{subdomain}:{expiry}")` → `{expiry}.{signature}`
2. Redirects user to `https://{subdomain}.engine.openlegion.ai/__auth/callback?token=...`
3. Auth gate (deployed via cloud-init) verifies HMAC, sets `ol_session` cookie (one-time-use token replay protection)
4. Caddy reverse proxy uses `forward_auth` to check cookie on every request

#### Exposed Endpoints

- `/mesh/agents` — health check target (provisioner hits via SSH + localhost)
- `/__auth/callback` — SSO callback (app redirects users here, handled by auth gate behind Caddy)
- Dashboard UI on :8420 — user-facing after SSO
- `/vnc/{path}` — reverse proxy to browser container's KasmVNC

### Patterns In Use

#### Naming
- snake_case for files, functions, variables
- PascalCase for classes and Pydantic models
- `setup_logging("component.module")` for loggers
- `@skill` decorator for agent capabilities
- `_UPPER_CASE` for module-level constants

#### Error Handling
- Domain-specific exceptions propagated with context
- Overly broad catches avoided — transient vs permanent distinguished
- `sanitize_for_prompt()` strips invisible Unicode across all input boundaries
- Security errors return generic messages (no leaking internals)

#### Async Patterns
- Async by default (FastAPI + asyncio). Blocking calls wrapped in `run_in_executor`.
- `TYPE_CHECKING` imports for circular dependency prevention

#### Config & Environment
- `.env` file loaded via python-dotenv at CLI startup
- `OPENLEGION_SYSTEM_<PROVIDER>_API_KEY` env vars for LLM provider keys (mesh-only)
- `OPENLEGION_CRED_<NAME>` env vars for agent-tier credentials
- `OPENLEGION_MAX_AGENTS`, `OPENLEGION_MAX_PROJECTS` for plan limits
- `OPENLEGION_LOG_FORMAT=json` for production

#### Logging
- `logger = setup_logging("component.module")` — every module
- JSON format in production, human-readable in dev

#### State
- All SQLite with WAL mode — blackboard, memory, costs, cron, traces. `busy_timeout=30000` for mesh/costs/memory/wallet; `busy_timeout=5000` for traces.
- No Redis, no external databases

#### Security Boundaries

- **Agents never hold API keys.** All LLM/API calls go through mesh credential vault.
- **No `eval()`/`exec()` on untrusted input.**
- **Permission checks on all mesh endpoints.** Default deny.
- **Rate limits on state-mutating mesh endpoints.** API proxy, vault, notify, spawn, cron_create, wallet (read/transfer/execute), image_gen all rate-limited. Defined in `server.py:_RATE_LIMITS`.
- **File path traversal protection.** Two-stage validation (reject `..` before resolution, then walk with symlink resolution via `lstat()`).
- **Container hardening.** Non-root (UID 1000), `no-new-privileges`, `cap_drop=[ALL]`, `read_only=True`, `tmpfs=/tmp` with `noexec,nosuid`, 384MB memory, 0.15 CPU, `pids_limit=256`.
- **All untrusted text sanitized** via `sanitize_for_prompt()` before reaching LLM context (72 call sites across 14 source files including inter-agent messages and daily logs).
- **VNC proxy blocks agent Bearer tokens.** Dashboard auth required (`ol_session` cookie on HTTP and WebSocket).
- **AST validation for skill self-authoring.** `_FORBIDDEN_ATTRS` (10 attributes), `_FORBIDDEN_IMPORTS` (22 modules including os, subprocess, pathlib, io, asyncio, gc, etc.), `_FORBIDDEN_CALLS` (16 functions including type, vars, dir, memoryview, super, eval, exec, open).
- **SSRF protection.** DNS pinning + IP blocking including `0.0.0.0` (`ip.is_unspecified`) and CGNAT (`100.64.0.0/10`).
- **Credential isolation.** Two-tier vault (SYSTEM_*/CRED_*), opaque handles.
- **Bounded execution.** 20 iterations for tasks, 30 tool rounds for chat, 200 total chat rounds, token budgets per task. Env-var bounds clamped with validation.
- **Write-then-compact.** Before discarding context, important facts flush to MEMORY.md. Empty summary falls back to hard prune.
- **CSRF protection.** Dashboard state-changing endpoints require `X-Requested-With` header.
- **Workspace file caps.** `_FILE_CAPS` enforced on workspace writes (HTTP 413).
- **Webhook body size limit.** 1MB with Content-Length pre-check.
- **Wallet seed protection.** Seed reveal endpoint returns HTTP 410 (seed shown once at init).

### Infrastructure

#### Key Dependencies
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

#### Optional Dependency Groups
- `channels` — python-telegram-bot, discord.py, slack-bolt
- `wallet` — web3, eth-account, mnemonic, solders, solana
- `mcp` — mcp (Model Context Protocol)
- `dev` — pytest, pytest-asyncio, pytest-cov, ruff

#### Runtime Infrastructure
- **Runtime**: Python 3.10+, FastAPI, asyncio
- **Isolation**: Docker containers per agent, bridge network (`openlegion_agents`). `Dockerfile.agent` and `Dockerfile.browser` in repo root.
- **Browser**: Shared container with KasmVNC (Xvnc + Openbox), Camoufox, per-agent X11 WID targeting
- **Dashboard**: Alpine.js SPA — no React, no build step
- **CI**: GitHub Actions — lint (ruff) + tests (pytest) on Python 3.11 and 3.12
- **Dependencies**: `pyproject.toml` uses minimum version bounds (`>=`), no lock file

### Known Tech Debt & Constraints

1. **Fleet model, not hierarchy.** No CEO agent. Users talk to agents directly. Agents coordinate through blackboard.
2. **Skills over features.** New agent capabilities added as `@skill` decorated functions, not loop changes.
3. **Module-level globals.** `_skill_staging` in skills.py (threading lock protected), `_client` in http_tool.py (connection pooling). Avoid adding more.
4. **Subagent browser concurrency.** Module-level state means subagents shouldn't use browser concurrently.
5. **VNC proxy creates httpx client per request** — acceptable at current usage levels.
6. **`src/shared/types.py` is the contract.** Every cross-component message is a Pydantic model here (~335 lines, 24 models).
7. **LLM tool-calling message roles must alternate.** `user → assistant(tool_calls) → tool(result) → assistant`. `_trim_context` merges summary into first user message to preserve this invariant.
8. **busy_timeout variance.** Traces uses 5000 while other SQLite connections use 30000.
9. **Monolithic server files.** `dashboard/server.py` (~3040 lines) and `host/server.py` (~1540 lines) are single function-scoped definitions.
10. **`containers.py` backward-compat alias.** Only consumed by E2E tests.

## Git Workflow

- **MANDATORY: Use worktrees for ALL code changes.** Every subagent that touches code MUST use `isolation: "worktree"`. Multiple agents work concurrently — without worktrees they overwrite each other's changes, cause conflicts, and break in-progress work. No exceptions. This applies to the main conversation dispatching subagents too — always set `isolation: "worktree"` on the Agent tool.
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

## Target State

### Architecture Decisions

1. **Monolithic server files.** `dashboard/server.py` (~3040 lines) and `host/server.py` (~1540 lines) are single function-scoped definitions. For production, consider whether these should be split into route groups — but only if there's a concrete maintainability problem, not just size.
2. **API key management.** `api_keys.py` stores hashes in a JSON file (`config/api_keys.json`). Evaluate whether this is sufficient for production or needs migration to the SQLite pattern used elsewhere.
3. **Coordination protocol maturity.** `coordination_tool.py` is new (added in recent commit). Verify it's fully integrated and tested before production load.

### Pattern Standards

- **SQLite WAL + busy_timeout**: Standardize on 30000ms everywhere. The traces.py exception (5000ms) should be justified or aligned.
- **sanitize_for_prompt()**: Applied at 72 call sites across 14 files. Audit should verify no input boundaries are missed.
- **Rate limiting**: Defined in `server.py:_RATE_LIMITS` dict. All state-mutating endpoints should be covered.

### Security Requirements

- All security boundaries documented above are implemented and verified.
- Orphaned `__pycache__` files exist for removed modules (captcha.py, webhook.py channel, orchestrator.py) — should be cleaned.
- Verify no secrets in logs at any log level.
- Verify container escape paths are blocked.

### Current → Target Gaps

- Stale `__pycache__` artifacts from removed modules
- `busy_timeout` inconsistency (traces.py vs everything else)
- Exact sanitize_for_prompt coverage needs boundary-by-boundary audit (not just call count)

## Review State

### Verification Log

**2026-03-24 (initial):** CLAUDE.md verified against codebase.
- Fixed host server.py endpoint count: 37 → 40
- Fixed host server.py line count: ~1420 → ~1540
- Fixed dashboard server.py line count: ~2940 → ~3040
- Fixed _FORBIDDEN_IMPORTS count: 25 → 22
- Fixed sanitize_for_prompt location count: 71 → 56 call sites across 14 files
- Fixed test file count: 63 → 62
- Added missing modules: `src/host/api_keys.py`, `src/agent/builtins/coordination_tool.py`
- Added missing test mappings: `test_api_keys.py`, `test_coordination.py`
- Added CLI `wallet` command group, `src/channels/__init__.py`, _FORBIDDEN_ATTRS count

**2026-03-24 (production review):** Re-verified via automated codebase walk.
- Fixed host server.py endpoint count: 40 → 42
- Fixed sanitize_for_prompt call site count: 56 → 72 (14 files unchanged)
- Fixed `_RATE_LIMITS` location: was documented as `runtime.py`, actually in `server.py`

### Plan

33 review units across 3 repos + cross-repo. Engine units E1-E17, Provisioner P1-P6, App A1-A7, Cross-repo X1-X3.

**Engine units (this repo):**
- E1: Shared Foundation (types.py, utils.py, trace.py, models.py)
- E2: Host Core (mesh.py, runtime.py, transport.py, permissions.py)
- E3: Host Security (credentials.py, api_keys.py, wallet.py) — SECURITY
- E4: Host Services A (costs.py, cron.py, lanes.py, health.py, containers.py)
- E5: Host Services B (failover.py, traces.py, transcript.py, webhooks.py, watchers.py)
- E6: Host Server (server.py) — wires E1-E5
- E7: Agent Foundation (server.py, llm.py, context.py, memory.py, workspace.py, skills.py)
- E8: Agent Support (mesh_client.py, loop_detector.py, mcp_client.py, attachments.py)
- E9: Agent Builtins A (exec_tool.py, file_tool.py, http_tool.py, skill_tool.py) — SECURITY
- E10: Agent Builtins B (mesh_tool.py, coordination_tool.py, memory_tool.py, vault_tool.py)
- E11: Agent Builtins C (browser_tool.py, wallet_tool.py, image_gen_tool.py, web_search_tool.py, subagent_tool.py, introspect_tool.py)
- E12: Agent Loop (loop.py)
- E13: Browser Service (__main__.py, server.py, service.py, redaction.py, stealth.py, timing.py)
- E14: Channels (__init__.py, base.py, telegram.py, discord.py, slack.py, whatsapp.py)
- E15: Dashboard (server.py, events.py, auth.py) — SECURITY
- E16: CLI (__main__.py, main.py, config.py, runtime.py, repl.py, channels.py, formatting.py)
- E17: Misc (marketplace.py, setup_wizard.py, templates/)

### Completed Units (2026-03-24)

All 17 engine units reviewed. 304 total findings across all repos (3 critical, 20 high, 103 medium, 178 low).

**Engine findings: 3 critical, 7 high, 44 medium, 88 low = 142 total**

Critical:
- E1-1: TokenBudget.record_usage() imports src.host.costs (unavailable in agent containers)
- E9-1: `builtins` missing from _FORBIDDEN_IMPORTS → full AST validation bypass
- E9-2: `__dict__` not in _FORBIDDEN_ATTRS → access any builtin by string key

High:
- E2-6: Auth tokens persist after agent stop (no cleanup in stop_agent)
- E3-7: Wallet policy TOCTOU race allows daily limit bypass under concurrency
- E4-2: Budget enforcement post-hoc only; track() never blocks over-budget
- E12-4: Confirms E1-1 cross-zone import
- E14-1: WhatsApp webhook processes messages without signature verification when secret unset
- E15-1: Dashboard POST /api/wallet/init returns BIP-39 seed in response
- E15-2: Dashboard GET /api/credentials/{name}/value returns raw credential values

### Outstanding Cross-References

- E1-1/E12-4: TokenBudget broken in containers → affects budget enforcement across agent loop
- E9-1: Skill self-authoring bypass → affects all agent containers
- E15-2: Dashboard credential exposure → app SSO flow should never pass raw creds client-side
- E2-6: Stale tokens → cross-ref with E6 server endpoint auth
- A3-4/A5-12: Paused subs treated as active → cross-ref with engine SSO auth gate

### Fixes Applied (2026-03-24)

**16 fixes across 3 repos. 0 test regressions (2512 passed, 4 pre-existing failures in test_credentials.py).**

Engine fixes:
- E9-1: Added `"builtins"` to `_FORBIDDEN_IMPORTS` in skill_tool.py (sandbox escape fix)
- E9-3: Added `"__dict__"` to `_FORBIDDEN_ATTRS` in skill_tool.py (defense-in-depth)
- E9-2: Added 6to4 IPv6 SSRF check in http_tool.py `_is_blocked_ip()`
- E9-6: Added Teredo IPv6 SSRF check in http_tool.py `_is_blocked_ip()`
- E1-1: Moved `estimate_cost` from `host/costs.py` to `shared/models.py`, updated import in `types.py`, re-exported from `costs.py` for backward compat
- E2-6: Added `auth_tokens.pop(agent_id, None)` to both `stop_agent()` implementations in runtime.py
- E15-2: Changed `/api/credentials/{name}/value` to return masked values (last 4 chars), removed system-tier access
- E15-1: Added `Cache-Control: no-store` + `Pragma: no-cache` headers to wallet init response
- E14-1: Added production check requiring `WHATSAPP_APP_SECRET` when `MESH_AUTH_TOKEN` is set
- E7-1: Added path traversal check (`resolve` + `is_relative_to`) to `workspace._read_file()`
- E11-3: Added `sanitize_for_prompt()` to web search result titles, URLs, and snippets
- E2-7: Added `env_file.chmod(0o600)` after writing `.agent.env` in `_prepare_workspace()`

App fixes:
- A3-4: Added `ne(subscriptions.status, "paused")` to `getActiveSubscription()` query
- A5-9: Made retry status transition atomic with `UPDATE...WHERE status IN (...) RETURNING`
- X2-1: Expanded app configure route `ALLOWED_KEYS` to match provisioner's 10-provider allowlist

Provisioner fixes:
- P5-1: Added `rate_limit("self-update:global", 1, 300)` to `/self-update` route
