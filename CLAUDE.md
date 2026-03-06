# CLAUDE.md — engine

## Current State

### Architecture (verified 2026-03-06)

OpenLegion is a container-isolated multi-agent runtime. LLM-powered agents run in Docker containers (or Sandbox microVMs), coordinated through a central mesh host. Fleet model — no CEO agent. Users talk to agents directly; agents coordinate via shared blackboard and YAML workflows.

```
User (CLI REPL / Telegram / Discord / Slack / WhatsApp / Webhook)
  → Mesh Host (FastAPI :8420) — routes messages, enforces permissions, proxies APIs, VNC proxy
    → Agent Containers (FastAPI :8400 each) — isolated execution with private memory
    → Browser Service Container (FastAPI :8421 + KasmVNC :6080) — shared Camoufox browser instances with live VNC viewing
```

Three trust zones: **User** (full trust), **Mesh** (trusted coordinator), **Agents** (untrusted, sandboxed). Everything between zones is HTTP + JSON with Pydantic contracts defined in `src/shared/types.py`.

### Module Map

| Path | What it does |
|---|---|
| **Shared** | |
| `src/shared/types.py` | THE contract — all Pydantic models shared between host and agents. `_generate_id()` helper for UUID generation. |
| `src/shared/utils.py` | Utilities: `sanitize_for_prompt()`, `setup_logging()`, misc helpers |
| `src/shared/trace.py` | Distributed trace-ID generation and propagation |
| `src/shared/models.py` | Model cost/context window registry backed by LiteLLM |
| **Agent Container** | |
| `src/agent/loop.py` | Agent execution loop (task mode + chat mode) |
| `src/agent/skills.py` | Skill registry and tool discovery |
| `src/agent/server.py` | Agent container FastAPI server (`/task`, `/chat`, `/status`, `/cancel`) |
| `src/agent/llm.py` | LLM client (routes through mesh proxy, never holds keys) |
| `src/agent/mesh_client.py` | Agent-side HTTP client for mesh communication |
| `src/agent/memory.py` | Per-agent SQLite + sqlite-vec + FTS5 memory store |
| `src/agent/workspace.py` | Persistent markdown workspace (SOUL.md, INSTRUCTIONS.md, MEMORY.md, daily logs, learnings) |
| `src/agent/context.py` | Context window management (write-then-compact, `_SUMMARIZATION_INPUT_LIMIT = 20_000`) |
| `src/agent/loop_detector.py` | Tool loop detection with escalating responses |
| `src/agent/mcp_client.py` | MCP tool server client and lifecycle |
| `src/agent/__main__.py` | Agent container entry point |
| **Agent Builtins** | |
| `src/agent/builtins/exec_tool.py` | Shell execution scoped to `/data` (`_MAX_TIMEOUT = 300`) |
| `src/agent/builtins/file_tool.py` | File I/O with two-stage path traversal protection (uses `lstat()` to prevent symlink info leak) |
| `src/agent/builtins/http_tool.py` | HTTP requests with CRED handles, DNS rebinding/SSRF protection (blocks `0.0.0.0`), cross-origin auth header stripping |
| `src/agent/builtins/browser_tool.py` | Browser automation via shared Camoufox service container |
| `src/agent/builtins/captcha.py` | CAPTCHA detection and solving via 2Captcha / CapSolver APIs (reCAPTCHA v2/v3/Enterprise, hCaptcha, Turnstile) |
| `src/agent/builtins/memory_tool.py` | Memory search with hierarchical fallback |
| `src/agent/builtins/mesh_tool.py` | Blackboard (with `sanitize_for_prompt()`), pub/sub, notify_user, list_agents |
| `src/agent/builtins/vault_tool.py` | Credential generation without returning actual values |
| `src/agent/builtins/web_search_tool.py` | DuckDuckGo search (no API key needed) |
| `src/agent/builtins/skill_tool.py` | Self-authoring with AST validation (`_FORBIDDEN_ATTRS` denylist, forbidden imports/calls blocked) |
| `src/agent/builtins/subagent_tool.py` | In-process subagents (MAX_DEPTH=2, MAX_CONCURRENT=3, DEFAULT_TTL=300s) |
| `src/agent/builtins/introspect_tool.py` | Runtime state query (permissions, budget, fleet, cron, health) |
| **Host/Mesh** | |
| `src/host/server.py` | Mesh FastAPI app factory — all endpoints enforce permissions. Includes VNC reverse proxy (`/vnc/{path}`) with agent token rejection. |
| `src/host/mesh.py` | Blackboard (SQLite WAL), PubSub, MessageRouter |
| `src/host/orchestrator.py` | DAG workflow executor with safe condition eval (regex, no eval()) |
| `src/host/runtime.py` | RuntimeBackend ABC → DockerBackend / SandboxBackend. DockerBackend manages agent network (`openlegion_agents` bridge), browser container (with KasmVNC + plan-based resource scaling), VNC URL tracking. |
| `src/host/transport.py` | Transport ABC → HttpTransport / SandboxTransport. `_AGENT_ID_RE` validation. |
| `src/host/credentials.py` | Two-tier credential vault (SYSTEM_*/CRED_*) + LLM API proxy |
| `src/host/permissions.py` | Per-agent ACL enforcement (glob patterns, deny-all default). Includes `can_spawn`, `can_manage_cron`. |
| `src/host/lanes.py` | Per-agent FIFO task queues (followup/steer/collect modes) |
| `src/host/health.py` | Health monitor with auto-restart and rate limiting |
| `src/host/costs.py` | Per-agent cost tracking + budget enforcement (SQLite) |
| `src/host/cron.py` | Persistent cron scheduler with heartbeat probes |
| `src/host/failover.py` | Model health tracking + failover chains |
| `src/host/traces.py` | Request tracing + grouped summaries |
| `src/host/transcript.py` | Provider-specific transcript sanitization |
| `src/host/webhooks.py` | Named webhook endpoints (payloads sanitized) |
| `src/host/watchers.py` | File watcher with polling (messages sanitized) |
| `src/host/containers.py` | Backward-compat alias for `DockerBackend` |
| **Browser Service** | |
| `src/browser/__main__.py` | Browser service entry point — starts KasmVNC (Xvnc), Openbox WM, and FastAPI command server |
| `src/browser/server.py` | Browser service FastAPI app (note: `/browser/{agent_id}/evaluate` removed for security) |
| `src/browser/service.py` | BrowserManager with per-agent Camoufox instances |
| `src/browser/redaction.py` | Credential redaction for browser output |
| `src/browser/stealth.py` | Anti-bot fingerprint building |
| `src/browser/timing.py` | Timing jitter for human-like behavior |
| **Channels** | |
| `src/channels/base.py` | Abstract Channel with PairingManager for unified REPL-like UX. All channel messages sanitized via `sanitize_for_prompt()`. |
| `src/channels/telegram.py` | Telegram bot channel adapter |
| `src/channels/discord.py` | Discord bot channel adapter |
| `src/channels/slack.py` | Slack channel adapter (Socket Mode) |
| `src/channels/whatsapp.py` | WhatsApp Cloud API channel adapter (`X-Hub-Signature-256` verification) |
| `src/channels/webhook.py` | HTTP webhook channel adapter (Bearer token auth with `hmac.compare_digest`) |
| **Dashboard** | |
| `src/dashboard/server.py` | Dashboard FastAPI router + API endpoints + VNC URL injection for browser viewing |
| `src/dashboard/events.py` | EventBus for real-time WebSocket streaming |
| `src/dashboard/auth.py` | Session cookie verification for dashboard access |
| `src/dashboard/templates/index.html` | Alpine.js SPA template |
| `src/dashboard/static/` | JS (app.js, websocket.js), CSS (dashboard.css), avatars, favicons |
| **CLI** | |
| `src/cli/main.py` | CLI entry point: start, stop, status, chat, version |
| `src/cli/config.py` | Config loading, Docker helpers, agent management, fleet template system (`_load_templates()`, `_apply_template()`, `_create_agent_from_template()`, `_load_skill_templates()`) |
| `src/cli/runtime.py` | RuntimeContext — full lifecycle management. `_RESERVED_AGENT_IDS` validation. |
| `src/cli/repl.py` | REPLSession — interactive command dispatch (includes template-based agent creation) |
| `src/cli/channels.py` | ChannelManager — messaging channel lifecycle |
| `src/cli/formatting.py` | Tool display, styled output, response rendering |
| **Templates** | |
| `src/templates/` | YAML fleet templates defining agent configurations, souls, heartbeats, permissions, budgets, and workflows. Templates: `starter`, `content`, `deep-research`, `devteam`, `monitor`, `sales`. |
| **Other** | |
| `src/setup_wizard.py` | Interactive setup wizard with validation |
| `src/marketplace.py` | Git-based skill marketplace (install/remove) |

### Cross-Repo Integration Points

**The engine is a standalone runtime with NO direct dependencies on app/ or provisioner/.** No imports, no calls, no shared code.

Integration happens externally:

1. **Provisioner → Engine**: Provisioner manages engine instances via Docker/systemd on Hetzner VPS. Provisioner deploys code, writes `.env`, starts services, checks health via SSH to `/mesh/agents`. Provisioner also forwards proxy env vars for browser anti-detection.

2. **App → Engine**: App generates HMAC tokens for SSO. Engine's auth gate (deployed via cloud-init) verifies HMAC tokens at `/__auth/callback?token={expiry}.{signature}` using the shared `access_token`.

3. **Mesh endpoints exposed** (that external systems interact with):
   - `/mesh/agents` — health check target (provisioner hits this via SSH)
   - `/__auth/callback` — SSO callback (app redirects users here)
   - Dashboard UI on port 8420 — user-facing after SSO
   - `/vnc/{path}` — reverse proxy to browser container's KasmVNC (accessed through dashboard)

### Patterns In Use

- **Pydantic for boundaries, plain dicts internally.** Cross-component messages use Pydantic models from `types.py`. Internal data flow within a module can use dicts.
- **Async by default.** Agent-side code is async (FastAPI + asyncio). Blocking calls wrapped in `run_in_executor`.
- **SQLite for all state.** Blackboard, agent memory, cost tracking, cron, traces — all SQLite with WAL mode and `busy_timeout=5000`.
- **`TYPE_CHECKING` imports** for circular dependency prevention (28 instances across codebase).
- **`setup_logging(name)`** — every module creates `logger = setup_logging("component.module")`.
- **Small, focused modules** — each file has a single responsibility.
- **Skills over features** — new agent capabilities added as `@skill` decorated functions, not loop changes.
- **YAML templates for fleet configuration** — agent definitions, permissions, budgets, and workflows defined declaratively in `src/templates/*.yaml`.

### Known Tech Debt

1. **Polling in orchestrator**: `_wait_for_task_result` uses polling instead of push-based notification.
2. **Module-level globals**: `_skill_staging` in `skills.py` (protected by threading lock), `_client` in browser tool (for connection pooling). Avoid adding more.
3. **Rate limiting**: Per-endpoint rate limiting via in-memory deque — not persistent across restarts.
4. **Subagent browser concurrency**: Module-level state means subagents shouldn't use browser concurrently.
5. **VNC proxy creates httpx client per request** in `host/server.py` — should pool if VNC usage becomes heavy.

### Infrastructure

- **Runtime**: Python 3.11+, FastAPI, asyncio
- **State**: All SQLite (WAL mode) — no Redis, no external databases
- **Isolation**: Docker containers (or Sandbox microVMs) per agent, bridge network (`openlegion_agents`)
- **Container hardening**: non-root (UID 1000), `no-new-privileges`, 384MB memory, 0.15 CPU (plan-scalable for browser container)
- **Browser service**: Shared container with KasmVNC (Xvnc + Openbox), Camoufox. Resources scaled by plan tier.
- **Dashboard**: Alpine.js SPA — no React, no build step
- **Dependencies**: Minimal — no LangChain, no Kubernetes
- **CI**: GitHub Actions (lint + tests on multiple Python versions)

## Target State

### Architecture Decisions

- Engine architecture is sound. No structural changes needed for production.
- The polling in orchestrator should eventually be replaced with push-based notification.
- Browser service architecture (shared container with KasmVNC) is correct for resource efficiency + debuggability.

### Pattern Standards

- All modules follow the same patterns consistently. No standardization needed.
- `setup_logging()`, `TYPE_CHECKING` imports, SQLite WAL — all uniformly applied.

### Security Requirements

All security boundaries are already enforced:
- Trust zones (User/Mesh/Agent) with permission checks on all 26+ mesh endpoints
- `sanitize_for_prompt()` at 35+ choke points for prompt injection prevention
- File path traversal protection (two-stage validation)
- SSRF protection (DNS pinning + IP blocking including `0.0.0.0`)
- Credential isolation (two-tier vault, opaque handles)
- Execution limits (iteration caps, token budgets, subagent depth/concurrency)
- Safe condition evaluation (regex parser, no eval)
- VNC proxy blocks agent Bearer tokens (prevents untrusted agent access)
- AST validation for skill self-authoring (forbidden attrs + imports)
- Webhook channel requires Bearer token auth
- WhatsApp verifies `X-Hub-Signature-256`

### Current → Target Gaps

- No critical gaps. Engine is production-ready.
- Minor: VNC proxy httpx client pooling (only matters under heavy usage).

## Git Workflow

- **Always use worktrees.** Multiple agents work on this codebase concurrently. Use `isolation: "worktree"` for subagents or `EnterWorktree` for interactive sessions. Never make changes directly on `main`.
- **Never `pip install` from a worktree.** It hijacks the global `openlegion` entry point. A runtime guard in `src/cli/__init__.py` catches this.
- **Never merge directly to main.** Always create a PR via `gh pr create` and merge through GitHub.
- **Wait for CI before merging.** Run `gh pr checks <number> --watch` and wait for all checks to pass.
- **Branch naming:** `feat/`, `fix/`, `refactor/`, `docs/`, etc.
- **Commit style:** descriptive subject line, body explains "why". No Co-Authored-By trailers.

## Non-Negotiable Rules

### Security boundaries — never violate these

1. **Agents never hold API keys.** All LLM/API calls go through the mesh credential vault (`src/host/credentials.py`). An agent's `LLMClient` posts to `/mesh/api` — the vault injects credentials server-side.

2. **No `eval()`, no `exec()` on untrusted input.** Workflow conditions use a regex-based safe parser (`src/host/orchestrator.py:_safe_evaluate_condition`).

3. **Permission checks before every cross-boundary operation.** Blackboard reads/writes, pub/sub, message routing, API proxy calls, spawn, and cron management all check the `PermissionMatrix` first. Default policy is deny.

4. **Path traversal protection in agent file tools.** Agent file operations confined to `/data` with two-stage validation (reject `..` before resolution, then walk component-by-component with symlink resolution).

5. **Container hardening is not optional.** Non-root (UID 1000), `no-new-privileges`, memory limits (384MB), CPU quotas (0.15 CPU).

6. **All untrusted text is sanitized before reaching LLM context.** `sanitize_for_prompt()` strips invisible Unicode at 35+ choke points: user input, tool results, workspace loading, mesh tools, dashboard input, channel messages, webhook payloads, file watcher messages, blackboard reads.

### Architectural invariants

7. **`src/shared/types.py` is the contract.** Every message and state object crossing component boundaries is a Pydantic model defined here.

8. **Fleet model, not hierarchy.** No CEO agent. Users talk to agents directly. Agents coordinate through blackboard and YAML workflows.

9. **Bounded execution.** 20 iterations for tasks (`MAX_ITERATIONS`), 30 tool rounds for chat (`CHAT_MAX_TOOL_ROUNDS`). Token budgets enforced per task.

10. **Write-then-compact.** Before discarding context, important facts flush to `MEMORY.md` via the workspace.

11. **LLM tool-calling message roles must alternate correctly.** `user → assistant(tool_calls) → tool(result) → assistant`. The `_trim_context` method groups them to preserve this invariant.

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
5. Add tests in `tests/test_mesh.py` or `tests/test_integration.py`

### Adding a new channel

1. Subclass `Channel` from `src/channels/base.py`
2. Implement `start()`, `stop()`, `send_notification()`
3. Message handling provided by base class (`handle_message`)
4. Add startup logic in `src/cli/channels.py:ChannelManager`
5. Add tests in `tests/test_channels.py`

### Adding a new fleet template

1. Create `src/templates/your_template.yaml`
2. Define agents with `role`, `model`, `instructions`, `soul`, `resources`
3. Optionally include `heartbeat_rules`, `permissions`, `budget`, `workflows`
4. Templates are auto-discovered by `_load_templates()` in `src/cli/config.py`
5. Add tests in `tests/test_templates.py`

## Testing

```bash
# Unit + integration (fast, no Docker needed)
pytest tests/ --ignore=tests/test_e2e.py --ignore=tests/test_e2e_chat.py --ignore=tests/test_e2e_memory.py --ignore=tests/test_e2e_triggering.py -x

# All tests including E2E (requires Docker + API key)
pytest tests/ -x

# Single test file
pytest tests/test_loop.py -x -v
```

### Conventions

- **Always run tests in a subagent.** Use the Agent tool to run `pytest` so the main context window isn't polluted with test output. This applies to both running existing tests and verifying new ones.
- Mock LLM responses, not the loop. See `tests/test_loop.py:_make_loop()`.
- Use `AsyncMock` for async methods.
- SQLite in-memory or `tmp_path` for database paths.
- E2E tests are optional (Docker + API key required, skip gracefully).
- Every new feature gets tests matching the corresponding test file.

### Test file mapping

| Source | Test file |
|---|---|
| `src/agent/loop.py` | `tests/test_loop.py`, `tests/test_chat.py` |
| `src/agent/memory.py` | `tests/test_memory.py`, `tests/test_memory_integration.py` |
| `src/agent/workspace.py` | `tests/test_workspace.py`, `tests/test_chat_workspace.py` |
| `src/agent/context.py` | `tests/test_context.py` |
| `src/agent/skills.py` + builtins | `tests/test_skills.py`, `tests/test_builtins.py`, `tests/test_memory_tools.py` |
| `src/agent/builtins/vault_tool.py` | `tests/test_vault.py` |
| `src/agent/builtins/subagent_tool.py` | `tests/test_subagent.py` |
| `src/agent/mcp_client.py` | `tests/test_mcp_client.py`, `tests/test_mcp_e2e.py` |
| `src/agent/loop_detector.py` | `tests/test_loop_detector.py` |
| `src/agent/server.py` | `tests/test_agent_server.py` |
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
| `src/host/server.py` (VNC proxy) | `tests/test_dashboard.py` |
| `src/dashboard/server.py` | `tests/test_dashboard.py`, `tests/test_dashboard_workspace.py` |
| `src/dashboard/auth.py` | `tests/test_dashboard_auth.py` |
| `src/browser/service.py` | `tests/test_browser_service.py` |
| `src/shared/models.py` | `tests/test_models.py` |
| `src/templates/` | `tests/test_templates.py` |
| `src/marketplace.py` | `tests/test_marketplace.py` |
| `src/channels/base.py` | `tests/test_channels.py` |
| `src/channels/discord.py` | `tests/test_discord.py` |
| `src/channels/slack.py` | `tests/test_slack.py` |
| `src/channels/whatsapp.py` | `tests/test_whatsapp.py` |
| `src/shared/types.py` | `tests/test_types.py` |
| `src/shared/utils.py` | `tests/test_sanitize.py` |
| `src/cli/` | `tests/test_cli_commands.py`, `tests/test_setup_wizard.py` |
| `src/cli/config.py` | `tests/test_projects.py` |
| `src/dashboard/events.py` | `tests/test_events.py` |
| Cross-component | `tests/test_integration.py` |
| Test fixtures | `tests/fixtures/echo_mcp_server.py` |

## Common Mistakes to Avoid

- **Running `pip install` from a worktree.** Overwrites the user's entry point.
- **Creating httpx clients per request.** Reuse clients with connection pooling.
- **Polling for task completion.** Prefer push-based patterns.
- **Breaking tool-call message grouping.** Never separate assistant `tool_calls` from `tool` results.
- **Putting secrets in agent code.** All keys belong in the vault (SYSTEM_*/CRED_*).
- **Using global mutable state.** Avoid new module-level mutable globals. Pass state through constructors.
- **Overly broad exception handling.** Log errors. Distinguish transient from permanent.
- **Monolithic functions.** Extract classes with clear lifecycle (init, start, stop).

## Design Philosophy

- **Act first, ask never.** Agents are autonomous executors.
- **The mesh is the only door.** Agents have no external network access except through the mesh.
- **Private by default, shared by promotion.** Agents keep knowledge in private memory.
- **Skills over features.** New capabilities as `@skill` functions, not loop changes.
- **Smallest thing that works.** No LangChain, no Redis, no Kubernetes.

## Review State

### Plan
Stage 3 complete (2026-03-06). Two full review passes completed. All CRITICAL and HIGH findings fixed. Test suite: 1779 passed, 0 regressions, 28 skipped, 2 pre-existing dashboard test failures (unrelated).

### Completed Units
- Units 1-16 (all engine units) — prior review as of commit `d840e4a`
- Units E1-E6 (2026-03-06 review): credentials/OAuth, workspace, loop, server, cron, subagent, mesh_tool, host/server, host/runtime, templates

### Fixes Applied (Stage 3, 2026-03-05 — prior review)

**Batch 1 — Dead Weight:**
- Deleted `src/shared/constants.py` (zero imports)
- Removed redundant SQLite index on `entries(key)` in `src/host/mesh.py` (M18)

**Batch 2 — Standardization:**
- Extracted `_SUMMARIZATION_INPUT_LIMIT = 20_000` constant in `src/agent/context.py` (M27)

**Batch 3 — Consolidation:**
- Added `_generate_id(prefix, length)` helper in `src/shared/types.py` replacing duplicated UUID lambdas (L1)
- Added `can_spawn` and `can_manage_cron` fields to `AgentPermissions` in `src/shared/types.py`

**Batch 5a — Security (Engine):**
- C1: Added `_FORBIDDEN_ATTRS` denylist + `ast.Attribute` check in `src/agent/builtins/skill_tool.py`
- C2: Removed `/browser/{agent_id}/evaluate` endpoint from `src/browser/server.py`
- C3: Added Bearer token auth with `hmac.compare_digest` to `src/channels/webhook.py`
- H1/M1: Added `permissions.can_spawn()` and `permissions.can_manage_cron()` checks in `src/host/server.py`
- H3: Added `sanitize_for_prompt()` to blackboard reads in `src/agent/builtins/mesh_tool.py`
- H6: Added cross-origin Authorization header stripping on redirects in `src/agent/builtins/http_tool.py`
- H7: Added `sanitize_for_prompt()` to webhook payloads in `src/host/webhooks.py`
- H8: Added `sanitize_for_prompt()` to file watcher messages in `src/host/watchers.py`
- H9: Added `sanitize_for_prompt()` to channel messages in `src/channels/base.py`
- H10: Added `X-Hub-Signature-256` verification in `src/channels/whatsapp.py`
- M5: Verified `/project` already has `x-mesh-internal` check (no change needed)
- M8: Added `_AGENT_ID_RE` validation in `src/host/transport.py`
- M19: Added `_RESERVED_AGENT_IDS` and `_validate_agent_id()` in `src/host/server.py` and `src/cli/runtime.py`
- M31: Added `ip.is_unspecified` to SSRF check in `src/agent/builtins/http_tool.py`
- M32: Replaced sequential `str.replace()` with `re.sub()` callback in `src/agent/builtins/http_tool.py`
- M33: Added `_MAX_TIMEOUT = 300` and timeout capping in `src/agent/builtins/exec_tool.py`
- M34: Changed `item.stat()` to `item.lstat()` in `src/agent/builtins/file_tool.py`

### Fixes Applied (Stage 3, 2026-03-06 — current review)

**Batch 1 — Dead Weight:**
- Removed dead `load_prompt_context()` method from `src/agent/workspace.py`
- Removed 4 dead tests for `load_prompt_context` from `tests/test_workspace.py`

**Batch 2 — Standardization:**
- Extracted `_FLEET_ROSTER_TTL`, `_FALLBACK_MAX_TOKENS`, `_TOOL_HISTORY_LIMIT` constants in `src/agent/loop.py`
- Added `days` parameter clamping (`max(1, min(days, 14))`) to `/history` endpoint in `src/agent/server.py`
- Added `_UPDATABLE_FIELDS` frozenset allowlist to `update_job` in `src/host/cron.py`
- Added `MAX_TTL = 600` constant and TTL clamping in `src/agent/builtins/subagent_tool.py`
- Added `monthly_usd` budget to all 6 templates in `src/templates/`

**Batch 4 — Bug Fixes:**
- **CRITICAL**: Added `_convert_messages_to_anthropic()` in `src/host/credentials.py` — converts OpenAI-format tool messages to Anthropic Messages API format for OAuth path. Integrated into `_build_anthropic_body`.
- Added 2 tests in `tests/test_credentials.py` for Anthropic message conversion

**Batch 5 — Security:**
- H2: Added `_sanitize_value(result)` to `read_agent_history` return in `src/agent/builtins/mesh_tool.py`
- M8: Added `sanitize_for_prompt()` to `list_shared_state` key field in `src/agent/builtins/mesh_tool.py`
- M9: Added localhost validation for `x-mesh-internal` header in `src/host/server.py`
- M10: Added `ol_session` cookie auth to VNC HTTP proxy in `src/host/server.py`
- M11: Added `ol_session` cookie auth to VNC WebSocket proxy in `src/host/server.py`
- M12: Added `_quote_env_value()` for proper .env value escaping in `src/host/runtime.py`
- Updated 3 test assertions in `tests/test_runtime.py` for quoted .env format

### Fixes Applied (Production Readiness Review #3, 2026-03-06)

**Batch 1 — Dead Weight:**
- Moved inline `from urllib.parse import urlparse` and `import base64` to module-level in `src/browser/service.py`
- Removed 4 redundant inner `import re`/`import os` in `src/dashboard/server.py`
- Removed 2 redundant inner `import json as _json` in `src/cli/main.py`

**Batch 2 — Standardization:**
- Added `autoescape=True` to SPA catchall Jinja2 Environment in `src/dashboard/server.py`
- Added `default-src 'self'; connect-src 'self'; frame-src 'self'` to CSP headers in `src/dashboard/server.py` (both main dashboard and catchall)
- Extracted `_verify_dashboard_auth` to module level, added as dependency to SPA catchall router (X1-4)

**Batch 4 — Bug Fixes:**
- E14-01 (HIGH): Fixed `_remove_project_blackboard_permissions` in `src/cli/config.py` — now only removes project-specific pattern instead of blanket-clearing all
- E14-02 (HIGH): Fixed log file handling in `_start_detached()` — append mode + try/finally for cleanup
- E11-7 (MEDIUM): Added `_MAX_WALK_DEPTH = 50` to `snapshot._walk()` in `src/browser/service.py`
- E11-10 (MEDIUM): Upgraded cleanup error logging from `debug` to `warning` in `src/browser/service.py`
- E11-13 (LOW): Added `proc.wait()` after `proc.kill()` in `src/browser/__main__.py` to reap zombies
- E13-4 (MEDIUM): Added `threading.Lock` to `EventBus.emit()` in `src/dashboard/events.py`

**Batch 5 — Security:**
- E12-C1 (CRITICAL): Added `sanitize_for_prompt()` to streaming dispatch in `src/channels/telegram.py`, `discord.py`, `slack.py`
- E11-1 (HIGH): Removed `"evaluate"` from `_ALLOWED_ACTIONS` in `src/host/server.py`, removed `browser_evaluate` skill from `src/agent/builtins/browser_tool.py` and formatting case from `src/cli/formatting.py`
- E11-2 (HIGH): Added startup warning in `src/browser/server.py` when `BROWSER_AUTH_TOKEN` is not set
- E11-3 (MEDIUM): Replaced hardcoded KasmVNC password with `secrets.token_urlsafe(16)` in `src/browser/__main__.py`

**Tests:** Removed `TestBrowserEvaluateHttpClient` class and `test_evaluate_no_mesh` from `tests/test_builtins.py` (skill was removed). 1779 passed, 0 regressions.

### Outstanding Findings (not fixed — deferred)
- H1: No OAuth token refresh/expiry handling (design decision needed)
- H3: Template pub/sub permissions overridden by collaboration-mode wildcards
- H4: No `_oauth_chat_stream` tests (test coverage gap)
- M1: OAuth bypasses failover chain (structural change)
- M2: Network failures during OAuth validation treated as valid
- M3: OAuth error messages may leak token data
- M4: Self-evolution feedback loop risk
- M5: Task mode doesn't refresh system prompt after workspace writes
- M13-M15: Template workflow issues (name collisions, feedback loops, missing conditions)
- Batch 3 (consolidation) skipped: duplicated tool call construction, duplicated OAuth auth-choice prompt, `search()` using direct file reads instead of `_read_file()`
- E8-1: `_prepare_llm_params` agent-controlled kwargs leak (needs design decision on kwarg allowlist)
- E11-9: Overly broad redaction regex false positives (needs UX testing)
- E11-4: KasmVNC bound to 0.0.0.0 (Docker networking constraint)
- E14-03: `select.select` on stdin doesn't work on Windows
- E14-16/E14-17: Mesh server binds 0.0.0.0 ignoring mesh.host config

### Pre-Existing Test Failures (not regressions)
- `test_dashboard.py::TestDashboardAgentsAPI::test_api_agents_returns_fleet` — expects 2 agents but gets 5 (reads actual mesh.yaml from disk instead of mocked config)
- `test_dashboard.py::TestDashboardAgentsAPI::test_api_agents_empty_registry` — same root cause
