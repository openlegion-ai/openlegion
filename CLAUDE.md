# CLAUDE.md — engine

## Current State

### Architecture (verified 2026-03-05)

OpenLegion is a container-isolated multi-agent runtime. LLM-powered agents run in Docker containers (or Sandbox microVMs), coordinated through a central mesh host. Fleet model — no CEO agent. Users talk to agents directly; agents coordinate via shared blackboard and YAML workflows.

```
User (CLI REPL / Telegram / Discord / Slack / WhatsApp / Webhook)
  → Mesh Host (FastAPI :8420) — routes messages, enforces permissions, proxies APIs
    → Agent Containers (FastAPI :8400 each) — isolated execution with private memory
    → Browser Service Container (FastAPI :8421) — shared Camoufox browser instances
```

Three trust zones: **User** (full trust), **Mesh** (trusted coordinator), **Agents** (untrusted, sandboxed). Everything between zones is HTTP + JSON with Pydantic contracts defined in `src/shared/types.py`.

### Module Map

| Path | What it does |
|---|---|
| **Shared** | |
| `src/shared/types.py` | THE contract — all Pydantic models shared between host and agents |
| `src/shared/utils.py` | Utilities: `sanitize_for_prompt()`, `setup_logging()`, misc helpers |
| `src/shared/trace.py` | Distributed trace-ID generation and propagation |
| `src/shared/models.py` | Model cost/context window registry backed by LiteLLM |
| ~~`src/shared/constants.py`~~ | *(Deleted — was dead code, zero imports)* |
| **Agent Container** | |
| `src/agent/loop.py` | Agent execution loop (task mode + chat mode) |
| `src/agent/skills.py` | Skill registry and tool discovery |
| `src/agent/server.py` | Agent container FastAPI server (`/task`, `/chat`, `/status`, `/cancel`) |
| `src/agent/llm.py` | LLM client (routes through mesh proxy, never holds keys) |
| `src/agent/mesh_client.py` | Agent-side HTTP client for mesh communication |
| `src/agent/memory.py` | Per-agent SQLite + sqlite-vec + FTS5 memory store |
| `src/agent/workspace.py` | Persistent markdown workspace (SOUL.md, INSTRUCTIONS.md, MEMORY.md, daily logs, learnings) |
| `src/agent/context.py` | Context window management (write-then-compact) |
| `src/agent/loop_detector.py` | Tool loop detection with escalating responses |
| `src/agent/mcp_client.py` | MCP tool server client and lifecycle |
| `src/agent/__main__.py` | Agent container entry point |
| **Agent Builtins** | |
| `src/agent/builtins/exec_tool.py` | Shell execution scoped to `/data` |
| `src/agent/builtins/file_tool.py` | File I/O with two-stage path traversal protection |
| `src/agent/builtins/http_tool.py` | HTTP requests with CRED handles and DNS rebinding/SSRF protection |
| `src/agent/builtins/browser_tool.py` | Browser automation via shared Camoufox service container |
| `src/agent/builtins/memory_tool.py` | Memory search with hierarchical fallback |
| `src/agent/builtins/mesh_tool.py` | Blackboard, pub/sub, notify_user, list_agents |
| `src/agent/builtins/vault_tool.py` | Credential generation without returning actual values |
| `src/agent/builtins/web_search_tool.py` | DuckDuckGo search (no API key needed) |
| `src/agent/builtins/skill_tool.py` | Self-authoring with AST validation (forbidden imports/calls blocked) |
| `src/agent/builtins/subagent_tool.py` | In-process subagents (MAX_DEPTH=2, MAX_CONCURRENT=3, DEFAULT_TTL=300s) |
| `src/agent/builtins/introspect_tool.py` | Runtime state query (permissions, budget, fleet, cron, health) |
| **Host/Mesh** | |
| `src/host/server.py` | Mesh FastAPI app factory — all endpoints enforce permissions |
| `src/host/mesh.py` | Blackboard (SQLite WAL), PubSub, MessageRouter |
| `src/host/orchestrator.py` | DAG workflow executor with safe condition eval (regex, no eval()) |
| `src/host/runtime.py` | RuntimeBackend ABC → DockerBackend / SandboxBackend |
| `src/host/transport.py` | Transport ABC → HttpTransport / SandboxTransport |
| `src/host/credentials.py` | Two-tier credential vault (SYSTEM_*/CRED_*) + LLM API proxy |
| `src/host/permissions.py` | Per-agent ACL enforcement (glob patterns, deny-all default) |
| `src/host/lanes.py` | Per-agent FIFO task queues (followup/steer/collect modes) |
| `src/host/health.py` | Health monitor with auto-restart and rate limiting |
| `src/host/costs.py` | Per-agent cost tracking + budget enforcement (SQLite) |
| `src/host/cron.py` | Persistent cron scheduler with heartbeat probes |
| `src/host/failover.py` | Model health tracking + failover chains |
| `src/host/traces.py` | Request tracing + grouped summaries |
| `src/host/transcript.py` | Provider-specific transcript sanitization |
| `src/host/webhooks.py` | Named webhook endpoints |
| `src/host/watchers.py` | File watcher with polling |
| `src/host/containers.py` | Backward-compat alias for `DockerBackend` |
| **Browser Service** | |
| `src/browser/server.py` | Browser service FastAPI app |
| `src/browser/service.py` | BrowserManager with per-agent Camoufox instances |
| `src/browser/redaction.py` | Credential redaction for browser output |
| `src/browser/stealth.py` | Anti-bot fingerprint building |
| `src/browser/timing.py` | Timing jitter for human-like behavior |
| **Channels** | |
| `src/channels/base.py` | Abstract Channel with PairingManager for unified REPL-like UX |
| `src/channels/telegram.py` | Telegram bot channel adapter |
| `src/channels/discord.py` | Discord bot channel adapter |
| `src/channels/slack.py` | Slack channel adapter (Socket Mode) |
| `src/channels/whatsapp.py` | WhatsApp Cloud API channel adapter |
| `src/channels/webhook.py` | HTTP webhook channel adapter |
| **Dashboard** | |
| `src/dashboard/server.py` | Dashboard FastAPI router + API endpoints |
| `src/dashboard/events.py` | EventBus for real-time WebSocket streaming |
| `src/dashboard/auth.py` | Session cookie verification for dashboard access |
| **CLI** | |
| `src/cli/main.py` | CLI entry point: start, stop, status, chat, version |
| `src/cli/config.py` | Config loading, Docker helpers, agent management |
| `src/cli/runtime.py` | RuntimeContext — full lifecycle management |
| `src/cli/repl.py` | REPLSession — interactive command dispatch |
| `src/cli/channels.py` | ChannelManager — messaging channel lifecycle |
| `src/cli/formatting.py` | Tool display, styled output, response rendering |
| **Other** | |
| `src/setup_wizard.py` | Interactive setup wizard with validation |
| `src/marketplace.py` | Git-based skill marketplace (install/remove) |

### Cross-Repo Integration Points

**The engine is a standalone runtime with NO direct dependencies on app/ or provisioner/.** No imports, no calls, no shared code.

Integration happens externally:

1. **Provisioner → Engine**: Provisioner manages engine instances via Docker/systemd on Hetzner VPS. Provisioner deploys code, writes `.env`, starts services, checks health via SSH to `/mesh/agents`.

2. **App → Engine**: App generates HMAC tokens for SSO. Engine's auth gate (deployed via cloud-init) verifies HMAC tokens at `/__auth/callback?token={expiry}.{signature}` using the shared `access_token`.

3. **Mesh endpoints exposed** (that external systems interact with):
   - `/mesh/agents` — health check target (provisioner hits this via SSH)
   - `/__auth/callback` — SSO callback (app redirects users here)
   - Dashboard UI on port 8420 — user-facing after SSO

### Patterns In Use

- **Pydantic for boundaries, plain dicts internally.** Cross-component messages use Pydantic models from `types.py`. Internal data flow within a module can use dicts.
- **Async by default.** Agent-side code is async (FastAPI + asyncio). Blocking calls wrapped in `run_in_executor`.
- **SQLite for all state.** Blackboard, agent memory, cost tracking, cron, traces — all SQLite with WAL mode and `busy_timeout=5000`.
- **`TYPE_CHECKING` imports** for circular dependency prevention (28 instances across codebase).
- **`setup_logging(name)`** — every module creates `logger = setup_logging("component.module")`.
- **Small, focused modules** — each file has a single responsibility.
- **Skills over features** — new agent capabilities added as `@skill` decorated functions, not loop changes.

### Known Tech Debt

1. **Polling in orchestrator**: `_wait_for_task_result` uses polling instead of push-based notification.
2. **Module-level globals**: `_skill_staging` in `skills.py` (protected by threading lock), `_client` in browser tool (for connection pooling). Avoid adding more.
3. **Rate limiting**: Per-endpoint rate limiting via in-memory deque — not persistent across restarts.
4. **Subagent browser concurrency**: Module-level state means subagents shouldn't use browser concurrently.

### Infrastructure

- **Runtime**: Python 3.11+, FastAPI, asyncio
- **State**: All SQLite (WAL mode) — no Redis, no external databases
- **Isolation**: Docker containers (or Sandbox microVMs) per agent
- **Container hardening**: non-root (UID 1000), `no-new-privileges`, 384MB memory, 0.15 CPU
- **Browser service**: Shared container with 2GB RAM, 1 CPU (Camoufox)
- **Dashboard**: Alpine.js SPA — no React, no build step
- **Dependencies**: Minimal — no LangChain, no Kubernetes
- **CI**: GitHub Actions (lint + tests on multiple Python versions)

## Target State

### Architecture Decisions

- Engine architecture is sound. No structural changes needed for production.
- The polling in orchestrator should eventually be replaced with push-based notification.
- Browser service architecture (shared container) is correct for resource efficiency.

### Pattern Standards

- All modules follow the same patterns consistently. No standardization needed.
- `setup_logging()`, `TYPE_CHECKING` imports, SQLite WAL — all uniformly applied.

### Security Requirements

All security boundaries are already enforced:
- Trust zones (User/Mesh/Agent) with permission checks on all 26+ mesh endpoints
- `sanitize_for_prompt()` at 35+ choke points for prompt injection prevention
- File path traversal protection (two-stage validation)
- SSRF protection (DNS pinning + IP blocking)
- Credential isolation (two-tier vault, opaque handles)
- Execution limits (iteration caps, token budgets, subagent depth/concurrency)
- Safe condition evaluation (regex parser, no eval)

### Current → Target Gaps

- No critical gaps identified. Engine is production-ready from architecture and security standpoints.
- Minor: document `src/shared/constants.py` and `src/dashboard/auth.py` (now done above).

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

3. **Permission checks before every cross-boundary operation.** Blackboard reads/writes, pub/sub, message routing, and API proxy calls all check the `PermissionMatrix` first. Default policy is deny.

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
| `src/dashboard/server.py` | `tests/test_dashboard.py`, `tests/test_dashboard_workspace.py` |
| `src/dashboard/auth.py` | `tests/test_dashboard_auth.py` |
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
Stage 2 complete. See `/REVIEW_FINDINGS.md` for full findings.

### Completed Units
Units 1-16 (all engine units)

### Findings Summary
- **CRITICAL (3)**: AST validation bypass (skill_tool.py), SSRF via browser evaluate (browser/server.py), webhook channel zero auth (channels/webhook.py)
- **HIGH (8)**: /mesh/spawn no permission check, streaming budget lock gap, blackboard not sanitized, auth header forwarding on redirects, webhook payloads not sanitized, file watcher not sanitized, channel messages not sanitized, WhatsApp missing signature verification
- **MEDIUM (19)**: /mesh/cron no permission check, input_data not sanitized, HTTP 500 not retried, forced final call waste, /project no x-mesh-internal, skill auto-discovery, list_by_prefix no filtering, agent_id not validated, PUA emoji stripping, orchestrator task_result bypass, redundant index, trusted agent IDs not reserved, cron range+step parsing bug, 0.0.0.0 SSRF bypass, credential double-resolution, exec timeout unbounded, list_files symlink info leak, context summarization truncation, cron tasks not tracked

### Outstanding Cross-References
- H2 (streaming budget) coupled with H5 (TokenBudget.can_spend ignoring max_cost_usd)
- H3 (blackboard sanitization) + H7/H8/H9 (webhook/watcher/channel sanitization) — same pattern, same fix

### Fixes Applied (Stage 3, 2026-03-05)

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
