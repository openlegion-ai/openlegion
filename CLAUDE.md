# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

OpenLegion is a container-isolated multi-agent runtime. LLM-powered agents run in Docker containers (or Sandbox microVMs), coordinated through a central mesh host. Fleet model — no CEO agent. Users talk to agents directly; agents coordinate via shared blackboard and YAML workflows.

## Architecture (read this first)

```
User (CLI REPL / Telegram / Discord / Webhook)
  → Mesh Host (FastAPI :8420) — routes messages, enforces permissions, proxies APIs
    → Agent Containers (FastAPI :8400 each) — isolated execution with private memory
```

Three trust zones: **User** (full trust), **Mesh** (trusted coordinator), **Agents** (untrusted, sandboxed). Everything between zones is HTTP + JSON with Pydantic contracts defined in `src/shared/types.py`.

### Key source paths

| Path | What it does |
|---|---|
| `src/shared/types.py` | THE contract — all Pydantic models shared between host and agents |
| `src/agent/loop.py` | Agent execution loop (task mode + chat mode) |
| `src/agent/skills.py` | Skill registry and tool discovery |
| `src/agent/builtins/` | Built-in tools (exec, file, browser, memory, mesh, web search) |
| `src/agent/memory.py` | Per-agent SQLite + sqlite-vec + FTS5 memory store |
| `src/agent/workspace.py` | Persistent markdown workspace (MEMORY.md, daily logs, learnings) |
| `src/agent/context.py` | Context window management (write-then-compact) |
| `src/agent/llm.py` | LLM client (routes through mesh proxy, never holds keys) |
| `src/host/mesh.py` | Blackboard (SQLite WAL), PubSub, MessageRouter |
| `src/host/orchestrator.py` | DAG workflow executor with safe condition eval |
| `src/host/runtime.py` | RuntimeBackend ABC → DockerBackend / SandboxBackend |
| `src/host/transport.py` | Transport ABC → HttpTransport / SandboxTransport |
| `src/host/credentials.py` | Credential vault + LLM API proxy |
| `src/host/permissions.py` | Per-agent ACL enforcement (glob patterns) |
| `src/host/server.py` | Mesh FastAPI app factory |
| `src/host/lanes.py` | Per-agent FIFO task queues |
| `src/host/health.py` | Health monitor with auto-restart |
| `src/host/costs.py` | Per-agent cost tracking + budget enforcement |
| `src/host/cron.py` | Cron scheduler with heartbeat support |
| `src/channels/base.py` | Abstract channel with unified REPL-like UX |
| `src/cli.py` | CLI entry point and interactive REPL |

## Non-Negotiable Rules

### Security boundaries — never violate these

1. **Agents never hold API keys.** All LLM/API calls go through the mesh credential vault (`src/host/credentials.py`). An agent's `LLMClient` posts to `/mesh/api` — the vault injects credentials server-side. If you're adding a new external service integration, add it as a vault handler, not as agent-side code.

2. **No `eval()`, no `exec()` on untrusted input.** Workflow conditions use a regex-based safe parser (`src/host/orchestrator.py:_safe_evaluate_condition`). If you need conditional logic, extend the parser — never use `eval()`.

3. **Permission checks before every cross-boundary operation.** Blackboard reads/writes, pub/sub, message routing, and API proxy calls all check the `PermissionMatrix` first. New mesh endpoints must enforce permissions. Default policy is deny.

4. **Path traversal protection in agent file tools.** Agent file operations are confined to `/data` inside the container. The `file_tool.py` tools must validate paths. Never expose host filesystem paths to agents.

5. **Container hardening is not optional.** Agents run as non-root (UID 1000), with `no-new-privileges`, memory limits (512MB), and CPU quotas. Don't weaken these defaults.

6. **All untrusted text is sanitized before reaching LLM context.** `sanitize_for_prompt()` in `src/shared/utils.py` strips invisible Unicode (bidi overrides, tag chars, zero-width chars, variation selectors, etc.) at three choke points: user input (`server.py`), tool results (`loop.py`), and system prompt context (`loop.py`). New paths from untrusted text to LLM context must call `sanitize_for_prompt()`. Never bypass these layers.

### Architectural invariants

6. **`src/shared/types.py` is the contract.** Every message, event, and state object crossing component boundaries is a Pydantic model defined here. When adding new inter-component communication, add the model here first. Agents and mesh share ONLY these types.

7. **Fleet model, not hierarchy.** There is no CEO agent that routes or delegates. Users talk to agents directly. Agents coordinate through the blackboard (shared state) and YAML workflows (deterministic DAGs). Do not introduce agent-to-agent direct messaging patterns that bypass the mesh.

8. **Bounded execution.** Agent loops have hard limits: 20 iterations for tasks (`AgentLoop.MAX_ITERATIONS`), 10 tool rounds for chat (`CHAT_MAX_TOOL_ROUNDS`). Token budgets are enforced per task. These prevent runaway agents. Do not remove these limits.

9. **Write-then-compact.** Before discarding any conversation context, important facts are flushed to `MEMORY.md` via the workspace (`src/agent/context.py`). No information should be silently lost during context management.

10. **LLM tool-calling message roles must alternate correctly.** The sequence is: `user → assistant(tool_calls) → tool(result) → assistant`. Never split a tool_call from its tool results. The `_trim_context` method in `loop.py` groups them to preserve this invariant. Breaking this causes LLM API errors.

## Code Patterns

### How we write code here

- **Small modules.** No file exceeds ~800 lines. If a module grows past that, split it by responsibility.
- **Pydantic for boundaries, plain dicts internally.** Cross-component messages use Pydantic models. Internal data flow within a module can use dicts.
- **Async by default.** Agent-side code is async (FastAPI + asyncio). Use `async def` for any I/O. Blocking calls must be wrapped in `run_in_executor`.
- **SQLite for all state.** Blackboard, agent memory, cost tracking, cron — all SQLite with WAL mode. No Redis, no external databases. Always set `PRAGMA journal_mode=WAL` and `PRAGMA busy_timeout=5000` on new connections.
- **`TYPE_CHECKING` imports for circular dependency prevention.** Heavy imports (other modules in the project) go behind `if TYPE_CHECKING:` guards. See any module for examples.
- **Logging via `setup_logging(name)`.** Every module creates a logger with `logger = setup_logging("component.module")`. Use structured logging: `logger.info("msg", extra={"extra_data": {...}})`.

### Adding a new built-in tool

1. Create `src/agent/builtins/your_tool.py`
2. Use the `@skill` decorator with `name`, `description`, and `parameters`
3. Parameters dict defines the JSON schema for LLM function calling
4. Accept `mesh_client` and/or `workspace_manager` as keyword-only args if needed (auto-injected by `SkillRegistry.execute`)
5. Return a dict (serialized to JSON for the LLM)
6. Add tests in `tests/test_builtins.py`

```python
@skill(
    name="your_tool",
    description="Clear description of what this does and when to use it",
    parameters={
        "param1": {"type": "string", "description": "What this param is for"},
        "param2": {"type": "integer", "description": "Optional param", "default": 10},
    },
)
async def your_tool(param1: str, param2: int = 10, *, mesh_client=None) -> dict:
    # Implementation here
    return {"result": "value"}
```

### Adding a new mesh endpoint

1. Add the route in `src/host/server.py` inside `create_mesh_app()`
2. Enforce permissions — check `permissions.can_*()` before acting
3. Use Pydantic models from `src/shared/types.py` for request/response
4. If agents need to call it, add a method to `src/agent/mesh_client.py`
5. Add tests in `tests/test_mesh.py` or `tests/test_integration.py`

### Adding a new channel

1. Subclass `Channel` from `src/channels/base.py`
2. Implement `start()`, `stop()`, `send_notification()`
3. Message handling is already provided by the base class (`handle_message`) — it handles @mentions, /commands, and agent routing
4. Add startup logic in `cli.py:_start_channels()`
5. Add tests in `tests/test_channels.py`

## Testing

### Run tests

```bash
# Unit + integration (fast, no Docker needed)
pytest tests/ --ignore=tests/test_e2e.py --ignore=tests/test_e2e_chat.py --ignore=tests/test_e2e_memory.py --ignore=tests/test_e2e_triggering.py -x

# All tests including E2E (requires Docker + API key)
pytest tests/ -x

# Single test file
pytest tests/test_loop.py -x -v
```

### Testing conventions

- **Mock LLM responses, not the loop.** Tests create `AgentLoop` with mock `LLMClient` that returns predetermined `LLMResponse` objects. See `tests/test_loop.py:_make_loop()` for the pattern.
- **Use `AsyncMock` for async methods.** All agent-side methods are async.
- **SQLite in-memory for tests.** Use `":memory:"` or `tmp_path` for database paths in tests.
- **E2E tests are optional.** They require Docker and an API key. They skip gracefully when unavailable. Unit tests must pass without Docker.
- **Every new feature gets tests.** New tools, new endpoints, new channel logic — all need test coverage. Match the existing patterns in the corresponding test file.

### Test structure mirrors source

| Source | Test file |
|---|---|
| `src/agent/loop.py` | `tests/test_loop.py` |
| `src/agent/memory.py` | `tests/test_memory.py` |
| `src/agent/workspace.py` | `tests/test_workspace.py` |
| `src/agent/context.py` | `tests/test_context.py` |
| `src/agent/skills.py` + builtins | `tests/test_skills.py`, `tests/test_builtins.py` |
| `src/host/mesh.py` | `tests/test_mesh.py` |
| `src/host/orchestrator.py` | `tests/test_orchestrator.py` |
| `src/host/credentials.py` | `tests/test_credentials.py` |
| `src/host/runtime.py` | `tests/test_runtime.py` |
| `src/host/transport.py` | `tests/test_transport.py` |
| `src/host/costs.py` | `tests/test_costs.py` |
| `src/host/cron.py` | `tests/test_cron.py` |
| `src/channels/base.py` | `tests/test_channels.py` |
| `src/cli.py` | `tests/test_cli_commands.py` |
| `src/shared/utils.py` (sanitization) | `tests/test_sanitize.py` |
| Chat mode | `tests/test_chat.py`, `tests/test_chat_workspace.py` |

## Common Mistakes to Avoid

- **Creating httpx clients per request.** Reuse clients with connection pooling. `LLMClient` already does this (`_get_client`). Follow the same pattern elsewhere — create the client once, close on shutdown.
- **Polling for task completion.** Prefer push-based patterns (agent posts result back to mesh) over polling loops. The current `_wait_for_task_result` in orchestrator.py is a known area for improvement.
- **Breaking tool-call message grouping.** When trimming context, never separate an `assistant` message with `tool_calls` from its corresponding `tool` result messages. The `_trim_context` method handles this — respect the grouping pattern.
- **Putting secrets in agent code.** Agents run in untrusted containers. API keys, tokens, credentials — all belong in the vault (`OPENLEGION_CRED_*` env vars loaded by `credentials.py`). New service integrations go through the vault proxy.
- **Using global mutable state.** The `_skill_registry` global in `skills.py` is a known issue. Avoid adding new module-level mutable globals. Pass state through constructors.
- **Overly broad exception handling.** Don't `except Exception: pass`. Log the error. Distinguish transient errors (network timeouts, rate limits — retry with backoff) from permanent errors (invalid input, missing config — fail fast).
- **Monolithic functions.** `cli.py:_start_interactive` is too large. When adding features, prefer composable components over growing existing functions. Extract classes with clear lifecycle (init, start, stop).

## Design Philosophy

- **Act first, ask never.** Agents are autonomous executors. System prompts instruct them to call tools immediately, not describe what they would do. This philosophy should extend to all UX decisions — prefer action over confirmation dialogs.
- **The mesh is the only door.** Agents have no external network access except through the mesh. This is a feature, not a limitation — it enables credential isolation, cost tracking, and permission enforcement in one place.
- **Private by default, shared by promotion.** Agents keep knowledge in private memory. Facts are explicitly promoted to the blackboard when they should be shared. Don't default to sharing everything.
- **Skills over features.** New agent capabilities should be added as skills (Python functions with `@skill` decorator), not as changes to the core loop. The loop is the execution engine; skills are the capabilities. Keep them separate.
- **Smallest thing that works.** No LangChain, no Redis, no Kubernetes, no web UI. Every dependency must justify its existence. SQLite handles all state. Docker handles all isolation. Three-line BM25 search beats a vector database dependency for 90% of use cases.
