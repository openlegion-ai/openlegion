# Development & Testing

Guide for contributing to OpenLegion -- testing conventions, code patterns, and how to add new components.

## Quick Start

```bash
# Clone and install
git clone https://github.com/openlegion-ai/openlegion.git
cd openlegion
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run tests (no Docker needed). The -x flag stops at the first failure.
pytest tests/ --ignore=tests/test_e2e.py --ignore=tests/test_e2e_chat.py \
  --ignore=tests/test_e2e_memory.py --ignore=tests/test_e2e_triggering.py -x

# Run a single test file
pytest tests/test_loop.py -x -v
```

Or use the bundled `Makefile`:

```bash
make install   # runs install.sh — checks deps, creates .venv, installs package
make test      # runs the unit + integration suite (same ignores as above)
make lint      # ruff check + ruff format on src/ and tests/
make start     # openlegion start (inside .venv)
make stop      # openlegion stop
make clean     # remove .venv and cache directories
```

### Worktree Workflow (Mandatory)

All code changes must happen on a `git worktree`, never directly on `main`. Multiple subagents work concurrently — without isolated worktrees they overwrite each other's changes. See [CLAUDE.md → Git Workflow](../CLAUDE.md) for the full policy. The short version:

```bash
git worktree add -b feat/your-change .claude/worktrees/your-change
cd .claude/worktrees/your-change
# work, commit, push, open a PR
```

Never run `pip install` from a worktree — it hijacks the global `openlegion` entry point. Always merge through a GitHub PR; do not push to `main` directly.

### Entry Point

The `openlegion` CLI is declared as `openlegion = "src.cli:cli"` in `pyproject.toml`. `src/cli/__init__.py` re-exports `cli` from `src/cli/main.py`; the Click group lives at `src/cli/main.py:86-106`. The CLI is also module-invocable via `python -m src.cli` (see `src/cli/__main__.py`).

## Testing

### Test Categories

| Category | Needs Docker | Needs API Key | Description |
|----------|-------------|---------------|-------------|
| Unit tests | No | No | Component-level tests with mocked dependencies |
| Integration tests | No | No | Cross-component interaction tests |
| E2E tests | Yes | Yes | Full runtime with real containers and LLM |

### Running Tests

```bash
# Unit + integration (fast, recommended during development)
pytest tests/ --ignore=tests/test_e2e.py --ignore=tests/test_e2e_chat.py \
  --ignore=tests/test_e2e_memory.py --ignore=tests/test_e2e_triggering.py -x

# Single file
pytest tests/test_tools.py -x -v

# E2E (requires Docker + API key)
pytest tests/test_e2e.py -x -v

# Full suite
pytest tests/ -x
```

`pyproject.toml` sets `asyncio_mode = "auto"`, so `async def` test functions do **not** need a `@pytest.mark.asyncio` decorator.

### Test Markers

A custom `browser` marker is defined in `pyproject.toml` for tests that drive a real Playwright browser:

```python
import pytest

@pytest.mark.browser
async def test_navigate_real_browser():
    ...
```

Playwright is intentionally **not** in the `[dev]` extra (browser binaries balloon the install). CI does not run these tests. Install Playwright manually if you need them: `pip install playwright && playwright install firefox`.

### CI Matrix

`.github/workflows/test.yml` runs on push to `main` and pull requests targeting `main`. `lint` and `test` run on both triggers; `coverage` is push-only:

| Job | Trigger | Setup | Command |
|-----|---------|-------|---------|
| `test` | PR + push | Python 3.11 **and** 3.12 × 3 shards (matrix of 6), `uv pip install --system -e ".[dev]" Pillow` | `pytest ... -n auto --dist=loadfile --splits 3 --group ${{ matrix.shard }}` |
| `coverage` | push only | Python 3.12, `uv pip install` | `pytest ... -n auto --cov=src --cov-report=term-missing --cov-report=xml` |
| `lint` | PR + push | Python 3.12 | `pip install ruff && ruff check src/ tests/` |

Sharding uses `pytest-split` (`--splits 3 --group N`). `--dist=loadfile` keeps all tests from one file on the same xdist worker so fixture caches are reused — important for `tests/test_browser_service.py` (≈500 tests with shared setup).

### Test File Map

| Source | Test File |
|--------|-----------|
| `src/agent/loop.py` | `tests/test_loop.py`, `tests/test_chat.py` |
| `src/agent/memory.py` | `tests/test_memory.py`, `tests/test_memory_integration.py` |
| `src/agent/workspace.py` | `tests/test_workspace.py`, `tests/test_chat_workspace.py` |
| `src/agent/context.py` | `tests/test_context.py` |
| `src/agent/tools.py` | `tests/test_tools.py` |
| `src/agent/builtins/*` | `tests/test_builtins.py`, `tests/test_memory_tools.py` |
| `src/agent/builtins/vault_tool.py` | `tests/test_vault.py` |
| `src/agent/builtins/subagent_tool.py` | `tests/test_subagent.py` |
| `src/agent/mcp_client.py` | `tests/test_mcp_client.py`, `tests/test_mcp_e2e.py` |
| `src/host/mesh.py` | `tests/test_mesh.py` |
| `src/host/credentials.py` | `tests/test_credentials.py` |
| `src/host/runtime.py` | `tests/test_runtime.py` |
| `src/host/transport.py` | `tests/test_transport.py` |
| `src/host/costs.py` | `tests/test_costs.py` |
| `src/host/cron.py` | `tests/test_cron.py` |
| `src/host/health.py` | `tests/test_health.py` |
| `src/host/lanes.py` | `tests/test_lanes.py` |
| `src/host/traces.py` | `tests/test_traces.py` |
| `src/host/transcript.py` | `tests/test_transcript.py` |
| `src/agent/server.py` | `tests/test_agent_server.py` |
| `src/dashboard/server.py` | `tests/test_dashboard.py`, `tests/test_dashboard_workspace.py` |
| `src/host/failover.py` | `tests/test_failover.py` |
| `src/host/webhooks.py` | `tests/test_webhooks.py` |
| `src/agent/loop_detector.py` | `tests/test_loop_detector.py` |
| `src/marketplace.py` | `tests/test_marketplace.py` |
| `src/channels/base.py` | `tests/test_channels.py` |
| `src/channels/discord.py` | `tests/test_discord.py` |
| `src/channels/slack.py` | `tests/test_slack.py` |
| `src/channels/whatsapp.py` | `tests/test_whatsapp.py` |
| `src/agent/attachments.py` | `tests/test_attachments.py` |
| `src/agent/llm.py` | `tests/test_llm_param_allowlist.py` |
| `src/agent/builtins/web_search_tool.py` | `tests/test_web_search_tool.py` |
| `src/browser/service.py` | `tests/test_browser_service.py` |
| `src/shared/types.py` | `tests/test_types.py` |
| `src/shared/utils.py` (sanitization) | `tests/test_sanitize.py` |
| `src/shared/models.py` | `tests/test_models.py` |
| `src/templates/` | `tests/test_templates.py` |
| `src/cli/` | `tests/test_cli_commands.py`, `tests/test_setup_wizard.py` |
| `src/cli/config.py` (teams) | `tests/test_teams.py` |
| `src/dashboard/auth.py` | `tests/test_dashboard_auth.py` |
| `src/dashboard/events.py` | `tests/test_events.py` |
| `src/agent/memory.py` (fallback) | `tests/test_embedding_fallback.py` |
| Cross-component | `tests/test_integration.py` |
| `src/agent/builtins/wallet_tool.py` | `tests/test_wallet.py`, `tests/test_wallet_tool.py` |
| `src/host/wallet.py` | `tests/test_wallet_endpoints.py` |
| `src/agent/builtins/image_gen_tool.py` | `tests/test_image_gen.py` |
| `src/agent/builtins/coordination_tool.py` | `tests/test_coordination.py` |
| `src/host/permissions.py` | `tests/test_permissions.py` |
| `src/host/api_keys.py` | `tests/test_api_keys.py` |

### Testing Conventions

**Mock LLM responses, not the loop.** Tests create `AgentLoop` with a mock `LLMClient` that returns predetermined `LLMResponse` objects. The canonical helper lives in `tests/test_loop.py:_make_loop()` — copy from there rather than rolling your own setup:

```python
async def _make_loop(tool_calls=None, text="Done"):
    llm = AsyncMock()
    llm.chat.return_value = LLMResponse(
        content=text,
        tool_calls=tool_calls or [],
        tokens_used=100,
    )
    loop = AgentLoop(
        agent_id="test",
        role="test",
        memory=mock_memory,
        tools=mock_tools,
        llm=llm,
        mesh_client=mock_mesh,
    )
    return loop
```

**Use `AsyncMock` for async methods.** All agent-side code is async:

```python
from unittest.mock import AsyncMock, MagicMock

mesh_client = AsyncMock()
mesh_client.read_blackboard.return_value = {"key": "value"}
```

**SQLite in-memory for tests.** Use `":memory:"` or `tmp_path` for databases:

```python
async def test_memory_store(tmp_path):
    store = MemoryStore(db_path=str(tmp_path / "test.db"))
    await store.store_fact(key="user_name", value="Alice")
    results = await store.search("name", top_k=5)
    assert results[0].value == "Alice"
```

**Every new feature gets tests.** New tools, endpoints, and channel logic all need coverage.

## Code Patterns

### Module Structure

- Keep modules focused by responsibility
- Async by default (`async def` for any I/O)
- `TYPE_CHECKING` imports for circular dependency prevention
- Logging via `setup_logging("component.module")`

```python
from __future__ import annotations

from typing import TYPE_CHECKING

from src.shared.utils import setup_logging

if TYPE_CHECKING:
    from src.agent.mcp_client import MCPClient

logger = setup_logging("agent.tools")
```

### Pydantic for Boundaries

Cross-component messages use Pydantic models from `src/shared/types.py`. Internal data flow within a module can use dicts.

```python
# In src/shared/types.py
class TaskAssignment(BaseModel):
    task_id: str                          # auto-generated if omitted
    workflow_id: str                      # required
    step_id: str                          # required
    task_type: str
    input_data: dict[str, Any]
    context: dict[str, Any] = {}
    timeout: int = 120
    max_retries: int = 0
    token_budget: TokenBudget | None = None

# Used at component boundaries — dispatching tasks to agents
assignment = TaskAssignment(
    workflow_id="wf_123",
    step_id="step_1",
    task_type="research",
    input_data={"company": "Acme Corp"},
)
```

### SQLite for All State

Blackboard, agent memory, cost tracking, cron -- all SQLite with WAL mode:

```python
conn = sqlite3.connect(db_path)
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA busy_timeout=30000")  # 30s for mesh/costs/memory; traces uses 5000
```

The shorter `5000`ms timeout on the traces DB is intentional: traces writes are append-only and high-volume, so blocking the request path for 30s on lock contention would mask real latency problems. Other stores can afford to wait. See `CLAUDE.md` → Known Constraints (8) for the rationale.

No Redis, no external databases.

### Runtime Constants

Bounded execution is enforced by a small set of constants in `src/agent/loop.py`:

| Constant | Value | Env override |
|---|---|---|
| `MAX_ITERATIONS` | 20 | `OPENLEGION_MAX_ITERATIONS` (clamped 1–100) |
| `CHAT_MAX_TOOL_ROUNDS` | 30 | env-clamped 1–200 |
| `CHAT_MAX_TOTAL_ROUNDS` | 200 | env-clamped 1–1000 |
| `HEARTBEAT_MAX_ITERATIONS` | 12 | — |
| `_TOOL_TIMEOUT` | 300 (seconds) | `OPENLEGION_TOOL_TIMEOUT` |
| `_MAX_SESSION_CONTINUES` | 5 | hardcoded |

Context manager thresholds (`src/agent/context.py`): 60% proactive fact flush, 70% compact (summarize + replace), 80% warning injected into the prompt, 90% emergency hard-prune. `_hard_prune()` keeps the first message plus the last 4 tool-call groups and bridges same-role sequences.

Workspace file caps (`_FILE_CAPS` in `src/agent/server.py`) are enforced on `PUT /workspace/{filename}` with HTTP 413 on overflow — see [`configuration.md`](configuration.md) for the per-file size table.

### Tool Discovery

Tools are auto-discovered at agent startup from three paths inside the container:

- `/app/agent/builtins` — built-in tools shipped with the engine
- `/data/custom_tools` — agent self-authored tools (persisted on the agent's data volume)
- `/app/marketplace_tools` — installed marketplace tools

Conflicts resolve in order: custom overrides built-in overrides marketplace.

## Adding New Components

### Adding a Built-in Tool

1. Create `src/agent/builtins/your_tool.py` (maps to `/app/agent/builtins/your_tool.py` inside the agent container at runtime)
2. Use the `@tool` decorator
3. Accept `mesh_client`/`workspace_manager`/`memory_store` as keyword-only args if needed (auto-injected)
4. Return a dict
5. Add tests in `tests/test_builtins.py`

```python
from src.agent.tools import tool

@tool(
    name="your_tool",
    description="Clear description of what this does",
    parameters={
        "param1": {"type": "string", "description": "What this param is for"},
        "param2": {"type": "integer", "description": "Optional param", "default": 10},
    },
)
async def your_tool(param1: str, param2: int = 10, *, mesh_client=None) -> dict:
    result = do_something(param1, param2)
    return {"result": result}
```

### Adding a Mesh Endpoint

1. Add the route in `src/host/server.py` inside `create_mesh_app()`
2. Enforce permissions -- check `permissions.can_*()` before acting
3. Use Pydantic models from `src/shared/types.py`
4. Add a method to `src/agent/mesh_client.py` if agents need to call it
5. Add HTTP endpoint tests in `tests/test_dashboard.py` (which covers `create_mesh_app()` HTTP handlers). Reserve `tests/test_mesh.py` for blackboard/pubsub layer tests.

All endpoints are registered inside the `create_mesh_app()` factory in `src/host/server.py` — `@app` is a local variable, so routes must live in that function's scope. Permission checks use the helpers defined alongside (`_require_any_auth`, `_require_operator_or_internal`) and the per-agent `permissions.can_*()` calls. Agent identity is taken from the request (typically a Bearer token resolved against `_auth_tokens`); there is no top-level `_get_agent_id(request)` helper.

```python
# Inside create_mesh_app() in src/host/server.py
@app.post("/mesh/your_endpoint")
async def your_endpoint(request: Request, body: YourRequest):
    _require_any_auth(request)
    agent_id = _agent_id_from_token(request)  # resolved against _auth_tokens
    if not permissions.can_do_thing(agent_id):
        raise HTTPException(403, "Not permitted")
    result = do_thing(body)
    return {"status": "ok", "data": result}
```

### Adding a Channel

1. Implement the adapter in `src/channels/your_channel.py` (subclass `Channel` from `src/channels/base.py`)
2. Implement `start()`, `stop()`, `send_notification()`
3. The base class `handle_message()` handles all command parsing
4. Wire startup/stop lifecycle in `src/cli/channels.py:ChannelManager` — channel adapters live under `src/channels/`; `src/cli/channels.py` only orchestrates lifecycle.
5. Add tests in `tests/test_channels.py`

### Self-Authored Tools and the Marketplace

Agents can author their own tools via `src/agent/builtins/tool_authoring.py`, and operators can install tools from git repositories via `src/marketplace.py`. Both paths run untrusted Python — the safety net is **AST validation** before the file is imported:

- `_FORBIDDEN_IMPORTS` (23 modules) — blocks `os`, `subprocess`, `socket`, etc.
- `_FORBIDDEN_CALLS` (16 functions) — blocks `eval`, `exec`, `open`, `compile`, `__import__`, etc.
- `_FORBIDDEN_ATTRS` (11 attributes) — blocks `__dict__`, `__subclasses__`, `__globals__`, etc.

Marketplace clones are also hardened against malicious repositories: `git clone` runs with `-c core.hooksPath=/dev/null`, `-c protocol.ext.allow=never`, `-c core.symlinks=false`, and an environment of `GIT_CONFIG_NOSYSTEM=1 GIT_TERMINAL_PROMPT=0`. Depth is `1`, only `https://` and `git@` URLs are accepted.

## Project Structure

```
openlegion/
├── src/
│   ├── agent/                   # Runs inside containers
│   │   ├── __main__.py          # Agent entry point
│   │   ├── server.py            # Agent FastAPI server
│   │   ├── loop.py              # Execution loop (task + chat)
│   │   ├── loop_detector.py     # Tool loop detection (warn/block/terminate)
│   │   ├── tools.py            # Tool registry
│   │   ├── memory.py            # SQLite + sqlite-vec memory
│   │   ├── workspace.py         # Persistent markdown workspace
│   │   ├── context.py           # Context window management
│   │   ├── llm.py               # LLM client (mesh proxy)
│   │   ├── mcp_client.py        # MCP tool server client
│   │   ├── mesh_client.py       # Mesh API client
│   │   └── builtins/            # Built-in tools
│   │       ├── exec_tool.py     # Shell execution
│   │       ├── file_tool.py     # File operations
│   │       ├── http_tool.py     # HTTP requests
│   │       ├── browser_tool.py  # Browser automation via shared Camoufox service
│   │       ├── memory_tool.py   # Memory search/save
│   │       ├── mesh_tool.py     # Blackboard, pub/sub, artifacts, cron
│   │       ├── vault_tool.py    # Credential vault (blind storage)
│   │       ├── introspect_tool.py # Live runtime state queries
│   │       ├── tool_authoring.py    # Custom tool creation + reload
│   │       ├── subagent_tool.py # In-process subagent spawning
│   │       ├── web_search_tool.py  # DuckDuckGo search
│   │       ├── wallet_tool.py   # Blockchain wallet operations (address, balance, transfer)
│   │       ├── image_gen_tool.py # Image generation via Gemini or DALL-E 3
│   │       ├── coordination_tool.py # Structured multi-agent coordination (hand_off, check_inbox)
│   │       ├── fleet_tool.py    # Fleet management (operator-agent only)
│   │       └── operator_tools.py # Operator-privileged tools (operator-agent only)
│   ├── host/                    # Runs on the host machine
│   │   ├── server.py            # Mesh HTTP endpoints (~109 routes) that expose the mesh.py classes
│   │   ├── mesh.py              # Blackboard, PubSub, MessageRouter, audit log (low-level classes)
│   │   ├── runtime.py           # Docker/Sandbox container management
│   │   ├── transport.py         # HTTP/Sandbox transport layer
│   │   ├── credentials.py       # Credential vault + API proxy
│   │   ├── permissions.py       # ACL enforcement
│   │   ├── health.py            # Health monitor with auto-restart
│   │   ├── costs.py             # Cost tracking + budgets
│   │   ├── cron.py              # Cron scheduler
│   │   ├── lanes.py             # Per-agent task queues
│   │   ├── failover.py          # LLM model failover logic
│   │   ├── webhooks.py          # Webhook manager
│   │   ├── traces.py            # Request tracing and diagnostics
│   │   ├── transcript.py        # Provider-specific transcript sanitization
│   │   ├── wallet.py            # Ethereum + Solana wallet signing service
│   │   └── api_keys.py          # Named API key management (salted SHA-256, config/api_keys.json)
│   ├── channels/                # Messaging adapters
│   │   ├── base.py              # Abstract channel
│   │   ├── telegram.py          # Telegram bot
│   │   ├── discord.py           # Discord bot
│   │   ├── slack.py             # Slack (Socket Mode)
│   │   └── whatsapp.py          # WhatsApp (Cloud API)
│   ├── shared/                  # Shared between host and agent
│   │   ├── types.py             # Pydantic contracts
│   │   ├── utils.py             # Logging, ID generation
│   │   ├── trace.py             # Distributed trace context
│   │   └── models.py            # Model cost/context window registry (LiteLLM-backed)
│   ├── dashboard/               # Web dashboard
│   │   ├── server.py            # FastAPI router + API endpoints
│   │   ├── events.py            # EventBus for real-time streaming
│   │   ├── auth.py              # Session cookie verification
│   │   ├── telemetry.py         # SPA telemetry event sink
│   │   ├── platform_success.py  # Per-tenant success scoring
│   │   ├── templates/           # Dashboard HTML (Alpine.js + Tailwind)
│   │   └── static/              # CSS + JS assets
│   ├── setup_wizard.py          # Guided setup wizard
│   ├── marketplace.py           # Tool marketplace
│   ├── templates/               # Agent setup templates
│   └── cli/                     # CLI package
│       ├── main.py              # Click commands and entry point
│       ├── config.py            # Config loading, Docker helpers
│       ├── runtime.py           # RuntimeContext lifecycle management
│       ├── repl.py              # REPLSession interactive dispatch
│       ├── channels.py          # ChannelManager messaging lifecycle
│       ├── formatting.py        # Tool display and styled output
│       └── proxy.py             # Proxy env var builder (build_proxy_env_vars)
├── config/                      # Runtime configuration
│   ├── agents.yaml
│   ├── mesh.yaml
│   ├── permissions.json
│   └── cron.json
├── tests/                       # Test suite (~193 test files)
│   └── fixtures/                # Test fixtures (echo MCP server, etc.)
├── tools/                       # Standalone dev tools
│   ├── behavior_analyze.py      # Browser/captcha behavior analysis
│   ├── behavior_baseline.jsonl  # Recorded baseline for comparison
│   └── captcha_validation/      # Captcha solver validation harness
├── Dockerfile.agent             # Agent container image
├── Dockerfile.browser           # Shared browser service image
├── Makefile                     # install / start / stop / test / lint / clean
└── pyproject.toml               # Project metadata
```

## Dependencies

### Host-Side (installed via `pip install -e .`)

| Package | Purpose |
|---------|---------|
| `fastapi` + `uvicorn` | Mesh HTTP server |
| `httpx` | HTTP client |
| `pydantic` | Type validation |
| `litellm` | LLM API abstraction (multi-provider) |
| `docker` | Container management |
| `click` | CLI framework |
| `pyyaml` | Config parsing |
| `python-dotenv` | `.env` file loading |
| `sqlite-vec` | Vector search for memory |
| `websockets` | Dashboard real-time streaming |
| `pypdf` | PDF text extraction for attachments |
| `anthropic` | Anthropic SDK (used by litellm) |
| `python-multipart` | Multipart form data parsing |
| `Pillow` | Image processing for browser screenshots |

### Agent-Side (installed in container via Dockerfile)

| Package | Purpose |
|---------|---------|
| `fastapi` + `uvicorn` | Agent HTTP server |
| `httpx` | Mesh API client |
| `pydantic` | Type validation |
| `sqlite-vec` | Vector search for memory |
| `mcp` | Model Context Protocol client |
| `pypdf` | PDF text extraction for attachments |
| `camoufox` | Stealth browser automation (in browser service container) |

### Optional

| Group | Packages | Purpose |
|-------|----------|---------|
| `dev` | `pytest`, `pytest-asyncio`, `pytest-cov`, `pytest-split`, `pytest-xdist`, `ruff` | Testing, coverage, sharding, and linting (`pytest-split` is what CI uses for `--splits 3`) |
| `mcp` | `mcp>=1.0` | MCP tool support |
| `channels` | `python-telegram-bot`, `discord.py`, `slack-bolt` | Messaging channels |
| `wallet` | `web3`, `eth-account`, `mnemonic`, `solders`, `solana` | Ethereum + Solana wallet support |

There is no `requirements.lock` / `Pipfile.lock` — version bounds in `pyproject.toml` are `>=` only. CI runs the test matrix on both Python 3.11 and 3.12 to catch regressions across the supported range.

## Common Mistakes

- **Creating httpx clients per request** -- reuse clients with connection pooling
- **Polling for task completion** -- prefer push-based patterns
- **Breaking tool-call message grouping** -- never split `assistant(tool_calls)` from `tool(result)` messages
- **Putting secrets in agent code** -- use the credential vault
- **Using global mutable state** -- pass state through constructors
- **Overly broad exception handling** -- log errors, distinguish transient from permanent
- **Monolithic functions** -- prefer composable components over growing existing functions
