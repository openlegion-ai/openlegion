# Development & Testing

Guide for contributing to OpenLegion -- testing conventions, code patterns, and how to add new components.

## Quick Start

```bash
# Clone and install
git clone https://github.com/openlegion-ai/openlegion.git
cd openlegion
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run tests (no Docker needed)
pytest tests/ --ignore=tests/test_e2e.py --ignore=tests/test_e2e_chat.py \
  --ignore=tests/test_e2e_memory.py --ignore=tests/test_e2e_triggering.py -x

# Run a single test file
pytest tests/test_loop.py -x -v
```

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
pytest tests/test_skills.py -x -v

# E2E (requires Docker + API key)
pytest tests/test_e2e.py -x -v

# Full suite
pytest tests/ -x
```

### Test File Map

| Source | Test File |
|--------|-----------|
| `src/agent/loop.py` | `tests/test_loop.py`, `tests/test_chat.py` |
| `src/agent/memory.py` | `tests/test_memory.py`, `tests/test_memory_integration.py` |
| `src/agent/workspace.py` | `tests/test_workspace.py`, `tests/test_chat_workspace.py` |
| `src/agent/context.py` | `tests/test_context.py` |
| `src/agent/skills.py` | `tests/test_skills.py` |
| `src/agent/builtins/*` | `tests/test_builtins.py`, `tests/test_memory_tools.py` |
| `src/agent/builtins/vault_tool.py` | `tests/test_vault.py` |
| `src/agent/builtins/subagent_tool.py` | `tests/test_subagent.py` |
| `src/agent/mcp_client.py` | `tests/test_mcp_client.py`, `tests/test_mcp_e2e.py` |
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
| `src/agent/server.py` | `tests/test_agent_server.py` |
| `src/dashboard/server.py` | `tests/test_dashboard.py`, `tests/test_dashboard_workspace.py` |
| `src/host/failover.py` | `tests/test_failover.py` |
| `src/host/webhooks.py` | `tests/test_webhooks.py` |
| `src/host/watchers.py` | `tests/test_watchers.py` |
| `src/agent/loop_detector.py` | `tests/test_loop_detector.py` |
| `src/marketplace.py` | `tests/test_marketplace.py` |
| `src/channels/base.py` | `tests/test_channels.py` |
| `src/channels/discord.py` | `tests/test_discord.py` |
| `src/channels/slack.py` | `tests/test_slack.py` |
| `src/channels/whatsapp.py` | `tests/test_whatsapp.py` |
| `src/shared/types.py` | `tests/test_types.py` |
| `src/shared/utils.py` (sanitization) | `tests/test_sanitize.py` |
| `src/cli/` | `tests/test_cli_commands.py`, `tests/test_setup_wizard.py` |
| Cross-component | `tests/test_integration.py`, `tests/test_events.py` |

### Testing Conventions

**Mock LLM responses, not the loop.** Tests create `AgentLoop` with a mock `LLMClient` that returns predetermined `LLMResponse` objects:

```python
async def _make_loop(tool_calls=None, text="Done"):
    llm = AsyncMock()
    llm.chat.return_value = LLMResponse(
        content=text,
        tool_calls=tool_calls or [],
        usage={"prompt_tokens": 10, "completion_tokens": 5},
    )
    loop = AgentLoop(
        agent_id="test",
        llm_client=llm,
        skill_registry=mock_skills,
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
def test_memory_store(tmp_path):
    store = MemoryStore(db_path=str(tmp_path / "test.db"))
    store.save("user_name", "Alice")
    results = store.recall("name")
    assert results[0]["value"] == "Alice"
```

**Every new feature gets tests.** New tools, endpoints, and channel logic all need coverage.

## Code Patterns

### Module Structure

- Small modules, ~800 lines max
- Async by default (`async def` for any I/O)
- `TYPE_CHECKING` imports for circular dependency prevention
- Logging via `setup_logging("component.module")`

```python
from __future__ import annotations

from typing import TYPE_CHECKING

from src.shared.utils import setup_logging

if TYPE_CHECKING:
    from src.agent.mcp_client import MCPClient

logger = setup_logging("agent.skills")
```

### Pydantic for Boundaries

Cross-component messages use Pydantic models from `src/shared/types.py`. Internal data flow within a module can use dicts.

```python
# In src/shared/types.py
class TaskAssignment(BaseModel):
    task_id: str
    workflow_id: str
    step_id: str
    task_type: str
    input_data: dict
    timeout: int = 120

# Used at component boundaries
assignment = TaskAssignment(workflow_id="wf1", step_id="s1", task_type="research", input_data={})
```

### SQLite for All State

Blackboard, agent memory, cost tracking, cron -- all SQLite with WAL mode:

```python
conn = sqlite3.connect(db_path)
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA busy_timeout=5000")
```

No Redis, no external databases.

## Adding New Components

### Adding a Built-in Tool

1. Create `src/agent/builtins/your_tool.py`
2. Use the `@skill` decorator
3. Accept `mesh_client`/`workspace_manager`/`memory_store` as keyword-only args if needed (auto-injected)
4. Return a dict
5. Add tests in `tests/test_builtins.py`

```python
from src.agent.skills import skill

@skill(
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
5. Add tests in `tests/test_mesh.py`

```python
@app.post("/mesh/your_endpoint")
async def your_endpoint(request: YourRequest):
    agent_id = _get_agent_id(request)
    if not permissions.can_do_thing(agent_id):
        raise HTTPException(403, "Not permitted")
    result = do_thing(request)
    return {"status": "ok", "data": result}
```

### Adding a Channel

1. Subclass `Channel` from `src/channels/base.py`
2. Implement `start()`, `stop()`, `send_notification()`
3. The base class `handle_message()` handles all command parsing
4. Add startup logic in `src/cli/channels.py`
5. Add tests in `tests/test_channels.py`

### Adding a Workflow

Create a YAML file in `config/workflows/`:

```yaml
name: my_workflow
trigger: my_event_topic
timeout: 300
steps:
  - id: step1
    agent: researcher
    task_type: research
    input_from: trigger.payload

  - id: step2
    agent: writer
    task_type: write_report
    depends_on: [step1]
    input_from: step1.result
```

The workflow starts when any agent publishes to the `my_event_topic` pub/sub topic.

## Project Structure

```
openlegion/
├── src/
│   ├── agent/                   # Runs inside containers
│   │   ├── __main__.py          # Agent entry point
│   │   ├── server.py            # Agent FastAPI server
│   │   ├── loop.py              # Execution loop (task + chat)
│   │   ├── loop_detector.py     # Tool loop detection (warn/block/terminate)
│   │   ├── skills.py            # Skill registry
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
│   │       ├── browser_tool.py  # Playwright browser
│   │       ├── memory_tool.py   # Memory search/save/recall
│   │       ├── mesh_tool.py     # Blackboard, pub/sub, artifacts, cron
│   │       ├── vault_tool.py    # Credential vault (blind storage)
│   │       ├── introspect_tool.py # Live runtime state queries
│   │       ├── skill_tool.py    # Custom skill creation + reload
│   │       ├── subagent_tool.py # In-process subagent spawning
│   │       └── web_search_tool.py  # DuckDuckGo search
│   ├── host/                    # Runs on the host machine
│   │   ├── server.py            # Mesh FastAPI app
│   │   ├── mesh.py              # Blackboard, PubSub, routing
│   │   ├── orchestrator.py      # DAG workflow executor
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
│   │   ├── watchers.py          # File watchers
│   │   ├── containers.py        # Container management facade
│   │   ├── traces.py            # Request tracing and diagnostics
│   │   └── transcript.py        # Conversation transcript storage
│   ├── channels/                # Messaging adapters
│   │   ├── base.py              # Abstract channel
│   │   ├── telegram.py          # Telegram bot
│   │   ├── discord.py           # Discord bot
│   │   ├── slack.py             # Slack (Socket Mode)
│   │   ├── whatsapp.py          # WhatsApp (Cloud API)
│   │   └── webhook.py           # HTTP webhooks
│   ├── shared/                  # Shared between host and agent
│   │   ├── types.py             # Pydantic contracts
│   │   ├── utils.py             # Logging, ID generation
│   │   └── trace.py             # Distributed trace context
│   ├── dashboard/               # Web dashboard
│   │   ├── server.py            # FastAPI router + API endpoints
│   │   ├── events.py            # EventBus for real-time streaming
│   │   ├── templates/           # Dashboard HTML (Alpine.js + Tailwind)
│   │   └── static/              # CSS + JS assets
│   ├── setup_wizard.py          # Guided setup wizard
│   ├── marketplace.py           # Skill marketplace
│   ├── templates/               # Agent setup templates
│   └── cli/                     # CLI package
│       ├── main.py              # Click commands and entry point
│       ├── config.py            # Config loading, Docker helpers
│       ├── runtime.py           # RuntimeContext lifecycle management
│       ├── repl.py              # REPLSession interactive dispatch
│       ├── channels.py          # ChannelManager messaging lifecycle
│       └── formatting.py        # Tool display and styled output
├── config/                      # Runtime configuration
│   ├── agents.yaml
│   ├── mesh.yaml
│   ├── permissions.json
│   ├── cron.json
│   └── workflows/
├── tests/                       # Test suite (1276 tests)
│   └── fixtures/                # Test fixtures (echo MCP server, etc.)
├── Dockerfile.agent             # Agent container image
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

### Agent-Side (installed in container via Dockerfile)

| Package | Purpose |
|---------|---------|
| `fastapi` + `uvicorn` | Agent HTTP server |
| `httpx` | Mesh API client |
| `pydantic` | Type validation |
| `sqlite-vec` | Vector search for memory |
| `mcp` | Model Context Protocol client |
| `playwright` | Browser automation — basic tier (Chromium) |
| `camoufox` | Anti-detect browser — stealth tier (Firefox) |

### Optional

| Group | Packages | Purpose |
|-------|----------|---------|
| `dev` | `pytest`, `pytest-asyncio`, `ruff` | Testing and linting |
| `mcp` | `mcp>=1.0` | MCP tool support |
| `channels` | `python-telegram-bot`, `discord.py`, `slack-bolt` | Messaging channels |

## Common Mistakes

- **Creating httpx clients per request** -- reuse clients with connection pooling
- **Polling for task completion** -- prefer push-based patterns
- **Breaking tool-call message grouping** -- never split `assistant(tool_calls)` from `tool(result)` messages
- **Putting secrets in agent code** -- use the credential vault
- **Using global mutable state** -- pass state through constructors
- **Overly broad exception handling** -- log errors, distinguish transient from permanent
- **Monolithic functions** -- prefer composable components over growing existing functions
