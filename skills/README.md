# Skills

Skills are Python functions that give agents custom capabilities beyond the
built-in tools (exec, file I/O, HTTP, browser, memory, mesh communication).

## How Skills Work

Each agent has a `skills_dir` in its config. At startup, the agent auto-discovers
all `.py` files in that directory and registers any functions decorated with `@skill`.

```python
from src.agent.skills import skill

@skill(
    name="my_tool",
    description="What this tool does (shown to the LLM)",
    parameters={
        "query": {"type": "string", "description": "The search query"},
        "limit": {"type": "integer", "description": "Max results", "default": 5},
    },
)
async def my_tool(query: str, limit: int = 5, *, mesh_client=None):
    """mesh_client is injected automatically if declared in the signature."""
    # Use mesh_client.api_call() for external APIs (routed through the vault)
    response = await mesh_client.api_call(
        service="my_service",
        action="search",
        params={"q": query, "limit": limit},
    )
    return response.data if response.success else {"error": response.error}
```

## Key Points

- Skills can be **sync or async**. Sync functions run in a thread executor.
- Declare `mesh_client` or `workspace_manager` as keyword-only parameters
  to have them injected at call time.
- **Never call external APIs directly.** Use `mesh_client.api_call()` which
  routes through the credential vault. Agents never see API keys.
- Parameters without a `"default"` key are required.
- The `description` is what the LLM reads to decide when to use the tool.

## Example Skills in This Directory

The skills in `research/`, `qualify/`, and `outreach/` are **stub examples**
showing the pattern for different use cases. They route through the mesh API
proxy but the external services (Apollo, Hunter, etc.) require API keys
configured in your `.env` file.

## Built-in Tools

Every agent automatically has access to built-in tools defined in
`src/agent/builtins/`. You don't need to add these to your skills directory:

- `exec` — Shell command execution
- `read_file`, `write_file`, `list_files` — File I/O
- `http_request` — HTTP requests
- `browser_navigate`, `browser_screenshot`, etc. — Browser automation
- `memory_search`, `memory_save` — Persistent memory
- `delegate_task`, `list_agents`, `send_message` — Team coordination
- `read_shared_state`, `write_shared_state`, `list_shared_state` — Shared blackboard
