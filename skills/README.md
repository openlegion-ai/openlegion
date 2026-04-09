# Skills

Skills are Python functions that give agents custom capabilities beyond the
built-in tools (shell, file I/O, HTTP, browser, memory, mesh communication).

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

## Built-in Tools

Every agent automatically has access to built-in tools defined in
`src/agent/builtins/`. You don't need to add these to your skills directory:

- `run_command` — Shell command execution
- `read_file`, `write_file`, `list_files` — File I/O
- `http_request` — HTTP requests with `$CRED{name}` handles
- `browser_navigate`, `browser_get_elements`, `browser_screenshot`, `browser_click`, `browser_type`, `browser_hover`, `browser_scroll`, `browser_wait_for`, `browser_press_key`, `browser_go_back`, `browser_go_forward`, `browser_switch_tab`, `browser_reset`, `browser_detect_captcha` — Browser automation
- `memory_search`, `memory_save` — Persistent memory
- `web_search` — Web search via DuckDuckGo
- `list_agents`, `spawn_fleet_agent`, `notify_user`, `publish_event`, `subscribe_event` — Fleet coordination
- `read_shared_state`, `write_shared_state`, `list_shared_state`, `watch_blackboard`, `claim_task`, `save_artifact` — Shared blackboard
- `update_workspace` — Persist learnings to workspace files
- `spawn_subagent`, `list_subagents`, `wait_for_subagent` — In-process subagents
- `set_cron`, `list_cron`, `remove_cron` — Scheduled jobs
- `vault_generate_secret`, `vault_list` — Credential vault
- `get_system_status` — Runtime state queries (permissions, budget, fleet, cron, health)
- `read_agent_history` — Read another agent's conversation logs
- `create_skill`, `reload_skills` — Self-extension
