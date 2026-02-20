# Agent Tools Reference

Agents interact with their environment through **skills** -- Python functions registered via the `@skill` decorator. Skills are auto-discovered at startup from built-in modules and custom skill directories.

## Built-in Tools

### Shell Execution

| Tool | Parameters | Description |
|------|-----------|-------------|
| `exec_command` | `command`, `workdir`, `timeout` | Execute shell commands with full Linux environment |

### File Operations

All file operations are scoped to `/data` inside the container. Path traversal is blocked.

| Tool | Parameters | Description |
|------|-----------|-------------|
| `read_file` | `path`, `offset`, `limit` | Read file contents with optional pagination |
| `write_file` | `path`, `content`, `append` | Write or append to a file (creates directories) |
| `list_files` | `path`, `pattern` | List files with optional glob pattern matching |

### HTTP

| Tool | Parameters | Description |
|------|-----------|-------------|
| `http_request` | `url`, `method`, `headers`, `body` | Make HTTP requests (GET/POST/PUT/DELETE/PATCH) |

### Browser Automation

Powered by Playwright with headless Chromium pre-installed in the agent container.

| Tool | Parameters | Description |
|------|-----------|-------------|
| `browser_navigate` | `url`, `wait_ms` | Open URL, wait, extract page text (default wait: 1000ms) |
| `browser_snapshot` | -- | Accessibility tree snapshot with element refs (e1, e2, ...) |
| `browser_screenshot` | `filename`, `full_page` | Save screenshot to /data (default: screenshot.png) |
| `browser_click` | `ref` or `selector` | Click element by accessibility ref or CSS selector |
| `browser_type` | `ref` or `selector`, `text` | Type into input field (supports `$CRED{name}` handles) |
| `browser_evaluate` | `script` | Run JavaScript in page context |

### Memory

| Tool | Parameters | Description |
|------|-----------|-------------|
| `memory_search` | `query`, `max_results` | Hybrid search across workspace files and structured DB |
| `memory_save` | `content` | Save a fact to workspace daily log and structured memory |
| `memory_recall` | `query`, `category`, `max_results` | Semantic search with optional category filtering |

### Mesh / Fleet

| Tool | Parameters | Description |
|------|-----------|-------------|
| `list_agents` | -- | Discover other agents in the fleet |
| `read_shared_state` | `key` | Read from the shared blackboard |
| `write_shared_state` | `key`, `value` | Write to the shared blackboard |
| `list_shared_state` | `prefix` | Browse blackboard entries by prefix |
| `publish_event` | `topic`, `data` | Publish event to mesh pub/sub |
| `save_artifact` | `name`, `content` | Save deliverable to workspace and register on blackboard |

### Scheduling & Automation

| Tool | Parameters | Description |
|------|-----------|-------------|
| `set_cron` | `schedule`, `message` | Schedule a recurring job (cron expression or interval) |
| `set_heartbeat` | `schedule` | Enable autonomous monitoring (probes run automatically) |
| `list_cron` | -- | List scheduled jobs |
| `remove_cron` | `job_id` | Remove a scheduled job |

### Web Search

| Tool | Parameters | Description |
|------|-----------|-------------|
| `web_search` | `query`, `max_results` | Search via DuckDuckGo (no API key needed) |

### Self-Extension

| Tool | Parameters | Description |
|------|-----------|-------------|
| `create_skill` | `name`, `code` | Write a new Python skill at runtime |
| `list_custom_skills` | -- | List all custom skills the agent has created |
| `reload_skills` | -- | Hot-reload all skills from disk |
| `spawn_agent` | `role`, `system_prompt`, `ttl` | Spawn an ephemeral sub-agent (default TTL: 3600s) |

### Credential Vault

Agents never see credential values. All operations return opaque `$CRED{name}` handles.

| Tool | Parameters | Description |
|------|-----------|-------------|
| `vault_generate_secret` | `name`, `length`, `charset` | Generate a random secret and store it (returns handle only) |
| `vault_capture_from_page` | `name`, `selector` or `ref` | Read text from a browser element and store as credential |
| `vault_list` | -- | List credential names stored in the vault (names only) |
| `vault_status` | `name` | Check if a credential exists in the vault |

### Agent History

| Tool | Parameters | Description |
|------|-----------|-------------|
| `read_agent_history` | `agent_id` | Read another agent's conversation logs (permission-checked) |

## MCP Tools

Agents can also use tools from external MCP (Model Context Protocol) servers. These are configured per-agent in `config/agents.yaml` and discovered automatically at startup. See [MCP Integration](mcp.md) for details.

## Custom Skills

Agents can create and load custom skills at runtime:

```python
from src.agent.skills import skill

@skill(
    name="analyze_csv",
    description="Parse and analyze a CSV file",
    parameters={
        "path": {"type": "string", "description": "Path to CSV file"},
        "query": {"type": "string", "description": "Analysis question"},
    },
)
async def analyze_csv(path: str, query: str, *, workspace_manager=None) -> dict:
    # Implementation here
    return {"result": "analysis output"}
```

### Auto-Injected Dependencies

Skills can request these keyword-only arguments (auto-injected by SkillRegistry):

| Argument | Type | Provides |
|----------|------|----------|
| `mesh_client` | `MeshClient` | Access to mesh APIs (blackboard, pub/sub) |
| `workspace_manager` | `WorkspaceManager` | Access to workspace files |
| `memory_store` | `MemoryStore` | Access to structured memory DB |
