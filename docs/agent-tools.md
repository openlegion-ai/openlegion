# Agent Tools Reference

Agents interact with their environment through **skills** -- Python functions registered via the `@skill` decorator. Skills are auto-discovered at startup from built-in modules and custom skill directories.

## Built-in Tools

### Shell Execution

| Tool | Parameters | Description |
|------|-----------|-------------|
| `exec` | `command`, `workdir`, `timeout` | Execute shell commands with full Linux environment |

### File Operations

All file operations are scoped to `/data` inside the container. Path traversal is blocked.

| Tool | Parameters | Description |
|------|-----------|-------------|
| `read_file` | `path`, `offset`, `limit` | Read file contents with optional pagination |
| `write_file` | `path`, `content`, `append` | Write or append to a file (creates directories) |
| `list_files` | `path`, `pattern`, `recursive` | List files with optional glob pattern matching (recursive: boolean, default false) |

### HTTP

| Tool | Parameters | Description |
|------|-----------|-------------|
| `http_request` | `url`, `method`, `headers`, `body`, `timeout` | Make HTTP requests (GET/POST/PUT/DELETE/PATCH). Supports `$CRED{name}` handles in URL, headers, and body for credential-blind API calls. Resolved credentials are redacted from responses. Timeout default: 30s. SSRF protection blocks requests to private/internal addresses (loopback, link-local, reserved ranges) including redirect targets. |

### Browser Automation

All agents share a single **browser service container** running Camoufox (a stealth Firefox fork) with KasmVNC. The browser service runs separately from agent containers — agents send browser commands via HTTP to the shared service. KasmVNC serves a live browser view on port 6080 (accessible via the dashboard). Browser profiles are preserved across sessions, maintaining login sessions and cookies.

| Tool | Parameters | Description |
|------|-----------|-------------|
| `browser_navigate` | `url`, `wait_ms` | Open URL, wait, extract page text (default wait: 1000ms). |
| `browser_snapshot` | -- | Accessibility tree snapshot with element refs (e1, e2, ...) |
| `browser_screenshot` | `full_page` | Take screenshot, return base64 PNG. |
| `browser_click` | `ref` or `selector` | Click element by accessibility ref or CSS selector |
| `browser_type` | `ref` or `selector`, `text` | Type into input field (supports `$CRED{name}` handles) |
| `browser_scroll` | `direction`, `amount`, `ref` | Scroll page up/down or scroll element into view. Default direction: `down`, default amount: one viewport height. |
| `browser_reset` | -- | Reset browser session (profile preserved) |
| `browser_solve_captcha` | -- | Manual CAPTCHA detection and solving. Usually not needed — `browser_navigate` auto-detects CAPTCHAs. |

### Memory

| Tool | Parameters | Description |
|------|-----------|-------------|
| `memory_search` | `query`, `category`, `max_results` | Hybrid search across workspace files and structured DB. Provide `category` to search only the fact database filtered to that category. |
| `memory_save` | `content` | Save a fact to workspace daily log and structured memory |

### Mesh / Fleet

| Tool | Parameters | Description |
|------|-----------|-------------|
| `list_agents` | -- | Discover agents in your project (standalone agents see only themselves) |
| `read_shared_state` | `key` | Read from the project-scoped blackboard. Keys are auto-namespaced under `projects/{name}/` — agents use natural keys like `tasks/abc` |
| `write_shared_state` | `key`, `value` | Write to the project-scoped blackboard |
| `list_shared_state` | `prefix` | Browse project blackboard entries by prefix |
| `publish_event` | `topic`, `data` | Publish event to mesh pub/sub |
| `subscribe_event` | `topic` | Subscribe to a pub/sub topic at runtime. Events arrive as steer messages between tool rounds. |
| `watch_blackboard` | `pattern` | Watch blackboard keys matching a glob pattern. Notifications arrive when matching keys are written. |
| `claim_task` | `key`, `claim_value` | Atomically claim a task from the blackboard (compare-and-swap). Prevents duplicate work. |
| `save_artifact` | `name`, `content` | Save deliverable to workspace and register on project blackboard |
| `notify_user` | `message` | Send a notification to the user across all connected channels (CLI, Telegram, Discord, Slack, etc.) |
| `read_agent_history` | `agent_id` | Read another agent's conversation logs (permission-checked) |

**Project isolation:** Blackboard tools (`read_shared_state`, `write_shared_state`, `list_shared_state`, `save_artifact`) are only available to agents assigned to a project. Standalone agents cannot access the blackboard — calls return an error explaining they must be added to a project first.

### Workspace

| Tool | Parameters | Description |
|------|-----------|-------------|
| `update_workspace` | `filename`, `content` | Update a writable workspace file (SOUL.md, INSTRUCTIONS.md, USER.md, or HEARTBEAT.md) to persist learnings across sessions |

### Scheduling & Automation

| Tool | Parameters | Description |
|------|-----------|-------------|
| `set_cron` | `schedule`, `message`, `heartbeat` | Schedule a recurring job (cron expression or interval). Set `heartbeat=true` to update your autonomous wakeup schedule. |
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
| `reload_skills` | -- | Hot-reload all skills from disk |
| `spawn_agent` | `role`, `system_prompt`, `ttl` | Spawn an ephemeral sub-agent in a new container (default TTL: 3600s) |

### Subagents (In-Container)

Lightweight subagents that run inside the same process as the parent agent, sharing LLM and mesh clients but with their own memory and workspace.

**Limits:** Max 3 concurrent subagents, max depth 2 (no grandchildren), default TTL 300s (max TTL 600s), max 10 iterations per subagent. Subagents cannot use `create_skill`, `reload_skills`, `spawn_subagent`, or `wait_for_subagent` (prevents recursion and nesting). Results are written to blackboard at `subagent_results/{parent_id}/{subagent_id}`.

| Tool | Parameters | Description |
|------|-----------|-------------|
| `spawn_subagent` | `task`, `role`, `ttl_seconds` | Spawn a lightweight subagent for parallel subtask execution |
| `list_subagents` | -- | List active subagents spawned by this agent and their status |
| `wait_for_subagent` | `subagent_id`, `timeout` | Wait for a subagent to complete and return its result |

### Credential Vault

Agents never see credential values. All operations return opaque `$CRED{name}` handles. Agents can only access credentials allowed by their `allowed_credentials` patterns in `config/permissions.json`. System credentials (LLM provider API keys) are never accessible to agents.

| Tool | Parameters | Description |
|------|-----------|-------------|
| `vault_generate_secret` | `name`, `length`, `charset` | Generate a random secret and store it (returns handle only) |
| `vault_capture_from_page` | `name`, `selector` or `ref` | Read text from a browser element and store as credential |
| `vault_list` | -- | List credential names the agent can access (names only, filtered by permissions) |

### System Introspection

| Tool | Parameters | Description |
|------|-----------|-------------|
| `introspect` | `section` | Query live runtime state: permissions, budget, fleet, cron, health, or all |

Agents receive system awareness through three layers:

1. **SYSTEM.md** — Generated at startup and refreshed every 5 minutes. Contains a static architecture guide (mesh concepts, context window mechanics, tool cost model, common errors) plus a compact snapshot of permissions and fleet. Loaded into the system prompt via workspace bootstrap.
2. **Runtime Context** — A compact block injected into the system prompt on each turn (5-minute cache). Shows live budget numbers, permission patterns, fleet roster, and cron schedule.
3. **`introspect` tool** — On-demand access to fresh data when agents need exact numbers mid-conversation.

The `section` parameter accepts: `permissions`, `budget`, `fleet`, `cron`, `health`, or `all` (default). Fleet data is filtered by `can_message` permissions — agents only see teammates they can interact with.

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
