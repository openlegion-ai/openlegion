# MCP Integration Guide

OpenLegion supports the **Model Context Protocol (MCP)** -- the emerging standard for LLM tool interoperability. Any MCP-compatible tool server can be plugged into an agent, with tools automatically discovered and exposed to the LLM alongside built-in skills.

## Overview

MCP servers are external processes that expose tools via a standardized protocol. OpenLegion launches them as subprocesses inside agent containers using stdio transport, discovers their tools, and routes LLM tool calls through the MCP protocol.

```
LLM -> tool_call("read_file", {path: "/data/report.csv"})
  -> SkillRegistry.execute()
    -> MCPClient.call_tool()
      -> stdio -> MCP Server subprocess
        -> result
```

## Configuration

Add `mcp_servers` to any agent in `config/agents.yaml`:

```yaml
agents:
  analyst:
    role: "Data analyst"
    model: "openai/gpt-4.1-mini"
    mcp_servers:
      - name: filesystem
        command: mcp-server-filesystem
        args: ["/data"]
      - name: sqlite
        command: mcp-server-sqlite
        args: ["--db", "/data/analytics.db"]
```

### Server Config Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Unique identifier for the server |
| `command` | string | Yes | Command to launch the server |
| `args` | list[string] | No | Command-line arguments |
| `env` | dict[string, string] | No | Environment variables for the server process |

## How It Works

### Startup Sequence

1. The runtime layer (`DockerBackend` in `src/host/runtime.py`) reads `mcp_servers` from agent config in `config/agents.yaml` and serializes it as JSON into the `MCP_SERVERS` environment variable passed to the agent container
2. Agent container starts; `src/agent/__main__.py` reads `MCP_SERVERS`
3. `MCPClient` is created and passed to `SkillRegistry`
4. During lifespan startup, `MCPClient.start()` launches each server:
   - Creates `StdioServerParameters` from config; any `env` dict in the server config is forwarded to the subprocess environment
   - Opens stdio transport via `AsyncExitStack`
   - Establishes `ClientSession` and calls `initialize()`
   - Calls `list_tools()` to discover available tools
5. Tools are registered in `SkillRegistry` alongside built-in skills (name conflicts resolved by renaming at this point)
6. Agent registers with mesh, reporting MCP tools in its capabilities

If a server fails to start at step 4, its tools are not registered but the agent continues normally with built-in skills and any successfully started MCP servers.

### Tool Call Routing

MCP tools are registered into `SkillRegistry` at startup alongside built-in skills. Name conflicts are resolved **at registration time** (not at call time) — if an MCP tool has the same name as a built-in or as another MCP tool, it is renamed to `mcp_{server_name}_{tool_name}` before insertion. This means by the time `execute()` runs, every tool has a unique name and there is no runtime priority check.

When the LLM calls a tool:

1. `SkillRegistry.execute()` looks up the name in the unified skill dict
2. If the entry has `"function": "mcp"`, it routes to `MCPClient.call_tool(name, arguments)`
3. `MCPClient` looks up which server provides the tool via its internal `_tool_to_server` map
4. Sends the call via the MCP session to the correct subprocess
5. Converts the MCP `CallToolResult` to a dict: text content blocks are concatenated under a `"result"` key; image and binary content blocks are silently dropped

Each `call_tool()` call has a **60-second timeout**. If the server does not respond in time, the call returns an error dict.

If an agent has `ALLOWED_TOOLS` configured (operator mode), MCP tool names must appear in that allowlist to be accessible — the restriction applies equally to built-ins and MCP tools.

### Name Conflict Resolution

Conflicts are resolved at registration time, before execution:

- If an MCP tool has the same name as a built-in skill, the MCP tool is renamed to `mcp_{server_name}_{tool_name}` and a warning is logged. The built-in always keeps its original name.
- If two MCP servers provide tools with the same name, the first server's tool keeps the original name and the second server's tool is prefixed with `mcp_{server_name}_{tool_name}`.

After registration every tool has a unique name, so there is no runtime priority resolution.

### Graceful Failure

- If one MCP server fails to start, others still work
- Built-in skills are always available regardless of MCP server status
- Failed servers are logged but don't crash the agent
- If a server crashes mid-session, tool calls return error dicts

### Shutdown

On agent shutdown, `MCPClient.stop()` closes the `AsyncExitStack`, which terminates all server subprocesses and cleans up resources.

## Architecture

### Source Files

| File | Role |
|------|------|
| `src/agent/mcp_client.py` | `MCPClient` class -- server lifecycle, tool discovery, call routing |
| `src/agent/skills.py` | `SkillRegistry` -- MCP tool registration and execution routing |
| `src/agent/__main__.py` | Reads `MCP_SERVERS` env var, creates MCPClient, manages lifespan |
| `src/host/runtime.py` | Passes `mcp_servers` config as container environment variable |
| `src/host/health.py` | Preserves `mcp_servers` across agent restarts |

### Key Classes

**`MCPClient`** (`src/agent/mcp_client.py`):
- `start(servers, builtin_names)` -- Launch servers, discover tools
- `stop()` -- Shut down all servers
- `list_tools()` -- Return discovered tools as schema dicts
- `call_tool(name, arguments)` -- Route call to correct server
- `has_tool(name)` -- Check if a tool exists

**`SkillRegistry`** (`src/agent/skills.py`):
- Accepts optional `mcp_client` parameter
- `_register_mcp_tools()` -- Register MCP tools in the skill dict
- `execute()` -- Dispatches tool calls by name from the unified skill dict; MCP tools are identified by the `"function": "mcp"` marker set at registration time
- `get_tool_definitions()` -- MCP tools included with full JSON Schema

## Popular MCP Servers

Some well-known MCP servers that work with OpenLegion:

| Server | Package | Tools |
|--------|---------|-------|
| Filesystem | `@modelcontextprotocol/server-filesystem` | Read/write/search files |
| SQLite | `@modelcontextprotocol/server-sqlite` | Query SQLite databases |
| PostgreSQL | `@modelcontextprotocol/server-postgres` | Query PostgreSQL databases |
| Brave Search | `@modelcontextprotocol/server-brave-search` | Web search via Brave API |
| GitHub | `@modelcontextprotocol/server-github` | GitHub API operations |
| Playwright | `@playwright/mcp` | Browser automation (70+ tools) |

**Note:** npm-based MCP servers require Node.js in the agent container. Python-based MCP servers work with the default container image — the `mcp` Python SDK is pre-installed in `Dockerfile.agent`. If you use a custom agent image, ensure the `mcp` package is included; without it the `MCPClient` import fails silently (the error is logged and MCP is disabled for that agent).

## Writing a Custom MCP Server

A minimal Python MCP server using the `mcp` SDK (requires `mcp >= 1.0` for the `FastMCP` high-level API):

```python
from mcp.server import FastMCP

server = FastMCP("my-tools")

@server.tool()
def analyze(data: str, format: str = "json") -> str:
    """Analyze data in the specified format."""
    # Your logic here
    return f"Analysis of {len(data)} bytes in {format} format"

if __name__ == "__main__":
    server.run(transport="stdio")
```

Configure it in `config/agents.yaml`:

```yaml
mcp_servers:
  - name: my-tools
    command: python
    args: ["/app/tools/my_server.py"]
```

## Troubleshooting

### Server fails to start

Check agent container logs:
```bash
openlegion status  # or docker logs openlegion_<agent_id>
```

Common causes:
- Command not found in container (missing dependency)
- Permission error (file not executable)
- Missing environment variable the server requires (add it via the `env` config field)

### Tools not discovered

- Verify the server runs correctly standalone: `python my_server.py` should start and accept stdin
- Check that `list_tools()` returns tools (use the MCP inspector)
- Look for "MCP server started" log messages

### Name conflicts

If your MCP tool has the same name as a built-in (e.g., `exec_command`), it will be prefixed. Check logs for "conflicts with existing tool, renamed to" warnings.
