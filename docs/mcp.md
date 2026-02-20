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
    model: "openai/gpt-4o-mini"
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

1. Host reads `mcp_servers` from agent config in `config/agents.yaml`
2. Host serializes it as JSON into `MCP_SERVERS` environment variable
3. Agent container starts, `__main__.py` reads `MCP_SERVERS`
4. `MCPClient` is created and passed to `SkillRegistry`
5. During lifespan startup, `MCPClient.start()` launches each server:
   - Creates `StdioServerParameters` from config
   - Opens stdio transport via `AsyncExitStack`
   - Establishes `ClientSession` and calls `initialize()`
   - Calls `list_tools()` to discover available tools
6. Tools are registered in `SkillRegistry` alongside built-in skills
7. Agent registers with mesh, reporting MCP tools in its capabilities

### Tool Call Routing

When the LLM calls an MCP tool:

1. `SkillRegistry.execute()` checks `MCPClient.has_tool(name)` first
2. If it's an MCP tool, routes to `MCPClient.call_tool(name, arguments)`
3. `MCPClient` looks up which server provides the tool
4. Sends the call via the MCP session to the correct subprocess
5. Converts the MCP `CallToolResult` to a dict for the LLM

### Name Conflict Resolution

If an MCP tool has the same name as a built-in skill:
- The MCP tool is renamed to `mcp_{server_name}_{tool_name}`
- A warning is logged
- The built-in skill keeps priority

If two MCP servers provide tools with the same name:
- The first server's tool keeps the original name
- The second server's tool is prefixed with `mcp_{server_name}_{tool_name}`

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
- `execute()` -- Routes MCP tools through MCPClient before checking builtins
- `get_tool_definitions()` -- MCP tools included with full JSON Schema

## Popular MCP Servers

Some well-known MCP servers that work with OpenLegion:

| Server | Package | Tools |
|--------|---------|-------|
| Filesystem | `@anthropic/mcp-server-filesystem` | Read/write/search files |
| SQLite | `@anthropic/mcp-server-sqlite` | Query SQLite databases |
| PostgreSQL | `@anthropic/mcp-server-postgres` | Query PostgreSQL databases |
| Brave Search | `@anthropic/mcp-server-brave-search` | Web search via Brave API |
| GitHub | `@anthropic/mcp-server-github` | GitHub API operations |
| Playwright | `@playwright/mcp` | Browser automation (70+ tools) |

**Note:** npm-based MCP servers require Node.js in the agent container. Python-based MCP servers work with the default container image.

## Writing a Custom MCP Server

A minimal Python MCP server using the `mcp` SDK:

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
- Port conflict (shouldn't happen with stdio transport)
- Permission error (file not executable)

### Tools not discovered

- Verify the server runs correctly standalone: `python my_server.py` should start and accept stdin
- Check that `list_tools()` returns tools (use the MCP inspector)
- Look for "MCP server started" log messages

### Name conflicts

If your MCP tool has the same name as a built-in (e.g., `exec_command`), it will be prefixed. Check logs for "conflicts with existing tool, renamed to" warnings.
