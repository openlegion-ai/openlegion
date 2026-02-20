"""MCP (Model Context Protocol) client for agent containers.

Manages MCP server lifecycles via stdio transport. Each agent can connect
to multiple MCP servers, discover their tools, and route tool calls through
the MCP protocol. Tools are exposed to the LLM alongside built-in skills.
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Any

from src.shared.utils import setup_logging

logger = setup_logging("agent.mcp")

# Lazy imports — mcp SDK is only installed in agent containers.
# These get populated on first use in start() and can be patched in tests.
try:
    from mcp import StdioServerParameters
    from mcp.client.session import ClientSession
    from mcp.client.stdio import stdio_client
except ImportError:  # pragma: no cover
    StdioServerParameters = None  # type: ignore[assignment,misc]
    ClientSession = None  # type: ignore[assignment,misc]
    stdio_client = None  # type: ignore[assignment]


class MCPClient:
    """Manages multiple MCP server connections via stdio transport."""

    def __init__(self) -> None:
        self._exit_stack = AsyncExitStack()
        self._sessions: dict[str, Any] = {}  # server_name → ClientSession
        self._tool_to_server: dict[str, str] = {}  # tool_name → server_name
        self._tool_schemas: dict[str, dict] = {}  # tool_name → schema dict

    async def start(self, servers: list[dict], builtin_names: set[str] | None = None) -> None:
        """Start MCP server subprocesses and discover their tools.

        Args:
            servers: List of server configs, each with keys:
                name (str): Server identifier
                command (str): Command to run
                args (list[str], optional): Command arguments
                env (dict[str, str], optional): Environment variables
            builtin_names: Set of built-in skill names for conflict detection.
        """
        if StdioServerParameters is None:
            logger.error("MCP SDK not installed — cannot start MCP servers")
            return

        builtin_names = builtin_names or set()

        for server_cfg in servers:
            name = server_cfg["name"]
            try:
                params = StdioServerParameters(
                    command=server_cfg["command"],
                    args=server_cfg.get("args", []),
                    env=server_cfg.get("env"),
                )

                transport = await self._exit_stack.enter_async_context(
                    stdio_client(params)
                )
                read_stream, write_stream = transport
                session = await self._exit_stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )
                await session.initialize()

                tools_result = await session.list_tools()
                self._sessions[name] = session

                for tool in tools_result.tools:
                    tool_name = tool.name
                    if tool_name in builtin_names or tool_name in self._tool_to_server:
                        prefixed = f"mcp_{name}_{tool_name}"
                        logger.warning(
                            f"MCP tool '{tool_name}' from server '{name}' "
                            f"conflicts with existing tool, renamed to '{prefixed}'"
                        )
                        tool_name = prefixed

                    self._tool_to_server[tool_name] = name
                    self._tool_schemas[tool_name] = {
                        "name": tool_name,
                        "description": tool.description or "",
                        "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                        "function": "mcp",
                        "_mcp_original_name": tool.name,
                    }

                logger.info(
                    f"MCP server '{name}' started: "
                    f"{len(tools_result.tools)} tools discovered"
                )
            except Exception as e:
                logger.error(f"Failed to start MCP server '{name}': {e}")

    async def stop(self) -> None:
        """Shut down all MCP server connections."""
        try:
            await self._exit_stack.aclose()
        except Exception as e:
            logger.warning(f"Error closing MCP connections: {e}")
        self._sessions.clear()
        self._tool_to_server.clear()
        self._tool_schemas.clear()

    def list_tools(self) -> list[dict]:
        """Return all discovered MCP tools as schema dicts."""
        return list(self._tool_schemas.values())

    async def call_tool(self, name: str, arguments: dict) -> dict:
        """Route a tool call to the correct MCP server.

        Returns:
            dict with 'result' key on success, or 'error' key on failure.
        """
        server_name = self._tool_to_server.get(name)
        if not server_name:
            return {"error": f"Unknown MCP tool: {name}"}

        session = self._sessions.get(server_name)
        if not session:
            return {"error": f"MCP server '{server_name}' not connected"}

        original_name = self._tool_schemas[name].get("_mcp_original_name", name)

        try:
            result = await session.call_tool(original_name, arguments)
            if result.isError:
                text = "\n".join(
                    block.text for block in result.content
                    if hasattr(block, "text")
                )
                return {"error": text or "MCP tool returned an error"}
            text_parts = []
            for block in result.content:
                if hasattr(block, "text"):
                    text_parts.append(block.text)
            return {"result": "\n".join(text_parts)}
        except Exception as e:
            logger.error(f"MCP tool '{name}' call failed: {e}")
            return {"error": str(e)}

    def has_tool(self, name: str) -> bool:
        """Check if a tool is provided by an MCP server."""
        return name in self._tool_to_server
