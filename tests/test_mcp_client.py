"""Tests for MCP (Model Context Protocol) client integration."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.mcp_client import MCPClient


def _make_mock_tool(name: str, description: str = "", input_schema: dict | None = None):
    """Create a mock MCP Tool object."""
    tool = MagicMock()
    tool.name = name
    tool.description = description or f"Mock tool: {name}"
    tool.inputSchema = input_schema or {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path"},
        },
        "required": ["path"],
    }
    return tool


def _make_mock_result(text: str, is_error: bool = False):
    """Create a mock MCP CallToolResult."""
    result = MagicMock()
    result.isError = is_error
    block = MagicMock()
    block.text = text
    result.content = [block]
    return result


def _mcp_patches():
    """Return a combined patch context for all MCP SDK symbols."""
    return (
        patch("src.agent.mcp_client.StdioServerParameters", MagicMock()),
        patch("src.agent.mcp_client.stdio_client"),
        patch("src.agent.mcp_client.ClientSession"),
    )


def _setup_mock_server(mock_stdio, mock_cs_cls, mock_session):
    """Wire up mock stdio_client and ClientSession to return mock_session."""
    mock_transport_cm = AsyncMock()
    mock_transport_cm.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
    mock_stdio.return_value = mock_transport_cm

    mock_session_cm = AsyncMock()
    mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
    mock_cs_cls.return_value = mock_session_cm


class TestMCPClientListTools:
    @pytest.mark.asyncio
    async def test_list_tools_returns_openai_format(self):
        """MCP tools are returned in OpenAI function-calling schema format."""
        client = MCPClient()

        mock_session = AsyncMock()
        mock_tools_result = MagicMock()
        mock_tools_result.tools = [
            _make_mock_tool("read_file", "Read a file", {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                },
                "required": ["path"],
            }),
            _make_mock_tool("write_file", "Write a file", {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "Content"},
                },
                "required": ["path", "content"],
            }),
        ]
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=mock_tools_result)

        p1, p2, p3 = _mcp_patches()
        with p1, p2 as mock_stdio, p3 as mock_cs_cls:
            _setup_mock_server(mock_stdio, mock_cs_cls, mock_session)
            await client.start([{"name": "fs", "command": "mcp-server-fs", "args": ["/data"]}])

        tools = client.list_tools()
        assert len(tools) == 2

        read_tool = next(t for t in tools if t["name"] == "read_file")
        assert read_tool["description"] == "Read a file"
        assert read_tool["parameters"]["type"] == "object"
        assert "path" in read_tool["parameters"]["properties"]
        assert read_tool["function"] == "mcp"


class TestMCPClientCallTool:
    @pytest.mark.asyncio
    async def test_call_tool_routes_to_correct_server(self):
        """Tool calls route to the correct MCP server by name."""
        client = MCPClient()

        # Set up two servers manually
        session_a = AsyncMock()
        session_b = AsyncMock()
        client._sessions = {"server_a": session_a, "server_b": session_b}
        client._tool_to_server = {"tool_a": "server_a", "tool_b": "server_b"}
        client._tool_schemas = {
            "tool_a": {"name": "tool_a", "description": "", "parameters": {}, "function": "mcp", "_mcp_original_name": "tool_a"},
            "tool_b": {"name": "tool_b", "description": "", "parameters": {}, "function": "mcp", "_mcp_original_name": "tool_b"},
        }

        session_a.call_tool = AsyncMock(return_value=_make_mock_result("result_a"))
        session_b.call_tool = AsyncMock(return_value=_make_mock_result("result_b"))

        result_a = await client.call_tool("tool_a", {"arg": "val"})
        assert result_a == {"result": "result_a"}
        session_a.call_tool.assert_awaited_once_with("tool_a", {"arg": "val"})
        session_b.call_tool.assert_not_awaited()

        result_b = await client.call_tool("tool_b", {"arg": "val2"})
        assert result_b == {"result": "result_b"}
        session_b.call_tool.assert_awaited_once_with("tool_b", {"arg": "val2"})

    @pytest.mark.asyncio
    async def test_call_tool_handles_error(self):
        """MCP result with isError=True returns error dict."""
        client = MCPClient()

        session = AsyncMock()
        client._sessions = {"srv": session}
        client._tool_to_server = {"broken": "srv"}
        client._tool_schemas = {
            "broken": {"name": "broken", "description": "", "parameters": {}, "function": "mcp", "_mcp_original_name": "broken"},
        }

        session.call_tool = AsyncMock(
            return_value=_make_mock_result("something went wrong", is_error=True)
        )

        result = await client.call_tool("broken", {})
        assert "error" in result
        assert "something went wrong" in result["error"]

    @pytest.mark.asyncio
    async def test_call_tool_unknown_tool(self):
        """Calling an unknown MCP tool returns an error dict."""
        client = MCPClient()
        result = await client.call_tool("nonexistent", {})
        assert "error" in result
        assert "Unknown MCP tool" in result["error"]

    @pytest.mark.asyncio
    async def test_call_tool_disconnected_server(self):
        """Calling a tool whose server session is missing returns error."""
        client = MCPClient()
        client._tool_to_server = {"orphan": "dead_server"}
        client._tool_schemas = {
            "orphan": {"name": "orphan", "description": "", "parameters": {}, "function": "mcp", "_mcp_original_name": "orphan"},
        }
        # No session for "dead_server"

        result = await client.call_tool("orphan", {})
        assert "error" in result
        assert "not connected" in result["error"]


class TestMCPClientNameConflict:
    @pytest.mark.asyncio
    async def test_tool_name_conflict_prefixed(self):
        """When an MCP tool name collides with a builtin, it gets prefixed."""
        client = MCPClient()

        mock_session = AsyncMock()
        mock_tools_result = MagicMock()
        mock_tools_result.tools = [
            _make_mock_tool("exec_command"),  # conflicts with builtin
        ]
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=mock_tools_result)

        p1, p2, p3 = _mcp_patches()
        with p1, p2 as mock_stdio, p3 as mock_cs_cls:
            _setup_mock_server(mock_stdio, mock_cs_cls, mock_session)
            await client.start(
                [{"name": "myserver", "command": "mcp-server"}],
                builtin_names={"exec_command"},
            )

        assert client.has_tool("mcp_myserver_exec_command")
        assert not client.has_tool("exec_command")

    @pytest.mark.asyncio
    async def test_tool_name_conflict_between_servers(self):
        """When two MCP servers provide same tool name, second gets prefixed."""
        client = MCPClient()

        mock_session = AsyncMock()
        mock_tools_result = MagicMock()
        mock_tools_result.tools = [_make_mock_tool("search")]
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=mock_tools_result)

        p1, p2, p3 = _mcp_patches()
        with p1, p2 as mock_stdio, p3 as mock_cs_cls:
            _setup_mock_server(mock_stdio, mock_cs_cls, mock_session)
            await client.start([
                {"name": "srv1", "command": "cmd1"},
                {"name": "srv2", "command": "cmd2"},
            ])

        # First gets the original name, second gets prefixed
        assert client.has_tool("search")
        assert client.has_tool("mcp_srv2_search")


class TestMCPClientServerFailure:
    @pytest.mark.asyncio
    async def test_start_server_failure_graceful(self):
        """When one server fails to start, others still work."""
        client = MCPClient()

        good_session = AsyncMock()
        good_tools_result = MagicMock()
        good_tools_result.tools = [_make_mock_tool("good_tool")]
        good_session.initialize = AsyncMock()
        good_session.list_tools = AsyncMock(return_value=good_tools_result)

        call_count = 0

        def mock_stdio_side_effect(params):
            nonlocal call_count
            call_count += 1
            cm = AsyncMock()
            if call_count == 1:
                # First server fails during enter_async_context
                cm.__aenter__ = AsyncMock(side_effect=RuntimeError("Server crashed"))
            else:
                cm.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
            return cm

        p1 = patch("src.agent.mcp_client.StdioServerParameters", MagicMock())
        p2 = patch("src.agent.mcp_client.stdio_client", side_effect=mock_stdio_side_effect)
        p3 = patch("src.agent.mcp_client.ClientSession")

        with p1, p2, p3 as mock_cs_cls:
            mock_session_cm = AsyncMock()
            mock_session_cm.__aenter__ = AsyncMock(return_value=good_session)
            mock_cs_cls.return_value = mock_session_cm

            await client.start([
                {"name": "bad_server", "command": "bad-cmd"},
                {"name": "good_server", "command": "good-cmd"},
            ])

        assert client.has_tool("good_tool")


class TestMCPClientLifecycle:
    @pytest.mark.asyncio
    async def test_stop_cleans_up(self):
        """stop() closes the AsyncExitStack and clears state."""
        client = MCPClient()
        client._sessions = {"test": MagicMock()}
        client._tool_to_server = {"tool": "test"}
        client._tool_schemas = {"tool": {"name": "tool"}}

        client._exit_stack = AsyncMock()
        client._exit_stack.aclose = AsyncMock()

        await client.stop()

        client._exit_stack.aclose.assert_awaited_once()
        assert len(client._sessions) == 0
        assert len(client._tool_to_server) == 0
        assert len(client._tool_schemas) == 0

    def test_has_tool(self):
        """has_tool returns True only for registered MCP tools."""
        client = MCPClient()
        client._tool_to_server = {"mcp_tool": "server"}

        assert client.has_tool("mcp_tool")
        assert not client.has_tool("other_tool")
