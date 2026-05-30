"""Tests for MCP (Model Context Protocol) client integration."""

import asyncio
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
            "tool_a": {
                "name": "tool_a", "description": "", "parameters": {},
                "function": "mcp", "_mcp_original_name": "tool_a",
            },
            "tool_b": {
                "name": "tool_b", "description": "", "parameters": {},
                "function": "mcp", "_mcp_original_name": "tool_b",
            },
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
            "broken": {
                "name": "broken", "description": "", "parameters": {},
                "function": "mcp", "_mcp_original_name": "broken",
            },
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
            "orphan": {
                "name": "orphan", "description": "", "parameters": {},
                "function": "mcp", "_mcp_original_name": "orphan",
            },
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


class TestMCPMetadataHardening:
    """M12: tool metadata from a (possibly malicious/rug-pulled) MCP server
    is sanitized + length-capped before it can enter the LLM tool payload.
    sanitize_for_prompt is lossless for normal text; the cap is the DoS
    control. Schema string values are sanitized but keys/structure stay
    intact so tool-calling is not broken.
    """

    @pytest.mark.asyncio
    async def test_description_sanitized_and_length_capped(self):
        from src.agent.mcp_client import _MCP_DESCRIPTION_MAX_BYTES

        # Invisible/control chars (zero-width space U+200B, BOM U+FEFF) plus
        # a description far larger than the cap.
        poisoned = "Read​a﻿file " + ("A" * (_MCP_DESCRIPTION_MAX_BYTES + 5000))

        client = MCPClient()
        session = AsyncMock()
        tools_result = MagicMock()
        tools_result.tools = [_make_mock_tool("read_file", poisoned)]
        session.initialize = AsyncMock()
        session.list_tools = AsyncMock(return_value=tools_result)

        p1, p2, p3 = _mcp_patches()
        with p1, p2 as mock_stdio, p3 as mock_cs_cls:
            _setup_mock_server(mock_stdio, mock_cs_cls, session)
            await client.start([{"name": "fs", "command": "x"}])

        desc = client.list_tools()[0]["description"]
        # Invisible chars stripped (lossless for the legit text around them).
        assert "​" not in desc
        assert "﻿" not in desc
        assert desc.startswith("Readafile ")
        # Length-capped well under the original size.
        assert len(desc.encode("utf-8")) <= _MCP_DESCRIPTION_MAX_BYTES
        assert desc.endswith("… [truncated]")

    @pytest.mark.asyncio
    async def test_legit_description_unaffected(self):
        """A normal description passes through unchanged (lossless)."""
        client = MCPClient()
        session = AsyncMock()
        tools_result = MagicMock()
        tools_result.tools = [
            _make_mock_tool("read_file", "Read a file from disk and return contents."),
        ]
        session.initialize = AsyncMock()
        session.list_tools = AsyncMock(return_value=tools_result)

        p1, p2, p3 = _mcp_patches()
        with p1, p2 as mock_stdio, p3 as mock_cs_cls:
            _setup_mock_server(mock_stdio, mock_cs_cls, session)
            await client.start([{"name": "fs", "command": "x"}])

        assert (
            client.list_tools()[0]["description"]
            == "Read a file from disk and return contents."
        )

    @pytest.mark.asyncio
    async def test_schema_string_values_sanitized_keys_and_structure_intact(self):
        client = MCPClient()
        session = AsyncMock()
        tools_result = MagicMock()
        tools_result.tools = [
            _make_mock_tool(
                "search",
                "Search",
                {
                    "type": "object",
                    "properties": {
                        # Key contains a name; value description is poisoned.
                        "query": {
                            "type": "string",
                            "description": "The​search query",
                            "title": "Que﻿ry",
                        },
                        "limit": {"type": "integer", "description": "Max​results"},
                    },
                    "required": ["query"],
                },
            ),
        ]
        session.initialize = AsyncMock()
        session.list_tools = AsyncMock(return_value=tools_result)

        p1, p2, p3 = _mcp_patches()
        with p1, p2 as mock_stdio, p3 as mock_cs_cls:
            _setup_mock_server(mock_stdio, mock_cs_cls, session)
            await client.start([{"name": "srv", "command": "x"}])

        params = client.list_tools()[0]["parameters"]
        props = params["properties"]
        # Structure + keys preserved exactly.
        assert params["type"] == "object"
        assert params["required"] == ["query"]
        assert set(props.keys()) == {"query", "limit"}
        assert props["query"]["type"] == "string"
        assert props["limit"]["type"] == "integer"
        # String VALUES sanitized (invisible chars stripped).
        assert props["query"]["description"] == "Thesearch query"
        assert props["query"]["title"] == "Query"
        assert props["limit"]["description"] == "Maxresults"

    @pytest.mark.asyncio
    async def test_invisible_char_in_tool_name_stripped_and_routable(self):
        """A poisoned tool name is sanitized and the routing key matches the
        emitted name, so call_tool() still resolves it.
        """
        client = MCPClient()
        session = AsyncMock()
        tools_result = MagicMock()
        tools_result.tools = [_make_mock_tool("read​file", "Read")]
        session.initialize = AsyncMock()
        session.list_tools = AsyncMock(return_value=tools_result)

        p1, p2, p3 = _mcp_patches()
        with p1, p2 as mock_stdio, p3 as mock_cs_cls:
            _setup_mock_server(mock_stdio, mock_cs_cls, session)
            await client.start([{"name": "fs", "command": "x"}])

        emitted_name = client.list_tools()[0]["name"]
        assert emitted_name == "readfile"
        assert client.has_tool("readfile")
        assert not client.has_tool("read​file")


class TestMCPStartupTimeout:
    """L17: a hung MCP server (accepts the connection but never completes
    the init/list_tools handshake) must time out and be marked failed so
    agent boot proceeds instead of blocking forever.
    """

    @pytest.mark.asyncio
    async def test_hung_server_times_out_and_boot_continues(self):
        import src.agent.mcp_client as mcp_mod

        # Drop the timeout so the test is fast; the production default is 30s.
        with patch.object(mcp_mod, "_MCP_STARTUP_TIMEOUT_SECONDS", 0.05):
            client = MCPClient()

            async def _never_returns():
                await asyncio.Event().wait()  # blocks forever

            hung_session = AsyncMock()
            hung_session.initialize = AsyncMock(side_effect=_never_returns)
            hung_session.list_tools = AsyncMock(side_effect=_never_returns)

            good_session = AsyncMock()
            good_tools = MagicMock()
            good_tools.tools = [_make_mock_tool("ok_tool")]
            good_session.initialize = AsyncMock()
            good_session.list_tools = AsyncMock(return_value=good_tools)

            call_count = 0

            def stdio_side_effect(params):
                cm = AsyncMock()
                cm.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
                return cm

            def session_side_effect(read, write):
                nonlocal call_count
                call_count += 1
                cm = AsyncMock()
                if call_count == 1:
                    cm.__aenter__ = AsyncMock(return_value=hung_session)
                else:
                    cm.__aenter__ = AsyncMock(return_value=good_session)
                return cm

            p1 = patch("src.agent.mcp_client.StdioServerParameters", MagicMock())
            p2 = patch(
                "src.agent.mcp_client.stdio_client", side_effect=stdio_side_effect
            )
            p3 = patch(
                "src.agent.mcp_client.ClientSession", side_effect=session_side_effect
            )
            with p1, p2, p3:
                await client.start([
                    {"name": "hung", "command": "x"},
                    {"name": "good", "command": "y"},
                ])

        # The hung server timed out → marked failed, boot proceeded to the
        # next server which started normally.
        statuses = {s["name"]: s for s in client.list_server_statuses()}
        assert statuses["hung"]["state"] == "failed"
        assert statuses["hung"]["tools_count"] == 0
        assert statuses["good"]["state"] == "running"
        assert client.has_tool("ok_tool")
        assert not client.has_tool("hung")


class TestMCPClientLifecycle:
    @pytest.mark.asyncio
    async def test_stop_cleans_up(self):
        """stop() closes the AsyncExitStack and clears state."""
        client = MCPClient()
        client._sessions = {"test": MagicMock()}
        client._tool_to_server = {"tool": "test"}
        client._tool_schemas = {"tool": {"name": "tool"}}
        client._server_status = {"test": {"state": "running"}}

        client._exit_stack = AsyncMock()
        client._exit_stack.aclose = AsyncMock()

        await client.stop()

        client._exit_stack.aclose.assert_awaited_once()
        assert len(client._sessions) == 0
        assert len(client._tool_to_server) == 0
        assert len(client._tool_schemas) == 0
        assert len(client._server_status) == 0

    def test_has_tool(self):
        """has_tool returns True only for registered MCP tools."""
        client = MCPClient()
        client._tool_to_server = {"mcp_tool": "server"}

        assert client.has_tool("mcp_tool")


# === Per-server status registry + tool-to-server mapping (T6) ===


class TestMCPServerStatusRegistry:
    """The MCPClient tracks per-server startup/discovery status so the
    dashboard can render status dots and surface the error string on a
    failed-start row click. NOT live health — registry reflects the
    last ``start()`` attempt only.
    """

    @pytest.mark.asyncio
    async def test_running_status_recorded_on_successful_start(self):
        client = MCPClient()

        session = AsyncMock()
        tools_result = MagicMock()
        tools_result.tools = [
            _make_mock_tool("read_file"),
            _make_mock_tool("write_file"),
        ]
        session.initialize = AsyncMock()
        session.list_tools = AsyncMock(return_value=tools_result)

        p1 = patch("src.agent.mcp_client.StdioServerParameters", MagicMock())
        p2 = patch("src.agent.mcp_client.stdio_client")
        p3 = patch("src.agent.mcp_client.ClientSession")

        with p1, p2 as mock_stdio, p3 as mock_cs_cls:
            _setup_mock_server(mock_stdio, mock_cs_cls, session)
            await client.start([{"name": "fs", "command": "x"}])

        statuses = client.list_server_statuses()
        assert len(statuses) == 1
        assert statuses[0] == {
            "name": "fs", "state": "running", "tools_count": 2, "error": None,
        }

    @pytest.mark.asyncio
    async def test_failed_status_captures_error_message(self):
        client = MCPClient()

        def boom(params):
            cm = AsyncMock()
            cm.__aenter__ = AsyncMock(
                side_effect=RuntimeError("command not found: linear-server"),
            )
            return cm

        p1 = patch("src.agent.mcp_client.StdioServerParameters", MagicMock())
        p2 = patch("src.agent.mcp_client.stdio_client", side_effect=boom)
        p3 = patch("src.agent.mcp_client.ClientSession")

        with p1, p2, p3:
            await client.start([{"name": "linear", "command": "linear-server"}])

        statuses = client.list_server_statuses()
        assert len(statuses) == 1
        s = statuses[0]
        assert s["name"] == "linear"
        assert s["state"] == "failed"
        assert s["tools_count"] == 0
        assert "command not found" in s["error"]

    @pytest.mark.asyncio
    async def test_mixed_success_failure_recorded_independently(self):
        """A failing server doesn't tank the registry entry for a
        successful sibling — both must appear with their respective
        states. Mirrors the existing graceful-failure behavior.
        """
        client = MCPClient()

        good_session = AsyncMock()
        good_tools = MagicMock()
        good_tools.tools = [_make_mock_tool("ok_tool")]
        good_session.initialize = AsyncMock()
        good_session.list_tools = AsyncMock(return_value=good_tools)

        call_count = 0

        def side_effect(params):
            nonlocal call_count
            call_count += 1
            cm = AsyncMock()
            if call_count == 1:
                cm.__aenter__ = AsyncMock(side_effect=RuntimeError("bad"))
            else:
                cm.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
            return cm

        p1 = patch("src.agent.mcp_client.StdioServerParameters", MagicMock())
        p2 = patch("src.agent.mcp_client.stdio_client", side_effect=side_effect)
        p3 = patch("src.agent.mcp_client.ClientSession")

        with p1, p2, p3 as mock_cs_cls:
            scm = AsyncMock()
            scm.__aenter__ = AsyncMock(return_value=good_session)
            mock_cs_cls.return_value = scm
            await client.start([
                {"name": "bad", "command": "x"},
                {"name": "good", "command": "y"},
            ])

        by_name = {s["name"]: s for s in client.list_server_statuses()}
        assert by_name["bad"]["state"] == "failed"
        assert by_name["bad"]["tools_count"] == 0
        assert by_name["good"]["state"] == "running"
        assert by_name["good"]["tools_count"] == 1


class TestMCPToolToServerMapping:
    """The ``get_tool_to_server`` side-channel lets the dashboard filter
    the existing tool list by MCP server when the user clicks a
    server's tool-count badge. Captures both non-conflicting tool
    names AND the conflict-rename case (``mcp_<server>_<tool>``).
    """

    @pytest.mark.asyncio
    async def test_non_conflict_tool_maps_to_server_under_original_name(self):
        client = MCPClient()

        session = AsyncMock()
        tools_result = MagicMock()
        tools_result.tools = [_make_mock_tool("read_file")]
        session.initialize = AsyncMock()
        session.list_tools = AsyncMock(return_value=tools_result)

        p1 = patch("src.agent.mcp_client.StdioServerParameters", MagicMock())
        p2 = patch("src.agent.mcp_client.stdio_client")
        p3 = patch("src.agent.mcp_client.ClientSession")
        with p1, p2 as mock_stdio, p3 as mock_cs_cls:
            _setup_mock_server(mock_stdio, mock_cs_cls, session)
            await client.start([{"name": "fs", "command": "x"}])

        mapping = client.get_tool_to_server()
        assert mapping == {"read_file": "fs"}

    @pytest.mark.asyncio
    async def test_conflict_renamed_tool_maps_under_prefixed_name(self):
        # A tool name that collides with a built-in gets renamed to
        # ``mcp_<server>_<tool>``; the mapping must use that key (the
        # one the LLM and dashboard see) so filtering works.
        client = MCPClient()

        session = AsyncMock()
        tools_result = MagicMock()
        tools_result.tools = [_make_mock_tool("exec_command")]
        session.initialize = AsyncMock()
        session.list_tools = AsyncMock(return_value=tools_result)

        p1 = patch("src.agent.mcp_client.StdioServerParameters", MagicMock())
        p2 = patch("src.agent.mcp_client.stdio_client")
        p3 = patch("src.agent.mcp_client.ClientSession")
        with p1, p2 as mock_stdio, p3 as mock_cs_cls:
            _setup_mock_server(mock_stdio, mock_cs_cls, session)
            # ``exec_command`` collides with a built-in name → renamed
            await client.start(
                [{"name": "shell", "command": "x"}],
                builtin_names={"exec_command"},
            )

        mapping = client.get_tool_to_server()
        # Renamed key, not the original ``exec_command``
        assert "mcp_shell_exec_command" in mapping
        assert mapping["mcp_shell_exec_command"] == "shell"
        # Original name is NOT in the mapping (it's the built-in's name)
        assert "exec_command" not in mapping

    def test_get_tool_to_server_returns_snapshot_not_alias(self):
        """Mutating the returned dict must not bleed into MCPClient state."""
        client = MCPClient()
        client._tool_to_server = {"a": "s1"}
        snap = client.get_tool_to_server()
        snap["b"] = "s2"
        assert "b" not in client._tool_to_server
        assert not client.has_tool("other_tool")
