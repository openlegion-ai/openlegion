"""End-to-end tests for MCP client using a real MCP server subprocess.

Launches tests/fixtures/echo_mcp_server.py via stdio transport and exercises
the full MCP protocol: initialization, tool discovery, tool calls, and error
handling. No mocks — this tests the real protocol plumbing.

Requires the `mcp` package to be installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

try:
    import mcp  # noqa: F401
    HAS_MCP = True
except ImportError:
    HAS_MCP = False

from src.agent.mcp_client import MCPClient
from src.agent.skills import SkillRegistry, _skill_staging

pytestmark = pytest.mark.skipif(not HAS_MCP, reason="mcp package not installed")

ECHO_SERVER = str(Path(__file__).parent / "fixtures" / "echo_mcp_server.py")


class TestMCPE2EProtocol:
    """Test the real MCP stdio protocol against a live server."""

    @pytest.mark.asyncio
    async def test_discovery_and_tool_call(self):
        """Start server, discover tools, call echo, verify result."""
        client = MCPClient()
        await client.start([{
            "name": "echo",
            "command": sys.executable,
            "args": [ECHO_SERVER],
        }])

        try:
            # Discovery
            tools = client.list_tools()
            tool_names = {t["name"] for t in tools}
            assert "echo" in tool_names
            assert "add" in tool_names
            assert "fail" in tool_names

            # Verify tool schemas have correct structure
            echo_tool = next(t for t in tools if t["name"] == "echo")
            assert echo_tool["function"] == "mcp"
            assert echo_tool["parameters"]["type"] == "object"
            assert "text" in echo_tool["parameters"]["properties"]

            # Call echo tool
            result = await client.call_tool("echo", {"text": "hello from openlegion"})
            assert "error" not in result
            assert "hello from openlegion" in result["result"]

            # Call add tool
            result = await client.call_tool("add", {"a": 17, "b": 25})
            assert "error" not in result
            assert "42" in result["result"]
        finally:
            await client.stop()

    @pytest.mark.asyncio
    async def test_error_tool(self):
        """Tool that raises an exception returns error dict."""
        client = MCPClient()
        await client.start([{
            "name": "echo",
            "command": sys.executable,
            "args": [ECHO_SERVER],
        }])

        try:
            result = await client.call_tool("fail", {})
            assert "error" in result
            assert "intentional test error" in result["error"].lower() or "error" in result["error"].lower()
        finally:
            await client.stop()

    @pytest.mark.asyncio
    async def test_has_tool(self):
        """has_tool returns True for discovered tools, False for unknown."""
        client = MCPClient()
        await client.start([{
            "name": "echo",
            "command": sys.executable,
            "args": [ECHO_SERVER],
        }])

        try:
            assert client.has_tool("echo")
            assert client.has_tool("add")
            assert not client.has_tool("nonexistent")
        finally:
            await client.stop()

    @pytest.mark.asyncio
    async def test_builtin_conflict_prefix(self):
        """Tool names conflicting with builtins get prefixed."""
        client = MCPClient()
        await client.start(
            [{
                "name": "echoserver",
                "command": sys.executable,
                "args": [ECHO_SERVER],
            }],
            builtin_names={"echo"},  # pretend "echo" is a builtin
        )

        try:
            # Original "echo" should be prefixed
            assert not client.has_tool("echo")
            assert client.has_tool("mcp_echoserver_echo")

            # Prefixed tool should still work
            result = await client.call_tool("mcp_echoserver_echo", {"text": "prefixed"})
            assert "error" not in result
            assert "prefixed" in result["result"]

            # "add" should keep its original name (no conflict)
            assert client.has_tool("add")
        finally:
            await client.stop()

    @pytest.mark.asyncio
    async def test_stop_cleans_up(self):
        """After stop(), all state is cleared."""
        client = MCPClient()
        await client.start([{
            "name": "echo",
            "command": sys.executable,
            "args": [ECHO_SERVER],
        }])

        assert len(client.list_tools()) > 0
        await client.stop()

        assert len(client.list_tools()) == 0
        assert not client.has_tool("echo")

    @pytest.mark.asyncio
    async def test_bad_server_graceful(self):
        """A server that fails to start doesn't crash the client."""
        client = MCPClient()
        await client.start([
            {"name": "bad", "command": sys.executable, "args": ["-c", "import sys; sys.exit(1)"]},
            {"name": "good", "command": sys.executable, "args": [ECHO_SERVER]},
        ])

        try:
            # Bad server failed, but good server should work
            assert client.has_tool("echo")
            result = await client.call_tool("echo", {"text": "survived"})
            assert "survived" in result["result"]
        finally:
            await client.stop()


class TestMCPE2ESkillRegistry:
    """Test MCPClient integration with SkillRegistry end-to-end."""

    def setup_method(self):
        _skill_staging.clear()

    @pytest.mark.asyncio
    async def test_mcp_tools_in_registry(self):
        """MCP tools appear in SkillRegistry and route correctly."""
        client = MCPClient()
        await client.start([{
            "name": "echo",
            "command": sys.executable,
            "args": [ECHO_SERVER],
        }])

        try:
            registry = SkillRegistry.__new__(SkillRegistry)
            registry.skills_dir = "/nonexistent"
            registry._mcp_client = client
            registry.skills = {}
            registry._register_mcp_tools()

            # Tools should appear in registry
            assert "echo" in registry.list_skills()
            assert "add" in registry.list_skills()

            # Tool definitions should be LLM-ready
            defs = registry.get_tool_definitions()
            echo_def = next(d for d in defs if d["function"]["name"] == "echo")
            assert echo_def["type"] == "function"
            assert echo_def["function"]["parameters"]["type"] == "object"

            # Execute through registry routes to MCP
            result = await registry.execute("echo", {"text": "via registry"})
            assert "via registry" in result["result"]

            result = await registry.execute("add", {"a": 3, "b": 7})
            assert "10" in result["result"]
        finally:
            await client.stop()
