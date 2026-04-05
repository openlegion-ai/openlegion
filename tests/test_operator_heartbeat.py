"""Tests for operator heartbeat tool restriction."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.agent.loop import AgentLoop, _HEARTBEAT_TOOLS
from src.shared.types import LLMResponse

from tests.test_loop import _make_loop


@pytest.mark.asyncio
async def test_heartbeat_restricts_operator_tools():
    """Operator heartbeat should only see heartbeat tools, not the full allowlist."""
    loop = _make_loop()
    loop.llm.chat = AsyncMock(return_value=LLMResponse(content="HEARTBEAT_OK", tokens_used=50))
    loop.mesh_client.introspect = AsyncMock(return_value={})

    # Simulate operator: set _allowed_tools to a superset
    operator_tools = frozenset({
        "list_agents", "get_agent_profile", "get_system_status",
        "notify_user", "save_observations", "hand_off", "propose_edit", "confirm_edit",
    })
    loop._allowed_tools = operator_tools
    original_allowed = loop._allowed_tools

    # Run heartbeat
    result = await loop.execute_heartbeat("Check fleet health")

    assert result["skipped"] is False
    assert result["outcome"] == "ok"

    # Verify get_tool_definitions was called with allowed=_HEARTBEAT_TOOLS
    for c in loop.skills.get_tool_definitions.call_args_list:
        if c.kwargs.get("allowed") is not None:
            assert c.kwargs["allowed"] == _HEARTBEAT_TOOLS

    # Verify _allowed_tools restored after heartbeat
    assert loop._allowed_tools == original_allowed


@pytest.mark.asyncio
async def test_heartbeat_no_restriction_for_regular_agents():
    """Regular agents (no _allowed_tools) should not be restricted."""
    loop = _make_loop()
    loop.llm.chat = AsyncMock(return_value=LLMResponse(content="HEARTBEAT_OK", tokens_used=50))
    loop.mesh_client.introspect = AsyncMock(return_value={})

    assert loop._allowed_tools is None  # Regular agent

    await loop.execute_heartbeat("Check health")

    # _allowed_tools should still be None
    assert loop._allowed_tools is None


@pytest.mark.asyncio
async def test_heartbeat_restores_tools_on_error():
    """Operator _allowed_tools must be restored even if heartbeat errors."""
    loop = _make_loop()
    loop.mesh_client.introspect = AsyncMock(return_value={})
    # Make the LLM call raise so heartbeat hits the except branch
    loop.llm.chat = AsyncMock(side_effect=RuntimeError("boom"))

    operator_tools = frozenset({"list_agents", "notify_user", "hand_off"})
    loop._allowed_tools = operator_tools

    result = await loop.execute_heartbeat("Check fleet health")

    # The heartbeat catches exceptions and returns an error dict
    assert result["outcome"] == "error"
    # _allowed_tools must still be restored
    assert loop._allowed_tools == operator_tools


@pytest.mark.asyncio
async def test_heartbeat_restores_tools_on_skip():
    """Operator _allowed_tools must be restored when heartbeat is skipped."""
    loop = _make_loop()
    loop.state = "working"  # Will cause skip

    operator_tools = frozenset({"list_agents", "notify_user", "hand_off"})
    loop._allowed_tools = operator_tools

    result = await loop.execute_heartbeat("Check fleet health")

    assert result["skipped"] is True
    assert loop._allowed_tools == operator_tools


def test_heartbeat_tools_constant():
    """Verify _HEARTBEAT_TOOLS has exactly the 5 expected tools."""
    assert _HEARTBEAT_TOOLS == frozenset({
        "list_agents", "get_agent_profile", "get_system_status",
        "notify_user", "save_observations",
    })
    assert len(_HEARTBEAT_TOOLS) == 5
