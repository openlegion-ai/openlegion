"""Tests for operator heartbeat tool restriction."""
from unittest.mock import AsyncMock

import pytest

from src.agent.loop import _HEARTBEAT_TOOLS
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
        "notify_user", "hand_off", "confirm_edit",
    })
    loop._allowed_tools = operator_tools
    original_allowed = loop._allowed_tools

    # Run heartbeat
    result = await loop.execute_heartbeat("Check fleet health")

    assert result["skipped"] is False
    assert result["outcome"] == "ok"

    # Verify get_tool_definitions was called with allowed=_HEARTBEAT_TOOLS
    for c in loop.tools.get_tool_definitions.call_args_list:
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
    """Verify _HEARTBEAT_TOOLS carries the heartbeat-reachable allowlist.

    Layer history:
    * v1 (initial): 4 read-only tools — ``list_agents``,
      ``get_agent_profile``, ``get_system_status``, ``notify_user``.
    * v2 (workflow awareness): + ``check_inbox``, ``workflow_snapshot``,
      ``await_task_event`` so the heartbeat can drive multi-stage
      chains without dropping out to a full /chat turn.
    * v3 (Work-tab rewrite PR 2): + ``rate_delivery``, ``manage_goals``
      so the heartbeat instructions that grade up to 10 oldest unrated
      done tasks and steward goal staleness are actually reachable —
      without these the loop denies the calls the instructions request.
    * v4 (PR 972 Codex follow-up): + ``inspect_agents``. Step 5 of
      the heartbeat procedure already called it for roster summary
      and drill-ins but the allowlist denied the call — a pre-existing
      contract mismatch Codex flagged during PR 972 review.
    * v5 (B5 task-run diagnostics): + ``inspect_task_run``. Read-only
      per-task execution summary (tokens, LLM calls, thinking, trace
      errors) so the rate-stale-deliverables step can look at HOW a
      task ran before grading it.
    """
    assert _HEARTBEAT_TOOLS == frozenset({
        "list_agents", "get_agent_profile", "get_system_status",
        "notify_user",
        "check_inbox", "workflow_snapshot", "await_task_event",
        "rate_delivery", "manage_goals",
        "inspect_agents", "inspect_task_run",
    })
    # v3 main was 9 tools (no save_observations — removed pre-merge);
    # v4 added inspect_agents = 10; v5 added inspect_task_run = 11.
    assert len(_HEARTBEAT_TOOLS) == 11
