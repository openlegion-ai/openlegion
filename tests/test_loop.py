"""Unit tests for the agent execution loop.

Uses mock LLM to verify:
- Proper message role alternation (user/assistant/tool)
- Bounded iterations
- Cancellation
- Token budget enforcement
- Final output parsing
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from src.agent.loop import AgentLoop
from src.shared.types import LLMResponse, TaskAssignment, TokenBudget, ToolCallInfo


def _make_loop(llm_responses: list[LLMResponse] | None = None) -> AgentLoop:
    """Create an AgentLoop with mock dependencies."""
    memory = MagicMock()
    memory.get_high_salience_facts = MagicMock(return_value=[])
    memory.search = AsyncMock(return_value=[])
    memory.store_fact = AsyncMock(return_value="fact_123")
    memory.log_action = AsyncMock()

    skills = MagicMock()
    skills.get_tool_definitions = MagicMock(return_value=[])
    skills.get_descriptions = MagicMock(return_value="- no tools")
    skills.list_skills = MagicMock(return_value=[])

    llm = MagicMock()
    if llm_responses:
        llm.chat = AsyncMock(side_effect=llm_responses)
    else:
        llm.chat = AsyncMock(return_value=LLMResponse(content='{"result": {"answer": "42"}}', tokens_used=100))
    llm.default_model = "test-model"

    mesh_client = MagicMock()
    mesh_client.send_message = AsyncMock(return_value={})

    loop = AgentLoop(
        agent_id="test_agent",
        role="research",
        memory=memory,
        skills=skills,
        llm=llm,
        mesh_client=mesh_client,
    )
    return loop


def test_simple_task_completes():
    """LLM returns final answer immediately (no tool calls)."""
    loop = _make_loop()
    assignment = TaskAssignment(
        workflow_id="wf1", step_id="s1", task_type="research", input_data={"query": "test"}
    )

    result = asyncio.get_event_loop().run_until_complete(loop.execute_task(assignment))
    assert result.status == "complete"
    assert result.result == {"answer": "42"}
    assert result.tokens_used == 100
    assert loop.tasks_completed == 1
    assert loop.state == "idle"


def test_tool_calling_message_roles():
    """Verify proper message role alternation: user -> assistant(tool_calls) -> tool -> assistant."""
    captured_messages = []

    tool_call_response = LLMResponse(
        content="",
        tool_calls=[ToolCallInfo(name="web_search", arguments={"query": "test"})],
        tokens_used=50,
    )
    final_response = LLMResponse(content='{"result": {"found": true}}', tokens_used=30)

    loop = _make_loop([tool_call_response, final_response])
    loop.skills.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "web_search"}}]
    )
    loop.skills.execute = AsyncMock(return_value={"results": ["r1"]})

    original_chat = loop.llm.chat

    async def capturing_chat(system, messages, tools=None, **kwargs):
        captured_messages.append([dict(m) for m in messages])
        return await original_chat(system=system, messages=messages, tools=tools, **kwargs)

    loop.llm.chat = capturing_chat

    assignment = TaskAssignment(
        workflow_id="wf1", step_id="s1", task_type="research", input_data={"q": "test"}
    )
    result = asyncio.get_event_loop().run_until_complete(loop.execute_task(assignment))

    assert result.status == "complete"
    assert len(captured_messages) == 2

    # Second call should have: user, assistant(tool_calls), tool
    second_call_msgs = captured_messages[1]
    assert second_call_msgs[0]["role"] == "user"
    assert second_call_msgs[1]["role"] == "assistant"
    assert "tool_calls" in second_call_msgs[1]
    assert second_call_msgs[2]["role"] == "tool"


def test_max_iterations_reached():
    """Loop should fail after MAX_ITERATIONS of tool calls."""
    always_tool = LLMResponse(
        content="",
        tool_calls=[ToolCallInfo(name="search", arguments={"q": "x"})],
        tokens_used=10,
    )
    responses = [always_tool] * 25

    loop = _make_loop(responses)
    loop.skills.get_tool_definitions = MagicMock(return_value=[{"type": "function", "function": {"name": "search"}}])
    loop.skills.execute = AsyncMock(return_value={"result": "ok"})

    assignment = TaskAssignment(
        workflow_id="wf1", step_id="s1", task_type="research", input_data={}
    )
    result = asyncio.get_event_loop().run_until_complete(loop.execute_task(assignment))

    assert result.status == "failed"
    assert "Max iterations" in result.error
    assert loop.tasks_failed == 1


def test_token_budget_exhausted():
    """Loop should stop when token budget is exceeded."""
    loop = _make_loop()
    budget = TokenBudget(max_tokens=100, used_tokens=99)
    assignment = TaskAssignment(
        workflow_id="wf1", step_id="s1", task_type="research", input_data={}, token_budget=budget
    )

    result = asyncio.get_event_loop().run_until_complete(loop.execute_task(assignment))
    assert result.status == "failed"
    assert "budget" in result.error.lower()


def test_cancellation():
    """Loop should respect cancellation flag."""
    loop = _make_loop()
    loop._cancel_requested = True

    assignment = TaskAssignment(
        workflow_id="wf1", step_id="s1", task_type="research", input_data={}
    )
    result = asyncio.get_event_loop().run_until_complete(loop.execute_task(assignment))
    assert result.status == "cancelled"


def test_parse_final_output_json():
    """Final output parser handles valid JSON."""
    loop = _make_loop()
    result_data, promotions = loop._parse_final_output('{"result": {"x": 1}, "promote": {"ctx/a": "b"}}')
    assert result_data == {"x": 1}
    assert promotions == {"ctx/a": "b"}


def test_parse_final_output_plain_text():
    """Final output parser handles non-JSON gracefully."""
    loop = _make_loop()
    result_data, promotions = loop._parse_final_output("Just a plain text response")
    assert result_data == {"raw": "Just a plain text response"}
    assert promotions == {}


def test_get_status():
    loop = _make_loop()
    status = loop.get_status()
    assert status.agent_id == "test_agent"
    assert status.role == "research"
    assert status.state == "idle"
