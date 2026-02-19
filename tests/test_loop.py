"""Unit tests for the agent execution loop.

Uses mock LLM to verify:
- Proper message role alternation (user/assistant/tool)
- Bounded iterations
- Cancellation
- Token budget enforcement
- Final output parsing
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.loop import AgentLoop
from src.shared.types import LLMResponse, TaskAssignment, TokenBudget, ToolCallInfo


def _make_loop(llm_responses: list[LLMResponse] | None = None) -> AgentLoop:
    """Create an AgentLoop with mock dependencies."""
    memory = MagicMock()
    memory.get_high_salience_facts = MagicMock(return_value=[])
    memory.search = AsyncMock(return_value=[])
    memory.search_hierarchical = AsyncMock(return_value=[])
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
    mesh_client.send_system_message = AsyncMock(return_value={})
    mesh_client.read_blackboard = AsyncMock(return_value=None)

    loop = AgentLoop(
        agent_id="test_agent",
        role="research",
        memory=memory,
        skills=skills,
        llm=llm,
        mesh_client=mesh_client,
    )
    return loop


@pytest.mark.asyncio
async def test_simple_task_completes():
    """LLM returns final answer immediately (no tool calls)."""
    loop = _make_loop()
    assignment = TaskAssignment(
        workflow_id="wf1", step_id="s1", task_type="research", input_data={"query": "test"}
    )

    result = await loop.execute_task(assignment)
    assert result.status == "complete"
    assert result.result == {"answer": "42"}
    assert result.tokens_used == 100
    assert loop.tasks_completed == 1
    assert loop.state == "idle"


@pytest.mark.asyncio
async def test_tool_calling_message_roles():
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
    result = await loop.execute_task(assignment)

    assert result.status == "complete"
    assert len(captured_messages) == 2

    second_call_msgs = captured_messages[1]
    assert second_call_msgs[0]["role"] == "user"
    assert second_call_msgs[1]["role"] == "assistant"
    assert "tool_calls" in second_call_msgs[1]
    assert second_call_msgs[2]["role"] == "tool"


@pytest.mark.asyncio
async def test_max_iterations_reached():
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
    result = await loop.execute_task(assignment)

    assert result.status == "failed"
    assert "Max iterations" in result.error
    assert loop.tasks_failed == 1


@pytest.mark.asyncio
async def test_token_budget_exhausted():
    """Loop should stop when token budget is exceeded."""
    loop = _make_loop()
    budget = TokenBudget(max_tokens=100, used_tokens=99)
    assignment = TaskAssignment(
        workflow_id="wf1", step_id="s1", task_type="research", input_data={}, token_budget=budget
    )

    result = await loop.execute_task(assignment)
    assert result.status == "failed"
    assert "budget" in result.error.lower()


@pytest.mark.asyncio
async def test_cancellation():
    """Loop should respect cancellation flag."""
    loop = _make_loop()
    loop._cancel_requested = True

    assignment = TaskAssignment(
        workflow_id="wf1", step_id="s1", task_type="research", input_data={}
    )
    result = await loop.execute_task(assignment)
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


# === LLM Retry Logic ===

from unittest.mock import patch

import httpx

from src.agent.loop import _llm_call_with_retry


@pytest.mark.asyncio
async def test_llm_retry_on_connect_error():
    """Retry on httpx.ConnectError, succeed on second attempt."""
    success = LLMResponse(content='{"result": {}}', tokens_used=50)
    mock_fn = AsyncMock(side_effect=[httpx.ConnectError("refused"), success])

    with patch("src.agent.loop.asyncio.sleep", new_callable=AsyncMock):
        result = await _llm_call_with_retry(mock_fn, system="s", messages=[], tools=None)

    assert result.content == '{"result": {}}'
    assert mock_fn.call_count == 2


@pytest.mark.asyncio
async def test_llm_retry_on_timeout():
    """Retry on httpx.TimeoutException."""
    success = LLMResponse(content='{"result": {}}', tokens_used=50)
    mock_fn = AsyncMock(side_effect=[httpx.ReadTimeout("timeout"), success])

    with patch("src.agent.loop.asyncio.sleep", new_callable=AsyncMock):
        result = await _llm_call_with_retry(mock_fn, system="s", messages=[], tools=None)

    assert result.tokens_used == 50
    assert mock_fn.call_count == 2


@pytest.mark.asyncio
async def test_llm_retry_on_429():
    """Retry on HTTP 429 status."""
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_request = MagicMock()
    error_429 = httpx.HTTPStatusError("rate limited", request=mock_request, response=mock_response)
    success = LLMResponse(content='{"result": {}}', tokens_used=50)
    mock_fn = AsyncMock(side_effect=[error_429, success])

    with patch("src.agent.loop.asyncio.sleep", new_callable=AsyncMock):
        result = await _llm_call_with_retry(mock_fn, system="s", messages=[], tools=None)

    assert result.tokens_used == 50


@pytest.mark.asyncio
async def test_llm_no_retry_on_permanent_error():
    """RuntimeError (budget exceeded) should not be retried."""
    mock_fn = AsyncMock(side_effect=RuntimeError("Budget exceeded"))

    with pytest.raises(RuntimeError, match="Budget exceeded"):
        await _llm_call_with_retry(mock_fn, system="s", messages=[], tools=None)

    assert mock_fn.call_count == 1


@pytest.mark.asyncio
async def test_llm_no_retry_on_non_retryable_status():
    """HTTP 400 should not be retried."""
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_request = MagicMock()
    error_400 = httpx.HTTPStatusError("bad request", request=mock_request, response=mock_response)
    mock_fn = AsyncMock(side_effect=error_400)

    with pytest.raises(httpx.HTTPStatusError):
        await _llm_call_with_retry(mock_fn, system="s", messages=[], tools=None)

    assert mock_fn.call_count == 1


@pytest.mark.asyncio
async def test_llm_retry_exhausted():
    """After MAX_RETRIES, the exception is raised."""
    mock_fn = AsyncMock(side_effect=httpx.ConnectError("refused"))

    with patch("src.agent.loop.asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(httpx.ConnectError):
            await _llm_call_with_retry(mock_fn, system="s", messages=[], tools=None)

    assert mock_fn.call_count == 4  # 1 initial + 3 retries


@pytest.mark.asyncio
async def test_llm_retry_in_chat_mode():
    """Chat mode LLM calls also benefit from retry."""
    connect_err = httpx.ConnectError("refused")
    success = LLMResponse(content="Hello!", tokens_used=50)
    responses = [connect_err, success]

    loop = _make_loop()
    loop.llm.chat = AsyncMock(side_effect=responses)
    loop.skills.get_tool_definitions = MagicMock(return_value=[])

    with patch("src.agent.loop.asyncio.sleep", new_callable=AsyncMock):
        result = await loop.chat("Hi")

    assert result["response"] == "Hello!"
    assert loop.llm.chat.call_count == 2
