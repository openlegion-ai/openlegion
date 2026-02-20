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


# === Silent Reply Token ===


@pytest.mark.asyncio
async def test_silent_reply_token_returns_empty():
    """When LLM returns __SILENT__, chat() should return empty string response."""
    silent_response = LLMResponse(content="__SILENT__", tokens_used=10)
    loop = _make_loop([silent_response])
    loop.skills.get_tool_definitions = MagicMock(return_value=[])

    result = await loop.chat("heartbeat ping")
    assert result["response"] == ""


@pytest.mark.asyncio
async def test_silent_reply_token_with_whitespace():
    """__SILENT__ with surrounding whitespace should still be detected."""
    silent_response = LLMResponse(content="  __SILENT__  \n", tokens_used=10)
    loop = _make_loop([silent_response])
    loop.skills.get_tool_definitions = MagicMock(return_value=[])

    result = await loop.chat("cron tick")
    assert result["response"] == ""


@pytest.mark.asyncio
async def test_non_silent_reply_passes_through():
    """Normal LLM responses should pass through unchanged."""
    normal_response = LLMResponse(content="Hello, how can I help?", tokens_used=20)
    loop = _make_loop([normal_response])
    loop.skills.get_tool_definitions = MagicMock(return_value=[])

    result = await loop.chat("hi")
    assert result["response"] == "Hello, how can I help?"


@pytest.mark.asyncio
async def test_silent_token_after_tool_rounds_exhausted():
    """__SILENT__ at max-rounds fallback should also be suppressed."""
    tool_call = LLMResponse(
        content="",
        tool_calls=[ToolCallInfo(name="search", arguments={"q": "x"})],
        tokens_used=10,
    )
    # Fill CHAT_MAX_TOOL_ROUNDS with tool calls, then final __SILENT__
    silent_final = LLMResponse(content="__SILENT__", tokens_used=10)
    responses = [tool_call] * 31 + [silent_final]

    loop = _make_loop(responses)
    loop.skills.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "search"}}]
    )
    loop.skills.execute = AsyncMock(return_value={"result": "ok"})

    result = await loop.chat("do something")
    assert result["response"] == ""


# === Steer Injection ===


@pytest.mark.asyncio
async def test_inject_steer_returns_working_state():
    """inject_steer returns True when agent is working."""
    loop = _make_loop()
    loop.state = "working"
    result = await loop.inject_steer("stop, do this instead")
    assert result is True


@pytest.mark.asyncio
async def test_inject_steer_returns_idle_state():
    """inject_steer returns False when agent is idle."""
    loop = _make_loop()
    loop.state = "idle"
    result = await loop.inject_steer("hello")
    assert result is False


@pytest.mark.asyncio
async def test_steer_to_idle_merges_with_next_message():
    """Steer message sent while idle gets merged into next user message."""
    loop = _make_loop([
        LLMResponse(content="Got it with context", tokens_used=50),
    ])
    loop.skills.get_tool_definitions = MagicMock(return_value=[])

    # Inject steer while idle
    await loop.inject_steer("extra context here")

    # Now send a chat message — steer should be merged
    result = await loop.chat("do the task")
    assert result["response"] == "Got it with context"

    # The first user message should contain the steer content
    user_msg = loop._chat_messages[0]
    assert "[Additional context]: extra context here" in user_msg["content"]


@pytest.mark.asyncio
async def test_steer_injection_between_tool_rounds():
    """Steer message appears in LLM context after tool results."""
    tool_response = LLMResponse(
        content="",
        tool_calls=[ToolCallInfo(name="search", arguments={"q": "x"})],
        tokens_used=10,
    )
    final_response = LLMResponse(content="Redirected!", tokens_used=20)
    captured_messages = []

    loop = _make_loop([tool_response, final_response])
    loop.skills.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "search"}}]
    )
    loop.skills.execute = AsyncMock(return_value={"result": "ok"})

    original_chat = loop.llm.chat

    async def capturing_chat(system, messages, tools=None, **kwargs):
        captured_messages.append([dict(m) for m in messages])
        return await original_chat(system=system, messages=messages, tools=tools, **kwargs)

    loop.llm.chat = capturing_chat

    # Inject steer after first tool call starts
    # We need to inject before the second LLM call
    original_execute = loop.skills.execute

    async def execute_with_steer(*args, **kwargs):
        # Inject steer message during tool execution
        await loop.inject_steer("stop, do this instead")
        return await original_execute(*args, **kwargs)

    loop.skills.execute = execute_with_steer

    result = await loop.chat("start task")
    assert result["response"] == "Redirected!"

    # The second LLM call should contain a user interjection message
    second_call = captured_messages[1]
    interjection_msgs = [m for m in second_call if "[User interjection]" in m.get("content", "")]
    assert len(interjection_msgs) == 1
    assert "stop, do this instead" in interjection_msgs[0]["content"]


@pytest.mark.asyncio
async def test_multiple_steers_combined():
    """Multiple steer messages are drained and joined together."""
    loop = _make_loop([
        LLMResponse(content="Acknowledged all", tokens_used=50),
    ])
    loop.skills.get_tool_definitions = MagicMock(return_value=[])

    # Queue multiple steer messages while idle
    await loop.inject_steer("first update")
    await loop.inject_steer("second update")

    result = await loop.chat("go")
    assert result["response"] == "Acknowledged all"

    # Both should be in the first user message
    user_msg = loop._chat_messages[0]["content"]
    assert "first update" in user_msg
    assert "second update" in user_msg


# === Context Warning Integration ===


def test_context_warning_in_chat_system_prompt():
    """When context >= 80%, _build_chat_system_prompt includes CONTEXT WARNING."""
    from src.agent.context import ContextManager

    loop = _make_loop()
    # Set up a context manager with very small max_tokens so any messages exceed 80%
    loop.context_manager = ContextManager(max_tokens=100)
    # Prefill chat messages to trigger the 80% threshold
    loop._chat_messages = [
        {"role": "user", "content": "x" * 500},
        {"role": "assistant", "content": "y" * 500},
    ]
    prompt = loop._build_chat_system_prompt()
    assert "CONTEXT WARNING" in prompt


def test_context_warning_absent_below_threshold():
    """When context < 80%, no warning in system prompt."""
    from src.agent.context import ContextManager

    loop = _make_loop()
    loop.context_manager = ContextManager(max_tokens=1_000_000)
    loop._chat_messages = [{"role": "user", "content": "Hello"}]
    prompt = loop._build_chat_system_prompt()
    assert "CONTEXT WARNING" not in prompt


def test_get_status_includes_context_fields():
    """get_status should return context_tokens, context_max, context_pct."""
    from src.agent.context import ContextManager

    loop = _make_loop()
    loop.context_manager = ContextManager(max_tokens=10_000)
    loop._chat_messages = [
        {"role": "user", "content": "Hello world " * 100},
    ]
    status = loop.get_status()
    assert status.context_max == 10_000
    assert status.context_tokens > 0
    assert 0 < status.context_pct < 1.0


def test_get_status_context_fields_without_context_manager():
    """get_status should return zero context fields when no context_manager."""
    loop = _make_loop()
    status = loop.get_status()
    assert status.context_tokens == 0
    assert status.context_max == 0
    assert status.context_pct == 0.0
