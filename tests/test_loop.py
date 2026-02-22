"""Unit tests for the agent execution loop.

Uses mock LLM to verify:
- Proper message role alternation (user/assistant/tool)
- Bounded iterations
- Cancellation
- Token budget enforcement
- Final output parsing
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.loop import AgentLoop
from src.shared.types import LLMResponse, TaskAssignment, TokenBudget, ToolCallInfo


def _make_loop(llm_responses: list[LLMResponse] | None = None, *, real_memory: bool = False) -> AgentLoop:
    """Create an AgentLoop with mock dependencies.

    If real_memory=True, uses a real in-memory MemoryStore instead of a mock
    (needed for tests that exercise tool outcome storage/retrieval).
    """
    if real_memory:
        from src.agent.memory import MemoryStore
        memory = MemoryStore(db_path=":memory:")
    else:
        memory = MagicMock()
        memory.get_high_salience_facts = MagicMock(return_value=[])
        memory.search = AsyncMock(return_value=[])
        memory.search_hierarchical = AsyncMock(return_value=[])
        memory.store_fact = AsyncMock(return_value="fact_123")
        memory.log_action = AsyncMock()
        memory.store_tool_outcome = MagicMock()
        memory.get_tool_history = MagicMock(return_value=[])

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
    # Use different arguments each iteration to avoid triggering loop detection
    responses = [
        LLMResponse(
            content="",
            tool_calls=[ToolCallInfo(name="search", arguments={"q": f"query_{i}"})],
            tokens_used=10,
        )
        for i in range(25)
    ]

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
    # Use different arguments each round to avoid triggering loop detection
    responses = [
        LLMResponse(
            content="",
            tool_calls=[ToolCallInfo(name="search", arguments={"q": f"q_{i}"})],
            tokens_used=10,
        )
        for i in range(31)
    ]
    # Fill CHAT_MAX_TOOL_ROUNDS with tool calls, then final __SILENT__
    silent_final = LLMResponse(content="__SILENT__", tokens_used=10)
    responses.append(silent_final)

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


# === Tool Memory ===


class TestToolMemory:
    @pytest.mark.asyncio
    async def test_learn_stores_tool_outcome(self):
        """_learn() stores a successful tool outcome in memory."""
        loop = _make_loop(real_memory=True)
        await loop._learn("exec", {"command": "ls"}, {"exit_code": 0, "stdout": "file.txt"})
        history = loop.memory.get_tool_history("exec")
        assert len(history) == 1
        assert history[0]["success"] is True

    @pytest.mark.asyncio
    async def test_record_failure_stores_tool_outcome(self):
        """_record_failure() stores a failed outcome in memory."""
        loop = _make_loop(real_memory=True)
        loop._record_failure("exec", "command not found", arguments={"command": "bad"})
        history = loop.memory.get_tool_history("exec")
        assert len(history) == 1
        assert history[0]["success"] is False

    @pytest.mark.asyncio
    async def test_chat_mode_learns_from_tools(self):
        """Chat mode calls _learn() on successful tool execution."""
        tool_response = LLMResponse(
            content="",
            tool_calls=[ToolCallInfo(name="exec", arguments={"command": "ls"})],
            tokens_used=30,
        )
        final_response = LLMResponse(content="Done", tokens_used=20)
        loop = _make_loop([tool_response, final_response], real_memory=True)
        loop.skills.execute = AsyncMock(return_value={"exit_code": 0})
        loop.skills.get_tool_definitions = MagicMock(
            return_value=[{"type": "function", "function": {"name": "exec"}}]
        )

        await loop.chat("Run ls")
        history = loop.memory.get_tool_history("exec")
        assert len(history) >= 1
        assert history[0]["success"] is True

    def test_build_tool_history_context_empty(self):
        """Returns empty string when no tool history."""
        loop = _make_loop(real_memory=True)
        assert loop._build_tool_history_context() == ""

    def test_build_tool_history_context_with_data(self):
        """Returns formatted section when tool history exists."""
        loop = _make_loop(real_memory=True)
        loop.memory.store_tool_outcome("exec", {"cmd": "ls"}, "file.txt", success=True)
        loop.memory.store_tool_outcome("exec", {"cmd": "bad"}, "error", success=False)
        ctx = loop._build_tool_history_context()
        assert "## Recent Tool History" in ctx
        assert "exec [OK]" in ctx
        assert "exec [FAILED]" in ctx

    def test_tool_history_in_chat_system_prompt(self):
        """Chat system prompt includes tool history when present."""
        loop = _make_loop(real_memory=True)
        loop.memory.store_tool_outcome("exec", {}, "ok", success=True)
        prompt = loop._build_chat_system_prompt()
        assert "Recent Tool History" in prompt


# === Memory Decay ===


@pytest.mark.asyncio
async def test_decay_all_called_on_task_start():
    """Salience decay runs at the start of each task execution."""
    loop = _make_loop()
    assignment = TaskAssignment(
        workflow_id="wf1", step_id="s1", task_type="research", input_data={"q": "test"}
    )
    await loop.execute_task(assignment)
    loop.memory.decay_all.assert_called_once()


# === Prompt Injection Sanitization (choke-point integration) ===


@pytest.mark.asyncio
async def test_tool_result_sanitized_in_execute_task():
    """Tool results with invisible chars are stripped before entering LLM context."""
    tool_call_response = LLMResponse(
        content="",
        tool_calls=[ToolCallInfo(name="web_search", arguments={"query": "test"})],
        tokens_used=50,
    )
    final_response = LLMResponse(content='{"result": {"ok": true}}', tokens_used=30)
    captured_messages = []

    loop = _make_loop([tool_call_response, final_response])
    loop.skills.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "web_search"}}]
    )
    # Return a string (non-dict) with invisible characters — goes through str() path
    loop.skills.execute = AsyncMock(return_value="clean\u200Bvalue\u202Ehere")

    original_chat = loop.llm.chat

    async def capturing_chat(system, messages, tools=None, **kwargs):
        captured_messages.append([dict(m) for m in messages])
        return await original_chat(system=system, messages=messages, tools=tools, **kwargs)

    loop.llm.chat = capturing_chat

    assignment = TaskAssignment(
        workflow_id="wf1", step_id="s1", task_type="research", input_data={"q": "test"}
    )
    await loop.execute_task(assignment)

    # The tool result message in the second LLM call should be sanitized
    second_call = captured_messages[1]
    tool_msg = next(m for m in second_call if m.get("role") == "tool")
    assert "\u200B" not in tool_msg["content"]
    assert "\u202E" not in tool_msg["content"]
    assert "cleanvaluehere" in tool_msg["content"]


@pytest.mark.asyncio
async def test_tool_result_sanitized_in_chat():
    """Chat mode tool results are sanitized before entering LLM context."""
    tool_call_response = LLMResponse(
        content="",
        tool_calls=[ToolCallInfo(name="exec", arguments={"command": "echo"})],
        tokens_used=30,
    )
    final_response = LLMResponse(content="Done", tokens_used=20)
    captured_messages = []

    loop = _make_loop([tool_call_response, final_response])
    loop.skills.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "exec"}}]
    )
    # Return a string (non-dict) with invisible characters — goes through str() path
    loop.skills.execute = AsyncMock(return_value="out\x00put\uFEFF")

    original_chat = loop.llm.chat

    async def capturing_chat(system, messages, tools=None, **kwargs):
        captured_messages.append([dict(m) for m in messages])
        return await original_chat(system=system, messages=messages, tools=tools, **kwargs)

    loop.llm.chat = capturing_chat

    await loop.chat("run something")

    second_call = captured_messages[1]
    tool_msg = next(m for m in second_call if m.get("role") == "tool")
    assert "\x00" not in tool_msg["content"]
    assert "\uFEFF" not in tool_msg["content"]
    assert "output" in tool_msg["content"]


@pytest.mark.asyncio
async def test_memory_autoload_sanitized_in_chat():
    """Workspace search snippets with invisible chars are sanitized before entering LLM context."""
    normal_response = LLMResponse(content="Got it", tokens_used=30)
    loop = _make_loop([normal_response])
    loop.skills.get_tool_definitions = MagicMock(return_value=[])

    # Set up workspace with search results containing invisible chars
    loop.workspace = MagicMock()
    loop.workspace.looks_like_correction = MagicMock(return_value=False)
    loop.workspace.search = MagicMock(return_value=[
        {"file": "MEMORY.md", "snippet": "secret\u200B data\u202E here"},
    ])
    loop.workspace.append_daily_log = MagicMock()

    await loop.chat("find info")

    # The user message in chat history should have sanitized memory context
    user_msg = loop._chat_messages[0]["content"]
    assert "\u200B" not in user_msg
    assert "\u202E" not in user_msg
    assert "secret data here" in user_msg


@pytest.mark.asyncio
async def test_system_prompt_goals_sanitized():
    """Blackboard goals with invisible chars are sanitized in system prompt.

    format_dict uses json.dumps which ASCII-escapes Unicode, so we verify
    sanitize_for_prompt is called by monkeypatching format_dict to preserve
    raw Unicode.
    """
    import src.agent.loop as loop_mod
    loop = _make_loop()
    goals = {"objective": "find\u200B data\u202E now"}
    # Monkeypatch format_dict to output raw Unicode (simulates non-ASCII-safe encoding)
    original_format = loop_mod.format_dict
    loop_mod.format_dict = lambda d: json.dumps(d, indent=2, default=str, ensure_ascii=False)
    try:
        prompt = loop._build_chat_system_prompt(goals=goals)
        assert "\u200B" not in prompt
        assert "\u202E" not in prompt
        assert "find data now" in prompt
    finally:
        loop_mod.format_dict = original_format


@pytest.mark.asyncio
async def test_system_prompt_learnings_sanitized():
    """Workspace learnings with invisible chars are sanitized in system prompt."""
    loop = _make_loop()
    loop.workspace = MagicMock()
    loop.workspace.get_bootstrap_content = MagicMock(return_value="")
    loop.workspace.get_learnings_context = MagicMock(
        return_value="lesson\u200B one\u202E important"
    )
    prompt = loop._build_chat_system_prompt()
    assert "\u200B" not in prompt
    assert "\u202E" not in prompt
    assert "lesson one important" in prompt


def test_chat_prompt_includes_memory_recall_instruction():
    """Chat system prompt instructs agent to search memory before answering."""
    loop = _make_loop()
    prompt = loop._build_chat_system_prompt()
    assert "memory_search" in prompt
    assert "prior work" in prompt or "Before answering" in prompt


# === Tool Loop Detection Integration ===


@pytest.mark.asyncio
async def test_task_loop_detection_warns():
    """3 identical tool calls in task mode → warning prepended to result."""
    tool_call = LLMResponse(
        content="",
        tool_calls=[ToolCallInfo(name="search", arguments={"q": "stuck"})],
        tokens_used=10,
    )
    final = LLMResponse(content='{"result": {"done": true}}', tokens_used=10)
    # 3 tool-call rounds then final answer
    responses = [tool_call, tool_call, tool_call, final]

    loop = _make_loop(responses)
    loop.skills.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "search"}}]
    )
    loop.skills.execute = AsyncMock(return_value={"result": "same"})

    captured_messages = []
    original_chat = loop.llm.chat

    async def capturing_chat(system, messages, tools=None, **kwargs):
        captured_messages.append([dict(m) for m in messages])
        return await original_chat(system=system, messages=messages, tools=tools, **kwargs)

    loop.llm.chat = capturing_chat

    assignment = TaskAssignment(
        workflow_id="wf1", step_id="s1", task_type="research", input_data={}
    )
    result = await loop.execute_task(assignment)
    assert result.status == "complete"

    # The warning from the 3rd tool execution is visible in the 4th LLM call
    # (the final answer call), since it was appended to messages after execution.
    fourth_call = captured_messages[3]
    tool_msgs = [m for m in fourth_call if m.get("role") == "tool"]
    assert any("WARNING" in m.get("content", "") for m in tool_msgs)


@pytest.mark.asyncio
async def test_task_loop_detection_blocks():
    """5+ identical calls in task mode → tool not executed, error returned."""
    tool_call = LLMResponse(
        content="",
        tool_calls=[ToolCallInfo(name="search", arguments={"q": "stuck"})],
        tokens_used=10,
    )
    final = LLMResponse(content='{"result": {"done": true}}', tokens_used=10)
    responses = [tool_call] * 6 + [final]

    loop = _make_loop(responses)
    loop.skills.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "search"}}]
    )
    execute_count = 0

    async def counting_execute(*args, **kwargs):
        nonlocal execute_count
        execute_count += 1
        return {"result": "same"}

    loop.skills.execute = counting_execute

    assignment = TaskAssignment(
        workflow_id="wf1", step_id="s1", task_type="research", input_data={}
    )
    result = await loop.execute_task(assignment)
    assert result.status == "complete"
    # Block starts at 5th call (after 4 identical), so calls 5 and 6 are blocked
    # Calls 1-4 execute normally, calls 5-6 are blocked
    assert execute_count == 4


@pytest.mark.asyncio
async def test_task_loop_detection_terminates():
    """10+ identical calls in task mode → TaskResult failed."""
    tool_call = LLMResponse(
        content="",
        tool_calls=[ToolCallInfo(name="search", arguments={"q": "stuck"})],
        tokens_used=10,
    )
    # Provide plenty of responses (terminate should happen before exhausting them)
    responses = [tool_call] * 20

    loop = _make_loop(responses)
    loop.skills.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "search"}}]
    )
    loop.skills.execute = AsyncMock(return_value={"result": "same"})

    assignment = TaskAssignment(
        workflow_id="wf1", step_id="s1", task_type="research", input_data={}
    )
    result = await loop.execute_task(assignment)
    assert result.status == "failed"
    assert "loop detected" in result.error.lower()


@pytest.mark.asyncio
async def test_chat_loop_detection_warns():
    """Chat mode: 3 identical tool calls → warning in result."""
    tool_call = LLMResponse(
        content="",
        tool_calls=[ToolCallInfo(name="exec", arguments={"command": "fail"})],
        tokens_used=10,
    )
    final = LLMResponse(content="Done", tokens_used=10)
    responses = [tool_call, tool_call, tool_call, final]

    loop = _make_loop(responses)
    loop.skills.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "exec"}}]
    )
    loop.skills.execute = AsyncMock(return_value={"error": "command not found"})

    result = await loop.chat("run it")
    assert result["response"] == "Done"
    # Check that at least one tool message contains the warning
    tool_msgs = [m for m in loop._chat_messages if m.get("role") == "tool"]
    assert any("WARNING" in m.get("content", "") for m in tool_msgs)


@pytest.mark.asyncio
async def test_chat_loop_detection_terminates():
    """Chat mode: 10+ identical calls → response contains 'Stopped'."""
    tool_call = LLMResponse(
        content="",
        tool_calls=[ToolCallInfo(name="exec", arguments={"command": "fail"})],
        tokens_used=10,
    )
    # Provide plenty (terminate should kick in before all are used)
    responses = [tool_call] * 20

    loop = _make_loop(responses)
    loop.skills.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "exec"}}]
    )
    loop.skills.execute = AsyncMock(return_value={"error": "command not found"})

    result = await loop.chat("run it")
    assert "Stopped" in result["response"]
    assert "loop detected" in result["response"].lower()


@pytest.mark.asyncio
async def test_detector_reset_on_new_task():
    """Loop detector is reset when starting a new task."""
    tool_call = LLMResponse(
        content="",
        tool_calls=[ToolCallInfo(name="search", arguments={"q": "x"})],
        tokens_used=10,
    )
    final = LLMResponse(content='{"result": {"ok": true}}', tokens_used=10)

    # First task: 3 identical calls (builds up warn state)
    loop = _make_loop([tool_call, tool_call, tool_call, final, tool_call, final])
    loop.skills.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "search"}}]
    )
    loop.skills.execute = AsyncMock(return_value={"result": "same"})

    assignment = TaskAssignment(
        workflow_id="wf1", step_id="s1", task_type="research", input_data={}
    )
    result1 = await loop.execute_task(assignment)
    assert result1.status == "complete"

    # Second task: 1 identical call should NOT warn (detector was reset)
    captured_messages = []
    original_chat = loop.llm.chat

    async def capturing_chat(system, messages, tools=None, **kwargs):
        captured_messages.append([dict(m) for m in messages])
        return await original_chat(system=system, messages=messages, tools=tools, **kwargs)

    loop.llm.chat = capturing_chat

    assignment2 = TaskAssignment(
        workflow_id="wf2", step_id="s2", task_type="research", input_data={}
    )
    result2 = await loop.execute_task(assignment2)
    assert result2.status == "complete"
    # No warning should be present in any tool result
    for call_msgs in captured_messages:
        for m in call_msgs:
            if m.get("role") == "tool":
                assert "WARNING" not in m.get("content", "")


@pytest.mark.asyncio
async def test_detector_reset_on_chat_reset():
    """Loop detector is reset when chat history is cleared."""
    tool_call = LLMResponse(
        content="",
        tool_calls=[ToolCallInfo(name="exec", arguments={"command": "fail"})],
        tokens_used=10,
    )
    final = LLMResponse(content="Done", tokens_used=10)

    # First chat: 3 identical calls (builds up warn state)
    loop = _make_loop([tool_call, tool_call, tool_call, final, tool_call, final])
    loop.skills.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "exec"}}]
    )
    loop.skills.execute = AsyncMock(return_value={"error": "command not found"})

    await loop.chat("run it")

    # Reset chat (should also reset detector)
    await loop.reset_chat()

    # Second chat: 1 identical call should NOT warn
    result2 = await loop.chat("run it again")
    assert result2["response"] == "Done"
    # Only tool messages from the second chat (after reset)
    tool_msgs = [m for m in loop._chat_messages if m.get("role") == "tool"]
    assert not any("WARNING" in m.get("content", "") for m in tool_msgs)


@pytest.mark.asyncio
async def test_task_terminate_mid_batch_blocks_tool():
    """If terminate threshold is hit mid-batch via recording, the tool is blocked (not executed)."""
    # Two identical tools per batch.  After 5 batches (10 calls), batch 5's
    # second tool should trigger terminate via check_before in per-tool loop.
    # But the pre-scan uses would_terminate which checks BEFORE any intra-batch
    # recording, so it may pass.  The per-tool check_before must still block.
    two_tools = LLMResponse(
        content="",
        tool_calls=[
            ToolCallInfo(name="search", arguments={"q": "stuck"}),
            ToolCallInfo(name="search", arguments={"q": "stuck"}),
        ],
        tokens_used=10,
    )
    final = LLMResponse(content='{"result": {"ok": true}}', tokens_used=10)
    # 5 batches of 2 = 10 tool calls, then final
    responses = [two_tools] * 6 + [final]

    loop = _make_loop(responses)
    loop.skills.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "search"}}]
    )
    execute_count = 0

    async def counting_execute(*args, **kwargs):
        nonlocal execute_count
        execute_count += 1
        return {"result": "same"}

    loop.skills.execute = counting_execute

    assignment = TaskAssignment(
        workflow_id="wf1", step_id="s1", task_type="research", input_data={}
    )
    result = await loop.execute_task(assignment)
    # Should either terminate or complete, but tools should NOT all execute
    # With 2 per batch: batch 1 (2 ok), batch 2 (2 ok=4 total), batch 3 (block+block),
    # batch 4 (block+block), batch 5 pre-scan: 8 recorded, not 9 → pass,
    # per-tool: first check = 8 → block, record → 9, second check = 9 → terminate (blocked)
    # The terminate path in per-tool is handled as block, so the loop continues.
    # Eventually the pre-scan catches it.
    assert execute_count == 4  # Only first 4 calls execute (block kicks in at 5th)


# === Richer Daily Logs ===


@pytest.fixture
def workspace_loop():
    """Yield (loop_factory, workspace) with a real workspace; clean up after."""
    import shutil
    import tempfile

    from src.agent.workspace import WorkspaceManager

    tmpdir = tempfile.mkdtemp()
    workspace = WorkspaceManager(workspace_dir=tmpdir)

    def factory(llm_responses=None):
        loop = _make_loop(llm_responses)
        loop.workspace = workspace
        return loop

    yield factory, workspace
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestLogChatTurn:
    @pytest.mark.asyncio
    async def test_log_chat_turn_captures_tools(self, workspace_loop):
        """_log_chat_turn includes tool names used during the turn."""
        factory, workspace = workspace_loop
        tool_response = LLMResponse(
            content="",
            tool_calls=[ToolCallInfo(name="web_search", arguments={"q": "test"})],
            tokens_used=20,
        )
        final = LLMResponse(content="Found it!", tokens_used=20)
        loop = factory([tool_response, final])
        loop.skills.get_tool_definitions = MagicMock(
            return_value=[{"type": "function", "function": {"name": "web_search"}}]
        )
        loop.skills.execute = AsyncMock(return_value={"results": ["r1"]})

        await loop.chat("find something")

        daily = workspace.load_daily_logs(days=1)
        assert "Tools: web_search" in daily
        assert "Chat:" in daily
        assert "Response:" in daily

    @pytest.mark.asyncio
    async def test_log_chat_turn_strips_memory_context(self, workspace_loop):
        """_log_chat_turn strips auto-loaded memory context from user summary."""
        factory, workspace = workspace_loop
        loop = factory([LLMResponse(content="Understood", tokens_used=20)])
        loop.skills.get_tool_definitions = MagicMock(return_value=[])

        msg = "Do the task\n[Relevant memory auto-loaded]\nSome old context here"
        await loop.chat(msg)

        daily = workspace.load_daily_logs(days=1)
        assert "Do the task" in daily
        assert "auto-loaded" not in daily

    @pytest.mark.asyncio
    async def test_silent_response_not_logged(self, workspace_loop):
        """__SILENT__ responses (empty string) should not be logged."""
        factory, workspace = workspace_loop
        loop = factory([LLMResponse(content="__SILENT__", tokens_used=10)])
        loop.skills.get_tool_definitions = MagicMock(return_value=[])

        await loop.chat("heartbeat ping")

        daily = workspace.load_daily_logs(days=1)
        assert "heartbeat ping" not in daily


class TestTaskCompletionLogging:
    @pytest.mark.asyncio
    async def test_task_completion_writes_daily_log(self, workspace_loop):
        """Successful task writes a completion entry to daily log."""
        factory, workspace = workspace_loop
        tool_response = LLMResponse(
            content="",
            tool_calls=[ToolCallInfo(name="exec", arguments={"cmd": "ls"})],
            tokens_used=50,
        )
        final = LLMResponse(content='{"result": {"done": true}}', tokens_used=30)
        loop = factory([tool_response, final])
        loop.skills.get_tool_definitions = MagicMock(
            return_value=[{"type": "function", "function": {"name": "exec"}}]
        )
        loop.skills.execute = AsyncMock(return_value={"exit_code": 0})

        assignment = TaskAssignment(
            workflow_id="wf1", step_id="s1", task_type="research",
            input_data={"query": "test"},
        )
        result = await loop.execute_task(assignment)
        assert result.status == "complete"

        daily = workspace.load_daily_logs(days=1)
        assert "Task complete: research" in daily
        assert "Tools: exec" in daily
        assert "iterations" in daily
        assert "tokens" in daily

    @pytest.mark.asyncio
    async def test_task_failure_writes_daily_log(self, workspace_loop):
        """Max-iterations failure writes a FAILED entry to daily log."""
        factory, workspace = workspace_loop
        responses = [
            LLMResponse(
                content="",
                tool_calls=[ToolCallInfo(name="search", arguments={"q": f"q_{i}"})],
                tokens_used=10,
            )
            for i in range(25)
        ]
        loop = factory(responses)
        loop.skills.get_tool_definitions = MagicMock(
            return_value=[{"type": "function", "function": {"name": "search"}}]
        )
        loop.skills.execute = AsyncMock(return_value={"result": "ok"})

        assignment = TaskAssignment(
            workflow_id="wf1", step_id="s1", task_type="analysis",
            input_data={"topic": "test"},
        )
        result = await loop.execute_task(assignment)
        assert result.status == "failed"

        daily = workspace.load_daily_logs(days=1)
        assert "Task FAILED (max iterations): analysis" in daily
        assert "tokens" in daily


class TestCollectToolNames:
    def test_extracts_unique_names_in_order(self):
        """_collect_tool_names returns unique tool names in first-appearance order."""
        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "search"}},
                {"function": {"name": "exec"}},
            ]},
            {"role": "tool", "content": "ok"},
            {"role": "tool", "content": "ok"},
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "search"}},  # duplicate
                {"function": {"name": "write_file"}},
            ]},
        ]
        names = AgentLoop._collect_tool_names(messages)
        assert names == ["search", "exec", "write_file"]

    def test_empty_messages(self):
        assert AgentLoop._collect_tool_names([]) == []

    def test_no_tool_calls(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        assert AgentLoop._collect_tool_names(messages) == []
