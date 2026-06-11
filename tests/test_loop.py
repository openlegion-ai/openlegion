"""Unit tests for the agent execution loop.

Uses mock LLM to verify:
- Proper message role alternation (user/assistant/tool)
- Bounded iterations
- Cancellation
- Token budget enforcement
- Final output parsing
"""

import asyncio
import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.loop import (
    _BLACKBOARD_TOOLS,
    _READ_ONLY_TOOLS,
    HEARTBEAT_MAX_ITERATIONS,
    AgentLoop,
    _heartbeat_mode,
)
from src.shared.types import (
    SILENT_REPLY_TOKEN,
    LLMResponse,
    TaskAssignment,
    TokenBudget,
    ToolCallInfo,
)

# Audit-pass: ``_has_outbound_effect`` was moved from module-level to a
# ``@staticmethod`` on ``AgentLoop`` so it sits with the other lazy-
# completion guards. Re-expose as a module name so test bodies that
# call it bare-name keep working without rewrites.
_has_outbound_effect = AgentLoop._has_outbound_effect


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
        memory.get_high_salience_facts = AsyncMock(return_value=[])
        memory.decay_all = AsyncMock()
        memory.search = AsyncMock(return_value=[])
        memory.search_hierarchical = AsyncMock(return_value=[])
        memory.log_action = AsyncMock()
        memory.store_tool_outcome = AsyncMock()
        memory.get_tool_history = MagicMock(return_value=[])
        memory._run_db = AsyncMock(return_value=None)

    tools = MagicMock()
    tools.get_tool_definitions = MagicMock(return_value=[])
    tools.get_descriptions = MagicMock(return_value="- no tools")
    tools.list_tools = MagicMock(return_value=[])
    tools.is_parallel_safe = MagicMock(return_value=True)
    tools.get_loop_exempt_tools = MagicMock(return_value=frozenset())
    tools.operator_only_tools = MagicMock(return_value=frozenset())

    llm = MagicMock()
    if llm_responses:
        llm.chat = AsyncMock(side_effect=llm_responses)
    else:
        llm.chat = AsyncMock(return_value=LLMResponse(content='{"result": {"answer": "42"}}', tokens_used=100))
    llm.default_model = "test-model"

    # Task/handoff paths call chat_collect (streaming). Delegate to whatever
    # llm.chat is at call time so existing mocks (incl. per-test reassignments
    # of loop.llm.chat) drive both the streaming and non-streaming paths.
    async def _chat_collect_delegate(*args, **kwargs):
        return await llm.chat(*args, **kwargs)
    llm.chat_collect = _chat_collect_delegate

    mesh_client = MagicMock()
    mesh_client.is_standalone = False
    mesh_client.send_system_message = AsyncMock(return_value={})
    mesh_client.read_blackboard = AsyncMock(return_value=None)
    mesh_client.list_agents = AsyncMock(return_value={})

    loop = AgentLoop(
        agent_id="test_agent",
        role="research",
        memory=memory,
        tools=tools,
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
    assert result.tokens_used == 100  # no tools available → no nudge, completes on iteration 0
    assert loop.tasks_completed == 1
    assert loop.state == "idle"


@pytest.mark.asyncio
async def test_task_nudge_fires_when_tools_available():
    """Agent gets nudged once when it responds with text on iteration 0 but has tools.

    After the nudge the LLM uses a tool, then produces a final answer —
    the healthy 3-iteration shape (text → nudge → tool → final).
    """
    responses = [
        # Iter 0: text-only — triggers the nudge.
        LLMResponse(content="I'll do this now.", tokens_used=50),
        # Iter 1: nudged — calls a tool.
        LLMResponse(
            content="",
            tool_calls=[ToolCallInfo(name="web_search", arguments={"q": "x"})],
            tokens_used=60,
        ),
        # Iter 2: final answer after seeing tool result.
        LLMResponse(content='{"result": {"answer": "42"}}', tokens_used=70),
    ]
    loop = _make_loop(responses)
    loop.tools.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "web_search"}}]
    )
    loop.tools.execute = AsyncMock(return_value={"results": ["r1"]})
    assignment = TaskAssignment(
        workflow_id="wf1", step_id="s1", task_type="research", input_data={"query": "test"}
    )

    result = await loop.execute_task(assignment)
    assert result.status == "complete"
    assert result.tokens_used == 180
    assert loop.tasks_completed == 1


@pytest.mark.asyncio
async def test_lazy_completion_guard_fails_text_only_after_nudge():
    """Bug F: LLM responds with text-only acknowledgment ("I'll do it now")
    on iteration 0, gets nudged, then responds with text-only again on
    iteration 1 — that's an acknowledgment with no real work performed.

    Pre-fix behavior: the loop fell through to "final answer is done" and
    marked the task complete with zero side effects. The PR #918
    pathological-success guard (iterations==1 + tokens==0 + empty content)
    didn't catch it because tokens > 0 and content is non-empty chatter.

    Post-fix: the lazy-completion guard trips on iterations_executed > 1
    AND tool_calls_count == 0 and downgrades the task to ``failed`` with
    error containing ``no_action_taken``.
    """
    responses = [
        # Iter 0: "I'll do it now" — text-only.
        LLMResponse(content="Acknowledged — I'll run the job now.", tokens_used=42),
        # Iter 1: still text-only after the nudge.
        LLMResponse(content="Task is complete.", tokens_used=18),
    ]
    loop = _make_loop(responses)
    # Tools must be available so the iter-0 nudge fires (and so the
    # LLM had every opportunity to call one).
    loop.tools.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "web_search"}}]
    )
    assignment = TaskAssignment(
        workflow_id="wf1", step_id="s1", task_type="research",
        input_data={"query": "test"},
    )

    result = await loop.execute_task(assignment)

    assert result.status == "failed"
    # Round-5 audit pass: execute_task's lazy-completion error renamed
    # from ``no_action_taken`` to ``no_outbound_effects`` to share
    # wording with the chat-path guard. The earlier wording is still
    # accepted for any pre-rename log scrapers.
    assert "no_outbound_effects" in (result.error or "")
    assert loop.tasks_failed == 1
    assert loop.tasks_completed == 0
    assert loop.state == "idle"


@pytest.mark.asyncio
async def test_lazy_guard_allows_structured_final_after_nudge():
    """Bug F counter-test (codex r4): a STRUCTURED final response after
    the nudge — e.g. ``{"result": {"status": "impossible", "reason":
    "..."}}`` — is a legitimate completion even with zero tool calls.

    The iter-0 nudge text explicitly invites a final JSON response for
    genuine completion or impossibility; the guard must let that
    through. Plain-text "I'll do it now" / "Done!" chatter still fails.
    """
    responses = [
        # Iter 0: text-only — triggers the nudge.
        LLMResponse(content="Looking at this...", tokens_used=30),
        # Iter 1: structured final response declaring impossibility.
        LLMResponse(
            content='{"result": {"status": "impossible", "reason": "needs API key"}}',
            tokens_used=40,
        ),
    ]
    loop = _make_loop(responses)
    loop.tools.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "web_search"}}]
    )
    assignment = TaskAssignment(
        workflow_id="wf1", step_id="s1", task_type="research",
        input_data={"query": "test"},
    )

    result = await loop.execute_task(assignment)

    # Structured final answer with "result" key → escapes the guard.
    assert result.status == "complete"
    assert loop.tasks_completed == 1
    assert loop.tasks_failed == 0
    # And the result_data reflects the structured answer.
    assert result.result == {"status": "impossible", "reason": "needs API key"}


@pytest.mark.asyncio
async def test_lazy_guard_rejects_empty_dict_result():
    """Bug F (codex r6): ``{"result": {}}`` is the same chatter-in-
    structure bypass as ``{"result": "..."}`` — it satisfies isinstance
    + dict but conveys nothing. The guard requires a non-empty dict."""
    responses = [
        LLMResponse(content="ok", tokens_used=20),
        LLMResponse(content='{"result": {}}', tokens_used=30),
    ]
    loop = _make_loop(responses)
    loop.tools.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "web_search"}}]
    )
    assignment = TaskAssignment(
        workflow_id="wf1", step_id="s1", task_type="research",
        input_data={"query": "test"},
    )
    result = await loop.execute_task(assignment)
    assert result.status == "failed"
    assert "no_outbound_effects" in (result.error or "")


@pytest.mark.asyncio
async def test_lazy_guard_rejects_scalar_result_chatter():
    """Bug F (codex r5): the structured-final escape hatch must require
    ``result`` to be a dict, not just present. Otherwise an LLM can
    paper over the guard by wrapping chatter in ``{"result": "I'll do
    it now"}`` — the prompt contract calls for ``{"result": {...}}``."""
    responses = [
        LLMResponse(content="Sure thing.", tokens_used=20),
        LLMResponse(content='{"result": "I will do it now"}', tokens_used=30),
    ]
    loop = _make_loop(responses)
    loop.tools.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "web_search"}}]
    )
    assignment = TaskAssignment(
        workflow_id="wf1", step_id="s1", task_type="research",
        input_data={"query": "test"},
    )
    result = await loop.execute_task(assignment)
    # Scalar result with no tool calls = chatter — guard trips.
    assert result.status == "failed"
    assert "no_outbound_effects" in (result.error or "")
    assert loop.tasks_failed == 1


@pytest.mark.asyncio
async def test_lazy_guard_survives_message_compaction():
    """Bug F (codex r4): summarizing compaction can drop the assistant-
    with-tool_calls entry from the live ``messages`` list. The guard
    uses a locally-tracked counter (seeded from messages at task start,
    then incremented at dispatch) so the tool-call signal survives.

    Simulated by clearing ``messages`` between the tool-call iteration
    and the final-answer iteration — equivalent to compaction dropping
    the assistant-with-tool_calls record.
    """
    responses = [
        # Iter 0: text-only — triggers the nudge.
        LLMResponse(content="thinking...", tokens_used=20),
        # Iter 1: nudged — calls a tool.
        LLMResponse(
            content="",
            tool_calls=[ToolCallInfo(name="web_search", arguments={"q": "x"})],
            tokens_used=30,
        ),
        # Iter 2: legitimate final summary referencing the tool result.
        LLMResponse(content='{"result": {"summary": "found 3"}}', tokens_used=40),
    ]
    loop = _make_loop(responses)
    loop.tools.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "web_search"}}]
    )
    loop.tools.execute = AsyncMock(return_value={"results": ["r1", "r2", "r3"]})

    # Stub maybe_compact to wipe message history right after the tool
    # iteration runs — proves the counter is independent of `messages`.
    compacted_once = {"done": False}
    if loop.context_manager is not None:
        async def _strip_history(_sys_prompt, msgs):
            if not compacted_once["done"] and any(
                m.get("role") == "tool" for m in msgs
            ):
                compacted_once["done"] = True
                # Keep only the latest user message; drop everything else
                # (worst-case compaction: tool_calls entries gone).
                msgs = [m for m in msgs if m.get("role") == "user"][-1:]
            return msgs, None
        loop.context_manager.maybe_compact = _strip_history

    assignment = TaskAssignment(
        workflow_id="wf1", step_id="s1", task_type="research",
        input_data={"query": "test"},
    )

    result = await loop.execute_task(assignment)

    # Despite the live messages losing the tool_calls record, the
    # counter remembered the dispatch — guard does NOT trip.
    assert result.status == "complete"
    assert loop.tasks_completed == 1
    assert loop.tasks_failed == 0


@pytest.mark.asyncio
async def test_lazy_guard_fires_on_iter0_text_only_with_tools_available():
    """Bug F regression follow-up (operator reported post-#932): the
    iter-0 nudge is gated on ``available_tools`` being truthy. If the
    LLM responds text-only on iter 0 and the nudge skips for any reason
    (e.g. responds before the nudge wraps), the prior guard's
    ``iterations_executed > 1`` precondition let the task complete as
    done. The fix drops that precondition — text-only + zero tools +
    not-structured is a hard fail regardless of iteration count.

    This test forces the scenario by feeding ONE LLM response and
    relying on the nudge path: the nudge appends a follow-up message
    and continues, the LLM mock raises StopAsyncIteration on the second
    call, which propagates as an exception out of execute_task. To
    exercise the guard directly we instead provide a single text-only
    response AND no tools available (so the nudge is skipped at iter 0
    and we fall straight to terminal). That's exactly the trend-scout
    failure mode.
    """
    text_only = LLMResponse(content="On it — running now.", tokens_used=42)
    loop = _make_loop([text_only])
    # available_tools=[] → nudge skipped at iter 0 → fall straight
    # through to terminal with iterations_executed=1 (the exact path
    # operator's trend-scout task hit in production).
    assignment = TaskAssignment(
        workflow_id="wf1", step_id="s1", task_type="research",
        input_data={"query": "test"},
    )
    result = await loop.execute_task(assignment)
    assert result.status == "failed", (
        f"text-only iter-0 completion must fail under the post-regression "
        f"guard; got status={result.status} error={result.error}"
    )
    assert "no_outbound_effects" in (result.error or "")
    assert loop.tasks_failed == 1
    assert loop.tasks_completed == 0


@pytest.mark.asyncio
async def test_lazy_guard_explained_deferral_closes_done():
    """Bug 3 (explained-deferral carve-out): an agent that calls a
    READ-ONLY tool to verify a prerequisite AND then explains its
    decision in plain prose (not the structured ``{"result": {...}}``
    envelope) is correctly DEFERRING, not ghosting. The task closes as
    ``complete`` with a ``deferred`` result payload — distinct from the
    genuine ghost case (zero tools / empty text) which still fails."""
    responses = [
        # Iter 0: call a read-only tool to check backpressure / dedup.
        LLMResponse(
            content="",
            tool_calls=[ToolCallInfo(name="read_blackboard", arguments={"key": "queue"})],
            tokens_used=20,
        ),
        # Iter 1: explain the deferral decision in plain prose (NOT a
        # structured result envelope).
        LLMResponse(
            content=(
                "The upstream queue already has a pending item for this "
                "request, so I'm deferring to avoid duplicate work."
            ),
            tokens_used=30,
        ),
    ]
    loop = _make_loop(responses)
    loop.tools.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "read_blackboard"}}]
    )
    loop.tools.execute = AsyncMock(return_value={"value": "pending"})
    assignment = TaskAssignment(
        workflow_id="wf1", step_id="s1", task_type="research",
        input_data={"query": "test"},
    )
    result = await loop.execute_task(assignment)
    assert result.status == "complete", (
        f"explained read-only deferral must close done, not failed; "
        f"got status={result.status} error={result.error}"
    )
    assert result.result["status"] == "deferred"
    assert "deferring" in result.result["reason"]
    # summary mirrors reason so the done back-edge surfaces the explanation
    assert result.result["summary"] == result.result["reason"]
    assert loop.tasks_completed == 1
    assert loop.tasks_failed == 0


@pytest.mark.asyncio
async def test_lazy_guard_read_only_empty_text_still_fails():
    """Ghost-case preservation: a read-only tool call followed by an
    EMPTY response (no explanation) is NOT a deferral — it's a ghost
    turn. The carve-out requires non-empty prose, so this still fails."""
    responses = [
        LLMResponse(
            content="",
            tool_calls=[ToolCallInfo(name="read_blackboard", arguments={"key": "queue"})],
            tokens_used=20,
        ),
        # Iter 1: empty / whitespace-only — no explanation.
        LLMResponse(content="   ", tokens_used=30),
    ]
    loop = _make_loop(responses)
    loop.tools.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "read_blackboard"}}]
    )
    loop.tools.execute = AsyncMock(return_value={"value": "pending"})
    assignment = TaskAssignment(
        workflow_id="wf1", step_id="s1", task_type="research",
        input_data={"query": "test"},
    )
    result = await loop.execute_task(assignment)
    assert result.status == "failed"
    assert "no_outbound_effects" in (result.error or "")
    assert loop.tasks_failed == 1
    assert loop.tasks_completed == 0


@pytest.mark.asyncio
async def test_lazy_guard_passes_iter0_structured_noop():
    """Counter-test to the iter-0 fix: a legitimate one-shot noop task
    must still complete when the LLM returns the documented contract
    ``{"result": {"status": "noop", "reason": "..."}}``. The
    structured-final escape handles this — guard does NOT trip."""
    structured_noop = LLMResponse(
        content='{"result": {"status": "noop", "reason": "queue empty"}}',
        tokens_used=42,
    )
    loop = _make_loop([structured_noop])
    assignment = TaskAssignment(
        workflow_id="wf1", step_id="s1", task_type="research",
        input_data={"query": "test"},
    )
    result = await loop.execute_task(assignment)
    assert result.status == "complete"
    assert result.result == {"status": "noop", "reason": "queue empty"}
    assert loop.tasks_completed == 1


@pytest.mark.asyncio
async def test_iter0_structured_final_with_tools_skips_nudge():
    """Codex r8: when the LLM returns a structured ``{"result": {...}}``
    on iter 0 AND tools are available, the prior code would still nudge
    (since the structured check was after the nudge branch). That wasted
    a turn and risked a false-positive lazy-completion failure if the
    LLM responded with text-only "I already told you" on iter 1.

    Fix: structured-final is detected up front. The nudge branch now
    requires ``not is_structured_final``, so a legitimate one-shot
    structured noop with tools available completes WITHOUT nudging.
    Verified by feeding a SINGLE LLM response: if the nudge fired,
    the test would hit StopAsyncIteration looking for a second response.
    """
    structured_only = LLMResponse(
        content='{"result": {"status": "noop", "reason": "queue empty"}}',
        tokens_used=42,
    )
    # Single response — if the nudge fires, the loop will call llm.chat
    # a second time and raise StopAsyncIteration.
    loop = _make_loop([structured_only])
    # Tools available — the OLD code would nudge here. New code detects
    # structured-final and skips the nudge.
    loop.tools.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "web_search"}}]
    )
    assignment = TaskAssignment(
        workflow_id="wf1", step_id="s1", task_type="research",
        input_data={"query": "test"},
    )
    result = await loop.execute_task(assignment)
    assert result.status == "complete"
    assert result.result == {"status": "noop", "reason": "queue empty"}
    # Exactly one LLM call — confirms the nudge did not fire.
    assert result.tokens_used == 42
    assert loop.tasks_completed == 1


@pytest.mark.asyncio
async def test_lazy_guard_passes_iter0_with_one_tool_call():
    """Counter-test: a one-iteration tool-call followed by a final
    text response must complete. The tool dispatch increments the
    counter and the guard skips."""
    responses = [
        LLMResponse(
            content="",
            tool_calls=[ToolCallInfo(name="web_search", arguments={"q": "x"})],
            tokens_used=30,
        ),
        # Final response after tool — text is acceptable here because
        # tool_calls_count > 0 (real work performed).
        LLMResponse(content="Done — see search results.", tokens_used=40),
    ]
    loop = _make_loop(responses)
    loop.tools.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "web_search"}}]
    )
    loop.tools.execute = AsyncMock(return_value={"results": ["r1"]})
    assignment = TaskAssignment(
        workflow_id="wf1", step_id="s1", task_type="research",
        input_data={"query": "test"},
    )
    result = await loop.execute_task(assignment)
    assert result.status == "complete"
    assert loop.tasks_completed == 1


@pytest.mark.asyncio
async def test_task_nudge_skipped_when_no_tools():
    """Agent completes immediately on iteration 0 when no tools are available."""
    loop = _make_loop()
    # Default mock returns [] for get_tool_definitions — no tools
    assignment = TaskAssignment(
        workflow_id="wf1", step_id="s1", task_type="research", input_data={"query": "test"}
    )

    result = await loop.execute_task(assignment)
    assert result.status == "complete"
    assert result.tokens_used == 100  # no nudge
    assert loop.tasks_completed == 1


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
    loop.tools.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "web_search"}}]
    )
    loop.tools.execute = AsyncMock(return_value={"results": ["r1"]})

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
    # Pin the cap below the supplied response count so the loop hits the
    # iteration ceiling (not the mock running dry). The default is now high.
    loop.MAX_ITERATIONS = 20
    loop.tools.get_tool_definitions = MagicMock(return_value=[{"type": "function", "function": {"name": "search"}}])
    loop.tools.execute = AsyncMock(return_value={"result": "ok"})

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

from unittest.mock import patch  # noqa: E402

import httpx  # noqa: E402

from src.agent.loop import _llm_call_with_retry  # noqa: E402


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
async def test_llm_retry_on_remote_protocol_error():
    """Retry on transient protocol disconnects during LLM calls."""
    success = LLMResponse(content='{"result": {}}', tokens_used=50)
    mock_fn = AsyncMock(side_effect=[httpx.RemoteProtocolError("incomplete chunked read"), success])

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
    loop.tools.get_tool_definitions = MagicMock(return_value=[])

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
    loop.tools.get_tool_definitions = MagicMock(return_value=[])

    result = await loop.chat("heartbeat ping")
    assert result["response"] == ""


@pytest.mark.asyncio
async def test_silent_reply_token_with_whitespace():
    """__SILENT__ with surrounding whitespace should still be detected."""
    silent_response = LLMResponse(content="  __SILENT__  \n", tokens_used=10)
    loop = _make_loop([silent_response])
    loop.tools.get_tool_definitions = MagicMock(return_value=[])

    result = await loop.chat("cron tick")
    assert result["response"] == ""


@pytest.mark.asyncio
async def test_non_silent_reply_passes_through():
    """Normal LLM responses should pass through unchanged."""
    normal_response = LLMResponse(content="Hello, how can I help?", tokens_used=20)
    loop = _make_loop([normal_response])
    loop.tools.get_tool_definitions = MagicMock(return_value=[])

    result = await loop.chat("hi")
    assert result["response"] == "Hello, how can I help?"


@pytest.mark.asyncio
async def test_silent_token_after_tool_rounds_exhausted():
    """__SILENT__ at the max-rounds force-compose exit must be honoured:
    empty reply, NO Bug-3 marker substituted.

    The loop runs ``CHAT_MAX_TOOL_ROUNDS`` tool rounds (consuming that
    many tool-call responses), breaks, then issues the force-compose
    call with ``tools=None`` — so the response that lands AT the
    force-compose exit must itself be the ``__SILENT__`` reply for this
    to exercise deliberate silence there.
    """
    loop = _make_loop()
    n = loop.CHAT_MAX_TOOL_ROUNDS
    # Use different arguments each round to avoid triggering loop detection.
    responses = [
        LLMResponse(
            content="",
            tool_calls=[ToolCallInfo(name="search", arguments={"q": f"q_{i}"})],
            tokens_used=10,
        )
        for i in range(n)
    ]
    # The (n+1)th call is the force-compose (tools withheld) — make it
    # the deliberate ``__SILENT__`` reply.
    responses.append(LLMResponse(content="__SILENT__", tokens_used=10))
    loop.llm.chat = AsyncMock(side_effect=responses)
    loop.tools.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "search"}}]
    )
    loop.tools.execute = AsyncMock(return_value={"result": "ok"})

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
    loop.tools.get_tool_definitions = MagicMock(return_value=[])

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
    loop.tools.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "search"}}]
    )
    loop.tools.execute = AsyncMock(return_value={"result": "ok"})

    original_chat = loop.llm.chat

    async def capturing_chat(system, messages, tools=None, **kwargs):
        captured_messages.append([dict(m) for m in messages])
        return await original_chat(system=system, messages=messages, tools=tools, **kwargs)

    loop.llm.chat = capturing_chat

    # Inject steer after first tool call starts
    # We need to inject before the second LLM call
    original_execute = loop.tools.execute

    async def execute_with_steer(*args, **kwargs):
        # Inject steer message during tool execution
        await loop.inject_steer("stop, do this instead")
        return await original_execute(*args, **kwargs)

    loop.tools.execute = execute_with_steer

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
    loop.tools.get_tool_definitions = MagicMock(return_value=[])

    # Queue multiple steer messages while idle
    await loop.inject_steer("first update")
    await loop.inject_steer("second update")

    result = await loop.chat("go")
    assert result["response"] == "Acknowledged all"

    # Both should be in the first user message
    user_msg = loop._chat_messages[0]["content"]
    assert "first update" in user_msg
    assert "second update" in user_msg


@pytest.mark.asyncio
async def test_steer_during_final_response_continues_loop():
    """Steer arriving during LLM's final answer re-enters the loop.

    When the LLM returns text (no tool calls), a pending steer should
    prevent immediate return — the assistant's response stays in context
    and the steer is injected so the next LLM call can adjust.
    """
    responses = [
        LLMResponse(content="Here is my original answer.", tokens_used=50),
        LLMResponse(content="Updated answer after steer.", tokens_used=50),
    ]

    loop = _make_loop(responses)

    original_chat = loop.llm.chat
    steer_injected = False

    async def chat_with_steer(system, messages, tools=None, **kwargs):
        nonlocal steer_injected
        result = await original_chat(system=system, messages=messages, tools=tools, **kwargs)
        if not steer_injected:
            steer_injected = True
            await loop.inject_steer("actually focus on Y instead")
        return result

    loop.llm.chat = chat_with_steer

    result = await loop.chat("Research X")

    # The final response should be the SECOND (steered) response
    assert result["response"] == "Updated answer after steer."


@pytest.mark.asyncio
async def test_steer_interrupt_limit():
    """After _MAX_STEER_INTERRUPTS, agent returns even with pending steers."""
    from src.agent.loop import _MAX_STEER_INTERRUPTS

    responses = [
        LLMResponse(content=f"attempt {i}", tokens_used=10)
        for i in range(_MAX_STEER_INTERRUPTS + 1)
    ]

    loop = _make_loop(responses)

    original_chat = loop.llm.chat

    async def always_steer(system, messages, tools=None, **kwargs):
        result = await original_chat(system=system, messages=messages, tools=tools, **kwargs)
        await loop.inject_steer("redirect again")
        return result

    loop.llm.chat = always_steer

    result = await loop.chat("start")

    # Should have returned the last response without looping forever
    assert result["response"] == f"attempt {_MAX_STEER_INTERRUPTS}"


# === Context Warning Integration ===


def test_context_warning_in_chat_system_prompt():
    """When context >= 80%, the CONTEXT WARNING is emitted as a VOLATILE fragment
    (relocated out of the cached system prefix, onto _volatile_prompt_suffix)."""
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
    # Volatile fragments live below the cache breakpoint, not in the system prompt.
    assert "CONTEXT WARNING" not in prompt
    assert "CONTEXT WARNING" in loop._volatile_prompt_suffix


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
        await loop._record_failure("exec", "command not found", arguments={"command": "bad"})
        history = loop.memory.get_tool_history("exec")
        assert len(history) == 1
        assert history[0]["success"] is False

    @pytest.mark.asyncio
    async def test_record_failure_logs_repeat_warning(self, caplog):
        """3+ failures with the same tool + args emit the grep-able
        ``repeat_failure`` warning (issue #1012 acceptance instrument);
        below the threshold stays silent."""
        loop = _make_loop(real_memory=True)
        with caplog.at_level(logging.WARNING, logger="agent.loop"):
            for _ in range(2):
                await loop._record_failure(
                    "exec", "command not found", arguments={"command": "bad"},
                )
            assert "repeat_failure" not in caplog.text  # 2 < threshold
            await loop._record_failure(
                "exec", "command not found", arguments={"command": "bad"},
            )
        assert "repeat_failure" in caplog.text
        assert "tool=exec count=3" in caplog.text

    @pytest.mark.asyncio
    async def test_record_failure_no_repeat_warning_for_different_args(self, caplog):
        """Same tool failing with DIFFERENT args is not a repeat — the
        counter keys on (tool_name, params_hash)."""
        loop = _make_loop(real_memory=True)
        with caplog.at_level(logging.WARNING, logger="agent.loop"):
            for i in range(4):
                await loop._record_failure(
                    "exec", "command not found", arguments={"command": f"bad_{i}"},
                )
        assert "repeat_failure" not in caplog.text

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
        loop.tools.execute = AsyncMock(return_value={"exit_code": 0})
        loop.tools.get_tool_definitions = MagicMock(
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

    @pytest.mark.asyncio
    async def test_build_tool_history_context_with_data(self):
        """Returns formatted section when tool history exists."""
        loop = _make_loop(real_memory=True)
        await loop.memory.store_tool_outcome("exec", {"cmd": "ls"}, "file.txt", success=True)
        await loop.memory.store_tool_outcome("exec", {"cmd": "bad"}, "error", success=False)
        ctx = loop._build_tool_history_context()
        assert "## Recent Tool History" in ctx
        assert "exec [OK]" in ctx
        assert "exec [FAILED]" in ctx

    @pytest.mark.asyncio
    async def test_tool_history_in_chat_system_prompt(self):
        """Chat system prompt includes tool history when present."""
        loop = _make_loop(real_memory=True)
        await loop.memory.store_tool_outcome("exec", {}, "ok", success=True)
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
    loop.tools.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "web_search"}}]
    )
    # Return a string (non-dict) with invisible characters — goes through str() path
    loop.tools.execute = AsyncMock(return_value="clean\u200Bvalue\u202Ehere")

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
    loop.tools.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "exec"}}]
    )
    # Return a string (non-dict) with invisible characters — goes through str() path
    loop.tools.execute = AsyncMock(return_value="out\x00put\uFEFF")

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
    loop.tools.get_tool_definitions = MagicMock(return_value=[])

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
    """Workspace learnings are pre-sanitized by the workspace cache.

    get_learnings_context() now pre-sanitizes via sanitize_for_prompt()
    before caching, so the loop receives clean content.
    """
    from src.shared.utils import sanitize_for_prompt
    loop = _make_loop()
    loop.workspace = MagicMock()
    loop.workspace.get_bootstrap_content = MagicMock(return_value="")
    raw = "lesson\u200B one\u202E important"
    loop.workspace.get_learnings_context = MagicMock(
        return_value=sanitize_for_prompt(raw)
    )
    prompt = loop._build_chat_system_prompt()
    assert "\u200B" not in prompt
    assert "\u202E" not in prompt
    assert "lesson one important" in prompt


def test_chat_prompt_includes_memory_search_instruction():
    """Chat system prompt instructs agent to search memory before answering."""
    loop = _make_loop()
    prompt = loop._build_chat_system_prompt()
    assert "memory_search" in prompt
    assert "Before answering" in prompt


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
    loop.tools.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "search"}}]
    )
    loop.tools.execute = AsyncMock(return_value={"result": "same"})

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
    loop.tools.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "search"}}]
    )
    execute_count = 0

    async def counting_execute(*args, **kwargs):
        nonlocal execute_count
        execute_count += 1
        return {"result": "same"}

    loop.tools.execute = counting_execute

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
    loop.tools.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "search"}}]
    )
    loop.tools.execute = AsyncMock(return_value={"result": "same"})

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
    loop.tools.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "exec"}}]
    )
    loop.tools.execute = AsyncMock(return_value={"error": "command not found"})

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
    loop.tools.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "exec"}}]
    )
    loop.tools.execute = AsyncMock(return_value={"error": "command not found"})

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
    loop.tools.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "search"}}]
    )
    loop.tools.execute = AsyncMock(return_value={"result": "same"})

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
    loop.tools.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "exec"}}]
    )
    loop.tools.execute = AsyncMock(return_value={"error": "command not found"})

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
    loop.tools.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "search"}}]
    )
    execute_count = 0

    async def counting_execute(*args, **kwargs):
        nonlocal execute_count
        execute_count += 1
        return {"result": "same"}

    loop.tools.execute = counting_execute

    assignment = TaskAssignment(
        workflow_id="wf1", step_id="s1", task_type="research", input_data={}
    )
    await loop.execute_task(assignment)
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
        loop.tools.get_tool_definitions = MagicMock(
            return_value=[{"type": "function", "function": {"name": "web_search"}}]
        )
        loop.tools.execute = AsyncMock(return_value={"results": ["r1"]})

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
        loop.tools.get_tool_definitions = MagicMock(return_value=[])

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
        loop.tools.get_tool_definitions = MagicMock(return_value=[])

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
        loop.tools.get_tool_definitions = MagicMock(
            return_value=[{"type": "function", "function": {"name": "exec"}}]
        )
        loop.tools.execute = AsyncMock(return_value={"exit_code": 0})

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
        # Pin the cap below the supplied response count so the loop hits the
        # iteration ceiling (not the mock running dry). The default is now high.
        loop.MAX_ITERATIONS = 20
        loop.tools.get_tool_definitions = MagicMock(
            return_value=[{"type": "function", "function": {"name": "search"}}]
        )
        loop.tools.execute = AsyncMock(return_value={"result": "ok"})

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


# ── Runtime context injection ────────────────────────────────────


class TestRuntimeContextInjection:
    """Verify ## Runtime Context is injected into system prompts."""

    _INTROSPECT_DATA = {
        "permissions": {
            "blackboard_read": ["context/*"],
            "blackboard_write": ["context/test_*"],
            "can_message": ["bob"],
            "can_publish": [],
            "can_subscribe": [],
            "allowed_apis": ["anthropic"],
            "allowed_credentials": ["brightdata_*"],
        },
        "budget": {
            "allowed": True,
            "daily_used": 0.50,
            "daily_limit": 5.00,
            "monthly_used": 3.00,
            "monthly_limit": 50.00,
        },
        "fleet": [
            {"id": "test_agent", "role": "research"},
            {"id": "bob", "role": "engineer"},
        ],
    }

    def test_task_mode_includes_runtime_context(self):
        """The ## Runtime Context block is emitted as a VOLATILE fragment
        (relocated below the cache breakpoint, onto _volatile_prompt_suffix)."""
        from src.shared.types import TaskAssignment

        loop = _make_loop()
        assignment = TaskAssignment(
            workflow_id="wf1", step_id="s1", task_type="test",
            input_data={"instruction": "do stuff"},
        )
        prompt = loop._build_system_prompt(assignment, introspect_data=self._INTROSPECT_DATA)
        # Runtime context lives below the cache breakpoint, not in the prompt.
        assert "## Runtime Context" not in prompt
        suffix = loop._volatile_prompt_suffix
        assert "## Runtime Context" in suffix
        assert "Budget: daily $0.50/$5.00" in suffix
        assert "allowed_credentials: brightdata_*" in suffix

    def test_chat_mode_includes_runtime_context(self):
        """The ## Runtime Context block is relocated onto _volatile_prompt_suffix."""
        loop = _make_loop()
        prompt = loop._build_chat_system_prompt(introspect_data=self._INTROSPECT_DATA)
        assert "## Runtime Context" not in prompt
        suffix = loop._volatile_prompt_suffix
        assert "## Runtime Context" in suffix
        assert "Budget: daily $0.50/$5.00" in suffix
        assert "allowed_credentials: brightdata_*" in suffix

    def test_chat_mode_excludes_fleet_from_runtime_when_fleet_ctx_present(self):
        """When fleet_roster is provided, fleet line is excluded from runtime context."""
        loop = _make_loop()
        roster = [{"name": "bob", "role": "engineer"}]
        prompt = loop._build_chat_system_prompt(
            fleet_roster=roster,
            introspect_data=self._INTROSPECT_DATA,
        )
        # Detailed fleet block should be present in the stable system prompt.
        assert "Your Team" in prompt
        # The relocated runtime context (now volatile) must NOT duplicate fleet.
        suffix = loop._volatile_prompt_suffix
        runtime_section = (
            suffix.split("## Runtime Context")[1]
            if "## Runtime Context" in suffix else ""
        )
        assert "Fleet:" not in runtime_section

    def test_task_mode_includes_fleet_in_runtime(self):
        """Task mode has no detailed fleet block, so fleet shows in the runtime
        context — which is now relocated onto _volatile_prompt_suffix."""
        from src.shared.types import TaskAssignment

        loop = _make_loop()
        assignment = TaskAssignment(
            workflow_id="wf1", step_id="s1", task_type="test",
            input_data={"instruction": "do stuff"},
        )
        loop._build_system_prompt(assignment, introspect_data=self._INTROSPECT_DATA)
        assert "Fleet: [test_agent, bob]" in loop._volatile_prompt_suffix

    def test_no_introspect_data_no_runtime_context(self):
        """Without introspect data, no runtime context block."""
        from src.shared.types import TaskAssignment

        loop = _make_loop()
        assignment = TaskAssignment(
            workflow_id="wf1", step_id="s1", task_type="test",
            input_data={"instruction": "do stuff"},
        )
        prompt = loop._build_system_prompt(assignment, introspect_data=None)
        assert "## Runtime Context" not in prompt


class TestFormatRuntimeContext:
    """Unit tests for _format_runtime_context static method."""

    def test_empty_data_returns_empty(self):
        result = AgentLoop._format_runtime_context({})
        assert result == ""

    def test_exclude_fleet(self):
        data = {
            "fleet": [{"id": "alice"}, {"id": "bob"}],
            "permissions": {"can_message": ["bob"]},
        }
        with_fleet = AgentLoop._format_runtime_context(data)
        assert "Fleet:" in with_fleet

        without_fleet = AgentLoop._format_runtime_context(data, exclude_fleet=True)
        assert "Fleet:" not in without_fleet
        # Permissions should still be present
        assert "can_message: bob" in without_fleet

    def test_budget_exceeded_flag(self):
        data = {
            "budget": {
                "allowed": False,
                "daily_used": 5.00,
                "daily_limit": 5.00,
                "monthly_used": 10.00,
                "monthly_limit": 50.00,
            },
        }
        result = AgentLoop._format_runtime_context(data)
        assert "[EXCEEDED]" in result

    def test_cron_formatting(self):
        data = {
            "cron": [
                {"schedule": "*/30 * * * *", "heartbeat": True},
                {"schedule": "0 9 * * *", "heartbeat": False},
            ],
        }
        result = AgentLoop._format_runtime_context(data)
        assert "*/30 * * * * (heartbeat)" in result
        assert "0 9 * * *" in result
        assert result.count("(heartbeat)") == 1

    def test_fleet_ids_sanitized(self):
        """Fleet IDs in runtime context are sanitized."""
        data = {
            "fleet": [{"id": "good"}, {"id": "evil\u200bagent"}],
        }
        result = AgentLoop._format_runtime_context(data)
        assert "\u200b" not in result


# === Standalone Agent Blackboard Isolation ===


class TestStandaloneBlackboardIsolation:
    """Standalone agents should not see blackboard tools or references."""

    def _make_standalone_loop(self):
        loop = _make_loop()
        loop.mesh_client.is_standalone = True
        # Re-compute excluded tools (normally set in __init__)
        loop._excluded_tools = _BLACKBOARD_TOOLS
        return loop

    def _make_project_loop(self):
        loop = _make_loop()
        loop.mesh_client.is_standalone = False
        loop._excluded_tools = None
        return loop

    def test_standalone_excludes_blackboard_tools(self):
        """Standalone agents exclude the blackboard tools (authoring tools may
        also be excluded by the default-off tool-authoring gate)."""
        mesh = MagicMock()
        mesh.is_standalone = True
        loop = AgentLoop(
            agent_id="test", role="helper",
            memory=MagicMock(get_tool_history=MagicMock(return_value=[])),
            tools=MagicMock(), llm=MagicMock(), mesh_client=mesh,
        )
        assert _BLACKBOARD_TOOLS <= loop._excluded_tools

    def test_project_agent_no_exclusion(self):
        """Project agents do NOT exclude blackboard tools (the authoring gate
        may still exclude create_tool/reload_tools — orthogonal to this test)."""
        mesh = MagicMock()
        mesh.is_standalone = False
        loop = AgentLoop(
            agent_id="test", role="helper",
            memory=MagicMock(get_tool_history=MagicMock(return_value=[])),
            tools=MagicMock(), llm=MagicMock(), mesh_client=mesh,
        )
        assert not (_BLACKBOARD_TOOLS & (loop._excluded_tools or frozenset()))

    def test_standalone_task_prompt_no_blackboard(self):
        """Standalone agent task prompt omits blackboard references."""
        loop = self._make_standalone_loop()
        assignment = TaskAssignment(
            workflow_id="wf1", step_id="s1", task_type="research", input_data={},
        )
        prompt = loop._build_system_prompt(assignment)
        assert "blackboard" not in prompt.lower()
        assert "notify_user to report results" in prompt

    def test_project_task_prompt_has_blackboard(self):
        """Project agent task prompt includes blackboard references."""
        loop = self._make_project_loop()
        assignment = TaskAssignment(
            workflow_id="wf1", step_id="s1", task_type="research", input_data={},
        )
        prompt = loop._build_system_prompt(assignment)
        assert "blackboard" in prompt.lower()
        assert "promote" in prompt.lower()

    def test_standalone_chat_prompt_no_blackboard(self):
        """Standalone agent chat prompt omits blackboard references."""
        loop = self._make_standalone_loop()
        prompt = loop._build_chat_system_prompt()
        assert "blackboard" not in prompt.lower()
        assert "notify_user to report results" in prompt

    def test_project_chat_prompt_has_blackboard(self):
        """Project agent chat prompt includes blackboard references."""
        loop = self._make_project_loop()
        prompt = loop._build_chat_system_prompt()
        assert "blackboard" in prompt.lower()

    def test_standalone_task_prompt_no_promote(self):
        """Standalone task prompt uses simple JSON format without 'promote'."""
        loop = self._make_standalone_loop()
        assignment = TaskAssignment(
            workflow_id="wf1", step_id="s1", task_type="research", input_data={},
        )
        prompt = loop._build_system_prompt(assignment)
        assert "promote" not in prompt.lower()

    def test_standalone_get_status_excludes_blackboard(self):
        """get_status capabilities list excludes blackboard tools for standalone."""
        loop = self._make_standalone_loop()
        # Use a real ToolRegistry-like mock that respects exclude
        loop.tools.list_tools = MagicMock(
            side_effect=lambda exclude=None: (
                [n for n in ["memory_save", "read_blackboard", "notify_user"]
                 if not exclude or n not in exclude]
            ),
        )
        status = loop.get_status()
        assert "read_blackboard" not in status.capabilities
        assert "notify_user" in status.capabilities
        assert "memory_save" in status.capabilities


# ---------------------------------------------------------------------------
# Multimodal steer-append regression
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_steer_appended_correctly_to_multimodal_content():
    """Steer suffix must append as a text block, not corrupt a list via +=.

    Regression test for: when _prepare_chat_turn enriches a user message into
    a list of content blocks (e.g. image attachment) and there are queued steer
    messages, the steer text must be added as a new {"type": "text"} block
    rather than extending the list with individual characters (which would
    happen if we naively did list += string).
    """
    loop = _make_loop(
        [LLMResponse(content="Sure, I can see the image.", tokens_used=50)],
    )
    # Inject a steer message as if the user typed it mid-turn
    loop._steer_queue.put_nowait(("Please focus on the chart legend.", False))

    # Inject a multimodal content list directly (simulates enrichment result)
    multimodal_content = [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        {"type": "text", "text": "What is in this image?"},
    ]
    loop._chat_messages.append({"role": "user", "content": multimodal_content})

    # Now drain the steer — the fix path runs here
    steered = loop._drain_steer_messages()
    assert steered  # steer was present

    combined = "\n\n".join(text for text, _ in steered)
    steer_suffix = f"\n\n[Additional context]: {combined}"
    current = loop._chat_messages[-1]["content"]
    if isinstance(current, list):
        loop._chat_messages[-1]["content"].append(
            {"type": "text", "text": steer_suffix.strip()}
        )
    else:
        loop._chat_messages[-1]["content"] += steer_suffix

    result_content = loop._chat_messages[-1]["content"]

    # Content must remain a proper list of dicts, NOT a list polluted with
    # individual string characters from an erroneous list += string.
    assert isinstance(result_content, list)
    assert all(isinstance(b, dict) for b in result_content), (
        "Content list was corrupted — contains non-dict elements "
        "(likely caused by list += string)"
    )
    # The steer text should be present as a text block
    text_blocks = [b["text"] for b in result_content if b.get("type") == "text"]
    assert any("focus on the chart legend" in t for t in text_blocks)
    # The original image block must be untouched
    image_blocks = [b for b in result_content if b.get("type") == "image_url"]
    assert len(image_blocks) == 1


# ── _run_tool multimodal image support ────────────────────────


@pytest.mark.asyncio
async def test_run_tool_pops_image_and_returns_multimodal_content():
    """When a tool result contains _image, _run_tool returns multimodal content."""
    loop = _make_loop()
    loop.tools.execute = AsyncMock(return_value={
        "status": "screenshot captured",
        "_image": {"data": "iVBORw0KGgo=", "media_type": "image/png"},
    })

    content, result_dict = await loop._run_tool(
        ToolCallInfo(name="browser_screenshot", arguments={})
    )

    # _image must be popped from the result dict
    assert "_image" not in result_dict
    assert result_dict == {"status": "screenshot captured"}

    # content must be a list with text + image_url blocks
    assert isinstance(content, list)
    assert len(content) == 2
    assert content[0]["type"] == "text"
    assert "screenshot captured" in content[0]["text"]
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[1]["image_url"]["url"].endswith("iVBORw0KGgo=")


@pytest.mark.asyncio
async def test_run_tool_no_image_returns_plain_string():
    """Normal tool results (no _image) return a plain string."""
    loop = _make_loop()
    loop.tools.execute = AsyncMock(return_value={"result": "ok"})

    content, result_dict = await loop._run_tool(
        ToolCallInfo(name="web_search", arguments={"q": "test"})
    )

    assert isinstance(content, str)
    assert "ok" in content


@pytest.mark.asyncio
async def test_run_tool_image_not_in_serialized_result():
    """The serialized result_str must not contain the base64 image data."""
    loop = _make_loop()
    image_b64 = "A" * 1000  # large payload
    loop.tools.execute = AsyncMock(return_value={
        "status": "screenshot captured",
        "_image": {"data": image_b64, "media_type": "image/png"},
    })

    content, result_dict = await loop._run_tool(
        ToolCallInfo(name="browser_screenshot", arguments={})
    )

    # The text portion of the content must not contain the image data
    text_part = content[0]["text"] if isinstance(content, list) else content
    assert image_b64 not in text_part
    # _image was popped
    assert "_image" not in result_dict


@pytest.mark.asyncio
async def test_task_mode_appends_multimodal_content():
    """Task mode correctly passes multimodal content into message history."""
    captured_messages = []

    tool_call_response = LLMResponse(
        content="",
        tool_calls=[ToolCallInfo(name="browser_screenshot", arguments={})],
        tokens_used=50,
    )
    final_response = LLMResponse(content='{"result": {"seen": true}}', tokens_used=30)

    loop = _make_loop([tool_call_response, final_response])
    loop.tools.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "browser_screenshot"}}]
    )
    loop.tools.execute = AsyncMock(return_value={
        "status": "screenshot captured",
        "_image": {"data": "iVBORw0KGgo=", "media_type": "image/png"},
    })

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
    # The tool result message in the second LLM call should have multimodal content
    tool_msg = captured_messages[1][2]  # user, assistant, tool
    assert tool_msg["role"] == "tool"
    assert isinstance(tool_msg["content"], list)
    assert tool_msg["content"][0]["type"] == "text"
    assert tool_msg["content"][1]["type"] == "image_url"


@pytest.mark.asyncio
async def test_trim_context_handles_multimodal_tool_content():
    """_trim_context extracts text from multimodal content for summaries."""
    loop = _make_loop()

    multimodal_content = [
        {"type": "text", "text": '{"status": "screenshot captured"}'},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
    ]

    messages = [
        {"role": "user", "content": "take a screenshot"},
        {
            "role": "assistant", "content": "",
            "tool_calls": [{"id": "c1", "type": "function",
                           "function": {"name": "browser_screenshot", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": multimodal_content},
        {
            "role": "assistant", "content": "",
            "tool_calls": [{"id": "c2", "type": "function",
                           "function": {"name": "web_search", "arguments": '{"q":"x"}'}}],
        },
        {"role": "tool", "tool_call_id": "c2", "content": "search results here"},
        {
            "role": "assistant", "content": "",
            "tool_calls": [{"id": "c3", "type": "function",
                           "function": {"name": "web_search", "arguments": '{"q":"y"}'}}],
        },
        {"role": "tool", "tool_call_id": "c3", "content": "more results"},
    ]

    # Force trimming with a very low max_tokens
    trimmed = loop._trim_context(messages, max_tokens=1)

    # Summary is merged into the first user message (no consecutive user messages)
    first_msg = trimmed[0]
    assert first_msg["role"] == "user"
    assert "Previous Actions" in first_msg["content"]
    assert "screenshot captured" in first_msg["content"]


# ── Heartbeat mode ─────────────────────────────────────────────


def test_heartbeat_max_iterations_constant():
    """PR-V — bumped from 10 to 12 for defensive headroom.

    Worst-case operator heartbeat (cap-3 drill-ins): 1 status + 1 roster +
    3 drill-ins + 1 stale fanout + 1 notify_user = 7 tool calls plus the
    final assistant turn — well inside the 12-iter ceiling, with budget
    for future playbook additions.
    """
    assert HEARTBEAT_MAX_ITERATIONS == 12


@pytest.mark.asyncio
async def test_heartbeat_simple_completion():
    """Heartbeat returns structured result when LLM gives final answer."""
    loop = _make_loop()
    loop.llm.chat = AsyncMock(return_value=LLMResponse(content="HEARTBEAT_OK", tokens_used=50))
    loop.mesh_client.introspect = AsyncMock(return_value={})

    result = await loop.execute_heartbeat("Check stuff")

    assert result["skipped"] is False
    assert result["outcome"] == "ok"
    assert result["response"] == "HEARTBEAT_OK"
    assert result["tokens_used"] == 50
    assert result["duration_ms"] >= 0
    assert result["tools_used"] == []
    assert loop.state == "idle"


@pytest.mark.asyncio
async def test_heartbeat_injects_tool_history():
    """The heartbeat system prompt carries Recent Tool History — the
    evidence the Self-Evolution nudge tells the agent to act on."""
    loop = _make_loop()
    loop.llm.chat = AsyncMock(return_value=LLMResponse(content="HEARTBEAT_OK", tokens_used=50))
    loop.mesh_client.introspect = AsyncMock(return_value={})
    loop.memory.get_tool_history = MagicMock(return_value=[
        {"tool_name": "exec", "params_hash": "h1", "outcome": "file.txt",
         "success": True, "created_at": "2026-06-11 09:00:00"},
        {"tool_name": "exec", "params_hash": "h2", "outcome": "command not found",
         "success": False, "created_at": "2026-06-11 09:01:00"},
    ])

    result = await loop.execute_heartbeat("Check stuff")

    assert result["skipped"] is False
    system_prompt = loop.llm.chat.call_args.kwargs["system"]
    assert "## Recent Tool History" in system_prompt
    assert "exec [OK]" in system_prompt
    assert "exec [FAILED]" in system_prompt


@pytest.mark.asyncio
async def test_heartbeat_omits_tool_history_when_empty():
    """No tool outcomes recorded → no Recent Tool History section."""
    loop = _make_loop()  # fixture stubs get_tool_history to []
    loop.llm.chat = AsyncMock(return_value=LLMResponse(content="HEARTBEAT_OK", tokens_used=50))
    loop.mesh_client.introspect = AsyncMock(return_value={})

    result = await loop.execute_heartbeat("Check stuff")

    assert result["skipped"] is False
    system_prompt = loop.llm.chat.call_args.kwargs["system"]
    assert "## Recent Tool History" not in system_prompt


@pytest.mark.asyncio
async def test_heartbeat_skips_when_busy():
    """Heartbeat returns skipped when agent state is not idle."""
    loop = _make_loop()
    loop.state = "working"  # simulate task in progress

    result = await loop.execute_heartbeat("Check stuff")

    assert result["skipped"] is True
    assert result["reason"] == "agent_busy"
    # State should not have been modified
    assert loop.state == "working"


@pytest.mark.asyncio
async def test_heartbeat_skips_when_chat_locked():
    """Heartbeat returns skipped when chat lock is held."""
    loop = _make_loop()
    await loop._chat_lock.acquire()
    try:
        result = await loop.execute_heartbeat("Check stuff")
        assert result["skipped"] is True
        assert result["reason"] == "agent_busy"
    finally:
        loop._chat_lock.release()


@pytest.mark.asyncio
async def test_run_maintenance_skips_when_busy_else_runs_under_lock():
    """The background maintenance pass must skip while a turn is in flight
    (same idle/lock guard the heartbeat uses) and otherwise run the
    ContextManager pass under the chat lock."""
    loop = _make_loop()
    loop.context_manager = MagicMock()
    loop.context_manager.run_maintenance = AsyncMock()

    # Busy with a task → skip.
    loop.state = "working"
    await loop.run_maintenance()
    loop.context_manager.run_maintenance.assert_not_awaited()

    # Idle but a chat turn holds the lock → skip.
    loop.state = "idle"
    await loop._chat_lock.acquire()
    try:
        await loop.run_maintenance()
    finally:
        loop._chat_lock.release()
    loop.context_manager.run_maintenance.assert_not_awaited()

    # Idle and unlocked → runs, then restores idle state (so it doesn't
    # leave the agent stuck "working").
    await loop.run_maintenance()
    loop.context_manager.run_maintenance.assert_awaited_once()
    assert loop.state == "idle"


@pytest.mark.asyncio
async def test_run_maintenance_loop_invokes_then_cancels(monkeypatch):
    """The production launch path (server.run_maintenance_loop) ticks
    loop.run_maintenance and stops cleanly on cancellation."""
    from src.agent import server

    loop = _make_loop()
    loop.run_maintenance = AsyncMock()
    monkeypatch.setattr(server, "_MAINTENANCE_INITIAL_DELAY_S", 0)
    monkeypatch.setattr(server, "_MAINTENANCE_TICK_S", 0.01)

    task = asyncio.create_task(server.run_maintenance_loop(loop))
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert loop.run_maintenance.await_count >= 1


@pytest.mark.asyncio
async def test_heartbeat_skips_when_no_rules():
    """Heartbeat skips LLM call when HEARTBEAT.md is empty and no goals set."""
    loop = _make_loop()
    loop.workspace = MagicMock()
    loop.workspace.load_heartbeat_rules = MagicMock(return_value="# Heartbeat Rules\n")

    result = await loop.execute_heartbeat("Check stuff")
    assert result["skipped"] is True
    assert result["reason"] == "no_heartbeat_rules"
    loop.llm.chat.assert_not_called()


@pytest.mark.asyncio
async def test_heartbeat_dispatch_handles_missing_file():
    """Lazy-create contract: missing HEARTBEAT.md must dispatch like empty.

    After the lazy-bootstrap refactor, ``load_heartbeat_rules()`` returns
    ``""`` (not a "# Heartbeat Rules\n" stub) when no file exists on disk.
    ``_is_heartbeat_empty("")`` must still treat that as "no rules",
    fire the ``no_heartbeat_rules`` skip without crashing, and never
    reach the LLM. This pins the boundary so a future reader that grows
    a strict-existence check can't silently break agents created after
    the lazy-bootstrap change.
    """
    loop = _make_loop()
    loop.workspace = MagicMock()
    # Simulate the missing-file state — load_heartbeat_rules returns "".
    loop.workspace.load_heartbeat_rules = MagicMock(return_value="")
    loop._fetch_goals = AsyncMock(return_value=None)

    result = await loop.execute_heartbeat("Check stuff")
    assert result["skipped"] is True
    assert result["reason"] == "no_heartbeat_rules"
    loop.llm.chat.assert_not_called()


@pytest.mark.asyncio
async def test_heartbeat_force_llm_bypasses_no_heartbeat_rules_skip():
    """Bug 6 (codex P2 r2): force_llm=True must reach the LLM even with empty rules.

    Pipeline-kicker agents intentionally have no HEARTBEAT.md content and
    no goals (their job IS to think on a schedule and decide what to
    do). Without force_llm propagation the agent-side
    ``no_heartbeat_rules`` skip would silence them. Pin the bypass.
    """
    loop = _make_loop()
    loop.llm.chat = AsyncMock(return_value=LLMResponse(content="kicked", tokens_used=10))
    loop.mesh_client.introspect = AsyncMock(return_value={})
    loop.mesh_client.read_blackboard = AsyncMock(return_value=None)
    loop.workspace = MagicMock()
    loop.workspace.get_bootstrap_content = MagicMock(return_value="")
    loop.workspace.get_learnings_context = MagicMock(return_value="")
    loop.workspace.load_heartbeat_rules = MagicMock(return_value="# Heartbeat Rules\n")
    loop.workspace.append_daily_log = MagicMock()
    loop.workspace.append_activity = MagicMock()

    result = await loop.execute_heartbeat("Check stuff", force_llm=True)
    assert not result.get("skipped", False)
    loop.llm.chat.assert_called()


@pytest.mark.asyncio
async def test_heartbeat_force_llm_still_respects_agent_busy():
    """Bug 6 (codex P2 r2): force_llm bypasses no_heartbeat_rules but NOT busy.

    A busy agent has work in flight — running the heartbeat would
    contend with the active state machine. force_llm is about
    overriding the skip-LLM cost optimization, not about ignoring
    safety interlocks. Pin that busy still wins.
    """
    loop = _make_loop()
    # Mark the agent as already working.
    await loop._chat_lock.acquire()
    try:
        result = await loop.execute_heartbeat("Check stuff", force_llm=True)
        assert result["skipped"] is True
        assert result["reason"] == "agent_busy"
    finally:
        loop._chat_lock.release()


@pytest.mark.asyncio
async def test_heartbeat_runs_when_empty_rules_but_goals_exist():
    """Heartbeat still runs when HEARTBEAT.md is empty but goals are set."""
    loop = _make_loop()
    loop.llm.chat = AsyncMock(return_value=LLMResponse(content="Done", tokens_used=10))
    loop.mesh_client.introspect = AsyncMock(return_value={})
    loop.mesh_client.read_blackboard = AsyncMock(return_value={"goal": "Monitor competitors"})
    loop.workspace = MagicMock()
    loop.workspace.get_bootstrap_content = MagicMock(return_value="")
    loop.workspace.get_learnings_context = MagicMock(return_value="")
    loop.workspace.load_heartbeat_rules = MagicMock(return_value="# Heartbeat Rules\n")
    loop.workspace.append_daily_log = MagicMock()
    loop.workspace.append_activity = MagicMock()

    result = await loop.execute_heartbeat("check")
    assert not result.get("skipped", False)
    loop.llm.chat.assert_called()


@pytest.mark.asyncio
async def test_heartbeat_with_tool_calls():
    """Heartbeat executes tools and tracks them in result."""
    tool_call_response = LLMResponse(
        content="",
        tool_calls=[ToolCallInfo(name="notify_user", arguments={"message": "Alert!"})],
        tokens_used=80,
    )
    final_response = LLMResponse(content="Done", tokens_used=30)

    loop = _make_loop([tool_call_response, final_response])
    loop.tools.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "notify_user"}}]
    )
    loop.tools.execute = AsyncMock(return_value={"sent": True})
    loop.mesh_client.introspect = AsyncMock(return_value={})
    loop.workspace = MagicMock()
    loop.workspace.get_bootstrap_content = MagicMock(return_value="")
    loop.workspace.get_learnings_context = MagicMock(return_value="")
    loop.workspace.load_heartbeat_rules = MagicMock(return_value="- Check alerts every hour")
    loop.workspace.append_daily_log = MagicMock()
    loop.workspace.append_activity = MagicMock()

    result = await loop.execute_heartbeat("Check alerts")

    assert result["outcome"] == "ok"
    assert "notify_user" in result["tools_used"]
    assert result["tokens_used"] == 110
    assert loop.state == "idle"
    # Notifications should be passed to append_activity
    call_kwargs = loop.workspace.append_activity.call_args[1]
    assert call_kwargs["notifications"] == ["Alert!"]


@pytest.mark.asyncio
async def test_heartbeat_does_not_touch_chat():
    """Heartbeat doesn't modify _chat_messages or call append_chat_message."""
    loop = _make_loop()
    loop.llm.chat = AsyncMock(return_value=LLMResponse(content="ok", tokens_used=10))
    loop.mesh_client.introspect = AsyncMock(return_value={})
    loop.workspace = MagicMock()
    loop.workspace.get_bootstrap_content = MagicMock(return_value="")
    loop.workspace.get_learnings_context = MagicMock(return_value="")
    loop.workspace.load_heartbeat_rules = MagicMock(return_value="- Check status")
    loop.workspace.append_daily_log = MagicMock()
    loop.workspace.append_activity = MagicMock()

    loop._chat_messages.append({"role": "user", "content": "existing chat"})

    await loop.execute_heartbeat("heartbeat check")

    # Chat messages should be untouched
    assert len(loop._chat_messages) == 1
    assert loop._chat_messages[0]["content"] == "existing chat"
    # append_chat_message should NOT be called
    loop.workspace.append_chat_message.assert_not_called()
    # But activity log should be written
    loop.workspace.append_activity.assert_called_once()


@pytest.mark.asyncio
async def test_heartbeat_logs_to_activity():
    """Heartbeat writes structured entry to workspace activity log."""
    loop = _make_loop()
    loop.llm.chat = AsyncMock(return_value=LLMResponse(content="All clear", tokens_used=25))
    loop.mesh_client.introspect = AsyncMock(return_value={})
    loop.workspace = MagicMock()
    loop.workspace.get_bootstrap_content = MagicMock(return_value="")
    loop.workspace.get_learnings_context = MagicMock(return_value="")
    loop.workspace.load_heartbeat_rules = MagicMock(return_value="- Monitor targets")
    loop.workspace.append_daily_log = MagicMock()
    loop.workspace.append_activity = MagicMock()

    await loop.execute_heartbeat("check")

    loop.workspace.append_activity.assert_called_once()
    call_kwargs = loop.workspace.append_activity.call_args
    assert call_kwargs.kwargs["trigger"] == "heartbeat"  # or positional
    # Also verify daily log was written
    loop.workspace.append_daily_log.assert_called_once()


@pytest.mark.asyncio
async def test_heartbeat_max_iterations():
    """Heartbeat wraps up gracefully when approaching iteration limit.

    On the last iteration tools are withheld, so the LLM produces a text
    answer and the heartbeat finishes with outcome "ok" instead of being
    cut off with "max_iterations".
    """
    tool_defs = [{"type": "function", "function": {"name": "search"}}]

    # Simulate an LLM that calls tools when they are available but gives
    # a text answer when tools are withheld (last iteration).
    call_count = 0

    async def _smart_llm(*, system, messages, tools=None, **kw):
        nonlocal call_count
        call_count += 1
        if tools:
            return LLMResponse(
                content="",
                tool_calls=[ToolCallInfo(name="search", arguments={"q": f"query_{call_count}"})],
                tokens_used=10,
            )
        # No tools → forced text answer
        return LLMResponse(content="Wrapping up.", tool_calls=[], tokens_used=10)

    loop = _make_loop([])  # responses unused — overridden by side_effect
    loop.llm.chat = AsyncMock(side_effect=_smart_llm)
    loop.tools.get_tool_definitions = MagicMock(return_value=tool_defs)
    loop.tools.execute = AsyncMock(return_value={"result": "ok"})
    loop.mesh_client.introspect = AsyncMock(return_value={})

    result = await loop.execute_heartbeat("infinite loop")

    # The last iteration withholds tools → text answer → outcome "ok"
    assert result["outcome"] == "ok"
    assert result["response"] == "Wrapping up."
    assert loop.state == "idle"
    assert call_count == HEARTBEAT_MAX_ITERATIONS


@pytest.mark.asyncio
async def test_heartbeat_windup_nudge_messages():
    """Nudge messages are injected at _remaining==2 and _remaining==1."""
    tool_defs = [{"type": "function", "function": {"name": "search"}}]
    captured_messages: list[list[dict]] = []
    # Vary arguments per call so the tool-loop detector's terminate
    # threshold (9 identical params) doesn't trip before the iteration
    # cap — the cap is what this test exercises.
    call_idx = {"n": 0}

    async def _capture_llm(*, system, messages, tools=None, **kw):
        # Snapshot the message list the LLM receives on each call
        captured_messages.append([m.copy() for m in messages])
        if tools:
            call_idx["n"] += 1
            return LLMResponse(
                content="",
                tool_calls=[ToolCallInfo(name="search", arguments={"q": f"x{call_idx['n']}"})],
                tokens_used=10,
            )
        return LLMResponse(content="Done.", tool_calls=[], tokens_used=10)

    loop = _make_loop([])
    loop.llm.chat = AsyncMock(side_effect=_capture_llm)
    loop.tools.get_tool_definitions = MagicMock(return_value=tool_defs)
    loop.tools.execute = AsyncMock(return_value={"result": "ok"})
    loop.mesh_client.introspect = AsyncMock(return_value={})

    await loop.execute_heartbeat("go")

    assert len(captured_messages) == HEARTBEAT_MAX_ITERATIONS

    # The second-to-last call (index MAX-2) should include the wrap-up nudge
    penultimate_msgs = captured_messages[HEARTBEAT_MAX_ITERATIONS - 2]
    nudge_texts = [m["content"] for m in penultimate_msgs if m["role"] == "user"]
    assert any("2 iterations remaining" in t for t in nudge_texts)

    # The last call (index MAX-1) should include the final nudge
    last_msgs = captured_messages[HEARTBEAT_MAX_ITERATIONS - 1]
    nudge_texts = [m["content"] for m in last_msgs if m["role"] == "user"]
    assert any("LAST iteration" in t for t in nudge_texts)


@pytest.mark.asyncio
async def test_heartbeat_forced_wrapup_on_unexpected_tool_calls():
    """If the LLM returns tool_calls on the last iteration (tools withheld),
    they are ignored and the text content is used as the final answer."""
    tool_defs = [{"type": "function", "function": {"name": "search"}}]
    # Vary arguments per call so the tool-loop detector's terminate
    # threshold doesn't trip before the iteration cap.
    call_idx = {"n": 0}

    async def _stubborn_llm(*, system, messages, tools=None, **kw):
        # Always returns tool_calls, even when tools=None
        call_idx["n"] += 1
        return LLMResponse(
            content="Almost done",
            tool_calls=[ToolCallInfo(name="search", arguments={"q": f"x{call_idx['n']}"})],
            tokens_used=10,
        )

    loop = _make_loop([])
    loop.llm.chat = AsyncMock(side_effect=_stubborn_llm)
    loop.tools.get_tool_definitions = MagicMock(return_value=tool_defs)
    loop.tools.execute = AsyncMock(return_value={"result": "ok"})
    loop.mesh_client.introspect = AsyncMock(return_value={})

    result = await loop.execute_heartbeat("go")

    # Should wrap up with "ok" — NOT "max_iterations"
    assert result["outcome"] == "ok"
    assert result["response"] == "Almost done"
    assert loop.llm.chat.call_count == HEARTBEAT_MAX_ITERATIONS


@pytest.mark.asyncio
async def test_heartbeat_forced_wrapup_empty_content():
    """When forced wrap-up discards tool_calls and content is empty,
    a fallback message is used."""
    tool_defs = [{"type": "function", "function": {"name": "search"}}]
    # Vary arguments per call so the tool-loop detector's terminate
    # threshold doesn't trip before the iteration cap.
    call_idx = {"n": 0}

    async def _empty_llm(*, system, messages, tools=None, **kw):
        call_idx["n"] += 1
        return LLMResponse(
            content="",
            tool_calls=[ToolCallInfo(name="search", arguments={"q": f"x{call_idx['n']}"})],
            tokens_used=10,
        )

    loop = _make_loop([])
    loop.llm.chat = AsyncMock(side_effect=_empty_llm)
    loop.tools.get_tool_definitions = MagicMock(return_value=tool_defs)
    loop.tools.execute = AsyncMock(return_value={"result": "ok"})
    loop.mesh_client.introspect = AsyncMock(return_value={})

    result = await loop.execute_heartbeat("go")

    assert result["outcome"] == "ok"
    # Fallback content used when LLM content was empty
    assert result["response"] == "Heartbeat complete."


@pytest.mark.asyncio
async def test_heartbeat_sets_contextvar():
    """_heartbeat_mode ContextVar is True during heartbeat and reset after."""
    captured = []

    async def spy_execute(name, args, **kwargs):
        captured.append(_heartbeat_mode.get(False))
        return {"sent": True}

    tool_call_response = LLMResponse(
        content="",
        tool_calls=[ToolCallInfo(name="notify_user", arguments={"message": "hi"})],
        tokens_used=10,
    )
    final_response = LLMResponse(content="done", tokens_used=10)

    loop = _make_loop([tool_call_response, final_response])
    loop.tools.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "notify_user"}}]
    )
    loop.tools.execute = spy_execute
    loop.mesh_client.introspect = AsyncMock(return_value={})

    await loop.execute_heartbeat("check")

    # During tool execution, _heartbeat_mode should have been True
    assert captured == [True]
    # After heartbeat, should be reset to False
    assert _heartbeat_mode.get(False) is False


@pytest.mark.asyncio
async def test_heartbeat_error_handling():
    """Heartbeat catches exceptions and returns error outcome."""
    loop = _make_loop()
    loop.llm.chat = AsyncMock(side_effect=RuntimeError("Budget exceeded"))
    loop.mesh_client.introspect = AsyncMock(return_value={})

    result = await loop.execute_heartbeat("check")

    assert result["outcome"] == "error"
    assert "Budget exceeded" in result["response"]
    assert loop.state == "idle"
    # ContextVar should be reset even on error
    assert _heartbeat_mode.get(False) is False


@pytest.mark.asyncio
async def test_heartbeat_includes_goals_in_system_prompt():
    """Heartbeat system prompt includes goals when they exist on the blackboard."""
    captured_system = []

    async def _capture_llm(*, system, messages, **kw):
        captured_system.append(system)
        return LLMResponse(content="Done", tokens_used=10)

    loop = _make_loop([])
    loop.llm.chat = AsyncMock(side_effect=_capture_llm)
    loop.mesh_client.introspect = AsyncMock(return_value={})
    loop.mesh_client.read_blackboard = AsyncMock(
        return_value={"value": {"primary": "Monitor sales pipeline"}}
    )

    await loop.execute_heartbeat("wakeup")

    assert len(captured_system) == 1
    assert "Your Current Goals" in captured_system[0]
    assert "Monitor sales pipeline" in captured_system[0]


@pytest.mark.asyncio
async def test_heartbeat_includes_fleet_roster():
    """Heartbeat system prompt includes fleet roster for multi-agent setups."""
    captured_system = []

    async def _capture_llm(*, system, messages, **kw):
        captured_system.append(system)
        return LLMResponse(content="Done", tokens_used=10)

    loop = _make_loop([])
    loop.llm.chat = AsyncMock(side_effect=_capture_llm)
    loop.mesh_client.is_standalone = False
    loop.mesh_client.introspect = AsyncMock(return_value={})
    loop.mesh_client.list_agents = AsyncMock(return_value={
        "writer": {"role": "content writer"},
        "test_agent": {"role": "research"},  # self — should be excluded
    })

    await loop.execute_heartbeat("wakeup")

    assert len(captured_system) == 1
    assert "Your Team" in captured_system[0]
    assert "writer" in captured_system[0]
    assert "content writer" in captured_system[0]


@pytest.mark.asyncio
async def test_heartbeat_skips_fleet_for_standalone():
    """Heartbeat does not call list_agents for standalone agents."""
    loop = _make_loop([])
    loop.llm.chat = AsyncMock(return_value=LLMResponse(content="ok", tokens_used=10))
    loop.mesh_client.is_standalone = True
    loop.mesh_client.introspect = AsyncMock(return_value={})

    await loop.execute_heartbeat("wakeup")

    loop.mesh_client.list_agents.assert_not_called()


@pytest.mark.asyncio
async def test_heartbeat_includes_learnings():
    """Heartbeat system prompt includes learnings from workspace."""
    captured_system = []

    async def _capture_llm(*, system, messages, **kw):
        captured_system.append(system)
        return LLMResponse(content="Done", tokens_used=10)

    loop = _make_loop([])
    loop.llm.chat = AsyncMock(side_effect=_capture_llm)
    loop.mesh_client.introspect = AsyncMock(return_value={})
    loop.workspace = MagicMock()
    loop.workspace.get_bootstrap_content = MagicMock(return_value="")
    loop.workspace.get_learnings_context = MagicMock(
        return_value="- Always verify API response status before parsing"
    )
    loop.workspace.load_heartbeat_rules = MagicMock(return_value="- Check alerts")
    loop.workspace.append_daily_log = MagicMock()
    loop.workspace.append_activity = MagicMock()

    await loop.execute_heartbeat("wakeup")

    assert len(captured_system) == 1
    assert "Learnings from Past Sessions" in captured_system[0]
    assert "Always verify API response status" in captured_system[0]
    # Heartbeat uses half the chat-mode cap (3000 → 1500)
    loop.workspace.get_learnings_context.assert_called_once_with(max_chars=1500)


@pytest.mark.asyncio
async def test_heartbeat_includes_self_evolution():
    """Heartbeat system prompt includes self-evolution nudge."""
    captured_system = []

    async def _capture_llm(*, system, messages, **kw):
        captured_system.append(system)
        return LLMResponse(content="Done", tokens_used=10)

    loop = _make_loop([])
    loop.llm.chat = AsyncMock(side_effect=_capture_llm)
    loop.mesh_client.introspect = AsyncMock(return_value={})

    await loop.execute_heartbeat("wakeup")

    assert len(captured_system) == 1
    assert "Self-Evolution" in captured_system[0]
    assert "INSTRUCTIONS.md" in captured_system[0]


@pytest.mark.asyncio
async def test_heartbeat_drains_steer_queue():
    """Heartbeat drains pending steer messages into the user message."""
    captured_messages = []

    async def _capture_llm(*, system, messages, **kw):
        captured_messages.append(messages)
        return LLMResponse(content="HEARTBEAT_OK", tokens_used=10)

    loop = _make_loop([])
    loop.llm.chat = AsyncMock(side_effect=_capture_llm)
    loop.mesh_client.introspect = AsyncMock(return_value={})

    # Inject a steer message before heartbeat runs
    await loop.inject_steer("coordination signal from writer")

    result = await loop.execute_heartbeat("heartbeat check")

    assert not result.get("skipped", False)
    assert len(captured_messages) >= 1
    user_content = captured_messages[0][0]["content"]
    assert "Pending Coordination Signals" in user_content
    assert "coordination signal from writer" in user_content


@pytest.mark.asyncio
async def test_heartbeat_check_inbox_in_system_prompt():
    """Heartbeat system prompt tells agent to call check_inbox()."""
    captured_system = []

    async def _capture_llm(*, system, messages, **kw):
        captured_system.append(system)
        return LLMResponse(content="HEARTBEAT_OK", tokens_used=10)

    loop = _make_loop([])
    loop.llm.chat = AsyncMock(side_effect=_capture_llm)
    loop.mesh_client.introspect = AsyncMock(return_value={})

    await loop.execute_heartbeat("wakeup")

    assert len(captured_system) == 1
    assert "check_inbox()" in captured_system[0]
    assert "goals, or inbox needs attention" in captured_system[0]


@pytest.mark.asyncio
async def test_heartbeat_standalone_no_check_inbox():
    """Standalone agents don't get check_inbox() in heartbeat prompt."""
    captured_system = []

    async def _capture_llm(*, system, messages, **kw):
        captured_system.append(system)
        return LLMResponse(content="HEARTBEAT_OK", tokens_used=10)

    loop = _make_loop([])
    loop.mesh_client.is_standalone = True
    loop.llm.chat = AsyncMock(side_effect=_capture_llm)
    loop.mesh_client.introspect = AsyncMock(return_value={})

    await loop.execute_heartbeat("wakeup")

    assert len(captured_system) == 1
    assert "check_inbox()" not in captured_system[0]
    assert "goals needs attention" in captured_system[0]


@pytest.mark.asyncio
async def test_tools_reload_rebuilds_system_prompt():
    """After reload_tools, the system prompt is rebuilt with new tool descriptions."""
    systems_seen = []
    call_count = 0

    async def _tracking_llm(*, system, messages, tools=None, **kw):
        nonlocal call_count
        call_count += 1
        systems_seen.append(system)
        if call_count == 1:
            # First call: LLM calls reload_tools
            from src.shared.types import ToolCallInfo
            return LLMResponse(
                content="",
                tool_calls=[ToolCallInfo(name="reload_tools", arguments={})],
                tokens_used=10,
            )
        # Second call: LLM gives final answer
        return LLMResponse(content="done", tokens_used=10)

    loop = _make_loop([])
    loop.llm.chat = AsyncMock(side_effect=_tracking_llm)

    # Inject a mock tool to appear after reload
    original_reload = loop.tools.reload

    def _mock_reload():
        count = original_reload()
        # After reload, descriptions cache is cleared.
        # The next get_descriptions() call will rebuild.
        return count

    loop.tools.reload = _mock_reload

    await loop.chat("test reload")

    # System prompt should have been rebuilt after reload
    assert len(systems_seen) >= 2
    # The flag should be consumed (not stuck True)
    assert loop._tools_reloaded is False


# ── Seam follow-up Fix 3: loop catches LLMAuthError / LLMConfigError ──


class TestLoopLLMErrorHandling:
    """execute_task / chat / chat_stream must catch LLMAuthError and
    LLMConfigError explicitly — auth errors self-report to mesh for
    quarantine; config errors fail the task with actionable error string
    but don't quarantine (operator misconfig, not broken credential).
    """

    @pytest.mark.asyncio
    async def test_execute_task_catches_llm_auth_error(self):
        """Loop catches LLMAuthError and fails the task.

        Codex P1 follow-up: the loop does NOT self-report — the mesh
        already recorded the failure inside execute_api_call before
        tagging the response, and self-reporting would double-count
        against the quarantine threshold.
        """
        from src.shared.errors import LLMAuthError
        loop = _make_loop()
        loop.llm.chat = AsyncMock(
            side_effect=LLMAuthError(
                "auth blown", provider="openai", model="openai/gpt-5",
                http_status=401,
            ),
        )
        # report_auth_failure must NOT be called from the loop.
        loop.mesh_client.report_auth_failure = AsyncMock(
            return_value={"recorded": True, "quarantined": False},
        )
        assignment = TaskAssignment(
            workflow_id="w", step_id="s", task_type="x", input_data={"q": "1"},
        )
        result = await loop.execute_task(assignment)
        assert result.status == "failed"
        assert "auth_failure" in (result.error or "")
        loop.mesh_client.report_auth_failure.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_execute_task_catches_llm_config_error(self):
        from src.shared.errors import LLMConfigError
        loop = _make_loop()
        loop.llm.chat = AsyncMock(
            side_effect=LLMConfigError(
                "model not supported", provider="openai",
                model="openai/gpt-99", http_status=400,
            ),
        )
        # Even though report_auth_failure exists, config errors must NOT
        # self-report (this is operator misconfig, not a broken credential).
        loop.mesh_client.report_auth_failure = AsyncMock()
        assignment = TaskAssignment(
            workflow_id="w", step_id="s", task_type="x", input_data={"q": "1"},
        )
        result = await loop.execute_task(assignment)
        assert result.status == "failed"
        assert "config_error" in (result.error or "")
        loop.mesh_client.report_auth_failure.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_mesh_client_report_auth_failure_swallows_errors(self):
        """The mesh_client.report_auth_failure helper exists for
        future/non-loop call sites (legacy clients, external scripts).
        It must never raise out — fire-and-forget semantics."""
        from unittest.mock import patch

        from src.agent.mesh_client import MeshClient

        mc = MeshClient(mesh_url="http://localhost:8420", agent_id="a")

        # Stub the underlying client.post to raise.
        with patch.object(
            mc, "_get_client", AsyncMock(side_effect=RuntimeError("mesh down")),
        ):
            result = await mc.report_auth_failure(
                provider="openai", model="x", http_status=401,
            )
        assert result.get("recorded") is False
        assert "error" in result


# === Chat empty-response fallback (Bug 3) ===


class TestChatEmptyResponseFallback:
    """When the LLM produces zero final text, the chat return value gets a
    synthetic notice so the dashboard renders the turn (instead of a blank
    panel). Bug 3: this now fires REGARDLESS of task_id / tool_limit and
    even with zero tool_outputs (generic marker in that case). Deliberate
    ``__SILENT__`` replies are still preserved (``silent_reply`` set
    upstream) and are NOT papered over."""

    @pytest.mark.asyncio
    async def test_chat_empty_response_with_tools_gets_fallback(self):
        loop = _make_loop()

        async def fake_inner(_msg, **_kw):
            return {
                "response": "",
                "tool_outputs": [{"name": "noop"}, {"name": "noop2"}],
                "tokens_used": 10,
            }
        loop._chat_inner = fake_inner

        result = await loop.chat("hi")

        assert result["tool_outputs"]
        assert result["response"].strip(), \
            "fallback must produce non-empty response"
        assert "2 tool calls" in result["response"]

    @pytest.mark.asyncio
    async def test_chat_empty_response_no_tools_gets_generic_marker(self):
        """Bug 3 (updated contract): a no-tool empty turn that is NOT a
        deliberate ``__SILENT__`` reply must NOT surface a blank bubble.
        The ``chat()`` final net substitutes the generic (no-tools)
        marker and flags ``silent_reply``. Deliberate silence is handled
        upstream in ``_chat_inner`` (it sets ``silent_reply`` so this net
        leaves it alone — covered by the silent-token tests)."""
        loop = _make_loop()

        async def fake_inner(_msg, **_kw):
            return {"response": "", "tool_outputs": [], "tokens_used": 0}
        loop._chat_inner = fake_inner

        result = await loop.chat("hi")
        assert result["response"].startswith("(No response")
        assert result.get("silent_reply") is True

    @pytest.mark.asyncio
    async def test_chat_with_task_id_still_rescued(self):
        """Bug 3 (updated contract): the empty-response rescue is NO
        LONGER gated on ``not task_id``. A handoff (task_id) turn that
        ends empty with tool_outputs must also be rescued so the user
        never sees a blank reply. ``_chat_inner`` normally substitutes
        the marker; here we fake an empty inner result to prove the
        ``chat()`` final net rescues it even in task_id mode."""
        loop = _make_loop()

        async def fake_inner(_msg, **_kw):
            return {
                "response": "",
                "tool_outputs": [{"name": "x"}],
                "tokens_used": 5,
            }
        loop._chat_inner = fake_inner

        # Mock the auto-close pipeline so the test doesn't depend on
        # mesh round-trips.
        loop._auto_close_task = AsyncMock(return_value=None)

        result = await loop.chat("hi", task_id="task_X")

        # The rescue fires regardless of task_id — response is non-empty.
        assert result["response"].strip() != ""
        assert "Completed" in result["response"]
        assert result.get("silent_reply") is True

    # --- Transcript-persistence regression tests (chat refresh bug) ---
    #
    # Before the fix, when the LLM ran tools and ended with empty final
    # text, ``_log_chat_turn`` skipped persisting the assistant entry to
    # ``chat_transcript.jsonl``. The dashboard's live ``done`` event
    # showed the synthetic notice fine, but the transcript loaded on
    # page refresh had no assistant row — "the message disappeared".
    # These tests pin the contract that the synthetic fallback now lands
    # in the transcript (and on the daily log) when there were tools,
    # and that truly silent turns (no text, no tools) still skip.

    def test_log_chat_turn_persists_fallback_when_empty_with_tools(
        self, tmp_path,
    ):
        from src.agent.workspace import WorkspaceManager

        loop = _make_loop()
        loop.workspace = WorkspaceManager(workspace_dir=str(tmp_path))

        loop._log_chat_turn(
            "user q",
            "",
            tool_outputs=[
                {"tool": "notify_user", "input": {}, "output": "ok"},
            ],
        )

        transcript = loop.workspace.load_chat_transcript()
        assistant_entries = [
            m for m in transcript if m.get("role") == "assistant"
        ]
        assert len(assistant_entries) == 1, \
            "fallback must persist exactly one assistant entry"
        content = assistant_entries[0]["content"]
        assert "Completed 1 tool call" in content
        assert content == AgentLoop._synthesize_empty_chat_fallback(
            [{"tool": "notify_user", "input": {}, "output": "ok"}],
        )

    def test_log_chat_turn_skips_when_empty_and_no_tools(self, tmp_path):
        """Truly silent turn (no text, no tools) — nothing to persist.
        Bug 3: ``_log_chat_turn`` only substitutes the marker when there
        ARE tool_outputs, so a deliberate ``__SILENT__`` reply (empty
        text, no tools) still skips persistence — no assistant row lands
        in the transcript."""
        from src.agent.workspace import WorkspaceManager

        loop = _make_loop()
        loop.workspace = WorkspaceManager(workspace_dir=str(tmp_path))

        loop._log_chat_turn("user q", "", tool_outputs=None)

        transcript = loop.workspace.load_chat_transcript()
        assistant_entries = [
            m for m in transcript if m.get("role") == "assistant"
        ]
        assert assistant_entries == []

    def test_synthesize_helper_pluralization(self):
        """Pure-function contract on the helper. Bug 3: Empty/None now
        returns the GENERIC (no-tools) marker, never ``""``. 1 tool →
        singular "tool call". 2+ tools → plural "tool calls"."""
        generic = AgentLoop._synthesize_empty_chat_fallback(None)
        assert generic.startswith("(No response")
        assert AgentLoop._synthesize_empty_chat_fallback([]) == generic

        one = AgentLoop._synthesize_empty_chat_fallback([{"tool": "x"}])
        assert "1 tool call" in one
        assert "1 tool calls" not in one  # no plural mis-fire

        two = AgentLoop._synthesize_empty_chat_fallback(
            [{"tool": "x"}, {"tool": "y"}],
        )
        assert "2 tool calls" in two

        five = AgentLoop._synthesize_empty_chat_fallback(
            [{"tool": "x"}] * 5,
        )
        assert "5 tool calls" in five

    def test_chat_fallback_wording_matches_log_chat_turn(self, tmp_path):
        """Anti-drift guard: the string written to the transcript by
        ``_log_chat_turn`` must equal the helper's return value for the
        same ``tool_outputs``. If someone re-inlines the wording in one
        path but not the other, this test catches the divergence."""
        from src.agent.workspace import WorkspaceManager

        loop = _make_loop()
        loop.workspace = WorkspaceManager(workspace_dir=str(tmp_path))

        tool_outputs = [
            {"tool": "web_search", "input": {"q": "x"}, "output": "r"},
            {"tool": "notify_user", "input": {}, "output": "ok"},
        ]
        expected = AgentLoop._synthesize_empty_chat_fallback(tool_outputs)
        assert expected, "helper must return non-empty when tools ran"

        loop._log_chat_turn("user q", "", tool_outputs=tool_outputs)

        transcript = loop.workspace.load_chat_transcript()
        assistant_entries = [
            m for m in transcript if m.get("role") == "assistant"
        ]
        assert len(assistant_entries) == 1
        assert assistant_entries[0]["content"] == expected


# === Round-4 structural fix: system-side hand_off failure enforcement ===


class TestChatHandoffFailureEnforcement:
    """``AgentLoop._chat_result_failure_reason`` scans ``tool_outputs`` for
    ``hand_off`` results and forces a failure reason when the downstream
    chain didn't land (``handed_off=False`` or any of ``create_failed`` /
    ``wake_failed`` / ``output_write_failed``). The previous design trusted
    the LLM to surface those failures in its text; models ignored the
    directive across three repro cycles, so enforcement is now system-side.
    These tests pin the contract on the pure staticmethod — no async needed.
    """

    def test_create_failed_handoff_fails_task(self):
        result = {
            "response": "Done, handed off to bob",
            "tool_outputs": [
                {
                    "tool": "hand_off",
                    "input": {"to": "bob"},
                    "output": {
                        "handed_off": False,
                        "create_failed": True,
                        "to": "bob",
                        "error": "create_failed: bob does not exist",
                    },
                },
            ],
        }
        reason = AgentLoop._chat_result_failure_reason(result)
        assert reason is not None
        assert reason.startswith("handoff_failed:")
        assert "bob" in reason

    def test_wake_failed_handoff_fails_task(self):
        result = {
            "response": "Handed off",
            "tool_outputs": [
                {
                    "tool": "hand_off",
                    "input": {"to": "carol"},
                    "output": {
                        "handed_off": False,
                        "wake_failed": True,
                        "to": "carol",
                        "error": "wake_failed: timeout",
                    },
                },
            ],
        }
        reason = AgentLoop._chat_result_failure_reason(result)
        assert reason is not None
        assert reason.startswith("handoff_failed:")
        assert "wake_failed" in reason or "handoff_failed:" in reason

    def test_output_write_failed_handoff_fails_task(self):
        result = {
            "response": "All good",
            "tool_outputs": [
                {
                    "tool": "hand_off",
                    "input": {"to": "dave"},
                    "output": {
                        "handed_off": False,
                        "output_write_failed": True,
                        "to": "dave",
                        "error": "output_write_failed: disk full",
                    },
                },
            ],
        }
        reason = AgentLoop._chat_result_failure_reason(result)
        assert reason is not None
        assert reason.startswith("handoff_failed:")
        assert "output_write_failed" in reason or "dave" in reason

    def test_handed_off_false_alone_fails_task(self):
        """No specific failure flag, just ``handed_off=False`` — still
        a failure (legacy/future payload shape)."""
        result = {
            "response": "Tried to hand off",
            "tool_outputs": [
                {
                    "tool": "hand_off",
                    "input": {"to": "eve"},
                    "output": {"handed_off": False, "to": "eve"},
                },
            ],
        }
        reason = AgentLoop._chat_result_failure_reason(result)
        assert reason is not None
        assert reason.startswith("handoff_failed:")

    def test_successful_handoff_does_not_fail_task(self):
        result = {
            "response": "Handed off to bob",
            "tool_outputs": [
                {
                    "tool": "hand_off",
                    "input": {"to": "bob"},
                    "output": {
                        "handed_off": True,
                        "to": "bob",
                        "task_id": "task_X",
                    },
                },
            ],
        }
        assert AgentLoop._chat_result_failure_reason(result) is None

    def test_non_handoff_tool_output_ignored(self):
        """Scan only triggers on ``tool == "hand_off"`` — same keys on
        a different tool's payload must NOT fail the task."""
        result = {
            "response": "Notified",
            "tool_outputs": [
                {
                    "tool": "notify_user",
                    "input": {},
                    "output": {"create_failed": True},
                },
            ],
        }
        assert AgentLoop._chat_result_failure_reason(result) is None

    def test_alternate_payload_key_result(self):
        """Some dispatchers expose the payload under ``result`` instead
        of ``output`` — both shapes must be recognized."""
        result = {
            "response": "...",
            "tool_outputs": [
                {
                    "tool": "hand_off",
                    "input": {"to": "bob"},
                    "result": {"create_failed": True, "to": "bob"},
                },
            ],
        }
        reason = AgentLoop._chat_result_failure_reason(result)
        assert reason is not None
        assert reason.startswith("handoff_failed:")

    def test_pre_existing_failures_take_precedence(self):
        """``tool_limit_reached`` is checked before the tool_outputs
        scan — even with a "successful" handoff in the same envelope,
        the iteration-limit failure wins."""
        result = {
            "tool_limit_reached": True,
            "tool_outputs": [
                {"tool": "hand_off", "input": {}, "output": {"handed_off": True}},
            ],
        }
        assert AgentLoop._chat_result_failure_reason(result) == \
            "max_iterations_reached"

    def test_real_tool_output_schema_with_create_failed(self):
        """Pin the integration with the actual ``_chat_inner`` schema
        ``{"tool": tool_name, "input": ..., "output": ...}`` from
        loop.py:2727. Guards against the PR #953 dead-code regression
        where the scanner used the wrong key (``name``) and matched
        nothing in production.
        """
        result = {
            "response": "tried to hand off to bob",
            "tool_outputs": [
                {
                    "tool": "hand_off",
                    "input": {"to": "bob", "context": "follow-up"},
                    "output": {
                        "handed_off": False,
                        "create_failed": True,
                        "to": "bob",
                    },
                },
            ],
        }
        reason = AgentLoop._chat_result_failure_reason(result)
        assert reason is not None
        assert reason.startswith("handoff_failed:")
        assert "bob" in reason


# === Outbound-effect lazy guard ===


class TestOutboundEffectLazyGuard:
    """``_has_outbound_effect`` flags any tool_outputs list that contains
    at least one tool name NOT in ``_READ_ONLY_TOOLS``. Used by the chat
    handoff lazy-completion guard so ghost-completions that only run
    diagnostic reads (``check_inbox`` + ``read_blackboard`` + …) get
    auto-closed as failed instead of slipping past as ``done``.

    All tests pure-function on the module-level helper — no fixtures.
    """

    def test_empty_tool_outputs_returns_false(self):
        assert _has_outbound_effect(None) is False
        assert _has_outbound_effect([]) is False

    def test_only_read_only_tools_returns_false(self):
        outputs = [
            {"tool": "check_inbox"},
            {"tool": "read_blackboard"},
        ]
        assert _has_outbound_effect(outputs) is False

    def test_any_outbound_tool_returns_true(self):
        outputs = [
            {"tool": "check_inbox"},
            {"tool": "hand_off"},
        ]
        assert _has_outbound_effect(outputs) is True

    def test_unknown_tool_treated_as_outbound(self):
        """Default-outbound policy: a brand-new tool the guard hasn't
        seen yet is conservatively counted as an outbound effect (lenient
        false negatives are preferred over false positives that would
        block real work)."""
        outputs = [{"tool": "some_brand_new_tool_we_havent_seen"}]
        assert _has_outbound_effect(outputs) is True

    def test_legacy_name_key_fallback(self):
        """Some older tool-output schemas use ``name`` instead of
        ``tool``; the helper falls back to ``name`` when ``tool`` is
        absent so both shapes are recognized."""
        outputs = [{"name": "hand_off"}]
        assert _has_outbound_effect(outputs) is True

    def test_hand_off_alone_is_outbound(self):
        assert _has_outbound_effect([{"tool": "hand_off"}]) is True

    def test_write_blackboard_is_outbound(self):
        assert _has_outbound_effect([{"tool": "write_blackboard"}]) is True

    def test_notify_user_is_outbound(self):
        assert _has_outbound_effect([{"tool": "notify_user"}]) is True

    def test_memory_search_is_read_only(self):
        assert _has_outbound_effect([{"tool": "memory_search"}]) is False

    def test_read_only_constant_includes_documented_tools(self):
        """Pin the documented read-only set so adding/removing entries
        is a deliberate change reviewed alongside the guard."""
        expected_subset = {
            "check_inbox",
            "read_blackboard",
            "list_blackboard",
            "list_agents",
            "get_agent_profile",
            "get_system_status",
            "workflow_snapshot",
            "await_task_event",
            "inspect_agents",
            "inspect_teams",
            "list_agent_queue",
            "memory_search",
            "read_user_notifications",
        }
        assert expected_subset.issubset(_READ_ONLY_TOOLS)

    @pytest.mark.asyncio
    async def test_chat_handoff_with_only_read_tools_auto_closes_failed(self):
        """Integration: a chat() handoff turn whose tool_outputs are ALL
        read-only (no outbound effect) and whose response is empty
        auto-closes the task as ``failed`` with a ``no_outbound_effects``
        error envelope. Mirrors the production repro that motivated the
        guard strengthening."""
        loop = _make_loop()

        async def fake_inner(_msg, **_kw):
            return {
                "response": "",
                "tool_outputs": [
                    {
                        "tool": "check_inbox",
                        "input": {},
                        "output": {"events": []},
                    },
                ],
                "tokens_used": 5,
            }
        loop._chat_inner = fake_inner
        loop._auto_close_task = AsyncMock(return_value=None)

        await loop.chat("hi", task_id="task_lazy")

        # Find the terminal close call (status="failed"). The first
        # auto_close_task call opens the task as ``working``; the second
        # is the terminal transition.
        terminal_calls = [
            c for c in loop._auto_close_task.await_args_list
            if len(c.args) >= 2 and c.args[1] == "failed"
        ]
        assert terminal_calls, (
            "expected an auto_close_task(..., 'failed') terminal call "
            f"but saw: {loop._auto_close_task.await_args_list}"
        )
        last_failed = terminal_calls[-1]
        assert last_failed.args[0] == "task_lazy"
        error_msg = last_failed.kwargs.get("error") or ""
        assert "no_outbound_effects" in error_msg, (
            f"expected error envelope to contain 'no_outbound_effects', "
            f"got: {error_msg!r}"
        )
        assert "check_inbox" in error_msg, (
            "error envelope should surface the read-only tools that were "
            f"called, got: {error_msg!r}"
        )

    @pytest.mark.asyncio
    async def test_chat_handoff_with_outbound_tool_auto_closes_done(self):
        """Counterpart to the read-only case: a handoff turn whose
        tool_outputs include at least one outbound effect (here:
        ``hand_off``) and whose response is empty auto-closes as
        ``done`` (the outbound work proves the turn wasn't lazy)."""
        loop = _make_loop()

        async def fake_inner(_msg, **_kw):
            return {
                "response": "",
                "tool_outputs": [
                    {
                        "tool": "hand_off",
                        "input": {"to": "bob"},
                        "output": {
                            "handed_off": True,
                            "to": "bob",
                            "task_id": "task_child",
                        },
                    },
                ],
                "tokens_used": 5,
            }
        loop._chat_inner = fake_inner
        loop._auto_close_task = AsyncMock(return_value=None)

        await loop.chat("hi", task_id="task_outbound")

        terminal_calls = [
            c for c in loop._auto_close_task.await_args_list
            if len(c.args) >= 2 and c.args[1] in ("done", "failed")
        ]
        assert terminal_calls
        last_terminal = terminal_calls[-1]
        assert last_terminal.args[1] == "done", (
            "outbound-effect turn must close as 'done', not 'failed'; "
            f"calls: {loop._auto_close_task.await_args_list}"
        )

    @pytest.mark.asyncio
    async def test_chat_handoff_read_only_with_prose_auto_closes_deferred(self):
        """Bug 3 (explained-deferral carve-out): a handoff turn whose
        tool_outputs are ALL read-only (no outbound effect) BUT whose
        response is non-empty prose is a correct DEFERRAL, not a ghost.
        It auto-closes as ``done`` with a ``deferred`` result payload
        instead of ``failed`` (which polluted failure metrics)."""
        loop = _make_loop()

        async def fake_inner(_msg, **_kw):
            return {
                "response": (
                    "Already a pending dedup entry on the blackboard for "
                    "this request — deferring to avoid duplicate work."
                ),
                "tool_outputs": [
                    {
                        "tool": "read_blackboard",
                        "input": {"key": "queue"},
                        "output": {"value": "pending"},
                    },
                ],
                "tokens_used": 5,
            }
        loop._chat_inner = fake_inner
        loop._auto_close_task = AsyncMock(return_value=None)

        await loop.chat("hi", task_id="task_deferred")

        terminal_calls = [
            c for c in loop._auto_close_task.await_args_list
            if len(c.args) >= 2 and c.args[1] in ("done", "failed")
        ]
        assert terminal_calls
        last_terminal = terminal_calls[-1]
        assert last_terminal.args[1] == "done", (
            "read-only turn WITH a prose explanation must close as 'done' "
            f"(deferral), not 'failed'; calls: "
            f"{loop._auto_close_task.await_args_list}"
        )
        payload = last_terminal.kwargs.get("result_payload") or {}
        assert payload.get("status") == "deferred"
        assert "deferring" in payload.get("reason", "")
        # summary feeds the done back-edge → originator's check_inbox
        assert payload.get("summary") == payload.get("reason")

    @pytest.mark.asyncio
    async def test_ghost_handoff_empty_response_closes_failed(self):
        """Regression (don't-let-marker-mask-ghost): a handoff turn that
        calls ONE read-only tool then produces NO final text is a ghost
        completion. ``_chat_inner`` substitutes a synthetic empty-turn
        marker into ``result["response"]`` (flagging ``silent_reply``).
        The explained-deferral carve-out MUST ignore marker turns, so the
        task auto-closes as ``failed`` (``no_outbound_effects``) — the
        marker must NOT be mistaken for a genuine deferral explanation.

        Drives the REAL ``_chat_inner`` (round 1: read-only tool call +
        empty text; round 2: empty final → triggers the empty-compose
        retry; retry: also empty → marker substituted) so the actual
        marker/silent_reply interaction with the carve-out is exercised.
        Without the ``not result.get("silent_reply")`` gate this closes as
        ``done``/deferred instead — defeating the ghost guard.
        """
        read_only_call = ToolCallInfo(name="check_inbox", arguments={})
        responses = [
            _resp("", tool_calls=[read_only_call]),  # round 1: read-only tool
            _resp(""),                               # round 2: empty final
            _resp(""),                               # empty-compose retry: empty
        ]
        loop = _make_loop(responses)
        loop.tools.get_tool_definitions = MagicMock(
            return_value=[{"name": "check_inbox"}]
        )
        loop.tools.execute = AsyncMock(return_value={"events": []})
        loop._auto_close_task = AsyncMock(return_value=None)

        result = await loop.chat("hi", task_id="t-ghost")

        # The marker was substituted (user never sees a blank bubble) —
        # but the task STATUS must still be a ghost failure.
        assert result.get("silent_reply") is True
        assert result["response"].strip() != ""

        terminal_calls = [
            c for c in loop._auto_close_task.await_args_list
            if len(c.args) >= 2 and c.args[1] in ("done", "failed")
        ]
        assert terminal_calls, (
            "expected a terminal auto_close_task call, saw: "
            f"{loop._auto_close_task.await_args_list}"
        )
        last_terminal = terminal_calls[-1]
        assert last_terminal.args[1] == "failed", (
            "ghost turn (read-only tool, empty real text → marker) must "
            "close as 'failed', not 'done'/deferred; the synthetic marker "
            "must not mask the ghost. calls: "
            f"{loop._auto_close_task.await_args_list}"
        )
        assert last_terminal.args[0] == "t-ghost"
        error_msg = last_terminal.kwargs.get("error") or ""
        assert "no_outbound_effects" in error_msg, (
            f"expected 'no_outbound_effects' in error, got: {error_msg!r}"
        )

    @pytest.mark.asyncio
    async def test_genuine_deferral_handoff_closes_done_deferred(self):
        """Counterpart: a handoff turn that calls ONE read-only tool then
        returns REAL prose (a genuine deferral explanation, ``silent_reply``
        NOT set) still takes the carve-out → auto-closes as ``done`` with a
        ``deferred`` result payload. The ``silent_reply`` gate only excludes
        synthetic marker turns; genuine explanations are unaffected.
        """
        read_only_call = ToolCallInfo(
            name="read_blackboard", arguments={"key": "queue"},
        )
        responses = [
            _resp("", tool_calls=[read_only_call]),  # round 1: read-only tool
            _resp("Reviewed the blackboard; no action needed because X."),
        ]
        loop = _make_loop(responses)
        loop.tools.get_tool_definitions = MagicMock(
            return_value=[{"name": "read_blackboard"}]
        )
        loop.tools.execute = AsyncMock(return_value={"value": "pending"})
        loop._auto_close_task = AsyncMock(return_value=None)

        result = await loop.chat("hi", task_id="t-deferral")

        # Genuine prose: no marker, silent_reply not set.
        assert result.get("silent_reply") is not True
        assert "no action needed" in result["response"]

        terminal_calls = [
            c for c in loop._auto_close_task.await_args_list
            if len(c.args) >= 2 and c.args[1] in ("done", "failed")
        ]
        assert terminal_calls
        last_terminal = terminal_calls[-1]
        assert last_terminal.args[1] == "done", (
            "genuine deferral (read-only tool + real prose) must close as "
            f"'done', not 'failed'. calls: {loop._auto_close_task.await_args_list}"
        )
        assert last_terminal.args[0] == "t-deferral"
        payload = last_terminal.kwargs.get("result_payload") or {}
        assert payload.get("status") == "deferred", (
            f"expected deferred result payload, got: {payload!r}"
        )


# === Bug 3 chain pin: chat() error → blocker_note wiring ===


class TestChatAutoCloseErrorPropagation:
    """Bug 3 chain pin. When an LLM error fires during a ``chat()`` turn
    that rode an ``x-task-id`` wake chain, the auto-close MUST pass the
    rejection reason through ``mesh_client.set_task_status(error=...)``.

    The 6-step propagation chain is:
        LLMConfigError raised inside ``_chat_inner``'s ``llm.chat`` call
            → ``_chat_inner`` catches and returns ``{config_error: True, ...}``
            → ``chat()`` post-call calls ``_chat_result_failure_reason``
            → ``chat()`` calls ``_auto_close_task(task_id, "failed", error=reason)``
            → ``_auto_close_task`` calls ``mesh_client.set_task_status(error=reason)``
            → ``POST /tasks/{id}/status`` → ``update_status`` promotes to ``blocker_note``.

    Each end has unit tests (see ``test_execute_task_catches_llm_config_error``
    on the loop side, ``test_update_status_failed_persists_blocker_note``
    on the orchestration side). This class pins the CONNECTING wiring
    inside ``chat()`` itself so a refactor cannot break it silently.
    """

    @pytest.mark.asyncio
    async def test_chat_with_task_id_propagates_llm_config_error_to_auto_close(
        self,
    ):
        """When chat() runs a task_id-bearing turn and the LLM raises
        LLMConfigError, mesh_client.set_task_status must be called with
        status='failed' AND error= containing the config_error reason.
        The mesh-side endpoint promotes that to blocker_note. Without
        this wiring the operator sees a bare 'failed' with no reason."""
        from src.shared.errors import LLMConfigError

        loop = _make_loop()
        # Force the LLM call inside _chat_inner to raise LLMConfigError.
        # _chat_inner catches it and returns a dict tagged config_error=True;
        # chat() then routes that through _chat_result_failure_reason →
        # _auto_close_task('failed', error=...) → set_task_status.
        loop.llm.chat = AsyncMock(
            side_effect=LLMConfigError(
                "OAuth-allowed models: ['openai/gpt-5']; got 'openai/gpt-4o-mini'",
                provider="openai",
                model="openai/gpt-4o-mini",
                allowed_models={"openai/gpt-5", "openai/gpt-5-mini"},
            ),
        )
        loop.tools.get_tool_definitions = MagicMock(return_value=[])

        # Spy on mesh_client.set_task_status — the terminal sink of the
        # propagation chain. The mesh-side handler reads the ``error``
        # kwarg and promotes it to blocker_note.
        loop.mesh_client.set_task_status = AsyncMock(
            return_value={"status": "failed"},
        )

        # Drive chat() with a task_id riding the wake chain. LLMConfigError
        # does NOT propagate out (caught inside _chat_inner) — chat returns
        # a config_error-tagged dict instead.
        result = await loop.chat("do the thing", task_id="task_abc")
        assert result.get("config_error") is True, (
            "_chat_inner must tag the result with config_error=True so "
            "chat()'s post-call failure-reason scan can detect it"
        )

        # Verify the wiring: set_task_status was called with status=failed
        # AND error= containing the config_error string. The error is what
        # downstream is promoted to blocker_note.
        calls = loop.mesh_client.set_task_status.await_args_list
        failed_calls = [
            c for c in calls
            if (len(c.args) >= 2 and c.args[1] == "failed")
            or c.kwargs.get("status") == "failed"
        ]
        assert len(failed_calls) >= 1, (
            f"Expected at least one set_task_status(failed) call, "
            f"got: {calls}"
        )
        err_kwarg = failed_calls[-1].kwargs.get("error")
        assert err_kwarg is not None, (
            "set_task_status MUST receive ``error=`` for chain to "
            "propagate the rejection reason to blocker_note"
        )
        # The PR contract is that BOTH pieces flow through: the
        # ``config_error`` prefix (so downstream UI can branch on it) AND
        # the actual LLMConfigError message content (so the operator
        # learns WHY their model was rejected). A bare ``config_error``
        # label with no body is exactly the regression this test guards
        # against — assert both pieces, not "either".
        assert "config_error" in err_kwarg, (
            f"expected ``config_error`` prefix, got: {err_kwarg!r}"
        )
        assert "OAuth-allowed models" in err_kwarg, (
            "expected the LLMConfigError message content to flow through — "
            "a generic 'config_error' label with no body means the actual "
            "rejection reason isn't reaching the operator; "
            f"got: {err_kwarg!r}"
        )
        # Bug 3 truncation contract: failure-reason text is sliced to 400
        # chars in _chat_result_failure_reason; the chain-tail _auto_close
        # truncation cap is 500 chars. Either way the kwarg stays bounded.
        assert len(err_kwarg) <= 500


# === Bug 2 fix: in-flight chat turn is finalized cleanly on errors/terminate ===
#
# Before the fix, exception and tool-loop-terminate paths wrote a NEW
# assistant transcript entry without ``turn_id``. The earlier ``partial``
# entry (written before tool dispatch) stayed on disk as ``partial=True``
# next to a separate "Error: ..." or "Stopped: ..." bubble. With
# ``load_chat_transcript`` deduping by turn_id, the partial was an
# orphan. These tests pin the ``_finalize_chat_turn`` helper that closes
# the turn cleanly: one entry, same turn_id as the partial, content =
# ``<accumulated text>\n\n<closing message>``.


class TestFinalizeChatTurn:
    """Helper-level pin. Exercises the contract directly without
    standing up the full chat() / _chat_inner async machinery."""

    def test_finalize_combines_accumulated_text_with_closing_message(
        self, tmp_path,
    ):
        from src.agent.workspace import WorkspaceManager
        loop = _make_loop()
        loop.workspace = WorkspaceManager(workspace_dir=str(tmp_path))

        loop._finalize_chat_turn(
            turn_id="turn-1",
            accumulated_content="streamed text so far",
            tool_names=["tool_a", "tool_b"],
            closing_message="Error: boom",
        )

        msgs = loop.workspace.load_chat_transcript()
        assert len(msgs) == 1
        entry = msgs[0]
        assert entry["role"] == "assistant"
        assert entry["content"] == "streamed text so far\n\nError: boom"
        assert entry.get("turn_id") == "turn-1"
        assert entry.get("tools") == ["tool_a", "tool_b"]
        # Final entry — NOT partial.
        assert entry.get("partial") is not True

    def test_finalize_supersedes_in_flight_partial(self, tmp_path):
        """The same turn_id as a prior partial entry must replace it in
        the transcript dedupe. Pins the orphan-partial fix."""
        from src.agent.workspace import WorkspaceManager
        loop = _make_loop()
        loop.workspace = WorkspaceManager(workspace_dir=str(tmp_path))

        # Simulate the partial write that happens before tool dispatch.
        loop.workspace.append_chat_message(
            "assistant", "streamed text so far",
            tool_names=["tool_a"], turn_id="turn-1", partial=True,
        )
        # Hit the error path → finalize.
        loop._finalize_chat_turn(
            turn_id="turn-1",
            accumulated_content="streamed text so far",
            tool_names=["tool_a"],
            closing_message="Error: boom",
        )

        msgs = loop.workspace.load_chat_transcript()
        # Dedupe by turn_id: ONE entry, the final one.
        assert len(msgs) == 1
        assert msgs[0]["content"] == "streamed text so far\n\nError: boom"
        assert msgs[0].get("partial") is not True

    def test_finalize_with_empty_accumulated_content_writes_only_closing(
        self, tmp_path,
    ):
        """No streamed text yet (exception fired before LLM emitted any
        tokens) → only the closing message lands, no leading blank lines."""
        from src.agent.workspace import WorkspaceManager
        loop = _make_loop()
        loop.workspace = WorkspaceManager(workspace_dir=str(tmp_path))

        loop._finalize_chat_turn(
            turn_id="turn-1",
            accumulated_content="",
            tool_names=[],
            closing_message="Error: boom",
        )
        msgs = loop.workspace.load_chat_transcript()
        assert len(msgs) == 1
        assert msgs[0]["content"] == "Error: boom"
        assert msgs[0].get("turn_id") == "turn-1"
        # Empty tool list → no ``tools`` key in entry.
        assert "tools" not in msgs[0] or msgs[0]["tools"] == []

    def test_finalize_no_workspace_is_noop(self):
        """Agents without a mounted workspace must skip silently — same
        pattern as the existing ``if self.workspace:`` guard."""
        loop = _make_loop()
        loop.workspace = None
        # Must not raise.
        loop._finalize_chat_turn(
            turn_id="turn-1",
            accumulated_content="text",
            tool_names=["a"],
            closing_message="Error: boom",
        )

    def test_finalize_rejects_empty_turn_id(self, tmp_path):
        """An empty/None turn_id breaks the partial-dedupe contract —
        the helper must refuse and raise so callers can't silently
        regress to the orphaned-partial bug."""
        from src.agent.workspace import WorkspaceManager
        loop = _make_loop()
        loop.workspace = WorkspaceManager(workspace_dir=str(tmp_path))
        with pytest.raises(ValueError, match="turn_id"):
            loop._finalize_chat_turn(
                turn_id="",
                accumulated_content="text",
                tool_names=["a"],
                closing_message="Error: boom",
            )


class TestChatExceptionPathsFinalizeCleanly:
    """End-to-end pin: the exception handlers in ``_chat_inner`` must
    route through ``_finalize_chat_turn`` (NOT the raw ``append_chat_message``)
    so the in-flight partial is superseded cleanly."""

    @pytest.mark.asyncio
    async def test_chat_inner_llm_config_error_finalizes_partial(
        self, tmp_path,
    ):
        from src.agent.workspace import WorkspaceManager
        from src.shared.errors import LLMConfigError

        loop = _make_loop()
        loop.workspace = WorkspaceManager(workspace_dir=str(tmp_path))
        loop.tools.get_tool_definitions = MagicMock(return_value=[])
        # First call: LLMConfigError mid-turn. Before this point a partial
        # MAY exist on disk if a prior round wrote one — simulate that
        # case so we verify supersedence.
        loop.workspace.append_chat_message(
            "assistant", "intermediate streamed text",
            tool_names=["search"], turn_id="will-be-overwritten",
            partial=True,
        )
        # Pre-state: one partial entry on disk.
        pre = loop.workspace.load_chat_transcript()
        assert len(pre) == 1
        assert pre[0].get("partial") is True

        loop.llm.chat = AsyncMock(side_effect=LLMConfigError(
            "OAuth-allowed models: [...]; got 'openai/gpt-4o-mini'",
            provider="openai",
            model="openai/gpt-4o-mini",
            allowed_models=set(),
        ))

        result = await loop._chat_inner("hello")
        assert result.get("config_error") is True

        msgs = loop.workspace.load_chat_transcript()
        # The legacy partial is unrelated to the new turn_id — it stays
        # as a separate entry. The NEW turn's error finalizes cleanly
        # (no orphan partial for it).
        # New entry's content must surface the error.
        error_entries = [
            m for m in msgs if "Config error" in m.get("content", "")
        ]
        assert len(error_entries) == 1
        assert error_entries[0].get("partial") is not True
        assert error_entries[0].get("turn_id"), (
            "error finalize entry must carry a turn_id so any future "
            "partial from the same turn dedupes against it"
        )

    @pytest.mark.asyncio
    async def test_chat_stream_inner_generic_exception_finalizes_partial(
        self, tmp_path,
    ):
        """Streaming path's generic Exception handler must finalize via
        the same helper. Verify by patching ``_finalize_chat_turn`` and
        checking it gets called with the error message."""
        from src.agent.workspace import WorkspaceManager

        loop = _make_loop()
        loop.workspace = WorkspaceManager(workspace_dir=str(tmp_path))
        loop.tools.get_tool_definitions = MagicMock(return_value=[])
        loop.llm.chat_stream = MagicMock(
            side_effect=RuntimeError("kaboom"),
        )
        loop.llm.chat = AsyncMock(side_effect=RuntimeError("kaboom"))

        finalize_spy = MagicMock()
        loop._finalize_chat_turn = finalize_spy

        events = []
        async for evt in loop._chat_stream_inner("hello"):
            events.append(evt)

        # Helper invoked exactly once for this turn.
        assert finalize_spy.call_count == 1
        call = finalize_spy.call_args
        assert "Error: kaboom" in call.kwargs["closing_message"]
        assert call.kwargs.get("turn_id"), (
            "finalize must receive the turn's turn_id so the partial "
            "supersedes cleanly"
        )


# ── Bug 3: empty-chat-turn recovery (retry-then-marker) ──────────────
#
# A chat turn that runs tools (or even none) but produces no final text
# must NEVER surface a blank reply unless the model deliberately emitted
# ``__SILENT__``. ``_chat_inner`` / ``_chat_stream_inner`` retry the
# final compose once with tools withheld; if still empty they substitute
# a marker via ``_synthesize_empty_chat_fallback`` and flag the result
# with ``silent_reply`` so the dashboard can badge the recovery. These
# tests pin that contract across BOTH surfaces, regardless of task_id /
# tool_limit_reached, and confirm deliberate silence is preserved.


def _resp(content, tool_calls=None, tokens=10):
    return LLMResponse(content=content, tool_calls=tool_calls, tokens_used=tokens)


def _empty_after_tools_loop(final_responses):
    """An AgentLoop wired so round 1 calls a tool, then subsequent
    responses are returned in order. One tool round populates
    ``tool_outputs`` before the empty/recovery exit."""
    tool_call = ToolCallInfo(name="search", arguments={"q": "x"})
    responses = [_resp("", tool_calls=[tool_call])] + list(final_responses)
    loop = _make_loop(responses)
    loop.tools.get_tool_definitions = MagicMock(return_value=[{"name": "search"}])
    loop.tools.execute = AsyncMock(return_value={"ok": True})
    return loop


@pytest.mark.asyncio
async def test_empty_chat_after_tools_retry_succeeds_returns_recovered_text():
    """Empty final text after tools, retry yields prose → that prose is
    the response; no marker; ``silent_reply`` not set."""
    # round 1: tool call. round 2: empty final → triggers retry.
    # retry compose (tools withheld): real text.
    loop = _empty_after_tools_loop([_resp(""), _resp("Recovered answer")])
    result = await loop.chat("ping")
    assert result["response"] == "Recovered answer"
    assert result.get("silent_reply") is not True
    assert loop._chat_messages[-1]["content"] == "Recovered answer"


@pytest.mark.asyncio
async def test_empty_chat_after_tools_retry_also_empty_returns_marker():
    """Empty final text after tools, retry also empty → marker response,
    ``silent_reply`` True."""
    loop = _empty_after_tools_loop([_resp(""), _resp("")])
    result = await loop.chat("ping")
    assert "Completed" in result["response"]
    assert "tool call" in result["response"]
    assert result.get("silent_reply") is True


@pytest.mark.asyncio
async def test_empty_chat_after_tools_with_task_id_is_rescued():
    """task_id set must NOT gate the rescue (the old gate is removed):
    the response is non-empty (recovered text or marker)."""
    loop = _empty_after_tools_loop([_resp(""), _resp("")])
    loop._auto_close_task = AsyncMock(return_value=None)
    result = await loop.chat("ping", task_id="task-123")
    assert result["response"].strip() != ""


@pytest.mark.asyncio
async def test_empty_chat_tool_limit_reached_is_rescued():
    """The tool-limit force-compose exit must also be rescued — an empty
    forced compose yields the marker, not a blank reply."""
    loop = _make_loop()
    loop.CHAT_MAX_TOOL_ROUNDS = 2
    tool_call = ToolCallInfo(name="search", arguments={"q": "x"})
    responses = [
        _resp("", tool_calls=[tool_call]),
        _resp("", tool_calls=[tool_call]),
        _resp(""),  # force-final compose: empty
    ]
    loop.llm.chat = AsyncMock(side_effect=responses)
    loop.tools.get_tool_definitions = MagicMock(return_value=[{"name": "search"}])
    loop.tools.execute = AsyncMock(return_value={"ok": True})
    result = await loop.chat("ping")
    assert result.get("tool_limit_reached") is True
    assert result["response"].strip() != ""
    assert result.get("silent_reply") is True


@pytest.mark.asyncio
async def test_empty_chat_zero_tools_retry_empty_returns_generic_marker():
    """Empty final text, ZERO tool_outputs, retry also empty → the
    generic (no-tools) marker, never an empty string."""
    # round 1 (no tools): empty → retry. retry: empty.
    loop = _make_loop([_resp(""), _resp("")])
    result = await loop.chat("ping")
    assert result["response"].startswith("(No response")
    assert result.get("silent_reply") is True
    assert result["response"] != ""


@pytest.mark.asyncio
async def test_deliberate_silent_after_tools_not_rescued_and_no_retry():
    """A deliberate ``__SILENT__`` final (after a tool round) yields an
    empty reply with NO marker, and the retry compose is NOT called
    (proven by the LLM mock call count)."""
    loop = _empty_after_tools_loop([_resp(SILENT_REPLY_TOKEN)])
    result = await loop.chat("ping")
    assert result["response"] == ""
    assert result.get("silent_reply") is True
    # 2 calls only: round-1 tool call + the silent final. No 3rd retry.
    assert loop.llm.chat.call_count == 2


@pytest.mark.asyncio
async def test_normal_nonempty_response_unchanged():
    """A normal text reply is returned unchanged (regression guard) and
    the retry compose is never invoked."""
    loop = _make_loop([_resp("All good.")])
    result = await loop.chat("ping")
    assert result["response"] == "All good."
    assert result.get("silent_reply") is not True
    assert loop.llm.chat.call_count == 1


@pytest.mark.asyncio
async def test_stream_empty_after_tools_emits_marker_text_event():
    """Streaming: empty final text after tools (retry also empty) emits a
    text_delta carrying the marker AND the done event carries it too."""
    tool_call = ToolCallInfo(name="search", arguments={"q": "x"})
    responses = [
        _resp("", tool_calls=[tool_call]),  # round 1: tool call
        _resp(""),                          # round 2: empty final
        _resp(""),                          # retry compose: empty
    ]
    loop = _make_loop(responses)
    # Force the non-streaming fallback path: chat_stream raises so the
    # loop falls back to ``self.llm.chat`` (our AsyncMock side_effect).
    loop.llm.chat_stream = MagicMock(side_effect=RuntimeError("no stream"))
    loop.tools.get_tool_definitions = MagicMock(return_value=[{"name": "search"}])
    loop.tools.execute = AsyncMock(return_value={"ok": True})
    events = [ev async for ev in loop.chat_stream("ping")]
    text_events = [e for e in events if e.get("type") == "text_delta"]
    done_events = [e for e in events if e.get("type") == "done"]
    assert any("Completed" in (e.get("content") or "") for e in text_events)
    assert done_events
    assert "Completed" in (done_events[-1].get("response") or "")


@pytest.mark.asyncio
async def test_stream_empty_after_tools_retry_succeeds_emits_recovered_text():
    """Streaming: empty final text after tools, retry yields prose → a
    text_delta and done carry the recovered prose, not a marker."""
    tool_call = ToolCallInfo(name="search", arguments={"q": "x"})
    responses = [
        _resp("", tool_calls=[tool_call]),   # round 1: tool call
        _resp(""),                           # round 2: empty final
        _resp("Streamed recovery"),          # retry compose: prose
    ]
    loop = _make_loop(responses)
    loop.llm.chat_stream = MagicMock(side_effect=RuntimeError("no stream"))
    loop.tools.get_tool_definitions = MagicMock(return_value=[{"name": "search"}])
    loop.tools.execute = AsyncMock(return_value={"ok": True})
    events = [ev async for ev in loop.chat_stream("ping")]
    text_events = [e for e in events if e.get("type") == "text_delta"]
    done_events = [e for e in events if e.get("type") == "done"]
    assert any("Streamed recovery" in (e.get("content") or "") for e in text_events)
    assert done_events[-1]["response"] == "Streamed recovery"


@pytest.mark.asyncio
async def test_empty_compose_retry_reraises_auth_error():
    """Regression: a credential outage DURING the empty-compose retry must
    surface as ``auth_failure`` — not be swallowed into a benign
    ``"(no response)"`` marker. The helper re-raises LLMAuthError /
    LLMConfigError; ``_chat_inner``'s typed ``except`` arm tags the result.
    """
    from src.shared.errors import LLMAuthError

    tool_call = ToolCallInfo(name="search", arguments={"q": "x"})
    # Round 1: tool call + empty text. Round 2: empty final → triggers retry.
    # The retry compose (3rd llm.chat) raises LLMAuthError.
    responses = [
        _resp("", tool_calls=[tool_call]),
        _resp(""),
        LLMAuthError("credentials revoked", provider="anthropic"),
    ]
    loop = _make_loop(responses)
    loop.tools.get_tool_definitions = MagicMock(return_value=[{"name": "search"}])
    loop.tools.execute = AsyncMock(return_value={"ok": True})
    result = await loop._chat_inner("ping")
    assert result.get("auth_failure") is True
    # Must NOT be papered over as a successful "(no response)" marker.
    assert "(no response" not in (result.get("response") or "").lower()


@pytest.mark.asyncio
async def test_stream_deliberate_silence_not_retried():
    """Streaming deliberate ``__SILENT__`` reply: NO compose retry and NO
    marker text event (mirror of the non-stream deliberate-silence test).
    """
    loop = _make_loop([_resp(SILENT_REPLY_TOKEN)])
    # Force the non-streaming fallback so llm.chat (our side_effect) is used.
    loop.llm.chat_stream = MagicMock(side_effect=RuntimeError("no stream"))
    # Patch the retry helper to a sentinel so we can assert it is NOT called.
    loop._retry_empty_compose = AsyncMock()

    events = [ev async for ev in loop._chat_stream_inner("ping")]

    assert loop._retry_empty_compose.call_count == 0
    text_events = [e for e in events if e.get("type") == "text_delta"]
    assert text_events == []
    assert all(
        "(no response" not in (e.get("content") or "").lower() for e in events
    )
    done = [e for e in events if e.get("type") == "done"]
    assert done and done[-1].get("response") == ""


@pytest.mark.asyncio
async def test_retry_returns_silent_token_is_honoured_not_marked():
    """Bug 3: the first no-tool reply is empty (an accidental blip, NOT the
    silent token), so the empty-compose retry fires; the retry itself emits
    ``__SILENT__``. That deliberate silence on the retry MUST be honoured:
    empty response, ``silent_reply`` flagged, and NO marker substituted.

    Before the tri-state fix the silence check in ``_retry_empty_compose``
    was dead (the flag is set inside ``_resolve_content`` which ran AFTER
    the check), so the retry's ``__SILENT__`` was stripped to ``""`` and the
    caller papered it over with a marker — this test would FAIL.
    """
    loop = _make_loop([_resp(""), _resp(SILENT_REPLY_TOKEN)])
    result = await loop._chat_inner("ping")
    assert result["response"] == ""
    assert result.get("silent_reply") is True
    resp = (result.get("response") or "")
    assert not resp.strip().startswith("(")
    assert "no text response" not in resp.lower()
    assert "no response" not in resp.lower()


@pytest.mark.asyncio
async def test_stream_retry_returns_silent_token_emits_no_marker():
    """Bug 3 stream: first no-tool reply empty (blip), retry emits
    ``__SILENT__``. The deliberate retry-silence is honoured: NO text_delta
    marker is emitted (ideally no text_delta at all) and the final ``done``
    event carries an empty ``response``.
    """
    loop = _make_loop([_resp(""), _resp(SILENT_REPLY_TOKEN)])
    # Force the non-streaming fallback so llm.chat (our side_effect) drives it.
    loop.llm.chat_stream = MagicMock(side_effect=RuntimeError("no stream"))
    events = [ev async for ev in loop._chat_stream_inner("ping")]
    text_events = [e for e in events if e.get("type") == "text_delta"]
    assert text_events == []
    assert all(
        "no response" not in (e.get("content") or "").lower() for e in events
    )
    assert all(
        "no text response" not in (e.get("content") or "").lower() for e in events
    )
    done = [e for e in events if e.get("type") == "done"]
    assert done and done[-1].get("response") == ""



# === RC-1 / RC-3: per-task convergence budget (reworked) ===
#
# Handed-off / woken tasks execute via the interactive chat path
# (``loop.chat(task_id=...)`` → ``_chat_inner``), which historically had
# NO per-task budget and NO convergence forcing function — agents could
# over-iterate up to ~CHAT_MAX_TOOL_ROUNDS × session-continues LLM calls,
# only stopped by the mesh-side 900s lane cap.
#
# PRE-ROUND-BOUNDARY mechanism (Codex r3 — 3rd iteration):
#   The loop runs over the NORMAL interactive bound (CHAT_MAX_TOOL_ROUNDS).
#   The per-task cap is a PRE-ROUND boundary: at the TOP of each iteration,
#   BEFORE the LLM call, if the task's persisted round count has reached
#   TASK_MAX_TOOL_ROUNDS the loop breaks and returns immediately. This is the
#   ONLY place ``task_convergence_capped`` is set for tasks.
#   #1 the convergence nudge lives in the SYSTEM PROMPT (the ``system``
#      kwarg of the per-round llm.chat call), NEVER a synthetic ``user``
#      chat message — so role-alternation is preserved.
#   #2 TOOLS REMAIN AVAILABLE on every round (no tool-withhold) — the agent
#      converges by CALLING complete_task / hand_off.
#   #3 the cap closes ``blocked`` ONLY if the agent did NOT converge; a
#      genuine completion (final text, or a coordination-tool close) returns
#      via the no-tool-calls path INSIDE the loop, before the next round's
#      top-of-loop cap check — so the cap can NEVER override it.
#   #4 ``_task_round_counts`` is popped only AFTER a successful non-working
#      status write, and is size-bounded (``_TASK_ROUND_COUNTS_MAX``).
#   #5 default TASK_MAX_TOOL_ROUNDS is 20.
#
# Codex r3 BLOCKERS this suite pins:
#   B1 a terminal complete_task / hand_off / final on the LAST budgeted round
#      closes done/handoff, NOT blocked (the cap never pre-empts it).
#   B2 an exhausted re-wake (count already >= cap) breaks at the first round
#      top with NO additional LLM call and NO tool round.


def _always_tool_calling_loop(*, task_max=4, tool_name="web_search"):
    """An AgentLoop whose mocked LLM ALWAYS returns a tool call (never
    converges on its own). The per-task PRE-ROUND boundary — not the response
    list — must be the binding limit. Tools are offered on EVERY round. Once
    the persisted round count reaches the cap, the top-of-round boundary
    breaks and ``_chat_inner`` returns immediately WITHOUT a force-compose
    LLM call. ``_auto_close_task`` is replaced with an AsyncMock so
    terminal-close calls can be asserted without standing up a real mesh
    client.
    """
    loop = _make_loop()
    loop.TASK_MAX_TOOL_ROUNDS = task_max
    def _llm_side_effect(*args, **kwargs):
        # Only the post-budget force-compose passes tools=None.
        if kwargs.get("tools") is None:
            return LLMResponse(content="Final answer, cut off.", tokens_used=5)
        return LLMResponse(
            content="",
            tool_calls=[ToolCallInfo(name=tool_name, arguments={"q": "x"})],
            tokens_used=10,
        )
    loop.llm.chat = AsyncMock(side_effect=_llm_side_effect)
    loop.tools.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": tool_name}}]
    )
    loop.tools.execute = AsyncMock(return_value={"results": ["r"]})
    loop._auto_close_task = AsyncMock(return_value=None)
    return loop


def test_task_max_tool_rounds_default_matches_limits_table():
    """The per-task budget default is sourced from the central limits table
    (raised to a high default; tune via env / per-agent config)."""
    from src.shared import limits
    loop = _make_loop()
    assert loop.TASK_MAX_TOOL_ROUNDS == limits.LIMIT_SPECS["task_max_tool_rounds"][0]


@pytest.mark.asyncio
async def test_task_never_converges_closes_blocked_not_working():
    """(d) A task whose LLM keeps calling tools and NEVER converges is
    bounded by the per-task budget and terminally closed as ``blocked``
    with the static convergence blocker_note — NOT left ``working`` and
    NOT allowed to run up to CHAT_MAX_TOOL_ROUNDS."""
    loop = _always_tool_calling_loop(task_max=4)

    result = await loop.chat("do the task", task_id="t-conv")

    assert result.get("task_convergence_capped") is True

    # Bounded EXACTLY by the per-task budget: the cap breaks at the top of the
    # round whose count has reached TASK_MAX, so there are exactly task_max
    # tool-round LLM calls and NO force-compose call. Far below the
    # interactive CHAT_MAX_TOOL_ROUNDS (30).
    assert loop.llm.chat.await_count == loop.TASK_MAX_TOOL_ROUNDS
    assert loop.llm.chat.await_count < loop.CHAT_MAX_TOOL_ROUNDS

    # Terminal close: blocked with the static convergence note.
    blocked_calls = [
        c for c in loop._auto_close_task.await_args_list
        if len(c.args) >= 2 and c.args[1] == "blocked"
    ]
    assert blocked_calls, (
        "a task hitting the convergence cap without converging must close "
        f"'blocked', not dangle 'working'. calls: "
        f"{loop._auto_close_task.await_args_list}"
    )
    note = blocked_calls[-1].kwargs.get("blocker_note") or ""
    assert "convergence_cap" in note
    assert blocked_calls[-1].args[0] == "t-conv"


@pytest.mark.asyncio
async def test_convergence_nudge_in_system_prompt_not_chat_messages():
    """(a) The convergence nudge is appended to the SYSTEM PROMPT passed to
    llm.chat (the ``system`` kwarg), NOT injected as a ``user`` chat
    message. Role-alternation is preserved: no user-after-tool and no
    user-after-user message is ever synthesized by the convergence path."""
    loop = _always_tool_calling_loop(task_max=3)

    await loop.chat("go", task_id="t-nudge")

    # No synthetic convergence chat message was ever appended (the old,
    # buggy mechanism). _chat_messages must contain NO system:task_convergence
    # marker and NO injected user nudge text.
    assert not [
        m for m in loop._chat_messages
        if isinstance(m, dict) and m.get("_origin") == "system:task_convergence"
    ], "convergence nudge must NOT be a chat message (role-alternation bug)"

    # The nudge text rode the SYSTEM kwarg of at least one llm.chat call.
    system_strs = [
        (c.kwargs.get("system") or "")
        for c in loop.llm.chat.await_args_list
    ]
    nudged = [s for s in system_strs if "complete_task or hand_off" in s]
    assert nudged, (
        "expected the convergence directive in the system prompt of a "
        f"budgeted round; systems seen: {[s[-80:] for s in system_strs]}"
    )
    # The FINAL-round escalation wording appears near the cap.
    assert any("final round for this task" in s.lower() for s in nudged)

    # Role-alternation invariant: a ``user`` message never directly follows
    # a ``tool`` message or another ``user`` message in the transcript the
    # convergence path produced.
    roles = [m.get("role") for m in loop._chat_messages if isinstance(m, dict)]
    for prev, cur in zip(roles, roles[1:]):
        assert not (cur == "user" and prev in ("tool", "user")), (
            f"role-alternation violated: {prev} → {cur} in {roles}"
        )


@pytest.mark.asyncio
async def test_tools_remain_available_on_every_budgeted_round():
    """(b) Tools are offered on EVERY round — the agent must be able to
    converge by CALLING complete_task / hand_off. With the pre-round-boundary
    redesign the cap path returns at the top of the round WITHOUT any
    tools=None force-compose call, so NO llm.chat call ever withholds tools on
    the task non-convergence path."""
    loop = _always_tool_calling_loop(task_max=3)

    await loop.chat("go", task_id="t-tools")

    tools_none = [
        c for c in loop.llm.chat.await_args_list
        if c.kwargs.get("tools") is None
    ]
    assert tools_none == [], (
        "tools must be offered on every round; the pre-round-boundary cap "
        "returns without a tools=None force-compose, so no call may withhold "
        "tools on the task path"
    )
    assert all(
        c.kwargs.get("tools") is not None for c in loop.llm.chat.await_args_list
    )
    # Exactly task_max tool-round calls — the cap broke at the next round top.
    assert loop.llm.chat.await_count == 3


@pytest.mark.asyncio
async def test_terminal_handoff_on_last_budgeted_round_closes_done_not_blocked():
    """(B1 — Codex blocker #1) A task that calls hand_off (outbound effect)
    and then returns its terminal text on the LAST budgeted round closes
    ``done``/handoff — NOT ``blocked``. Drives the count to cap-1 right at the
    terminal so the OLD shrunk-range bug (fall-through forcing
    ``task_convergence_capped=True``) would have force-blocked it. With the
    pre-round boundary the terminal no-tool-calls reply returns from INSIDE the
    loop, before the next round's top-of-loop cap check can run. Drives the
    REAL ``_auto_close_task``."""
    loop = _make_loop()
    loop.TASK_MAX_TOOL_ROUNDS = 2
    loop.mesh_client.set_task_status = AsyncMock(return_value={"status": "done"})
    # Budget 2. Round idx0 (count 0→1): hand_off (outbound effect). Round idx1
    # (entry count 1 == cap-1, the LAST budgeted round): terminal text →
    # converges via the no-tool-calls branch BEFORE the cap boundary, which
    # would next fire at entry count 2.
    responses = [
        _resp("", tool_calls=[ToolCallInfo(name="hand_off", arguments={"to": "bob"})]),
        _resp("Handed off to bob; done."),
    ]
    loop.llm.chat = AsyncMock(side_effect=responses)
    loop.tools.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "web_search"}}]
    )

    async def _exec(name, *a, **k):
        if name == "hand_off":
            return {"handed_off": True, "to": "bob"}
        return {"results": ["r"]}
    loop.tools.execute = AsyncMock(side_effect=_exec)

    result = await loop.chat("delegate", task_id="t-converge")

    # NOT capped — converged genuinely on the last budgeted round.
    assert result.get("task_convergence_capped") is not True
    statuses = [
        (c.args[1] if len(c.args) >= 2 else c.kwargs.get("status"))
        for c in loop.mesh_client.set_task_status.await_args_list
    ]
    assert "blocked" not in statuses, (
        f"a converging task must NOT be force-closed blocked; statuses={statuses}"
    )
    assert "done" in statuses
    # Per-task state cleared by the terminal done close.
    assert "t-converge" not in loop._task_round_counts


@pytest.mark.asyncio
async def test_terminal_complete_task_on_last_budgeted_round_closes_done():
    """(B1 — Codex blocker #1, complete_task variant) Same protection, with
    ``complete_task`` as the terminal outbound tool on the last budgeted
    round. Closes ``done``, never ``blocked``."""
    loop = _make_loop()
    loop.TASK_MAX_TOOL_ROUNDS = 2
    loop.mesh_client.set_task_status = AsyncMock(return_value={"status": "done"})
    # Round idx0 (count 0→1): complete_task. Round idx1 (entry count 1 ==
    # cap-1, LAST budgeted round): terminal text → no-tool-calls return.
    responses = [
        _resp("", tool_calls=[ToolCallInfo(name="complete_task", arguments={})]),
        _resp("All done."),
    ]
    loop.llm.chat = AsyncMock(side_effect=responses)
    loop.tools.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "complete_task"}}]
    )

    async def _exec(name, *a, **k):
        if name == "complete_task":
            return {"completed": True}
        return {"ok": True}
    loop.tools.execute = AsyncMock(side_effect=_exec)

    result = await loop.chat("finish it", task_id="t-complete")

    assert result.get("task_convergence_capped") is not True
    statuses = [
        (c.args[1] if len(c.args) >= 2 else c.kwargs.get("status"))
        for c in loop.mesh_client.set_task_status.await_args_list
    ]
    assert "blocked" not in statuses
    assert "done" in statuses
    assert "t-complete" not in loop._task_round_counts


@pytest.mark.asyncio
async def test_complete_task_on_exact_cap_round_closes_done_not_blocked():
    """(Item 1 — Codex r4 BLOCKER) The genuine completion lands on the EXACT
    round whose increment reaches the cap, AND the model keeps emitting
    tool_calls (it does NOT volunteer a clean final-text round). Without the
    loop-side early return, the count would increment to the cap, the loop
    would continue, and the NEXT top-of-round cap check would break →
    ``task_convergence_capped`` → chat() closes ``blocked`` — overriding a real
    completion. The fix returns immediately after a successful terminal
    coordination tool so chat() runs its done/handoff close."""
    loop = _make_loop()
    loop.TASK_MAX_TOOL_ROUNDS = 1  # cap reached by the FIRST round's increment
    loop.mesh_client.set_task_status = AsyncMock(return_value={"status": "done"})
    # The model would call complete_task on round idx0 (count 0→1 == cap) and,
    # if allowed to continue, keep tool-calling forever. The early return must
    # fire after round idx0 so this second response is NEVER consumed.
    responses = [
        _resp("", tool_calls=[ToolCallInfo(name="complete_task", arguments={})]),
        _resp("", tool_calls=[ToolCallInfo(name="web_search", arguments={"q": "x"})]),
        _resp("", tool_calls=[ToolCallInfo(name="web_search", arguments={"q": "y"})]),
    ]
    loop.llm.chat = AsyncMock(side_effect=responses)
    loop.tools.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "complete_task"}}]
    )

    async def _exec(name, *a, **k):
        if name == "complete_task":
            return {"completed": True}
        return {"results": ["r"]}
    loop.tools.execute = AsyncMock(side_effect=_exec)

    result = await loop.chat("finish it", task_id="t-exact")

    # The completion wins: NOT capped, closes done, never blocked.
    assert result.get("task_convergence_capped") is not True
    statuses = [
        (c.args[1] if len(c.args) >= 2 else c.kwargs.get("status"))
        for c in loop.mesh_client.set_task_status.await_args_list
    ]
    assert "blocked" not in statuses, (
        f"a terminal complete_task on the cap round must NOT be force-blocked; "
        f"statuses={statuses}"
    )
    assert "done" in statuses
    assert "t-exact" not in loop._task_round_counts
    # Early return fired after exactly ONE LLM round — the extra tool-calling
    # responses were never consumed.
    assert loop.llm.chat.await_count == 1
    # The closing assistant message persisted to durable history must be
    # NON-empty: the Anthropic Messages API rejects empty content on
    # non-final assistant messages, so an empty closer 400s the NEXT turn
    # (chat-bubble suppression is handled via ``silent_reply``, not here).
    # Assistant messages WITH tool_calls may legitimately be empty.
    for msg in loop._chat_messages:
        if msg.get("role") == "assistant" and not msg.get("tool_calls"):
            assert str(msg.get("content", "")).strip(), (
                f"empty persisted assistant message would 400 the next "
                f"Anthropic turn: {msg!r}"
            )


@pytest.mark.asyncio
async def test_handoff_on_exact_cap_round_closes_done_not_blocked():
    """(Item 1 — Codex r4 BLOCKER, hand_off variant) A successful hand_off on
    the exact cap-reaching round closes done/handoff, not blocked."""
    loop = _make_loop()
    loop.TASK_MAX_TOOL_ROUNDS = 1
    loop.mesh_client.set_task_status = AsyncMock(return_value={"status": "done"})
    responses = [
        _resp("", tool_calls=[ToolCallInfo(name="hand_off", arguments={"to": "bob"})]),
        _resp("", tool_calls=[ToolCallInfo(name="web_search", arguments={"q": "x"})]),
    ]
    loop.llm.chat = AsyncMock(side_effect=responses)
    loop.tools.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "hand_off"}}]
    )

    async def _exec(name, *a, **k):
        if name == "hand_off":
            return {"handed_off": True, "to": "bob"}
        return {"results": ["r"]}
    loop.tools.execute = AsyncMock(side_effect=_exec)

    result = await loop.chat("delegate", task_id="t-exact-ho")

    assert result.get("task_convergence_capped") is not True
    statuses = [
        (c.args[1] if len(c.args) >= 2 else c.kwargs.get("status"))
        for c in loop.mesh_client.set_task_status.await_args_list
    ]
    assert "blocked" not in statuses
    assert "done" in statuses
    assert "t-exact-ho" not in loop._task_round_counts
    assert loop.llm.chat.await_count == 1


@pytest.mark.asyncio
async def test_notify_user_on_cap_round_still_blocks_not_completion():
    """(Item 1 — over-broadening guard) A NON-terminal outbound (``notify_user``)
    on the cap round is NOT a completion. It must NOT short-circuit the
    convergence cap: the task still hits the cap and closes ``blocked``.
    Distinguishes ``_last_round_terminal_completion`` from the broader
    ``_has_outbound_effect``."""
    loop = _make_loop()
    loop.TASK_MAX_TOOL_ROUNDS = 1
    loop.mesh_client.set_task_status = AsyncMock(return_value={"status": "blocked"})
    # Round idx0 (count 0→1 == cap): notify_user (outbound, NOT terminal).
    # Then the model keeps tool-calling — the cap must fire on round idx1's top.
    responses = [
        _resp("", tool_calls=[ToolCallInfo(name="notify_user", arguments={"message": "fyi"})]),
        _resp("", tool_calls=[ToolCallInfo(name="web_search", arguments={"q": "x"})]),
    ]
    loop.llm.chat = AsyncMock(side_effect=responses)
    loop.tools.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "notify_user"}}]
    )

    async def _exec(name, *a, **k):
        if name == "notify_user":
            return {"notified": True}
        return {"results": ["r"]}
    loop.tools.execute = AsyncMock(side_effect=_exec)

    result = await loop.chat("work", task_id="t-notify")

    # notify_user is NOT a completion → cap fires → blocked.
    assert result.get("task_convergence_capped") is True
    statuses = [
        (c.args[1] if len(c.args) >= 2 else c.kwargs.get("status"))
        for c in loop.mesh_client.set_task_status.await_args_list
    ]
    assert "blocked" in statuses, (
        f"a non-terminal outbound must NOT count as completion; the task must "
        f"still close blocked at the cap; statuses={statuses}"
    )
    assert "done" not in statuses


def test_last_round_terminal_completion_precision():
    """Unit: the completion detector counts ONLY successful terminal
    coordination tools, never a non-terminal outbound or a FAILED terminal."""
    f = AgentLoop._last_round_terminal_completion
    # Successful terminals.
    assert f([{"tool": "complete_task", "output": {"completed": True}}]) is True
    assert f([{"tool": "hand_off", "output": {"handed_off": True}}]) is True
    # Non-terminal outbound — must NOT count.
    assert f([{"tool": "notify_user", "output": {"notified": True}}]) is False
    assert f([{"tool": "write_blackboard", "output": {"ok": True}}]) is False
    # FAILED terminals — must NOT count (handled as failures elsewhere).
    assert f([{"tool": "hand_off", "output": {"handed_off": False}}]) is False
    assert f([{"tool": "complete_task", "output": {"completed": False}}]) is False
    assert f([{"tool": "complete_task", "output": {"error": "x"}}]) is False
    # Empty / malformed.
    assert f([]) is False
    assert f(None) is False
    assert f([{"tool": "complete_task", "output": "not-a-dict"}]) is False


@pytest.mark.asyncio
async def test_task_max_clamped_to_chat_max():
    """(Item 3 — Codex r4) The effective per-task budget is clamped to
    CHAT_MAX_TOOL_ROUNDS. A misconfigured TASK_MAX > CHAT_MAX must not let a
    task outrun the interactive ceiling and fall through to the
    ``tool_limit_reached`` exit before its per-task cap fires."""
    import os
    from unittest.mock import patch
    with patch.dict(os.environ, {
        "OPENLEGION_CHAT_MAX_TOOL_ROUNDS": "5",
        "OPENLEGION_TASK_MAX_TOOL_ROUNDS": "40",
    }):
        loop = _make_loop()
    assert loop.CHAT_MAX_TOOL_ROUNDS == 5
    assert loop.TASK_MAX_TOOL_ROUNDS == 5, (
        "TASK_MAX must be clamped down to CHAT_MAX when misconfigured higher"
    )


@pytest.mark.asyncio
async def test_round_counts_bound_never_resets_live_task_budget():
    """(Item 4 — Codex r4) When ``_task_round_counts`` is at its size bound, a
    brand-new task is simply NOT tracked — an EXISTING (live) task's count is
    never evicted, so a still-working task can never regain a fresh per-task
    budget. Asserts no live entry is dropped when a new task arrives at bound."""
    loop = _make_loop()
    loop._TASK_ROUND_COUNTS_MAX = 2
    # Pre-seed two live tracked tasks at the bound, with meaningful counts.
    loop._task_round_counts = {"live-a": 7, "live-b": 3}

    async def _exec(name, *a, **k):
        return {"results": ["r"]}
    loop.tools.execute = AsyncMock(side_effect=_exec)
    loop.tools.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "web_search"}}]
    )
    # New task does one tool round then converges with final text.
    loop.llm.chat = AsyncMock(side_effect=[
        _resp("", tool_calls=[ToolCallInfo(name="web_search", arguments={"q": "x"})]),
        _resp("done"),
    ])
    loop.mesh_client.set_task_status = AsyncMock(return_value={"status": "done"})

    await loop.chat("go", task_id="new-task")

    # Neither live task was evicted, and neither had its count reset.
    assert loop._task_round_counts.get("live-a") == 7
    assert loop._task_round_counts.get("live-b") == 3
    # The new task was NOT tracked (table was at bound on its first round).
    assert "new-task" not in loop._task_round_counts


@pytest.mark.asyncio
async def test_exhausted_rewake_blocks_immediately_no_extra_tool_round():
    """(B2 — Codex blocker #2) A re-wake of a task whose persisted round count
    is ALREADY at the cap must break at the FIRST round top and close
    ``blocked`` WITHOUT executing any further LLM call or tool round. The old
    mechanism (``_task_loop_rounds = max(1, 0) = 1``) granted one more
    tool-enabled round on every re-wake — unbounded across re-wakes. The
    pre-round boundary forbids it. Drives the REAL ``_auto_close_task``."""
    loop = _make_loop()
    loop.TASK_MAX_TOOL_ROUNDS = 3
    loop.mesh_client.set_task_status = AsyncMock(return_value={"status": "blocked"})
    # Pre-seed the persisted count AT the cap (a prior wake spent the budget;
    # a 5xx kept the task 'working' so the count survived).
    loop._task_round_counts["t-exhausted"] = 3

    loop.llm.chat = AsyncMock(
        side_effect=AssertionError("LLM must NOT be called on an exhausted re-wake")
    )
    loop.tools.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "web_search"}}]
    )
    loop.tools.execute = AsyncMock(
        side_effect=AssertionError("no tool round may run on an exhausted re-wake")
    )

    result = await loop.chat("continue", task_id="t-exhausted")

    # Capped immediately, with ZERO LLM calls and ZERO tool executions.
    assert result.get("task_convergence_capped") is True
    assert loop.llm.chat.await_count == 0, (
        "an exhausted re-wake must break at the first round top BEFORE the LLM "
        f"call; llm calls={loop.llm.chat.await_count}"
    )
    assert loop.tools.execute.await_count == 0
    # Terminal blocked close with the static convergence note.
    statuses = [
        (c.args[1] if len(c.args) >= 2 else c.kwargs.get("status"))
        for c in loop.mesh_client.set_task_status.await_args_list
    ]
    assert statuses and statuses[-1] == "blocked"


@pytest.mark.asyncio
async def test_task_round_counts_size_bounded_under_many_tasks():
    """(#3 — Codex SHOULD) ``_task_round_counts`` stays bounded even when many
    distinct tasks accumulate counts without their terminal close ever popping
    them (the sustained mesh-write-failure regime). Adding a new task beyond
    ``_TASK_ROUND_COUNTS_MAX`` evicts an existing entry instead of growing
    unbounded. Exercised at the increment site directly via repeated
    single-round task turns whose close is stubbed to a no-op (never pops)."""
    loop = _make_loop()
    loop.TASK_MAX_TOOL_ROUNDS = 5
    loop._TASK_ROUND_COUNTS_MAX = 8     # small bound for the test
    # Stub the terminal close to a no-op so entries are NEVER popped — this is
    # the degenerate sustained-write-failure regime the bound guards.
    loop._auto_close_task = AsyncMock(return_value=None)
    loop.tools.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "web_search"}}]
    )
    loop.tools.execute = AsyncMock(return_value={"results": ["r"]})

    # Each task: one tool round then a final text (converges) — one increment
    # per distinct task_id. The final close is a no-op stub, so nothing pops.
    for i in range(40):
        loop.llm.chat = AsyncMock(side_effect=[
            _resp("", tool_calls=[ToolCallInfo(name="web_search", arguments={})]),
            _resp("done"),
        ])
        loop._chat_messages = []
        await loop.chat("go", task_id=f"task-{i}")
        assert len(loop._task_round_counts) <= loop._TASK_ROUND_COUNTS_MAX, (
            f"_task_round_counts grew past the bound at iteration {i}: "
            f"size={len(loop._task_round_counts)}"
        )

    # Stays at/under the bound after 40 distinct tasks despite no pops.
    assert len(loop._task_round_counts) <= loop._TASK_ROUND_COUNTS_MAX


@pytest.mark.asyncio
async def test_task_converges_via_final_text_on_last_round_closes_done():
    """(c) A task that converges by returning a STRUCTURED final result on
    its last budgeted round closes ``done`` — NOT ``blocked``."""
    loop = _make_loop()
    loop.TASK_MAX_TOOL_ROUNDS = 2
    loop.mesh_client.set_task_status = AsyncMock(return_value={"status": "done"})
    # Round 1: tool. Round 2 (last): structured final result (no tool call) —
    # converges via the no-tool-calls branch, NOT the cap fall-through.
    responses = [
        _resp("", tool_calls=[ToolCallInfo(name="web_search", arguments={})]),
        _resp('{"result": {"answer": "42"}}'),
    ]
    loop.llm.chat = AsyncMock(side_effect=responses)
    loop.tools.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "web_search"}}]
    )
    loop.tools.execute = AsyncMock(return_value={"results": ["r"]})

    result = await loop.chat("answer it", task_id="t-final")

    assert result.get("task_convergence_capped") is not True
    statuses = [
        (c.args[1] if len(c.args) >= 2 else c.kwargs.get("status"))
        for c in loop.mesh_client.set_task_status.await_args_list
    ]
    assert "blocked" not in statuses
    assert "done" in statuses
    assert "t-final" not in loop._task_round_counts


@pytest.mark.asyncio
async def test_task_round_count_persists_across_wakes_tighten_only():
    """(g) The per-task round count PERSISTS across two chat(task_id=X)
    calls (a re-wake). The second wake does NOT get a fresh full budget —
    it hits the cap with fewer rounds than a fresh window would allow."""
    loop = _make_loop()
    loop.TASK_MAX_TOOL_ROUNDS = 6
    loop._auto_close_task = AsyncMock(return_value=None)
    loop.tools.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "web_search"}}]
    )
    loop.tools.execute = AsyncMock(return_value={"results": ["r"]})

    # Simulate 4 of 6 rounds already spent on a prior wake that kept the task
    # 'working' across the wake boundary (a lane followup).
    loop._task_round_counts["t-persist"] = 4

    def _always(*a, **k):
        if k.get("tools") is None:
            return LLMResponse(content="cut off", tokens_used=5)
        return LLMResponse(
            content="",
            tool_calls=[ToolCallInfo(name="web_search", arguments={})],
            tokens_used=10,
        )
    loop.llm.chat = AsyncMock(side_effect=_always)
    result = await loop.chat("continue", task_id="t-persist")

    assert result.get("task_convergence_capped") is True
    # Only 2 rounds remained (6 - 4) → exactly 2 tool rounds (counts 5, 6),
    # then the next round top hits the cap and breaks (no force-compose).
    # NOT a fresh full budget of 6.
    assert loop.llm.chat.await_count == 2, (
        "second wake must resume from the persisted per-task count, not get "
        f"a fresh full budget; llm calls={loop.llm.chat.await_count}"
    )


@pytest.mark.asyncio
async def test_task_count_not_popped_when_status_write_raises_5xx():
    """(e) When the terminal status write FAILS with a 5xx (mesh unhealthy),
    ``_task_round_counts`` is NOT popped — the task is still 'working' on the
    mesh and must keep its accumulated budget so a re-wake doesn't get a
    fresh window. Drives the REAL ``_auto_close_task``."""
    import httpx

    loop = _make_loop()
    loop._task_round_counts["t-5xx"] = 5

    # set_task_status raises a 5xx HTTPStatusError.
    req = httpx.Request("PUT", "http://mesh/tasks/t-5xx/status")
    resp = httpx.Response(503, request=req)
    err = httpx.HTTPStatusError("server error", request=req, response=resp)
    loop.mesh_client.set_task_status = AsyncMock(side_effect=err)

    await loop._auto_close_task("t-5xx", "blocked", blocker_note="x")

    assert "t-5xx" in loop._task_round_counts, (
        "a 5xx write failure must KEEP the per-task count (task still working "
        "on the mesh); popping it would hand a re-wake a fresh budget"
    )
    assert loop._task_round_counts["t-5xx"] == 5


@pytest.mark.asyncio
async def test_task_count_cleared_on_successful_terminal_close():
    """(e) A successful non-working status write clears the per-task count.
    A 4xx (already-terminal race) also clears it. Drives the REAL
    ``_auto_close_task``."""
    import httpx

    loop = _make_loop()

    # Successful terminal close → cleared.
    loop._task_round_counts["t-ok"] = 3
    loop.mesh_client.set_task_status = AsyncMock(return_value={"status": "done"})
    await loop._auto_close_task("t-ok", "done", result_payload={"summary": "s"})
    assert "t-ok" not in loop._task_round_counts

    # A 'working' (non-terminal) open transition must NOT clear it.
    loop._task_round_counts["t-work"] = 2
    await loop._auto_close_task("t-work", "working")
    assert loop._task_round_counts.get("t-work") == 2

    # 4xx already-terminal race → cleared (involvement over).
    loop._task_round_counts["t-4xx"] = 4
    req = httpx.Request("PUT", "http://mesh/tasks/t-4xx/status")
    resp = httpx.Response(409, request=req)
    err = httpx.HTTPStatusError("conflict", request=req, response=resp)
    loop.mesh_client.set_task_status = AsyncMock(side_effect=err)
    await loop._auto_close_task("t-4xx", "done")
    assert "t-4xx" not in loop._task_round_counts


@pytest.mark.asyncio
async def test_interactive_chat_no_task_id_completely_unaffected():
    """(f) Interactive chat with NO task_id is COMPLETELY unaffected: no
    per-task cap, no convergence nudge in the system prompt, no tool-withhold
    from the task path. It runs the full interactive CHAT_MAX_TOOL_ROUNDS
    path."""
    loop = _make_loop()
    loop.TASK_MAX_TOOL_ROUNDS = 2          # tiny — would cap a task hard
    loop.CHAT_MAX_TOOL_ROUNDS = 5          # bound the interactive loop
    loop._auto_close_task = AsyncMock(return_value=None)
    loop.tools.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "web_search"}}]
    )
    loop.tools.execute = AsyncMock(return_value={"results": ["r"]})

    def _always(*a, **k):
        if k.get("tools") is None:
            return LLMResponse(content="done", tokens_used=5)
        return LLMResponse(
            content="",
            tool_calls=[ToolCallInfo(name="web_search", arguments={})],
            tokens_used=10,
        )
    loop.llm.chat = AsyncMock(side_effect=_always)

    # NO task_id — pure interactive chat.
    result = await loop.chat("just chatting")

    # Ran MORE tool rounds than the tiny TASK_MAX would have permitted.
    assert loop.llm.chat.await_count > loop.TASK_MAX_TOOL_ROUNDS + 1

    # No convergence directive in ANY system prompt.
    assert not [
        (c.kwargs.get("system") or "")
        for c in loop.llm.chat.await_args_list
        if "complete_task or hand_off" in (c.kwargs.get("system") or "")
    ], "interactive chat must NOT get the per-task convergence directive"

    # No convergence-cap marker, no per-task state, no blocked close.
    assert result.get("task_convergence_capped") is not True
    assert loop._task_round_counts == {}
    assert not [
        c for c in loop._auto_close_task.await_args_list
        if len(c.args) >= 2 and c.args[1] == "blocked"
    ]


def test_trim_context_preserves_multimodal_first_message() -> None:
    """Regression: when the initial user message is multimodal (list content,
    e.g. an uploaded image enriched by _build_initial_context), trimming must
    fold the summary into that message as a trailing text block — not append a
    second consecutive user message, which would violate the LLM
    role-alternation invariant (Constraint #7)."""
    loop = _make_loop()
    img_block = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    messages: list[dict] = [
        {"role": "user", "content": [{"type": "text", "text": "Review this"}, img_block]},
    ]
    # >3 tool-call groups so trimming engages and the middle groups summarize.
    for i in range(4):
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": f"c{i}", "type": "function",
                "function": {"name": "noop", "arguments": "{}"},
            }],
        })
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"result {i}"})

    trimmed = loop._trim_context(messages, max_tokens=5)

    # No two consecutive user-role messages anywhere in the result.
    roles = [m["role"] for m in trimmed]
    assert not any(
        roles[i] == "user" and roles[i + 1] == "user" for i in range(len(roles) - 1)
    ), f"consecutive user messages: {roles}"
    # First message stays the multimodal user turn with the image intact.
    assert trimmed[0]["role"] == "user"
    assert isinstance(trimmed[0]["content"], list)
    assert img_block in trimmed[0]["content"]
    # The summary was folded in as a trailing text block.
    assert any(
        isinstance(b, dict) and b.get("type") == "text"
        and "Previous Actions" in b.get("text", "")
        for b in trimmed[0]["content"]
    )


# === Context-window wedge guard (defense in depth) ===


@pytest.mark.asyncio
async def test_restore_session_prunes_oversized_checkpoint():
    """A fat checkpoint reloaded by _maybe_restore_session must be pruned under
    the window so the agent can't re-wedge on the first call after restart."""
    from src.agent.context import ContextManager

    loop = _make_loop()
    loop.context_manager = ContextManager(max_tokens=8_000, model="anthropic/claude")
    loop._chat_messages = []

    # Build a checkpoint far over the window: many big tool-call groups.
    fat: list[dict] = [{"role": "user", "content": "INITIAL " + "i" * 4000}]
    for g in range(10):
        fat.append({
            "role": "assistant", "content": "",
            "tool_calls": [{"id": f"c{g}", "function": {"name": "do", "arguments": "{}"}}],
        })
        fat.append({"role": "tool", "tool_call_id": f"c{g}", "content": "r" * 4000})
        fat.append({"role": "assistant", "content": "a" * 4000})
        fat.append({"role": "user", "content": "u" * 4000})

    before = loop.context_manager.estimate_request_tokens(fat)
    assert before > loop.context_manager.max_tokens

    loop.memory._run_db = AsyncMock(return_value={
        "messages": fat,
        "total_rounds": 3,
        "auto_continues": 0,
        "flush_triggered": False,
    })

    await loop._maybe_restore_session()

    after = loop.context_manager.estimate_request_tokens(loop._chat_messages)
    assert after <= loop.context_manager.max_tokens * 0.90
    # First (initial) message preserved.
    assert loop._chat_messages[0]["content"].startswith("INITIAL")


@pytest.mark.asyncio
async def test_chat_self_heals_on_context_overflow():
    """A context-overflow on the first chat call triggers an emergency prune +
    successful retry rather than aborting the turn."""
    from src.agent.context import ContextManager
    from src.agent.llm import LLMContextOverflowError

    final = LLMResponse(content="recovered answer", tokens_used=20)

    loop = _make_loop()
    loop.context_manager = ContextManager(max_tokens=8_000, model="anthropic/claude")

    # Oversized inherited context so prune_to_fit has groups to shed.
    seed: list[dict] = [{"role": "user", "content": "INITIAL " + "i" * 4000}]
    for g in range(10):
        seed.append({
            "role": "assistant", "content": "",
            "tool_calls": [{"id": f"c{g}", "function": {"name": "do", "arguments": "{}"}}],
        })
        seed.append({"role": "tool", "tool_call_id": f"c{g}", "content": "r" * 4000})
        seed.append({"role": "assistant", "content": "a" * 4000})
        seed.append({"role": "user", "content": "u" * 4000})
    loop._chat_messages = list(seed)

    # First call overflows, second returns a normal response.
    loop.llm.chat_collect = AsyncMock(
        side_effect=[LLMContextOverflowError("prompt is too long: 9000 > 8000"), final]
    )
    loop.tools.get_tool_definitions = MagicMock(return_value=[])

    resp = await loop._chat_call_self_healing(system="sys", tools=None)
    assert resp is final
    assert loop.llm.chat_collect.call_count == 2
    # The context was pruned in the process.
    assert len(loop._chat_messages) < len(seed)


@pytest.mark.asyncio
async def test_chat_self_heal_reraises_when_prune_cannot_help():
    """If pruning makes no progress (already minimal), the overflow is
    re-raised so the user sees a clear failure rather than an infinite loop."""
    from src.agent.context import ContextManager
    from src.agent.llm import LLMContextOverflowError

    loop = _make_loop()
    loop.context_manager = ContextManager(max_tokens=8_000, model="anthropic/claude")
    # Single small message: nothing to prune.
    loop._chat_messages = [{"role": "user", "content": "hi"}]
    loop.llm.chat_collect = AsyncMock(
        side_effect=LLMContextOverflowError("prompt is too long")
    )
    loop.tools.get_tool_definitions = MagicMock(return_value=[])

    with pytest.raises(LLMContextOverflowError):
        await loop._chat_call_self_healing(system="sys", tools=None)


@pytest.mark.asyncio
async def test_apply_task_thinking_sets_validates_and_survives_errors():
    """B4: the loop pins the task's thinking level as the LLM override,
    ignores invalid levels, and treats lookup failures as best-effort."""
    loop = _make_loop()
    loop.llm.VALID_THINKING_LEVELS = {"off", "low", "medium", "high"}
    loop.llm.thinking_override = None

    loop.mesh_client.get_task = AsyncMock(
        return_value={"id": "t1", "thinking": "high"},
    )
    await loop._apply_task_thinking("t1")
    assert loop.llm.thinking_override == "high"

    loop.llm.thinking_override = None
    loop.mesh_client.get_task = AsyncMock(
        return_value={"id": "t2", "thinking": "bogus"},
    )
    await loop._apply_task_thinking("t2")
    assert loop.llm.thinking_override is None

    loop.mesh_client.get_task = AsyncMock(side_effect=RuntimeError("down"))
    await loop._apply_task_thinking("t3")
    assert loop.llm.thinking_override is None

    loop.mesh_client.get_task = AsyncMock(return_value=None)
    await loop._apply_task_thinking("t4")
    assert loop.llm.thinking_override is None


# ── system_note: honest transcript roles for mesh-composed messages ──


class TestSystemNoteTranscriptRole:
    """A system wake (``system_note=True``) persists to the transcript with
    role ``system`` — never as the user's own bubble. The in-memory LLM
    message stays role ``user`` (model API requirement)."""

    @pytest.mark.asyncio
    async def test_system_note_chat_persists_system_role(self, tmp_path):
        from src.agent.workspace import WorkspaceManager

        loop = _make_loop([LLMResponse(content="verified", tokens_used=10)])
        loop.workspace = WorkspaceManager(workspace_dir=str(tmp_path))
        loop.tools.get_tool_definitions = MagicMock(return_value=[])

        await loop.chat("WAKE-MARKER chain completed, verify it", system_note=True)

        rows = loop.workspace.load_chat_transcript()
        inbound = [r for r in rows if "WAKE-MARKER" in (r.get("content") or "")]
        assert inbound, "wake message missing from transcript"
        assert inbound[0]["role"] == "system"
        assert not [
            r for r in rows
            if r["role"] == "user" and "WAKE-MARKER" in (r.get("content") or "")
        ], "wake message must not persist as a user row"
        # LLM still saw it as a user message (API requires the role).
        assert loop._chat_messages[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_default_chat_persists_user_role(self, tmp_path):
        from src.agent.workspace import WorkspaceManager

        loop = _make_loop([LLMResponse(content="hi", tokens_used=10)])
        loop.workspace = WorkspaceManager(workspace_dir=str(tmp_path))
        loop.tools.get_tool_definitions = MagicMock(return_value=[])

        await loop.chat("USER-MARKER hello")

        rows = loop.workspace.load_chat_transcript()
        inbound = [r for r in rows if "USER-MARKER" in (r.get("content") or "")]
        assert inbound and inbound[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_system_note_skips_correction_recording(self, tmp_path):
        """Wake boilerplate must not pollute the corrections store even when
        it pattern-matches the correction detector."""
        from src.agent.workspace import WorkspaceManager

        loop = _make_loop([LLMResponse(content="ok", tokens_used=10)])
        loop.workspace = WorkspaceManager(workspace_dir=str(tmp_path))
        loop.tools.get_tool_definitions = MagicMock(return_value=[])
        loop._chat_messages = [{"role": "assistant", "content": "prior reply"}]
        loop.workspace.looks_like_correction = MagicMock(return_value=True)
        loop.workspace.record_correction = MagicMock()

        await loop.chat("no, that's wrong — do NOT re-announce", system_note=True)

        loop.workspace.looks_like_correction.assert_not_called()
        loop.workspace.record_correction.assert_not_called()

    @pytest.mark.asyncio
    async def test_system_note_skips_memory_auto_search(self, tmp_path):
        """First-message memory auto-search is boilerplate pollution for a
        wake — skipped when system_note is set, kept for real users."""
        from src.agent.workspace import WorkspaceManager

        loop = _make_loop([LLMResponse(content="ok", tokens_used=10)])
        loop.workspace = WorkspaceManager(workspace_dir=str(tmp_path))
        loop.tools.get_tool_definitions = MagicMock(return_value=[])
        loop.workspace.search = MagicMock(return_value=[])

        await loop.chat("wake text", system_note=True)
        loop.workspace.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_busy_system_wake_queues_flagged_tuple(self):
        """A system wake hitting a busy agent (current_task set) rides the
        steer queue WITH its flag — the drain must not render it as a
        '[steer]' user bubble."""
        loop = _make_loop()
        loop.current_task = "task_in_flight"

        result = await loop.chat("wake while busy", system_note=True)

        assert "queued" in result["response"].lower()
        assert loop._drain_steer_messages() == [("wake while busy", True)]

    @pytest.mark.asyncio
    async def test_inject_steer_round_trips_flag(self):
        loop = _make_loop()
        await loop.inject_steer("watch update", system_note=True)
        await loop.inject_steer("human ping")
        assert loop._drain_steer_messages() == [
            ("watch update", True), ("human ping", False),
        ]

    def test_persist_steer_entries_uses_honest_roles(self):
        loop = _make_loop()
        loop.workspace = MagicMock()
        loop._persist_steer_entries([("sys note", True), ("human", False)])
        calls = loop.workspace.append_chat_message.call_args_list
        assert calls[0].args == ("system", "sys note")
        assert calls[1].args == ("user", "[steer] human")

    def test_steer_interjection_labels_by_provenance(self):
        from src.agent.loop import AgentLoop
        out = AgentLoop._steer_interjection([("a", True), ("b", False)])
        assert "[System]: a" in out
        assert "[User interjection]: b" in out
