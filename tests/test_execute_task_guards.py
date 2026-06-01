"""Tests for the Bug 5 defensive guards in ``AgentLoop.execute_task``.

Three guards:

1. Stale-checkpoint rejection: a checkpoint whose ``iteration`` is at or
   past ``MAX_ITERATIONS`` would otherwise fall through to the
   max-iterations branch immediately, producing a 0-token "completion".
   The guard clears the checkpoint and starts fresh.

2. Pathological-success rejection: if the final-response branch is hit
   on the first iteration with ``total_tokens == 0`` AND empty content,
   downgrade to ``failed`` with a clear error rather than reporting a
   ghost-success (mocked no-op LLM, checkpoint resurrection of a
   finished state, double-completion race). Empty content is required
   because some providers (Ollama, proxies that strip usage) report
   ``tokens_used=0`` for legitimate completions.

3. Branch-exit logging: every ``return TaskResult(...)`` in
   ``execute_task`` is preceded by a structured ``logger.info`` so a
   future ghost-complete is debuggable from logs alone.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.loop import AgentLoop
from src.agent.memory import MemoryStore
from src.shared.types import LLMResponse, TaskAssignment


def _make_loop(
    llm_responses: list[LLMResponse] | None = None,
    *,
    real_memory: bool = False,
) -> AgentLoop:
    """Minimal AgentLoop tailored to the guard tests."""
    if real_memory:
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

    llm = MagicMock()
    if llm_responses:
        llm.chat = AsyncMock(side_effect=llm_responses)
    else:
        llm.chat = AsyncMock(return_value=LLMResponse(
            content='{"result": {"answer": "ok"}}', tokens_used=42,
        ))
    llm.default_model = "test-model"

    mesh_client = MagicMock()
    mesh_client.is_standalone = False
    mesh_client.send_system_message = AsyncMock(return_value={})
    mesh_client.read_blackboard = AsyncMock(return_value=None)
    mesh_client.list_agents = AsyncMock(return_value={})

    return AgentLoop(
        agent_id="test_agent",
        role="research",
        memory=memory,
        tools=tools,
        llm=llm,
        mesh_client=mesh_client,
    )


# ── Guard 1: stale checkpoint ────────────────────────────────────


@pytest.mark.asyncio
async def test_stale_checkpoint_at_max_iterations_starts_fresh():
    """A checkpoint with iteration==MAX-1 implies start_iteration==MAX,
    which the for-loop would skip entirely. Guard must clear it and run
    a fresh task to completion."""
    loop = _make_loop(real_memory=True)
    task_id = "task_stale_cp"

    # Pre-populate a checkpoint at the maximum iteration value.
    assignment = TaskAssignment(
        task_id=task_id, workflow_id="wf_test", step_id="step_1",
        task_type="research", input_data={"query": "x"},
    )

    def _save():
        loop.memory.save_task_checkpoint(
            task_id=task_id,
            messages=[{"role": "user", "content": "old"}],
            iteration=loop.MAX_ITERATIONS - 1,  # start_iteration = MAX
            tokens_used=0,
            assignment_json=assignment.model_dump_json(),
            budget_used_tokens=0,
            budget_estimated_cost=0.0,
            flush_triggered=False,
        )

    await loop.memory._run_db(_save)

    # Sanity: checkpoint really is there.
    pre = await loop.memory._run_db(loop.memory.load_task_checkpoint)
    assert pre is not None
    assert pre["iteration"] == loop.MAX_ITERATIONS - 1

    # Run — the guard should discard and start fresh, letting the LLM
    # respond with a final answer on iteration 0.
    result = await loop.execute_task(assignment)

    # If the guard didn't fire we'd hit "Max iterations reached" with
    # 0 tokens and 0 iterations. Instead we should get a real completion.
    assert result.status == "complete", (
        f"expected complete, got {result.status} (error={result.error})"
    )
    assert result.tokens_used > 0


# ── Guard 2: pathological success rejection ──────────────────────


@pytest.mark.asyncio
async def test_pathological_zero_token_empty_content_downgraded_to_failed():
    """A first-iteration LLM response with zero tokens AND empty content
    is the ghost-completion smoking gun — guard must downgrade success
    to failed. Empty content is required because some providers (Ollama,
    proxies that strip usage) legitimately report tokens_used=0 for real
    answers; content + tokens together is the unambiguous signal."""
    ghost = LLMResponse(content="", tokens_used=0)
    loop = _make_loop(llm_responses=[ghost])

    assignment = TaskAssignment(
        task_id="task_ghost", workflow_id="wf", step_id="s1",
        task_type="research", input_data={"q": "anything"},
    )
    result = await loop.execute_task(assignment)

    assert result.status == "failed"
    assert "without doing work" in (result.error or "")


@pytest.mark.asyncio
async def test_pathological_zero_token_empty_content_logs_guard_branch(caplog):
    """The guard should emit the pathological_success_guard branch-exit log,
    not the normal final_response label."""
    ghost = LLMResponse(content="   ", tokens_used=0)  # whitespace-only also empty
    loop = _make_loop(llm_responses=[ghost])

    assignment = TaskAssignment(
        task_id="task_ghost_log", workflow_id="wf", step_id="s1",
        task_type="research", input_data={"q": "x"},
    )

    with caplog.at_level(logging.INFO, logger="agent.loop"):
        result = await loop.execute_task(assignment)

    assert result.status == "failed"
    branch_lines = [
        r.message for r in caplog.records
        if r.name == "agent.loop"
        and "execute_task exit branch=" in r.message
    ]
    assert any("branch=pathological_success_guard" in m for m in branch_lines)


@pytest.mark.asyncio
async def test_first_iteration_completion_with_tokens_stays_complete():
    """Zero-token guard must NOT fire when the LLM actually did work."""
    real = LLMResponse(content='{"result": {"answer": "real"}}', tokens_used=42)
    loop = _make_loop(llm_responses=[real])
    assignment = TaskAssignment(
        task_id="task_real", workflow_id="wf", step_id="s1",
        task_type="research", input_data={"q": "real"},
    )
    result = await loop.execute_task(assignment)
    assert result.status == "complete"
    assert result.tokens_used == 42


@pytest.mark.asyncio
async def test_lazy_guard_conservative_seed_on_resume_after_compaction():
    """Bug F (codex r5): a checkpoint can be saved AFTER ``maybe_compact``
    stripped the assistant-with-tool_calls entries from the message
    tail. On resume, seeding ``tool_calls_count`` from the compacted
    messages reads as 0 even though tools were called pre-checkpoint —
    the original undercount bug. The fix conservatively sets the seed
    to 1 when ``start_iteration > 0 AND tool_calls_count == 0``, so a
    legitimately-busy task that resumes from a compacted checkpoint
    does NOT trip the lazy-completion guard on its post-resume reply.
    """
    # The resumed LLM call returns a plain text final answer — would
    # normally trip the guard (text-only, zero tool_calls visible in
    # the compacted messages). The conservative seed must let it pass.
    resumed_reply = LLMResponse(content="Done.", tokens_used=15)
    loop = _make_loop(llm_responses=[resumed_reply], real_memory=True)
    task_id = "task_compacted_cp"
    assignment = TaskAssignment(
        task_id=task_id, workflow_id="wf", step_id="s1",
        task_type="research", input_data={"q": "x"},
    )

    # Persist a checkpoint at iteration=3 with messages that contain NO
    # assistant-with-tool_calls entries (simulating compaction stripping
    # them out). Only user + tool-response messages remain.
    compacted_messages = [
        {"role": "user", "content": "original task brief"},
        # NOTE: the assistant-with-tool_calls record that would have
        # carried `"tool_calls": [...]` has been compacted away. Only
        # the user message survives in the worst-case tail.
    ]

    def _save():
        loop.memory.save_task_checkpoint(
            task_id=task_id,
            messages=compacted_messages,
            iteration=3,  # start_iteration will be 4 — well past the guard's threshold
            tokens_used=120,
            assignment_json=assignment.model_dump_json(),
            budget_used_tokens=120,
            budget_estimated_cost=0.0,
            flush_triggered=False,
        )

    await loop.memory._run_db(_save)

    # Give the agent tools so the nudge would normally fire — but the
    # nudge only fires at iteration 0, and we're resuming at iteration 4.
    loop.tools.get_tool_definitions = MagicMock(
        return_value=[{"type": "function", "function": {"name": "web_search"}}]
    )

    result = await loop.execute_task(assignment)

    # The lazy-completion guard MUST NOT trip — the conservative seed
    # remembered that tools were called pre-compaction.
    assert result.status == "complete", (
        f"expected complete (conservative seed should suppress guard), got "
        f"status={result.status} error={result.error}"
    )
    assert loop.tasks_completed == 1
    assert loop.tasks_failed == 0


@pytest.mark.asyncio
async def test_first_iteration_completion_with_content_but_zero_tokens_stays_complete():
    """Codex P2 regression guard: a provider that omits usage metadata
    (e.g. Ollama, some proxies) returns tokens_used=0 for legitimate
    completions. As long as content is present, the guard must NOT fire."""
    legit = LLMResponse(content='{"result": {"answer": "real"}}', tokens_used=0)
    loop = _make_loop(llm_responses=[legit])
    assignment = TaskAssignment(
        task_id="task_no_usage", workflow_id="wf", step_id="s1",
        task_type="research", input_data={"q": "ok"},
    )
    result = await loop.execute_task(assignment)
    assert result.status == "complete", (
        f"providers without usage metadata must not be flagged; got {result.status} "
        f"(error={result.error})"
    )


# ── Guard 3: branch-exit logging ─────────────────────────────────


@pytest.mark.asyncio
async def test_branch_exit_logging_fires_for_final_response(caplog):
    """A normal completion logs an ``execute_task exit branch=...`` line
    so future ghost-completes are diagnosable from logs."""
    final = LLMResponse(content='{"result": {"answer": "ok"}}', tokens_used=42)
    loop = _make_loop(llm_responses=[final])

    assignment = TaskAssignment(
        task_id="task_logger_final", workflow_id="wf_log", step_id="s1",
        task_type="research", input_data={"q": "hello"},
    )

    with caplog.at_level(logging.INFO, logger="agent.loop"):
        result = await loop.execute_task(assignment)

    assert result.status == "complete"
    branch_lines = [
        r.message for r in caplog.records
        if r.name == "agent.loop"
        and "execute_task exit branch=" in r.message
    ]
    assert branch_lines, (
        f"expected branch-exit log, got: "
        f"{[r.message for r in caplog.records if r.name == 'agent.loop']}"
    )
    # The matching branch label should mention final_response.
    assert any("branch=final_response" in m for m in branch_lines)


@pytest.mark.asyncio
async def test_branch_exit_logging_fires_for_exception(caplog):
    """An exception path also produces an exit-branch log so we can
    distinguish unrelated raises from the success path."""
    # Force an LLM error mid-flight.
    llm_resp = AsyncMock(side_effect=RuntimeError("llm exploded"))
    loop = _make_loop()
    loop.llm.chat = llm_resp

    assignment = TaskAssignment(
        task_id="task_logger_err", workflow_id="wf_log", step_id="s1",
        task_type="research", input_data={"q": "x"},
    )

    with caplog.at_level(logging.INFO, logger="agent.loop"):
        result = await loop.execute_task(assignment)

    assert result.status == "failed"
    branch_lines = [
        r.message for r in caplog.records
        if r.name == "agent.loop"
        and "execute_task exit branch=" in r.message
    ]
    assert any("branch=exception" in m for m in branch_lines)
