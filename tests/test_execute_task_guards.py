"""Tests for the Bug 5 defensive guards in ``AgentLoop.execute_task``.

Three guards:

1. Stale-checkpoint rejection: a checkpoint whose ``iteration`` is at or
   past ``MAX_ITERATIONS`` would otherwise fall through to the
   max-iterations branch immediately, producing a 0-token "completion".
   The guard clears the checkpoint and starts fresh.

2. Pathological-success rejection: if the final-response branch is hit
   with both ``iterations_executed == 0`` and ``total_tokens == 0`` we
   downgrade to ``failed`` with a clear error rather than reporting a
   ghost-success that masks corruption.

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

    skills = MagicMock()
    skills.get_tool_definitions = MagicMock(return_value=[])
    skills.get_descriptions = MagicMock(return_value="- no tools")
    skills.list_skills = MagicMock(return_value=[])
    skills.is_parallel_safe = MagicMock(return_value=True)
    skills.get_loop_exempt_tools = MagicMock(return_value=frozenset())

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
        skills=skills,
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


def test_pathological_guard_present_in_source():
    """Belt-and-suspenders guard: the final-response branch will downgrade
    to ``failed`` when both ``iterations_executed`` and ``total_tokens``
    are zero — a combination that can't arise from a healthy run
    (iterations_executed is always at least 1 by the time we reach the
    success branch). Cannot be exercised via the normal loop path; we
    pin the guard's presence and contract in source to prevent silent
    removal during future refactors."""
    from pathlib import Path

    loop_src = Path("src/agent/loop.py").read_text()
    # The guard's loud-log line, the operator-facing error message,
    # and the corruption hint must all be present. The error string is
    # split across source lines so we look for prefix + suffix.
    assert "pathological success guard tripped" in loop_src
    assert "execute_task returned success without doing" in loop_src
    assert "checkpoint corruption" in loop_src
    # And the guard must downgrade to failed, not silently log.
    assert 'status="failed"' in loop_src


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
