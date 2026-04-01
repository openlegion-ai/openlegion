"""Tests for task execution checkpointing (crash recovery).

Tests the memory-layer persistence (save/load/clear) and the loop-layer
integration (checkpoint after iterations, resume on restart, cleanup on exit).
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.memory import MemoryStore, _TASK_CHECKPOINT_VERSION
from src.agent.loop import AgentLoop
from src.shared.types import LLMResponse, TaskAssignment, TaskResult, TokenBudget, ToolCallInfo


# ── Helpers ──────────────────────────────────────────────────────


def _make_loop(llm_responses: list[LLMResponse] | None = None, *, real_memory: bool = False) -> AgentLoop:
    """Create an AgentLoop with mock dependencies.

    If real_memory=True, uses a real in-memory MemoryStore instead of a mock
    (needed for tests that exercise checkpoint storage/retrieval).
    """
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
        llm.chat = AsyncMock(return_value=LLMResponse(content='{"result": {"answer": "42"}}', tokens_used=100))
    llm.default_model = "test-model"

    mesh_client = MagicMock()
    mesh_client.is_standalone = False
    mesh_client.send_system_message = AsyncMock(return_value={})
    mesh_client.read_blackboard = AsyncMock(return_value=None)
    mesh_client.list_agents = AsyncMock(return_value={})

    loop = AgentLoop(
        agent_id="test_agent",
        role="research",
        memory=memory,
        skills=skills,
        llm=llm,
        mesh_client=mesh_client,
    )
    return loop


# ── Memory layer tests ───────────────────────────────────────────


class TestTaskCheckpointMemory:
    """Tests for MemoryStore task checkpoint persistence."""

    def _store(self) -> MemoryStore:
        return MemoryStore(db_path=":memory:", embed_fn=None)

    def test_save_load_task_checkpoint(self):
        """Save, load, verify all fields round-trip."""
        store = self._store()
        messages = [
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": "ok", "tool_calls": []},
        ]
        store.save_task_checkpoint(
            task_id="task_123",
            assignment_json='{"task_id": "task_123"}',
            messages=messages,
            iteration=3,
            tokens_used=5000,
            budget_used_tokens=4000,
            budget_estimated_cost=0.05,
            flush_triggered=True,
        )
        cp = store.load_task_checkpoint()
        assert cp is not None
        assert cp["task_id"] == "task_123"
        assert cp["assignment_json"] == '{"task_id": "task_123"}'
        assert cp["messages"] == messages
        assert cp["iteration"] == 3
        assert cp["tokens_used"] == 5000
        assert cp["budget_used_tokens"] == 4000
        assert cp["budget_estimated_cost"] == 0.05
        assert cp["flush_triggered"] is True

    def test_save_overwrites_task_checkpoint(self):
        """Save twice, load gets latest."""
        store = self._store()
        store.save_task_checkpoint(
            task_id="task_1",
            assignment_json="{}",
            messages=[{"role": "user", "content": "first"}],
            iteration=0,
            tokens_used=100,
            budget_used_tokens=0,
            budget_estimated_cost=0.0,
            flush_triggered=False,
        )
        store.save_task_checkpoint(
            task_id="task_2",
            assignment_json='{"new": true}',
            messages=[{"role": "user", "content": "second"}],
            iteration=5,
            tokens_used=9000,
            budget_used_tokens=8000,
            budget_estimated_cost=1.23,
            flush_triggered=True,
        )
        cp = store.load_task_checkpoint()
        assert cp is not None
        assert cp["task_id"] == "task_2"
        assert cp["iteration"] == 5
        assert cp["tokens_used"] == 9000
        assert cp["flush_triggered"] is True

    def test_clear_task_checkpoint(self):
        """Save, clear, load returns None."""
        store = self._store()
        store.save_task_checkpoint(
            task_id="task_1",
            assignment_json="{}",
            messages=[],
            iteration=0,
            tokens_used=0,
            budget_used_tokens=0,
            budget_estimated_cost=0.0,
            flush_triggered=False,
        )
        store.clear_task_checkpoint()
        assert store.load_task_checkpoint() is None

    def test_load_task_checkpoint_empty(self):
        """Fresh DB, load returns None."""
        store = self._store()
        assert store.load_task_checkpoint() is None

    def test_load_task_checkpoint_version_mismatch(self):
        """Manually set wrong version, load returns None and clears."""
        store = self._store()
        store.save_task_checkpoint(
            task_id="task_1",
            assignment_json="{}",
            messages=[{"role": "user", "content": "hi"}],
            iteration=2,
            tokens_used=500,
            budget_used_tokens=0,
            budget_estimated_cost=0.0,
            flush_triggered=False,
        )
        # Corrupt the version
        store.db.execute("UPDATE task_checkpoint SET version = 999 WHERE id = 1")
        store.db.commit()
        assert store.load_task_checkpoint() is None
        # Verify it was cleared
        row = store.db.execute("SELECT COUNT(*) FROM task_checkpoint").fetchone()
        assert row[0] == 0

    def test_load_task_checkpoint_corrupt_json(self):
        """Manually corrupt messages column, load returns None and clears."""
        store = self._store()
        store.save_task_checkpoint(
            task_id="task_1",
            assignment_json="{}",
            messages=[{"role": "user", "content": "hi"}],
            iteration=1,
            tokens_used=200,
            budget_used_tokens=0,
            budget_estimated_cost=0.0,
            flush_triggered=False,
        )
        # Corrupt messages JSON
        store.db.execute("UPDATE task_checkpoint SET messages = 'NOT_VALID_JSON{{{' WHERE id = 1")
        store.db.commit()
        assert store.load_task_checkpoint() is None
        # Verify it was cleared
        row = store.db.execute("SELECT COUNT(*) FROM task_checkpoint").fetchone()
        assert row[0] == 0


# ── Loop layer tests ─────────────────────────────────────────────


class TestTaskCheckpointLoop:
    """Tests for AgentLoop task checkpoint integration."""

    @pytest.mark.asyncio
    async def test_task_checkpoint_saved_after_iteration(self):
        """Run a task that takes 2 iterations, verify checkpoint exists after first."""
        tool_call_response = LLMResponse(
            content="",
            tool_calls=[ToolCallInfo(name="web_search", arguments={"query": "test"})],
            tokens_used=50,
        )
        final_response = LLMResponse(content='{"result": {"done": true}}', tokens_used=30)
        loop = _make_loop([tool_call_response, final_response], real_memory=True)
        loop.skills.get_tool_definitions = MagicMock(
            return_value=[{"type": "function", "function": {"name": "web_search"}}]
        )
        loop.skills.execute = AsyncMock(return_value={"results": ["r1"]})

        # Track checkpoint saves
        saved_checkpoints = []
        original_save = loop.memory.save_task_checkpoint

        def tracking_save(*args, **kwargs):
            result = original_save(*args, **kwargs)
            cp = loop.memory.load_task_checkpoint()
            saved_checkpoints.append(cp)
            return result

        loop.memory.save_task_checkpoint = tracking_save

        assignment = TaskAssignment(
            workflow_id="wf1", step_id="s1", task_type="research", input_data={"q": "test"}
        )
        result = await loop.execute_task(assignment)
        assert result.status == "complete"
        # Checkpoint was saved at least once (after tool call iteration)
        assert len(saved_checkpoints) >= 1
        assert saved_checkpoints[0]["iteration"] == 0

    @pytest.mark.asyncio
    async def test_task_checkpoint_cleared_on_success(self):
        """Run task to completion, verify checkpoint cleared."""
        loop = _make_loop(real_memory=True)
        assignment = TaskAssignment(
            workflow_id="wf1", step_id="s1", task_type="research", input_data={"q": "test"}
        )
        result = await loop.execute_task(assignment)
        assert result.status == "complete"
        # Checkpoint should be cleared
        assert loop.memory.load_task_checkpoint() is None

    @pytest.mark.asyncio
    async def test_task_checkpoint_cleared_on_failure(self):
        """Force task exception, verify checkpoint cleared."""
        loop = _make_loop(real_memory=True)
        loop.llm.chat = AsyncMock(side_effect=RuntimeError("LLM exploded"))

        assignment = TaskAssignment(
            workflow_id="wf1", step_id="s1", task_type="research", input_data={"q": "test"}
        )
        result = await loop.execute_task(assignment)
        assert result.status == "failed"
        # Checkpoint should be cleared even on failure
        assert loop.memory.load_task_checkpoint() is None

    @pytest.mark.asyncio
    async def test_task_resume_from_checkpoint(self):
        """Pre-populate checkpoint, run execute_task with matching task_id, verify resume."""
        loop = _make_loop(real_memory=True)

        task_id = "task_resume_test"
        assignment = TaskAssignment(
            workflow_id="wf1", step_id="s1", task_type="research",
            input_data={"q": "test"}, task_id=task_id,
        )

        # Pre-populate checkpoint at iteration 2 with some messages
        checkpoint_messages = [
            {"role": "user", "content": "do research on test"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "web_search", "arguments": '{"query": "test"}'}}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": '{"results": ["r1"]}'},
        ]
        loop.memory.save_task_checkpoint(
            task_id=task_id,
            assignment_json=assignment.model_dump_json(),
            messages=checkpoint_messages,
            iteration=2,
            tokens_used=500,
            budget_used_tokens=0,
            budget_estimated_cost=0.0,
            flush_triggered=False,
        )

        # LLM returns final answer immediately on resume
        loop.llm.chat = AsyncMock(
            return_value=LLMResponse(content='{"result": {"answer": "resumed"}}', tokens_used=100)
        )

        result = await loop.execute_task(assignment)
        assert result.status == "complete"
        assert result.result == {"answer": "resumed"}
        # Tokens should include checkpoint's 500 + new 100
        assert result.tokens_used == 600

    @pytest.mark.asyncio
    async def test_task_resume_stale_checkpoint_cleared(self):
        """Pre-populate checkpoint with different task_id, verify cleared and fresh start."""
        loop = _make_loop(real_memory=True)

        # Checkpoint from a different task
        loop.memory.save_task_checkpoint(
            task_id="old_task",
            assignment_json="{}",
            messages=[{"role": "user", "content": "old stuff"}],
            iteration=5,
            tokens_used=9999,
            budget_used_tokens=0,
            budget_estimated_cost=0.0,
            flush_triggered=False,
        )

        assignment = TaskAssignment(
            workflow_id="wf1", step_id="s1", task_type="research",
            input_data={"q": "test"}, task_id="new_task",
        )
        result = await loop.execute_task(assignment)
        assert result.status == "complete"
        # Should NOT carry forward old checkpoint's 9999 tokens
        assert result.tokens_used == 100  # just the one LLM call
        # Checkpoint should be cleared
        assert loop.memory.load_task_checkpoint() is None

    @pytest.mark.asyncio
    async def test_task_resume_continuation_prompt(self):
        """Pre-populate checkpoint where last message is assistant role, verify continuation user message."""
        loop = _make_loop(real_memory=True)

        task_id = "task_cont_test"
        assignment = TaskAssignment(
            workflow_id="wf1", step_id="s1", task_type="research",
            input_data={"q": "test"}, task_id=task_id,
        )

        # Last message is assistant (tool call results were last, then compaction happened
        # with assistant being the last message)
        checkpoint_messages = [
            {"role": "user", "content": "do research"},
            {"role": "assistant", "content": "working on it"},
        ]
        loop.memory.save_task_checkpoint(
            task_id=task_id,
            assignment_json=assignment.model_dump_json(),
            messages=checkpoint_messages,
            iteration=1,
            tokens_used=200,
            budget_used_tokens=0,
            budget_estimated_cost=0.0,
            flush_triggered=False,
        )

        # Capture messages sent to LLM
        captured_messages = []
        original_chat = loop.llm.chat

        async def capturing_chat(system, messages, tools=None, **kwargs):
            captured_messages.append([dict(m) for m in messages])
            return LLMResponse(content='{"result": {"done": true}}', tokens_used=50)

        loop.llm.chat = capturing_chat

        result = await loop.execute_task(assignment)
        assert result.status == "complete"
        # The messages sent to LLM should include the continuation prompt
        assert len(captured_messages) >= 1
        first_call = captured_messages[0]
        # Last message before LLM call should be the continuation user message
        last_msg = first_call[-1]
        assert last_msg["role"] == "user"
        assert "interrupted" in last_msg["content"]

    @pytest.mark.asyncio
    async def test_task_resume_no_continuation_when_last_is_user(self):
        """Pre-populate checkpoint where last message is user role, verify NO continuation."""
        loop = _make_loop(real_memory=True)

        task_id = "task_no_cont_test"
        assignment = TaskAssignment(
            workflow_id="wf1", step_id="s1", task_type="research",
            input_data={"q": "test"}, task_id=task_id,
        )

        # Last message is already a user message
        checkpoint_messages = [
            {"role": "user", "content": "do research"},
            {"role": "assistant", "content": "I need more info"},
            {"role": "user", "content": "here is more info"},
        ]
        loop.memory.save_task_checkpoint(
            task_id=task_id,
            assignment_json=assignment.model_dump_json(),
            messages=checkpoint_messages,
            iteration=1,
            tokens_used=300,
            budget_used_tokens=0,
            budget_estimated_cost=0.0,
            flush_triggered=False,
        )

        captured_messages = []

        async def capturing_chat(system, messages, tools=None, **kwargs):
            captured_messages.append([dict(m) for m in messages])
            return LLMResponse(content='{"result": {"done": true}}', tokens_used=50)

        loop.llm.chat = capturing_chat

        result = await loop.execute_task(assignment)
        assert result.status == "complete"
        assert len(captured_messages) >= 1
        first_call = captured_messages[0]
        # Last message should be the original "here is more info", NOT a continuation prompt
        last_msg = first_call[-1]
        assert last_msg["role"] == "user"
        assert last_msg["content"] == "here is more info"
        assert "interrupted" not in last_msg["content"]
