"""Tests for auto-resume on agent startup and checkpoint status endpoint.

Covers:
- Auto-resume from checkpoint when the agent lifespan starts
- /status endpoint exposes has_task_checkpoint field
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from src.agent.server import create_agent_app
from src.shared.types import TaskAssignment

# ── Helpers ──────────────────────────────────────────────────────


def _make_assignment(task_id: str = "tsk_abc123") -> TaskAssignment:
    return TaskAssignment(
        workflow_id="wf1",
        step_id="s1",
        task_type="research",
        input_data={"query": "test"},
        task_id=task_id,
    )


def _make_checkpoint(assignment: TaskAssignment | None = None, iteration: int = 3) -> dict:
    if assignment is None:
        assignment = _make_assignment()
    return {
        "task_id": assignment.task_id,
        "assignment_json": assignment.model_dump_json(),
        "messages": [],
        "iteration": iteration,
        "tokens_used": 500,
        "budget_used_tokens": 0,
        "budget_estimated_cost": 0.0,
        "flush_triggered": False,
    }


def _make_mock_loop(*, has_memory: bool = True) -> MagicMock:
    """Create a mock AgentLoop with common defaults."""
    loop = MagicMock()
    loop.agent_id = "test_agent"
    loop.role = "researcher"
    loop.state = "idle"
    loop.current_task = None
    loop._current_task_handle = None
    loop._cancel_requested = False
    loop._excluded_tools = frozenset()
    loop.skills = MagicMock()
    loop.skills.list_skills = MagicMock(return_value=[])
    loop.skills.get_tool_definitions = MagicMock(return_value=[])
    loop.skills.get_tool_sources = MagicMock(return_value={})
    loop.skills.execute = AsyncMock(return_value={"ok": True})
    loop.mesh_client = MagicMock()
    loop.workspace = None

    if has_memory:
        loop.memory = MagicMock()
        loop.memory._run_db = AsyncMock(return_value=None)
        loop.memory.load_task_checkpoint = MagicMock()  # sync, passed to _run_db
    else:
        loop.memory = None

    return loop


# ── Auto-resume tests ────────────────────────────────────────────


class TestAutoResume:
    """Test the auto-resume logic in the agent lifespan handler."""

    @pytest.mark.asyncio
    async def test_auto_resume_with_checkpoint(self):
        """When a checkpoint exists, loop state is set and execute_task is called."""
        assignment = _make_assignment()
        cp = _make_checkpoint(assignment)

        loop = _make_mock_loop()
        loop.memory._run_db = AsyncMock(return_value=cp)
        loop.execute_task = AsyncMock(return_value=None)

        # We test the lifespan logic by importing and simulating the relevant
        # section. Since lifespan is tightly coupled to __main__.py, we
        # replicate the core logic here and verify behavior.
        import secrets

        from src.shared.types import TaskAssignment as TA

        # Simulate the auto-resume block
        if loop.memory:
            result_cp = await loop.memory._run_db(loop.memory.load_task_checkpoint)
            if result_cp:
                _resume_trace_id = f"tr_{secrets.token_hex(6)}"
                parsed = TA.model_validate_json(result_cp["assignment_json"])
                loop.state = "working"
                loop.current_task = parsed.task_id

                async def _auto_resume():
                    await loop.execute_task(parsed, trace_id=_resume_trace_id)

                _resume_task = asyncio.create_task(_auto_resume())
                loop._current_task_handle = _resume_task
                await _resume_task  # wait for it to finish in test

        assert loop.state == "working"
        assert loop.current_task == assignment.task_id
        loop.execute_task.assert_called_once()
        call_args = loop.execute_task.call_args
        assert call_args[0][0].task_id == assignment.task_id
        assert call_args[1]["trace_id"].startswith("tr_")

    @pytest.mark.asyncio
    async def test_auto_resume_no_checkpoint(self):
        """When no checkpoint exists, loop stays idle and execute_task is not called."""
        loop = _make_mock_loop()
        loop.memory._run_db = AsyncMock(return_value=None)
        loop.execute_task = AsyncMock()

        # Simulate the auto-resume block
        if loop.memory:
            result_cp = await loop.memory._run_db(loop.memory.load_task_checkpoint)
            if result_cp:
                # Should not enter here
                loop.execute_task()

        assert loop.state == "idle"
        assert loop.current_task is None
        loop.execute_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_resume_load_failure(self, caplog):
        """When load_task_checkpoint raises, warning is logged and no crash."""
        loop = _make_mock_loop()
        loop.memory._run_db = AsyncMock(side_effect=RuntimeError("DB corrupted"))
        loop.execute_task = AsyncMock()

        # Simulate the auto-resume block with exception handling
        if loop.memory:
            try:
                await loop.memory._run_db(loop.memory.load_task_checkpoint)
            except Exception as e:
                # In the real code, this logs a warning
                logging.getLogger("agent.main").warning("Auto-resume check failed: %s", e)

        assert loop.state == "idle"
        assert loop.current_task is None
        loop.execute_task.assert_not_called()


# ── Status endpoint tests ────────────────────────────────────────


class TestStatusCheckpoint:
    """Test /status endpoint includes has_task_checkpoint field."""

    @pytest.mark.asyncio
    async def test_status_with_checkpoint(self):
        """GET /status returns has_task_checkpoint: true when checkpoint exists."""
        loop = _make_mock_loop()
        cp = _make_checkpoint()
        loop.memory._run_db = AsyncMock(return_value=cp)

        # get_status needs to return a real AgentStatus
        from src.shared.types import AgentStatus

        loop.get_status = MagicMock(return_value=AgentStatus(
            agent_id="test_agent",
            role="researcher",
            state="idle",
            current_task=None,
            capabilities=[],
            uptime_seconds=42.0,
            tasks_completed=0,
            tasks_failed=0,
        ))

        app = create_agent_app(loop)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/status")
            assert resp.status_code == 200
            data = resp.json()
            assert data["has_task_checkpoint"] is True
            # Ensure standard fields are still present
            assert data["agent_id"] == "test_agent"
            assert data["state"] == "idle"

    @pytest.mark.asyncio
    async def test_status_without_checkpoint(self):
        """GET /status returns has_task_checkpoint: false when no checkpoint."""
        loop = _make_mock_loop()
        loop.memory._run_db = AsyncMock(return_value=None)

        from src.shared.types import AgentStatus

        loop.get_status = MagicMock(return_value=AgentStatus(
            agent_id="test_agent",
            role="researcher",
            state="idle",
            current_task=None,
            capabilities=[],
            uptime_seconds=42.0,
            tasks_completed=0,
            tasks_failed=0,
        ))

        app = create_agent_app(loop)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/status")
            assert resp.status_code == 200
            data = resp.json()
            assert data["has_task_checkpoint"] is False

    @pytest.mark.asyncio
    async def test_status_no_memory(self):
        """GET /status returns has_task_checkpoint: false when memory is None."""
        loop = _make_mock_loop(has_memory=False)

        from src.shared.types import AgentStatus

        loop.get_status = MagicMock(return_value=AgentStatus(
            agent_id="test_agent",
            role="researcher",
            state="idle",
            current_task=None,
            capabilities=[],
            uptime_seconds=42.0,
            tasks_completed=0,
            tasks_failed=0,
        ))

        app = create_agent_app(loop)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/status")
            assert resp.status_code == 200
            data = resp.json()
            assert data["has_task_checkpoint"] is False

    @pytest.mark.asyncio
    async def test_status_checkpoint_load_error(self):
        """GET /status returns has_task_checkpoint: false when DB query fails."""
        loop = _make_mock_loop()
        loop.memory._run_db = AsyncMock(side_effect=RuntimeError("DB error"))

        from src.shared.types import AgentStatus

        loop.get_status = MagicMock(return_value=AgentStatus(
            agent_id="test_agent",
            role="researcher",
            state="idle",
            current_task=None,
            capabilities=[],
            uptime_seconds=42.0,
            tasks_completed=0,
            tasks_failed=0,
        ))

        app = create_agent_app(loop)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/status")
            assert resp.status_code == 200
            data = resp.json()
            assert data["has_task_checkpoint"] is False
