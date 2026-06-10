"""Tests for the operator ``inspect_task_run`` tool (B5).

Operator-only diagnostic over ``GET /mesh/tasks/{id}/run`` — surfaces how
a task actually executed (thinking level, LLM calls, tokens, trace
errors, status timeline) so the operator can diagnose shallow or failed
deliverables before re-dispatching or rating.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture(autouse=True)
def _set_operator_env(monkeypatch):
    """inspect_task_run requires ALLOWED_TOOLS to be set (defence-in-depth)."""
    monkeypatch.setenv("ALLOWED_TOOLS", "inspect_task_run")


@pytest.mark.asyncio
async def test_inspect_task_run_returns_payload():
    from src.agent.builtins.operator_tools import inspect_task_run

    payload = {
        "task": {"id": "task_1", "thinking": "high", "status": "done"},
        "execution": {"llm_calls": 4, "tokens_used": 9000, "models": ["m"]},
        "events": [{"event_kind": "created"}],
    }
    mc = MagicMock()
    mc.get_task_run = AsyncMock(return_value=payload)

    result = await inspect_task_run("task_1", mesh_client=mc)
    assert result == payload
    mc.get_task_run.assert_awaited_once_with("task_1")


@pytest.mark.asyncio
async def test_inspect_task_run_not_found():
    from src.agent.builtins.operator_tools import inspect_task_run

    mc = MagicMock()
    mc.get_task_run = AsyncMock(return_value=None)

    result = await inspect_task_run("task_missing", mesh_client=mc)
    assert result["error"] == "not_found"
    assert result["task_id"] == "task_missing"


@pytest.mark.asyncio
async def test_inspect_task_run_transport_error_envelope():
    from src.agent.builtins.operator_tools import inspect_task_run

    mc = MagicMock()
    mc.get_task_run = AsyncMock(side_effect=RuntimeError("mesh down"))

    result = await inspect_task_run("task_1", mesh_client=mc)
    assert "error" in result
    assert "mesh down" in result["error"]


@pytest.mark.asyncio
async def test_inspect_task_run_rejects_non_operator(monkeypatch):
    monkeypatch.setenv("ALLOWED_TOOLS", "")
    from src.agent.builtins.operator_tools import inspect_task_run

    result = await inspect_task_run("task_1", mesh_client=MagicMock())
    assert "error" in result
    assert "operator" in result["error"]
