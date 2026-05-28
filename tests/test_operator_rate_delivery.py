"""Tests for the ``rate_delivery`` operator tool (PR 2 of Work tab rewrite).

Covers:

* Operator-only gating (non-operator rejected)
* Missing mesh_client returns error
* Missing / empty task_id rejected
* Unknown outcome rejected
* feedback type validation
* feedback required for rework / rejected
* feedback length cap (2000 chars)
* Happy path → POSTs to mesh and returns ok dict
* rework outcome → surfaces ``rework_task_id`` from mesh response
* Mesh exception → returns ``{error}`` (no crash)
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest


@pytest.fixture(autouse=True)
def _set_operator_env(monkeypatch):
    monkeypatch.setenv("ALLOWED_TOOLS", "rate_delivery")


def _make_mesh(response: dict | None = None, raises: Exception | None = None):
    mesh = AsyncMock()
    if raises is not None:
        mesh.set_task_outcome.side_effect = raises
    else:
        mesh.set_task_outcome.return_value = response or {"ok": True}
    return mesh


@pytest.mark.asyncio
async def test_rate_delivery_non_operator_rejected(monkeypatch):
    monkeypatch.delenv("ALLOWED_TOOLS", raising=False)
    from src.agent.builtins.operator_tools import rate_delivery

    result = await rate_delivery(
        task_id="task_1", outcome="accepted", mesh_client=_make_mesh(),
    )
    assert "error" in result
    assert "operator" in result["error"].lower()


@pytest.mark.asyncio
async def test_rate_delivery_no_mesh_client():
    from src.agent.builtins.operator_tools import rate_delivery

    result = await rate_delivery(task_id="task_1", outcome="accepted")
    assert "error" in result
    assert "mesh_client" in result["error"]


@pytest.mark.asyncio
async def test_rate_delivery_missing_task_id():
    from src.agent.builtins.operator_tools import rate_delivery

    mesh = _make_mesh()
    result = await rate_delivery(
        task_id="", outcome="accepted", mesh_client=mesh,
    )
    assert "error" in result
    assert "task_id" in result["error"]
    mesh.set_task_outcome.assert_not_called()


@pytest.mark.asyncio
async def test_rate_delivery_unknown_outcome_rejected():
    from src.agent.builtins.operator_tools import rate_delivery

    mesh = _make_mesh()
    result = await rate_delivery(
        task_id="task_1", outcome="loved_it", mesh_client=mesh,
    )
    assert "error" in result
    assert "outcome" in result["error"]
    mesh.set_task_outcome.assert_not_called()


@pytest.mark.asyncio
async def test_rate_delivery_feedback_must_be_string():
    from src.agent.builtins.operator_tools import rate_delivery

    mesh = _make_mesh()
    # The skill is typed as ``feedback: str`` but the LLM could pass
    # garbage; we defend at the boundary.
    result = await rate_delivery(
        task_id="task_1",
        outcome="accepted",
        feedback=123,  # type: ignore[arg-type]
        mesh_client=mesh,
    )
    assert "error" in result
    assert "feedback" in result["error"]
    mesh.set_task_outcome.assert_not_called()


@pytest.mark.asyncio
async def test_rate_delivery_rework_requires_feedback():
    from src.agent.builtins.operator_tools import rate_delivery

    mesh = _make_mesh()
    result = await rate_delivery(
        task_id="task_1", outcome="rework", feedback="   ",
        mesh_client=mesh,
    )
    assert "error" in result
    assert "feedback is required" in result["error"]
    mesh.set_task_outcome.assert_not_called()


@pytest.mark.asyncio
async def test_rate_delivery_rejected_requires_feedback():
    from src.agent.builtins.operator_tools import rate_delivery

    mesh = _make_mesh()
    result = await rate_delivery(
        task_id="task_1", outcome="rejected", feedback="",
        mesh_client=mesh,
    )
    assert "error" in result
    assert "feedback is required" in result["error"]
    mesh.set_task_outcome.assert_not_called()


@pytest.mark.asyncio
async def test_rate_delivery_feedback_length_capped():
    from src.agent.builtins.operator_tools import rate_delivery

    mesh = _make_mesh()
    # 2001 chars — one over the cap.
    big = "x" * 2001
    result = await rate_delivery(
        task_id="task_1", outcome="rework", feedback=big,
        mesh_client=mesh,
    )
    assert "error" in result
    assert "2000" in result["error"]
    mesh.set_task_outcome.assert_not_called()


@pytest.mark.asyncio
async def test_rate_delivery_accepted_happy_path():
    from src.agent.builtins.operator_tools import rate_delivery

    mesh = _make_mesh(response={"ok": True, "task": {"id": "task_1"}})
    result = await rate_delivery(
        task_id="task_1", outcome="accepted", mesh_client=mesh,
    )
    assert result["ok"] is True
    assert result["task_id"] == "task_1"
    assert result["outcome"] == "accepted"
    mesh.set_task_outcome.assert_awaited_once_with("task_1", "accepted", "")


@pytest.mark.asyncio
async def test_rate_delivery_acknowledged_silent_feedback_ok():
    """acknowledged with empty feedback is the safe default and must work."""
    from src.agent.builtins.operator_tools import rate_delivery

    mesh = _make_mesh(response={"ok": True})
    result = await rate_delivery(
        task_id="task_1", outcome="acknowledged", mesh_client=mesh,
    )
    assert result["ok"] is True
    assert result["outcome"] == "acknowledged"


@pytest.mark.asyncio
async def test_rate_delivery_rework_surfaces_rework_task_id():
    from src.agent.builtins.operator_tools import rate_delivery

    mesh = _make_mesh(response={
        "ok": True,
        "task": {"id": "task_1"},
        "rework_task_id": "task_2",
        "rework_assignee": "scout",
    })
    result = await rate_delivery(
        task_id="task_1",
        outcome="rework",
        feedback="tighten the intro paragraph",
        mesh_client=mesh,
    )
    assert result["ok"] is True
    assert result["rework_task_id"] == "task_2"
    assert result["rework_assignee"] == "scout"
    mesh.set_task_outcome.assert_awaited_once_with(
        "task_1", "rework", "tighten the intro paragraph",
    )


@pytest.mark.asyncio
async def test_rate_delivery_mesh_failure_returns_error():
    from src.agent.builtins.operator_tools import rate_delivery

    mesh = _make_mesh(raises=RuntimeError("mesh down"))
    result = await rate_delivery(
        task_id="task_1", outcome="accepted", mesh_client=mesh,
    )
    assert "error" in result
    assert "mesh down" in result["error"]


@pytest.mark.asyncio
async def test_rate_delivery_feedback_sanitized_before_transit():
    """Invisible Unicode / zero-width chars must be stripped via
    ``sanitize_for_prompt`` before the mesh receives the string.

    Without this guard a corrupted upstream feedback could poison the
    receiving agent's memory write.
    """
    from src.agent.builtins.operator_tools import rate_delivery

    mesh = _make_mesh(response={"ok": True})
    # Zero-width space + bidi marker injected in feedback.
    raw = "needs more depth​text‮"
    result = await rate_delivery(
        task_id="task_1", outcome="rework", feedback=raw,
        mesh_client=mesh,
    )
    assert result["ok"] is True
    # The mesh-side feedback must not contain the raw smuggled chars.
    call = mesh.set_task_outcome.await_args
    sent_feedback = call.args[2]
    assert "​" not in sent_feedback
    assert "‮" not in sent_feedback
    assert "needs more depth" in sent_feedback
