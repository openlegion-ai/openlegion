"""Tests for the ``compose_work_summary`` operator skill.

The skill is deterministic (no LLM call) and composes from the existing
``team_summary`` endpoint + prior rated feedback. Tests pin:

- Operator gate (other agents get a clean error, not a 403).
- Param validation (scope_kind, scope_id, period_hours).
- Narrative is built from ``team_summary`` data (mocked).
- Recommendations prioritize top blockers, then prior 👎 feedback,
  then a "stay the course" fallback.
- Persistence goes through ``mesh_client.create_work_summary``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_mesh(*, base_summary=None, list_response=None, create_response=None):
    """Mesh client mock with the three methods compose_work_summary calls."""
    mc = MagicMock()
    mc.team_summary = AsyncMock(
        return_value=base_summary or {
            "status_text": "team is healthy",
            "counts": {"active": 2, "blocked": 0, "done": 5, "failed": 0},
            "top_blockers": [],
            "recent_completions": [],
        },
    )
    mc.list_work_summaries = AsyncMock(
        return_value=list_response or {"summaries": []},
    )
    mc.create_work_summary = AsyncMock(
        return_value=create_response or {"id": "ws_test", "scope_id": "x"},
    )
    return mc


@pytest.fixture
def as_operator(monkeypatch):
    """Pretend the running agent IS operator for the @skill is_operator gate."""
    monkeypatch.setenv("AGENT_ID", "operator")
    monkeypatch.setenv("ALLOWED_TOOLS", "compose_work_summary,foo")
    return None


@pytest.mark.asyncio
async def test_non_operator_blocked(monkeypatch):
    """Workers must get a clean error, not a server-side 403."""
    monkeypatch.setenv("AGENT_ID", "scout")
    monkeypatch.delenv("ALLOWED_TOOLS", raising=False)
    from src.agent.builtins.operator_tools import compose_work_summary
    result = await compose_work_summary(
        scope_kind="team", scope_id="x", mesh_client=MagicMock(),
    )
    assert "error" in result
    assert "operator" in result["error"].lower()


@pytest.mark.asyncio
async def test_invalid_scope_kind_returns_error(as_operator):
    from src.agent.builtins.operator_tools import compose_work_summary
    result = await compose_work_summary(
        scope_kind="bogus", scope_id="x", mesh_client=_make_mesh(),
    )
    assert "error" in result
    assert "scope_kind" in result["error"]


@pytest.mark.asyncio
async def test_missing_scope_id_returns_error(as_operator):
    from src.agent.builtins.operator_tools import compose_work_summary
    result = await compose_work_summary(
        scope_kind="team", scope_id="", mesh_client=_make_mesh(),
    )
    assert "error" in result
    assert "scope_id" in result["error"]


@pytest.mark.asyncio
async def test_period_hours_clamped(as_operator):
    """period_hours must clamp to [1, 720]."""
    from src.agent.builtins.operator_tools import compose_work_summary
    mc = _make_mesh()
    # 99999 → clamp to 720
    await compose_work_summary(
        scope_kind="team", scope_id="x", period_hours=99999, mesh_client=mc,
    )
    call = mc.create_work_summary.call_args
    metrics = call.kwargs["metrics"]
    assert metrics["period_hours"] == 720
    # 0 → clamp to 1
    mc.create_work_summary.reset_mock()
    await compose_work_summary(
        scope_kind="team", scope_id="x", period_hours=0, mesh_client=mc,
    )
    call = mc.create_work_summary.call_args
    assert call.kwargs["metrics"]["period_hours"] == 1


@pytest.mark.asyncio
async def test_metrics_built_from_team_summary(as_operator):
    from src.agent.builtins.operator_tools import compose_work_summary
    mc = _make_mesh(
        base_summary={
            "status_text": "ok",
            "counts": {"active": 4, "blocked": 2, "done": 7, "failed": 1},
            "top_blockers": [{"task_id": "t1", "title": "stuck", "assignee": "a"}],
            "recent_completions": [{"title": "shipped", "assignee": "b"}],
        },
    )
    await compose_work_summary(
        scope_kind="team", scope_id="content-seo", mesh_client=mc,
    )
    call = mc.create_work_summary.call_args
    metrics = call.kwargs["metrics"]
    assert metrics["tasks_active"] == 4
    assert metrics["tasks_blocked"] == 2
    assert metrics["tasks_done"] == 7
    assert metrics["tasks_failed"] == 1
    assert metrics["top_blocker_count"] == 1
    assert metrics["recent_completion_count"] == 1


@pytest.mark.asyncio
async def test_narrative_includes_team_label_and_status(as_operator):
    from src.agent.builtins.operator_tools import compose_work_summary
    mc = _make_mesh(
        base_summary={
            "status_text": "Quiet week, no blockers.",
            "counts": {"active": 1, "blocked": 0, "done": 3, "failed": 0},
            "top_blockers": [],
            "recent_completions": [],
        },
    )
    await compose_work_summary(
        scope_kind="team", scope_id="content-seo", mesh_client=mc,
    )
    narrative = mc.create_work_summary.call_args.kwargs["narrative_md"]
    assert "content-seo" in narrative
    assert "Quiet week, no blockers." in narrative
    assert "**Activity**" in narrative
    assert "1 active" in narrative


@pytest.mark.asyncio
async def test_narrative_for_solo_uses_solo_label(as_operator):
    from src.agent.builtins.operator_tools import compose_work_summary
    mc = _make_mesh()
    await compose_work_summary(
        scope_kind="solo", scope_id="standalone-agent", mesh_client=mc,
    )
    narrative = mc.create_work_summary.call_args.kwargs["narrative_md"]
    assert "Solo agent" in narrative
    assert "standalone-agent" in narrative
    # Solo path should NOT have called team_summary.
    mc.team_summary.assert_not_called()


@pytest.mark.asyncio
async def test_recommendations_prioritize_top_blockers(as_operator):
    from src.agent.builtins.operator_tools import compose_work_summary
    mc = _make_mesh(
        base_summary={
            "status_text": "blocked",
            "counts": {"active": 0, "blocked": 1, "done": 0, "failed": 0},
            "top_blockers": [
                {"task_id": "t1", "title": "stage-4 deadlock", "assignee": "dev-pub"},
            ],
            "recent_completions": [],
        },
    )
    await compose_work_summary(
        scope_kind="team", scope_id="content-seo", mesh_client=mc,
    )
    recs = mc.create_work_summary.call_args.kwargs["recommendations"]
    assert len(recs) >= 1
    assert "stage-4 deadlock" in recs[0] or "Unblock" in recs[0]


@pytest.mark.asyncio
async def test_recommendations_inject_prior_rework_feedback(as_operator):
    from src.agent.builtins.operator_tools import compose_work_summary
    mc = _make_mesh(
        list_response={
            "summaries": [
                {
                    "rating": "rework",
                    "feedback": "Focus more on stage-4 publishing throughput",
                    "rated_at": 1.0,
                },
            ],
        },
    )
    await compose_work_summary(
        scope_kind="team", scope_id="content-seo", mesh_client=mc,
    )
    recs = mc.create_work_summary.call_args.kwargs["recommendations"]
    joined = " ".join(recs)
    assert "stage-4 publishing throughput" in joined
    assert "👎" in joined or "prior" in joined.lower()


@pytest.mark.asyncio
async def test_quiet_team_gets_stay_the_course_recommendation(as_operator):
    from src.agent.builtins.operator_tools import compose_work_summary
    mc = _make_mesh()  # no blockers, no prior feedback
    await compose_work_summary(
        scope_kind="team", scope_id="content-seo", mesh_client=mc,
    )
    recs = mc.create_work_summary.call_args.kwargs["recommendations"]
    assert len(recs) >= 1
    assert any("stay the course" in r.lower() for r in recs)


@pytest.mark.asyncio
async def test_persistence_failure_returned_as_error(as_operator):
    from src.agent.builtins.operator_tools import compose_work_summary
    mc = _make_mesh()
    mc.create_work_summary = AsyncMock(side_effect=RuntimeError("db blew up"))
    result = await compose_work_summary(
        scope_kind="team", scope_id="x", mesh_client=mc,
    )
    assert "error" in result
    assert "db blew up" in result["error"]


@pytest.mark.asyncio
async def test_prior_feedback_fetch_failure_does_not_block_compose(as_operator):
    """If list_work_summaries blows up, compose still proceeds."""
    from src.agent.builtins.operator_tools import compose_work_summary
    mc = _make_mesh()
    mc.list_work_summaries = AsyncMock(side_effect=RuntimeError("transient"))
    result = await compose_work_summary(
        scope_kind="team", scope_id="x", mesh_client=mc,
    )
    # Should successfully create (using the create_response default).
    assert "error" not in result
    mc.create_work_summary.assert_awaited_once()
