"""Tests for ``/mesh/work-summaries`` endpoints + operator tool +
``WorkSummariesStore`` end-to-end via the mesh app.

Covers create / list / get / rate paths + permission boundaries
(operator-only create + rate; scope-filtered visibility on read).
"""

from __future__ import annotations

import importlib
import time

import pytest
from httpx import ASGITransport, AsyncClient

from src.host.costs import CostTracker
from src.host.mesh import Blackboard, MessageRouter, PubSub
from src.host.permissions import AgentPermissions, PermissionMatrix
from src.host.traces import TraceStore


def _reload_server():
    import src.host.server as server_module
    importlib.reload(server_module)
    return server_module


@pytest.fixture
def mesh_setup(tmp_path, monkeypatch):
    """Mesh app with auth tokens + tasks store + summaries store.

    Permissions intentionally narrow so operator's success on the
    bypass paths is a real signal, not an accident of grant coverage.
    Team membership is seeded on the app's TeamStore so the server-side
    ``_is_team_member`` check sees scout as a member of content-seo.
    """
    monkeypatch.setenv("OPENLEGION_TEAM_SCOPE_MODE", "warn")
    monkeypatch.setenv(
        "OPENLEGION_ORCHESTRATION_TASKS_DB", str(tmp_path / "tasks.db"),
    )
    monkeypatch.setenv(
        "OPENLEGION_WORK_SUMMARIES_DB", str(tmp_path / "summaries.db"),
    )
    # Pin the track-record ledger DB too (plan §8 #18) — otherwise this
    # module's app shares data/track_record.db with every other test
    # file that doesn't override it, making count assertions flaky.
    monkeypatch.setenv(
        "OPENLEGION_TRACK_RECORD_DB", str(tmp_path / "track_record.db"),
    )
    server = _reload_server()

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {
        "operator": AgentPermissions(agent_id="operator"),
        "scout": AgentPermissions(agent_id="scout"),
        "writer": AgentPermissions(agent_id="writer"),
    }
    perms._config_path = str(tmp_path / "perms.json")

    router = MessageRouter(permissions=perms, agent_registry={})
    router.register_agent("operator", "http://operator:8400", [])
    router.register_agent("scout", "http://scout:8400", [])
    router.register_agent("writer", "http://writer:8400", [])

    costs = CostTracker(str(tmp_path / "costs.db"))
    traces = TraceStore(str(tmp_path / "traces.db"))

    auth_tokens = {
        "operator": "operator-secret",
        "scout": "scout-secret",
        "writer": "writer-secret",
    }
    app = server.create_mesh_app(
        blackboard=bb,
        pubsub=pubsub,
        router=router,
        permissions=perms,
        cost_tracker=costs,
        trace_store=traces,
        auth_tokens=auth_tokens,
    )
    # scout is a member of "content-seo"; writer is solo (no membership).
    app.teams_store.create_team("content-seo", description="Test team")
    app.teams_store.add_member("content-seo", "scout")
    yield {
        "app": app,
        "store": app.summaries_store,
        "tokens": auth_tokens,
    }
    bb.close()
    costs.close()
    traces.close()
    app.summaries_store.close()
    app.track_record_store.close()
    monkeypatch.delenv("OPENLEGION_TEAM_SCOPE_MODE", raising=False)
    monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_DB", raising=False)
    monkeypatch.delenv("OPENLEGION_WORK_SUMMARIES_DB", raising=False)
    monkeypatch.delenv("OPENLEGION_TRACK_RECORD_DB", raising=False)
    _reload_server()


def _hdr(token: str) -> dict:
    return {"authorization": f"Bearer {token}"}


def _payload(scope_id: str = "content-seo", **overrides) -> dict:
    now = time.time()
    base = {
        "scope_kind": "team",
        "scope_id": scope_id,
        "period_start": now - 86400,
        "period_end": now,
        "narrative_md": "## Test team\n\n3 delivered, 1 blocked.",
        "metrics": {"created": 5, "delivered": 3, "blocked": 1},
        "recommendations": ["Unblock stage-4 stall"],
    }
    base.update(overrides)
    return base


# =============================================================================
# Create
# =============================================================================


@pytest.mark.asyncio
async def test_operator_can_create_summary(mesh_setup):
    app = mesh_setup["app"]
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post(
            "/mesh/work-summaries",
            json=_payload(),
            headers=_hdr(mesh_setup["tokens"]["operator"]),
        )
    assert resp.status_code == 200, resp.text
    row = resp.json()
    assert row["id"].startswith("ws_")
    assert row["scope_id"] == "content-seo"
    assert row["rating"] is None
    assert row["generated_by"] == "operator"


@pytest.mark.asyncio
async def test_worker_cannot_create_summary(mesh_setup):
    """Workers must NOT create summaries — operator-or-internal only."""
    app = mesh_setup["app"]
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post(
            "/mesh/work-summaries",
            json=_payload(),
            headers=_hdr(mesh_setup["tokens"]["scout"]),
        )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_create_rejects_missing_fields(mesh_setup):
    app = mesh_setup["app"]
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post(
            "/mesh/work-summaries",
            json={"scope_kind": "team"},  # missing scope_id, narrative, periods
            headers=_hdr(mesh_setup["tokens"]["operator"]),
        )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_create_dedupes_same_period(mesh_setup):
    app = mesh_setup["app"]
    payload = _payload()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        a = await client.post(
            "/mesh/work-summaries", json=payload,
            headers=_hdr(mesh_setup["tokens"]["operator"]),
        )
        b = await client.post(
            "/mesh/work-summaries", json=payload,
            headers=_hdr(mesh_setup["tokens"]["operator"]),
        )
    assert a.status_code == 200
    # UNIQUE collision surfaces as 409 so crons can retry-safely.
    assert b.status_code == 409


# =============================================================================
# List + visibility
# =============================================================================


@pytest.mark.asyncio
async def test_operator_lists_all_summaries(mesh_setup):
    store = mesh_setup["store"]
    now = time.time()
    store.create(
        scope_kind="team", scope_id="content-seo",
        period_start=now - 100, period_end=now,
        narrative_md="x", metrics={}, generated_by="operator",
    )
    store.create(
        scope_kind="team", scope_id="growth",
        period_start=now - 100, period_end=now,
        narrative_md="x", metrics={}, generated_by="operator",
    )
    app = mesh_setup["app"]
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.get(
            "/mesh/work-summaries",
            headers=_hdr(mesh_setup["tokens"]["operator"]),
        )
    assert resp.status_code == 200
    summaries = resp.json()["summaries"]
    scope_ids = {s["scope_id"] for s in summaries}
    assert scope_ids == {"content-seo", "growth"}


@pytest.mark.asyncio
async def test_worker_sees_only_own_team_summaries(mesh_setup):
    store = mesh_setup["store"]
    now = time.time()
    store.create(
        scope_kind="team", scope_id="content-seo",
        period_start=now - 100, period_end=now,
        narrative_md="x", metrics={}, generated_by="operator",
    )
    store.create(
        scope_kind="team", scope_id="growth",  # scout not a member
        period_start=now - 100, period_end=now,
        narrative_md="x", metrics={}, generated_by="operator",
    )
    app = mesh_setup["app"]
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.get(
            "/mesh/work-summaries",
            headers=_hdr(mesh_setup["tokens"]["scout"]),
        )
    assert resp.status_code == 200
    visible = resp.json()["summaries"]
    assert len(visible) == 1
    assert visible[0]["scope_id"] == "content-seo"


@pytest.mark.asyncio
async def test_solo_summary_visible_only_to_self_and_operator(mesh_setup):
    store = mesh_setup["store"]
    now = time.time()
    store.create(
        scope_kind="solo", scope_id="writer",
        period_start=now - 100, period_end=now,
        narrative_md="solo work", metrics={}, generated_by="operator",
    )
    app = mesh_setup["app"]
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        # writer (the solo agent) sees its own summary
        w_resp = await client.get(
            "/mesh/work-summaries",
            headers=_hdr(mesh_setup["tokens"]["writer"]),
        )
        # scout (different agent, no team match for solo scope) does NOT
        s_resp = await client.get(
            "/mesh/work-summaries",
            headers=_hdr(mesh_setup["tokens"]["scout"]),
        )
    assert len(w_resp.json()["summaries"]) == 1
    assert len(s_resp.json()["summaries"]) == 0


# =============================================================================
# Get single
# =============================================================================


@pytest.mark.asyncio
async def test_get_summary_by_id(mesh_setup):
    store = mesh_setup["store"]
    now = time.time()
    row = store.create(
        scope_kind="team", scope_id="content-seo",
        period_start=now - 100, period_end=now,
        narrative_md="detail test", metrics={"x": 1},
        generated_by="operator",
    )
    app = mesh_setup["app"]
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.get(
            f"/mesh/work-summaries/{row['id']}",
            headers=_hdr(mesh_setup["tokens"]["operator"]),
        )
    assert resp.status_code == 200
    assert resp.json()["narrative_md"] == "detail test"


@pytest.mark.asyncio
async def test_get_missing_summary_returns_404(mesh_setup):
    app = mesh_setup["app"]
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.get(
            "/mesh/work-summaries/ws_does_not_exist",
            headers=_hdr(mesh_setup["tokens"]["operator"]),
        )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_summary_scope_404_for_outsider(mesh_setup):
    # L16: an unauthorized caller gets the SAME 404 as a non-existent
    # summary so the endpoint can't be used as an existence oracle.
    store = mesh_setup["store"]
    now = time.time()
    row = store.create(
        scope_kind="team", scope_id="growth",  # scout not a member
        period_start=now - 100, period_end=now,
        narrative_md="cross-team", metrics={}, generated_by="operator",
    )
    app = mesh_setup["app"]
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.get(
            f"/mesh/work-summaries/{row['id']}",
            headers=_hdr(mesh_setup["tokens"]["scout"]),
        )
    assert resp.status_code == 404


# =============================================================================
# Rating
# =============================================================================


@pytest.mark.asyncio
async def test_operator_can_rate_summary(mesh_setup):
    store = mesh_setup["store"]
    now = time.time()
    row = store.create(
        scope_kind="team", scope_id="content-seo",
        period_start=now - 100, period_end=now,
        narrative_md="x", metrics={}, generated_by="operator",
    )
    app = mesh_setup["app"]
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post(
            f"/mesh/work-summaries/{row['id']}/rating",
            json={"rating": "accepted", "feedback": "good"},
            headers=_hdr(mesh_setup["tokens"]["operator"]),
        )
    assert resp.status_code == 200, resp.text
    rated = resp.json()
    assert rated["rating"] == "accepted"
    assert rated["feedback"] == "good"
    assert rated["rated_by"] == "operator"


@pytest.mark.asyncio
async def test_rate_team_summary_writes_track_record_event_team_scoped(mesh_setup):
    """Plan §8 #18: a team-scoped rating has no single rated agent —
    ``agent_id`` is None, ``team_id`` carries the scope. This is the
    mesh (agent-reachable) path, so ``rater_kind`` is "operator_agent"."""
    store = mesh_setup["store"]
    now = time.time()
    row = store.create(
        scope_kind="team", scope_id="content-seo",
        period_start=now - 100, period_end=now,
        narrative_md="x", metrics={}, generated_by="operator",
    )
    app = mesh_setup["app"]
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post(
            f"/mesh/work-summaries/{row['id']}/rating",
            json={"rating": "accepted", "feedback": "good"},
            headers=_hdr(mesh_setup["tokens"]["operator"]),
        )
    assert resp.status_code == 200, resp.text
    events = app.track_record_store.recent_events("content-seo")
    assert events == []  # team scope: nothing under the team's own "agent" bucket
    counts = app.track_record_store.counts_for_agent("content-seo")
    assert counts == {}
    # Query by team_id directly via a raw event since counts_for_agent
    # is agent-keyed; confirm the row landed with team_id set instead.
    with app.track_record_store._conn() as conn:
        row_db = conn.execute(
            "SELECT agent_id, team_id, rater_kind FROM outcome_events WHERE source='summary_rating'"
        ).fetchone()
    assert row_db == (None, "content-seo", "operator_agent")


@pytest.mark.asyncio
async def test_rate_solo_summary_writes_track_record_event_agent_scoped(mesh_setup):
    """Plan §8 #18: a solo-scoped rating's scope_id IS the agent id."""
    store = mesh_setup["store"]
    now = time.time()
    row = store.create(
        scope_kind="solo", scope_id="writer",
        period_start=now - 100, period_end=now,
        narrative_md="solo work", metrics={}, generated_by="operator",
    )
    app = mesh_setup["app"]
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post(
            f"/mesh/work-summaries/{row['id']}/rating",
            json={"rating": "accepted"},
            headers=_hdr(mesh_setup["tokens"]["operator"]),
        )
    assert resp.status_code == 200, resp.text
    events = app.track_record_store.recent_events("writer")
    assert len(events) == 1
    assert events[0]["source"] == "summary_rating"
    assert events[0]["outcome"] == "accepted"
    assert events[0]["rater_kind"] == "operator_agent"
    assert events[0]["team_id"] is None


@pytest.mark.asyncio
async def test_worker_cannot_rate_summary(mesh_setup):
    store = mesh_setup["store"]
    now = time.time()
    row = store.create(
        scope_kind="team", scope_id="content-seo",
        period_start=now - 100, period_end=now,
        narrative_md="x", metrics={}, generated_by="operator",
    )
    app = mesh_setup["app"]
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post(
            f"/mesh/work-summaries/{row['id']}/rating",
            json={"rating": "accepted"},
            headers=_hdr(mesh_setup["tokens"]["scout"]),
        )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_rating_invalid_value_rejected(mesh_setup):
    store = mesh_setup["store"]
    now = time.time()
    row = store.create(
        scope_kind="team", scope_id="content-seo",
        period_start=now - 100, period_end=now,
        narrative_md="x", metrics={}, generated_by="operator",
    )
    app = mesh_setup["app"]
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post(
            f"/mesh/work-summaries/{row['id']}/rating",
            json={"rating": "fantastic"},
            headers=_hdr(mesh_setup["tokens"]["operator"]),
        )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_rating_missing_summary_404(mesh_setup):
    app = mesh_setup["app"]
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post(
            "/mesh/work-summaries/ws_ghost/rating",
            json={"rating": "accepted"},
            headers=_hdr(mesh_setup["tokens"]["operator"]),
        )
    assert resp.status_code == 404
