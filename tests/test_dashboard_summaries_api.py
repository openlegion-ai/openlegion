"""Dashboard /dashboard/api/workplace/summaries proxy tests (PR-B).

Mirrors the existing dashboard test pattern: build a router with a real
``WorkSummariesStore`` and exercise the proxy routes through ``TestClient``.
The mesh-side endpoint surface is already tested in ``test_summaries_api``;
this file pins the dashboard-side translation layer + CSRF gate.
"""

from __future__ import annotations

import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.host.costs import CostTracker
from src.host.mesh import Blackboard
from src.host.summaries import WorkSummariesStore
from src.host.traces import TraceStore


class _CSRFTestClient(TestClient):
    """Auto-inject X-Requested-With for state-changing requests."""
    def request(self, method, url, **kwargs):
        if method.upper() not in ("GET", "HEAD", "OPTIONS"):
            headers = kwargs.get("headers") or {}
            headers.setdefault("X-Requested-With", "XMLHttpRequest")
            kwargs["headers"] = headers
        return super().request(method, url, **kwargs)


@pytest.fixture
def client(tmp_path):
    from src.dashboard.server import create_dashboard_router

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    costs = CostTracker(str(tmp_path / "costs.db"))
    traces = TraceStore(str(tmp_path / "traces.db"))
    summaries = WorkSummariesStore(":memory:")

    router = create_dashboard_router(
        blackboard=bb,
        health_monitor=None,
        cost_tracker=costs,
        trace_store=traces,
        event_bus=None,
        agent_registry={},
        summaries_store=summaries,
    )
    app = FastAPI()
    app.include_router(router)
    yield _CSRFTestClient(app), summaries
    bb.close()
    costs.close()
    traces.close()
    summaries.close()


def _seed(store, *, scope_id="content-seo", scope_kind="team", offset=0):
    now = time.time()
    return store.create(
        scope_kind=scope_kind,
        scope_id=scope_id,
        period_start=now - 86400 + offset,
        period_end=now + offset,
        narrative_md=f"## {scope_kind} {scope_id}\n\nTest narrative.",
        metrics={"created": 3, "delivered": 1},
        recommendations=["Test recommendation"],
        generated_by="operator",
    )


# =============================================================================
# List endpoint
# =============================================================================


def test_list_returns_enabled_true_and_summaries(client):
    c, store = client
    _seed(store, scope_id="x")
    _seed(store, scope_id="y", offset=1)
    resp = c.get("/dashboard/api/workplace/summaries")
    assert resp.status_code == 200
    data = resp.json()
    assert data["enabled"] is True
    assert len(data["summaries"]) == 2


def test_list_without_summaries_store_returns_disabled(tmp_path):
    """When summaries_store isn't wired, the proxy returns
    ``{enabled: False, summaries: []}`` so the UI degrades cleanly
    instead of 500-ing."""
    from src.dashboard.server import create_dashboard_router

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    costs = CostTracker(str(tmp_path / "costs.db"))
    traces = TraceStore(str(tmp_path / "traces.db"))
    router = create_dashboard_router(
        blackboard=bb, health_monitor=None, cost_tracker=costs,
        trace_store=traces, event_bus=None, agent_registry={},
        # no summaries_store
    )
    app = FastAPI()
    app.include_router(router)
    c = _CSRFTestClient(app)
    resp = c.get("/dashboard/api/workplace/summaries")
    assert resp.status_code == 200
    data = resp.json()
    assert data["enabled"] is False
    assert data["summaries"] == []
    bb.close()
    costs.close()
    traces.close()


def test_list_filters_by_scope_kind(client):
    c, store = client
    _seed(store, scope_id="team-a", scope_kind="team")
    _seed(store, scope_id="solo-a", scope_kind="solo", offset=1)
    resp = c.get("/dashboard/api/workplace/summaries?scope_kind=solo")
    assert resp.status_code == 200
    rows = resp.json()["summaries"]
    assert len(rows) == 1
    assert rows[0]["scope_kind"] == "solo"


# =============================================================================
# Detail endpoint
# =============================================================================


def test_detail_returns_summary(client):
    c, store = client
    row = _seed(store)
    resp = c.get(f"/dashboard/api/workplace/summaries/{row['id']}")
    assert resp.status_code == 200
    detail = resp.json()
    assert detail["id"] == row["id"]
    assert detail["scope_id"] == "content-seo"


def test_detail_missing_summary_404(client):
    c, _ = client
    resp = c.get("/dashboard/api/workplace/summaries/ws_doesnotexist")
    assert resp.status_code == 404


# =============================================================================
# Rating endpoint
# =============================================================================


def test_rate_accepted_persists(client):
    c, store = client
    row = _seed(store)
    resp = c.post(
        f"/dashboard/api/workplace/summaries/{row['id']}/rating",
        json={"rating": "accepted"},
    )
    assert resp.status_code == 200, resp.text
    rated = resp.json()
    assert rated["rating"] == "accepted"
    assert rated["rated_by"] == "operator"  # dashboard persona


def test_rate_rework_with_feedback(client):
    c, store = client
    row = _seed(store)
    resp = c.post(
        f"/dashboard/api/workplace/summaries/{row['id']}/rating",
        json={"rating": "rework", "feedback": "Focus on stage-4 publish"},
    )
    assert resp.status_code == 200
    rated = resp.json()
    assert rated["rating"] == "rework"
    assert "stage-4" in rated["feedback"]


def test_rate_rework_without_feedback_400(client):
    c, store = client
    row = _seed(store)
    resp = c.post(
        f"/dashboard/api/workplace/summaries/{row['id']}/rating",
        json={"rating": "rework"},
    )
    assert resp.status_code == 400


def test_rate_invalid_rating_400(client):
    c, store = client
    row = _seed(store)
    resp = c.post(
        f"/dashboard/api/workplace/summaries/{row['id']}/rating",
        json={"rating": "amazing"},
    )
    assert resp.status_code == 400


def test_rate_missing_summary_404(client):
    c, _ = client
    resp = c.post(
        "/dashboard/api/workplace/summaries/ws_ghost/rating",
        json={"rating": "accepted"},
    )
    assert resp.status_code == 404


def test_rate_requires_csrf_header(tmp_path):
    """State-changing requests without X-Requested-With are blocked
    by the router-level CSRF guard."""
    from src.dashboard.server import create_dashboard_router

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    costs = CostTracker(str(tmp_path / "costs.db"))
    traces = TraceStore(str(tmp_path / "traces.db"))
    summaries = WorkSummariesStore(":memory:")
    row = summaries.create(
        scope_kind="team", scope_id="x",
        period_start=0, period_end=1,
        narrative_md="n", metrics={}, generated_by="operator",
    )
    router = create_dashboard_router(
        blackboard=bb, health_monitor=None, cost_tracker=costs,
        trace_store=traces, event_bus=None, agent_registry={},
        summaries_store=summaries,
    )
    app = FastAPI()
    app.include_router(router)
    # Plain TestClient — no auto X-Requested-With header.
    plain = TestClient(app)
    resp = plain.post(
        f"/dashboard/api/workplace/summaries/{row['id']}/rating",
        json={"rating": "accepted"},
    )
    assert resp.status_code == 403  # CSRF rejection
    bb.close()
    costs.close()
    traces.close()
    summaries.close()
