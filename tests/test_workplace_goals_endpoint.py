"""Tests for ``GET /api/workplace/goals`` (PR 1 of Work tab rewrite).

The endpoint reads ``GOALS.json`` from the operator container's workspace
via the transport proxy. These tests pin the four states the frontend
needs to handle: transport unavailable, operator not registered,
transport raises, and the happy path.
"""

from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.host.costs import CostTracker
from src.host.mesh import Blackboard
from src.host.summaries import WorkSummariesStore
from src.host.traces import TraceStore


class _FakeTransport:
    """Minimal stand-in for HttpTransport."""

    def __init__(self, response=None, exc: Exception | None = None):
        self._response = response
        self._exc = exc
        self.calls: list[tuple[str, str, str]] = []

    async def request(self, agent_id, method, path, **_kwargs):
        self.calls.append((agent_id, method, path))
        if self._exc is not None:
            raise self._exc
        return self._response


def _build_app(*, transport=None, agent_registry=None, tmp_path=None):
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
        agent_registry=agent_registry or {},
        summaries_store=summaries,
        transport=transport,
    )
    app = FastAPI()
    app.include_router(router)

    def _cleanup():
        bb.close()
        costs.close()
        traces.close()
        summaries.close()

    return app, _cleanup


def test_goals_endpoint_transport_unavailable(tmp_path):
    app, cleanup = _build_app(transport=None, tmp_path=tmp_path)
    try:
        with TestClient(app) as c:
            r = c.get("/dashboard/api/workplace/goals")
            assert r.status_code == 200
            assert r.json() == {"enabled": False, "goals": []}
    finally:
        cleanup()


def test_goals_endpoint_operator_not_registered(tmp_path):
    # transport present but the operator container isn't in the registry
    fake = _FakeTransport(response={"content": "{}"})
    app, cleanup = _build_app(
        transport=fake, agent_registry={}, tmp_path=tmp_path,
    )
    try:
        with TestClient(app) as c:
            r = c.get("/dashboard/api/workplace/goals")
            assert r.status_code == 200
            assert r.json() == {"enabled": False, "goals": []}
            # Endpoint short-circuits before hitting transport.
            assert fake.calls == []
    finally:
        cleanup()


def test_goals_endpoint_happy_path(tmp_path):
    payload = {
        "goals": [
            {
                "name": "Launch newsletter",
                "status": "in_progress",
                "progress_note": "drafts approved",
                "updated_at": "2026-05-28T12:00:00+00:00",
            },
            {
                "name": "Audit accounts",
                "status": "on_track",
                "progress_note": "",
                "updated_at": "2026-05-28T13:00:00+00:00",
            },
        ],
    }
    fake = _FakeTransport(response={"content": json.dumps(payload)})
    app, cleanup = _build_app(
        transport=fake,
        agent_registry={"operator": "http://operator:8400"},
        tmp_path=tmp_path,
    )
    try:
        with TestClient(app) as c:
            r = c.get("/dashboard/api/workplace/goals")
            assert r.status_code == 200
            body = r.json()
            assert body["enabled"] is True
            assert len(body["goals"]) == 2
            assert body["goals"][0]["name"] == "Launch newsletter"
            assert body["goals"][0]["status"] == "in_progress"
            assert body["goals"][1]["name"] == "Audit accounts"
            # Endpoint hit the operator workspace.
            assert fake.calls == [("operator", "GET", "/workspace/GOALS.json")]
    finally:
        cleanup()


def test_goals_endpoint_empty_file(tmp_path):
    """No goals tracked yet — file present but empty/whitespace content."""
    fake = _FakeTransport(response={"content": ""})
    app, cleanup = _build_app(
        transport=fake,
        agent_registry={"operator": "http://operator:8400"},
        tmp_path=tmp_path,
    )
    try:
        with TestClient(app) as c:
            r = c.get("/dashboard/api/workplace/goals")
            assert r.status_code == 200
            assert r.json() == {"enabled": True, "goals": []}
    finally:
        cleanup()


def test_goals_endpoint_malformed_json(tmp_path):
    """Corrupt sidecar shouldn't blow up the Work tab."""
    fake = _FakeTransport(response={"content": "this is not json"})
    app, cleanup = _build_app(
        transport=fake,
        agent_registry={"operator": "http://operator:8400"},
        tmp_path=tmp_path,
    )
    try:
        with TestClient(app) as c:
            r = c.get("/dashboard/api/workplace/goals")
            assert r.status_code == 200
            assert r.json() == {"enabled": True, "goals": []}
    finally:
        cleanup()


def test_goals_endpoint_transport_raises(tmp_path):
    """Transport failure surfaces but doesn't 500, and the error message
    is generic so internal hostnames / paths from httpx don't leak."""
    fake = _FakeTransport(
        exc=RuntimeError("Connection refused at http://operator-internal:8400"),
    )
    app, cleanup = _build_app(
        transport=fake,
        agent_registry={"operator": "http://operator:8400"},
        tmp_path=tmp_path,
    )
    try:
        with TestClient(app) as c:
            r = c.get("/dashboard/api/workplace/goals")
            assert r.status_code == 200
            body = r.json()
            assert body["enabled"] is True
            assert body["goals"] == []
            assert body.get("error") == "unable to reach operator"
            # Internal host/path must NOT leak to the browser.
            assert "operator-internal" not in body.get("error", "")
    finally:
        cleanup()


def test_dashboard_workspace_put_rejects_goals_files(tmp_path):
    """Lock in the security fix: the workspace editor's PUT proxy must
    refuse GOALS.md and GOALS.json so a cookie-authed user can't write
    raw JSON around the ``manage_goals`` tool's validation. Goals are
    read via the dedicated /api/workplace/goals endpoint; there is no
    legitimate write path through the workspace proxy.
    """
    fake = _FakeTransport(response={"filename": "x", "size": 0})
    app, cleanup = _build_app(
        transport=fake,
        agent_registry={"operator": "http://operator:8400"},
        tmp_path=tmp_path,
    )
    try:
        with TestClient(app):
            # Use the CSRF-injecting client so the request itself isn't
            # rejected on CSRF grounds — we want the allowlist check
            # to fire.
            from fastapi.testclient import TestClient as _TC

            class _CSRFClient(_TC):
                def request(self, method, url, **kw):
                    if method.upper() not in ("GET", "HEAD", "OPTIONS"):
                        h = kw.get("headers") or {}
                        h.setdefault("X-Requested-With", "XMLHttpRequest")
                        kw["headers"] = h
                    return super().request(method, url, **kw)

            csrf_client = _CSRFClient(app)
            for filename in ("GOALS.json", "GOALS.md"):
                r = csrf_client.put(
                    f"/dashboard/api/agents/operator/workspace/{filename}",
                    json={"content": '{"goals": []}'},
                )
                assert r.status_code == 400, (
                    f"PUT {filename} should be rejected by allowlist, "
                    f"got {r.status_code}: {r.text}"
                )
                assert "not allowed" in r.text.lower()
            # Transport must NOT have been called for the rejected PUTs.
            assert all(call[1] != "PUT" for call in fake.calls)
    finally:
        cleanup()


def test_goals_endpoint_drops_malformed_entries(tmp_path):
    """Entries that aren't dicts get filtered, not 500'd."""
    payload = {
        "goals": [
            {"name": "Good", "status": "in_progress"},
            "not a dict",
            {"name": "Also good", "status": "done"},
        ],
    }
    fake = _FakeTransport(response={"content": json.dumps(payload)})
    app, cleanup = _build_app(
        transport=fake,
        agent_registry={"operator": "http://operator:8400"},
        tmp_path=tmp_path,
    )
    try:
        with TestClient(app) as c:
            r = c.get("/dashboard/api/workplace/goals")
            assert r.status_code == 200
            body = r.json()
            assert body["enabled"] is True
            names = [g["name"] for g in body["goals"]]
            assert names == ["Good", "Also good"]
    finally:
        cleanup()


def test_goals_endpoint_response_shape_is_stable(tmp_path):
    """Frontend depends on every goal having the four documented keys."""
    payload = {
        "goals": [
            # Missing progress_note + updated_at; should default to "".
            {"name": "Partial", "status": "blocked"},
        ],
    }
    fake = _FakeTransport(response={"content": json.dumps(payload)})
    app, cleanup = _build_app(
        transport=fake,
        agent_registry={"operator": "http://operator:8400"},
        tmp_path=tmp_path,
    )
    try:
        with TestClient(app) as c:
            r = c.get("/dashboard/api/workplace/goals")
            body = r.json()
            assert body["goals"] == [{
                "name": "Partial",
                "status": "blocked",
                "progress_note": "",
                "updated_at": "",
            }]
    finally:
        cleanup()
