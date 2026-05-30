"""HTTP-edge security hardening (M18 / L11 / M19).

Covers the mesh-app baseline security headers (L11) and the env-gated
API-docs / OpenAPI exposure (M19). M18 (dashboard clickjacking headers)
is covered in tests/test_dashboard.py.
"""

from fastapi.testclient import TestClient

from src.host.costs import CostTracker
from src.host.mesh import Blackboard, MessageRouter, PubSub
from src.host.permissions import PermissionMatrix
from src.host.traces import TraceStore


def _make_mesh_app(tmp_path):
    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix()
    router = MessageRouter(perms, {})
    costs = CostTracker(str(tmp_path / "costs.db"))
    traces = TraceStore(str(tmp_path / "traces.db"))
    router.register_agent("operator", "http://operator:8400", [])

    # The docs gate (M19) is read inside create_mesh_app, so the current
    # process env applies at construction time — no module reload needed.
    from src.host.server import create_mesh_app

    app = create_mesh_app(
        blackboard=bb, pubsub=pubsub, router=router, permissions=perms,
        cost_tracker=costs, trace_store=traces,
    )
    cleanup = lambda: (bb.close(), costs.close(), traces.close())
    return app, cleanup


def test_mesh_response_has_security_headers(tmp_path):
    # L11: every mesh response carries the baseline security headers.
    app, cleanup = _make_mesh_app(tmp_path)
    try:
        client = TestClient(app)
        resp = client.get("/mesh/agents", headers={"x-mesh-internal": "1"})
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        assert (
            resp.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
        )
        assert "Permissions-Policy" in resp.headers
    finally:
        cleanup()


def test_mesh_security_headers_on_error_response(tmp_path):
    # Headers must be stamped even on non-2xx / unknown routes — the outer
    # middleware runs regardless of route outcome.
    app, cleanup = _make_mesh_app(tmp_path)
    try:
        client = TestClient(app)
        resp = client.get("/mesh/this-route-does-not-exist")
        assert resp.status_code == 404
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
    finally:
        cleanup()


def test_openapi_disabled_by_default(tmp_path, monkeypatch):
    # M19: OpenAPI schema + docs are off unless OPENLEGION_ENABLE_DOCS is set.
    monkeypatch.delenv("OPENLEGION_ENABLE_DOCS", raising=False)
    app, cleanup = _make_mesh_app(tmp_path)
    try:
        client = TestClient(app)
        assert client.get("/openapi.json").status_code == 404
        assert client.get("/docs").status_code == 404
        assert client.get("/redoc").status_code == 404
    finally:
        cleanup()


def test_openapi_enabled_via_env(tmp_path, monkeypatch):
    # M19: flipping the env flag restores the default FastAPI docs for dev.
    monkeypatch.setenv("OPENLEGION_ENABLE_DOCS", "1")
    app, cleanup = _make_mesh_app(tmp_path)
    try:
        client = TestClient(app)
        assert client.get("/openapi.json").status_code == 200
        assert client.get("/docs").status_code == 200
    finally:
        cleanup()
