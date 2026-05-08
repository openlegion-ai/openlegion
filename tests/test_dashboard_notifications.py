"""Tests for the persistent dashboard notifications store + API.

Covers Phase 2 of the Board UX overhaul (`docs/plans/2026-05-08-board-ux-overhaul.md`).
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.dashboard.events import EventBus
from src.dashboard.notifications import NotificationStore
from src.host.costs import CostTracker
from src.host.health import HealthMonitor
from src.host.mesh import Blackboard
from src.host.traces import TraceStore

# ── Pure-store unit tests ────────────────────────────────────────────


class TestNotificationStore:
    def setup_method(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(self._tmpdir, "notifications.db")
        self.store = NotificationStore(db_path=db_path)

    def teardown_method(self) -> None:
        self.store.close()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_add_and_list_returns_inserted_row(self):
        nid = self.store.add(kind="delivered", title="Researcher delivered brief")
        assert nid > 0
        rows = self.store.list_recent()
        assert len(rows) == 1
        assert rows[0]["id"] == nid
        assert rows[0]["title"] == "Researcher delivered brief"
        assert rows[0]["kind"] == "delivered"
        assert rows[0]["read_at"] is None

    def test_list_orders_unread_first_then_ts_desc(self):
        # Insert: A unread (oldest), B read, C unread (newest)
        a = self.store.add(kind="info", title="A", ts=100.0)
        b = self.store.add(kind="info", title="B", ts=200.0)
        c = self.store.add(kind="info", title="C", ts=300.0)
        self.store.mark_read(b)
        rows = self.store.list_recent()
        # Expect: C (unread, newest), A (unread, oldest), B (read)
        assert [r["id"] for r in rows] == [c, a, b]

    def test_mark_read_persists_and_idempotent(self):
        nid = self.store.add(kind="info", title="hi")
        assert self.store.mark_read(nid) is True
        # Second call should be a no-op (already read).
        assert self.store.mark_read(nid) is False
        rows = self.store.list_recent()
        assert rows[0]["read_at"] is not None

    def test_mark_all_read(self):
        ids = [self.store.add(kind="info", title=f"n{i}") for i in range(5)]
        count = self.store.mark_all_read()
        assert count == 5
        for r in self.store.list_recent():
            assert r["read_at"] is not None
        # Idempotent.
        assert self.store.mark_all_read() == 0
        # Manual sanity-check on the inserted ids.
        assert len(ids) == 5

    def test_unread_count(self):
        self.store.add(kind="info", title="a")
        self.store.add(kind="info", title="b")
        nid = self.store.add(kind="info", title="c")
        self.store.mark_read(nid)
        assert self.store.unread_count() == 2

    def test_payload_roundtrip(self):
        self.store.add(
            kind="delivered",
            title="task done",
            payload={"task_id": "t-1", "agent_id": "writer"},
        )
        rows = self.store.list_recent()
        assert rows[0]["payload"] == {"task_id": "t-1", "agent_id": "writer"}

    def test_limit_clamped(self):
        for i in range(15):
            self.store.add(kind="info", title=f"n{i}")
        rows = self.store.list_recent(limit=5)
        assert len(rows) == 5

    def test_unknown_kind_accepted(self):
        # Unknown kinds are accepted (logged) — see notifications.py docstring.
        nid = self.store.add(kind="experimental_kind", title="hello")
        assert nid > 0

    def test_empty_title_rejected(self):
        with pytest.raises(ValueError):
            self.store.add(kind="info", title="")


# ── HTTP endpoint tests ──────────────────────────────────────────────


def _make_components(tmp_path: str) -> dict:
    bb = Blackboard(db_path=os.path.join(tmp_path, "bb.db"))
    cost_tracker = CostTracker(db_path=os.path.join(tmp_path, "costs.db"))
    trace_store = TraceStore(db_path=os.path.join(tmp_path, "traces.db"))
    event_bus = EventBus()

    from unittest.mock import MagicMock

    runtime_mock = MagicMock()
    runtime_mock.browser_vnc_url = None
    runtime_mock.browser_service_url = None
    runtime_mock.browser_auth_token = ""
    transport_mock = MagicMock()
    router_mock = MagicMock()
    health_monitor = HealthMonitor(
        runtime=runtime_mock, transport=transport_mock, router=router_mock,
    )
    notification_store = NotificationStore(
        db_path=os.path.join(tmp_path, "notifications.db"),
    )
    return {
        "blackboard": bb,
        "health_monitor": health_monitor,
        "cost_tracker": cost_tracker,
        "trace_store": trace_store,
        "event_bus": event_bus,
        "agent_registry": {},
        "notification_store": notification_store,
    }


def _make_client(components: dict) -> TestClient:
    from src.dashboard.server import create_dashboard_router

    router = create_dashboard_router(**components, mesh_port=8420)
    app = FastAPI()
    app.include_router(router)

    class _CSRFTestClient(TestClient):
        def request(self, method, url, **kwargs):
            if method.upper() not in ("GET", "HEAD", "OPTIONS"):
                headers = kwargs.get("headers") or {}
                if "X-Requested-With" not in headers:
                    headers["X-Requested-With"] = "XMLHttpRequest"
                    kwargs["headers"] = headers
            return super().request(method, url, **kwargs)

    return _CSRFTestClient(app)


def _teardown(components: dict) -> None:
    components["cost_tracker"].close()
    components["trace_store"].close()
    components["blackboard"].close()
    components["notification_store"].close()


class TestNotificationsAPI:
    def setup_method(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self.components = _make_components(self._tmpdir)
        self.store: NotificationStore = self.components["notification_store"]
        self.client = _make_client(self.components)

    def teardown_method(self) -> None:
        _teardown(self.components)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_get_notifications_empty(self):
        resp = self.client.get("/dashboard/api/notifications")
        assert resp.status_code == 200
        body = resp.json()
        assert body == {"notifications": [], "unread_count": 0}

    def test_get_notifications_returns_inserted_row(self):
        nid = self.store.add(kind="delivered", title="Brief delivered", body="By writer")
        resp = self.client.get("/dashboard/api/notifications")
        assert resp.status_code == 200
        body = resp.json()
        assert body["unread_count"] == 1
        assert len(body["notifications"]) == 1
        row = body["notifications"][0]
        assert row["id"] == nid
        assert row["title"] == "Brief delivered"
        assert row["body"] == "By writer"
        assert row["read_at"] is None

    def test_post_mark_read_updates_unread_count(self):
        nid = self.store.add(kind="info", title="hi")
        resp = self.client.post(f"/dashboard/api/notifications/{nid}/read")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["id"] == nid
        assert body["changed"] is True
        # Subsequent GET reflects the read state.
        snap = self.client.get("/dashboard/api/notifications").json()
        assert snap["unread_count"] == 0
        assert snap["notifications"][0]["read_at"] is not None

    def test_post_mark_read_idempotent_returns_ok(self):
        nid = self.store.add(kind="info", title="hi")
        self.store.mark_read(nid)
        # Already read — endpoint still returns OK with changed=False.
        resp = self.client.post(f"/dashboard/api/notifications/{nid}/read")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["changed"] is False

    def test_post_read_all(self):
        for i in range(3):
            self.store.add(kind="info", title=f"n{i}")
        resp = self.client.post("/dashboard/api/notifications/read-all")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["marked"] == 3
        assert self.client.get("/dashboard/api/notifications").json()["unread_count"] == 0

    def test_post_mark_read_requires_csrf_header(self):
        nid = self.store.add(kind="info", title="hi")
        # Bypass the auto-CSRF wrapper to assert the dashboard CSRF guard.
        from fastapi.testclient import TestClient

        from src.dashboard.server import create_dashboard_router

        router = create_dashboard_router(**self.components, mesh_port=8420)
        app = FastAPI()
        app.include_router(router)
        plain = TestClient(app)
        resp = plain.post(f"/dashboard/api/notifications/{nid}/read")
        assert resp.status_code == 403


# ── Auto-instantiation ────────────────────────────────────────────────


class TestNotificationsAutoInstantiation:
    """When no notification_store is passed, the dashboard router still
    exposes the endpoints — the constructor builds a default store."""

    def setup_method(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self.cwd = os.getcwd()
        os.chdir(self._tmpdir)  # ensure the default ``data/`` path is local
        from unittest.mock import MagicMock

        from src.dashboard.events import EventBus
        from src.host.costs import CostTracker
        from src.host.health import HealthMonitor
        from src.host.mesh import Blackboard
        from src.host.traces import TraceStore

        self.bb = Blackboard(db_path=os.path.join(self._tmpdir, "bb.db"))
        self.cost = CostTracker(db_path=os.path.join(self._tmpdir, "costs.db"))
        self.traces = TraceStore(db_path=os.path.join(self._tmpdir, "traces.db"))
        self.event_bus = EventBus()
        runtime_mock = MagicMock()
        runtime_mock.browser_vnc_url = None
        runtime_mock.browser_service_url = None
        runtime_mock.browser_auth_token = ""
        self.health = HealthMonitor(
            runtime=runtime_mock, transport=MagicMock(), router=MagicMock(),
        )

        from src.dashboard.server import create_dashboard_router
        # Note: no notification_store passed.
        router = create_dashboard_router(
            blackboard=self.bb,
            health_monitor=self.health,
            cost_tracker=self.cost,
            trace_store=self.traces,
            event_bus=self.event_bus,
            agent_registry={},
            mesh_port=8420,
        )
        app = FastAPI()
        app.include_router(router)
        self.client = TestClient(app)

    def teardown_method(self) -> None:
        self.cost.close()
        self.traces.close()
        self.bb.close()
        os.chdir(self.cwd)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_auto_instantiated_endpoints_return_empty(self):
        resp = self.client.get("/dashboard/api/notifications")
        assert resp.status_code == 200
        body = resp.json()
        assert body == {"notifications": [], "unread_count": 0}


def test_default_db_path_directory_created(tmp_path: Path):
    db_path = tmp_path / "subdir" / "notifications.db"
    store = NotificationStore(db_path=str(db_path))
    try:
        assert db_path.exists()
        store.add(kind="info", title="hi")
        assert store.unread_count() == 1
    finally:
        store.close()
