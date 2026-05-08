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


# ── Producer wiring (EventBus → NotificationStore) ───────────────────


class TestNotificationsProducerWiring:
    """When the dashboard router is built with an EventBus, it registers
    a listener that translates relevant events into notification rows.

    These tests exercise the listener directly via ``event_bus.emit``
    (which calls listeners synchronously on the caller's stack) and
    then read the store back through the public API. We never assert on
    private state — the contract is "fire event → row appears in
    ``list_recent``".
    """

    def setup_method(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self.components = _make_components(self._tmpdir)
        self.store: NotificationStore = self.components["notification_store"]
        # Building the router registers the producer listener as a side
        # effect; we don't need to keep the client/router around.
        from src.dashboard.server import create_dashboard_router
        create_dashboard_router(**self.components, mesh_port=8420)
        self.bus = self.components["event_bus"]

    def teardown_method(self) -> None:
        _teardown(self.components)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_task_outcome_delivered_creates_notification(self):
        self.bus.emit("task_outcome", agent="operator", data={
            "task_id": "t-1",
            "project_id": "proj-a",
            "assignee": "writer",
            "status": "done",
            "outcome": "delivered",
            "feedback": "Looks good",
        })
        rows = self.store.list_recent()
        assert len(rows) == 1
        row = rows[0]
        assert row["kind"] == "delivered"
        assert "writer" in row["title"]
        assert "delivered" in row["title"].lower()
        assert row["payload"]["task_id"] == "t-1"

    def test_task_outcome_rejected_does_not_create_delivery_notification(self):
        # Only "delivered" outcomes surface — rejection / rework / re-rate
        # are operator-initiated and surface elsewhere (task drawer).
        self.bus.emit("task_outcome", agent="operator", data={
            "task_id": "t-2",
            "outcome": "rejected",
            "assignee": "writer",
        })
        self.bus.emit("task_outcome", agent="operator", data={
            "task_id": "t-3",
            "outcome": "needs_rework",
            "assignee": "writer",
        })
        assert self.store.list_recent() == []

    def test_pending_action_created_creates_notification(self):
        self.bus.emit("pending_action_created", agent="operator", data={
            "nonce": "abc123",
            "actor": "operator",
            "target_kind": "agent",
            "target_id": "writer",
            "action_kind": "edit_agent",
            "summary": "Switch writer's model from gpt-4o to claude-opus",
            "preview_diff": "- model: gpt-4o\n+ model: claude-opus",
        })
        rows = self.store.list_recent()
        assert len(rows) == 1
        assert rows[0]["kind"] == "approval"
        assert "Approval needed" in rows[0]["title"]
        assert rows[0]["payload"]["nonce"] == "abc123"

    def test_credential_request_creates_notification(self):
        self.bus.emit("credential_request", agent="researcher", data={
            "name": "google_api_key",
            "service": "Google API",
            "description": "Needed for Custom Search.",
            "request_id": "req-1",
        })
        rows = self.store.list_recent()
        assert len(rows) == 1
        assert rows[0]["kind"] == "credential"
        assert "researcher" in rows[0]["title"]
        assert "Google API" in rows[0]["title"]
        assert rows[0]["payload"]["request_id"] == "req-1"

    def test_browser_login_request_creates_notification(self):
        self.bus.emit("browser_login_request", agent="researcher", data={
            "url": "https://www.linkedin.com/login",
            "service": "LinkedIn",
            "description": "Needed for outreach research.",
            "request_id": "br-1",
        })
        rows = self.store.list_recent()
        assert len(rows) == 1
        assert rows[0]["kind"] == "credential"
        assert "researcher" in rows[0]["title"]
        assert "sign-in" in rows[0]["title"]
        assert "LinkedIn" in rows[0]["title"]
        assert rows[0]["payload"]["request_id"] == "br-1"

    def test_health_change_to_degraded_creates_notification(self):
        self.bus.emit("health_change", agent="researcher", data={
            "previous": "healthy",
            "current": "unhealthy",
            "failures": 3,
        })
        rows = self.store.list_recent()
        assert len(rows) == 1
        assert rows[0]["kind"] == "alert"
        assert "researcher" in rows[0]["title"]
        assert "unhealthy" in rows[0]["title"]

    def test_health_change_to_healthy_does_not_create_notification(self):
        # Recovery flips ARE an event but they don't need user
        # attention — the bell only surfaces degradations.
        self.bus.emit("health_change", agent="researcher", data={
            "previous": "unhealthy",
            "current": "healthy",
            "failures": 0,
        })
        assert self.store.list_recent() == []

    def test_credit_exhausted_creates_notification(self):
        self.bus.emit("credit_exhausted", agent="writer", data={
            "error": "Insufficient credits",
        })
        rows = self.store.list_recent()
        assert len(rows) == 1
        assert rows[0]["kind"] == "alert"
        assert "writer" in rows[0]["title"]
        assert "out of credit" in rows[0]["title"]

    def test_unrelated_event_does_not_create_notification(self):
        # Sanity check — events outside the frozen mapping table never
        # create rows. Without this we'd be tempted to grow the mapping
        # ad-hoc.
        self.bus.emit("tool_start", agent="writer", data={"tool": "browser_navigate"})
        self.bus.emit("blackboard_write", agent="writer", data={"key": "shared/note"})
        assert self.store.list_recent() == []
