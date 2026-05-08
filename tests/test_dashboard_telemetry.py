"""Tests for the Phase -1 onboarding-wizard telemetry surface.

Covers:
  * ``DashboardTelemetry`` SQLite store — record, recent, rate limit,
    payload caps.
  * ``POST /dashboard/api/telemetry`` HTTP endpoint — auth, CSRF,
    validation, rate limit at 60/min per session.
"""

from __future__ import annotations

import os
import shutil
import tempfile

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.dashboard.events import EventBus
from src.dashboard.telemetry import (
    RATE_LIMIT_EVENTS_PER_MIN,
    DashboardTelemetry,
)
from src.host.costs import CostTracker
from src.host.health import HealthMonitor
from src.host.mesh import Blackboard
from src.host.traces import TraceStore


def _make_components(tmp_path: str) -> dict:
    from unittest.mock import MagicMock

    bb = Blackboard(db_path=os.path.join(tmp_path, "bb.db"))
    cost_tracker = CostTracker(db_path=os.path.join(tmp_path, "costs.db"))
    trace_store = TraceStore(db_path=os.path.join(tmp_path, "traces.db"))
    event_bus = EventBus()
    runtime_mock = MagicMock()
    runtime_mock.browser_vnc_url = None
    runtime_mock.browser_service_url = None
    runtime_mock.browser_auth_token = ""
    transport_mock = MagicMock()
    router_mock = MagicMock()
    health_monitor = HealthMonitor(
        runtime=runtime_mock, transport=transport_mock, router=router_mock,
    )
    return {
        "blackboard": bb,
        "health_monitor": health_monitor,
        "cost_tracker": cost_tracker,
        "trace_store": trace_store,
        "event_bus": event_bus,
        "agent_registry": {},
    }


class _CSRFTestClient(TestClient):
    def request(self, method, url, **kwargs):
        if method.upper() not in ("GET", "HEAD", "OPTIONS"):
            headers = kwargs.get("headers") or {}
            if "X-Requested-With" not in headers:
                headers["X-Requested-With"] = "XMLHttpRequest"
                kwargs["headers"] = headers
        return super().request(method, url, **kwargs)


def _make_client(components: dict, telemetry: DashboardTelemetry) -> TestClient:
    from src.dashboard.server import create_dashboard_router

    router = create_dashboard_router(
        **components, mesh_port=8420, telemetry=telemetry,
    )
    app = FastAPI()
    app.include_router(router)
    return _CSRFTestClient(app)


def _teardown(components: dict) -> None:
    components["cost_tracker"].close()
    components["trace_store"].close()
    components["blackboard"].close()


# ── Store-level tests ────────────────────────────────────────────


class TestDashboardTelemetryStore:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.store = DashboardTelemetry(
            db_path=os.path.join(self._tmpdir, "telemetry.db"),
        )

    def teardown_method(self):
        self.store.close()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_record_persists_event(self):
        row_id = self.store.record(
            event_name="wizard_started",
            session_id="op-test",
            props={"startedAt": 12345},
        )
        assert row_id > 0
        events = self.store.recent(event_name="wizard_started")
        assert len(events) == 1
        assert events[0]["event_name"] == "wizard_started"
        assert events[0]["session_id"] == "op-test"
        assert events[0]["props"] == {"startedAt": 12345}

    def test_record_rejects_empty_event_name(self):
        try:
            self.store.record(event_name="", session_id="op")
        except ValueError:
            return
        raise AssertionError("expected ValueError for empty event_name")

    def test_record_rejects_oversized_props(self):
        big = {"data": "x" * 5000}
        try:
            self.store.record(
                event_name="wizard_chip_clicked",
                session_id="op", props=big,
            )
        except ValueError:
            return
        raise AssertionError("expected ValueError for oversized props")

    def test_record_rejects_long_event_name(self):
        try:
            self.store.record(
                event_name="x" * 200, session_id="op",
            )
        except ValueError:
            return
        raise AssertionError("expected ValueError for long event_name")

    def test_rate_limit_allows_under_threshold(self):
        for _ in range(5):
            allowed, retry = self.store.check_rate_limit("op-test")
            assert allowed
            assert retry == 0

    def test_rate_limit_blocks_over_threshold(self):
        # Burn through the budget.
        for _ in range(RATE_LIMIT_EVENTS_PER_MIN):
            allowed, _ = self.store.check_rate_limit("op-burst")
            assert allowed
        # Next call must be denied.
        allowed, retry = self.store.check_rate_limit("op-burst")
        assert not allowed
        assert retry > 0

    def test_rate_limit_isolated_per_session(self):
        for _ in range(RATE_LIMIT_EVENTS_PER_MIN):
            self.store.check_rate_limit("op-a")
        # op-b is independent.
        allowed, _ = self.store.check_rate_limit("op-b")
        assert allowed

    def test_recent_filters_by_event_name(self):
        self.store.record(event_name="wizard_started", session_id="op")
        self.store.record(
            event_name="wizard_chip_clicked",
            session_id="op",
            props={"label": "Build a content team"},
        )
        chips = self.store.recent(event_name="wizard_chip_clicked")
        assert len(chips) == 1
        assert chips[0]["props"]["label"] == "Build a content team"


# ── HTTP endpoint tests ──────────────────────────────────────────


class TestDashboardTelemetryEndpoint:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.components = _make_components(self._tmpdir)
        self.telemetry = DashboardTelemetry(
            db_path=os.path.join(self._tmpdir, "telemetry.db"),
        )
        self.client = _make_client(self.components, self.telemetry)

    def teardown_method(self):
        self.telemetry.close()
        _teardown(self.components)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_post_telemetry_records_event(self):
        resp = self.client.post(
            "/dashboard/api/telemetry",
            json={
                "event_name": "wizard_started",
                "props": {"startedAt": 1700000000},
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["recorded"] is True
        assert body["id"] >= 1
        # Verify it landed in the store.
        rows = self.telemetry.recent(event_name="wizard_started")
        assert len(rows) == 1

    def test_post_telemetry_requires_csrf_header(self):
        # Use raw TestClient to bypass auto-CSRF injection.
        plain = TestClient(self.client.app)
        resp = plain.post(
            "/dashboard/api/telemetry",
            json={"event_name": "wizard_started"},
        )
        assert resp.status_code == 403

    def test_post_telemetry_rejects_missing_event_name(self):
        resp = self.client.post(
            "/dashboard/api/telemetry",
            json={"props": {"foo": "bar"}},
        )
        assert resp.status_code == 400
        assert "event_name" in resp.json()["detail"]

    def test_post_telemetry_rejects_invalid_json(self):
        resp = self.client.post(
            "/dashboard/api/telemetry",
            data="not-json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400

    def test_post_telemetry_rejects_non_object_props(self):
        resp = self.client.post(
            "/dashboard/api/telemetry",
            json={"event_name": "foo", "props": "not-an-object"},
        )
        assert resp.status_code == 400

    def test_post_telemetry_rate_limits_at_60_per_min(self):
        # Burn through the per-session budget. We share a session_id
        # because the dashboard uses ``ol_session`` cookie for that and
        # the TestClient doesn't set one (so all calls map to the same
        # 'operator' session bucket).
        last_status = 200
        for _ in range(RATE_LIMIT_EVENTS_PER_MIN):
            resp = self.client.post(
                "/dashboard/api/telemetry",
                json={"event_name": "wizard_chip_clicked"},
            )
            last_status = resp.status_code
        assert last_status == 200
        # Next call must hit 429.
        resp = self.client.post(
            "/dashboard/api/telemetry",
            json={"event_name": "wizard_chip_clicked"},
        )
        assert resp.status_code == 429
        body = resp.json()
        assert body["error"]["code"] == "rate_limited"
        assert body["error"]["retry_after_ms"] > 0
        assert "Retry-After" in resp.headers

    def test_post_telemetry_lazy_init_when_omitted(self):
        # When the caller doesn't pass ``telemetry`` at all (default
        # parameter), the dashboard auto-creates a DashboardTelemetry
        # against ``data/telemetry.db`` so the endpoint always works.
        # We verify this by building a router without our explicit
        # store and observing that POST still records (status 200).
        import tempfile as _tempfile

        from src.dashboard.server import create_dashboard_router

        # Run from a tmp cwd so the auto-init's ``data/telemetry.db``
        # lands somewhere we can clean up.
        prev_cwd = os.getcwd()
        sandbox = _tempfile.mkdtemp()
        try:
            os.chdir(sandbox)
            router = create_dashboard_router(
                **self.components, mesh_port=8420,
            )
            app = FastAPI()
            app.include_router(router)
            client = _CSRFTestClient(app)
            resp = client.post(
                "/dashboard/api/telemetry",
                json={"event_name": "wizard_started"},
            )
            assert resp.status_code == 200, resp.text
            assert resp.json()["recorded"] is True
        finally:
            os.chdir(prev_cwd)
            shutil.rmtree(sandbox, ignore_errors=True)
