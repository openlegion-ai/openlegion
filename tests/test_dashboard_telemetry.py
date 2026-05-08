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
from pathlib import Path

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

_REPO_ROOT = Path(__file__).resolve().parent.parent
_APP_JS = _REPO_ROOT / "src/dashboard/static/js/app.js"
_TEMPLATE = _REPO_ROOT / "src/dashboard/templates/index.html"


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


# ── Phase 0 baseline events — endpoint contract ─────────────────


class TestPhase0EndpointEvents:
    """The Phase 0 telemetry baseline emits seven event names against the
    same generic endpoint. The store has no schema-per-event constraint,
    so the contract is: each event records, persists props verbatim, and
    survives a ``recent()`` round-trip with the right shape.
    """

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

    def test_tab_view_event_persists_with_props(self):
        resp = self.client.post(
            "/dashboard/api/telemetry",
            json={
                "event_name": "tab_view",
                "props": {"tab_id": "fleet", "from_tab_id": "chat"},
            },
        )
        assert resp.status_code == 200, resp.text
        rows = self.telemetry.recent(event_name="tab_view")
        assert len(rows) == 1
        assert rows[0]["props"]["tab_id"] == "fleet"
        assert rows[0]["props"]["from_tab_id"] == "chat"

    def test_time_to_first_action_event_persists(self):
        resp = self.client.post(
            "/dashboard/api/telemetry",
            json={
                "event_name": "time_to_first_action",
                "props": {
                    "action_type": "tab_view",
                    "seconds_since_load": 4.21,
                },
            },
        )
        assert resp.status_code == 200, resp.text
        rows = self.telemetry.recent(event_name="time_to_first_action")
        assert len(rows) == 1
        assert rows[0]["props"]["action_type"] == "tab_view"
        assert rows[0]["props"]["seconds_since_load"] == 4.21

    def test_needs_you_click_event_persists(self):
        resp = self.client.post(
            "/dashboard/api/telemetry",
            json={
                "event_name": "needs_you_click",
                "props": {
                    "item_count": 3,
                    "item_kind": "pending",
                    "action_label": "Confirm",
                },
            },
        )
        assert resp.status_code == 200, resp.text
        rows = self.telemetry.recent(event_name="needs_you_click")
        assert rows[0]["props"]["item_count"] == 3
        assert rows[0]["props"]["item_kind"] == "pending"

    def test_subtab_usage_event_persists(self):
        resp = self.client.post(
            "/dashboard/api/telemetry",
            json={
                "event_name": "subtab_usage",
                "props": {"from_subtab": "feed", "to_subtab": "task-board"},
            },
        )
        assert resp.status_code == 200, resp.text
        rows = self.telemetry.recent(event_name="subtab_usage")
        assert rows[0]["props"]["from_subtab"] == "feed"
        assert rows[0]["props"]["to_subtab"] == "task-board"

    def test_empty_state_cta_click_event_persists(self):
        resp = self.client.post(
            "/dashboard/api/telemetry",
            json={
                "event_name": "empty_state_cta_click",
                "props": {"section_id": "operator_intent_content"},
            },
        )
        assert resp.status_code == 200, resp.text
        rows = self.telemetry.recent(event_name="empty_state_cta_click")
        assert rows[0]["props"]["section_id"] == "operator_intent_content"

    def test_dock_open_and_close_events_persist(self):
        # Dock helpers have no UI in Phase 0 but are wireable. Confirm
        # the endpoint accepts both event names so Phase 1 can ship the
        # surface without a backend change.
        for name, props in (
            ("dock_open", {"from_tab_id": "fleet"}),
            ("dock_close", {"from_tab_id": "fleet"}),
        ):
            resp = self.client.post(
                "/dashboard/api/telemetry",
                json={"event_name": name, "props": props},
            )
            assert resp.status_code == 200, resp.text
        opens = self.telemetry.recent(event_name="dock_open")
        closes = self.telemetry.recent(event_name="dock_close")
        assert len(opens) == 1
        assert len(closes) == 1
        assert opens[0]["props"]["from_tab_id"] == "fleet"


# ── Phase 0 baseline events — JS / template wiring ──────────────


class TestPhase0FrontendWiring:
    """The track() helper landed in Phase -1; Phase 0 adds named helpers
    and wires them to existing UI. Without these wiring tests, a future
    refactor could quietly delete a hook and the events would silently
    stop flowing — exactly the bug a baseline is supposed to catch.
    """

    def test_app_js_defines_phase0_helpers(self):
        js = _APP_JS.read_text(encoding="utf-8")
        for helper in (
            "_trackFirstAction",
            "_handleNeedsYouAction",
            "trackEmptyStateCta",
            "trackSubtabUsage",
            "dockOpen",
            "dockClose",
        ):
            assert helper in js, f"missing helper in app.js: {helper}"

    def test_first_action_tracker_fires_at_most_once(self):
        # The flag-guard pattern is load-bearing: every interactive
        # wrapper calls this helper, but only the first call should
        # actually emit the event. Assert the flag-set lives in the
        # helper itself rather than in each call site.
        js = _APP_JS.read_text(encoding="utf-8")
        assert "_firstActionTracked: false" in js
        # The early-return on the flag must precede the track() call.
        block_start = js.find("_trackFirstAction(actionType)")
        assert block_start >= 0
        block = js[block_start : block_start + 600]
        assert "if (this._firstActionTracked) return;" in block
        assert "this._firstActionTracked = true;" in block
        assert "this.track('time_to_first_action'" in block

    def test_switch_tab_emits_tab_view_with_from_id(self):
        js = _APP_JS.read_text(encoding="utf-8")
        # The body of switchTab() should record fromTab BEFORE mutating
        # activeTab, then call track('tab_view', {tab_id, from_tab_id}).
        idx = js.find("switchTab(tab) {")
        assert idx >= 0
        body = js[idx : idx + 1200]
        assert "const fromTab = this.activeTab;" in body
        assert "this.track('tab_view'" in body
        assert "from_tab_id: fromTab" in body
        # Self-switches must be filtered so refresh doesn't double-fire.
        assert "if (fromTab !== tab)" in body

    def test_dock_helpers_are_idempotent(self):
        js = _APP_JS.read_text(encoding="utf-8")
        # Both helpers must short-circuit on the shadowed _dockOpen flag
        # before emitting telemetry, so accidental double-fires (ESC +
        # click-outside) don't double-count.
        assert "_dockOpen: false" in js
        open_idx = js.find("dockOpen(opts)")
        assert open_idx >= 0
        open_body = js[open_idx : open_idx + 500]
        assert "if (this._dockOpen) return;" in open_body
        assert "this._dockOpen = true;" in open_body
        close_idx = js.find("dockClose(opts)")
        assert close_idx >= 0
        close_body = js[close_idx : close_idx + 500]
        assert "if (!this._dockOpen) return;" in close_body
        assert "this._dockOpen = false;" in close_body

    def test_workplace_subtab_buttons_emit_subtab_usage(self):
        html = _TEMPLATE.read_text(encoding="utf-8")
        # Phase 3 collapsed the four-sub-tab bar into a single-scroll
        # Home layout (`homeTab === 'main'`) plus a kanban sub-page
        # (`homeTab === 'tasks'`). The legacy
        # ``trackSubtabUsage(workplaceTab, wt.id)`` wiring is gone with
        # the bar; the equivalent transition handler is now
        # ``switchHomeTab('main' | 'tasks')`` invoked from the back-link
        # and "See full task board" CTA. trackSubtabUsage itself is kept
        # in app.js for the hidden legacy renders + for empty-state CTA
        # telemetry — see test_empty_state_cta_buttons_emit_telemetry.
        assert "switchHomeTab('main')" in html
        assert "switchHomeTab('tasks')" in html

    def test_needs_you_action_button_uses_telemetry_wrapper(self):
        html = _TEMPLATE.read_text(encoding="utf-8")
        # The action-row click handler should funnel through the wrapper
        # (which both records the click AND invokes the original handler)
        # rather than calling action.handler() directly.
        assert "_handleNeedsYouAction(item, action)" in html
        # Defensively guard against a regression that re-introduces the
        # raw call path.
        assert '@click="action.handler()"' not in html

    def test_empty_state_cta_buttons_emit_telemetry(self):
        html = _TEMPLATE.read_text(encoding="utf-8")
        # Each empty-state intent chip on the operator empty state has a
        # stable section_id so we can compare CTA traction across them.
        # ``recently_delivered_view_all`` was retired in Phase 3 when the
        # Workplace sub-tab bar collapsed into the single-scroll Home
        # layout — there's no "View all →" button to instrument anymore.
        for section_id in (
            "operator_intent_content",
            "operator_intent_research",
            "operator_intent_sales",
            "operator_intent_devteam",
            "operator_intent_other",
        ):
            assert f"trackEmptyStateCta('{section_id}')" in html, (
                f"missing trackEmptyStateCta call for: {section_id}"
            )
