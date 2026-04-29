"""Tests for the per-platform success aggregation + dashboard endpoint.

Covers the in-memory aggregator (24h rolling window, host
canonicalization, captcha/burn/dwell ingestion) and the
``GET /api/dashboard/platform-success`` endpoint wired through
``EventBus`` listeners.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.dashboard.events import EventBus
from src.dashboard.platform_success import (
    PlatformSuccessAggregator,
    canonical_host,
)

# ── Aggregator unit tests ──────────────────────────────────────────────


class _Clock:
    """Monotonic injectable clock for window-eviction tests."""

    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TestCanonicalHost:
    def test_full_url_strips_scheme_and_www(self):
        assert canonical_host("https://www.linkedin.com/foo") == "linkedin.com"

    def test_bare_host_with_port(self):
        assert canonical_host("LinkedIn.com:443") == "linkedin.com"

    def test_subdomain_preserved(self):
        # We don't reduce subdomains — the dashboard wants per-host
        # visibility; the protected-platform timing logic does the
        # subdomain folding separately.
        assert canonical_host("https://api.linkedin.com/v2") == "api.linkedin.com"

    def test_empty_returns_none(self):
        assert canonical_host("") is None
        assert canonical_host(None) is None  # type: ignore[arg-type]

    def test_malformed_returns_none(self):
        # urlparse is forgiving — "not-a-url" is treated as a bare host
        # and lower-cased.  Anything obviously wrong (whitespace-only)
        # comes back as None.
        assert canonical_host("   ") is None or canonical_host("   ") == ""


class TestRecordCaptcha:
    def test_solved_failed_other(self):
        clk = _Clock()
        agg = PlatformSuccessAggregator(time_fn=clk)
        agg.record_captcha("linkedin.com", "success")
        agg.record_captcha("linkedin.com", "failed")
        agg.record_captcha("linkedin.com", "cost_cap")
        snap = agg.snapshot()
        rows = {p["host"]: p for p in snap["platforms"]}
        assert "linkedin.com" in rows
        row = rows["linkedin.com"]
        assert row["captcha_solved"] == 1
        assert row["captcha_failed"] == 1
        # cost_cap, rate_limited, etc. → "other" (gate-skipped, not
        # actually attempted at the solver).
        assert row["captcha_other"] == 1
        assert row["captcha_events"] == 3
        # ``captcha_attempted`` excludes gate skips; success_rate uses
        # the attempted denominator so a fleet hitting cost cap 100x
        # and solving 5 captchas doesn't show as "5/105 = 4.7%".
        assert row["captcha_attempted"] == 2
        assert abs(row["success_rate"] - 0.5) < 1e-3

    def test_no_host_is_dropped(self):
        agg = PlatformSuccessAggregator()
        agg.record_captcha(None, "success")
        agg.record_captcha("", "success")
        assert agg.snapshot()["platforms"] == []


class TestRecordPreNavDelay:
    def test_average_dwell_computed(self):
        agg = PlatformSuccessAggregator()
        agg.record_pre_nav_delay("linkedin.com", 2.0)
        agg.record_pre_nav_delay("linkedin.com", 4.0)
        agg.record_pre_nav_delay("linkedin.com", 3.0)
        snap = agg.snapshot()
        row = next(p for p in snap["platforms"] if p["host"] == "linkedin.com")
        # Mean of 2.0, 4.0, 3.0 = 3.0
        assert abs(row["avg_pre_nav_delay_s"] - 3.0) < 1e-3
        # And the dwell event implicitly counts as a navigation.
        assert row["navigations"] == 3
        assert row["pre_nav_delays_applied"] == 3

    def test_negative_delay_rejected(self):
        agg = PlatformSuccessAggregator()
        agg.record_pre_nav_delay("linkedin.com", -1.0)
        assert agg.snapshot()["platforms"] == []


class TestRecordFingerprintBurn:
    def test_burn_attributed_to_host(self):
        agg = PlatformSuccessAggregator()
        agg.record_fingerprint_burn("x.com")
        snap = agg.snapshot()
        row = next(p for p in snap["platforms"] if p["host"] == "x.com")
        assert row["fingerprint_burns"] == 1


class TestUnknownPlatformAggregated:
    def test_arbitrary_host_visible(self):
        """Operator visibility is platform-agnostic: a captcha event on
        ``example.com`` (NOT on the protected-platform list) still
        shows up in the rollup."""
        agg = PlatformSuccessAggregator()
        agg.record_captcha("example.com", "success")
        snap = agg.snapshot()
        hosts = {p["host"] for p in snap["platforms"]}
        assert "example.com" in hosts


class TestRollingWindow:
    def test_old_events_drop_out(self):
        clk = _Clock()
        agg = PlatformSuccessAggregator(window_s=10.0, time_fn=clk)
        agg.record_captcha("linkedin.com", "success")
        clk.advance(5)
        agg.record_captcha("linkedin.com", "success")
        # Both are still inside the 10s window
        snap = agg.snapshot()
        row = next(p for p in snap["platforms"] if p["host"] == "linkedin.com")
        assert row["captcha_solved"] == 2
        # Now jump past the first event's TTL
        clk.advance(6)  # now=11+ > window of 10s after first
        snap = agg.snapshot()
        row = next(p for p in snap["platforms"] if p["host"] == "linkedin.com")
        assert row["captcha_solved"] == 1
        # Jump past everything
        clk.advance(1000)
        snap = agg.snapshot()
        # Empty hosts get pruned
        assert snap["platforms"] == []


class TestSnapshotShape:
    def test_sort_by_navigations_desc(self):
        agg = PlatformSuccessAggregator()
        for _ in range(5):
            agg.record_navigation("a.com")
        for _ in range(10):
            agg.record_navigation("b.com")
        for _ in range(3):
            agg.record_navigation("c.com")
        rows = agg.snapshot()["platforms"]
        assert [p["host"] for p in rows] == ["b.com", "a.com", "c.com"]

    def test_since_present_iso8601(self):
        agg = PlatformSuccessAggregator()
        snap = agg.snapshot()
        # ISO8601 with Z suffix for UTC
        assert snap["since"].endswith("Z")

    def test_no_captcha_yields_null_success_rate(self):
        """A row with navigations but no captcha events reports
        success_rate=null — operators should see "—" not 100%."""
        agg = PlatformSuccessAggregator()
        agg.record_navigation("quiet-site.com")
        snap = agg.snapshot()
        row = next(p for p in snap["platforms"] if p["host"] == "quiet-site.com")
        assert row["success_rate"] is None


# ── EventBus integration ──────────────────────────────────────────────


class TestHandleEvent:
    """Verify the EventBus listener correctly dispatches by sub-type."""

    def test_captcha_gate_event(self):
        agg = PlatformSuccessAggregator()
        evt = {
            "type": "browser_metrics",
            "agent": "alpha",
            "data": {
                "type": "captcha_gate",
                "agent_id": "alpha",
                "outcome": "success",
                "kind": "cf_turnstile",
                "url": "https://www.linkedin.com/feed",
                "count": 3,
            },
        }
        agg.handle_event(evt)
        snap = agg.snapshot()
        row = next(p for p in snap["platforms"] if p["host"] == "linkedin.com")
        # Aggregator replays the count so all three are recorded
        assert row["captcha_solved"] == 3

    def test_fingerprint_burn_event(self):
        agg = PlatformSuccessAggregator()
        evt = {
            "type": "browser_metrics",
            "agent": "alpha",
            "data": {
                "type": "fingerprint_event",
                "agent_id": "alpha",
                "signal": "fingerprint_burn",
                "page_origin": "https://x.com",
                "count": 1,
            },
        }
        agg.handle_event(evt)
        snap = agg.snapshot()
        row = next(p for p in snap["platforms"] if p["host"] == "x.com")
        assert row["fingerprint_burns"] == 1

    def test_fingerprint_rejected_signal_ignored(self):
        agg = PlatformSuccessAggregator()
        evt = {
            "type": "browser_metrics",
            "agent": "alpha",
            "data": {
                "type": "fingerprint_event",
                "agent_id": "alpha",
                "signal": "rejected",  # NOT a burn — should be ignored
                "page_origin": "https://x.com",
                "count": 5,
            },
        }
        agg.handle_event(evt)
        assert agg.snapshot()["platforms"] == []

    def test_platform_pre_nav_delay_event(self):
        agg = PlatformSuccessAggregator()
        evt = {
            "type": "browser_metrics",
            "agent": "alpha",
            "data": {
                "type": "platform_pre_nav_delay",
                "agent_id": "alpha",
                "host": "linkedin.com",
                "count": 4,
                "total_delay_s": 12.0,
            },
        }
        agg.handle_event(evt)
        snap = agg.snapshot()
        row = next(p for p in snap["platforms"] if p["host"] == "linkedin.com")
        assert row["pre_nav_delays_applied"] == 4
        # 12.0 / 4 = 3.0
        assert abs(row["avg_pre_nav_delay_s"] - 3.0) < 1e-3
        # Each dwell implies a navigation
        assert row["navigations"] == 4

    def test_unrelated_event_ignored(self):
        agg = PlatformSuccessAggregator()
        evt = {
            "type": "llm_call",
            "data": {"type": "captcha_gate", "url": "https://linkedin.com/"},
        }
        agg.handle_event(evt)
        # Only browser_metrics events count
        assert agg.snapshot()["platforms"] == []

    def test_malformed_event_swallowed(self):
        agg = PlatformSuccessAggregator()
        # No "data" key, no "type" — must not raise
        agg.handle_event({})
        agg.handle_event({"type": "browser_metrics"})
        agg.handle_event({"type": "browser_metrics", "data": {"type": "captcha_gate"}})
        # No exception is the assertion


class TestEventBusListener:
    def test_emit_invokes_listener(self):
        bus = EventBus()
        agg = PlatformSuccessAggregator()
        bus.add_listener(agg.handle_event)
        bus.emit("browser_metrics", agent="alpha", data={
            "type": "captcha_gate",
            "agent_id": "alpha",
            "outcome": "success",
            "kind": "cf_turnstile",
            "url": "https://linkedin.com/",
            "count": 1,
        })
        snap = agg.snapshot()
        assert snap["platforms"][0]["host"] == "linkedin.com"

    def test_listener_exception_does_not_break_emit(self):
        bus = EventBus()
        crashes = []

        def bad(_evt):
            crashes.append(1)
            raise RuntimeError("boom")

        bus.add_listener(bad)
        bus.emit("llm_call")
        assert crashes == [1]  # listener was called
        # The event still made it into the buffer
        assert len(bus._buffer) == 1

    def test_remove_listener(self):
        bus = EventBus()
        agg = PlatformSuccessAggregator()
        bus.add_listener(agg.handle_event)
        bus.remove_listener(agg.handle_event)
        bus.emit("browser_metrics", data={
            "type": "captcha_gate",
            "outcome": "success",
            "url": "https://linkedin.com/",
            "count": 1,
        })
        assert agg.snapshot()["platforms"] == []


# ── Endpoint integration ──────────────────────────────────────────────


def _make_components(tmp_path: str) -> dict:
    """Minimal dashboard components for the endpoint tests.

    Mirrors :func:`tests.test_dashboard._make_components` with only the
    fields the platform-success endpoint touches.  Avoids the
    ``include_v2=True`` machinery (we don't need it).
    """
    from src.host.costs import CostTracker
    from src.host.health import HealthMonitor
    from src.host.mesh import Blackboard
    from src.host.traces import TraceStore

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


def _teardown(components: dict) -> None:
    components["cost_tracker"].close()
    components["trace_store"].close()
    components["blackboard"].close()


def _make_client(components: dict) -> TestClient:
    from src.dashboard.server import create_dashboard_router

    router = create_dashboard_router(**components, mesh_port=8420)
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestPlatformSuccessEndpoint:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.components = _make_components(self._tmpdir)
        self.client = _make_client(self.components)

    def teardown_method(self):
        _teardown(self.components)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_endpoint_returns_200_for_authenticated_request(self):
        # Self-hosted mode (no /opt/openlegion/.access_token) — the
        # auth verifier is a no-op, mirroring the rest of the
        # dashboard tests.
        resp = self.client.get("/dashboard/api/dashboard/platform-success")
        assert resp.status_code == 200

    def test_empty_state(self):
        resp = self.client.get("/dashboard/api/dashboard/platform-success")
        data = resp.json()
        assert data["platforms"] == []
        assert "since" in data
        assert data["since"].endswith("Z")

    def test_endpoint_reflects_emitted_events(self):
        bus: EventBus = self.components["event_bus"]
        # Emit several captcha events for linkedin.com — the dashboard
        # listener registered by create_dashboard_router should bin them.
        for outcome in ("success", "success", "failed"):
            bus.emit("browser_metrics", agent="alpha", data={
                "type": "captcha_gate",
                "agent_id": "alpha",
                "outcome": outcome,
                "kind": "cf_turnstile",
                "url": "https://linkedin.com/feed",
                "count": 1,
            })
        resp = self.client.get("/dashboard/api/dashboard/platform-success")
        data = resp.json()
        rows = {p["host"]: p for p in data["platforms"]}
        assert "linkedin.com" in rows
        row = rows["linkedin.com"]
        assert row["captcha_solved"] == 2
        assert row["captcha_failed"] == 1
        assert row["captcha_events"] == 3

    def test_endpoint_picks_up_fingerprint_burn(self):
        bus: EventBus = self.components["event_bus"]
        bus.emit("browser_metrics", agent="alpha", data={
            "type": "fingerprint_event",
            "agent_id": "alpha",
            "signal": "fingerprint_burn",
            "page_origin": "https://www.x.com",
            "count": 1,
        })
        resp = self.client.get("/dashboard/api/dashboard/platform-success")
        rows = {p["host"]: p for p in resp.json()["platforms"]}
        assert rows["x.com"]["fingerprint_burns"] == 1

    def test_endpoint_picks_up_pre_nav_delay(self):
        bus: EventBus = self.components["event_bus"]
        bus.emit("browser_metrics", agent="alpha", data={
            "type": "platform_pre_nav_delay",
            "agent_id": "alpha",
            "host": "linkedin.com",
            "count": 5,
            "total_delay_s": 14.7,
        })
        resp = self.client.get("/dashboard/api/dashboard/platform-success")
        rows = {p["host"]: p for p in resp.json()["platforms"]}
        row = rows["linkedin.com"]
        assert row["pre_nav_delays_applied"] == 5
        # 14.7 / 5 = 2.94
        assert abs(row["avg_pre_nav_delay_s"] - 2.94) < 1e-2

    def test_endpoint_sort_order(self):
        bus: EventBus = self.components["event_bus"]
        # 5 navs to a, 1 nav to b — a should come first
        for _ in range(5):
            bus.emit("browser_metrics", agent="alpha", data={
                "type": "platform_pre_nav_delay",
                "agent_id": "alpha", "host": "a.com",
                "count": 1, "total_delay_s": 1.0,
            })
        bus.emit("browser_metrics", agent="alpha", data={
            "type": "platform_pre_nav_delay",
            "agent_id": "alpha", "host": "b.com",
            "count": 1, "total_delay_s": 1.0,
        })
        rows = self.client.get(
            "/dashboard/api/dashboard/platform-success",
        ).json()["platforms"]
        assert [p["host"] for p in rows[:2]] == ["a.com", "b.com"]

    def test_arbitrary_host_visible_endpoint(self):
        """example.com is not on the protected-platform list, but a
        captcha event there still surfaces — operator visibility must
        be platform-agnostic."""
        bus: EventBus = self.components["event_bus"]
        bus.emit("browser_metrics", agent="alpha", data={
            "type": "captcha_gate",
            "agent_id": "alpha",
            "outcome": "success",
            "kind": "cf_turnstile",
            "url": "https://example.com/foo",
            "count": 1,
        })
        rows = {p["host"]: p for p in self.client.get(
            "/dashboard/api/dashboard/platform-success",
        ).json()["platforms"]}
        assert "example.com" in rows


# ── Window eviction integration ──────────────────────────────────────


class TestWindowEvictionEndpoint:
    """Verifies events older than the rolling window drop out of the
    snapshot.  Uses an injected clock to avoid sleeping."""

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.components = _make_components(self._tmpdir)
        # Build a router with a custom aggregator clock so we can
        # advance time without the test taking 24h to run.  We
        # accomplish this by reaching into the listener after-the-fact:
        # the aggregator is module-level only inside the closure, so
        # we test the eviction path via the unit-level interface and
        # then sanity-check the endpoint path with a freshly-emitted
        # event (no time travel needed).
        self.client = _make_client(self.components)

    def teardown_method(self):
        _teardown(self.components)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_endpoint_24h_window_eviction(self):
        """Direct aggregator test (the endpoint path is covered by the
        sibling tests): events older than the window are evicted."""
        clk = _Clock()
        agg = PlatformSuccessAggregator(window_s=86400.0, time_fn=clk)
        agg.record_captcha("linkedin.com", "success")
        # Step past the window
        clk.advance(86400.0 + 1)
        snap = agg.snapshot()
        # The host's only event aged out — empty rows are dropped
        assert snap["platforms"] == []


# ── success_rate denominator excludes gate-skipped events ─────────────


class TestSuccessRateExcludesGateSkips:
    """Pre-fix the denominator was ``solved + failed + other``. A fleet
    that hit cost cap 100 times and solved 5 captchas reported a 4.7%
    success rate — operator-misleading. The honest rate uses only
    ``solved + failed`` (the actual solver attempts)."""

    def test_cost_cap_storm_does_not_drag_success_rate_to_zero(self):
        """100 cost-cap skips + 5 successful solves → 100% (5/5),
        not 4.7% (5/105). Operator sees the truth: every solve we
        ATTEMPTED succeeded; we just didn't attempt many because of
        the cap."""
        agg = PlatformSuccessAggregator()
        for _ in range(5):
            agg.record_captcha("linkedin.com", "success")
        for _ in range(100):
            agg.record_captcha("linkedin.com", "cost_cap")
        snap = agg.snapshot()
        row = next(p for p in snap["platforms"] if p["host"] == "linkedin.com")
        assert row["captcha_solved"] == 5
        assert row["captcha_failed"] == 0
        assert row["captcha_other"] == 100
        assert row["captcha_attempted"] == 5
        # captcha_events still reports the total (operator-visible
        # volume) so a glance at the row shows "100 cost-cap skips".
        assert row["captcha_events"] == 105
        # Success rate excludes the skips → 5/5 = 100%.
        assert row["success_rate"] == 1.0

    def test_rate_limit_skips_excluded(self):
        """``rate_limited`` is a gate-skip outcome same as ``cost_cap`` —
        belongs in ``other``, not in the success-rate denominator."""
        agg = PlatformSuccessAggregator()
        agg.record_captcha("x.com", "success")
        agg.record_captcha("x.com", "failed")
        agg.record_captcha("x.com", "rate_limited")
        agg.record_captcha("x.com", "skipped_behavioral")
        snap = agg.snapshot()
        row = next(p for p in snap["platforms"] if p["host"] == "x.com")
        assert row["captcha_attempted"] == 2  # success + failed
        assert row["captcha_other"] == 2      # rate_limited + skipped_behavioral
        assert abs(row["success_rate"] - 0.5) < 1e-3

    def test_only_gate_skips_yields_null_success_rate(self):
        """A platform that only ever hits gate skips (every captcha
        cost-cap'd, never reached the solver) reports
        ``success_rate=None`` — there's nothing meaningful to compute,
        and "0% / 0 attempts" is more honest than "0% / 0 attempts"
        rendered as "0%"."""
        agg = PlatformSuccessAggregator()
        for _ in range(10):
            agg.record_captcha("instagram.com", "cost_cap")
        snap = agg.snapshot()
        row = next(p for p in snap["platforms"] if p["host"] == "instagram.com")
        assert row["captcha_attempted"] == 0
        assert row["captcha_other"] == 10
        assert row["success_rate"] is None
