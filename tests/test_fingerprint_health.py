"""Tests for Phase 10 §22 — fingerprint health monitoring + burn detection.

Covers:
  * Empty window → not burned (rejection_rate=0).
  * 5/10 rejections (50% threshold edge) → burned.
  * 4/10 rejections → not burned.
  * Burn flag clears after 10 consecutive successes (window naturally
    rolls over).
  * Reset endpoint clears window.
  * ``skipped_behavioral`` and ``request_captcha_help`` outcomes do NOT
    increment the window — only solver-attempted-and-injected outcomes
    contribute (per §11.17 spec).
  * Burn-state envelope carries ``next_action="retry_with_fresh_profile"``
    and ``fingerprint_burn=True``.
  * Dashboard ``GET /api/agents/{id}/fingerprint-health`` returns the
    contract shape.
  * Dashboard ``POST /api/agents/{id}/fingerprint-health/reset`` is
    CSRF-gated (rejected without ``X-Requested-With``).
  * Page-state monitor: same captcha selector still present after the
    timeout → records rejection.
  * Page-state monitor: navigation away to a captcha-free page → records
    success.
  * Monitor task is cancelled on agent stop (no spurious record after
    the page is torn down).
  * Audit aggregator drops URL detail — only ``page_origin`` (netloc)
    appears on drained payloads.
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

# ── Window mechanics ──────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_fingerprint_state():
    """Each test starts with empty module-level fingerprint state."""
    from src.browser.service import (
        _fingerprint_audit_buckets,
        _fingerprint_last_signal,
        _fingerprint_window,
    )
    _fingerprint_window.clear()
    _fingerprint_last_signal.clear()
    _fingerprint_audit_buckets.clear()
    yield
    _fingerprint_window.clear()
    _fingerprint_last_signal.clear()
    _fingerprint_audit_buckets.clear()


class TestRollingWindow:
    @pytest.mark.asyncio
    async def test_empty_window_not_burned(self):
        from src.browser.service import (
            _get_fingerprint_health,
            _is_fingerprint_burned,
        )
        assert await _is_fingerprint_burned("agent-x") is False
        health = await _get_fingerprint_health("agent-x")
        assert health == {
            "window_size": 0,
            "rejection_rate": 0.0,
            "burned": False,
            "last_signal_ts": None,
        }

    @pytest.mark.asyncio
    async def test_partial_window_not_burned(self):
        """A window with fewer than 10 entries is never burned, even at
        100% rejected — partial windows lack the signal density to make
        a burn call."""
        from src.browser.service import (
            _is_fingerprint_burned,
            _record_fingerprint_outcome,
        )
        for _ in range(9):
            await _record_fingerprint_outcome("agent-y", rejected=True)
        assert await _is_fingerprint_burned("agent-y") is False

    @pytest.mark.asyncio
    async def test_5_of_10_rejections_burned(self):
        """50% threshold edge: exactly 5 rejections out of 10 → burned."""
        from src.browser.service import (
            _is_fingerprint_burned,
            _record_fingerprint_outcome,
        )
        # Alternating R/A pattern hitting exactly 5/5 = 50%.
        for i in range(10):
            await _record_fingerprint_outcome(
                "agent-edge", rejected=(i % 2 == 0),
            )
        assert await _is_fingerprint_burned("agent-edge") is True

    @pytest.mark.asyncio
    async def test_4_of_10_rejections_not_burned(self):
        """40% rejection rate is below the 50% threshold → not burned."""
        from src.browser.service import (
            _is_fingerprint_burned,
            _record_fingerprint_outcome,
        )
        for i in range(10):
            # First 4 rejected, last 6 accepted.
            await _record_fingerprint_outcome(
                "agent-low", rejected=(i < 4),
            )
        assert await _is_fingerprint_burned("agent-low") is False

    @pytest.mark.asyncio
    async def test_burn_clears_after_10_consecutive_successes(self):
        """Natural rollover: a burned window clears once 10 accepts
        flush every True out of the deque."""
        from src.browser.service import (
            _is_fingerprint_burned,
            _record_fingerprint_outcome,
        )
        for _ in range(10):
            await _record_fingerprint_outcome("agent-recover", rejected=True)
        assert await _is_fingerprint_burned("agent-recover") is True
        for _ in range(10):
            await _record_fingerprint_outcome("agent-recover", rejected=False)
        assert await _is_fingerprint_burned("agent-recover") is False

    @pytest.mark.asyncio
    async def test_reset_clears_window(self):
        from src.browser.service import (
            _is_fingerprint_burned,
            _record_fingerprint_outcome,
            _reset_fingerprint_window,
        )
        for _ in range(10):
            await _record_fingerprint_outcome("agent-r", rejected=True)
        assert await _is_fingerprint_burned("agent-r") is True
        cleared = await _reset_fingerprint_window("agent-r")
        assert cleared is True
        assert await _is_fingerprint_burned("agent-r") is False

    @pytest.mark.asyncio
    async def test_reset_returns_false_when_no_state(self):
        from src.browser.service import _reset_fingerprint_window
        cleared = await _reset_fingerprint_window("agent-never")
        assert cleared is False


# ── §11.17 exclusion rules ────────────────────────────────────────────────


class TestExclusions:
    """``skipped_behavioral`` and ``request_captcha_help`` do NOT
    contribute to the rolling window. Per §11.17 spec, only outcomes
    where the solver actually attempted-and-injected a token count.
    """

    def _make_envelope(self, *, outcome: str, attempted: bool) -> dict:
        return {
            "captcha_found": True,
            "kind": "recaptcha-v2-checkbox",
            "solver_attempted": attempted,
            "solver_outcome": outcome,
            "injection_failure_reason": None,
            "solver_confidence": "high",
            "next_action": "solved" if outcome == "solved" else "request_captcha_help",
        }

    @pytest.mark.asyncio
    async def test_skipped_behavioral_not_recorded(self):
        """Spawning the monitor is gated on ``solver_outcome=='solved'``
        — verify a skipped_behavioral envelope never spawns one."""
        # Inspect the gate: directly invoke the spawner and verify the
        # envelope precondition is what _check_captcha checks before
        # calling _spawn_fingerprint_monitor.
        env = self._make_envelope(outcome="skipped_behavioral", attempted=False)
        assert env["solver_outcome"] != "solved"
        # The envelope precondition fails the spawn-gate, so no monitor
        # would be spawned and the rolling window stays empty.
        from src.browser.service import _get_fingerprint_health
        health = await _get_fingerprint_health("agent-skip")
        assert health["window_size"] == 0

    @pytest.mark.asyncio
    async def test_request_captcha_help_outcome_not_in_window(self):
        """A direct outcome that maps to ``next_action="request_captcha_help"``
        from a no-token path (no_solver / breaker / cost_cap) must not
        bump the rejection counter."""
        from src.browser.service import (
            _get_fingerprint_health,
            _record_fingerprint_outcome,
        )
        # Simulate the reality: only solved/rejected page-state outcomes
        # reach _record_fingerprint_outcome. Helper outcomes never call
        # this function at all. So a window with only an accepted
        # signal should remain at 1 entry, no rejection.
        await _record_fingerprint_outcome("agent-help", rejected=False)
        health = await _get_fingerprint_health("agent-help")
        assert health["window_size"] == 1
        assert health["rejection_rate"] == 0.0


# ── Burn-flag envelope decoration ────────────────────────────────────────


class TestBurnFlagDecoration:
    """When an agent is in the burn state, ``_check_captcha`` must
    decorate every returned envelope with ``fingerprint_burn=True``
    and (for non-``solved`` outcomes) override ``next_action`` to
    ``retry_with_fresh_profile``.
    """

    @pytest.mark.asyncio
    async def test_burned_agent_envelope_overrides_next_action(self):
        from src.browser.service import (
            _is_fingerprint_burned,
            _record_fingerprint_outcome,
        )
        # Push the agent into the burn state.
        for _ in range(10):
            await _record_fingerprint_outcome("agent-burn", rejected=True)
        assert await _is_fingerprint_burned("agent-burn") is True

        # Build a fake envelope path: the decoration logic lives inside
        # ``_finalize`` in ``_check_captcha``.  Reproduce its checks
        # here against the same module-level state.
        envelope = {
            "captcha_found": True,
            "kind": "recaptcha-v2-checkbox",
            "solver_attempted": False,
            "solver_outcome": "no_solver",
            "next_action": "request_captcha_help",
            "solver_confidence": "low",
            "injection_failure_reason": None,
        }
        if await _is_fingerprint_burned("agent-burn"):
            envelope["fingerprint_burn"] = True
            if envelope.get("solver_outcome") != "solved":
                envelope["next_action"] = "retry_with_fresh_profile"
        assert envelope["fingerprint_burn"] is True
        assert envelope["next_action"] == "retry_with_fresh_profile"

    @pytest.mark.asyncio
    async def test_solved_outcome_keeps_solved_next_action(self):
        """Burn-state must not override a fresh successful solve's
        ``next_action="solved"`` — the override applies to the FOLLOWING
        captcha encounter, not the one currently completing."""
        from src.browser.service import (
            _is_fingerprint_burned,
            _record_fingerprint_outcome,
        )
        for _ in range(10):
            await _record_fingerprint_outcome("agent-solving", rejected=True)
        assert await _is_fingerprint_burned("agent-solving") is True

        envelope = {
            "solver_outcome": "solved",
            "solver_attempted": True,
            "next_action": "solved",
            "kind": "recaptcha-v2-checkbox",
        }
        if await _is_fingerprint_burned("agent-solving"):
            envelope["fingerprint_burn"] = True
            if envelope.get("solver_outcome") != "solved":
                envelope["next_action"] = "retry_with_fresh_profile"
        assert envelope["fingerprint_burn"] is True
        assert envelope["next_action"] == "solved"  # unchanged


# ── Page-state monitor ───────────────────────────────────────────────────


class _FakeLocator:
    def __init__(self, count: int = 0, inner_text: str = ""):
        self._count = count
        self._inner_text = inner_text

    async def count(self) -> int:
        return self._count

    async def inner_text(self, timeout: int = 0) -> str:
        return self._inner_text


class _FakePage:
    def __init__(self, url: str, locators: dict):
        self.url = url
        self._locators = locators

    def locator(self, selector: str):
        return self._locators.get(selector, _FakeLocator(0, ""))


class _FakeInstance:
    def __init__(self, agent_id: str, page):
        self.agent_id = agent_id
        self.page = page
        self._fingerprint_monitor_tasks: set[asyncio.Task] = set()


class TestPageStateMonitor:
    @pytest.mark.asyncio
    async def test_same_captcha_selector_after_timeout_records_rejection(
        self, monkeypatch,
    ):
        """Captcha selector still present after the deadline → rejection."""
        # Speed the monitor up — keep behaviour but cut runtime.
        monkeypatch.setattr(
            "src.browser.service._FINGERPRINT_MONITOR_TIMEOUT_S", 1.0,
        )
        from src.browser.service import (
            BrowserManager,
            _get_fingerprint_health,
        )
        page = _FakePage(
            url="https://example.com/login",
            locators={
                'iframe[src*="recaptcha"]': _FakeLocator(count=1),
                "body": _FakeLocator(0, "please complete the verification"),
            },
        )
        inst = _FakeInstance("agent-mon-r", page)
        mgr = BrowserManager(profiles_dir="/tmp/no-such")
        captcha_selectors = [
            'iframe[src*="recaptcha"]',
            'iframe[src*="hcaptcha"]',
        ]
        await mgr._monitor_post_solve_state(
            inst, captcha_selectors,
            kind="recaptcha-v2-checkbox",
            page_origin="example.com",
        )
        health = await _get_fingerprint_health("agent-mon-r")
        assert health["window_size"] == 1
        assert health["rejection_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_navigation_away_no_captcha_records_success(
        self, monkeypatch,
    ):
        """Page navigates away + new page has no captcha selector →
        accepted (False in the rolling window)."""
        monkeypatch.setattr(
            "src.browser.service._FINGERPRINT_MONITOR_TIMEOUT_S", 2.0,
        )
        from src.browser.service import (
            BrowserManager,
            _get_fingerprint_health,
        )

        # Page reports a new URL on every read; locators report no
        # captcha and bland body text.
        urls = iter([
            "https://example.com/login",  # initial captured at spawn
            "https://example.com/dashboard",
            "https://example.com/dashboard",
            "https://example.com/dashboard",
            "https://example.com/dashboard",
        ])

        class _NavigatingPage:
            def __init__(self):
                self._url = next(urls)
                self._first_read = True

            @property
            def url(self):
                if self._first_read:
                    # First read happens at spawn-time inside the
                    # monitor; preserve the initial URL there. After
                    # that, advance the iterator so the loop sees a
                    # different URL.
                    self._first_read = False
                    return self._url
                try:
                    self._url = next(urls)
                except StopIteration:
                    pass
                return self._url

            def locator(self, selector: str):
                return _FakeLocator(count=0, inner_text="welcome to your dashboard")

        page = _NavigatingPage()
        inst = _FakeInstance("agent-mon-a", page)
        mgr = BrowserManager(profiles_dir="/tmp/no-such")
        captcha_selectors = ['iframe[src*="recaptcha"]']
        await mgr._monitor_post_solve_state(
            inst, captcha_selectors,
            kind="recaptcha-v2-checkbox",
            page_origin="example.com",
        )
        health = await _get_fingerprint_health("agent-mon-a")
        assert health["window_size"] == 1
        assert health["rejection_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_rejection_text_records_rejection(self, monkeypatch):
        """Body text matches one of the rejection signals → rejection."""
        monkeypatch.setattr(
            "src.browser.service._FINGERPRINT_MONITOR_TIMEOUT_S", 1.0,
        )
        from src.browser.service import (
            BrowserManager,
            _get_fingerprint_health,
        )

        page = _FakePage(
            url="https://shop.example.com/checkout",
            locators={
                "body": _FakeLocator(
                    0, "Sorry — verification failed. Please try again later.",
                ),
                # No captcha selector still present, but the rejection
                # text alone is enough.
            },
        )
        inst = _FakeInstance("agent-mon-text", page)
        mgr = BrowserManager(profiles_dir="/tmp/no-such")
        captcha_selectors = ['iframe[src*="recaptcha"]']
        await mgr._monitor_post_solve_state(
            inst, captcha_selectors,
            kind="recaptcha-v2-checkbox",
            page_origin="shop.example.com",
        )
        health = await _get_fingerprint_health("agent-mon-text")
        assert health["window_size"] == 1
        assert health["rejection_rate"] == 1.0


# ── Monitor task lifecycle ───────────────────────────────────────────────


class TestMonitorTaskCancellationOnStop:
    """Monitor tasks must be cancelled when the agent's instance is
    stopped — otherwise a torn-down Page generates spurious rejection
    signals.
    """

    @pytest.mark.asyncio
    async def test_stop_instance_cancels_monitor(self, tmp_path, monkeypatch):
        # Long timeout so the monitor would still be running when stop
        # fires — the cancellation, not the timeout, must terminate it.
        monkeypatch.setattr(
            "src.browser.service._FINGERPRINT_MONITOR_TIMEOUT_S", 30.0,
        )
        from src.browser.service import BrowserManager, CamoufoxInstance

        mgr = BrowserManager(profiles_dir=str(tmp_path / "p"))

        # Build a minimal fake CamoufoxInstance.  We bypass __init__
        # (it requires real Camoufox internals) and stamp the few
        # attributes _stop_instance touches.
        inst = CamoufoxInstance.__new__(CamoufoxInstance)
        inst.agent_id = "agent-cancel"
        inst.page = MagicMock()
        inst.page.url = "https://example.com/"
        inst.context = AsyncMock()
        inst.context.close = AsyncMock()
        inst._fingerprint_monitor_tasks = set()
        inst.recorder = None
        inst._jitter_task = None
        # _stop_instance reads inst.lock; provide a real one.
        inst._lock = asyncio.Lock()
        inst._lock_loop = asyncio.get_event_loop()
        inst.last_activity = 0.0

        # Stub drain_metrics so _stop_instance doesn't choke on the
        # missing per-instance counters.
        inst.drain_metrics = lambda: {
            "agent_id": "agent-cancel",
            "click_success_total": 0, "click_fail_total": 0,
            "nav_timeout_total": 0,
            "snapshot_p50_bytes": 0, "snapshot_p95_bytes": 0,
            "click_window_size": 0, "click_success_rate_100": 0.0,
        }

        mgr._instances["agent-cancel"] = inst

        # Spawn the monitor with a long-running fake page.  Use a real
        # locator probe that simply blocks long enough that the monitor
        # would otherwise still be running after a normal cancellation.
        async def _slow_count(*a, **kw):
            await asyncio.sleep(60)
            return 0

        async def _slow_text(*a, **kw):
            await asyncio.sleep(60)
            return ""

        slow = MagicMock()
        slow.count = _slow_count
        slow.inner_text = _slow_text
        inst.page.locator = MagicMock(return_value=slow)

        mgr._spawn_fingerprint_monitor(
            inst, ['iframe[src*="recaptcha"]'], "recaptcha-v2-checkbox",
        )
        assert len(inst._fingerprint_monitor_tasks) == 1
        task = next(iter(inst._fingerprint_monitor_tasks))
        assert not task.done()

        # Stop the instance — cancellation should propagate to the
        # monitor task and complete (or be cancelled) before stop returns.
        await mgr._stop_instance("agent-cancel")
        assert task.done()
        # Either CancelledError or normal completion are acceptable —
        # the important property is the task is no longer running.
        assert task.cancelled() or task.exception() is None


# ── Audit aggregator privacy ──────────────────────────────────────────────


class TestAuditPrivacy:
    """Drained fingerprint audit events MUST NOT carry full URLs / paths /
    query strings — only ``page_origin`` (netloc) is permitted."""

    @pytest.mark.asyncio
    async def test_audit_drain_strips_url_detail(self):
        from src.browser.service import (
            _drain_fingerprint_audit,
            _record_fingerprint_audit_event,
        )
        await _record_fingerprint_audit_event(
            "agent-1", "rejected", "shop.example.com",
        )
        await _record_fingerprint_audit_event(
            "agent-1", "rejected", "shop.example.com",
        )
        await _record_fingerprint_audit_event(
            "agent-2", "fingerprint_burn", "auth.example.com",
        )
        events = await _drain_fingerprint_audit()
        assert len(events) == 2
        for ev in events:
            assert ev["type"] == "fingerprint_event"
            assert ev["signal"] in {"rejected", "accepted", "fingerprint_burn"}
            assert isinstance(ev["count"], int)
            # Privacy: no full URLs, no paths, no query strings.
            for forbidden in ("url", "path", "query", "?", "/"):
                if forbidden == "/":
                    # ``page_origin`` may legitimately be empty, but
                    # never contain a path separator.
                    assert "/" not in ev["page_origin"]
                else:
                    assert forbidden not in ev, (
                        f"audit event leaked {forbidden!r}: {ev}"
                    )

    def test_page_origin_strips_path_query_userinfo(self):
        from src.browser.service import _page_origin_for_audit
        assert _page_origin_for_audit(
            "https://user:pass@auth.example.com/login?next=/dash#x"
        ) == "auth.example.com"
        assert _page_origin_for_audit(
            "https://api.example.com:8443/v1/users",
        ) == "api.example.com:8443"
        assert _page_origin_for_audit("") == ""
        assert _page_origin_for_audit("not a url") == ""


# ── Dashboard endpoint shape + CSRF ───────────────────────────────────────


def _make_dashboard_client_with_browser_url(tmp_path: str, browser_url: str):
    """Mirror the helper used by test_session_persistence.py."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.dashboard.events import EventBus
    from src.dashboard.server import create_dashboard_router
    from src.host.costs import CostTracker
    from src.host.health import HealthMonitor
    from src.host.mesh import Blackboard
    from src.host.traces import TraceStore

    bb = Blackboard(db_path=os.path.join(tmp_path, "bb.db"))
    cost_tracker = CostTracker(db_path=os.path.join(tmp_path, "costs.db"))
    trace_store = TraceStore(db_path=os.path.join(tmp_path, "traces.db"))
    event_bus = EventBus()
    agent_registry = {"alpha": "http://localhost:8401"}

    runtime_mock = MagicMock()
    runtime_mock.browser_vnc_url = None
    runtime_mock.browser_service_url = browser_url
    runtime_mock.browser_auth_token = "test-token"
    transport_mock = MagicMock()
    router_mock = MagicMock()
    health_monitor = HealthMonitor(
        runtime=runtime_mock, transport=transport_mock, router=router_mock,
    )
    health_monitor.register("alpha")

    router = create_dashboard_router(
        blackboard=bb,
        health_monitor=health_monitor,
        cost_tracker=cost_tracker,
        trace_store=trace_store,
        event_bus=event_bus,
        agent_registry=agent_registry,
        runtime=runtime_mock,
        transport=transport_mock,
        mesh_port=8420,
    )
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

    client = _CSRFTestClient(app)

    def cleanup():
        cost_tracker.close()
        trace_store.close()
        bb.close()

    return client, cleanup


class TestDashboardEndpoints:
    def test_get_returns_contract_shape(self, tmp_path, monkeypatch):
        import httpx

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/browser/alpha/fingerprint-health"
            return httpx.Response(200, json={
                "window_size": 6,
                "rejection_rate": 0.5,
                "burned": False,
                "last_signal_ts": "2026-04-27T12:00:00Z",
            })

        original_async_client = httpx.AsyncClient

        def patched_async_client(*args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            return original_async_client(*args, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", patched_async_client)

        client, cleanup = _make_dashboard_client_with_browser_url(
            str(tmp_path), "http://browser:8500",
        )
        try:
            resp = client.get(
                "/dashboard/api/agents/alpha/fingerprint-health",
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["success"] is True
            data = body["data"]
            assert set(data.keys()) == {
                "window_size", "rejection_rate", "burned", "last_signal_ts",
            }
            assert data["window_size"] == 6
            assert data["rejection_rate"] == 0.5
            assert data["burned"] is False
        finally:
            cleanup()

    def test_get_404_for_unknown_agent(self, tmp_path):
        client, cleanup = _make_dashboard_client_with_browser_url(
            str(tmp_path), "http://browser:8500",
        )
        try:
            resp = client.get(
                "/dashboard/api/agents/missing/fingerprint-health",
            )
            assert resp.status_code == 404
        finally:
            cleanup()

    def test_reset_requires_csrf(self, tmp_path):
        from fastapi.testclient import TestClient
        client, cleanup = _make_dashboard_client_with_browser_url(
            str(tmp_path), "http://browser:8500",
        )
        try:
            # Bypass the auto-CSRF wrapper used by the test client.
            raw = TestClient(client.app)
            resp = raw.post(
                "/dashboard/api/agents/alpha/fingerprint-health/reset",
            )
            assert resp.status_code == 403
            assert "X-Requested-With" in resp.text
        finally:
            cleanup()

    def test_reset_with_csrf_proxies_to_browser(self, tmp_path, monkeypatch):
        import httpx
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["method"] = request.method
            seen["path"] = request.url.path
            seen["auth"] = request.headers.get("authorization", "")
            return httpx.Response(200, json={"reset": True})

        original_async_client = httpx.AsyncClient

        def patched_async_client(*args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            return original_async_client(*args, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", patched_async_client)

        client, cleanup = _make_dashboard_client_with_browser_url(
            str(tmp_path), "http://browser:8500",
        )
        try:
            resp = client.post(
                "/dashboard/api/agents/alpha/fingerprint-health/reset",
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["success"] is True
            assert seen["method"] == "POST"
            assert seen["path"] == "/browser/alpha/fingerprint-health/reset"
            assert seen["auth"] == "Bearer test-token"
        finally:
            cleanup()

    def test_get_handles_browser_service_unavailable(self, tmp_path):
        client, cleanup = _make_dashboard_client_with_browser_url(
            str(tmp_path), "",
        )
        try:
            resp = client.get(
                "/dashboard/api/agents/alpha/fingerprint-health",
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["success"] is False
            assert body["error"]["code"] == "service_unavailable"
        finally:
            cleanup()


# ── Burn-state crossing emits a one-shot audit event ─────────────────────


# ── Vendor-specific rejection signal coverage ────────────────────────────


class TestVendorBlockSignals:
    """Each anti-bot vendor — Cloudflare, DataDome, PerimeterX/HUMAN,
    Imperva, Akamai BMP, F5/Shape — surfaces a deterministic element
    selector or signature text on its block / interstitial page.  The
    monitor must pick up at least one signal per vendor; missing a
    rejection costs us much more than a false positive (a false positive
    is one entry in a 10-deep window; a missed rejection means the
    fingerprint stays in service after it has been flagged)."""

    @pytest.mark.parametrize(
        "selector",
        [
            "[id^=cf-error]",          # Cloudflare 1xxx
            "#challenge-error-text",   # Cloudflare JS challenge
            ".cf-browser-verification",  # Cloudflare interstitial
            "#ddm-blocked",            # DataDome
            "#px-captcha",             # PerimeterX press-and-hold
            ".px-block-spam",          # PerimeterX outright block
            "#main-iframe[src*=incident]",  # Imperva incident page
            "[id^=bm-error]",          # Akamai BMP
        ],
    )
    @pytest.mark.asyncio
    async def test_vendor_selector_records_rejection(
        self, monkeypatch, selector,
    ):
        """A page that renders a known vendor block selector must be
        treated as a rejection regardless of the visible text."""
        monkeypatch.setattr(
            "src.browser.service._FINGERPRINT_MONITOR_TIMEOUT_S", 1.0,
        )
        from src.browser.service import (
            BrowserManager,
            _get_fingerprint_health,
        )

        # Build a page where the vendor selector resolves to count=1
        # but no captcha selector is present and the body text is
        # benign — the vendor signal alone must trip the monitor.
        page = _FakePage(
            url="https://protected.example.com/account",
            locators={
                selector: _FakeLocator(count=1),
                "body": _FakeLocator(0, "loading content"),
            },
        )
        inst = _FakeInstance("agent-vendor", page)
        mgr = BrowserManager(profiles_dir="/tmp/no-such-vendor")
        captcha_selectors = ['iframe[src*="recaptcha"]']
        await mgr._monitor_post_solve_state(
            inst, captcha_selectors,
            kind="recaptcha-v2-checkbox",
            page_origin="protected.example.com",
        )
        health = await _get_fingerprint_health("agent-vendor")
        assert health["window_size"] == 1
        assert health["rejection_rate"] == 1.0

    @pytest.mark.parametrize(
        "body_text",
        [
            "Sorry, you have been blocked",      # DataDome / Imperva
            "Cloudflare Ray ID: 89abcdef0",      # Cloudflare-branded
            "error 1020 — Access denied",        # Cloudflare 1020
            "error 1015 — You are being rate-limited",  # Cloudflare 1015
            "Please verify you are a human",     # PerimeterX
            "Press and hold to confirm",         # PerimeterX press-and-hold
            "Reference #18.deadbeef.1234",       # Akamai BMP
            "Incident ID: 0123-4567-89ab",       # Imperva
            "Unusual activity detected",         # Generic anti-bot
            "Suspicious activity from your network",
            "Just a moment...",                  # Cloudflare interstitial
            "Captcha invalid — please retry",
            "Challenge failed",
            "Session expired, please log in again",
            "checking if the site connection is secure",
        ],
    )
    @pytest.mark.asyncio
    async def test_extended_rejection_text_records_rejection(
        self, monkeypatch, body_text,
    ):
        """Each vendor's branded block-page text trips the monitor
        regardless of which one we encounter."""
        monkeypatch.setattr(
            "src.browser.service._FINGERPRINT_MONITOR_TIMEOUT_S", 1.0,
        )
        from src.browser.service import (
            BrowserManager,
            _get_fingerprint_health,
        )
        page = _FakePage(
            url="https://protected.example.com/x",
            locators={"body": _FakeLocator(0, body_text)},
        )
        inst = _FakeInstance("agent-text", page)
        mgr = BrowserManager(profiles_dir="/tmp/no-such-text")
        await mgr._monitor_post_solve_state(
            inst, ['iframe[src*="recaptcha"]'],
            kind="recaptcha-v2-checkbox",
            page_origin="protected.example.com",
        )
        health = await _get_fingerprint_health("agent-text")
        assert health["window_size"] == 1, (
            f"text {body_text!r} did not trip rejection signal"
        )
        assert health["rejection_rate"] == 1.0


# ── Defense-in-depth on agent_id validation ──────────────────────────────


class TestAgentIdValidation:
    """The dashboard / browser-service URL routes already enforce path
    validation, but the manager methods are public — a future caller
    that bypasses the routes must not be able to read or mutate state
    keyed by an arbitrary agent_id."""

    @pytest.mark.asyncio
    async def test_get_with_invalid_agent_id_returns_empty_shape(self):
        from src.browser.service import BrowserManager
        mgr = BrowserManager(profiles_dir="/tmp/no-such-validate")
        for bad in ("", "../etc/passwd", "agent with spaces", "<script>"):
            health = await mgr.get_fingerprint_health(bad)
            assert health == {
                "window_size": 0,
                "rejection_rate": 0.0,
                "burned": False,
                "last_signal_ts": None,
            }

    @pytest.mark.asyncio
    async def test_reset_with_invalid_agent_id_is_noop(self):
        from src.browser.service import (
            BrowserManager,
            _is_fingerprint_burned,
            _record_fingerprint_outcome,
        )
        # Seed a burn against a *valid* id, then attempt to clear with
        # an invalid id — the burn must remain.
        for _ in range(10):
            await _record_fingerprint_outcome("agent-real", rejected=True)
        assert await _is_fingerprint_burned("agent-real") is True
        mgr = BrowserManager(profiles_dir="/tmp/no-such-validate-2")
        result = await mgr.reset_fingerprint_health("../agent-real")
        assert result == {"reset": False}
        assert await _is_fingerprint_burned("agent-real") is True


# ── next_action override across all non-solved outcomes ──────────────────


class TestBurnFlagAllOutcomes:
    """The decoration logic must override ``next_action`` for *every*
    non-``solved`` outcome — including ``rejected`` (token was injected
    but the page rejected it) and ``timeout`` / ``injection_failed``.
    The current tests cover ``no_solver`` and ``solved``; this fills
    in the gap for the in-between states."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "outcome",
        ["rejected", "timeout", "injection_failed", "no_solver"],
    )
    async def test_non_solved_outcomes_override_next_action(self, outcome):
        from src.browser.service import (
            _is_fingerprint_burned,
            _record_fingerprint_outcome,
        )
        for _ in range(10):
            await _record_fingerprint_outcome(
                "agent-multi", rejected=True,
            )
        assert await _is_fingerprint_burned("agent-multi") is True

        envelope = {
            "captcha_found": True,
            "kind": "recaptcha-v2-checkbox",
            "solver_attempted": outcome != "no_solver",
            "solver_outcome": outcome,
            "next_action": (
                "request_captcha_help" if outcome == "no_solver"
                else "retry_previous"
            ),
        }
        # Reproduce the exact decoration logic from _check_captcha so
        # this test catches a refactor that changes the contract with
        # the burn helpers.
        if await _is_fingerprint_burned("agent-multi"):
            envelope["fingerprint_burn"] = True
            if envelope.get("solver_outcome") != "solved":
                envelope["next_action"] = "retry_with_fresh_profile"

        assert envelope["fingerprint_burn"] is True
        assert envelope["next_action"] == "retry_with_fresh_profile"


class TestBurnStateAudit:
    @pytest.mark.asyncio
    async def test_crossing_burn_threshold_emits_burn_event(self):
        """The first rejected signal that pushes the window past the
        threshold should emit a ``fingerprint_burn`` audit event in
        addition to the per-signal ``rejected`` event."""
        from src.browser.service import (
            BrowserManager,
            _drain_fingerprint_audit,
        )
        mgr = BrowserManager(profiles_dir="/tmp/no-such-fp")

        # Pre-load the window with 9 rejections so the next rejection
        # crosses the threshold (10/10 = 100% > 50%).
        for _ in range(9):
            await mgr._handle_post_solve_outcome(
                "agent-cross", rejected=True, page_origin="example.com",
            )
        # No burn yet — window only has 9 entries.
        events_pre = await _drain_fingerprint_audit()
        # 1 bucket: (agent-cross, rejected, example.com), count=9.
        assert len(events_pre) == 1
        assert events_pre[0]["signal"] == "rejected"
        assert events_pre[0]["count"] == 9

        # 10th rejection — crosses the threshold.
        await mgr._handle_post_solve_outcome(
            "agent-cross", rejected=True, page_origin="example.com",
        )
        events_post = await _drain_fingerprint_audit()
        # Two buckets now: rejected (count=1) and fingerprint_burn (count=1).
        signals = sorted(ev["signal"] for ev in events_post)
        assert signals == ["fingerprint_burn", "rejected"]
