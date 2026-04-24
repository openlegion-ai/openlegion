"""Tests for the mesh-side browser-metrics poll loop (Phase 2 §5.1/§5.2).

The browser service container exposes ``GET /browser/metrics?since=<seq>``.
The mesh host polls it every 60s and fans each new per-agent payload out
as a ``browser_metrics`` event on the dashboard EventBus.

These tests exercise ``_poll_browser_metrics_once`` in isolation: the
cadence loop itself is just ``asyncio.sleep`` + retries and not worth
testing separately.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _build_mesh(tmp_path):
    """Minimal mesh app with a fake container_manager pointing at a
    fake browser service URL. Returns ``(app, event_bus, container_manager)``.

    We build the app via ``create_mesh_app`` so the poller inside its
    closure is wired normally. The tests then patch the module-level
    ``httpx.AsyncClient.get`` to return canned metric payloads.
    """
    from src.host.costs import CostTracker
    from src.host.mesh import Blackboard, MessageRouter, PubSub
    from src.host.permissions import PermissionMatrix
    from src.host.server import create_mesh_app
    from src.host.traces import TraceStore

    blackboard = Blackboard(str(tmp_path / "bb.db"))
    pubsub = PubSub()
    permissions = PermissionMatrix()
    router = MessageRouter(permissions, {})
    costs = CostTracker(str(tmp_path / "costs.db"))
    traces = TraceStore(str(tmp_path / "traces.db"))

    cm = MagicMock()
    cm.browser_service_url = "http://browser-svc:8500"
    cm.browser_auth_token = "t0k"

    event_bus = MagicMock()

    app = create_mesh_app(
        blackboard=blackboard,
        pubsub=pubsub,
        router=router,
        permissions=permissions,
        cost_tracker=costs,
        trace_store=traces,
        event_bus=event_bus,
        container_manager=cm,
    )
    return app, event_bus, cm


class TestMetricsPoll:
    @pytest.mark.asyncio
    async def test_poll_emits_browser_metrics_events(self, tmp_path, monkeypatch):
        """A fresh poll fetches from /browser/metrics and emits one
        browser_metrics event per payload agent to the EventBus."""
        import httpx

        # Canned response from the fake browser service.
        canned = {
            "current_seq": 2,
            "boot_id": "boot-1",
            "metrics": [
                {
                    "seq": 1,
                    "ts": 1000.0,
                    "agent_id": "a1",
                    "click_success": 5,
                    "click_fail": 0,
                    "nav_timeout": 0,
                    "snapshot_count": 3,
                    "snapshot_bytes_p50": 400,
                    "snapshot_bytes_p95": 900,
                    "click_window_size": 10,
                    "click_success_rate_100": 1.0,
                },
                {
                    "seq": 2,
                    "ts": 1001.0,
                    "agent_id": "a2",
                    "click_success": 1,
                    "click_fail": 4,
                    "nav_timeout": 2,
                    "snapshot_count": 1,
                    "snapshot_bytes_p50": 200,
                    "snapshot_bytes_p95": 200,
                    "click_window_size": 20,
                    "click_success_rate_100": 0.5,
                },
            ],
        }

        async def fake_get(self, url, *args, **kwargs):
            req = httpx.Request("GET", url)
            return httpx.Response(200, json=canned, request=req)

        monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

        # Build app and manually invoke the poll. We avoid relying on
        # FastAPI startup events firing: instead, find the captured
        # closure via the app's routes' lifespan hook by calling the
        # registered startup function directly.
        app, event_bus, _cm = _build_mesh(tmp_path)

        # FastAPI stores startup handlers on app.router.on_startup.
        handlers = list(app.router.on_startup)
        assert handlers, "expected at least one startup handler"

        # Poll function is closed over by _browser_metrics_loop which is
        # scheduled by one of those handlers. Rather than scheduling the
        # loop (which sleeps 5s), we reach into the closure directly.
        loop_handler = None
        for h in handlers:
            # The handler that schedules the metrics loop is named
            # ``_start_browser_metrics_poll`` in the source.
            if h.__name__ == "_start_browser_metrics_poll":
                loop_handler = h
                break
        assert loop_handler is not None, "missing _start_browser_metrics_poll"

        # Walk the closure to find _poll_browser_metrics_once (via the
        # loop function closure). The loop function is created inside
        # create_mesh_app, so we grab it from the start handler's
        # co_freevars binding.
        poll_fn = _extract_poll_fn(app)
        assert poll_fn is not None

        await poll_fn()

        # Both agents should have emitted a browser_metrics event.
        calls = [c for c in event_bus.emit.call_args_list
                 if c.args and c.args[0] == "browser_metrics"]
        assert len(calls) == 2
        agents = [c.kwargs.get("agent") or (c.args[1] if len(c.args) > 1 else "")
                  for c in calls]
        assert set(agents) == {"a1", "a2"}
        # Payload round-trip
        for c in calls:
            data = c.kwargs.get("data") or (c.args[2] if len(c.args) > 2 else {})
            assert "click_success_rate_100" in data

    @pytest.mark.asyncio
    async def test_poll_skips_already_seen_seqs(self, tmp_path, monkeypatch):
        """Second poll with same seqs emits nothing (idempotent)."""
        import httpx

        canned = {
            "current_seq": 1,
            "boot_id": "b1",
            "metrics": [{
                "seq": 1, "ts": 1.0, "agent_id": "a1",
                "click_success": 1, "click_fail": 0, "nav_timeout": 0,
                "snapshot_count": 0, "snapshot_bytes_p50": 0,
                "snapshot_bytes_p95": 0, "click_window_size": 0,
                "click_success_rate_100": None,
            }],
        }

        # On the second call the browser would return {metrics: []} with
        # the same current_seq since nothing new happened.
        call_count = {"n": 0}
        responses = [canned, {"current_seq": 1, "boot_id": "b1", "metrics": []}]

        async def fake_get(self, url, *args, **kwargs):
            idx = min(call_count["n"], len(responses) - 1)
            call_count["n"] += 1
            req = httpx.Request("GET", url)
            return httpx.Response(200, json=responses[idx], request=req)

        monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

        app, event_bus, _cm = _build_mesh(tmp_path)
        poll = _extract_poll_fn(app)

        await poll()
        await poll()

        metric_emits = [c for c in event_bus.emit.call_args_list
                        if c.args and c.args[0] == "browser_metrics"]
        assert len(metric_emits) == 1  # only the first tick carried a new seq

    @pytest.mark.asyncio
    async def test_browser_restart_no_data_loss_on_first_poll(
        self, tmp_path, monkeypatch,
    ):
        """Regression (Codex #2 P1): after a browser restart the mesh's
        cached high-water `since` is stale — post-restart seqs start at
        1, which fails the browser's `seq > since` filter. The mesh
        must re-poll with `since=0` the moment it detects the new
        boot_id, before those payloads scroll off the browser's history.
        """
        import httpx

        responses = [
            # First poll: stale since=100 ; browser restarted and seqs
            # now go 1..3 but the filter drops them all.
            {"current_seq": 3, "boot_id": "boot-B", "metrics": []},
            # Second request (the immediate re-poll triggered by boot_id
            # change) with since=0 returns the post-restart data.
            {
                "current_seq": 3, "boot_id": "boot-B", "metrics": [
                    {"seq": 1, "ts": 1.0, "agent_id": "a1",
                     "click_success": 5, "click_fail": 0, "nav_timeout": 0,
                     "snapshot_count": 0, "snapshot_bytes_p50": 0,
                     "snapshot_bytes_p95": 0, "click_window_size": 0,
                     "click_success_rate_100": None},
                ],
            },
        ]
        call_count = {"n": 0}

        async def fake_get(self, url, *args, **kwargs):
            idx = min(call_count["n"], len(responses) - 1)
            call_count["n"] += 1
            req = httpx.Request("GET", url)
            return httpx.Response(200, json=responses[idx], request=req)

        monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

        app, event_bus, _cm = _build_mesh(tmp_path)
        # Seed the mesh with a stale high-water so the first poll
        # hits the filter-drops-everything case.
        app.state.browser_metrics_poll_state["boot_id"] = "boot-A"
        app.state.browser_metrics_poll_state["last_seen_seq"] = 100

        poll = _extract_poll_fn(app)
        await poll()

        # The post-restart payload must have reached the EventBus.
        metric_emits = [c for c in event_bus.emit.call_args_list
                        if c.args and c.args[0] == "browser_metrics"]
        assert len(metric_emits) == 1, (
            "expected 1 browser_metrics event after restart recovery; "
            "got " + repr(metric_emits)
        )
        payload = metric_emits[0].kwargs["data"]
        assert payload["seq"] == 1
        assert payload["click_success"] == 5
        # And the HTTP client was actually called twice (original + re-poll).
        assert call_count["n"] == 2

    @pytest.mark.asyncio
    async def test_poll_resets_watermark_on_boot_id_change(
        self, tmp_path, monkeypatch,
    ):
        """Browser service restart → boot_id changes → seq counter resets
        to 0 inside the browser. The poller must detect this and re-emit
        post-restart seqs (which would otherwise be <= high-water)."""
        import httpx

        responses = [
            {
                "current_seq": 5,
                "boot_id": "boot-A",
                "metrics": [{
                    "seq": 5, "ts": 1.0, "agent_id": "a1",
                    "click_success": 1, "click_fail": 0, "nav_timeout": 0,
                    "snapshot_count": 0, "snapshot_bytes_p50": 0,
                    "snapshot_bytes_p95": 0, "click_window_size": 0,
                    "click_success_rate_100": None,
                }],
            },
            # Service restarted — boot_id changed, seq reset
            {
                "current_seq": 1,
                "boot_id": "boot-B",
                "metrics": [{
                    "seq": 1, "ts": 2.0, "agent_id": "a1",
                    "click_success": 2, "click_fail": 0, "nav_timeout": 0,
                    "snapshot_count": 0, "snapshot_bytes_p50": 0,
                    "snapshot_bytes_p95": 0, "click_window_size": 0,
                    "click_success_rate_100": None,
                }],
            },
        ]
        call_count = {"n": 0}

        async def fake_get(self, url, *args, **kwargs):
            idx = min(call_count["n"], len(responses) - 1)
            call_count["n"] += 1
            req = httpx.Request("GET", url)
            return httpx.Response(200, json=responses[idx], request=req)

        monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

        app, event_bus, _cm = _build_mesh(tmp_path)
        poll = _extract_poll_fn(app)

        await poll()
        await poll()

        metric_emits = [c for c in event_bus.emit.call_args_list
                        if c.args and c.args[0] == "browser_metrics"]
        # Both post-boot-A seq=5 and post-boot-B seq=1 should have been
        # surfaced — restart must not swallow metrics.
        assert len(metric_emits) == 2

    @pytest.mark.asyncio
    async def test_poll_tolerates_browser_service_error(
        self, tmp_path, monkeypatch,
    ):
        """A 5xx or transport error must not crash the poll loop — metrics
        are best-effort observability, not correctness."""
        import httpx

        async def fake_get(self, url, *args, **kwargs):
            raise httpx.ConnectError("simulated down")

        monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

        app, event_bus, _cm = _build_mesh(tmp_path)
        poll = _extract_poll_fn(app)

        await poll()  # must not raise
        metric_emits = [c for c in event_bus.emit.call_args_list
                        if c.args and c.args[0] == "browser_metrics"]
        assert len(metric_emits) == 0

    @pytest.mark.asyncio
    async def test_first_seen_collapses_to_latest_per_agent(
        self, tmp_path, monkeypatch,
    ):
        """On boot-id-first-seen (fresh mesh OR browser restart), the
        browser's history buffer may contain hours of stale payloads per
        agent. Replaying all of them would flood the dashboard's 500-event
        ring buffer and evict live signals. Only the latest seq per agent
        should emit.
        """
        import httpx

        # Browser has 3 entries for a1, 2 entries for a2 — all from the
        # same boot. Only the highest-seq payload per agent should reach
        # the EventBus on the first poll.
        canned = {
            "current_seq": 5,
            "boot_id": "boot-new",
            "metrics": [
                {"seq": 1, "ts": 1.0, "agent_id": "a1", "click_success": 1,
                 "click_fail": 0, "nav_timeout": 0, "snapshot_count": 0,
                 "snapshot_bytes_p50": 0, "snapshot_bytes_p95": 0,
                 "click_window_size": 0, "click_success_rate_100": None},
                {"seq": 2, "ts": 2.0, "agent_id": "a2", "click_success": 2,
                 "click_fail": 0, "nav_timeout": 0, "snapshot_count": 0,
                 "snapshot_bytes_p50": 0, "snapshot_bytes_p95": 0,
                 "click_window_size": 0, "click_success_rate_100": None},
                {"seq": 3, "ts": 3.0, "agent_id": "a1", "click_success": 3,
                 "click_fail": 0, "nav_timeout": 0, "snapshot_count": 0,
                 "snapshot_bytes_p50": 0, "snapshot_bytes_p95": 0,
                 "click_window_size": 0, "click_success_rate_100": None},
                {"seq": 4, "ts": 4.0, "agent_id": "a2", "click_success": 4,
                 "click_fail": 0, "nav_timeout": 0, "snapshot_count": 0,
                 "snapshot_bytes_p50": 0, "snapshot_bytes_p95": 0,
                 "click_window_size": 0, "click_success_rate_100": None},
                {"seq": 5, "ts": 5.0, "agent_id": "a1", "click_success": 5,
                 "click_fail": 0, "nav_timeout": 0, "snapshot_count": 0,
                 "snapshot_bytes_p50": 0, "snapshot_bytes_p95": 0,
                 "click_window_size": 0, "click_success_rate_100": None},
            ],
        }

        async def fake_get(self, url, *args, **kwargs):
            req = httpx.Request("GET", url)
            return httpx.Response(200, json=canned, request=req)

        monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

        app, event_bus, _cm = _build_mesh(tmp_path)
        poll = _extract_poll_fn(app)
        await poll()

        metric_emits = [c for c in event_bus.emit.call_args_list
                        if c.args and c.args[0] == "browser_metrics"]
        # Exactly 2 events (latest per agent), not 5.
        assert len(metric_emits) == 2
        agents_emitted = {}
        for c in metric_emits:
            data = c.kwargs.get("data") or {}
            agents_emitted[c.kwargs.get("agent")] = data["seq"]
        assert agents_emitted == {"a1": 5, "a2": 4}

    @pytest.mark.asyncio
    async def test_persistent_failures_escalate_to_warning(
        self, tmp_path, monkeypatch, caplog,
    ):
        """Silent debug-level failures hide real outages. After the
        threshold (5 ticks), the mesh must log at WARNING so operators
        can diagnose 'why no metrics'."""
        import logging

        import httpx

        async def always_fails(self, url, *args, **kwargs):
            raise httpx.ConnectError("simulated down")

        monkeypatch.setattr(httpx.AsyncClient, "get", always_fails)

        app, _event_bus, _cm = _build_mesh(tmp_path)
        poll = _extract_poll_fn(app)

        caplog.set_level(logging.WARNING)
        # Hit the threshold (5 consecutive) — fifth call must warn.
        for _ in range(5):
            await poll()

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("consecutive" in r.getMessage() for r in warnings), (
            "expected consecutive-failure warning; got: "
            + repr([r.getMessage() for r in warnings])
        )

    @pytest.mark.asyncio
    async def test_recovery_logs_info(self, tmp_path, monkeypatch, caplog):
        """After a noisy failure streak, a success must log at INFO so
        operators see the all-clear."""
        import logging

        import httpx

        call_count = {"n": 0}

        async def fail_then_succeed(self, url, *args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 5:
                raise httpx.ConnectError("down")
            req = httpx.Request("GET", url)
            return httpx.Response(200, json={
                "current_seq": 0, "boot_id": "b", "metrics": [],
            }, request=req)

        monkeypatch.setattr(httpx.AsyncClient, "get", fail_then_succeed)

        app, _event_bus, _cm = _build_mesh(tmp_path)
        poll = _extract_poll_fn(app)

        caplog.set_level(logging.INFO)
        for _ in range(6):
            await poll()

        info = [r for r in caplog.records if r.levelno == logging.INFO]
        assert any("recovered" in r.getMessage() for r in info)

    @pytest.mark.asyncio
    async def test_poll_state_accessible_via_app_state(self, tmp_path):
        """The poll primitive + state are exposed on ``app.state`` so
        tests and future admin endpoints can reach them without walking
        closure cells."""
        app, _event_bus, _cm = _build_mesh(tmp_path)
        assert callable(getattr(app.state, "poll_browser_metrics_once", None))
        state = getattr(app.state, "browser_metrics_poll_state", None)
        assert isinstance(state, dict)
        assert "consecutive_failures" in state
        assert "last_seen_seq" in state

    @pytest.mark.asyncio
    async def test_poll_noop_without_browser_service(self, tmp_path, monkeypatch):
        """If container_manager has no browser_service_url, the poll is
        a silent no-op — mesh configurations without a browser shouldn't
        log spurious errors."""
        import httpx

        def boom(*_a, **_k):
            raise AssertionError("HTTP client should not be called")

        monkeypatch.setattr(httpx.AsyncClient, "get", boom)

        app, event_bus, cm = _build_mesh(tmp_path)
        cm.browser_service_url = None
        poll = _extract_poll_fn(app)
        await poll()
        assert event_bus.emit.call_count == 0


# ── Helpers ─────────────────────────────────────────────────────────


def _extract_poll_fn(app):
    """Return ``_poll_browser_metrics_once`` from the mesh app.

    The poll primitive is exposed on ``app.state`` precisely so tests
    (and future admin endpoints) can reach it without walking closure
    cells. Raises AssertionError if missing — a noisy failure here
    means production wiring broke.
    """
    fn = getattr(app.state, "poll_browser_metrics_once", None)
    assert fn is not None, (
        "app.state.poll_browser_metrics_once missing — did create_mesh_app "
        "stop publishing the poll primitive?"
    )
    return fn
