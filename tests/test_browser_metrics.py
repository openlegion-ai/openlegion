"""Tests for per-agent browser metrics counters + minute-tick emitter (§4.6)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.browser.service import BrowserManager, CamoufoxInstance


def _new_instance(agent_id: str = "a1") -> CamoufoxInstance:
    """Build a CamoufoxInstance with MagicMock-backed page/context.

    Only the counter fields are exercised; the rest is mocked out.
    """
    return CamoufoxInstance(
        agent_id=agent_id,
        browser=MagicMock(),
        context=MagicMock(),
        page=MagicMock(),
    )


class TestCounterInitialState:
    def test_all_counters_start_at_zero(self):
        inst = _new_instance()
        assert inst.m_click_success == 0
        assert inst.m_click_fail == 0
        assert inst.m_nav_timeout == 0
        assert inst.m_snapshot_bytes == []
        assert len(inst.click_window) == 0
        # Empty window must read as "unknown", not 0% — would falsely
        # suggest catastrophic failure on a freshly-booted agent.
        assert inst.rolling_click_success_rate() is None


class TestRollingClickWindow:
    def test_counts_success_rate(self):
        inst = _new_instance()
        for _ in range(8):
            inst.click_window.append(True)
        for _ in range(2):
            inst.click_window.append(False)
        assert inst.rolling_click_success_rate() == 0.8
        assert len(inst.click_window) == 10

    def test_window_capped_at_100(self):
        inst = _new_instance()
        for i in range(150):
            inst.click_window.append(True)
        assert len(inst.click_window) == 100
        assert inst.rolling_click_success_rate() == 1.0

    def test_window_evicts_old_entries(self):
        """Older failures age out past the 100-sample window."""
        inst = _new_instance()
        # 50 failures, then 100 successes — the failures fall off the
        # head once the deque reaches its maxlen.
        for _ in range(50):
            inst.click_window.append(False)
        for _ in range(100):
            inst.click_window.append(True)
        assert len(inst.click_window) == 100
        assert inst.rolling_click_success_rate() == 1.0

    def test_window_survives_drain(self):
        """Rolling window persists across per-minute emits — it's a
        longer-horizon health signal than the per-minute counters."""
        inst = _new_instance()
        for _ in range(10):
            inst.click_window.append(True)
        payload = inst.drain_metrics()
        assert payload["click_window_size"] == 10
        assert payload["click_success_rate_100"] == 1.0
        # Window survives the drain; per-minute counters reset.
        assert len(inst.click_window) == 10

    def test_drain_reports_none_when_empty(self):
        inst = _new_instance()
        payload = inst.drain_metrics()
        assert payload["click_window_size"] == 0
        assert payload["click_success_rate_100"] is None


class TestDrainMetrics:
    def test_returns_snapshot_and_resets_counters(self):
        inst = _new_instance()
        inst.m_click_success = 7
        inst.m_click_fail = 2
        inst.m_nav_timeout = 1
        inst.m_snapshot_bytes = [100, 200, 300]

        payload = inst.drain_metrics()

        assert payload["agent_id"] == "a1"
        assert payload["click_success"] == 7
        assert payload["click_fail"] == 2
        assert payload["nav_timeout"] == 1
        assert payload["snapshot_count"] == 3
        # p50 of [100,200,300] is the middle value (200).
        assert payload["snapshot_bytes_p50"] == 200
        # p95 index: int(3 * 0.95) = 2 → value 300.
        assert payload["snapshot_bytes_p95"] == 300

        # Counters reset after drain.
        assert inst.m_click_success == 0
        assert inst.m_click_fail == 0
        assert inst.m_nav_timeout == 0
        assert inst.m_snapshot_bytes == []

    def test_empty_snapshot_bytes_returns_zero_percentiles(self):
        inst = _new_instance()
        payload = inst.drain_metrics()
        assert payload["snapshot_count"] == 0
        assert payload["snapshot_bytes_p50"] == 0
        assert payload["snapshot_bytes_p95"] == 0

    def test_second_drain_sees_fresh_counts(self):
        inst = _new_instance()
        inst.m_click_success = 5
        inst.drain_metrics()
        inst.m_click_success = 2
        assert inst.drain_metrics()["click_success"] == 2


class TestEmitMetrics:
    @pytest.mark.asyncio
    async def test_emit_calls_sink_per_instance(self, tmp_path):
        sink_calls: list[dict] = []
        mgr = BrowserManager(
            profiles_dir=str(tmp_path / "profiles"),
            metrics_sink=sink_calls.append,
        )
        mgr._instances = {
            "a1": _new_instance("a1"),
            "a2": _new_instance("a2"),
        }
        mgr._instances["a1"].m_click_success = 3
        mgr._instances["a2"].m_click_fail = 1

        await mgr._emit_metrics()

        assert len(sink_calls) == 2
        by_agent = {c["agent_id"]: c for c in sink_calls}
        assert by_agent["a1"]["click_success"] == 3
        assert by_agent["a2"]["click_fail"] == 1

    @pytest.mark.asyncio
    async def test_emit_no_sink_still_resets(self, tmp_path):
        """When no sink is wired, counters still reset so memory doesn't grow."""
        mgr = BrowserManager(
            profiles_dir=str(tmp_path / "profiles"),
            metrics_sink=None,
        )
        inst = _new_instance()
        mgr._instances = {"a1": inst}
        inst.m_click_success = 99
        await mgr._emit_metrics()
        assert inst.m_click_success == 0

    @pytest.mark.asyncio
    async def test_emit_without_instances_is_noop(self, tmp_path):
        sink_calls: list[dict] = []
        mgr = BrowserManager(
            profiles_dir=str(tmp_path / "profiles"),
            metrics_sink=sink_calls.append,
        )
        await mgr._emit_metrics()
        assert sink_calls == []

    @pytest.mark.asyncio
    async def test_sink_exception_does_not_block_other_agents(self, tmp_path):
        """A sink that raises for one agent must not suppress others."""
        delivered: list[str] = []

        def flaky_sink(payload):
            if payload["agent_id"] == "bad":
                raise RuntimeError("sink failure")
            delivered.append(payload["agent_id"])

        mgr = BrowserManager(
            profiles_dir=str(tmp_path / "profiles"),
            metrics_sink=flaky_sink,
        )
        mgr._instances = {
            "bad": _new_instance("bad"),
            "good": _new_instance("good"),
        }
        # Both need activity to avoid the _is_empty_payload skip —
        # zero-counter payloads are intentionally filtered out.
        mgr._instances["bad"].m_click_success = 1
        mgr._instances["good"].m_click_success = 1
        await mgr._emit_metrics()
        assert "good" in delivered


class TestStopInstanceDrainsCounters:
    """_stop_instance() must emit counters before popping the instance from
    the fleet — otherwise idle cleanup / LRU eviction silently drops the
    last interval's metrics."""

    @pytest.mark.asyncio
    async def test_stop_drains_metrics_to_sink(self, tmp_path):
        sink_calls: list[dict] = []
        mgr = BrowserManager(
            profiles_dir=str(tmp_path / "profiles"),
            metrics_sink=sink_calls.append,
        )
        inst = _new_instance("bye")
        inst.context = MagicMock()
        inst.context.close = AsyncMock()
        mgr._instances["bye"] = inst
        inst.m_click_success = 17
        inst.m_snapshot_bytes = [100, 200]

        # Stop under the manager's lock as callers do.
        async with mgr._lock:
            await mgr._stop_instance("bye")

        # Instance removed AND its counters made it to the sink.
        assert "bye" not in mgr._instances
        assert len(sink_calls) == 1
        assert sink_calls[0]["agent_id"] == "bye"
        assert sink_calls[0]["click_success"] == 17
        assert sink_calls[0]["snapshot_count"] == 2

    @pytest.mark.asyncio
    async def test_stop_with_no_sink_does_not_raise(self, tmp_path):
        """Metrics-sink-less BrowserManager must still stop cleanly."""
        mgr = BrowserManager(
            profiles_dir=str(tmp_path / "profiles"),
            metrics_sink=None,
        )
        inst = _new_instance("bye")
        inst.context = MagicMock()
        inst.context.close = AsyncMock()
        mgr._instances["bye"] = inst
        inst.m_click_success = 5
        async with mgr._lock:
            await mgr._stop_instance("bye")
        assert "bye" not in mgr._instances


class TestCleanupLoopIntegration:
    @pytest.mark.asyncio
    async def test_cleanup_loop_calls_emit_metrics(self, tmp_path, monkeypatch):
        """Regression guard: the minute-tick path runs both idle cleanup AND
        metric emission. Break either order and this fails."""
        sink_calls: list[dict] = []
        mgr = BrowserManager(
            profiles_dir=str(tmp_path / "profiles"),
            metrics_sink=sink_calls.append,
        )
        mgr._instances = {"a1": _new_instance()}
        mgr._instances["a1"].m_click_success = 1

        # Let the first sleep return normally (so the loop body runs: idle
        # cleanup + metric emit), then raise on the second sleep to break
        # the infinite loop.
        call_count = {"n": 0}

        async def _sleep_then_break(_):
            call_count["n"] += 1
            if call_count["n"] >= 2:
                raise asyncio.CancelledError

        monkeypatch.setattr(
            "src.browser.service.asyncio.sleep", _sleep_then_break,
        )

        with pytest.raises(asyncio.CancelledError):
            await mgr._cleanup_loop()
        assert len(sink_calls) == 1
        assert sink_calls[0]["click_success"] == 1


class TestMetricsHistoryBuffer:
    """§5.1/§5.2: the mesh polls /browser/metrics?since=<seq> to forward
    aggregate payloads into the EventBus. This test class covers the
    browser-side buffer + cursor semantics the poller relies on.
    """

    @pytest.mark.asyncio
    async def test_emit_writes_to_history(self, tmp_path):
        mgr = BrowserManager(profiles_dir=str(tmp_path / "profiles"))
        mgr._instances = {"a1": _new_instance("a1")}
        mgr._instances["a1"].m_click_success = 3

        await mgr._emit_metrics()

        snap = mgr.get_recent_metrics(since_seq=0)
        assert snap["current_seq"] == 1
        assert snap["boot_id"] == mgr.boot_id
        assert len(snap["metrics"]) == 1
        assert snap["metrics"][0]["agent_id"] == "a1"
        assert snap["metrics"][0]["click_success"] == 3
        assert snap["metrics"][0]["seq"] == 1
        assert "ts" in snap["metrics"][0]

    @pytest.mark.asyncio
    async def test_since_filter_returns_only_new(self, tmp_path):
        """The poller passes back current_seq as ``since`` — we must
        return only payloads strictly newer than that."""
        mgr = BrowserManager(profiles_dir=str(tmp_path / "profiles"))
        inst = _new_instance("a1")
        mgr._instances = {"a1": inst}

        # Each drain needs activity — otherwise ``_is_empty_payload``
        # (correctly) skips the emit to keep idle history tidy.
        for _ in range(3):
            inst.m_click_success = 1
            await mgr._emit_metrics()

        snap = mgr.get_recent_metrics(since_seq=2)
        assert snap["current_seq"] == 3
        assert [m["seq"] for m in snap["metrics"]] == [3]

    @pytest.mark.asyncio
    async def test_since_beyond_current_returns_empty(self, tmp_path):
        mgr = BrowserManager(profiles_dir=str(tmp_path / "profiles"))
        inst = _new_instance("a1")
        mgr._instances = {"a1": inst}
        inst.m_click_success = 1
        await mgr._emit_metrics()
        snap = mgr.get_recent_metrics(since_seq=999)
        assert snap["metrics"] == []
        assert snap["current_seq"] == 1

    @pytest.mark.asyncio
    async def test_idle_agent_skipped_from_history(self, tmp_path):
        """Regression: agents with zero activity on a tick must not
        consume seqs or pollute the history buffer."""
        mgr = BrowserManager(profiles_dir=str(tmp_path / "profiles"))
        mgr._instances = {"idle": _new_instance("idle")}
        await mgr._emit_metrics()
        snap = mgr.get_recent_metrics(since_seq=0)
        assert snap["current_seq"] == 0
        assert snap["metrics"] == []

    @pytest.mark.asyncio
    async def test_post_click_idle_interval_still_filtered(self, tmp_path):
        """Regression (Codex #1 P1): the rolling click window persists
        across drains, so if ``_is_empty_payload`` treated a non-empty
        window as "activity" the filter would be permanently bypassed
        for any agent that ever clicked. This test would silently
        pass even with the bug if the agent had an active minute; we
        specifically exercise an idle minute *after* activity.
        """
        mgr = BrowserManager(profiles_dir=str(tmp_path / "profiles"))
        inst = _new_instance("chatty")
        mgr._instances = {"chatty": inst}
        # Minute 1: has a click.
        inst.m_click_success = 1
        inst.click_window.append(True)
        await mgr._emit_metrics()
        assert mgr.get_recent_metrics(since_seq=0)["current_seq"] == 1
        # Minute 2: no new clicks, but window is still non-empty from M1.
        await mgr._emit_metrics()
        # Seq must NOT advance — the idle minute is correctly filtered.
        assert mgr.get_recent_metrics(since_seq=0)["current_seq"] == 1

    @pytest.mark.asyncio
    async def test_history_records_final_stop_drain(self, tmp_path):
        """An agent stopping must still surface its last minute of data
        to the next poll, even without a metrics_sink wired in-process."""
        mgr = BrowserManager(profiles_dir=str(tmp_path / "profiles"))
        inst = _new_instance("goodbye")
        inst.context = MagicMock()
        inst.context.close = AsyncMock()
        mgr._instances["goodbye"] = inst
        inst.m_click_success = 9
        async with mgr._lock:
            await mgr._stop_instance("goodbye")

        snap = mgr.get_recent_metrics(since_seq=0)
        assert len(snap["metrics"]) == 1
        assert snap["metrics"][0]["agent_id"] == "goodbye"
        assert snap["metrics"][0]["click_success"] == 9


class TestBrowserMetricsEndpoint:
    """``GET /browser/metrics?since=<seq>`` is the mesh's polling target.

    These tests construct ``BrowserManager`` inside an async context so the
    ``asyncio.Lock()`` in its ``__init__`` binds cleanly on Python 3.9 — on
    3.10+ this is a no-op but it also avoids test-order leakage when
    ``asyncio.run`` closes the default event loop between test cases.
    """

    def _mk_app(self, monkeypatch, manager):
        monkeypatch.delenv("BROWSER_AUTH_TOKEN", raising=False)
        monkeypatch.delenv("MESH_AUTH_TOKEN", raising=False)
        from src.browser.server import create_browser_app
        return create_browser_app(manager)

    @pytest.mark.asyncio
    async def test_endpoint_returns_buffered_metrics(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient

        mgr = BrowserManager(profiles_dir=str(tmp_path / "profiles"))
        mgr._instances = {"a1": _new_instance("a1")}
        mgr._instances["a1"].m_click_success = 2
        await mgr._emit_metrics()

        app = self._mk_app(monkeypatch, mgr)
        with TestClient(app) as client:
            resp = client.get("/browser/metrics?since=0")
        assert resp.status_code == 200
        body = resp.json()
        assert body["current_seq"] == 1
        assert body["boot_id"] == mgr.boot_id
        assert len(body["metrics"]) == 1
        assert body["metrics"][0]["agent_id"] == "a1"
        assert body["metrics"][0]["click_success"] == 2

    @pytest.mark.asyncio
    async def test_endpoint_respects_since_filter(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient

        mgr = BrowserManager(profiles_dir=str(tmp_path / "profiles"))
        inst = _new_instance("a1")
        mgr._instances = {"a1": inst}
        # Activity on each tick so both payloads survive the
        # empty-payload filter.
        for _ in range(2):
            inst.m_click_success = 1
            await mgr._emit_metrics()

        app = self._mk_app(monkeypatch, mgr)
        with TestClient(app) as client:
            resp = client.get("/browser/metrics?since=1")
        assert resp.status_code == 200
        body = resp.json()
        assert [m["seq"] for m in body["metrics"]] == [2]

    @pytest.mark.asyncio
    async def test_endpoint_handles_invalid_since(self, tmp_path, monkeypatch):
        """A non-integer ``since`` must not crash the server; FastAPI
        returns 422 at the validation layer."""
        from fastapi.testclient import TestClient

        mgr = BrowserManager(profiles_dir=str(tmp_path / "profiles"))
        app = self._mk_app(monkeypatch, mgr)
        with TestClient(app) as client:
            resp = client.get("/browser/metrics?since=abc")
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_endpoint_requires_auth_when_configured(
        self, tmp_path, monkeypatch,
    ):
        """With BROWSER_AUTH_TOKEN set, metrics require a Bearer token —
        same posture as every other /browser/* endpoint."""
        from fastapi.testclient import TestClient

        monkeypatch.setenv("BROWSER_AUTH_TOKEN", "secret-t0k")
        monkeypatch.delenv("MESH_AUTH_TOKEN", raising=False)
        from src.browser.server import create_browser_app

        mgr = BrowserManager(profiles_dir=str(tmp_path / "profiles"))
        app = create_browser_app(mgr)
        with TestClient(app) as client:
            unauth = client.get("/browser/metrics")
            assert unauth.status_code == 401
            authed = client.get(
                "/browser/metrics",
                headers={"Authorization": "Bearer secret-t0k"},
            )
            assert authed.status_code == 200
