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
