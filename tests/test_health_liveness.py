"""Tests for the loop-liveness signal (Bug 1).

The health monitor previously only checked HTTP 200 on /status — a dead
inner loop with a live FastAPI thread would report healthy. The fix
plumbs ``last_iteration_ts`` + ``iterations_since_boot`` through
``AgentStatus`` and adds a staleness branch in ``HealthMonitor`` that
combines /status with the lane queue depth.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.host.health import HealthMonitor


def _make_monitor(agents_info: dict | None = None, *, queue_depth_fn=None):
    runtime = MagicMock()
    runtime.agents = agents_info or {}
    transport = MagicMock()
    router = MagicMock()
    event_bus = MagicMock()
    monitor = HealthMonitor(
        runtime=runtime, transport=transport, router=router,
        event_bus=event_bus, queue_depth_fn=queue_depth_fn,
    )
    return monitor


class TestLivenessFields:
    """`AgentLoop._bump_liveness` must update the two attrs that get
    surfaced on /status."""

    def test_bump_liveness_updates_fields(self):
        from src.agent.loop import AgentLoop

        # Bypass __init__ — we only need the two attrs and the helper.
        loop = AgentLoop.__new__(AgentLoop)
        loop._last_iteration_ts = None
        loop._iterations_since_boot = 0

        before = time.time()
        loop._bump_liveness()
        after = time.time()

        assert loop._iterations_since_boot == 1
        assert loop._last_iteration_ts is not None
        assert before <= loop._last_iteration_ts <= after

        loop._bump_liveness()
        assert loop._iterations_since_boot == 2

    def test_agent_status_carries_liveness_fields(self):
        """``AgentStatus`` must serialise the new optional fields."""
        from src.shared.types import AgentStatus

        ts = time.time()
        s = AgentStatus(
            agent_id="x", role="r", state="idle",
            last_iteration_ts=ts,
            iterations_since_boot=7,
        )
        dumped = s.model_dump()
        assert dumped["last_iteration_ts"] == ts
        assert dumped["iterations_since_boot"] == 7

    def test_agent_status_defaults_safe(self):
        """Old payloads without the fields must still deserialise."""
        from src.shared.types import AgentStatus

        s = AgentStatus.model_validate({
            "agent_id": "x", "role": "r", "state": "idle",
        })
        assert s.last_iteration_ts is None
        assert s.iterations_since_boot == 0


class TestHealthLivenessCheck:
    """The staleness rule: reachable + queue_depth>0 + stale iteration =>
    unhealthy. Reachable + idle queue => healthy regardless of staleness."""

    @pytest.mark.asyncio
    async def test_reachable_busy_stale_marks_unhealthy(self):
        """Wedged inner loop with queued work is treated as unhealthy."""
        depth = {"agent-a": 3}
        monitor = _make_monitor(queue_depth_fn=lambda a: depth.get(a, 0))
        monitor.register("agent-a")
        monitor.transport.is_reachable = AsyncMock(return_value=True)
        # /status reports last_iteration_ts way in the past.
        ancient = time.time() - 9999
        monitor.transport.request = AsyncMock(return_value={
            "agent_id": "agent-a", "state": "working",
            "last_iteration_ts": ancient,
            "iterations_since_boot": 12,
        })

        await monitor._check_agent("agent-a")
        # First strike — not yet restarted but marked unhealthy.
        h = monitor.agents["agent-a"]
        assert h.status == "unhealthy", (
            f"expected unhealthy, got {h.status}"
        )
        assert h.consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_reachable_idle_stale_left_alone(self):
        """No work in the queue → idle agent isn't expected to tick.
        Must stay healthy even with no recent iteration."""
        monitor = _make_monitor(queue_depth_fn=lambda a: 0)
        monitor.register("agent-b")
        monitor.transport.is_reachable = AsyncMock(return_value=True)
        monitor.transport.request = AsyncMock(return_value={
            "agent_id": "agent-b", "state": "idle",
            "last_iteration_ts": time.time() - 9999,
            "iterations_since_boot": 0,
        })

        await monitor._check_agent("agent-b")
        h = monitor.agents["agent-b"]
        assert h.status == "healthy"
        assert h.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_reachable_busy_fresh_iteration_stays_healthy(self):
        """Queue has work but the loop ticked recently — not stale."""
        monitor = _make_monitor(queue_depth_fn=lambda a: 2)
        monitor.register("agent-c")
        monitor.transport.is_reachable = AsyncMock(return_value=True)
        monitor.transport.request = AsyncMock(return_value={
            "agent_id": "agent-c", "state": "working",
            "last_iteration_ts": time.time() - 1,
            "iterations_since_boot": 5,
        })

        await monitor._check_agent("agent-c")
        h = monitor.agents["agent-c"]
        assert h.status == "healthy"
        assert h.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_no_queue_depth_fn_disables_staleness_check(self):
        """Default wiring (no callback) must not regress reachability behaviour."""
        monitor = _make_monitor()  # no queue_depth_fn
        monitor.register("agent-d")
        monitor.transport.is_reachable = AsyncMock(return_value=True)
        # Even an obviously stale /status is ignored.
        monitor.transport.request = AsyncMock(return_value={
            "agent_id": "agent-d", "last_iteration_ts": 0,
        })

        await monitor._check_agent("agent-d")
        h = monitor.agents["agent-d"]
        assert h.status == "healthy"

    @pytest.mark.asyncio
    async def test_set_queue_depth_fn_after_construction(self):
        """``set_queue_depth_fn`` enables the check post-hoc."""
        monitor = _make_monitor()
        monitor.set_queue_depth_fn(lambda a: 1)
        monitor.register("agent-e")
        monitor.transport.is_reachable = AsyncMock(return_value=True)
        # Stale by 1200s (> default 900s threshold post codex P2 r2).
        monitor.transport.request = AsyncMock(return_value={
            "agent_id": "agent-e", "last_iteration_ts": time.time() - 1200,
        })

        await monitor._check_agent("agent-e")
        h = monitor.agents["agent-e"]
        assert h.status == "unhealthy"

    @pytest.mark.asyncio
    async def test_status_fetch_failure_falls_back_to_healthy(self):
        """A failed /status fetch must not flip an otherwise reachable
        agent to unhealthy — best-effort branch."""
        monitor = _make_monitor(queue_depth_fn=lambda a: 5)
        monitor.register("agent-f")
        monitor.transport.is_reachable = AsyncMock(return_value=True)
        monitor.transport.request = AsyncMock(
            side_effect=RuntimeError("status fetch died"),
        )

        await monitor._check_agent("agent-f")
        h = monitor.agents["agent-f"]
        assert h.status == "healthy"

    @pytest.mark.asyncio
    async def test_missing_last_iteration_ts_treated_as_not_stale(self):
        """A freshly booted agent that hasn't ticked yet must not be
        flagged stale."""
        monitor = _make_monitor(queue_depth_fn=lambda a: 1)
        monitor.register("agent-g")
        monitor.transport.is_reachable = AsyncMock(return_value=True)
        monitor.transport.request = AsyncMock(return_value={
            "agent_id": "agent-g", "last_iteration_ts": None,
        })

        await monitor._check_agent("agent-g")
        h = monitor.agents["agent-g"]
        assert h.status == "healthy"

    @pytest.mark.asyncio
    async def test_queue_depth_fn_raises_handled_gracefully(self):
        """If the lane callback raises, fall through to healthy."""
        def boom(_):
            raise RuntimeError("lane manager exploded")

        monitor = _make_monitor(queue_depth_fn=boom)
        monitor.register("agent-h")
        monitor.transport.is_reachable = AsyncMock(return_value=True)
        monitor.transport.request = AsyncMock(return_value={
            "agent_id": "agent-h", "last_iteration_ts": 0,
        })

        # Pre-condition: status is "unknown" (just registered)
        assert monitor.agents["agent-h"].status == "unknown"
        await monitor._check_agent("agent-h")
        h = monitor.agents["agent-h"]
        assert h.status == "healthy"


class TestLaneQueueDepthHelper:
    """``LaneManager.get_queue_depth`` underpins the staleness branch."""

    def test_unknown_agent_returns_zero(self):
        from src.host.lanes import LaneManager

        async def _dispatch(*args, **kw):
            return ""

        lm = LaneManager(dispatch_fn=_dispatch)
        assert lm.get_queue_depth("never-seen") == 0

    @pytest.mark.asyncio
    async def test_busy_lane_adds_one(self):
        """Busy flag counts as 1 even if the queue is empty."""
        from src.host.lanes import LaneManager

        async def _dispatch(*args, **kw):
            return ""

        lm = LaneManager(dispatch_fn=_dispatch)
        lm._ensure_lane("agent")
        # No queued tasks, but mark busy.
        lm._busy["agent"] = True
        assert lm.get_queue_depth("agent") == 1
        lm._busy["agent"] = False
        assert lm.get_queue_depth("agent") == 0
