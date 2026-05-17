"""Tests for the health monitor, including ephemeral agent cleanup."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.host.health import AgentHealth, HealthMonitor


def _make_monitor(agents_info: dict | None = None):
    """Create a HealthMonitor with mocked dependencies."""
    runtime = MagicMock()
    runtime.agents = agents_info or {}
    transport = MagicMock()
    router = MagicMock()
    event_bus = MagicMock()
    monitor = HealthMonitor(
        runtime=runtime, transport=transport, router=router, event_bus=event_bus,
    )
    return monitor


class TestEphemeralCleanup:
    @pytest.mark.asyncio
    async def test_cleanup_ephemeral_expired(self):
        """Agent past TTL is stopped and unregistered."""
        monitor = _make_monitor({
            "spawn-abc": {"ephemeral": True, "ttl": 60, "spawned_at": time.time() - 120},
        })
        monitor.agents["spawn-abc"] = AgentHealth(agent_id="spawn-abc")
        await monitor._cleanup_ephemeral_agents()
        monitor.runtime.stop_agent.assert_called_once_with("spawn-abc")
        monitor.router.unregister_agent.assert_called_once_with("spawn-abc")
        assert "spawn-abc" not in monitor.agents

    @pytest.mark.asyncio
    async def test_cleanup_ephemeral_not_expired(self):
        """Agent within TTL is kept."""
        monitor = _make_monitor({
            "spawn-def": {"ephemeral": True, "ttl": 3600, "spawned_at": time.time()},
        })
        monitor.agents["spawn-def"] = AgentHealth(agent_id="spawn-def")
        await monitor._cleanup_ephemeral_agents()
        monitor.runtime.stop_agent.assert_not_called()
        assert "spawn-def" in monitor.agents

    @pytest.mark.asyncio
    async def test_cleanup_non_ephemeral_untouched(self):
        """Regular (non-ephemeral) agents are not affected by cleanup."""
        monitor = _make_monitor({
            "regular": {"role": "assistant"},
        })
        monitor.agents["regular"] = AgentHealth(agent_id="regular")
        await monitor._cleanup_ephemeral_agents()
        monitor.runtime.stop_agent.assert_not_called()
        assert "regular" in monitor.agents

    @pytest.mark.asyncio
    async def test_cleanup_emits_event(self):
        """Removing an expired ephemeral agent emits an agent_state event."""
        monitor = _make_monitor({
            "spawn-xyz": {"ephemeral": True, "ttl": 10, "spawned_at": time.time() - 100},
        })
        monitor.agents["spawn-xyz"] = AgentHealth(agent_id="spawn-xyz")
        await monitor._cleanup_ephemeral_agents()
        monitor._event_bus.emit.assert_called_once_with(
            "agent_state", agent="spawn-xyz",
            data={"state": "removed", "reason": "ttl_expired"},
        )


class TestHealthRestartMissingConfig:
    @pytest.mark.asyncio
    async def test_restart_skipped_when_no_config(self):
        """Restart with missing agent metadata sets status to 'failed' and does NOT call start_agent."""
        monitor = _make_monitor({})  # empty agents dict — no config
        monitor.register("ghost-agent")
        health = monitor.agents["ghost-agent"]
        health.consecutive_failures = 3
        health.status = "unhealthy"

        await monitor._try_restart("ghost-agent")

        assert health.status == "failed"
        monitor.runtime.start_agent.assert_not_called()

    @pytest.mark.asyncio
    async def test_restart_succeeds_with_config(self):
        """Restart with valid agent metadata proceeds normally."""
        monitor = _make_monitor({
            "good-agent": {"role": "coder", "skills_dir": "/skills"},
        })
        monitor.register("good-agent")
        health = monitor.agents["good-agent"]
        health.consecutive_failures = 3
        health.status = "unhealthy"
        monitor.runtime.start_agent.return_value = "http://localhost:8401"
        monitor.runtime.wait_for_agent = AsyncMock(return_value=True)

        await monitor._try_restart("good-agent")

        monitor.runtime.start_agent.assert_called_once()
        assert health.status == "healthy"
        assert health.restart_count == 1


class TestParallelHealthChecks:
    @pytest.mark.asyncio
    async def test_check_all_runs_concurrently(self):
        """_check_all dispatches health checks concurrently via asyncio.gather."""
        import asyncio

        call_times = []

        async def slow_reachable(agent_id, timeout=5):
            call_times.append(time.monotonic())
            await asyncio.sleep(0.1)
            return True

        monitor = _make_monitor({
            "a": {"role": "assistant"},
            "b": {"role": "assistant"},
            "c": {"role": "assistant"},
        })
        monitor.register("a")
        monitor.register("b")
        monitor.register("c")
        monitor.transport.is_reachable = slow_reachable

        t0 = time.monotonic()
        await monitor._check_all()
        elapsed = time.monotonic() - t0

        assert len(call_times) == 3
        # Parallel: total ~0.1s, not ~0.3s
        assert elapsed < 0.25, f"Expected parallel but took {elapsed:.2f}s"
        # All calls started at roughly the same time
        assert max(call_times) - min(call_times) < 0.05


class TestHealthRecoveryEvent:
    @pytest.mark.asyncio
    async def test_recovery_emits_health_change(self):
        """When an unhealthy agent becomes reachable, a health_change event fires."""
        monitor = _make_monitor({"agent-a": {"role": "assistant"}})
        monitor.register("agent-a")
        # Simulate prior unhealthy state
        monitor.agents["agent-a"].status = "unhealthy"
        monitor.agents["agent-a"].consecutive_failures = 2
        monitor.transport.is_reachable = AsyncMock(return_value=True)
        await monitor._check_agent("agent-a")
        assert monitor.agents["agent-a"].status == "healthy"
        monitor._event_bus.emit.assert_called_once_with(
            "health_change", agent="agent-a",
            data={
                "previous": "unhealthy", "current": "healthy",
                "failures": 0, "restart_count": 0,
            },
        )


# ── Seam follow-up Fix 4: quarantine on consecutive auth failures ──


class TestQuarantine:
    """Tests for HealthMonitor.record_auth_failure / clear_quarantine
    / is_quarantined / auto-expiry — the credential-failure quarantine path.

    The mesh quarantines an agent after AUTH_FAILURE_THRESHOLD consecutive
    auth failures so the lane stops dispatching work that will obviously
    fail. Clear is implicit on edit_agent(model) or after the auto-expiry
    TTL — no separate operator tool needed (operator UX principle).
    """

    def test_record_auth_failure_increments_counter(self):
        monitor = _make_monitor({})
        monitor.register("agent-a")
        quarantined = monitor.record_auth_failure(
            "agent-a", provider="openai", model="openai/gpt-5", http_status=401,
        )
        assert quarantined is False
        assert monitor.agents["agent-a"].consecutive_auth_failures == 1
        assert monitor.agents["agent-a"].quarantined is False

    def test_quarantine_triggers_at_threshold(self):
        monitor = _make_monitor({})
        monitor.register("agent-a")
        # Default threshold is 3 — first two stay below.
        monitor.record_auth_failure("agent-a", provider="openai", model="x", http_status=401)
        monitor.record_auth_failure("agent-a", provider="openai", model="x", http_status=401)
        assert monitor.agents["agent-a"].quarantined is False
        just_quarantined = monitor.record_auth_failure(
            "agent-a", provider="openai", model="x", http_status=401,
        )
        assert just_quarantined is True
        assert monitor.agents["agent-a"].quarantined is True
        assert monitor.agents["agent-a"].status == "quarantined"
        assert monitor.agents["agent-a"].quarantine_reason is not None
        assert "openai" in monitor.agents["agent-a"].quarantine_reason

    def test_quarantine_emits_event(self):
        monitor = _make_monitor({})
        monitor.register("agent-a")
        for _ in range(3):
            monitor.record_auth_failure(
                "agent-a", provider="openai", model="x", http_status=401,
            )
        # The event bus is called with the quarantine transition.
        emit_calls = monitor._event_bus.emit.call_args_list
        # Find the health_change → quarantined emit
        quarantine_emit = [
            c for c in emit_calls
            if c.args and c.args[0] == "health_change"
            and c.kwargs.get("data", {}).get("current") == "quarantined"
        ]
        assert len(quarantine_emit) == 1

    def test_quarantine_writes_notification(self):
        monitor = _make_monitor({})
        notif_store = MagicMock()
        monitor.set_notifications_store(notif_store)
        monitor.register("agent-a")
        for _ in range(3):
            monitor.record_auth_failure(
                "agent-a", provider="openai", model="openai/gpt-5", http_status=401,
            )
        notif_store.add.assert_called_once()
        kwargs = notif_store.add.call_args.kwargs
        assert kwargs["kind"] == "credential"
        assert kwargs["agent_id"] == "agent-a"

    def test_is_quarantined_query(self):
        monitor = _make_monitor({})
        monitor.register("agent-a")
        monitor.register("agent-b")
        for _ in range(3):
            monitor.record_auth_failure(
                "agent-a", provider="openai", model="x", http_status=401,
            )
        assert monitor.is_quarantined("agent-a") is True
        assert monitor.is_quarantined("agent-b") is False
        assert monitor.is_quarantined("unknown") is False

    def test_clear_quarantine_resets_state(self):
        monitor = _make_monitor({})
        monitor.register("agent-a")
        for _ in range(3):
            monitor.record_auth_failure(
                "agent-a", provider="openai", model="x", http_status=401,
            )
        cleared = monitor.clear_quarantine("agent-a", reason="model changed")
        assert cleared is True
        assert monitor.agents["agent-a"].quarantined is False
        assert monitor.agents["agent-a"].quarantine_reason is None
        assert monitor.agents["agent-a"].consecutive_auth_failures == 0
        assert monitor.agents["agent-a"].status == "healthy"
        # Emits a clear event.
        emit_calls = monitor._event_bus.emit.call_args_list
        clear_emits = [
            c for c in emit_calls
            if c.args and c.args[0] == "health_change"
            and c.kwargs.get("data", {}).get("current") == "healthy"
            and c.kwargs.get("data", {}).get("previous") == "quarantined"
        ]
        assert len(clear_emits) == 1

    def test_clear_quarantine_noop_when_not_quarantined(self):
        monitor = _make_monitor({})
        monitor.register("agent-a")
        assert monitor.clear_quarantine("agent-a", reason="test") is False

    def test_auto_expiry_clears_old_quarantine(self):
        monitor = _make_monitor({})
        monitor.register("agent-a")
        for _ in range(3):
            monitor.record_auth_failure(
                "agent-a", provider="openai", model="x", http_status=401,
            )
        # Backdate the quarantine timestamp past the TTL.
        monitor.agents["agent-a"].quarantined_at = time.time() - (
            monitor.QUARANTINE_AUTO_CLEAR_SECONDS + 60
        )
        monitor._maybe_expire_quarantines(time.time())
        assert monitor.agents["agent-a"].quarantined is False

    def test_get_status_surfaces_quarantine_fields(self):
        monitor = _make_monitor({})
        monitor.register("agent-a")
        for _ in range(3):
            monitor.record_auth_failure(
                "agent-a", provider="openai", model="x", http_status=401,
            )
        statuses = monitor.get_status()
        agent_status = next(s for s in statuses if s["agent"] == "agent-a")
        assert agent_status["quarantined"] is True
        assert agent_status["quarantine_reason"] is not None
        assert agent_status["consecutive_auth_failures"] == 3

    @pytest.mark.asyncio
    async def test_reachability_poll_preserves_quarantined_status(self):
        """Codex P2 follow-up: a successful reachability poll must NOT
        flip a quarantined agent back to healthy — the agent is
        reachable but its credentials are broken, which is what lane/cron
        are skipping on. Only clear_quarantine should flip status."""
        monitor = _make_monitor({"agent-a": {"role": "x"}})
        monitor.register("agent-a")
        # Quarantine the agent.
        for _ in range(3):
            monitor.record_auth_failure(
                "agent-a", provider="openai", model="x", http_status=401,
            )
        assert monitor.agents["agent-a"].status == "quarantined"
        # Successful reachability poll.
        monitor.transport.is_reachable = AsyncMock(return_value=True)
        await monitor._check_agent("agent-a")
        # Status must remain quarantined.
        assert monitor.agents["agent-a"].status == "quarantined"
        assert monitor.agents["agent-a"].quarantined is True

    @pytest.mark.asyncio
    async def test_unreachable_poll_preserves_quarantined_status(self):
        """Same for unreachable polls — the reachability counter ticks
        but the status string stays quarantined."""
        monitor = _make_monitor({"agent-a": {"role": "x"}})
        monitor.register("agent-a")
        for _ in range(3):
            monitor.record_auth_failure(
                "agent-a", provider="openai", model="x", http_status=401,
            )
        assert monitor.agents["agent-a"].status == "quarantined"
        monitor.transport.is_reachable = AsyncMock(return_value=False)
        await monitor._check_agent("agent-a")
        # Reachability counter still ticks.
        assert monitor.agents["agent-a"].consecutive_failures >= 1
        # Status string stays quarantined.
        assert monitor.agents["agent-a"].status == "quarantined"
