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
