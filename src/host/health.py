"""Health monitor with auto-restart for agent runtimes.

Runs as a background task in the mesh host. Polls each registered agent's
/status endpoint every POLL_INTERVAL seconds via the Transport layer (works
for both Docker containers and Sandbox microVMs). If an agent fails
MAX_FAILURES consecutive checks, it is restarted via the RuntimeBackend.

Restart policy: max RESTART_LIMIT restarts per RESTART_WINDOW seconds.
After the limit is hit, the agent is marked as 'failed' and left stopped.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.shared.utils import setup_logging

if TYPE_CHECKING:
    from src.host.mesh import MessageRouter
    from src.host.runtime import RuntimeBackend
    from src.host.transport import Transport

logger = setup_logging("host.health")


@dataclass
class AgentHealth:
    agent_id: str
    consecutive_failures: int = 0
    restart_count: int = 0
    restart_timestamps: list[float] = field(default_factory=list)
    last_check: float = 0.0
    last_healthy: float = 0.0
    status: str = "unknown"


class HealthMonitor:
    """Polls agents and restarts unhealthy ones."""

    POLL_INTERVAL = 30
    MAX_FAILURES = 3
    RESTART_LIMIT = 3
    RESTART_WINDOW = 3600

    def __init__(
        self,
        runtime: RuntimeBackend,
        transport: Transport,
        router: MessageRouter,
        event_bus=None,
    ):
        self.runtime = runtime
        self.transport = transport
        self.router = router
        self._event_bus = event_bus
        self.agents: dict[str, AgentHealth] = {}
        self._running = False

    def register(self, agent_id: str, url: str = "") -> None:
        self.agents[agent_id] = AgentHealth(agent_id=agent_id)

    async def start(self) -> None:
        self._running = True
        logger.info(f"Health monitor started for {len(self.agents)} agents")
        while self._running:
            await self._check_all()
            await asyncio.sleep(self.POLL_INTERVAL)

    def stop(self) -> None:
        self._running = False

    async def _check_all(self) -> None:
        for agent_id in list(self.agents.keys()):
            await self._check_agent(agent_id)

    async def _check_agent(self, agent_id: str) -> None:
        health = self.agents.get(agent_id)
        if not health or health.status == "failed":
            return

        prev_status = health.status
        now = time.time()
        health.last_check = now

        try:
            reachable = await self.transport.is_reachable(agent_id, timeout=5)
            if reachable:
                health.consecutive_failures = 0
                health.last_healthy = now
                health.status = "healthy"
                if health.status != prev_status and self._event_bus:
                    self._event_bus.emit("health_change", agent=agent_id, data={
                        "previous": prev_status, "current": health.status,
                        "failures": health.consecutive_failures,
                        "restart_count": health.restart_count,
                    })
                return
        except Exception as e:
            logger.debug("Health check transport error for '%s': %s", agent_id, e)

        health.consecutive_failures += 1
        health.status = "unhealthy"
        if health.status != prev_status and self._event_bus:
            self._event_bus.emit("health_change", agent=agent_id, data={
                "previous": prev_status, "current": health.status,
                "failures": health.consecutive_failures,
                "restart_count": health.restart_count,
            })
        logger.debug(
            f"Agent '{agent_id}' health check failed "
            f"({health.consecutive_failures}/{self.MAX_FAILURES})"
        )

        if health.consecutive_failures >= self.MAX_FAILURES:
            logger.warning(f"Agent '{agent_id}' unreachable, restarting...")
            await self._try_restart(agent_id)

    async def _try_restart(self, agent_id: str) -> None:
        health = self.agents[agent_id]
        now = time.time()

        health.restart_timestamps = [
            t for t in health.restart_timestamps
            if now - t < self.RESTART_WINDOW
        ]

        if len(health.restart_timestamps) >= self.RESTART_LIMIT:
            health.status = "failed"
            if self._event_bus:
                self._event_bus.emit("health_change", agent=agent_id, data={
                    "previous": "unhealthy", "current": "failed",
                    "failures": health.consecutive_failures,
                })
            logger.error(
                f"Agent '{agent_id}' exceeded restart limit "
                f"({self.RESTART_LIMIT} in {self.RESTART_WINDOW}s). Marking as failed."
            )
            return

        logger.info(f"Restarting agent '{agent_id}'...")
        info = self.runtime.agents.get(agent_id, {})
        try:
            self.runtime.stop_agent(agent_id)
        except Exception as e:
            logger.warning(f"Error stopping agent '{agent_id}' during restart: {e}")

        try:
            url = self.runtime.start_agent(
                agent_id=agent_id,
                role=info.get("role", ""),
                skills_dir=info.get("skills_dir", ""),
                system_prompt=info.get("system_prompt", ""),
                model=info.get("model", ""),
                mcp_servers=info.get("mcp_servers"),
            )

            self.router.register_agent(agent_id, url)
            health.consecutive_failures = 0
            health.restart_count += 1
            health.restart_timestamps.append(now)
            health.status = "restarting"
            if self._event_bus:
                self._event_bus.emit("health_change", agent=agent_id, data={
                    "previous": "unhealthy", "current": "restarting",
                    "failures": 0, "restart_count": health.restart_count,
                })

            ready = await self.runtime.wait_for_agent(agent_id, timeout=60)
            prev = health.status
            health.status = "healthy" if ready else "unhealthy"
            if self._event_bus:
                self._event_bus.emit("health_change", agent=agent_id, data={
                    "previous": prev, "current": health.status,
                    "failures": health.consecutive_failures,
                    "restart_count": health.restart_count,
                })

            logger.info(
                f"Agent '{agent_id}' restarted "
                f"(attempt {health.restart_count}, {'ready' if ready else 'not ready'})"
            )
        except Exception as e:
            health.status = "unhealthy"
            if self._event_bus:
                self._event_bus.emit("health_change", agent=agent_id, data={
                    "previous": "restarting", "current": "unhealthy",
                    "failures": health.consecutive_failures,
                    "error": str(e),
                })
            logger.error(f"Failed to restart agent '{agent_id}': {e}")

    def get_status(self) -> list[dict]:
        """Return health status for all monitored agents."""
        return [
            {
                "agent": h.agent_id,
                "status": h.status,
                "failures": h.consecutive_failures,
                "restarts": h.restart_count,
                "last_check": h.last_check,
                "last_healthy": h.last_healthy,
            }
            for h in self.agents.values()
        ]
