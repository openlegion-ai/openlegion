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
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.cli.config import _load_config
from src.cli.proxy import build_proxy_env_vars, resolve_agent_proxy
from src.shared.utils import set_llm_max_tokens_env, setup_logging

if TYPE_CHECKING:
    from src.host.mesh import MessageRouter
    from src.host.runtime import RuntimeBackend
    from src.host.transport import Transport

logger = setup_logging("host.health")

# Liveness staleness threshold (Bug 1) — a reachable agent with a non-empty
# lane queue but no loop iteration in this window is treated as having a
# dead inner loop. Aligned with the per-task lane watchdog default (900s)
# so the lane watchdog gets to fail a single stuck task before the health
# monitor escalates to an agent restart. Codex P2 r2: 300s was too
# aggressive — the loop bumps at iteration boundaries, post-LLM, and
# post-tool, so a single deep reasoning call or a full 300s tool timeout
# can exceed the old threshold even on a perfectly healthy turn. With the
# 900s default, the staleness check only fires when truly nothing has
# happened in the same window that the lane watchdog uses to cancel the
# task itself.
_STALE_LOOP_SECONDS = int(os.environ.get("OPENLEGION_LOOP_STALE_SECONDS", "900"))


@dataclass
class AgentHealth:
    agent_id: str
    consecutive_failures: int = 0
    restart_count: int = 0
    restart_timestamps: list[float] = field(default_factory=list)
    last_check: float = 0.0
    last_healthy: float = 0.0
    status: str = "unknown"
    # Quarantine state (Fix 4 in seam follow-up). Separate from the
    # reachability-failure counter — an agent with a working /status but
    # a broken LLM credential is healthy from the runtime's POV yet must
    # not consume more work. Cleared by edit_agent on model change, or
    # automatically after ``QUARANTINE_AUTO_CLEAR_SECONDS``.
    consecutive_auth_failures: int = 0
    quarantined: bool = False
    quarantine_reason: str | None = None
    quarantined_at: float | None = None


class HealthMonitor:
    """Polls agents and restarts unhealthy ones."""

    POLL_INTERVAL = 30
    MAX_FAILURES = 3
    RESTART_LIMIT = 3
    RESTART_WINDOW = 3600
    # Quarantine thresholds (Fix 4 in seam follow-up).
    AUTH_FAILURE_THRESHOLD = 3
    QUARANTINE_AUTO_CLEAR_SECONDS = int(
        os.environ.get("OPENLEGION_QUARANTINE_AUTO_CLEAR_SECONDS", "1800"),
    )

    def __init__(
        self,
        runtime: RuntimeBackend,
        transport: Transport,
        router: MessageRouter,
        event_bus=None,
        cleanup_agent_fn=None,
        blackboard=None,
        queue_depth_fn: Callable[[str], int] | None = None,
        notifications_store=None,
    ):
        self.runtime = runtime
        self.transport = transport
        self.router = router
        self._event_bus = event_bus
        self._cleanup_agent = cleanup_agent_fn
        self._blackboard = blackboard
        # ``queue_depth_fn`` (Bug 1) — optional callable that returns the
        # lane queue depth for an agent_id. When wired, the health monitor
        # combines /status liveness (last_iteration_ts) with queue depth to
        # detect a dead inner loop with a live FastAPI thread. Without it
        # the staleness check is a no-op (reachability behaviour unchanged).
        self._queue_depth_fn = queue_depth_fn
        # ``notifications_store`` (Fix 4) — optional dashboard notifications
        # sink. When wired, quarantine transitions emit a ``credential`` kind
        # notification so the bell + Work tab surface it to the operator.
        self._notifications_store = notifications_store
        self.agents: dict[str, AgentHealth] = {}
        self._agent_lock: asyncio.Lock | None = None
        self._agent_lock_loop: asyncio.AbstractEventLoop | None = None
        self._running = False

    def _get_agent_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._agent_lock is None or self._agent_lock_loop is not loop:
            self._agent_lock = asyncio.Lock()
            self._agent_lock_loop = loop
        return self._agent_lock

    def register(self, agent_id: str, url: str = "") -> None:
        self.agents[agent_id] = AgentHealth(agent_id=agent_id)

    def unregister(self, agent_id: str) -> None:
        self.agents.pop(agent_id, None)

    def set_queue_depth_fn(self, fn: Callable[[str], int] | None) -> None:
        """Wire (or unwire) the lane queue-depth callback after construction.

        ``HealthMonitor`` is built before ``LaneManager`` in the host
        bootstrap path; this setter lets the lane wire itself in once
        constructed. ``None`` disables the staleness branch.
        """
        self._queue_depth_fn = fn

    def set_notifications_store(self, store) -> None:
        """Wire (or unwire) the dashboard notifications sink after construction.

        Mirrors the ``set_queue_depth_fn`` injection pattern (PR #918) —
        notifications store can be constructed after ``HealthMonitor``
        in the bootstrap path. ``None`` disables quarantine notifications.
        """
        self._notifications_store = store

    # ── Quarantine API (Fix 4 in seam follow-up) ──────────────

    def record_auth_failure(
        self, agent_id: str, *, provider: str, model: str, http_status: int,
    ) -> bool:
        """Increment the per-agent auth-failure counter.

        Returns ``True`` if this call transitioned the agent into
        quarantine. The counter is cleared by ``clear_quarantine`` or by
        the auto-expiry sweeper after ``QUARANTINE_AUTO_CLEAR_SECONDS``.
        """
        h = self.agents.get(agent_id)
        if not h:
            return False
        h.consecutive_auth_failures += 1
        just_quarantined = False
        if (
            h.consecutive_auth_failures >= self.AUTH_FAILURE_THRESHOLD
            and not h.quarantined
        ):
            h.quarantined = True
            h.quarantine_reason = (
                f"{provider}/{model} auth failing (status={http_status}, "
                f"{h.consecutive_auth_failures} consecutive failures)"
            )
            h.quarantined_at = time.time()
            h.status = "quarantined"
            just_quarantined = True
            logger.warning(
                "Agent '%s' quarantined: %s — fix credential or change "
                "model via edit_agent",
                agent_id, h.quarantine_reason,
            )
            if self._notifications_store is not None:
                try:
                    self._notifications_store.add(
                        kind="credential",
                        title=f"Agent '{agent_id}' quarantined: credential broken",
                        body=(
                            (h.quarantine_reason or "") + ". Lane has stopped "
                            "dispatching new work. Rotate the credential or "
                            "run edit_agent to pick a compatible model."
                        ),
                        agent_id=agent_id,
                    )
                except Exception as notif_err:
                    logger.warning(
                        "Failed to write quarantine notification: %s",
                        notif_err,
                    )
            if self._event_bus:
                try:
                    self._event_bus.emit("health_change", agent=agent_id, data={
                        "previous": "healthy", "current": "quarantined",
                        "reason": h.quarantine_reason,
                    })
                except Exception as ev_err:
                    logger.debug("health_change emit failed: %s", ev_err)
        return just_quarantined

    def clear_quarantine(self, agent_id: str, *, reason: str) -> bool:
        """Lift quarantine.

        Called by ``edit_agent`` (on a successful ``model`` change) and by
        the auto-expiry sweeper. Returns ``True`` when the agent was
        actually quarantined and got cleared, ``False`` when it was a no-op
        with respect to the quarantine flag itself.

        Codex P2 r3: always reset the pre-threshold auth-failure counter
        too — when the credential or model changes, any partial-failure
        history is no longer relevant. Without this reset, an agent that
        accumulated 2 auth failures (below threshold), then had its model
        switched, would quarantine on the very first failure on the new
        model. The boolean return value still reflects the quarantine
        flag transition, NOT the counter reset.
        """
        h = self.agents.get(agent_id)
        if not h:
            return False
        # Reset the pre-threshold counter regardless of quarantine state —
        # any pending failures are no longer relevant after a credential
        # or model change.
        if h.consecutive_auth_failures > 0:
            logger.debug(
                "Resetting auth-failure counter for '%s' (was %d): %s",
                agent_id, h.consecutive_auth_failures, reason,
            )
            h.consecutive_auth_failures = 0
        if not h.quarantined:
            return False  # counter already reset above; nothing more to do
        prev_status = h.status
        h.quarantined = False
        h.quarantine_reason = None
        h.quarantined_at = None
        # Codex r4 (principal-eng): don't clobber a more-severe runtime
        # state. ``failed`` is terminal (MAX_FAILURES restarts exhausted
        # — see :440/:471) and must never be revived to ``healthy`` by a
        # credential-side clear. ``restarting`` is an in-flight state
        # (set in :534) the restart path will reconcile on its own.
        # ``unhealthy`` means reachability is genuinely broken — the
        # next ``_check_agent`` poll will flip it back to ``healthy`` if
        # the agent actually responds. Only flip to ``healthy`` when no
        # other negative signal is in flight.
        if h.status in ("failed", "restarting"):
            pass  # preserve — terminal or in-flight runtime state
        elif h.consecutive_failures > 0:
            h.status = "unhealthy"
        else:
            h.status = "healthy"
        logger.info("Agent '%s' quarantine cleared: %s", agent_id, reason)
        if self._event_bus:
            try:
                self._event_bus.emit("health_change", agent=agent_id, data={
                    "previous": prev_status, "current": h.status,
                    "reason": reason,
                })
            except Exception as ev_err:
                logger.debug("health_change emit failed: %s", ev_err)
        return True

    def is_quarantined(self, agent_id: str) -> bool:
        h = self.agents.get(agent_id)
        return bool(h and h.quarantined)

    def _maybe_expire_quarantines(self, now: float) -> None:
        """Auto-clear quarantines older than the TTL."""
        for agent_id, h in list(self.agents.items()):
            if (
                h.quarantined and h.quarantined_at
                and (now - h.quarantined_at) > self.QUARANTINE_AUTO_CLEAR_SECONDS
            ):
                self.clear_quarantine(agent_id, reason="auto-expiry after timeout")

    async def start(self) -> None:
        self._running = True
        logger.info(f"Health monitor started for {len(self.agents)} agents")
        while self._running:
            # Auto-expire quarantines BEFORE the reachability check so a
            # newly-cleared agent's status doesn't get clobbered.
            self._maybe_expire_quarantines(time.time())
            await self._check_all()
            await asyncio.sleep(self.POLL_INTERVAL)

    def stop(self) -> None:
        self._running = False

    async def _check_all(self) -> None:
        await self._cleanup_ephemeral_agents()
        if self._blackboard:
            try:
                self._blackboard.gc_expired()
            except Exception as e:
                logger.debug("Blackboard TTL cleanup failed: %s", e)
        agent_ids = list(self.agents.keys())
        if agent_ids:
            await asyncio.gather(
                *(self._check_agent(aid) for aid in agent_ids),
                return_exceptions=True,
            )

    async def _cleanup_ephemeral_agents(self) -> None:
        """Remove ephemeral (spawned) agents that have exceeded their TTL."""
        now = time.time()
        async with self._get_agent_lock():
            for agent_id in list(self.agents.keys()):
                info = self.runtime.agents.get(agent_id, {})
                if not info.get("ephemeral"):
                    continue
                ttl = info.get("ttl", 3600)
                spawned_at = info.get("spawned_at", 0)
                if now - spawned_at < ttl:
                    continue
                logger.info("Ephemeral agent '%s' exceeded TTL (%ss), removing", agent_id, ttl)
                try:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, self.runtime.stop_agent, agent_id)
                except Exception as e:
                    logger.warning("Error stopping ephemeral agent '%s': %s", agent_id, e)
                self.router.unregister_agent(agent_id)
                if self._cleanup_agent:
                    self._cleanup_agent(agent_id)
                del self.agents[agent_id]
                if self._event_bus:
                    self._event_bus.emit("agent_state", agent=agent_id, data={
                        "state": "removed", "reason": "ttl_expired",
                    })

    async def _is_loop_stale(self, agent_id: str, now: float) -> bool:
        """Return True if the agent has queued work but its loop hasn't ticked.

        Bug 1 fix: ``Transport.is_reachable`` only checks the /status HTTP
        response — a wedged inner loop with a live FastAPI thread reports
        healthy. We compare the loop's ``last_iteration_ts`` against the
        wall clock and only flag a problem when (a) the lane has actual
        work waiting (``queue_depth > 0``), and (b) the loop hasn't ticked
        in ``_STALE_LOOP_SECONDS``. An idle agent with an empty queue is
        fine — it shouldn't be bumping the counter.

        Best-effort: any error in the /status fetch or queue lookup
        returns ``False`` so this never spuriously flags an otherwise
        healthy agent.
        """
        if self._queue_depth_fn is None:
            return False
        try:
            depth = self._queue_depth_fn(agent_id)
        except Exception as e:
            logger.debug("queue_depth_fn raised for '%s': %s", agent_id, e)
            return False
        if depth <= 0:
            return False
        try:
            status_resp = await self.transport.request(
                agent_id, "GET", "/status", timeout=5,
            )
        except Exception as e:
            logger.debug("Liveness /status fetch failed for '%s': %s", agent_id, e)
            return False
        if not isinstance(status_resp, dict):
            return False
        if "error" in status_resp:
            return False
        last_ts = status_resp.get("last_iteration_ts")
        if last_ts is None:
            # Agent hasn't ticked yet (just booted) — not stale.
            return False
        try:
            return (now - float(last_ts)) > _STALE_LOOP_SECONDS
        except (TypeError, ValueError):
            return False

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
                # Liveness check (Bug 1): an agent can be reachable on
                # /status (FastAPI thread alive) while its inner loop is
                # hung. Only flag stale when the lane queue actually has
                # work waiting — an idle agent legitimately stops bumping
                # the iteration counter.
                stale = await self._is_loop_stale(agent_id, now)
                if stale:
                    logger.warning(
                        "Agent '%s' is reachable with queued work but loop "
                        "appears stale (no iteration in >%ds) — treating as "
                        "unhealthy",
                        agent_id, _STALE_LOOP_SECONDS,
                    )
                    # Fall through to the unhealthy bookkeeping below.
                else:
                    health.consecutive_failures = 0
                    health.last_healthy = now
                    # Codex P2 follow-up: a successful reachability poll
                    # must NOT clobber quarantined status — the agent is
                    # reachable but its credentials are broken, which is
                    # what lane/cron are skipping on. Status flips back
                    # to healthy only via clear_quarantine (operator
                    # edit_agent or TTL expiry).
                    if not health.quarantined:
                        new_status = "healthy"
                        health.status = new_status
                        if new_status != prev_status and self._event_bus:
                            self._event_bus.emit("health_change", agent=agent_id, data={
                                "previous": prev_status, "current": health.status,
                                "failures": health.consecutive_failures,
                                "restart_count": health.restart_count,
                            })
                    return
        except Exception as e:
            logger.debug("Health check transport error for '%s': %s", agent_id, e)

        health.consecutive_failures += 1
        # Codex P2 follow-up: don't flip quarantined → unhealthy on an
        # unreachable poll either. The reachability tracker continues
        # (failures counter ticks toward the restart path) but the
        # status string stays "quarantined" so the dashboard and the
        # lane stay in sync. Restart escalation below still fires if
        # the agent is unreachable AND quarantined — the restart
        # itself doesn't clear quarantine, only edit_agent or TTL does.
        if not health.quarantined:
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

        # Exponential backoff: wait 30s * 2^(restart_count) between restarts
        if health.restart_timestamps:
            last_restart = max(health.restart_timestamps)
            backoff = min(30 * (2 ** len(health.restart_timestamps)), 600)
            if now - last_restart < backoff:
                logger.debug(
                    "Restart backoff for '%s': %.0fs remaining",
                    agent_id, backoff - (now - last_restart),
                )
                return

        logger.info(f"Restarting agent '{agent_id}'...")
        info = self.runtime.agents.get(agent_id)
        if not info:
            logger.error(
                "Cannot restart agent '%s': no stored config. "
                "Manual restart required.", agent_id,
            )
            health.status = "failed"
            return
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.runtime.stop_agent, agent_id)
        except Exception as e:
            logger.warning(f"Error stopping agent '{agent_id}' during restart: {e}")

        try:
            # Preserve operator's ALLOWED_TOOLS on restart
            from src.cli.config import (
                _OPERATOR_AGENT_ID,
                _OPERATOR_ALLOWED_TOOLS,
                _load_permissions,
            )
            restart_env: dict[str, str] = {}
            if agent_id == _OPERATOR_AGENT_ID:
                restart_env["ALLOWED_TOOLS"] = ",".join(_OPERATOR_ALLOWED_TOOLS)
                # Re-seed the internet-access flag so a health-restart
                # doesn't undo the user's toggle. Same logic as
                # cli/runtime.py.
                try:
                    _op_perms = _load_permissions().get(
                        "permissions", {},
                    ).get(_OPERATOR_AGENT_ID, {})
                    restart_env["OL_INTERNET_ACCESS_ENABLED"] = (
                        "true" if _op_perms.get("can_use_internet", True) else "false"
                    )
                except Exception:
                    restart_env["OL_INTERNET_ACCESS_ENABLED"] = "true"

            loop = asyncio.get_running_loop()
            # Load fresh config for proxy resolution
            fresh_cfg = _load_config()
            _agents_cfg = fresh_cfg.get("agents", {})
            _network_cfg = fresh_cfg.get("network", {})
            _proxy_url = resolve_agent_proxy(agent_id, _agents_cfg, _network_cfg)
            _proxy_env = build_proxy_env_vars(
                _proxy_url, _network_cfg.get("no_proxy", ""),
            )
            self.runtime.extra_env.update(_proxy_env)
            # Per-agent output-token cap → LLM_MAX_TOKENS so an operator's
            # max_output_tokens edit survives an automatic crash-recovery
            # restart. Read from fresh YAML (the registry ``info`` dict
            # doesn't carry it); no-op when unset → LLMClient default 8192.
            set_llm_max_tokens_env(restart_env, _agents_cfg.get(agent_id, {}))
            try:
                url = await loop.run_in_executor(
                    None,
                    lambda: self.runtime.start_agent(
                        agent_id=agent_id,
                        role=info.get("role", ""),
                        tools_dir=info.get("tools_dir", ""),
                        model=info.get("model", ""),
                        mcp_servers=info.get("mcp_servers"),
                        thinking=info.get("thinking", ""),
                        env_overrides=restart_env,
                    ),
                )
            finally:
                self.runtime.extra_env.pop("HTTP_PROXY", None)
                self.runtime.extra_env.pop("HTTPS_PROXY", None)
                self.runtime.extra_env.pop("NO_PROXY", None)

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
            # Codex P2 follow-up: restart restores the runtime but does NOT
            # rotate a broken credential. If the agent is still quarantined,
            # snap the status string back to "quarantined" — symmetric with
            # the ``_check_agent`` guards at lines 393 and 414 — so the
            # dashboard and the lane stay in sync. ``unhealthy`` still wins
            # when the runtime didn't come up, because the lane gate (bool
            # flag) is unchanged either way and the operator needs to see
            # the more urgent "still broken" signal.
            if health.quarantined and ready:
                health.status = "quarantined"
            else:
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
                "quarantined": h.quarantined,
                "quarantine_reason": h.quarantine_reason,
                "consecutive_auth_failures": h.consecutive_auth_failures,
            }
            for h in self.agents.values()
        ]
