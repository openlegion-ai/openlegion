"""Cron scheduler for autonomous agent triggering.

Runs in the mesh host (not agent containers) so schedules survive
container restarts. Dispatches messages to agents via POST /chat or
invokes tools directly via POST /invoke (no LLM involved).

Supports:
  - Standard 5-field cron expressions: "0 9 * * 1-5"
  - Interval shorthand: "every 30m", "every 2h", "every 1d"
  - Heartbeat pattern: runs deterministic probes, fetches agent
    workspace context (HEARTBEAT.md, daily logs), and skips
    LLM dispatch when there is nothing actionable (no custom
    rules, no recent activity, no triggered probes).

State persisted to config/cron.json, hot-reloadable.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.shared.utils import dumps_safe, generate_id, setup_logging

logger = setup_logging("host.cron")

_EMPTY_RESPONSES = frozenset({"", "ok", "heartbeat_ok", "nothing to do", "no updates"})
_MAX_CRON_SCAN_MINUTES = 43200  # 30 days
_DEFAULT_MAX_CRON_JOBS_PER_AGENT = 50


def _max_cron_jobs_per_agent() -> int:
    """Per-agent cron-job cap (M11). Env-configurable via
    ``OPENLEGION_MAX_CRON_JOBS_PER_AGENT``; falls back to the default on
    unset / non-integer / non-positive values."""
    raw = os.environ.get("OPENLEGION_MAX_CRON_JOBS_PER_AGENT")
    if raw is None:
        return _DEFAULT_MAX_CRON_JOBS_PER_AGENT
    try:
        val = int(raw)
    except ValueError:
        return _DEFAULT_MAX_CRON_JOBS_PER_AGENT
    return val if val > 0 else _DEFAULT_MAX_CRON_JOBS_PER_AGENT


@dataclass
class HeartbeatProbeResult:
    """Result of a single deterministic probe."""
    name: str
    triggered: bool
    detail: str = ""
    entries: list | None = None  # cached blackboard entries to avoid re-query


@dataclass
class CronJob:
    id: str
    agent: str
    schedule: str
    message: str
    timezone: str = "UTC"
    enabled: bool = True
    suppress_empty: bool = True
    heartbeat: bool = False
    tool_name: str | None = None    # invoke this tool directly — no LLM involved
    tool_params: str | None = None  # JSON-encoded params dict for the tool
    last_run: str | None = None
    next_run: str | None = None
    run_count: int = 0
    error_count: int = 0


class CronScheduler:
    """Persistent cron scheduler that lives in the mesh host.

    Heartbeat jobs run cheap deterministic probes before dispatching to
    the agent, gating the agenda turn on a non-empty plate (pending
    tasks / inbox events / probe alerts / standing goals). A truly-empty
    plate never reaches the LLM — the tick stays a ~zero-cost probe-only
    check.
    """

    TICK_INTERVAL = 5
    DEFAULT_HEARTBEAT_SCHEDULE = "every 15m"

    def __init__(
        self,
        config_path: str = "config/cron.json",
        dispatch_fn: Callable | None = None,
        invoke_fn: Callable | None = None,
        blackboard: Any = None,
        trace_store: Any = None,
        context_fn: Callable | None = None,
        heartbeat_dispatch_fn: Callable | None = None,
        event_bus: Any = None,
        health_monitor: Any = None,
        utility_model_fn: Callable | None = None,
        goals_fn: Callable | None = None,
    ):
        self.config_path = Path(config_path)
        self.jobs: dict[str, CronJob] = {}
        self.dispatch_fn = dispatch_fn
        self.invoke_fn = invoke_fn
        self.blackboard = blackboard
        self._trace_store = trace_store
        self.context_fn = context_fn
        self.heartbeat_dispatch_fn = heartbeat_dispatch_fn
        self._event_bus = event_bus
        # Plate-gate inputs read mesh-side (never from the untrusted
        # container). ``utility_model_fn() -> str`` returns the
        # deployment ``llm.utility_model`` ("" when unset); ``goals_fn(
        # agent) -> dict | None`` returns the agent's standing-goals
        # record from the Team store. Both gate the goal-only initiative
        # escalation: a plate with no actionable items but standing goals
        # dispatches only when a utility (coordination) model exists.
        self.utility_model_fn = utility_model_fn
        self.goals_fn = goals_fn
        # ``health_monitor`` (Fix 5 in seam follow-up): when wired, the
        # scheduler skips dispatch for quarantined agents. Heartbeat
        # cron jobs would otherwise keep ticking on a broken credential.
        self.health_monitor = health_monitor
        self._running = False
        self._job_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._load()

    def set_health_monitor(self, health_monitor: Any) -> None:
        """Wire the health monitor after construction (Fix 5).

        Mirrors the ``LaneManager.set_tasks_store`` injection pattern —
        ``CronScheduler`` may be built before ``HealthMonitor`` in some
        bootstrap orderings. ``None`` disables the quarantine skip.
        """
        self.health_monitor = health_monitor

    def _load(self) -> None:
        if not self.config_path.exists():
            return
        try:
            data = json.loads(self.config_path.read_text())
            if not isinstance(data, dict):
                return
            _valid_fields = {f.name for f in CronJob.__dataclass_fields__.values()}
            for job_data in data.get("jobs", []):
                # Strip unknown keys (e.g. removed "workflow" fields) for compat
                job_data = {k: v for k, v in job_data.items() if k in _valid_fields}
                job = CronJob(**job_data)
                self._compute_next_run(job)
                self.jobs[job.id] = job
        except Exception as e:
            logger.warning(f"Failed to load cron config: {e}")

    def _save(self) -> None:
        """Persist jobs to cron.json atomically (write-to-temp + rename)."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"jobs": [asdict(j) for j in self.jobs.values()]}
        content = json.dumps(data, indent=2) + "\n"
        # Atomic write: write to a temp file in the same directory, then rename.
        # This prevents partial reads if two async tasks save concurrently.
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.config_path.parent), suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w") as f:
                f.write(content)
        except BaseException:
            # os.fdopen failed or write failed — close fd if still open
            try:
                os.close(fd)
            except OSError:
                pass  # already closed by fdopen
            Path(tmp_path).unlink(missing_ok=True)
            raise
        try:
            Path(tmp_path).replace(self.config_path)
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise

    def _compute_next_run(self, job: CronJob) -> None:
        """Compute and set next_run for a job based on its schedule."""
        schedule = job.schedule.strip()
        now = datetime.now(timezone.utc)

        interval_match = re.match(r"every\s+(\d+)([smhd])", schedule, re.IGNORECASE)
        if interval_match:
            amount = int(interval_match.group(1))
            unit = interval_match.group(2).lower()
            seconds = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit] * amount
            if job.last_run:
                last = datetime.fromisoformat(job.last_run)
                if last.tzinfo is None:
                    last = last.replace(tzinfo=timezone.utc)
                job.next_run = (last + timedelta(seconds=seconds)).isoformat()
            else:
                job.next_run = now.isoformat()
            return

        parts = schedule.split()
        if len(parts) == 5:
            candidate = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
            for _ in range(_MAX_CRON_SCAN_MINUTES):  # scan up to 30 days
                if all(
                    _match_cron_field(f, c)
                    for f, c in zip(parts, [
                        candidate.minute, candidate.hour, candidate.day,
                        candidate.month, candidate.isoweekday() % 7,
                    ], strict=True)
                ):
                    job.next_run = candidate.isoformat()
                    return
                candidate += timedelta(minutes=1)

        job.next_run = None

    def add_job(
        self, agent: str, schedule: str, message: str = "",
        heartbeat: bool = False,
        tool_name: str | None = None,
        tool_params: str | None = None,
        **kwargs: Any,
    ) -> CronJob:
        error = self._validate_schedule(schedule)
        if error:
            raise ValueError(error)
        # M11: cap the number of cron jobs per agent so a single agent
        # can't exhaust the scheduler (or the config file) by registering
        # unbounded jobs. Default 50, env-overridable.
        cap = _max_cron_jobs_per_agent()
        existing = sum(1 for j in self.jobs.values() if j.agent == agent)
        if existing >= cap:
            raise ValueError(
                f"Agent '{agent}' has reached the cron job limit ({cap}). "
                f"Remove an existing job before adding another."
            )
        job = CronJob(
            id=generate_id("cron"),
            agent=agent,
            schedule=schedule,
            message=message,
            heartbeat=heartbeat,
            tool_name=tool_name,
            tool_params=tool_params,
            **kwargs,
        )
        self._compute_next_run(job)
        self.jobs[job.id] = job
        self._save()
        if heartbeat:
            kind = "heartbeat"
        elif tool_name:
            kind = "tool"
        else:
            kind = "message"
        logger.info(f"Added {kind} job {job.id}: agent={agent} schedule={schedule}")
        self._emit_cron_change("created", job)
        return job

    def find_heartbeat_job(self, agent: str) -> CronJob | None:
        """Find existing heartbeat job for an agent."""
        for job in self.jobs.values():
            if job.agent == agent and job.heartbeat:
                return job
        return None

    def ensure_heartbeat(self, agent: str, schedule: str | None = None) -> CronJob:
        """Idempotently ensure an agent has a heartbeat cron job.

        Returns the existing job if one exists, otherwise creates one.
        """
        existing = self.find_heartbeat_job(agent)
        if existing:
            return existing
        return self.add_job(
            agent=agent, schedule=schedule or self.DEFAULT_HEARTBEAT_SCHEDULE,
            message=f"Heartbeat check for {agent}", heartbeat=True,
        )

    # Default daily-summary schedule. 9am cron format. Per-team override
    # via team metadata ``summary_schedule``. Operator can also update
    # via ``PUT /mesh/cron/{id}`` once the job exists.
    DEFAULT_SUMMARY_SCHEDULE = "0 9 * * *"

    def find_summary_job(
        self, scope_kind: str, scope_id: str,
    ) -> CronJob | None:
        """Find existing daily-summary cron job for a given scope.

        Matches on ``tool_name == "compose_work_summary"`` plus the
        ``scope_kind`` / ``scope_id`` pair inside the job's
        ``tool_params`` JSON. Returns None when no match.
        """
        for job in self.jobs.values():
            if job.tool_name != "compose_work_summary":
                continue
            try:
                params = json.loads(job.tool_params or "{}")
            except (json.JSONDecodeError, TypeError):
                continue
            if (params.get("scope_kind") == scope_kind
                    and params.get("scope_id") == scope_id):
                return job
        return None

    def ensure_summary_job(
        self,
        *,
        scope_kind: str,
        scope_id: str,
        schedule: str | None = None,
        operator_id: str = "operator",
    ) -> CronJob:
        """Idempotently ensure a daily ``compose_work_summary`` cron
        job exists for a given scope.

        Fires the tool directly (``tool_name="compose_work_summary"``)
        so each tick is a deterministic tool invocation — no LLM call
        per fire. Per-team schedule customization arrives via the
        ``schedule`` arg (sourced from team metadata at boot) or via
        the existing ``PUT /mesh/cron/{id}`` operator endpoint.
        """
        existing = self.find_summary_job(scope_kind, scope_id)
        if existing:
            return existing
        return self.add_job(
            agent=operator_id,
            schedule=schedule or self.DEFAULT_SUMMARY_SCHEDULE,
            tool_name="compose_work_summary",
            tool_params=json.dumps({
                "scope_kind": scope_kind,
                "scope_id": scope_id,
            }),
        )

    _UPDATABLE_FIELDS = frozenset({
        "schedule", "message", "enabled", "suppress_empty",
        "tool_name", "tool_params",
    })

    async def update_job(self, job_id: str, **kwargs) -> CronJob | None:
        """Update fields on an existing cron job. Returns updated job or None."""
        async with self._job_locks[job_id]:
            job = self.jobs.get(job_id)
            if not job:
                return None
            # H8 validate-before-mutate: a bad schedule must never persist
            # in the live job. Validate the candidate BEFORE setattr so a
            # rejected update leaves the job's schedule untouched (the old
            # ordering setattr'd job.schedule first, so a poison value
            # stuck in memory even when _compute_next_run later choked).
            if "schedule" in kwargs:
                error = self._validate_schedule(kwargs["schedule"])
                if error:
                    raise ValueError(error)
            for k, v in kwargs.items():
                if k in self._UPDATABLE_FIELDS and hasattr(job, k):
                    setattr(job, k, v)
            if "schedule" in kwargs:
                self._compute_next_run(job)
            self._save()
            self._emit_cron_change("updated", job)
            return job

    def remove_job(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job is None:
            return False
        del self.jobs[job_id]
        self._save()
        self._emit_cron_change("removed", job)
        return True

    def remove_agent_jobs(self, agent_id: str) -> int:
        """Remove ALL jobs for a given agent. Returns count of removed jobs."""
        to_remove = [jid for jid, job in self.jobs.items() if job.agent == agent_id]
        for jid in to_remove:
            del self.jobs[jid]
        if to_remove:
            self._save()
            if self._event_bus:
                try:
                    self._event_bus.emit("cron_change", agent=agent_id, data={
                        "action": "removed_all", "count": len(to_remove),
                    })
                except Exception:
                    logger.debug("Failed to emit cron_change event", exc_info=True)
        return len(to_remove)

    async def pause_job(self, job_id: str) -> bool:
        async with self._job_locks[job_id]:
            if job_id not in self.jobs:
                return False
            self.jobs[job_id].enabled = False
            self._save()
            self._emit_cron_change("paused", self.jobs[job_id])
            return True

    async def resume_job(self, job_id: str) -> bool:
        async with self._job_locks[job_id]:
            if job_id not in self.jobs:
                return False
            self.jobs[job_id].enabled = True
            self._compute_next_run(self.jobs[job_id])
            self._save()
            self._emit_cron_change("resumed", self.jobs[job_id])
            return True

    def _emit_cron_change(self, action: str, job: CronJob) -> None:
        """Emit a cron_change event so the dashboard updates without polling."""
        if self._event_bus:
            try:
                self._event_bus.emit("cron_change", agent=job.agent, data={
                    "action": action, "job_id": job.id,
                    "schedule": job.schedule,
                })
            except Exception:
                logger.debug("Failed to emit cron_change event", exc_info=True)

    async def run_job(self, job_id: str) -> str | None:
        """Manually trigger a job. Returns the agent response."""
        if job_id not in self.jobs:
            return None
        return await self._execute_job(self.jobs[job_id], manual=True)

    async def start(self) -> None:
        self._running = True
        logger.info(f"Cron scheduler started with {len(self.jobs)} jobs")
        while self._running:
            await self._tick()
            await asyncio.sleep(self.TICK_INTERVAL)

    def stop(self) -> None:
        self._running = False

    async def _tick(self) -> None:
        now = datetime.now(timezone.utc)
        for job in list(self.jobs.values()):
            # H8 crash containment: one poison job (e.g. a malformed
            # schedule that slipped past validation) must never kill the
            # scheduler task and silence ALL heartbeats / jobs fleet-wide.
            # Wrap the per-job body so a raise here logs + continues to
            # the next job instead of propagating out of the while loop.
            try:
                if not job.enabled:
                    continue
                if self._is_due(job, now):
                    asyncio.create_task(self._execute_job(job))
            except Exception:
                logger.error(
                    "Cron tick: job %s raised during scheduling — skipping",
                    getattr(job, "id", "<unknown>"), exc_info=True,
                )
                continue

    async def _execute_job(self, job: CronJob, manual: bool = False) -> str | None:
        lock = self._job_locks[job.id]
        if lock.locked():
            logger.debug("Job %s already running, skipping this tick", job.id)
            return None
        # Fix 5 (seam follow-up): quarantined agents are skipped — the
        # cron tick is the cheapest place to bail out, before we burn
        # probes / context / LLM cost on an agent that can't run. Skip
        # is silent (debug-level) because quarantine itself already
        # surfaces a dashboard notification.
        if (
            self.health_monitor is not None
            and self.health_monitor.is_quarantined(job.agent)
        ):
            logger.debug(
                "Cron %s: agent '%s' quarantined — skipping dispatch "
                "(clear via edit_agent)",
                job.id, job.agent,
            )
            return None
        async with lock:
            try:
                job.last_run = datetime.now(timezone.utc).isoformat()
                job.run_count += 1
                self._compute_next_run(job)
                self._save()
                self._emit_cron_change("started", job)
                if self._trace_store:
                    from src.shared.trace import new_trace_id
                    self._trace_store.record(
                        trace_id=new_trace_id(), source="cron", agent=job.agent,
                        event_type="cron_trigger",
                        detail=f"job={job.id} schedule={job.schedule}",
                    )

                response = None
                if job.tool_name:
                    if not self.invoke_fn:
                        logger.error(
                            "Cron %s: tool '%s' configured but no invoke_fn available — skipping",
                            job.id, job.tool_name,
                        )
                        return None
                    params: dict = {}
                    if job.tool_params:
                        try:
                            params = json.loads(job.tool_params)
                        except json.JSONDecodeError:
                            logger.warning(
                                "Cron %s: invalid tool_params JSON, invoking with no params", job.id,
                            )
                    result = await self.invoke_fn(job.agent, job.tool_name, params)
                    response = json.dumps(result) if isinstance(result, dict) else str(result)
                    logger.info(
                        "Cron %s invoked tool '%s' on agent '%s'",
                        job.id, job.tool_name, job.agent,
                    )
                elif self.dispatch_fn:
                    if job.heartbeat:
                        probes = self._run_heartbeat_probes(job.agent)
                        triggered = [p for p in probes if p.triggered]

                        # Fetch agent workspace context (HEARTBEAT.md, daily logs)
                        ctx = {}
                        if self.context_fn:
                            try:
                                ctx = await self.context_fn(job.agent)
                            except Exception as e:
                                logger.warning(
                                    "context_fn failed for '%s': %s", job.agent, e,
                                )

                        is_default = ctx.get("is_default_heartbeat", True)
                        has_activity = ctx.get("has_recent_activity", False)

                        # Plate gate (Phase-3 unit 2): the heartbeat becomes an
                        # agenda dispatch. ACTIONABLE items — triggered probes
                        # (disk / pending signals / pending tasks), recent
                        # activity, or custom HEARTBEAT.md rules — always
                        # escalate to an agenda turn (work responding to work;
                        # pre-B2 behavior already dispatched on activity/probes).
                        # A manual trigger always runs (the user expects it).
                        #
                        # With NO actionable items, only standing goals could
                        # justify a tick. Goal-only initiative REQUIRES the
                        # coordination tier (§8 #11): without a configured
                        # utility model, a speculative goal-only tick would bill
                        # the WORK ledger on the strong model — the exact B2
                        # starvation vector — so it stays a probe-only tick.
                        # A truly-empty plate (no actionable items, no goals, or
                        # no utility model) never reaches the LLM.
                        actionable = bool(triggered) or has_activity or not is_default
                        if not manual and not actionable:
                            has_goals = self._agent_has_goals(job.agent)
                            utility_ready = self._utility_model_configured()
                            if not (has_goals and utility_ready):
                                logger.debug(
                                    "Heartbeat %s: probe-only tick for '%s' "
                                    "(no actionable plate; goals=%s, "
                                    "utility_model=%s)",
                                    job.id, job.agent, has_goals, utility_ready,
                                )
                                return None

                        # Build rich heartbeat message. The leading line
                        # carries the current ISO timestamp so the LLM
                        # has a concrete "now" for date math (e.g. the
                        # 7-day re-ask throttle on goal seeding which
                        # otherwise has to guess against its training
                        # cutoff). Codex r3 (PR 972) flagged the gap.
                        # ``datetime`` + ``timezone`` are imported at
                        # module scope (line 29) — re-importing here
                        # would shadow the module-level binding and
                        # break earlier same-function references.
                        now_iso = datetime.now(timezone.utc).isoformat()
                        sections: list[str] = [
                            f"Heartbeat for {job.agent} at {now_iso}."
                        ]

                        # Hardcoded operating rules — always included, cannot be
                        # overridden by agent-editable HEARTBEAT.md. One rule
                        # set for every agent (solo = team-of-one): a solo
                        # agent's blackboard is its own private namespace, so
                        # the blackboard rule is accurate for it too.
                        hb_rules = (
                            "## Heartbeat Operating Rules (non-negotiable)\n\n"
                            "1. This is your workday tick. Review your plate — "
                            "pending tasks, inbox events, probe alerts, and your "
                            "standing goals — then prioritize and act on what "
                            "matters most.\n"
                            "2. You MAY create goal-directed work toward your "
                            "standing goals (hand_off to yourself) once your "
                            "plate has capacity — check your budget (introspect's "
                            "budget section, including the coordination tier) "
                            "before adding new self-directed work. Your budget is "
                            "the governor — work the plate until it is clear or "
                            "the budget is spent.\n"
                            "3. Report what you worked on to the USER via "
                            "notify_user.\n"
                            "4. The blackboard is for sharing data with other "
                            "agents. Do NOT write status updates or progress "
                            "reports there.\n"
                            "5. If the plate is genuinely empty and nothing "
                            "advances your goals, end the turn without making "
                            "tool calls."
                        )
                        sections.append(hb_rules)

                        rules = ctx.get("heartbeat_rules", "")
                        if rules and not is_default:
                            sections.append(
                                f"## Your Heartbeat Rules\n\n{rules.strip()}"
                            )

                        daily = ctx.get("daily_logs", "")
                        if daily and daily.strip():
                            capped = daily[:4000]
                            if len(daily) > 4000:
                                capped += "\n\n... (truncated)"
                            sections.append(
                                f"## Your Recent Activity\n\n{capped}"
                            )

                        if triggered:
                            probe_lines = "\n".join(
                                f"- [{p.name}] {p.detail}" for p in triggered
                            )
                            sections.append(
                                f"## Probe Alerts\n\n{probe_lines}"
                            )

                        pending = self._get_pending_details(job.agent, probes)
                        if pending:
                            sections.append(pending)

                        if is_default:
                            sections.append(
                                "Review your plate and work toward your goals. If "
                                "the plate is empty and nothing advances your "
                                "goals, end the turn."
                            )
                        else:
                            sections.append(
                                "Follow your HEARTBEAT.md rules and work your "
                                "plate. If nothing needs attention, end the turn."
                            )

                        message = "\n\n".join(sections)

                        # Use dedicated heartbeat endpoint when available.
                        if self.heartbeat_dispatch_fn:
                            hb_result = await self.heartbeat_dispatch_fn(
                                job.agent, message,
                            )
                            # hb_result is a structured dict from execute_heartbeat
                            if isinstance(hb_result, dict):
                                if hb_result.get("skipped"):
                                    logger.debug(
                                        "Heartbeat %s: agent '%s' busy (%s), skipped",
                                        job.id, job.agent,
                                        hb_result.get("reason", "unknown"),
                                    )
                                    return None
                                response = hb_result.get("response", "")
                                if self._event_bus:
                                    self._event_bus.emit(
                                        "heartbeat_complete", agent=job.agent,
                                        data={
                                            "summary": hb_result.get("summary", ""),
                                            "tools_used": hb_result.get("tools_used", []),
                                            "duration_ms": hb_result.get("duration_ms", 0),
                                            "tokens_used": hb_result.get("tokens_used", 0),
                                            "outcome": hb_result.get("outcome", "ok"),
                                        },
                                    )
                            else:
                                response = str(hb_result) if hb_result else ""
                        else:
                            response = await self.dispatch_fn(job.agent, message)

                        logger.info(
                            f"Heartbeat {job.id}: dispatched for '{job.agent}' "
                            f"({len(triggered)} probes triggered)"
                        )
                    else:
                        response = await self.dispatch_fn(job.agent, job.message)

                    if job.suppress_empty and _is_empty_response(response):
                        logger.debug(f"Cron {job.id}: suppressed empty response")
                        self._emit_cron_change("executed", job)
                        return response
                    logger.info(f"Cron {job.id} executed for agent '{job.agent}'")

                self._emit_cron_change("executed", job)
                return response
            except Exception as e:
                job.error_count += 1
                self._save()
                logger.error(f"Cron {job.id} failed: {e}")
                self._emit_cron_change("error", job)
                return None

    def _agent_has_goals(self, agent: str) -> bool:
        """Whether the agent has any standing goals (operator-set).

        Read mesh-side via the same Team-store goals surface the agent's
        own ``_fetch_goals`` reads — never from the untrusted container.
        Missing wiring or a read failure degrades to "no goals" so the
        goal-only escalation stays conservative.
        """
        if self.goals_fn is None:
            return False
        try:
            record = self.goals_fn(agent)
        except Exception as e:
            logger.debug("goals_fn failed for '%s': %s", agent, e)
            return False
        return bool(record and record.get("goals"))

    def _utility_model_configured(self) -> bool:
        """Whether a deployment utility (coordination) model is configured.

        Reads the same ``llm.utility_model`` config the mesh model pin /
        proxy classifier use. Unset (or unwired) ⇒ no coordination tier,
        so goal-only initiative ticks stay probe-only (§8 #11).
        """
        if self.utility_model_fn is None:
            return False
        try:
            return bool(self.utility_model_fn())
        except Exception as e:
            logger.debug("utility_model_fn failed: %s", e)
            return False

    def _run_heartbeat_probes(self, agent: str) -> list[HeartbeatProbeResult]:
        """Run cheap, deterministic probes before invoking the LLM."""
        results: list[HeartbeatProbeResult] = []

        # Probe 1: Disk usage on agent data volume
        try:
            data_dir = Path(tempfile.gettempdir()) / f"openlegion_data_{agent}"
            if not data_dir.exists():
                data_dir = Path(".")
            usage = shutil.disk_usage(str(data_dir))
            pct = (usage.used / usage.total) * 100 if usage.total else 0
            results.append(HeartbeatProbeResult(
                name="disk_usage",
                triggered=pct > 85,
                detail=f"{pct:.0f}% used ({usage.free // (1024**2)}MB free)",
            ))
        except OSError as e:
            logger.debug("Disk usage probe failed for '%s': %s", agent, e)

        # Probe 2: Pending signals on blackboard
        if self.blackboard:
            try:
                signals = self.blackboard.list_by_prefix(f"signals/{agent}")
                results.append(HeartbeatProbeResult(
                    name="pending_signals",
                    triggered=len(signals) > 0,
                    detail=f"{len(signals)} pending signal(s)",
                    entries=signals if signals else None,
                ))
            except Exception as e:
                logger.debug("Pending signals probe failed for '%s': %s", agent, e)

            # Probe 3: Pending tasks on blackboard
            try:
                tasks = self.blackboard.list_by_prefix(f"tasks/{agent}")
                results.append(HeartbeatProbeResult(
                    name="pending_tasks",
                    triggered=len(tasks) > 0,
                    detail=f"{len(tasks)} pending task(s)",
                    entries=tasks if tasks else None,
                ))
            except Exception as e:
                logger.debug("Pending tasks probe failed for '%s': %s", agent, e)

        return results

    def _get_pending_details(
        self,
        agent: str,
        probes: list[HeartbeatProbeResult],
        max_items: int = 5,
    ) -> str:
        """Format actual blackboard entry content for pending signals/tasks.

        Uses cached entries from probe results instead of re-querying the
        blackboard.
        """
        probe_map = {p.name: p for p in probes}
        parts: list[str] = []

        for probe_name, heading in [
            ("pending_signals", "## Pending Signals"),
            ("pending_tasks", "## Pending Tasks"),
        ]:
            probe = probe_map.get(probe_name)
            if probe and probe.triggered and probe.entries:
                lines = []
                for entry in probe.entries[:max_items]:
                    val = dumps_safe(entry.value)
                    if len(val) > 200:
                        val = val[:200] + "..."
                    lines.append(f"- `{entry.key}`: {val}")
                parts.append(f"{heading}\n\n" + "\n".join(lines))

        return "\n\n".join(parts)

    def _is_due(self, job: CronJob, now: datetime) -> bool:
        schedule = job.schedule.strip()

        interval_match = re.match(r"every\s+(\d+)([smhd])", schedule, re.IGNORECASE)
        if interval_match:
            amount = int(interval_match.group(1))
            unit = interval_match.group(2).lower()
            seconds = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit] * amount
            if job.last_run:
                last = datetime.fromisoformat(job.last_run)
                if last.tzinfo is None:
                    last = last.replace(tzinfo=timezone.utc)
                return (now - last).total_seconds() >= seconds
            return True

        parts = schedule.split()
        if len(parts) == 5:
            return self._match_cron(parts, now, job)

        return False

    # Per-field (min, max) inclusive bounds for a 5-field cron expression:
    # minute, hour, day-of-month, month, day-of-week (0=Sun..6=Sat).
    _CRON_FIELD_BOUNDS: tuple[tuple[int, int], ...] = (
        (0, 59),  # minute
        (0, 23),  # hour
        (1, 31),  # day of month
        (1, 12),  # month
        (0, 6),   # day of week
    )

    @staticmethod
    def _validate_cron_field(field: str, low: int, high: int) -> bool:
        """Return True iff ``field`` is a structurally valid cron field
        within [low, high]. Rejects step 0, malformed ranges (``1-``),
        non-numeric values, and out-of-range numbers."""
        if field == "*":
            return True
        for segment in field.split(","):
            if not segment:
                return False
            try:
                if "/" in segment:
                    base, step_str = segment.split("/", 1)
                    step = int(step_str)
                    if step <= 0:
                        return False
                    # Base may be "*" or a numeric start within bounds.
                    if base != "*":
                        base_val = int(base)
                        if not (low <= base_val <= high):
                            return False
                elif "-" in segment:
                    start_str, end_str = segment.split("-", 1)
                    start, end = int(start_str), int(end_str)
                    if start > end or start < low or end > high:
                        return False
                else:
                    val = int(segment)
                    if not (low <= val <= high):
                        return False
            except (ValueError, ZeroDivisionError):
                return False
        return True

    @classmethod
    def _validate_schedule(cls, schedule: str) -> str | None:
        """Validate a schedule string. Returns error message or None.

        Parses each of the 5 cron fields (not just field count) so a
        poison schedule like ``*/0 * * * *`` or ``1- * * * *`` is rejected
        at config-write time (H8) instead of crashing the scheduler loop.
        """
        schedule = schedule.strip()
        if re.match(r"every\s+(\d+)([smhd])", schedule, re.IGNORECASE):
            return None
        parts = schedule.split()
        if len(parts) == 6:
            return (
                "6-field (seconds) cron is not supported. "
                "Use 5-field cron (minute resolution) or 'every Ns' for seconds. "
                "Example: 'every 5s' or '*/1 * * * *'"
            )
        if len(parts) == 5:
            for field, (low, high) in zip(
                parts, cls._CRON_FIELD_BOUNDS, strict=True,
            ):
                if not cls._validate_cron_field(field, low, high):
                    return (
                        f"Invalid cron field '{field}' in schedule "
                        f"'{schedule}'. Each field must be '*', a number, "
                        f"a range (a-b), a list (a,b,c), or a step (*/n with "
                        f"n>0), within range [{low}-{high}]."
                    )
            return None
        return f"Invalid schedule: '{schedule}'. Use 5-field cron or 'every N[s/m/h/d]'"

    def _match_cron(self, parts: list[str], now: datetime, job: CronJob) -> bool:
        """Match a 5-field cron expression, firing at most once per minute."""
        if now.second > self.TICK_INTERVAL:
            return False

        # Dedup: don't fire again within the same minute
        if job.last_run:
            last = datetime.fromisoformat(job.last_run)
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            if (now - last).total_seconds() < 60:
                return False

        fields = [
            (now.minute, 0, 59),
            (now.hour, 0, 23),
            (now.day, 1, 31),
            (now.month, 1, 12),
            (now.isoweekday() % 7, 0, 6),  # 0=Sun to match cron convention
        ]

        return all(
            _match_cron_field(field_str, current)
            for field_str, (current, _low, _high) in zip(parts, fields, strict=True)
        )

    def list_jobs(self) -> list[dict]:
        return [asdict(j) for j in self.jobs.values()]


def _match_cron_field(field: str, current: int) -> bool:
    if field == "*":
        return True
    for segment in field.split(","):
        # H8: a malformed field (step 0, bad range like "1-", non-numeric)
        # must return a non-match rather than raise — a raise here would
        # bubble up through ``_is_due`` and (pre-fix) kill the scheduler.
        try:
            if "/" in segment:
                base, step_str = segment.split("/", 1)
                step = int(step_str)
                # Guard the modulo: step 0 would ZeroDivisionError.
                if step <= 0:
                    continue
                if base == "*" and current % step == 0:
                    return True
            elif "-" in segment:
                start, end = map(int, segment.split("-", 1))
                if start <= current <= end:
                    return True
            else:
                if int(segment) == current:
                    return True
        except (ValueError, ZeroDivisionError):
            continue
    return False


def _is_empty_response(response: Any) -> bool:
    if not response:
        return True
    if isinstance(response, str):
        return response.strip().lower() in _EMPTY_RESPONSES
    return False
