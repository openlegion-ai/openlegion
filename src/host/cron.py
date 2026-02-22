"""Cron scheduler for autonomous agent triggering.

Runs in the mesh host (not agent containers) so schedules survive
container restarts. Dispatches messages to agents via POST /chat.

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
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Optional

from collections import defaultdict

from src.shared.utils import generate_id, setup_logging

logger = setup_logging("host.cron")

_EMPTY_RESPONSES = frozenset({"", "ok", "heartbeat_ok", "nothing to do", "no updates"})


@dataclass
class HeartbeatProbeResult:
    """Result of a single deterministic probe."""
    name: str
    triggered: bool
    detail: str = ""


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
    workflow: Optional[str] = None
    workflow_payload: Optional[str] = None
    last_run: Optional[str] = None
    next_run: Optional[str] = None
    run_count: int = 0
    error_count: int = 0


class CronScheduler:
    """Persistent cron scheduler that lives in the mesh host.

    Heartbeat jobs run cheap deterministic probes before dispatching to
    the agent. This keeps autonomous operation economical — the LLM is
    only invoked when there's actually something to act on.
    """

    TICK_INTERVAL = 5

    def __init__(
        self,
        config_path: str = "config/cron.json",
        dispatch_fn: Optional[Callable] = None,
        workflow_trigger_fn: Optional[Callable] = None,
        blackboard: Any = None,
        trace_store: Any = None,
        context_fn: Optional[Callable] = None,
    ):
        self.config_path = Path(config_path)
        self.jobs: dict[str, CronJob] = {}
        self.dispatch_fn = dispatch_fn
        self.workflow_trigger_fn = workflow_trigger_fn
        self.blackboard = blackboard
        self._trace_store = trace_store
        self.context_fn = context_fn
        self._running = False
        self._job_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._load()

    def _load(self) -> None:
        if not self.config_path.exists():
            return
        try:
            data = json.loads(self.config_path.read_text())
            if not isinstance(data, dict):
                return
            for job_data in data.get("jobs", []):
                job = CronJob(**job_data)
                self.jobs[job.id] = job
        except Exception as e:
            logger.warning(f"Failed to load cron config: {e}")

    def _save(self) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"jobs": [asdict(j) for j in self.jobs.values()]}
        self.config_path.write_text(json.dumps(data, indent=2) + "\n")

    def add_job(
        self, agent: str, schedule: str, message: str,
        heartbeat: bool = False, **kwargs: Any,
    ) -> CronJob:
        error = self._validate_schedule(schedule)
        if error:
            raise ValueError(error)
        job = CronJob(
            id=generate_id("cron"),
            agent=agent,
            schedule=schedule,
            message=message,
            heartbeat=heartbeat,
            **kwargs,
        )
        self.jobs[job.id] = job
        self._save()
        kind = "heartbeat" if heartbeat else "cron"
        logger.info(f"Added {kind} job {job.id}: agent={agent} schedule={schedule}")
        return job

    def find_heartbeat_job(self, agent: str) -> CronJob | None:
        """Find existing heartbeat job for an agent."""
        for job in self.jobs.values():
            if job.agent == agent and job.heartbeat:
                return job
        return None

    async def update_job(self, job_id: str, **kwargs) -> CronJob | None:
        """Update fields on an existing cron job. Returns updated job or None."""
        async with self._job_locks[job_id]:
            job = self.jobs.get(job_id)
            if not job:
                return None
            for k, v in kwargs.items():
                if hasattr(job, k) and k != "id":
                    setattr(job, k, v)
            self._save()
            return job

    def remove_job(self, job_id: str) -> bool:
        if job_id not in self.jobs:
            return False
        del self.jobs[job_id]
        self._save()
        return True

    def remove_agent_jobs(self, agent_id: str) -> int:
        """Remove ALL jobs for a given agent. Returns count of removed jobs."""
        to_remove = [jid for jid, job in self.jobs.items() if job.agent == agent_id]
        for jid in to_remove:
            del self.jobs[jid]
        if to_remove:
            self._save()
        return len(to_remove)

    async def pause_job(self, job_id: str) -> bool:
        async with self._job_locks[job_id]:
            if job_id not in self.jobs:
                return False
            self.jobs[job_id].enabled = False
            self._save()
            return True

    async def resume_job(self, job_id: str) -> bool:
        async with self._job_locks[job_id]:
            if job_id not in self.jobs:
                return False
            self.jobs[job_id].enabled = True
            self._save()
            return True

    async def run_job(self, job_id: str) -> str | None:
        """Manually trigger a job. Returns the agent response."""
        if job_id not in self.jobs:
            return None
        return await self._execute_job(self.jobs[job_id])

    async def start(self) -> None:
        self._running = True
        logger.info(f"Cron scheduler started with {len(self.jobs)} jobs")
        while self._running:
            await self._tick()
            await asyncio.sleep(self.TICK_INTERVAL)

    def stop(self) -> None:
        self._running = False

    async def _tick(self) -> None:
        now = datetime.now(UTC)
        for job in list(self.jobs.values()):
            if not job.enabled:
                continue
            if self._is_due(job, now):
                asyncio.create_task(self._execute_job(job))

    async def _execute_job(self, job: CronJob) -> str | None:
        try:
            async with self._job_locks[job.id]:
                job.last_run = datetime.now(UTC).isoformat()
                job.run_count += 1
                self._save()
            if self._trace_store:
                from src.shared.trace import new_trace_id
                self._trace_store.record(
                    trace_id=new_trace_id(), source="cron", agent=job.agent,
                    event_type="cron_trigger",
                    detail=f"job={job.id} schedule={job.schedule}",
                )

            response = None
            if job.workflow and self.workflow_trigger_fn:
                payload = json.loads(job.workflow_payload) if job.workflow_payload else {}
                payload.setdefault("date", datetime.now(UTC).strftime("%Y-%m-%d"))
                response = await self.workflow_trigger_fn(job.workflow, payload)
                logger.info(f"Cron {job.id} triggered workflow '{job.workflow}'")
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

                    # Skip-LLM optimization: no custom rules, no activity, no probes
                    if is_default and not has_activity and not triggered:
                        logger.debug(
                            "Heartbeat %s: skipped (default rules, no activity, "
                            "no probes) for '%s'", job.id, job.agent,
                        )
                        return None

                    # Build rich heartbeat message
                    sections: list[str] = [f"Heartbeat for {job.agent}."]

                    # Hardcoded operating rules — always included, cannot be
                    # overridden by agent-editable HEARTBEAT.md
                    sections.append(
                        "## Heartbeat Operating Rules (non-negotiable)\n\n"
                        "1. Be ECONOMICAL. Each heartbeat costs API credits. "
                        "Only call tools if there is actual work to do.\n"
                        "2. If nothing needs attention, respond HEARTBEAT_OK "
                        "immediately. Do NOT make unnecessary tool calls.\n"
                        "3. Report what you worked on to the USER via "
                        "notify_user — not the blackboard.\n"
                        "4. The blackboard is for sharing data with other "
                        "agents. Do NOT write status updates or progress "
                        "reports there.\n"
                        "5. Do NOT change your heartbeat schedule to run "
                        "more frequently unless the user asked you to. "
                        "More frequent heartbeats waste credits."
                    )

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
                            "Work toward your goals. If nothing needs attention, "
                            "respond HEARTBEAT_OK."
                        )
                    else:
                        sections.append(
                            "Follow your HEARTBEAT.md rules. If nothing needs "
                            "attention, respond HEARTBEAT_OK."
                        )

                    message = "\n\n".join(sections)
                    response = await self.dispatch_fn(job.agent, message)
                    logger.info(
                        f"Heartbeat {job.id}: dispatched for '{job.agent}' "
                        f"({len(triggered)} probes triggered)"
                    )
                else:
                    response = await self.dispatch_fn(job.agent, job.message)

                if job.suppress_empty and _is_empty_response(response):
                    logger.debug(f"Cron {job.id}: suppressed empty response")
                    return response
                logger.info(f"Cron {job.id} executed for agent '{job.agent}'")

            return response
        except Exception as e:
            async with self._job_locks[job.id]:
                job.error_count += 1
                self._save()
            logger.error(f"Cron {job.id} failed: {e}")
            return None

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

        Reuses probe results to determine which prefixes have entries,
        avoiding duplicate blackboard queries.
        """
        if not self.blackboard:
            return ""
        # Only fetch details for prefixes that probes already found entries for
        probe_map = {p.name: p for p in probes}
        parts: list[str] = []

        if probe_map.get("pending_signals", HeartbeatProbeResult("", False)).triggered:
            try:
                signals = self.blackboard.list_by_prefix(f"signals/{agent}")
                if signals:
                    lines = []
                    for entry in signals[:max_items]:
                        val = json.dumps(entry.value, default=str)
                        if len(val) > 200:
                            val = val[:200] + "..."
                        lines.append(f"- `{entry.key}`: {val}")
                    parts.append("## Pending Signals\n\n" + "\n".join(lines))
            except Exception as e:
                logger.debug("Failed to get signal details for '%s': %s", agent, e)

        if probe_map.get("pending_tasks", HeartbeatProbeResult("", False)).triggered:
            try:
                tasks = self.blackboard.list_by_prefix(f"tasks/{agent}")
                if tasks:
                    lines = []
                    for entry in tasks[:max_items]:
                        val = json.dumps(entry.value, default=str)
                        if len(val) > 200:
                            val = val[:200] + "..."
                        lines.append(f"- `{entry.key}`: {val}")
                    parts.append("## Pending Tasks\n\n" + "\n".join(lines))
            except Exception as e:
                logger.debug("Failed to get task details for '%s': %s", agent, e)
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
                    last = last.replace(tzinfo=UTC)
                return (now - last).total_seconds() >= seconds
            return True

        parts = schedule.split()
        if len(parts) == 5:
            return self._match_cron(parts, now, job)

        return False

    @staticmethod
    def _validate_schedule(schedule: str) -> str | None:
        """Validate a schedule string. Returns error message or None."""
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
                last = last.replace(tzinfo=UTC)
            if (now - last).total_seconds() < 60:
                return False

        fields = [
            (now.minute, 0, 59),
            (now.hour, 0, 23),
            (now.day, 1, 31),
            (now.month, 1, 12),
            (now.isoweekday() % 7, 0, 6),  # 0=Sun to match cron convention
        ]

        for field_str, (current, _low, _high) in zip(parts, fields):
            if not _match_cron_field(field_str, current):
                return False
        return True

    def list_jobs(self) -> list[dict]:
        return [asdict(j) for j in self.jobs.values()]


def _match_cron_field(field: str, current: int) -> bool:
    if field == "*":
        return True
    for segment in field.split(","):
        if "/" in segment:
            base, step_str = segment.split("/", 1)
            step = int(step_str)
            if base == "*" and current % step == 0:
                return True
        elif "-" in segment:
            start, end = map(int, segment.split("-", 1))
            if start <= current <= end:
                return True
        else:
            try:
                if int(segment) == current:
                    return True
            except ValueError:
                pass
    return False


def _is_empty_response(response: Any) -> bool:
    if not response:
        return True
    if isinstance(response, str):
        return response.strip().lower() in _EMPTY_RESPONSES
    return False
