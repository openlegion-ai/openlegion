"""Cron scheduler for autonomous agent triggering.

Runs in the mesh host (not agent containers) so schedules survive
container restarts. Dispatches messages to agents via POST /chat.

Supports:
  - Standard 5-field cron expressions: "0 9 * * 1-5"
  - Interval shorthand: "every 30m", "every 2h", "every 1d"
  - Heartbeat pattern: runs deterministic probes first, only
    dispatches to agent (costing LLM tokens) when probes find
    actionable items.

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
    ):
        self.config_path = Path(config_path)
        self.jobs: dict[str, CronJob] = {}
        self.dispatch_fn = dispatch_fn
        self.workflow_trigger_fn = workflow_trigger_fn
        self.blackboard = blackboard
        self._running = False
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

    def remove_job(self, job_id: str) -> bool:
        if job_id not in self.jobs:
            return False
        del self.jobs[job_id]
        self._save()
        return True

    def pause_job(self, job_id: str) -> bool:
        if job_id not in self.jobs:
            return False
        self.jobs[job_id].enabled = False
        self._save()
        return True

    def resume_job(self, job_id: str) -> bool:
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
            job.last_run = datetime.now(UTC).isoformat()
            job.run_count += 1
            self._save()

            response = None
            if job.workflow and self.workflow_trigger_fn:
                payload = json.loads(job.workflow_payload) if job.workflow_payload else {}
                payload.setdefault("date", datetime.now(UTC).strftime("%Y-%m-%d"))
                response = await self.workflow_trigger_fn(job.workflow, payload)
                logger.info(f"Cron {job.id} triggered workflow '{job.workflow}'")
            elif self.dispatch_fn:
                # Heartbeat: run cheap probes first, only dispatch if triggered
                if job.heartbeat:
                    probes = self._run_heartbeat_probes(job.agent)
                    triggered = [p for p in probes if p.triggered]
                    if not triggered:
                        logger.debug(f"Heartbeat {job.id}: all probes clean, skipping LLM")
                        return None
                    probe_summary = "\n".join(
                        f"- [{p.name}] {p.detail}" for p in triggered
                    )
                    message = (
                        f"Heartbeat alert — the following probes triggered:\n"
                        f"{probe_summary}\n\n"
                        f"Check HEARTBEAT.md for your autonomous rules and take action."
                    )
                    response = await self.dispatch_fn(job.agent, message)
                    logger.info(
                        f"Heartbeat {job.id}: {len(triggered)} probes triggered for '{job.agent}'"
                    )
                else:
                    response = await self.dispatch_fn(job.agent, job.message)

                if job.suppress_empty and _is_empty_response(response):
                    logger.debug(f"Cron {job.id}: suppressed empty response")
                    return response
                logger.info(f"Cron {job.id} executed for agent '{job.agent}'")

            return response
        except Exception as e:
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
