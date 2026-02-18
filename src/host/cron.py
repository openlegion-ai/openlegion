"""Cron scheduler for autonomous agent triggering.

Runs in the mesh host (not agent containers) so schedules survive
container restarts. Dispatches messages to agents via POST /chat.

Supports:
  - Standard 5-field cron expressions: "0 9 * * 1-5"
  - Interval shorthand: "every 30m", "every 2h", "every 1d"
  - Heartbeat pattern: sends "Check HEARTBEAT.md", suppresses OK replies

State persisted to config/cron.json, hot-reloadable.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Optional

from src.shared.utils import generate_id, setup_logging

logger = setup_logging("host.cron")

_EMPTY_RESPONSES = frozenset({"", "ok", "heartbeat_ok", "nothing to do", "no updates"})


@dataclass
class CronJob:
    id: str
    agent: str
    schedule: str
    message: str
    timezone: str = "UTC"
    enabled: bool = True
    suppress_empty: bool = True
    last_run: Optional[str] = None
    next_run: Optional[str] = None
    run_count: int = 0
    error_count: int = 0


class CronScheduler:
    """Persistent cron scheduler that lives in the mesh host."""

    TICK_INTERVAL = 15

    def __init__(
        self,
        config_path: str = "config/cron.json",
        dispatch_fn: Optional[Callable] = None,
    ):
        self.config_path = Path(config_path)
        self.jobs: dict[str, CronJob] = {}
        self.dispatch_fn = dispatch_fn
        self._running = False
        self._load()

    def _load(self) -> None:
        if not self.config_path.exists():
            return
        try:
            data = json.loads(self.config_path.read_text())
            for job_data in data.get("jobs", []):
                job = CronJob(**job_data)
                self.jobs[job.id] = job
        except Exception as e:
            logger.warning(f"Failed to load cron config: {e}")

    def _save(self) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"jobs": [asdict(j) for j in self.jobs.values()]}
        self.config_path.write_text(json.dumps(data, indent=2) + "\n")

    def add_job(self, agent: str, schedule: str, message: str, **kwargs: Any) -> CronJob:
        job = CronJob(
            id=generate_id("cron"),
            agent=agent,
            schedule=schedule,
            message=message,
            **kwargs,
        )
        self.jobs[job.id] = job
        self._save()
        logger.info(f"Added cron job {job.id}: agent={agent} schedule={schedule}")
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
            if self.dispatch_fn:
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

    def _is_due(self, job: CronJob, now: datetime) -> bool:
        schedule = job.schedule.strip()

        interval_match = re.match(r"every\s+(\d+)([mhd])", schedule, re.IGNORECASE)
        if interval_match:
            amount = int(interval_match.group(1))
            unit = interval_match.group(2).lower()
            seconds = {"m": 60, "h": 3600, "d": 86400}[unit] * amount
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
