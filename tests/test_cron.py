"""Tests for CronScheduler: scheduling, intervals, dispatch, persistence, heartbeat."""

from __future__ import annotations

import shutil
import tempfile
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.host.cron import CronScheduler, _match_cron_field


class TestCronFieldMatching:
    def test_wildcard(self):
        assert _match_cron_field("*", 0)
        assert _match_cron_field("*", 59)

    def test_exact(self):
        assert _match_cron_field("5", 5)
        assert not _match_cron_field("5", 6)

    def test_range(self):
        assert _match_cron_field("1-5", 3)
        assert not _match_cron_field("1-5", 6)

    def test_list(self):
        assert _match_cron_field("1,3,5", 3)
        assert not _match_cron_field("1,3,5", 2)

    def test_step(self):
        assert _match_cron_field("*/5", 10)
        assert _match_cron_field("*/5", 0)
        assert not _match_cron_field("*/5", 3)


class TestCronScheduler:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.config_path = f"{self._tmpdir}/cron.json"

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_add_job(self):
        sched = CronScheduler(config_path=self.config_path)
        job = sched.add_job(agent="researcher", schedule="0 9 * * 1-5", message="Morning check")
        assert job.id.startswith("cron_")
        assert job.agent == "researcher"
        assert job.enabled is True
        assert len(sched.jobs) == 1

    def test_persistence(self):
        sched = CronScheduler(config_path=self.config_path)
        sched.add_job(agent="test", schedule="every 30m", message="ping")

        sched2 = CronScheduler(config_path=self.config_path)
        assert len(sched2.jobs) == 1
        job = list(sched2.jobs.values())[0]
        assert job.agent == "test"
        assert job.schedule == "every 30m"

    def test_remove_job(self):
        sched = CronScheduler(config_path=self.config_path)
        job = sched.add_job(agent="test", schedule="every 1h", message="ping")
        assert sched.remove_job(job.id)
        assert len(sched.jobs) == 0

    def test_remove_nonexistent(self):
        sched = CronScheduler(config_path=self.config_path)
        assert not sched.remove_job("nonexistent")

    def test_pause_resume(self):
        sched = CronScheduler(config_path=self.config_path)
        job = sched.add_job(agent="test", schedule="every 1h", message="ping")
        assert sched.pause_job(job.id)
        assert sched.jobs[job.id].enabled is False
        assert sched.resume_job(job.id)
        assert sched.jobs[job.id].enabled is True

    def test_list_jobs(self):
        sched = CronScheduler(config_path=self.config_path)
        sched.add_job(agent="a", schedule="every 1h", message="m1")
        sched.add_job(agent="b", schedule="every 2h", message="m2")
        jobs = sched.list_jobs()
        assert len(jobs) == 2
        agents = {j["agent"] for j in jobs}
        assert agents == {"a", "b"}


class TestCronIntervalDue:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.config_path = f"{self._tmpdir}/cron.json"

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_interval_due_first_time(self):
        sched = CronScheduler(config_path=self.config_path)
        job = sched.add_job(agent="test", schedule="every 30m", message="ping")
        assert sched._is_due(job, datetime.now(UTC))

    def test_interval_not_due_too_soon(self):
        sched = CronScheduler(config_path=self.config_path)
        job = sched.add_job(agent="test", schedule="every 30m", message="ping")
        job.last_run = datetime.now(UTC).isoformat()
        assert not sched._is_due(job, datetime.now(UTC))


class TestCronDispatch:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.config_path = f"{self._tmpdir}/cron.json"

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_execute_job_calls_dispatch(self):
        dispatch = AsyncMock(return_value="Done!")
        sched = CronScheduler(config_path=self.config_path, dispatch_fn=dispatch)
        job = sched.add_job(agent="test", schedule="every 1m", message="hello")

        result = await sched._execute_job(job)
        dispatch.assert_called_once_with("test", "hello")
        assert result == "Done!"
        assert job.run_count == 1

    @pytest.mark.asyncio
    async def test_execute_job_increments_error_on_failure(self):
        dispatch = AsyncMock(side_effect=RuntimeError("connection failed"))
        sched = CronScheduler(config_path=self.config_path, dispatch_fn=dispatch)
        job = sched.add_job(agent="test", schedule="every 1m", message="hello")

        result = await sched._execute_job(job)
        assert result is None
        assert job.error_count == 1

    @pytest.mark.asyncio
    async def test_suppress_empty_response(self):
        dispatch = AsyncMock(return_value="HEARTBEAT_OK")
        sched = CronScheduler(config_path=self.config_path, dispatch_fn=dispatch)
        job = sched.add_job(agent="test", schedule="every 1m", message="check", suppress_empty=True)

        result = await sched._execute_job(job)
        assert result == "HEARTBEAT_OK"

    @pytest.mark.asyncio
    async def test_manual_run(self):
        dispatch = AsyncMock(return_value="Manual result")
        sched = CronScheduler(config_path=self.config_path, dispatch_fn=dispatch)
        job = sched.add_job(agent="test", schedule="every 1h", message="hello")

        result = await sched.run_job(job.id)
        assert result == "Manual result"
        dispatch.assert_called_once()

    @pytest.mark.asyncio
    async def test_manual_run_nonexistent(self):
        sched = CronScheduler(config_path=self.config_path)
        result = await sched.run_job("nonexistent")
        assert result is None


class TestHeartbeat:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.config_path = f"{self._tmpdir}/cron.json"

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_add_heartbeat_job(self):
        sched = CronScheduler(config_path=self.config_path)
        job = sched.add_job(agent="test", schedule="every 15m", message="heartbeat", heartbeat=True)
        assert job.heartbeat is True

    def test_heartbeat_persists(self):
        sched = CronScheduler(config_path=self.config_path)
        sched.add_job(agent="test", schedule="every 15m", message="heartbeat", heartbeat=True)
        sched2 = CronScheduler(config_path=self.config_path)
        job = list(sched2.jobs.values())[0]
        assert job.heartbeat is True

    def test_heartbeat_probes_no_blackboard(self):
        sched = CronScheduler(config_path=self.config_path)
        results = sched._run_heartbeat_probes("test")
        # Should still return disk usage probe
        assert any(r.name == "disk_usage" for r in results)

    def test_heartbeat_probes_with_signals(self):
        mock_bb = MagicMock()
        mock_bb.list_by_prefix.return_value = [MagicMock()]
        sched = CronScheduler(config_path=self.config_path, blackboard=mock_bb)
        results = sched._run_heartbeat_probes("test")
        signal_probes = [r for r in results if r.name == "pending_signals"]
        assert len(signal_probes) == 1
        assert signal_probes[0].triggered is True

    def test_heartbeat_probes_no_signals(self):
        mock_bb = MagicMock()
        mock_bb.list_by_prefix.return_value = []
        sched = CronScheduler(config_path=self.config_path, blackboard=mock_bb)
        results = sched._run_heartbeat_probes("test")
        signal_probes = [r for r in results if r.name == "pending_signals"]
        assert len(signal_probes) == 1
        assert signal_probes[0].triggered is False

    @pytest.mark.asyncio
    async def test_heartbeat_skips_dispatch_when_clean(self):
        dispatch = AsyncMock(return_value="should not be called")
        mock_bb = MagicMock()
        mock_bb.list_by_prefix.return_value = []
        sched = CronScheduler(
            config_path=self.config_path, dispatch_fn=dispatch, blackboard=mock_bb,
        )
        job = sched.add_job(agent="test", schedule="every 15m", message="heartbeat", heartbeat=True)
        result = await sched._execute_job(job)
        dispatch.assert_not_called()
        assert result is None

    @pytest.mark.asyncio
    async def test_heartbeat_dispatches_when_triggered(self):
        dispatch = AsyncMock(return_value="Taking action")
        mock_bb = MagicMock()
        mock_bb.list_by_prefix.side_effect = lambda prefix: [MagicMock()] if "signals" in prefix else []
        sched = CronScheduler(
            config_path=self.config_path, dispatch_fn=dispatch, blackboard=mock_bb,
        )
        job = sched.add_job(agent="test", schedule="every 15m", message="heartbeat", heartbeat=True)
        result = await sched._execute_job(job)
        dispatch.assert_called_once()
        assert result == "Taking action"
