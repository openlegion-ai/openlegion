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

    def test_remove_agent_jobs(self):
        sched = CronScheduler(config_path=self.config_path)
        sched.add_job(agent="a", schedule="every 1h", message="m1")
        sched.add_job(agent="a", schedule="every 2h", message="m2")
        sched.add_job(agent="b", schedule="every 1h", message="m3")
        assert sched.remove_agent_jobs("a") == 2
        assert len(sched.jobs) == 1
        assert list(sched.jobs.values())[0].agent == "b"

    def test_remove_agent_jobs_none(self):
        sched = CronScheduler(config_path=self.config_path)
        sched.add_job(agent="a", schedule="every 1h", message="m1")
        assert sched.remove_agent_jobs("nonexistent") == 0
        assert len(sched.jobs) == 1

    def test_remove_agent_jobs_persists(self):
        sched = CronScheduler(config_path=self.config_path)
        sched.add_job(agent="a", schedule="every 1h", message="m1")
        sched.add_job(agent="b", schedule="every 1h", message="m2")
        sched.remove_agent_jobs("a")
        sched2 = CronScheduler(config_path=self.config_path)
        assert len(sched2.jobs) == 1
        assert list(sched2.jobs.values())[0].agent == "b"


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
    async def test_heartbeat_skips_when_clean_no_context(self):
        """Heartbeat skips dispatch when no context_fn, no probes, default rules."""
        dispatch = AsyncMock(return_value="Checked in")
        mock_bb = MagicMock()
        mock_bb.list_by_prefix.return_value = []
        sched = CronScheduler(
            config_path=self.config_path, dispatch_fn=dispatch, blackboard=mock_bb,
        )
        job = sched.add_job(agent="test", schedule="every 15m", message="heartbeat", heartbeat=True)
        result = await sched._execute_job(job)
        # Skip-LLM optimization: no custom rules, no activity, no probes → skip
        dispatch.assert_not_called()
        assert result is None

    @pytest.mark.asyncio
    async def test_heartbeat_dispatches_with_probe_context(self):
        """Heartbeat includes probe details when probes trigger."""
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
        # Message should mention probes
        call_msg = dispatch.call_args[0][1]
        assert "Probe Alerts" in call_msg

    def test_find_heartbeat_job(self):
        sched = CronScheduler(config_path=self.config_path)
        sched.add_job(agent="researcher", schedule="every 30m", message="check", heartbeat=False)
        sched.add_job(agent="researcher", schedule="every 15m", message="heartbeat", heartbeat=True)
        sched.add_job(agent="analyst", schedule="every 1h", message="heartbeat", heartbeat=True)

        hb = sched.find_heartbeat_job("researcher")
        assert hb is not None
        assert hb.heartbeat is True
        assert hb.agent == "researcher"

        assert sched.find_heartbeat_job("nonexistent") is None

    def test_update_job(self):
        sched = CronScheduler(config_path=self.config_path)
        job = sched.add_job(agent="test", schedule="every 15m", message="ping")
        original_id = job.id

        updated = sched.update_job(job.id, schedule="every 30m", message="pong")
        assert updated is not None
        assert updated.schedule == "every 30m"
        assert updated.message == "pong"
        assert updated.id == original_id  # id should not change

        # Verify persistence
        sched2 = CronScheduler(config_path=self.config_path)
        reloaded = sched2.jobs[original_id]
        assert reloaded.schedule == "every 30m"

    def test_update_job_nonexistent(self):
        sched = CronScheduler(config_path=self.config_path)
        assert sched.update_job("nonexistent", schedule="every 1h") is None

    def test_update_job_cannot_change_id(self):
        sched = CronScheduler(config_path=self.config_path)
        job = sched.add_job(agent="test", schedule="every 15m", message="ping")
        original_id = job.id

        updated = sched.update_job(job.id, id="hacked_id")
        assert updated is not None
        assert updated.id == original_id  # id should not change


class TestEnrichedHeartbeat:
    """Tests for the enriched heartbeat message with context_fn and skip-LLM."""

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.config_path = f"{self._tmpdir}/cron.json"

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_heartbeat_skip_no_rules_no_activity_no_probes(self):
        """Skip dispatch when default rules, no activity, and no probes triggered."""
        dispatch = AsyncMock(return_value="response")
        context_fn = AsyncMock(return_value={
            "heartbeat_rules": "",
            "daily_logs": "",
            "is_default_heartbeat": True,
            "has_recent_activity": False,
        })
        mock_bb = MagicMock()
        mock_bb.list_by_prefix.return_value = []
        sched = CronScheduler(
            config_path=self.config_path, dispatch_fn=dispatch,
            blackboard=mock_bb, context_fn=context_fn,
        )
        job = sched.add_job(
            agent="test", schedule="every 15m", message="heartbeat", heartbeat=True,
        )
        result = await sched._execute_job(job)
        assert result is None
        dispatch.assert_not_called()

    @pytest.mark.asyncio
    async def test_heartbeat_dispatches_custom_rules(self):
        """Message includes HEARTBEAT.md content when custom rules exist."""
        dispatch = AsyncMock(return_value="Done")
        context_fn = AsyncMock(return_value={
            "heartbeat_rules": "# My Rules\n\nCheck email every hour.",
            "daily_logs": "",
            "is_default_heartbeat": False,
            "has_recent_activity": False,
        })
        mock_bb = MagicMock()
        mock_bb.list_by_prefix.return_value = []
        sched = CronScheduler(
            config_path=self.config_path, dispatch_fn=dispatch,
            blackboard=mock_bb, context_fn=context_fn,
        )
        job = sched.add_job(
            agent="test", schedule="every 15m", message="heartbeat", heartbeat=True,
        )
        result = await sched._execute_job(job)
        dispatch.assert_called_once()
        call_msg = dispatch.call_args[0][1]
        assert "Check email every hour" in call_msg
        assert "Your Heartbeat Rules" in call_msg
        assert "Follow your HEARTBEAT.md rules" in call_msg

    @pytest.mark.asyncio
    async def test_heartbeat_dispatches_on_probes(self):
        """Skip optimization bypassed when probes trigger, even with default rules."""
        dispatch = AsyncMock(return_value="Acting")
        context_fn = AsyncMock(return_value={
            "heartbeat_rules": "",
            "daily_logs": "",
            "is_default_heartbeat": True,
            "has_recent_activity": False,
        })
        mock_bb = MagicMock()
        mock_bb.list_by_prefix.side_effect = (
            lambda prefix: [MagicMock()] if "signals" in prefix else []
        )
        sched = CronScheduler(
            config_path=self.config_path, dispatch_fn=dispatch,
            blackboard=mock_bb, context_fn=context_fn,
        )
        job = sched.add_job(
            agent="test", schedule="every 15m", message="heartbeat", heartbeat=True,
        )
        result = await sched._execute_job(job)
        dispatch.assert_called_once()
        call_msg = dispatch.call_args[0][1]
        assert "Probe Alerts" in call_msg

    @pytest.mark.asyncio
    async def test_heartbeat_dispatches_on_activity(self):
        """Skip optimization bypassed when has_recent_activity=True."""
        dispatch = AsyncMock(return_value="Continued")
        context_fn = AsyncMock(return_value={
            "heartbeat_rules": "",
            "daily_logs": "Did some work",
            "is_default_heartbeat": True,
            "has_recent_activity": True,
        })
        mock_bb = MagicMock()
        mock_bb.list_by_prefix.return_value = []
        sched = CronScheduler(
            config_path=self.config_path, dispatch_fn=dispatch,
            blackboard=mock_bb, context_fn=context_fn,
        )
        job = sched.add_job(
            agent="test", schedule="every 15m", message="heartbeat", heartbeat=True,
        )
        result = await sched._execute_job(job)
        dispatch.assert_called_once()
        call_msg = dispatch.call_args[0][1]
        assert "Your Recent Activity" in call_msg
        assert "Did some work" in call_msg

    @pytest.mark.asyncio
    async def test_heartbeat_message_order(self):
        """Agent context sections appear before blackboard sections."""
        dispatch = AsyncMock(return_value="Ok")
        context_fn = AsyncMock(return_value={
            "heartbeat_rules": "# My Rules\n\nDo stuff.",
            "daily_logs": "Some activity",
            "is_default_heartbeat": False,
            "has_recent_activity": True,
        })
        mock_bb = MagicMock()
        mock_bb.list_by_prefix.side_effect = lambda prefix: (
            [MagicMock(key=f"signals/test/s1", value={"msg": "hi"})]
            if "signals" in prefix else []
        )
        sched = CronScheduler(
            config_path=self.config_path, dispatch_fn=dispatch,
            blackboard=mock_bb, context_fn=context_fn,
        )
        job = sched.add_job(
            agent="test", schedule="every 15m", message="heartbeat", heartbeat=True,
        )
        await sched._execute_job(job)
        call_msg = dispatch.call_args[0][1]

        rules_pos = call_msg.index("Your Heartbeat Rules")
        activity_pos = call_msg.index("Your Recent Activity")
        signals_pos = call_msg.index("Pending Signals")
        # Agent context (rules, activity) should come before blackboard (signals)
        assert rules_pos < signals_pos
        assert activity_pos < signals_pos

    @pytest.mark.asyncio
    async def test_heartbeat_pending_details(self):
        """Heartbeat message includes actual pending signal/task values."""
        dispatch = AsyncMock(return_value="Ok")
        context_fn = AsyncMock(return_value={
            "heartbeat_rules": "# Rules\n\nCheck tasks.",
            "daily_logs": "",
            "is_default_heartbeat": False,
            "has_recent_activity": False,
        })
        mock_signal = MagicMock()
        mock_signal.key = "signals/test/alert1"
        mock_signal.value = {"message": "urgent update needed"}
        mock_task = MagicMock()
        mock_task.key = "tasks/test/todo1"
        mock_task.value = {"action": "review PR #42"}
        mock_bb = MagicMock()
        mock_bb.list_by_prefix.side_effect = lambda prefix: (
            [mock_signal] if "signals" in prefix else
            [mock_task] if "tasks" in prefix else []
        )
        sched = CronScheduler(
            config_path=self.config_path, dispatch_fn=dispatch,
            blackboard=mock_bb, context_fn=context_fn,
        )
        job = sched.add_job(
            agent="test", schedule="every 15m", message="heartbeat", heartbeat=True,
        )
        await sched._execute_job(job)
        call_msg = dispatch.call_args[0][1]
        assert "urgent update needed" in call_msg
        assert "review PR #42" in call_msg

    @pytest.mark.asyncio
    async def test_heartbeat_context_fn_failure_graceful(self):
        """Dispatch still happens when context_fn raises."""
        dispatch = AsyncMock(return_value="Ok")
        context_fn = AsyncMock(side_effect=RuntimeError("transport down"))
        mock_bb = MagicMock()
        mock_bb.list_by_prefix.return_value = []
        sched = CronScheduler(
            config_path=self.config_path, dispatch_fn=dispatch,
            blackboard=mock_bb, context_fn=context_fn,
        )
        job = sched.add_job(
            agent="test", schedule="every 15m", message="heartbeat", heartbeat=True,
        )
        # context_fn fails → ctx={} → is_default=True, has_activity=False
        # but no probes triggered either → would skip…
        # Actually, since ctx is empty, is_default=True, has_activity=False,
        # and with empty blackboard → no probes triggered → SKIP.
        # That's the correct behavior when we have no context and no probes.
        # To test graceful fallback, we need at least one probe to trigger.
        mock_bb.list_by_prefix.side_effect = (
            lambda prefix: [MagicMock()] if "signals" in prefix else []
        )
        result = await sched._execute_job(job)
        dispatch.assert_called_once()
        # Should still dispatch despite context_fn failure

    @pytest.mark.asyncio
    async def test_heartbeat_includes_operating_rules(self):
        """Heartbeat messages always include non-negotiable operating rules."""
        dispatch = AsyncMock(return_value="Ok")
        context_fn = AsyncMock(return_value={
            "heartbeat_rules": "# My Rules\n\nDo stuff.",
            "daily_logs": "Some activity",
            "is_default_heartbeat": False,
            "has_recent_activity": True,
        })
        mock_bb = MagicMock()
        mock_bb.list_by_prefix.return_value = []
        sched = CronScheduler(
            config_path=self.config_path, dispatch_fn=dispatch,
            blackboard=mock_bb, context_fn=context_fn,
        )
        job = sched.add_job(
            agent="test", schedule="every 15m", message="heartbeat", heartbeat=True,
        )
        await sched._execute_job(job)
        call_msg = dispatch.call_args[0][1]
        # Non-negotiable rules must be present
        assert "Operating Rules" in call_msg
        assert "ECONOMICAL" in call_msg
        assert "notify_user" in call_msg
        assert "HEARTBEAT_OK" in call_msg
        assert "Do NOT change your heartbeat schedule" in call_msg
        # Operating rules should appear before agent's custom rules
        rules_pos = call_msg.index("Operating Rules")
        custom_pos = call_msg.index("Your Heartbeat Rules")
        assert rules_pos < custom_pos

