"""Tests for CronScheduler: scheduling, intervals, dispatch, persistence, heartbeat."""

from __future__ import annotations

import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.host.cron import CronJob, CronScheduler, _match_cron_field


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

    # ── H8: malformed fields must not raise (poison-job containment) ──

    def test_step_zero_does_not_divide(self):
        """``*/0`` would ZeroDivisionError on ``current % step`` — guard it."""
        # Must return False (no match) rather than raise.
        assert _match_cron_field("*/0", 0) is False
        assert _match_cron_field("*/0", 5) is False

    def test_negative_step_safe(self):
        assert _match_cron_field("*/-1", 0) is False

    def test_malformed_range_safe(self):
        """``1-`` (missing end) must not raise."""
        assert _match_cron_field("1-", 3) is False

    def test_non_numeric_field_safe(self):
        assert _match_cron_field("abc", 3) is False
        assert _match_cron_field("1,foo,5", 3) is False
        # A valid segment alongside a junk one still matches the valid one.
        assert _match_cron_field("foo,5", 5) is True


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

    @pytest.mark.asyncio
    async def test_pause_resume(self):
        sched = CronScheduler(config_path=self.config_path)
        job = sched.add_job(agent="test", schedule="every 1h", message="ping")
        assert await sched.pause_job(job.id)
        assert sched.jobs[job.id].enabled is False
        assert await sched.resume_job(job.id)
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
        assert sched._is_due(job, datetime.now(timezone.utc))

    def test_interval_not_due_too_soon(self):
        sched = CronScheduler(config_path=self.config_path)
        job = sched.add_job(agent="test", schedule="every 30m", message="ping")
        job.last_run = datetime.now(timezone.utc).isoformat()
        assert not sched._is_due(job, datetime.now(timezone.utc))


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

    # ── Task 2b: cron_dispatch wrapper stamps kind="cron" origin ────

    @pytest.mark.asyncio
    async def test_cron_dispatch_wrapper_stamps_typed_origin(self):
        """``RuntimeContext._create_cron_scheduler``'s ``cron_dispatch``
        wrapper must pass a typed ``MessageOrigin(kind="cron")`` through
        to ``async_dispatch`` so the lane sees a typed origin instead
        of ``None``.
        """
        from src.cli.runtime import RuntimeContext
        from src.shared.types import MessageOrigin

        # Synthesize a minimal RuntimeContext with stub dependencies.
        ctx = RuntimeContext.__new__(RuntimeContext)
        captured = {}

        async def _fake_async_dispatch(agent, message, **kwargs):
            captured["agent"] = agent
            captured["message"] = message
            captured.update(kwargs)
            return "ok"

        ctx.async_dispatch = _fake_async_dispatch
        ctx.transport = MagicMock()
        ctx.blackboard = None
        ctx.trace_store = None
        ctx.event_bus = None
        ctx.cfg = {}
        # Fix 5 (seam follow-up): RuntimeContext now passes
        # ``health_monitor`` to CronScheduler so the quarantine skip
        # works. Synthetic contexts must provide the attr (None is OK
        # — disables the skip).
        ctx.health_monitor = None

        ctx._create_cron_scheduler()
        # The cron_dispatch closure was registered as ``dispatch_fn``.
        await ctx.cron_scheduler.dispatch_fn("agent1", "tick")
        origin = captured.get("origin")
        assert isinstance(origin, MessageOrigin)
        assert origin.kind == "cron"
        assert origin.channel == "cron"
        assert origin.user == ""


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
        # Mock probes to return nothing triggered (isolate from host disk usage)
        with patch.object(sched, "_run_heartbeat_probes", return_value=[]):
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

    def test_ensure_heartbeat_creates_when_missing(self):
        sched = CronScheduler(config_path=self.config_path)
        job = sched.ensure_heartbeat("researcher", "every 20m")
        assert job.agent == "researcher"
        assert job.heartbeat is True
        assert job.schedule == "every 20m"
        assert len(sched.jobs) == 1

    def test_ensure_heartbeat_returns_existing(self):
        sched = CronScheduler(config_path=self.config_path)
        first = sched.ensure_heartbeat("researcher", "every 20m")
        second = sched.ensure_heartbeat("researcher", "every 1h")
        assert first.id == second.id  # same job, not duplicated
        assert second.schedule == "every 20m"  # schedule unchanged
        assert len(sched.jobs) == 1

    @pytest.mark.asyncio
    async def test_update_job(self):
        sched = CronScheduler(config_path=self.config_path)
        job = sched.add_job(agent="test", schedule="every 15m", message="ping")
        original_id = job.id

        updated = await sched.update_job(job.id, schedule="every 30m", message="pong")
        assert updated is not None
        assert updated.schedule == "every 30m"
        assert updated.message == "pong"
        assert updated.id == original_id  # id should not change

        # Verify persistence
        sched2 = CronScheduler(config_path=self.config_path)
        reloaded = sched2.jobs[original_id]
        assert reloaded.schedule == "every 30m"

    @pytest.mark.asyncio
    async def test_update_job_nonexistent(self):
        sched = CronScheduler(config_path=self.config_path)
        assert await sched.update_job("nonexistent", schedule="every 1h") is None

    @pytest.mark.asyncio
    async def test_update_job_cannot_change_id(self):
        sched = CronScheduler(config_path=self.config_path)
        job = sched.add_job(agent="test", schedule="every 15m", message="ping")
        original_id = job.id

        updated = await sched.update_job(job.id, id="hacked_id")
        assert updated is not None
        assert updated.id == original_id  # id should not change


class TestCronConcurrentUpdate:
    """Tests for per-job locking under concurrent mutation."""

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.config_path = f"{self._tmpdir}/cron.json"

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_concurrent_update_during_execution(self):
        """update_job during _execute_job: both mutations persist."""
        import asyncio

        async def slow_dispatch(agent, msg):
            await asyncio.sleep(0.1)

        dispatch = AsyncMock(side_effect=slow_dispatch)
        sched = CronScheduler(config_path=self.config_path, dispatch_fn=dispatch)
        job = sched.add_job(agent="test", schedule="every 1h", message="original")

        async def run_execute():
            await sched._execute_job(job)

        async def run_update():
            # Small delay so _execute_job starts first
            await asyncio.sleep(0.05)
            await sched.update_job(job.id, message="updated")

        await asyncio.gather(run_execute(), run_update())

        # Both mutations should have persisted
        assert sched.jobs[job.id].run_count == 1  # from _execute_job
        assert sched.jobs[job.id].message == "updated"  # from update_job

    @pytest.mark.asyncio
    async def test_concurrent_pause_during_execution(self):
        """pause_job during _execute_job: both mutations persist."""
        import asyncio

        async def slow_dispatch(agent, msg):
            await asyncio.sleep(0.1)

        dispatch = AsyncMock(side_effect=slow_dispatch)
        sched = CronScheduler(config_path=self.config_path, dispatch_fn=dispatch)
        job = sched.add_job(agent="test", schedule="every 1h", message="ping")

        async def run_execute():
            await sched._execute_job(job)

        async def run_pause():
            await asyncio.sleep(0.05)
            await sched.pause_job(job.id)

        await asyncio.gather(run_execute(), run_pause())

        assert sched.jobs[job.id].run_count == 1
        assert sched.jobs[job.id].enabled is False


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
        # Mock probes to return nothing triggered (isolate from host disk usage)
        with patch.object(sched, "_run_heartbeat_probes", return_value=[]):
            result = await sched._execute_job(job)
        assert result is None
        dispatch.assert_not_called()

    @pytest.mark.asyncio
    async def test_heartbeat_manual_trigger_bypasses_skip(self):
        """Manual trigger should always dispatch, even with default rules and no activity."""
        dispatch = AsyncMock(return_value="manual response")
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
        with patch.object(sched, "_run_heartbeat_probes", return_value=[]):
            result = await sched.run_job(job.id)
        assert result == "manual response"
        dispatch.assert_called_once()
        call_msg = dispatch.call_args[0][1]
        assert "Heartbeat for test" in call_msg
        assert "Heartbeat Operating Rules" in call_msg
        # PR 972 Codex r3 — the trigger line carries a current ISO
        # timestamp so the LLM has a concrete "now" for date math
        # (e.g. the 7-day re-ask throttle on goal seeding).
        import re
        # ISO 8601 with timezone offset, e.g. 2026-05-29T01:23:45.678901+00:00
        assert re.search(
            r"Heartbeat for test at \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",
            call_msg,
        ), f"missing ISO timestamp in heartbeat trigger: {call_msg[:120]}"

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
        await sched._execute_job(job)
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
        await sched._execute_job(job)
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
        await sched._execute_job(job)
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
            [MagicMock(key="signals/test/s1", value={"msg": "hi"})]
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
        await sched._execute_job(job)
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


class TestHeartbeatDispatchFn:
    """Tests for dedicated heartbeat_dispatch_fn path."""

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.config_path = f"{self._tmpdir}/cron.json"

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_heartbeat_uses_dispatch_fn(self):
        """When heartbeat_dispatch_fn is set, heartbeats use it instead of dispatch_fn."""
        dispatch = AsyncMock(return_value="should not be called")
        hb_dispatch = AsyncMock(return_value={
            "response": "All clear",
            "summary": "Nothing to do",
            "tools_used": [],
            "duration_ms": 500,
            "tokens_used": 100,
            "outcome": "ok",
            "skipped": False,
        })
        context_fn = AsyncMock(return_value={
            "heartbeat_rules": "# Rules\nDo stuff",
            "is_default_heartbeat": False,
            "has_recent_activity": True,
        })
        sched = CronScheduler(
            config_path=self.config_path,
            dispatch_fn=dispatch,
            heartbeat_dispatch_fn=hb_dispatch,
            context_fn=context_fn,
        )
        job = sched.add_job(
            agent="test", schedule="every 15m", message="heartbeat", heartbeat=True,
        )
        await sched._execute_job(job)

        hb_dispatch.assert_called_once()
        dispatch.assert_not_called()

    @pytest.mark.asyncio
    async def test_heartbeat_skipped_returns_none(self):
        """When agent is busy, heartbeat_dispatch_fn returns skipped and cron returns None."""
        hb_dispatch = AsyncMock(return_value={
            "skipped": True,
            "reason": "agent_busy",
        })
        context_fn = AsyncMock(return_value={
            "heartbeat_rules": "# Rules",
            "is_default_heartbeat": False,
            "has_recent_activity": True,
        })
        sched = CronScheduler(
            config_path=self.config_path,
            dispatch_fn=AsyncMock(),
            heartbeat_dispatch_fn=hb_dispatch,
            context_fn=context_fn,
        )
        job = sched.add_job(
            agent="test", schedule="every 15m", message="heartbeat", heartbeat=True,
        )
        result = await sched._execute_job(job)
        assert result is None

    @pytest.mark.asyncio
    async def test_heartbeat_prompt_uses_silent_ok_rule(self):
        """Heartbeat prompt should not force user notifications for routine OK states."""
        dispatch = AsyncMock(return_value="should not be called")
        hb_dispatch = AsyncMock(return_value={
            "response": "HEARTBEAT_OK",
            "summary": "Nothing to do",
            "tools_used": [],
            "duration_ms": 100,
            "tokens_used": 10,
            "outcome": "ok",
            "skipped": False,
        })
        context_fn = AsyncMock(return_value={
            "heartbeat_rules": "# Rules\nStay quiet on no-op heartbeats",
            "is_default_heartbeat": False,
            "has_recent_activity": True,
        })
        sched = CronScheduler(
            config_path=self.config_path,
            dispatch_fn=dispatch,
            heartbeat_dispatch_fn=hb_dispatch,
            context_fn=context_fn,
        )
        job = sched.add_job(
            agent="test", schedule="every 15m", message="heartbeat", heartbeat=True,
        )
        await sched._execute_job(job)

        msg = hb_dispatch.call_args.args[1]
        assert "stay silent and return HEARTBEAT_OK" in msg
        assert "Report what you worked on to the USER via notify_user" not in msg

    @pytest.mark.asyncio
    async def test_heartbeat_emits_event(self):
        """heartbeat_complete event is emitted with structured data."""
        hb_dispatch = AsyncMock(return_value={
            "response": "Done",
            "summary": "Checked alerts",
            "tools_used": ["http_request"],
            "duration_ms": 2000,
            "tokens_used": 300,
            "outcome": "ok",
            "skipped": False,
        })
        context_fn = AsyncMock(return_value={
            "is_default_heartbeat": False,
            "has_recent_activity": True,
        })
        event_bus = MagicMock()
        sched = CronScheduler(
            config_path=self.config_path,
            dispatch_fn=AsyncMock(),
            heartbeat_dispatch_fn=hb_dispatch,
            context_fn=context_fn,
            event_bus=event_bus,
        )
        job = sched.add_job(
            agent="test", schedule="every 15m", message="heartbeat", heartbeat=True,
        )
        event_bus.reset_mock()
        await sched._execute_job(job)

        # Should emit cron_change(started) + heartbeat_complete + cron_change(executed)
        hb_calls = [c for c in event_bus.emit.call_args_list if c[0][0] == "heartbeat_complete"]
        assert len(hb_calls) == 1
        call_args = hb_calls[0]
        assert call_args[1]["agent"] == "test"
        assert call_args[1]["data"]["summary"] == "Checked alerts"
        assert call_args[1]["data"]["outcome"] == "ok"
        # cron_change should fire twice: started + executed
        cron_calls = [c for c in event_bus.emit.call_args_list if c[0][0] == "cron_change"]
        assert len(cron_calls) == 2
        actions = [c[1]["data"]["action"] for c in cron_calls]
        assert actions == ["started", "executed"]

    @pytest.mark.asyncio
    async def test_heartbeat_falls_back_to_dispatch_fn(self):
        """Without heartbeat_dispatch_fn, heartbeats use dispatch_fn."""
        dispatch = AsyncMock(return_value="Handled")
        context_fn = AsyncMock(return_value={
            "heartbeat_rules": "# Rules\nStuff",
            "is_default_heartbeat": False,
            "has_recent_activity": True,
        })
        sched = CronScheduler(
            config_path=self.config_path,
            dispatch_fn=dispatch,
            context_fn=context_fn,
        )
        job = sched.add_job(
            agent="test", schedule="every 15m", message="heartbeat", heartbeat=True,
        )
        await sched._execute_job(job)
        dispatch.assert_called_once()


class TestToolInvoke:
    """Tests for tool-type cron jobs — direct tool execution without LLM."""

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.config_path = f"{self._tmpdir}/cron.json"

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_tool_job_calls_invoke_fn_not_dispatch(self):
        invoke = AsyncMock(return_value={"result": 42})
        dispatch = AsyncMock(return_value="should not be called")
        sched = CronScheduler(
            config_path=self.config_path, dispatch_fn=dispatch, invoke_fn=invoke,
        )
        job = sched.add_job(agent="test", schedule="every 1m", tool_name="random_number")

        result = await sched._execute_job(job)
        invoke.assert_called_once_with("test", "random_number", {})
        dispatch.assert_not_called()
        assert job.run_count == 1
        assert result is not None

    @pytest.mark.asyncio
    async def test_tool_job_passes_parsed_params(self):
        invoke = AsyncMock(return_value={"sent": True})
        sched = CronScheduler(config_path=self.config_path, invoke_fn=invoke)
        job = sched.add_job(
            agent="test", schedule="every 1m",
            tool_name="notify_user", tool_params='{"message": "hello"}',
        )

        await sched._execute_job(job)
        invoke.assert_called_once_with("test", "notify_user", {"message": "hello"})

    @pytest.mark.asyncio
    async def test_tool_job_invalid_params_json_falls_back_to_empty_dict(self):
        invoke = AsyncMock(return_value={"sent": True})
        sched = CronScheduler(config_path=self.config_path, invoke_fn=invoke)
        job = sched.add_job(
            agent="test", schedule="every 1m",
            tool_name="notify_user", tool_params="not-valid-json",
        )

        await sched._execute_job(job)
        invoke.assert_called_once_with("test", "notify_user", {})

    @pytest.mark.asyncio
    async def test_tool_job_no_invoke_fn_returns_none_does_not_dispatch(self):
        """tool_name set but invoke_fn is None → skip cleanly, never dispatch."""
        dispatch = AsyncMock(return_value="should not be called")
        sched = CronScheduler(
            config_path=self.config_path, dispatch_fn=dispatch, invoke_fn=None,
        )
        job = sched.add_job(agent="test", schedule="every 1m", tool_name="some_tool")

        result = await sched._execute_job(job)
        assert result is None
        dispatch.assert_not_called()
        assert job.run_count == 1

    @pytest.mark.asyncio
    async def test_tool_job_invoke_error_increments_error_count(self):
        invoke = AsyncMock(side_effect=RuntimeError("tool failed"))
        sched = CronScheduler(config_path=self.config_path, invoke_fn=invoke)
        job = sched.add_job(agent="test", schedule="every 1m", tool_name="broken_tool")

        result = await sched._execute_job(job)
        assert result is None
        assert job.error_count == 1

    def test_tool_job_persists_and_reloads(self):
        sched = CronScheduler(config_path=self.config_path)
        sched.add_job(
            agent="test", schedule="every 5m",
            tool_name="notify_user", tool_params='{"message": "ping"}',
        )

        sched2 = CronScheduler(config_path=self.config_path)
        loaded = list(sched2.jobs.values())[0]
        assert loaded.tool_name == "notify_user"
        assert loaded.tool_params == '{"message": "ping"}'
        assert loaded.message == ""

    @pytest.mark.asyncio
    async def test_update_job_tool_params(self):
        sched = CronScheduler(config_path=self.config_path)
        job = sched.add_job(
            agent="test", schedule="every 5m",
            tool_name="notify_user", tool_params='{"message": "old"}',
        )

        updated = await sched.update_job(job.id, tool_params='{"message": "new"}')
        assert updated is not None
        assert updated.tool_params == '{"message": "new"}'

        sched2 = CronScheduler(config_path=self.config_path)
        assert list(sched2.jobs.values())[0].tool_params == '{"message": "new"}'


class TestComputeNextRun:
    """Tests for CronScheduler._compute_next_run."""

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.config_path = f"{self._tmpdir}/cron.json"

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_interval_with_last_run(self):
        """next_run = last_run + interval for interval schedules."""
        sched = CronScheduler(config_path=self.config_path)
        last = datetime(2026, 3, 9, 10, 0, 0, tzinfo=timezone.utc)
        job = CronJob(
            id="cron_test", agent="a", schedule="every 15m",
            message="ping", last_run=last.isoformat(),
        )
        sched._compute_next_run(job)
        expected = (last + timedelta(minutes=15)).isoformat()
        assert job.next_run == expected

    def test_interval_no_last_run(self):
        """next_run = now when no last_run (job is immediately due)."""
        sched = CronScheduler(config_path=self.config_path)
        job = CronJob(
            id="cron_test", agent="a", schedule="every 1h", message="ping",
        )
        before = datetime.now(timezone.utc)
        sched._compute_next_run(job)
        after = datetime.now(timezone.utc)
        assert job.next_run is not None
        next_dt = datetime.fromisoformat(job.next_run)
        assert before <= next_dt <= after

    def test_cron_expression(self):
        """Cron expression computes a future next_run."""
        sched = CronScheduler(config_path=self.config_path)
        job = CronJob(
            id="cron_test", agent="a", schedule="0 9 * * *", message="daily",
        )
        sched._compute_next_run(job)
        assert job.next_run is not None
        next_dt = datetime.fromisoformat(job.next_run)
        now = datetime.now(timezone.utc)
        assert next_dt > now
        assert next_dt.minute == 0
        assert next_dt.hour == 9

    def test_add_job_sets_next_run(self):
        """add_job automatically computes next_run."""
        sched = CronScheduler(config_path=self.config_path)
        job = sched.add_job(agent="a", schedule="every 30m", message="ping")
        assert job.next_run is not None

    def test_load_computes_next_run(self):
        """Loading from disk recomputes next_run for all jobs."""
        sched = CronScheduler(config_path=self.config_path)
        job = sched.add_job(agent="a", schedule="every 10m", message="ping")
        assert job.next_run is not None  # sanity check before reload

        # Reload — should recompute
        sched2 = CronScheduler(config_path=self.config_path)
        loaded_job = list(sched2.jobs.values())[0]
        assert loaded_job.next_run is not None

    @pytest.mark.asyncio
    async def test_execute_updates_next_run(self):
        """After execution, next_run advances past now."""
        dispatch = AsyncMock(return_value="ok")
        sched = CronScheduler(
            config_path=self.config_path, dispatch_fn=dispatch,
        )
        job = sched.add_job(agent="a", schedule="every 5m", message="ping")
        await sched._execute_job(job)
        assert job.next_run is not None
        next_dt = datetime.fromisoformat(job.next_run)
        now = datetime.now(timezone.utc)
        assert next_dt > now

    @pytest.mark.asyncio
    async def test_resume_updates_next_run(self):
        """Resuming a paused job recomputes next_run."""
        sched = CronScheduler(config_path=self.config_path)
        job = sched.add_job(agent="a", schedule="every 20m", message="ping")
        await sched.pause_job(job.id)
        await sched.resume_job(job.id)
        assert sched.jobs[job.id].next_run is not None


class TestCronForceLlm:
    """Bug 4 (force_llm) — pipeline-kicker agents have no probes and an
    empty HEARTBEAT.md, so the standard skip-LLM check would never wake
    them. ``force_llm=True`` bypasses the skip so the LLM runs every tick."""

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.config_path = f"{self._tmpdir}/cron.json"

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_force_llm_dispatches_through_all_skip_conditions(self):
        """All four skip conditions hold but force_llm=True → dispatch fires."""
        dispatch = AsyncMock(return_value="kicked the pipeline")
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
            agent="kicker", schedule="every 15m",
            message="heartbeat", heartbeat=True,
            force_llm=True,
        )
        with patch.object(sched, "_run_heartbeat_probes", return_value=[]):
            result = await sched._execute_job(job)
        # Even with all skip conditions met, dispatch fires.
        dispatch.assert_called_once()
        assert result is not None

    @pytest.mark.asyncio
    async def test_force_llm_default_unset_keeps_skipping(self):
        """Without force_llm the skip-LLM optimization runs unchanged."""
        dispatch = AsyncMock(return_value="should not see this")
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
        # No force_llm — default False.
        job = sched.add_job(
            agent="regular", schedule="every 15m",
            message="heartbeat", heartbeat=True,
        )
        assert job.force_llm is False
        with patch.object(sched, "_run_heartbeat_probes", return_value=[]):
            result = await sched._execute_job(job)
        assert result is None
        dispatch.assert_not_called()

    def test_force_llm_persists_across_save_load_cycle(self):
        """``CronJob.force_llm`` round-trips through the JSON config file."""
        sched = CronScheduler(config_path=self.config_path)
        sched.add_job(
            agent="kicker", schedule="every 1h",
            message="heartbeat", heartbeat=True,
            force_llm=True,
        )

        # Reload from disk — confirm force_llm survived.
        sched2 = CronScheduler(config_path=self.config_path)
        loaded = list(sched2.jobs.values())[0]
        assert loaded.force_llm is True

    @pytest.mark.asyncio
    async def test_force_llm_updatable_via_update_job(self):
        """``_UPDATABLE_FIELDS`` allows toggling force_llm at runtime."""
        sched = CronScheduler(config_path=self.config_path)
        job = sched.add_job(
            agent="kicker", schedule="every 1h",
            message="heartbeat", heartbeat=True,
        )
        assert job.force_llm is False

        updated = await sched.update_job(job.id, force_llm=True)
        assert updated is not None
        assert updated.force_llm is True

        # And toggling it back off works too.
        updated = await sched.update_job(job.id, force_llm=False)
        assert updated.force_llm is False

    @pytest.mark.asyncio
    async def test_force_llm_propagates_to_heartbeat_dispatch_fn(self):
        """Bug 6 (codex P2 r2): force_llm must flow past cron into the dispatch.

        The cron-side skip is only half the path — the agent's
        ``execute_heartbeat`` runs its own ``no_heartbeat_rules`` skip on
        empty HEARTBEAT.md. If ``force_llm`` doesn't reach the dispatch
        callable, the loop-side skip still silences pipeline-kicker
        agents. Pin the propagation so a future refactor can't drop it.
        """
        hb_dispatch = AsyncMock(return_value={
            "response": "kicked", "summary": "", "tools_used": [],
            "duration_ms": 10, "tokens_used": 5, "outcome": "ok",
            "skipped": False,
        })
        context_fn = AsyncMock(return_value={
            "heartbeat_rules": "",
            "daily_logs": "",
            "is_default_heartbeat": True,
            "has_recent_activity": False,
        })
        mock_bb = MagicMock()
        mock_bb.list_by_prefix.return_value = []
        sched = CronScheduler(
            config_path=self.config_path, dispatch_fn=AsyncMock(),
            heartbeat_dispatch_fn=hb_dispatch,
            blackboard=mock_bb, context_fn=context_fn,
        )
        job = sched.add_job(
            agent="kicker", schedule="every 15m",
            message="heartbeat", heartbeat=True, force_llm=True,
        )
        with patch.object(sched, "_run_heartbeat_probes", return_value=[]):
            await sched._execute_job(job)
        hb_dispatch.assert_called_once()
        # force_llm=True must reach the dispatcher as a kwarg so it can
        # set the x-force-llm header on the /heartbeat request.
        assert hb_dispatch.call_args.kwargs.get("force_llm") is True

    @pytest.mark.asyncio
    async def test_force_llm_default_does_not_propagate_true(self):
        """Default force_llm=False stays False through the dispatch."""
        hb_dispatch = AsyncMock(return_value={
            "response": "ok", "summary": "", "tools_used": [],
            "duration_ms": 10, "tokens_used": 5, "outcome": "ok",
            "skipped": False,
        })
        context_fn = AsyncMock(return_value={
            "heartbeat_rules": "# Rules\nstuff",
            "is_default_heartbeat": False,
            "has_recent_activity": True,
        })
        sched = CronScheduler(
            config_path=self.config_path, dispatch_fn=AsyncMock(),
            heartbeat_dispatch_fn=hb_dispatch, context_fn=context_fn,
        )
        job = sched.add_job(
            agent="regular", schedule="every 15m",
            message="heartbeat", heartbeat=True,
        )
        await sched._execute_job(job)
        hb_dispatch.assert_called_once()
        assert hb_dispatch.call_args.kwargs.get("force_llm") is False


# ── Seam follow-up Fix 5: cron skips dispatch to quarantined agent ──


class TestCronQuarantineGate:
    """CronScheduler skips dispatch when the target agent is quarantined.

    The heartbeat cron tick is the cheapest place to bail — skip is silent
    (debug-level) because quarantine itself already surfaces a dashboard
    notification via HealthMonitor.
    """

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.config_path = f"{self._tmpdir}/cron.json"

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_cron_skips_quarantined_agent(self):
        dispatch = AsyncMock(return_value="ok")
        health_monitor = MagicMock()
        health_monitor.is_quarantined = MagicMock(return_value=True)
        sched = CronScheduler(
            config_path=self.config_path,
            dispatch_fn=dispatch,
            health_monitor=health_monitor,
        )
        job = sched.add_job(
            agent="quarantined-agent", schedule="every 15m",
            message="ping",
        )
        result = await sched._execute_job(job)
        assert result is None
        dispatch.assert_not_called()

    @pytest.mark.asyncio
    async def test_cron_dispatches_to_healthy_agent(self):
        dispatch = AsyncMock(return_value="ok")
        health_monitor = MagicMock()
        health_monitor.is_quarantined = MagicMock(return_value=False)
        sched = CronScheduler(
            config_path=self.config_path,
            dispatch_fn=dispatch,
            health_monitor=health_monitor,
        )
        job = sched.add_job(
            agent="healthy", schedule="every 15m", message="ping",
        )
        await sched._execute_job(job)
        dispatch.assert_called_once()

    @pytest.mark.asyncio
    async def test_cron_set_health_monitor_setter(self):
        dispatch = AsyncMock(return_value="ok")
        sched = CronScheduler(
            config_path=self.config_path,
            dispatch_fn=dispatch,
        )
        # No health monitor → no skip.
        job = sched.add_job(
            agent="agent-a", schedule="every 15m", message="ping",
        )
        await sched._execute_job(job)
        dispatch.assert_called_once()
        # Wire after construction.
        hm = MagicMock()
        hm.is_quarantined = MagicMock(return_value=True)
        sched.set_health_monitor(hm)
        dispatch.reset_mock()
        await sched._execute_job(job)
        dispatch.assert_not_called()


# ── H8 / M11: scheduler resilience + per-agent cap ──────────────────


class TestScheduleValidation:
    """Content-aware ``_validate_schedule`` (H8). Both the create and the
    update mesh endpoints surface this as HTTP 400 — create via
    ``add_job``'s ValueError, update via the direct ``_validate_schedule``
    call before ``update_job``."""

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.config_path = f"{self._tmpdir}/cron.json"

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_step_zero_rejected(self):
        sched = CronScheduler(config_path=self.config_path)
        # The error string is what the update endpoint turns into a 400.
        assert sched._validate_schedule("*/0 * * * *") is not None

    def test_step_zero_rejected_by_create(self):
        """add_job raises ValueError → create endpoint maps to 400."""
        sched = CronScheduler(config_path=self.config_path)
        with pytest.raises(ValueError):
            sched.add_job(agent="a", schedule="*/0 * * * *", message="x")
        assert len(sched.jobs) == 0

    @pytest.mark.asyncio
    async def test_step_zero_rejected_by_update(self):
        """update_job validate-before-mutate: poison value never persists."""
        sched = CronScheduler(config_path=self.config_path)
        job = sched.add_job(agent="a", schedule="0 9 * * *", message="x")
        # Endpoint validates first; assert _validate_schedule flags it.
        assert sched._validate_schedule("*/0 * * * *") is not None
        # And update_job itself refuses to persist the poison schedule.
        with pytest.raises(ValueError):
            await sched.update_job(job.id, schedule="*/0 * * * *")
        # Original schedule untouched.
        assert sched.jobs[job.id].schedule == "0 9 * * *"

    def test_malformed_range_rejected(self):
        sched = CronScheduler(config_path=self.config_path)
        assert sched._validate_schedule("1- * * * *") is not None
        assert sched._validate_schedule("5-2 * * * *") is not None  # start>end

    def test_out_of_range_rejected(self):
        sched = CronScheduler(config_path=self.config_path)
        assert sched._validate_schedule("99 * * * *") is not None  # minute>59
        assert sched._validate_schedule("* 25 * * *") is not None  # hour>23
        assert sched._validate_schedule("* * 0 * *") is not None   # dom<1

    def test_non_numeric_rejected(self):
        sched = CronScheduler(config_path=self.config_path)
        assert sched._validate_schedule("foo * * * *") is not None

    def test_valid_schedules_accepted(self):
        sched = CronScheduler(config_path=self.config_path)
        for good in [
            "0 9 * * *", "*/5 * * * *", "0 9 * * 1-5",
            "15,45 * * * *", "* * * * *", "0 0 1 1 *",
            "every 5m", "every 30s", "every 2h", "every 1d",
        ]:
            assert sched._validate_schedule(good) is None, good

    def test_valid_step_and_interval_still_work(self):
        """A normal cron and an ``every 5m`` job add + compute next_run."""
        sched = CronScheduler(config_path=self.config_path)
        cron_job = sched.add_job(agent="a", schedule="*/5 * * * *", message="x")
        assert cron_job.next_run is not None
        interval_job = sched.add_job(agent="b", schedule="every 5m", message="y")
        assert interval_job.next_run is not None


class TestSchedulerResilience:
    """H8: a poison schedule forced into a live job must NOT crash the
    scheduler. ``_tick`` wraps each job; ``_is_due``/``_match_cron`` no
    longer raise."""

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.config_path = f"{self._tmpdir}/cron.json"

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_is_due_does_not_raise_on_poison_schedule(self):
        """Forcing ``*/0 * * * *`` onto a job: _is_due returns False, no raise."""
        sched = CronScheduler(config_path=self.config_path)
        job = sched.add_job(agent="a", schedule="0 9 * * *", message="x")
        # Bypass validation by mutating the live job directly (simulates a
        # poison value loaded from an old/corrupt config file).
        job.schedule = "*/0 * * * *"
        job.last_run = None
        # Force the once-per-minute window to be open.
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        # Should not raise ZeroDivisionError.
        assert sched._is_due(job, now) is False

    @pytest.mark.asyncio
    async def test_tick_survives_poison_job(self):
        """A poison job in _tick must not kill the scheduler — the other
        (healthy) job must still be scheduled."""
        dispatch = AsyncMock(return_value="ok")
        sched = CronScheduler(config_path=self.config_path, dispatch_fn=dispatch)
        poison = sched.add_job(agent="a", schedule="0 9 * * *", message="x")
        healthy = sched.add_job(agent="b", schedule="every 1s", message="y")
        # Poison the first job post-validation.
        poison.schedule = "*/0 * * * *"
        poison.last_run = None
        # _tick must not raise even with the poison job present.
        await sched._tick()
        # Healthy interval job (no last_run) is due → an execute task fired.
        # Give the created task a chance to run.
        import asyncio
        await asyncio.sleep(0)
        # Sanity: scheduler is still usable after the poison tick.
        assert sched.jobs[healthy.id].enabled is True

    @pytest.mark.asyncio
    async def test_tick_continues_when_is_due_raises(self):
        """If _is_due itself raises (defensive), _tick logs + continues."""
        dispatch = AsyncMock(return_value="ok")
        sched = CronScheduler(config_path=self.config_path, dispatch_fn=dispatch)
        sched.add_job(agent="a", schedule="0 9 * * *", message="x")
        with patch.object(
            sched, "_is_due", side_effect=RuntimeError("boom"),
        ):
            # Must swallow the raise rather than propagate out of the loop.
            await sched._tick()


class TestPerAgentCronCap:
    """M11: per-agent cron-job cap, env-overridable."""

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.config_path = f"{self._tmpdir}/cron.json"

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_cap_enforced(self, monkeypatch):
        monkeypatch.setenv("OPENLEGION_MAX_CRON_JOBS_PER_AGENT", "3")
        sched = CronScheduler(config_path=self.config_path)
        for i in range(3):
            sched.add_job(agent="a", schedule="every 1h", message=f"m{i}")
        with pytest.raises(ValueError, match="cron job limit"):
            sched.add_job(agent="a", schedule="every 1h", message="overflow")
        assert sum(1 for j in sched.jobs.values() if j.agent == "a") == 3

    def test_cap_is_per_agent(self, monkeypatch):
        monkeypatch.setenv("OPENLEGION_MAX_CRON_JOBS_PER_AGENT", "2")
        sched = CronScheduler(config_path=self.config_path)
        sched.add_job(agent="a", schedule="every 1h", message="m1")
        sched.add_job(agent="a", schedule="every 1h", message="m2")
        # Different agent has its own budget.
        b = sched.add_job(agent="b", schedule="every 1h", message="m1")
        assert b.agent == "b"

    def test_default_cap(self):
        from src.host.cron import (
            _DEFAULT_MAX_CRON_JOBS_PER_AGENT,
            _max_cron_jobs_per_agent,
        )
        assert _max_cron_jobs_per_agent() == _DEFAULT_MAX_CRON_JOBS_PER_AGENT == 50

    def test_invalid_env_falls_back_to_default(self, monkeypatch):
        from src.host.cron import _max_cron_jobs_per_agent
        monkeypatch.setenv("OPENLEGION_MAX_CRON_JOBS_PER_AGENT", "not-a-number")
        assert _max_cron_jobs_per_agent() == 50
        monkeypatch.setenv("OPENLEGION_MAX_CRON_JOBS_PER_AGENT", "0")
        assert _max_cron_jobs_per_agent() == 50

