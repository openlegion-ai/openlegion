"""Tests for ``RuntimeContext._reconcile_heartbeats`` (plan §8 #24 prereq
ii).

Regression guard: unlike its siblings ``_reconcile_work_summary_jobs``
and ``_reconcile_standup_jobs`` (both already skip archived rows), this
reconcile recreated a heartbeat cron job for an archived agent on every
mesh boot — the archive endpoint (``_archive_agent_core``) already
removed that job via ``cron_scheduler.remove_agent_jobs``, so recreating
it fought the archive. Mirrors the ``TestReconcileStandupJobs`` pattern
in ``tests/test_cron_standup.py``.
"""

from __future__ import annotations

import shutil
import tempfile

from src.cli.runtime import RuntimeContext
from src.host.cron import CronScheduler


class TestReconcileHeartbeats:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.config_path = f"{self._tmpdir}/cron.json"

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _stub(self, scheduler, cfg):
        class _Stub:
            cron_scheduler = scheduler

        stub = _Stub()
        stub.cfg = cfg
        return stub

    def test_creates_heartbeat_for_active_agent(self):
        scheduler = CronScheduler(config_path=self.config_path)
        cfg = {"agents": {"worker-1": {"role": "worker", "status": "active"}}}
        RuntimeContext._reconcile_heartbeats(self._stub(scheduler, cfg))
        assert scheduler.find_heartbeat_job("worker-1") is not None

    def test_skips_archived_agent(self):
        scheduler = CronScheduler(config_path=self.config_path)
        cfg = {"agents": {"worker-archived": {"role": "worker", "status": "archived"}}}
        RuntimeContext._reconcile_heartbeats(self._stub(scheduler, cfg))
        assert scheduler.find_heartbeat_job("worker-archived") is None

    def test_mixed_roster_only_active_gets_job(self):
        scheduler = CronScheduler(config_path=self.config_path)
        cfg = {
            "agents": {
                "worker-active": {"role": "worker", "status": "active"},
                "worker-archived": {"role": "worker", "status": "archived"},
            },
        }
        RuntimeContext._reconcile_heartbeats(self._stub(scheduler, cfg))
        assert scheduler.find_heartbeat_job("worker-active") is not None
        assert scheduler.find_heartbeat_job("worker-archived") is None

    def test_legacy_row_missing_status_defaults_active(self):
        scheduler = CronScheduler(config_path=self.config_path)
        cfg = {"agents": {"worker-legacy": {"role": "worker"}}}
        RuntimeContext._reconcile_heartbeats(self._stub(scheduler, cfg))
        assert scheduler.find_heartbeat_job("worker-legacy") is not None

    def test_no_cron_scheduler_is_a_noop(self):
        class _Stub:
            cron_scheduler = None
            cfg = {"agents": {"worker-1": {"status": "active"}}}

        RuntimeContext._reconcile_heartbeats(_Stub())  # must not raise
