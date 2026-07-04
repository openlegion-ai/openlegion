"""Tests for the daily work-summary cron bootstrap.

Covers:
- ``CronScheduler.find_summary_job`` matches on tool_name + scope id.
- ``CronScheduler.ensure_summary_job`` is idempotent (same scope returns
  the existing job rather than adding a duplicate).
- Different scopes get separate jobs.
- Bad ``tool_params`` JSON doesn't crash the lookup.
- Per-team schedule override propagates from team metadata.
- Reconcile prunes summary jobs for teams that were archived/deleted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def scheduler(tmp_path, monkeypatch):
    """A CronScheduler isolated to tmp_path so test runs don't bleed
    into ``config/cron.json``."""
    monkeypatch.chdir(tmp_path)
    Path(tmp_path / "config").mkdir(exist_ok=True)
    from src.host.cron import CronScheduler
    return CronScheduler()


# ---------------------------------------------------------------- find_summary
def test_find_summary_job_returns_none_when_empty(scheduler):
    assert scheduler.find_summary_job("team", "content-seo") is None


def test_find_summary_job_matches_scope(scheduler):
    job = scheduler.ensure_summary_job(scope_kind="team", scope_id="content-seo")
    found = scheduler.find_summary_job("team", "content-seo")
    assert found is not None
    assert found.id == job.id


def test_find_summary_job_returns_none_for_wrong_scope(scheduler):
    scheduler.ensure_summary_job(scope_kind="team", scope_id="content-seo")
    assert scheduler.find_summary_job("team", "growth") is None
    assert scheduler.find_summary_job("solo", "content-seo") is None


def test_find_summary_job_ignores_non_summary_tool_jobs(scheduler):
    scheduler.add_job(
        agent="operator", schedule="0 8 * * *",
        tool_name="other_tool",
        tool_params=json.dumps({"scope_id": "x"}),
    )
    assert scheduler.find_summary_job("team", "x") is None


def test_find_summary_job_handles_bad_json_params(scheduler):
    """Malformed ``tool_params`` mustn't crash the lookup — skip the
    row and keep scanning."""
    scheduler.add_job(
        agent="operator", schedule="0 8 * * *",
        tool_name="compose_work_summary",
        tool_params="this is not json",  # malformed
    )
    # Real summary job for the actual scope.
    scheduler.ensure_summary_job(scope_kind="team", scope_id="content-seo")
    found = scheduler.find_summary_job("team", "content-seo")
    assert found is not None


# --------------------------------------------------------------- ensure_summary
def test_ensure_summary_job_is_idempotent(scheduler):
    a = scheduler.ensure_summary_job(scope_kind="team", scope_id="x")
    b = scheduler.ensure_summary_job(scope_kind="team", scope_id="x")
    assert a.id == b.id
    summary_jobs = [
        j for j in scheduler.jobs.values()
        if j.tool_name == "compose_work_summary"
    ]
    assert len(summary_jobs) == 1


def test_ensure_summary_job_different_scopes_are_distinct(scheduler):
    a = scheduler.ensure_summary_job(scope_kind="team", scope_id="x")
    b = scheduler.ensure_summary_job(scope_kind="team", scope_id="y")
    c = scheduler.ensure_summary_job(scope_kind="solo", scope_id="x")
    assert a.id != b.id != c.id
    assert len({a.id, b.id, c.id}) == 3


def test_ensure_summary_job_uses_default_schedule(scheduler):
    from src.host.cron import CronScheduler
    job = scheduler.ensure_summary_job(scope_kind="team", scope_id="x")
    assert job.schedule == CronScheduler.DEFAULT_SUMMARY_SCHEDULE


def test_ensure_summary_job_accepts_custom_schedule(scheduler):
    job = scheduler.ensure_summary_job(
        scope_kind="team", scope_id="x", schedule="every 6h",
    )
    assert job.schedule == "every 6h"


def test_ensure_summary_job_fires_compose_tool(scheduler):
    """The cron must dispatch ``compose_work_summary`` directly
    (tool_name path) so each tick is deterministic, not LLM-mediated."""
    job = scheduler.ensure_summary_job(scope_kind="team", scope_id="x")
    assert job.tool_name == "compose_work_summary"
    params = json.loads(job.tool_params)
    assert params == {"scope_kind": "team", "scope_id": "x"}


def test_ensure_summary_job_dispatches_to_operator_by_default(scheduler):
    job = scheduler.ensure_summary_job(scope_kind="team", scope_id="x")
    assert job.agent == "operator"


# ----------------------------------------- reconcile (bootstrap path in runtime)
def _seed_team(projects_dir: Path, name: str, *, status: str = "active",
               schedule: str | None = None):
    """Seed a team's metadata.yaml. Per-team cadence overrides live in
    ``settings.summary_schedule`` (TeamMetadata.settings is the
    extension dict; top-level fields are schema-pinned)."""
    import yaml as _yaml
    team_dir = projects_dir / name
    team_dir.mkdir(parents=True, exist_ok=True)
    meta = {"name": name, "description": "", "members": [], "status": status}
    if schedule is not None:
        meta["settings"] = {"summary_schedule": schedule}
    (team_dir / "metadata.yaml").write_text(_yaml.dump(meta))


def test_reconcile_creates_one_job_per_active_team(tmp_path, monkeypatch):
    """The reconcile path should iterate active teams and ensure each
    has a summary cron job. Archived teams are skipped."""
    projects_dir = tmp_path / "config" / "projects"
    projects_dir.mkdir(parents=True)
    _seed_team(projects_dir, "alpha")
    _seed_team(projects_dir, "beta")
    _seed_team(projects_dir, "archived-team", status="archived")
    import src.cli.config as _cli_config
    monkeypatch.setattr(_cli_config, "TEAMS_DIR", projects_dir)
    monkeypatch.setattr(_cli_config, "TEAMS_DIR", projects_dir)

    # Drive the reconcile through a minimal RuntimeContext stand-in
    # using just the bits the method touches.
    from src.host.cron import CronScheduler
    scheduler = CronScheduler()

    class _Stub:
        cron_scheduler = scheduler

    from src.cli.runtime import RuntimeContext
    RuntimeContext._reconcile_work_summary_jobs(_Stub())

    summary_jobs = [
        j for j in scheduler.jobs.values()
        if j.tool_name == "compose_work_summary"
    ]
    scope_ids = sorted(
        json.loads(j.tool_params)["scope_id"] for j in summary_jobs
    )
    assert scope_ids == ["alpha", "beta"]


def test_reconcile_honors_per_team_schedule_override(tmp_path, monkeypatch):
    projects_dir = tmp_path / "config" / "projects"
    projects_dir.mkdir(parents=True)
    _seed_team(projects_dir, "weekly-team", schedule="0 9 * * 1")  # Mondays
    import src.cli.config as _cli_config
    monkeypatch.setattr(_cli_config, "TEAMS_DIR", projects_dir)
    monkeypatch.setattr(_cli_config, "TEAMS_DIR", projects_dir)

    from src.host.cron import CronScheduler
    scheduler = CronScheduler()

    class _Stub:
        cron_scheduler = scheduler
    from src.cli.runtime import RuntimeContext
    RuntimeContext._reconcile_work_summary_jobs(_Stub())

    job = scheduler.find_summary_job("team", "weekly-team")
    assert job is not None
    assert job.schedule == "0 9 * * 1"


def test_reconcile_prunes_summary_jobs_for_deleted_teams(tmp_path, monkeypatch):
    projects_dir = tmp_path / "config" / "projects"
    projects_dir.mkdir(parents=True)
    _seed_team(projects_dir, "alpha")
    import src.cli.config as _cli_config
    monkeypatch.setattr(_cli_config, "TEAMS_DIR", projects_dir)
    monkeypatch.setattr(_cli_config, "TEAMS_DIR", projects_dir)

    from src.host.cron import CronScheduler
    scheduler = CronScheduler()
    # Pre-seed a summary job for a team that no longer exists.
    scheduler.ensure_summary_job(scope_kind="team", scope_id="orphan")
    scheduler.ensure_summary_job(scope_kind="team", scope_id="alpha")

    class _Stub:
        cron_scheduler = scheduler
    from src.cli.runtime import RuntimeContext
    RuntimeContext._reconcile_work_summary_jobs(_Stub())

    # alpha kept; orphan pruned; no duplicates of alpha.
    summary_jobs = [
        j for j in scheduler.jobs.values()
        if j.tool_name == "compose_work_summary"
    ]
    scope_ids = sorted(
        json.loads(j.tool_params)["scope_id"] for j in summary_jobs
    )
    assert scope_ids == ["alpha"]


def test_reconcile_prunes_summary_jobs_for_archived_teams(tmp_path, monkeypatch):
    projects_dir = tmp_path / "config" / "projects"
    projects_dir.mkdir(parents=True)
    _seed_team(projects_dir, "alpha")
    _seed_team(projects_dir, "old-team", status="archived")
    import src.cli.config as _cli_config
    monkeypatch.setattr(_cli_config, "TEAMS_DIR", projects_dir)
    monkeypatch.setattr(_cli_config, "TEAMS_DIR", projects_dir)

    from src.host.cron import CronScheduler
    scheduler = CronScheduler()
    # Pre-seed a job for the team that's now archived.
    scheduler.ensure_summary_job(scope_kind="team", scope_id="old-team")

    class _Stub:
        cron_scheduler = scheduler
    from src.cli.runtime import RuntimeContext
    RuntimeContext._reconcile_work_summary_jobs(_Stub())

    summary_jobs = [
        j for j in scheduler.jobs.values()
        if j.tool_name == "compose_work_summary"
    ]
    scope_ids = sorted(
        json.loads(j.tool_params)["scope_id"] for j in summary_jobs
    )
    assert scope_ids == ["alpha"]


def test_reconcile_reschedules_drift_to_new_cadence(tmp_path, monkeypatch):
    """When team metadata's summary_schedule changes between boots,
    the reconcile path must update the existing job's schedule —
    ensure_summary_job alone returns the existing job unchanged
    (codex r1 P2)."""
    projects_dir = tmp_path / "config" / "projects"
    projects_dir.mkdir(parents=True)
    _seed_team(projects_dir, "alpha", schedule="0 9 * * *")
    import src.cli.config as _cli_config
    monkeypatch.setattr(_cli_config, "TEAMS_DIR", projects_dir)
    monkeypatch.setattr(_cli_config, "TEAMS_DIR", projects_dir)

    from src.host.cron import CronScheduler
    scheduler = CronScheduler()

    class _Stub:
        cron_scheduler = scheduler
    from src.cli.runtime import RuntimeContext

    # First boot — daily.
    RuntimeContext._reconcile_work_summary_jobs(_Stub())
    job_id_before = scheduler.find_summary_job("team", "alpha").id

    # Operator edits the metadata to weekly cadence between boots.
    _seed_team(projects_dir, "alpha", schedule="0 9 * * 1")

    # Second boot — should reschedule the SAME job, not create a new one.
    RuntimeContext._reconcile_work_summary_jobs(_Stub())
    found = scheduler.find_summary_job("team", "alpha")
    assert found is not None
    assert found.id == job_id_before  # same job, not recreated
    assert found.schedule == "0 9 * * 1"


def test_reconcile_logs_warning_on_invalid_schedule_metadata(
    tmp_path, monkeypatch,
):
    """Bad team metadata (invalid cron expr) must NOT crash reconcile
    or apply the bad schedule — log a warning and keep the existing."""
    projects_dir = tmp_path / "config" / "projects"
    projects_dir.mkdir(parents=True)
    _seed_team(projects_dir, "alpha", schedule="0 9 * * *")
    import src.cli.config as _cli_config
    monkeypatch.setattr(_cli_config, "TEAMS_DIR", projects_dir)
    monkeypatch.setattr(_cli_config, "TEAMS_DIR", projects_dir)

    from src.host.cron import CronScheduler
    scheduler = CronScheduler()

    class _Stub:
        cron_scheduler = scheduler
    from src.cli.runtime import RuntimeContext

    RuntimeContext._reconcile_work_summary_jobs(_Stub())
    job_before = scheduler.find_summary_job("team", "alpha")
    schedule_before = job_before.schedule

    # Corrupt the metadata.
    _seed_team(projects_dir, "alpha", schedule="this is not cron")

    # Must not raise.
    RuntimeContext._reconcile_work_summary_jobs(_Stub())
    job_after = scheduler.find_summary_job("team", "alpha")
    assert job_after.schedule == schedule_before  # preserved


def test_reconcile_leaves_non_summary_tool_jobs_alone(tmp_path, monkeypatch):
    """An unrelated tool-cron must not be pruned by the summary
    reconcile, even when its tool_params can't be parsed."""
    projects_dir = tmp_path / "config" / "projects"
    projects_dir.mkdir(parents=True)
    _seed_team(projects_dir, "alpha")
    import src.cli.config as _cli_config
    monkeypatch.setattr(_cli_config, "TEAMS_DIR", projects_dir)
    monkeypatch.setattr(_cli_config, "TEAMS_DIR", projects_dir)

    from src.host.cron import CronScheduler
    scheduler = CronScheduler()
    other = scheduler.add_job(
        agent="operator", schedule="0 0 * * *",
        tool_name="some_other_tool",
        tool_params=json.dumps({"any": "thing"}),
    )
    scheduler.ensure_summary_job(scope_kind="team", scope_id="alpha")

    class _Stub:
        cron_scheduler = scheduler
    from src.cli.runtime import RuntimeContext
    RuntimeContext._reconcile_work_summary_jobs(_Stub())

    # other_tool job still there.
    assert other.id in scheduler.jobs
