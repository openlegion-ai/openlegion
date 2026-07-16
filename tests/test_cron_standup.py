"""Tests for the lead standup cron job (plan §8 #14).

Covers:
- ``CronScheduler.find_standup_job`` / ``ensure_standup_job`` /
  ``remove_standup_job``: idempotent create, in-place repoint on lead
  change, in-place reschedule on cadence drift.
- ``_execute_job``'s host-published channel post: posts a non-empty/
  non-error response into the team's channel thread via
  ``thread_store``; skips on empty/error responses; never crashes the
  job on a ``thread_store`` hiccup (best-effort, mirrors the mesh's
  other host-side thread writers).
- ``post_to_channel`` is HOST-SIDE ONLY: no agent-facing cron surface
  (the mesh create/update endpoints, the ``set_cron`` tool) can ever
  set or change it — the security invariant behind the "thread writers
  are host-side only" rule (src/host/threads.py).
- ``RuntimeContext._reconcile_standup_jobs`` mirrors
  ``_reconcile_work_summary_jobs``: ensure for led teams, prune for
  teams without a lead (or archived/deleted), reschedule on drift.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def scheduler(tmp_path, monkeypatch):
    """A CronScheduler isolated to tmp_path, mirroring the summary-job
    test fixture."""
    monkeypatch.chdir(tmp_path)
    Path(tmp_path / "config").mkdir(exist_ok=True)
    from src.host.cron import CronScheduler
    return CronScheduler()


# ── find_standup_job / ensure_standup_job / remove_standup_job ──────


def test_find_standup_job_returns_none_when_empty(scheduler):
    assert scheduler.find_standup_job("alpha") is None


def test_ensure_standup_job_creates(scheduler):
    from src.host.cron import CronScheduler

    job = scheduler.ensure_standup_job("alpha", "lead-1")
    assert job.agent == "lead-1"
    assert job.post_to_channel == "alpha"
    assert job.tool_name is None
    assert job.heartbeat is False
    assert job.schedule == CronScheduler.DEFAULT_STANDUP_SCHEDULE
    # Distinct from the 9:00 daily summary default.
    from src.host.cron import CronScheduler as _CS
    assert job.schedule != _CS.DEFAULT_SUMMARY_SCHEDULE


def test_ensure_standup_job_is_idempotent(scheduler):
    a = scheduler.ensure_standup_job("alpha", "lead-1")
    b = scheduler.ensure_standup_job("alpha", "lead-1")
    assert a.id == b.id
    standup_jobs = [j for j in scheduler.jobs.values() if j.post_to_channel == "alpha"]
    assert len(standup_jobs) == 1


def test_ensure_standup_job_different_teams_distinct(scheduler):
    a = scheduler.ensure_standup_job("alpha", "lead-1")
    b = scheduler.ensure_standup_job("beta", "lead-2")
    assert a.id != b.id


def test_ensure_standup_job_accepts_custom_schedule(scheduler):
    job = scheduler.ensure_standup_job("alpha", "lead-1", schedule="0 10 * * *")
    assert job.schedule == "0 10 * * *"


def test_ensure_standup_job_repoints_on_lead_change(scheduler):
    """A lead reassignment must repoint the SAME job, not duplicate it."""
    first = scheduler.ensure_standup_job("alpha", "lead-1")
    second = scheduler.ensure_standup_job("alpha", "lead-2")
    assert first.id == second.id
    assert second.agent == "lead-2"
    standup_jobs = [j for j in scheduler.jobs.values() if j.post_to_channel == "alpha"]
    assert len(standup_jobs) == 1


def test_ensure_standup_job_reschedules_on_drift(scheduler):
    first = scheduler.ensure_standup_job("alpha", "lead-1", schedule="0 9 * * *")
    second = scheduler.ensure_standup_job("alpha", "lead-1", schedule="0 11 * * *")
    assert first.id == second.id
    assert second.schedule == "0 11 * * *"


def test_ensure_standup_job_persists(scheduler, tmp_path):
    from src.host.cron import CronScheduler

    scheduler.ensure_standup_job("alpha", "lead-1")
    reloaded = CronScheduler(config_path=str(tmp_path / "config" / "cron.json"))
    job = reloaded.find_standup_job("alpha")
    assert job is not None
    assert job.agent == "lead-1"
    assert job.post_to_channel == "alpha"


def test_remove_standup_job(scheduler):
    scheduler.ensure_standup_job("alpha", "lead-1")
    assert scheduler.remove_standup_job("alpha") is True
    assert scheduler.find_standup_job("alpha") is None


def test_remove_standup_job_missing_returns_false(scheduler):
    assert scheduler.remove_standup_job("ghost") is False


def test_find_standup_job_ignores_summary_and_heartbeat_jobs(scheduler):
    """A tool-fired summary job or a heartbeat job must never be
    mistaken for a standup job even if it somehow carried
    ``post_to_channel`` (defense in depth for the matcher)."""
    scheduler.ensure_summary_job(scope_kind="team", scope_id="alpha")
    scheduler.ensure_heartbeat("alpha")
    assert scheduler.find_standup_job("alpha") is None


# ── _execute_job: host-published channel post ────────────────────────


class TestPostToChannelExecutor:
    @pytest.mark.asyncio
    async def test_posts_on_success(self, scheduler):
        dispatch = AsyncMock(return_value="Team shipped the report today.")
        thread_store = MagicMock()
        thread_store.ensure_channel.return_value = {"id": "channel:alpha"}
        scheduler.dispatch_fn = dispatch
        scheduler.thread_store = thread_store
        job = scheduler.add_job(
            agent="lead-1", schedule="every 1h", message="standup", post_to_channel="alpha",
        )
        result = await scheduler._execute_job(job)
        assert result == "Team shipped the report today."
        thread_store.ensure_channel.assert_called_once_with("alpha")
        thread_store.post_message.assert_called_once_with(
            "channel:alpha", sender="lead-1", body="Team shipped the report today.",
        )

    @pytest.mark.asyncio
    async def test_skips_on_empty_response(self, scheduler):
        dispatch = AsyncMock(return_value="")
        thread_store = MagicMock()
        scheduler.dispatch_fn = dispatch
        scheduler.thread_store = thread_store
        job = scheduler.add_job(
            agent="lead-1", schedule="every 1h", message="standup",
            post_to_channel="alpha", suppress_empty=False,
        )
        await scheduler._execute_job(job)
        thread_store.ensure_channel.assert_not_called()
        thread_store.post_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_on_suppressed_empty_response(self, scheduler):
        dispatch = AsyncMock(return_value="heartbeat_ok")
        thread_store = MagicMock()
        scheduler.dispatch_fn = dispatch
        scheduler.thread_store = thread_store
        job = scheduler.add_job(
            agent="lead-1", schedule="every 1h", message="standup", post_to_channel="alpha",
        )
        await scheduler._execute_job(job)
        thread_store.post_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_response_never_posts(self, scheduler):
        """A dispatch failure (caught by _execute_job's outer try/except)
        must not reach the channel-post code — the job's error path
        returns before it."""
        dispatch = AsyncMock(side_effect=RuntimeError("agent unreachable"))
        thread_store = MagicMock()
        scheduler.dispatch_fn = dispatch
        scheduler.thread_store = thread_store
        job = scheduler.add_job(
            agent="lead-1", schedule="every 1h", message="standup", post_to_channel="alpha",
        )
        result = await scheduler._execute_job(job)
        assert result is None
        thread_store.post_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_thread_store_hiccup_never_crashes_job(self, scheduler):
        """A ThreadStore exception must be swallowed — the standup turn
        already ran; a posting failure must not surface as a cron error."""
        dispatch = AsyncMock(return_value="Standup update.")
        thread_store = MagicMock()
        thread_store.ensure_channel.side_effect = RuntimeError("db locked")
        scheduler.dispatch_fn = dispatch
        scheduler.thread_store = thread_store
        job = scheduler.add_job(
            agent="lead-1", schedule="every 1h", message="standup", post_to_channel="alpha",
        )
        result = await scheduler._execute_job(job)
        assert result == "Standup update."
        assert job.error_count == 0

    @pytest.mark.asyncio
    async def test_no_thread_store_wired_is_a_noop(self, scheduler):
        dispatch = AsyncMock(return_value="Standup update.")
        scheduler.dispatch_fn = dispatch
        scheduler.thread_store = None
        job = scheduler.add_job(
            agent="lead-1", schedule="every 1h", message="standup", post_to_channel="alpha",
        )
        result = await scheduler._execute_job(job)
        assert result == "Standup update."

    @pytest.mark.asyncio
    async def test_non_string_response_is_json_encoded(self, scheduler):
        """A dict response (defensive — dispatch_fn is documented to
        return str, but the post helper must not crash on odd input)."""
        dispatch = AsyncMock(return_value={"summary": "ok"})
        thread_store = MagicMock()
        thread_store.ensure_channel.return_value = {"id": "channel:alpha"}
        scheduler.dispatch_fn = dispatch
        scheduler.thread_store = thread_store
        job = scheduler.add_job(
            agent="lead-1", schedule="every 1h", message="standup", post_to_channel="alpha",
        )
        await scheduler._execute_job(job)
        _, kwargs = thread_store.post_message.call_args
        assert "ok" in kwargs["body"]

    @pytest.mark.asyncio
    async def test_job_without_post_to_channel_never_touches_thread_store(self, scheduler):
        dispatch = AsyncMock(return_value="regular message job")
        thread_store = MagicMock()
        scheduler.dispatch_fn = dispatch
        scheduler.thread_store = thread_store
        job = scheduler.add_job(agent="lead-1", schedule="every 1h", message="hi")
        await scheduler._execute_job(job)
        thread_store.ensure_channel.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "returned",
        ["__SILENT__", "(no response)", "dispatch_error: connection reset"],
    )
    async def test_lane_dispatch_non_success_shapes_never_posted(self, scheduler, returned):
        """The lane dispatcher returns three non-success shapes AS SUCCESS
        strings (SILENT sentinel, "(no response)", "dispatch_error:"). A
        briefly-unreachable lead must never post any of them into the team
        channel — the shared usable-reply gate rejects all three."""
        dispatch = AsyncMock(return_value=returned)
        thread_store = MagicMock()
        thread_store.ensure_channel.return_value = {"id": "channel:alpha"}
        scheduler.dispatch_fn = dispatch
        scheduler.thread_store = thread_store
        job = scheduler.add_job(
            agent="lead-1", schedule="every 1h", message="standup",
            post_to_channel="alpha", suppress_empty=False,
        )
        await scheduler._execute_job(job)
        thread_store.ensure_channel.assert_not_called()
        thread_store.post_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_job_removed_mid_dispatch_does_not_resurrect_channel(self, scheduler):
        """A standup job removed WHILE its (slow) dispatch runs — the team was
        deleted/archived mid-turn — must not post its late reply, which would
        resurrect the deleted team's channel via ``ensure_channel``."""
        thread_store = MagicMock()
        thread_store.ensure_channel.return_value = {"id": "channel:alpha"}
        job = scheduler.add_job(
            agent="lead-1", schedule="every 1h", message="standup", post_to_channel="alpha",
        )

        async def _slow_dispatch(agent, message):
            # Team deleted mid-turn — the job is removed from the table.
            scheduler.remove_job(job.id)
            return "Standup update that arrived after the team was deleted."

        scheduler.dispatch_fn = _slow_dispatch
        scheduler.thread_store = thread_store
        await scheduler._execute_job(job)
        thread_store.ensure_channel.assert_not_called()
        thread_store.post_message.assert_not_called()


# ── Security: post_to_channel is HOST-SIDE ONLY ──────────────────────


class TestPostToChannelNotAgentSettable:
    """post_to_channel must never be reachable from an agent-facing
    cron surface — an agent proxy-posting into the team channel thread
    would violate the writers-host-side-only invariant."""

    def test_not_in_updatable_fields(self):
        from src.host.cron import CronScheduler
        assert "post_to_channel" not in CronScheduler._UPDATABLE_FIELDS

    @pytest.mark.asyncio
    async def test_update_job_ignores_post_to_channel(self, scheduler):
        job = scheduler.add_job(agent="a", schedule="every 1h", message="hi")
        updated = await scheduler.update_job(job.id, post_to_channel="alpha")
        assert updated.post_to_channel is None

    def test_set_cron_tool_schema_has_no_post_to_channel_param(self):
        """The LLM-facing tool schema must not even expose the
        parameter — the agent can't ask for it, let alone set it."""
        import importlib

        import src.agent.builtins.mesh_tool as mesh_tool
        # Reload so registration re-runs even when another test file in
        # this worker cleared the module-global ``_tool_staging`` (e.g.
        # test_grouped_tools/test_mcp_e2e setup) — a bare re-import is a
        # no-op and this security pin would KeyError on registry state
        # instead of testing the schema. Same idiom as test_builtins'
        # set_cron copy test.
        importlib.reload(mesh_tool)
        from src.agent.tools import _tool_staging

        params = _tool_staging["set_cron"]["parameters"]
        assert "post_to_channel" not in params

    @pytest.mark.asyncio
    async def test_mesh_create_cron_endpoint_ignores_post_to_channel(self, tmp_path, monkeypatch):
        """POST /mesh/cron: even if a caller sneaks ``post_to_channel``
        into the JSON body, the created job never carries it — the
        endpoint only forwards a fixed allowlist of fields to
        ``add_job``."""
        import importlib

        from httpx import ASGITransport, AsyncClient

        monkeypatch.chdir(tmp_path)
        import src.host.server as server_module

        importlib.reload(server_module)
        from src.host.cron import CronScheduler
        from src.host.mesh import Blackboard, MessageRouter, PubSub
        from src.host.permissions import PermissionMatrix

        permissions = PermissionMatrix()
        router = MessageRouter(permissions, {"operator": "http://op:8400"})
        blackboard = Blackboard(str(tmp_path / "bb.db"))
        cron_scheduler = CronScheduler(config_path=str(tmp_path / "cron.json"))
        app = server_module.create_mesh_app(
            blackboard=blackboard,
            pubsub=PubSub(),
            router=router,
            permissions=permissions,
            cron_scheduler=cron_scheduler,
            auth_tokens={"operator": "op-token"},
        )
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
                resp = await c.post(
                    "/mesh/cron",
                    json={
                        # "operator" is a reserved agent_id and 400s the
                        # request-body validator before resolution — use
                        # a plain worker name; the verified BEARER
                        # identity (operator, via auth_tokens) is what
                        # actually lands in job.agent either way.
                        "agent_id": "worker1",
                        "schedule": "every 1h",
                        "message": "hi",
                        "post_to_channel": "alpha",
                    },
                    headers={"Authorization": "Bearer op-token", "X-Agent-ID": "operator"},
                )
            assert resp.status_code == 200, resp.text
            job_id = resp.json()["id"]
            assert cron_scheduler.jobs[job_id].post_to_channel is None
        finally:
            blackboard.close()
            importlib.reload(server_module)

    @pytest.mark.asyncio
    async def test_mesh_update_cron_endpoint_ignores_post_to_channel(self, tmp_path, monkeypatch):
        """PUT /mesh/cron/{id}: the same sneak-it-into-the-body attempt
        against an existing job must also be a no-op."""
        import importlib

        from httpx import ASGITransport, AsyncClient

        monkeypatch.chdir(tmp_path)
        import src.host.server as server_module

        importlib.reload(server_module)
        from src.host.cron import CronScheduler
        from src.host.mesh import Blackboard, MessageRouter, PubSub
        from src.host.permissions import PermissionMatrix

        permissions = PermissionMatrix()
        router = MessageRouter(permissions, {"operator": "http://op:8400"})
        blackboard = Blackboard(str(tmp_path / "bb.db"))
        cron_scheduler = CronScheduler(config_path=str(tmp_path / "cron.json"))
        job = cron_scheduler.add_job(agent="operator", schedule="every 1h", message="hi")
        app = server_module.create_mesh_app(
            blackboard=blackboard,
            pubsub=PubSub(),
            router=router,
            permissions=permissions,
            cron_scheduler=cron_scheduler,
            auth_tokens={"operator": "op-token"},
        )
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
                resp = await c.put(
                    f"/mesh/cron/{job.id}",
                    json={"post_to_channel": "alpha"},
                    headers={"Authorization": "Bearer op-token", "X-Agent-ID": "operator"},
                )
            assert resp.status_code == 200, resp.text
            assert cron_scheduler.jobs[job.id].post_to_channel is None
        finally:
            blackboard.close()
            importlib.reload(server_module)


class TestAddMemberAutoAppointWiresStandup:
    """Phase-1 leadership loop (docs/plans/2026-07-16-autonomous-team-
    delivery.md §1/§3): the ``POST /mesh/teams/{name}/members`` endpoint
    (what ``add_agents_to_team`` calls) must wire the auto-appointed lead's
    standup cron LIVE — not defer it to the next mesh reboot's reconcile."""

    @pytest.mark.asyncio
    async def test_add_member_crossing_to_two_wires_standup_job(self, tmp_path, monkeypatch):
        import importlib
        import json as _json

        import yaml as _yaml
        from httpx import ASGITransport, AsyncClient

        monkeypatch.chdir(tmp_path)
        import src.cli.config as cli_cfg

        perms_file = tmp_path / "permissions.json"
        perms_file.write_text(
            _json.dumps(
                {
                    "permissions": {
                        "agent1": {"blackboard_read": [], "blackboard_write": []},
                        "agent2": {"blackboard_read": [], "blackboard_write": []},
                    }
                }
            )
        )
        monkeypatch.setattr(cli_cfg, "PERMISSIONS_FILE", perms_file)
        agents_file = tmp_path / "agents.yaml"
        agents_file.write_text(
            _yaml.dump(
                {
                    "agents": {
                        "agent1": {"role": "a"},
                        "agent2": {"role": "b"},
                        "operator": {"role": "operator"},
                    }
                }
            )
        )
        monkeypatch.setattr(cli_cfg, "AGENTS_FILE", agents_file)

        import src.host.server as server_module

        importlib.reload(server_module)
        from src.host.cron import CronScheduler
        from src.host.mesh import Blackboard, MessageRouter, PubSub
        from src.host.permissions import PermissionMatrix
        from src.host.teams import TeamStore

        permissions = PermissionMatrix()
        router = MessageRouter(
            permissions,
            {"operator": "http://op:8400", "agent1": "http://a1:8400", "agent2": "http://a2:8400"},
        )
        blackboard = Blackboard(str(tmp_path / "bb.db"))
        cron_scheduler = CronScheduler(config_path=str(tmp_path / "cron.json"))
        teams_store = TeamStore(db_path=str(tmp_path / "teams.db"), teams_dir=tmp_path / "teams")
        teams_store.create_team("squad")
        app = server_module.create_mesh_app(
            blackboard=blackboard,
            pubsub=PubSub(),
            router=router,
            permissions=permissions,
            teams_store=teams_store,
            cron_scheduler=cron_scheduler,
            auth_tokens={"operator": "op-token"},
        )
        hdr = {"Authorization": "Bearer op-token", "X-Agent-ID": "operator"}
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
                r1 = await c.post("/mesh/teams/squad/members", json={"agent": "agent1"}, headers=hdr)
                assert r1.status_code == 200, r1.text
                # Solo so far: no lead, no standup job yet.
                assert teams_store.get_team("squad")["lead_agent_id"] is None
                assert cron_scheduler.find_standup_job("squad") is None
                r2 = await c.post("/mesh/teams/squad/members", json={"agent": "agent2"}, headers=hdr)
                assert r2.status_code == 200, r2.text
            # Crossed to two members: first member auto-appointed + standup wired live.
            assert teams_store.get_team("squad")["lead_agent_id"] == "agent1"
            job = cron_scheduler.find_standup_job("squad")
            assert job is not None
            assert job.agent == "agent1"
            assert job.post_to_channel == "squad"
        finally:
            blackboard.close()
            importlib.reload(server_module)


# ── Boot reconcile (cli/runtime.RuntimeContext._reconcile_standup_jobs) ──


def _team_store(tmp_path):
    from src.host.teams import TeamStore
    return TeamStore(db_path=str(tmp_path / "teams.db"))


def _seed_led_team(store, name: str, lead: str | None, *, status: str = "active", schedule: str | None = None):
    if not store.team_exists(name):
        store.create_team(name)
    store.set_status(name, status)
    if lead is not None:
        store.add_member(name, lead)
        store.set_lead(name, lead)
    if schedule is not None:
        store.set_settings(name, {"standup_schedule": schedule})


class TestReconcileStandupJobs:
    def test_reconcile_creates_job_only_for_led_teams(self, tmp_path):
        store = _team_store(tmp_path)
        _seed_led_team(store, "alpha", "lead-1")
        _seed_led_team(store, "beta", None)  # no lead — no standup job

        from src.host.cron import CronScheduler
        scheduler = CronScheduler(config_path=str(tmp_path / "cron.json"))

        class _Stub:
            cron_scheduler = scheduler
            teams_store = store

        from src.cli.runtime import RuntimeContext
        RuntimeContext._reconcile_standup_jobs(_Stub())

        assert scheduler.find_standup_job("alpha") is not None
        assert scheduler.find_standup_job("beta") is None

    def test_reconcile_skips_archived_teams(self, tmp_path):
        store = _team_store(tmp_path)
        _seed_led_team(store, "alpha", "lead-1", status="archived")

        from src.host.cron import CronScheduler
        scheduler = CronScheduler(config_path=str(tmp_path / "cron.json"))

        class _Stub:
            cron_scheduler = scheduler
            teams_store = store

        from src.cli.runtime import RuntimeContext
        RuntimeContext._reconcile_standup_jobs(_Stub())

        assert scheduler.find_standup_job("alpha") is None

    def test_reconcile_prunes_job_when_lead_cleared(self, tmp_path):
        store = _team_store(tmp_path)
        _seed_led_team(store, "alpha", "lead-1")

        from src.host.cron import CronScheduler
        scheduler = CronScheduler(config_path=str(tmp_path / "cron.json"))

        class _Stub:
            cron_scheduler = scheduler
            teams_store = store

        from src.cli.runtime import RuntimeContext
        RuntimeContext._reconcile_standup_jobs(_Stub())
        assert scheduler.find_standup_job("alpha") is not None

        store.set_lead("alpha", None)
        RuntimeContext._reconcile_standup_jobs(_Stub())
        assert scheduler.find_standup_job("alpha") is None

    def test_reconcile_prunes_job_for_deleted_team(self, tmp_path):
        store = _team_store(tmp_path)
        _seed_led_team(store, "alpha", "lead-1")

        from src.host.cron import CronScheduler
        scheduler = CronScheduler(config_path=str(tmp_path / "cron.json"))

        class _Stub:
            cron_scheduler = scheduler
            teams_store = store

        from src.cli.runtime import RuntimeContext
        RuntimeContext._reconcile_standup_jobs(_Stub())
        assert scheduler.find_standup_job("alpha") is not None

        store.delete_team("alpha")
        RuntimeContext._reconcile_standup_jobs(_Stub())
        assert scheduler.find_standup_job("alpha") is None

    def test_reconcile_repoints_on_lead_change(self, tmp_path):
        store = _team_store(tmp_path)
        _seed_led_team(store, "alpha", "lead-1")

        from src.host.cron import CronScheduler
        scheduler = CronScheduler(config_path=str(tmp_path / "cron.json"))

        class _Stub:
            cron_scheduler = scheduler
            teams_store = store

        from src.cli.runtime import RuntimeContext
        RuntimeContext._reconcile_standup_jobs(_Stub())
        job_id_before = scheduler.find_standup_job("alpha").id

        store.add_member("alpha", "lead-2")
        store.set_lead("alpha", "lead-2")
        RuntimeContext._reconcile_standup_jobs(_Stub())
        found = scheduler.find_standup_job("alpha")
        assert found.id == job_id_before
        assert found.agent == "lead-2"

    def test_reconcile_reschedules_drift(self, tmp_path):
        store = _team_store(tmp_path)
        _seed_led_team(store, "alpha", "lead-1", schedule="0 9 * * *")

        from src.host.cron import CronScheduler
        scheduler = CronScheduler(config_path=str(tmp_path / "cron.json"))

        class _Stub:
            cron_scheduler = scheduler
            teams_store = store

        from src.cli.runtime import RuntimeContext
        RuntimeContext._reconcile_standup_jobs(_Stub())
        job_id_before = scheduler.find_standup_job("alpha").id

        _seed_led_team(store, "alpha", None, schedule="0 11 * * *")
        RuntimeContext._reconcile_standup_jobs(_Stub())
        found = scheduler.find_standup_job("alpha")
        assert found.id == job_id_before
        assert found.schedule == "0 11 * * *"

    def test_reconcile_does_not_touch_summary_jobs(self, tmp_path):
        """The two reconciles must coexist without cross-pruning."""
        store = _team_store(tmp_path)
        _seed_led_team(store, "alpha", "lead-1")

        from src.host.cron import CronScheduler
        scheduler = CronScheduler(config_path=str(tmp_path / "cron.json"))
        scheduler.ensure_summary_job(scope_kind="team", scope_id="alpha")

        class _Stub:
            cron_scheduler = scheduler
            teams_store = store

        from src.cli.runtime import RuntimeContext
        RuntimeContext._reconcile_standup_jobs(_Stub())

        assert scheduler.find_summary_job("team", "alpha") is not None
        assert scheduler.find_standup_job("alpha") is not None
        assert len(scheduler.jobs) == 2

    def test_reconcile_no_cron_scheduler_is_a_noop(self, tmp_path):
        store = _team_store(tmp_path)

        class _Stub:
            cron_scheduler = None
            teams_store = store

        from src.cli.runtime import RuntimeContext
        RuntimeContext._reconcile_standup_jobs(_Stub())  # must not raise

    def test_reconcile_skips_archived_lead_agent_but_keeps_active(self, tmp_path, monkeypatch):
        """A team whose LEAD agent's config status is 'archived' (offboarded/
        paused) must NOT get a standup job recreated at boot — a ghost lead
        firing a daily LLM turn. A team led by an ACTIVE agent still does."""
        store = _team_store(tmp_path)
        _seed_led_team(store, "alpha", "lead-archived")
        _seed_led_team(store, "beta", "lead-active")

        from src.host.cron import CronScheduler
        scheduler = CronScheduler(config_path=str(tmp_path / "cron.json"))

        class _Stub:
            cron_scheduler = scheduler
            teams_store = store

        import src.cli.runtime as runtime_mod
        fake_cfg = {"agents": {
            "lead-archived": {"role": "lead", "status": "archived"},
            "lead-active": {"role": "lead", "status": "active"},
        }}
        monkeypatch.setattr(runtime_mod, "_load_config", lambda *a, **k: fake_cfg)

        from src.cli.runtime import RuntimeContext
        RuntimeContext._reconcile_standup_jobs(_Stub())

        assert scheduler.find_standup_job("alpha") is None      # archived lead → skipped
        assert scheduler.find_standup_job("beta") is not None   # active lead → created

    def test_reconcile_prunes_existing_job_when_lead_agent_archived(self, tmp_path, monkeypatch):
        """An existing standup job whose lead becomes archived (offboarded)
        is pruned on the next boot reconcile."""
        store = _team_store(tmp_path)
        _seed_led_team(store, "alpha", "lead-1")

        from src.host.cron import CronScheduler
        scheduler = CronScheduler(config_path=str(tmp_path / "cron.json"))

        class _Stub:
            cron_scheduler = scheduler
            teams_store = store

        import src.cli.runtime as runtime_mod
        from src.cli.runtime import RuntimeContext

        # First boot: lead active → job created.
        monkeypatch.setattr(
            runtime_mod, "_load_config",
            lambda *a, **k: {"agents": {"lead-1": {"status": "active"}}},
        )
        RuntimeContext._reconcile_standup_jobs(_Stub())
        assert scheduler.find_standup_job("alpha") is not None

        # Lead offboarded (config status archived) → next boot prunes it.
        monkeypatch.setattr(
            runtime_mod, "_load_config",
            lambda *a, **k: {"agents": {"lead-1": {"status": "archived"}}},
        )
        RuntimeContext._reconcile_standup_jobs(_Stub())
        assert scheduler.find_standup_job("alpha") is None


# ── Boot lead backfill (cli/runtime.RuntimeContext._backfill_team_leads) ──


def _seed_team(store, name: str, members: list[str], *, status: str = "active"):
    """Create a team with ``members`` and NO lead (pre-Phase-1 shape)."""
    store.create_team(name)
    for m in members:
        store.add_member(name, m)
    if status != "active":
        store.set_status(name, status)


class TestBackfillTeamLeads:
    """``_backfill_team_leads`` (Phase-1 autonomous-team-delivery): appoint
    the first non-operator member as lead for every active NON-SOLO team
    that predates auto-appointment and has a NULL lead. Solo/team-of-one
    self-leads (no lead row); existing leads and archived teams untouched."""

    def _stub(self, store):
        class _Stub:
            teams_store = store
        return _Stub()

    def test_backfill_appoints_first_member_for_leaderless_team(self, tmp_path):
        store = _team_store(tmp_path)
        _seed_team(store, "alpha", ["ada", "bob"])
        assert store.get_team("alpha")["lead_agent_id"] is None

        from src.cli.runtime import RuntimeContext
        RuntimeContext._backfill_team_leads(self._stub(store))

        assert store.get_team("alpha")["lead_agent_id"] == "ada"

    def test_backfill_skips_solo_team_of_one(self, tmp_path):
        store = _team_store(tmp_path)
        _seed_team(store, "solo", ["ada"])  # one member = self-leads

        from src.cli.runtime import RuntimeContext
        RuntimeContext._backfill_team_leads(self._stub(store))

        assert store.get_team("solo")["lead_agent_id"] is None

    def test_backfill_skips_empty_team(self, tmp_path):
        store = _team_store(tmp_path)
        _seed_team(store, "ghost", [])

        from src.cli.runtime import RuntimeContext
        RuntimeContext._backfill_team_leads(self._stub(store))

        assert store.get_team("ghost")["lead_agent_id"] is None

    def test_backfill_is_idempotent_and_keeps_existing_lead(self, tmp_path):
        store = _team_store(tmp_path)
        _seed_team(store, "alpha", ["ada", "bob"])
        store.set_lead("alpha", "bob")  # already led

        from src.cli.runtime import RuntimeContext
        RuntimeContext._backfill_team_leads(self._stub(store))
        # Existing lead preserved, not overwritten with the first member.
        assert store.get_team("alpha")["lead_agent_id"] == "bob"
        # Second run is a no-op.
        RuntimeContext._backfill_team_leads(self._stub(store))
        assert store.get_team("alpha")["lead_agent_id"] == "bob"

    def test_backfill_skips_archived_team(self, tmp_path):
        store = _team_store(tmp_path)
        _seed_team(store, "old", ["ada", "bob"], status="archived")

        from src.cli.runtime import RuntimeContext
        RuntimeContext._backfill_team_leads(self._stub(store))

        assert store.get_team("old")["lead_agent_id"] is None

    def test_backfill_then_standup_reconcile_creates_job(self, tmp_path):
        """End-to-end: a leaderless multi-member team gets a lead at boot,
        and the standup reconcile (run after) then wires its standup job."""
        store = _team_store(tmp_path)
        _seed_team(store, "alpha", ["ada", "bob"])

        from src.host.cron import CronScheduler
        scheduler = CronScheduler(config_path=str(tmp_path / "cron.json"))

        class _Stub:
            cron_scheduler = scheduler
            teams_store = store

        from src.cli.runtime import RuntimeContext
        stub = _Stub()
        # Boot order: backfill BEFORE the standup reconcile.
        RuntimeContext._backfill_team_leads(stub)
        RuntimeContext._reconcile_standup_jobs(stub)

        assert store.get_team("alpha")["lead_agent_id"] == "ada"
        job = scheduler.find_standup_job("alpha")
        assert job is not None
        assert job.agent == "ada"
