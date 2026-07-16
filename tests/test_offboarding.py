"""Offboarding-with-handover (plan §8 #15) — the data-loss hot spot.

Covers:
  - ``_offboard_agent`` core: solo/teamless no-op, live-container handover
    commit (author = the departing agent), unreachable-container snapshot-
    only, bounded handover-turn timeout, drive-commit-failure resilience,
    audit logging.
  - The new ``POST /mesh/agents/{id}/offboard`` endpoint: auth (operator-
    or-internal only, worker 403, operator target rejected), and that it
    runs offboard THEN archive.
  - ORDER PROOF: the mesh delete-confirm chain attempts the offboard
    (handover turn) strictly BEFORE ``stop_agent(..., remove_data=True)``.
  - ``manage_agent(action="offboard")`` operator-tool wiring.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import time
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from httpx import ASGITransport, AsyncClient

from src.host import drive as drive_mod
from src.host.mesh import Blackboard, MessageRouter, PubSub
from src.host.permissions import PermissionMatrix
from src.host.teams import TeamStore
from src.shared.types import MessageOrigin


class _RecordingLaneManager:
    """Duck-typed lane manager stand-in — ``_offboard_agent`` only ever
    calls ``.enqueue`` on it."""

    def __init__(
        self, *, reply="Handover doc: current work lives in the drive.",
        raise_exc=None, delay=0.0, order=None,
    ):
        self.calls: list[dict] = []
        self.reply = reply
        self.raise_exc = raise_exc
        self.delay = delay
        self.order = order

    async def enqueue(self, agent, message, *, mode="followup", system_note=False, **kw):
        self.calls.append({"agent": agent, "message": message, "mode": mode, "system_note": system_note})
        if self.order is not None:
            self.order.append("handover_turn")
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.raise_exc:
            raise self.raise_exc
        return self.reply

    def remove_lane(self, agent: str) -> None:
        """No-op — satisfies ``cleanup_agent``'s best-effort lane teardown."""


def _human_origin_headers(agent_id: str = "operator") -> dict:
    origin = MessageOrigin(kind="human", channel="cli", user="u1")
    return {
        "Authorization": "Bearer op-token",
        "X-Agent-ID": agent_id,
        "X-Origin": origin.to_header_value(),
    }


def _op_headers() -> dict:
    return {"Authorization": "Bearer op-token", "X-Agent-ID": "operator"}


def _worker_headers(agent: str, token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "X-Agent-ID": agent}


def _build_app(
    tmp_path,
    monkeypatch,
    *,
    lane_manager=None,
    container_manager=None,
    with_team_drive=True,
    extra_agents=None,
    cron_scheduler=None,
):
    """Mesh app with a real disk-backed TeamStore (drive provisioner
    wired so ``_offboard_agent`` commits to a REAL bare git repo), a
    real on-disk agents.yaml/permissions.json (archive/delete need
    ``AGENTS_FILE`` to exist), and bearer auth for the operator + a
    worker."""
    monkeypatch.chdir(tmp_path)
    drives_dir = tmp_path / "drives"
    if with_team_drive:
        monkeypatch.setenv("OPENLEGION_TEAM_DRIVES_DIR", str(drives_dir))

    agents = {"scout": {"role": "researcher", "model": "openai/gpt-4o-mini"}}
    if extra_agents:
        agents.update(extra_agents)

    config_dir = tmp_path / "config"
    config_dir.mkdir(exist_ok=True)
    agents_file = config_dir / "agents.yaml"
    agents_file.write_text(yaml.dump({"agents": agents}))
    perms_file = config_dir / "permissions.json"
    perms_file.write_text(json.dumps({"permissions": {}}))
    teams_dir = config_dir / "teams"
    teams_dir.mkdir(parents=True, exist_ok=True)

    import src.cli.config as cli_cfg

    monkeypatch.setattr(cli_cfg, "AGENTS_FILE", agents_file)
    monkeypatch.setattr(cli_cfg, "PERMISSIONS_FILE", perms_file)
    monkeypatch.setattr(cli_cfg, "TEAMS_DIR", teams_dir)

    import src.host.server as server_module

    importlib.reload(server_module)

    perms = PermissionMatrix()
    blackboard = Blackboard(str(tmp_path / "bb.db"))
    pubsub = PubSub()
    router = MessageRouter(perms, {"operator": "http://op:8400"})
    teams_store = TeamStore(db_path=str(tmp_path / "teams.db"), teams_dir=tmp_path / "teams")
    if with_team_drive:
        teams_store.set_drive_provisioner(drive_mod.ensure_team_drive, drive_mod.remove_team_drive)
    teams_store.create_team("research")
    teams_store.add_member("research", "scout")

    app = server_module.create_mesh_app(
        blackboard=blackboard,
        pubsub=pubsub,
        router=router,
        permissions=perms,
        teams_store=teams_store,
        container_manager=container_manager,
        lane_manager=lane_manager,
        cron_scheduler=cron_scheduler,
        auth_tokens={"operator": "op-token", "scout": "scout-token"},
    )
    return app, blackboard, teams_store, drives_dir


# ── _offboard_agent core ────────────────────────────────────────────


class TestOffboardAgentCore:
    @pytest.mark.asyncio
    async def test_solo_agent_no_team_drive_clean_manifest_no_exception(self, tmp_path, monkeypatch):
        """Solo/teamless agents have no Team Drive — skipped, never crashes."""
        app, bb, store, _ = _build_app(
            tmp_path, monkeypatch, extra_agents={"loner": {"role": "solo"}},
        )
        try:
            manifest = await app._offboard_agent("loner", reason="delete")
        finally:
            bb.close()
        assert manifest["team_id"] is None
        assert manifest["skipped"] == "no team drive"
        assert manifest["handover_committed"] is False
        assert manifest["snapshot_committed"] is False
        assert manifest["errors"] == []

    @pytest.mark.asyncio
    async def test_live_container_commits_handover_authored_by_departing_agent(self, tmp_path, monkeypatch):
        lane = _RecordingLaneManager(reply="Here is my handover: task X lives on branch y.")
        app, bb, store, drives_dir = _build_app(tmp_path, monkeypatch, lane_manager=lane)
        try:
            manifest = await app._offboard_agent("scout", reason="offboard")
        finally:
            bb.close()
        assert manifest["team_id"] == "research"
        assert manifest["handover_committed"] is True
        assert manifest["handover_ref"].startswith("drive://research/handovers/scout/")
        assert manifest["snapshot_committed"] is True
        assert manifest["snapshot_ref"].startswith("drive://research/handovers/scout/")
        assert manifest["errors"] == []
        # The lane got the SYSTEM-composed handover turn.
        assert lane.calls and lane.calls[0]["agent"] == "scout"
        assert lane.calls[0]["system_note"] is True

        # Author verification — a real git log against the bare repo.
        import subprocess

        repo = drives_dir / "research.git"
        log = subprocess.run(
            ["git", "log", "--format=%an <%ae>%n%s", "main"],
            cwd=repo, capture_output=True, text=True,
        )
        assert log.returncode == 0
        assert "scout <scout@agents.local>" in log.stdout
        assert "offboard handover for scout" in log.stdout
        assert "offboard snapshot for scout" in log.stdout

    @pytest.mark.asyncio
    async def test_unreachable_container_handover_skipped_snapshot_still_commits(self, tmp_path, monkeypatch):
        """A dispatch failure (unreachable container) must not block the
        snapshot commit — only the handover is skipped."""
        lane = _RecordingLaneManager(raise_exc=RuntimeError("connection refused"))
        app, bb, store, _ = _build_app(tmp_path, monkeypatch, lane_manager=lane)
        try:
            manifest = await app._offboard_agent("scout", reason="delete")
        finally:
            bb.close()
        assert manifest["handover_committed"] is False
        assert manifest["handover_ref"] is None
        assert manifest["snapshot_committed"] is True
        assert any("handover turn failed" in e for e in manifest["errors"])

    @pytest.mark.asyncio
    async def test_no_lane_manager_wired_handover_skipped_snapshot_still_commits(self, tmp_path, monkeypatch):
        """No lane manager (e.g. a bare mesh construction) — handover step
        is skipped outright, snapshot still commits."""
        app, bb, store, _ = _build_app(tmp_path, monkeypatch, lane_manager=None)
        try:
            manifest = await app._offboard_agent("scout", reason="delete")
        finally:
            bb.close()
        assert manifest["handover_committed"] is False
        assert manifest["snapshot_committed"] is True

    @pytest.mark.asyncio
    async def test_handover_turn_timeout_is_bounded_never_hangs(self, tmp_path, monkeypatch):
        import src.host.server as server_module

        real_resolve = server_module.limits_mod.resolve

        def _fake_resolve(key, *a, **kw):
            if key == "offboard_handover_timeout_seconds":
                return 0.05
            return real_resolve(key, *a, **kw)

        monkeypatch.setattr(server_module.limits_mod, "resolve", _fake_resolve)

        lane = _RecordingLaneManager(delay=5.0)
        app, bb, store, _ = _build_app(tmp_path, monkeypatch, lane_manager=lane)
        try:
            t0 = time.monotonic()
            manifest = await app._offboard_agent("scout", reason="delete")
            elapsed = time.monotonic() - t0
        finally:
            bb.close()
        assert elapsed < 3.0, f"offboard handover did not honor the bound (took {elapsed:.2f}s)"
        assert manifest["handover_committed"] is False
        assert any("timed out" in e for e in manifest["errors"])
        # Snapshot still attempted despite the handover timeout.
        assert manifest["snapshot_committed"] is True

    @pytest.mark.asyncio
    async def test_empty_handover_reply_records_no_handover_doc(self, tmp_path, monkeypatch):
        lane = _RecordingLaneManager(reply="   ")
        app, bb, store, _ = _build_app(tmp_path, monkeypatch, lane_manager=lane)
        try:
            manifest = await app._offboard_agent("scout", reason="delete")
        finally:
            bb.close()
        assert manifest["handover_committed"] is False
        assert manifest["snapshot_committed"] is True

    @pytest.mark.asyncio
    async def test_no_response_sentinel_records_no_handover_doc(self, tmp_path, monkeypatch):
        """The lane dispatcher returns the literal "(no response)" as a
        SUCCESS string for an unreachable agent (runtime._direct_dispatch)
        — it must never be committed as if the agent authored it."""
        lane = _RecordingLaneManager(reply="(no response)")
        app, bb, store, _ = _build_app(tmp_path, monkeypatch, lane_manager=lane)
        try:
            manifest = await app._offboard_agent("scout", reason="delete")
        finally:
            bb.close()
        assert manifest["handover_committed"] is False
        assert manifest["snapshot_committed"] is True
        assert any("nothing usable" in e for e in manifest["errors"])

    @pytest.mark.asyncio
    async def test_silent_token_records_no_handover_doc(self, tmp_path, monkeypatch):
        """The lane dispatcher returns SILENT_REPLY_TOKEN ("__SILENT__") for a
        transport failure — the shared usable-reply gate rejects it so it is
        never committed as if the agent authored it."""
        from src.shared.types import SILENT_REPLY_TOKEN

        lane = _RecordingLaneManager(reply=SILENT_REPLY_TOKEN)
        app, bb, store, _ = _build_app(tmp_path, monkeypatch, lane_manager=lane)
        try:
            manifest = await app._offboard_agent("scout", reason="delete")
        finally:
            bb.close()
        assert manifest["handover_committed"] is False
        assert manifest["snapshot_committed"] is True
        assert any("nothing usable" in e for e in manifest["errors"])

    @pytest.mark.asyncio
    async def test_dispatch_error_note_records_no_handover_doc(self, tmp_path, monkeypatch):
        """The lane dispatcher's except-branch returns a
        "dispatch_error: <redacted>" note — the shared gate rejects any string
        starting with that prefix so the note is never committed as a handover
        doc (the pre-fix gap that COMMITTED it verbatim)."""
        lane = _RecordingLaneManager(reply="dispatch_error: Server disconnected")
        app, bb, store, _ = _build_app(tmp_path, monkeypatch, lane_manager=lane)
        try:
            manifest = await app._offboard_agent("scout", reason="delete")
        finally:
            bb.close()
        assert manifest["handover_committed"] is False
        assert manifest["snapshot_committed"] is True
        assert any("nothing usable" in e for e in manifest["errors"])

    @pytest.mark.asyncio
    async def test_drive_commit_failure_records_error_never_raises(self, tmp_path, monkeypatch):
        """A drive/commit failure must land in the manifest, not raise —
        teardown (whatever the caller does next) must still proceed."""
        import src.host.server as server_module

        async def _boom(*a, **kw):
            raise server_module.team_drive.DriveError("disk full")

        monkeypatch.setattr(server_module.team_drive, "commit_file", _boom)

        lane = _RecordingLaneManager(reply="Handover text.")
        app, bb, store, _ = _build_app(tmp_path, monkeypatch, lane_manager=lane)
        try:
            manifest = await app._offboard_agent("scout", reason="delete")
        finally:
            bb.close()
        assert manifest["handover_committed"] is False
        assert manifest["snapshot_committed"] is False
        assert any("commit failed" in e for e in manifest["errors"])

    @pytest.mark.asyncio
    async def test_no_team_drive_backend_wired_records_error_never_raises(self, tmp_path, monkeypatch):
        """Team exists but no drive provisioner is wired (``_drive_repo``
        503s internally) — recorded as an error, not an exception."""
        lane = _RecordingLaneManager(reply="Handover text.")
        app, bb, store, _ = _build_app(tmp_path, monkeypatch, lane_manager=lane, with_team_drive=False)
        try:
            manifest = await app._offboard_agent("scout", reason="delete")
        finally:
            bb.close()
        assert manifest["handover_committed"] is False
        assert manifest["snapshot_committed"] is False
        assert any("drive" in e.lower() for e in manifest["errors"])

    @pytest.mark.asyncio
    async def test_offboard_audit_logged(self, tmp_path, monkeypatch):
        lane = _RecordingLaneManager(reply="Handover text.")
        app, bb, store, _ = _build_app(tmp_path, monkeypatch, lane_manager=lane)
        try:
            await app._offboard_agent("scout", reason="delete")
            log = bb.get_audit_log(agent_id="scout", action="offboard_agent")
        finally:
            bb.close()
        assert log["entries"], "expected an offboard_agent audit row"
        assert log["entries"][0]["field"] == "delete"


# ── POST /mesh/agents/{id}/offboard endpoint ────────────────────────


class TestOffboardEndpoint:
    @pytest.mark.asyncio
    async def test_requires_operator_or_internal_worker_403(self, tmp_path, monkeypatch):
        lane = _RecordingLaneManager(reply="Handover text.")
        app, bb, store, _ = _build_app(tmp_path, monkeypatch, lane_manager=lane)
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
                r = await c.post(
                    "/mesh/agents/scout/offboard",
                    headers=_worker_headers("scout", "scout-token"),
                )
            assert r.status_code == 403
        finally:
            bb.close()

    @pytest.mark.asyncio
    async def test_operator_target_rejected(self, tmp_path, monkeypatch):
        lane = _RecordingLaneManager(reply="Handover text.")
        app, bb, store, _ = _build_app(
            tmp_path, monkeypatch, lane_manager=lane,
            extra_agents={"operator": {"role": "operator"}},
        )
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
                r = await c.post("/mesh/agents/operator/offboard", headers=_op_headers())
            assert r.status_code == 400
        finally:
            bb.close()

    @pytest.mark.asyncio
    async def test_offboard_then_archive_success(self, tmp_path, monkeypatch):
        lane = _RecordingLaneManager(reply="Handover text.")
        container_manager = MagicMock()
        app, bb, store, _ = _build_app(
            tmp_path, monkeypatch, lane_manager=lane, container_manager=container_manager,
        )
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
                r = await c.post("/mesh/agents/scout/offboard", headers=_op_headers())
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["offboarded"] is True
            assert body["archived"] is True
            assert body["manifest"]["handover_committed"] is True
        finally:
            bb.close()
        # The offboard's handover turn ran BEFORE archive's container stop.
        container_manager.stop_agent.assert_called_once()
        assert lane.calls, "handover turn should have been dispatched"

    @pytest.mark.asyncio
    async def test_unknown_agent_404(self, tmp_path, monkeypatch):
        app, bb, store, _ = _build_app(tmp_path, monkeypatch)
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
                r = await c.post("/mesh/agents/ghost/offboard", headers=_op_headers())
            assert r.status_code == 404
        finally:
            bb.close()


# ── Lead-orphan seam: departure re-appoint (offboard + delete) ──────
#
# Mirror of the ADD-side auto-appoint (docs/plans/2026-07-16-autonomous-
# team-delivery.md §1/§3): a departing lead must not leave the team
# leaderless until the next reboot's backfill.


class TestOffboardLeadReappoint:
    @pytest.mark.asyncio
    async def test_offboard_lead_reappoints_remaining_and_wires_standup(self, tmp_path, monkeypatch):
        """Offboarding the lead re-appoints the first OTHER real member and
        rewires the standup cron. The offboarded agent is archived but still
        in ``team_members`` — it is EXCLUDED from candidates (never re-picked
        as its own successor)."""
        from src.host.cron import CronScheduler

        cron = CronScheduler(config_path=str(tmp_path / "cron.json"))
        lane = _RecordingLaneManager(reply="Handover text.")
        app, bb, store, _ = _build_app(
            tmp_path, monkeypatch, lane_manager=lane, cron_scheduler=cron,
            extra_agents={"analyst": {"role": "a"}, "editor": {"role": "e"}},
        )
        # research already has scout; add two more real members, scout leads.
        store.add_member("research", "analyst")
        store.add_member("research", "editor")
        store.set_lead("research", "scout")
        cron.ensure_standup_job("research", "scout")
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
                r = await c.post("/mesh/agents/scout/offboard", headers=_op_headers())
            assert r.status_code == 200, r.text
        finally:
            bb.close()
        # scout excluded → the first OTHER real member (analyst) re-appointed.
        assert store.get_team("research")["lead_agent_id"] == "analyst"
        job = cron.find_standup_job("research")
        assert job is not None
        assert job.agent == "analyst"

    @pytest.mark.asyncio
    async def test_offboard_lead_down_to_one_leaves_no_lead(self, tmp_path, monkeypatch):
        """Offboarding a lead when only one OTHER real member remains leaves
        the team leaderless (the solo remainder self-leads)."""
        lane = _RecordingLaneManager(reply="Handover text.")
        app, bb, store, _ = _build_app(
            tmp_path, monkeypatch, lane_manager=lane,
            extra_agents={"analyst": {"role": "a"}},
        )
        store.add_member("research", "analyst")  # research = {scout, analyst}
        store.set_lead("research", "scout")
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
                r = await c.post("/mesh/agents/scout/offboard", headers=_op_headers())
            assert r.status_code == 200, r.text
        finally:
            bb.close()
        assert store.get_team("research")["lead_agent_id"] is None

    @pytest.mark.asyncio
    async def test_offboard_non_lead_leaves_lead_unchanged(self, tmp_path, monkeypatch):
        """Offboarding a NON-lead member never touches the surviving lead."""
        lane = _RecordingLaneManager(reply="Handover text.")
        app, bb, store, _ = _build_app(
            tmp_path, monkeypatch, lane_manager=lane,
            extra_agents={"analyst": {"role": "a"}, "editor": {"role": "e"}},
        )
        store.add_member("research", "analyst")
        store.add_member("research", "editor")
        store.set_lead("research", "analyst")  # analyst leads; offboard scout
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
                r = await c.post("/mesh/agents/scout/offboard", headers=_op_headers())
            assert r.status_code == 200, r.text
        finally:
            bb.close()
        assert store.get_team("research")["lead_agent_id"] == "analyst"

    @pytest.mark.asyncio
    async def test_offboard_reappoint_failure_does_not_fail_offboard(self, tmp_path, monkeypatch):
        """Best-effort: a ``set_lead`` failure during the departure re-appoint
        must never fail the offboard response."""
        from unittest.mock import patch

        lane = _RecordingLaneManager(reply="Handover text.")
        app, bb, store, _ = _build_app(
            tmp_path, monkeypatch, lane_manager=lane,
            extra_agents={"analyst": {"role": "a"}, "editor": {"role": "e"}},
        )
        store.add_member("research", "analyst")
        store.add_member("research", "editor")
        store.set_lead("research", "scout")
        try:
            with patch.object(store, "set_lead", side_effect=ValueError("boom")):
                async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
                    r = await c.post("/mesh/agents/scout/offboard", headers=_op_headers())
            assert r.status_code == 200, r.text
            assert r.json()["offboarded"] is True
        finally:
            bb.close()


class TestDeleteLeadReappoint:
    @pytest.mark.asyncio
    async def test_delete_lead_reappoints_remaining_member(self, tmp_path, monkeypatch):
        """Deleting a lead (archive → propose-delete → confirm) re-appoints
        the first remaining real member for its former team — the delete path
        clears the lead via ``remove_agent`` and would otherwise orphan it."""
        # ``_remove_agent`` opens its OWN TeamStore handle via
        # ``_open_teams_store()`` (``OPENLEGION_TEAMS_DB``); point it at the
        # SAME file the app's store uses so the delete's membership write is
        # visible to the endpoint's re-appoint (in production both resolve to
        # one ``data/teams.db``).
        monkeypatch.setenv("OPENLEGION_TEAMS_DB", str(tmp_path / "teams.db"))
        lane = _RecordingLaneManager(reply="Handover text.")
        container_manager = MagicMock()
        app, bb, store, _ = _build_app(
            tmp_path, monkeypatch, lane_manager=lane, container_manager=container_manager,
            extra_agents={"analyst": {"role": "a"}, "editor": {"role": "e"}},
        )
        store.add_member("research", "analyst")
        store.add_member("research", "editor")
        store.set_lead("research", "scout")  # scout leads; archive keeps the pointer
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
                r = await c.post("/mesh/agents/scout/archive", headers=_op_headers())
                assert r.status_code == 200, r.text
                r = await c.post(
                    "/mesh/agents/scout/propose-delete", headers=_human_origin_headers(),
                )
                assert r.status_code == 200, r.text
                nonce = r.json()["change_id"]
                digest = r.json()["payload_digest"]
                r = await c.post(
                    "/mesh/config/confirm",
                    json={"change_id": nonce, "payload_digest": digest},
                    headers=_human_origin_headers(),
                )
                assert r.status_code == 200, r.text
                assert r.json()["deleted"] == "agent"
        finally:
            bb.close()
        assert store.team_of("scout") is None
        assert store.get_team("research")["lead_agent_id"] == "analyst"


# ── ORDER PROOF: mesh delete-confirm chain ──────────────────────────


class TestMeshDeleteOrderProof:
    @pytest.mark.asyncio
    async def test_offboard_attempted_before_volume_destruction(self, tmp_path, monkeypatch):
        """THE order invariant: the handover turn (offboard) must be
        dispatched strictly before ``stop_agent(..., remove_data=True)``
        in the mesh propose/confirm delete chain."""
        order: list[str] = []
        lane = _RecordingLaneManager(reply="Handover text.", order=order)
        container_manager = MagicMock()

        def _stop_agent(agent_id, remove_data=False):
            if remove_data:
                order.append("stop_agent_remove_data_true")

        container_manager.stop_agent = MagicMock(side_effect=_stop_agent)

        app, bb, store, _ = _build_app(
            tmp_path, monkeypatch, lane_manager=lane, container_manager=container_manager,
        )
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
                r = await c.post("/mesh/agents/scout/archive", headers=_op_headers())
                assert r.status_code == 200, r.text
                r = await c.post(
                    "/mesh/agents/scout/propose-delete", headers=_human_origin_headers(),
                )
                assert r.status_code == 200, r.text
                nonce = r.json()["change_id"]
                digest = r.json()["payload_digest"]
                r = await c.post(
                    "/mesh/config/confirm",
                    json={"change_id": nonce, "payload_digest": digest},
                    headers=_human_origin_headers(),
                )
                assert r.status_code == 200, r.text
                body = r.json()
        finally:
            bb.close()
        assert body["deleted"] == "agent"
        assert body["offboard"]["handover_committed"] is True
        assert order == ["handover_turn", "stop_agent_remove_data_true"], (
            f"offboard must precede volume destruction, got order={order}"
        )
        # The archive step (earlier in this same flow) also calls
        # stop_agent WITHOUT remove_data — assert the DELETE step's call
        # (the volume-destroying one) landed, in addition to the ordering
        # already proven above via ``order``.
        container_manager.stop_agent.assert_any_call("scout", remove_data=True)


# ── manage_agent(action="offboard") ─────────────────────────────────


class TestManageAgentOffboardAction:
    @pytest.fixture(autouse=True)
    def _set_operator_env(self, monkeypatch):
        monkeypatch.setenv("ALLOWED_TOOLS", "manage_agent")

    @pytest.mark.asyncio
    async def test_manage_agent_offboard_calls_mesh_client(self):
        from src.agent.builtins.operator_tools import manage_agent

        mc = MagicMock()
        mc.offboard_agent = AsyncMock(
            return_value={"offboarded": True, "manifest": {"handover_committed": True}, "archived": True},
        )
        messages = [{"role": "user", "content": "yes", "_origin": "user"}]
        result = await manage_agent("scout", "offboard", mesh_client=mc, _messages=messages)
        mc.offboard_agent.assert_awaited_once_with("scout")
        assert result["offboarded"] is True

    @pytest.mark.asyncio
    async def test_manage_agent_offboard_blocks_operator_target(self):
        from src.agent.builtins.operator_tools import manage_agent

        messages = [{"role": "user", "content": "yes", "_origin": "user"}]
        result = await manage_agent("operator", "offboard", mesh_client=MagicMock(), _messages=messages)
        assert "operator" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_manage_agent_offboard_surfaces_mesh_error(self):
        from src.agent.builtins.operator_tools import manage_agent

        mc = MagicMock()
        mc.offboard_agent = AsyncMock(side_effect=RuntimeError("500: boom"))
        messages = [{"role": "user", "content": "yes", "_origin": "user"}]
        result = await manage_agent("scout", "offboard", mesh_client=mc, _messages=messages)
        assert "error" in result


# ── FIX 8: RefMoved retry on offboard drive commits ─────────────────


class TestOffboardCommitRefMovedRetry:
    @pytest.mark.asyncio
    async def test_refmoved_twice_then_success_still_commits_no_error(self, tmp_path, monkeypatch):
        """A concurrent commit that loses the CAS (RefMoved) on the first two
        attempts is retried; the third succeeds, so the handover/snapshot land
        with NO manifest error (pre-fix: a single RefMoved dropped the doc)."""
        import src.host.server as server_module

        real_commit = server_module.team_drive.commit_file
        state = {"remaining_failures": 2}

        async def _flaky_commit(*a, **kw):
            if state["remaining_failures"] > 0:
                state["remaining_failures"] -= 1
                raise server_module.team_drive.RefMoved("main moved — retry")
            return await real_commit(*a, **kw)

        monkeypatch.setattr(server_module.team_drive, "commit_file", _flaky_commit)

        lane = _RecordingLaneManager(reply="Handover text.")
        app, bb, store, _ = _build_app(tmp_path, monkeypatch, lane_manager=lane)
        try:
            manifest = await app._offboard_agent("scout", reason="delete")
        finally:
            bb.close()
        assert manifest["handover_committed"] is True
        assert manifest["snapshot_committed"] is True
        assert manifest["errors"] == []

    @pytest.mark.asyncio
    async def test_refmoved_always_records_error_after_exactly_three_attempts(self, tmp_path, monkeypatch):
        """A commit that always loses the CAS is retried EXACTLY 3 times per
        call before the loss surfaces as a manifest error (never an
        exception); both the handover and snapshot paths make 3 attempts."""
        import src.host.server as server_module

        attempts: list[str] = []

        async def _always_moved(*a, **kw):
            msg = kw.get("message", "")
            attempts.append("handover" if "handover" in msg else "snapshot")
            raise server_module.team_drive.RefMoved("main moved — retry")

        monkeypatch.setattr(server_module.team_drive, "commit_file", _always_moved)

        lane = _RecordingLaneManager(reply="Handover text.")
        app, bb, store, _ = _build_app(tmp_path, monkeypatch, lane_manager=lane)
        try:
            manifest = await app._offboard_agent("scout", reason="delete")
        finally:
            bb.close()
        assert manifest["handover_committed"] is False
        assert manifest["snapshot_committed"] is False
        assert attempts.count("handover") == 3
        assert attempts.count("snapshot") == 3
        assert sum("commit failed" in e for e in manifest["errors"]) == 2


# ── FIX 5a: offboarding a LEAD clears the leadership pointer ─────────


class TestOffboardClearsLeadership:
    @pytest.mark.asyncio
    async def test_offboard_of_lead_clears_pointer_and_removes_standup_job(self, tmp_path, monkeypatch):
        """Offboard = departure: a departing LEAD stops being lead. The
        endpoint must clear ``teams.lead_agent_id`` and remove the team's
        standup cron (otherwise a ghost lead lingers in the Team Room and the
        boot reconcile recreates the standup for the archived agent)."""
        cron = MagicMock()
        lane = _RecordingLaneManager(reply="Handover text.")
        app, bb, store, _ = _build_app(
            tmp_path, monkeypatch, lane_manager=lane, cron_scheduler=cron,
        )
        store.set_lead("research", "scout")
        assert store.get_team("research")["lead_agent_id"] == "scout"
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
                r = await c.post("/mesh/agents/scout/offboard", headers=_op_headers())
            assert r.status_code == 200, r.text
            # Pointer cleared — no ghost lead.
            assert store.get_team("research")["lead_agent_id"] is None
            # Standup cron removed for the team (via _sync_standup_job_on_lead_change).
            cron.remove_standup_job.assert_any_call("research")
            # Audit row records the offboard-driven lead clear
            # (get_audit_log's ``agent_id`` filter matches on ``target``).
            log = bb.get_audit_log(agent_id="research", action="team_lead_cleared")
            assert log["entries"], "expected a team_lead_cleared audit row"
        finally:
            bb.close()

    @pytest.mark.asyncio
    async def test_offboard_of_non_lead_leaves_other_lead_untouched(self, tmp_path, monkeypatch):
        """Offboarding a plain member must NOT clear a different agent's
        leadership pointer."""
        cron = MagicMock()
        lane = _RecordingLaneManager(reply="Handover text.")
        app, bb, store, _ = _build_app(
            tmp_path, monkeypatch, lane_manager=lane, cron_scheduler=cron,
            extra_agents={"lead2": {"role": "lead"}},
        )
        store.add_member("research", "lead2")
        store.set_lead("research", "lead2")
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
                r = await c.post("/mesh/agents/scout/offboard", headers=_op_headers())
            assert r.status_code == 200, r.text
            # lead2 is still the lead — offboarding scout (a non-lead) is a no-op here.
            assert store.get_team("research")["lead_agent_id"] == "lead2"
        finally:
            bb.close()
