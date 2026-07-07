"""Team Drive (Phase-2 unit 1) — mesh-hosted git + review-before-integrate.

Covers, per the unit spec:

  * TeamStore drive lifecycle: create provisions + writes ``drive_ref``;
    delete removes the repo; delete→recreate never adopts a stale drive;
    provision failure never fails team creation; boot backfill.
  * The smart-HTTP transport against a REAL ``git`` client over a live
    uvicorn server: clone → commit → push branch OK; push to main
    REJECTED for a worker but allowed for the operator; per-push cap and
    disk-quota 413s; gzip-encoded POST bodies.
  * The auth matrix: member OK, cross-team 403, solo 403, operator OK,
    unauthenticated 401, unknown team 404.
  * Reviews: submit/list, supersede-on-resubmit, operator merge
    (including a merge-conflict 409 and the resolved-review 409), reject.
  * The agent-side ``team_drive`` tool: solo failure envelope,
    clone/branch/sync/submit against the live mesh, main-protection,
    token redaction, teammate-text sanitization.
"""

from __future__ import annotations

import gzip
import importlib
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import uvicorn
import yaml
from httpx import ASGITransport, AsyncClient

from src.host import drive as drive_mod
from src.host.mesh import Blackboard, MessageRouter, PubSub
from src.host.permissions import PermissionMatrix
from src.host.runtime import RuntimeBackend
from src.host.teams import TeamNotFound, TeamStore

TOKENS = {
    "operator": "op-token",
    "member1": "m1-token",
    "member2": "m2-token",
    "outsider": "out-token",
    "solo-a": "solo-token",
}


def _headers(agent: str) -> dict:
    return {"Authorization": f"Bearer {TOKENS[agent]}", "X-Agent-ID": agent}


def _git(cwd, *args, agent: str = "member1", check: bool = True) -> subprocess.CompletedProcess:
    """Run a REAL git client with the per-invocation mesh auth headers."""
    cmd = [
        "git",
        "-c", f"http.extraHeader=Authorization: Bearer {TOKENS[agent]}",
        "-c", f"http.extraHeader=X-Agent-ID: {agent}",
        "-c", "user.name=Test User",
        "-c", "user.email=test@example.com",
        *args,
    ]
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    if check:
        assert result.returncode == 0, f"git {args} failed:\n{result.stdout}\n{result.stderr}"
    return result


@pytest.fixture
def drive_env(tmp_path, monkeypatch):
    """Mesh app + TeamStore with a wired drive provisioner and bearer auth.

    team-x: member1 + member2. team-y: outsider. solo-a: no team.
    """
    monkeypatch.chdir(tmp_path)
    drives_dir = tmp_path / "drives"
    monkeypatch.setenv("OPENLEGION_TEAM_DRIVES_DIR", str(drives_dir))

    perms_file = tmp_path / "permissions.json"
    perms_file.write_text(
        json.dumps(
            {
                "permissions": {
                    "operator": {"blackboard_read": ["*"], "blackboard_write": ["*"]},
                    "member1": {},
                    "member2": {},
                    "outsider": {},
                    "solo-a": {},
                },
            }
        )
    )
    agents_file = tmp_path / "agents.yaml"
    agents_file.write_text(
        yaml.dump(
            {
                "agents": {
                    "operator": {"role": "operator"},
                    "member1": {"role": "w"},
                    "member2": {"role": "w"},
                    "outsider": {"role": "w"},
                    "solo-a": {"role": "w"},
                },
            }
        )
    )
    import src.cli.config as cli_cfg

    monkeypatch.setattr(cli_cfg, "PERMISSIONS_FILE", perms_file)
    monkeypatch.setattr(cli_cfg, "AGENTS_FILE", agents_file)

    import src.host.server as server_module

    importlib.reload(server_module)

    permissions = PermissionMatrix(config_path=str(perms_file))
    router = MessageRouter(permissions, {"operator": "http://op:8400"})
    blackboard = Blackboard(str(tmp_path / "bb.db"))
    pubsub = PubSub()
    store = TeamStore(db_path=str(tmp_path / "teams.db"), teams_dir=tmp_path / "teams")
    store.set_drive_provisioner(drive_mod.ensure_team_drive, drive_mod.remove_team_drive)
    store.create_team("team-x")
    store.add_member("team-x", "member1")
    store.add_member("team-x", "member2")
    store.create_team("team-y")
    store.add_member("team-y", "outsider")

    app = server_module.create_mesh_app(
        blackboard=blackboard,
        pubsub=pubsub,
        router=router,
        permissions=permissions,
        teams_store=store,
        container_manager=MagicMock(),
        auth_tokens=dict(TOKENS),
    )
    yield {
        "app": app,
        "store": store,
        "blackboard": blackboard,
        "drives_dir": drives_dir,
        "tmp_path": tmp_path,
    }
    blackboard.close()
    importlib.reload(server_module)


@pytest.fixture
def live_server(drive_env):
    """Real uvicorn server on an ephemeral loopback port — the git smart-HTTP
    round trip needs a genuine HTTP listener for the real git client."""
    config = uvicorn.Config(
        drive_env["app"], host="127.0.0.1", port=0, log_level="warning", lifespan="off"
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 20
    while not server.started:
        if time.time() > deadline:
            raise RuntimeError("uvicorn test server failed to start")
        time.sleep(0.05)
    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)


def _drive_url(base: str, team: str = "team-x") -> str:
    return f"{base}/mesh/teams/{team}/drive"


# ── TeamStore lifecycle ──────────────────────────────────────────────


class TestDriveProvisioning:
    def test_create_team_provisions_bare_repo_and_drive_ref(self, drive_env):
        store, drives = drive_env["store"], drive_env["drives_dir"]
        team = store.get_team("team-x")
        repo = drives / "team-x.git"
        assert team["drive_ref"] == str(repo.resolve())
        assert (repo / "HEAD").exists()
        # HEAD points at main and main is seeded.
        head = (repo / "HEAD").read_text().strip()
        assert head == "ref: refs/heads/main"
        log = subprocess.run(
            ["git", "log", "--oneline", "main"], cwd=repo, capture_output=True, text=True
        )
        assert log.returncode == 0 and "Initialize Team Drive" in log.stdout

    def test_pre_receive_hook_installed_executable(self, drive_env):
        hook = drive_env["drives_dir"] / "team-x.git" / "hooks" / "pre-receive"
        assert hook.exists()
        assert os.access(hook, os.X_OK)
        text = hook.read_text()
        assert text.startswith("#!/bin/sh")
        assert "OL_DRIVE_PRIVILEGED" in text and "refs/heads/main" in text

    def test_receive_max_input_size_configured(self, drive_env):
        repo = drive_env["drives_dir"] / "team-x.git"
        out = subprocess.run(
            ["git", "config", "receive.maxInputSize"], cwd=repo, capture_output=True, text=True
        )
        assert int(out.stdout.strip()) == 64 * 1024 * 1024

    def test_ensure_is_idempotent(self, drive_env):
        store = drive_env["store"]
        ref1 = store.get_team("team-x")["drive_ref"]
        ref2 = drive_mod.ensure_team_drive("team-x")
        assert ref1 == ref2
        # Seeded commit not duplicated.
        log = subprocess.run(
            ["git", "rev-list", "--count", "main"], cwd=ref1, capture_output=True, text=True
        )
        assert log.stdout.strip() == "1"

    def test_delete_team_removes_repo_and_reviews(self, drive_env):
        store, drives = drive_env["store"], drive_env["drives_dir"]
        # Seed a review row so the sweep is observable.
        repo = Path(store.get_team("team-y")["drive_ref"])
        subprocess.run(
            ["git", "update-ref", "refs/heads/feat", "refs/heads/main"], cwd=repo, check=True
        )
        store.create_review("team-y", "feat", "outsider", "t")
        store.delete_team("team-y")
        assert not (drives / "team-y.git").exists()
        assert store.list_reviews("team-y") == []

    def test_delete_recreate_wipes_stale_drive(self, drive_env):
        """A leftover repo dir from a failed delete must never be adopted
        by a NEW team of the same name (Phase-1 finding #4 precedent)."""
        store, drives = drive_env["store"], drive_env["drives_dir"]
        repo = drives / "team-y.git"
        sentinel = repo / "stale-sentinel"
        sentinel.write_text("old team data")
        # Simulate a failed drive removal during delete.
        real_remove = store._drive_remove
        store._drive_remove = lambda team_id: (_ for _ in ()).throw(OSError("disk"))
        store.delete_team("team-y")
        assert sentinel.exists(), "precondition: stale dir survived the delete"
        store._drive_remove = real_remove
        store.create_team("team-y")
        assert not sentinel.exists(), "stale drive adopted by the recreated team"
        assert (repo / "HEAD").exists()

    def test_provision_failure_does_not_fail_create(self, drive_env):
        store = drive_env["store"]

        def _boom(team_id):
            raise RuntimeError("no disk")

        store.set_drive_provisioner(_boom, lambda team_id: None)
        team = store.create_team("team-z")
        assert team["id"] == "team-z"
        assert team["drive_ref"] is None

    def test_boot_backfill_provisions_missing_drives(self, drive_env):
        store, drives = drive_env["store"], drive_env["drives_dir"]

        def _boom(team_id):
            raise RuntimeError("no disk")

        store.set_drive_provisioner(_boom, lambda team_id: None)
        store.create_team("team-late")
        assert store.get_team("team-late")["drive_ref"] is None
        # Wire the real provisioner back (the boot path) and backfill.
        store.set_drive_provisioner(drive_mod.ensure_team_drive, drive_mod.remove_team_drive)
        provisioned = store.backfill_drives()
        assert "team-late" in provisioned
        assert (drives / "team-late.git" / "HEAD").exists()
        assert store.get_team("team-late")["drive_ref"] == str((drives / "team-late.git").resolve())

    def test_backfill_adopts_existing_repo_without_wiping(self, drive_env):
        """Backfill is non-destructive: an existing repo (drive_ref lost,
        e.g. the UPDATE failed post-provision) is adopted, not re-inited."""
        store = drive_env["store"]
        repo = Path(store.get_team("team-x")["drive_ref"])
        subprocess.run(
            ["git", "update-ref", "refs/heads/precious", "refs/heads/main"], cwd=repo, check=True
        )
        store._set_drive_ref("team-x", None)
        provisioned = store.backfill_drives()
        assert "team-x" in provisioned
        check = subprocess.run(
            ["git", "rev-parse", "--verify", "refs/heads/precious"],
            cwd=repo, capture_output=True,
        )
        assert check.returncode == 0, "backfill wiped an existing drive"

    def test_ensure_drive_reprovisions_when_dir_missing(self, drive_env):
        store, drives = drive_env["store"], drive_env["drives_dir"]
        import shutil

        shutil.rmtree(drives / "team-x.git")
        ref = store.ensure_drive("team-x")
        assert ref and (Path(ref) / "HEAD").exists()

    def test_ensure_drive_unknown_team_raises(self, drive_env):
        with pytest.raises(TeamNotFound):
            drive_env["store"].ensure_drive("nope")

    def test_runtime_backend_concrete_volume_methods(self, drive_env):
        """The ABC ships CONCRETE host-dir implementations shared by both
        backends — no subclass override needed."""

        class _Minimal(RuntimeBackend):
            def start_agent(self, *a, **k):
                raise NotImplementedError

            def stop_agent(self, *a, **k):
                raise NotImplementedError

            def health_check(self, agent_id):
                return False

            def get_logs(self, agent_id, tail=40):
                return ""

            async def wait_for_agent(self, agent_id, timeout=30):
                return False

        backend = _Minimal()
        ref = backend.ensure_team_volume("team-vol")
        assert ref == str((drive_env["drives_dir"] / "team-vol.git").resolve())
        assert (Path(ref) / "HEAD").exists()
        backend.remove_team_volume("team-vol")
        assert not Path(ref).exists()

    def test_pure_db_mode_skips_provisioning(self, tmp_path):
        store = TeamStore(db_path=":memory:")
        team = store.create_team("quiet")
        assert team["drive_ref"] is None
        assert store.backfill_drives() == []


class TestDriveReviewStore:
    def test_supersede_on_resubmit_same_branch(self, drive_env):
        store = drive_env["store"]
        r1 = store.create_review("team-x", "feat-a", "member1", "first")
        r2 = store.create_review("team-x", "feat-a", "member1", "second")
        assert store.get_review(r1["id"])["status"] == "superseded"
        assert store.get_review(r1["id"])["resolved_at"] is not None
        assert store.get_review(r2["id"])["status"] == "open"
        # A different branch does not supersede.
        r3 = store.create_review("team-x", "feat-b", "member2", "other")
        assert store.get_review(r2["id"])["status"] == "open"
        assert store.get_review(r3["id"])["status"] == "open"

    def test_list_reviews_filters_by_status(self, drive_env):
        store = drive_env["store"]
        r1 = store.create_review("team-x", "feat-a", "member1", "t1")
        store.create_review("team-x", "feat-b", "member1", "t2")
        store.resolve_review(r1["id"], "rejected", reviewer="operator")
        assert {r["branch"] for r in store.list_reviews("team-x", "open")} == {"feat-b"}
        assert {r["branch"] for r in store.list_reviews("team-x", "rejected")} == {"feat-a"}
        assert len(store.list_reviews("team-x")) == 2

    def test_resolve_guards(self, drive_env):
        store = drive_env["store"]
        r = store.create_review("team-x", "feat-a", "member1", "t")
        with pytest.raises(ValueError):
            store.resolve_review(r["id"], "banana", reviewer="operator")
        store.resolve_review(r["id"], "merged", reviewer="operator", merge_commit="abc123")
        row = store.get_review(r["id"])
        assert row["status"] == "merged" and row["merge_commit"] == "abc123"
        with pytest.raises(ValueError):
            store.resolve_review(r["id"], "rejected", reviewer="operator")
        with pytest.raises(ValueError):
            store.resolve_review("rev_missing", "merged", reviewer="operator")

    def test_create_review_unknown_team(self, drive_env):
        with pytest.raises(TeamNotFound):
            drive_env["store"].create_review("ghost", "b", "a", "t")


# ── Auth matrix (ASGI level) ─────────────────────────────────────────


class TestDriveAuthMatrix:
    async def _info_refs(self, app, team: str, agent: str | None, service="git-upload-pack"):
        headers = _headers(agent) if agent else {}
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            return await c.get(
                f"/mesh/teams/{team}/drive/info/refs",
                params={"service": service},
                headers=headers,
            )

    @pytest.mark.asyncio
    async def test_member_ok(self, drive_env):
        resp = await self._info_refs(drive_env["app"], "team-x", "member1")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/x-git-upload-pack-advertisement")
        assert resp.content.startswith(b"001e# service=git-upload-pack\n0000")

    @pytest.mark.asyncio
    async def test_cross_team_403(self, drive_env):
        resp = await self._info_refs(drive_env["app"], "team-x", "outsider")
        assert resp.status_code == 403
        assert "not a member" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_solo_403_directive(self, drive_env):
        resp = await self._info_refs(drive_env["app"], "team-x", "solo-a")
        assert resp.status_code == 403
        assert "not on a team" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_operator_ok(self, drive_env):
        resp = await self._info_refs(drive_env["app"], "team-x", "operator")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_unauthenticated_401(self, drive_env):
        resp = await self._info_refs(drive_env["app"], "team-x", None)
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_unknown_team_404(self, drive_env):
        resp = await self._info_refs(drive_env["app"], "ghost-team", "operator")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_bad_service_400(self, drive_env):
        resp = await self._info_refs(drive_env["app"], "team-x", "member1", service="rm -rf")
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_review_submit_cross_team_403(self, drive_env):
        async with AsyncClient(transport=ASGITransport(app=drive_env["app"]), base_url="http://t") as c:
            resp = await c.post(
                "/mesh/teams/team-x/drive/reviews",
                json={"branch": "feat", "title": "t"},
                headers=_headers("outsider"),
            )
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_merge_reject_are_operator_only(self, drive_env):
        store = drive_env["store"]
        review = store.create_review("team-x", "feat-gate", "member1", "t")
        async with AsyncClient(transport=ASGITransport(app=drive_env["app"]), base_url="http://t") as c:
            for verb in ("merge", "reject"):
                resp = await c.post(
                    f"/mesh/teams/team-x/drive/reviews/{review['id']}/{verb}",
                    headers=_headers("member1"),
                )
                assert resp.status_code == 403, f"{verb} must be operator-or-internal"

    @pytest.mark.asyncio
    async def test_review_list_member_ok(self, drive_env):
        async with AsyncClient(transport=ASGITransport(app=drive_env["app"]), base_url="http://t") as c:
            resp = await c.get("/mesh/teams/team-x/drive/reviews", headers=_headers("member2"))
        assert resp.status_code == 200
        assert "reviews" in resp.json()

    @pytest.mark.asyncio
    async def test_review_validation(self, drive_env):
        async with AsyncClient(transport=ASGITransport(app=drive_env["app"]), base_url="http://t") as c:
            for payload, needle in (
                ({"branch": "main", "title": "t"}, "integration target"),
                ({"branch": "--evil", "title": "t"}, "branch"),
                ({"branch": "a..b", "title": "t"}, "branch"),
                ({"branch": "feat", "title": ""}, "title"),
                ({"branch": "never-pushed", "title": "t"}, "does not exist"),
            ):
                resp = await c.post(
                    "/mesh/teams/team-x/drive/reviews",
                    json=payload,
                    headers=_headers("member1"),
                )
                assert resp.status_code == 400, payload
                assert needle in str(resp.json()["detail"])


# ── Real git smart-HTTP round trip (live uvicorn) ────────────────────


class TestSmartHttpRoundTrip:
    def test_clone_commit_push_branch(self, live_server, drive_env, tmp_path):
        clone = tmp_path / "wc-m1"
        _git(None, "clone", _drive_url(live_server), str(clone))
        assert (clone / "README.md").exists()
        _git(clone, "switch", "-c", "feat-doc")
        (clone / "notes.md").write_text("# findings\n")
        _git(clone, "add", "-A")
        _git(clone, "commit", "-m", "add notes")
        _git(clone, "push", "origin", "feat-doc")
        # The branch landed on the bare repo.
        repo = drive_env["store"].get_team("team-x")["drive_ref"]
        check = subprocess.run(
            ["git", "rev-parse", "--verify", "refs/heads/feat-doc"],
            cwd=repo, capture_output=True,
        )
        assert check.returncode == 0
        # Audit trail recorded the push.
        audit = drive_env["blackboard"].get_audit_log(action="drive_push")
        assert audit["total"] >= 1

    def test_worker_push_to_main_rejected(self, live_server, tmp_path):
        clone = tmp_path / "wc-main"
        _git(None, "clone", _drive_url(live_server), str(clone))
        (clone / "sneak.md").write_text("straight to main\n")
        _git(clone, "add", "-A")
        _git(clone, "commit", "-m", "sneaky")
        result = _git(clone, "push", "origin", "main", check=False)
        assert result.returncode != 0
        combined = result.stdout + result.stderr
        assert "protected" in combined
        # main did not move on a fresh clone.
        verify = tmp_path / "wc-verify"
        _git(None, "clone", _drive_url(live_server), str(verify), agent="member2")
        assert not (verify / "sneak.md").exists()

    def test_operator_can_push_main(self, live_server, tmp_path):
        clone = tmp_path / "wc-op"
        _git(None, "clone", _drive_url(live_server), str(clone), agent="operator")
        (clone / "policy.md").write_text("operator-integrated\n")
        _git(clone, "add", "-A", agent="operator")
        _git(clone, "commit", "-m", "operator update", agent="operator")
        _git(clone, "push", "origin", "main", agent="operator")

    def test_cross_team_clone_denied(self, live_server, tmp_path):
        result = _git(
            None, "clone", _drive_url(live_server), str(tmp_path / "wc-out"),
            agent="outsider", check=False,
        )
        assert result.returncode != 0

    def test_push_cap_413(self, live_server, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENLEGION_DRIVE_PUSH_MAX_MB", "1")
        clone = tmp_path / "wc-big"
        _git(None, "clone", _drive_url(live_server), str(clone))
        _git(clone, "switch", "-c", "feat-big")
        (clone / "blob.bin").write_bytes(os.urandom(3 * 1024 * 1024))
        _git(clone, "add", "-A")
        _git(clone, "commit", "-m", "big blob")
        result = _git(clone, "push", "origin", "feat-big", check=False)
        assert result.returncode != 0
        assert "413" in (result.stdout + result.stderr)

    def test_quota_reject_413(self, live_server, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENLEGION_DRIVE_QUOTA_MB", "1")
        clone = tmp_path / "wc-quota"
        _git(None, "clone", _drive_url(live_server), str(clone))
        _git(clone, "switch", "-c", "feat-fill")
        # First push: repo still under 1 MB → allowed, lands ~2 MB.
        (clone / "fill.bin").write_bytes(os.urandom(2 * 1024 * 1024))
        _git(clone, "add", "-A")
        _git(clone, "commit", "-m", "fill")
        _git(clone, "push", "origin", "feat-fill")
        # Second push: repo now over quota → rejected before receive-pack.
        (clone / "more.bin").write_bytes(os.urandom(256 * 1024))
        _git(clone, "add", "-A")
        _git(clone, "commit", "-m", "more")
        result = _git(clone, "push", "origin", "feat-fill", check=False)
        assert result.returncode != 0
        assert "413" in (result.stdout + result.stderr)


class TestGzipBodies:
    @pytest.mark.asyncio
    async def test_gzip_upload_pack_body_accepted(self, drive_env):
        # A gzip-compressed flush-pkt: upload-pack ends the session cleanly.
        async with AsyncClient(transport=ASGITransport(app=drive_env["app"]), base_url="http://t") as c:
            resp = await c.post(
                "/mesh/teams/team-x/drive/git-upload-pack",
                content=gzip.compress(b"0000"),
                headers={**_headers("member1"), "Content-Encoding": "gzip"},
            )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_gzip_bomb_413_on_decompressed_size(self, drive_env, monkeypatch):
        monkeypatch.setenv("OPENLEGION_DRIVE_PUSH_MAX_MB", "1")
        bomb = gzip.compress(b"\x00" * (8 * 1024 * 1024))  # tiny on the wire
        async with AsyncClient(transport=ASGITransport(app=drive_env["app"]), base_url="http://t") as c:
            resp = await c.post(
                "/mesh/teams/team-x/drive/git-receive-pack",
                content=bomb,
                headers={**_headers("member1"), "Content-Encoding": "gzip"},
            )
        assert resp.status_code == 413
        assert "decompressed" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_invalid_gzip_400(self, drive_env):
        async with AsyncClient(transport=ASGITransport(app=drive_env["app"]), base_url="http://t") as c:
            resp = await c.post(
                "/mesh/teams/team-x/drive/git-receive-pack",
                content=b"definitely not gzip",
                headers={**_headers("member1"), "Content-Encoding": "gzip"},
            )
        assert resp.status_code == 400

    def test_gunzip_capped_helper(self):
        data = b"x" * 1000
        assert drive_mod.gunzip_capped(gzip.compress(data), 1000) == data
        with pytest.raises(ValueError):
            drive_mod.gunzip_capped(gzip.compress(data), 999)

    def test_pkt_line(self):
        assert drive_mod.pkt_line(b"# service=git-upload-pack\n") == b"001e# service=git-upload-pack\n"


# ── Review flow end-to-end ───────────────────────────────────────────


class TestReviewFlow:
    def _push_branch(self, live_server, workdir, branch: str, filename: str, content: str,
                     agent: str = "member1", base: str | None = None):
        clone = workdir / f"wc-{branch}-{agent}"
        _git(None, "clone", _drive_url(live_server), str(clone), agent=agent)
        if base:
            _git(clone, "reset", "--hard", base, agent=agent)
        _git(clone, "switch", "-c", branch, agent=agent)
        (clone / filename).write_text(content)
        _git(clone, "add", "-A", agent=agent)
        _git(clone, "commit", "-m", f"work on {branch}", agent=agent)
        _git(clone, "push", "origin", branch, agent=agent)
        return clone

    def _post(self, base: str, path: str, agent: str, payload: dict | None = None):
        import httpx

        return httpx.post(
            f"{base}{path}",
            json=payload,
            headers=_headers(agent),
            timeout=30,
        )

    def test_submit_merge_and_integration_visible(self, live_server, drive_env, tmp_path):
        self._push_branch(live_server, tmp_path, "feat-report", "report.md", "# Q3 report\n")
        resp = self._post(
            live_server, "/mesh/teams/team-x/drive/reviews", "member1",
            {"branch": "feat-report", "title": "Q3 report", "summary": "adds the report"},
        )
        assert resp.status_code == 200
        review = resp.json()["review"]
        assert review["status"] == "open" and review["author"] == "member1"

        merge = self._post(
            live_server, f"/mesh/teams/team-x/drive/reviews/{review['id']}/merge", "operator",
        )
        assert merge.status_code == 200, merge.text
        body = merge.json()
        assert body["merged"] is True and body["merge_commit"]
        assert body["review"]["status"] == "merged"
        assert body["review"]["reviewer"] == "operator"

        # A teammate's fresh clone sees the integrated file on main.
        verify = tmp_path / "wc-after-merge"
        _git(None, "clone", _drive_url(live_server), str(verify), agent="member2")
        assert (verify / "report.md").read_text() == "# Q3 report\n"

        # The merged review cannot be re-merged.
        again = self._post(
            live_server, f"/mesh/teams/team-x/drive/reviews/{review['id']}/merge", "operator",
        )
        assert again.status_code == 409

    def test_merge_conflict_409(self, live_server, drive_env, tmp_path):
        # Two branches from the same base editing the same file differently;
        # merge the first, then the second conflicts.
        self._push_branch(live_server, tmp_path, "conf-a", "README.md", "version A\n")
        self._push_branch(live_server, tmp_path, "conf-b", "README.md", "version B\n")
        r_a = self._post(
            live_server, "/mesh/teams/team-x/drive/reviews", "member1",
            {"branch": "conf-a", "title": "A"},
        ).json()["review"]
        r_b = self._post(
            live_server, "/mesh/teams/team-x/drive/reviews", "member2",
            {"branch": "conf-b", "title": "B"},
        ).json()["review"]
        assert self._post(
            live_server, f"/mesh/teams/team-x/drive/reviews/{r_a['id']}/merge", "operator",
        ).status_code == 200
        conflict = self._post(
            live_server, f"/mesh/teams/team-x/drive/reviews/{r_b['id']}/merge", "operator",
        )
        assert conflict.status_code == 409
        detail = conflict.json()["detail"]
        assert detail["error"] == "merge_conflict"
        assert "README.md" in detail["files"]
        # The conflicted review stays open for rework.
        assert drive_env["store"].get_review(r_b["id"])["status"] == "open"

    def test_resubmit_supersedes_via_endpoint(self, live_server, drive_env, tmp_path):
        self._push_branch(live_server, tmp_path, "feat-again", "a.md", "v1\n")
        first = self._post(
            live_server, "/mesh/teams/team-x/drive/reviews", "member1",
            {"branch": "feat-again", "title": "v1"},
        ).json()["review"]
        second = self._post(
            live_server, "/mesh/teams/team-x/drive/reviews", "member1",
            {"branch": "feat-again", "title": "v2"},
        ).json()["review"]
        store = drive_env["store"]
        assert store.get_review(first["id"])["status"] == "superseded"
        assert store.get_review(second["id"])["status"] == "open"
        # Merging the superseded one 409s.
        stale = self._post(
            live_server, f"/mesh/teams/team-x/drive/reviews/{first['id']}/merge", "operator",
        )
        assert stale.status_code == 409

    def test_reject_flow(self, live_server, drive_env, tmp_path):
        self._push_branch(live_server, tmp_path, "feat-nope", "n.md", "no\n")
        review = self._post(
            live_server, "/mesh/teams/team-x/drive/reviews", "member1",
            {"branch": "feat-nope", "title": "nope"},
        ).json()["review"]
        rej = self._post(
            live_server, f"/mesh/teams/team-x/drive/reviews/{review['id']}/reject", "operator",
        )
        assert rej.status_code == 200
        assert rej.json()["review"]["status"] == "rejected"
        # Terminal: a second reject 409s; the branch survives for rework.
        assert self._post(
            live_server, f"/mesh/teams/team-x/drive/reviews/{review['id']}/reject", "operator",
        ).status_code == 409


# ── Agent-side team_drive tool ───────────────────────────────────────


class TestDriveTool:
    @pytest.fixture
    def tool_env(self, live_server, tmp_path, monkeypatch):
        from src.agent.builtins import drive_tool

        clone_dir = tmp_path / "agent-clone" / "drive"
        monkeypatch.setattr(drive_tool, "_DRIVE_PATH", str(clone_dir))
        monkeypatch.setenv("MESH_URL", live_server)
        monkeypatch.setenv("TEAM_NAME", "team-x")
        monkeypatch.setenv("AGENT_ID", "member1")
        monkeypatch.setenv("MESH_AUTH_TOKEN", TOKENS["member1"])
        return drive_tool

    @pytest.mark.asyncio
    async def test_solo_agent_gets_directive_envelope(self, monkeypatch):
        from src.agent.builtins import drive_tool

        monkeypatch.setenv("MESH_URL", "http://mesh:8420")
        monkeypatch.setenv("TEAM_NAME", "solo-a")
        monkeypatch.setenv("AGENT_ID", "solo-a")
        result = await drive_tool.team_drive("clone")
        assert result["ok"] is False
        assert "not on a team" in result["error"]
        assert "recovery_hint" in result

    @pytest.mark.asyncio
    async def test_unknown_action_envelope(self, tool_env):
        result = await tool_env.team_drive("explode")
        assert result["ok"] is False and "recovery_hint" in result

    @pytest.mark.asyncio
    async def test_clone_branch_sync_flow(self, tool_env, drive_env):
        result = await tool_env.team_drive("clone")
        assert result["ok"] is True, result
        assert (Path(tool_env._DRIVE_PATH) / "README.md").exists()
        # Idempotent clone.
        again = await tool_env.team_drive("clone")
        assert again["ok"] is True and "already" in again.get("note", "")

        # Sync from main is refused before any branch exists.
        on_main = await tool_env.team_drive("sync", message="oops")
        assert on_main["ok"] is False and "main" in on_main["error"]

        assert (await tool_env.team_drive("branch", branch="tool-work"))["ok"] is True
        (Path(tool_env._DRIVE_PATH) / "tool.md").write_text("from the tool\n")
        synced = await tool_env.team_drive("sync", message="tool output")
        assert synced["ok"] is True and synced["branch"] == "tool-work"
        repo = drive_env["store"].get_team("team-x")["drive_ref"]
        check = subprocess.run(
            ["git", "rev-parse", "--verify", "refs/heads/tool-work"],
            cwd=repo, capture_output=True,
        )
        assert check.returncode == 0

        status = await tool_env.team_drive("status")
        assert status["ok"] is True
        log = await tool_env.team_drive("log")
        assert log["ok"] is True and "tool output" in log["log"]
        pulled = await tool_env.team_drive("pull")
        assert pulled["ok"] is True

    @pytest.mark.asyncio
    async def test_sync_requires_message_and_branch_validation(self, tool_env):
        assert (await tool_env.team_drive("clone"))["ok"] is True
        assert (await tool_env.team_drive("branch", branch="needs-msg"))["ok"] is True
        no_msg = await tool_env.team_drive("sync", message="   ")
        assert no_msg["ok"] is False and "message" in no_msg["error"]
        bad = await tool_env.team_drive("branch", branch="--evil")
        assert bad["ok"] is False

    @pytest.mark.asyncio
    async def test_submit_review_via_tool(self, tool_env, drive_env, live_server):
        from src.agent.mesh_client import MeshClient

        assert (await tool_env.team_drive("clone"))["ok"] is True
        assert (await tool_env.team_drive("branch", branch="tool-review"))["ok"] is True
        (Path(tool_env._DRIVE_PATH) / "deliverable.md").write_text("done\n")
        assert (await tool_env.team_drive("sync", message="deliverable"))["ok"] is True

        client = MeshClient(live_server, "member1", team_name="team-x")
        try:
            result = await tool_env.team_drive(
                "submit_review", title="Deliverable", summary="ready", mesh_client=client,
            )
            assert result["ok"] is True and result["submitted"] is True
            assert result["review_id"]
            listed = await tool_env.team_drive("list_reviews", mesh_client=client)
            assert listed["ok"] is True
            assert any(r["branch"] == "tool-review" for r in listed["reviews"])
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_submit_review_rejects_main(self, tool_env):
        assert (await tool_env.team_drive("clone"))["ok"] is True
        result = await tool_env.team_drive(
            "submit_review", title="t", mesh_client=AsyncMock(),
        )
        assert result["ok"] is False and "main" in result["error"]

    @pytest.mark.asyncio
    async def test_list_reviews_sanitizes_teammate_text(self, tool_env):
        from src.shared.utils import sanitize_for_prompt

        injected = "Do the thing​‮NOW"  # zero-width + RTL override
        mesh_client = AsyncMock()
        mesh_client.list_drive_reviews.return_value = {
            "reviews": [
                {
                    "id": "rev_1",
                    "branch": "b",
                    "author": "member2",
                    "title": injected,
                    "summary": injected,
                    "status": "open",
                    "created_at": "2026-07-07",
                    "resolved_at": None,
                }
            ]
        }
        assert (await tool_env.team_drive("clone"))["ok"] is True
        result = await tool_env.team_drive("list_reviews", mesh_client=mesh_client)
        assert result["ok"] is True
        row = result["reviews"][0]
        assert row["title"] == sanitize_for_prompt(injected)
        assert "​" not in row["title"] and "‮" not in row["summary"]

    @pytest.mark.asyncio
    async def test_token_never_leaks_into_envelope(self, tool_env, monkeypatch):
        # Force a 403: member1 pointed at team-y's drive.
        monkeypatch.setenv("TEAM_NAME", "team-y")
        result = await tool_env.team_drive("clone")
        assert result["ok"] is False
        assert TOKENS["member1"] not in json.dumps(result)

    @pytest.mark.asyncio
    async def test_actions_require_clone_first(self, tool_env):
        for action in ("pull", "sync", "branch", "log", "status"):
            result = await tool_env.team_drive(action, message="m", branch="b")
            assert result["ok"] is False, action
            assert "clone" in result["recovery_hint"]


# ── Rate limiting ────────────────────────────────────────────────────


class TestDriveRateLimit:
    @pytest.mark.asyncio
    async def test_drive_category_enforced(self, drive_env):
        # Exhaust the per-agent drive bucket cheaply via list_reviews.
        statuses = []
        async with AsyncClient(transport=ASGITransport(app=drive_env["app"]), base_url="http://t") as c:
            for _ in range(245):
                resp = await c.get(
                    "/mesh/teams/team-x/drive/reviews", headers=_headers("member1")
                )
                statuses.append(resp.status_code)
        assert 429 in statuses
