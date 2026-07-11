"""Kernel-executed auto-merge consumer (plan §8 #20, Phase-5 unit 4).

Two layers, mirroring the module split:

  * ``repo_env`` — a real bare git drive + a real (disk-backed) TeamStore
    + an in-memory TrackRecordStore, driving ``auto_merge.consider_auto_merge``
    DIRECTLY (no FastAPI) — the trust-floor / self-approval / rate-cap /
    concurrency / sampling decision logic, against real git plumbing
    (mirrors ``test_team_drive.py``'s bare-repo fixture pattern, minus the
    live uvicorn server: branches are pushed via a LOCAL filesystem clone,
    since the pre-receive hook only blocks ``refs/heads/main``).
  * ``app_env`` — a full ``create_mesh_app`` ASGI app (mirrors
    ``test_team_drive.py``'s ``drive_env``) for endpoint-level behavior:
    the verdict endpoint scheduling the consumer task, the response
    surviving a consumer explosion, ``lead_verdict_by`` recorded +
    surfaced + feeding ``_record_drive_review_outcome``, and the
    flag-auto-merge / revert-merge endpoints' operator-only gate.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import random
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml
from httpx import ASGITransport, AsyncClient

from src.host import auto_merge
from src.host import drive as drive_mod
from src.host.mesh import Blackboard, MessageRouter, PubSub
from src.host.permissions import PermissionMatrix
from src.host.teams import TeamStore
from src.host.track_record import TrackRecordStore

_MERGE_TREE_OK = drive_mod.git_supports_merge_tree()
_requires_merge_tree = pytest.mark.skipif(
    not _MERGE_TREE_OK, reason="host git too old for `git merge-tree --write-tree` (needs >= 2.38)"
)

_GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "Test",
    "GIT_AUTHOR_EMAIL": "test@example.com",
    "GIT_COMMITTER_NAME": "Test",
    "GIT_COMMITTER_EMAIL": "test@example.com",
    "GIT_TERMINAL_PROMPT": "0",
}


def _push_branch_locally(repo: Path, branch: str, filename: str, content: str) -> str:
    """Create ``branch`` with one commit adding ``filename``, pushed
    directly into the bare repo via a LOCAL filesystem clone (no mesh
    HTTP / live server needed — the pre-receive hook still runs for a
    local push, but only blocks ``refs/heads/main``). Returns the new
    commit sha."""
    with tempfile.TemporaryDirectory() as tmp:
        clone = Path(tmp) / "wc"
        subprocess.run(
            ["git", "-c", "protocol.file.allow=always", "clone", str(repo), str(clone)],
            check=True, capture_output=True, text=True, env=_GIT_ENV,
        )
        subprocess.run(["git", "switch", "-c", branch], cwd=clone, check=True, capture_output=True, env=_GIT_ENV)
        (clone / filename).write_text(content)
        subprocess.run(["git", "add", "-A"], cwd=clone, check=True, capture_output=True, env=_GIT_ENV)
        subprocess.run(
            ["git", "commit", "-m", f"work on {branch}"],
            cwd=clone, check=True, capture_output=True, env=_GIT_ENV,
        )
        result = subprocess.run(
            ["git", "-c", "protocol.file.allow=always", "push", "origin", branch],
            cwd=clone, capture_output=True, text=True, env=_GIT_ENV,
        )
        assert result.returncode == 0, result.stderr
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=clone, capture_output=True, text=True, env=_GIT_ENV,
        ).stdout.strip()
    return sha


def _seed_events(
    track: TrackRecordStore, *, submitter: str, lead: str, outcome: str, count: int = 1,
    rater_kind: str = "human", verdict: str | None = "approve", team_id: str = "team-x",
) -> None:
    for i in range(count):
        details = {"lead_agent_id": lead, "resolution": outcome}
        if verdict is not None:
            details["lead_verdict"] = verdict
        track.record(
            source="drive_review",
            ref_id=f"seed-{submitter}-{lead}-{outcome}-{i}",
            outcome=outcome,
            rater_kind=rater_kind,
            agent_id=submitter,
            team_id=team_id,
            rated_by="operator" if rater_kind == "human" else "policy_engine",
            details=details,
        )


# ── direct-pipeline fixture (no FastAPI) ──────────────────────────────


@pytest.fixture
def repo_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENLEGION_TEAM_DRIVES_DIR", str(tmp_path / "drives"))
    store = TeamStore(db_path=str(tmp_path / "teams.db"), teams_dir=tmp_path / "teams")
    store.set_drive_provisioner(drive_mod.ensure_team_drive, drive_mod.remove_team_drive)
    store.create_team("team-x")
    store.add_member("team-x", "member1")
    store.add_member("team-x", "member2")
    store.set_lead("team-x", "member2")
    track = TrackRecordStore(":memory:")
    repo = Path(store.get_team("team-x")["drive_ref"])
    yield {"store": store, "track": track, "repo": repo, "tmp_path": tmp_path}
    track.close()


def _make_review(repo_env, branch: str, filename: str = "f.md", content: str = "content\n",
                  author: str = "member1") -> dict:
    sha = _push_branch_locally(repo_env["repo"], branch, filename, content)
    return repo_env["store"].create_review("team-x", branch, author, "title", head_sha=sha)


async def _run(repo_env, review, lead="member2", **kw):
    return await auto_merge.consider_auto_merge(
        team_id="team-x",
        review=review,
        lead_verdict_by=lead,
        teams_store=repo_env["store"],
        track_record_store=repo_env["track"],
        repo=repo_env["repo"],
        **kw,
    )


class TestTrustFloor:
    @_requires_merge_tree
    async def test_floor_unmet_no_merge(self, repo_env):
        review = _make_review(repo_env, "feat-unmet")
        await _run(repo_env, review)
        assert repo_env["store"].get_review(review["id"])["status"] == "open"
        assert repo_env["track"].count_events(source="drive_review", outcome="auto_merged") == 0

    @_requires_merge_tree
    async def test_floor_met_lands_auto_merge(self, repo_env):
        store, track, repo = repo_env["store"], repo_env["track"], repo_env["repo"]
        _seed_events(track, submitter="member1", lead="member2", outcome="merged", count=5)
        review = _make_review(repo_env, "feat-met", "a.md", "hello\n")

        await _run(repo_env, review)

        resolved = store.get_review(review["id"])
        assert resolved["status"] == "merged"
        assert resolved["reviewer"] == auto_merge.AUTO_MERGE_RATER
        assert resolved["merge_commit"]

        # The content actually landed on main.
        with tempfile.TemporaryDirectory() as tmp:
            verify = Path(tmp) / "verify"
            subprocess.run(
                ["git", "-c", "protocol.file.allow=always", "clone", str(repo), str(verify)],
                check=True, capture_output=True, env=_GIT_ENV,
            )
            assert (verify / "a.md").read_text() == "hello\n"

        events = [e for e in track.recent_events("member1") if e["outcome"] == "auto_merged"]
        assert len(events) == 1
        assert events[0]["rater_kind"] == "system"
        assert events[0]["rated_by"] == "policy_engine"
        assert events[0]["details"]["lead_agent_id"] == "member2"
        assert events[0]["details"]["lead_verdict_by"] == "member2"

    @_requires_merge_tree
    async def test_self_approval_never_merges(self, repo_env):
        track = repo_env["track"]
        _seed_events(track, submitter="member1", lead="member1", outcome="merged", count=10)
        review = _make_review(repo_env, "feat-self")

        await _run(repo_env, review, lead="member1")

        assert repo_env["store"].get_review(review["id"])["status"] == "open"

    @_requires_merge_tree
    async def test_rejected_after_approve_blocks(self, repo_env):
        track = repo_env["track"]
        _seed_events(track, submitter="member1", lead="member2", outcome="merged", count=5)
        _seed_events(track, submitter="member1", lead="member2", outcome="rejected", count=1)
        review = _make_review(repo_env, "feat-rejafter")

        await _run(repo_env, review)

        assert repo_env["store"].get_review(review["id"])["status"] == "open"

    @_requires_merge_tree
    @pytest.mark.parametrize("decay_outcome", ["auto_merge_flagged", "auto_merge_reverted"])
    async def test_flagged_or_reverted_blocks(self, repo_env, decay_outcome):
        track = repo_env["track"]
        _seed_events(track, submitter="member1", lead="member2", outcome="merged", count=5)
        _seed_events(track, submitter="member1", lead="member2", outcome=decay_outcome, verdict=None, count=1)
        review = _make_review(repo_env, f"feat-{decay_outcome}")

        await _run(repo_env, review)

        assert repo_env["store"].get_review(review["id"])["status"] == "open"

    @_requires_merge_tree
    async def test_system_rated_auto_merges_never_count_toward_floor(self, repo_env):
        """The self-reinforcement pin at the full-pipeline level: seed one
        merge SHORT of the floor plus a pile of system-rated `auto_merged`
        events for the same pair — the pair must stay ineligible."""
        track = repo_env["track"]
        _seed_events(track, submitter="member1", lead="member2", outcome="merged", count=4)
        _seed_events(
            track, submitter="member1", lead="member2", outcome="auto_merged",
            rater_kind="system", count=20,
        )
        review = _make_review(repo_env, "feat-selfreinforce")

        await _run(repo_env, review)

        assert repo_env["store"].get_review(review["id"])["status"] == "open"


class TestDailyCap:
    @_requires_merge_tree
    async def test_daily_cap_blocks(self, repo_env, monkeypatch):
        monkeypatch.setenv("OPENLEGION_AUTO_MERGE_DAILY_CAP", "1")
        track = repo_env["track"]
        _seed_events(track, submitter="member1", lead="member2", outcome="merged", count=5)
        _seed_events(
            track, submitter="somebody-else", lead="lead-x", outcome="auto_merged",
            rater_kind="system", count=1,
        )
        review = _make_review(repo_env, "feat-cap")

        await _run(repo_env, review)

        assert repo_env["store"].get_review(review["id"])["status"] == "open"

    @_requires_merge_tree
    async def test_daily_cap_zero_disables_auto_merge_entirely(self, repo_env, monkeypatch):
        monkeypatch.setenv("OPENLEGION_AUTO_MERGE_DAILY_CAP", "0")
        track = repo_env["track"]
        _seed_events(track, submitter="member1", lead="member2", outcome="merged", count=50)
        review = _make_review(repo_env, "feat-killswitch")

        await _run(repo_env, review)

        assert repo_env["store"].get_review(review["id"])["status"] == "open"


class TestConcurrencyAndSampling:
    @_requires_merge_tree
    async def test_concurrent_human_merge_loses_claim_harmlessly(self, repo_env):
        store, track = repo_env["store"], repo_env["track"]
        _seed_events(track, submitter="member1", lead="member2", outcome="merged", count=5)
        review = _make_review(repo_env, "feat-race")
        # Simulate a human merge that claimed the review first.
        store.claim_review_for_merge(review["id"])

        await _run(repo_env, review)

        assert store.get_review(review["id"])["status"] == "merging"
        assert track.count_events(source="drive_review", outcome="auto_merged") == 0

    @_requires_merge_tree
    async def test_sampled_flag_matches_injected_rng(self, repo_env):
        track = repo_env["track"]
        _seed_events(track, submitter="member1", lead="member2", outcome="merged", count=5)
        review = _make_review(repo_env, "feat-sampled")

        await _run(repo_env, review, rng=random.Random(7))

        events = [e for e in track.recent_events("member1") if e["outcome"] == "auto_merged"]
        assert len(events) == 1
        # Recompute the exact same deterministic draw independently and
        # compare — proves the injected rng (not the module's own
        # `random.Random()` default) drove the recorded `sampled` flag.
        from src.shared import limits as limits_mod

        rate = auto_merge.sample_rate_for(
            0,
            initial=limits_mod.auto_merge_sample_rate_initial(),
            decay_after=limits_mod.resolve("auto_merge_sample_decay_after"),
            floor=limits_mod.auto_merge_sample_rate_floor(),
        )
        assert events[0]["details"]["sampled"] == (random.Random(7).random() < rate)

    @_requires_merge_tree
    async def test_sample_decays_after_threshold_for_the_pair(self, repo_env):
        track = repo_env["track"]
        _seed_events(track, submitter="member1", lead="member2", outcome="merged", count=5)
        decay_after = 3
        _seed_events(
            track, submitter="member1", lead="member2", outcome="auto_merged",
            rater_kind="system", count=decay_after,
        )
        review = _make_review(repo_env, "feat-decayed")

        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("OPENLEGION_AUTO_MERGE_DAILY_CAP", "100")  # the seeded events must not trip the cap
            mp.setenv("OPENLEGION_AUTO_MERGE_SAMPLE_DECAY_AFTER", str(decay_after))
            mp.setenv("OPENLEGION_AUTO_MERGE_SAMPLE_RATE_INITIAL", "1.0")
            mp.setenv("OPENLEGION_AUTO_MERGE_SAMPLE_RATE_FLOOR", "0.0")
            await _run(repo_env, review, rng=random.Random(0))

        events = [e for e in track.recent_events("member1") if e["outcome"] == "auto_merged"]
        assert events[0]["details"]["sampled"] is False  # decayed to the 0.0 floor


class TestSampleRateHelpers:
    def test_sample_rate_for_decays_after_threshold(self):
        assert auto_merge.sample_rate_for(0, initial=0.2, decay_after=10, floor=0.05) == 0.2
        assert auto_merge.sample_rate_for(9, initial=0.2, decay_after=10, floor=0.05) == 0.2
        assert auto_merge.sample_rate_for(10, initial=0.2, decay_after=10, floor=0.05) == 0.05
        assert auto_merge.sample_rate_for(100, initial=0.2, decay_after=10, floor=0.05) == 0.05

    def test_should_sample_deterministic_with_seeded_rng(self):
        results_a = [auto_merge.should_sample(0.5, random.Random(42)) for _ in range(1)]
        rng = random.Random(42)
        results_b = [auto_merge.should_sample(0.5, rng) for _ in range(20)]
        rng2 = random.Random(42)
        results_c = [auto_merge.should_sample(0.5, rng2) for _ in range(20)]
        assert results_b == results_c
        assert len(results_a) == 1

    def test_should_sample_rate_zero_never_samples(self):
        rng = random.Random(1)
        assert all(not auto_merge.should_sample(0.0, rng) for _ in range(50))

    def test_should_sample_rate_one_always_samples(self):
        rng = random.Random(1)
        assert all(auto_merge.should_sample(1.0, rng) for _ in range(50))

    def test_evaluate_pair_trust_floor(self):
        gate = auto_merge.evaluate_pair_trust(
            {"merged": 5, "rejected_after_approve": 0, "flagged": 0}, trust_floor=5,
        )
        assert gate.eligible is True

        below = auto_merge.evaluate_pair_trust(
            {"merged": 4, "rejected_after_approve": 0, "flagged": 0}, trust_floor=5,
        )
        assert below.eligible is False

        rejected = auto_merge.evaluate_pair_trust(
            {"merged": 100, "rejected_after_approve": 1, "flagged": 0}, trust_floor=5,
        )
        assert rejected.eligible is False

        flagged = auto_merge.evaluate_pair_trust(
            {"merged": 100, "rejected_after_approve": 0, "flagged": 1}, trust_floor=5,
        )
        assert flagged.eligible is False


class TestLimitsDefaults:
    def test_auto_merge_limit_defaults(self):
        from src.shared import limits as limits_mod

        assert limits_mod.LIMIT_SPECS["auto_merge_daily_cap"][0] == 3
        assert limits_mod.LIMIT_SPECS["auto_merge_daily_cap"][1] == 0  # 0 = kill switch, unclamped up
        assert limits_mod.LIMIT_SPECS["auto_merge_trust_floor"][0] == 5
        assert limits_mod.LIMIT_SPECS["auto_merge_sample_decay_after"][0] == 10
        assert limits_mod.auto_merge_sample_rate_initial() == 0.20
        assert limits_mod.auto_merge_sample_rate_floor() == 0.05

    def test_auto_merge_sample_rate_env_override_and_clamp(self, monkeypatch):
        from src.shared import limits as limits_mod

        monkeypatch.setenv("OPENLEGION_AUTO_MERGE_SAMPLE_RATE_INITIAL", "0.5")
        assert limits_mod.auto_merge_sample_rate_initial() == 0.5
        monkeypatch.setenv("OPENLEGION_AUTO_MERGE_SAMPLE_RATE_INITIAL", "5.0")
        assert limits_mod.auto_merge_sample_rate_initial() == 1.0  # clamped
        monkeypatch.setenv("OPENLEGION_AUTO_MERGE_SAMPLE_RATE_INITIAL", "not-a-float")
        assert limits_mod.auto_merge_sample_rate_initial() == 0.20  # invalid -> default


# ── full-app fixture (FastAPI, mirrors test_team_drive.py's drive_env) ─

TOKENS = {"operator": "op-token", "member1": "m1-token", "member2": "m2-token"}


def _headers(agent: str) -> dict:
    return {"Authorization": f"Bearer {TOKENS[agent]}", "X-Agent-ID": agent}


class _FakeTransport:
    """Duck-typed ``Transport`` — records every ``/chat/note`` call."""

    def __init__(self, *, fail: bool = False):
        self.calls: list[dict] = []
        self.fail = fail

    async def request(self, agent_id, method, path, json=None, timeout=None):
        self.calls.append({"agent_id": agent_id, "method": method, "path": path, "json": json})
        if self.fail:
            return {"error": "boom"}
        return {"ok": True}


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENLEGION_TEAM_DRIVES_DIR", str(tmp_path / "drives"))

    perms_file = tmp_path / "permissions.json"
    perms_file.write_text(json.dumps({
        "permissions": {
            "operator": {"blackboard_read": ["*"], "blackboard_write": ["*"]},
            "member1": {}, "member2": {},
        },
    }))
    agents_file = tmp_path / "agents.yaml"
    agents_file.write_text(yaml.dump({
        "agents": {
            "operator": {"role": "operator"},
            "member1": {"role": "w"},
            "member2": {"role": "w"},
        },
    }))
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
    store.set_lead("team-x", "member2")

    transport = _FakeTransport()
    app = server_module.create_mesh_app(
        blackboard=blackboard,
        pubsub=pubsub,
        router=router,
        permissions=permissions,
        teams_store=store,
        container_manager=MagicMock(),
        auth_tokens=dict(TOKENS),
        transport=transport,
    )
    yield {"app": app, "store": store, "blackboard": blackboard, "transport": transport, "tmp_path": tmp_path}
    blackboard.close()
    importlib.reload(server_module)


async def _drain(app) -> None:
    """Await every fire-and-forget auto-merge task to completion —
    deterministic (no polling): the tasks were scheduled on THIS event
    loop by the request we just awaited."""
    tasks = list(app._auto_merge_tasks)
    if tasks:
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=30)


def _seed_pair_trust(app, *, submitter, lead, count=5):
    _seed_events(app.track_record_store, submitter=submitter, lead=lead, outcome="merged", count=count)


class TestVerdictEndpointTriggersConsumer:
    @_requires_merge_tree
    @pytest.mark.asyncio
    async def test_approve_verdict_schedules_and_lands_auto_merge(self, app_env):
        app, store, transport = app_env["app"], app_env["store"], app_env["transport"]
        _seed_pair_trust(app, submitter="member1", lead="member2")
        sha = _push_branch_locally(Path(store.get_team("team-x")["drive_ref"]), "feat-e2e", "e.md", "x\n")
        review = store.create_review("team-x", "feat-e2e", "member1", "t", head_sha=sha)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            resp = await c.post(
                f"/mesh/teams/team-x/drive/reviews/{review['id']}/verdict",
                json={"verdict": "approve"},
                headers=_headers("member2"),
            )
        assert resp.status_code == 200

        await _drain(app)

        resolved = store.get_review(review["id"])
        assert resolved["status"] == "merged"
        assert resolved["reviewer"] == auto_merge.AUTO_MERGE_RATER
        assert resolved["lead_verdict_by"] == "member2"
        # An operator-chat note went out for the landed auto-merge.
        note_calls = [c for c in transport.calls if c["path"] == "/chat/note"]
        assert len(note_calls) == 1
        assert review["id"] in note_calls[0]["json"]["message"]

    @pytest.mark.asyncio
    async def test_reject_verdict_never_schedules_consumer(self, app_env):
        app, store = app_env["app"], app_env["store"]
        _seed_pair_trust(app, submitter="member1", lead="member2")
        sha = _push_branch_locally(Path(store.get_team("team-x")["drive_ref"]), "feat-noauto", "n.md", "x\n")
        review = store.create_review("team-x", "feat-noauto", "member1", "t", head_sha=sha)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            resp = await c.post(
                f"/mesh/teams/team-x/drive/reviews/{review['id']}/verdict",
                json={"verdict": "reject"},
                headers=_headers("member2"),
            )
        assert resp.status_code == 200
        assert len(app._auto_merge_tasks) == 0

    @pytest.mark.asyncio
    async def test_verdict_endpoint_survives_consumer_exception(self, app_env, monkeypatch):
        """Exception injected deep in the pipeline (pair_trust) — the
        verdict endpoint must still 200, and draining the task must not
        raise (consider_auto_merge's own outer net absorbs it)."""
        app, store = app_env["app"], app_env["store"]
        sha = _push_branch_locally(Path(store.get_team("team-x")["drive_ref"]), "feat-explode", "x.md", "x\n")
        review = store.create_review("team-x", "feat-explode", "member1", "t", head_sha=sha)

        def _boom(*a, **kw):
            raise RuntimeError("injected failure")

        monkeypatch.setattr(app.track_record_store, "count_events", _boom)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            resp = await c.post(
                f"/mesh/teams/team-x/drive/reviews/{review['id']}/verdict",
                json={"verdict": "approve"},
                headers=_headers("member2"),
            )
        assert resp.status_code == 200

        await _drain(app)  # must not raise

        assert store.get_review(review["id"])["status"] == "open"

    @pytest.mark.asyncio
    async def test_lead_verdict_by_recorded_and_surfaced(self, app_env):
        app, store = app_env["app"], app_env["store"]
        sha = _push_branch_locally(Path(store.get_team("team-x")["drive_ref"]), "feat-lvb", "lvb.md", "x\n")
        review = store.create_review("team-x", "feat-lvb", "member1", "t", head_sha=sha)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            resp = await c.post(
                f"/mesh/teams/team-x/drive/reviews/{review['id']}/verdict",
                json={"verdict": "approve", "note": "lgtm"},
                headers=_headers("member2"),
            )
        assert resp.status_code == 200
        assert resp.json()["review"]["lead_verdict_by"] == "member2"
        assert store.get_review(review["id"])["lead_verdict_by"] == "member2"
        listed = store.list_reviews("team-x", status="open")
        assert any(r["id"] == review["id"] and r["lead_verdict_by"] == "member2" for r in listed)

        await _drain(app)  # floor unmet — stays open; no crash on drain

    @_requires_merge_tree
    @pytest.mark.asyncio
    async def test_record_drive_review_outcome_uses_lead_verdict_by_not_current_lead(self, app_env):
        """U1's approximation bug, fixed by §8 #20: a lead SWAP between
        verdict and (human) merge must not mis-attribute the pair — the
        recorded event's `lead_agent_id` must be the row's `lead_verdict_by`
        (member2, who verdicted), not member1 (the new current lead)."""
        app, store = app_env["app"], app_env["store"]
        store.add_member("team-x", "member3")
        sha = _push_branch_locally(Path(store.get_team("team-x")["drive_ref"]), "feat-swap", "s.md", "x\n")
        review = store.create_review("team-x", "feat-swap", "member1", "t", head_sha=sha)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            verdict_resp = await c.post(
                f"/mesh/teams/team-x/drive/reviews/{review['id']}/verdict",
                json={"verdict": "approve"},
                headers=_headers("member2"),
            )
            assert verdict_resp.status_code == 200
            await _drain(app)  # floor unmet — no auto-merge fires

            # Lead swap AFTER the verdict, BEFORE the (human) merge.
            store.set_lead("team-x", "member3")

            merge_resp = await c.post(
                f"/mesh/teams/team-x/drive/reviews/{review['id']}/merge",
                headers=_headers("operator"),
            )
        assert merge_resp.status_code == 200, merge_resp.text

        events = app.track_record_store.recent_events("member1")
        assert len(events) == 1
        assert events[0]["details"]["lead_agent_id"] == "member2"  # NOT member3


class TestFlagAndRevertEndpoints:
    async def _auto_merged_review(self, app_env):
        app, store = app_env["app"], app_env["store"]
        _seed_pair_trust(app, submitter="member1", lead="member2")
        sha = _push_branch_locally(Path(store.get_team("team-x")["drive_ref"]), "feat-flaggable", "fl.md", "x\n")
        review = store.create_review("team-x", "feat-flaggable", "member1", "t", head_sha=sha)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            await c.post(
                f"/mesh/teams/team-x/drive/reviews/{review['id']}/verdict",
                json={"verdict": "approve"}, headers=_headers("member2"),
            )
        await _drain(app)
        resolved = store.get_review(review["id"])
        assert resolved["status"] == "merged"
        return resolved

    @_requires_merge_tree
    @pytest.mark.asyncio
    async def test_flag_auto_merge_operator_only(self, app_env):
        review = await self._auto_merged_review(app_env)
        app = app_env["app"]
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            agent_resp = await c.post(
                f"/mesh/teams/team-x/drive/reviews/{review['id']}/flag-auto-merge",
                headers=_headers("member1"),
            )
            assert agent_resp.status_code == 403

            op_resp = await c.post(
                f"/mesh/teams/team-x/drive/reviews/{review['id']}/flag-auto-merge",
                headers=_headers("operator"),
            )
        assert op_resp.status_code == 200, op_resp.text

        pair = app.track_record_store.pair_trust("member2", "member1")
        assert pair["flagged"] == 1

    @_requires_merge_tree
    @pytest.mark.asyncio
    async def test_flag_auto_merge_404_unknown_and_409_non_auto_merged(self, app_env):
        app, store = app_env["app"], app_env["store"]
        sha = _push_branch_locally(Path(store.get_team("team-x")["drive_ref"]), "feat-plain", "p.md", "x\n")
        plain_review = store.create_review("team-x", "feat-plain", "member1", "t", head_sha=sha)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            missing = await c.post(
                "/mesh/teams/team-x/drive/reviews/rev_missing/flag-auto-merge",
                headers=_headers("operator"),
            )
            assert missing.status_code == 404

            not_merged = await c.post(
                f"/mesh/teams/team-x/drive/reviews/{plain_review['id']}/flag-auto-merge",
                headers=_headers("operator"),
            )
        assert not_merged.status_code == 409

    @_requires_merge_tree
    @pytest.mark.asyncio
    async def test_revert_merge_operator_only(self, app_env):
        review = await self._auto_merged_review(app_env)
        app = app_env["app"]
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            agent_resp = await c.post(
                f"/mesh/teams/team-x/drive/reviews/{review['id']}/revert-merge",
                headers=_headers("member1"),
            )
            assert agent_resp.status_code == 403

    @_requires_merge_tree
    @pytest.mark.asyncio
    async def test_revert_merge_inverts_the_diff_and_records_decay(self, app_env):
        review = await self._auto_merged_review(app_env)
        app, store = app_env["app"], app_env["store"]
        repo = Path(store.get_team("team-x")["drive_ref"])

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            resp = await c.post(
                f"/mesh/teams/team-x/drive/reviews/{review['id']}/revert-merge",
                headers=_headers("operator"),
            )
        assert resp.status_code == 200, resp.text
        revert_sha = resp.json()["revert_commit"]
        assert revert_sha

        # The file the auto-merge introduced is gone from main post-revert.
        with tempfile.TemporaryDirectory() as tmp:
            verify = Path(tmp) / "verify"
            subprocess.run(
                ["git", "-c", "protocol.file.allow=always", "clone", str(repo), str(verify)],
                check=True, capture_output=True, env=_GIT_ENV,
            )
            assert not (verify / "fl.md").exists()

        # The review row itself is untouched (still "merged" — the git
        # history, not the review record, carries the undo).
        assert store.get_review(review["id"])["status"] == "merged"

        pair = app.track_record_store.pair_trust("member2", "member1")
        assert pair["flagged"] == 1

    @_requires_merge_tree
    @pytest.mark.asyncio
    async def test_revert_merge_404_non_auto_merged(self, app_env):
        app, store = app_env["app"], app_env["store"]
        sha = _push_branch_locally(Path(store.get_team("team-x")["drive_ref"]), "feat-plain2", "p2.md", "x\n")
        plain_review = store.create_review("team-x", "feat-plain2", "member1", "t", head_sha=sha)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            resp = await c.post(
                f"/mesh/teams/team-x/drive/reviews/{plain_review['id']}/revert-merge",
                headers=_headers("operator"),
            )
        assert resp.status_code == 409
