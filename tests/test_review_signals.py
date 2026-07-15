"""Tests for the team signal ledger's review-outcome signals (Item 2).

Each Team-Drive review outcome (merge / reject / lead approve-verdict /
supersede / merge-preflight failure) must close the feedback loop to the
review AUTHOR: a host-side ``kind='event'`` row on the team CHANNEL thread
addressed to the author, plus a direct low-priority author wake. The
actionable outcomes (``review_rejected`` / ``review_merge_failed``) also
join ``ACTIONABLE_EVENT_KINDS`` so Item 1's heartbeat backstop surfaces
them.

Two layers:
  * the exposed ``app._signal_review_author`` helper — every outcome kind +
    the negative controls (recipient is always the author, self-notify
    skip, coalescing), fast and git-free;
  * the reject + lead-verdict ENDPOINTS end-to-end (no git needed) — proves
    the real wiring and the "author, never the operator/lead" control.
"""

from __future__ import annotations

import asyncio
import importlib
import threading
import time

import pytest
import yaml
from httpx import ASGITransport, AsyncClient

from src.host.mesh import Blackboard, MessageRouter, PubSub
from src.host.permissions import PermissionMatrix
from src.host.teams import TeamStore
from src.host.threads import ThreadStore

TOKENS = {
    "operator": "op-token",
    "member1": "m1-token",
    "member2": "m2-token",
    "outsider": "out-token",
}


def _headers(agent: str) -> dict:
    return {"Authorization": f"Bearer {TOKENS[agent]}", "X-Agent-ID": agent}


class _FakeLane:
    def __init__(self):
        self.calls: list[dict] = []

    async def enqueue(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        return "ok"


@pytest.fixture
def review_env(tmp_path, monkeypatch):
    """Mesh app wired with a real TeamStore + ThreadStore + a fake lane and a
    background dispatch loop, so review-outcome signals can write events AND
    fire author wakes. team-x: member1 (author) + member2. All of member1 /
    member2 / operator are registered so wakes can target them."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENLEGION_ORCHESTRATION_TASKS_DB", str(tmp_path / "tasks.db"))

    perms_file = tmp_path / "permissions.json"
    perms_file.write_text(
        yaml.dump({"permissions": {a: {} for a in ("operator", "member1", "member2", "outsider")}})
    )
    agents_file = tmp_path / "agents.yaml"
    agents_file.write_text(
        yaml.dump({"agents": {a: {"role": "w"} for a in ("operator", "member1", "member2", "outsider")}})
    )
    import src.cli.config as cli_cfg

    monkeypatch.setattr(cli_cfg, "PERMISSIONS_FILE", perms_file)
    monkeypatch.setattr(cli_cfg, "AGENTS_FILE", agents_file)

    import src.host.server as server_module

    importlib.reload(server_module)

    permissions = PermissionMatrix(config_path=str(perms_file))
    router = MessageRouter(permissions, {
        "operator": "http://operator:8400",
        "member1": "http://member1:8400",
        "member2": "http://member2:8400",
    })
    blackboard = Blackboard(str(tmp_path / "bb.db"))
    pubsub = PubSub()
    store = TeamStore(db_path=str(tmp_path / "teams.db"), teams_dir=tmp_path / "teams")
    store.create_team("team-x")
    store.add_member("team-x", "member1")
    store.add_member("team-x", "member2")
    thread_store = ThreadStore(":memory:")

    lane = _FakeLane()
    loop = asyncio.new_event_loop()

    def _run():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    app = server_module.create_mesh_app(
        blackboard=blackboard,
        pubsub=pubsub,
        router=router,
        permissions=permissions,
        teams_store=store,
        thread_store=thread_store,
        lane_manager=lane,  # type: ignore[arg-type]
        dispatch_loop=loop,
        auth_tokens=dict(TOKENS),
    )
    yield {
        "app": app, "store": store, "thread_store": thread_store, "lane": lane,
    }
    loop.call_soon_threadsafe(loop.stop)
    t.join(timeout=2)
    blackboard.close()
    thread_store.close()
    loop.close()
    importlib.reload(server_module)


def _settle(seconds: float = 0.05) -> None:
    time.sleep(seconds)


def _events(thread_store, recipient):
    return thread_store.list_events_for(recipient)


# ── helper-level: every outcome kind + negative controls ─────────

def _review(author="member1", rid="rev_1", branch="feat-a"):
    return {"id": rid, "author": author, "branch": branch, "title": "work"}


class TestSignalReviewAuthorHelper:
    def test_merged_informational_event_and_wake(self, review_env):
        app, ts, lane = review_env["app"], review_env["thread_store"], review_env["lane"]
        app._signal_review_author("team-x", _review(), "review_merged", resolved_by="operator")
        _settle()
        ev = next(e for e in _events(ts, "member1") if e.get("review_id") == "rev_1")
        assert ev["kind"] == "review_merged"
        assert ev["team_id"] == "team-x"
        assert [c for c in lane.calls if c["args"][0] == "member1"]
        # Informational — NOT surfaced by the Item 1 actionable probe.
        assert ts.count_unseen_actionable("member1") == 0

    def test_rejected_is_actionable(self, review_env):
        app, ts, lane = review_env["app"], review_env["thread_store"], review_env["lane"]
        app._signal_review_author("team-x", _review(), "review_rejected", resolved_by="operator")
        _settle()
        assert any(e.get("kind") == "review_rejected" for e in _events(ts, "member1"))
        assert [c for c in lane.calls if c["args"][0] == "member1"]
        # Actionable — the heartbeat backstop will force-surface it.
        assert ts.count_unseen_actionable("member1") == 1

    def test_merge_failed_is_actionable(self, review_env):
        app, ts = review_env["app"], review_env["thread_store"]
        app._signal_review_author(
            "team-x", _review(), "review_merge_failed",
            note="branch changed", resolved_by="operator",
        )
        _settle()
        ev = next(e for e in _events(ts, "member1") if e.get("kind") == "review_merge_failed")
        assert ev["note"] == "branch changed"
        assert ts.count_unseen_actionable("member1") == 1

    def test_approved_and_superseded_informational(self, review_env):
        app, ts = review_env["app"], review_env["thread_store"]
        app._signal_review_author("team-x", _review(rid="rA"), "review_approved", resolved_by="member2")
        app._signal_review_author("team-x", _review(rid="rS"), "review_superseded", resolved_by="member1")
        _settle()
        kinds = {e.get("kind") for e in _events(ts, "member1")}
        assert {"review_approved", "review_superseded"} <= kinds
        assert ts.count_unseen_actionable("member1") == 0

    def test_negative_control_recipient_is_only_the_author(self, review_env):
        """The operator/lead who RESOLVED the review never receives the
        event — only the author does."""
        app, ts = review_env["app"], review_env["thread_store"]
        app._signal_review_author("team-x", _review(author="member1"), "review_rejected", resolved_by="operator")
        _settle()
        assert any(e.get("review_id") == "rev_1" for e in _events(ts, "member1"))
        assert not any(e.get("review_id") == "rev_1" for e in _events(ts, "operator"))
        assert not any(e.get("review_id") == "rev_1" for e in _events(ts, "member2"))

    def test_self_resolution_records_event_but_no_wake(self, review_env):
        app, ts, lane = review_env["app"], review_env["thread_store"], review_env["lane"]
        app._signal_review_author("team-x", _review(author="member1"), "review_merged", resolved_by="member1")
        _settle()
        # Event still recorded (durable ledger)...
        assert any(e.get("review_id") == "rev_1" for e in _events(ts, "member1"))
        # ...but no self-wake.
        assert lane.calls == []

    def test_unregistered_author_records_event_no_wake(self, review_env):
        app, ts, lane = review_env["app"], review_env["thread_store"], review_env["lane"]
        app._signal_review_author("team-x", _review(author="ghost"), "review_rejected", resolved_by="operator")
        _settle()
        assert any(e.get("review_id") == "rev_1" for e in _events(ts, "ghost"))
        assert lane.calls == []

    def test_wake_coalesced_per_review(self, review_env):
        app, lane = review_env["app"], review_env["lane"]
        app._signal_review_author("team-x", _review(), "review_rejected", resolved_by="operator")
        app._signal_review_author("team-x", _review(), "review_merge_failed", resolved_by="operator")
        _settle()
        # Same review id twice inside the window → ONE author wake.
        assert len([c for c in lane.calls if c["args"][0] == "member1"]) == 1


# ── endpoint-level: reject + lead verdict (no git) ───────────────

class TestReviewEndpointSignals:
    @pytest.mark.asyncio
    async def test_reject_signals_author_and_makes_plate_actionable(self, review_env):
        store, app, ts, lane = (
            review_env["store"], review_env["app"], review_env["thread_store"], review_env["lane"],
        )
        review = store.create_review("team-x", "feat-a", "member1", "t")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            resp = await c.post(
                f"/mesh/teams/team-x/drive/reviews/{review['id']}/reject",
                headers=_headers("operator"),
            )
        assert resp.status_code == 200, resp.text
        _settle()
        # Author got the actionable review_rejected event + a wake.
        ev = next(e for e in _events(ts, "member1") if e.get("review_id") == review["id"])
        assert ev["kind"] == "review_rejected"
        assert ev["resolved_by"] == "operator"
        assert [wk for wk in lane.calls if wk["args"][0] == "member1"]
        # Cross-item: surfaces via Item 1's unseen-actionable probe.
        assert ts.count_unseen_actionable("member1") >= 1
        # Negative control: the operator (resolver) is NOT a recipient.
        assert not any(e.get("review_id") == review["id"] for e in _events(ts, "operator"))

    @pytest.mark.asyncio
    async def test_reject_by_author_self_no_wake(self, review_env):
        """An internal caller acting AS the author (author == resolver) still
        records the event but does not self-wake."""
        store, app, ts, lane = (
            review_env["store"], review_env["app"], review_env["thread_store"], review_env["lane"],
        )
        review = store.create_review("team-x", "feat-a", "member1", "t")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            resp = await c.post(
                f"/mesh/teams/team-x/drive/reviews/{review['id']}/reject",
                headers={"x-mesh-internal": "1", "X-Agent-ID": "member1"},
            )
        assert resp.status_code == 200, resp.text
        _settle()
        assert any(e.get("review_id") == review["id"] for e in _events(ts, "member1"))
        assert lane.calls == []

    @pytest.mark.asyncio
    async def test_lead_approve_verdict_signals_author_informational(self, review_env):
        store, app, ts, lane = (
            review_env["store"], review_env["app"], review_env["thread_store"], review_env["lane"],
        )
        store.set_lead("team-x", "member2")
        review = store.create_review("team-x", "feat-a", "member1", "t")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            resp = await c.post(
                f"/mesh/teams/team-x/drive/reviews/{review['id']}/verdict",
                json={"verdict": "approve", "note": "ship it"},
                headers=_headers("member2"),
            )
        assert resp.status_code == 200, resp.text
        _settle()
        ev = next(e for e in _events(ts, "member1") if e.get("review_id") == review["id"])
        assert ev["kind"] == "review_approved"
        assert ev["note"] == "ship it"
        assert [wk for wk in lane.calls if wk["args"][0] == "member1"]
        # Informational — does NOT make the plate actionable.
        assert ts.count_unseen_actionable("member1") == 0
        # Negative control: the lead (member2) is NOT a recipient.
        assert not any(e.get("review_id") == review["id"] for e in _events(ts, "member2"))

    @pytest.mark.asyncio
    async def test_lead_reject_verdict_does_not_signal(self, review_env):
        """A lead REJECT verdict is advisory (review stays OPEN); it is
        deliberately NOT signalled — the operator's eventual reject/merge is
        the terminal author signal."""
        store, app, ts, lane = (
            review_env["store"], review_env["app"], review_env["thread_store"], review_env["lane"],
        )
        store.set_lead("team-x", "member2")
        review = store.create_review("team-x", "feat-a", "member1", "t")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            resp = await c.post(
                f"/mesh/teams/team-x/drive/reviews/{review['id']}/verdict",
                json={"verdict": "reject", "note": "needs work"},
                headers=_headers("member2"),
            )
        assert resp.status_code == 200, resp.text
        _settle()
        assert not any(e.get("review_id") == review["id"] for e in _events(ts, "member1"))
        assert lane.calls == []
