"""Onboarding wake on team JOIN (plan §8 #15).

Covers:
  - Mesh ``POST /mesh/teams/{team}/members`` fires the onboarding wake
    fire-and-forget (never blocks the response) for a running, non-
    operator joiner: an intro turn is dispatched, a non-empty reply is
    posted HOST-SIDE into the team channel thread (sender = the new
    member — no agent-facing posting endpoint), and a probationary-
    first-task nudge goes to the lead (or the operator if none).
  - Guard rails: no wake for the operator, none when the joining agent
    isn't running, exceptions logged + swallowed (join never fails).
  - The dashboard's ``POST /api/teams/{team}/members`` equivalent wires
    the SAME injected callback.
"""

from __future__ import annotations

import asyncio
import importlib
from unittest.mock import MagicMock

import pytest
import yaml
from httpx import ASGITransport, AsyncClient

from src.host.mesh import Blackboard, MessageRouter, PubSub
from src.host.permissions import PermissionMatrix
from src.host.teams import TeamStore
from src.host.threads import ThreadStore


class _RecordingLaneManager:
    """Duck-typed lane manager — records every ``enqueue`` and replies
    per-agent (empty string / omission = silent)."""

    def __init__(self, replies: dict[str, str] | None = None, raise_for: set[str] | None = None,
                 block_for: set[str] | None = None):
        self.calls: list[dict] = []
        self.replies = replies or {}
        self.raise_for = raise_for or set()
        self.block_for = block_for or set()
        self._events: dict[str, asyncio.Event] = {}

    def release(self, agent: str) -> None:
        self._events.setdefault(agent, asyncio.Event()).set()

    async def enqueue(self, agent, message, *, mode="followup", system_note=False, **kw):
        self.calls.append({"agent": agent, "message": message, "system_note": system_note})
        if agent in self.block_for:
            ev = self._events.setdefault(agent, asyncio.Event())
            await ev.wait()
        if agent in self.raise_for:
            raise RuntimeError(f"dispatch to {agent} failed")
        return self.replies.get(agent, "")


async def _drain(iterations: int = 50) -> None:
    """Let fire-and-forget background tasks run to completion on this
    same event loop (no real delay in the fakes below)."""
    for _ in range(iterations):
        await asyncio.sleep(0)


def _op_headers() -> dict:
    return {"Authorization": "Bearer op-token", "X-Agent-ID": "operator"}


def _build_app(tmp_path, monkeypatch, *, lane_manager=None, extra_registry=None):
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "config"
    config_dir.mkdir(exist_ok=True)
    agents_file = config_dir / "agents.yaml"
    agents_file.write_text(yaml.dump({
        "agents": {
            "agent1": {"role": "worker"},
            "agent2": {"role": "worker"},
            "operator": {"role": "operator"},
        },
    }))
    perms_file = config_dir / "permissions.json"
    perms_file.write_text("{}")

    import src.cli.config as cli_cfg

    monkeypatch.setattr(cli_cfg, "AGENTS_FILE", agents_file)
    monkeypatch.setattr(cli_cfg, "PERMISSIONS_FILE", perms_file)
    monkeypatch.setattr(cli_cfg, "TEAMS_DIR", config_dir / "teams")

    import src.host.server as server_module

    importlib.reload(server_module)

    perms = PermissionMatrix()
    blackboard = Blackboard(str(tmp_path / "bb.db"))
    pubsub = PubSub()
    registry = {"operator": "http://op:8400"}
    if extra_registry:
        registry.update(extra_registry)
    router = MessageRouter(perms, registry)
    teams_store = TeamStore(db_path=str(tmp_path / "teams.db"), teams_dir=tmp_path / "teams")
    teams_store.create_team("research", description="d")
    teams_store.set_goal("research", "Ship the launch.", ["10 leads/week"])
    thread_store = ThreadStore(db_path=":memory:")

    app = server_module.create_mesh_app(
        blackboard=blackboard,
        pubsub=pubsub,
        router=router,
        permissions=perms,
        teams_store=teams_store,
        lane_manager=lane_manager,
        thread_store=thread_store,
        auth_tokens={"operator": "op-token"},
    )
    return app, blackboard, teams_store, thread_store, router


# ── Mesh join endpoint ───────────────────────────────────────────────


class TestMeshOnboardingWake:
    @pytest.mark.asyncio
    async def test_join_dispatches_intro_posts_to_channel_and_nudges_operator(self, tmp_path, monkeypatch):
        lane = _RecordingLaneManager(replies={"agent1": "Hi team, I'm agent1, glad to be here!"})
        app, bb, store, threads, router = _build_app(
            tmp_path, monkeypatch, lane_manager=lane, extra_registry={"agent1": "http://a1:8400"},
        )
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
                r = await c.post(
                    "/mesh/teams/research/members", json={"agent": "agent1"}, headers=_op_headers(),
                )
            assert r.status_code == 200, r.text
            await _drain()

            # Intro turn dispatched to the new member, system-composed.
            intro_calls = [c for c in lane.calls if c["agent"] == "agent1"]
            assert len(intro_calls) == 1
            assert intro_calls[0]["system_note"] is True
            assert "research" in intro_calls[0]["message"]
            assert "Ship the launch." in intro_calls[0]["message"]

            # Posted HOST-SIDE into the team channel, sender = the new member.
            channel = threads.ensure_channel("research")
            msgs = threads.list_messages(channel["id"])
            bodies = [m["body"] for m in msgs if m["sender"] == "agent1"]
            assert any("glad to be here" in b for b in bodies)

            # No lead on this team — nudge falls back to the operator.
            nudge_calls = [c for c in lane.calls if c["agent"] == "operator"]
            assert len(nudge_calls) == 1
            assert "agent1" in nudge_calls[0]["message"]
            assert "research" in nudge_calls[0]["message"]
        finally:
            bb.close()
            threads.close()

    @pytest.mark.asyncio
    async def test_no_response_sentinel_is_never_posted_to_channel(self, tmp_path, monkeypatch):
        """An unreachable new member's intro dispatch returns the literal
        "(no response)" success sentinel — it must not be posted to the
        team channel as if the member said it."""
        lane = _RecordingLaneManager(replies={"agent1": "(no response)"})
        app, bb, store, threads, router = _build_app(
            tmp_path, monkeypatch, lane_manager=lane, extra_registry={"agent1": "http://a1:8400"},
        )
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
                r = await c.post(
                    "/mesh/teams/research/members", json={"agent": "agent1"}, headers=_op_headers(),
                )
            assert r.status_code == 200, r.text
            await _drain()
            channel = threads.ensure_channel("research")
            msgs = threads.list_messages(channel["id"])
            assert not [m for m in msgs if m["sender"] == "agent1"]
        finally:
            bb.close()
            threads.close()

    @pytest.mark.asyncio
    async def test_silent_token_is_never_posted_to_channel(self, tmp_path, monkeypatch):
        """SILENT_REPLY_TOKEN from the lane dispatcher must never be posted to
        the team channel as if the new member said it (shared usable-reply
        gate)."""
        from src.shared.types import SILENT_REPLY_TOKEN

        lane = _RecordingLaneManager(replies={"agent1": SILENT_REPLY_TOKEN})
        app, bb, store, threads, router = _build_app(
            tmp_path, monkeypatch, lane_manager=lane, extra_registry={"agent1": "http://a1:8400"},
        )
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
                r = await c.post(
                    "/mesh/teams/research/members", json={"agent": "agent1"}, headers=_op_headers(),
                )
            assert r.status_code == 200, r.text
            await _drain()
            channel = threads.ensure_channel("research")
            msgs = threads.list_messages(channel["id"])
            assert not [m for m in msgs if m["sender"] == "agent1"]
        finally:
            bb.close()
            threads.close()

    @pytest.mark.asyncio
    async def test_dispatch_error_note_is_never_posted_to_channel(self, tmp_path, monkeypatch):
        """A "dispatch_error: <redacted>" note from the lane dispatcher's
        except-branch must never be posted to the team channel."""
        lane = _RecordingLaneManager(replies={"agent1": "dispatch_error: connection reset"})
        app, bb, store, threads, router = _build_app(
            tmp_path, monkeypatch, lane_manager=lane, extra_registry={"agent1": "http://a1:8400"},
        )
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
                r = await c.post(
                    "/mesh/teams/research/members", json={"agent": "agent1"}, headers=_op_headers(),
                )
            assert r.status_code == 200, r.text
            await _drain()
            channel = threads.ensure_channel("research")
            msgs = threads.list_messages(channel["id"])
            assert not [m for m in msgs if m["sender"] == "agent1"]
        finally:
            bb.close()
            threads.close()

    @pytest.mark.asyncio
    async def test_join_nudges_lead_not_operator_when_lead_present(self, tmp_path, monkeypatch):
        lane = _RecordingLaneManager(replies={"agent1": "hello"})
        app, bb, store, threads, router = _build_app(
            tmp_path, monkeypatch, lane_manager=lane,
            extra_registry={"agent1": "http://a1:8400", "agent2": "http://a2:8400"},
        )
        try:
            store.add_member("research", "agent2")
            store.set_lead("research", "agent2")
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
                r = await c.post(
                    "/mesh/teams/research/members", json={"agent": "agent1"}, headers=_op_headers(),
                )
            assert r.status_code == 200, r.text
            await _drain()
        finally:
            bb.close()
            threads.close()

        nudge_targets = {c["agent"] for c in lane.calls} - {"agent1"}
        assert nudge_targets == {"agent2"}

    @pytest.mark.asyncio
    async def test_no_wake_when_joining_agent_not_running(self, tmp_path, monkeypatch):
        """agent1 exists in config but has no live URL registration."""
        lane = _RecordingLaneManager(replies={"agent1": "hello"})
        app, bb, store, threads, router = _build_app(tmp_path, monkeypatch, lane_manager=lane)
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
                r = await c.post(
                    "/mesh/teams/research/members", json={"agent": "agent1"}, headers=_op_headers(),
                )
            assert r.status_code == 200, r.text
            await _drain()
        finally:
            bb.close()
            threads.close()
        assert lane.calls == []

    @pytest.mark.asyncio
    async def test_no_wake_for_operator_join(self, tmp_path, monkeypatch):
        """Defensive: the mesh app's scheduler itself refuses the operator
        (membership already rejects an operator JOIN, so this pins the
        guard directly rather than relying on that upstream rejection)."""
        lane = _RecordingLaneManager()
        app, bb, store, threads, router = _build_app(
            tmp_path, monkeypatch, lane_manager=lane, extra_registry={"operator": "http://op:8400"},
        )
        try:
            app._schedule_onboarding_wake("operator", "research")
            await _drain()
        finally:
            bb.close()
            threads.close()
        assert lane.calls == []

    @pytest.mark.asyncio
    async def test_join_response_never_blocks_on_the_wake(self, tmp_path, monkeypatch):
        """Fire-and-forget: the endpoint must return before a slow intro
        turn resolves."""
        lane = _RecordingLaneManager(replies={"agent1": "hello"}, block_for={"agent1"})
        app, bb, store, threads, router = _build_app(
            tmp_path, monkeypatch, lane_manager=lane, extra_registry={"agent1": "http://a1:8400"},
        )
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
                r = await asyncio.wait_for(
                    c.post("/mesh/teams/research/members", json={"agent": "agent1"}, headers=_op_headers()),
                    timeout=5,
                )
            assert r.status_code == 200, r.text
            # The endpoint returned even though agent1's intro turn is
            # still blocked — prove the wake really is in-flight.
            await _drain(5)
            assert any(c["agent"] == "agent1" for c in lane.calls)
            lane.release("agent1")
            await _drain()
        finally:
            bb.close()
            threads.close()

    @pytest.mark.asyncio
    async def test_intro_turn_failure_swallowed_nudge_still_fires(self, tmp_path, monkeypatch):
        """A crash in the intro turn must not skip the lead/operator nudge,
        and must never propagate out of the join endpoint."""
        lane = _RecordingLaneManager(raise_for={"agent1"})
        app, bb, store, threads, router = _build_app(
            tmp_path, monkeypatch, lane_manager=lane, extra_registry={"agent1": "http://a1:8400"},
        )
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
                r = await c.post(
                    "/mesh/teams/research/members", json={"agent": "agent1"}, headers=_op_headers(),
                )
            assert r.status_code == 200, r.text
            await _drain()
        finally:
            bb.close()
            threads.close()
        # The nudge (to the operator, no lead set) still happened.
        assert any(c["agent"] == "operator" for c in lane.calls)


# ── Dashboard join endpoint (same injected seam) ────────────────────


class TestDashboardOnboardingWakeWiring:
    def test_dashboard_join_calls_injected_onboarding_wake(self, tmp_path):
        """The dashboard endpoint doesn't reimplement the wake logic — it
        calls the SAME callback the mesh app exposes. This test only pins
        the wiring; the dispatch logic itself is covered above."""
        import os

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from src.dashboard.server import create_dashboard_router
        from src.host.costs import CostTracker
        from src.host.health import HealthMonitor
        from src.host.traces import TraceStore

        bb = Blackboard(str(tmp_path / "bb.db"))
        cost_tracker = CostTracker(db_path=str(tmp_path / "costs.db"))
        trace_store = TraceStore(db_path=str(tmp_path / "traces.db"))
        teams_store = TeamStore(db_path=str(tmp_path / "teams.db"), teams_dir=tmp_path / "teams")
        teams_store.create_team("research")
        runtime_mock = MagicMock()
        transport_mock = MagicMock()
        router_mock = MagicMock()
        health_monitor = HealthMonitor(runtime=runtime_mock, transport=transport_mock, router=router_mock)
        agent_registry = {"agent1": "http://a1:8400"}
        onboarding_wake = MagicMock()

        router = create_dashboard_router(
            blackboard=bb,
            health_monitor=health_monitor,
            cost_tracker=cost_tracker,
            trace_store=trace_store,
            event_bus=None,
            agent_registry=agent_registry,
            teams_store=teams_store,
            permissions=MagicMock(),
            transport=transport_mock,
            onboarding_wake=onboarding_wake,
        )
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        try:
            resp = client.post(
                "/dashboard/api/teams/research/members",
                json={"agent": "agent1"},
                headers={"X-Requested-With": "XMLHttpRequest"},
            )
            assert resp.status_code == 200, resp.text
        finally:
            bb.close()
            cost_tracker.close()
            trace_store.close()
        onboarding_wake.assert_called_once_with("agent1", "research")
        os.environ.pop("OPENLEGION_MAX_TEAMS", None)


# ── FIX 6: create-team paths fire onboarding for INITIAL members ─────


class TestCreateTeamOnboardingWake:
    @pytest.mark.asyncio
    async def test_mesh_create_team_fires_wake_for_initial_members(self, tmp_path, monkeypatch):
        """``POST /mesh/teams`` must fire the onboarding wake for each INITIAL
        member (previously only add-member did), so a seeded member gets the
        same intro turn + channel post that a later-added member gets."""
        lane = _RecordingLaneManager(replies={"agent1": "Hi team, agent1 here!"})
        app, bb, store, threads, router = _build_app(
            tmp_path, monkeypatch, lane_manager=lane, extra_registry={"agent1": "http://a1:8400"},
        )
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
                r = await c.post(
                    "/mesh/teams",
                    json={"name": "newteam", "members": ["agent1"]},
                    headers=_op_headers(),
                )
            assert r.status_code == 200, r.text
            await _drain()
            # Intro turn dispatched to the seeded member.
            intro_calls = [c for c in lane.calls if c["agent"] == "agent1"]
            assert len(intro_calls) == 1
            assert intro_calls[0]["system_note"] is True
            assert "newteam" in intro_calls[0]["message"]
            # Reply posted host-side into the new team's channel.
            channel = threads.ensure_channel("newteam")
            msgs = threads.list_messages(channel["id"])
            assert any("agent1 here" in m["body"] for m in msgs if m["sender"] == "agent1")
        finally:
            bb.close()
            threads.close()

    def test_dashboard_create_team_calls_injected_wake_per_member(self, tmp_path, monkeypatch):
        """The dashboard ``POST /api/teams`` create path must invoke the
        injected ``onboarding_wake`` for each initial member (same seam the
        add-member endpoint uses)."""
        import os

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from src.dashboard.server import create_dashboard_router
        from src.host.costs import CostTracker
        from src.host.health import HealthMonitor
        from src.host.traces import TraceStore

        config_dir = tmp_path / "config"
        config_dir.mkdir(exist_ok=True)
        agents_file = config_dir / "agents.yaml"
        agents_file.write_text(yaml.dump({"agents": {"agent1": {"role": "worker"}}}))
        perms_file = config_dir / "permissions.json"
        perms_file.write_text("{}")
        import src.cli.config as cli_cfg

        monkeypatch.setattr(cli_cfg, "AGENTS_FILE", agents_file)
        monkeypatch.setattr(cli_cfg, "PERMISSIONS_FILE", perms_file)
        monkeypatch.setattr(cli_cfg, "TEAMS_DIR", config_dir / "teams")

        bb = Blackboard(str(tmp_path / "bb.db"))
        cost_tracker = CostTracker(db_path=str(tmp_path / "costs.db"))
        trace_store = TraceStore(db_path=str(tmp_path / "traces.db"))
        teams_store = TeamStore(db_path=str(tmp_path / "teams.db"), teams_dir=tmp_path / "teams")
        runtime_mock = MagicMock()
        transport_mock = MagicMock()
        router_mock = MagicMock()
        health_monitor = HealthMonitor(runtime=runtime_mock, transport=transport_mock, router=router_mock)
        agent_registry = {"agent1": "http://a1:8400"}
        onboarding_wake = MagicMock()

        router = create_dashboard_router(
            blackboard=bb,
            health_monitor=health_monitor,
            cost_tracker=cost_tracker,
            trace_store=trace_store,
            event_bus=None,
            agent_registry=agent_registry,
            teams_store=teams_store,
            permissions=MagicMock(),
            transport=transport_mock,
            onboarding_wake=onboarding_wake,
        )
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        try:
            resp = client.post(
                "/dashboard/api/teams",
                json={"name": "newteam", "members": ["agent1"]},
                headers={"X-Requested-With": "XMLHttpRequest"},
            )
            assert resp.status_code == 200, resp.text
        finally:
            bb.close()
            cost_tracker.close()
            trace_store.close()
        onboarding_wake.assert_called_once_with("agent1", "newteam")
        os.environ.pop("OPENLEGION_MAX_TEAMS", None)
