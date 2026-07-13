"""Regression tests for M10 + M13 (Phase 0-5 integration review).

M10 — the mesh add-member endpoint checked ``team_exists`` but not that the
agent id is a real agent, so an operator typo silently created a ghost
``team_members`` row that the Phase-4 lead/standup machinery then treated as
real (ghost lead, ghost standup cron, ghost plate row).

M13 (security) — a membership change (remove / move / team delete) rewired the
blackboard ACL but never purged the departing agent's pubsub subscriptions or
blackboard watches on the OLD team's ``teams/{old}/`` namespace. ``publish`` and
the watcher fan-out read the STORED subscriber/watcher list with no current-ACL
recheck, so the departed agent kept receiving the old team's signals — a real
cross-team event leak.
"""

import importlib
import json

import pytest
import yaml
from httpx import ASGITransport, AsyncClient

from src.host.mesh import Blackboard, MessageRouter, PubSub
from src.host.permissions import PermissionMatrix
from src.host.teams import TeamStore

# ── mesh.py scoped-purge unit tests (M13 primitives) ─────────────────


def test_pubsub_unsubscribe_agent_prefix_scoped(tmp_path):
    ps = PubSub(str(tmp_path / "ps.db"))
    ps.subscribe("teams/x/signals/a", "agent1")
    ps.subscribe("teams/x/status/b", "agent1")
    ps.subscribe("teams/agent1/self", "agent1")  # retained self-scope
    ps.subscribe("teams/x/signals/a", "agent2")  # another agent, untouched
    removed = ps.unsubscribe_agent_prefix("agent1", "teams/x/")
    assert removed == 2
    assert "agent1" not in ps.get_subscribers("teams/x/signals/a")
    assert "agent1" not in ps.get_subscribers("teams/x/status/b")
    # The agent's own self-scope subscription survives.
    assert "agent1" in ps.get_subscribers("teams/agent1/self")
    # Another agent's subscription to the same topic is untouched.
    assert "agent2" in ps.get_subscribers("teams/x/signals/a")
    ps.close()


def test_pubsub_prefix_purge_persists(tmp_path):
    db = str(tmp_path / "ps.db")
    ps = PubSub(db)
    ps.subscribe("teams/x/signals/a", "agent1")
    ps.unsubscribe_agent_prefix("agent1", "teams/x/")
    ps.close()
    # Reload from disk — the DELETE was committed, not just in-memory.
    ps2 = PubSub(db)
    assert "agent1" not in ps2.get_subscribers("teams/x/signals/a")
    ps2.close()


def test_blackboard_remove_agent_watches_prefix_scoped(tmp_path):
    bb = Blackboard(str(tmp_path / "bb.db"))
    bb.add_watch("agent1", "teams/x/*")
    bb.add_watch("agent1", "teams/x/status/*")
    bb.add_watch("agent1", "teams/agent1/*")  # retained self-scope
    removed = bb.remove_agent_watches_prefix("agent1", "teams/x/")
    assert removed == 2
    remaining = bb.get_agent_watches("agent1")
    assert remaining == ["teams/agent1/*"]
    bb.close()


# ── endpoint integration (M10 + M13) ─────────────────────────────────


def _write_perms(tmp_path, permissions: dict):
    perms_file = tmp_path / "permissions.json"
    perms_file.write_text(json.dumps({"permissions": permissions}))
    return perms_file


@pytest.fixture
def team_app(tmp_path, monkeypatch):
    """Mesh app exposing its pubsub + blackboard so the leak-purge is
    observable (mirrors ``tests/test_teams.py::team_app``)."""
    monkeypatch.chdir(tmp_path)
    perms_file = _write_perms(
        tmp_path,
        {
            "agent1": {"blackboard_read": [], "blackboard_write": []},
            "agent2": {"blackboard_read": [], "blackboard_write": []},
        },
    )
    import src.cli.config as cli_cfg

    monkeypatch.setattr(cli_cfg, "PERMISSIONS_FILE", perms_file)
    agents_file = tmp_path / "agents.yaml"
    agents_file.write_text(
        yaml.dump(
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

    permissions = PermissionMatrix()
    router = MessageRouter(permissions, {"operator": "http://op:8400"})
    blackboard = Blackboard(str(tmp_path / "bb.db"))
    pubsub = PubSub(str(tmp_path / "ps.db"))
    teams_store = TeamStore(
        db_path=str(tmp_path / "teams.db"),
        teams_dir=tmp_path / "teams",
    )
    app = server_module.create_mesh_app(
        blackboard=blackboard,
        pubsub=pubsub,
        router=router,
        permissions=permissions,
        teams_store=teams_store,
        auth_tokens={"operator": "op-token"},
    )
    yield app, teams_store, pubsub, blackboard
    blackboard.close()
    pubsub.close()
    importlib.reload(server_module)


def _op_headers() -> dict:
    return {"Authorization": "Bearer op-token", "X-Agent-ID": "operator"}


async def _post(app, path, json_body):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        return await c.post(path, json=json_body, headers=_op_headers())


async def _delete(app, path):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        return await c.delete(path, headers=_op_headers())


class TestAddMemberValidationM10:
    @pytest.mark.asyncio
    async def test_unknown_agent_rejected(self, team_app):
        app, store, _, _ = team_app
        store.create_team("proj1")
        r = await _post(app, "/mesh/teams/proj1/members", {"agent": "ghost"})
        assert r.status_code == 400
        assert "Unknown agent" in r.json()["detail"]
        # No ghost membership row was written.
        assert store.members("proj1") == []

    @pytest.mark.asyncio
    async def test_known_agent_still_accepted(self, team_app):
        app, store, _, _ = team_app
        store.create_team("proj1")
        r = await _post(app, "/mesh/teams/proj1/members", {"agent": "agent1"})
        assert r.status_code == 200
        assert store.members("proj1") == ["agent1"]


class TestMembershipLeakPurgeM13:
    @pytest.mark.asyncio
    async def test_remove_member_purges_subscriptions_and_watches(self, team_app):
        app, store, pubsub, blackboard = team_app
        store.create_team("proj1")
        await _post(app, "/mesh/teams/proj1/members", {"agent": "agent1"})
        pubsub.subscribe("teams/proj1/signals/x", "agent1")
        blackboard.add_watch("agent1", "teams/proj1/*")

        r = await _delete(app, "/mesh/teams/proj1/members/agent1")
        assert r.status_code == 200
        # The leak: agent1 must no longer be a subscriber/watcher of proj1.
        assert "agent1" not in pubsub.get_subscribers("teams/proj1/signals/x")
        assert "teams/proj1/*" not in blackboard.get_agent_watches("agent1")

    @pytest.mark.asyncio
    async def test_move_purges_old_team_subscriptions(self, team_app):
        app, store, pubsub, blackboard = team_app
        store.create_team("proj1")
        store.create_team("proj2")
        await _post(app, "/mesh/teams/proj1/members", {"agent": "agent1"})
        pubsub.subscribe("teams/proj1/signals/x", "agent1")
        blackboard.add_watch("agent1", "teams/proj1/status/*")

        # Move agent1 → proj2 (evicts from proj1).
        await _post(app, "/mesh/teams/proj2/members", {"agent": "agent1"})
        assert store.team_of("agent1") == "proj2"
        assert "agent1" not in pubsub.get_subscribers("teams/proj1/signals/x")
        assert blackboard.get_agent_watches("agent1") == []

    @pytest.mark.asyncio
    async def test_delete_team_purges_members(self, team_app):
        app, store, pubsub, blackboard = team_app
        store.create_team("proj1")
        await _post(app, "/mesh/teams/proj1/members", {"agent": "agent1"})
        pubsub.subscribe("teams/proj1/signals/x", "agent1")
        blackboard.add_watch("agent1", "teams/proj1/*")

        r = await _delete(app, "/mesh/teams/proj1")
        assert r.status_code == 200
        assert "agent1" not in pubsub.get_subscribers("teams/proj1/signals/x")
        assert "teams/proj1/*" not in blackboard.get_agent_watches("agent1")
