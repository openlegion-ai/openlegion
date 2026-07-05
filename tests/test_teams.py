"""Tests for multi-team support: membership wiring, permissions, isolation.

Team identity/membership now lives in the SQLite-backed ``TeamStore``
(src/host/teams.py, covered structurally by ``tests/test_teams_store.py``).
What this module pins is the layer AROUND the store:

- the mesh team endpoints (create/delete/members) rewiring blackboard
  ACLs in ``permissions.json`` on every membership change,
- membership eviction + operator rejection surfacing as HTTP errors,
- ``_remove_agent`` cleaning up store membership + team permissions,
- MeshClient key scoping and PermissionMatrix cross-team isolation
  (unchanged behavior, kept from the pre-store suite).
"""

import importlib
import json
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.cli.config import (
    _add_team_blackboard_permissions,
    _remove_agent,
    _remove_team_blackboard_permissions,
)
from src.host.mesh import Blackboard, MessageRouter, PubSub
from src.host.permissions import PermissionMatrix
from src.host.teams import TeamStore, validate_team_id


class TestValidateTeamId:
    def test_valid_names(self):
        assert validate_team_id("my-project") == "my-project"
        assert validate_team_id("project1") == "project1"
        assert validate_team_id("A_B-C") == "A_B-C"

    def test_invalid_empty(self):
        with pytest.raises(ValueError, match="Invalid team name"):
            validate_team_id("")

    def test_invalid_special_chars(self):
        with pytest.raises(ValueError, match="Invalid team name"):
            validate_team_id("my project")

    def test_invalid_start_char(self):
        with pytest.raises(ValueError, match="Invalid team name"):
            validate_team_id("-start")

    def test_path_traversal_rejected(self):
        for name in ["../escape", "foo/bar", "..", "./current", "a/../b"]:
            with pytest.raises(ValueError, match="Invalid team name"):
                validate_team_id(name)

    def test_max_length_boundary(self):
        # 64 chars should pass
        assert validate_team_id("a" * 64) == "a" * 64
        # 65 chars should fail
        with pytest.raises(ValueError, match="Invalid team name"):
            validate_team_id("a" * 65)

    def test_underscore_start_rejected(self):
        with pytest.raises(ValueError, match="Invalid team name"):
            validate_team_id("_underscore")

    def test_reserved_ids_rejected(self):
        # Teams and agents share downstream namespaces (blackboard
        # prefixes, TEAM_NAME env, _caller_teams sentinels) — a team named
        # after a system identity would shadow it.
        for name in ("operator", "mesh", "default", "canary-probe"):
            with pytest.raises(ValueError, match="reserved"):
                validate_team_id(name)

    def test_trailing_newline_rejected(self):
        with pytest.raises(ValueError, match="Invalid team name"):
            validate_team_id("myteam\n")


# ── Mesh endpoints: store ops + permissions.json wiring ────────────


def _write_perms(tmp_path, permissions: dict) -> "Path":  # noqa: F821
    perms_file = tmp_path / "permissions.json"
    perms_file.write_text(json.dumps({"permissions": permissions}))
    return perms_file


@pytest.fixture
def team_app(tmp_path, monkeypatch):
    """Mesh app with an operator-routed registry, tmp permissions.json,
    and the app's own (in-memory) TeamStore with a scaffold dir."""
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
    import yaml as _yaml

    agents_file.write_text(
        _yaml.dump(
            {
                "agents": {
                    "agent1": {"role": "a"},
                    "agent2": {"role": "b"},
                    "operator": {"role": "operator"},
                },
            }
        )
    )
    monkeypatch.setattr(cli_cfg, "AGENTS_FILE", agents_file)

    import src.host.server as server_module

    importlib.reload(server_module)

    permissions = PermissionMatrix()
    router = MessageRouter(permissions, {"operator": "http://op:8400"})
    blackboard = Blackboard(str(tmp_path / "bb.db"))
    teams_store = TeamStore(
        db_path=str(tmp_path / "teams.db"),
        teams_dir=tmp_path / "teams",
    )
    app = server_module.create_mesh_app(
        blackboard=blackboard,
        pubsub=PubSub(),
        router=router,
        permissions=permissions,
        teams_store=teams_store,
        # Bearer auth so ``_resolve_agent_id`` verifies the operator
        # identity (the team-write endpoints 403 an unverifiable "").
        auth_tokens={"operator": "op-token"},
    )
    yield app, teams_store, perms_file, tmp_path
    blackboard.close()
    importlib.reload(server_module)


def _op_headers() -> dict:
    return {"Authorization": "Bearer op-token", "X-Agent-ID": "operator"}


async def _post(app, path, json_body):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        return await c.post(path, json=json_body, headers=_op_headers())


async def _delete(app, path):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        return await c.delete(path, headers=_op_headers())


class TestMeshTeamEndpoints:
    @pytest.mark.asyncio
    async def test_create_with_members_wires_permissions(self, team_app):
        app, store, perms_file, tmp_path = team_app
        r = await _post(
            app,
            "/mesh/teams",
            {
                "name": "my-proj",
                "description": "Test",
                "members": ["agent1", "agent2"],
            },
        )
        assert r.status_code == 200, r.text
        assert store.members("my-proj") == ["agent1", "agent2"]
        # On-disk scaffold exists (team.md + workflows/).
        assert (tmp_path / "teams" / "my-proj" / "team.md").exists()
        assert (tmp_path / "teams" / "my-proj" / "workflows").is_dir()
        # Only the team-specific pattern is granted.
        perms = json.loads(perms_file.read_text())
        for agent in ("agent1", "agent2"):
            assert perms["permissions"][agent]["blackboard_read"] == ["teams/my-proj/*"]
            assert perms["permissions"][agent]["blackboard_write"] == ["teams/my-proj/*"]

    @pytest.mark.asyncio
    async def test_create_with_member_from_other_team_strips_old_acl(self, team_app):
        """Creating a team whose initial members include an agent already
        on another team must evict the membership AND swap the blackboard
        patterns — the moved agent must not keep teams/old/* reach."""
        app, store, perms_file, _ = team_app
        store.create_team("old-proj")
        await _post(app, "/mesh/teams/old-proj/members", {"agent": "agent1"})
        r = await _post(
            app,
            "/mesh/teams",
            {"name": "new-proj", "members": ["agent1"]},
        )
        assert r.status_code == 200, r.text
        assert store.members("old-proj") == []
        assert store.team_of("agent1") == "new-proj"
        perms = json.loads(perms_file.read_text())
        assert perms["permissions"]["agent1"]["blackboard_read"] == ["teams/new-proj/*"]
        assert perms["permissions"]["agent1"]["blackboard_write"] == ["teams/new-proj/*"]

    @pytest.mark.asyncio
    async def test_create_duplicate_400(self, team_app):
        app, store, _, _ = team_app
        store.create_team("existing")
        r = await _post(app, "/mesh/teams", {"name": "existing"})
        assert r.status_code == 400
        assert "already exists" in r.json()["detail"]

    @pytest.mark.asyncio
    async def test_create_with_operator_member_rejected_no_partial_state(self, team_app):
        """Operator in initial members must 400 BEFORE the team row is
        created (no partial state)."""
        app, store, _, _ = team_app
        r = await _post(
            app,
            "/mesh/teams",
            {
                "name": "new-proj",
                "members": ["agent1", "operator"],
            },
        )
        assert r.status_code == 400
        assert "system agent" in r.json()["detail"]
        assert not store.team_exists("new-proj")

    @pytest.mark.asyncio
    async def test_add_member_wires_permissions(self, team_app):
        app, store, perms_file, _ = team_app
        store.create_team("proj1")
        r = await _post(app, "/mesh/teams/proj1/members", {"agent": "agent1"})
        assert r.status_code == 200
        assert store.team_of("agent1") == "proj1"
        perms = json.loads(perms_file.read_text())
        assert perms["permissions"]["agent1"]["blackboard_read"] == ["teams/proj1/*"]
        assert perms["permissions"]["agent1"]["blackboard_write"] == ["teams/proj1/*"]

    @pytest.mark.asyncio
    async def test_add_member_idempotent(self, team_app):
        app, store, perms_file, _ = team_app
        store.create_team("proj1")
        await _post(app, "/mesh/teams/proj1/members", {"agent": "agent1"})
        await _post(app, "/mesh/teams/proj1/members", {"agent": "agent1"})
        assert store.members("proj1").count("agent1") == 1
        perms = json.loads(perms_file.read_text())
        read_patterns = perms["permissions"]["agent1"]["blackboard_read"]
        assert read_patterns.count("teams/proj1/*") == 1

    @pytest.mark.asyncio
    async def test_add_member_unknown_team_400(self, team_app):
        app, _, _, _ = team_app
        r = await _post(app, "/mesh/teams/ghost/members", {"agent": "agent1"})
        assert r.status_code == 400
        assert "not found" in r.json()["detail"]

    @pytest.mark.asyncio
    async def test_add_operator_rejected(self, team_app):
        app, store, _, _ = team_app
        store.create_team("proj1")
        r = await _post(app, "/mesh/teams/proj1/members", {"agent": "operator"})
        assert r.status_code == 400
        assert "system agent" in r.json()["detail"]
        assert store.members("proj1") == []

    @pytest.mark.asyncio
    async def test_move_agent_between_teams_evicts_and_rewires(self, team_app):
        """One-team-per-agent: adding to proj2 evicts from proj1 and
        swaps the blackboard patterns."""
        app, store, perms_file, _ = team_app
        store.create_team("proj1")
        store.create_team("proj2")
        await _post(app, "/mesh/teams/proj1/members", {"agent": "agent1"})
        await _post(app, "/mesh/teams/proj2/members", {"agent": "agent1"})
        assert store.members("proj1") == []
        assert store.team_of("agent1") == "proj2"
        perms = json.loads(perms_file.read_text())
        assert perms["permissions"]["agent1"]["blackboard_read"] == ["teams/proj2/*"]
        assert perms["permissions"]["agent1"]["blackboard_write"] == ["teams/proj2/*"]

    @pytest.mark.asyncio
    async def test_remove_member_clears_permissions(self, team_app):
        app, store, perms_file, _ = team_app
        store.create_team("proj1")
        await _post(app, "/mesh/teams/proj1/members", {"agent": "agent1"})
        r = await _delete(app, "/mesh/teams/proj1/members/agent1")
        assert r.status_code == 200
        assert store.team_of("agent1") is None
        perms = json.loads(perms_file.read_text())
        assert perms["permissions"]["agent1"]["blackboard_read"] == []
        assert perms["permissions"]["agent1"]["blackboard_write"] == []

    @pytest.mark.asyncio
    async def test_remove_member_unknown_team_400(self, team_app):
        app, _, _, _ = team_app
        r = await _delete(app, "/mesh/teams/ghost/members/agent1")
        assert r.status_code == 400

    @pytest.mark.asyncio
    async def test_delete_team_strips_all_member_permissions(self, team_app):
        app, store, perms_file, tmp_path = team_app
        await _post(
            app,
            "/mesh/teams",
            {
                "name": "doomed",
                "members": ["agent1", "agent2"],
            },
        )
        r = await _delete(app, "/mesh/teams/doomed")
        assert r.status_code == 200
        assert not store.team_exists("doomed")
        assert not (tmp_path / "teams" / "doomed").exists()
        perms = json.loads(perms_file.read_text())
        for agent in ("agent1", "agent2"):
            assert perms["permissions"][agent]["blackboard_read"] == []
            assert perms["permissions"][agent]["blackboard_write"] == []

    @pytest.mark.asyncio
    async def test_delete_nonexistent_404(self, team_app):
        app, _, _, _ = team_app
        r = await _delete(app, "/mesh/teams/ghost")
        assert r.status_code == 404


class TestBlackboardPermissions:
    def test_add_permissions(self, tmp_path):
        """Adding team permissions grants only the team namespace pattern."""
        perms_file = tmp_path / "permissions.json"
        # Start with empty blackboard (standalone agent)
        perms_file.write_text(
            json.dumps(
                {
                    "permissions": {
                        "bot": {"blackboard_read": [], "blackboard_write": []},
                    }
                }
            )
        )

        with patch("src.cli.config.PERMISSIONS_FILE", perms_file):
            _add_team_blackboard_permissions("bot", "marketing")

        perms = json.loads(perms_file.read_text())
        read = perms["permissions"]["bot"]["blackboard_read"]
        write = perms["permissions"]["bot"]["blackboard_write"]
        assert read == ["teams/marketing/*"]
        assert write == ["teams/marketing/*"]

    def test_add_permissions_appends_alongside_existing_patterns(self, tmp_path):
        """``_add_team_blackboard_permissions`` appends the team
        pattern; pre-existing narrower patterns survive. Pin: the helper
        is additive, never a full replacement of the agent's existing
        ACL. Other tests rely on this when an agent is moved between
        teams (the endpoint move sequence calls remove-then-add, so the
        appended pattern is the only team pattern that ever co-exists
        with non-team patterns)."""
        perms_file = tmp_path / "permissions.json"
        perms_file.write_text(
            json.dumps(
                {
                    "permissions": {
                        "bot": {
                            "blackboard_read": ["context/global", "tasks/bot/*"],
                            "blackboard_write": ["output/bot/*"],
                        },
                    },
                }
            )
        )

        with patch("src.cli.config.PERMISSIONS_FILE", perms_file):
            _add_team_blackboard_permissions("bot", "marketing")

        perms = json.loads(perms_file.read_text())
        read = perms["permissions"]["bot"]["blackboard_read"]
        write = perms["permissions"]["bot"]["blackboard_write"]
        # Existing patterns survived.
        assert "context/global" in read
        assert "tasks/bot/*" in read
        assert "output/bot/*" in write
        # Team pattern was appended.
        assert "teams/marketing/*" in read
        assert "teams/marketing/*" in write

    def test_remove_permissions(self, tmp_path):
        """Removing team permissions clears ALL blackboard access."""
        perms_file = tmp_path / "permissions.json"
        perms_file.write_text(
            json.dumps(
                {
                    "permissions": {
                        "bot": {
                            "blackboard_read": ["teams/marketing/*"],
                            "blackboard_write": ["teams/marketing/*"],
                        },
                    }
                }
            )
        )

        with patch("src.cli.config.PERMISSIONS_FILE", perms_file):
            _remove_team_blackboard_permissions("bot", "marketing")

        perms = json.loads(perms_file.read_text())
        assert perms["permissions"]["bot"]["blackboard_read"] == []
        assert perms["permissions"]["bot"]["blackboard_write"] == []

    def test_remove_permissions_only_strips_target_project(self, tmp_path):
        """``_remove_team_blackboard_permissions`` is targeted: it strips
        the named team's pattern only. Other patterns (including other
        teams' patterns and non-team patterns like ``context/global``)
        survive. Pin: leaving team A does not nuke an agent's access
        to team B or to fleet-wide / per-agent patterns."""
        perms_file = tmp_path / "permissions.json"
        perms_file.write_text(
            json.dumps(
                {
                    "permissions": {
                        "bot": {
                            "blackboard_read": [
                                "context/global",
                                "teams/marketing/*",
                                "teams/sales/*",
                            ],
                            "blackboard_write": [
                                "output/bot/*",
                                "teams/marketing/*",
                            ],
                        },
                    },
                }
            )
        )

        with patch("src.cli.config.PERMISSIONS_FILE", perms_file):
            _remove_team_blackboard_permissions("bot", "marketing")

        perms = json.loads(perms_file.read_text())
        read = perms["permissions"]["bot"]["blackboard_read"]
        write = perms["permissions"]["bot"]["blackboard_write"]
        # Marketing is gone.
        assert "teams/marketing/*" not in read
        assert "teams/marketing/*" not in write
        # Sales survives (other team).
        assert "teams/sales/*" in read
        # Non-team patterns survive.
        assert "context/global" in read
        assert "output/bot/*" in write


class TestRemoveAgentCleansTeam:
    def test_remove_agent_also_removes_from_team(self, tmp_path, monkeypatch):
        import yaml

        teams_db = tmp_path / "teams.db"
        monkeypatch.setenv("OPENLEGION_TEAMS_DB", str(teams_db))
        store = TeamStore(db_path=str(teams_db))
        store.create_team("proj1")
        store.add_member("proj1", "agent1")
        store.add_member("proj1", "agent2")
        store.set_agent_goals("agent1", ["standing goal"])

        agents_file = tmp_path / "agents.yaml"
        agents_file.write_text(
            yaml.dump(
                {
                    "agents": {"agent1": {"role": "test"}, "agent2": {"role": "test2"}},
                }
            )
        )

        perms_file = tmp_path / "permissions.json"
        perms_file.write_text(
            json.dumps(
                {
                    "permissions": {
                        "agent1": {
                            "blackboard_read": ["teams/proj1/*"],
                            "blackboard_write": ["teams/proj1/*"],
                        },
                        "agent2": {
                            "blackboard_read": ["teams/proj1/*"],
                            "blackboard_write": ["teams/proj1/*"],
                        },
                    }
                }
            )
        )

        with (
            patch("src.cli.config.AGENTS_FILE", agents_file),
            patch("src.cli.config.PERMISSIONS_FILE", perms_file),
        ):
            _remove_agent("agent1")

        # agent1 removed from team membership (agent2 untouched);
        # standing goals cleared with it.
        assert store.team_of("agent1") is None
        assert store.team_of("agent2") == "proj1"
        assert store.get_agent_goals("agent1") is None

        # agent1 removed from agents.yaml
        agents = yaml.safe_load(agents_file.read_text())
        assert "agent1" not in agents["agents"]

        # agent1 removed from permissions
        perms = json.loads(perms_file.read_text())
        assert "agent1" not in perms["permissions"]


class TestMeshClientKeyScoping:
    """MeshClient transparently prefixes blackboard keys by team."""

    def test_scope_key_with_project(self):
        from src.agent.mesh_client import MeshClient

        client = MeshClient("http://mesh:8420", "bot1", team_name="alpha")
        assert client._scope_key("context/market") == "teams/alpha/context/market"
        assert client._scope_key("goals/researcher") == "teams/alpha/goals/researcher"

    def test_scope_key_standalone(self):
        from src.agent.mesh_client import MeshClient

        client = MeshClient("http://mesh:8420", "bot1", team_name=None)
        assert client._scope_key("context/market") == "context/market"

    def test_scope_key_empty(self):
        from src.agent.mesh_client import MeshClient

        client = MeshClient("http://mesh:8420", "bot1", team_name="alpha")
        assert client._scope_key("") == "teams/alpha/"

    def _mock_client(self, project_name, response_data, status_code=200):
        """Create a MeshClient with mocked httpx transport."""
        from unittest.mock import AsyncMock, MagicMock

        from src.agent.mesh_client import MeshClient

        client = MeshClient("http://mesh:8420", "bot1", team_name=project_name)
        mock_http = MagicMock()
        mock_http.is_closed = False
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json.return_value = response_data
        mock_resp.raise_for_status = MagicMock()
        mock_http.get = AsyncMock(return_value=mock_resp)
        mock_http.put = AsyncMock(return_value=mock_resp)
        mock_http.delete = AsyncMock(return_value=mock_resp)
        client._client = mock_http
        client._trace_headers = lambda: {}
        return client, mock_http

    @pytest.mark.asyncio
    async def test_read_blackboard_uses_scoped_key(self):
        """read_blackboard sends the team-prefixed key in the URL."""
        client, mock_http = self._mock_client(
            "alpha",
            {"key": "teams/alpha/context/market", "value": {"data": 1}},
        )
        result = await client.read_blackboard("context/market")
        url = mock_http.get.call_args[0][0]
        assert "teams/alpha/context/market" in url
        assert result["value"] == {"data": 1}

    @pytest.mark.asyncio
    async def test_write_blackboard_uses_scoped_key(self):
        """write_blackboard sends the team-prefixed key in the URL."""
        client, mock_http = self._mock_client(
            "alpha",
            {"key": "teams/alpha/goals/v1", "version": 1},
        )
        await client.write_blackboard("goals/v1", {"objective": "test"})
        url = mock_http.put.call_args[0][0]
        assert "teams/alpha/goals/v1" in url

    @pytest.mark.asyncio
    async def test_delete_blackboard_project_scoping(self):
        """delete_blackboard mirrors write_blackboard's three-way scoping."""
        client, mock_http = self._mock_client("alpha", {"deleted": True})
        # Default: caller's own team prefix.
        await client.delete_blackboard("goals/researcher")
        url = mock_http.delete.call_args[0][0]
        assert "teams/alpha/goals/researcher" in url
        # Explicit team= override (operator clearing a team agent's key).
        await client.delete_blackboard("goals/researcher", team="beta")
        url = mock_http.delete.call_args[0][0]
        assert "teams/beta/goals/researcher" in url
        # global_scope bypasses both.
        await client.delete_blackboard("goals/researcher", global_scope=True)
        url = mock_http.delete.call_args[0][0]
        assert "teams/" not in url
        assert "goals/researcher" in url

    @pytest.mark.asyncio
    async def test_list_blackboard_scopes_prefix_and_strips_keys(self):
        """list_blackboard scopes the prefix and strips team prefix from returned keys."""
        client, mock_http = self._mock_client(
            "alpha",
            [
                {"key": "teams/alpha/context/market", "value": {"data": 1}},
                {"key": "teams/alpha/context/competitor", "value": {"data": 2}},
            ],
        )
        entries = await client.list_blackboard("context/")
        # Prefix was scoped in the request
        params = mock_http.get.call_args[1]["params"]
        assert params["prefix"] == "teams/alpha/context/"
        # Keys were stripped in the response
        assert entries[0]["key"] == "context/market"
        assert entries[1]["key"] == "context/competitor"

    @pytest.mark.asyncio
    async def test_standalone_blackboard_no_scoping(self):
        """Standalone agent's blackboard calls use raw keys (no prefix)."""
        client, mock_http = self._mock_client(
            None,
            {"key": "context/market", "value": {"data": 1}},
        )
        await client.read_blackboard("context/market")
        url = mock_http.get.call_args[0][0]
        # No team prefix — raw key in URL
        assert "teams/" not in url
        assert "context/market" in url


class TestCrossProjectPermissionIsolation:
    """Verify PermissionMatrix enforces cross-team isolation."""

    def test_project_agent_can_access_own_namespace(self, tmp_path):
        """Agent with teams/alpha/* can read/write under that namespace."""
        from src.host.permissions import PermissionMatrix

        perms_file = tmp_path / "perms.json"
        perms_file.write_text(
            json.dumps(
                {
                    "permissions": {
                        "bot1": {
                            "blackboard_read": ["teams/alpha/*"],
                            "blackboard_write": ["teams/alpha/*"],
                        },
                    }
                }
            )
        )
        pm = PermissionMatrix(config_path=str(perms_file))

        assert pm.can_read_blackboard("bot1", "teams/alpha/context/market")
        assert pm.can_read_blackboard("bot1", "teams/alpha/goals/v1")
        assert pm.can_write_blackboard("bot1", "teams/alpha/tasks/todo")

    def test_project_agent_cannot_access_other_project(self, tmp_path):
        """Agent with teams/alpha/* CANNOT access teams/beta/*."""
        from src.host.permissions import PermissionMatrix

        perms_file = tmp_path / "perms.json"
        perms_file.write_text(
            json.dumps(
                {
                    "permissions": {
                        "bot1": {
                            "blackboard_read": ["teams/alpha/*"],
                            "blackboard_write": ["teams/alpha/*"],
                        },
                    }
                }
            )
        )
        pm = PermissionMatrix(config_path=str(perms_file))

        assert not pm.can_read_blackboard("bot1", "teams/beta/context/market")
        assert not pm.can_write_blackboard("bot1", "teams/beta/goals/v1")
        assert not pm.can_read_blackboard("bot1", "teams/beta/anything")

    def test_project_agent_cannot_access_global_keys(self, tmp_path):
        """Agent with teams/alpha/* CANNOT access unprefixed global keys."""
        from src.host.permissions import PermissionMatrix

        perms_file = tmp_path / "perms.json"
        perms_file.write_text(
            json.dumps(
                {
                    "permissions": {
                        "bot1": {
                            "blackboard_read": ["teams/alpha/*"],
                            "blackboard_write": ["teams/alpha/*"],
                        },
                    }
                }
            )
        )
        pm = PermissionMatrix(config_path=str(perms_file))

        assert not pm.can_read_blackboard("bot1", "context/market")
        assert not pm.can_read_blackboard("bot1", "tasks/todo")
        assert not pm.can_write_blackboard("bot1", "goals/v1")

    def test_standalone_agent_has_no_blackboard_access(self, tmp_path):
        """Standalone agent (empty patterns) cannot access any blackboard key."""
        from src.host.permissions import PermissionMatrix

        perms_file = tmp_path / "perms.json"
        perms_file.write_text(
            json.dumps(
                {
                    "permissions": {
                        "solo": {
                            "blackboard_read": [],
                            "blackboard_write": [],
                        },
                    }
                }
            )
        )
        pm = PermissionMatrix(config_path=str(perms_file))

        assert not pm.can_read_blackboard("solo", "context/anything")
        assert not pm.can_read_blackboard("solo", "teams/alpha/data")
        assert not pm.can_write_blackboard("solo", "anything")

    def test_two_projects_fully_isolated(self, tmp_path):
        """Two agents in different teams have zero overlap in blackboard access."""
        from src.host.permissions import PermissionMatrix

        perms_file = tmp_path / "perms.json"
        perms_file.write_text(
            json.dumps(
                {
                    "permissions": {
                        "alpha_worker": {
                            "blackboard_read": ["teams/alpha/*"],
                            "blackboard_write": ["teams/alpha/*"],
                        },
                        "beta_worker": {
                            "blackboard_read": ["teams/beta/*"],
                            "blackboard_write": ["teams/beta/*"],
                        },
                    }
                }
            )
        )
        pm = PermissionMatrix(config_path=str(perms_file))

        # alpha_worker can access alpha, not beta
        assert pm.can_read_blackboard("alpha_worker", "teams/alpha/data")
        assert not pm.can_read_blackboard("alpha_worker", "teams/beta/data")

        # beta_worker can access beta, not alpha
        assert pm.can_read_blackboard("beta_worker", "teams/beta/data")
        assert not pm.can_read_blackboard("beta_worker", "teams/alpha/data")

        # Neither can access global keys
        assert not pm.can_read_blackboard("alpha_worker", "context/global")
        assert not pm.can_read_blackboard("beta_worker", "context/global")
