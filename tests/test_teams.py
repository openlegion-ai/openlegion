"""Tests for multi-team support: CRUD, config loading, permissions wiring.

These tests exercise the underlying ``_create_team`` / ``_load_teams``
helpers in ``src/cli/config.py``, which still carry their legacy names as
the back-compat surface for the project→team rename. The behaviour they
verify IS the team-domain logic; only the Python identifiers and the
filesystem layout (``config/projects/``) stay project-named for one more
release. PR 3 will sunset both.
"""

import json
from unittest.mock import patch

import pytest
import yaml

from src.cli.config import (
    _add_agent_to_team,
    _add_team_blackboard_permissions,
    _create_team,
    _delete_team,
    _get_agent_team,
    _load_config,
    _load_teams,
    _remove_agent,
    _remove_agent_from_team,
    _remove_team_blackboard_permissions,
    _validate_team_name,
)


class TestValidateProjectName:
    def test_valid_names(self):
        assert _validate_team_name("my-project") == "my-project"
        assert _validate_team_name("project1") == "project1"
        assert _validate_team_name("A_B-C") == "A_B-C"

    def test_invalid_empty(self):
        with pytest.raises(ValueError, match="Invalid team name"):
            _validate_team_name("")

    def test_invalid_special_chars(self):
        with pytest.raises(ValueError, match="Invalid team name"):
            _validate_team_name("my project")

    def test_invalid_start_char(self):
        with pytest.raises(ValueError, match="Invalid team name"):
            _validate_team_name("-start")

    def test_path_traversal_rejected(self):
        for name in ["../escape", "foo/bar", "..", "./current", "a/../b"]:
            with pytest.raises(ValueError, match="Invalid team name"):
                _validate_team_name(name)

    def test_max_length_boundary(self):
        # 64 chars should pass
        assert _validate_team_name("a" * 64) == "a" * 64
        # 65 chars should fail
        with pytest.raises(ValueError, match="Invalid team name"):
            _validate_team_name("a" * 65)

    def test_underscore_start_rejected(self):
        with pytest.raises(ValueError, match="Invalid team name"):
            _validate_team_name("_underscore")


class TestLoadProjects:
    def test_no_projects_dir(self, tmp_path):
        with patch("src.cli.config.TEAMS_DIR", tmp_path / "nonexistent"):
            result = _load_teams()
        assert result == {}

    def test_load_single_project(self, tmp_path):
        projects_dir = tmp_path / "projects"
        proj_dir = projects_dir / "my-proj"
        proj_dir.mkdir(parents=True)
        meta = {"name": "my-proj", "description": "Test", "members": ["agent1"]}
        (proj_dir / "metadata.yaml").write_text(yaml.dump(meta))

        with patch("src.cli.config.TEAMS_DIR", projects_dir):
            result = _load_teams()

        assert "my-proj" in result
        assert result["my-proj"]["description"] == "Test"
        assert result["my-proj"]["members"] == ["agent1"]

    def test_load_multiple_projects(self, tmp_path):
        projects_dir = tmp_path / "projects"
        for name in ["alpha", "beta"]:
            d = projects_dir / name
            d.mkdir(parents=True)
            (d / "metadata.yaml").write_text(yaml.dump({"name": name, "members": []}))

        with patch("src.cli.config.TEAMS_DIR", projects_dir):
            result = _load_teams()

        assert len(result) == 2
        assert "alpha" in result
        assert "beta" in result

    def test_keyed_by_directory_name(self, tmp_path):
        """Projects are keyed by directory name, not metadata 'name' field."""
        projects_dir = tmp_path / "projects"
        proj_dir = projects_dir / "dir-name"
        proj_dir.mkdir(parents=True)
        # metadata has a different 'name' than directory
        (proj_dir / "metadata.yaml").write_text(yaml.dump({
            "name": "metadata-name", "members": [],
        }))

        with patch("src.cli.config.TEAMS_DIR", projects_dir):
            result = _load_teams()

        assert "dir-name" in result
        assert "metadata-name" not in result

    def test_corrupted_metadata_skipped(self, tmp_path):
        projects_dir = tmp_path / "projects"
        d = projects_dir / "bad"
        d.mkdir(parents=True)
        (d / "metadata.yaml").write_text("{{invalid yaml")

        with patch("src.cli.config.TEAMS_DIR", projects_dir):
            result = _load_teams()

        assert result == {}


class TestGetAgentProject:
    def test_agent_in_project(self, tmp_path):
        projects_dir = tmp_path / "projects"
        d = projects_dir / "proj1"
        d.mkdir(parents=True)
        (d / "metadata.yaml").write_text(yaml.dump({
            "name": "proj1", "members": ["agent1", "agent2"],
        }))

        with patch("src.cli.config.TEAMS_DIR", projects_dir):
            assert _get_agent_team("agent1") == "proj1"
            assert _get_agent_team("agent2") == "proj1"

    def test_standalone_agent(self, tmp_path):
        projects_dir = tmp_path / "projects"
        projects_dir.mkdir(parents=True)

        with patch("src.cli.config.TEAMS_DIR", projects_dir):
            assert _get_agent_team("standalone") is None


class TestCreateProject:
    def test_create_basic(self, tmp_path):
        projects_dir = tmp_path / "projects"
        perms_file = tmp_path / "permissions.json"
        perms_file.write_text(json.dumps({"permissions": {}}))

        with (
            patch("src.cli.config.TEAMS_DIR", projects_dir),
            patch("src.cli.config.PERMISSIONS_FILE", perms_file),
        ):
            _create_team("test-proj", description="A test project")

        meta_file = projects_dir / "test-proj" / "metadata.yaml"
        assert meta_file.exists()
        data = yaml.safe_load(meta_file.read_text())
        assert data["name"] == "test-proj"
        assert data["description"] == "A test project"
        assert data["members"] == []

        team_md = projects_dir / "test-proj" / "team.md"
        assert team_md.exists()
        assert "test-proj" in team_md.read_text()

        workflows_dir = projects_dir / "test-proj" / "workflows"
        assert workflows_dir.is_dir()

    def test_create_with_members(self, tmp_path):
        projects_dir = tmp_path / "projects"
        perms_file = tmp_path / "permissions.json"
        # Agents start with empty blackboard (standalone)
        perms_file.write_text(json.dumps({
            "permissions": {
                "agent1": {"blackboard_read": [], "blackboard_write": []},
                "agent2": {"blackboard_read": [], "blackboard_write": []},
            }
        }))

        with (
            patch("src.cli.config.TEAMS_DIR", projects_dir),
            patch("src.cli.config.PERMISSIONS_FILE", perms_file),
        ):
            _create_team("my-proj", members=["agent1", "agent2"])

        meta = yaml.safe_load((projects_dir / "my-proj" / "metadata.yaml").read_text())
        assert meta["members"] == ["agent1", "agent2"]

        # Only the project-specific pattern is granted (no shared patterns)
        perms = json.loads(perms_file.read_text())
        assert perms["permissions"]["agent1"]["blackboard_read"] == ["teams/my-proj/*"]
        assert perms["permissions"]["agent1"]["blackboard_write"] == ["teams/my-proj/*"]
        assert perms["permissions"]["agent2"]["blackboard_read"] == ["teams/my-proj/*"]
        assert perms["permissions"]["agent2"]["blackboard_write"] == ["teams/my-proj/*"]

    def test_create_moves_agent_from_existing_project(self, tmp_path):
        """Creating a project with members already in another project moves them."""
        projects_dir = tmp_path / "projects"
        # Pre-existing project with agent1
        old_dir = projects_dir / "old-proj"
        old_dir.mkdir(parents=True)
        (old_dir / "metadata.yaml").write_text(yaml.dump({
            "name": "old-proj", "members": ["agent1"],
        }))

        perms_file = tmp_path / "permissions.json"
        perms_file.write_text(json.dumps({
            "permissions": {
                "agent1": {
                    "blackboard_read": ["teams/old-proj/*"],
                    "blackboard_write": ["teams/old-proj/*"],
                },
            }
        }))

        with (
            patch("src.cli.config.TEAMS_DIR", projects_dir),
            patch("src.cli.config.PERMISSIONS_FILE", perms_file),
        ):
            _create_team("new-proj", members=["agent1"])

        # Agent should be in new project only
        new_meta = yaml.safe_load((projects_dir / "new-proj" / "metadata.yaml").read_text())
        assert "agent1" in new_meta["members"]

        # Removed from old project
        old_meta = yaml.safe_load((old_dir / "metadata.yaml").read_text())
        assert "agent1" not in old_meta["members"]

        # Permissions updated: new project pattern present, old one gone
        perms = json.loads(perms_file.read_text())
        assert perms["permissions"]["agent1"]["blackboard_read"] == ["teams/new-proj/*"]
        assert perms["permissions"]["agent1"]["blackboard_write"] == ["teams/new-proj/*"]

    def test_create_duplicate_raises(self, tmp_path):
        projects_dir = tmp_path / "projects"
        (projects_dir / "existing").mkdir(parents=True)
        perms_file = tmp_path / "permissions.json"
        perms_file.write_text(json.dumps({"permissions": {}}))

        with (
            patch("src.cli.config.TEAMS_DIR", projects_dir),
            patch("src.cli.config.PERMISSIONS_FILE", perms_file),
        ):
            with pytest.raises(ValueError, match="already exists"):
                _create_team("existing")

    def test_create_with_operator_member_rejected_no_partial_state(self, tmp_path):
        """Operator in initial members must raise BEFORE the project dir is created (no partial state)."""
        projects_dir = tmp_path / "projects"
        projects_dir.mkdir(parents=True)
        perms_file = tmp_path / "permissions.json"
        perms_file.write_text(json.dumps({"permissions": {}}))

        with (
            patch("src.cli.config.TEAMS_DIR", projects_dir),
            patch("src.cli.config.PERMISSIONS_FILE", perms_file),
        ):
            with pytest.raises(ValueError, match="system agent"):
                _create_team("new-proj", members=["agent1", "operator"])
        # Project directory must not exist — rejection happens before any filesystem writes
        assert not (projects_dir / "new-proj").exists()


class TestDeleteProject:
    def test_delete_team(self, tmp_path):
        projects_dir = tmp_path / "projects"
        proj_dir = projects_dir / "doomed"
        proj_dir.mkdir(parents=True)
        (proj_dir / "metadata.yaml").write_text(yaml.dump({
            "name": "doomed", "members": ["agent1"],
        }))
        (proj_dir / "team.md").write_text("# doomed")

        perms_file = tmp_path / "permissions.json"
        perms_file.write_text(json.dumps({
            "permissions": {
                "agent1": {
                    "blackboard_read": ["teams/doomed/*"],
                    "blackboard_write": ["teams/doomed/*"],
                },
            }
        }))

        with (
            patch("src.cli.config.TEAMS_DIR", projects_dir),
            patch("src.cli.config.PERMISSIONS_FILE", perms_file),
        ):
            _delete_team("doomed")

        assert not proj_dir.exists()

        # Agent becomes standalone — all blackboard permissions cleared
        perms = json.loads(perms_file.read_text())
        assert perms["permissions"]["agent1"]["blackboard_read"] == []
        assert perms["permissions"]["agent1"]["blackboard_write"] == []

    def test_delete_nonexistent_raises(self, tmp_path):
        with patch("src.cli.config.TEAMS_DIR", tmp_path / "nope"):
            with pytest.raises(ValueError, match="not found"):
                _delete_team("ghost")


class TestAddRemoveAgentProject:
    def _setup(self, tmp_path):
        projects_dir = tmp_path / "projects"
        proj_dir = projects_dir / "proj1"
        proj_dir.mkdir(parents=True)
        (proj_dir / "metadata.yaml").write_text(yaml.dump({
            "name": "proj1", "members": [],
        }))

        perms_file = tmp_path / "permissions.json"
        # Standalone agent: no blackboard permissions
        perms_file.write_text(json.dumps({
            "permissions": {
                "agent1": {
                    "blackboard_read": [],
                    "blackboard_write": [],
                },
            }
        }))
        return projects_dir, perms_file

    def test_add_agent(self, tmp_path):
        projects_dir, perms_file = self._setup(tmp_path)

        with (
            patch("src.cli.config.TEAMS_DIR", projects_dir),
            patch("src.cli.config.PERMISSIONS_FILE", perms_file),
        ):
            _add_agent_to_team("proj1", "agent1")

        meta = yaml.safe_load((projects_dir / "proj1" / "metadata.yaml").read_text())
        assert "agent1" in meta["members"]

        # Agent gets only project-scoped blackboard permissions
        perms = json.loads(perms_file.read_text())
        assert perms["permissions"]["agent1"]["blackboard_read"] == ["teams/proj1/*"]
        assert perms["permissions"]["agent1"]["blackboard_write"] == ["teams/proj1/*"]

    def test_add_agent_idempotent(self, tmp_path):
        projects_dir, perms_file = self._setup(tmp_path)

        with (
            patch("src.cli.config.TEAMS_DIR", projects_dir),
            patch("src.cli.config.PERMISSIONS_FILE", perms_file),
        ):
            _add_agent_to_team("proj1", "agent1")
            _add_agent_to_team("proj1", "agent1")

        meta = yaml.safe_load((projects_dir / "proj1" / "metadata.yaml").read_text())
        assert meta["members"].count("agent1") == 1

        # Pattern should not be duplicated
        perms = json.loads(perms_file.read_text())
        read_patterns = perms["permissions"]["agent1"]["blackboard_read"]
        assert read_patterns.count("teams/proj1/*") == 1

    def test_add_to_nonexistent_project_raises(self, tmp_path):
        projects_dir = tmp_path / "projects"
        projects_dir.mkdir(parents=True)
        perms_file = tmp_path / "permissions.json"
        perms_file.write_text(json.dumps({"permissions": {"agent1": {}}}))

        with (
            patch("src.cli.config.TEAMS_DIR", projects_dir),
            patch("src.cli.config.PERMISSIONS_FILE", perms_file),
        ):
            with pytest.raises(ValueError, match="not found"):
                _add_agent_to_team("ghost", "agent1")

    def test_add_operator_rejected(self, tmp_path):
        """Operator is a system agent and must never be assignable to a project."""
        projects_dir = tmp_path / "projects"
        proj_dir = projects_dir / "proj1"
        proj_dir.mkdir(parents=True)
        (proj_dir / "metadata.yaml").write_text(yaml.dump({"name": "proj1", "members": []}))
        perms_file = tmp_path / "permissions.json"
        perms_file.write_text(json.dumps({"permissions": {"operator": {}}}))

        with (
            patch("src.cli.config.TEAMS_DIR", projects_dir),
            patch("src.cli.config.PERMISSIONS_FILE", perms_file),
        ):
            with pytest.raises(ValueError, match="system agent"):
                _add_agent_to_team("proj1", "operator")
        # Members list must remain empty — no partial write before the rejection
        meta = yaml.safe_load((proj_dir / "metadata.yaml").read_text())
        assert meta["members"] == []

    def test_remove_agent(self, tmp_path):
        projects_dir = tmp_path / "projects"
        proj_dir = projects_dir / "proj1"
        proj_dir.mkdir(parents=True)
        (proj_dir / "metadata.yaml").write_text(yaml.dump({
            "name": "proj1", "members": ["agent1"],
        }))

        perms_file = tmp_path / "permissions.json"
        perms_file.write_text(json.dumps({
            "permissions": {
                "agent1": {
                    "blackboard_read": ["teams/proj1/*"],
                    "blackboard_write": ["teams/proj1/*"],
                },
            }
        }))

        with (
            patch("src.cli.config.TEAMS_DIR", projects_dir),
            patch("src.cli.config.PERMISSIONS_FILE", perms_file),
        ):
            _remove_agent_from_team("proj1", "agent1")

        meta = yaml.safe_load((proj_dir / "metadata.yaml").read_text())
        assert "agent1" not in meta["members"]

        # Agent becomes standalone — all blackboard permissions cleared
        perms = json.loads(perms_file.read_text())
        assert perms["permissions"]["agent1"]["blackboard_read"] == []
        assert perms["permissions"]["agent1"]["blackboard_write"] == []

    def test_move_agent_between_projects(self, tmp_path):
        projects_dir = tmp_path / "projects"
        for pname in ("proj1", "proj2"):
            d = projects_dir / pname
            d.mkdir(parents=True)
            members = ["agent1"] if pname == "proj1" else []
            (d / "metadata.yaml").write_text(yaml.dump({
                "name": pname, "members": members,
            }))

        perms_file = tmp_path / "permissions.json"
        perms_file.write_text(json.dumps({
            "permissions": {
                "agent1": {
                    "blackboard_read": ["teams/proj1/*"],
                    "blackboard_write": ["teams/proj1/*"],
                },
            }
        }))

        with (
            patch("src.cli.config.TEAMS_DIR", projects_dir),
            patch("src.cli.config.PERMISSIONS_FILE", perms_file),
        ):
            _add_agent_to_team("proj2", "agent1")

        # Removed from proj1
        meta1 = yaml.safe_load((projects_dir / "proj1" / "metadata.yaml").read_text())
        assert "agent1" not in meta1["members"]

        # Added to proj2
        meta2 = yaml.safe_load((projects_dir / "proj2" / "metadata.yaml").read_text())
        assert "agent1" in meta2["members"]

        # Permissions updated: new project pattern only
        perms = json.loads(perms_file.read_text())
        assert perms["permissions"]["agent1"]["blackboard_read"] == ["teams/proj2/*"]
        assert perms["permissions"]["agent1"]["blackboard_write"] == ["teams/proj2/*"]


class TestBlackboardPermissions:
    def test_add_permissions(self, tmp_path):
        """Adding project permissions grants only the project namespace pattern."""
        perms_file = tmp_path / "permissions.json"
        # Start with empty blackboard (standalone agent)
        perms_file.write_text(json.dumps({
            "permissions": {
                "bot": {"blackboard_read": [], "blackboard_write": []},
            }
        }))

        with patch("src.cli.config.PERMISSIONS_FILE", perms_file):
            _add_team_blackboard_permissions("bot", "marketing")

        perms = json.loads(perms_file.read_text())
        read = perms["permissions"]["bot"]["blackboard_read"]
        write = perms["permissions"]["bot"]["blackboard_write"]
        assert read == ["teams/marketing/*"]
        assert write == ["teams/marketing/*"]

    def test_add_permissions_appends_alongside_existing_patterns(self, tmp_path):
        """``_add_team_blackboard_permissions`` appends the project
        pattern; pre-existing narrower patterns survive. Pin: the helper
        is additive, never a full replacement of the agent's existing
        ACL. Other tests rely on this when an agent is moved between
        projects (the ``_add_agent_to_team`` move sequence calls
        remove-then-add, so the appended pattern is the only project
        pattern that ever co-exists with non-project patterns)."""
        perms_file = tmp_path / "permissions.json"
        perms_file.write_text(json.dumps({
            "permissions": {
                "bot": {
                    "blackboard_read": ["context/global", "tasks/bot/*"],
                    "blackboard_write": ["output/bot/*"],
                },
            },
        }))

        with patch("src.cli.config.PERMISSIONS_FILE", perms_file):
            _add_team_blackboard_permissions("bot", "marketing")

        perms = json.loads(perms_file.read_text())
        read = perms["permissions"]["bot"]["blackboard_read"]
        write = perms["permissions"]["bot"]["blackboard_write"]
        # Existing patterns survived.
        assert "context/global" in read
        assert "tasks/bot/*" in read
        assert "output/bot/*" in write
        # Project pattern was appended.
        assert "teams/marketing/*" in read
        assert "teams/marketing/*" in write

    def test_remove_permissions(self, tmp_path):
        """Removing project permissions clears ALL blackboard access."""
        perms_file = tmp_path / "permissions.json"
        perms_file.write_text(json.dumps({
            "permissions": {
                "bot": {
                    "blackboard_read": ["teams/marketing/*"],
                    "blackboard_write": ["teams/marketing/*"],
                },
            }
        }))

        with patch("src.cli.config.PERMISSIONS_FILE", perms_file):
            _remove_team_blackboard_permissions("bot", "marketing")

        perms = json.loads(perms_file.read_text())
        assert perms["permissions"]["bot"]["blackboard_read"] == []
        assert perms["permissions"]["bot"]["blackboard_write"] == []

    def test_remove_permissions_only_strips_target_project(self, tmp_path):
        """``_remove_team_blackboard_permissions`` is targeted: it strips
        the named project's pattern only. Other patterns (including other
        projects' patterns and non-project patterns like ``context/global``)
        survive. Pin: leaving project A does not nuke an agent's access
        to project B or to fleet-wide / per-agent patterns."""
        perms_file = tmp_path / "permissions.json"
        perms_file.write_text(json.dumps({
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
        }))

        with patch("src.cli.config.PERMISSIONS_FILE", perms_file):
            _remove_team_blackboard_permissions("bot", "marketing")

        perms = json.loads(perms_file.read_text())
        read = perms["permissions"]["bot"]["blackboard_read"]
        write = perms["permissions"]["bot"]["blackboard_write"]
        # Marketing is gone.
        assert "teams/marketing/*" not in read
        assert "teams/marketing/*" not in write
        # Sales survives (other project).
        assert "teams/sales/*" in read
        # Non-project patterns survive.
        assert "context/global" in read
        assert "output/bot/*" in write


class TestLoadConfigWithProjects:
    def test_config_includes_projects(self, tmp_path):
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"
        projects_dir = tmp_path / "config" / "projects"

        config_file.write_text(yaml.dump({"mesh": {"port": 8420}}))
        agents_file.write_text(yaml.dump({"agents": {"bot1": {"role": "test"}}}))

        proj_dir = projects_dir / "myproject"
        proj_dir.mkdir(parents=True)
        (proj_dir / "metadata.yaml").write_text(yaml.dump({
            "name": "myproject", "members": ["bot1"],
        }))

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
            patch("src.cli.config.TEAMS_DIR", projects_dir),
        ):
            cfg = _load_config(config_file)

        assert "myproject" in cfg["teams"]
        assert cfg["_agent_teams"]["bot1"] == "myproject"

    def test_config_no_projects(self, tmp_path):
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"
        projects_dir = tmp_path / "nonexistent"

        config_file.write_text(yaml.dump({"mesh": {"port": 8420}}))
        agents_file.write_text(yaml.dump({"agents": {}}))

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
            patch("src.cli.config.TEAMS_DIR", projects_dir),
        ):
            cfg = _load_config(config_file)

        assert cfg["teams"] == {}
        assert cfg["_agent_teams"] == {}


class TestRemoveAgentCleansProject:
    def test_remove_agent_also_removes_from_project(self, tmp_path):
        projects_dir = tmp_path / "projects"
        proj_dir = projects_dir / "proj1"
        proj_dir.mkdir(parents=True)
        (proj_dir / "metadata.yaml").write_text(yaml.dump({
            "name": "proj1", "members": ["agent1", "agent2"],
        }))

        agents_file = tmp_path / "agents.yaml"
        agents_file.write_text(yaml.dump({
            "agents": {"agent1": {"role": "test"}, "agent2": {"role": "test2"}},
        }))

        perms_file = tmp_path / "permissions.json"
        perms_file.write_text(json.dumps({
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
        }))

        with (
            patch("src.cli.config.TEAMS_DIR", projects_dir),
            patch("src.cli.config.AGENTS_FILE", agents_file),
            patch("src.cli.config.PERMISSIONS_FILE", perms_file),
        ):
            _remove_agent("agent1")

        # agent1 removed from project
        meta = yaml.safe_load((proj_dir / "metadata.yaml").read_text())
        assert "agent1" not in meta["members"]
        assert "agent2" in meta["members"]

        # agent1 removed from agents.yaml
        agents = yaml.safe_load(agents_file.read_text())
        assert "agent1" not in agents["agents"]

        # agent1 removed from permissions
        perms = json.loads(perms_file.read_text())
        assert "agent1" not in perms["permissions"]


class TestMeshClientKeyScoping:
    """MeshClient transparently prefixes blackboard keys by project."""

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
        """read_blackboard sends the project-prefixed key in the URL."""
        client, mock_http = self._mock_client(
            "alpha", {"key": "teams/alpha/context/market", "value": {"data": 1}},
        )
        result = await client.read_blackboard("context/market")
        url = mock_http.get.call_args[0][0]
        assert "teams/alpha/context/market" in url
        assert result["value"] == {"data": 1}

    @pytest.mark.asyncio
    async def test_write_blackboard_uses_scoped_key(self):
        """write_blackboard sends the project-prefixed key in the URL."""
        client, mock_http = self._mock_client(
            "alpha", {"key": "teams/alpha/goals/v1", "version": 1},
        )
        await client.write_blackboard("goals/v1", {"objective": "test"})
        url = mock_http.put.call_args[0][0]
        assert "teams/alpha/goals/v1" in url

    @pytest.mark.asyncio
    async def test_delete_blackboard_project_scoping(self):
        """delete_blackboard mirrors write_blackboard's three-way scoping."""
        client, mock_http = self._mock_client("alpha", {"deleted": True})
        # Default: caller's own project prefix.
        await client.delete_blackboard("goals/researcher")
        url = mock_http.delete.call_args[0][0]
        assert "teams/alpha/goals/researcher" in url
        # Explicit project= override (operator clearing a team agent's key).
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
        """list_blackboard scopes the prefix and strips project prefix from returned keys."""
        client, mock_http = self._mock_client("alpha", [
            {"key": "teams/alpha/context/market", "value": {"data": 1}},
            {"key": "teams/alpha/context/competitor", "value": {"data": 2}},
        ])
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
            None, {"key": "context/market", "value": {"data": 1}},
        )
        await client.read_blackboard("context/market")
        url = mock_http.get.call_args[0][0]
        # No project prefix — raw key in URL
        assert "teams/" not in url
        assert "context/market" in url


class TestCrossProjectPermissionIsolation:
    """Verify PermissionMatrix enforces cross-project isolation."""

    def test_project_agent_can_access_own_namespace(self, tmp_path):
        """Agent with teams/alpha/* can read/write under that namespace."""
        from src.host.permissions import PermissionMatrix

        perms_file = tmp_path / "perms.json"
        perms_file.write_text(json.dumps({
            "permissions": {
                "bot1": {
                    "blackboard_read": ["teams/alpha/*"],
                    "blackboard_write": ["teams/alpha/*"],
                },
            }
        }))
        pm = PermissionMatrix(config_path=str(perms_file))

        assert pm.can_read_blackboard("bot1", "teams/alpha/context/market")
        assert pm.can_read_blackboard("bot1", "teams/alpha/goals/v1")
        assert pm.can_write_blackboard("bot1", "teams/alpha/tasks/todo")

    def test_project_agent_cannot_access_other_project(self, tmp_path):
        """Agent with teams/alpha/* CANNOT access teams/beta/*."""
        from src.host.permissions import PermissionMatrix

        perms_file = tmp_path / "perms.json"
        perms_file.write_text(json.dumps({
            "permissions": {
                "bot1": {
                    "blackboard_read": ["teams/alpha/*"],
                    "blackboard_write": ["teams/alpha/*"],
                },
            }
        }))
        pm = PermissionMatrix(config_path=str(perms_file))

        assert not pm.can_read_blackboard("bot1", "teams/beta/context/market")
        assert not pm.can_write_blackboard("bot1", "teams/beta/goals/v1")
        assert not pm.can_read_blackboard("bot1", "teams/beta/anything")

    def test_project_agent_cannot_access_global_keys(self, tmp_path):
        """Agent with teams/alpha/* CANNOT access unprefixed global keys."""
        from src.host.permissions import PermissionMatrix

        perms_file = tmp_path / "perms.json"
        perms_file.write_text(json.dumps({
            "permissions": {
                "bot1": {
                    "blackboard_read": ["teams/alpha/*"],
                    "blackboard_write": ["teams/alpha/*"],
                },
            }
        }))
        pm = PermissionMatrix(config_path=str(perms_file))

        assert not pm.can_read_blackboard("bot1", "context/market")
        assert not pm.can_read_blackboard("bot1", "tasks/todo")
        assert not pm.can_write_blackboard("bot1", "goals/v1")

    def test_standalone_agent_has_no_blackboard_access(self, tmp_path):
        """Standalone agent (empty patterns) cannot access any blackboard key."""
        from src.host.permissions import PermissionMatrix

        perms_file = tmp_path / "perms.json"
        perms_file.write_text(json.dumps({
            "permissions": {
                "solo": {
                    "blackboard_read": [],
                    "blackboard_write": [],
                },
            }
        }))
        pm = PermissionMatrix(config_path=str(perms_file))

        assert not pm.can_read_blackboard("solo", "context/anything")
        assert not pm.can_read_blackboard("solo", "teams/alpha/data")
        assert not pm.can_write_blackboard("solo", "anything")

    def test_two_projects_fully_isolated(self, tmp_path):
        """Two agents in different projects have zero overlap in blackboard access."""
        from src.host.permissions import PermissionMatrix

        perms_file = tmp_path / "perms.json"
        perms_file.write_text(json.dumps({
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
        }))
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
