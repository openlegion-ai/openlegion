"""Tests for POST /mesh/agents/create endpoint."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.host.mesh import Blackboard, MessageRouter, PubSub
from src.host.permissions import PermissionMatrix
from src.host.server import create_mesh_app
from src.shared.types import AgentPermissions


@pytest.fixture()
def _mesh_env(tmp_path, monkeypatch):
    """Patch config module paths so agent creation writes to tmp_path."""
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()

    monkeypatch.setattr("src.cli.config.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("src.cli.config.AGENTS_FILE", cfg_dir / "agents.yaml")
    monkeypatch.setattr("src.cli.config.PERMISSIONS_FILE", cfg_dir / "permissions.json")
    monkeypatch.setattr("src.cli.config.CONFIG_FILE", cfg_dir / "mesh.yaml")


@pytest.fixture()
def container_mgr():
    mgr = MagicMock()
    mgr.start_agent.return_value = "http://localhost:9001"
    mgr.wait_for_agent = AsyncMock(return_value=True)
    mgr.agents = {}
    return mgr


@pytest.fixture()
def mesh_app(tmp_path, _mesh_env, container_mgr):
    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {
        "operator": AgentPermissions(
            agent_id="operator",
            can_message=["*"],
            can_spawn=True,
            blackboard_read=["*"],
            blackboard_write=["*"],
        ),
        "other": AgentPermissions(
            agent_id="other",
            can_message=["mesh"],
            can_spawn=False,
        ),
    }
    router = MessageRouter(permissions=perms, agent_registry={})
    app = create_mesh_app(
        bb, pubsub, router, perms,
        container_manager=container_mgr,
    )
    client = TestClient(app)
    return {
        "client": client,
        "bb": bb,
        "router": router,
        "perms": perms,
        "container_mgr": container_mgr,
    }


class TestCreateCustomAgent:
    """Tests for POST /mesh/agents/create."""

    def test_success(self, mesh_app):
        client = mesh_app["client"]
        resp = client.post(
            "/mesh/agents/create",
            json={
                "agent_id": "operator",
                "name": "researcher",
                "role": "research assistant",
                "model": "openai/gpt-4o-mini",
                "instructions": "You research topics.",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["agent_id"] == "researcher"
        assert body["role"] == "research assistant"
        assert body["ready"] is True

        # Container was started with correct args
        mgr = mesh_app["container_mgr"]
        mgr.start_agent.assert_called_once()
        call_kwargs = mgr.start_agent.call_args
        assert call_kwargs.kwargs.get("agent_id") or call_kwargs[1].get("agent_id") or \
            (call_kwargs.args[0] if call_kwargs.args else None) == "researcher"

    def test_name_required(self, mesh_app):
        client = mesh_app["client"]
        resp = client.post(
            "/mesh/agents/create",
            json={"agent_id": "operator", "role": "test"},
        )
        assert resp.status_code == 400
        assert "name is required" in resp.json()["detail"]

    def test_invalid_name(self, mesh_app):
        client = mesh_app["client"]
        resp = client.post(
            "/mesh/agents/create",
            json={"agent_id": "operator", "name": "../bad-name", "role": "test"},
        )
        assert resp.status_code == 400
        assert "Invalid agent name" in resp.json()["detail"]

    def test_duplicate_agent(self, mesh_app):
        client = mesh_app["client"]
        # Create first
        resp = client.post(
            "/mesh/agents/create",
            json={"agent_id": "operator", "name": "myagent", "role": "test"},
        )
        assert resp.status_code == 200

        # Try to create duplicate
        resp = client.post(
            "/mesh/agents/create",
            json={"agent_id": "operator", "name": "myagent", "role": "test2"},
        )
        assert resp.status_code == 409
        assert "already exists" in resp.json()["detail"]

    def test_spawn_permission_denied(self, mesh_app):
        client = mesh_app["client"]
        resp = client.post(
            "/mesh/agents/create",
            json={"agent_id": "other", "name": "sneaky", "role": "test"},
        )
        assert resp.status_code == 403
        assert "not allowed to create agents" in resp.json()["detail"]

    def test_plan_limit_enforced(self, mesh_app, monkeypatch):
        monkeypatch.setenv("OPENLEGION_MAX_AGENTS", "1")
        client = mesh_app["client"]

        # Create first agent (reaches limit)
        resp = client.post(
            "/mesh/agents/create",
            json={"agent_id": "operator", "name": "agent1", "role": "first"},
        )
        assert resp.status_code == 200

        # Second should be blocked
        resp = client.post(
            "/mesh/agents/create",
            json={"agent_id": "operator", "name": "agent2", "role": "second"},
        )
        assert resp.status_code == 409
        assert "Plan limit reached" in resp.json()["detail"]

    def test_plan_limit_excludes_operator(self, mesh_app, tmp_path, monkeypatch):
        """Operator itself should not count toward the plan agent limit."""
        import yaml

        # Pre-populate agents.yaml with operator entry
        agents_file = tmp_path / "config" / "agents.yaml"
        agents_file.parent.mkdir(parents=True, exist_ok=True)
        agents_file.write_text(yaml.dump({
            "agents": {"operator": {"role": "operator", "model": "openai/gpt-4o-mini"}},
        }))

        monkeypatch.setenv("OPENLEGION_MAX_AGENTS", "1")
        client = mesh_app["client"]

        # Should succeed because operator doesn't count
        resp = client.post(
            "/mesh/agents/create",
            json={"agent_id": "operator", "name": "worker1", "role": "worker"},
        )
        assert resp.status_code == 200

    def test_default_model_from_config(self, mesh_app, tmp_path):
        """When model is omitted, default should come from config."""
        import yaml

        # Write mesh config with a specific default model
        cfg_file = tmp_path / "config" / "mesh.yaml"
        cfg_file.parent.mkdir(parents=True, exist_ok=True)
        cfg_file.write_text(yaml.dump({"llm": {"default_model": "anthropic/claude-haiku"}}))

        client = mesh_app["client"]
        resp = client.post(
            "/mesh/agents/create",
            json={"agent_id": "operator", "name": "nomodel", "role": "test"},
        )
        assert resp.status_code == 200

        mgr = mesh_app["container_mgr"]
        call_kw = mgr.start_agent.call_args
        # model should be the config default
        assert "anthropic/claude-haiku" in str(call_kw)

    def test_container_manager_unavailable(self, tmp_path, _mesh_env):
        """503 when container manager is None."""
        bb = Blackboard(db_path=str(tmp_path / "bb2.db"))
        pubsub = PubSub()
        perms = PermissionMatrix.__new__(PermissionMatrix)
        perms.permissions = {
            "operator": AgentPermissions(
                agent_id="operator",
                can_message=["*"],
                can_spawn=True,
            ),
        }
        router = MessageRouter(permissions=perms, agent_registry={})
        app = create_mesh_app(bb, pubsub, router, perms, container_manager=None)
        client = TestClient(app)
        resp = client.post(
            "/mesh/agents/create",
            json={"agent_id": "operator", "name": "test", "role": "test"},
        )
        assert resp.status_code == 503
        bb.close()

    def test_container_start_failure(self, mesh_app):
        """500 when container_manager.start_agent raises."""
        mgr = mesh_app["container_mgr"]
        mgr.start_agent.side_effect = RuntimeError("Docker not available")

        client = mesh_app["client"]
        resp = client.post(
            "/mesh/agents/create",
            json={"agent_id": "operator", "name": "failbot", "role": "test"},
        )
        assert resp.status_code == 500
        assert "Failed to start agent" in resp.json()["detail"]

    def test_env_overrides_passed(self, mesh_app):
        """Instructions and soul should be passed as env_overrides, not shared env."""
        client = mesh_app["client"]
        resp = client.post(
            "/mesh/agents/create",
            json={
                "agent_id": "operator",
                "name": "envtest",
                "role": "test",
                "instructions": "Do great things",
                "soul": "Be kind",
            },
        )
        assert resp.status_code == 200

        mgr = mesh_app["container_mgr"]
        call_kw = mgr.start_agent.call_args
        env_overrides = call_kw.kwargs.get("env_overrides") or call_kw[1].get("env_overrides", {})
        assert env_overrides.get("INITIAL_INSTRUCTIONS") == "Do great things"
        assert env_overrides.get("INITIAL_SOUL") == "Be kind"

    def test_skills_dir_created(self, mesh_app, tmp_path):
        """Skills directory should be created for the new agent."""
        client = mesh_app["client"]
        resp = client.post(
            "/mesh/agents/create",
            json={"agent_id": "operator", "name": "skillbot", "role": "test"},
        )
        assert resp.status_code == 200
        assert (tmp_path / "skills" / "skillbot").is_dir()

    def test_agent_registered_in_router(self, mesh_app):
        """New agent should appear in the router's registry."""
        client = mesh_app["client"]
        resp = client.post(
            "/mesh/agents/create",
            json={"agent_id": "operator", "name": "routed", "role": "test"},
        )
        assert resp.status_code == 200
        router = mesh_app["router"]
        assert "routed" in router.agent_registry


class TestCreateCustomAgentMeshClient:
    """Tests for MeshClient.create_custom_agent."""

    @pytest.mark.asyncio
    async def test_create_custom_agent_builds_correct_request(self):
        from src.agent.mesh_client import MeshClient

        mc = MeshClient("http://localhost:8420", "operator")
        captured: dict = {}

        async def mock_post(url, json=None, timeout=None, headers=None):
            captured["url"] = url
            captured["json"] = json
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            resp.json.return_value = {"agent_id": "test", "role": "test", "ready": True}
            return resp

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = mock_post
        mc._client = mock_client

        result = await mc.create_custom_agent(
            name="test", role="tester", model="openai/gpt-4o",
            instructions="be helpful",
        )
        assert captured["url"] == "http://localhost:8420/mesh/agents/create"
        assert captured["json"]["name"] == "test"
        assert captured["json"]["role"] == "tester"
        assert captured["json"]["model"] == "openai/gpt-4o"
        assert captured["json"]["instructions"] == "be helpful"
        assert "soul" not in captured["json"]  # empty soul should be omitted
        assert result["agent_id"] == "test"
        await mc.close()

    @pytest.mark.asyncio
    async def test_create_custom_agent_minimal(self):
        from src.agent.mesh_client import MeshClient

        mc = MeshClient("http://localhost:8420", "operator")
        captured: dict = {}

        async def mock_post(url, json=None, timeout=None, headers=None):
            captured["json"] = json
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            resp.json.return_value = {"agent_id": "min", "role": "min", "ready": True}
            return resp

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = mock_post
        mc._client = mock_client

        result = await mc.create_custom_agent(name="min", role="min")
        assert captured["json"] == {"name": "min", "role": "min"}
        assert result["ready"] is True
        await mc.close()
