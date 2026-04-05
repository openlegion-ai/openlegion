"""Tests for fleet template endpoints, fleet_tool skills, and mesh client methods."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _set_operator_env(monkeypatch):
    """Fleet tools require ALLOWED_TOOLS to be set (defence-in-depth guard)."""
    monkeypatch.setenv("ALLOWED_TOOLS", "list_templates,apply_template")


# ── _last_message_is_user_origin helper ────────────────────────


class TestLastMessageIsUserOrigin:
    def test_empty_messages(self):
        from src.agent.loop import _last_message_is_user_origin

        assert _last_message_is_user_origin([]) is False

    def test_no_user_messages(self):
        from src.agent.loop import _last_message_is_user_origin

        messages = [{"role": "assistant", "content": "hi"}]
        assert _last_message_is_user_origin(messages) is False

    def test_user_message_without_origin(self):
        from src.agent.loop import _last_message_is_user_origin

        messages = [{"role": "user", "content": "hello"}]
        assert _last_message_is_user_origin(messages) is True

    def test_user_message_with_user_origin(self):
        from src.agent.loop import _last_message_is_user_origin

        messages = [{"role": "user", "content": "hello", "_origin": "user"}]
        assert _last_message_is_user_origin(messages) is True

    def test_user_message_with_system_origin(self):
        from src.agent.loop import _last_message_is_user_origin

        messages = [{"role": "user", "content": "hello", "_origin": "system"}]
        assert _last_message_is_user_origin(messages) is False

    def test_user_message_with_heartbeat_origin(self):
        from src.agent.loop import _last_message_is_user_origin

        messages = [{"role": "user", "content": "hello", "_origin": "heartbeat"}]
        assert _last_message_is_user_origin(messages) is False

    def test_last_user_message_checked(self):
        from src.agent.loop import _last_message_is_user_origin

        messages = [
            {"role": "user", "content": "first", "_origin": "user"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second", "_origin": "system"},
        ]
        assert _last_message_is_user_origin(messages) is False


# ── fleet_tool: list_templates ─────────────────────────────────


class TestListTemplates:
    @pytest.mark.asyncio
    async def test_no_mesh_client(self):
        from src.agent.builtins.fleet_tool import list_templates

        result = await list_templates(mesh_client=None)
        assert "error" in result
        assert "mesh_client" in result["error"]

    @pytest.mark.asyncio
    async def test_success(self):
        from src.agent.builtins.fleet_tool import list_templates

        mock_client = AsyncMock()
        mock_client.list_fleet_templates.return_value = {
            "templates": [
                {"name": "sales", "description": "Sales team",
                 "agent_count": 3, "agents": ["sdr", "closer", "analyst"]},
            ]
        }
        result = await list_templates(mesh_client=mock_client)
        assert "templates" in result
        assert len(result["templates"]) == 1
        mock_client.list_fleet_templates.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_exception_handling(self):
        from src.agent.builtins.fleet_tool import list_templates

        mock_client = AsyncMock()
        mock_client.list_fleet_templates.side_effect = RuntimeError("network error")
        result = await list_templates(mesh_client=mock_client)
        assert "error" in result
        assert "network error" in result["error"]


# ── fleet_tool: apply_template ─────────────────────────────────


class TestApplyTemplate:
    @pytest.mark.asyncio
    async def test_no_mesh_client(self):
        from src.agent.builtins.fleet_tool import apply_template

        result = await apply_template(template="sales", mesh_client=None)
        assert "error" in result
        assert "mesh_client" in result["error"]

    @pytest.mark.asyncio
    async def test_provenance_blocks_non_user(self):
        from src.agent.builtins.fleet_tool import apply_template

        mock_client = AsyncMock()
        messages = [{"role": "user", "content": "do it", "_origin": "system"}]
        result = await apply_template(template="sales", mesh_client=mock_client, _messages=messages)
        assert result["error"] == "provenance_check_failed"
        mock_client.apply_fleet_template.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_provenance_allows_user(self):
        from src.agent.builtins.fleet_tool import apply_template

        mock_client = AsyncMock()
        mock_client.apply_fleet_template.return_value = {
            "template": "sales",
            "created": [{"agent_id": "sdr", "role": "SDR", "ready": True}],
            "failed": [],
            "skipped": [],
        }
        messages = [{"role": "user", "content": "apply sales template", "_origin": "user"}]
        result = await apply_template(template="sales", mesh_client=mock_client, _messages=messages)
        assert "created" in result
        mock_client.apply_fleet_template.assert_awaited_once_with("sales", model="")

    @pytest.mark.asyncio
    async def test_provenance_allows_when_no_messages(self):
        """When _messages is None, provenance check is skipped."""
        from src.agent.builtins.fleet_tool import apply_template

        mock_client = AsyncMock()
        mock_client.apply_fleet_template.return_value = {
            "template": "sales", "created": [], "failed": [], "skipped": [],
        }
        result = await apply_template(template="sales", mesh_client=mock_client, _messages=None)
        assert "error" not in result
        mock_client.apply_fleet_template.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_model_override(self):
        from src.agent.builtins.fleet_tool import apply_template

        mock_client = AsyncMock()
        mock_client.apply_fleet_template.return_value = {
            "template": "sales", "created": [], "failed": [], "skipped": [],
        }
        messages = [{"role": "user", "content": "go"}]
        await apply_template(template="sales", model="openai/gpt-4o", mesh_client=mock_client, _messages=messages)
        mock_client.apply_fleet_template.assert_awaited_once_with("sales", model="openai/gpt-4o")

    @pytest.mark.asyncio
    async def test_exception_handling(self):
        from src.agent.builtins.fleet_tool import apply_template

        mock_client = AsyncMock()
        mock_client.apply_fleet_template.side_effect = RuntimeError("server error")
        messages = [{"role": "user", "content": "go"}]
        result = await apply_template(template="sales", mesh_client=mock_client, _messages=messages)
        assert "error" in result
        assert "server error" in result["error"]


# ── MeshClient fleet methods ──────────────────────────────────


class TestMeshClientFleetMethods:
    @pytest.mark.asyncio
    async def test_list_fleet_templates(self):
        from src.agent.mesh_client import MeshClient

        client = MeshClient(mesh_url="http://localhost:8420", agent_id="test")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"templates": [{"name": "sales"}]}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_with_retry", new_callable=AsyncMock, return_value=mock_response):
            result = await client.list_fleet_templates()
            assert result == {"templates": [{"name": "sales"}]}

    @pytest.mark.asyncio
    async def test_apply_fleet_template(self):
        from src.agent.mesh_client import MeshClient

        client = MeshClient(mesh_url="http://localhost:8420", agent_id="test")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"template": "sales", "created": []}
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response

        with patch.object(client, "_get_client", new_callable=AsyncMock, return_value=mock_http):
            result = await client.apply_fleet_template("sales", model="openai/gpt-4o")
            assert result == {"template": "sales", "created": []}
            mock_http.post.assert_awaited_once()
            call_kwargs = mock_http.post.call_args
            assert call_kwargs.kwargs["json"] == {"template": "sales", "model": "openai/gpt-4o"}

    @pytest.mark.asyncio
    async def test_apply_fleet_template_no_model(self):
        from src.agent.mesh_client import MeshClient

        client = MeshClient(mesh_url="http://localhost:8420", agent_id="test")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"template": "starter", "created": []}
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response

        with patch.object(client, "_get_client", new_callable=AsyncMock, return_value=mock_http):
            await client.apply_fleet_template("starter")
            call_kwargs = mock_http.post.call_args
            # model not included when empty
            assert call_kwargs.kwargs["json"] == {"template": "starter"}


# ── Mesh endpoint: GET /mesh/fleet/templates ───────────────────


class TestMeshFleetTemplatesEndpoint:
    """Test the mesh server fleet template endpoints using FastAPI TestClient."""

    def _make_app(self):
        """Create a minimal mesh app with mocked dependencies."""
        from src.host.mesh import Blackboard, MessageRouter, PubSub
        from src.host.permissions import PermissionMatrix
        from src.host.server import create_mesh_app

        bb = Blackboard(db_path=":memory:")
        pubsub = PubSub()
        perms = PermissionMatrix.__new__(PermissionMatrix)
        perms.permissions = {}
        perms._config_path = ""
        perms._reload_lock = __import__("threading").Lock()
        agent_registry: dict[str, str] = {}
        router = MessageRouter(permissions=perms, agent_registry=agent_registry)

        app = create_mesh_app(
            blackboard=bb,
            pubsub=pubsub,
            router=router,
            permissions=perms,
        )
        return app

    def test_list_templates(self):
        from fastapi.testclient import TestClient

        app = self._make_app()
        client = TestClient(app)
        resp = client.get("/mesh/fleet/templates")
        assert resp.status_code == 200
        data = resp.json()
        assert "templates" in data
        assert isinstance(data["templates"], list)
        # Should include built-in templates
        names = [t["name"] for t in data["templates"]]
        assert "starter" in names

    def test_list_templates_structure(self):
        from fastapi.testclient import TestClient

        app = self._make_app()
        client = TestClient(app)
        resp = client.get("/mesh/fleet/templates")
        data = resp.json()
        for tpl in data["templates"]:
            assert "name" in tpl
            assert "description" in tpl
            assert "agent_count" in tpl
            assert "agents" in tpl
            assert isinstance(tpl["agents"], list)
            assert tpl["agent_count"] == len(tpl["agents"])


class TestMeshFleetApplyEndpoint:
    """Test POST /mesh/fleet/apply with mocked container manager."""

    def _make_app(self, *, max_agents: int = 0, existing_agents: dict | None = None):
        """Create a mesh app with container manager mock."""
        from pathlib import Path

        from src.host.mesh import Blackboard, MessageRouter, PubSub
        from src.host.permissions import PermissionMatrix
        from src.host.server import create_mesh_app

        bb = Blackboard(db_path=":memory:")
        pubsub = PubSub()
        perms = PermissionMatrix.__new__(PermissionMatrix)
        perms.permissions = {}
        perms._config_path = ""
        perms._reload_lock = __import__("threading").Lock()
        agent_registry: dict[str, str] = {}
        router = MessageRouter(permissions=perms, agent_registry=agent_registry)

        # Mock container manager
        cm = MagicMock()
        cm.project_root = Path("/tmp/test_project")
        cm.extra_env = {}
        cm.start_agent.return_value = "http://localhost:8401"
        cm.wait_for_agent = AsyncMock(return_value=True)

        if existing_agents:
            for aid, url in existing_agents.items():
                router.register_agent(aid, url)

        app = create_mesh_app(
            blackboard=bb,
            pubsub=pubsub,
            router=router,
            permissions=perms,
            container_manager=cm,
        )
        return app, cm, router

    def test_apply_missing_template(self):
        from fastapi.testclient import TestClient

        app, _, _ = self._make_app()
        client = TestClient(app)
        resp = client.post("/mesh/fleet/apply", json={"template": ""})
        assert resp.status_code == 400

    def test_apply_nonexistent_template(self):
        from fastapi.testclient import TestClient

        app, _, _ = self._make_app()
        client = TestClient(app)
        resp = client.post("/mesh/fleet/apply", json={"template": "nonexistent_template_xyz"})
        assert resp.status_code == 404

    @patch.dict("os.environ", {"OPENLEGION_MAX_AGENTS": "1"})
    def test_apply_exceeds_plan_limit(self):
        from fastapi.testclient import TestClient

        # Already have one non-operator agent
        app, _, _ = self._make_app(existing_agents={"existing": "http://localhost:8401"})
        client = TestClient(app)
        # Starter template has 1 agent, and we already have 1 (limit is 1)
        resp = client.post("/mesh/fleet/apply", json={"template": "starter"})
        assert resp.status_code == 403
        assert "limit" in resp.json()["detail"].lower()

    @patch.dict("os.environ", {"OPENLEGION_MAX_AGENTS": "10"})
    def test_apply_excludes_operator_from_count(self):
        from fastapi.testclient import TestClient

        # Operator exists but should not count against limit
        app, _, router = self._make_app(existing_agents={"operator": "http://localhost:8400"})
        client = TestClient(app)
        resp = client.post("/mesh/fleet/apply", json={"template": "starter"})
        # Should succeed — operator doesn't count, so 0 + 1 <= 10
        assert resp.status_code == 200
        data = resp.json()
        assert data["template"] == "starter"

    def test_apply_no_container_manager(self):
        """Without container manager, apply returns 503."""
        from fastapi.testclient import TestClient

        from src.host.mesh import Blackboard, MessageRouter, PubSub
        from src.host.permissions import PermissionMatrix
        from src.host.server import create_mesh_app

        bb = Blackboard(db_path=":memory:")
        pubsub = PubSub()
        perms = PermissionMatrix.__new__(PermissionMatrix)
        perms.permissions = {}
        perms._config_path = ""
        perms._reload_lock = __import__("threading").Lock()
        agent_registry: dict[str, str] = {}
        router = MessageRouter(permissions=perms, agent_registry=agent_registry)

        app = create_mesh_app(
            blackboard=bb, pubsub=pubsub, router=router, permissions=perms,
            container_manager=None,
        )
        client = TestClient(app)
        resp = client.post("/mesh/fleet/apply", json={"template": "starter"})
        assert resp.status_code == 503


# ── Dashboard proxy: GET /dashboard/api/fleet/templates ────────


class TestDashboardFleetTemplatesProxy:
    def _make_dashboard(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from src.dashboard.server import create_dashboard_router
        from src.host.costs import CostTracker
        from src.host.mesh import Blackboard

        bb = Blackboard(db_path=":memory:")
        cost_tracker = CostTracker(db_path=":memory:")

        router = create_dashboard_router(
            blackboard=bb,
            health_monitor=None,
            cost_tracker=cost_tracker,
            trace_store=None,
            event_bus=None,
            agent_registry={},
        )
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_fleet_templates_endpoint(self):
        client = self._make_dashboard()
        resp = client.get("/dashboard/api/fleet/templates", headers={"X-Requested-With": "XMLHttpRequest"})
        assert resp.status_code == 200
        data = resp.json()
        assert "templates" in data
        names = [t["name"] for t in data["templates"]]
        assert "starter" in names
