"""Tests for fleet template endpoints, fleet_tool tools, and mesh client methods."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml


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
        mock_client.apply_fleet_template.assert_awaited_once_with(
            "sales", model="", agent_overrides=None,
        )

    @pytest.mark.asyncio
    async def test_provenance_fails_closed_when_no_messages(self):
        """When _messages is None, provenance fails closed (security: no bypass via cron/invoke)."""
        from src.agent.builtins.fleet_tool import apply_template

        mock_client = AsyncMock()
        mock_client.apply_fleet_template.return_value = {
            "template": "sales", "created": [], "failed": [], "skipped": [],
        }
        result = await apply_template(template="sales", mesh_client=mock_client, _messages=None)
        assert result["error"] == "provenance_check_failed"
        mock_client.apply_fleet_template.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_model_override(self):
        from src.agent.builtins.fleet_tool import apply_template

        mock_client = AsyncMock()
        mock_client.apply_fleet_template.return_value = {
            "template": "sales", "created": [], "failed": [], "skipped": [],
        }
        messages = [{"role": "user", "content": "go"}]
        await apply_template(template="sales", model="openai/gpt-4o", mesh_client=mock_client, _messages=messages)
        mock_client.apply_fleet_template.assert_awaited_once_with(
            "sales", model="openai/gpt-4o", agent_overrides=None,
        )

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

    @pytest.mark.asyncio
    async def test_apply_fleet_template_with_agent_overrides(self):
        """PR-N: ``agent_overrides`` is forwarded as part of the JSON body."""
        from src.agent.mesh_client import MeshClient

        client = MeshClient(mesh_url="http://localhost:8420", agent_id="test")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"template": "sales", "created": []}
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response

        overrides = {"writer": {"model": "openai/gpt-4o", "instructions": "Be terse."}}
        with patch.object(client, "_get_client", new_callable=AsyncMock, return_value=mock_http):
            await client.apply_fleet_template("sales", agent_overrides=overrides)
            call_kwargs = mock_http.post.call_args
            assert call_kwargs.kwargs["json"] == {
                "template": "sales",
                "agent_overrides": overrides,
            }

    @pytest.mark.asyncio
    async def test_apply_fleet_template_omits_empty_overrides(self):
        """Empty / None overrides are NOT included in the body (back-compat)."""
        from src.agent.mesh_client import MeshClient

        client = MeshClient(mesh_url="http://localhost:8420", agent_id="test")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"template": "starter", "created": []}
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response

        with patch.object(client, "_get_client", new_callable=AsyncMock, return_value=mock_http):
            # None
            await client.apply_fleet_template("starter", agent_overrides=None)
            assert mock_http.post.call_args.kwargs["json"] == {"template": "starter"}
            # Empty dict
            await client.apply_fleet_template("starter", agent_overrides={})
            assert mock_http.post.call_args.kwargs["json"] == {"template": "starter"}


class TestApplyTemplateAgentOverridesTool:
    """PR-N: ``apply_template`` tool forwards ``agent_overrides`` and back-compat."""

    @pytest.mark.asyncio
    async def test_overrides_forwarded(self):
        from src.agent.builtins.fleet_tool import apply_template

        mock_client = AsyncMock()
        mock_client.apply_fleet_template.return_value = {
            "template": "sales", "created": [], "failed": [], "skipped": [],
        }
        messages = [{"role": "user", "content": "go"}]
        overrides = {"writer": {"model": "openai/gpt-4o"}}
        await apply_template(
            template="sales",
            agent_overrides=overrides,
            mesh_client=mock_client,
            _messages=messages,
        )
        mock_client.apply_fleet_template.assert_awaited_once_with(
            "sales", model="", agent_overrides=overrides,
        )

    @pytest.mark.asyncio
    async def test_omitted_overrides_pass_none(self):
        """Back-compat: callers passing only template + model still work."""
        from src.agent.builtins.fleet_tool import apply_template

        mock_client = AsyncMock()
        mock_client.apply_fleet_template.return_value = {
            "template": "starter", "created": [], "failed": [], "skipped": [],
        }
        messages = [{"role": "user", "content": "go"}]
        await apply_template(
            template="starter",
            mesh_client=mock_client,
            _messages=messages,
        )
        mock_client.apply_fleet_template.assert_awaited_once_with(
            "starter", model="", agent_overrides=None,
        )


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
        # Point at a path that doesn't exist — _load() handles missing files
        # gracefully (warns + clears). The fleet apply route now reloads
        # permissions after writing to disk, so _config_path must be set.
        perms._config_path = "/tmp/__nonexistent_permissions_for_test.json"
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

    @patch.dict("os.environ", {"OPENLEGION_MAX_AGENTS": "10"})
    def test_apply_reloads_permissions_after_disk_write(self):
        """Regression: _apply_template calls _add_agent_permissions for
        every new agent (writing to config/permissions.json on disk). The
        live PermissionMatrix has to reload, otherwise every template-
        spawned agent's /mesh/register call falls through to default/
        deny-all and they're locked out of coordination until restart."""
        from fastapi.testclient import TestClient

        app, _, _ = self._make_app(existing_agents={"operator": "http://localhost:8400"})
        # Find the live PermissionMatrix bound to the app and stub reload
        # so we can assert it ran. (state_for the matrix instance so we can
        # find it again — _make_app stores it on perms.)
        # The mesh app holds permissions in a closure; we pick it up via
        # the perms passed into _make_app via attribute access on router.
        router = app.state if hasattr(app, "state") else None  # noqa: F841
        # Easier: rebuild with a stubbed perms.reload
        from src.host.mesh import Blackboard, MessageRouter, PubSub
        from src.host.permissions import PermissionMatrix
        from src.host.server import create_mesh_app

        bb = Blackboard(db_path=":memory:")
        pubsub = PubSub()
        perms = PermissionMatrix.__new__(PermissionMatrix)
        perms.permissions = {}
        perms._config_path = ""
        perms._reload_lock = __import__("threading").Lock()
        perms.reload = MagicMock()

        agent_registry: dict[str, str] = {"operator": "http://localhost:8400"}
        router = MessageRouter(permissions=perms, agent_registry=agent_registry)
        from pathlib import Path
        cm = MagicMock()
        cm.project_root = Path("/tmp/test_project")
        cm.extra_env = {}
        cm.start_agent.return_value = "http://localhost:8401"
        cm.wait_for_agent = AsyncMock(return_value=True)

        app2 = create_mesh_app(
            blackboard=bb, pubsub=pubsub, router=router,
            permissions=perms, container_manager=cm,
        )
        client = TestClient(app2)
        resp = client.post("/mesh/fleet/apply", json={"template": "starter"})
        assert resp.status_code == 200
        perms.reload.assert_called()

    # ── PR-N: agent_overrides validation ─────────────────────

    @patch.dict("os.environ", {"OPENLEGION_MAX_AGENTS": "10"})
    def test_apply_unknown_agent_in_overrides_returns_400(self):
        """Override key referencing an agent not in the template → 400, no creation."""
        from fastapi.testclient import TestClient

        app, cm, _ = self._make_app(existing_agents={"operator": "http://localhost:8400"})
        client = TestClient(app)
        resp = client.post(
            "/mesh/fleet/apply",
            json={
                "template": "starter",
                "agent_overrides": {"nonexistent_xyz": {"model": "openai/gpt-4o"}},
            },
        )
        assert resp.status_code == 400
        detail = resp.json()["detail"].lower()
        assert "unknown" in detail
        assert "nonexistent_xyz" in detail
        # Validation happens BEFORE template application
        cm.start_agent.assert_not_called()

    @patch.dict("os.environ", {"OPENLEGION_MAX_AGENTS": "10"})
    def test_apply_unknown_override_field_returns_400(self):
        """Override field outside {'model','instructions'} → 400."""
        from fastapi.testclient import TestClient

        app, cm, _ = self._make_app(existing_agents={"operator": "http://localhost:8400"})
        client = TestClient(app)
        resp = client.post(
            "/mesh/fleet/apply",
            json={
                "template": "starter",
                "agent_overrides": {"assistant": {"foo": "bar"}},
            },
        )
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert "foo" in detail
        cm.start_agent.assert_not_called()

    @patch.dict("os.environ", {"OPENLEGION_MAX_AGENTS": "10"})
    def test_apply_invalid_model_in_override_returns_400(self):
        """Unknown model id → 400 from upfront validation, no agents created."""
        from fastapi.testclient import TestClient

        app, cm, _ = self._make_app(existing_agents={"operator": "http://localhost:8400"})
        client = TestClient(app)
        resp = client.post(
            "/mesh/fleet/apply",
            json={
                "template": "starter",
                "agent_overrides": {
                    "assistant": {"model": "fakeprovider/totally-not-real-model-xyz"},
                },
            },
        )
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert "fakeprovider/totally-not-real-model-xyz" in detail
        cm.start_agent.assert_not_called()

    @patch.dict("os.environ", {"OPENLEGION_MAX_AGENTS": "10"})
    def test_apply_oversized_instructions_returns_413(self):
        """Instructions > 12000 chars → 413, no agents created."""
        from fastapi.testclient import TestClient

        app, cm, _ = self._make_app(existing_agents={"operator": "http://localhost:8400"})
        client = TestClient(app)
        oversized = "x" * 12001
        resp = client.post(
            "/mesh/fleet/apply",
            json={
                "template": "starter",
                "agent_overrides": {"assistant": {"instructions": oversized}},
            },
        )
        assert resp.status_code == 413
        detail = resp.json()["detail"]
        assert "assistant" in detail
        assert "12000" in detail
        cm.start_agent.assert_not_called()

    # ── PR-N v2: soul/heartbeat/interface validation ─────────

    @patch.dict("os.environ", {"OPENLEGION_MAX_AGENTS": "10"})
    def test_apply_oversized_soul_returns_413(self):
        """Soul > 4000 chars → 413, names offending agent, no creation."""
        from fastapi.testclient import TestClient

        app, cm, _ = self._make_app(existing_agents={"operator": "http://localhost:8400"})
        client = TestClient(app)
        oversized = "s" * 4001
        resp = client.post(
            "/mesh/fleet/apply",
            json={
                "template": "starter",
                "agent_overrides": {"assistant": {"soul": oversized}},
            },
        )
        assert resp.status_code == 413
        detail = resp.json()["detail"]
        assert "assistant" in detail
        assert "soul" in detail
        assert "4000" in detail
        cm.start_agent.assert_not_called()

    @patch.dict("os.environ", {"OPENLEGION_MAX_AGENTS": "10"})
    def test_apply_oversized_interface_returns_413(self):
        """Interface > 4000 chars → 413, names offending agent."""
        from fastapi.testclient import TestClient

        app, cm, _ = self._make_app(existing_agents={"operator": "http://localhost:8400"})
        client = TestClient(app)
        oversized = "i" * 4001
        resp = client.post(
            "/mesh/fleet/apply",
            json={
                "template": "starter",
                "agent_overrides": {"assistant": {"interface": oversized}},
            },
        )
        assert resp.status_code == 413
        detail = resp.json()["detail"]
        assert "assistant" in detail
        assert "interface" in detail
        assert "4000" in detail
        cm.start_agent.assert_not_called()

    @patch.dict("os.environ", {"OPENLEGION_MAX_AGENTS": "10"})
    def test_apply_role_override_field_accepted(self, tmp_path, monkeypatch):
        """`role` is a hiring-wizard-v2 unfreeze (§8 #16b): a per-slot
        override lands in agents.yaml AND the started container's env
        (AGENT_ROLE), letting a hire's job-description-derived role win
        over the template slot's default role."""
        from fastapi.testclient import TestClient

        # Isolate agents.yaml/permissions.json for this test — other tests
        # in this class share the real repo config (gitignored, tolerated
        # because they never assert on `created` contents), but this test
        # DOES assert on-disk content, so it needs a clean slate rather
        # than depending on whether a prior test run already created
        # (and thus template-skipped) the "assistant" slot.
        import src.cli.config as cli_config

        monkeypatch.setattr(cli_config, "AGENTS_FILE", tmp_path / "agents.yaml")
        monkeypatch.setattr(cli_config, "PERMISSIONS_FILE", tmp_path / "permissions.json")

        app, cm, _ = self._make_app(existing_agents={"operator": "http://localhost:8400"})
        client = TestClient(app)
        resp = client.post(
            "/mesh/fleet/apply",
            json={
                "template": "starter",
                "agent_overrides": {"assistant": {"role": "Senior support specialist"}},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        created_ids = [c["agent_id"] for c in data["created"]]
        assert "assistant" in created_ids
        assert next(c for c in data["created"] if c["agent_id"] == "assistant")["role"] == (
            "Senior support specialist"
        )

        cm.start_agent.assert_called_once()
        _, kwargs = cm.start_agent.call_args
        assert kwargs["role"] == "Senior support specialist"

        with open(cli_config.AGENTS_FILE) as f:
            agents_cfg = yaml.safe_load(f)
        assert agents_cfg["agents"]["assistant"]["role"] == "Senior support specialist"

    @patch.dict("os.environ", {"OPENLEGION_MAX_AGENTS": "10"})
    def test_apply_still_unknown_override_field_rejected(self):
        """A field that was never allowed (and still isn't) stays a 400 —
        `role` moving to the allowed set doesn't loosen the unknown-field
        gate for everything else."""
        from fastapi.testclient import TestClient

        app, cm, _ = self._make_app(existing_agents={"operator": "http://localhost:8400"})
        client = TestClient(app)
        resp = client.post(
            "/mesh/fleet/apply",
            json={
                "template": "starter",
                "agent_overrides": {"assistant": {"nickname": "renamed"}},
            },
        )
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert "nickname" in detail
        assert "role" in detail  # allowed list is echoed in the error
        cm.start_agent.assert_not_called()

    @patch.dict("os.environ", {"OPENLEGION_MAX_AGENTS": "10"})
    def test_apply_non_string_soul_returns_400(self):
        """soul must be a string."""
        from fastapi.testclient import TestClient

        app, cm, _ = self._make_app(existing_agents={"operator": "http://localhost:8400"})
        client = TestClient(app)
        resp = client.post(
            "/mesh/fleet/apply",
            json={
                "template": "starter",
                "agent_overrides": {"assistant": {"soul": 123}},
            },
        )
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert "soul" in detail
        assert "string" in detail
        cm.start_agent.assert_not_called()

    @patch.dict("os.environ", {"OPENLEGION_MAX_AGENTS": "10"})
    def test_apply_large_heartbeat_accepted(self):
        """heartbeat has no length cap — validation lets large strings through."""
        from fastapi.testclient import TestClient

        app, _, _ = self._make_app(existing_agents={"operator": "http://localhost:8400"})
        client = TestClient(app)
        # 8000 chars -- well above SOUL/INTERFACE 4000 cap. Reflects
        # HEARTBEAT.md _FILE_CAPS = None (uncapped).
        big_heartbeat = "h" * 8000
        # Patch _apply_template so we don't pollute real config/agents.yaml;
        # we only care that validation passes.
        with patch(
            "src.cli.config._apply_template", return_value=["assistant"],
        ) as mock_apply:
            resp = client.post(
                "/mesh/fleet/apply",
                json={
                    "template": "starter",
                    "agent_overrides": {"assistant": {"heartbeat": big_heartbeat}},
                },
            )
        assert resp.status_code == 200, resp.text
        # Confirm the override was forwarded intact
        _, kwargs = mock_apply.call_args
        assert kwargs["agent_overrides"]["assistant"]["heartbeat"] == big_heartbeat

    @patch.dict("os.environ", {"OPENLEGION_MAX_AGENTS": "10"})
    def test_apply_empty_overrides_is_noop(self):
        """Empty agent_overrides={} behaves identically to no overrides."""
        from fastapi.testclient import TestClient

        app, _, _ = self._make_app(existing_agents={"operator": "http://localhost:8400"})
        client = TestClient(app)
        resp = client.post(
            "/mesh/fleet/apply",
            json={"template": "starter", "agent_overrides": {}},
        )
        assert resp.status_code == 200
        assert resp.json()["template"] == "starter"

    @patch.dict("os.environ", {"OPENLEGION_MAX_AGENTS": "10"})
    def test_apply_non_dict_overrides_returns_400(self):
        """agent_overrides must be an object."""
        from fastapi.testclient import TestClient

        app, _, _ = self._make_app(existing_agents={"operator": "http://localhost:8400"})
        client = TestClient(app)
        resp = client.post(
            "/mesh/fleet/apply",
            json={"template": "starter", "agent_overrides": ["not", "a", "dict"]},
        )
        assert resp.status_code == 400

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
        perms._config_path = "/tmp/__nonexistent_permissions_for_test.json"
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
