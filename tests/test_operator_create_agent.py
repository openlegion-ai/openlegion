"""Tests for POST /mesh/agents/create endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

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
    tools_dir = tmp_path / "agent_tools"
    tools_dir.mkdir()

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
    # Seed the on-disk permissions file with operator + other so that
    # permissions.reload() (called by the create route after writing the
    # new agent's perms) preserves them across the reload.
    import json as _json
    perms_file = tmp_path / "config" / "permissions.json"
    perms_file.parent.mkdir(parents=True, exist_ok=True)
    perms_file.write_text(_json.dumps({
        "permissions": {
            "operator": {
                "can_message": ["*"],
                "can_spawn": True,
                "can_manage_fleet": True,
                "blackboard_read": ["*"],
                "blackboard_write": ["*"],
            },
            "other": {
                "can_message": ["mesh"],
                "can_spawn": False,
            },
        },
    }))
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {
        "operator": AgentPermissions(
            agent_id="operator",
            can_message=["*"],
            can_spawn=True,
            can_manage_fleet=True,
            blackboard_read=["*"],
            blackboard_write=["*"],
        ),
        "other": AgentPermissions(
            agent_id="other",
            can_message=["mesh"],
            can_spawn=False,
        ),
    }
    # _config_path points at the seeded permissions file so reload()
    # in the create route picks up the on-disk write without losing the
    # operator/other entries the fixture set in memory.
    perms._config_path = str(perms_file)
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

    def test_tools_dir_created(self, mesh_app, tmp_path):
        """Tools directory should be created for the new agent."""
        client = mesh_app["client"]
        resp = client.post(
            "/mesh/agents/create",
            json={"agent_id": "operator", "name": "toolbot", "role": "test"},
        )
        assert resp.status_code == 200
        assert (tmp_path / "agent_tools" / "toolbot").is_dir()

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

    def test_permissions_reload_after_disk_write(self, mesh_app):
        """Regression: _add_agent_permissions writes to disk; the live
        PermissionMatrix has to reload, otherwise the agent's imminent
        /mesh/register call falls through to default/deny-all and the
        coordination defaults applied here are silently negated until
        process restart (cf. PR #656)."""
        perms = mesh_app["perms"]
        perms.reload = MagicMock()

        client = mesh_app["client"]
        resp = client.post(
            "/mesh/agents/create",
            json={"agent_id": "operator", "name": "reloadtest", "role": "test"},
        )
        assert resp.status_code == 200
        perms.reload.assert_called()

    def test_default_capabilities_seed_full_coordination_protocol(
        self, mesh_app, tmp_path,
    ):
        """Operator-created agents get broad coordination defaults written
        to ``permissions.json`` so their imminent ``/mesh/register`` call does
        not fall through to deny-all. Regression guard for ``3b90a0a``.
        Specifically:

          * ``blackboard_read``  ⊇ ``["*"]``
          * ``blackboard_write`` ⊇ the five coordination namespaces
          * ``can_publish``      ⊇ ``["*"]``
          * ``can_subscribe``    ⊇ ``["*"]``
          * ``can_use_browser``  is True
          * ``can_manage_cron``  is True
          * ``allowed_apis``     ⊇ ``{"llm", "image_gen"}``
        """
        import json as _json

        client = mesh_app["client"]
        resp = client.post(
            "/mesh/agents/create",
            json={
                "agent_id": "operator",
                "name": "freshie",
                "role": "fresh agent",
                "model": "openai/gpt-4o-mini",
                "instructions": "do things",
            },
        )
        assert resp.status_code == 200, resp.text

        # Inspect the on-disk permissions written by _add_agent_permissions.
        perms_path = tmp_path / "config" / "permissions.json"
        on_disk = _json.loads(perms_path.read_text())
        agent_perms = on_disk["permissions"]["freshie"]

        # Blackboard reads — the private team-of-one pattern ONLY (ratified
        # #5 review fix: no worker starts with a read wildcard; the host
        # ACL is the read boundary and joins add team patterns).
        assert agent_perms["blackboard_read"] == ["teams/freshie/*"]
        assert "*" not in agent_perms["blackboard_read"]

        # Blackboard writes — the signal coordination namespaces. Phase-2
        # unit 4 removed output/* + artifacts/* (payloads moved to the Team
        # Drive; the blackboard is signals-only).
        write = set(agent_perms["blackboard_write"])
        assert {
            "tasks/*", "context/*", "status/*",
        }.issubset(write), f"missing namespaces from {write!r}"
        assert "output/*" not in write
        assert "artifacts/*" not in write

        # Pubsub.
        assert "*" in agent_perms["can_publish"]
        assert "*" in agent_perms["can_subscribe"]

        # Browser + cron capability bits.
        assert agent_perms["can_use_browser"] is True
        assert agent_perms["can_manage_cron"] is True

        # API surface — at minimum LLM (so the agent can run) + image_gen.
        allowed_apis = set(agent_perms.get("allowed_apis", []))
        assert {"llm", "image_gen"}.issubset(allowed_apis), \
            f"expected llm + image_gen in {allowed_apis!r}"

        # And the LIVE PermissionMatrix must agree once it's been reloaded
        # — the in-memory object backs every mesh permission gate the
        # newly-registered agent will hit. Without reload the agent
        # registers and immediately gets deny-all.
        perms = mesh_app["perms"]
        perms.reload()
        live = perms.get_permissions("freshie")
        assert live.blackboard_read == ["teams/freshie/*"]
        assert "tasks/*" in live.blackboard_write
        # Full default capability set: browser + internet + schedules ON,
        # wallet OFF (parity with the human `_create_agent` path).
        assert live.can_use_browser is True
        assert live.can_use_internet is True
        assert live.can_manage_cron is True
        assert live.can_use_wallet is False

    def test_default_capabilities_include_can_spawn_but_not_wallet(
        self, mesh_app, tmp_path,
    ):
        """Pin: a newly-created agent gets ``can_spawn=True`` by default
        (ephemeral fleet-spawn is a baseline capability now) but NOT
        ``can_use_wallet`` (spending money requires explicit human setup).
        The recursion wall is preserved elsewhere: an ephemeral ``spawn-*``
        agent is never written to permissions.json, so it resolves to the
        field default (False) and cannot itself re-spawn.
        """
        import json as _json

        client = mesh_app["client"]
        resp = client.post(
            "/mesh/agents/create",
            json={
                "agent_id": "operator", "name": "noprivs", "role": "fresh",
                "model": "openai/gpt-4o-mini",
            },
        )
        assert resp.status_code == 200, resp.text

        perms_path = tmp_path / "config" / "permissions.json"
        on_disk = _json.loads(perms_path.read_text())
        agent_perms = on_disk["permissions"]["noprivs"]
        # Spawn ON by default; wallet OFF (the sole privileged-off default).
        assert agent_perms.get("can_spawn") is True
        assert agent_perms.get("can_use_wallet", False) is False

        # And the live matrix agrees.
        mesh_app["perms"].reload()
        assert mesh_app["perms"].can_spawn("noprivs") is True


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


class TestConcurrentCreateNoOvershoot:
    """M15 — concurrent create requests must not overshoot MAX_AGENTS.

    The shared creation lock serializes the count-check→config-write so
    two requests can't both observe ``current < max`` and both proceed.
    We inject an ``await`` into the locked section (via a patched
    ``_load_config`` that yields control) to force the event loop to
    interleave the requests — without the lock both would pass the cap
    check; with it, exactly MAX_AGENTS land.
    """

    @pytest.mark.asyncio
    async def test_concurrent_creates_respect_max_agents(
        self, mesh_app, monkeypatch,
    ):
        import asyncio as _asyncio

        import httpx

        monkeypatch.setenv("OPENLEGION_MAX_AGENTS", "3")

        import src.cli.config as _cfgmod
        _orig_load = _cfgmod._load_config

        app = mesh_app["client"].app
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test",
        ) as ac:
            async def _create(n: int):
                return await ac.post(
                    "/mesh/agents/create",
                    json={
                        "agent_id": "operator",
                        "name": f"worker{n}",
                        "role": "worker",
                    },
                )

            # Fire 8 concurrent creates against a cap of 3.
            results = await _asyncio.gather(*[_create(i) for i in range(8)])

        ok = [r for r in results if r.status_code == 200]
        rejected = [r for r in results if r.status_code == 409]
        assert len(ok) == 3, (
            f"expected exactly 3 successful creates, got {len(ok)} "
            f"(overshoot means the lock failed)"
        )
        assert len(rejected) == 5
        # On-disk config must hold exactly 3 worker agents.
        cfg = _orig_load()
        workers = [a for a in cfg.get("agents", {}) if a.startswith("worker")]
        assert len(workers) == 3
