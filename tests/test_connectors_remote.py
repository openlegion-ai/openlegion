"""Phase 2b integration tests: mesh endpoints, agent-side registration
and dispatch, dashboard probe + cache-invalidation hooks.

The mesh tests run a real ``MCPGateway`` over a real ``ConnectorStore``
with the fake streamable-HTTP factory from ``test_mcp_gateway`` — no
``mcp`` SDK required. The D11 pin lives here: the operator trust-tier
carve-out does NOT extend to ``/mesh/connectors/call`` (assignment is
the gate, operator included), and because assignment isn't a
``permissions.can_*`` gate, the trust-tier grep trip-wire can't cover
it — this HTTP-level test is the only pin.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.host.connectors import ConnectorStore
from src.host.mcp_gateway import MCPGateway
from src.shared.types import ConnectorAuth, HttpConnector
from tests.test_mcp_gateway import FakeOpenClient, FakeSession, _factory


@pytest.fixture(autouse=True)
def _reset_fakes():
    FakeSession.instances = []
    FakeSession.current_behavior = {}
    FakeOpenClient.opened = []
    FakeOpenClient.closed = 0
    yield


def _store_with_linear(tmp_path, agents=("alpha",)) -> ConnectorStore:
    store = ConnectorStore(str(tmp_path / "connectors.json"))
    store.upsert(HttpConnector(
        transport="http", name="linear",
        url="https://93.184.216.34/mcp",
        auth=ConnectorAuth(), agents=list(agents),
    ))
    return store


# ── mesh endpoints ───────────────────────────────────────────


@pytest.fixture
def mesh_env(tmp_path):
    """Mesh app with a real store + gateway (fake transport). Dev-mode
    auth (no auth_tokens) → X-Agent-ID is trusted, which lets tests
    impersonate the operator and workers directly."""
    from src.host.mesh import Blackboard, MessageRouter, PubSub
    from src.host.permissions import PermissionMatrix
    from src.host.server import create_mesh_app

    store = _store_with_linear(tmp_path)
    gateway = MCPGateway(store, None, client_factory=_factory)
    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {}
    app = create_mesh_app(
        bb, PubSub(), MessageRouter(permissions=perms, agent_registry={}),
        perms, connector_store=store, mcp_gateway=gateway,
    )
    client = TestClient(app)
    yield client, store, gateway, bb
    client.close()
    bb.close()


class TestMeshConnectorEndpoints:
    def _call(self, client, agent, connector="linear", tool="t"):
        # agent_id rides the body like every agent-authenticated POST
        # (vault_resolve pattern); with auth_tokens enforced the mesh
        # ignores it in favor of the verified bearer identity.
        return client.post(
            "/mesh/connectors/call",
            json={
                "agent_id": agent, "connector": connector,
                "tool": tool, "arguments": {},
            },
            headers={"X-Agent-ID": agent},
        )

    def test_assigned_agent_round_trips(self, mesh_env):
        client, *_ = mesh_env
        resp = self._call(client, "alpha")
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"result": "done"}

    def test_unassigned_agent_403(self, mesh_env):
        client, *_ = mesh_env
        assert self._call(client, "beta").status_code == 403

    def test_operator_not_in_bypass_set(self, mesh_env):
        # D11 pin: connectors front third-party credentials — the
        # operator is gated by assignment exactly like a worker. If a
        # future refactor adds connector calls to the operator
        # trust-tier carve-out, this fails.
        client, store, *_ = mesh_env
        assert self._call(client, "operator").status_code == 403
        # ...and assignment (not identity) is what grants access:
        widened = store.get("linear").model_copy(update={"agents": ["*"]})
        store.upsert(widened)
        assert self._call(client, "operator").status_code == 200

    def test_unknown_connector_404(self, mesh_env):
        client, *_ = mesh_env
        assert self._call(client, "alpha", connector="ghost").status_code == 404

    def test_bad_bodies_400(self, mesh_env):
        client, *_ = mesh_env
        resp = client.post(
            "/mesh/connectors/call",
            json={"connector": "linear", "arguments": {}},
            headers={"X-Agent-ID": "alpha"},
        )
        assert resp.status_code == 400
        resp = client.post(
            "/mesh/connectors/call",
            json={"connector": "linear", "tool": "t", "arguments": "nope"},
            headers={"X-Agent-ID": "alpha"},
        )
        assert resp.status_code == 400

    def test_tools_endpoint_caller_scoped(self, mesh_env):
        client, *_ = mesh_env
        data = client.get(
            "/mesh/connectors/tools",
            params={"agent_id": "alpha"},
            headers={"X-Agent-ID": "alpha"},
        ).json()
        assert list(data["connectors"]) == ["linear"]
        assert data["connectors"]["linear"]["tools"][0]["name"] == "create_issue"
        data = client.get(
            "/mesh/connectors/tools",
            params={"agent_id": "beta"},
            headers={"X-Agent-ID": "beta"},
        ).json()
        assert data["connectors"] == {}

    def test_upstream_error_masked_and_502(self, mesh_env):
        client, *_ = mesh_env
        FakeSession.current_behavior = {
            "call_tool": RuntimeError("500 internal: secret stack XYZ"),
        }
        resp = self._call(client, "alpha")
        assert resp.status_code == 502
        assert "XYZ" not in resp.text

    def test_audit_row_with_truncated_args(self, mesh_env):
        client, _, _, bb = mesh_env
        bb.log_audit = MagicMock()
        self._call(client, "alpha")
        kwargs = bb.log_audit.call_args.kwargs
        assert kwargs["action"] == "connector_call"
        assert kwargs["target"] == "linear:t"
        assert kwargs["actor"] == "alpha"
        assert len(kwargs["after_value"]) <= 500

    def test_rate_limit_category_registered(self):
        # _RATE_LIMITS is closure-scoped inside create_mesh_app; pin
        # the bucket's existence via source (same spirit as the
        # operator trust-tier grep trip-wire).
        import inspect

        import src.host.server as mesh_server
        assert '"connectors": (6000, 60)' in inspect.getsource(mesh_server)

    def test_gateway_unavailable_503(self, mesh_env):
        client, _, gateway, _ = mesh_env
        from src.host.mcp_gateway import GatewayUnavailable

        def _broken():
            raise GatewayUnavailable("mcp SDK not installed — re-run ./install.sh")

        gateway._client_factory = _broken
        resp = self._call(client, "alpha")
        assert resp.status_code == 503
        assert "install" in resp.json()["detail"]


# ── agent-side registry ──────────────────────────────────────


def _payload(connector="linear", name="create_issue"):
    return {connector: {"tools": [{
        "name": name,
        "description": "Create an issue",
        "parameters": {"type": "object", "properties": {"title": {"type": "string"}}},
    }]}}


@pytest.fixture
def registry(tmp_path):
    from src.agent.tools import ToolRegistry
    return ToolRegistry(tools_dir=str(tmp_path / "tools"))


class TestAgentRemoteRegistration:
    def test_registers_with_remote_marker(self, registry):
        registry.register_remote_tools(_payload())
        info = registry.tools["create_issue"]
        assert info["function"] == "mcp_remote"
        assert info["_connector"] == "linear"
        assert info["_remote_original_name"] == "create_issue"
        assert registry.get_tool_sources()["create_issue"] == "mcp"

    def test_conflict_with_builtin_prefixed(self, registry):
        builtin_name = next(iter(registry.tools))
        registry.register_remote_tools(_payload(name=builtin_name))
        prefixed = f"mcp_linear_{builtin_name}"
        assert prefixed in registry.tools
        assert registry.tools[builtin_name]["function"] != "mcp_remote"

    def test_conflict_with_stdio_prefixed(self, registry):
        # The stdio has_tool short-circuit runs FIRST in execute(), so
        # an unprefixed same-named remote tool would be unreachable.
        stdio = MagicMock()
        stdio.has_tool.side_effect = lambda n: n == "create_issue"
        stdio.list_tools.return_value = []
        registry._mcp_client = stdio
        registry.register_remote_tools(_payload())
        assert "mcp_linear_create_issue" in registry.tools
        assert "create_issue" not in registry.tools

    def test_reload_preserves_remote_tools(self, registry):
        registry.register_remote_tools(_payload())
        registry.reload()
        assert registry.tools["create_issue"]["function"] == "mcp_remote"

    def test_statuses_for_capabilities(self, registry):
        payload = _payload()
        payload["broken"] = {"tools": [], "error": "upstream unreachable"}
        registry.register_remote_tools(payload)
        statuses = {s["name"]: s for s in registry.remote_connector_statuses()}
        assert statuses["linear"]["state"] == "running"
        assert statuses["linear"]["tools_count"] == 1
        assert statuses["broken"]["state"] == "error"

    def test_tool_definitions_use_full_schema(self, registry):
        registry.register_remote_tools(_payload())
        defs = registry.get_tool_definitions()
        entry = next(
            d for d in defs if d["function"]["name"] == "create_issue"
        )
        assert entry["function"]["parameters"]["properties"]["title"] == {
            "type": "string",
        }


class TestAgentRemoteDispatch:
    @pytest.mark.asyncio
    async def test_routes_through_mesh_client(self, registry):
        stdio = MagicMock()
        stdio.has_tool.side_effect = lambda n: n == "create_issue"
        stdio.list_tools.return_value = []
        registry._mcp_client = stdio
        registry.register_remote_tools(_payload())  # → prefixed name
        mesh = MagicMock()
        mesh.call_connector_tool = AsyncMock(return_value={"result": "ok"})
        out = await registry.execute(
            "mcp_linear_create_issue", {"title": "x"}, mesh_client=mesh,
        )
        assert out == {"result": "ok"}
        # The ORIGINAL tool name goes over the wire, not the prefixed.
        mesh.call_connector_tool.assert_awaited_once_with(
            "linear", "create_issue", {"title": "x"},
        )

    @pytest.mark.asyncio
    async def test_no_mesh_client_is_error_dict(self, registry):
        registry.register_remote_tools(_payload())
        out = await registry.execute("create_issue", {})
        assert "error" in out

    @pytest.mark.asyncio
    async def test_mesh_failure_is_error_dict_not_raise(self, registry):
        registry.register_remote_tools(_payload())
        mesh = MagicMock()
        mesh.call_connector_tool = AsyncMock(side_effect=RuntimeError("502"))
        out = await registry.execute("create_issue", {}, mesh_client=mesh)
        assert "error" in out


# ── dashboard probe + invalidation hooks ─────────────────────


class _CSRFClient(TestClient):
    def request(self, method, url, **kwargs):
        if method.upper() not in ("GET", "HEAD", "OPTIONS"):
            headers = kwargs.get("headers") or {}
            headers.setdefault("X-Requested-With", "XMLHttpRequest")
            kwargs["headers"] = headers
        return super().request(method, url, **kwargs)


@pytest.fixture
def dash_env(tmp_path):
    from src.dashboard.events import EventBus
    from src.dashboard.server import create_dashboard_router
    from src.host.costs import CostTracker
    from src.host.mesh import Blackboard
    from src.host.traces import TraceStore

    store = _store_with_linear(tmp_path, agents=("alpha",))
    gateway = MagicMock()
    gateway.probe = AsyncMock(return_value={"ok": True, "tools_count": 3})
    vault = MagicMock()
    vault.resolve_credential.return_value = "secret-value"
    vault.list_agent_credential_names.return_value = ["linear_token"]
    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    router = create_dashboard_router(
        blackboard=bb,
        health_monitor=None,
        cost_tracker=CostTracker(db_path=str(tmp_path / "costs.db")),
        trace_store=TraceStore(db_path=str(tmp_path / "traces.db")),
        event_bus=EventBus(),
        agent_registry={"alpha": "http://x:1"},
        permissions=MagicMock(),
        credential_vault=vault,
        connector_store=store,
        mcp_gateway=gateway,
    )
    app = FastAPI()
    app.include_router(router)
    client = _CSRFClient(app)
    yield client, store, gateway
    client.close()
    bb.close()


class TestDashboardProbeAndInvalidate:
    def test_probe_passthrough(self, dash_env):
        client, _, gateway = dash_env
        resp = client.post("/dashboard/api/connectors/linear/probe")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "tools_count": 3}
        gateway.probe.assert_awaited_once_with("linear")

    def test_auth_mode_change_invalidates_and_prompts_restart(self, dash_env):
        # none -> bearer is a MODE change: cache invalidated (auth-only
        # at the store layer) AND restart prompted (agents registered
        # zero tools for the 401ing connector at their last boot).
        client, _, gateway = dash_env
        resp = client.put("/dashboard/api/connectors/linear", json={
            "transport": "http",
            "url": "https://93.184.216.34/mcp",
            "auth": {"kind": "bearer", "cred": "linear_token"},
            "agents": ["alpha"],
        })
        assert resp.status_code == 200, resp.text
        assert resp.json()["restart_required"] is True
        gateway.invalidate.assert_called_once_with("linear")

    def test_auth_rotation_invalidates_without_restart(self, dash_env):
        client, _, gateway = dash_env
        for cred in ("linear_token", "linear_token_v2"):
            resp = client.put("/dashboard/api/connectors/linear", json={
                "transport": "http",
                "url": "https://93.184.216.34/mcp",
                "auth": {"kind": "bearer", "cred": cred},
                "agents": ["alpha"],
            })
            assert resp.status_code == 200, resp.text
        # Second PUT was same-kind rotation: no restart, cache dropped.
        assert resp.json()["restart_required"] is False
        assert gateway.invalidate.call_count == 2

    def test_restart_relevant_edit_does_not_invalidate(self, dash_env):
        # Generation-keyed cache handles those; invalidate is the
        # auth-only side channel.
        client, _, gateway = dash_env
        resp = client.put("/dashboard/api/connectors/linear", json={
            "transport": "http",
            "url": "https://93.184.216.34:8443/mcp",
            "agents": ["alpha"],
        })
        assert resp.status_code == 200, resp.text
        assert resp.json()["restart_required"] is True
        gateway.invalidate.assert_not_called()

    def test_partial_put_inherits_transport(self, dash_env):
        # Review follow-up: a partial PUT against an http record must
        # not be re-tagged stdio by the union default (confusing 400 /
        # silent morph). Absent transport inherits the existing one.
        client, store, _ = dash_env
        resp = client.put("/dashboard/api/connectors/linear", json={
            "url": "https://93.184.216.34:9443/mcp",
        })
        assert resp.status_code == 200, resp.text
        replaced = store.get("linear")
        assert isinstance(replaced, HttpConnector)
        assert replaced.url.endswith(":9443/mcp")
        assert replaced.agents == ["alpha"]  # preserved

    def test_delete_invalidates(self, dash_env):
        client, _, gateway = dash_env
        resp = client.request(
            "DELETE", "/dashboard/api/connectors/linear",
        )
        assert resp.status_code == 200
        gateway.invalidate.assert_called_once_with("linear")

    def test_url_userinfo_rejected(self, dash_env):
        client, *_ = dash_env
        resp = client.put("/dashboard/api/connectors/leaky", json={
            "transport": "http",
            "url": "https://user:tok3n@example.com/mcp",
        })
        assert resp.status_code == 400
        errors = resp.json()["detail"]["errors"]
        # The safe-error mapping strips ctx/input — the secret embedded
        # in the rejected URL must not echo through the 400.
        assert "tok3n" not in json.dumps(errors)
        assert any(e["loc"][0] == "url" for e in errors)
