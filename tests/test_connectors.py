"""Tests for the fleet MCP connector catalog.

Covers the three layers:

* ``ConnectorStore`` — persistence, atomicity, fail-closed loading,
  assignment resolution, pending-restart tracking.
* ``RuntimeBackend`` integration — the catalog is the single source of
  MCP servers at agent start (``_mcp_servers_for``), degrading to no
  servers when the store is unwired or unreadable.
* Dashboard API — ``GET/PUT/DELETE /api/connectors`` and
  ``POST /api/agents/restart-batch``.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.host.connectors import ConnectorStore
from src.shared.types import CONNECTOR_ALL_AGENTS, MCPConnector

# ── MCPConnector model ───────────────────────────────────────


class TestMCPConnectorModel:
    def test_inherits_server_validators(self):
        with pytest.raises(ValueError):
            MCPConnector(name="x", command="run $CRED{tok}")  # handle in command

    def test_star_exclusive(self):
        with pytest.raises(ValueError):
            MCPConnector(name="x", command="y", agents=["*", "a"])

    def test_invalid_agent_id_rejected(self):
        with pytest.raises(ValueError):
            MCPConnector(name="x", command="y", agents=["bad id!"])

    def test_agents_deduped_order_preserved(self):
        c = MCPConnector(name="x", command="y", agents=["b", "a", "b"])
        assert c.agents == ["b", "a"]

    def test_applies_to(self):
        c = MCPConnector(name="x", command="y", agents=[CONNECTOR_ALL_AGENTS])
        assert c.applies_to("anyone")
        c2 = MCPConnector(name="x", command="y", agents=["a"])
        assert c2.applies_to("a") and not c2.applies_to("b")

    def test_server_dict_strips_assignment(self):
        c = MCPConnector(name="x", command="y", args=["-v"], agents=["a"])
        assert c.server_dict() == {
            "name": "x", "command": "y", "args": ["-v"], "env": None,
        }


# ── ConnectorStore ───────────────────────────────────────────


class TestConnectorStore:
    def _store(self, tmp_path) -> ConnectorStore:
        return ConnectorStore(str(tmp_path / "connectors.json"))

    def test_empty_when_file_missing(self, tmp_path):
        assert self._store(tmp_path).list() == []

    def test_upsert_persists_and_reloads(self, tmp_path):
        s = self._store(tmp_path)
        s.upsert(MCPConnector(name="fs", command="mcp-fs", agents=["*"]))
        s2 = self._store(tmp_path)
        assert [c.name for c in s2.list()] == ["fs"]

    def test_upsert_replaces_case_insensitive(self, tmp_path):
        s = self._store(tmp_path)
        s.upsert(MCPConnector(name="FS", command="old"))
        previous = s.upsert(MCPConnector(name="fs", command="new"))
        assert previous is not None and previous.command == "old"
        assert len(s.list()) == 1 and s.list()[0].command == "new"

    def test_remove(self, tmp_path):
        s = self._store(tmp_path)
        s.upsert(MCPConnector(name="fs", command="x"))
        assert s.remove("FS") is not None
        assert s.remove("fs") is None
        assert self._store(tmp_path).list() == []

    def test_stdio_for_agent_assignment_and_order(self, tmp_path):
        s = self._store(tmp_path)
        s.upsert(MCPConnector(name="one", command="c1", agents=["a"]))
        s.upsert(MCPConnector(name="two", command="c2", agents=["*"]))
        s.upsert(MCPConnector(name="three", command="c3", agents=["b"]))
        assert [d["name"] for d in s.stdio_for_agent("a")] == ["one", "two"]
        assert [d["name"] for d in s.stdio_for_agent("b")] == ["two", "three"]
        # Catalog-only fields never reach MCP_SERVERS shape.
        for d in s.stdio_for_agent("a"):
            assert set(d) == {"name", "command", "args", "env"}

    def test_unassigned_connector_reaches_nobody(self, tmp_path):
        s = self._store(tmp_path)
        s.upsert(MCPConnector(name="staged", command="c"))
        assert s.stdio_for_agent("a") == []

    def test_assigned_agents_expansion(self, tmp_path):
        s = self._store(tmp_path)
        s.upsert(MCPConnector(name="all", command="c", agents=["*"]))
        s.upsert(MCPConnector(name="some", command="c", agents=["a", "ghost"]))
        assert s.assigned_agents("all", ["b", "a"]) == ["a", "b"]
        # Explicit ids intersect with the live registry.
        assert s.assigned_agents("some", ["a", "b"]) == ["a"]
        assert s.assigned_agents("missing", ["a"]) == []

    def test_corrupt_file_fails_closed(self, tmp_path):
        path = tmp_path / "connectors.json"
        path.write_text("{not json")
        assert ConnectorStore(str(path)).list() == []

    def test_malformed_entry_dropped_not_fatal(self, tmp_path):
        path = tmp_path / "connectors.json"
        path.write_text(json.dumps({"connectors": [
            {"name": "ok", "command": "c"},
            {"name": "bad name!", "command": "c"},
            {"name": "OK", "command": "dup"},
        ]}))
        s = ConnectorStore(str(path))
        assert [c.name for c in s.list()] == ["ok"]

    def test_save_is_atomic_no_tmp_left(self, tmp_path):
        s = self._store(tmp_path)
        s.upsert(MCPConnector(name="fs", command="c"))
        leftovers = [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]
        assert leftovers == []

    def test_dirty_tracking(self, tmp_path):
        s = self._store(tmp_path)
        s.mark_dirty(["b", "a"])
        assert s.pending_restart() == ["a", "b"]
        s.mark_clean("a")
        assert s.pending_restart() == ["b"]
        s.mark_clean("never-dirty")  # no-op, no raise


# ── RuntimeBackend integration ───────────────────────────────


class TestRuntimeCatalogIntegration:
    def _backend(self):
        from src.host.runtime import DockerBackend
        b = DockerBackend.__new__(DockerBackend)
        return b

    def test_unwired_store_means_no_servers(self):
        b = self._backend()
        assert b._mcp_servers_for("a") == []

    def test_wired_store_resolves_assignment(self, tmp_path):
        b = self._backend()
        store = ConnectorStore(str(tmp_path / "c.json"))
        store.upsert(MCPConnector(name="fs", command="mcp-fs", agents=["*"]))
        b.set_connector_store(store)
        assert [d["name"] for d in b._mcp_servers_for("anyone")] == ["fs"]

    def test_store_error_degrades_to_no_servers(self):
        b = self._backend()
        broken = MagicMock()
        broken.stdio_for_agent.side_effect = RuntimeError("disk gone")
        b.set_connector_store(broken)
        assert b._mcp_servers_for("a") == []

    def test_start_agent_signature_has_no_mcp_param(self):
        import inspect

        from src.host.runtime import DockerBackend, RuntimeBackend, SandboxBackend
        for cls in (RuntimeBackend, DockerBackend, SandboxBackend):
            sig = inspect.signature(cls.start_agent)
            assert "mcp_servers" not in sig.parameters, cls.__name__


# ── Dashboard API ────────────────────────────────────────────


class _CSRFClient(TestClient):
    def request(self, method, url, **kwargs):
        if method.upper() not in ("GET", "HEAD", "OPTIONS"):
            headers = kwargs.get("headers") or {}
            headers.setdefault("X-Requested-With", "XMLHttpRequest")
            kwargs["headers"] = headers
        return super().request(method, url, **kwargs)


@pytest.fixture
def connector_env(tmp_path):
    """Dashboard router with a real ConnectorStore and two agents."""
    from src.dashboard.events import EventBus
    from src.dashboard.server import create_dashboard_router
    from src.host.costs import CostTracker
    from src.host.mesh import Blackboard
    from src.host.traces import TraceStore

    store = ConnectorStore(str(tmp_path / "connectors.json"))
    permissions = MagicMock()
    permissions.can_access_credential.return_value = True
    vault = MagicMock()
    vault.resolve_credential.return_value = "secret-value"
    vault.list_agent_credential_names.return_value = ["linear_token"]
    router = create_dashboard_router(
        blackboard=Blackboard(db_path=str(tmp_path / "bb.db")),
        health_monitor=None,
        cost_tracker=CostTracker(db_path=str(tmp_path / "costs.db")),
        trace_store=TraceStore(db_path=str(tmp_path / "traces.db")),
        event_bus=EventBus(),
        agent_registry={"alpha": "http://x:1", "beta": "http://x:2"},
        permissions=permissions,
        credential_vault=vault,
        connector_store=store,
    )
    app = FastAPI()
    app.include_router(router)
    client = _CSRFClient(app)
    yield client, store, permissions, vault
    client.close()


class TestConnectorEndpoints:
    def test_list_empty(self, connector_env):
        client, *_ = connector_env
        data = client.get("/dashboard/api/connectors").json()
        assert data["connectors"] == []
        assert data["pending_restart"] == []
        assert data["agents"] == ["alpha", "beta"]
        assert data["available_credentials"] == ["linear_token"]

    def test_upsert_create_and_masking(self, connector_env):
        client, store, *_ = connector_env
        resp = client.put("/dashboard/api/connectors/linear", json={
            "command": "mcp-server-linear",
            "args": ["--workspace", "ol"],
            "env": {"LINEAR_API_KEY": "$CRED{linear_token}"},
            "agents": ["*"],
        })
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["restart_required"] is True
        assert data["affected_agents"] == ["alpha", "beta"]
        # env never returned; env_keys stand in.
        assert "env" not in data["connector"]
        assert data["connector"]["env_keys"] == ["LINEAR_API_KEY"]
        # Persisted with the real env.
        assert store.get("linear").env == {"LINEAR_API_KEY": "$CRED{linear_token}"}
        # Pending restart recorded for the affected agents.
        assert client.get("/dashboard/api/connectors").json()["pending_restart"] == ["alpha", "beta"]

    def test_upsert_env_and_agents_absent_preserve(self, connector_env):
        client, store, *_ = connector_env
        client.put("/dashboard/api/connectors/linear", json={
            "command": "c", "env": {"K": "v"}, "agents": ["alpha"],
        })
        resp = client.put("/dashboard/api/connectors/linear", json={
            "command": "c2",
        })
        assert resp.status_code == 200
        assert store.get("linear").env == {"K": "v"}
        assert store.get("linear").command == "c2"
        assert store.get("linear").agents == ["alpha"]

    def test_upsert_noop_returns_no_restart(self, connector_env):
        client, store, *_ = connector_env
        body = {"command": "c", "agents": ["alpha"]}
        client.put("/dashboard/api/connectors/fs", json=body)
        store.mark_clean("alpha")
        resp = client.put("/dashboard/api/connectors/fs", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["restart_required"] is False
        assert data["affected_agents"] == []
        assert store.pending_restart() == []

    def test_upsert_validation_error_shape(self, connector_env):
        client, *_ = connector_env
        resp = client.put("/dashboard/api/connectors/bad", json={
            "command": "run $CRED{tok}", "agents": [],
        })
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert detail["field"] == "connector"
        assert any("command" in e["loc"] for e in detail["errors"])

    def test_upsert_name_mismatch_rejected(self, connector_env):
        client, *_ = connector_env
        resp = client.put("/dashboard/api/connectors/a", json={"name": "b", "command": "c"})
        assert resp.status_code == 400

    def test_cred_missing_from_vault_rejected(self, connector_env):
        client, _, _, vault = connector_env
        vault.resolve_credential.return_value = None
        resp = client.put("/dashboard/api/connectors/x", json={
            "command": "c", "env": {"K": "$CRED{ghost}"}, "agents": ["alpha"],
        })
        assert resp.status_code == 400
        assert "ghost" in resp.json()["detail"]

    def test_cred_permission_blocked_agent_named(self, connector_env):
        client, _, permissions, _ = connector_env
        permissions.can_access_credential.side_effect = (
            lambda agent, cred: agent != "beta"
        )
        resp = client.put("/dashboard/api/connectors/x", json={
            "command": "c", "env": {"K": "$CRED{linear_token}"}, "agents": ["*"],
        })
        assert resp.status_code == 400
        assert "beta" in resp.json()["detail"]

    def test_delete_marks_previous_assignment_dirty(self, connector_env):
        client, store, *_ = connector_env
        client.put("/dashboard/api/connectors/fs", json={"command": "c", "agents": ["alpha"]})
        store.mark_clean("alpha")
        resp = client.request("DELETE", "/dashboard/api/connectors/fs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["removed"] is True
        assert data["affected_agents"] == ["alpha"]
        assert store.get("fs") is None
        assert store.pending_restart() == ["alpha"]

    def test_delete_unknown_404(self, connector_env):
        client, *_ = connector_env
        assert client.request("DELETE", "/dashboard/api/connectors/ghost").status_code == 404

    def test_restart_batch_validates_body(self, connector_env):
        client, *_ = connector_env
        assert client.post("/dashboard/api/agents/restart-batch", json={"agents": []}).status_code == 400
        assert client.post(
            "/dashboard/api/agents/restart-batch", json={"agents": ["a"] * 33},
        ).status_code == 400
        assert client.post(
            "/dashboard/api/agents/restart-batch", json={"agents": [1]},
        ).status_code == 400

    def test_restart_batch_reports_per_agent_results(self, connector_env):
        client, *_ = connector_env
        # No runtime wired in this fixture → the single-agent restart
        # handler 503s for known agents, 404s for unknown; either way
        # the batch reports per-agent failure instead of raising.
        resp = client.post(
            "/dashboard/api/agents/restart-batch", json={"agents": ["alpha", "ghost"]},
        )
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert set(results) == {"alpha", "ghost"}
        assert all(v.startswith("failed:") for v in results.values())

    def test_unavailable_store_503(self, tmp_path):
        from src.dashboard.events import EventBus
        from src.dashboard.server import create_dashboard_router
        from src.host.costs import CostTracker
        from src.host.mesh import Blackboard
        from src.host.traces import TraceStore
        router = create_dashboard_router(
            blackboard=Blackboard(db_path=str(tmp_path / "bb.db")),
            health_monitor=None,
            cost_tracker=CostTracker(db_path=str(tmp_path / "costs.db")),
            trace_store=TraceStore(db_path=str(tmp_path / "traces.db")),
            event_bus=EventBus(),
            agent_registry={},
        )
        app = FastAPI()
        app.include_router(router)
        with _CSRFClient(app) as client:
            assert client.get("/dashboard/api/connectors").status_code == 503
