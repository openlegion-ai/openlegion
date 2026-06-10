"""Tests for the fleet MCP connector catalog.

Covers the three layers:

* ``ConnectorStore`` — persistence, atomicity, fail-closed loading,
  assignment resolution, pending-restart tracking.
* ``RuntimeBackend`` integration — the catalog is the single source of
  MCP servers at agent start (``_mcp_snapshot_for``), degrading to no
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
        s.upsert(MCPConnector(name="fs", command="new"))
        assert len(s.list()) == 1 and s.list()[0].command == "new"

    def test_remove(self, tmp_path):
        s = self._store(tmp_path)
        s.upsert(MCPConnector(name="fs", command="x"))
        assert s.remove("FS") is True
        assert s.remove("fs") is False
        assert self._store(tmp_path).list() == []

    def test_remove_agent_strips_explicit_assignments(self, tmp_path):
        s = self._store(tmp_path)
        s.upsert(MCPConnector(name="one", command="c", agents=["a", "b"]))
        s.upsert(MCPConnector(name="all", command="c", agents=["*"]))
        s.mark_dirty(["a"])
        s.remove_agent("a")
        assert s.get("one").agents == ["b"]
        # "*" semantics untouched — it means "whatever agents exist".
        assert s.get("all").agents == ["*"]
        assert "a" not in s.pending_restart()
        # Persisted: a fresh load sees the stripped assignment.
        assert ConnectorStore(str(tmp_path / "connectors.json")).get("one").agents == ["b"]

    def test_external_edit_reloads_and_touches(self, tmp_path):
        path = tmp_path / "connectors.json"
        s = ConnectorStore(str(path))
        s.upsert(MCPConnector(name="fs", command="old", agents=["a"]))
        servers, gen = s.snapshot_for_agent("a")
        s.record_agent_start("a", gen)
        assert s.pending_restart() == []
        # Hand-edit the file (headless operator) with a different mtime.
        data = json.loads(path.read_text())
        data["connectors"][0]["command"] = "new"
        path.write_text(json.dumps(data))
        os.utime(path, ns=(1, 1))  # force a visible stat change
        assert s.get("fs").command == "new"
        assert s.pending_restart() == ["a"]

    def test_transport_key_tolerated_in_stored_records(self, tmp_path):
        # Forward/rollback tolerance: records written with the Phase-2
        # transport field must load (stdio) rather than be dropped.
        path = tmp_path / "connectors.json"
        path.write_text(json.dumps({"connectors": [
            {"name": "fs", "transport": "stdio", "command": "c", "agents": ["*"]},
        ]}))
        assert [c.name for c in ConnectorStore(str(path)).list()] == ["fs"]

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

    def test_pending_restart_generation_derivation(self, tmp_path):
        s = self._store(tmp_path)
        s.mark_dirty(["b", "a"])
        assert s.pending_restart() == ["a", "b"]
        # A restart records the snapshot generation it was built from.
        _, gen = s.snapshot_for_agent("a")
        s.record_agent_start("a", gen)
        assert s.pending_restart() == ["b"]
        s.record_agent_start("never-dirty", 0)  # no-op, no raise

    def test_edit_during_container_build_stays_dirty(self, tmp_path):
        """The race the generation model exists to close: a catalog
        edit landing between the env-build snapshot and the
        post-start record must keep the agent pending-restart."""
        s = self._store(tmp_path)
        s.upsert(MCPConnector(name="fs", command="v1", agents=["a"]))
        s.mark_dirty(["a"])
        servers, snapshot_gen = s.snapshot_for_agent("a")  # env built here
        # Edit lands while the container is still being created.
        s.upsert(MCPConnector(name="fs", command="v2", agents=["a"]))
        s.mark_dirty(["a"])
        s.record_agent_start("a", snapshot_gen)  # start completes late
        assert s.pending_restart() == ["a"]
        # After a restart against the current catalog, it goes clean.
        _, gen2 = s.snapshot_for_agent("a")
        s.record_agent_start("a", gen2)
        assert s.pending_restart() == []


# ── RuntimeBackend integration ───────────────────────────────


class TestRuntimeCatalogIntegration:
    def _backend(self):
        from src.host.runtime import DockerBackend
        b = DockerBackend.__new__(DockerBackend)
        return b

    def test_unwired_store_means_no_servers(self):
        b = self._backend()
        assert b._mcp_snapshot_for("a") == ([], 0)

    def test_wired_store_resolves_assignment(self, tmp_path):
        b = self._backend()
        store = ConnectorStore(str(tmp_path / "c.json"))
        store.upsert(MCPConnector(name="fs", command="mcp-fs", agents=["*"]))
        b.set_connector_store(store)
        servers, gen = b._mcp_snapshot_for("anyone")
        assert [d["name"] for d in servers] == ["fs"]
        assert gen == store.snapshot_for_agent("anyone")[1]

    def test_store_error_degrades_to_no_servers(self):
        b = self._backend()
        broken = MagicMock()
        broken.snapshot_for_agent.side_effect = RuntimeError("disk gone")
        b.set_connector_store(broken)
        assert b._mcp_snapshot_for("a") == ([], 0)

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
    blackboard = Blackboard(db_path=str(tmp_path / "bb.db"))
    router = create_dashboard_router(
        blackboard=blackboard,
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
    yield client, store, permissions, vault, blackboard
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

    def test_affected_agents_intersect_live_registry(self, connector_env):
        client, *_ = connector_env
        resp = client.put("/dashboard/api/connectors/x", json={
            "command": "c", "agents": ["alpha", "ghost"],
        })
        assert resp.status_code == 200
        data = resp.json()
        # 'ghost' has no running container to restart; raw assignment kept.
        assert data["affected_agents"] == ["alpha"]
        assert data["connector"]["agents"] == ["alpha", "ghost"]
        assert data["connector"]["assigned_agents"] == ["alpha"]

    def test_upsert_noop_returns_no_restart(self, connector_env):
        client, store, *_ = connector_env
        body = {"command": "c", "agents": ["alpha"]}
        client.put("/dashboard/api/connectors/fs", json=body)
        # Simulate the agent restarting against the current catalog.
        _, gen = store.snapshot_for_agent("alpha")
        store.record_agent_start("alpha", gen)
        resp = client.put("/dashboard/api/connectors/fs", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["restart_required"] is False
        assert data["affected_agents"] == []
        assert store.pending_restart() == []

    def test_cred_check_skips_ghost_explicit_agents(self, connector_env):
        # A deleted agent's leftover id in an explicit assignment must
        # not block edits forever — the check runs against live agents.
        client, _, permissions, *_ = connector_env
        permissions.can_access_credential.side_effect = (
            lambda agent, cred: agent != "ghost"
        )
        resp = client.put("/dashboard/api/connectors/x", json={
            "command": "c", "env": {"K": "$CRED{linear_token}"},
            "agents": ["alpha", "ghost"],
        })
        assert resp.status_code == 200, resp.text

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
        client, _, _, vault, _ = connector_env
        vault.resolve_credential.return_value = None
        resp = client.put("/dashboard/api/connectors/x", json={
            "command": "c", "env": {"K": "$CRED{ghost}"}, "agents": ["alpha"],
        })
        assert resp.status_code == 400
        assert "ghost" in resp.json()["detail"]

    def test_cred_permission_blocked_agent_named(self, connector_env):
        client, _, permissions, *_ = connector_env
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
        _, gen = store.snapshot_for_agent("alpha")
        store.record_agent_start("alpha", gen)
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

    def test_pending_restart_filters_to_live_agents(self, connector_env):
        client, store, *_ = connector_env
        store.mark_dirty(["alpha", "long-gone"])
        data = client.get("/dashboard/api/connectors").json()
        assert data["pending_restart"] == ["alpha"]

    def test_audit_row_redacts_connector_env_values(self, connector_env):
        # Pin: env VALUES never reach the audit table; keys survive as
        # env_keys (the 'covert plaintext secret leak' guard).
        client, _, _, _, blackboard = connector_env
        client.put("/dashboard/api/connectors/linear", json={
            "command": "c", "env": {"LINEAR_API_KEY": "$CRED{linear_token}"},
            "agents": ["alpha"],
        })
        rows = blackboard.get_audit_log(per_page=10)
        flat = json.dumps(rows)
        assert "edit_connector" in flat
        assert "$CRED{linear_token}" not in flat
        assert "LINEAR_API_KEY" in flat

    def test_restart_batch_validates_body(self, connector_env):
        client, *_ = connector_env
        assert client.post("/dashboard/api/agents/restart-batch", json={"agents": []}).status_code == 400
        assert client.post(
            "/dashboard/api/agents/restart-batch", json={"agents": [1]},
        ).status_code == 400

    def test_restart_batch_starts_known_skips_unknown_and_dedupes(self, connector_env):
        client, *_ = connector_env
        resp = client.post(
            "/dashboard/api/agents/restart-batch",
            json={"agents": ["alpha", "ghost", "alpha"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["started"] == ["alpha"]
        assert data["skipped"] == {"ghost": "unknown agent"}

    def test_restart_batch_no_cap_for_large_fleets(self, connector_env):
        # The old 32-cap made 'Restart now' unusable for exactly the
        # fleet-wide connectors the feature exists for. Unknown ids are
        # skipped, not rejected.
        client, *_ = connector_env
        agents = [f"a{i}" for i in range(40)]
        resp = client.post(
            "/dashboard/api/agents/restart-batch", json={"agents": agents},
        )
        assert resp.status_code == 200
        assert resp.json()["started"] == []
        assert len(resp.json()["skipped"]) == 40

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


# ── Deletion-path cleanup ────────────────────────────────────
#
# Every permanent agent-deletion path must strip the agent from
# connector assignments, or a future agent recreated under the same
# name silently inherits the deleted agent's MCP connectors (and
# their $CRED-bearing env). The dashboard delete path is pinned in
# the endpoint tests above; these pin the mesh confirm-delete path
# (via ``app.cleanup_agent``) and the CLI REPL remove path.


class TestDeletionPathCleanup:
    def test_mesh_cleanup_agent_strips_connector_assignments(self, tmp_path):
        from src.host.mesh import Blackboard, MessageRouter, PubSub
        from src.host.permissions import PermissionMatrix
        from src.host.server import create_mesh_app

        store = ConnectorStore(str(tmp_path / "connectors.json"))
        store.upsert(MCPConnector(name="fs", command="c", agents=["worker", "other"]))
        store.mark_dirty(["worker"])
        bb = Blackboard(db_path=str(tmp_path / "bb.db"))
        perms = PermissionMatrix.__new__(PermissionMatrix)
        perms.permissions = {}
        app = create_mesh_app(
            bb, PubSub(), MessageRouter(permissions=perms, agent_registry={}),
            perms, connector_store=store,
        )
        app.cleanup_agent("worker")
        assert store.get("fs").agents == ["other"]
        assert "worker" not in store.pending_restart()
        bb.close()

    def test_repl_remove_strips_connector_assignments(self, tmp_path, monkeypatch):
        import click

        from src.cli.repl import REPLSession

        store = ConnectorStore(str(tmp_path / "connectors.json"))
        store.upsert(MCPConnector(name="fs", command="c", agents=["worker", "other"]))
        sess = REPLSession.__new__(REPLSession)
        sess.ctx = MagicMock()
        sess.ctx.agents = {"worker": "http://x"}
        sess.ctx.connector_store = store
        sess.current = None
        monkeypatch.setattr(click, "confirm", lambda *a, **k: True)
        monkeypatch.setattr("src.cli.config._remove_agent", lambda *a, **k: None)
        sess._cmd_remove("worker")
        assert store.get("fs").agents == ["other"]
