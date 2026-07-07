"""Per-agent standing goals mesh endpoints (ratified #7 / C.3-b).

``GET/PUT/DELETE /mesh/agents/{id}/goals`` back the Team store's
``agent_goals`` table — the single home for standing goals since the
blackboard ``goals/`` key path was deleted. Read is self-or-operator
(plus loopback internal); writes are operator/internal only.
"""

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient


def _build_app(tmp_path, monkeypatch):
    import src.cli.config as cli_config
    from src.host.mesh import Blackboard, MessageRouter, PubSub
    from src.host.permissions import PermissionMatrix
    from src.host.server import create_mesh_app

    monkeypatch.setattr(
        cli_config,
        "_load_config",
        lambda: {"agents": {"scout": {"role": "researcher"}, "other": {"role": "writer"}}},
    )

    perms = PermissionMatrix()
    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    registry = {
        "operator": "http://operator:8400",
        "scout": "http://scout:8400",
        "other": "http://other:8400",
    }
    router = MessageRouter(perms, registry)

    app = create_mesh_app(
        blackboard=bb,
        pubsub=pubsub,
        router=router,
        permissions=perms,
        auth_tokens={
            "operator": "tok-op",
            "scout": "tok-scout",
            "other": "tok-other",
        },
    )
    return app, bb


_OP = {"Authorization": "Bearer tok-op"}
_SCOUT = {"Authorization": "Bearer tok-scout"}
_OTHER = {"Authorization": "Bearer tok-other"}


# ── read side ────────────────────────────────────────────────────────


def test_get_unset_returns_empty_record(tmp_path, monkeypatch):
    app, bb = _build_app(tmp_path, monkeypatch)
    try:
        resp = TestClient(app).get("/mesh/agents/scout/goals", headers=_OP)
        assert resp.status_code == 200, resp.text
        assert resp.json() == {
            "agent_id": "scout",
            "goals": [],
            "set_by": None,
            "updated_at": None,
        }
    finally:
        bb.close()


def test_agent_can_read_own_goals(tmp_path, monkeypatch):
    app, bb = _build_app(tmp_path, monkeypatch)
    try:
        app.teams_store.set_agent_goals("scout", ["Find leads."], set_by="operator")
        resp = TestClient(app).get("/mesh/agents/scout/goals", headers=_SCOUT)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["goals"] == ["Find leads."]
        assert body["set_by"] == "operator"
        assert body["updated_at"]
    finally:
        bb.close()


def test_agent_cannot_read_peer_goals(tmp_path, monkeypatch):
    app, bb = _build_app(tmp_path, monkeypatch)
    try:
        app.teams_store.set_agent_goals("scout", ["Find leads."])
        resp = TestClient(app).get("/mesh/agents/scout/goals", headers=_OTHER)
        assert resp.status_code == 403, resp.text
    finally:
        bb.close()


def test_operator_can_read_any_agents_goals(tmp_path, monkeypatch):
    app, bb = _build_app(tmp_path, monkeypatch)
    try:
        app.teams_store.set_agent_goals("scout", ["Find leads."])
        resp = TestClient(app).get("/mesh/agents/scout/goals", headers=_OP)
        assert resp.status_code == 200, resp.text
        assert resp.json()["goals"] == ["Find leads."]
    finally:
        bb.close()


def test_unauthenticated_read_rejected(tmp_path, monkeypatch):
    app, bb = _build_app(tmp_path, monkeypatch)
    try:
        resp = TestClient(app).get("/mesh/agents/scout/goals")
        assert resp.status_code == 401, resp.text
    finally:
        bb.close()


@pytest.mark.asyncio
async def test_internal_loopback_caller_can_read_and_write(tmp_path, monkeypatch):
    # httpx ASGITransport presents the client as 127.0.0.1, so the
    # loopback + x-mesh-internal predicate passes (mirrors the mesh's
    # own dispatcher / dashboard proxy path).
    app, bb = _build_app(tmp_path, monkeypatch)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            put = await client.put(
                "/mesh/agents/scout/goals",
                headers={"x-mesh-internal": "1"},
                json={"goals": ["Watch the inbox."], "set_by": "user"},
            )
            assert put.status_code == 200, put.text
            assert put.json() == {"agent_id": "scout", "set": True, "count": 1}
            get = await client.get(
                "/mesh/agents/scout/goals",
                headers={"x-mesh-internal": "1"},
            )
            assert get.status_code == 200, get.text
            assert get.json()["goals"] == ["Watch the inbox."]
            assert get.json()["set_by"] == "user"
    finally:
        bb.close()


# ── write side ───────────────────────────────────────────────────────


def test_operator_put_get_delete_roundtrip(tmp_path, monkeypatch):
    app, bb = _build_app(tmp_path, monkeypatch)
    client = TestClient(app)
    try:
        put = client.put(
            "/mesh/agents/scout/goals",
            headers=_OP,
            json={"goals": ["Find 10 qualified leads per week."]},
        )
        assert put.status_code == 200, put.text
        assert put.json() == {"agent_id": "scout", "set": True, "count": 1}
        # set_by defaults to "operator".
        record = app.teams_store.get_agent_goals("scout")
        assert record["goals"] == ["Find 10 qualified leads per week."]
        assert record["set_by"] == "operator"

        dele = client.delete("/mesh/agents/scout/goals", headers=_OP)
        assert dele.status_code == 200, dele.text
        assert dele.json()["cleared"] is True
        assert dele.json()["existed"] is True
        assert app.teams_store.get_agent_goals("scout") is None

        # DELETE is idempotent — clearing unset goals is fine.
        again = client.delete("/mesh/agents/scout/goals", headers=_OP)
        assert again.status_code == 200
        assert again.json()["existed"] is False
    finally:
        bb.close()


def test_worker_cannot_write_even_own_goals(tmp_path, monkeypatch):
    app, bb = _build_app(tmp_path, monkeypatch)
    client = TestClient(app)
    try:
        resp = client.put(
            "/mesh/agents/scout/goals",
            headers=_SCOUT,
            json={"goals": ["x"]},
        )
        assert resp.status_code == 403, resp.text
        resp = client.delete("/mesh/agents/scout/goals", headers=_SCOUT)
        assert resp.status_code == 403, resp.text
    finally:
        bb.close()


def test_put_empty_list_clears(tmp_path, monkeypatch):
    app, bb = _build_app(tmp_path, monkeypatch)
    client = TestClient(app)
    try:
        app.teams_store.set_agent_goals("scout", ["Old goal."])
        resp = client.put(
            "/mesh/agents/scout/goals",
            headers=_OP,
            json={"goals": []},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"agent_id": "scout", "cleared": True}
        assert app.teams_store.get_agent_goals("scout") is None
    finally:
        bb.close()


def test_put_rejects_operator_target(tmp_path, monkeypatch):
    app, bb = _build_app(tmp_path, monkeypatch)
    try:
        resp = TestClient(app).put(
            "/mesh/agents/operator/goals",
            headers=_OP,
            json={"goals": ["x"]},
        )
        assert resp.status_code == 400, resp.text
        assert "manage_goals" in resp.json()["detail"]
    finally:
        bb.close()


def test_put_unknown_agent_404_names_available(tmp_path, monkeypatch):
    app, bb = _build_app(tmp_path, monkeypatch)
    try:
        resp = TestClient(app).put(
            "/mesh/agents/ghost/goals",
            headers=_OP,
            json={"goals": ["x"]},
        )
        assert resp.status_code == 404, resp.text
        detail = resp.json()["detail"]
        assert "not found" in detail
        assert "scout" in detail
    finally:
        bb.close()


def test_put_validation_400s(tmp_path, monkeypatch):
    app, bb = _build_app(tmp_path, monkeypatch)
    client = TestClient(app)
    try:
        # Not a list.
        resp = client.put(
            "/mesh/agents/scout/goals",
            headers=_OP,
            json={"goals": "do it"},
        )
        assert resp.status_code == 400
        # Too many entries (max 5).
        resp = client.put(
            "/mesh/agents/scout/goals",
            headers=_OP,
            json={"goals": [f"goal {i}" for i in range(6)]},
        )
        assert resp.status_code == 400
        # A single goal over 300 chars.
        resp = client.put(
            "/mesh/agents/scout/goals",
            headers=_OP,
            json={"goals": ["a" * 301]},
        )
        assert resp.status_code == 400
        # Blank after strip.
        resp = client.put(
            "/mesh/agents/scout/goals",
            headers=_OP,
            json={"goals": [" "]},
        )
        assert resp.status_code == 400
        # Non-string entry.
        resp = client.put(
            "/mesh/agents/scout/goals",
            headers=_OP,
            json={"goals": [42]},
        )
        assert resp.status_code == 400
        # Nothing persisted by any rejected payload.
        assert app.teams_store.get_agent_goals("scout") is None
    finally:
        bb.close()


def test_delete_unknown_agent_404_names_available(tmp_path, monkeypatch):
    """A typo'd clear must NOT report success while the real target's
    goals stay in force (adversarial-review finding)."""
    app, bb = _build_app(tmp_path, monkeypatch)
    try:
        resp = TestClient(app).delete("/mesh/agents/ghost/goals", headers=_OP)
        assert resp.status_code == 404, resp.text
        detail = resp.json()["detail"]
        assert "not found" in detail
        assert "scout" in detail
    finally:
        bb.close()


def test_delete_rejects_operator_target(tmp_path, monkeypatch):
    app, bb = _build_app(tmp_path, monkeypatch)
    try:
        resp = TestClient(app).delete("/mesh/agents/operator/goals", headers=_OP)
        assert resp.status_code == 400, resp.text
        assert "manage_goals" in resp.json()["detail"]
    finally:
        bb.close()


def test_goals_writes_leave_audit_trail(tmp_path, monkeypatch):
    """Goals are prompt-injected standing instructions — LLM-driven
    rewrites must be reviewable in the audit log (parity with the
    dashboard's human write path)."""
    app, bb = _build_app(tmp_path, monkeypatch)
    client = TestClient(app)
    try:
        client.put(
            "/mesh/agents/scout/goals",
            headers=_OP,
            json={"goals": ["New goal."]},
        )
        client.delete("/mesh/agents/scout/goals", headers=_OP)
        entries = bb.get_audit_log(per_page=10)["entries"]
        actions = [e["action"] for e in entries if e["target"] == "scout"]
        assert "edit_goals" in actions
        assert "clear_goals" in actions
        edit = next(e for e in entries if e["action"] == "edit_goals")
        assert "New goal." in edit["after_value"]
    finally:
        bb.close()
