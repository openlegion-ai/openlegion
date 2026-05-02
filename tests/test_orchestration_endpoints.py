"""HTTP endpoint tests for orchestration tasks v2 (Task 6).

Boots the mesh app with ``OPENLEGION_ORCHESTRATION_TASKS_V2=1``, drives
the new ``/mesh/tasks*`` routes through ``ASGITransport``, and verifies
the permission gates and 503 fall-through.
"""

from __future__ import annotations

import importlib

import pytest
import yaml
from httpx import ASGITransport, AsyncClient

from src.host.mesh import Blackboard, MessageRouter, PubSub
from src.host.permissions import PermissionMatrix
from src.shared.types import AgentPermissions


def _reload_server(monkeypatch, *, v2: bool, tasks_db: str):
    """Reload ``src.host.server`` after pinning the env vars.

    The orchestration v2 flag is read at module import. Tests that
    flip the flag must reload the module so ``_ORCHESTRATION_TASKS_V2``
    picks up the new value.
    """
    if v2:
        monkeypatch.setenv("OPENLEGION_ORCHESTRATION_TASKS_V2", "1")
        monkeypatch.setenv("OPENLEGION_ORCHESTRATION_TASKS_DB", tasks_db)
    else:
        monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_V2", raising=False)
    import src.host.server as server_module
    importlib.reload(server_module)
    return server_module


def _build_app(tmp_path, server_module, *, perms_map, agents=None):
    blackboard = Blackboard(str(tmp_path / "bb.db"))
    pubsub = PubSub()
    permissions = PermissionMatrix()
    for aid, perms in perms_map.items():
        permissions.permissions[aid] = AgentPermissions(agent_id=aid, **perms)
    router = MessageRouter(permissions, agents or {})
    app = server_module.create_mesh_app(
        blackboard=blackboard,
        pubsub=pubsub,
        router=router,
        permissions=permissions,
    )
    return app, blackboard


def _projects_layout(tmp_path):
    """Two projects: research with [scout, analyst]; ops with [tracker]."""
    pdir = tmp_path / "projects"
    research = pdir / "research"
    research.mkdir(parents=True)
    (research / "metadata.yaml").write_text(yaml.dump({
        "name": "research",
        "members": ["scout", "analyst"],
        "created_at": "2026-05-02T00:00:00+00:00",
    }))
    ops = pdir / "ops"
    ops.mkdir(parents=True)
    (ops / "metadata.yaml").write_text(yaml.dump({
        "name": "ops",
        "members": ["tracker"],
        "created_at": "2026-05-02T00:00:00+00:00",
    }))
    return pdir


@pytest.fixture
def v2_app(tmp_path, monkeypatch):
    """Build a fresh v2-enabled mesh app for each test."""
    server_module = _reload_server(
        monkeypatch, v2=True, tasks_db=str(tmp_path / "tasks.db"),
    )
    perms_map = {
        "operator": {"can_route_tasks": True},
        "scout":    {"can_route_tasks": True, "can_message": ["analyst", "operator"]},
        "analyst":  {"can_route_tasks": False, "can_message": ["scout"]},
        "tracker":  {"can_route_tasks": False, "can_message": []},
    }
    app, bb = _build_app(
        tmp_path, server_module,
        perms_map=perms_map,
        agents={
            "scout": "http://scout:8400",
            "analyst": "http://analyst:8400",
            "tracker": "http://tracker:8400",
            "operator": "http://operator:8400",
        },
    )
    yield app, server_module, tmp_path
    bb.close()
    # Reset the v2 flag so other tests see legacy default.
    monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_V2", raising=False)
    monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_DB", raising=False)
    importlib.reload(server_module)


@pytest.mark.asyncio
async def test_status_endpoint_when_disabled_returns_503(tmp_path, monkeypatch):
    server_module = _reload_server(monkeypatch, v2=False, tasks_db="")
    app, bb = _build_app(
        tmp_path, server_module, perms_map={"scout": {}},
        agents={"scout": "http://scout:8400"},
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/mesh/orchestration/status", headers={"X-Agent-ID": "scout"})
    assert r.status_code == 503
    bb.close()


@pytest.mark.asyncio
async def test_status_endpoint_when_enabled_returns_200(v2_app):
    app, server_module, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/mesh/orchestration/status", headers={"X-Agent-ID": "scout"})
    assert r.status_code == 200
    assert r.json() == {"enabled": True}


@pytest.mark.asyncio
async def test_tasks_endpoints_503_when_disabled(tmp_path, monkeypatch):
    server_module = _reload_server(monkeypatch, v2=False, tasks_db="")
    app, bb = _build_app(
        tmp_path, server_module, perms_map={"scout": {"can_route_tasks": True}},
        agents={"scout": "http://scout:8400"},
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/tasks",
            json={"assignee": "analyst", "title": "x"},
            headers={"X-Agent-ID": "scout"},
        )
    assert r.status_code == 503
    bb.close()


@pytest.mark.asyncio
async def test_create_task_success(v2_app):
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/tasks",
            json={
                "assignee": "analyst",
                "title": "research handoff",
                "description": "dig into X",
            },
            headers={"X-Agent-ID": "scout"},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["assignee"] == "analyst"
    assert body["creator"] == "scout"
    assert body["status"] == "pending"
    assert body["id"].startswith("task_")


@pytest.mark.asyncio
async def test_create_task_requires_can_route_tasks(v2_app):
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/tasks",
            json={"assignee": "scout", "title": "no permission"},
            headers={"X-Agent-ID": "analyst"},
        )
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_create_task_invalid_assignee(v2_app):
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/tasks",
            json={"assignee": "../../etc", "title": "evil"},
            headers={"X-Agent-ID": "scout"},
        )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_get_task_404(v2_app):
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get(
            "/mesh/tasks/missing",
            headers={"X-Agent-ID": "scout"},
        )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_get_task_visible_to_creator_assignee_operator(v2_app):
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        # Scout creates a task for analyst.
        r = await c.post(
            "/mesh/tasks",
            json={"assignee": "analyst", "title": "x"},
            headers={"X-Agent-ID": "scout"},
        )
        tid = r.json()["id"]
        # Creator can read.
        r = await c.get(f"/mesh/tasks/{tid}", headers={"X-Agent-ID": "scout"})
        assert r.status_code == 200
        # Assignee can read.
        r = await c.get(f"/mesh/tasks/{tid}", headers={"X-Agent-ID": "analyst"})
        assert r.status_code == 200
        # Operator can read.
        r = await c.get(f"/mesh/tasks/{tid}", headers={"X-Agent-ID": "operator"})
        assert r.status_code == 200
        # Outsider cannot.
        r = await c.get(f"/mesh/tasks/{tid}", headers={"X-Agent-ID": "tracker"})
        assert r.status_code == 403


@pytest.mark.asyncio
async def test_inbox_visibility_assignee_only(v2_app):
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        # Scout creates two tasks for analyst.
        for title in ("task1", "task2"):
            await c.post(
                "/mesh/tasks",
                json={"assignee": "analyst", "title": title},
                headers={"X-Agent-ID": "scout"},
            )
        # Assignee can read own inbox.
        r = await c.get(
            "/mesh/tasks/inbox/analyst",
            headers={"X-Agent-ID": "analyst"},
        )
        assert r.status_code == 200
        assert r.json()["count"] == 2
        # Operator can read any inbox.
        r = await c.get(
            "/mesh/tasks/inbox/analyst",
            headers={"X-Agent-ID": "operator"},
        )
        assert r.status_code == 200
        # Other worker cannot.
        r = await c.get(
            "/mesh/tasks/inbox/analyst",
            headers={"X-Agent-ID": "tracker"},
        )
        assert r.status_code == 403


@pytest.mark.asyncio
async def test_project_list_requires_membership(v2_app, monkeypatch):
    app, _, tmp_path = v2_app
    pdir = _projects_layout(tmp_path)
    monkeypatch.setattr("src.cli.config.PROJECTS_DIR", pdir)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        # Scout (member of research) creates a task in research.
        await c.post(
            "/mesh/tasks",
            json={"assignee": "analyst", "title": "scoped", "project": "research"},
            headers={"X-Agent-ID": "scout"},
        )
        # Member can list project tasks.
        r = await c.get(
            "/mesh/tasks/project/research",
            headers={"X-Agent-ID": "scout"},
        )
        assert r.status_code == 200
        assert r.json()["count"] == 1
        # Non-member is denied.
        r = await c.get(
            "/mesh/tasks/project/research",
            headers={"X-Agent-ID": "tracker"},
        )
        assert r.status_code == 403
        # Operator gets through.
        r = await c.get(
            "/mesh/tasks/project/research",
            headers={"X-Agent-ID": "operator"},
        )
        assert r.status_code == 200


@pytest.mark.asyncio
async def test_status_update_creator_assignee_operator_outsider(v2_app):
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/tasks",
            json={"assignee": "analyst", "title": "transition"},
            headers={"X-Agent-ID": "scout"},
        )
        tid = r.json()["id"]
        # Outsider denied.
        r = await c.post(
            f"/mesh/tasks/{tid}/status",
            json={"status": "working"},
            headers={"X-Agent-ID": "tracker"},
        )
        assert r.status_code == 403
        # Assignee can transition pending → working.
        r = await c.post(
            f"/mesh/tasks/{tid}/status",
            json={"status": "working"},
            headers={"X-Agent-ID": "analyst"},
        )
        assert r.status_code == 200
        assert r.json()["status"] == "working"
        # Invalid transition → 400.
        r = await c.post(
            f"/mesh/tasks/{tid}/status",
            json={"status": "pending"},
            headers={"X-Agent-ID": "analyst"},
        )
        assert r.status_code == 400


@pytest.mark.asyncio
async def test_reroute_requires_can_route_tasks(v2_app):
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/tasks",
            json={"assignee": "analyst", "title": "routed"},
            headers={"X-Agent-ID": "scout"},
        )
        tid = r.json()["id"]
        # Operator can reroute.
        r = await c.post(
            f"/mesh/tasks/{tid}/reroute",
            json={"new_assignee": "tracker", "reason": "load balance"},
            headers={"X-Agent-ID": "operator"},
        )
        assert r.status_code == 200
        # Worker without can_route_tasks denied.
        r = await c.post(
            f"/mesh/tasks/{tid}/reroute",
            json={"new_assignee": "analyst"},
            headers={"X-Agent-ID": "analyst"},
        )
        assert r.status_code == 403
        # Worker WITH can_route_tasks allowed (scout has it).
        r = await c.post(
            f"/mesh/tasks/{tid}/reroute",
            json={"new_assignee": "scout"},
            headers={"X-Agent-ID": "scout"},
        )
        assert r.status_code == 200


@pytest.mark.asyncio
async def test_cancel_creator_or_assignee_or_operator(v2_app):
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/tasks",
            json={"assignee": "analyst", "title": "doomed"},
            headers={"X-Agent-ID": "scout"},
        )
        tid = r.json()["id"]
        # Outsider denied.
        r = await c.post(
            f"/mesh/tasks/{tid}/cancel",
            json={"reason": "no"},
            headers={"X-Agent-ID": "tracker"},
        )
        assert r.status_code == 403
        # Creator can cancel.
        r = await c.post(
            f"/mesh/tasks/{tid}/cancel",
            json={"reason": "user changed mind"},
            headers={"X-Agent-ID": "scout"},
        )
        assert r.status_code == 200
        assert r.json()["status"] == "cancelled"


@pytest.mark.asyncio
async def test_events_endpoint_returns_audit_history(v2_app):
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/tasks",
            json={"assignee": "analyst", "title": "audited"},
            headers={"X-Agent-ID": "scout"},
        )
        tid = r.json()["id"]
        await c.post(
            f"/mesh/tasks/{tid}/status",
            json={"status": "working"},
            headers={"X-Agent-ID": "analyst"},
        )
        r = await c.get(
            f"/mesh/tasks/{tid}/events",
            headers={"X-Agent-ID": "analyst"},
        )
        assert r.status_code == 200
        kinds = [e["event_kind"] for e in r.json()["events"]]
        assert kinds == ["created", "status_changed"]


# ── Task 9 — /mesh/pending/* endpoints ─────────────────────────────


@pytest.mark.asyncio
async def test_pending_list_endpoint_returns_open_nonces(v2_app):
    """``GET /mesh/pending`` returns rows from the SQLite store."""
    app, _, _ = v2_app
    pa = app.pending_actions
    pa.store(
        nonce="n-test", actor="operator", target_kind="agent",
        target_id="alpha", action_kind="model", payload={"x": 1},
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/mesh/pending", headers={"X-Mesh-Internal": "1"})
        assert r.status_code == 200
        rows = r.json()["pending"]
        assert any(p["nonce"] == "n-test" for p in rows)


@pytest.mark.asyncio
async def test_pending_cancel_endpoint_deletes_row(v2_app):
    """``POST /mesh/pending/{nonce}/cancel`` deletes the row."""
    app, _, _ = v2_app
    pa = app.pending_actions
    pa.store(
        nonce="n-cancel", actor="operator", target_kind="agent",
        target_id="alpha", action_kind="model", payload={"x": 1},
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/pending/n-cancel/cancel",
            headers={"X-Agent-ID": "operator", "X-Mesh-Internal": "1"},
        )
        assert r.status_code == 200
        assert r.json()["nonce"] == "n-cancel"
    # Row gone — peek confirms.
    assert pa.peek("n-cancel") is None


@pytest.mark.asyncio
async def test_pending_cancel_unknown_returns_404(v2_app):
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/pending/missing-nonce/cancel",
            headers={"X-Agent-ID": "operator", "X-Mesh-Internal": "1"},
        )
        assert r.status_code == 404
