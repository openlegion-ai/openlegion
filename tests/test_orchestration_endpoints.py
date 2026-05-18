"""HTTP endpoint tests for orchestration tasks v2 (Task 6).

Boots the mesh app, drives the ``/mesh/tasks*`` routes through
``ASGITransport``, and verifies the permission gates and 503
fall-through.
"""

from __future__ import annotations

import importlib

import pytest
import yaml
from httpx import ASGITransport, AsyncClient

from src.host.mesh import Blackboard, MessageRouter, PubSub
from src.host.permissions import PermissionMatrix
from src.shared.types import AgentPermissions


def _reload_server(monkeypatch, *, tasks_db: str):
    """Reload ``src.host.server`` after pinning the tasks DB path.

    The tasks-store DB path is read at module import via env var so a
    test that wants to point it at ``tmp_path`` must reload the module
    after ``setenv``.
    """
    monkeypatch.setenv("OPENLEGION_ORCHESTRATION_TASKS_DB", tasks_db)
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
        monkeypatch, tasks_db=str(tmp_path / "tasks.db"),
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
    monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_DB", raising=False)
    importlib.reload(server_module)


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
            "/mesh/tasks/team/research",
            headers={"X-Agent-ID": "scout"},
        )
        assert r.status_code == 200
        assert r.json()["count"] == 1
        # Non-member is denied.
        r = await c.get(
            "/mesh/tasks/team/research",
            headers={"X-Agent-ID": "tracker"},
        )
        assert r.status_code == 403
        # Operator gets through.
        r = await c.get(
            "/mesh/tasks/team/research",
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


# ── Hotfix: _extract_verified_agent_id honors internal callers ──────
#
# In production (auth tokens configured), the dashboard cancel proxy
# hits ``/mesh/tasks/{id}/cancel`` over loopback with
# ``x-mesh-internal: 1`` + ``X-Agent-ID: operator`` but NO Bearer token
# (Bearer tokens are server-side secrets the dashboard doesn't have).
# The previous ``_extract_verified_agent_id`` rejected those callers
# with 401 "Missing authentication token". Trust the loopback boundary
# instead.


@pytest.fixture
def v2_app_with_auth(tmp_path, monkeypatch):
    """v2 mesh app with auth tokens configured (production-like)."""
    server_module = _reload_server(
        monkeypatch, tasks_db=str(tmp_path / "tasks.db"),
    )
    perms_map = {
        "operator": {"can_route_tasks": True},
        "scout":    {"can_route_tasks": True, "can_message": ["analyst", "operator"]},
        "analyst":  {"can_route_tasks": False, "can_message": ["scout"]},
    }
    blackboard = Blackboard(str(tmp_path / "bb.db"))
    pubsub = PubSub()
    permissions = PermissionMatrix()
    for aid, perms in perms_map.items():
        permissions.permissions[aid] = AgentPermissions(agent_id=aid, **perms)
    router = MessageRouter(permissions, {
        "scout": "http://scout:8400",
        "analyst": "http://analyst:8400",
        "operator": "http://operator:8400",
    })
    app = server_module.create_mesh_app(
        blackboard=blackboard,
        pubsub=pubsub,
        router=router,
        permissions=permissions,
        # Auth tokens configured — public callers must Bearer-auth.
        auth_tokens={
            "operator": "tok_operator",
            "scout":    "tok_scout",
            "analyst":  "tok_analyst",
        },
    )
    yield app, server_module
    blackboard.close()
    monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_DB", raising=False)
    importlib.reload(server_module)


@pytest.mark.asyncio
async def test_internal_caller_skips_bearer_requirement(v2_app_with_auth):
    """Loopback + ``x-mesh-internal: 1`` + ``X-Agent-ID`` succeeds
    without a Bearer token. This is the dashboard cancel proxy path.
    """
    app, _ = v2_app_with_auth
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        # Internal caller creates a task on behalf of operator…
        r = await c.post(
            "/mesh/tasks",
            json={"assignee": "analyst", "title": "internal-create"},
            headers={"x-mesh-internal": "1", "X-Agent-ID": "operator"},
        )
        assert r.status_code == 200, r.text
        tid = r.json()["id"]
        # …and then cancels it the same way (no Bearer token).
        r = await c.post(
            f"/mesh/tasks/{tid}/cancel",
            json={"reason": "ops decided no"},
            headers={"x-mesh-internal": "1", "X-Agent-ID": "operator"},
        )
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "cancelled"


@pytest.mark.asyncio
async def test_cancel_task_via_dashboard_proxy_succeeds_in_production_auth_mode(
    v2_app_with_auth,
):
    """Regression for the kanban Cancel button. With auth tokens
    configured, a loopback + ``x-mesh-internal`` POST to
    ``/mesh/tasks/{id}/cancel`` (no Bearer) must NOT 401.
    """
    app, _ = v2_app_with_auth
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        # Seed via a Bearer-authed worker so the task exists with a
        # creator we can verify the cancel against.
        r = await c.post(
            "/mesh/tasks",
            json={"assignee": "analyst", "title": "kanban-cancel-target"},
            headers={
                "X-Agent-ID": "scout",
                "Authorization": "Bearer tok_scout",
            },
        )
        assert r.status_code == 200, r.text
        tid = r.json()["id"]
        # Dashboard proxies the cancel: loopback + x-mesh-internal,
        # NO Authorization header.
        r = await c.post(
            f"/mesh/tasks/{tid}/cancel",
            json={"reason": "user clicked Cancel"},
            headers={"x-mesh-internal": "1", "X-Agent-ID": "operator"},
        )
        assert r.status_code != 401, (
            "Dashboard cancel proxy must not require Bearer when caller "
            "is loopback + x-mesh-internal"
        )
        assert r.status_code == 200, r.text


@pytest.mark.asyncio
async def test_public_caller_without_bearer_still_rejected(v2_app_with_auth):
    """The fix only exempts internal callers — a public caller
    without a Bearer token still gets 401.
    """
    app, _ = v2_app_with_auth
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/tasks",
            json={"assignee": "analyst", "title": "public-no-bearer"},
            headers={"X-Agent-ID": "scout"},
        )
        assert r.status_code == 401


@pytest.mark.asyncio
async def test_public_caller_with_valid_bearer_succeeds(v2_app_with_auth):
    """Sanity check: existing Bearer auth path still works."""
    app, _ = v2_app_with_auth
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/tasks",
            json={"assignee": "analyst", "title": "with-bearer"},
            headers={
                "X-Agent-ID": "scout",
                "Authorization": "Bearer tok_scout",
            },
        )
        assert r.status_code == 200, r.text
