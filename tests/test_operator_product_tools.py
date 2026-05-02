"""Tests for Task 7 operator product tools.

Covers the new operator surface (read tools + action tools), plus the
HTTP endpoint integration tests for project/agent archive + delete flows
and the cost-aware reroute/retry path.

The unit tests at the top mock ``mesh_client`` directly so they exercise
the tool wiring + flag gating without booting a mesh app. The endpoint
tests at the bottom build a real ``create_mesh_app`` and drive the
routes through ``ASGITransport``.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import yaml
from httpx import ASGITransport, AsyncClient

from src.host.mesh import Blackboard, MessageRouter, PubSub
from src.host.permissions import PermissionMatrix
from src.shared.types import AgentPermissions, MessageOrigin


@pytest.fixture(autouse=True)
def _set_operator_env(monkeypatch):
    monkeypatch.setenv("ALLOWED_TOOLS", "list_project_status,list_agent_queue")


# ── Helper: simulate an HTTP error wrapping an over_budget body ────


def _fake_budget_http_error(agent: str, monthly_used: float = 50.0) -> httpx.HTTPStatusError:
    """Build an ``HTTPStatusError`` carrying the structured budget detail."""
    request = httpx.Request("POST", "http://t/mesh/tasks/x/reroute")
    response = httpx.Response(
        400,
        request=request,
        json={
            "detail": (
                '{"error": "over_budget", '
                '"detail": "test", '
                '"budget": {"agent": "' + agent + '", "monthly_used": '
                + str(monthly_used) + '}}'
            ),
        },
    )
    return httpx.HTTPStatusError("400 Bad Request", request=request, response=response)


# ── v2 flag gate ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_project_status_requires_v2_flag(monkeypatch):
    """Read tools that consume tasks return a clean error when v2 off."""
    monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_V2", raising=False)
    from src.agent.builtins.operator_tools import list_project_status
    result = await list_project_status(mesh_client=MagicMock())
    assert "error" in result
    assert "OPENLEGION_ORCHESTRATION_TASKS_V2" in result["error"]


@pytest.mark.asyncio
async def test_list_agent_queue_requires_v2_flag(monkeypatch):
    monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_V2", raising=False)
    from src.agent.builtins.operator_tools import list_agent_queue
    result = await list_agent_queue("a1", mesh_client=MagicMock())
    assert "error" in result
    assert "OPENLEGION_ORCHESTRATION_TASKS_V2" in result["error"]


@pytest.mark.asyncio
async def test_get_team_outputs_requires_v2_flag(monkeypatch):
    monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_V2", raising=False)
    from src.agent.builtins.operator_tools import get_team_outputs
    result = await get_team_outputs("p1", mesh_client=MagicMock())
    assert "error" in result


@pytest.mark.asyncio
async def test_summarize_project_progress_requires_v2_flag(monkeypatch):
    monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_V2", raising=False)
    from src.agent.builtins.operator_tools import summarize_project_progress
    result = await summarize_project_progress("p1", mesh_client=MagicMock())
    assert "error" in result


@pytest.mark.asyncio
async def test_reroute_task_requires_v2_flag(monkeypatch):
    monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_V2", raising=False)
    from src.agent.builtins.operator_tools import reroute_task
    result = await reroute_task("t1", "writer", mesh_client=MagicMock())
    assert "error" in result


@pytest.mark.asyncio
async def test_cancel_task_requires_v2_flag(monkeypatch):
    monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_V2", raising=False)
    from src.agent.builtins.operator_tools import cancel_task
    result = await cancel_task("t1", mesh_client=MagicMock())
    assert "error" in result


@pytest.mark.asyncio
async def test_retry_failed_task_requires_v2_flag(monkeypatch):
    monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_V2", raising=False)
    from src.agent.builtins.operator_tools import retry_failed_task
    result = await retry_failed_task("t1", mesh_client=MagicMock())
    assert "error" in result


@pytest.mark.asyncio
async def test_archive_project_works_without_v2_flag(monkeypatch):
    """Archive/delete project/agent tools work regardless of the v2 flag."""
    monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_V2", raising=False)
    from src.agent.builtins.operator_tools import archive_project
    mc = MagicMock()
    mc.archive_project = AsyncMock(return_value={"archived": True, "project": "growth"})
    messages = [{"role": "user", "content": "yes", "_origin": "user"}]
    result = await archive_project("growth", mesh_client=mc, _messages=messages)
    assert result["archived"] is True


# ── Read tool happy path ──────────────────────────────────────


@pytest.mark.asyncio
async def test_list_project_status_all_projects(monkeypatch):
    monkeypatch.setenv("OPENLEGION_ORCHESTRATION_TASKS_V2", "1")
    from src.agent.builtins.operator_tools import list_project_status
    mc = MagicMock()
    mc.all_projects_status = AsyncMock(
        return_value={"projects": [
            {"project": {"name": "p1"}, "counts": {"active": 2}},
        ]},
    )
    result = await list_project_status(mesh_client=mc)
    assert "projects" in result
    mc.all_projects_status.assert_awaited_once()


@pytest.mark.asyncio
async def test_list_project_status_single_project(monkeypatch):
    monkeypatch.setenv("OPENLEGION_ORCHESTRATION_TASKS_V2", "1")
    from src.agent.builtins.operator_tools import list_project_status
    mc = MagicMock()
    mc.project_status = AsyncMock(
        return_value={"project": {"name": "growth"}, "counts": {"active": 1}},
    )
    result = await list_project_status("growth", mesh_client=mc)
    mc.project_status.assert_awaited_once_with("growth")
    assert result["project"]["name"] == "growth"


@pytest.mark.asyncio
async def test_list_agent_queue_calls_mesh(monkeypatch):
    monkeypatch.setenv("OPENLEGION_ORCHESTRATION_TASKS_V2", "1")
    from src.agent.builtins.operator_tools import list_agent_queue
    mc = MagicMock()
    mc.agent_queue = AsyncMock(
        return_value={"agent_id": "writer", "queue": {"active": []}},
    )
    result = await list_agent_queue("writer", limit=20, mesh_client=mc)
    mc.agent_queue.assert_awaited_once_with("writer", limit=20)
    assert result["agent_id"] == "writer"


@pytest.mark.asyncio
async def test_get_team_outputs_passes_since(monkeypatch):
    monkeypatch.setenv("OPENLEGION_ORCHESTRATION_TASKS_V2", "1")
    from src.agent.builtins.operator_tools import get_team_outputs
    mc = MagicMock()
    mc.project_outputs = AsyncMock(return_value={"outputs": []})
    await get_team_outputs("p1", since="24h", mesh_client=mc)
    mc.project_outputs.assert_awaited_once_with("p1", since="24h")


@pytest.mark.asyncio
async def test_summarize_project_progress_calls_mesh(monkeypatch):
    monkeypatch.setenv("OPENLEGION_ORCHESTRATION_TASKS_V2", "1")
    from src.agent.builtins.operator_tools import summarize_project_progress
    mc = MagicMock()
    mc.project_summary = AsyncMock(
        return_value={"status_text": "all good", "counts": {"active": 0}},
    )
    result = await summarize_project_progress("p1", mesh_client=mc)
    assert "status_text" in result


@pytest.mark.asyncio
async def test_get_agent_profile_calls_mesh():
    """get_agent_profile is not gated on v2 — calls /mesh/agents/{id}/profile."""
    from src.agent.builtins.operator_tools import get_agent_profile
    mc = MagicMock()
    mc.get_agent_profile = AsyncMock(return_value={"agent_id": "writer", "role": "writer"})
    result = await get_agent_profile("writer", mesh_client=mc)
    assert result["agent_id"] == "writer"


@pytest.mark.asyncio
async def test_get_agent_profile_returns_structured_routing_fields():
    """Task 8 — operator profile read surfaces the new structured routing
    fields (capabilities are tool list; the four siblings + the
    interface_capabilities field carry the human-routing data)."""
    from src.agent.builtins.operator_tools import get_agent_profile
    mc = MagicMock()
    mc.get_agent_profile = AsyncMock(return_value={
        "agent_id": "researcher",
        "role": "researcher",
        "capabilities": ["browser_navigate", "web_search"],  # tool list
        "interface_capabilities": ["Web research", "Synthesize findings"],
        "preferred_inputs": ["User questions"],
        "expected_outputs": ["Research reports"],
        "escalation_to": "operator",
        "forbidden": ["Speculative findings as fact"],
    })
    result = await get_agent_profile("researcher", mesh_client=mc)
    assert result["interface_capabilities"] == ["Web research", "Synthesize findings"]
    assert result["preferred_inputs"] == ["User questions"]
    assert result["expected_outputs"] == ["Research reports"]
    assert result["escalation_to"] == "operator"
    assert result["forbidden"] == ["Speculative findings as fact"]
    # Tool capabilities still distinct.
    assert result["capabilities"] == ["browser_navigate", "web_search"]


# ── Action tool: reroute_task with cost gate ──────────────────


@pytest.mark.asyncio
async def test_reroute_task_success(monkeypatch):
    monkeypatch.setenv("OPENLEGION_ORCHESTRATION_TASKS_V2", "1")
    from src.agent.builtins.operator_tools import reroute_task
    mc = MagicMock()
    mc.reroute_task = AsyncMock(
        return_value={"id": "task_1", "assignee": "writer", "status": "pending"},
    )
    result = await reroute_task("task_1", "writer", reason="capacity",
                                mesh_client=mc)
    mc.reroute_task.assert_awaited_once_with("task_1", "writer", reason="capacity")
    assert result["assignee"] == "writer"


@pytest.mark.asyncio
async def test_reroute_task_over_budget_returns_structured_error(monkeypatch):
    """Over-budget surface from the mesh wraps a structured payload."""
    monkeypatch.setenv("OPENLEGION_ORCHESTRATION_TASKS_V2", "1")
    from src.agent.builtins.operator_tools import reroute_task
    mc = MagicMock()
    mc.reroute_task = AsyncMock(side_effect=_fake_budget_http_error("writer"))
    result = await reroute_task("task_1", "writer", mesh_client=mc)
    assert result["error"] == "over_budget"
    assert "writer" in (result.get("budget") or {}).get("agent", "")


@pytest.mark.asyncio
async def test_cancel_task_success(monkeypatch):
    monkeypatch.setenv("OPENLEGION_ORCHESTRATION_TASKS_V2", "1")
    from src.agent.builtins.operator_tools import cancel_task
    mc = MagicMock()
    mc.cancel_task = AsyncMock(return_value={"id": "task_1", "status": "cancelled"})
    result = await cancel_task("task_1", reason="bad scope", mesh_client=mc)
    assert result["status"] == "cancelled"
    mc.cancel_task.assert_awaited_once_with("task_1", reason="bad scope")


@pytest.mark.asyncio
async def test_retry_failed_task_with_changes(monkeypatch):
    monkeypatch.setenv("OPENLEGION_ORCHESTRATION_TASKS_V2", "1")
    from src.agent.builtins.operator_tools import retry_failed_task
    mc = MagicMock()
    mc.retry_task = AsyncMock(
        return_value={"clone": {"id": "task_2"}, "original_id": "task_1"},
    )
    result = await retry_failed_task(
        "task_1", with_changes={"assignee": "scout", "title": "v2"},
        mesh_client=mc,
    )
    assert result["original_id"] == "task_1"
    mc.retry_task.assert_awaited_once_with(
        "task_1", title="v2", description=None, assignee="scout",
    )


@pytest.mark.asyncio
async def test_retry_failed_task_over_budget(monkeypatch):
    monkeypatch.setenv("OPENLEGION_ORCHESTRATION_TASKS_V2", "1")
    from src.agent.builtins.operator_tools import retry_failed_task
    mc = MagicMock()
    mc.retry_task = AsyncMock(side_effect=_fake_budget_http_error("scout"))
    result = await retry_failed_task("task_1", mesh_client=mc)
    assert result["error"] == "over_budget"
    assert (result.get("budget") or {}).get("agent") == "scout"


# ── Archive / delete tools ────────────────────────────────────


@pytest.mark.asyncio
async def test_archive_project_provenance_required():
    from src.agent.builtins.operator_tools import archive_project
    messages = [{"role": "user", "content": "x", "_origin": "system:heartbeat"}]
    result = await archive_project("p1", mesh_client=MagicMock(), _messages=messages)
    assert result["error"] == "provenance_check_failed"


@pytest.mark.asyncio
async def test_archive_project_success():
    from src.agent.builtins.operator_tools import archive_project
    mc = MagicMock()
    mc.archive_project = AsyncMock(return_value={"archived": True, "project": "p1"})
    messages = [{"role": "user", "content": "yes", "_origin": "user"}]
    result = await archive_project("p1", mesh_client=mc, _messages=messages)
    assert result["archived"] is True


@pytest.mark.asyncio
async def test_archive_agent_blocks_operator():
    from src.agent.builtins.operator_tools import archive_agent
    messages = [{"role": "user", "content": "yes", "_origin": "user"}]
    result = await archive_agent("operator", mesh_client=MagicMock(), _messages=messages)
    assert "operator" in result["error"].lower()


@pytest.mark.asyncio
async def test_archive_agent_success():
    from src.agent.builtins.operator_tools import archive_agent
    mc = MagicMock()
    mc.archive_agent = AsyncMock(return_value={"archived": True, "agent_id": "writer"})
    messages = [{"role": "user", "content": "yes", "_origin": "user"}]
    result = await archive_agent("writer", mesh_client=mc, _messages=messages)
    assert result["archived"] is True


@pytest.mark.asyncio
async def test_delete_project_returns_nonce_for_confirmation():
    from src.agent.builtins.operator_tools import delete_project
    mc = MagicMock()
    mc.propose_delete_project = AsyncMock(return_value={
        "change_id": "abc-123",
        "summary": "Delete project 'growth' and 2 agent(s).",
        "expires_at": "2026-05-02T00:15:00+00:00",
        "payload_digest": "deadbeef",
        "requires_confirmation": True,
    })
    messages = [{"role": "user", "content": "yes", "_origin": "user"}]
    result = await delete_project("growth", mesh_client=mc, _messages=messages)
    assert result["requires_confirmation"] is True
    assert result["change_id"] == "abc-123"
    assert "summary" in result


@pytest.mark.asyncio
async def test_delete_project_provenance_required():
    from src.agent.builtins.operator_tools import delete_project
    messages = [{"role": "user", "content": "hb", "_origin": "system:heartbeat"}]
    result = await delete_project("growth", mesh_client=MagicMock(), _messages=messages)
    assert result["error"] == "provenance_check_failed"


@pytest.mark.asyncio
async def test_delete_project_archive_required():
    """If the mesh rejects with 400, the tool surfaces a friendly hint."""
    from src.agent.builtins.operator_tools import delete_project
    mc = MagicMock()
    mc.propose_delete_project = AsyncMock(side_effect=RuntimeError("400: Project must be archived"))
    messages = [{"role": "user", "content": "yes", "_origin": "user"}]
    result = await delete_project("growth", mesh_client=mc, _messages=messages)
    assert result["error"] == "archive_required"


@pytest.mark.asyncio
async def test_delete_agent_blocks_operator():
    from src.agent.builtins.operator_tools import delete_agent
    messages = [{"role": "user", "content": "yes", "_origin": "user"}]
    result = await delete_agent("operator", mesh_client=MagicMock(), _messages=messages)
    assert "operator" in result["error"].lower()


@pytest.mark.asyncio
async def test_delete_agent_returns_nonce():
    from src.agent.builtins.operator_tools import delete_agent
    mc = MagicMock()
    mc.propose_delete_agent = AsyncMock(return_value={
        "change_id": "n1",
        "summary": "Delete agent 'writer'",
        "payload_digest": "abc",
        "expires_at": "2026-05-02T00:15:00+00:00",
        "requires_confirmation": True,
    })
    messages = [{"role": "user", "content": "yes", "_origin": "user"}]
    result = await delete_agent("writer", mesh_client=mc, _messages=messages)
    assert result["change_id"] == "n1"
    assert result["requires_confirmation"] is True


# ── HTTP endpoint integration tests ────────────────────────────


def _reload_server(monkeypatch, *, v2: bool, tasks_db: str):
    if v2:
        monkeypatch.setenv("OPENLEGION_ORCHESTRATION_TASKS_V2", "1")
        monkeypatch.setenv("OPENLEGION_ORCHESTRATION_TASKS_DB", tasks_db)
    else:
        monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_V2", raising=False)
    import src.host.server as server_module
    importlib.reload(server_module)
    return server_module


def _build_app(tmp_path, server_module, *, perms_map, agents=None,
               cost_tracker=None, container_manager=None):
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
        cost_tracker=cost_tracker,
        container_manager=container_manager,
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
        "status": "active",
    }))
    ops = pdir / "ops"
    ops.mkdir(parents=True)
    (ops / "metadata.yaml").write_text(yaml.dump({
        "name": "ops",
        "members": ["tracker"],
        "created_at": "2026-05-02T00:00:00+00:00",
        "status": "active",
    }))
    return pdir


def _agents_yaml(tmp_path, names: list[str]) -> Path:
    """Write a minimal agents.yaml with named agents."""
    (tmp_path / "config").mkdir(exist_ok=True)
    afile = tmp_path / "config" / "agents.yaml"
    cfg = {"agents": {name: {"role": name, "model": "gpt-4o-mini"} for name in names}}
    afile.write_text(yaml.dump(cfg))
    return afile


@pytest.fixture
def v2_app(tmp_path, monkeypatch):
    """Mesh app with v2 on, projects + agents on disk, operator routed."""
    # Pin the cli/config paths into tmp_path so PROJECTS_DIR + AGENTS_FILE
    # use the test layout.
    pdir = _projects_layout(tmp_path)
    afile = _agents_yaml(tmp_path, names=["scout", "analyst", "tracker"])
    monkeypatch.chdir(tmp_path)
    import src.cli.config as cli_cfg
    monkeypatch.setattr(cli_cfg, "PROJECTS_DIR", pdir)
    monkeypatch.setattr(cli_cfg, "AGENTS_FILE", afile)
    monkeypatch.setattr(cli_cfg, "PERMISSIONS_FILE", tmp_path / "config" / "permissions.json")

    server_module = _reload_server(
        monkeypatch, v2=True, tasks_db=str(tmp_path / "tasks.db"),
    )

    perms_map = {
        "operator": {"can_route_tasks": True, "can_manage_projects": True},
        "scout":    {"can_route_tasks": True, "can_message": ["analyst", "operator"]},
        "analyst":  {"can_route_tasks": False, "can_message": ["scout"]},
        "tracker":  {"can_route_tasks": False, "can_message": []},
    }
    # CostTracker stub: scout over budget, analyst within
    cost_tracker = MagicMock()
    cost_tracker.check_budget = MagicMock(side_effect=lambda agent: (
        {"allowed": False, "daily_used": 20.0, "daily_limit": 10.0,
         "monthly_used": 250.0, "monthly_limit": 200.0}
        if agent == "scout"
        else {"allowed": True, "daily_used": 1.0, "daily_limit": 10.0,
              "monthly_used": 5.0, "monthly_limit": 200.0}
    ))
    container_manager = MagicMock()
    container_manager.stop_agent = MagicMock(return_value=True)

    app, bb = _build_app(
        tmp_path, server_module,
        perms_map=perms_map,
        agents={
            "scout": "http://scout:8400",
            "analyst": "http://analyst:8400",
            "tracker": "http://tracker:8400",
            "operator": "http://operator:8400",
        },
        cost_tracker=cost_tracker,
        container_manager=container_manager,
    )
    yield app, server_module, tmp_path
    bb.close()
    monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_V2", raising=False)
    monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_DB", raising=False)
    importlib.reload(server_module)


def _human_origin_headers(agent_id: str = "operator", channel: str = "cli", user: str = "u1") -> dict:
    """Headers for a request originating from a paired/CLI human."""
    origin = MessageOrigin(kind="human", channel=channel, user=user)
    return {
        "X-Agent-ID": agent_id,
        "X-Origin": origin.to_header_value(),
    }


@pytest.mark.asyncio
async def test_endpoint_project_status_returns_counts(v2_app):
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        # Create a few tasks
        await c.post("/mesh/tasks",
                     json={"assignee": "analyst", "title": "t1", "project": "research"},
                     headers={"X-Agent-ID": "operator"})
        await c.post("/mesh/tasks",
                     json={"assignee": "scout", "title": "t2", "project": "research"},
                     headers={"X-Agent-ID": "operator"})
        r = await c.get("/mesh/projects/research/status",
                        headers={"X-Agent-ID": "operator"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["counts"]["active"] == 2
    assert body["project"]["name"] == "research"
    assert body["project"]["status"] == "active"


@pytest.mark.asyncio
async def test_endpoint_project_status_403_for_non_member(v2_app):
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/mesh/projects/research/status",
                        headers={"X-Agent-ID": "tracker"})
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_endpoint_all_projects_status_operator_sees_all(v2_app):
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/mesh/projects/status",
                        headers={"X-Agent-ID": "operator"})
    assert r.status_code == 200
    names = {p["project"]["name"] for p in r.json()["projects"]}
    assert names == {"research", "ops"}


@pytest.mark.asyncio
async def test_endpoint_agent_queue_buckets(v2_app):
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        # Create one task for analyst, then mark it done
        r = await c.post("/mesh/tasks",
                         json={"assignee": "analyst", "title": "t1", "project": "research"},
                         headers={"X-Agent-ID": "operator"})
        tid = r.json()["id"]
        await c.post(f"/mesh/tasks/{tid}/status",
                     json={"status": "working"},
                     headers={"X-Agent-ID": "analyst"})
        await c.post(f"/mesh/tasks/{tid}/status",
                     json={"status": "done"},
                     headers={"X-Agent-ID": "analyst"})
        r = await c.get("/mesh/agents/analyst/queue",
                        headers={"X-Agent-ID": "operator"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["agent_id"] == "analyst"
    assert len(body["queue"]["done"]) == 1
    assert len(body["queue"]["active"]) == 0


@pytest.mark.asyncio
async def test_endpoint_project_outputs_filters_by_since(v2_app):
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/mesh/tasks",
                         json={"assignee": "analyst", "title": "report", "project": "research"},
                         headers={"X-Agent-ID": "operator"})
        tid = r.json()["id"]
        await c.post(f"/mesh/tasks/{tid}/status",
                     json={"status": "working"},
                     headers={"X-Agent-ID": "analyst"})
        await c.post(f"/mesh/tasks/{tid}/status",
                     json={"status": "done"},
                     headers={"X-Agent-ID": "analyst"})
        # Last 24h includes the task
        r = await c.get("/mesh/projects/research/outputs",
                        params={"since": "24h"},
                        headers={"X-Agent-ID": "operator"})
    assert r.status_code == 200
    assert len(r.json()["outputs"]) == 1


@pytest.mark.asyncio
async def test_endpoint_project_summary(v2_app):
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/mesh/projects/research/summary",
                        headers={"X-Agent-ID": "operator"})
    assert r.status_code == 200
    body = r.json()
    assert "status_text" in body
    assert "counts" in body
    assert "ask_for_user" in body


@pytest.mark.asyncio
async def test_endpoint_reroute_blocks_when_target_over_budget(v2_app):
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/mesh/tasks",
                         json={"assignee": "analyst", "title": "t1", "project": "research"},
                         headers={"X-Agent-ID": "operator"})
        tid = r.json()["id"]
        # Reroute to scout (over budget) — should fail with structured error
        r = await c.post(f"/mesh/tasks/{tid}/reroute",
                         json={"new_assignee": "scout"},
                         headers={"X-Agent-ID": "operator"})
    assert r.status_code == 400
    body = r.json()
    detail = body.get("detail")
    if isinstance(detail, str):
        import json as _json
        detail = _json.loads(detail)
    assert detail["error"] == "over_budget"
    assert detail["budget"]["agent"] == "scout"


@pytest.mark.asyncio
async def test_endpoint_retry_failed_task_clones(v2_app):
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/mesh/tasks",
                         json={"assignee": "analyst", "title": "t1", "project": "research"},
                         headers={"X-Agent-ID": "operator"})
        tid = r.json()["id"]
        await c.post(f"/mesh/tasks/{tid}/status",
                     json={"status": "working"},
                     headers={"X-Agent-ID": "analyst"})
        await c.post(f"/mesh/tasks/{tid}/status",
                     json={"status": "failed"},
                     headers={"X-Agent-ID": "analyst"})
        r = await c.post(f"/mesh/tasks/{tid}/retry",
                         json={"title": "t1 redux"},
                         headers={"X-Agent-ID": "operator"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["original_id"] == tid
    assert body["clone"]["status"] == "pending"
    assert body["clone"]["title"] == "t1 redux"


@pytest.mark.asyncio
async def test_endpoint_retry_only_failed(v2_app):
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/mesh/tasks",
                         json={"assignee": "analyst", "title": "t1", "project": "research"},
                         headers={"X-Agent-ID": "operator"})
        tid = r.json()["id"]
        # Task is still pending — retry must refuse
        r = await c.post(f"/mesh/tasks/{tid}/retry",
                         headers={"X-Agent-ID": "operator"})
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_endpoint_archive_project(v2_app):
    app, _, tmp_path = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/mesh/projects/research/archive",
                         headers={"X-Agent-ID": "operator"})
    assert r.status_code == 200
    # Default list should now exclude archived
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/mesh/projects",
                        headers={"X-Agent-ID": "operator"})
    body = r.json()
    names = {p["name"] for p in body["projects"]}
    assert "research" not in names
    assert "ops" in names


@pytest.mark.asyncio
async def test_endpoint_delete_project_requires_archive(v2_app):
    """Delete on a live project must be rejected with 400."""
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/mesh/projects/research/propose-delete",
                         headers=_human_origin_headers())
    assert r.status_code == 400
    assert "archived" in r.text.lower()


@pytest.mark.asyncio
async def test_endpoint_delete_project_happy_path(v2_app):
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        # Archive first
        r = await c.post("/mesh/projects/research/archive",
                         headers={"X-Agent-ID": "operator"})
        assert r.status_code == 200
        # Propose delete
        r = await c.post("/mesh/projects/research/propose-delete",
                         headers=_human_origin_headers())
        assert r.status_code == 200, r.text
        nonce = r.json()["change_id"]
        digest = r.json()["payload_digest"]
        # Confirm with human origin succeeds
        r = await c.post("/mesh/config/confirm",
                         json={"change_id": nonce, "payload_digest": digest},
                         headers=_human_origin_headers())
    assert r.status_code == 200
    body = r.json()
    assert body["deleted"] == "project"
    assert body["name"] == "research"


@pytest.mark.asyncio
async def test_endpoint_delete_project_confirm_with_agent_origin_403(v2_app):
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        await c.post("/mesh/projects/research/archive",
                     headers={"X-Agent-ID": "operator"})
        r = await c.post("/mesh/projects/research/propose-delete",
                         headers=_human_origin_headers())
        nonce = r.json()["change_id"]
        digest = r.json()["payload_digest"]
        # Agent origin → 403 from the confirm-side gate
        agent_origin = MessageOrigin(kind="agent", channel="", user="").to_header_value()
        r = await c.post("/mesh/config/confirm",
                         json={"change_id": nonce, "payload_digest": digest},
                         headers={"X-Agent-ID": "operator", "X-Origin": agent_origin})
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_endpoint_delete_project_expired_or_unknown(v2_app):
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/mesh/config/confirm",
                         json={"change_id": "does-not-exist"},
                         headers=_human_origin_headers())
    assert r.status_code == 400
    assert "invalid or expired" in r.text.lower()


@pytest.mark.asyncio
async def test_endpoint_archive_agent_then_delete(v2_app):
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        # Archive scout
        r = await c.post("/mesh/agents/scout/archive",
                         headers={"X-Agent-ID": "operator"})
        assert r.status_code == 200
        # Propose delete
        r = await c.post("/mesh/agents/scout/propose-delete",
                         headers=_human_origin_headers())
        assert r.status_code == 200, r.text
        nonce = r.json()["change_id"]
        digest = r.json()["payload_digest"]
        # Confirm
        r = await c.post("/mesh/config/confirm",
                         json={"change_id": nonce, "payload_digest": digest},
                         headers=_human_origin_headers())
    assert r.status_code == 200
    body = r.json()
    assert body["deleted"] == "agent"


@pytest.mark.asyncio
async def test_endpoint_delete_agent_requires_archive_first(v2_app):
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/mesh/agents/scout/propose-delete",
                         headers=_human_origin_headers())
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_endpoint_delete_agent_blocks_operator(v2_app):
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/mesh/agents/operator/propose-delete",
                         headers=_human_origin_headers())
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_endpoint_archive_agent_blocks_operator(v2_app):
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/mesh/agents/operator/archive",
                         headers={"X-Agent-ID": "operator"})
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_endpoint_list_projects_includes_archived_when_flagged(v2_app):
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        await c.post("/mesh/projects/research/archive",
                     headers={"X-Agent-ID": "operator"})
        r = await c.get("/mesh/projects",
                        params={"include_archived": True},
                        headers={"X-Agent-ID": "operator"})
    body = r.json()
    names = {p["name"] for p in body["projects"]}
    assert "research" in names
