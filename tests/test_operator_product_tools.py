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
    monkeypatch.setenv("ALLOWED_TOOLS", "inspect_projects,list_agent_queue")


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
async def test_inspect_projects_status_requires_v2_flag(monkeypatch):
    """Read tools that consume tasks return a clean error when v2 off.

    The flag now defaults to ``1`` (rollout) so the off path must
    explicitly set ``OPENLEGION_ORCHESTRATION_TASKS_V2=0`` rather than
    relying on the env var being unset.
    """
    monkeypatch.setenv("OPENLEGION_ORCHESTRATION_TASKS_V2", "0")
    from src.agent.builtins.operator_tools import inspect_projects
    result = await inspect_projects(detail="status", mesh_client=MagicMock())
    assert "error" in result
    assert "OPENLEGION_ORCHESTRATION_TASKS_V2" in result["error"]


@pytest.mark.asyncio
async def test_list_agent_queue_requires_v2_flag(monkeypatch):
    monkeypatch.setenv("OPENLEGION_ORCHESTRATION_TASKS_V2", "0")
    from src.agent.builtins.operator_tools import list_agent_queue
    result = await list_agent_queue("a1", mesh_client=MagicMock())
    assert "error" in result
    assert "OPENLEGION_ORCHESTRATION_TASKS_V2" in result["error"]


@pytest.mark.asyncio
async def test_get_team_outputs_requires_v2_flag(monkeypatch):
    monkeypatch.setenv("OPENLEGION_ORCHESTRATION_TASKS_V2", "0")
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
async def test_manage_task_reroute_requires_v2_flag(monkeypatch):
    monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_V2", raising=False)
    from src.agent.builtins.operator_tools import manage_task
    result = await manage_task("t1", "reroute", new_assignee="writer",
                                mesh_client=MagicMock())
    assert "error" in result


@pytest.mark.asyncio
async def test_manage_task_cancel_requires_v2_flag(monkeypatch):
    monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_V2", raising=False)
    from src.agent.builtins.operator_tools import manage_task
    result = await manage_task("t1", "cancel", mesh_client=MagicMock())
    assert "error" in result


@pytest.mark.asyncio
async def test_manage_task_retry_requires_v2_flag(monkeypatch):
    monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_V2", raising=False)
    from src.agent.builtins.operator_tools import manage_task
    result = await manage_task("t1", "retry", mesh_client=MagicMock())
    assert "error" in result


@pytest.mark.asyncio
async def test_manage_project_archive_works_without_v2_flag(monkeypatch):
    """Archive/delete project tools work regardless of the v2 flag."""
    monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_V2", raising=False)
    from src.agent.builtins.operator_tools import manage_project
    mc = MagicMock()
    mc.archive_project = AsyncMock(return_value={"archived": True, "project": "growth"})
    messages = [{"role": "user", "content": "yes", "_origin": "user"}]
    result = await manage_project("growth", "archive",
                                   mesh_client=mc, _messages=messages)
    assert result["archived"] is True


# ── Read tool happy path ──────────────────────────────────────


@pytest.mark.asyncio
async def test_inspect_projects_status_all_projects(monkeypatch):
    monkeypatch.setenv("OPENLEGION_ORCHESTRATION_TASKS_V2", "1")
    from src.agent.builtins.operator_tools import inspect_projects
    mc = MagicMock()
    mc.all_projects_status = AsyncMock(
        return_value={"projects": [
            {"project": {"name": "p1"}, "counts": {"active": 2}},
        ]},
    )
    result = await inspect_projects(detail="status", mesh_client=mc)
    assert "projects" in result
    mc.all_projects_status.assert_awaited_once()


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
async def test_inspect_agents_profile_calls_mesh():
    """inspect_agents(depth='profile') calls /mesh/agents/{id}/profile."""
    from src.agent.builtins.operator_tools import inspect_agents
    mc = MagicMock()
    mc.get_agent_profile = AsyncMock(return_value={"agent_id": "writer", "role": "writer"})
    result = await inspect_agents("writer", depth="profile", mesh_client=mc)
    assert result["agent_id"] == "writer"


@pytest.mark.asyncio
async def test_inspect_agents_profile_returns_structured_routing_fields():
    """Task 8 — operator profile read surfaces the new structured routing
    fields (capabilities are tool list; the four siblings + the
    interface_capabilities field carry the human-routing data)."""
    from src.agent.builtins.operator_tools import inspect_agents
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
    result = await inspect_agents("researcher", depth="profile", mesh_client=mc)
    assert result["interface_capabilities"] == ["Web research", "Synthesize findings"]
    assert result["preferred_inputs"] == ["User questions"]
    assert result["expected_outputs"] == ["Research reports"]
    assert result["escalation_to"] == "operator"
    assert result["forbidden"] == ["Speculative findings as fact"]
    # Tool capabilities still distinct.
    assert result["capabilities"] == ["browser_navigate", "web_search"]


# ── Action tool: manage_task with cost gate ──────────────────


@pytest.mark.asyncio
async def test_manage_task_reroute_success(monkeypatch):
    monkeypatch.setenv("OPENLEGION_ORCHESTRATION_TASKS_V2", "1")
    from src.agent.builtins.operator_tools import manage_task
    mc = MagicMock()
    mc.reroute_task = AsyncMock(
        return_value={"id": "task_1", "assignee": "writer", "status": "pending"},
    )
    result = await manage_task("task_1", "reroute", new_assignee="writer",
                                reason="capacity", mesh_client=mc)
    mc.reroute_task.assert_awaited_once_with("task_1", "writer", reason="capacity")
    assert result["assignee"] == "writer"


@pytest.mark.asyncio
async def test_manage_task_reroute_requires_new_assignee(monkeypatch):
    """`reroute` action without new_assignee returns a clear error."""
    monkeypatch.setenv("OPENLEGION_ORCHESTRATION_TASKS_V2", "1")
    from src.agent.builtins.operator_tools import manage_task
    result = await manage_task("task_1", "reroute", mesh_client=MagicMock())
    assert "error" in result
    assert "new_assignee" in result["error"]


@pytest.mark.asyncio
async def test_manage_task_reroute_over_budget_returns_structured_error(monkeypatch):
    """Over-budget surface from the mesh wraps a structured payload."""
    monkeypatch.setenv("OPENLEGION_ORCHESTRATION_TASKS_V2", "1")
    from src.agent.builtins.operator_tools import manage_task
    mc = MagicMock()
    mc.reroute_task = AsyncMock(side_effect=_fake_budget_http_error("writer"))
    result = await manage_task("task_1", "reroute", new_assignee="writer",
                                mesh_client=mc)
    assert result["error"] == "over_budget"
    assert "writer" in (result.get("budget") or {}).get("agent", "")


@pytest.mark.asyncio
async def test_manage_task_cancel_success(monkeypatch):
    monkeypatch.setenv("OPENLEGION_ORCHESTRATION_TASKS_V2", "1")
    from src.agent.builtins.operator_tools import manage_task
    mc = MagicMock()
    mc.cancel_task = AsyncMock(return_value={"id": "task_1", "status": "cancelled"})
    result = await manage_task("task_1", "cancel", reason="bad scope",
                                mesh_client=mc)
    assert result["status"] == "cancelled"
    mc.cancel_task.assert_awaited_once_with("task_1", reason="bad scope")


@pytest.mark.asyncio
async def test_manage_task_retry_with_changes(monkeypatch):
    monkeypatch.setenv("OPENLEGION_ORCHESTRATION_TASKS_V2", "1")
    from src.agent.builtins.operator_tools import manage_task
    mc = MagicMock()
    mc.retry_task = AsyncMock(
        return_value={"clone": {"id": "task_2"}, "original_id": "task_1"},
    )
    result = await manage_task(
        "task_1", "retry",
        with_changes={"assignee": "scout", "title": "v2"},
        mesh_client=mc,
    )
    assert result["original_id"] == "task_1"
    mc.retry_task.assert_awaited_once_with(
        "task_1", title="v2", description=None, assignee="scout",
    )


@pytest.mark.asyncio
async def test_manage_task_retry_over_budget(monkeypatch):
    monkeypatch.setenv("OPENLEGION_ORCHESTRATION_TASKS_V2", "1")
    from src.agent.builtins.operator_tools import manage_task
    mc = MagicMock()
    mc.retry_task = AsyncMock(side_effect=_fake_budget_http_error("scout"))
    result = await manage_task("task_1", "retry", mesh_client=mc)
    assert result["error"] == "over_budget"
    assert (result.get("budget") or {}).get("agent") == "scout"


# ── Archive / delete tools ────────────────────────────────────


@pytest.mark.asyncio
async def test_manage_project_archive_autonomous_allowed():
    """The provenance gate was dropped — operator can archive a project
    autonomously (e.g. during heartbeat). The "undo" comes from
    archive being reversible via the dedicated unarchive endpoint."""
    from src.agent.builtins.operator_tools import manage_project
    mc = MagicMock()
    mc.archive_project = AsyncMock(return_value={"archived": True, "project": "p1"})
    messages = [{"role": "user", "content": "x", "_origin": "system:heartbeat"}]
    result = await manage_project("p1", "archive",
                                    mesh_client=mc, _messages=messages)
    assert result["archived"] is True


@pytest.mark.asyncio
async def test_manage_project_archive_success():
    from src.agent.builtins.operator_tools import manage_project
    mc = MagicMock()
    mc.archive_project = AsyncMock(return_value={"archived": True, "project": "p1"})
    messages = [{"role": "user", "content": "yes", "_origin": "user"}]
    result = await manage_project("p1", "archive",
                                    mesh_client=mc, _messages=messages)
    assert result["archived"] is True


@pytest.mark.asyncio
async def test_manage_agent_archive_blocks_operator():
    from src.agent.builtins.operator_tools import manage_agent
    messages = [{"role": "user", "content": "yes", "_origin": "user"}]
    result = await manage_agent("operator", "archive",
                                  mesh_client=MagicMock(), _messages=messages)
    assert "operator" in result["error"].lower()


@pytest.mark.asyncio
async def test_manage_agent_archive_success():
    from src.agent.builtins.operator_tools import manage_agent
    mc = MagicMock()
    mc.archive_agent = AsyncMock(return_value={"archived": True, "agent_id": "writer"})
    messages = [{"role": "user", "content": "yes", "_origin": "user"}]
    result = await manage_agent("writer", "archive",
                                  mesh_client=mc, _messages=messages)
    assert result["archived"] is True


@pytest.mark.asyncio
async def test_manage_project_delete_returns_nonce_for_confirmation():
    from src.agent.builtins.operator_tools import manage_project
    mc = MagicMock()
    mc.propose_delete_project = AsyncMock(return_value={
        "change_id": "abc-123",
        "summary": "Delete project 'growth' and 2 agent(s).",
        "expires_at": "2026-05-02T00:15:00+00:00",
        "payload_digest": "deadbeef",
        "requires_confirmation": True,
    })
    messages = [{"role": "user", "content": "yes", "_origin": "user"}]
    result = await manage_project("growth", "delete",
                                    mesh_client=mc, _messages=messages)
    assert result["requires_confirmation"] is True
    assert result["change_id"] == "abc-123"
    assert "summary" in result


@pytest.mark.asyncio
async def test_manage_project_delete_autonomous_allowed():
    """Delete provenance gate was dropped at the operator-tool layer.
    Delete still has a mesh-side confirmation TTL window so the brief
    Confirm step in the dashboard remains; that's tested separately by
    the propose-delete endpoint tests. The operator-tool itself just
    forwards the call without checking message origin."""
    from src.agent.builtins.operator_tools import manage_project
    mc = MagicMock()
    mc.propose_delete_project = AsyncMock(return_value={
        "change_id": "n1",
        "summary": "Delete project 'growth'",
        "requires_confirmation": True,
    })
    messages = [{"role": "user", "content": "hb", "_origin": "system:heartbeat"}]
    result = await manage_project("growth", "delete",
                                    mesh_client=mc, _messages=messages)
    assert result["change_id"] == "n1"
    assert result["requires_confirmation"] is True


@pytest.mark.asyncio
async def test_manage_project_delete_archive_required():
    """If the mesh rejects with 400, the tool surfaces a friendly hint."""
    from src.agent.builtins.operator_tools import manage_project
    mc = MagicMock()
    mc.propose_delete_project = AsyncMock(side_effect=RuntimeError("400: Project must be archived"))
    messages = [{"role": "user", "content": "yes", "_origin": "user"}]
    result = await manage_project("growth", "delete",
                                    mesh_client=mc, _messages=messages)
    assert result["error"] == "archive_required"


@pytest.mark.asyncio
async def test_manage_agent_delete_blocks_operator():
    from src.agent.builtins.operator_tools import manage_agent
    messages = [{"role": "user", "content": "yes", "_origin": "user"}]
    result = await manage_agent("operator", "delete",
                                  mesh_client=MagicMock(), _messages=messages)
    assert "operator" in result["error"].lower()


@pytest.mark.asyncio
async def test_manage_agent_delete_returns_nonce():
    from src.agent.builtins.operator_tools import manage_agent
    mc = MagicMock()
    mc.propose_delete_agent = AsyncMock(return_value={
        "change_id": "n1",
        "summary": "Delete agent 'writer'",
        "payload_digest": "abc",
        "expires_at": "2026-05-02T00:15:00+00:00",
        "requires_confirmation": True,
    })
    messages = [{"role": "user", "content": "yes", "_origin": "user"}]
    result = await manage_agent("writer", "delete",
                                  mesh_client=mc, _messages=messages)
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
               cost_tracker=None, container_manager=None, event_bus=None):
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
        event_bus=event_bus,
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
        body = r.json()
        nonce = body["change_id"]
        digest = body["payload_digest"]
        # PR #2: response carries the human-readable summary so the
        # inline pending-action card can render without a follow-up
        # round-trip.
        assert "delete project" in body["summary"].lower()
        assert "'research'" in body["summary"]
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
        body = r.json()
        nonce = body["change_id"]
        digest = body["payload_digest"]
        # PR #2: agent-delete summary is short and names the target
        # so the inline pending-action card is self-describing.
        assert "delete agent" in body["summary"].lower()
        assert "'scout'" in body["summary"]
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


# ── PR — close EventBus coverage gaps ─────────────────────────────────


@pytest.fixture
def v2_app_with_bus(tmp_path, monkeypatch):
    """Variant of ``v2_app`` that wires a real EventBus so tests can
    assert which events fire on archive/unarchive/project-CRUD/blackboard
    delete endpoints. Yields ``(app, server_module, tmp_path, bus)``."""
    from src.dashboard.events import EventBus

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
    bus = EventBus()
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
        event_bus=bus,
    )
    yield app, server_module, tmp_path, bus
    bb.close()
    monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_V2", raising=False)
    monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_DB", raising=False)
    importlib.reload(server_module)


def _capture(bus) -> list[dict]:
    captured: list[dict] = []
    bus.add_listener(lambda e: captured.append(e))
    return captured


@pytest.mark.asyncio
async def test_archive_agent_emits_agent_archived(v2_app_with_bus):
    app, _, _, bus = v2_app_with_bus
    captured = _capture(bus)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/mesh/agents/scout/archive",
                         headers={"X-Agent-ID": "operator"})
    assert r.status_code == 200
    archived = [e for e in captured if e["type"] == "agent_archived"]
    assert len(archived) == 1
    assert archived[0]["agent"] == "scout"
    assert archived[0]["data"]["agent_id"] == "scout"


@pytest.mark.asyncio
async def test_unarchive_agent_emits_agent_unarchived(v2_app_with_bus):
    app, _, _, bus = v2_app_with_bus
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        await c.post("/mesh/agents/scout/archive",
                     headers={"X-Agent-ID": "operator"})
        captured = _capture(bus)  # capture only post-archive events
        r = await c.post("/mesh/agents/scout/unarchive",
                         headers={"X-Agent-ID": "operator"})
    assert r.status_code == 200
    unarchived = [e for e in captured if e["type"] == "agent_unarchived"]
    assert len(unarchived) == 1
    assert unarchived[0]["data"]["agent_id"] == "scout"


@pytest.mark.asyncio
async def test_archive_project_emits_project_archived(v2_app_with_bus):
    app, _, _, bus = v2_app_with_bus
    captured = _capture(bus)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/mesh/projects/research/archive",
                         headers={"X-Agent-ID": "operator"})
    assert r.status_code == 200
    arch = [e for e in captured if e["type"] == "project_archived"]
    assert len(arch) == 1
    assert arch[0]["data"]["project_id"] == "research"


@pytest.mark.asyncio
async def test_unarchive_project_emits_project_unarchived(v2_app_with_bus):
    app, _, _, bus = v2_app_with_bus
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        await c.post("/mesh/projects/research/archive",
                     headers={"X-Agent-ID": "operator"})
        captured = _capture(bus)
        r = await c.post("/mesh/projects/research/unarchive",
                         headers={"X-Agent-ID": "operator"})
    assert r.status_code == 200
    unarch = [e for e in captured if e["type"] == "project_unarchived"]
    assert len(unarch) == 1
    assert unarch[0]["data"]["project_id"] == "research"


# NOTE: ``mesh_create_project`` / ``mesh_delete_project`` use
# ``_resolve_agent_id("", request)`` which only consults the bearer
# token when ``_auth_tokens`` is configured. The v2 fixture runs
# without auth tokens (dev/test mode) so the endpoint returns
# ``Only the operator can manage projects`` regardless of the
# ``X-Agent-ID`` header. The equivalent dashboard endpoint —
# ``POST /api/projects`` / ``DELETE /api/projects/{name}`` — is
# covered in ``test_dashboard.py::TestDashboardEventBusCoverage`` and
# uses the same ``_create_project`` / ``_delete_project`` helpers, so
# the emit logic is exercised there.


@pytest.mark.asyncio
async def test_mesh_set_project_goal_emits_project_updated(v2_app_with_bus):
    app, _, _, bus = v2_app_with_bus
    captured = _capture(bus)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/projects/research/goal",
            json={"north_star": "Win this quarter", "success_criteria": ["x"]},
            headers={"X-Agent-ID": "operator"},
        )
    assert r.status_code == 200, r.text
    updated = [e for e in captured if e["type"] == "project_updated"]
    assert len(updated) == 1
    assert updated[0]["data"]["field"] == "goal"
    assert updated[0]["data"]["project_id"] == "research"


@pytest.mark.asyncio
async def test_blackboard_delete_emits_blackboard_delete(v2_app_with_bus):
    """The blackboard delete endpoint emits ``blackboard_delete`` so the
    SPA can drop the entry from the viewer without a polling round-trip.
    Mirrors the existing ``blackboard_write`` emit on the write path."""
    app, _, _, bus = v2_app_with_bus
    # Add a permission for "scout" to write/read its own keys, since
    # the v2_app perms_map doesn't grant blackboard ACL by default.
    # We seed the entry via the Blackboard directly to bypass the
    # write endpoint's ACL gate, then exercise the delete endpoint.
    from src.host.permissions import AgentPermissions
    # Use the permissions matrix on the app's router. We pull it
    # out of the closure on a registered endpoint since the v2_app
    # fixture doesn't expose the matrix directly.
    perms = None
    for route in app.routes:
        endpoint = getattr(route, "endpoint", None)
        if endpoint is None:
            continue
        # The closure captures ``permissions``; expose via an
        # attribute so we can flip the ACL for this test only.
        closure = getattr(endpoint, "__closure__", None) or ()
        names = getattr(getattr(endpoint, "__code__", None), "co_freevars", ())
        for name, cell in zip(names, closure):
            if name == "permissions":
                perms = cell.cell_contents
                break
        if perms is not None:
            break
    assert perms is not None, "could not locate permission matrix on app"
    perms.permissions["operator"] = AgentPermissions(
        agent_id="operator",
        can_route_tasks=True,
        can_manage_projects=True,
        blackboard_write=["*"],
    )

    # Seed the entry directly via Blackboard (bypassing the write
    # endpoint ACL gate isn't what we're testing here — the delete
    # emit is).
    blackboard = None
    for route in app.routes:
        endpoint = getattr(route, "endpoint", None)
        if endpoint is None:
            continue
        closure = getattr(endpoint, "__closure__", None) or ()
        names = getattr(getattr(endpoint, "__code__", None), "co_freevars", ())
        for name, cell in zip(names, closure):
            if name == "blackboard":
                blackboard = cell.cell_contents
                break
        if blackboard is not None:
            break
    assert blackboard is not None
    blackboard.write("foo/bar", {"hi": 1}, written_by="operator")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        captured = _capture(bus)
        r = await c.request(
            "DELETE", "/mesh/blackboard/foo/bar",
            params={"agent_id": "operator"},
            headers={"X-Agent-ID": "operator"},
        )
    assert r.status_code == 200, r.text
    deletes = [e for e in captured if e["type"] == "blackboard_delete"]
    assert len(deletes) == 1
    assert deletes[0]["data"]["key"] == "foo/bar"
    assert deletes[0]["data"]["deleted_by"] == "operator"


@pytest.mark.asyncio
async def test_soft_edit_does_not_emit_agent_config_updated(v2_app_with_bus):
    """Soft edits ride on ``operator_action_receipt`` (which carries the
    diff). Firing ``agent_config_updated`` on top would be redundant
    noise the SPA would have to dedupe — gate the emit to hard fields
    only."""
    app, _, _, bus = v2_app_with_bus
    captured = _capture(bus)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/agents/scout/edit-soft",
            json={
                "field": "instructions",
                "value": "Be exceedingly polite.",
                "summary": "tighten tone",
            },
            headers=_human_origin_headers(),
        )
    assert r.status_code == 200, r.text
    # No ``agent_config_updated`` for soft fields — the receipt covers it.
    assert not [e for e in captured if e["type"] == "agent_config_updated"]
    # The receipt itself still fires.
    assert [e for e in captured if e["type"] == "operator_action_receipt"]


@pytest.fixture
def v2_app_with_bus_and_transport(tmp_path, monkeypatch):
    """Variant of ``v2_app_with_bus`` that wires a controllable transport
    so tests can drive the hot-reload success / failure branches inside
    ``_apply_pending_change``.

    Yields ``(app, server_module, tmp_path, bus, transport)``.
    """
    from src.dashboard.events import EventBus

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
    cost_tracker = MagicMock()
    cost_tracker.check_budget = MagicMock(return_value={
        "allowed": True, "daily_used": 1.0, "daily_limit": 10.0,
        "monthly_used": 5.0, "monthly_limit": 200.0,
    })
    container_manager = MagicMock()
    container_manager.stop_agent = MagicMock(return_value=True)
    bus = EventBus()
    transport = MagicMock()
    transport.request = AsyncMock(return_value={"status": "ok"})

    blackboard = Blackboard(str(tmp_path / "bb.db"))
    pubsub = PubSub()
    permissions = PermissionMatrix()
    for aid, perms in perms_map.items():
        permissions.permissions[aid] = AgentPermissions(agent_id=aid, **perms)
    agents = {
        "scout": "http://scout:8400",
        "analyst": "http://analyst:8400",
        "tracker": "http://tracker:8400",
        "operator": "http://operator:8400",
    }
    router = MessageRouter(permissions, agents)
    # ``auth_tokens`` must be non-empty for ``_resolve_agent_id`` to
    # resolve through ``_extract_verified_agent_id`` (which respects
    # the ``x-mesh-internal`` + ``X-Agent-ID`` headers internal callers
    # use). The ``/mesh/agents/{id}/propose`` endpoint specifically
    # routes through ``_resolve_agent_id`` rather than the verified
    # extractor, so without auth_tokens it returns "" → 403. The token
    # value itself is irrelevant because we go through the internal
    # path; we just need the dict to be truthy.
    app = server_module.create_mesh_app(
        blackboard=blackboard,
        pubsub=pubsub,
        router=router,
        permissions=permissions,
        cost_tracker=cost_tracker,
        container_manager=container_manager,
        event_bus=bus,
        transport=transport,
        auth_tokens={"operator": "test-token-not-used-via-internal-path"},
    )
    yield app, server_module, tmp_path, bus, transport
    blackboard.close()
    monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_V2", raising=False)
    monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_DB", raising=False)
    importlib.reload(server_module)


def _internal_operator_headers() -> dict:
    """Headers for an internal-caller operator request.

    The hard-edit ``propose`` + ``config`` endpoints route through
    ``_resolve_agent_id("", request)`` rather than
    ``_extract_verified_agent_id``, which means in test mode (no
    ``_auth_tokens`` configured) they return ``""`` from a plain
    ``X-Agent-ID: operator`` header — and 403 from the operator gate.
    The ``x-mesh-internal: 1`` header (loopback peer is provided by
    ASGITransport) flips the resolver to read X-Agent-ID, restoring
    the dev-mode operator persona.
    """
    return {
        "X-Agent-ID": "operator",
        "X-Origin": MessageOrigin(
            kind="human", channel="cli", user="u1",
        ).to_header_value(),
        "x-mesh-internal": "1",
    }


@pytest.mark.asyncio
async def test_hard_edit_live_true_when_hot_reload_succeeds(
    v2_app_with_bus_and_transport,
):
    """Hot-reload success → ``live: true`` so the SPA doesn't show the
    "saved — restart to apply" hint."""
    app, _, _, bus, transport = v2_app_with_bus_and_transport
    transport.request = AsyncMock(return_value={"status": "ok"})
    captured = _capture(bus)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        propose = await c.post(
            "/mesh/agents/scout/propose",
            json={"field": "thinking", "value": "high"},
            headers=_internal_operator_headers(),
        )
        assert propose.status_code == 200, propose.text
        change_id = propose.json()["change_id"]
        confirm = await c.post(
            "/mesh/agents/scout/config",
            json={"change_id": change_id},
            headers=_internal_operator_headers(),
        )
    assert confirm.status_code == 200, confirm.text
    updated = [e for e in captured if e["type"] == "agent_config_updated"]
    assert len(updated) == 1
    assert updated[0]["data"]["live"] is True


@pytest.mark.asyncio
async def test_hard_edit_live_false_when_hot_reload_errors(
    v2_app_with_bus_and_transport,
):
    """Hot-reload error path → ``live: false``. The YAML write is
    durable; the running agent has the old config until restart. The
    SPA reads ``live`` and renders a "saved — restart to apply" hint.
    """
    app, _, _, bus, transport = v2_app_with_bus_and_transport
    # transport.request returns ``{"error": ...}`` rather than raising
    # for HTTP / timeout failures (see HttpTransport semantics).
    transport.request = AsyncMock(return_value={"error": "Connection refused"})
    captured = _capture(bus)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        propose = await c.post(
            "/mesh/agents/scout/propose",
            json={"field": "thinking", "value": "high"},
            headers=_internal_operator_headers(),
        )
        assert propose.status_code == 200, propose.text
        change_id = propose.json()["change_id"]
        confirm = await c.post(
            "/mesh/agents/scout/config",
            json={"change_id": change_id},
            headers=_internal_operator_headers(),
        )
    assert confirm.status_code == 200, confirm.text
    updated = [e for e in captured if e["type"] == "agent_config_updated"]
    assert len(updated) == 1
    assert updated[0]["data"]["live"] is False


@pytest.mark.asyncio
async def test_hard_edit_emits_agent_config_updated_without_value_diff(
    v2_app_with_bus_and_transport,
):
    """Hard-field confirms emit ``agent_config_updated`` (no Revert
    receipt) so the SPA flips the card. Payload deliberately omits
    ``new_value`` / ``old_value`` — the SPA refetches the agent
    detail anyway, and ``permissions`` ACL diffs are structurally
    sensitive (keeping them out of WS frames)."""
    app, _, _, bus, transport = v2_app_with_bus_and_transport
    transport.request = AsyncMock(return_value={"status": "ok"})
    captured = _capture(bus)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        # Stage a hard-field propose first, then confirm.
        propose = await c.post(
            "/mesh/agents/scout/propose",
            json={
                "field": "model",
                "value": "openai/gpt-4.1-mini",
            },
            headers=_internal_operator_headers(),
        )
        assert propose.status_code == 200, propose.text
        change_id = propose.json().get("change_id")
        assert change_id, propose.json()
        confirm = await c.post(
            "/mesh/agents/scout/config",
            json={"change_id": change_id},
            headers=_internal_operator_headers(),
        )
    assert confirm.status_code == 200, confirm.text
    updated = [e for e in captured if e["type"] == "agent_config_updated"]
    assert len(updated) == 1
    assert updated[0]["agent"] == "scout"
    assert updated[0]["data"]["field"] == "model"
    # The diff fields are intentionally absent.
    assert "new_value" not in updated[0]["data"]
    assert "old_value" not in updated[0]["data"]
    # ``live`` rides along so the SPA can render a "saved — restart to
    # apply" hint when the running agent still has the old config.
    assert "live" in updated[0]["data"]
    assert isinstance(updated[0]["data"]["live"], bool)


# ── Operator task-action wake propagation ─────────────────────────────


def _build_app_with_lanes(tmp_path, server_module, *, perms_map, agents,
                          cost_tracker, container_manager):
    """Build the mesh app with a stub lane manager + dedicated dispatch loop.

    Returns ``(app, blackboard, lane_calls, teardown)``. ``lane_calls``
    is the list of (args, kwargs) every ``lane_manager.enqueue`` call
    is recorded into so tests can assert the operator-driven wake.
    """
    import asyncio
    import threading

    blackboard = Blackboard(str(tmp_path / "bb.db"))
    pubsub = PubSub()
    permissions = PermissionMatrix()
    for aid, perms in perms_map.items():
        permissions.permissions[aid] = AgentPermissions(agent_id=aid, **perms)
    router = MessageRouter(permissions, agents)

    lane_calls: list[tuple[tuple, dict]] = []

    def _enqueue(*args, **kwargs):
        lane_calls.append((args, kwargs))

        async def _noop():
            return None

        return _noop()

    lane_manager = MagicMock()
    lane_manager.enqueue = _enqueue

    dispatch_loop = asyncio.new_event_loop()
    thread = threading.Thread(target=dispatch_loop.run_forever, daemon=True)
    thread.start()

    app = server_module.create_mesh_app(
        blackboard=blackboard,
        pubsub=pubsub,
        router=router,
        permissions=permissions,
        cost_tracker=cost_tracker,
        container_manager=container_manager,
        lane_manager=lane_manager,
        dispatch_loop=dispatch_loop,
    )

    def _teardown():
        blackboard.close()
        dispatch_loop.call_soon_threadsafe(dispatch_loop.stop)
        thread.join(timeout=2)
        dispatch_loop.close()

    return app, lane_calls, _teardown


@pytest.fixture
def v2_app_with_lanes(tmp_path, monkeypatch):
    """``v2_app`` variant wired to a stub lane manager.

    Used to verify operator state changes (reroute / retry / cancel)
    actually enqueue a wake for the affected agent instead of leaving
    them idle until the next heartbeat.
    """
    pdir = _projects_layout(tmp_path)
    afile = _agents_yaml(tmp_path, names=["scout", "analyst", "tracker"])
    monkeypatch.chdir(tmp_path)
    import src.cli.config as cli_cfg
    monkeypatch.setattr(cli_cfg, "PROJECTS_DIR", pdir)
    monkeypatch.setattr(cli_cfg, "AGENTS_FILE", afile)
    monkeypatch.setattr(
        cli_cfg, "PERMISSIONS_FILE", tmp_path / "config" / "permissions.json",
    )

    server_module = _reload_server(
        monkeypatch, v2=True, tasks_db=str(tmp_path / "tasks.db"),
    )

    perms_map = {
        "operator": {"can_route_tasks": True, "can_manage_projects": True},
        "scout":    {"can_route_tasks": True, "can_message": ["analyst", "operator"]},
        "analyst":  {"can_route_tasks": False, "can_message": ["scout"]},
        "tracker":  {"can_route_tasks": False, "can_message": []},
    }
    cost_tracker = MagicMock()
    cost_tracker.check_budget = MagicMock(
        return_value={
            "allowed": True, "daily_used": 1.0, "daily_limit": 10.0,
            "monthly_used": 5.0, "monthly_limit": 200.0,
        },
    )
    container_manager = MagicMock()
    container_manager.stop_agent = MagicMock(return_value=True)

    app, lane_calls, teardown = _build_app_with_lanes(
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
    try:
        yield app, lane_calls
    finally:
        teardown()
        monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_V2", raising=False)
        monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_DB", raising=False)
        importlib.reload(server_module)


@pytest.mark.asyncio
async def test_reroute_wakes_new_assignee(v2_app_with_lanes):
    """Operator reroute should fire-and-forget a lane enqueue on the
    new assignee so it starts work immediately."""
    app, lane_calls = v2_app_with_lanes
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/tasks",
            json={"assignee": "analyst", "title": "investigate", "project": "research"},
            headers={"X-Agent-ID": "operator"},
        )
        tid = r.json()["id"]
        r = await c.post(
            f"/mesh/tasks/{tid}/reroute",
            json={"new_assignee": "tracker", "reason": "load balance"},
            headers={"X-Agent-ID": "operator"},
        )
    assert r.status_code == 200, r.text
    targets = [call[0][0] for call in lane_calls]
    assert "tracker" in targets, f"expected tracker wake, got {lane_calls!r}"
    # Pull the matching call and check the message + mode.
    tracker_call = next(c for c in lane_calls if c[0][0] == "tracker")
    message = tracker_call[0][1]
    assert "rerouted" in message.lower()
    assert "investigate" in message
    assert "check_inbox" in message
    assert tracker_call[1].get("mode") == "followup"


@pytest.mark.asyncio
async def test_retry_wakes_clone_assignee(v2_app_with_lanes):
    """Operator retry on a failed task should wake the clone's assignee."""
    app, lane_calls = v2_app_with_lanes
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/tasks",
            json={"assignee": "analyst", "title": "build report", "project": "research"},
            headers={"X-Agent-ID": "operator"},
        )
        tid = r.json()["id"]
        await c.post(
            f"/mesh/tasks/{tid}/status", json={"status": "working"},
            headers={"X-Agent-ID": "analyst"},
        )
        await c.post(
            f"/mesh/tasks/{tid}/status", json={"status": "failed"},
            headers={"X-Agent-ID": "analyst"},
        )
        lane_calls.clear()
        r = await c.post(
            f"/mesh/tasks/{tid}/retry",
            json={"assignee": "tracker"},
            headers={"X-Agent-ID": "operator"},
        )
    assert r.status_code == 200, r.text
    targets = [call[0][0] for call in lane_calls]
    assert "tracker" in targets, f"expected tracker wake, got {lane_calls!r}"
    tracker_call = next(c for c in lane_calls if c[0][0] == "tracker")
    message = tracker_call[0][1]
    assert "retried" in message.lower()
    assert "build report" in message
    assert "check_inbox" in message


@pytest.mark.asyncio
async def test_cancel_wakes_prior_assignee(v2_app_with_lanes):
    """Operator cancelling an active task should wake the assignee so
    they stop work instead of churning until the next heartbeat."""
    app, lane_calls = v2_app_with_lanes
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/tasks",
            json={"assignee": "analyst", "title": "scan logs", "project": "research"},
            headers={"X-Agent-ID": "operator"},
        )
        tid = r.json()["id"]
        await c.post(
            f"/mesh/tasks/{tid}/status", json={"status": "working"},
            headers={"X-Agent-ID": "analyst"},
        )
        lane_calls.clear()
        r = await c.post(
            f"/mesh/tasks/{tid}/cancel",
            json={"reason": "scope changed"},
            headers={"X-Agent-ID": "operator"},
        )
    assert r.status_code == 200, r.text
    targets = [call[0][0] for call in lane_calls]
    assert "analyst" in targets, f"expected analyst wake, got {lane_calls!r}"
    analyst_call = next(c for c in lane_calls if c[0][0] == "analyst")
    message = analyst_call[0][1]
    assert "cancelled" in message.lower()
    assert "scan logs" in message


@pytest.mark.asyncio
async def test_cancel_does_not_wake_self_canceller(v2_app_with_lanes):
    """If the assignee cancels their own task, no self-wake fires —
    you don't need to tell yourself to stop."""
    app, lane_calls = v2_app_with_lanes
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/tasks",
            json={"assignee": "analyst", "title": "tail logs", "project": "research"},
            headers={"X-Agent-ID": "operator"},
        )
        tid = r.json()["id"]
        await c.post(
            f"/mesh/tasks/{tid}/status", json={"status": "working"},
            headers={"X-Agent-ID": "analyst"},
        )
        lane_calls.clear()
        r = await c.post(
            f"/mesh/tasks/{tid}/cancel",
            json={"reason": "redundant"},
            headers={"X-Agent-ID": "analyst"},
        )
    assert r.status_code == 200, r.text
    targets = [call[0][0] for call in lane_calls]
    assert "analyst" not in targets, (
        f"self-cancel should not wake self, got {lane_calls!r}"
    )
