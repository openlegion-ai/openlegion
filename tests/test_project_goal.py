"""Tests for the PR 5 north-star path: ``set_project_goal`` operator
tool, ``POST /mesh/projects/{name}/goal`` endpoint, and the
``ProjectMetadata`` schema bump that lets a project carry its goal as a
first-class artifact.

These exercise:
- The tool's happy path + validation gates (length limits, types).
- The mesh endpoint's persistence into ``metadata.yaml`` (round-trip
  through ``_load_projects`` so the dashboard sees the new fields).
- Backwards compat — projects predating the new fields load with
  ``north_star=None`` / ``success_criteria=None``.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from httpx import ASGITransport, AsyncClient

from src.host.mesh import Blackboard, MessageRouter, PubSub
from src.host.permissions import PermissionMatrix
from src.shared.types import AgentPermissions, ProjectMetadata


@pytest.fixture(autouse=True)
def _set_operator_env(monkeypatch):
    """Operator tools require ALLOWED_TOOLS to be set."""
    monkeypatch.setenv("ALLOWED_TOOLS", "set_project_goal")


# ── Schema: ProjectMetadata accepts north_star + success_criteria ─────


class TestProjectMetadataSchema:
    def test_defaults_are_none(self):
        pm = ProjectMetadata(name="x")
        assert pm.north_star is None
        assert pm.success_criteria is None

    def test_round_trip_with_goal(self):
        pm = ProjectMetadata(
            name="x",
            north_star="Ship $10k MRR landing page in 2 weeks",
            success_criteria=["100 daily uniques", "5 demo bookings/wk"],
        )
        dumped = pm.model_dump()
        assert dumped["north_star"] == "Ship $10k MRR landing page in 2 weeks"
        assert dumped["success_criteria"] == [
            "100 daily uniques", "5 demo bookings/wk",
        ]
        # Round-trip back through the model so the YAML on-disk shape works.
        pm2 = ProjectMetadata(**dumped)
        assert pm2.north_star == pm.north_star
        assert pm2.success_criteria == pm.success_criteria

    def test_legacy_metadata_without_goal_loads(self):
        """A project written before this PR has no north_star / success_criteria
        and must still load cleanly (defaulting to ``None``)."""
        legacy = {
            "name": "legacy",
            "description": "old project",
            "members": ["a"],
            "created_at": "2025-12-01T00:00:00+00:00",
            "status": "active",
            "settings": {},
        }
        pm = ProjectMetadata(**legacy)
        assert pm.north_star is None
        assert pm.success_criteria is None


# ── set_project_goal tool unit tests (mocked mesh client) ─────────────


@pytest.mark.asyncio
async def test_set_project_goal_happy_path():
    from src.agent.builtins.operator_tools import set_project_goal

    mc = MagicMock()
    mc.set_project_goal = AsyncMock(return_value={
        "success": True,
        "project_name": "growth",
        "north_star": "Ship $10k MRR",
        "success_criteria": ["100 visits/day"],
    })
    result = await set_project_goal(
        "growth", "Ship $10k MRR",
        success_criteria=["100 visits/day"],
        mesh_client=mc,
    )
    assert result["success"] is True
    assert result["project_name"] == "growth"
    mc.set_project_goal.assert_awaited_once_with(
        "growth", "Ship $10k MRR", ["100 visits/day"],
    )


@pytest.mark.asyncio
async def test_set_project_goal_no_mesh_client():
    from src.agent.builtins.operator_tools import set_project_goal

    result = await set_project_goal("growth", "ship it")
    assert "error" in result
    assert "mesh_client" in result["error"].lower()


@pytest.mark.asyncio
async def test_set_project_goal_requires_operator(monkeypatch):
    """Defence-in-depth: tool refuses to run when ALLOWED_TOOLS is unset."""
    monkeypatch.delenv("ALLOWED_TOOLS", raising=False)
    from src.agent.builtins.operator_tools import set_project_goal
    result = await set_project_goal("growth", "ship it", mesh_client=MagicMock())
    assert "operator" in result["error"].lower()


@pytest.mark.asyncio
async def test_set_project_goal_validates_north_star_length():
    from src.agent.builtins.operator_tools import set_project_goal

    too_long = "x" * 501
    result = await set_project_goal("growth", too_long, mesh_client=MagicMock())
    assert "error" in result
    assert "500" in result["error"]


@pytest.mark.asyncio
async def test_set_project_goal_validates_success_criteria_count():
    from src.agent.builtins.operator_tools import set_project_goal

    result = await set_project_goal(
        "growth", "ship it",
        success_criteria=[f"sc-{i}" for i in range(11)],
        mesh_client=MagicMock(),
    )
    assert "error" in result
    assert "10" in result["error"]


@pytest.mark.asyncio
async def test_set_project_goal_validates_success_criteria_length():
    from src.agent.builtins.operator_tools import set_project_goal

    result = await set_project_goal(
        "growth", "ship it",
        success_criteria=["x" * 201],
        mesh_client=MagicMock(),
    )
    assert "error" in result
    assert "200" in result["error"]


@pytest.mark.asyncio
async def test_set_project_goal_strips_blank_criteria():
    """Empty / whitespace-only entries should silently drop, not fail."""
    from src.agent.builtins.operator_tools import set_project_goal

    mc = MagicMock()
    mc.set_project_goal = AsyncMock(return_value={"success": True})
    await set_project_goal(
        "growth", "ship it",
        success_criteria=["one", "  ", "", "two"],
        mesh_client=mc,
    )
    args = mc.set_project_goal.await_args.args
    assert args == ("growth", "ship it", ["one", "two"])


@pytest.mark.asyncio
async def test_set_project_goal_mesh_error_surfaces():
    from src.agent.builtins.operator_tools import set_project_goal

    mc = MagicMock()
    mc.set_project_goal = AsyncMock(side_effect=RuntimeError("boom"))
    result = await set_project_goal(
        "growth", "ship it", mesh_client=mc,
    )
    assert "error" in result
    assert "boom" in result["error"]


# ── Endpoint integration: POST /mesh/projects/{name}/goal ────────────


def _reload_server(monkeypatch):
    """Fresh import of the server module with a clean env."""
    import src.host.server as server_module
    importlib.reload(server_module)
    return server_module


def _projects_layout(tmp_path) -> Path:
    """One project on disk so the goal endpoint has something to update."""
    pdir = tmp_path / "projects"
    growth = pdir / "growth"
    growth.mkdir(parents=True)
    (growth / "metadata.yaml").write_text(yaml.dump({
        "name": "growth",
        "description": "growth project",
        "members": ["writer"],
        "created_at": "2026-05-02T00:00:00+00:00",
        "status": "active",
    }))
    return pdir


def _build_app(tmp_path, server_module, perms_map):
    blackboard = Blackboard(str(tmp_path / "bb.db"))
    pubsub = PubSub()
    permissions = PermissionMatrix()
    for aid, perms in perms_map.items():
        permissions.permissions[aid] = AgentPermissions(agent_id=aid, **perms)
    router = MessageRouter(permissions, {"operator": "http://op:8400"})
    app = server_module.create_mesh_app(
        blackboard=blackboard,
        pubsub=pubsub,
        router=router,
        permissions=permissions,
    )
    return app, blackboard


@pytest.fixture
def goal_app(tmp_path, monkeypatch):
    pdir = _projects_layout(tmp_path)
    monkeypatch.chdir(tmp_path)
    import src.cli.config as cli_cfg
    monkeypatch.setattr(cli_cfg, "PROJECTS_DIR", pdir)
    server_module = _reload_server(monkeypatch)
    perms_map = {
        "operator": {"can_route_tasks": True, "can_manage_projects": True},
        "writer": {"can_route_tasks": False, "can_message": []},
    }
    app, bb = _build_app(tmp_path, server_module, perms_map)
    yield app, pdir
    bb.close()
    importlib.reload(server_module)


@pytest.mark.asyncio
async def test_endpoint_set_goal_persists(goal_app):
    app, pdir = goal_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/projects/growth/goal",
            json={
                "north_star": "Ship $10k MRR landing page in 2 weeks",
                "success_criteria": ["100 visits/day", "5 demo bookings/wk"],
            },
            headers={"X-Agent-ID": "operator"},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True
    assert body["project_name"] == "growth"
    assert body["north_star"] == "Ship $10k MRR landing page in 2 weeks"
    assert body["success_criteria"] == ["100 visits/day", "5 demo bookings/wk"]

    # Round-trip via _load_projects so the dashboard would see the same.
    from src.cli.config import _load_projects
    projects = _load_projects()
    assert projects["growth"]["north_star"] == (
        "Ship $10k MRR landing page in 2 weeks"
    )
    assert projects["growth"]["success_criteria"] == [
        "100 visits/day", "5 demo bookings/wk",
    ]


@pytest.mark.asyncio
async def test_endpoint_set_goal_unknown_project_returns_404(goal_app):
    app, _ = goal_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/projects/no-such/goal",
            json={"north_star": "x"},
            headers={"X-Agent-ID": "operator"},
        )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_endpoint_set_goal_non_operator_forbidden(goal_app):
    app, _ = goal_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/projects/growth/goal",
            json={"north_star": "x"},
            headers={"X-Agent-ID": "writer"},
        )
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_endpoint_set_goal_north_star_too_long_400(goal_app):
    app, _ = goal_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/projects/growth/goal",
            json={"north_star": "x" * 501},
            headers={"X-Agent-ID": "operator"},
        )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_endpoint_set_goal_too_many_criteria_400(goal_app):
    app, _ = goal_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/projects/growth/goal",
            json={
                "north_star": "x",
                "success_criteria": [f"sc-{i}" for i in range(11)],
            },
            headers={"X-Agent-ID": "operator"},
        )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_endpoint_set_goal_success_criterion_too_long_400(goal_app):
    app, _ = goal_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/projects/growth/goal",
            json={
                "north_star": "x",
                "success_criteria": ["x" * 201],
            },
            headers={"X-Agent-ID": "operator"},
        )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_endpoint_set_goal_clears_when_empty(goal_app):
    """Sending blank values clears the goal (north_star=None, success_criteria=None)."""
    app, _ = goal_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        # First set something.
        r = await c.post(
            "/mesh/projects/growth/goal",
            json={"north_star": "first", "success_criteria": ["one"]},
            headers={"X-Agent-ID": "operator"},
        )
        assert r.status_code == 200
        # Now clear it.
        r = await c.post(
            "/mesh/projects/growth/goal",
            json={"north_star": "", "success_criteria": []},
            headers={"X-Agent-ID": "operator"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["north_star"] is None
    assert body["success_criteria"] is None
