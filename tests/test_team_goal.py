"""Tests for the north-star path on the team-domain entity: the
``set_team_goal`` operator tool and the ``POST /mesh/teams/{name}/goal``
endpoint, persisting through the TeamStore (the first-class team row).

These exercise:
- The tool's happy path + validation gates (length limits, types).
- The mesh endpoint's persistence into the TeamStore (round-trip
  through ``teams_store.get_team`` so the dashboard sees the new fields).
- Teams created without a goal read back ``north_star=None`` /
  ``success_criteria=None``.
"""

from __future__ import annotations

import importlib
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from src.host.mesh import Blackboard, MessageRouter, PubSub
from src.host.permissions import PermissionMatrix
from src.shared.types import AgentPermissions


@pytest.fixture(autouse=True)
def _set_operator_env(monkeypatch):
    """Operator tools require ALLOWED_TOOLS to be set."""
    monkeypatch.setenv("ALLOWED_TOOLS", "set_team_goal")


# ── Store: teams default to no goal until one is set ───────────────


class TestTeamGoalDefaults:
    def test_defaults_are_none(self):
        from src.host.teams import TeamStore
        store = TeamStore(db_path=":memory:")
        store.create_team("x")
        team = store.get_team("x")
        assert team["north_star"] is None
        assert team["success_criteria"] is None


# ── set_project_goal tool unit tests (mocked mesh client) ─────────────


@pytest.mark.asyncio
async def test_set_team_goal_happy_path():
    from src.agent.builtins.operator_tools import set_team_goal

    mc = MagicMock()
    mc.set_team_goal = AsyncMock(return_value={
        "success": True,
        "team_name": "growth",
        "north_star": "Ship $10k MRR",
        "success_criteria": ["100 visits/day"],
    })
    result = await set_team_goal(
        "growth", "Ship $10k MRR",
        success_criteria=["100 visits/day"],
        mesh_client=mc,
    )
    assert result["success"] is True
    assert result["team_name"] == "growth"
    mc.set_team_goal.assert_awaited_once_with(
        "growth", "Ship $10k MRR", ["100 visits/day"],
    )


@pytest.mark.asyncio
async def test_set_project_goal_no_mesh_client():
    from src.agent.builtins.operator_tools import set_team_goal

    result = await set_team_goal("growth", "ship it")
    assert "error" in result
    assert "mesh_client" in result["error"].lower()


@pytest.mark.asyncio
async def test_set_project_goal_requires_operator(monkeypatch):
    """Defence-in-depth: tool refuses to run when ALLOWED_TOOLS is unset."""
    monkeypatch.delenv("ALLOWED_TOOLS", raising=False)
    from src.agent.builtins.operator_tools import set_team_goal
    result = await set_team_goal("growth", "ship it", mesh_client=MagicMock())
    assert "operator" in result["error"].lower()


@pytest.mark.asyncio
async def test_set_project_goal_validates_north_star_length():
    from src.agent.builtins.operator_tools import set_team_goal

    too_long = "x" * 2001
    result = await set_team_goal("growth", too_long, mesh_client=MagicMock())
    assert "error" in result
    assert "2000" in result["error"]


@pytest.mark.asyncio
async def test_set_project_goal_accepts_long_vision_statement():
    """A real vision statement at the 2000-char ceiling must round-trip."""
    from src.agent.builtins.operator_tools import set_team_goal

    mc = MagicMock()
    mc.set_team_goal = AsyncMock(return_value={"success": True})
    long_vision = "ship it. " * 200  # ~1800 chars
    assert 500 < len(long_vision) <= 2000
    result = await set_team_goal(
        "growth", long_vision, mesh_client=mc,
    )
    assert "error" not in result


@pytest.mark.asyncio
async def test_set_project_goal_validates_success_criteria_count():
    from src.agent.builtins.operator_tools import set_team_goal

    result = await set_team_goal(
        "growth", "ship it",
        success_criteria=[f"sc-{i}" for i in range(11)],
        mesh_client=MagicMock(),
    )
    assert "error" in result
    assert "10" in result["error"]


@pytest.mark.asyncio
async def test_set_project_goal_validates_success_criteria_length():
    from src.agent.builtins.operator_tools import set_team_goal

    result = await set_team_goal(
        "growth", "ship it",
        success_criteria=["x" * 201],
        mesh_client=MagicMock(),
    )
    assert "error" in result
    assert "200" in result["error"]


@pytest.mark.asyncio
async def test_set_project_goal_strips_blank_criteria():
    """Empty / whitespace-only entries should silently drop, not fail."""
    from src.agent.builtins.operator_tools import set_team_goal

    mc = MagicMock()
    mc.set_team_goal = AsyncMock(return_value={"success": True})
    await set_team_goal(
        "growth", "ship it",
        success_criteria=["one", "  ", "", "two"],
        mesh_client=mc,
    )
    args = mc.set_team_goal.await_args.args
    assert args == ("growth", "ship it", ["one", "two"])


@pytest.mark.asyncio
async def test_set_project_goal_mesh_error_surfaces():
    from src.agent.builtins.operator_tools import set_team_goal

    mc = MagicMock()
    mc.set_team_goal = AsyncMock(side_effect=RuntimeError("boom"))
    result = await set_team_goal(
        "growth", "ship it", mesh_client=mc,
    )
    assert "error" in result
    assert "boom" in result["error"]


# ── Endpoint integration: POST /mesh/teams/{name}/goal ───────────────


def _reload_server(monkeypatch):
    """Fresh import of the server module with a clean env."""
    import src.host.server as server_module
    importlib.reload(server_module)
    return server_module


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
    monkeypatch.chdir(tmp_path)
    server_module = _reload_server(monkeypatch)
    perms_map = {
        "operator": {"can_route_tasks": True, "can_manage_teams": True},
        "writer": {"can_route_tasks": False, "can_message": []},
    }
    app, bb = _build_app(tmp_path, server_module, perms_map)
    # One team seeded so the goal endpoint has something to update.
    app.teams_store.create_team("growth", description="growth project")
    app.teams_store.add_member("growth", "writer")
    yield app, app.teams_store
    bb.close()
    importlib.reload(server_module)


@pytest.mark.asyncio
async def test_endpoint_set_goal_persists(goal_app):
    app, teams_store = goal_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/teams/growth/goal",
            json={
                "north_star": "Ship $10k MRR landing page in 2 weeks",
                "success_criteria": ["100 visits/day", "5 demo bookings/wk"],
            },
            headers={"X-Agent-ID": "operator"},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True
    assert body["team_name"] == "growth"
    assert body["north_star"] == "Ship $10k MRR landing page in 2 weeks"
    assert body["success_criteria"] == ["100 visits/day", "5 demo bookings/wk"]

    # Round-trip via the store so the dashboard would see the same.
    team = teams_store.get_team("growth")
    assert team["north_star"] == (
        "Ship $10k MRR landing page in 2 weeks"
    )
    assert team["success_criteria"] == [
        "100 visits/day", "5 demo bookings/wk",
    ]


@pytest.mark.asyncio
async def test_endpoint_set_goal_unknown_project_returns_404(goal_app):
    app, _ = goal_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/teams/no-such/goal",
            json={"north_star": "x"},
            headers={"X-Agent-ID": "operator"},
        )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_endpoint_set_goal_non_operator_forbidden(goal_app):
    app, _ = goal_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/teams/growth/goal",
            json={"north_star": "x"},
            headers={"X-Agent-ID": "writer"},
        )
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_endpoint_set_goal_north_star_too_long_400(goal_app):
    app, _ = goal_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/teams/growth/goal",
            json={"north_star": "x" * 2001},
            headers={"X-Agent-ID": "operator"},
        )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_endpoint_set_goal_too_many_criteria_400(goal_app):
    app, _ = goal_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/teams/growth/goal",
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
            "/mesh/teams/growth/goal",
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
            "/mesh/teams/growth/goal",
            json={"north_star": "first", "success_criteria": ["one"]},
            headers={"X-Agent-ID": "operator"},
        )
        assert r.status_code == 200
        # Now clear it.
        r = await c.post(
            "/mesh/teams/growth/goal",
            json={"north_star": "", "success_criteria": []},
            headers={"X-Agent-ID": "operator"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["north_star"] is None
    assert body["success_criteria"] is None
