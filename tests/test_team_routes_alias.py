"""Tests for the ``/mesh/teams/*`` and ``/api/teams/*`` route aliases.

PR 1 of the project→team rename scaffolds these as thin aliases over
the existing project routes. The tests below verify each new alias:

  * is reachable and returns the same response shape (plus ``team_*``
    keys alongside the existing ``project_*`` keys),
  * exercises the same underlying ``_load_projects`` / ``_create_project``
    / ``_delete_project`` / ``_add_agent_to_project`` / etc. helpers, so
    the side effect lands on disk identically,
  * preserves the project route's auth and validation behavior.

The mesh-side tests reuse the ``v2_app`` fixture from
``tests/test_operator_product_tools.py``; the dashboard-side tests reuse
the ``_make_components`` / ``_make_client`` helpers from
``tests/test_dashboard.py``.
"""

# ruff: noqa: F811
# F811 silenced module-wide: the ``v2_app`` fixture is imported from
# tests/test_operator_product_tools.py and then bound as a parameter on
# each test function — this is the standard pytest fixture-sharing
# pattern and not an actual redefinition.

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from httpx import ASGITransport, AsyncClient

# Reuse the existing fixtures rather than re-implementing them so the
# tests stay aligned as the project routes evolve.
from tests.test_dashboard import _make_client, _make_components, _teardown
from tests.test_operator_product_tools import v2_app  # noqa: F401

# ── Mesh /mesh/teams/* aliases ─────────────────────────────────────


@pytest.mark.asyncio
async def test_mesh_teams_list(v2_app):
    """GET /mesh/teams returns the same set of names as /mesh/projects."""
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        projects_resp = await c.get(
            "/mesh/projects", headers={"X-Agent-ID": "operator"},
        )
        teams_resp = await c.get(
            "/mesh/teams", headers={"X-Agent-ID": "operator"},
        )
    assert projects_resp.status_code == 200
    assert teams_resp.status_code == 200
    teams_body = teams_resp.json()
    # New ``teams`` key, plus legacy ``projects`` alias for back-compat.
    assert "teams" in teams_body
    assert "projects" in teams_body
    proj_names = {p["name"] for p in projects_resp.json()["projects"]}
    team_names = {t["name"] for t in teams_body["teams"]}
    assert proj_names == team_names == {"research", "ops"}
    # Each row carries both ``name`` and ``team_name``.
    for row in teams_body["teams"]:
        assert row["name"] == row["team_name"]


@pytest.mark.asyncio
async def test_mesh_teams_create_parity_with_projects(v2_app):
    """POST /mesh/teams matches POST /mesh/projects behavior.

    The v2_app fixture runs without auth tokens, so ``_resolve_agent_id``
    returns the (empty) header value rather than ``"operator"`` and both
    endpoints reject with 403. The test asserts parity — same status,
    same error shape — which proves the alias is wired up correctly.
    See the note in ``test_operator_product_tools.py`` around line 1000
    for why ``_resolve_agent_id``-gated endpoints can't be exercised
    end-to-end from this fixture.
    """
    app, _, _ = v2_app
    payload = {"name": "neoteam", "description": "via alias", "members": []}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        proj_resp = await c.post(
            "/mesh/projects", json=payload,
            headers={"X-Agent-ID": "operator"},
        )
        team_resp = await c.post(
            "/mesh/teams", json=payload,
            headers={"X-Agent-ID": "operator"},
        )
    assert proj_resp.status_code == team_resp.status_code == 403
    assert "operator" in proj_resp.json()["detail"].lower()
    assert "operator" in team_resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_mesh_teams_add_member_parity_with_projects(v2_app):
    """POST /mesh/teams/{name}/members mirrors /mesh/projects/{name}/members.

    Same caveat as ``test_mesh_teams_create_parity_with_projects`` —
    ``_resolve_agent_id`` gate forces 403 from this fixture for both
    routes; we assert parity.
    """
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        proj_resp = await c.post(
            "/mesh/projects/research/members",
            json={"agent": "tracker"},
            headers={"X-Agent-ID": "operator"},
        )
        team_resp = await c.post(
            "/mesh/teams/research/members",
            json={"agent": "tracker"},
            headers={"X-Agent-ID": "operator"},
        )
    assert proj_resp.status_code == team_resp.status_code == 403


@pytest.mark.asyncio
async def test_mesh_teams_remove_member_parity_with_projects(v2_app):
    """DELETE /mesh/teams/{name}/members/{agent} mirrors the project route."""
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        proj_resp = await c.delete(
            "/mesh/projects/research/members/scout",
            headers={"X-Agent-ID": "operator"},
        )
        team_resp = await c.delete(
            "/mesh/teams/research/members/scout",
            headers={"X-Agent-ID": "operator"},
        )
    assert proj_resp.status_code == team_resp.status_code == 403


@pytest.mark.asyncio
async def test_mesh_teams_delete_parity_with_projects(v2_app):
    """DELETE /mesh/teams/{name} mirrors DELETE /mesh/projects/{name}."""
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        proj_resp = await c.delete(
            "/mesh/projects/ops", headers={"X-Agent-ID": "operator"},
        )
        team_resp = await c.delete(
            "/mesh/teams/ops", headers={"X-Agent-ID": "operator"},
        )
    assert proj_resp.status_code == team_resp.status_code == 403


@pytest.mark.asyncio
async def test_mesh_teams_update_context_parity_with_projects(v2_app):
    """PUT /mesh/teams/{name}/context mirrors PUT /mesh/projects/{name}/context."""
    app, _, _ = v2_app
    payload = {"context": "Hunt for opportunities in fintech."}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        proj_resp = await c.put(
            "/mesh/projects/research/context", json=payload,
            headers={"X-Agent-ID": "operator"},
        )
        team_resp = await c.put(
            "/mesh/teams/research/context", json=payload,
            headers={"X-Agent-ID": "operator"},
        )
    assert proj_resp.status_code == team_resp.status_code == 403


@pytest.mark.asyncio
async def test_mesh_teams_set_goal(v2_app):
    """POST /mesh/teams/{team_name}/goal persists north_star + criteria."""
    app, _, tmp_path = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/teams/research/goal",
            json={
                "north_star": "Ship the alpha",
                "success_criteria": ["First demo by Friday"],
            },
            headers={"X-Agent-ID": "operator"},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True
    assert body["team_name"] == "research"
    assert body["north_star"] == "Ship the alpha"
    meta = yaml.safe_load(
        (tmp_path / "projects" / "research" / "metadata.yaml").read_text()
    )
    assert meta["north_star"] == "Ship the alpha"


@pytest.mark.asyncio
async def test_mesh_tasks_team_id(v2_app):
    """GET /mesh/tasks/team/{team_id} lists the same tasks as /mesh/tasks/project."""
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        await c.post(
            "/mesh/tasks",
            json={"assignee": "analyst", "title": "t1", "project": "research"},
            headers={"X-Agent-ID": "operator"},
        )
        proj_resp = await c.get(
            "/mesh/tasks/project/research",
            headers={"X-Agent-ID": "operator"},
        )
        team_resp = await c.get(
            "/mesh/tasks/team/research",
            headers={"X-Agent-ID": "operator"},
        )
    assert proj_resp.status_code == 200
    assert team_resp.status_code == 200
    assert proj_resp.json()["count"] == team_resp.json()["count"] == 1


@pytest.mark.asyncio
async def test_mesh_teams_status(v2_app):
    """GET /mesh/teams/{team_id}/status returns counts + ``team``/``project`` meta."""
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        await c.post(
            "/mesh/tasks",
            json={"assignee": "analyst", "title": "t1", "project": "research"},
            headers={"X-Agent-ID": "operator"},
        )
        r = await c.get(
            "/mesh/teams/research/status",
            headers={"X-Agent-ID": "operator"},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["counts"]["active"] == 1
    assert body["team"]["name"] == "research"
    assert body["project"]["name"] == "research"


@pytest.mark.asyncio
async def test_mesh_teams_status_all(v2_app):
    """GET /mesh/teams/status returns the same projects as /mesh/projects/status."""
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get(
            "/mesh/teams/status", headers={"X-Agent-ID": "operator"},
        )
    assert r.status_code == 200
    body = r.json()
    # Carries both the legacy ``projects`` key and the new ``teams`` key.
    assert "projects" in body
    assert "teams" in body
    names = {p["project"]["name"] for p in body["projects"]}
    assert names == {"research", "ops"}


@pytest.mark.asyncio
async def test_mesh_teams_outputs(v2_app):
    """GET /mesh/teams/{team_id}/outputs returns completed tasks."""
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/tasks",
            json={"assignee": "analyst", "title": "report", "project": "research"},
            headers={"X-Agent-ID": "operator"},
        )
        tid = r.json()["id"]
        await c.post(
            f"/mesh/tasks/{tid}/status",
            json={"status": "working"},
            headers={"X-Agent-ID": "analyst"},
        )
        await c.post(
            f"/mesh/tasks/{tid}/status",
            json={"status": "done"},
            headers={"X-Agent-ID": "analyst"},
        )
        r = await c.get(
            "/mesh/teams/research/outputs",
            params={"since": "24h"},
            headers={"X-Agent-ID": "operator"},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["team_id"] == "research"
    assert body["project_id"] == "research"
    assert len(body["outputs"]) == 1


@pytest.mark.asyncio
async def test_mesh_teams_summary(v2_app):
    """GET /mesh/teams/{team_id}/summary returns the synthesized status text."""
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get(
            "/mesh/teams/research/summary",
            headers={"X-Agent-ID": "operator"},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["team"]["name"] == "research"
    assert body["project"]["name"] == "research"
    assert "status_text" in body


@pytest.mark.asyncio
async def test_mesh_teams_archive_and_unarchive(v2_app):
    """POST /mesh/teams/{team_name}/archive then unarchive flips the status."""
    app, _, _ = v2_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/teams/research/archive",
            headers={"X-Agent-ID": "operator"},
        )
        assert r.status_code == 200, r.text
        assert r.json()["archived"] is True
        assert r.json()["team_name"] == "research"
        # Default list excludes archived
        r = await c.get(
            "/mesh/teams", headers={"X-Agent-ID": "operator"},
        )
        names = {t["name"] for t in r.json()["teams"]}
        assert "research" not in names
        # Unarchive restores it
        r = await c.post(
            "/mesh/teams/research/unarchive",
            headers={"X-Agent-ID": "operator"},
        )
        assert r.status_code == 200, r.text
        assert r.json()["archived"] is False
        assert r.json()["team_name"] == "research"


@pytest.mark.asyncio
async def test_mesh_teams_propose_delete_requires_archive(v2_app):
    """propose-delete on a live team must fail with 400."""
    app, _, _ = v2_app
    from src.shared.types import MessageOrigin

    origin = MessageOrigin(kind="human", channel="cli", user="u1")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/teams/research/propose-delete",
            headers={
                "X-Agent-ID": "operator",
                "X-Origin": origin.to_header_value(),
            },
        )
    assert r.status_code == 400
    assert "archived" in r.text.lower()


@pytest.mark.asyncio
async def test_mesh_teams_propose_delete_happy_path(v2_app):
    """propose-delete on an archived team returns a nonce + summary."""
    app, _, _ = v2_app
    from src.shared.types import MessageOrigin

    origin = MessageOrigin(kind="human", channel="cli", user="u1")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        await c.post(
            "/mesh/teams/research/archive",
            headers={"X-Agent-ID": "operator"},
        )
        r = await c.post(
            "/mesh/teams/research/propose-delete",
            headers={
                "X-Agent-ID": "operator",
                "X-Origin": origin.to_header_value(),
            },
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["requires_confirmation"] is True
    assert "delete team" in body["summary"].lower()
    assert "'research'" in body["summary"]


@pytest.mark.asyncio
async def test_mesh_costs_team(v2_app):
    """GET /mesh/costs/team/{team} proxies to the same cost helper as /mesh/costs/project.

    Asserts parity with ``/mesh/costs/project`` — both routes call
    ``cost_tracker.get_project_spend`` and surface its return shape.
    """
    app, _, _ = v2_app
    # Locate the cost_tracker MagicMock in the app's closure and pin
    # ``get_project_spend`` to return a real dict so we can compare
    # the proxied output between the two routes.
    cost_tracker = None
    for route in app.routes:
        endpoint = getattr(route, "endpoint", None)
        if endpoint is None:
            continue
        closure = getattr(endpoint, "__closure__", None) or ()
        names = getattr(getattr(endpoint, "__code__", None), "co_freevars", ())
        for name, cell in zip(names, closure):
            if name == "cost_tracker":
                cost_tracker = cell.cell_contents
                break
        if cost_tracker is not None:
            break
    assert cost_tracker is not None, "cost_tracker missing from app closure"
    cost_tracker.get_project_spend = MagicMock(
        return_value={"period": "today", "total_usd": 1.23},
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        proj_resp = await c.get(
            "/mesh/costs/project/research",
            headers={"X-Agent-ID": "operator"},
        )
        team_resp = await c.get(
            "/mesh/costs/team/research",
            headers={"X-Agent-ID": "operator"},
        )
    assert proj_resp.status_code == team_resp.status_code == 200
    assert proj_resp.json() == team_resp.json()


# ── Dashboard /api/teams/* aliases ─────────────────────────────────


def _setup_dashboard_team_fixture(include_v2: bool = True):
    """Build a dashboard test client + tmp dir + components."""
    tmpdir = tempfile.mkdtemp()
    components = _make_components(tmpdir, include_v2=include_v2)
    if "runtime" in components:
        components["runtime"].project_root = MagicMock()
    client = _make_client(components)
    return tmpdir, components, client


def _seed_team_on_disk(tmpdir: str, name: str, members: list[str] | None = None) -> str:
    """Create a project (== team) dir + metadata.yaml under tmpdir."""
    projects_dir = os.path.join(tmpdir, "projects")
    pd = os.path.join(projects_dir, name)
    os.makedirs(pd, exist_ok=True)
    with open(os.path.join(pd, "metadata.yaml"), "w") as f:
        yaml.dump({"name": name, "members": members or []}, f)
    return projects_dir


def _seed_agents_yaml(tmpdir: str, names: list[str]) -> tuple[str, str, str]:
    """Build a minimal config/mesh.yaml + agents.yaml for member validation."""
    config_dir = os.path.join(tmpdir, "config")
    os.makedirs(config_dir, exist_ok=True)
    config_file = os.path.join(config_dir, "mesh.yaml")
    with open(config_file, "w") as f:
        yaml.dump({"agents": {n: {"role": "test"} for n in names}}, f)
    agents_file = os.path.join(config_dir, "agents.yaml")
    with open(agents_file, "w") as f:
        yaml.dump({n: {"role": "test"} for n in names}, f)
    return config_dir, config_file, agents_file


def test_api_teams_list():
    """GET /api/teams returns the same projects as /api/projects."""
    tmpdir, components, client = _setup_dashboard_team_fixture()
    try:
        projects_dir = _seed_team_on_disk(tmpdir, "alpha", ["bot1"])
        with patch("src.cli.config.PROJECTS_DIR", Path(projects_dir)):
            proj_resp = client.get("/dashboard/api/projects")
            team_resp = client.get("/dashboard/api/teams")
        assert proj_resp.status_code == 200
        assert team_resp.status_code == 200
        body = team_resp.json()
        assert "teams" in body
        assert "projects" in body
        proj_names = {p["name"] for p in proj_resp.json()["projects"]}
        team_names = {t["name"] for t in body["teams"]}
        assert proj_names == team_names == {"alpha"}
        for row in body["teams"]:
            assert row["name"] == row["team_name"]
    finally:
        _teardown(components)
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_api_teams_create():
    """POST /api/teams creates a project directory just like /api/projects."""
    tmpdir, components, client = _setup_dashboard_team_fixture()
    try:
        projects_dir = os.path.join(tmpdir, "projects")
        os.makedirs(projects_dir, exist_ok=True)
        config_dir, config_file, agents_file = _seed_agents_yaml(
            tmpdir, ["alpha", "beta"],
        )
        with patch("src.cli.config.PROJECTS_DIR", Path(projects_dir)), \
             patch("src.cli.config.CONFIG_FILE", Path(config_file)), \
             patch("src.cli.config.AGENTS_FILE", Path(agents_file)), \
             patch("src.cli.config.PERMISSIONS_FILE", Path(tmpdir) / "perms.json"):
            resp = client.post(
                "/dashboard/api/teams",
                json={"name": "myteam", "description": "via alias", "members": []},
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["created"] is True
        assert body["name"] == "myteam"
        assert body["team_name"] == "myteam"
        assert os.path.isdir(os.path.join(projects_dir, "myteam"))
    finally:
        _teardown(components)
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_api_teams_delete():
    """DELETE /api/teams/{team_name} removes the team directory."""
    tmpdir, components, client = _setup_dashboard_team_fixture()
    try:
        projects_dir = _seed_team_on_disk(tmpdir, "doomed")
        with patch("src.cli.config.PROJECTS_DIR", Path(projects_dir)), \
             patch("src.cli.config.PERMISSIONS_FILE", Path(tmpdir) / "perms.json"):
            resp = client.delete("/dashboard/api/teams/doomed")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["deleted"] is True
        assert body["team_name"] == "doomed"
        assert not os.path.exists(os.path.join(projects_dir, "doomed"))
    finally:
        _teardown(components)
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_api_teams_add_member():
    """POST /api/teams/{team_name}/members adds the agent to metadata."""
    tmpdir, components, client = _setup_dashboard_team_fixture()
    try:
        projects_dir = _seed_team_on_disk(tmpdir, "squad", members=[])
        with patch("src.cli.config.PROJECTS_DIR", Path(projects_dir)), \
             patch("src.cli.config.PERMISSIONS_FILE", Path(tmpdir) / "perms.json"):
            resp = client.post(
                "/dashboard/api/teams/squad/members", json={"agent": "alpha"},
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["added"] is True
        assert body["team_name"] == "squad"
        assert body["agent"] == "alpha"
        meta = yaml.safe_load(
            Path(projects_dir, "squad", "metadata.yaml").read_text()
        )
        assert "alpha" in meta["members"]
    finally:
        _teardown(components)
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_api_teams_remove_member():
    """DELETE /api/teams/{team_name}/members/{agent} removes the agent."""
    tmpdir, components, client = _setup_dashboard_team_fixture()
    try:
        projects_dir = _seed_team_on_disk(tmpdir, "squad", members=["alpha"])
        with patch("src.cli.config.PROJECTS_DIR", Path(projects_dir)), \
             patch("src.cli.config.PERMISSIONS_FILE", Path(tmpdir) / "perms.json"):
            resp = client.delete("/dashboard/api/teams/squad/members/alpha")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["removed"] is True
        assert body["team_name"] == "squad"
        meta = yaml.safe_load(
            Path(projects_dir, "squad", "metadata.yaml").read_text()
        )
        assert "alpha" not in meta["members"]
    finally:
        _teardown(components)
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_api_team_read():
    """GET /api/team?team=... reads the team's project.md."""
    tmpdir, components, client = _setup_dashboard_team_fixture()
    try:
        projects_dir = _seed_team_on_disk(tmpdir, "alpha", members=["bot1"])
        # Drop a project.md in place
        Path(projects_dir, "alpha", "project.md").write_text(
            "# Alpha Team\nShared context here.\n"
        )
        with patch("src.cli.config.PROJECTS_DIR", Path(projects_dir)):
            resp = client.get(
                "/dashboard/api/team", params={"team": "alpha"},
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["team"] == "alpha"
        assert body["project"] == "alpha"
        assert "Alpha Team" in body["content"]
    finally:
        _teardown(components)
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_api_team_write():
    """PUT /api/team?team=... writes the team's project.md."""
    tmpdir, components, client = _setup_dashboard_team_fixture()
    try:
        projects_dir = _seed_team_on_disk(tmpdir, "alpha", members=[])
        with patch("src.cli.config.PROJECTS_DIR", Path(projects_dir)):
            resp = client.put(
                "/dashboard/api/team", params={"team": "alpha"},
                json={"content": "Updated team body."},
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["saved"] is True
        text = Path(projects_dir, "alpha", "project.md").read_text()
        assert "Updated team body." in text
    finally:
        _teardown(components)
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_api_workplace_teams_disabled_without_v2():
    """GET /api/workplace/teams returns enabled=False when tasks_store is missing."""
    # Use the default (no v2) fixture so tasks_store is None.
    tmpdir, components, client = _setup_dashboard_team_fixture(include_v2=False)
    try:
        resp = client.get("/dashboard/api/workplace/teams")
        assert resp.status_code == 200
        body = resp.json()
        assert body["enabled"] is False
        # Both keys are present even when disabled, for client back-compat.
        assert body["teams"] == []
        assert body["projects"] == []
    finally:
        _teardown(components)
        shutil.rmtree(tmpdir, ignore_errors=True)
