"""P2 tests — fleet-wide knowledge propagation via section-scoped TEAM.md
updates, and rating history in team summaries.

Covers:
* ``replace_markdown_section`` (pure helper).
* ``PUT /mesh/teams/{team}/brief`` — auth, validation, section
  replace/append, push to running members.
* ``update_team_brief`` operator tool surface.
* ``GET /mesh/teams/{id}/summary?hours=N`` → ``outcomes_window``.
"""

from __future__ import annotations

import importlib
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from src.host.mesh import Blackboard, MessageRouter, PubSub
from src.host.permissions import PermissionMatrix
from src.shared.utils import replace_markdown_section

# ── replace_markdown_section ──────────────────────────────────────


class TestReplaceMarkdownSection:
    def test_appends_when_absent(self):
        out = replace_markdown_section(
            "# team\n\nintro text\n", "User Preferences", "- formal tone",
        )
        assert out.endswith("## User Preferences\n\n- formal tone\n")
        assert "intro text" in out

    def test_replaces_only_the_named_section(self):
        text = (
            "# team\n\n"
            "## User Preferences\n\nold prefs\n\n"
            "## Workflow\n\nkeep me\n"
        )
        out = replace_markdown_section(text, "User Preferences", "new prefs")
        assert "old prefs" not in out
        assert "new prefs" in out
        assert "keep me" in out
        assert out.count("## User Preferences") == 1

    def test_replaces_trailing_section(self):
        text = "# team\n\n## Workflow\n\nkeep\n\n## User Preferences\n\nold\n"
        out = replace_markdown_section(text, "User Preferences", "new")
        assert "old" not in out
        assert "keep" in out
        assert out.rstrip().endswith("new")

    def test_backslashes_in_content_are_literal(self):
        out = replace_markdown_section(
            "## X\n\nold\n", "X", r"path C:\new\g<0>",
        )
        assert r"C:\new" in out

    def test_empty_document(self):
        out = replace_markdown_section("", "User Preferences", "prefs")
        assert out == "## User Preferences\n\nprefs\n"


# ── mesh endpoint ─────────────────────────────────────────────────


class _FakeTransport:
    def __init__(self):
        self.calls: list[dict] = []

    async def request(self, agent_id, method, path, json=None, timeout=120, headers=None):
        self.calls.append({
            "agent_id": agent_id, "method": method,
            "path": path, "json": json,
        })
        return {"updated": True}


@pytest.fixture
def brief_app(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "OPENLEGION_ORCHESTRATION_TASKS_DB", str(tmp_path / "tasks.db"),
    )
    # Point TEAMS_DIR at tmp by patching the config module attribute.
    import src.cli.config as config_module
    import src.host.server as server_module
    importlib.reload(server_module)
    projects_dir = tmp_path / "projects"
    research = projects_dir / "research"
    research.mkdir(parents=True)
    (research / "team.md").write_text(
        "# research\n\nShared context.\n\n## Workflow\n\nscout -> analyst\n",
    )
    monkeypatch.setattr(config_module, "TEAMS_DIR", projects_dir)

    blackboard = Blackboard(str(tmp_path / "bb.db"))
    permissions = PermissionMatrix()
    router = MessageRouter(permissions, {
        "scout": "http://scout:8400",
        # analyst is a member but NOT running (not in registry).
        "operator": "http://operator:8400",
    })
    fake_transport = _FakeTransport()
    app = server_module.create_mesh_app(
        blackboard=blackboard,
        pubsub=PubSub(),
        router=router,
        permissions=permissions,
        transport=fake_transport,  # type: ignore[arg-type]
    )
    app.teams_store.create_team("research")
    app.teams_store.add_member("research", "scout")
    app.teams_store.add_member("research", "analyst")
    yield app, fake_transport, research / "team.md"
    blackboard.close()
    monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_DB", raising=False)
    importlib.reload(server_module)


@pytest.mark.asyncio
async def test_brief_update_replaces_section_and_pushes(brief_app):
    app, transport, project_md = brief_app
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t",
    ) as c:
        r = await c.put(
            "/mesh/teams/research/brief",
            json={
                "section": "User Preferences",
                "content": "- Formal tone\n- UK English",
            },
            headers={"X-Agent-ID": "operator"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["updated"] is True
    assert body["section"] == "User Preferences"
    # Only running members got the push (scout, not analyst).
    assert body["pushed"] == {"scout": True}
    assert len(transport.calls) == 1
    assert transport.calls[0]["agent_id"] == "scout"
    assert transport.calls[0]["path"] == "/team"
    # File: new section appended, existing sections preserved.
    on_disk = project_md.read_text()
    assert "## User Preferences" in on_disk
    assert "- Formal tone" in on_disk
    assert "scout -> analyst" in on_disk
    # The pushed content matches the file.
    assert transport.calls[0]["json"]["content"] == on_disk


@pytest.mark.asyncio
async def test_brief_update_is_section_scoped_on_second_write(brief_app):
    app, _transport, project_md = brief_app
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t",
    ) as c:
        await c.put(
            "/mesh/teams/research/brief",
            json={"section": "User Preferences", "content": "v1"},
            headers={"X-Agent-ID": "operator"},
        )
        await c.put(
            "/mesh/teams/research/brief",
            json={"section": "User Preferences", "content": "v2"},
            headers={"X-Agent-ID": "operator"},
        )
    on_disk = project_md.read_text()
    assert "v2" in on_disk
    assert "v1" not in on_disk
    assert on_disk.count("## User Preferences") == 1
    assert "scout -> analyst" in on_disk


@pytest.mark.asyncio
async def test_brief_update_validation(brief_app):
    app, _transport, _ = brief_app
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t",
    ) as c:
        # Worker callers rejected.
        r = await c.put(
            "/mesh/teams/research/brief",
            json={"section": "X", "content": "y"},
            headers={"X-Agent-ID": "scout"},
        )
        assert r.status_code == 403
        # Bad section name.
        r = await c.put(
            "/mesh/teams/research/brief",
            json={"section": "## sneaky\nheader", "content": "y"},
            headers={"X-Agent-ID": "operator"},
        )
        assert r.status_code == 400
        # Oversized content.
        r = await c.put(
            "/mesh/teams/research/brief",
            json={"section": "X", "content": "z" * 2001},
            headers={"X-Agent-ID": "operator"},
        )
        assert r.status_code == 400
        # Unknown team.
        r = await c.put(
            "/mesh/teams/nope/brief",
            json={"section": "X", "content": "y"},
            headers={"X-Agent-ID": "operator"},
        )
        assert r.status_code == 404


@pytest.mark.asyncio
async def test_team_context_update_now_pushes(brief_app):
    """P2 gap fix: the mesh-side context writer pushes to members."""
    app, transport, project_md = brief_app
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t",
    ) as c:
        r = await c.put(
            "/mesh/teams/research/context",
            json={"context": "fresh shared context"},
            headers={"X-Agent-ID": "operator"},
        )
    assert r.status_code == 200
    assert r.json()["pushed"] == {"scout": True}
    assert "fresh shared context" in project_md.read_text()


# ── outcomes_window on team summary ───────────────────────────────


@pytest.mark.asyncio
async def test_team_summary_hours_param_adds_outcomes_window(brief_app):
    app, _transport, _ = brief_app
    store = app.tasks_store
    rec = store.create(
        creator="operator", assignee="scout", title="t1",
        team_id="research",
    )
    store.update_status(rec["id"], "working", actor="scout")
    store.update_status(rec["id"], "done", actor="scout")
    store.set_outcome(rec["id"], "rework", "redo", actor="operator")

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t",
    ) as c:
        with_hours = await c.get(
            "/mesh/teams/research/summary",
            params={"hours": 24},
            headers={"X-Agent-ID": "operator"},
        )
        without_hours = await c.get(
            "/mesh/teams/research/summary",
            headers={"X-Agent-ID": "operator"},
        )
    assert with_hours.status_code == 200
    assert with_hours.json()["outcomes_window"] == {"rework": 1}
    assert "outcomes_window" not in without_hours.json()


# ── operator tool surface ─────────────────────────────────────────


@pytest.fixture(autouse=True)
def _set_operator_env(monkeypatch):
    monkeypatch.setenv("ALLOWED_TOOLS", "update_team_brief")


@pytest.mark.asyncio
async def test_update_team_brief_tool_passthrough():
    from src.agent.builtins.operator_tools import update_team_brief

    mc = MagicMock()
    mc.update_team_brief = AsyncMock(return_value={"updated": True})

    result = await update_team_brief(
        team_name="research", section="User Preferences",
        content="- formal", mesh_client=mc,
    )
    assert result == {"updated": True}
    mc.update_team_brief.assert_awaited_once_with(
        "research", "User Preferences", "- formal",
    )


@pytest.mark.asyncio
async def test_update_team_brief_tool_requires_all_args():
    from src.agent.builtins.operator_tools import update_team_brief

    result = await update_team_brief(
        team_name="research", section="", content="x",
        mesh_client=MagicMock(),
    )
    assert "error" in result
