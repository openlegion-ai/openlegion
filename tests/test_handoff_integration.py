"""Integration tests: ``hand_off`` → ``list_task_inbox``.

Canaries for Bug 1: agent A's hand_off must produce a row that the
recipient's inbox returns. Two layers of coverage so a regression
at EITHER layer fails loudly:

* Store-level canary (the original test below): real :class:`Tasks`
  store wired to a mock ``mesh_client``. Drives ``hand_off`` →
  ``store.create`` → ``store.list_inbox`` without going through HTTP,
  catching store-side parent_task_id / assignee regressions.
* HTTP-level canary (the ASGI test below): real mesh app behind
  ``ASGITransport``. Drives POST ``/mesh/tasks`` → endpoint → store →
  GET ``/mesh/tasks/inbox/{assignee}``, catching endpoint-side strip /
  verify / permission regressions that the store-level test would miss.
"""

from __future__ import annotations

import importlib
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from src.host.mesh import Blackboard, MessageRouter, PubSub
from src.host.permissions import PermissionMatrix
from src.shared.types import AgentPermissions


@pytest.mark.asyncio
async def test_hand_off_creates_task_visible_to_recipient_inbox(monkeypatch):
    """End-to-end: ``hand_off`` from scout to bob lands a row that
    ``list_inbox(bob)`` returns immediately.

    Assertions:
      * exactly one row in bob's inbox
      * ``assignee == "bob"`` (whitespace-stripped, byte-exact)
      * ``parent_task_id`` matches the seeded contextvar
      * ``status == "pending"``
      * ``hand_off`` reports ``handed_off=True`` so the caller does not
        emit a ``create_failed`` envelope.
    """
    from src.agent.builtins.coordination_tool import hand_off
    from src.host.orchestration import Tasks
    from src.shared.trace import current_task_id

    # Real Tasks store, shared in-memory connection so writes/reads see
    # the same SQLite database across both hand_off's create + our inbox
    # readback.
    store = Tasks(db_path=":memory:")

    # Mock mesh_client whose async surface proxies the relevant calls
    # into the durable store.
    mc = MagicMock()
    mc.agent_id = "scout"
    mc.is_standalone = False
    mc.team_name = "default"
    mc.project_name = "default"

    # Roster — bob has to exist for hand_off's pre-flight validation.
    mc.list_agents = AsyncMock(return_value={
        "scout": {"role": "scout", "team": "default"},
        "bob": {"role": "analyst", "team": "default"},
    })

    # ``create_task`` proxies into ``store.create``. ``hand_off`` invokes
    # this with kw-args matching the MeshClient.create_task signature.
    async def fake_create_task(*, assignee, title, description=None,
                               team_id=None, parent_task_id=None,
                               priority=0, dependencies=None,
                               artifact_refs=None, origin=None,
                               thinking=None):
        return store.create(
            creator=mc.agent_id,
            assignee=assignee,
            title=title,
            description=description,
            team_id=team_id,
            parent_task_id=parent_task_id,
            priority=priority,
            dependencies=dependencies,
            artifact_refs=artifact_refs,
            origin=None,
            thinking=thinking,
        )
    mc.create_task = AsyncMock(side_effect=fake_create_task)

    # Wake/write are best-effort decorations — return success-shaped dicts
    # so hand_off proceeds through the happy path.
    mc.wake_agent = AsyncMock(return_value={"woken": True})
    mc.write_blackboard = AsyncMock(return_value={"version": 1})

    # Seed a known parent task contextvar so the propagation invariant
    # can be asserted on the stored row.
    parent_token = current_task_id.set("task_root_parent")
    try:
        result = await hand_off(
            to="bob", summary="do the thing", mesh_client=mc,
        )
    finally:
        current_task_id.reset(parent_token)

    # hand_off succeeded.
    assert result.get("handed_off") is True, result
    assert result.get("create_failed") is not True

    # Recipient's inbox sees exactly one row with the right shape.
    rows = store.list_inbox("bob")
    assert len(rows) == 1, rows
    row = rows[0]
    assert row["assignee"] == "bob"
    assert row["parent_task_id"] == "task_root_parent"
    assert row["status"] == "pending"
    assert row["creator"] == "scout"


# ── HTTP-level integration canary (codex R5) ─────────────────────


def _reload_mesh_server(monkeypatch, *, tasks_db: str):
    monkeypatch.setenv("OPENLEGION_ORCHESTRATION_TASKS_DB", tasks_db)
    import src.host.server as server_module
    importlib.reload(server_module)
    return server_module


@pytest.fixture
def asgi_mesh(tmp_path, monkeypatch):
    """Mesh app behind ASGITransport with two agents in one project."""
    server_module = _reload_mesh_server(
        monkeypatch, tasks_db=str(tmp_path / "tasks.db"),
    )
    blackboard = Blackboard(str(tmp_path / "bb.db"))
    pubsub = PubSub()
    permissions = PermissionMatrix()
    for aid in ("scout", "analyst"):
        permissions.permissions[aid] = AgentPermissions(
            agent_id=aid, can_route_tasks=True,
            can_message=["scout", "analyst"],
        )
    router = MessageRouter(permissions, {
        "scout": "http://scout:8400",
        "analyst": "http://analyst:8400",
    })
    app = server_module.create_mesh_app(
        blackboard=blackboard, pubsub=pubsub,
        router=router, permissions=permissions,
    )
    app.teams_store.create_team("research")
    app.teams_store.add_member("research", "scout")
    app.teams_store.add_member("research", "analyst")
    yield app
    blackboard.close()
    monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_DB", raising=False)
    importlib.reload(server_module)


@pytest.mark.asyncio
async def test_http_handoff_round_trip_lands_in_recipient_inbox(asgi_mesh):
    """End-to-end through the HTTP layer: POST /mesh/tasks as scout for
    analyst, then GET /mesh/tasks/inbox/analyst — the row must be there.

    This is the codex R5 canary. The store-level test above can pass
    while an endpoint-side regression (e.g. someone reverts the post-
    write verify or breaks the assignee.strip normalization) silently
    breaks every real handoff in production. Drive the full HTTP chain
    so that class of regression fails loudly here.
    """
    transport = ASGITransport(app=asgi_mesh)
    async with AsyncClient(transport=transport, base_url="http://t") as c:
        # scout creates a task for analyst.
        post = await c.post(
            "/mesh/tasks",
            json={
                "assignee": "analyst",
                "title": "Investigate Q3 funnel",
                "parent_task_id": "task_kickoff_root",
            },
            headers={"X-Agent-ID": "scout"},
        )
        assert post.status_code == 200, post.text
        created = post.json()
        assert created["assignee"] == "analyst"
        assert created["creator"] == "scout"
        assert created["parent_task_id"] == "task_kickoff_root"
        assert created["status"] == "pending"

        # analyst's inbox must surface the row immediately.
        inbox = await c.get(
            "/mesh/tasks/inbox/analyst",
            headers={"X-Agent-ID": "analyst"},
        )
        assert inbox.status_code == 200, inbox.text
        payload = inbox.json()
        assert payload["count"] == 1
        assert payload["tasks"][0]["id"] == created["id"]
        assert payload["tasks"][0]["assignee"] == "analyst"
        assert payload["tasks"][0]["status"] == "pending"


@pytest.mark.asyncio
async def test_http_handoff_assignee_with_whitespace_normalized_and_visible(
    asgi_mesh,
):
    """Whitespace-padded assignee at the wire is stripped before storage,
    so the recipient's byte-exact SQLite ``=`` lookup matches. Bug 1
    repros consistent with a single stray space breaking ``list_inbox``."""
    transport = ASGITransport(app=asgi_mesh)
    async with AsyncClient(transport=transport, base_url="http://t") as c:
        post = await c.post(
            "/mesh/tasks",
            json={"assignee": "  analyst  ", "title": "Padded"},
            headers={"X-Agent-ID": "scout"},
        )
        assert post.status_code == 200, post.text
        assert post.json()["assignee"] == "analyst"
        inbox = await c.get(
            "/mesh/tasks/inbox/analyst",
            headers={"X-Agent-ID": "analyst"},
        )
        assert inbox.status_code == 200
        assert inbox.json()["count"] == 1
