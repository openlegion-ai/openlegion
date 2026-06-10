"""Tests for the M3 wake task-id binding gate on ``POST /mesh/wake``.

The wake endpoint threads an ``x-task-id`` header into the lane payload
so the recipient's loop auto-closes the task on return. Finding M3
requires that this only happens when the named task is actually assigned
to the wake ``target`` — otherwise a caller could ride a forged/mismatched
task_id and cause the recipient's loop to auto-close an UNRELATED agent's
task.

Binding rule:
  * ``assignee == target`` → task_id threaded (legit handoff auto-close).
  * ``assignee != target`` or task missing → task_id dropped, but the
    wake itself still proceeds (no rejection).

The fixture boots a real mesh app with a fake lane + a real dispatch
loop (so ``run_coroutine_threadsafe`` lands) and inspects the enqueue
kwargs the endpoint produced.
"""

from __future__ import annotations

import asyncio
import importlib
import threading
import time

import pytest
from httpx import ASGITransport, AsyncClient

from src.host.mesh import Blackboard, MessageRouter, PubSub
from src.host.permissions import PermissionMatrix
from src.shared.types import AgentPermissions


def _reload_server(monkeypatch, *, tasks_db: str):
    monkeypatch.setenv("OPENLEGION_ORCHESTRATION_TASKS_DB", tasks_db)
    import src.host.server as server_module
    importlib.reload(server_module)
    return server_module


class _FakeLane:
    """Minimal LaneManager stand-in that records enqueue calls."""

    def __init__(self):
        self.calls: list[dict] = []

    async def enqueue(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        return "ok"


@pytest.fixture
def wake_app(tmp_path, monkeypatch):
    server_module = _reload_server(
        monkeypatch, tasks_db=str(tmp_path / "tasks.db"),
    )

    blackboard = Blackboard(str(tmp_path / "bb.db"))
    pubsub = PubSub()
    permissions = PermissionMatrix()
    # ``can_message=["*"]`` so the wake's can_message gate is satisfied;
    # the test targets the task-id binding, not the messaging gate.
    for aid in ("scout", "writer", "analyst"):
        permissions.permissions[aid] = AgentPermissions(
            agent_id=aid, can_message=["*"],
        )
    router = MessageRouter(permissions, {
        "scout":   "http://scout:8400",
        "writer":  "http://writer:8400",
        "analyst": "http://analyst:8400",
    })

    lane = _FakeLane()
    loop = asyncio.new_event_loop()

    def _run():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    app = server_module.create_mesh_app(
        blackboard=blackboard,
        pubsub=pubsub,
        router=router,
        permissions=permissions,
        lane_manager=lane,  # type: ignore[arg-type]
        dispatch_loop=loop,
    )
    # Grab the tasks store the app wired so the test can seed task rows.
    tasks_store = app.tasks_store
    yield app, lane, tasks_store
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=2)
    blackboard.close()
    loop.close()
    monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_DB", raising=False)
    importlib.reload(server_module)


def _settle(seconds: float = 0.05) -> None:
    time.sleep(seconds)


@pytest.mark.asyncio
async def test_legit_handoff_threads_task_id(wake_app):
    """assignee == target → task_id is threaded so the recipient's loop
    auto-closes the task (the legitimate hand_off auto-close path)."""
    app, lane, tasks_store = wake_app
    # scout hands off to writer: create_task(assignee=writer), then wake
    # writer with that task_id.
    rec = tasks_store.create(
        creator="scout", assignee="writer", title="stage work",
    )
    task_id = rec["id"]

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post(
            "/mesh/wake",
            params={"target": "writer", "message": "new task"},
            headers={"X-Agent-ID": "scout", "x-task-id": task_id},
        )
    assert resp.status_code == 200
    assert resp.json()["woken"] is True
    _settle()

    assert len(lane.calls) == 1
    # task_id was threaded — auto-close works for the legit handoff.
    assert lane.calls[0]["kwargs"].get("task_id") == task_id


@pytest.mark.asyncio
async def test_mismatched_task_id_dropped_not_rejected(wake_app):
    """assignee != target → wake still succeeds but the task_id is
    dropped so no UNRELATED task is auto-closed."""
    app, lane, tasks_store = wake_app
    # Task is assigned to analyst, but the wake targets writer.
    rec = tasks_store.create(
        creator="scout", assignee="analyst", title="analyst's task",
    )
    other_task_id = rec["id"]

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post(
            "/mesh/wake",
            params={"target": "writer", "message": "nudge"},
            headers={"X-Agent-ID": "scout", "x-task-id": other_task_id},
        )
    # Wake is NOT rejected.
    assert resp.status_code == 200
    assert resp.json()["woken"] is True
    _settle()

    assert len(lane.calls) == 1
    # task_id was DROPPED — analyst's task will not be auto-closed by
    # writer's loop.
    assert lane.calls[0]["kwargs"].get("task_id") is None


@pytest.mark.asyncio
async def test_unknown_task_id_dropped(wake_app):
    """A task_id that does not resolve to any task is dropped (wake still
    proceeds)."""
    app, lane, _tasks_store = wake_app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post(
            "/mesh/wake",
            params={"target": "writer", "message": "nudge"},
            headers={"X-Agent-ID": "scout", "x-task-id": "does-not-exist"},
        )
    assert resp.status_code == 200
    _settle()

    assert len(lane.calls) == 1
    assert lane.calls[0]["kwargs"].get("task_id") is None


@pytest.mark.asyncio
async def test_wake_message_carries_task_brief(wake_app):
    """B2: when the wake is bound to a task (M3 check passed), the lane
    message is enriched with the task description so the recipient's chat
    turn starts from the full brief, not the title-sized wake stub."""
    app, lane, tasks_store = wake_app
    brief = (
        "## Objective\nDeep SEO audit of example.com\n"
        "## Deliverable\nFull written audit saved as an artifact."
    )
    rec = tasks_store.create(
        creator="scout", assignee="writer",
        title="Deep SEO audit", description=brief,
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post(
            "/mesh/wake",
            params={"target": "writer", "message": "New task from scout: Deep SEO audit"},
            headers={"X-Agent-ID": "scout", "x-task-id": rec["id"]},
        )
    assert resp.status_code == 200
    _settle()

    assert len(lane.calls) == 1
    lane_msg = lane.calls[0]["args"][1]
    assert "## Task Brief" in lane_msg
    assert "Deep SEO audit of example.com" in lane_msg
    assert "saved as an artifact" in lane_msg


@pytest.mark.asyncio
async def test_wake_message_skips_brief_when_description_is_title(wake_app):
    """Legacy handoffs (description == title) gain no redundant brief block."""
    app, lane, tasks_store = wake_app
    rec = tasks_store.create(
        creator="scout", assignee="writer",
        title="quick ping", description="quick ping",
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        await client.post(
            "/mesh/wake",
            params={"target": "writer", "message": "New task from scout: quick ping"},
            headers={"X-Agent-ID": "scout", "x-task-id": rec["id"]},
        )
    _settle()

    lane_msg = lane.calls[0]["args"][1]
    assert "## Task Brief" not in lane_msg


@pytest.mark.asyncio
async def test_wake_message_lists_artifact_refs(wake_app):
    """Data-payload pointers ride the wake so the recipient knows to fetch."""
    app, lane, tasks_store = wake_app
    rec = tasks_store.create(
        creator="scout", assignee="writer", title="stage work",
        artifact_refs=["output/scout/ho_123"],
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        await client.post(
            "/mesh/wake",
            params={"target": "writer", "message": "New task from scout: stage work"},
            headers={"X-Agent-ID": "scout", "x-task-id": rec["id"]},
        )
    _settle()

    lane_msg = lane.calls[0]["args"][1]
    assert "read_blackboard" in lane_msg
    assert "output/scout/ho_123" in lane_msg


@pytest.mark.asyncio
async def test_dropped_task_id_means_no_brief_enrichment(wake_app):
    """A mismatched task_id is dropped by the M3 gate — the brief of an
    UNRELATED task must not leak into the wake message either."""
    app, lane, tasks_store = wake_app
    rec = tasks_store.create(
        creator="scout", assignee="analyst",  # NOT the wake target
        title="other work", description="secret brief for analyst",
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        await client.post(
            "/mesh/wake",
            params={"target": "writer", "message": "nudge"},
            headers={"X-Agent-ID": "scout", "x-task-id": rec["id"]},
        )
    _settle()

    lane_msg = lane.calls[0]["args"][1]
    assert "secret brief" not in lane_msg
    assert lane.calls[0]["kwargs"].get("task_id") is None
