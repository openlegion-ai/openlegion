"""End-to-end integration test for the operator ``rate_delivery`` tool.

Closes the Codex r1 nice-to-have gap from PR 2 of the Work tab rewrite:
the existing ``test_operator_rate_delivery.py`` suite mocks the mesh
client, so the rework-spawn path is never exercised end-to-end.

This single test drives the full chain:

    rate_delivery(outcome="rework", ...)
        → MeshClient.set_task_outcome
        → POST /mesh/tasks/{id}/outcome on the in-process mesh app
        → Tasks.set_outcome + Tasks.create_rework_task

and asserts both side effects:

* The original task row carries ``outcome="rework"`` and the feedback.
* A follow-up task was spawned with ``previous_task_id`` pointing back
  to the original.
"""

from __future__ import annotations

import importlib
import os
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.host.mesh import Blackboard, MessageRouter, PubSub
from src.host.orchestration import Tasks
from src.host.permissions import PermissionMatrix


@pytest.fixture
def _operator_env(monkeypatch):
    """Operator gating is environment-driven — pin the operator id +
    expose ``rate_delivery`` so the per-tool ALLOWED_TOOLS filter on
    skill execution doesn't intercept the call.
    """
    monkeypatch.setenv("ALLOWED_TOOLS", "rate_delivery")
    yield


@pytest.fixture
def _mesh_app(tmp_path, monkeypatch):
    """Boot a fresh mesh app pinned to ``tmp_path/tasks.db`` so the
    test can read what ``set_outcome`` + ``create_rework_task`` wrote.
    """
    monkeypatch.setenv(
        "OPENLEGION_ORCHESTRATION_TASKS_DB",
        str(tmp_path / "tasks.db"),
    )
    import src.host.server as server_mod
    importlib.reload(server_mod)

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix()
    router = MessageRouter(perms, {})

    app = server_mod.create_mesh_app(
        blackboard=bb, pubsub=pubsub, router=router, permissions=perms,
    )
    # Use the same db file the mesh wired up so test asserts see the
    # writes the endpoint handler made.
    tasks_store = Tasks(db_path=str(tmp_path / "tasks.db"))

    try:
        yield app, tasks_store
    finally:
        try:
            tasks_store.close()
        finally:
            bb.close()
            os.environ.pop("OPENLEGION_ORCHESTRATION_TASKS_DB", None)
            importlib.reload(server_mod)


@pytest.mark.asyncio
async def test_rate_delivery_rework_spawns_linked_task_end_to_end(
    _operator_env, _mesh_app,
):
    """Tool → mesh client → mesh endpoint → tasks store, fully wired.

    The mesh client's internal httpx.AsyncClient is patched onto an
    ASGI transport so the call lands on the actual FastAPI app without
    needing a live server. ``rate_delivery`` MUST surface
    ``rework_task_id`` and the persisted row MUST carry
    ``previous_task_id`` linking back to the rated task — that's the
    machine-loop guarantee the mock-only PR 2 tests don't cover.
    """
    app, tasks_store = _mesh_app

    # Seed a completed task that's eligible to rate.
    seed = tasks_store.create(
        creator="operator", assignee="analyst",
        title="research the legal angle",
        project_id="research",
        description="initial brief",
    )
    tasks_store.update_status(seed["id"], "working", actor="analyst")
    tasks_store.update_status(seed["id"], "done", actor="analyst")

    from src.agent.builtins.operator_tools import rate_delivery
    from src.agent.mesh_client import MeshClient

    transport = ASGITransport(app=app)

    @asynccontextmanager
    async def _client_ctx():
        async with AsyncClient(
            transport=transport, base_url="http://mesh",
            headers={"x-mesh-internal": "1", "X-Agent-ID": "operator"},
        ) as c:
            yield c

    mesh_client = MeshClient(mesh_url="http://mesh", agent_id="operator")

    # The mesh client lazily builds an httpx.AsyncClient. Patch
    # ``_get_client`` to return our ASGI-backed client so every call
    # the tool makes goes straight to the in-process app.
    async with _client_ctx() as asgi_client:
        with patch.object(
            mesh_client, "_get_client",
            new=AsyncMock(return_value=asgi_client),
        ):
            result = await rate_delivery(
                task_id=seed["id"],
                outcome="rework",
                feedback="please redo with deeper legal sourcing",
                mesh_client=mesh_client,
            )

    # Tool-level surface.
    assert result.get("ok") is True
    assert result["task_id"] == seed["id"]
    assert result["outcome"] == "rework"
    rework_id = result.get("rework_task_id")
    assert rework_id, f"rework_task_id missing from result: {result!r}"
    assert result.get("rework_assignee") == "analyst"

    # Original task row: outcome + feedback persisted.
    rated = tasks_store.get(seed["id"])
    assert rated is not None
    assert rated["outcome"] == "rework"
    assert rated["feedback_text"] == (
        "please redo with deeper legal sourcing"
    )

    # Spawned rework task: links back to the original.
    spawned = tasks_store.get(rework_id)
    assert spawned is not None, "rework task not persisted"
    assert spawned["previous_task_id"] == seed["id"]
    assert spawned["assignee"] == "analyst"
    assert spawned["project_id"] == "research"
    # Title prefix is the contract the dashboard's rework-card UI relies
    # on (see ``test_workplace_outcome.py::test_outcome_rework_spawns_linked_task``).
    assert spawned["title"].startswith("Rework: ")
    # The feedback becomes the new task's description so the assignee
    # has the brief without a round-trip.
    assert spawned["description"] == (
        "please redo with deeper legal sourcing"
    )
