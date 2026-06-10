"""Workstream A tests — closing the rating → learning feedback loop.

Covers:
* A1 — ``push_outcome_feedback`` helper + agent ``POST /learnings/feedback``
  endpoint + the mesh outcome endpoint pushing on rework/rejected.
* A3 — rework briefs carry the original description; retry briefs carry
  the previous attempt's blocker_note.
* A4 — ``_format_fleet_health`` digest rendering.
* A6 — task-level failed/blocked closes write the learnings errors file.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.host.feedback_push import push_outcome_feedback

# ── A1: push helper ───────────────────────────────────────────────


class _FakeTransport:
    def __init__(self, response=None, exc=None):
        self.calls: list[dict] = []
        self._response = response if response is not None else {"recorded": True}
        self._exc = exc

    async def request(self, agent_id, method, path, json=None, timeout=120, headers=None):
        self.calls.append({
            "agent_id": agent_id, "method": method, "path": path, "json": json,
        })
        if self._exc:
            raise self._exc
        return self._response


@pytest.mark.asyncio
async def test_push_feedback_on_rework():
    transport = _FakeTransport()
    record = {"id": "task_1", "title": "SEO audit", "assignee": "analyst"}

    status = await push_outcome_feedback(transport, record, "rework", "too shallow")

    assert status == "recorded"
    assert len(transport.calls) == 1
    call = transport.calls[0]
    assert call["agent_id"] == "analyst"
    assert call["path"] == "/learnings/feedback"
    assert call["json"]["outcome"] == "rework"
    assert call["json"]["feedback"] == "too shallow"


@pytest.mark.asyncio
async def test_push_skipped_for_non_actionable_outcomes():
    transport = _FakeTransport()
    record = {"id": "task_1", "assignee": "analyst"}

    assert await push_outcome_feedback(transport, record, "accepted", "nice") is None
    assert await push_outcome_feedback(transport, record, "acknowledged", "ok") is None
    assert await push_outcome_feedback(transport, record, "rework", "") is None
    assert transport.calls == []


@pytest.mark.asyncio
async def test_push_failure_is_best_effort():
    transport = _FakeTransport(exc=RuntimeError("container down"))
    record = {"id": "task_1", "assignee": "analyst"}

    status = await push_outcome_feedback(transport, record, "rejected", "wrong data")
    assert status == "failed"

    # transport returning an {"error": ...} dict (HTTPTransport contract)
    transport2 = _FakeTransport(response={"error": "HTTP 503"})
    status2 = await push_outcome_feedback(transport2, record, "rejected", "wrong data")
    assert status2 == "failed"


@pytest.mark.asyncio
async def test_push_skipped_without_transport_or_assignee():
    assert await push_outcome_feedback(None, {"assignee": "a"}, "rework", "x") is None
    transport = _FakeTransport()
    assert await push_outcome_feedback(transport, {}, "rework", "x") is None
    assert transport.calls == []


# ── A3: rework brief carries the original ask ─────────────────────


def _make_store(tmp_path):
    from src.host.orchestration import Tasks
    return Tasks(db_path=str(tmp_path / "tasks.db"))


def test_rework_brief_contains_feedback_then_original(tmp_path):
    store = _make_store(tmp_path)
    rec = store.create(
        creator="operator", assignee="analyst",
        title="SEO audit",
        description="Audit example.com with per-keyword data.",
    )
    store.update_status(rec["id"], "working", actor="analyst")
    store.update_status(rec["id"], "done", actor="analyst")
    store.set_outcome(rec["id"], "rework", "Too shallow, add data", actor="operator")

    rework = store.create_rework_task(rec["id"], "Too shallow, add data")

    desc = rework["description"]
    assert desc.startswith("Too shallow, add data")
    assert "## Original brief" in desc
    assert "Audit example.com with per-keyword data." in desc


def test_rework_brief_skips_redundant_original(tmp_path):
    store = _make_store(tmp_path)
    rec = store.create(
        creator="operator", assignee="analyst",
        title="quick ping", description="quick ping",
    )
    store.update_status(rec["id"], "working", actor="analyst")
    store.update_status(rec["id"], "done", actor="analyst")

    rework = store.create_rework_task(rec["id"], "do it properly")

    assert rework["description"] == "do it properly"


# ── A4: fleet health digest formatting ────────────────────────────


def test_fleet_health_renders_only_nonzero_signals():
    from src.agent.loop import AgentLoop

    block = AgentLoop._format_fleet_health({
        "execution_failures_24h_count": {"writer": 3, "scout": 0},
        "outcome_rejected_24h_count": {},
        "outcome_rework_24h_count": {"writer": 4},
        "stale_tasks_24h_count": {},
        "chain_breaks_24h_count": {},
        "inbox_stale_count": 2,
        "per_agent_cost_today_usd": {"writer": 4.5, "scout": 0.2, "ed": 1.1},
        "agents_needing_attention": ["writer"],
    })

    assert "## Fleet Health" in block
    assert "Failures (24h): writer: 3" in block
    assert "Rework outcomes (24h): writer: 4" in block
    assert "Your stale inbox tasks (>24h): 2" in block
    assert "writer: $4.50" in block
    assert "Needs attention: writer" in block
    # zero-valued entries don't render
    assert "scout: 0" not in block
    assert "Rejected outcomes" not in block
    assert len(block) <= 600


def test_fleet_health_empty_for_healthy_fleet():
    from src.agent.loop import AgentLoop

    assert AgentLoop._format_fleet_health(None) == ""
    assert AgentLoop._format_fleet_health({}) == ""
    assert AgentLoop._format_fleet_health({
        "execution_failures_24h_count": {},
        "inbox_stale_count": 0,
        "per_agent_cost_today_usd": {},
    }) == ""


# ── A6: failed/blocked closes write learnings ─────────────────────


@pytest.mark.asyncio
async def test_auto_close_failed_records_task_error(tmp_path):
    from tests.test_loop import _make_loop

    loop = _make_loop()
    workspace = MagicMock()
    loop.workspace = workspace
    loop.mesh_client.set_task_status = AsyncMock(return_value={})

    await loop._auto_close_task("task_9", "failed", error="model rejected request")

    workspace.record_error.assert_called_once()
    args, kwargs = workspace.record_error.call_args
    assert args[0] == "task"
    assert "model rejected request" in args[1]


@pytest.mark.asyncio
async def test_auto_close_done_does_not_record(tmp_path):
    from tests.test_loop import _make_loop

    loop = _make_loop()
    workspace = MagicMock()
    loop.workspace = workspace
    loop.mesh_client.set_task_status = AsyncMock(return_value={})

    await loop._auto_close_task("task_9", "done")
    await loop._auto_close_task("task_9", "failed")  # no reason → no write

    workspace.record_error.assert_not_called()


# ── A1 end-to-end: mesh outcome endpoint pushes to the assignee ────


@pytest.mark.asyncio
async def test_outcome_endpoint_pushes_feedback_to_assignee(tmp_path, monkeypatch):
    import importlib

    from httpx import ASGITransport as HTTPXASGITransport
    from httpx import AsyncClient

    from src.host.mesh import Blackboard, MessageRouter, PubSub
    from src.host.permissions import PermissionMatrix

    monkeypatch.setenv(
        "OPENLEGION_ORCHESTRATION_TASKS_DB", str(tmp_path / "tasks.db"),
    )
    import src.host.server as server_module
    importlib.reload(server_module)

    blackboard = Blackboard(str(tmp_path / "bb.db"))
    permissions = PermissionMatrix()
    router = MessageRouter(permissions, {"analyst": "http://analyst:8400"})
    fake_transport = _FakeTransport()
    app = server_module.create_mesh_app(
        blackboard=blackboard,
        pubsub=PubSub(),
        router=router,
        permissions=permissions,
        transport=fake_transport,  # type: ignore[arg-type]
    )
    try:
        rec = app.tasks_store.create(
            creator="operator", assignee="analyst", title="SEO audit",
            description="full brief",
        )
        app.tasks_store.update_status(rec["id"], "working", actor="analyst")
        app.tasks_store.update_status(rec["id"], "done", actor="analyst")

        async with AsyncClient(
            transport=HTTPXASGITransport(app=app), base_url="http://t",
        ) as c:
            r = await c.post(
                f"/mesh/tasks/{rec['id']}/outcome",
                json={"outcome": "rework", "feedback": "too shallow"},
                headers={"X-Agent-ID": "operator"},
            )
        assert r.status_code == 200
        body = r.json()
        assert body["feedback_push"] == "recorded"
        assert body["rework_task_id"]
        # The push hit the assignee's learnings endpoint.
        assert len(fake_transport.calls) == 1
        call = fake_transport.calls[0]
        assert call["agent_id"] == "analyst"
        assert call["path"] == "/learnings/feedback"
        assert call["json"]["feedback"] == "too shallow"

        # accepted → no push, no feedback_push key.
        rec2 = app.tasks_store.create(
            creator="operator", assignee="analyst", title="other",
        )
        app.tasks_store.update_status(rec2["id"], "working", actor="analyst")
        app.tasks_store.update_status(rec2["id"], "done", actor="analyst")
        async with AsyncClient(
            transport=HTTPXASGITransport(app=app), base_url="http://t",
        ) as c:
            r2 = await c.post(
                f"/mesh/tasks/{rec2['id']}/outcome",
                json={"outcome": "accepted", "feedback": ""},
                headers={"X-Agent-ID": "operator"},
            )
        assert r2.status_code == 200
        assert "feedback_push" not in r2.json()
        assert len(fake_transport.calls) == 1
    finally:
        blackboard.close()
        monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_DB", raising=False)
        importlib.reload(server_module)


# ── A3: retry brief carries blocker_note ──────────────────────────


@pytest.mark.asyncio
async def test_retry_brief_carries_blocker_note(tmp_path, monkeypatch):
    import importlib

    from httpx import ASGITransport as HTTPXASGITransport
    from httpx import AsyncClient

    from src.host.mesh import Blackboard, MessageRouter, PubSub
    from src.host.permissions import PermissionMatrix

    monkeypatch.setenv(
        "OPENLEGION_ORCHESTRATION_TASKS_DB", str(tmp_path / "tasks.db"),
    )
    import src.host.server as server_module
    importlib.reload(server_module)

    blackboard = Blackboard(str(tmp_path / "bb.db"))
    permissions = PermissionMatrix()
    router = MessageRouter(permissions, {"analyst": "http://analyst:8400"})
    app = server_module.create_mesh_app(
        blackboard=blackboard,
        pubsub=PubSub(),
        router=router,
        permissions=permissions,
    )
    try:
        rec = app.tasks_store.create(
            creator="operator", assignee="analyst", title="SEO audit",
            description="audit example.com",
        )
        app.tasks_store.update_status(rec["id"], "working", actor="analyst")
        app.tasks_store.update_status(
            rec["id"], "failed", actor="analyst",
            blocker_note="GSC credentials expired",
        )

        async with AsyncClient(
            transport=HTTPXASGITransport(app=app), base_url="http://t",
        ) as c:
            r = await c.post(
                f"/mesh/tasks/{rec['id']}/retry",
                headers={"X-Agent-ID": "operator"},
            )
        assert r.status_code == 200
        clone = r.json()["clone"]
        assert "audit example.com" in clone["description"]
        assert "## Previous attempt failed" in clone["description"]
        assert "GSC credentials expired" in clone["description"]

        # Caller-supplied description override skips the blocker append.
        async with AsyncClient(
            transport=HTTPXASGITransport(app=app), base_url="http://t",
        ) as c:
            r2 = await c.post(
                f"/mesh/tasks/{rec['id']}/retry",
                json={"description": "fresh rewritten brief"},
                headers={"X-Agent-ID": "operator"},
            )
        assert r2.status_code == 200
        assert r2.json()["clone"]["description"] == "fresh rewritten brief"
    finally:
        blackboard.close()
        monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_DB", raising=False)
        importlib.reload(server_module)
