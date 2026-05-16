"""Integration + unit tests for the Bug 2/3 task-lifecycle plumbing.

Bug 2 (handed-off tasks never auto-close after recipient's loop finishes)
and Bug 3 (originator gets no back-edge for terminal transitions) are
bundled here because the fixes are co-dependent: the back-edge only
fires when ``/mesh/tasks/{id}/status`` lands, and that endpoint only
lands automatically when the recipient agent's loop knows it's handling
a specific task. Both arms of the plumbing are exercised below.

Tests:

1. wake_agent forwards ``x-task-id`` as an HTTP header when set.
2. loop.chat() auto-calls set_task_status("done") with a summary on success.
3. loop.chat() auto-calls set_task_status("failed") with the error on raise.
4. loop.chat() without task_id never touches set_task_status.
5. /mesh/tasks/{id}/status writes back-edge for ``agent`` origin.
6. /mesh/tasks/{id}/status skips back-edge for ``human`` origin.
7. /mesh/tasks/{id}/status skips back-edge for self-handoff
   (origin_user == assignee — no inbox spam).
8. coordination_tool._check_inbox_v2 surfaces back-edge events.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

# ── 1. wake_agent propagates x-task-id ───────────────────────────

@pytest.mark.asyncio
async def test_wake_agent_propagates_task_id_header():
    """wake_agent must include x-task-id when task_id is passed."""
    from src.agent.mesh_client import MeshClient

    mc = MeshClient(
        mesh_url="http://mesh:8420", agent_id="alpha",
    )

    captured_headers: dict = {}

    class _FakeResp:
        status_code = 200

        def json(self) -> dict:
            return {"woken": True, "target": "beta"}

        def raise_for_status(self) -> None:
            return None

    class _FakeClient:
        async def post(self, url, *, params=None, headers=None, **_kw):
            captured_headers.update(headers or {})
            return _FakeResp()

    mc._get_client = AsyncMock(return_value=_FakeClient())  # type: ignore[assignment]

    await mc.wake_agent("beta", "hi", task_id="task_abc123")

    assert captured_headers.get("x-task-id") == "task_abc123"

    # Negative: omitting task_id leaves the header off.
    captured_headers.clear()
    await mc.wake_agent("beta", "hi")
    assert "x-task-id" not in captured_headers


# ── 2. loop.chat() auto-closes on success ────────────────────────

def _make_loop_with_mocks():
    """Minimal AgentLoop with mock-of-everything, suitable for chat tests."""
    from src.agent.loop import AgentLoop
    from src.shared.types import LLMResponse

    memory = MagicMock()
    memory.get_high_salience_facts = AsyncMock(return_value=[])
    memory.decay_all = AsyncMock()
    memory.search = AsyncMock(return_value=[])
    memory.search_hierarchical = AsyncMock(return_value=[])
    memory.log_action = AsyncMock()
    memory.store_tool_outcome = AsyncMock()
    memory.get_tool_history = MagicMock(return_value=[])
    memory._run_db = AsyncMock(return_value=None)

    skills = MagicMock()
    skills.get_tool_definitions = MagicMock(return_value=[])
    skills.get_descriptions = MagicMock(return_value="- no tools")
    skills.list_skills = MagicMock(return_value=[])
    skills.is_parallel_safe = MagicMock(return_value=True)
    skills.get_loop_exempt_tools = MagicMock(return_value=frozenset())

    llm = MagicMock()
    llm.chat = AsyncMock(return_value=LLMResponse(
        content="Hello, I finished.", tokens_used=10,
    ))
    llm.default_model = "test-model"

    mesh_client = MagicMock()
    mesh_client.agent_id = "test_agent"
    mesh_client.is_standalone = True
    mesh_client.send_system_message = AsyncMock(return_value={})
    mesh_client.read_blackboard = AsyncMock(return_value=None)
    mesh_client.list_blackboard = AsyncMock(return_value=[])
    mesh_client.list_agents = AsyncMock(return_value={})
    mesh_client.set_task_status = AsyncMock(return_value={})

    return AgentLoop(
        agent_id="test_agent",
        role="research",
        memory=memory,
        skills=skills,
        llm=llm,
        mesh_client=mesh_client,
    )


@pytest.mark.asyncio
async def test_chat_auto_closes_task_on_success():
    """A successful chat with task_id triggers set_task_status("done")."""
    loop = _make_loop_with_mocks()

    result = await loop.chat("hi there", task_id="task_xyz")

    assert "response" in result
    # Two transitions: pending→working at start, working→done on completion.
    # State machine forbids pending→done directly, so the working step is
    # load-bearing for production handoffs.
    calls = loop.mesh_client.set_task_status.await_args_list
    assert len(calls) == 2, f"expected working+done, got {calls}"
    assert calls[0].args == ("task_xyz", "working")
    assert calls[1].args == ("task_xyz", "done")
    # Summary should carry the response prefix (truncated to 500 chars).
    assert calls[1].kwargs.get("result", {}).get("summary", "").startswith(
        "Hello, I finished",
    )


# ── 3. loop.chat() auto-fails on exception ───────────────────────

@pytest.mark.asyncio
async def test_chat_auto_closes_task_as_failed_on_exception():
    """If the inner chat body raises, set_task_status("failed") fires first."""
    loop = _make_loop_with_mocks()
    # Force _chat_inner to raise so the except branch in chat() runs.
    loop._chat_inner = AsyncMock(side_effect=RuntimeError("boom inside chat"))

    with pytest.raises(RuntimeError, match="boom inside chat"):
        await loop.chat("hi", task_id="task_fail")

    # Two transitions: pending→working at start, working→failed on exception.
    calls = loop.mesh_client.set_task_status.await_args_list
    assert len(calls) == 2, f"expected working+failed, got {calls}"
    assert calls[0].args == ("task_fail", "working")
    assert calls[1].args == ("task_fail", "failed")
    assert "boom inside chat" in (calls[1].kwargs.get("error") or "")


# ── 4. legacy path (no task_id) skips auto-close ─────────────────

@pytest.mark.asyncio
async def test_chat_without_task_id_skips_auto_close():
    """Legacy / heartbeat / manual chat callers never trigger set_task_status."""
    loop = _make_loop_with_mocks()

    await loop.chat("hi there")  # no task_id

    loop.mesh_client.set_task_status.assert_not_awaited()


# ── 5/6/7: back-edge emission on terminal transition ─────────────

def _setup_mesh_app(tmp_path):
    """Build a minimal create_mesh_app() with the v2 tasks store enabled."""
    import yaml as yaml_mod

    from src.host.mesh import Blackboard, MessageRouter, PubSub
    from src.host.orchestration import Tasks
    from src.host.permissions import PermissionMatrix

    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(exist_ok=True)
    (cfg_dir / "agents.yaml").write_text(yaml_mod.dump({"agents": {
        "alpha": {"role": "writer", "model": "gpt-4o"},
        "beta": {"role": "writer", "model": "gpt-4o"},
    }}))
    (cfg_dir / "mesh.yaml").write_text(yaml_mod.dump({"mesh": {"port": 8420}}))
    (cfg_dir / "projects").mkdir(exist_ok=True)

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix()
    router = MessageRouter(perms, {})
    router.register_agent("alpha", "http://alpha:8400", [])
    router.register_agent("beta", "http://beta:8400", [])

    # v2 tasks store is on by default ("1"). Point its DB at tmp_path so
    # we don't pollute the repo root with a real ``data/tasks.db``. Also
    # patch the module-level flag in case a sibling test set the env var
    # to "0" earlier in the suite (the flag is captured at import time).
    os.environ["OPENLEGION_ORCHESTRATION_TASKS_V2"] = "1"
    os.environ["OPENLEGION_ORCHESTRATION_TASKS_DB"] = str(
        tmp_path / "tasks.db",
    )

    import src.host.server as server_mod
    server_mod._ORCHESTRATION_TASKS_V2 = True

    app = server_mod.create_mesh_app(
        blackboard=bb, pubsub=pubsub, router=router, permissions=perms,
    )
    # Use the same DB the mesh wired up so writes from the test setup are
    # visible to the endpoint handler.
    tasks_store = Tasks(db_path=str(tmp_path / "tasks.db"))
    return app, bb, tasks_store, cfg_dir


def _teardown_mesh(bb, tasks_store):
    try:
        tasks_store.close()
    except Exception:
        pass
    try:
        bb.close()
    except Exception:
        pass
    os.environ.pop("OPENLEGION_ORCHESTRATION_TASKS_V2", None)
    os.environ.pop("OPENLEGION_ORCHESTRATION_TASKS_DB", None)


async def _advance_to_working(client, task_id):
    """Helper: pending → working so the state machine allows → done."""
    resp = await client.post(
        f"/mesh/tasks/{task_id}/status",
        json={"status": "working"},
        headers={"x-mesh-internal": "1"},
    )
    assert resp.status_code == 200, resp.text


@pytest.mark.asyncio
async def test_pending_to_working_to_done_end_to_end(tmp_path):
    """Production handoff: a freshly-created (pending) task auto-advances
    to working, then to done via two POSTs to /mesh/tasks/{id}/status.

    Regression guard: the state machine forbids pending → done directly,
    so the chat-start working transition is load-bearing. Without it,
    the auto-close would 4xx in production and tasks would still dangle —
    silently re-creating Bug 2 even with all the plumbing in place.
    """
    from httpx import ASGITransport, AsyncClient

    app, bb, tasks_store, _ = _setup_mesh_app(tmp_path)
    try:
        rec = tasks_store.create(
            creator="alpha",
            assignee="beta",
            title="Do the thing",
            origin={"kind": "agent", "channel": "", "user": "alpha"},
        )
        task_id = rec["id"]
        assert rec["status"] == "pending"

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            # Step 1: pending → working (what chat() does on entry).
            r1 = await client.post(
                f"/mesh/tasks/{task_id}/status",
                json={"status": "working"},
                headers={"x-mesh-internal": "1"},
            )
            assert r1.status_code == 200, r1.text
            # Step 2: working → done (what chat() does on success).
            r2 = await client.post(
                f"/mesh/tasks/{task_id}/status",
                json={"status": "done", "result": {"summary": "ok"}},
                headers={"x-mesh-internal": "1"},
            )
            assert r2.status_code == 200, r2.text

        # Task closed cleanly + back-edge written.
        fresh = tasks_store.get(task_id)
        assert fresh["status"] == "done"
        assert bb.read(f"inbox/alpha/task_event/{task_id}") is not None
    finally:
        _teardown_mesh(bb, tasks_store)


@pytest.mark.asyncio
async def test_terminal_transition_writes_back_edge_for_agent_origin(tmp_path):
    """origin_kind=='agent' triggers a blackboard back-edge on terminal status."""
    from httpx import ASGITransport, AsyncClient

    app, bb, tasks_store, _ = _setup_mesh_app(tmp_path)
    try:
        rec = tasks_store.create(
            creator="alpha",
            assignee="beta",
            title="Do the thing",
            origin={"kind": "agent", "channel": "", "user": "alpha"},
        )
        task_id = rec["id"]

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            await _advance_to_working(client, task_id)
            resp = await client.post(
                f"/mesh/tasks/{task_id}/status",
                json={"status": "done", "result": {"summary": "all done"}},
                headers={"x-mesh-internal": "1"},
            )
        assert resp.status_code == 200, resp.text

        entry = bb.read(f"inbox/alpha/task_event/{task_id}")
        assert entry is not None, "back-edge missing"
        payload = entry.value
        assert payload["kind"] == "task_completed"
        assert payload["task_id"] == task_id
        assert payload["recipient"] == "beta"
        assert payload["status"] == "done"
        assert payload["summary"] == "all done"
        assert entry.ttl == 604800
    finally:
        _teardown_mesh(bb, tasks_store)


@pytest.mark.asyncio
async def test_terminal_transition_skips_back_edge_for_human_origin(tmp_path):
    """origin_kind=='human' must NOT write a back-edge (humans use lane forward path)."""
    from httpx import ASGITransport, AsyncClient

    app, bb, tasks_store, _ = _setup_mesh_app(tmp_path)
    try:
        rec = tasks_store.create(
            creator="alpha",
            assignee="beta",
            title="User asked",
            origin={"kind": "human", "channel": "telegram", "user": "u_42"},
        )
        task_id = rec["id"]

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            await _advance_to_working(client, task_id)
            resp = await client.post(
                f"/mesh/tasks/{task_id}/status",
                json={"status": "done", "result": {"summary": "yo"}},
                headers={"x-mesh-internal": "1"},
            )
        assert resp.status_code == 200, resp.text

        # No back-edge for any plausible origin_user variant.
        for candidate in ("u_42", "telegram", "alpha", "beta"):
            assert bb.read(f"inbox/{candidate}/task_event/{task_id}") is None
    finally:
        _teardown_mesh(bb, tasks_store)


@pytest.mark.asyncio
async def test_terminal_transition_skips_back_edge_for_self(tmp_path):
    """origin_user == assignee (self-handoff) must NOT pollute the inbox."""
    from httpx import ASGITransport, AsyncClient

    app, bb, tasks_store, _ = _setup_mesh_app(tmp_path)
    try:
        rec = tasks_store.create(
            creator="beta",
            assignee="beta",
            title="Self-note",
            origin={"kind": "agent", "channel": "", "user": "beta"},
        )
        task_id = rec["id"]

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            await _advance_to_working(client, task_id)
            resp = await client.post(
                f"/mesh/tasks/{task_id}/status",
                json={"status": "done", "result": {"summary": "ok"}},
                headers={"x-mesh-internal": "1"},
            )
        assert resp.status_code == 200, resp.text

        assert bb.read(f"inbox/beta/task_event/{task_id}") is None
    finally:
        _teardown_mesh(bb, tasks_store)


# ── 8. check_inbox surfaces back-edge events ─────────────────────

@pytest.mark.asyncio
async def test_check_inbox_surfaces_back_edge_events():
    """_check_inbox_v2 returns events alongside tasks."""
    from src.agent.builtins.coordination_tool import _check_inbox_v2

    mesh_client = MagicMock()
    mesh_client.agent_id = "alpha"
    mesh_client.list_task_inbox = AsyncMock(return_value=[])
    # Synthetic back-edge payload as the mesh would write.
    mesh_client.list_blackboard = AsyncMock(return_value=[{
        "key": "inbox/alpha/task_event/task_abc",
        "value": {
            "kind": "task_completed",
            "task_id": "task_abc",
            "recipient": "beta",
            "title": "Do the thing",
            "status": "done",
            "ts": 1700000000,
            "summary": "finished it",
        },
    }])

    result = await _check_inbox_v2(mesh_client=mesh_client)

    assert result["count"] == 0
    assert result["event_count"] == 1
    assert len(result["events"]) == 1
    ev = result["events"][0]
    assert ev["kind"] == "task_completed"
    assert ev["task_id"] == "task_abc"
    assert ev["summary"] == "finished it"
    assert ev["key"] == "inbox/alpha/task_event/task_abc"

    # The blackboard list call used the right prefix + global_scope.
    mesh_client.list_blackboard.assert_awaited_once()
    call = mesh_client.list_blackboard.await_args
    assert call.args[0] == "inbox/alpha/task_event/"
    assert call.kwargs.get("global_scope") is True


@pytest.mark.asyncio
async def test_check_inbox_degrades_when_event_fetch_fails():
    """A blackboard hiccup must not fail the whole check_inbox call."""
    from src.agent.builtins.coordination_tool import _check_inbox_v2

    mesh_client = MagicMock()
    mesh_client.agent_id = "alpha"
    mesh_client.list_task_inbox = AsyncMock(return_value=[])
    mesh_client.list_blackboard = AsyncMock(side_effect=RuntimeError("boom"))

    result = await _check_inbox_v2(mesh_client=mesh_client)

    assert "error" not in result
    assert result["count"] == 0
    assert result["event_count"] == 0
    assert result["events"] == []
