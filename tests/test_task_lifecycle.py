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
8. coordination_tool.check_inbox surfaces back-edge events.
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
        # ``is_success`` mirrors ``httpx.Response.is_success`` (2xx) so the
        # ``_raise_with_body`` helper in ``mesh_client`` can short-circuit
        # without inspecting status semantics.
        is_success = True

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

    tools = MagicMock()
    tools.get_tool_definitions = MagicMock(return_value=[])
    tools.get_descriptions = MagicMock(return_value="- no tools")
    tools.list_tools = MagicMock(return_value=[])
    tools.is_parallel_safe = MagicMock(return_value=True)
    tools.get_loop_exempt_tools = MagicMock(return_value=frozenset())

    llm = MagicMock()
    # Default mock returns a STRUCTURED final answer so the chat-path
    # handoff lazy-completion guard (codex r9 follow-up) lets the
    # auto-close → done transition land. Lazy text-only mocks now
    # auto-fail under that guard — see ``test_chat_lazy_completion_*``
    # below for the failure-path tests.
    llm.chat = AsyncMock(return_value=LLMResponse(
        content='{"result": {"summary": "Hello, I finished."}}',
        tokens_used=10,
    ))
    llm.default_model = "test-model"

    async def _chat_collect_delegate(*args, **kwargs):
        return await llm.chat(*args, **kwargs)
    llm.chat_collect = _chat_collect_delegate

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
        tools=tools,
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
    # The structured ``{"result": ...}`` mock content survives intact —
    # auto-close packs the raw response prefix as ``summary``.
    assert "Hello, I finished" in calls[1].kwargs.get("result", {}).get(
        "summary", "",
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


# ── 3b. Bug F (codex r9) chat-path lazy-completion guard ─────────

@pytest.mark.asyncio
async def test_chat_handoff_text_only_no_tools_auto_closes_as_failed():
    """Codex r9 finding: the lazy-completion guard in execute_task did
    not cover the production handoff path. Lane dispatch posts the work
    through ``/chat`` → ``loop.chat(task_id=...)``, which used to
    auto-close as ``done`` regardless of whether real work was done.

    The trend-scout bug operator hit (text-only "On it — running now"
    reply with zero tool calls) was on THIS path, not execute_task.
    Mirror the contract: a handoff-bound chat must either call ≥1
    tool OR return a structured ``{"result": {...}}`` payload.
    """
    from src.shared.types import LLMResponse

    loop = _make_loop_with_mocks()
    # Override default mock with the operator's reproduction shape:
    # lazy text-only response, no tools.
    loop.llm.chat = AsyncMock(return_value=LLMResponse(
        content="On it — running now. I'll pick a topic and hand off shortly.",
        tokens_used=42,
    ))

    result = await loop.chat("Please pick a fresh topic.", task_id="task_lazy")

    # Two status calls: pending→working at start, working→failed via the
    # lazy-completion guard at the auto-close site.
    calls = loop.mesh_client.set_task_status.await_args_list
    assert len(calls) == 2, f"expected working+failed, got {calls}"
    assert calls[0].args == ("task_lazy", "working")
    assert calls[1].args == ("task_lazy", "failed")
    error_msg = calls[1].kwargs.get("error") or ""
    # Round-5: error envelope renamed from ``no_action_taken`` to
    # ``no_outbound_effects`` when the guard was strengthened to count
    # read-only tool calls (``check_inbox`` etc.) as still-lazy. Same
    # semantic — the LLM produced no downstream effect for the handoff.
    assert "no_outbound_effects" in error_msg
    # The chat call itself still returns the original response — the
    # auto-close failure annotates the task row, doesn't crash the
    # request.
    assert "response" in result


@pytest.mark.asyncio
async def test_chat_handoff_text_only_with_tool_calls_completes_as_done():
    """Counter-test: a chat-path handoff that calls at least one tool
    AND then summarizes in text MUST still complete. The tool dispatch
    is the work signal — text final after tools is the normal pattern.
    """
    from src.shared.types import LLMResponse, ToolCallInfo

    loop = _make_loop_with_mocks()
    # Two-response sequence: iter 0 tool call → iter 1 text summary.
    loop.llm.chat = AsyncMock(side_effect=[
        LLMResponse(
            content="", tokens_used=20,
            tool_calls=[ToolCallInfo(
                name="memory_save", arguments={"content": "noted"},
            )],
        ),
        LLMResponse(
            content="Saved the brief and notified the next agent.",
            tokens_used=30,
        ),
    ])
    loop.tools.get_tool_definitions = MagicMock(return_value=[
        {"type": "function", "function": {"name": "memory_save"}},
    ])
    loop.tools.execute = AsyncMock(return_value={"ok": True})

    await loop.chat("save the brief", task_id="task_real_work")

    calls = loop.mesh_client.set_task_status.await_args_list
    assert calls[-1].args == ("task_real_work", "done"), (
        f"task with real tool calls must auto-close as done; got {calls}"
    )
    # The summary carries the text-final response.
    assert "Saved the brief" in calls[-1].kwargs.get("result", {}).get(
        "summary", "",
    )


@pytest.mark.asyncio
async def test_chat_handoff_exception_after_tool_calls_auto_closes_as_failed():
    """Codex r10 HIGH: _chat_inner catches LLMAuthError, LLMConfigError,
    and bare Exception and returns ordinary result dicts (not raises).
    If an exception fires AFTER one or more tool dispatches, the
    result has non-empty tool_outputs and would slip past the
    lazy-completion guard, auto-closing the task as done despite the
    failure.

    Fix: _chat_inner tags the bare-exception return with
    exception_caught=True; the auto-close site checks
    _chat_result_failure_reason BEFORE the lazy guard so any failure
    marker (tool_limit_reached / auth_failure / config_error /
    exception_caught) routes the task to failed.
    """

    loop = _make_loop_with_mocks()

    # Simulate _chat_inner having partially executed (one tool output)
    # and then hitting an exception that got swallowed into a result
    # dict with the new exception_caught marker. The handler at
    # ``chat()`` must detect this BEFORE the lazy guard.
    swallowed_failure_result = {
        "response": "Error: provider down",
        "tool_outputs": [{"tool": "memory_save", "result": {"ok": True}}],
        "tokens_used": 50,
        "exception_caught": True,
    }
    loop._chat_inner = AsyncMock(return_value=swallowed_failure_result)

    await loop.chat("save the brief", task_id="task_partial")

    calls = loop.mesh_client.set_task_status.await_args_list
    # working then failed (NOT done despite non-empty tool_outputs).
    assert calls[-1].args == ("task_partial", "failed"), (
        f"exception_caught marker must route to failed; got {calls}"
    )
    assert "exception" in (calls[-1].kwargs.get("error") or "")


@pytest.mark.asyncio
async def test_chat_handoff_auth_failure_routes_to_failed():
    """Codex r10: auth_failure flag from _chat_inner must close the
    task as failed regardless of whether tool calls preceded the
    error (an auth failure mid-task is still a failure)."""
    loop = _make_loop_with_mocks()
    loop._chat_inner = AsyncMock(return_value={
        "response": "Auth failure: token expired",
        "tool_outputs": [{"tool": "memory_save", "result": {"ok": True}}],
        "tokens_used": 50,
        "auth_failure": True,
    })

    await loop.chat("do the thing", task_id="task_auth_fail")

    calls = loop.mesh_client.set_task_status.await_args_list
    assert calls[-1].args == ("task_auth_fail", "failed")
    assert "auth_failure" in (calls[-1].kwargs.get("error") or "")


@pytest.mark.asyncio
async def test_chat_handoff_cancelled_error_closes_task_as_cancelled():
    """Codex r10 MEDIUM: _chat_inner re-raises CancelledError, which is
    BaseException (NOT Exception). The original try/except Exception in
    chat() didn't catch it, so the durable task stayed at ``working``
    forever after a cancellation. Fix adds an explicit
    ``except asyncio.CancelledError`` branch that auto-closes as
    cancelled before re-raising."""
    import asyncio

    loop = _make_loop_with_mocks()
    loop._chat_inner = AsyncMock(side_effect=asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await loop.chat("interrupt me", task_id="task_cancelled")

    calls = loop.mesh_client.set_task_status.await_args_list
    # Two transitions: pending→working, working→cancelled.
    assert len(calls) == 2, f"expected working+cancelled, got {calls}"
    assert calls[0].args == ("task_cancelled", "working")
    assert calls[1].args == ("task_cancelled", "cancelled")


@pytest.mark.asyncio
async def test_chat_handoff_structured_final_no_tools_completes_as_done():
    """Counter-test: a chat-path handoff that returns a structured
    ``{"result": {...}}`` payload WITHOUT calling any tools MUST still
    complete — that's the documented contract for legitimate noop /
    impossibility outcomes."""
    from src.shared.types import LLMResponse

    loop = _make_loop_with_mocks()
    loop.llm.chat = AsyncMock(return_value=LLMResponse(
        content='{"result": {"status": "noop", "reason": "queue empty"}}',
        tokens_used=40,
    ))

    await loop.chat("check queue", task_id="task_noop")

    calls = loop.mesh_client.set_task_status.await_args_list
    assert calls[-1].args == ("task_noop", "done"), (
        f"structured-final handoff must auto-close as done; got {calls}"
    )


@pytest.mark.asyncio
async def test_chat_handoff_structured_done_surfaces_extracted_summary():
    """The answer-delivery contract: a chat-path handoff returning
    ``{"result": {"status": "done", "summary": "<answer>"}}`` must close
    ``done`` and surface the EXTRACTED answer as the deliverable — not the
    raw JSON envelope. Regression guard for the Codex-caught gap where the
    chat close stored ``response_text[:500]`` verbatim (operator/originator
    saw ``{"result": {...}}`` instead of the answer). Mirrors execute_task's
    _parse_final_output extraction."""
    from src.shared.types import LLMResponse

    loop = _make_loop_with_mocks()
    answer = "Q3 revenue was $1.2M, up 18% QoQ."
    loop.llm.chat = AsyncMock(return_value=LLMResponse(
        content='{"result": {"status": "done", "summary": "%s"}}' % answer,
        tokens_used=55,
    ))

    await loop.chat("what was Q3 revenue?", task_id="task_ans")

    call = loop.mesh_client.set_task_status.await_args
    assert call.args == ("task_ans", "done")
    surfaced = call.kwargs.get("result") or {}
    # The deliverable carries the extracted answer, not the JSON wrapper.
    assert surfaced.get("summary") == answer, (
        f"expected extracted answer as summary, got {surfaced!r}"
    )
    assert not str(surfaced.get("summary", "")).lstrip().startswith("{"), (
        "summary must not be the raw {\"result\": ...} envelope"
    )


# ── 4. legacy path (no task_id) skips auto-close ─────────────────

@pytest.mark.asyncio
async def test_chat_without_task_id_skips_auto_close():
    """Legacy / heartbeat / manual chat callers never trigger set_task_status."""
    loop = _make_loop_with_mocks()

    await loop.chat("hi there")  # no task_id

    loop.mesh_client.set_task_status.assert_not_awaited()


# ── 4b. agent-busy handoff rejection — no steer queue + back-edge ─

@pytest.mark.asyncio
async def test_chat_with_task_id_while_busy_rejects_handoff_without_queueing():
    """Codex P2: previous attempt at this fix put the message on the
    steer queue AND closed the task as failed. The originator saw
    task_failed and could retry/reroute, while THIS agent later
    drained the queue and processed the (now-failed) message —
    duplicate / conflicting work.

    Correct semantics: a handoff with task_id arriving on a busy
    agent must be CLEANLY rejected — no steer-queue enqueue. The
    originator gets a clear back-edge via set_task_status(failed)
    and is the only party that decides what to do next.

    Legacy free-form chat (no task_id) keeps queueing — there's
    no durable task in that path so no double-execution risk.
    """
    loop = _make_loop_with_mocks()
    # Sentinel async-mock for the steer queue so we can assert NO put.
    loop._steer_queue.put = AsyncMock(return_value=None)
    # Simulate agent already executing a task.
    loop.current_task = "task_in_flight"

    result = await loop.chat("handoff plz", task_id="task_arriving")

    # The handoff was rejected — response surfaces that to the caller.
    assert "rejected" in result["response"].lower()
    # The arriving task got a clear failed back-edge with a specific code.
    loop.mesh_client.set_task_status.assert_awaited_once()
    call = loop.mesh_client.set_task_status.await_args
    assert call.args == ("task_arriving", "failed")
    assert call.kwargs.get("error") == "agent_busy_handoff_rejected"
    # CRITICAL: must NOT have queued the message (would cause dup-exec).
    loop._steer_queue.put.assert_not_awaited()


@pytest.mark.asyncio
async def test_chat_without_task_id_while_busy_still_queues_for_chat_ux():
    """Legacy chat path is unchanged: a free-form chat message (no
    task_id) on a busy agent still goes onto the steer queue so the
    user's mid-conversation steering survives. Only the durable-task
    handoff path skips the queue (see test above)."""
    loop = _make_loop_with_mocks()
    loop._steer_queue.put = AsyncMock(return_value=None)
    loop.current_task = "task_in_flight"

    result = await loop.chat("hey can you also do X?")  # no task_id

    assert "queued" in result["response"].lower()
    # No task to close (no task_id).
    loop.mesh_client.set_task_status.assert_not_awaited()
    # And the queue WAS exercised.
    loop._steer_queue.put.assert_awaited_once_with("hey can you also do X?")


# ── 4c. _auto_close_task log-severity discriminates 4xx vs 5xx ────

@pytest.mark.asyncio
async def test_auto_close_logs_5xx_at_error_not_warning(caplog):
    """A mesh 5xx on auto-close means the task will silently dangle in
    pending forever — must surface at ERROR so standard log monitors
    page the operator. State-machine 4xx (benign concurrent-transition
    races) stays at WARNING so it doesn't trigger noise alerts.
    """
    import logging
    from unittest.mock import MagicMock as _MM

    loop = _make_loop_with_mocks()

    # Simulate httpx.HTTPStatusError shape: exception with a .response
    # that carries a status_code int.
    fake_resp_5xx = _MM()
    fake_resp_5xx.status_code = 503
    fake_resp_5xx.text = "Service Unavailable"
    err_5xx = Exception("mesh exploded")
    err_5xx.response = fake_resp_5xx
    loop.mesh_client.set_task_status = AsyncMock(side_effect=err_5xx)

    caplog.set_level(logging.WARNING, logger="agent.loop")
    await loop._auto_close_task("task_x", "done")

    # Exactly one log record, at ERROR level (not WARNING).
    err_records = [
        r for r in caplog.records
        if r.levelno >= logging.ERROR and "task_x" in r.getMessage()
    ]
    assert len(err_records) == 1, (
        f"expected 1 ERROR log for 5xx, got {len(err_records)}: "
        f"{[r.getMessage() for r in caplog.records]}"
    )
    assert "FAILED" in err_records[0].getMessage()


@pytest.mark.asyncio
async def test_auto_close_logs_4xx_at_warning_not_error(caplog):
    """State-machine rejections (e.g. already done, transition not
    allowed) are 400-ish and benign. Must NOT escalate to ERROR or
    we'll page on every concurrent-transition race.
    """
    import logging
    from unittest.mock import MagicMock as _MM

    loop = _make_loop_with_mocks()

    fake_resp_4xx = _MM()
    fake_resp_4xx.status_code = 400
    fake_resp_4xx.text = "Invalid transition"
    err_4xx = Exception("state machine said no")
    err_4xx.response = fake_resp_4xx
    loop.mesh_client.set_task_status = AsyncMock(side_effect=err_4xx)

    caplog.set_level(logging.WARNING, logger="agent.loop")
    await loop._auto_close_task("task_y", "done")

    err_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(err_records) == 0, (
        f"4xx should not emit ERROR — got {[r.getMessage() for r in err_records]}"
    )


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

    # Point the tasks store at tmp_path so we don't pollute the repo
    # root with a real ``data/tasks.db``.
    os.environ["OPENLEGION_ORCHESTRATION_TASKS_DB"] = str(
        tmp_path / "tasks.db",
    )

    import src.host.server as server_mod

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
        # task_completed is an informational back-edge kind → 24h TTL (the
        # actionable kinds task_failed/task_blocked keep the 7-day window).
        assert entry.ttl == 86400
    finally:
        _teardown_mesh(bb, tasks_store)


@pytest.mark.asyncio
async def test_back_edge_tolerates_non_dict_result(tmp_path):
    """Before the fix, a /status call with ``result="ok"`` (a string,
    not a dict) crashed the back-edge writer with AttributeError —
    the status update committed but the originator's inbox stayed
    empty. Test guards against the regression by sending a plain
    string result and verifying:
      (a) the status transition succeeds (200),
      (b) the back-edge IS written (just with empty summary).
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

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            await _advance_to_working(client, task_id)
            resp = await client.post(
                f"/mesh/tasks/{task_id}/status",
                # Misuse: result is a string, not a dict.
                json={"status": "done", "result": "ok"},
                headers={"x-mesh-internal": "1"},
            )
        assert resp.status_code == 200, resp.text
        entry = bb.read(f"inbox/alpha/task_event/{task_id}")
        assert entry is not None, "back-edge dropped on non-dict result"
        # Summary defaults to "" — the writer didn't crash extracting it.
        assert entry.value.get("summary") == ""
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
    """check_inbox returns events alongside tasks."""
    from src.agent.builtins.coordination_tool import check_inbox

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

    result = await check_inbox(mesh_client=mesh_client)

    assert result["count"] == 0
    assert result["event_count"] == 1
    assert len(result["events"]) == 1
    ev = result["events"][0]
    assert ev["kind"] == "task_completed"
    assert ev["task_id"] == "task_abc"
    assert ev["summary"] == "finished it"
    assert ev["key"] == "inbox/alpha/task_event/task_abc"

    # The durable task inbox endpoint is called, then the blackboard
    # back-edge prefix.
    mesh_client.list_task_inbox.assert_awaited_once_with("alpha")
    mesh_client.list_blackboard.assert_awaited_once()
    call = mesh_client.list_blackboard.await_args
    assert call.args[0] == "inbox/alpha/task_event/"
    assert call.kwargs.get("global_scope") is True


@pytest.mark.asyncio
async def test_check_inbox_degrades_when_event_fetch_fails():
    """A blackboard hiccup must not fail the whole check_inbox call."""
    from src.agent.builtins.coordination_tool import check_inbox

    mesh_client = MagicMock()
    mesh_client.agent_id = "alpha"
    mesh_client.list_task_inbox = AsyncMock(return_value=[])
    mesh_client.list_blackboard = AsyncMock(side_effect=RuntimeError("boom"))

    result = await check_inbox(mesh_client=mesh_client)

    assert "error" not in result
    assert result["count"] == 0
    assert result["event_count"] == 0
    assert result["events"] == []
