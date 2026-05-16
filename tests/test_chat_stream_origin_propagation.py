"""Regression tests for ContextVar propagation through the agent
/chat/stream endpoint.

Root cause (from operator bug report 2026-05-16): the dashboard chat
uses /chat/stream, and the streaming event_generator wraps each
``__anext__()`` in ``asyncio.ensure_future``, which creates a fresh
asyncio.Task per iteration. Each task copies the PARENT task's
context at creation time — so a ``current_origin.set()`` performed
INSIDE the generator body (iteration 1) is NOT visible in iteration
2+'s tasks. Tool dispatches happen in iterations 2+, so
``current_origin.get()`` returned ``None`` and
``mesh_client.propose_delete_agent`` skipped the X-Origin header
even when the request carried a verified human origin.

PR #902 fixed the mesh_client → mesh header forwarding but didn't
fix this contextvar-loss path. The unit tests for #902 set the
contextvar in the test's own task BEFORE awaiting propose_delete,
which masked the issue. The fix is in src/agent/server.py:
hoist the contextvar set OUT of loop.chat_stream's body and into
the /chat/stream endpoint's event_generator, so all per-iteration
ensure_future child tasks inherit the origin.
"""
from __future__ import annotations

import asyncio
import contextvars

import pytest

# Standalone unit test of the contextvar propagation pattern. Doesn't
# need any of the engine machinery — exercises the exact failure
# mode in isolation so the fix's contract is unambiguous.

_test_origin: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_test_origin", default=None,
)


async def _producer_no_internal_set():
    """An async generator that ONLY reads the contextvar — does not
    set it. Mimics what loop.chat_stream would look like after we
    move the set() to the outer caller (or, equivalently, the
    behavior any downstream tool dispatch would see)."""
    yield ("iter1_read", _test_origin.get())
    yield ("iter2_read", _test_origin.get())
    yield ("iter3_read", _test_origin.get())


async def _drive_via_ensure_future(gen):
    """Mimics the exact pattern in agent /chat/stream's event_generator
    — each __anext__ wrapped in asyncio.ensure_future."""
    out = []
    stream_iter = gen.__aiter__()
    next_event = asyncio.ensure_future(stream_iter.__anext__())
    try:
        while True:
            done, _ = await asyncio.wait({next_event}, timeout=2)
            if not done:
                break
            try:
                event = next_event.result()
            except StopAsyncIteration:
                break
            out.append(event)
            next_event = asyncio.ensure_future(stream_iter.__anext__())
    finally:
        if not next_event.done():
            next_event.cancel()
    return out


@pytest.mark.asyncio
async def test_outer_task_set_propagates_to_all_ensure_future_iterations():
    """Setting the contextvar in the OUTER task BEFORE the first
    ensure_future call propagates it to every subsequent child task.
    This is the contract the /chat/stream fix relies on.
    """
    _test_origin.set("human-from-outer")
    events = await _drive_via_ensure_future(_producer_no_internal_set())

    assert events == [
        ("iter1_read", "human-from-outer"),
        ("iter2_read", "human-from-outer"),
        ("iter3_read", "human-from-outer"),
    ]


# Higher-level: exercise the actual /chat/stream endpoint and assert
# that a tool dispatched in a LATE iteration reads the human origin
# from current_origin (the production failure path).

class _FakeLoop:
    """Minimal AgentLoop stub that yields a stream simulating
    text deltas then a tool dispatch. The "tool" records what
    current_origin sees at dispatch time."""

    def __init__(self, agent_id: str = "operator"):
        self.agent_id = agent_id
        self.state = "idle"
        self.current_task = None
        self.recorded_origin_at_tool_time: str | None = "SENTINEL_NOT_RECORDED"

    async def chat_stream(self, message: str, *, trace_id=None, origin=None, task_id=None):
        # Yield a couple of text_delta-shaped events first. Each yield
        # represents a streaming SSE chunk, and the event_generator
        # in the endpoint will spawn a fresh ensure_future task per
        # __anext__. So when we reach the "tool dispatch" event below
        # we're in iteration 3 — a new task than iteration 1. The fix
        # in /chat/stream must propagate current_origin to this task.
        from src.shared.trace import current_origin
        yield {"type": "text_delta", "content": "thinking..."}
        yield {"type": "text_delta", "content": "more text..."}
        # Tool dispatch — record what current_origin actually says at
        # this point. With the bug, this is None (iteration 3 task
        # inherited the parent context which never had origin set).
        # With the fix, this is the human MessageOrigin.
        origin_at_dispatch = current_origin.get()
        self.recorded_origin_at_tool_time = (
            origin_at_dispatch.kind if origin_at_dispatch is not None else None
        )
        yield {"type": "tool_start", "tool": "manage_agent"}
        yield {"type": "done", "response": "ok", "tool_outputs": [], "tokens_used": 0}


@pytest.mark.asyncio
async def test_chat_stream_endpoint_propagates_origin_to_late_iteration_tools():
    """End-to-end regression for the operator delete-confirm bug.

    POST to the real /chat/stream endpoint with a human X-Origin and
    ``x-mesh-internal: 1``. The fake loop records what
    ``current_origin.get()`` returns when its 3rd-iteration "tool
    dispatch" event fires. Without the fix this is ``None`` (the
    yield happens in a fresh ensure_future task that inherited the
    handler's context, which never set the contextvar). With the
    fix this is ``"human"``.
    """
    from httpx import ASGITransport, AsyncClient

    from src.agent.server import create_agent_app
    from src.shared.trace import origin_header
    from src.shared.types import MessageOrigin

    fake_loop = _FakeLoop()
    app = create_agent_app(fake_loop)  # type: ignore[arg-type]

    human = MessageOrigin(kind="human", channel="dashboard", user="user1")
    headers = {"x-mesh-internal": "1"}
    headers.update(origin_header(human))

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        async with client.stream(
            "POST", "/chat/stream",
            json={"message": "delete the scout agent"},
            headers=headers,
            timeout=10,
        ) as resp:
            assert resp.status_code == 200
            # Drain the stream.
            async for _ in resp.aiter_lines():
                pass

    # The fake loop's tool-dispatch event ran in iteration 3 (after
    # two yielded text deltas). Before the fix it saw current_origin
    # as None; after the fix it sees "human".
    assert fake_loop.recorded_origin_at_tool_time == "human", (
        f"current_origin at tool dispatch was "
        f"{fake_loop.recorded_origin_at_tool_time!r}; expected "
        f"'human' (the X-Origin sent on the request). The streaming "
        f"endpoint is not propagating origin to per-iteration child "
        f"tasks — the operator manage_agent → propose_delete_agent "
        f"path will fail confirm with 'did not carry a verified "
        f"human origin'."
    )


# Latent ValueError guard: loop.chat_stream's old inner set/reset
# pattern raised at end-of-stream because the token was created in
# iteration 1's task and reset() ran in iteration N's task. Removing
# the inner set/reset (now redundant after the outer hoist) eliminated
# that. This test pins it down.

class _CleanStreamLoop:
    """Yields a few events and exits cleanly. With the bug, the inner
    finally's reset(token) would raise ValueError at stream end."""

    def __init__(self):
        self.agent_id = "operator"
        self.state = "idle"
        self.current_task = None

    async def chat_stream(self, message, *, trace_id=None, origin=None, task_id=None):
        yield {"type": "text_delta", "content": "hello "}
        yield {"type": "text_delta", "content": "world"}
        yield {"type": "done", "response": "hello world", "tool_outputs": [], "tokens_used": 5}


@pytest.mark.asyncio
async def test_chat_stream_no_valueerror_at_stream_end():
    """Regression for the latent ValueError: loop.chat_stream used to
    do an inner ``current_origin.set()`` in iteration 1's task and
    match it with ``current_origin.reset(token)`` at its finally —
    which runs in iteration N's task, a different context. Python's
    ``ContextVar.reset()`` rejects cross-context resets with
    ValueError. The exception surfaced through ``next_e.result()``
    and broke out of event_generator's loop with an unhandled
    exception. Removing the inner set/reset (now redundant after
    PR #906's outer hoist) fixes this. Test asserts the stream
    completes without any exception leaking into the response.
    """
    import logging

    from httpx import ASGITransport, AsyncClient

    from src.agent.server import create_agent_app
    from src.shared.trace import origin_header
    from src.shared.types import MessageOrigin

    app = create_agent_app(_CleanStreamLoop())  # type: ignore[arg-type]
    human = MessageOrigin(kind="human", channel="dashboard", user="user1")
    headers = {"x-mesh-internal": "1"}
    headers.update(origin_header(human))

    lines: list[str] = []
    caplog_records: list[logging.LogRecord] = []

    class _CaptureHandler(logging.Handler):
        def emit(self, record):
            caplog_records.append(record)

    root_logger = logging.getLogger()
    handler = _CaptureHandler(level=logging.WARNING)
    root_logger.addHandler(handler)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            async with client.stream(
                "POST", "/chat/stream",
                json={"message": "hi"},
                headers=headers,
                timeout=10,
            ) as resp:
                assert resp.status_code == 200
                async for line in resp.aiter_lines():
                    if line:
                        lines.append(line)
    finally:
        root_logger.removeHandler(handler)

    # All three events landed.
    assert any('"text_delta"' in line for line in lines), lines
    assert any('"done"' in line for line in lines), lines
    # No ValueError-from-Context-reset in the captured warning logs.
    valueerror_records = [
        r for r in caplog_records
        if "different Context" in r.getMessage()
        or "Token" in r.getMessage() and "Context" in r.getMessage()
    ]
    assert not valueerror_records, (
        f"ValueError leaked from inner reset: {[r.getMessage() for r in valueerror_records]}"
    )


# task_id plumbing through /chat/stream — closes the gap PR #903
# left open. The streaming chat now auto-closes the originating
# handoff task on completion (mirrors the non-streaming /chat).

class _RecordingMeshClient:
    """Records all set_task_status calls for assertion."""

    def __init__(self):
        self.calls: list[tuple[str, str, dict]] = []

    async def set_task_status(self, task_id, status, *, result=None, error=None):
        self.calls.append((task_id, status, {"result": result, "error": error}))
        return {}


class _LoopWithTaskClose:
    """Yields a clean stream, then auto-close should fire via task_id."""

    def __init__(self):
        self.agent_id = "operator"
        self.state = "idle"
        self.current_task = None
        self.mesh_client = _RecordingMeshClient()
        # Async lock + queue used by the busy-handoff branch — even
        # though this test doesn't exercise it, the real chat_stream
        # checks ``self.current_task`` first.
        self._chat_lock = asyncio.Lock()
        # _checkpoint_chat_session is called in the finally.
        self._checkpointed = False

    async def _maybe_restore_session(self):
        return None

    async def _checkpoint_chat_session(self):
        self._checkpointed = True

    async def _chat_stream_inner(self, user_message):
        yield {"type": "text_delta", "content": "ok"}
        yield {"type": "done", "response": "all done", "tool_outputs": [], "tokens_used": 3}

    async def _auto_close_task(self, task_id, status, *, result_payload=None, error=None):
        # Delegate to the recording mesh_client so assertions stay simple.
        await self.mesh_client.set_task_status(
            task_id, status, result=result_payload, error=error,
        )


@pytest.mark.asyncio
async def test_chat_stream_auto_closes_task_on_clean_done():
    """When ``x-task-id`` rides the streaming wake, ``loop.chat_stream``
    must auto-call ``set_task_status(done)`` on clean completion —
    same contract as the non-streaming ``loop.chat`` (PR #903).
    Without this, dashboard-routed handoffs that happen to land on
    streaming chat never close.
    """
    # Use the real loop.chat_stream method on a stand-in that has
    # the helpers it needs but a simple inner generator. We invoke
    # chat_stream directly (not via the endpoint) — the contract
    # under test is the loop-side auto-close, not the endpoint
    # plumbing (covered by the test above).
    from src.agent.loop import AgentLoop

    stand_in = _LoopWithTaskClose()
    # Steal the real chat_stream method, bind it to our stand-in.
    method = AgentLoop.chat_stream.__get__(stand_in, AgentLoop)
    events = []
    async for ev in method("hi", task_id="task_alpha"):
        events.append(ev)

    # Stream produced both events.
    assert any(e.get("type") == "done" for e in events), events
    # Auto-close fired exactly once, with the right shape.
    assert len(stand_in.mesh_client.calls) == 1
    task_id, status, payload = stand_in.mesh_client.calls[0]
    assert task_id == "task_alpha"
    assert status == "done"
    assert payload["result"] == {"summary": "all done"}
    assert payload["error"] is None
    # Checkpoint still ran.
    assert stand_in._checkpointed
