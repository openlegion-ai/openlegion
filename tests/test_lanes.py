"""Tests for per-agent lane queues and message queue modes.

Covers:
- Followup: basic dispatch, serial per agent, parallel across agents
- Steer: calls steer_fn, falls back to followup, steer to idle agent
- Status: queue depth and busy flag
- Stop: cancels workers cleanly
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.host.lanes import SILENT_REPLY_TOKEN, LaneManager

# ── Followup mode ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_followup_basic_dispatch():
    """Basic dispatch returns response."""
    dispatch = AsyncMock(return_value="hello")
    lm = LaneManager(dispatch_fn=dispatch)

    result = await lm.enqueue("agent1", "hi")
    assert result == "hello"
    dispatch.assert_awaited_once_with("agent1", "hi")


@pytest.mark.asyncio
async def test_followup_serial_per_agent():
    """Tasks for the same agent execute serially."""
    call_order = []

    async def slow_dispatch(agent: str, message: str) -> str:
        call_order.append(message)
        await asyncio.sleep(0.05)
        return f"done:{message}"

    lm = LaneManager(dispatch_fn=slow_dispatch)

    r1, r2 = await asyncio.gather(
        lm.enqueue("agent1", "first"),
        lm.enqueue("agent1", "second"),
    )
    assert r1 == "done:first"
    assert r2 == "done:second"
    assert call_order == ["first", "second"]


@pytest.mark.asyncio
async def test_followup_parallel_across_agents():
    """Different agents run in parallel."""
    started = []

    async def tracking_dispatch(agent: str, message: str) -> str:
        started.append(agent)
        await asyncio.sleep(0.05)
        return f"done:{agent}"

    lm = LaneManager(dispatch_fn=tracking_dispatch)

    r1, r2 = await asyncio.gather(
        lm.enqueue("agent1", "msg1"),
        lm.enqueue("agent2", "msg2"),
    )
    assert r1 == "done:agent1"
    assert r2 == "done:agent2"
    # Both should have started before either finished
    assert set(started) == {"agent1", "agent2"}


# ── Steer mode ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_steer_calls_steer_fn():
    """Steer mode calls steer_fn directly when available."""
    dispatch = AsyncMock(return_value="normal")
    steer = AsyncMock(return_value={"injected": True})
    lm = LaneManager(dispatch_fn=dispatch, steer_fn=steer)

    # Ensure lane exists
    result = await lm.enqueue("agent1", "redirect now", mode="steer")
    assert "injected" in result.lower() or "Steered" in result
    steer.assert_awaited_once_with("agent1", "redirect now")
    # dispatch_fn should NOT have been called
    dispatch.assert_not_awaited()


@pytest.mark.asyncio
async def test_steer_falls_back_without_steer_fn():
    """Without steer_fn, steer falls back to followup."""
    dispatch = AsyncMock(return_value="followup response")
    lm = LaneManager(dispatch_fn=dispatch, steer_fn=None)

    result = await lm.enqueue("agent1", "redirect now", mode="steer")
    assert result == "followup response"
    dispatch.assert_awaited_once_with("agent1", "redirect now")


@pytest.mark.asyncio
async def test_steer_to_idle_agent_falls_back_to_followup():
    """Steer to idle agent (injected=False) dispatches via followup to wake it."""
    dispatch = AsyncMock(return_value="woke up")
    steer = AsyncMock(return_value={"injected": False})
    lm = LaneManager(dispatch_fn=dispatch, steer_fn=steer)

    result = await lm.enqueue("agent1", "hey", mode="steer")
    # Should have called steer_fn first, then fallen back to dispatch
    steer.assert_awaited_once_with("agent1", "hey")
    dispatch.assert_awaited_once_with("agent1", "hey")
    assert result == "woke up"


@pytest.mark.asyncio
async def test_steer_busy_agent_returns_injected_message():
    """Steer to a busy agent returns 'injected' confirmation."""
    steer = AsyncMock(return_value={"injected": True})
    dispatch_event = asyncio.Event()
    dispatch_done = asyncio.Event()

    async def blocking_dispatch(agent: str, message: str) -> str:
        dispatch_event.set()
        await dispatch_done.wait()
        return "ok"

    lm = LaneManager(dispatch_fn=blocking_dispatch, steer_fn=steer)

    # Make agent busy
    task1 = asyncio.create_task(lm.enqueue("agent1", "work"))
    await dispatch_event.wait()
    assert lm.get_status()["agent1"]["busy"] is True

    # Steer while busy
    result = await lm.enqueue("agent1", "redirect", mode="steer")
    assert "injected" in result.lower()
    steer.assert_awaited_once_with("agent1", "redirect")

    dispatch_done.set()
    await task1


@pytest.mark.asyncio
async def test_steer_active_agent_does_not_dispatch():
    """Steer to an active agent (injected=True) does NOT call dispatch_fn."""
    dispatch = AsyncMock(return_value="normal")
    steer = AsyncMock(return_value={"injected": True})
    lm = LaneManager(dispatch_fn=dispatch, steer_fn=steer)

    result = await lm.enqueue("agent1", "redirect", mode="steer")
    steer.assert_awaited_once_with("agent1", "redirect")
    dispatch.assert_not_awaited()
    assert "injected" in result.lower()


@pytest.mark.asyncio
async def test_steer_wakeup_rate_limiting():
    """Rate limiting prevents excess followup wakeups from idle steers."""
    dispatch = AsyncMock(return_value="woke")
    steer = AsyncMock(return_value={"injected": False})
    lm = LaneManager(dispatch_fn=dispatch, steer_fn=steer)

    results = []
    for i in range(15):
        r = await lm.enqueue("agent1", f"msg{i}", mode="steer")
        results.append(r)

    # First 10 should dispatch (followup wakeup)
    woke_count = sum(1 for r in results if r == "woke")
    assert woke_count == 10
    assert dispatch.await_count == 10

    # Remaining 5 should be rate-limited (SILENT_REPLY_TOKEN)
    silent_count = sum(1 for r in results if r == SILENT_REPLY_TOKEN)
    assert silent_count == 5


@pytest.mark.asyncio
async def test_steer_wakeup_rate_window_resets():
    """Rate limit prunes old timestamps so the window resets."""

    dispatch = AsyncMock(return_value="woke")
    steer = AsyncMock(return_value={"injected": False})
    lm = LaneManager(dispatch_fn=dispatch, steer_fn=steer)

    # Use up all 10 wakeups
    for i in range(10):
        await lm.enqueue("agent1", f"msg{i}", mode="steer")
    assert dispatch.await_count == 10

    # Next one should be rate-limited
    r = await lm.enqueue("agent1", "blocked", mode="steer")
    assert r == SILENT_REPLY_TOKEN
    assert dispatch.await_count == 10

    # Simulate time passing beyond the window by backdating all timestamps
    import time

    from src.host.lanes import _STEER_WAKEUP_WINDOW

    old_time = time.monotonic() - _STEER_WAKEUP_WINDOW - 1
    lm._steer_wakeup_ts["agent1"] = [old_time] * 10

    # Now it should allow again
    r = await lm.enqueue("agent1", "allowed", mode="steer")
    assert r == "woke"
    assert dispatch.await_count == 11


@pytest.mark.asyncio
async def test_steer_fn_exception_falls_back_to_followup():
    """When steer_fn raises an exception, falls back to followup dispatch."""
    dispatch = AsyncMock(return_value="dispatched")
    steer = AsyncMock(side_effect=Exception("connection refused"))
    lm = LaneManager(dispatch_fn=dispatch, steer_fn=steer)

    result = await lm.enqueue("agent1", "important msg", mode="steer")

    steer.assert_awaited_once()
    dispatch.assert_awaited_once()
    assert result == "dispatched"


# ── Status ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_status_reports_busy_and_queue_depth():
    """Status reports busy flag and queue depth per agent."""
    dispatch_event = asyncio.Event()
    dispatch_done = asyncio.Event()

    async def blocking_dispatch(agent: str, message: str) -> str:
        dispatch_event.set()
        await dispatch_done.wait()
        return "ok"

    lm = LaneManager(dispatch_fn=blocking_dispatch)

    task1 = asyncio.create_task(lm.enqueue("agent1", "work"))
    await dispatch_event.wait()

    status = lm.get_status()
    assert "agent1" in status
    assert status["agent1"]["busy"] is True
    assert status["agent1"]["queued"] == 0
    assert status["agent1"]["pending"] == 1
    # collect mode removed — no collected key should appear
    assert "collected" not in status["agent1"]

    dispatch_done.set()
    await task1


# ── EventBus integration (queue_changed) ─────────────────────


@pytest.mark.asyncio
async def test_queue_changed_emitted_when_bus_wired():
    """A wired EventBus receives queue_changed on enqueue, dequeue, and completion."""
    dispatch = AsyncMock(return_value="ok")
    bus = MagicMock()
    lm = LaneManager(dispatch_fn=dispatch)
    lm.set_event_bus(bus)

    await lm.enqueue("agent1", "hi")
    await asyncio.sleep(0.05)  # let the worker's finally block run

    queue_emits = [
        c for c in bus.emit.call_args_list
        if c.args and c.args[0] == "queue_changed"
    ]
    # enqueue + busy=True + terminal-finally → at least 3 emits for one task
    assert len(queue_emits) >= 3
    for c in queue_emits:
        assert c.kwargs.get("agent") == "agent1"
        assert c.kwargs.get("data") == {"agent": "agent1"}


@pytest.mark.asyncio
async def test_no_bus_wired_does_not_break_lane():
    """Without an EventBus the lane works identically (emits are no-ops)."""
    dispatch = AsyncMock(return_value="ok")
    lm = LaneManager(dispatch_fn=dispatch)  # no set_event_bus

    result = await lm.enqueue("agent1", "hi")
    assert result == "ok"
    dispatch.assert_awaited_once_with("agent1", "hi")


# ── Stop ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_remove_lane():
    """remove_lane cleans up all state for an agent."""
    lm = LaneManager(dispatch_fn=AsyncMock(return_value="ok"))

    await lm.enqueue("agent1", "hi")
    await lm.enqueue("agent2", "hi")
    assert "agent1" in lm._workers
    assert "agent2" in lm._workers

    lm.remove_lane("agent1")
    assert "agent1" not in lm._workers
    assert "agent1" not in lm._queues
    assert "agent1" not in lm._pending
    assert "agent1" not in lm._busy
    # agent2 should be unaffected
    assert "agent2" in lm._workers


@pytest.mark.asyncio
async def test_remove_lane_nonexistent():
    """remove_lane on nonexistent agent doesn't raise."""
    lm = LaneManager(dispatch_fn=AsyncMock(return_value="ok"))
    lm.remove_lane("nonexistent")  # should not raise


@pytest.mark.asyncio
async def test_stop_cancels_workers():
    """Stop cancels all worker tasks."""
    lm = LaneManager(dispatch_fn=AsyncMock(return_value="ok"))

    # Create a lane so a worker exists
    await lm.enqueue("agent1", "hi")
    assert "agent1" in lm._workers

    await lm.stop()
    assert len(lm._workers) == 0


# ── Fix 4: origin / auto_notify / notify_fn ─────────────────────


@pytest.mark.asyncio
async def test_auto_notify_triggers_notify_fn():
    """notify_fn called with origin+result+agent when auto_notify=True and result is non-empty."""
    dispatch = AsyncMock(return_value="task done!")
    notify = AsyncMock()
    lm = LaneManager(dispatch_fn=dispatch, notify_fn=notify)

    origin = {"channel": "whatsapp", "user": "+1234"}
    result = await lm.enqueue(
        "agent1", "do work", origin=origin, auto_notify=True,
    )

    assert result == "task done!"
    # Give the background task a chance to run
    await asyncio.sleep(0.05)
    notify.assert_awaited_once_with(origin, "task done!", "agent1")


@pytest.mark.asyncio
async def test_auto_notify_false_does_not_trigger():
    """notify_fn not called when auto_notify=False."""
    dispatch = AsyncMock(return_value="done")
    notify = AsyncMock()
    lm = LaneManager(dispatch_fn=dispatch, notify_fn=notify)

    origin = {"channel": "whatsapp", "user": "+1234"}
    await lm.enqueue("agent1", "msg", origin=origin, auto_notify=False)
    await asyncio.sleep(0.05)
    notify.assert_not_awaited()


@pytest.mark.asyncio
async def test_auto_notify_no_origin_does_not_trigger():
    """notify_fn not called when origin=None even if auto_notify=True."""
    dispatch = AsyncMock(return_value="done")
    notify = AsyncMock()
    lm = LaneManager(dispatch_fn=dispatch, notify_fn=notify)

    await lm.enqueue("agent1", "msg", origin=None, auto_notify=True)
    await asyncio.sleep(0.05)
    notify.assert_not_awaited()


@pytest.mark.asyncio
async def test_auto_notify_silent_reply_does_not_trigger():
    """notify_fn not called when result is SILENT_REPLY_TOKEN."""
    dispatch = AsyncMock(return_value=SILENT_REPLY_TOKEN)
    notify = AsyncMock()
    lm = LaneManager(dispatch_fn=dispatch, notify_fn=notify)

    origin = {"channel": "whatsapp", "user": "+1234"}
    await lm.enqueue("agent1", "msg", origin=origin, auto_notify=True)
    await asyncio.sleep(0.05)
    notify.assert_not_awaited()


@pytest.mark.asyncio
async def test_auto_notify_empty_result_does_not_trigger():
    """notify_fn not called when result is empty."""
    dispatch = AsyncMock(return_value="   ")
    notify = AsyncMock()
    lm = LaneManager(dispatch_fn=dispatch, notify_fn=notify)

    origin = {"channel": "whatsapp", "user": "+1234"}
    await lm.enqueue("agent1", "msg", origin=origin, auto_notify=True)
    await asyncio.sleep(0.05)
    notify.assert_not_awaited()


@pytest.mark.asyncio
async def test_origin_passed_to_dispatch_fn():
    """When origin is set, dispatch_fn receives it as a kwarg."""
    received_kwargs = {}

    async def recording_dispatch(agent, message, **kwargs):
        received_kwargs.update(kwargs)
        return "ok"

    lm = LaneManager(dispatch_fn=recording_dispatch)
    origin = {"channel": "telegram", "user": "42"}
    await lm.enqueue("agent1", "hello", origin=origin)

    assert received_kwargs.get("origin") == origin


@pytest.mark.asyncio
async def test_trace_id_propagated_to_dispatch_context():
    """Session observability keystone: a followup enqueued with a trace_id
    runs its dispatch_fn under that trace in ``current_trace_id`` — so a
    handoff-woken agent's outbound /chat carries the ORIGINATING session's
    X-Trace-Id and a multi-agent chain reconstructs as one session. Before the
    fix the lane enqueued with no trace_id, so every handoff minted a fresh,
    disconnected trace downstream."""
    from src.shared.trace import current_trace_id

    seen: dict = {}

    async def recording_dispatch(agent, message, **kwargs):
        seen["trace"] = current_trace_id.get()
        return "ok"

    lm = LaneManager(dispatch_fn=recording_dispatch)
    await lm.enqueue("agent1", "handoff work", trace_id="tr_chain0000001")
    for _ in range(50):
        if "trace" in seen:
            break
        await asyncio.sleep(0.01)

    assert seen.get("trace") == "tr_chain0000001"


# ── Per-task lane watchdog (Bug 4) ─────────────────────────────


class TestLaneWatchdog:
    """A hung dispatch_fn previously blocked the lane forever — every
    subsequent task for that agent piled up indefinitely. The watchdog
    wraps dispatch in ``asyncio.wait_for`` with a configurable timeout
    and marks the durable task ``failed`` so the originator sees a
    back-edge event."""

    @pytest.mark.asyncio
    async def test_hanging_dispatch_times_out_and_lane_recovers(self):
        """Stuck dispatch is cancelled at the timeout; the next task runs."""
        call_log = []

        async def hanging_then_normal(agent, message, **kwargs):
            call_log.append(message)
            if message == "stuck":
                await asyncio.sleep(60)  # never returns within the timeout
                return "should never reach here"
            return f"ok:{message}"

        # Tight timeout so the test runs quickly.
        lm = LaneManager(
            dispatch_fn=hanging_then_normal,
            task_timeout_seconds=1,
        )

        # First task hangs and should fail with TimeoutError.
        with pytest.raises(asyncio.TimeoutError):
            await lm.enqueue("agent1", "stuck")

        # Second task on the same lane must still be picked up — the
        # lane worker survived.
        result = await lm.enqueue("agent1", "next")
        assert result == "ok:next"
        assert call_log == ["stuck", "next"]

    @pytest.mark.asyncio
    async def test_normal_dispatch_within_timeout_unchanged(self):
        """Fast dispatch behaves exactly as before."""
        dispatch = AsyncMock(return_value="hi")
        lm = LaneManager(dispatch_fn=dispatch, task_timeout_seconds=5)

        result = await lm.enqueue("agent1", "ping")
        assert result == "hi"
        dispatch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_env_var_overrides_default_timeout(self, monkeypatch):
        """``OPENLEGION_LANE_TIMEOUT_SECONDS`` is honoured at module load.
        Re-importing the module after setting the env var picks it up."""
        import importlib

        monkeypatch.setenv("OPENLEGION_LANE_TIMEOUT_SECONDS", "42")
        import src.host.lanes as lanes_mod

        importlib.reload(lanes_mod)
        try:
            lm = lanes_mod.LaneManager(dispatch_fn=AsyncMock(return_value=""))
            assert lm._task_timeout_seconds == 42
        finally:
            # Restore the default for downstream tests.
            monkeypatch.delenv("OPENLEGION_LANE_TIMEOUT_SECONDS", raising=False)
            importlib.reload(lanes_mod)

    @pytest.mark.asyncio
    async def test_timeout_marks_durable_task_failed(self):
        """When the queued task has a ``task_id`` AND a tasks_store is
        wired, the watchdog calls ``update_status('failed')`` so the
        originator's back-edge inbox sees ``lane_timeout``."""
        async def hanging(agent, message, **kwargs):
            await asyncio.sleep(60)
            return "never"

        tasks_store = MagicMock()
        tasks_store.update_status = MagicMock()
        lm = LaneManager(
            dispatch_fn=hanging,
            task_timeout_seconds=1,
            tasks_store=tasks_store,
        )

        with pytest.raises(asyncio.TimeoutError):
            await lm.enqueue("agent1", "stuck", task_id="task-abc-123")

        # update_status should have been invoked with the failed status
        # and the lane_timeout marker.
        tasks_store.update_status.assert_called_once()
        args, kwargs = tasks_store.update_status.call_args
        assert args[0] == "task-abc-123"
        assert args[1] == "failed"
        assert kwargs.get("actor") == "lane_watchdog"
        extra = kwargs.get("extra_payload", {})
        assert extra.get("error") == "lane_timeout"

    @pytest.mark.asyncio
    async def test_timeout_without_task_id_skips_task_update(self):
        """Free-form lane messages (no task_id) just time out — no
        durable-task bookkeeping to do."""
        async def hanging(agent, message, **kwargs):
            await asyncio.sleep(60)
            return "never"

        tasks_store = MagicMock()
        tasks_store.update_status = MagicMock()
        lm = LaneManager(
            dispatch_fn=hanging,
            task_timeout_seconds=1,
            tasks_store=tasks_store,
        )

        with pytest.raises(asyncio.TimeoutError):
            await lm.enqueue("agent1", "stuck")  # no task_id

        tasks_store.update_status.assert_not_called()


# ── Seam follow-up Fix 5: lane refuses dispatch to quarantined agent ──


class TestLaneQuarantineGate:
    """LaneManager.quarantine_check shorts the dispatch when the agent
    is quarantined — protects the broken-credential path from queuing
    more work onto an agent that can't run."""

    @pytest.mark.asyncio
    async def test_lane_refuses_dispatch_for_quarantined_agent(self):
        dispatch = AsyncMock(return_value="should not be called")
        quarantined = {"agent-a"}
        lm = LaneManager(
            dispatch_fn=dispatch,
            quarantine_check=lambda a: a in quarantined,
        )
        with pytest.raises(RuntimeError) as ei:
            await lm.enqueue("agent-a", "hello")
        assert "quarantined" in str(ei.value).lower()
        assert "edit_agent" in str(ei.value)
        dispatch.assert_not_called()

    @pytest.mark.asyncio
    async def test_lane_dispatches_when_not_quarantined(self):
        dispatch = AsyncMock(return_value="ok")
        quarantined: set[str] = set()
        lm = LaneManager(
            dispatch_fn=dispatch,
            quarantine_check=lambda a: a in quarantined,
        )
        result = await lm.enqueue("agent-a", "hello")
        assert result == "ok"
        dispatch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_lane_set_quarantine_check_setter(self):
        dispatch = AsyncMock(return_value="ok")
        lm = LaneManager(dispatch_fn=dispatch)
        # First call goes through — no quarantine check wired.
        r1 = await lm.enqueue("agent-a", "first")
        assert r1 == "ok"
        # Wire the check after construction.
        lm.set_quarantine_check(lambda a: True)
        with pytest.raises(RuntimeError):
            await lm.enqueue("agent-a", "second")

    @pytest.mark.asyncio
    async def test_lane_closes_durable_task_on_quarantine_reject(self):
        """Codex P2 follow-up: when a queued wake carries a task_id and
        the assignee is quarantined, the durable task must be marked
        failed so the originating agent's back-edge inbox sees a
        terminal event instead of the task dangling in 'working'."""
        dispatch = AsyncMock(return_value="ok")
        tasks_store = MagicMock()
        tasks_store.update_status = MagicMock()
        lm = LaneManager(
            dispatch_fn=dispatch,
            tasks_store=tasks_store,
            quarantine_check=lambda a: True,
        )
        with pytest.raises(RuntimeError):
            await lm.enqueue(
                "agent-a", "hello", task_id="task-abc",
            )
        tasks_store.update_status.assert_called_once()
        call_kwargs = tasks_store.update_status.call_args
        assert call_kwargs.args[0] == "task-abc"
        assert call_kwargs.args[1] == "failed"
        assert call_kwargs.kwargs.get("actor") == "lane_quarantine"
        assert call_kwargs.kwargs.get("extra_payload", {}).get("error") == "agent_quarantined"


# ── Per-agent timeout + back-edge integration (operator workflow awareness) ──


class TestLanePerAgentTimeout:
    """The per-agent override is the workflow-awareness layer's way to
    give workflow-stage agents (page-writer etc.) a tight cap while
    leaving deep-research agents on the generous default."""

    @pytest.mark.asyncio
    async def test_default_used_when_no_override(self):
        lm = LaneManager(
            dispatch_fn=AsyncMock(return_value=""),
            task_timeout_seconds=300,
        )
        assert lm._timeout_for("any-agent") == 300

    @pytest.mark.asyncio
    async def test_per_agent_timeout_override_honored(self):
        """``set_agent_timeout`` overrides the module default for the
        watchdog wait. The override forces a timeout for one agent while
        the default cap is preserved for others."""
        async def slow(agent, message, **kwargs):
            await asyncio.sleep(5)
            return "would-have-completed"

        # Default 300s — would let dispatch complete normally. The
        # per-agent override is clamped to 60s, but that's still well
        # over the dispatch hang we want to validate. We assert the
        # resolver math directly + use a separately built lane with a
        # 1s default (the lane will pick it up via _timeout_for).
        lm = LaneManager(dispatch_fn=slow, task_timeout_seconds=300)
        lm.set_agent_timeout("agent-tight", 60)
        # Untouched agents still see the default.
        assert lm._timeout_for("agent-loose") == 300
        assert lm._timeout_for("agent-tight") == 60

        # Functional check: tight cap on the dispatch path.
        lm2 = LaneManager(dispatch_fn=slow, task_timeout_seconds=1)
        with pytest.raises(asyncio.TimeoutError):
            await lm2.enqueue("agent-x", "stuck")

    @pytest.mark.asyncio
    async def test_set_agent_timeout_clamps_below_60(self):
        """``seconds`` < 60 is clamped to 60 — a 1-second cap would make
        the watchdog the bottleneck instead of a safety net."""
        lm = LaneManager(dispatch_fn=AsyncMock(return_value=""))
        lm.set_agent_timeout("agent-a", 5)
        assert lm._timeout_for("agent-a") == 60
        lm.set_agent_timeout("agent-a", 120)
        assert lm._timeout_for("agent-a") == 120
        # None drops back to the default.
        lm.set_agent_timeout("agent-a", None)
        assert lm._timeout_for("agent-a") == lm._task_timeout_seconds

    @pytest.mark.asyncio
    async def test_timeout_fires_back_edge_with_task_failed(self):
        """When a tasks_store is wired AND a back_edge_fn is set, the
        watchdog calls the back-edge writer with event_kind=task_failed
        and ``payload_extras={'error':'lane_timeout','timeout_seconds':N}``
        so the originator's inbox sees the failure."""
        async def hanging(agent, message, **kwargs):
            await asyncio.sleep(60)
            return "never"

        # Pre-flight ``tasks_store.get`` returns the still-working row so
        # the watchdog's "already terminal" guard does not skip the
        # update + back-edge. The post-update ``get`` returns the failed
        # record that should flow through to the back-edge fn.
        working_record = {
            "id": "task_xyz",
            "assignee": "agent1",
            "title": "stuck work",
            "status": "working",
            "origin": {"kind": "agent", "channel": "", "user": "scout"},
        }
        failed_record = {
            "id": "task_xyz",
            "assignee": "agent1",
            "title": "stuck work",
            "status": "failed",
            "origin": {"kind": "agent", "channel": "", "user": "scout"},
        }
        tasks_store = MagicMock()
        tasks_store.update_status = MagicMock()
        tasks_store.get = MagicMock(side_effect=[working_record, failed_record])

        back_edge_calls = []

        def back_edge(task_record, *, event_kind, payload_extras=None):
            back_edge_calls.append((task_record, event_kind, payload_extras))

        lm = LaneManager(
            dispatch_fn=hanging,
            task_timeout_seconds=1,
            tasks_store=tasks_store,
        )
        lm.set_back_edge_fn(back_edge)

        with pytest.raises(asyncio.TimeoutError):
            await lm.enqueue("agent1", "stuck", task_id="task_xyz")

        tasks_store.update_status.assert_called_once()
        assert len(back_edge_calls) == 1
        rec, kind, extras = back_edge_calls[0]
        assert rec == failed_record
        assert kind == "task_failed"
        assert extras.get("error") == "lane_timeout"
        assert extras.get("timeout_seconds") == 1

    @pytest.mark.asyncio
    async def test_timeout_skips_back_edge_on_invalid_status_transition(self):
        """When ``update_status`` raises ``InvalidStatusTransition`` — the
        race where the assignee terminated the task between our SELECT
        and UPDATE — the worker logs the race and SKIPS the back-edge,
        avoiding a double-notification."""
        from src.host.orchestration import InvalidStatusTransition

        async def hanging(agent, message, **kwargs):
            await asyncio.sleep(60)
            return "never"

        tasks_store = MagicMock()
        tasks_store.update_status = MagicMock(
            side_effect=InvalidStatusTransition("already done"),
        )
        tasks_store.get = MagicMock(return_value={"id": "task_x"})

        back_edge_calls = []

        def back_edge(task_record, *, event_kind, payload_extras=None):
            back_edge_calls.append((task_record, event_kind, payload_extras))

        lm = LaneManager(
            dispatch_fn=hanging,
            task_timeout_seconds=1,
            tasks_store=tasks_store,
        )
        lm.set_back_edge_fn(back_edge)

        with pytest.raises(asyncio.TimeoutError):
            await lm.enqueue("agent1", "stuck", task_id="task_x")

        # update_status fired (and raised), back-edge was skipped (race).
        tasks_store.update_status.assert_called_once()
        assert back_edge_calls == []

    @pytest.mark.asyncio
    async def test_set_back_edge_fn_clears_with_none(self):
        """Setter accepts None to clear the wired function — used by
        teardown paths."""
        lm = LaneManager(dispatch_fn=AsyncMock(return_value=""))
        lm.set_back_edge_fn(lambda *a, **kw: None)
        assert lm._back_edge_fn is not None
        lm.set_back_edge_fn(None)
        assert lm._back_edge_fn is None

    @pytest.mark.asyncio
    async def test_lane_watchdog_skips_back_edge_when_task_already_terminal(self):
        """Same-status race: the assignee finished the task (terminal
        status) between dispatch and the watchdog's timeout, so the
        pre-flight ``tasks_store.get`` sees ``status in {done, failed,
        cancelled}``. The watchdog MUST skip both ``update_status`` AND
        the back-edge fire — otherwise the assignee's actual back-edge
        payload would be clobbered with a synthetic ``lane_timeout``."""
        async def hanging(agent, message, **kwargs):
            await asyncio.sleep(60)
            return "never"

        already_failed_record = {
            "id": "task_race",
            "assignee": "agent1",
            "title": "completed before watchdog",
            "status": "failed",
            "origin": {"kind": "agent", "channel": "", "user": "scout"},
        }
        tasks_store = MagicMock()
        tasks_store.update_status = MagicMock()
        tasks_store.get = MagicMock(return_value=already_failed_record)

        back_edge_calls = []

        def back_edge(task_record, *, event_kind, payload_extras=None):
            back_edge_calls.append((task_record, event_kind, payload_extras))

        lm = LaneManager(
            dispatch_fn=hanging,
            task_timeout_seconds=1,
            tasks_store=tasks_store,
        )
        lm.set_back_edge_fn(back_edge)

        with pytest.raises(asyncio.TimeoutError):
            await lm.enqueue("agent1", "stuck", task_id="task_race")

        # Pre-flight read happened; update_status + back-edge skipped.
        tasks_store.get.assert_called_once_with("task_race")
        tasks_store.update_status.assert_not_called()
        assert back_edge_calls == []

    @pytest.mark.asyncio
    async def test_lane_watchdog_fires_back_edge_when_task_still_working(self):
        """Opposite of the race case: the task is still ``working`` when
        the watchdog fires, so the pre-check falls through and the
        watchdog transitions the row + fires the back-edge exactly once
        with ``event_kind='task_failed'`` and the ``lane_timeout``
        marker in ``payload_extras``."""
        async def hanging(agent, message, **kwargs):
            await asyncio.sleep(60)
            return "never"

        working_record = {
            "id": "task_live",
            "assignee": "agent1",
            "title": "still running",
            "status": "working",
            "origin": {"kind": "agent", "channel": "", "user": "scout"},
        }
        failed_record = dict(working_record, status="failed")
        tasks_store = MagicMock()
        tasks_store.update_status = MagicMock()
        # First ``get`` (pre-check) → working. Second ``get`` (after
        # update_status) → failed (what flows into the back-edge).
        tasks_store.get = MagicMock(side_effect=[working_record, failed_record])

        back_edge_calls = []

        def back_edge(task_record, *, event_kind, payload_extras=None):
            back_edge_calls.append((task_record, event_kind, payload_extras))

        lm = LaneManager(
            dispatch_fn=hanging,
            task_timeout_seconds=1,
            tasks_store=tasks_store,
        )
        lm.set_back_edge_fn(back_edge)

        with pytest.raises(asyncio.TimeoutError):
            await lm.enqueue("agent1", "stuck", task_id="task_live")

        tasks_store.update_status.assert_called_once()
        assert len(back_edge_calls) == 1
        rec, kind, extras = back_edge_calls[0]
        assert rec == failed_record
        assert kind == "task_failed"
        assert extras.get("error") == "lane_timeout"
        assert extras.get("timeout_seconds") == 1


# ── Per-agent watchdog YAML wiring ─────────────────────────────────


class TestPerAgentWatchdogYamlWiring:
    """The runtime's startup loop reads
    ``agents.yaml.<id>.settings.watchdog_ttl_seconds`` and pushes each
    valid entry into ``LaneManager.set_agent_timeout``. The loop must:

    * dispatch ints unchanged into the per-agent overlay,
    * coerce numeric strings,
    * log + skip non-numeric values without aborting other entries,
    * skip entries that omit the field entirely,
    * be defensive against a malformed ``agents.yaml`` (the outer
      try/except in runtime.py wraps the loop so startup is not
      aborted by a parse error).
    """

    def _apply_yaml_overrides(self, lm: LaneManager, cfg: dict) -> list[str]:
        """Mirror the runtime.py loop body — same conditions, same
        coercion order. Returns a list of agent ids that were skipped
        (for assertion). Kept inline so the test pins the exact logic
        rather than monkey-patching the runtime module which would
        drag in the full ``RuntimeContext`` startup graph."""
        skipped: list[str] = []
        try:
            agents_cfg = cfg.get("agents", {}) or {}
            for aid, entry in agents_cfg.items():
                if not isinstance(entry, dict):
                    skipped.append(aid)
                    continue
                settings = entry.get("settings") or {}
                ttl = settings.get("watchdog_ttl_seconds")
                if ttl is None:
                    continue
                try:
                    lm.set_agent_timeout(aid, int(ttl))
                except (TypeError, ValueError):
                    skipped.append(aid)
        except Exception:
            pass
        return skipped

    def test_single_agent_ttl_lands_in_overlay(self):
        lm = LaneManager(dispatch_fn=AsyncMock(return_value=""))
        cfg = {
            "agents": {
                "alpha": {"settings": {"watchdog_ttl_seconds": 120}},
            },
        }
        skipped = self._apply_yaml_overrides(lm, cfg)
        assert skipped == []
        assert lm._per_agent_timeouts == {"alpha": 120}
        # _timeout_for resolves to the override, not the module default.
        assert lm._timeout_for("alpha") == 120

    def test_multiple_agents_independent_overrides(self):
        lm = LaneManager(dispatch_fn=AsyncMock(return_value=""))
        cfg = {
            "agents": {
                "alpha": {"settings": {"watchdog_ttl_seconds": 600}},
                "beta":  {"settings": {"watchdog_ttl_seconds": 900}},
                # gamma has no ttl → no override; falls back to default.
                "gamma": {"settings": {}},
                # delta omits ``settings`` entirely.
                "delta": {},
            },
        }
        self._apply_yaml_overrides(lm, cfg)
        assert lm._per_agent_timeouts == {"alpha": 600, "beta": 900}
        # gamma + delta fall back to module default (not in overlay dict).
        assert "gamma" not in lm._per_agent_timeouts
        assert "delta" not in lm._per_agent_timeouts

    def test_numeric_string_ttl_is_coerced(self):
        """YAML can yield a string when the user quotes the number;
        the runtime loop coerces via int() before passing through."""
        lm = LaneManager(dispatch_fn=AsyncMock(return_value=""))
        cfg = {
            "agents": {
                "alpha": {"settings": {"watchdog_ttl_seconds": "300"}},
            },
        }
        skipped = self._apply_yaml_overrides(lm, cfg)
        assert skipped == []
        assert lm._per_agent_timeouts["alpha"] == 300

    def test_bad_ttl_value_skipped_other_entries_apply(self):
        """A non-numeric value logs and skips that one agent. Other
        agents in the same dict still get their overrides applied."""
        lm = LaneManager(dispatch_fn=AsyncMock(return_value=""))
        cfg = {
            "agents": {
                "alpha": {"settings": {"watchdog_ttl_seconds": "not_a_number"}},
                "beta":  {"settings": {"watchdog_ttl_seconds": 420}},
            },
        }
        skipped = self._apply_yaml_overrides(lm, cfg)
        assert "alpha" in skipped
        # Alpha was skipped; beta still applied.
        assert "alpha" not in lm._per_agent_timeouts
        assert lm._per_agent_timeouts == {"beta": 420}

    def test_clamps_below_minimum_floor(self):
        """``set_agent_timeout`` clamps to a 60s floor so a tiny TTL
        in agents.yaml can't make the watchdog the bottleneck."""
        lm = LaneManager(dispatch_fn=AsyncMock(return_value=""))
        cfg = {
            "agents": {
                "alpha": {"settings": {"watchdog_ttl_seconds": 5}},
            },
        }
        self._apply_yaml_overrides(lm, cfg)
        # Clamped to 60 by ``set_agent_timeout``.
        assert lm._per_agent_timeouts["alpha"] == 60


# ── H7: lane queue depth cap (backpressure) ──────────────────────


@pytest.mark.asyncio
async def test_lane_queue_full_raises_backpressure():
    """When the followup queue is at its depth cap, enqueue raises
    LaneQueueFull instead of silently dropping or blocking forever."""
    from src.host.lanes import LaneQueueFull

    release = asyncio.Event()
    started = asyncio.Event()

    async def blocking_dispatch(agent: str, message: str) -> str:
        started.set()  # signal the worker pulled the first item
        await release.wait()  # then pin the worker so the queue fills
        return "done"

    maxsize = 3
    lm = LaneManager(dispatch_fn=blocking_dispatch, queue_maxsize=maxsize)

    # First enqueue: the worker pulls it and blocks in dispatch. Fire it
    # and wait until the worker is confirmed pinned so the queue won't
    # drain underneath us.
    first = asyncio.ensure_future(lm.enqueue("a", "m0"))
    await asyncio.wait_for(started.wait(), timeout=2.0)

    # Now buffer exactly ``maxsize`` more items — the worker is pinned, so
    # nothing drains and the queue fills to its cap deterministically.
    buffered = [
        asyncio.ensure_future(lm.enqueue("a", f"buf{i}"))
        for i in range(maxsize)
    ]
    await asyncio.sleep(0.05)
    assert lm.lane_full("a") is True

    # A further enqueue is rejected with backpressure, not dropped.
    with pytest.raises(LaneQueueFull):
        await lm.enqueue("a", "overflow")

    # Drain cleanly.
    release.set()
    await asyncio.gather(first, *buffered)
    await lm.stop()


@pytest.mark.asyncio
async def test_lane_unbounded_when_maxsize_zero():
    """maxsize<=0 disables the bound (lane_full always False)."""
    release = asyncio.Event()

    async def blocking_dispatch(agent: str, message: str) -> str:
        await release.wait()
        return "done"

    lm = LaneManager(dispatch_fn=blocking_dispatch, queue_maxsize=0)
    pending = [asyncio.ensure_future(lm.enqueue("a", f"m{i}")) for i in range(20)]
    await asyncio.sleep(0.05)
    assert lm.lane_full("a") is False
    release.set()
    await asyncio.gather(*pending)
    await lm.stop()


# ── system_note threading (PR: system wakes render as system rows) ──


@pytest.mark.asyncio
async def test_followup_system_note_threads_to_dispatch():
    """Flagged followups pass system_note=True to the dispatch fn."""
    dispatch = AsyncMock(return_value="ok")
    lm = LaneManager(dispatch_fn=dispatch)

    await lm.enqueue("agent1", "wake msg", system_note=True)
    dispatch.assert_awaited_once_with("agent1", "wake msg", system_note=True)


@pytest.mark.asyncio
async def test_followup_unflagged_keeps_legacy_signature():
    """Unflagged dispatches must not pass the kwarg (legacy dispatch fns
    accept only (agent, message))."""
    dispatch = AsyncMock(return_value="ok")
    lm = LaneManager(dispatch_fn=dispatch)

    await lm.enqueue("agent1", "hi")
    dispatch.assert_awaited_once_with("agent1", "hi")


@pytest.mark.asyncio
async def test_steer_system_note_reaches_steer_fn():
    """Flagged steers pass system_note to steer_fn so a busy agent's
    steer queue records the honest role."""
    steer = AsyncMock(return_value={"injected": True})
    lm = LaneManager(dispatch_fn=AsyncMock(), steer_fn=steer)

    await lm.enqueue("agent1", "watch update", mode="steer", system_note=True)
    steer.assert_awaited_once_with("agent1", "watch update", system_note=True)


@pytest.mark.asyncio
async def test_steer_unflagged_uses_legacy_two_arg_call():
    """Human steers keep the legacy (agent, message) steer_fn call."""
    steer = AsyncMock(return_value={"injected": True})
    lm = LaneManager(dispatch_fn=AsyncMock(), steer_fn=steer)

    await lm.enqueue("agent1", "user steer", mode="steer")
    steer.assert_awaited_once_with("agent1", "user steer")


@pytest.mark.asyncio
async def test_steer_idle_fallback_carries_system_note():
    """The idle-agent steer→followup conversion must not drop the flag —
    a blackboard wake to an idle operator would otherwise render as a
    fake user bubble."""
    dispatch = AsyncMock(return_value="ok")
    steer = AsyncMock(return_value={"injected": False})
    lm = LaneManager(dispatch_fn=dispatch, steer_fn=steer)

    await lm.enqueue("agent1", "watch update", mode="steer", system_note=True)
    dispatch.assert_awaited_once_with("agent1", "watch update", system_note=True)


@pytest.mark.asyncio
async def test_steer_error_fallback_carries_system_note():
    """The steer-error followup fallback must not drop the flag either."""
    dispatch = AsyncMock(return_value="ok")
    steer = AsyncMock(side_effect=RuntimeError("boom"))
    lm = LaneManager(dispatch_fn=dispatch, steer_fn=steer)

    await lm.enqueue("agent1", "watch update", mode="steer", system_note=True)
    dispatch.assert_awaited_once_with("agent1", "watch update", system_note=True)
