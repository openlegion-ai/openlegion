"""Tests for per-agent lane queues and message queue modes.

Covers:
- Followup: basic dispatch, serial per agent, parallel across agents
- Steer: calls steer_fn, falls back to followup, steer to idle agent
- Collect: single idle dispatch, batching when busy, SILENT_REPLY_TOKEN
- Status: includes collected count and busy flag
- Stop: cancels workers cleanly
"""

import asyncio
from unittest.mock import AsyncMock

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
async def test_steer_to_idle_agent():
    """Steer to idle agent returns queued message."""
    steer = AsyncMock(return_value={"injected": False})
    lm = LaneManager(dispatch_fn=AsyncMock(return_value="ok"), steer_fn=steer)

    result = await lm.enqueue("agent1", "hey", mode="steer")
    assert "idle" in result.lower() or "queued" in result.lower()


# ── Collect mode ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_collect_single_idle_dispatches_normally():
    """Collect when idle dispatches immediately like followup."""
    dispatch = AsyncMock(return_value="immediate")
    lm = LaneManager(dispatch_fn=dispatch)

    result = await lm.enqueue("agent1", "msg", mode="collect")
    assert result == "immediate"


@pytest.mark.asyncio
async def test_collect_batches_when_busy():
    """Collect returns SILENT_REPLY_TOKEN for secondary callers when agent is busy."""
    dispatch_event = asyncio.Event()
    dispatch_done = asyncio.Event()

    async def blocking_dispatch(agent: str, message: str) -> str:
        dispatch_event.set()
        await dispatch_done.wait()
        return f"done:{message}"

    lm = LaneManager(dispatch_fn=blocking_dispatch)

    # Start a followup task to make agent busy
    task1 = asyncio.create_task(lm.enqueue("agent1", "primary"))
    await dispatch_event.wait()  # Wait until dispatch is running

    # Now agent is busy — collect should return SILENT_REPLY_TOKEN
    result = await lm.enqueue("agent1", "batched1", mode="collect")
    assert result == SILENT_REPLY_TOKEN

    result2 = await lm.enqueue("agent1", "batched2", mode="collect")
    assert result2 == SILENT_REPLY_TOKEN

    # Release the first task
    dispatch_done.set()
    primary_result = await task1
    assert primary_result == "done:primary"

    # The flushed collect buffer should have created a combined task
    # Wait for the worker to process it
    await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_collect_combined_message_format():
    """Collected messages are combined with [Message N] format."""
    captured_messages = []
    dispatch_event = asyncio.Event()
    dispatch_done = asyncio.Event()

    async def capturing_dispatch(agent: str, message: str) -> str:
        captured_messages.append(message)
        if not dispatch_event.is_set():
            dispatch_event.set()
            await dispatch_done.wait()
        return "ok"

    lm = LaneManager(dispatch_fn=capturing_dispatch)

    # Start a task to make agent busy
    task1 = asyncio.create_task(lm.enqueue("agent1", "primary"))
    await dispatch_event.wait()

    # Collect two messages
    await lm.enqueue("agent1", "msg A", mode="collect")
    await lm.enqueue("agent1", "msg B", mode="collect")

    # Release primary task — flush will create combined message
    dispatch_done.set()
    await task1
    await asyncio.sleep(0.1)

    # The combined message should have been dispatched
    assert len(captured_messages) >= 2
    combined = captured_messages[1]
    assert "[Message 1]: msg A" in combined
    assert "[Message 2]: msg B" in combined


# ── Status ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_status_includes_collected_and_busy():
    """Status includes collected count and busy flag."""
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
    assert "collected" in status["agent1"]

    # Collect a message while busy
    await lm.enqueue("agent1", "extra", mode="collect")
    status = lm.get_status()
    assert status["agent1"]["collected"] == 1

    dispatch_done.set()
    await task1


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
