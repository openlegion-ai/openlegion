"""M17 regression — a wait-reply chat steer whose caller ALREADY timed
out must not be silently dropped when the ending turn is an agenda /
heartbeat turn.

Background (plan §8 #10 + #1213 fix-2): a busy-chat steer rides the
``_steer_queue`` as a 3-tuple ``(text, system_note, reply_future)``. The
caller waits on ``reply_future`` via ``asyncio.wait_for`` — which, on
timeout, CANCELS the future (so ``reply_future.done()`` is True) and
returns ``(True, None)`` to the caller. ``(True, None)`` means "genuinely
injected, do NOT dispatch a followup" — so if the heartbeat drain/sweep
then discards the queue tuple, the message text is lost forever.

#1213 fix-2 handled the LIVE-future subcase (caller still waiting →
resolve None → dedicated followup). This covers the narrower ALREADY-
TIMED-OUT subcase: the entry must be REQUEUED so the next real (non-
heartbeat) turn's catch-all drain delivers the text.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.loop import AgentLoop


def _make_loop() -> AgentLoop:
    memory = MagicMock()
    memory.get_high_salience_facts = MagicMock(return_value=[])
    memory.search = AsyncMock(return_value=[])
    memory.log_action = AsyncMock()

    tools = MagicMock()
    tools.get_tool_definitions = MagicMock(return_value=[])
    tools.get_descriptions = MagicMock(return_value="- no tools")
    tools.list_tools = MagicMock(return_value=[])

    llm = MagicMock()
    llm.default_model = "test-model"

    mesh_client = MagicMock()

    return AgentLoop(
        agent_id="test_agent",
        role="assistant",
        memory=memory,
        tools=tools,
        llm=llm,
        mesh_client=mesh_client,
    )


async def _timed_out_entry(loop: AgentLoop, text: str) -> asyncio.Future:
    """Reproduce the production race: enqueue a wait-reply steer while the
    agent is working, let the caller's ``wait_for`` time out (cancelling
    the future) — leaving the 3-tuple on the queue with a done future."""
    loop.state = "working"
    loop.current_task = None
    injected, reply = await loop.inject_steer_and_wait(text, timeout=0.01)
    # The caller genuinely timed out: injected True, reply None, and the
    # future was cancelled by asyncio.wait_for.
    assert (injected, reply) == (True, None)
    return injected


@pytest.mark.asyncio
async def test_sweep_requeues_timed_out_future_entry():
    """The end-of-heartbeat sweep must REQUEUE a timed-out 3-tuple (not
    drop it), so a later real turn's drain delivers the message."""
    loop = _make_loop()
    await _timed_out_entry(loop, "answer me after the agenda turn")

    loop._sweep_heartbeat_steer_futures()

    # Requeued — the next real turn's catch-all drain delivers the text.
    assert not loop._steer_queue.empty()
    drained = loop._drain_steer_messages()
    assert drained == [("answer me after the agenda turn", False)]
    # A cancelled future is never folded into the next turn's resolve set.
    assert loop._turn_steer_futures == []


@pytest.mark.asyncio
async def test_heartbeat_drain_requeues_timed_out_future_entry():
    """The initial heartbeat drain must also requeue (not drop) a timed-out
    3-tuple, and never inject its text into the agenda prompt."""
    loop = _make_loop()
    await _timed_out_entry(loop, "delivered on the next turn")

    steered = loop._drain_steer_messages(heartbeat=True)

    # Never injected into the agenda turn ...
    assert steered == []
    # ... but requeued for the next real turn to deliver.
    assert not loop._steer_queue.empty()
    assert loop._drain_steer_messages() == [("delivered on the next turn", False)]


@pytest.mark.asyncio
async def test_live_future_still_consumed_and_resolved_none():
    """Regression guard: a LIVE (not-done) wait-reply future is still
    consumed and resolved None on the heartbeat path (the #1213 fix-2
    behavior) — only the ALREADY-DONE case changed."""
    loop = _make_loop()
    live = asyncio.get_running_loop().create_future()
    await loop._steer_queue.put(("live chat", False, live))

    loop._sweep_heartbeat_steer_futures()

    assert live.done() and live.result() is None
    # Consumed, not requeued — the live caller dispatches its own followup.
    assert loop._steer_queue.empty()
