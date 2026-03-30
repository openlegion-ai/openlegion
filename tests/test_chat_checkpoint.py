"""Tests for chat session checkpointing (persist/restore across restarts)."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.memory import MemoryStore


# ── MemoryStore checkpoint methods ────────────────────────────


@pytest.fixture()
def memory(tmp_path):
    store = MemoryStore(db_path=str(tmp_path / "test.db"))
    yield store
    store.close()


def test_save_and_load_checkpoint(memory):
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    memory.save_chat_checkpoint(messages, total_rounds=5, auto_continues=1, flush_triggered=True)

    cp = memory.load_chat_checkpoint()
    assert cp is not None
    assert cp["messages"] == messages
    assert cp["total_rounds"] == 5
    assert cp["auto_continues"] == 1
    assert cp["flush_triggered"] is True


def test_load_returns_none_when_empty(memory):
    assert memory.load_chat_checkpoint() is None


def test_clear_checkpoint(memory):
    messages = [{"role": "user", "content": "test"}]
    memory.save_chat_checkpoint(messages, 0, 0, False)
    assert memory.load_chat_checkpoint() is not None

    memory.clear_chat_checkpoint()
    assert memory.load_chat_checkpoint() is None


def test_save_overwrites_previous(memory):
    memory.save_chat_checkpoint([{"role": "user", "content": "first"}], 1, 0, False)
    memory.save_chat_checkpoint([{"role": "user", "content": "second"}], 2, 1, True)

    cp = memory.load_chat_checkpoint()
    assert cp["messages"] == [{"role": "user", "content": "second"}]
    assert cp["total_rounds"] == 2
    assert cp["auto_continues"] == 1
    assert cp["flush_triggered"] is True


def test_version_mismatch_clears_checkpoint(memory):
    messages = [{"role": "user", "content": "old version"}]
    memory.save_chat_checkpoint(messages, 0, 0, False)

    # Manually tamper with version to simulate upgrade
    memory.db.execute("UPDATE chat_checkpoint SET version = 999 WHERE id = 1")
    memory.db.commit()

    cp = memory.load_chat_checkpoint()
    assert cp is None
    # Verify it was actually cleared
    assert memory.load_chat_checkpoint() is None


def test_checkpoint_with_tool_calls(memory):
    """Ensure tool_calls and tool results survive serialization."""
    messages = [
        {"role": "user", "content": "search for info"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": '{"query": "test"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_abc123", "content": '{"results": []}'},
        {"role": "assistant", "content": "No results found."},
    ]
    memory.save_chat_checkpoint(messages, 3, 0, False)

    cp = memory.load_chat_checkpoint()
    assert cp["messages"] == messages
    assert cp["messages"][1]["tool_calls"][0]["id"] == "call_abc123"


def test_checkpoint_with_multimodal_content(memory):
    """Multimodal content blocks (with base64) survive round-trip."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
            ],
        },
        {"role": "assistant", "content": "It's an image."},
    ]
    memory.save_chat_checkpoint(messages, 1, 0, False)

    cp = memory.load_chat_checkpoint()
    assert cp["messages"] == messages
    assert cp["messages"][0]["content"][1]["type"] == "image_url"


# ── AgentLoop integration ─────────────────────────────────────


def _make_loop(tmp_path, *, chat_messages=None):
    """Create a minimal AgentLoop with a real MemoryStore for checkpoint tests."""
    from src.agent.loop import AgentLoop

    memory = MemoryStore(db_path=str(tmp_path / "agent.db"))

    llm = MagicMock()
    llm.model = "openai/gpt-4o-mini"
    skills = MagicMock()
    skills.get_tool_definitions.return_value = []
    skills.get_descriptions.return_value = ""
    skills.get_loop_exempt_tools.return_value = set()
    mesh_client = MagicMock()
    mesh_client.is_standalone = False
    mesh_client.list_agents = AsyncMock(return_value={})

    loop = AgentLoop(
        agent_id="test-agent",
        role="tester",
        memory=memory,
        skills=skills,
        llm=llm,
        mesh_client=mesh_client,
        workspace=None,
        context_manager=None,
    )
    if chat_messages:
        loop._chat_messages = list(chat_messages)
    return loop, memory


@pytest.mark.asyncio
async def test_restore_on_first_chat_call(tmp_path):
    """Simulate restart: save checkpoint, create new loop, verify restore."""
    # First "session": populate and checkpoint
    loop1, memory = _make_loop(tmp_path)
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    loop1._chat_messages = messages
    loop1._chat_total_rounds = 7
    loop1._chat_auto_continues = 1
    await loop1._checkpoint_chat_session()
    memory.close()

    # Second "session": new loop, same DB — should restore
    loop2, memory2 = _make_loop(tmp_path)
    assert loop2._chat_messages == []

    await loop2._maybe_restore_session()
    assert loop2._chat_messages == messages
    assert loop2._chat_total_rounds == 7
    assert loop2._chat_auto_continues == 1
    memory2.close()


@pytest.mark.asyncio
async def test_no_restore_when_messages_exist(tmp_path):
    """If _chat_messages is already populated, skip restore."""
    loop, memory = _make_loop(tmp_path)
    memory.save_chat_checkpoint(
        [{"role": "user", "content": "old"}], 5, 0, False,
    )
    loop._chat_messages = [{"role": "user", "content": "current"}]

    await loop._maybe_restore_session()
    assert loop._chat_messages == [{"role": "user", "content": "current"}]
    assert loop._chat_total_rounds == 0  # unchanged
    memory.close()


@pytest.mark.asyncio
async def test_reset_clears_checkpoint(tmp_path):
    """reset_chat should clear the persisted checkpoint."""
    loop, memory = _make_loop(tmp_path)
    loop._chat_messages = [{"role": "user", "content": "test"}]
    await loop._checkpoint_chat_session()
    assert memory.load_chat_checkpoint() is not None

    await loop.reset_chat()
    assert memory.load_chat_checkpoint() is None
    memory.close()


@pytest.mark.asyncio
async def test_checkpoint_empty_messages_clears(tmp_path):
    """Checkpointing with empty messages should clear the row."""
    loop, memory = _make_loop(tmp_path)
    # First save something
    loop._chat_messages = [{"role": "user", "content": "test"}]
    await loop._checkpoint_chat_session()
    assert memory.load_chat_checkpoint() is not None

    # Now clear and checkpoint
    loop._chat_messages = []
    await loop._checkpoint_chat_session()
    assert memory.load_chat_checkpoint() is None
    memory.close()


@pytest.mark.asyncio
async def test_no_restore_without_memory(tmp_path):
    """If memory store is None, restore is a no-op."""
    from src.agent.loop import AgentLoop

    skills = MagicMock()
    skills.get_tool_definitions.return_value = []
    skills.get_descriptions.return_value = ""
    skills.get_loop_exempt_tools.return_value = set()
    mesh_client = MagicMock()
    mesh_client.is_standalone = False

    loop = AgentLoop(
        agent_id="test-agent",
        role="tester",
        memory=None,
        skills=skills,
        llm=MagicMock(),
        mesh_client=mesh_client,
        workspace=None,
        context_manager=None,
    )
    # Should not raise
    await loop._maybe_restore_session()
    assert loop._chat_messages == []

    # Checkpoint should also be a no-op
    await loop._checkpoint_chat_session()


@pytest.mark.asyncio
async def test_flush_triggered_restored(tmp_path):
    """The context manager's flush_triggered flag should be restored."""
    loop1, memory = _make_loop(tmp_path)
    context_mgr = MagicMock()
    context_mgr._flush_triggered = True
    loop1.context_manager = context_mgr
    loop1._chat_messages = [{"role": "user", "content": "hello"}]
    await loop1._checkpoint_chat_session()
    memory.close()

    loop2, memory2 = _make_loop(tmp_path)
    context_mgr2 = MagicMock()
    context_mgr2._flush_triggered = False
    loop2.context_manager = context_mgr2

    await loop2._maybe_restore_session()
    assert context_mgr2._flush_triggered is True
    memory2.close()
