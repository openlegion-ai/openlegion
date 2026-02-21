"""Tests for Slack channel adapter.

Verifies Socket Mode integration, composite user key for per-thread
agent tracking, pairing flow, command translation, and notifications.
All tests mock slack_bolt -- no live API needed.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.channels.slack import SlackChannel, _get_user_key


# ── helpers ──────────────────────────────────────────────────────

def _make_channel(
    paired: dict | None = None,
    agents: list[str] | None = None,
    **overrides,
) -> SlackChannel:
    agents = agents or ["alpha", "beta"]

    async def dispatch_fn(agent: str, message: str) -> str:
        return f"reply from {agent}"

    def list_agents_fn():
        return {a: {} for a in agents}

    def status_fn(name: str):
        return {"state": "running", "tasks_completed": 5}

    def costs_fn():
        return [{"agent": "alpha", "tokens": 100, "cost": 0.01}]

    defaults = {
        "bot_token": "xoxb-test",
        "app_token": "xapp-test",
        "dispatch_fn": dispatch_fn,
        "default_agent": "alpha",
        "list_agents_fn": list_agents_fn,
        "status_fn": status_fn,
        "costs_fn": costs_fn,
    }
    defaults.update(overrides)
    ch = SlackChannel(**defaults)
    if paired is not None:
        ch._pairing._data = paired
    return ch


def _make_say() -> AsyncMock:
    return AsyncMock()


# ── composite user key ──────────────────────────────────────────

class TestCompositeUserKey:
    def test_with_thread(self):
        assert _get_user_key("U123", "1234.5678") == "U123:1234.5678"

    def test_without_thread(self):
        assert _get_user_key("U123", None) == "U123"

    @pytest.mark.asyncio
    async def test_independent_per_thread_agent_tracking(self):
        """Different threads should maintain independent active agents."""
        ch = _make_channel(paired={"owner": "U1", "allowed": []})
        say = _make_say()

        # Thread A: switch to beta
        event_a = {"user": "U1", "text": "!use beta", "channel": "C1", "ts": "1.0", "thread_ts": "1.0"}
        await ch._on_message(event_a, say)
        await asyncio.sleep(0.05)  # let create_task complete
        user_key_a = _get_user_key("U1", "1.0")
        assert ch._get_active_agent(user_key_a) == "beta"

        # Thread B: should still be on default (alpha)
        user_key_b = _get_user_key("U1", "2.0")
        assert ch._get_active_agent(user_key_b) == "alpha"


# ── pairing flow ────────────────────────────────────────────────

class TestPairing:
    @pytest.mark.asyncio
    async def test_pairing_with_correct_code(self):
        ch = _make_channel(
            paired={"owner": None, "allowed": [], "pairing_code": "abc123"},
        )
        say = _make_say()
        event = {"user": "U1", "text": "!start abc123", "channel": "C1", "ts": "1.0"}
        await ch._on_message(event, say)
        assert ch._pairing.owner == "U1"
        say.assert_called()

    @pytest.mark.asyncio
    async def test_pairing_with_wrong_code(self):
        ch = _make_channel(
            paired={"owner": None, "allowed": [], "pairing_code": "abc123"},
        )
        say = _make_say()
        event = {"user": "U1", "text": "!start wrongcode", "channel": "C1", "ts": "1.0"}
        await ch._on_message(event, say)
        assert ch._pairing.owner is None
        # Should inform about pairing requirement
        say.assert_called()
        msg = say.call_args[1]["text"]
        assert "pairing" in msg.lower()

    @pytest.mark.asyncio
    async def test_unpaired_bot_rejects_regular_messages(self):
        ch = _make_channel(
            paired={"owner": None, "allowed": [], "pairing_code": "abc123"},
        )
        say = _make_say()
        event = {"user": "U1", "text": "hello", "channel": "C1", "ts": "1.0"}
        await ch._on_message(event, say)
        assert ch._pairing.owner is None
        msg = say.call_args[1]["text"]
        assert "pairing" in msg.lower()

    @pytest.mark.asyncio
    async def test_disallowed_user_rejected(self):
        ch = _make_channel(paired={"owner": "U_OWNER", "allowed": []})
        say = _make_say()
        event = {"user": "U_OTHER", "text": "!start code", "channel": "C1", "ts": "1.0"}
        await ch._on_message(event, say)
        msg = say.call_args[1]["text"]
        assert "denied" in msg.lower()

    @pytest.mark.asyncio
    async def test_disallowed_user_ignored_for_regular_messages(self):
        ch = _make_channel(paired={"owner": "U_OWNER", "allowed": []})
        say = _make_say()
        event = {"user": "U_OTHER", "text": "hello", "channel": "C1", "ts": "1.0"}
        await ch._on_message(event, say)
        say.assert_not_called()

    @pytest.mark.asyncio
    async def test_allowed_user_can_chat(self):
        ch = _make_channel(paired={"owner": "U_OWNER", "allowed": ["U_FRIEND"]})
        say = _make_say()
        event = {"user": "U_FRIEND", "text": "hello", "channel": "C1", "ts": "1.0"}
        await ch._on_message(event, say)
        # Should dispatch (the response goes via create_task, but say is called)
        await asyncio.sleep(0.05)
        say.assert_called()

    @pytest.mark.asyncio
    async def test_owner_allow_command(self):
        ch = _make_channel(paired={"owner": "U_OWNER", "allowed": []})
        say = _make_say()
        event = {"user": "U_OWNER", "text": "!allow U_FRIEND", "channel": "C1", "ts": "1.0"}
        await ch._on_message(event, say)
        assert ch._pairing.is_allowed("U_FRIEND")

    @pytest.mark.asyncio
    async def test_non_owner_allow_rejected(self):
        ch = _make_channel(paired={"owner": "U_OWNER", "allowed": ["U_OTHER"]})
        say = _make_say()
        event = {"user": "U_OTHER", "text": "!allow U_SOMEONE", "channel": "C1", "ts": "1.0"}
        await ch._on_message(event, say)
        assert not ch._pairing.is_allowed("U_SOMEONE")
        msg = say.call_args[1]["text"]
        assert "owner" in msg.lower()

    @pytest.mark.asyncio
    async def test_owner_revoke_command(self):
        ch = _make_channel(paired={"owner": "U_OWNER", "allowed": ["U_FRIEND"]})
        say = _make_say()
        event = {"user": "U_OWNER", "text": "!revoke U_FRIEND", "channel": "C1", "ts": "1.0"}
        await ch._on_message(event, say)
        assert not ch._pairing.is_allowed("U_FRIEND")

    @pytest.mark.asyncio
    async def test_owner_paired_command(self):
        ch = _make_channel(paired={"owner": "U_OWNER", "allowed": ["U_A"]})
        say = _make_say()
        event = {"user": "U_OWNER", "text": "!paired", "channel": "C1", "ts": "1.0"}
        await ch._on_message(event, say)
        msg = say.call_args[1]["text"]
        assert "U_OWNER" in msg
        assert "U_A" in msg


# ── message routing ─────────────────────────────────────────────

class TestMessageRouting:
    @pytest.mark.asyncio
    async def test_message_dispatches_to_handle_message(self):
        """Regular messages should go through handle_message and get a response."""
        ch = _make_channel(paired={"owner": "U1", "allowed": []})
        say = _make_say()
        event = {"user": "U1", "text": "hello world", "channel": "C1", "ts": "1.0"}
        await ch._on_message(event, say)
        await asyncio.sleep(0.05)
        say.assert_called()
        msg = say.call_args[1]["text"]
        assert "[alpha]" in msg

    @pytest.mark.asyncio
    async def test_bot_messages_ignored(self):
        ch = _make_channel(paired={"owner": "U1", "allowed": []})
        say = _make_say()
        event = {"user": "U1", "text": "hello", "channel": "C1", "ts": "1.0", "bot_id": "B123"}
        await ch._on_message(event, say)
        say.assert_not_called()

    @pytest.mark.asyncio
    async def test_subtype_messages_ignored(self):
        ch = _make_channel(paired={"owner": "U1", "allowed": []})
        say = _make_say()
        event = {"user": "U1", "text": "hello", "channel": "C1", "ts": "1.0", "subtype": "channel_join"}
        await ch._on_message(event, say)
        say.assert_not_called()


# ── command translation ─────────────────────────────────────────

class TestCommandTranslation:
    @pytest.mark.asyncio
    async def test_exclamation_translated_to_slash(self):
        """! commands should be translated to / for base class handling."""
        ch = _make_channel(paired={"owner": "U1", "allowed": []})
        say = _make_say()
        event = {"user": "U1", "text": "!agents", "channel": "C1", "ts": "1.0"}
        await ch._on_message(event, say)
        await asyncio.sleep(0.05)
        say.assert_called()
        msg = say.call_args[1]["text"]
        assert "alpha" in msg
        assert "beta" in msg


# ── workspace filtering (thread_ts) ─────────────────────────────

class TestThreadRouting:
    @pytest.mark.asyncio
    async def test_thread_ts_used_as_key(self):
        ch = _make_channel(paired={"owner": "U1", "allowed": []})
        say = _make_say()
        # Message in a thread
        event = {
            "user": "U1", "text": "!use beta",
            "channel": "C1", "ts": "2.0", "thread_ts": "1.0",
        }
        await ch._on_message(event, say)
        await asyncio.sleep(0.05)

        # thread_ts=1.0 should be the key
        key = _get_user_key("U1", "1.0")
        assert ch._get_active_agent(key) == "beta"

        # A different thread keeps default
        key2 = _get_user_key("U1", "3.0")
        assert ch._get_active_agent(key2) == "alpha"


# ── send_notification ────────────────────────────────────────────

class TestSendNotification:
    @pytest.mark.asyncio
    async def test_notification_calls_chat_post_message(self):
        ch = _make_channel(paired={"owner": "U1", "allowed": []})
        ch._channel_ids = {"C1", "C2"}
        mock_client = MagicMock()
        mock_client.chat_postMessage = AsyncMock()
        ch._bolt_app = MagicMock()
        ch._bolt_app.client = mock_client

        await ch.send_notification("test notification")
        assert mock_client.chat_postMessage.call_count == 2

    @pytest.mark.asyncio
    async def test_notification_no_channels_is_noop(self):
        ch = _make_channel(paired={"owner": "U1", "allowed": []})
        ch._channel_ids = set()
        ch._bolt_app = MagicMock()
        await ch.send_notification("test")
        # No calls should be made

    @pytest.mark.asyncio
    async def test_notification_no_app_is_noop(self):
        ch = _make_channel(paired={"owner": "U1", "allowed": []})
        ch._bolt_app = None
        await ch.send_notification("test")
