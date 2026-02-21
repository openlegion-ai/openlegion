"""Tests for messaging channel adapters (Telegram, Discord).

Verifies the unified multi-agent chat interface works identically
across channel adapters: per-user active agent, @mentions, /commands,
agent name labels on responses, and notification push.
"""

from __future__ import annotations

import pytest

from src.channels.base import Channel, chunk_text

# ── Test concrete channel ─────────────────────────────────────

class StubChannel(Channel):
    """Concrete channel for testing base class behaviour."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.notifications: list[str] = []

    async def start(self):
        pass

    async def stop(self):
        pass

    async def send_notification(self, text: str) -> None:
        self.notifications.append(text)


def _make_channel(agents: list[str] | None = None, **overrides):
    agents = agents or ["alpha", "beta"]

    async def dispatch_fn(agent: str, message: str) -> str:
        return f"reply from {agent}"

    def list_agents_fn():
        return {a: {} for a in agents}

    def status_fn(name: str):
        return {"state": "running", "tasks_completed": 5}

    def costs_fn():
        return [{"agent": "alpha", "tokens": 100, "cost": 0.01}]

    resets: list[str] = []

    def reset_fn(agent: str) -> bool:
        resets.append(agent)
        return True

    defaults = {
        "dispatch_fn": dispatch_fn,
        "default_agent": "alpha",
        "list_agents_fn": list_agents_fn,
        "status_fn": status_fn,
        "costs_fn": costs_fn,
        "reset_fn": reset_fn,
    }
    defaults.update(overrides)
    ch = StubChannel(**defaults)
    ch._resets = resets
    return ch


# ── chunk_text ────────────────────────────────────────────────

class TestChunkText:
    def test_short_text_unchanged(self):
        assert chunk_text("hello", 100) == ["hello"]

    def test_splits_at_newline(self):
        text = "line1\nline2\nline3\nline4"
        chunks = chunk_text(text, 12)
        assert len(chunks) >= 2
        assert all(len(c) <= 12 for c in chunks)

    def test_splits_at_max_when_no_newline(self):
        text = "a" * 20
        chunks = chunk_text(text, 10)
        assert len(chunks) == 2
        assert chunks[0] == "a" * 10
        assert chunks[1] == "a" * 10


# ── dispatch ──────────────────────────────────────────────────

class TestDispatch:
    @pytest.mark.asyncio
    async def test_dispatch_default_agent(self):
        ch = _make_channel()
        result = await ch.dispatch("", "hi")
        assert result == "reply from alpha"

    @pytest.mark.asyncio
    async def test_dispatch_specific_agent(self):
        ch = _make_channel()
        result = await ch.dispatch("beta", "hi")
        assert result == "reply from beta"

    @pytest.mark.asyncio
    async def test_dispatch_no_agent(self):
        ch = _make_channel(default_agent="")
        result = await ch.dispatch("", "hi")
        assert "No agent" in result


# ── handle_message: @mention routing ──────────────────────────

class TestMentionRouting:
    @pytest.mark.asyncio
    async def test_at_mention_routes_to_agent(self):
        ch = _make_channel()
        result = await ch.handle_message("u1", "@beta what is AI?")
        assert "[beta]" in result
        assert "reply from beta" in result

    @pytest.mark.asyncio
    async def test_unknown_mention_returns_error(self):
        ch = _make_channel()
        result = await ch.handle_message("u1", "@unknown hi")
        assert "Unknown agent" in result

    @pytest.mark.asyncio
    async def test_plain_message_uses_active_agent(self):
        ch = _make_channel()
        result = await ch.handle_message("u1", "hello")
        assert "[alpha]" in result


# ── handle_message: /commands ─────────────────────────────────

class TestCommands:
    @pytest.mark.asyncio
    async def test_use_command(self):
        ch = _make_channel()
        result = await ch.handle_message("u1", "/use beta")
        assert "beta" in result
        # Subsequent messages go to beta
        result2 = await ch.handle_message("u1", "test")
        assert "[beta]" in result2

    @pytest.mark.asyncio
    async def test_use_unknown_agent(self):
        ch = _make_channel()
        result = await ch.handle_message("u1", "/use nonexistent")
        assert "Unknown agent" in result

    @pytest.mark.asyncio
    async def test_agents_command(self):
        ch = _make_channel()
        result = await ch.handle_message("u1", "/agents")
        assert "alpha" in result
        assert "beta" in result
        assert "(active)" in result

    @pytest.mark.asyncio
    async def test_status_command(self):
        ch = _make_channel()
        result = await ch.handle_message("u1", "/status")
        assert "running" in result
        assert "alpha" in result

    @pytest.mark.asyncio
    async def test_costs_command(self):
        ch = _make_channel()
        result = await ch.handle_message("u1", "/costs")
        assert "$" in result
        assert "alpha" in result

    @pytest.mark.asyncio
    async def test_reset_command(self):
        ch = _make_channel()
        result = await ch.handle_message("u1", "/reset")
        assert "reset" in result.lower()
        assert ch._resets == ["alpha"]

    @pytest.mark.asyncio
    async def test_broadcast_command(self):
        ch = _make_channel()
        result = await ch.handle_message("u1", "/broadcast hi everyone")
        assert "[alpha]" in result
        assert "[beta]" in result
        assert "Broadcast" in result

    @pytest.mark.asyncio
    async def test_help_command(self):
        ch = _make_channel()
        result = await ch.handle_message("u1", "/help")
        assert "/use" in result
        assert "/agents" in result
        assert "/broadcast" in result

    @pytest.mark.asyncio
    async def test_unknown_command(self):
        ch = _make_channel()
        result = await ch.handle_message("u1", "/bogus")
        assert "Unknown command" in result


# ── per-user active agent ─────────────────────────────────────

class TestPerUserAgent:
    @pytest.mark.asyncio
    async def test_independent_user_agents(self):
        ch = _make_channel()
        await ch.handle_message("u1", "/use beta")
        result_u1 = await ch.handle_message("u1", "msg")
        result_u2 = await ch.handle_message("u2", "msg")
        assert "[beta]" in result_u1
        assert "[alpha]" in result_u2  # u2 still on default


# ── notifications ─────────────────────────────────────────────

class TestNotifications:
    @pytest.mark.asyncio
    async def test_send_notification(self):
        ch = _make_channel()
        await ch.send_notification("[cron -> alpha] done")
        assert len(ch.notifications) == 1
        assert "cron" in ch.notifications[0]


# ── /addkey command ────────────────────────────────────────────

class TestAddKeyCommand:
    @pytest.mark.asyncio
    async def test_addkey_command(self):
        stored = []
        def addkey_fn(svc, key):
            stored.append((svc, key))
        ch = _make_channel(addkey_fn=addkey_fn)
        result = await ch.handle_message("u1", "/addkey brave_search sk-test-123")
        assert "stored" in result.lower()
        assert stored == [("brave_search", "sk-test-123")]

    @pytest.mark.asyncio
    async def test_addkey_not_forwarded(self):
        """The /addkey command must be consumed at channel layer, never dispatched."""
        dispatch_called = []

        async def dispatch_fn(agent: str, message: str) -> str:
            dispatch_called.append(message)
            return "reply"

        def list_agents_fn():
            return {"alpha": {}}

        stored = []
        def addkey_fn(svc, key):
            stored.append((svc, key))

        ch = StubChannel(
            dispatch_fn=dispatch_fn,
            default_agent="alpha",
            list_agents_fn=list_agents_fn,
            addkey_fn=addkey_fn,
        )
        await ch.handle_message("u1", "/addkey my_svc my_key")
        assert len(dispatch_called) == 0  # NOT forwarded to agent
        assert len(stored) == 1

    @pytest.mark.asyncio
    async def test_addkey_missing_args(self):
        stored = []
        def addkey_fn(svc, key):
            stored.append((svc, key))
        ch = _make_channel(addkey_fn=addkey_fn)
        result = await ch.handle_message("u1", "/addkey")
        assert "usage" in result.lower()
        assert len(stored) == 0

    @pytest.mark.asyncio
    async def test_addkey_missing_key(self):
        stored = []
        def addkey_fn(svc, key):
            stored.append((svc, key))
        ch = _make_channel(addkey_fn=addkey_fn)
        result = await ch.handle_message("u1", "/addkey service_only")
        assert "usage" in result.lower()
        assert len(stored) == 0

    @pytest.mark.asyncio
    async def test_addkey_in_help(self):
        ch = _make_channel()
        result = await ch.handle_message("u1", "/help")
        assert "/addkey" in result


# ── Discord !-to-/ command translation ────────────────────────

class TestDiscordCommandTranslation:
    """Verify that ! commands are handled the same as / commands."""

    @pytest.mark.asyncio
    async def test_handle_exclamation_commands(self):
        """Simulate what DiscordChannel does: translate ! to / before handle_message."""
        ch = _make_channel()
        text = "!agents"
        if text.startswith("!"):
            text = "/" + text[1:]
        result = await ch.handle_message("u1", text)
        assert "alpha" in result
        assert "beta" in result


# ── empty/silent response suppression ────────────────────────

class TestSilentResponseSuppression:
    @pytest.mark.asyncio
    async def test_empty_response_suppressed(self):
        """When dispatch returns empty string, handle_message returns empty."""
        async def silent_dispatch(agent: str, message: str) -> str:
            return ""

        ch = _make_channel(dispatch_fn=silent_dispatch)
        result = await ch.handle_message("u1", "heartbeat ping")
        assert result == ""

    @pytest.mark.asyncio
    async def test_whitespace_only_response_suppressed(self):
        """Whitespace-only responses are suppressed."""
        async def whitespace_dispatch(agent: str, message: str) -> str:
            return "   \n  "

        ch = _make_channel(dispatch_fn=whitespace_dispatch)
        result = await ch.handle_message("u1", "cron tick")
        assert result == ""

    @pytest.mark.asyncio
    async def test_none_response_suppressed(self):
        """None responses are suppressed."""
        async def none_dispatch(agent: str, message: str) -> str:
            return None

        ch = _make_channel(dispatch_fn=none_dispatch)
        result = await ch.handle_message("u1", "silent message")
        assert result == ""

    @pytest.mark.asyncio
    async def test_normal_response_not_suppressed(self):
        """Normal responses still appear with agent label."""
        ch = _make_channel()
        result = await ch.handle_message("u1", "hello")
        assert "[alpha]" in result
        assert "reply from alpha" in result


# ── /steer command ────────────────────────────────────────────

class TestSteerCommand:
    @pytest.mark.asyncio
    async def test_steer_command(self):
        """'/steer msg' calls steer_fn with (current_agent, msg)."""
        steered = []
        def steer_fn(agent: str, msg: str) -> None:
            steered.append((agent, msg))
        ch = _make_channel(steer_fn=steer_fn)
        result = await ch.handle_message("u1", "/steer focus on task X")
        assert "Steered" in result
        assert steered == [("alpha", "focus on task X")]

    @pytest.mark.asyncio
    async def test_steer_no_fn(self):
        """Graceful message when steer_fn is None."""
        ch = _make_channel()
        result = await ch.handle_message("u1", "/steer do something")
        assert "not available" in result.lower()

    @pytest.mark.asyncio
    async def test_steer_no_args(self):
        """Usage hint when no message provided."""
        steered = []
        def steer_fn(agent: str, msg: str) -> None:
            steered.append((agent, msg))
        ch = _make_channel(steer_fn=steer_fn)
        result = await ch.handle_message("u1", "/steer")
        assert "usage" in result.lower()
        assert len(steered) == 0


# ── /debug command ────────────────────────────────────────────

class TestDebugCommand:
    @pytest.mark.asyncio
    async def test_debug_command(self):
        """'/debug' returns formatted trace list with 'Recent traces:' header."""
        def debug_fn(trace_id=None):
            return [
                {"trace_id": "abc123def456", "agent": "alpha", "event_type": "chat"},
                {"trace_id": "xyz789000111", "agent": "beta", "event_type": "tool_start"},
            ]
        ch = _make_channel(debug_fn=debug_fn)
        result = await ch.handle_message("u1", "/debug")
        assert "Recent traces:" in result
        assert "abc123def456"[:12] in result
        assert "alpha" in result
        assert "chat" in result

    @pytest.mark.asyncio
    async def test_debug_with_trace_id(self):
        """'/debug <id>' passes trace_id to debug_fn and uses trace-specific header."""
        received_ids = []
        def debug_fn(trace_id=None):
            received_ids.append(trace_id)
            return [{"trace_id": "abc123", "agent": "alpha", "event_type": "chat"}]
        ch = _make_channel(debug_fn=debug_fn)
        result = await ch.handle_message("u1", "/debug abc123")
        assert received_ids == ["abc123"]
        assert "Trace abc123:" in result
        assert "Recent traces:" not in result

    @pytest.mark.asyncio
    async def test_debug_no_fn(self):
        """Graceful message when debug_fn is None."""
        ch = _make_channel()
        result = await ch.handle_message("u1", "/debug")
        assert "not available" in result.lower()

    @pytest.mark.asyncio
    async def test_steer_and_debug_in_help(self):
        """'/steer' and '/debug' appear in /help output."""
        ch = _make_channel()
        result = await ch.handle_message("u1", "/help")
        assert "/steer" in result
        assert "/debug" in result
