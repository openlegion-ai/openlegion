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
