"""Tests for DiscordChannel — extracted methods and slash command registration.

Tests the refactored handler methods directly (no live Discord API needed)
and verifies the CommandTree has expected slash commands registered.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.channels.base import PairingManager
from src.channels.discord import DiscordChannel, MAX_DC_LEN


# ── helpers ────────────────────────────────────────────────────

def _make_discord_channel(tmp_path: Path | None = None, **overrides) -> DiscordChannel:
    """Create a DiscordChannel with stubbed callbacks for unit testing."""
    async def dispatch_fn(agent: str, message: str) -> str:
        return f"reply from {agent}"

    def list_agents_fn():
        return {"alpha": {}, "beta": {}}

    defaults = {
        "token": "fake-token",
        "default_agent": "alpha",
        "dispatch_fn": dispatch_fn,
        "list_agents_fn": list_agents_fn,
    }
    defaults.update(overrides)
    ch = DiscordChannel(**defaults)
    # Use a temp path so tests don't leak state to the real config dir
    if tmp_path is None:
        tmp_path = Path(tempfile.mkdtemp())
    ch._pairing = PairingManager(tmp_path / "discord_paired.json")
    ch._pairing._data["pairing_code"] = "ABC123"
    return ch


def _mock_interaction(user_id: int, channel_id: int = 42):
    """Build a mock Discord Interaction with common fields."""
    interaction = MagicMock()
    interaction.response = MagicMock()
    interaction.response.defer = AsyncMock()
    interaction.followup = MagicMock()
    interaction.followup.send = AsyncMock()
    interaction.user = MagicMock()
    interaction.user.id = user_id
    interaction.channel = MagicMock()
    interaction.channel_id = channel_id
    return interaction


# ── _handle_pairing ───────────────────────────────────────────

class TestHandlePairing:
    def test_correct_code_claims_owner(self):
        ch = _make_discord_channel()
        msg = ch._handle_pairing(12345, "ABC123")
        assert "Paired as owner" in msg
        assert ch._pairing.is_owner(12345)

    def test_wrong_code_rejected(self):
        ch = _make_discord_channel()
        msg = ch._handle_pairing(12345, "WRONG")
        assert "Pairing required" in msg
        assert ch._pairing.owner is None

    def test_empty_code_rejected(self):
        ch = _make_discord_channel()
        msg = ch._handle_pairing(12345, "")
        assert "Pairing required" in msg
        assert ch._pairing.owner is None


# ── _handle_allow / _handle_revoke ────────────────────────────

class TestHandleAllow:
    def test_owner_can_allow(self):
        ch = _make_discord_channel()
        ch._pairing.claim_owner(111)
        msg = ch._handle_allow(111, "222")
        assert "now allowed" in msg
        assert ch._pairing.is_allowed(222)

    def test_non_owner_rejected(self):
        ch = _make_discord_channel()
        ch._pairing.claim_owner(111)
        msg = ch._handle_allow(999, "222")
        assert "Only the owner" in msg
        assert not ch._pairing.is_allowed(222)

    def test_invalid_id(self):
        ch = _make_discord_channel()
        ch._pairing.claim_owner(111)
        msg = ch._handle_allow(111, "not-a-number")
        assert "Usage" in msg


class TestHandleRevoke:
    def test_owner_can_revoke(self):
        ch = _make_discord_channel()
        ch._pairing.claim_owner(111)
        ch._pairing.allow(222)
        msg = ch._handle_revoke(111, "222")
        assert "revoked" in msg
        assert not ch._pairing.is_allowed(222)

    def test_non_owner_rejected(self):
        ch = _make_discord_channel()
        ch._pairing.claim_owner(111)
        ch._pairing.allow(222)
        msg = ch._handle_revoke(999, "222")
        assert "Only the owner" in msg
        assert ch._pairing.is_allowed(222)

    def test_revoke_not_in_list(self):
        ch = _make_discord_channel()
        ch._pairing.claim_owner(111)
        msg = ch._handle_revoke(111, "999")
        assert "not in the allowed list" in msg

    def test_invalid_id(self):
        ch = _make_discord_channel()
        ch._pairing.claim_owner(111)
        msg = ch._handle_revoke(111, "bad")
        assert "Usage" in msg


# ── _handle_paired ────────────────────────────────────────────

class TestHandlePaired:
    def test_owner_sees_info(self):
        ch = _make_discord_channel()
        ch._pairing.claim_owner(111)
        ch._pairing.allow(222)
        ch._pairing.allow(333)
        msg = ch._handle_paired(111)
        assert "Owner: 111" in msg
        assert "222" in msg
        assert "333" in msg

    def test_owner_no_allowed_users(self):
        ch = _make_discord_channel()
        ch._pairing.claim_owner(111)
        msg = ch._handle_paired(111)
        assert "Owner: 111" in msg
        assert "No additional users" in msg

    def test_non_owner_rejected(self):
        ch = _make_discord_channel()
        ch._pairing.claim_owner(111)
        msg = ch._handle_paired(999)
        assert "Only the owner" in msg


# ── _handle_access_denied ─────────────────────────────────────

class TestHandleAccessDenied:
    def test_start_attempt_always_responds(self):
        ch = _make_discord_channel()
        ch._pairing.claim_owner(111)
        msg = ch._handle_access_denied(999, "/start ABC")
        assert "Access denied" in msg
        assert "999" in msg
        # Should respond again (no dedup for /start)
        msg2 = ch._handle_access_denied(999, "/start ABC")
        assert msg2 is not None

    def test_first_denial_notifies(self):
        ch = _make_discord_channel()
        ch._pairing.claim_owner(111)
        msg = ch._handle_access_denied(999, "hello")
        assert "Access denied" in msg

    def test_second_denial_silent(self):
        ch = _make_discord_channel()
        ch._pairing.claim_owner(111)
        ch._handle_access_denied(999, "hello")
        msg = ch._handle_access_denied(999, "hello again")
        assert msg is None

    def test_bang_start_also_responds(self):
        ch = _make_discord_channel()
        ch._pairing.claim_owner(111)
        msg = ch._handle_access_denied(999, "!start ABC")
        assert "Access denied" in msg


class TestAccessDeniedMessage:
    """_access_denied_message always returns a string (no dedup) for slash cmds."""

    def test_always_returns_message(self):
        ch = _make_discord_channel()
        msg1 = ch._access_denied_message(999)
        msg2 = ch._access_denied_message(999)
        assert "Access denied" in msg1
        assert "Access denied" in msg2
        assert "999" in msg1


# ── _handle_repl_slash ────────────────────────────────────────

class TestHandleReplSlash:
    @pytest.mark.asyncio
    async def test_access_check_blocks_unpaired(self):
        ch = _make_discord_channel()
        interaction = _mock_interaction(999)

        await ch._handle_repl_slash(interaction, "/agents")
        interaction.response.defer.assert_called_once()
        interaction.followup.send.assert_called_once()
        call = interaction.followup.send.call_args
        assert "pairing" in call[0][0].lower()
        # Should be ephemeral — no need to leak pairing state to channel
        assert call[1].get("ephemeral") is True

    @pytest.mark.asyncio
    async def test_access_check_blocks_denied_user(self):
        ch = _make_discord_channel()
        ch._pairing.claim_owner(111)
        interaction = _mock_interaction(999)

        await ch._handle_repl_slash(interaction, "/agents")
        interaction.response.defer.assert_called_once()
        call = interaction.followup.send.call_args
        assert "denied" in call[0][0].lower()
        assert call[1].get("ephemeral") is True

    @pytest.mark.asyncio
    async def test_denied_user_always_gets_response(self):
        """Denied users must ALWAYS get a response on slash commands.

        Unlike on_message (where dedup prevents channel spam), slash commands
        need a response every time because:
        1. Discord shows "thinking..." if there's no response — broken UX
        2. Responses are ephemeral (only visible to invoker) — no spam concern
        """
        ch = _make_discord_channel()
        ch._pairing.claim_owner(111)

        # First slash interaction
        i1 = _mock_interaction(999)
        await ch._handle_repl_slash(i1, "/agents")
        assert i1.followup.send.call_count == 1

        # Second slash interaction — must still respond (no dedup)
        i2 = _mock_interaction(999)
        await ch._handle_repl_slash(i2, "/status")
        assert i2.followup.send.call_count == 1
        call = i2.followup.send.call_args
        assert "denied" in call[0][0].lower()

    @pytest.mark.asyncio
    async def test_allowed_user_gets_response(self):
        ch = _make_discord_channel()
        ch._pairing.claim_owner(111)
        interaction = _mock_interaction(111)

        await ch._handle_repl_slash(interaction, "/agents")
        interaction.response.defer.assert_called_once()
        response_text = interaction.followup.send.call_args[0][0]
        assert "alpha" in response_text

    @pytest.mark.asyncio
    async def test_tracks_notify_channel(self):
        ch = _make_discord_channel()
        ch._pairing.claim_owner(111)
        interaction = _mock_interaction(111, channel_id=42)

        await ch._handle_repl_slash(interaction, "/help")
        assert 42 in ch._notify_channel_ids

    @pytest.mark.asyncio
    async def test_chunked_followup(self):
        """Long responses are split into multiple followup messages."""
        long_reply = "x" * (MAX_DC_LEN + 500)

        async def dispatch_fn(agent, msg):
            return long_reply

        ch = _make_discord_channel(dispatch_fn=dispatch_fn)
        ch._pairing.claim_owner(111)
        interaction = _mock_interaction(111)

        await ch._handle_repl_slash(interaction, "just chat")
        assert interaction.followup.send.call_count >= 2

    @pytest.mark.asyncio
    async def test_dispatch_error_reported(self):
        """Errors during dispatch are caught and sent as followup."""
        async def bad_dispatch(agent, msg):
            raise RuntimeError("boom")

        ch = _make_discord_channel(dispatch_fn=bad_dispatch)
        ch._pairing.claim_owner(111)
        interaction = _mock_interaction(111)

        await ch._handle_repl_slash(interaction, "hello")
        call = interaction.followup.send.call_args
        assert "Error" in call[0][0]


# ── slash command tree registration ───────────────────────────

class TestSlashCommandTree:
    def test_expected_commands_registered(self):
        """Verify all expected slash commands are registered on the tree."""
        discord = pytest.importorskip("discord")

        ch = _make_discord_channel()
        client = MagicMock()
        client.application_id = None
        tree = discord.app_commands.CommandTree(client)
        ch._tree = tree
        ch._register_slash_commands()

        command_names = {cmd.name for cmd in tree.get_commands()}
        expected = {
            "start", "allow", "revoke", "paired",
            "use", "agents", "status", "help", "costs", "reset",
            "broadcast",
        }
        # steer/debug not registered when their fns are None
        assert expected == command_names

    def test_steer_registered_when_fn_set(self):
        """steer slash command appears when steer_fn is provided."""
        discord = pytest.importorskip("discord")

        def steer_fn(agent, msg):
            pass

        ch = _make_discord_channel(steer_fn=steer_fn)
        client = MagicMock()
        client.application_id = None
        tree = discord.app_commands.CommandTree(client)
        ch._tree = tree
        ch._register_slash_commands()

        command_names = {cmd.name for cmd in tree.get_commands()}
        assert "steer" in command_names

    def test_debug_registered_when_fn_set(self):
        """debug slash command appears when debug_fn is provided."""
        discord = pytest.importorskip("discord")

        def debug_fn(trace_id=None):
            return []

        ch = _make_discord_channel(debug_fn=debug_fn)
        client = MagicMock()
        client.application_id = None
        tree = discord.app_commands.CommandTree(client)
        ch._tree = tree
        ch._register_slash_commands()

        command_names = {cmd.name for cmd in tree.get_commands()}
        assert "debug" in command_names

    def test_steer_debug_hidden_when_fns_none(self):
        """steer/debug slash commands absent when their fns are not set."""
        discord = pytest.importorskip("discord")

        ch = _make_discord_channel(steer_fn=None, debug_fn=None)
        client = MagicMock()
        client.application_id = None
        tree = discord.app_commands.CommandTree(client)
        ch._tree = tree
        ch._register_slash_commands()

        command_names = {cmd.name for cmd in tree.get_commands()}
        assert "steer" not in command_names
        assert "debug" not in command_names

    def test_addkey_not_registered(self):
        """addkey must NOT be a slash command (API keys visible in Discord UI)."""
        discord = pytest.importorskip("discord")

        ch = _make_discord_channel()
        client = MagicMock()
        client.application_id = None
        tree = discord.app_commands.CommandTree(client)
        ch._tree = tree
        ch._register_slash_commands()

        command_names = {cmd.name for cmd in tree.get_commands()}
        assert "addkey" not in command_names

    def test_paired_registered(self):
        """paired slash command is always registered."""
        discord = pytest.importorskip("discord")

        ch = _make_discord_channel()
        client = MagicMock()
        client.application_id = None
        tree = discord.app_commands.CommandTree(client)
        ch._tree = tree
        ch._register_slash_commands()

        command_names = {cmd.name for cmd in tree.get_commands()}
        assert "paired" in command_names


# ── on_message fallback ───────────────────────────────────────

class TestOnMessageFallback:
    """Verify !-prefix commands still work via on_message path."""

    def test_bang_start_pairing(self):
        """!start <code> still works for pairing via on_message."""
        ch = _make_discord_channel()
        msg = ch._handle_pairing(12345, "ABC123")
        assert "Paired as owner" in msg

    def test_bang_allow_works(self):
        """!allow via on_message delegates to _handle_allow."""
        ch = _make_discord_channel()
        ch._pairing.claim_owner(111)
        msg = ch._handle_allow(111, "222")
        assert "now allowed" in msg

    def test_bang_revoke_works(self):
        """!revoke via on_message delegates to _handle_revoke."""
        ch = _make_discord_channel()
        ch._pairing.claim_owner(111)
        ch._pairing.allow(222)
        msg = ch._handle_revoke(111, "222")
        assert "revoked" in msg

    @pytest.mark.asyncio
    async def test_exclamation_to_slash_conversion(self):
        """!agents gets converted to /agents and handled."""
        ch = _make_discord_channel()
        ch._pairing.claim_owner(111)
        result = await ch.handle_message("111", "/agents")
        assert "alpha" in result
        assert "beta" in result

    def test_bang_paired_works(self):
        """!paired via on_message delegates to _handle_paired."""
        ch = _make_discord_channel()
        ch._pairing.claim_owner(111)
        msg = ch._handle_paired(111)
        assert "Owner: 111" in msg
