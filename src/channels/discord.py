"""Discord channel adapter.

Bridges Discord to the OpenLegion mesh with the same UX as the CLI REPL:
  - Per-user active agent tracking
  - @agent mentions for routing to specific agents
  - /use, /agents, /status, /broadcast, /costs, /reset, /help commands
  - Agent name labels on all responses: [agent_name] response
  - Push notifications for cron/heartbeat results
  - Pairing: owner must send /start <pairing_code> to claim the bot.
    Code is generated during `openlegion start`. Others need /allow.
  - Discord slash commands registered via CommandTree (fixes #123)

Requires: pip install discord.py>=2.0
Config: DISCORD_BOT_TOKEN in .env, channels.discord in mesh.yaml
"""

from __future__ import annotations

import asyncio
import re
import time

from src.channels.base import Channel, PairingManager, chunk_text
from src.shared.utils import setup_logging

logger = setup_logging("channels.discord")

MAX_DC_LEN = 1900

_AT_MENTION_RE = re.compile(r"^@(\w+)\s+(.+)$", re.DOTALL)


class DiscordChannel(Channel):
    """Discord bot adapter for OpenLegion with pairing code security."""

    def __init__(
        self,
        token: str,
        default_agent: str = "",
        allowed_guilds: list[int] | None = None,
        **kwargs,
    ):
        super().__init__(default_agent=default_agent, **kwargs)
        self.token = token
        self.allowed_guilds = set(allowed_guilds) if allowed_guilds else None
        self._client = None
        self._tree = None
        self._notify_channel_ids: set[int] = set()
        self._denied_notified: set[int] = set()
        self._pairing = PairingManager("config/discord_paired.json")
        self._synced = False

    def _is_allowed(self, user_id: int) -> bool:
        return self._pairing.is_allowed(user_id)

    def _is_owner(self, user_id: int) -> bool:
        return self._pairing.is_owner(user_id)

    # ── extracted command handlers ─────────────────────────────

    def _handle_pairing(self, author_id: int, code_arg: str) -> str:
        """Validate pairing code and claim owner. Returns response message."""
        expected = self._pairing.pairing_code
        if not expected or code_arg != expected:
            logger.warning(
                f"Rejected /start without valid pairing code (id: {author_id})"
            )
            return (
                "Pairing required. Send:  `/start <pairing_code>`\n"
                "The code was shown during `openlegion start`."
            )
        self._pairing.claim_owner(author_id)
        logger.info(f"Paired owner via code (id: {author_id})")
        return (
            f"Paired as owner. Your Discord ID: {author_id}\n"
            f"Only you can use this bot. Use `/allow <user_id>` to grant access."
        )

    def _handle_access_denied(self, author_id: int, text: str) -> str | None:
        """Return access-denied message, or None if already notified.

        Used by on_message where dedup prevents channel spam.
        Slash commands should use _access_denied_message() instead — they
        always need a response and ephemeral means no spam concern.
        """
        if text.lower().startswith("!start") or text.lower().startswith("/start"):
            return (
                f"Access denied. This bot is paired to its owner.\n"
                f"Your Discord ID: {author_id}\n"
                f"Ask the owner to run `/allow {author_id}` to grant you access."
            )
        if author_id not in self._denied_notified:
            self._denied_notified.add(author_id)
            return (
                f"Access denied. Ask the bot owner to grant you access.\n"
                f"Your Discord ID: {author_id}"
            )
        return None

    def _access_denied_message(self, author_id: int) -> str:
        """Always-present denial message for slash commands (no dedup)."""
        return (
            f"Access denied. Ask the bot owner to grant you access.\n"
            f"Your Discord ID: {author_id}"
        )

    def _handle_allow(self, author_id: int, target_id_str: str) -> str:
        """Owner-only: allow a user. Returns response message."""
        if not self._is_owner(author_id):
            return "Only the owner can use /allow."
        try:
            target_id = int(target_id_str)
        except (ValueError, TypeError):
            return "Usage: /allow <discord_user_id>"
        self._pairing.allow(target_id)
        return f"User {target_id} is now allowed."

    def _handle_revoke(self, author_id: int, target_id_str: str) -> str:
        """Owner-only: revoke a user. Returns response message."""
        if not self._is_owner(author_id):
            return "Only the owner can use /revoke."
        try:
            target_id = int(target_id_str)
        except (ValueError, TypeError):
            return "Usage: /revoke <discord_user_id>"
        if self._pairing.revoke(target_id):
            return f"User {target_id} access revoked."
        return f"User {target_id} was not in the allowed list."

    def _handle_paired(self, author_id: int) -> str:
        """Owner-only: show pairing info."""
        if not self._is_owner(author_id):
            return "Only the owner can view pairing info."
        owner = self._pairing.owner
        allowed = self._pairing.allowed_list()
        lines = [f"Owner: {owner}"]
        if allowed:
            lines.append(f"Allowed users: {', '.join(str(u) for u in allowed)}")
        else:
            lines.append("No additional users allowed.")
        return "\n".join(lines)

    async def _handle_repl_slash(self, interaction, command_text: str) -> None:
        """Generic handler for REPL-style slash commands.

        Defers the interaction immediately (3-second Discord deadline),
        checks access, calls handle_message, and sends chunked followup.
        """
        await interaction.response.defer()
        author_id = interaction.user.id

        if self._pairing.owner is None:
            await interaction.followup.send(
                "This bot requires pairing. Send `/start <pairing_code>` to begin.",
                ephemeral=True,
            )
            return

        if not self._is_allowed(author_id):
            # Always respond to slash commands — ephemeral means no spam concern,
            # and Discord requires a response to deferred interactions.
            await interaction.followup.send(
                self._access_denied_message(author_id), ephemeral=True,
            )
            return

        # Track channel for notifications
        if interaction.channel:
            self._notify_channel_ids.add(interaction.channel_id)

        user_id = str(author_id)
        try:
            response = await self.handle_message(user_id, command_text)
        except Exception as e:
            logger.error(f"Slash dispatch failed for user {user_id}: {e}")
            response = f"Error: {e}"

        if response:
            for chunk in chunk_text(response, MAX_DC_LEN):
                await interaction.followup.send(chunk)

    # ── slash command registration ─────────────────────────────

    def _register_slash_commands(self) -> None:
        """Register Discord slash commands on self._tree."""
        import discord

        tree = self._tree
        channel_ref = self

        @tree.command(name="start", description="Pair this bot with your account")
        @discord.app_commands.describe(code="The pairing code shown during openlegion start")
        async def slash_start(interaction: discord.Interaction, code: str):
            author_id = interaction.user.id
            if channel_ref._pairing.owner is None:
                msg = channel_ref._handle_pairing(author_id, code)
                await interaction.response.send_message(msg, ephemeral=True)
                # Send help + track channel after successful pairing
                if channel_ref._is_owner(author_id):
                    if interaction.channel:
                        channel_ref._notify_channel_ids.add(interaction.channel_id)
                    try:
                        help_text = await channel_ref.handle_message(
                            str(author_id), "/help"
                        )
                        if help_text:
                            await interaction.followup.send(
                                help_text[:MAX_DC_LEN], ephemeral=True
                            )
                    except Exception:
                        pass
            else:
                if channel_ref._is_allowed(author_id):
                    if interaction.channel:
                        channel_ref._notify_channel_ids.add(interaction.channel_id)
                    await interaction.response.send_message(
                        "Already paired and you have access.", ephemeral=True
                    )
                else:
                    await interaction.response.send_message(
                        f"Access denied. This bot is paired to its owner.\n"
                        f"Your Discord ID: {author_id}\n"
                        f"Ask the owner to run `/allow {author_id}` to grant you access.",
                        ephemeral=True,
                    )

        @tree.command(name="allow", description="Grant a user access to this bot (owner only)")
        @discord.app_commands.describe(user_id="Discord user ID to allow")
        async def slash_allow(interaction: discord.Interaction, user_id: str):
            msg = channel_ref._handle_allow(interaction.user.id, user_id)
            await interaction.response.send_message(msg, ephemeral=True)

        @tree.command(name="revoke", description="Revoke a user's access (owner only)")
        @discord.app_commands.describe(user_id="Discord user ID to revoke")
        async def slash_revoke(interaction: discord.Interaction, user_id: str):
            msg = channel_ref._handle_revoke(interaction.user.id, user_id)
            await interaction.response.send_message(msg, ephemeral=True)

        @tree.command(name="paired", description="Show pairing info (owner only)")
        async def slash_paired(interaction: discord.Interaction):
            msg = channel_ref._handle_paired(interaction.user.id)
            await interaction.response.send_message(msg, ephemeral=True)

        # REPL-style commands — delegate to handle_message via _handle_repl_slash
        @tree.command(name="use", description="Switch active agent")
        @discord.app_commands.describe(agent="Agent name to switch to")
        async def slash_use(interaction: discord.Interaction, agent: str):
            await channel_ref._handle_repl_slash(interaction, f"/use {agent}")

        @tree.command(name="agents", description="List all agents")
        async def slash_agents(interaction: discord.Interaction):
            await channel_ref._handle_repl_slash(interaction, "/agents")

        @tree.command(name="status", description="Show agent health")
        async def slash_status(interaction: discord.Interaction):
            await channel_ref._handle_repl_slash(interaction, "/status")

        @tree.command(name="help", description="Show available commands")
        async def slash_help(interaction: discord.Interaction):
            await channel_ref._handle_repl_slash(interaction, "/help")

        @tree.command(name="costs", description="Show today's LLM spend")
        async def slash_costs(interaction: discord.Interaction):
            await channel_ref._handle_repl_slash(interaction, "/costs")

        @tree.command(name="reset", description="Clear conversation with active agent")
        async def slash_reset(interaction: discord.Interaction):
            await channel_ref._handle_repl_slash(interaction, "/reset")

        @tree.command(name="broadcast", description="Send a message to all agents")
        @discord.app_commands.describe(message="Message to broadcast")
        async def slash_broadcast(interaction: discord.Interaction, message: str):
            await channel_ref._handle_repl_slash(interaction, f"/broadcast {message}")

        # Conditionally register /steer and /debug — only show in command list
        # when the features are actually available.
        if channel_ref.steer_fn:
            @tree.command(name="steer", description="Inject a message into the active agent's context")
            @discord.app_commands.describe(message="Steering message")
            async def slash_steer(interaction: discord.Interaction, message: str):
                await channel_ref._handle_repl_slash(interaction, f"/steer {message}")

        if channel_ref.debug_fn:
            @tree.command(name="debug", description="Show recent traces or trace detail")
            @discord.app_commands.describe(trace_id="Optional trace ID to inspect")
            async def slash_debug(
                interaction: discord.Interaction, trace_id: str = "",
            ):
                cmd = f"/debug {trace_id}".strip()
                await channel_ref._handle_repl_slash(interaction, cmd)

    # ── lifecycle ──────────────────────────────────────────────

    async def start(self) -> None:
        try:
            import discord
        except ImportError:
            logger.error(
                "discord.py not installed. "
                "Install with: pip install 'openlegion[channels]'"
            )
            return

        intents = discord.Intents.default()
        intents.message_content = True
        self._client = discord.Client(intents=intents)
        self._tree = discord.app_commands.CommandTree(self._client)
        self._register_slash_commands()
        channel_ref = self

        @self._client.event
        async def on_ready():
            if not channel_ref._synced:
                channel_ref._synced = True
                try:
                    if channel_ref.allowed_guilds:
                        for gid in channel_ref.allowed_guilds:
                            guild = discord.Object(id=gid)
                            channel_ref._tree.copy_global_to(guild=guild)
                            await channel_ref._tree.sync(guild=guild)
                        logger.info(
                            f"Synced slash commands to {len(channel_ref.allowed_guilds)} guild(s)"
                        )
                    else:
                        await channel_ref._tree.sync()
                        logger.info("Synced slash commands globally")
                except Exception as e:
                    logger.error(f"Failed to sync slash commands: {e}")
            logger.info(f"Discord channel connected as {channel_ref._client.user}")

        @self._client.event
        async def on_message(message):
            if message.author == channel_ref._client.user:
                return
            if channel_ref.allowed_guilds and message.guild:
                if message.guild.id not in channel_ref.allowed_guilds:
                    return

            text = message.content.strip()
            if not text:
                return

            author_id = message.author.id

            # Pairing via code: /start <code> or !start <code>
            if channel_ref._pairing.owner is None:
                if text.lower().startswith("!start") or text.lower().startswith(
                    "/start"
                ):
                    parts = text.split(None, 1)
                    code_arg = parts[1].strip() if len(parts) > 1 else ""
                    msg = channel_ref._handle_pairing(author_id, code_arg)
                    await message.channel.send(msg)
                    # Send help + track channel after successful pairing
                    if channel_ref._is_owner(author_id):
                        channel_ref._notify_channel_ids.add(message.channel.id)
                        try:
                            help_text = await channel_ref.handle_message(
                                str(author_id), "/help"
                            )
                            if help_text:
                                await message.channel.send(help_text[:MAX_DC_LEN])
                        except Exception:
                            pass
                else:
                    await message.channel.send(
                        "This bot requires pairing. Send `/start <pairing_code>` to begin."
                    )
                return

            if not channel_ref._is_allowed(author_id):
                msg = channel_ref._handle_access_denied(author_id, text)
                if msg:
                    await message.channel.send(msg)
                return

            # Owner-only commands: /allow, /revoke, /paired
            if text.startswith("!allow ") or text.startswith("/allow "):
                parts = text.split(None, 1)
                target_str = parts[1] if len(parts) > 1 else ""
                msg = channel_ref._handle_allow(author_id, target_str)
                await message.channel.send(msg)
                return

            if text.startswith("!revoke ") or text.startswith("/revoke "):
                parts = text.split(None, 1)
                target_str = parts[1] if len(parts) > 1 else ""
                msg = channel_ref._handle_revoke(author_id, target_str)
                await message.channel.send(msg)
                return

            low = text.lower()
            if low in ("!paired", "/paired"):
                msg = channel_ref._handle_paired(author_id)
                await message.channel.send(msg)
                return

            channel_ref._notify_channel_ids.add(message.channel.id)
            user_id = str(author_id)

            if text.startswith("!"):
                text = "/" + text[1:]

            asyncio.create_task(
                channel_ref._dispatch_and_reply(message.channel, user_id, text)
            )

        await self._client.start(self.token)

    async def _dispatch_and_reply(self, channel, user_id: str, text: str) -> None:
        """Process a message in the background with a typing indicator."""
        current = self._get_active_agent(user_id)
        target = current

        match = _AT_MENTION_RE.match(text)
        if match:
            agents = self._get_agent_names()
            if match.group(1) in agents:
                target = match.group(1)

        # Use streaming dispatch if available for regular messages (not commands)
        # Note: ! commands are already converted to / before reaching here
        first_word = text.lstrip("/").split()[0] if text.strip() else ""
        is_command = text.startswith("/") and first_word in (
            "use", "agents", "status", "broadcast", "costs", "reset", "help",
            "addkey", "steer", "paired", "allow", "revoke", "debug",
        )
        if self.stream_dispatch_fn and target and not is_command:
            await self._stream_dispatch_and_reply(channel, user_id, target, text)
            return

        try:
            async with channel.typing():
                response = await self.handle_message(user_id, text)
        except Exception as e:
            logger.error(f"Dispatch failed for user {user_id}: {e}")
            response = f"Error: {e}"
        if response:
            for chunk in chunk_text(response, MAX_DC_LEN):
                try:
                    await channel.send(chunk)
                except Exception as e:
                    logger.warning(f"Failed to send response: {e}")

    async def _stream_dispatch_and_reply(
        self, channel, user_id: str, target: str, text: str,
    ) -> None:
        """Dispatch with streaming — progressive text updates + tool progress."""
        streaming_msg = None
        response_text = ""
        tool_lines: list[str] = []
        tool_count = 0
        last_edit_time = 0.0
        _EDIT_INTERVAL = 0.5

        try:
            async with channel.typing():
                async for event in self.stream_dispatch_fn(target, text):
                    if not isinstance(event, dict):
                        continue
                    etype = event.get("type", "")

                    if etype == "tool_start":
                        tool_count += 1
                        name = event.get("name", "?")
                        tool_lines.append(f"{tool_count}. {name}")

                    elif etype == "tool_result":
                        output = event.get("output", {})
                        if tool_lines:
                            hint = " ✗" if isinstance(output, dict) and output.get("error") else " ✓"
                            tool_lines[-1] += hint

                    elif etype == "text_delta":
                        response_text += event.get("content", "")
                        now = time.monotonic()
                        if now - last_edit_time >= _EDIT_INTERVAL and response_text.strip():
                            last_edit_time = now
                            display = f"[{target}] {response_text}..."
                            try:
                                if streaming_msg is None:
                                    streaming_msg = await channel.send(display[:MAX_DC_LEN])
                                else:
                                    await streaming_msg.edit(content=display[:MAX_DC_LEN])
                            except Exception:
                                pass

                    elif etype == "done":
                        response_text = event.get("response", response_text)

        except Exception as e:
            logger.error(f"Stream dispatch failed for user {user_id}: {e}")
            response_text = f"Error: {e}"

        if response_text:
            if tool_lines:
                names = [line.split(". ", 1)[1].split(" ")[0] if ". " in line else line for line in tool_lines]
                final_text = f"[{target}] Tools: {', '.join(names)}\n\n{response_text}"
            else:
                final_text = f"[{target}] {response_text}"
            if streaming_msg:
                try:
                    await streaming_msg.edit(content=final_text[:MAX_DC_LEN])
                    if len(final_text) > MAX_DC_LEN:
                        for chunk in chunk_text(final_text[MAX_DC_LEN:], MAX_DC_LEN):
                            await channel.send(chunk)
                except Exception:
                    for chunk in chunk_text(final_text, MAX_DC_LEN):
                        await channel.send(chunk)
            else:
                for chunk in chunk_text(final_text, MAX_DC_LEN):
                    await channel.send(chunk)

    async def stop(self) -> None:
        if self._client:
            await self._client.close()
            logger.info("Discord channel stopped")

    async def send_notification(self, text: str) -> None:
        """Push a cron/heartbeat notification to all active Discord channels."""
        if not self._client or not self._notify_channel_ids:
            return
        for ch_id in self._notify_channel_ids:
            try:
                channel = self._client.get_channel(ch_id)
                if channel:
                    for c in chunk_text(text, MAX_DC_LEN):
                        await channel.send(c)
            except Exception as e:
                logger.warning(f"Failed to notify channel {ch_id}: {e}")
