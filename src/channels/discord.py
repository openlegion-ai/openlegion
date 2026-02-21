"""Discord channel adapter.

Bridges Discord to the OpenLegion mesh with the same UX as the CLI REPL:
  - Per-user active agent tracking
  - @agent mentions for routing to specific agents
  - /use, /agents, /status, /broadcast, /costs, /reset, /help commands
  - Agent name labels on all responses: [agent_name] response
  - Push notifications for cron/heartbeat results
  - Pairing: owner must send !start <pairing_code> to claim the bot.
    Code is generated during `openlegion setup`. Others need !allow.

Requires: pip install discord.py>=2.0
Config: DISCORD_BOT_TOKEN in .env, channels.discord in mesh.yaml
"""

from __future__ import annotations

import asyncio

from src.channels.base import Channel, PairingManager, chunk_text
from src.shared.utils import setup_logging

logger = setup_logging("channels.discord")

MAX_DC_LEN = 1900


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
        self._notify_channel_ids: set[int] = set()
        self._pairing = PairingManager("config/discord_paired.json")

    def _is_allowed(self, user_id: int) -> bool:
        return self._pairing.is_allowed(user_id)

    def _is_owner(self, user_id: int) -> bool:
        return self._pairing.is_owner(user_id)

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
        channel_ref = self

        @self._client.event
        async def on_ready():
            logger.info(f"Discord channel connected as {self._client.user}")

        @self._client.event
        async def on_message(message):
            if message.author == self._client.user:
                return
            if self.allowed_guilds and message.guild:
                if message.guild.id not in self.allowed_guilds:
                    return

            text = message.content.strip()
            if not text:
                return

            author_id = message.author.id

            # Pairing via code: !start <code>
            if channel_ref._pairing.owner is None:
                if text.lower().startswith("!start") or text.lower().startswith("/start"):
                    parts = text.split(None, 1)
                    code_arg = parts[1].strip() if len(parts) > 1 else ""
                    expected = channel_ref._pairing.pairing_code
                    if not expected or code_arg != expected:
                        await message.channel.send(
                            "Pairing required. Send:  `!start <pairing_code>`\n"
                            "The code was shown during `openlegion setup`."
                        )
                        logger.warning(
                            f"Rejected !start without valid pairing code "
                            f"from {message.author} (id: {author_id})"
                        )
                        return
                    channel_ref._pairing.claim_owner(author_id)
                    logger.info(
                        f"Paired owner via code: {message.author} (id: {author_id})"
                    )
                    await message.channel.send(
                        f"Paired as owner. Your Discord ID: {author_id}\n"
                        f"Only you can use this bot. Use `!allow <user_id>` to grant access."
                    )
                else:
                    await message.channel.send(
                        "This bot requires pairing. Send `!start <pairing_code>` to begin."
                    )
                return

            if not channel_ref._is_allowed(author_id):
                if text.lower().startswith("!start") or text.lower().startswith("/start"):
                    await message.channel.send(
                        f"Access denied. This bot is paired to its owner.\n"
                        f"Your Discord ID: {author_id}\n"
                        f"Ask the owner to run `!allow {author_id}` to grant you access."
                    )
                return

            # Owner-only commands
            if text.startswith("!allow ") or text.startswith("/allow "):
                if not channel_ref._is_owner(author_id):
                    await message.channel.send("Only the owner can use !allow.")
                    return
                try:
                    target_id = int(text.split(None, 1)[1])
                except (ValueError, IndexError):
                    await message.channel.send("Usage: !allow <discord_user_id>")
                    return
                channel_ref._pairing.allow(target_id)
                await message.channel.send(f"User {target_id} is now allowed.")
                return

            if text.startswith("!revoke ") or text.startswith("/revoke "):
                if not channel_ref._is_owner(author_id):
                    await message.channel.send("Only the owner can use !revoke.")
                    return
                try:
                    target_id = int(text.split(None, 1)[1])
                except (ValueError, IndexError):
                    await message.channel.send("Usage: !revoke <discord_user_id>")
                    return
                if channel_ref._pairing.revoke(target_id):
                    await message.channel.send(f"User {target_id} access revoked.")
                else:
                    await message.channel.send(f"User {target_id} was not in the allowed list.")
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
