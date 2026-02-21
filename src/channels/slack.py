"""Slack channel adapter.

Bridges Slack to the OpenLegion mesh with the same UX as the CLI REPL:
  - Per-thread active agent tracking (composite key: user_id:thread_ts)
  - @agent mentions for routing to specific agents
  - /use, /agents, /status, /broadcast, /costs, /reset, /help commands
  - Agent name labels on all responses: [agent_name] response
  - Push notifications for cron/heartbeat results
  - Pairing: owner must send !start <pairing_code> to claim the bot.
    Code is generated during `openlegion setup`. Others need !allow.

Requires: pip install slack-bolt>=1.18
Config: SLACK_BOT_TOKEN + SLACK_APP_TOKEN in .env, channels.slack in mesh.yaml
Uses Socket Mode (no public URL needed).
"""

from __future__ import annotations

import asyncio
import time

from src.channels.base import Channel, PairingManager, chunk_text
from src.shared.utils import setup_logging

logger = setup_logging("channels.slack")

MAX_SLACK_LEN = 3000


def _get_user_key(user_id: str, thread_ts: str | None) -> str:
    """Composite key for per-thread agent tracking."""
    if thread_ts:
        return f"{user_id}:{thread_ts}"
    return user_id


class SlackChannel(Channel):
    """Slack bot adapter for OpenLegion with Socket Mode and pairing code security."""

    def __init__(
        self,
        bot_token: str,
        app_token: str,
        default_agent: str = "",
        **kwargs,
    ):
        super().__init__(default_agent=default_agent, **kwargs)
        self.bot_token = bot_token
        self.app_token = app_token
        self._bolt_app = None
        self._handler = None
        self._channel_ids: set[str] = set()
        self._pairing = PairingManager("config/slack_paired.json")

    async def start(self) -> None:
        try:
            from slack_bolt.adapter.socket_mode.async_handler import (
                AsyncSocketModeHandler,
            )
            from slack_bolt.async_app import AsyncApp
        except ImportError:
            logger.error(
                "slack-bolt not installed. "
                "Install with: pip install 'openlegion[channels]'"
            )
            return

        self._bolt_app = AsyncApp(token=self.bot_token)
        channel_ref = self

        @self._bolt_app.event("message")
        async def handle_message_event(event, say):
            await channel_ref._on_message(event, say)

        self._handler = AsyncSocketModeHandler(self._bolt_app, self.app_token)
        await self._handler.start_async()

        owner = self._pairing.owner
        if owner:
            logger.info(f"Slack channel started (owner: {owner})")
        elif self._pairing.pairing_code:
            logger.info("Slack channel started (awaiting pairing code)")
        else:
            logger.info("Slack channel started (no pairing code -- run setup again)")

    async def stop(self) -> None:
        if self._handler:
            await self._handler.close_async()
            logger.info("Slack channel stopped")

    async def send_notification(self, text: str) -> None:
        """Push a cron/heartbeat notification to all known channel IDs."""
        if not self._bolt_app or not self._channel_ids:
            return
        for ch_id in self._channel_ids:
            try:
                for part in chunk_text(text, MAX_SLACK_LEN):
                    await self._bolt_app.client.chat_postMessage(
                        channel=ch_id, text=part,
                    )
            except Exception as e:
                logger.warning(f"Failed to notify channel {ch_id}: {e}")

    def _is_allowed(self, user_id: str) -> bool:
        return self._pairing.is_allowed(user_id)

    def _is_owner(self, user_id: str) -> bool:
        return self._pairing.is_owner(user_id)

    async def _on_message(self, event: dict, say) -> None:
        """Handle incoming Slack message events."""
        # Ignore bot messages and message subtypes (edits, joins, etc.)
        if event.get("bot_id") or event.get("subtype"):
            return

        user_id = event.get("user", "")
        text = (event.get("text") or "").strip()
        channel_id = event.get("channel", "")
        thread_ts = event.get("thread_ts")

        if not text or not user_id:
            return

        # Pairing: !start <code>
        if self._pairing.owner is None:
            if text.lower().startswith("!start") or text.lower().startswith("/start"):
                parts = text.split(None, 1)
                code_arg = parts[1].strip() if len(parts) > 1 else ""
                expected = self._pairing.pairing_code
                if not expected or code_arg != expected:
                    await say(
                        text=(
                            "Pairing required. Send:  `!start <pairing_code>`\n"
                            "The code was shown during `openlegion setup`."
                        ),
                        thread_ts=thread_ts,
                    )
                    logger.warning(
                        f"Rejected !start without valid pairing code from {user_id}"
                    )
                    return
                self._pairing.claim_owner(user_id)
                logger.info(f"Paired owner via code: {user_id}")
                self._channel_ids.add(channel_id)
                await say(
                    text=(
                        f"Paired as owner. Your Slack ID: {user_id}\n"
                        f"Only you can use this bot. Use `!allow <user_id>` to grant access."
                    ),
                    thread_ts=thread_ts,
                )
                return
            else:
                await say(
                    text="This bot requires pairing. Send `!start <pairing_code>` to begin.",
                    thread_ts=thread_ts,
                )
                return

        if not self._is_allowed(user_id):
            if text.lower().startswith("!start") or text.lower().startswith("/start"):
                await say(
                    text=(
                        f"Access denied. This bot is paired to its owner.\n"
                        f"Your Slack ID: {user_id}\n"
                        f"Ask the owner to run `!allow {user_id}` to grant you access."
                    ),
                    thread_ts=thread_ts,
                )
            return

        # Owner-only commands
        if text.startswith("!allow ") or text.startswith("/allow "):
            if not self._is_owner(user_id):
                await say(text="Only the owner can use !allow.", thread_ts=thread_ts)
                return
            parts = text.split(None, 1)
            if len(parts) < 2 or not parts[1].strip():
                await say(text="Usage: !allow <slack_user_id>", thread_ts=thread_ts)
                return
            target_id = parts[1].strip()
            self._pairing.allow(target_id)
            await say(text=f"User {target_id} is now allowed.", thread_ts=thread_ts)
            logger.info(f"Owner allowed user {target_id}")
            return

        if text.startswith("!revoke ") or text.startswith("/revoke "):
            if not self._is_owner(user_id):
                await say(text="Only the owner can use !revoke.", thread_ts=thread_ts)
                return
            parts = text.split(None, 1)
            if len(parts) < 2 or not parts[1].strip():
                await say(text="Usage: !revoke <slack_user_id>", thread_ts=thread_ts)
                return
            target_id = parts[1].strip()
            if self._pairing.revoke(target_id):
                await say(text=f"User {target_id} access revoked.", thread_ts=thread_ts)
            else:
                await say(
                    text=f"User {target_id} was not in the allowed list.",
                    thread_ts=thread_ts,
                )
            return

        if text.lower() in ("!paired", "/paired"):
            if not self._is_owner(user_id):
                await say(
                    text="Only the owner can view pairing info.",
                    thread_ts=thread_ts,
                )
                return
            owner = self._pairing.owner
            allowed = self._pairing.allowed_list()
            lines = [f"Owner: {owner}"]
            if allowed:
                lines.append(f"Allowed users: {', '.join(str(u) for u in allowed)}")
            else:
                lines.append("No additional users allowed.")
            await say(text="\n".join(lines), thread_ts=thread_ts)
            return

        self._channel_ids.add(channel_id)

        # Translate ! commands to / for base class handling
        if text.startswith("!"):
            text = "/" + text[1:]

        user_key = _get_user_key(user_id, thread_ts)

        asyncio.create_task(
            self._dispatch_and_reply(say, user_key, text, thread_ts)
        )

    async def _dispatch_and_reply(
        self, say, user_key: str, text: str, thread_ts: str | None,
    ) -> None:
        """Process a message in the background."""
        # Resolve target agent for streaming
        current = self._get_active_agent(user_key)
        target = current

        import re as _re
        match = _re.match(r"^@(\w+)\s+(.+)$", text, _re.DOTALL)
        if match:
            agents = self._get_agent_names()
            if match.group(1) in agents:
                target = match.group(1)

        # Use streaming for regular messages (not commands)
        first_word = text.lstrip("/").split()[0] if text.strip() else ""
        is_command = text.startswith("/") and first_word in (
            "use", "agents", "status", "broadcast", "costs", "reset", "help",
            "addkey", "steer", "paired", "allow", "revoke", "debug",
        )
        if self.stream_dispatch_fn and target and not is_command:
            await self._stream_dispatch_and_reply(say, user_key, target, text, thread_ts)
            return

        try:
            response = await self.handle_message(user_key, text)
        except Exception as e:
            logger.error(f"Dispatch failed for user {user_key}: {e}")
            response = f"Error: {e}"
        if response:
            for part in chunk_text(response, MAX_SLACK_LEN):
                try:
                    await say(text=part, thread_ts=thread_ts)
                except Exception as e:
                    logger.warning(f"Failed to send response: {e}")

    async def _stream_dispatch_and_reply(
        self, say, user_key: str, target: str, text: str, thread_ts: str | None,
    ) -> None:
        """Dispatch with streaming — progressive text updates via chat.update."""
        streaming_ts = None  # Slack message ts for editing
        streaming_channel = None  # Channel where the streaming message lives
        response_text = ""
        tool_lines: list[str] = []
        tool_count = 0
        last_edit_time = 0.0
        _EDIT_INTERVAL = 0.5

        try:
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
                        hint = " :x:" if isinstance(output, dict) and output.get("error") else " :white_check_mark:"
                        tool_lines[-1] += hint

                elif etype == "text_delta":
                    response_text += event.get("content", "")
                    now = time.monotonic()
                    if now - last_edit_time >= _EDIT_INTERVAL and response_text.strip():
                        last_edit_time = now
                        display = f"[{target}] {response_text}..."
                        try:
                            if streaming_ts is None:
                                result = await say(
                                    text=display[:MAX_SLACK_LEN], thread_ts=thread_ts,
                                )
                                # Extract ts and channel from say() response
                                if hasattr(result, "data"):
                                    result = result.data  # SlackResponse → dict
                                if isinstance(result, dict):
                                    streaming_ts = result.get("ts")
                                    streaming_channel = result.get("channel")
                            elif self._bolt_app and streaming_ts and streaming_channel:
                                await self._bolt_app.client.chat_update(
                                    channel=streaming_channel,
                                    ts=streaming_ts,
                                    text=display[:MAX_SLACK_LEN],
                                )
                        except Exception:
                            pass

                elif etype == "done":
                    response_text = event.get("response", response_text)

        except Exception as e:
            logger.error(f"Stream dispatch failed for user {user_key}: {e}")
            response_text = f"Error: {e}"

        if response_text:
            final_text = f"[{target}] {response_text}"
            if streaming_ts and streaming_channel and self._bolt_app:
                try:
                    await self._bolt_app.client.chat_update(
                        channel=streaming_channel,
                        ts=streaming_ts,
                        text=final_text[:MAX_SLACK_LEN],
                    )
                    if len(final_text) > MAX_SLACK_LEN:
                        for part in chunk_text(final_text[MAX_SLACK_LEN:], MAX_SLACK_LEN):
                            await say(text=part, thread_ts=thread_ts)
                except Exception:
                    for part in chunk_text(final_text, MAX_SLACK_LEN):
                        await say(text=part, thread_ts=thread_ts)
            else:
                for part in chunk_text(final_text, MAX_SLACK_LEN):
                    try:
                        await say(text=part, thread_ts=thread_ts)
                    except Exception as e:
                        logger.warning(f"Failed to send response: {e}")
