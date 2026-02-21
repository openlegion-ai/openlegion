"""Telegram channel adapter.

Bridges Telegram to the OpenLegion mesh with the same UX as the CLI REPL:
  - Per-user active agent tracking
  - @agent mentions for routing to specific agents
  - /use, /agents, /status, /broadcast, /costs, /reset, /help commands
  - Agent name labels on all responses: [agent_name] response
  - Push notifications for cron/heartbeat results
  - Pairing: owner must send /start <pairing_code> to claim the bot.
    Code is generated during `openlegion setup`. Others need /allow.

Requires: pip install python-telegram-bot>=21.0
Config: TELEGRAM_BOT_TOKEN in .env, channels.telegram in mesh.yaml
"""

from __future__ import annotations

import asyncio
import time

from src.channels.base import Channel, PairingManager, chunk_text
from src.shared.utils import setup_logging

logger = setup_logging("channels.telegram")

MAX_TG_LEN = 4000


def _md_to_html(text: str) -> str:
    """Best-effort conversion of common Markdown to Telegram-safe HTML.

    Handles: bold, italic, inline code, code blocks, headers.
    Anything that fails parsing is sent as plain text by the caller.
    """
    import re as _re

    # Fenced code blocks: ```lang\n...\n``` → <pre>...</pre>
    text = _re.sub(
        r"```(?:\w+)?\n(.*?)```",
        lambda m: f"<pre>{m.group(1).rstrip()}</pre>",
        text,
        flags=_re.DOTALL,
    )
    # Inline code: `...` → <code>...</code>
    text = _re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    # Bold: **...** → <b>...</b>
    text = _re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    # Italic: *...* → <i>...</i>  (but not inside <b> tags)
    text = _re.sub(r"(?<!</b)\*(.+?)\*", r"<i>\1</i>", text)
    # Headers: # ... → <b>...</b> (Telegram has no header tag)
    text = _re.sub(r"^#{1,6}\s+(.+)$", r"<b>\1</b>", text, flags=_re.MULTILINE)
    return text


class TelegramChannel(Channel):
    """Telegram bot adapter for OpenLegion with pairing code security."""

    def __init__(
        self,
        token: str,
        default_agent: str = "",
        allowed_users: list[int] | None = None,
        **kwargs,
    ):
        super().__init__(default_agent=default_agent, **kwargs)
        self.token = token
        self._explicit_allowed = set(allowed_users) if allowed_users else None
        self._app = None
        self._chat_ids: set[int] = set()
        self._pairing = PairingManager("config/telegram_paired.json")

    async def start(self) -> None:
        try:
            from telegram.ext import (
                Application,
                CommandHandler,
                MessageHandler,
                filters,
            )
        except ImportError:
            logger.error(
                "python-telegram-bot not installed. "
                "Install with: pip install 'openlegion[channels]'"
            )
            return

        self._app = Application.builder().token(self.token).build()
        self._app.add_handler(CommandHandler("start", self._cmd_start))
        self._app.add_handler(CommandHandler("allow", self._cmd_allow))
        self._app.add_handler(CommandHandler("revoke", self._cmd_revoke))
        self._app.add_handler(CommandHandler("paired", self._cmd_paired))
        # Route OpenLegion REPL commands (/status, /agents, /costs, etc.)
        # through the base Channel.handle_message() handler.
        _repl_cmds = ("use", "agents", "status", "broadcast", "costs", "reset", "addkey", "help")
        for cmd in _repl_cmds:
            self._app.add_handler(CommandHandler(cmd, self._on_repl_command))
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_message)
        )

        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        owner = self._pairing.owner
        if owner:
            logger.info(f"Telegram channel started (owner: {owner})")
        elif self._pairing.pairing_code:
            logger.info("Telegram channel started (awaiting pairing code)")
        else:
            logger.info("Telegram channel started (no pairing code — run setup again)")

    async def stop(self) -> None:
        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
            logger.info("Telegram channel stopped")

    async def send_notification(self, text: str) -> None:
        """Push a cron/heartbeat notification to all known chat IDs."""
        if not self._app or not self._chat_ids:
            return
        for chat_id in self._chat_ids:
            try:
                for chunk in chunk_text(text, MAX_TG_LEN):
                    await self._app.bot.send_message(chat_id=chat_id, text=chunk)
            except Exception as e:
                logger.warning(f"Failed to notify chat {chat_id}: {e}")

    def _is_allowed(self, user_id: int) -> bool:
        if self._explicit_allowed is not None:
            return user_id in self._explicit_allowed
        return self._pairing.is_allowed(user_id)

    def _is_owner(self, user_id: int) -> bool:
        return self._pairing.is_owner(user_id)

    async def _cmd_start(self, update, context) -> None:
        user_id = update.effective_user.id
        username = update.effective_user.username or update.effective_user.first_name

        if self._pairing.owner is None:
            code_arg = context.args[0] if context.args else ""
            expected = self._pairing.pairing_code
            if not expected or code_arg != expected:
                await update.message.reply_text(
                    "Pairing required. Send:  /start <pairing_code>\n"
                    "The code was shown during `openlegion setup`."
                )
                logger.warning(
                    f"Rejected /start without valid pairing code from {username} (id: {user_id})"
                )
                return
            self._pairing.claim_owner(user_id)
            logger.info(f"Paired owner via code: {username} (id: {user_id})")
            self._chat_ids.add(update.effective_chat.id)
            str_id = str(user_id)
            response = await self.handle_message(str_id, "/help")
            welcome = (
                f"Paired as owner. Your Telegram ID: {user_id}\n"
                f"Active agent: {self._get_active_agent(str_id) or '(none)'}\n\n"
                f"Only you can use this bot. Use /allow <user_id> to grant access.\n\n"
            )
            await update.message.reply_text(welcome + response)
            return

        if not self._is_allowed(user_id):
            await update.message.reply_text(
                "Access denied. This bot is paired to its owner.\n"
                f"Your Telegram ID: {user_id}\n"
                "Ask the owner to run /allow in the bot to grant you access."
            )
            logger.warning(f"Rejected /start from unpaired user: {username} (id: {user_id})")
            return

        self._chat_ids.add(update.effective_chat.id)
        str_id = str(user_id)
        response = await self.handle_message(str_id, "/help")
        welcome = (
            f"OpenLegion connected. Active agent: "
            f"{self._get_active_agent(str_id) or '(none)'}\n\n"
        )
        await update.message.reply_text(welcome + response)

    async def _cmd_allow(self, update, context) -> None:
        """Owner-only: /allow <telegram_user_id> to grant access."""
        if not self._is_owner(update.effective_user.id):
            await update.message.reply_text("Only the owner can use /allow.")
            return
        args = context.args
        if not args:
            await update.message.reply_text("Usage: /allow <telegram_user_id>")
            return
        try:
            target_id = int(args[0])
        except ValueError:
            await update.message.reply_text("Invalid user ID. Must be a number.")
            return
        self._pairing.allow(target_id)
        await update.message.reply_text(f"User {target_id} is now allowed.")
        logger.info(f"Owner allowed user {target_id}")

    async def _cmd_revoke(self, update, context) -> None:
        """Owner-only: /revoke <telegram_user_id> to remove access."""
        if not self._is_owner(update.effective_user.id):
            await update.message.reply_text("Only the owner can use /revoke.")
            return
        args = context.args
        if not args:
            await update.message.reply_text("Usage: /revoke <telegram_user_id>")
            return
        try:
            target_id = int(args[0])
        except ValueError:
            await update.message.reply_text("Invalid user ID. Must be a number.")
            return
        if self._pairing.revoke(target_id):
            await update.message.reply_text(f"User {target_id} access revoked.")
        else:
            await update.message.reply_text(f"User {target_id} was not in the allowed list.")

    async def _cmd_paired(self, update, context) -> None:
        """Owner-only: show pairing info."""
        if not self._is_owner(update.effective_user.id):
            await update.message.reply_text("Only the owner can view pairing info.")
            return
        owner = self._pairing.owner
        allowed = self._pairing.allowed_list()
        lines = [f"Owner: {owner}"]
        if allowed:
            lines.append(f"Allowed users: {', '.join(str(u) for u in allowed)}")
        else:
            lines.append("No additional users allowed.")
        await update.message.reply_text("\n".join(lines))

    async def _on_repl_command(self, update, context) -> None:
        """Handle OpenLegion REPL commands (/status, /agents, /costs, etc.)."""
        if not self._is_allowed(update.effective_user.id):
            return
        self._chat_ids.add(update.effective_chat.id)
        user_id = str(update.effective_user.id)
        # Reconstruct the /command [args] text from Telegram's parsed command
        cmd = update.message.text or ""
        if not cmd.strip():
            return
        chat_id = update.effective_chat.id
        try:
            response = await self.handle_message(user_id, cmd)
        except Exception as e:
            logger.error(f"REPL command failed for user {user_id}: {e}")
            response = f"Error: {e}"
        if response:
            await self._send_reply(chat_id, response)

    async def _on_message(self, update, context) -> None:
        if not self._is_allowed(update.effective_user.id):
            return
        text = update.message.text or ""
        if not text.strip():
            return
        self._chat_ids.add(update.effective_chat.id)
        user_id = str(update.effective_user.id)
        chat_id = update.effective_chat.id
        asyncio.create_task(self._dispatch_and_reply(chat_id, user_id, text))

    async def _dispatch_and_reply(
        self, chat_id: int, user_id: str, text: str,
    ) -> None:
        """Process a message in the background with tool progress updates."""
        current = self._get_active_agent(user_id)
        target = current

        # Check for @agent mention
        import re as _re
        match = _re.match(r"^@(\w+)\s+(.+)$", text, _re.DOTALL)
        if match:
            agents = self._get_agent_names()
            if match.group(1) in agents:
                target = match.group(1)

        # Use streaming dispatch if available for tool progress
        if self.stream_dispatch_fn and target:
            await self._stream_dispatch_and_reply(chat_id, user_id, target, text)
            return

        # Fallback: non-streaming with typing indicator
        typing_task = asyncio.create_task(self._typing_loop(chat_id))
        try:
            response = await self.handle_message(user_id, text)
        except Exception as e:
            logger.error(f"Dispatch failed for user {user_id}: {e}")
            response = f"Error: {e}"
        finally:
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass
        if response:
            await self._send_reply(chat_id, response)

    async def _stream_dispatch_and_reply(
        self, chat_id: int, user_id: str, target: str, text: str,
    ) -> None:
        """Dispatch with streaming — progressive text updates + tool progress."""
        progress_msg = None
        tool_lines: list[str] = []
        response_text = ""
        tool_count = 0
        streaming_msg = None  # Message for progressive text streaming
        last_edit_time = 0.0
        _EDIT_INTERVAL = 0.5  # Debounce edits to avoid Telegram rate limits

        try:
            async for event in self.stream_dispatch_fn(target, text):
                if not isinstance(event, dict):
                    continue
                etype = event.get("type", "")

                if etype == "tool_start":
                    tool_count += 1
                    name = event.get("name", "?")
                    tool_lines.append(f"{tool_count}. {name}")
                    progress = f"Working...\n" + "\n".join(tool_lines)
                    # Delete streaming text msg when tools start (any round)
                    if streaming_msg:
                        try:
                            await self._app.bot.delete_message(
                                chat_id=chat_id, message_id=streaming_msg.message_id,
                            )
                        except Exception:
                            pass
                        streaming_msg = None
                    try:
                        if progress_msg is None:
                            progress_msg = await self._app.bot.send_message(
                                chat_id=chat_id, text=progress,
                            )
                        else:
                            await self._app.bot.edit_message_text(
                                chat_id=chat_id,
                                message_id=progress_msg.message_id,
                                text=progress,
                            )
                    except Exception:
                        pass  # edit may fail if text unchanged

                elif etype == "tool_result":
                    name = event.get("name", "?")
                    output = event.get("output", {})
                    if tool_lines:
                        hint = ""
                        if isinstance(output, dict) and output.get("error"):
                            hint = " ✗"
                        else:
                            hint = " ✓"
                        tool_lines[-1] += hint
                        progress = f"Working...\n" + "\n".join(tool_lines)
                        try:
                            if progress_msg:
                                await self._app.bot.edit_message_text(
                                    chat_id=chat_id,
                                    message_id=progress_msg.message_id,
                                    text=progress,
                                )
                        except Exception:
                            pass

                elif etype == "text_delta":
                    response_text += event.get("content", "")
                    # Progressive text update (debounced)
                    now = time.monotonic()
                    if now - last_edit_time >= _EDIT_INTERVAL and response_text.strip():
                        last_edit_time = now
                        display = f"[{target}] {response_text}..."
                        try:
                            if streaming_msg is None:
                                streaming_msg = await self._app.bot.send_message(
                                    chat_id=chat_id, text=display[:4096],
                                )
                            else:
                                await self._app.bot.edit_message_text(
                                    chat_id=chat_id,
                                    message_id=streaming_msg.message_id,
                                    text=display[:4096],
                                )
                        except Exception:
                            pass

                elif etype == "done":
                    response_text = event.get("response", response_text)

        except Exception as e:
            logger.error(f"Stream dispatch failed for user {user_id}: {e}")
            response_text = f"Error: {e}"

        # Delete progress message
        if progress_msg:
            try:
                await self._app.bot.delete_message(
                    chat_id=chat_id, message_id=progress_msg.message_id,
                )
            except Exception:
                pass

        # Edit streaming message with final text, or send new if none exists
        if response_text:
            final_text = f"[{target}] {response_text}"
            if streaming_msg:
                try:
                    await self._app.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=streaming_msg.message_id,
                        text=final_text[:4096],
                    )
                    # Send overflow as separate messages
                    if len(final_text) > 4096:
                        for chunk in chunk_text(final_text[4096:], MAX_TG_LEN):
                            await self._app.bot.send_message(
                                chat_id=chat_id, text=chunk,
                            )
                except Exception:
                    await self._send_reply(chat_id, final_text)
            else:
                await self._send_reply(chat_id, final_text)

    async def _send_reply(self, chat_id: int, text: str) -> None:
        """Send a response, trying HTML formatting first, then plain text."""
        for chunk in chunk_text(text, MAX_TG_LEN):
            try:
                await self._app.bot.send_message(
                    chat_id=chat_id, text=_md_to_html(chunk), parse_mode="HTML",
                )
            except Exception:
                # HTML parse failed — send as plain text
                try:
                    await self._app.bot.send_message(chat_id=chat_id, text=chunk)
                except Exception as e:
                    logger.warning(f"Failed to send response to {chat_id}: {e}")

    async def _typing_loop(self, chat_id: int) -> None:
        """Send typing indicator every 4s until cancelled."""
        try:
            from telegram.constants import ChatAction
        except ImportError:
            return
        while True:
            try:
                await self._app.bot.send_chat_action(
                    chat_id=chat_id, action=ChatAction.TYPING,
                )
            except Exception:
                pass
            await asyncio.sleep(4)
