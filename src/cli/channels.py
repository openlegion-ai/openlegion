"""Channel lifecycle management for messaging integrations."""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import threading

from src.cli.config import PROJECT_ROOT, _ensure_pairing_code

logger = logging.getLogger("cli")


class ChannelManager:
    """Manages the lifecycle of messaging channels (Telegram, Discord, etc.)."""

    CHANNEL_TYPES = ("telegram", "discord", "slack", "whatsapp")

    def __init__(self, cfg: dict, dispatch_fn, agent_registry, **callbacks):
        self.cfg = cfg
        self.dispatch_fn = dispatch_fn
        self.agent_registry = agent_registry
        self.callbacks = callbacks  # status_fn, costs_fn, reset_fn, stream_dispatch_fn, addkey_fn
        self.active: list = []
        self.channel_status: list[tuple[str, bool]] = []  # (message, needs_pairing)
        self.pairing_instructions: list[str] = []
        self._channel_map: dict[str, object] = {}  # "telegram" -> Channel instance

    def start_all(self) -> list:
        """Start all configured channels. Returns webhook routers."""
        channels_cfg = self.cfg.get("channels", {})
        all_agents = self.cfg.get("agents", {})
        first_agent = next(iter(all_agents), "")
        webhook_routers: list = []

        def list_agents_fn():
            return dict(self.agent_registry)

        common = {
            "dispatch_fn": self.dispatch_fn,
            "list_agents_fn": list_agents_fn,
            **self.callbacks,
        }

        # Telegram
        tg_cfg = channels_cfg.get("telegram", {})
        tg_token = (
            tg_cfg.get("token")
            or os.environ.get("OPENLEGION_SYSTEM_TELEGRAM_BOT_TOKEN", "")
            or os.environ.get("OPENLEGION_CRED_TELEGRAM_BOT_TOKEN", "")
            or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        )
        if tg_token:
            tg_code = _ensure_pairing_code(PROJECT_ROOT / "config" / "telegram_paired.json")
            from src.channels.telegram import TelegramChannel
            tg = TelegramChannel(
                token=tg_token,
                default_agent=tg_cfg.get("default_agent", first_agent),
                allowed_users=tg_cfg.get("allowed_users"),
                **common,
            )
            self._start_async_channel(tg)
            self._channel_map["telegram"] = tg
            self._record_pairing("Telegram", "send to your bot \u2192  /start", tg_code)

        # Discord
        dc_cfg = channels_cfg.get("discord", {})
        dc_token = (
            dc_cfg.get("token")
            or os.environ.get("OPENLEGION_SYSTEM_DISCORD_BOT_TOKEN", "")
            or os.environ.get("OPENLEGION_CRED_DISCORD_BOT_TOKEN", "")
            or os.environ.get("DISCORD_BOT_TOKEN", "")
        )
        if dc_token:
            dc_code = _ensure_pairing_code(PROJECT_ROOT / "config" / "discord_paired.json")
            from src.channels.discord import DiscordChannel
            dc = DiscordChannel(
                token=dc_token,
                default_agent=dc_cfg.get("default_agent", first_agent),
                allowed_guilds=dc_cfg.get("allowed_guilds"),
                **common,
            )
            self._start_async_channel(dc)
            self._channel_map["discord"] = dc
            self._record_pairing("Discord", "DM your bot \u2192  /start", dc_code)

        # Slack
        sl_cfg = channels_cfg.get("slack", {})
        sl_bot_token = (
            sl_cfg.get("bot_token")
            or os.environ.get("OPENLEGION_SYSTEM_SLACK_BOT_TOKEN", "")
            or os.environ.get("OPENLEGION_CRED_SLACK_BOT_TOKEN", "")
            or os.environ.get("SLACK_BOT_TOKEN", "")
        )
        sl_app_token = (
            sl_cfg.get("app_token")
            or os.environ.get("OPENLEGION_SYSTEM_SLACK_APP_TOKEN", "")
            or os.environ.get("OPENLEGION_CRED_SLACK_APP_TOKEN", "")
            or os.environ.get("SLACK_APP_TOKEN", "")
        )
        if sl_bot_token and sl_app_token:
            sl_code = _ensure_pairing_code(PROJECT_ROOT / "config" / "slack_paired.json")
            from src.channels.slack import SlackChannel
            sl = SlackChannel(
                bot_token=sl_bot_token,
                app_token=sl_app_token,
                default_agent=sl_cfg.get("default_agent", first_agent),
                **common,
            )
            self._start_async_channel(sl)
            self._channel_map["slack"] = sl
            self._record_pairing("Slack", "message your bot \u2192  /start", sl_code)

        # WhatsApp
        wa_cfg = channels_cfg.get("whatsapp", {})
        wa_token = (
            wa_cfg.get("access_token")
            or os.environ.get("OPENLEGION_SYSTEM_WHATSAPP_ACCESS_TOKEN", "")
            or os.environ.get("OPENLEGION_CRED_WHATSAPP_ACCESS_TOKEN", "")
            or os.environ.get("WHATSAPP_ACCESS_TOKEN", "")
        )
        wa_phone_id = (
            wa_cfg.get("phone_number_id")
            or os.environ.get("OPENLEGION_SYSTEM_WHATSAPP_PHONE_NUMBER_ID", "")
            or os.environ.get("OPENLEGION_CRED_WHATSAPP_PHONE_NUMBER_ID", "")
            or os.environ.get("WHATSAPP_PHONE_NUMBER_ID", "")
        )
        if wa_token and wa_phone_id:
            wa_code = _ensure_pairing_code(PROJECT_ROOT / "config" / "whatsapp_paired.json")
            wa_verify = (
                wa_cfg.get("verify_token")
                or os.environ.get("OPENLEGION_SYSTEM_WHATSAPP_VERIFY_TOKEN", "")
                or os.environ.get("OPENLEGION_CRED_WHATSAPP_VERIFY_TOKEN", "")
                or secrets.token_hex(16)
            )
            from src.channels.whatsapp import WhatsAppChannel
            wa = WhatsAppChannel(
                access_token=wa_token,
                phone_number_id=wa_phone_id,
                verify_token=wa_verify,
                default_agent=wa_cfg.get("default_agent", first_agent),
                **common,
            )
            asyncio.run(wa.start())
            webhook_routers.append(wa.create_router())
            self.active.append(wa)
            self._channel_map["whatsapp"] = wa
            self._record_pairing("WhatsApp", "send to your number \u2192  /start", wa_code)

        return webhook_routers

    def stop_all(self) -> None:
        """Stop all active messaging channels."""
        for ch in self.active:
            try:
                loop = getattr(ch, "_channel_loop", None)
                if loop and loop.is_running():
                    try:
                        future = asyncio.run_coroutine_threadsafe(ch.stop(), loop)
                        future.result(timeout=10)
                    finally:
                        loop.call_soon_threadsafe(loop.stop)
                else:
                    asyncio.run(ch.stop())
            except Exception as e:
                logger.debug("Error stopping channel %s: %s", type(ch).__name__, e)

    def _start_async_channel(self, channel) -> None:
        """Start an async channel in a background thread with retry."""
        max_retries = 3
        name = type(channel).__name__

        def _run():
            import time

            for attempt in range(max_retries):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                channel._channel_loop = loop
                try:
                    loop.run_until_complete(channel.start())
                    break
                except RuntimeError as e:
                    if "Event loop stopped" in str(e) and getattr(channel, "_handler", None) is not None:
                        # slack-bolt background tasks stop the loop after a
                        # successful Socket Mode connect, racing with
                        # run_until_complete().  The handler exists so the
                        # connection succeeded — proceed to run_forever().
                        break
                    try:
                        loop.close()
                    except Exception:
                        pass
                    if attempt < max_retries - 1:
                        delay = 5 * (attempt + 1)
                        logger.warning(
                            "%s failed to connect (attempt %d/%d): %s. Retrying in %ds...",
                            name, attempt + 1, max_retries, e, delay,
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            "%s failed to connect after %d attempts: %s. "
                            "Channel will not be available. Check your network connection.",
                            name, max_retries, e,
                        )
                        return
                except Exception as e:
                    try:
                        loop.close()
                    except Exception:
                        pass
                    if attempt < max_retries - 1:
                        delay = 5 * (attempt + 1)
                        logger.warning(
                            "%s failed to connect (attempt %d/%d): %s. Retrying in %ds...",
                            name, attempt + 1, max_retries, e, delay,
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            "%s failed to connect after %d attempts: %s. "
                            "Channel will not be available. Check your network connection.",
                            name, max_retries, e,
                        )
                        return
            loop.run_forever()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        self.active.append(channel)

    def get_channel_status(self) -> list[dict]:
        """Return status for all 4 channel types."""
        result = []
        for ch_type in self.CHANNEL_TYPES:
            ch = self._channel_map.get(ch_type)
            entry: dict = {"type": ch_type, "connected": False, "paired": False, "pairing_code": None}
            if ch is not None:
                entry["connected"] = True
                pairing = getattr(ch, "_pairing", None)
                if pairing is not None:
                    if pairing.owner:
                        entry["paired"] = True
                    else:
                        entry["pairing_code"] = pairing.pairing_code or None
            result.append(entry)
        return result

    def start_channel(self, channel_type: str, tokens: dict) -> list:
        """Start a single channel dynamically. Returns webhook routers (for WhatsApp)."""
        if channel_type not in self.CHANNEL_TYPES:
            raise ValueError(f"Unknown channel type: {channel_type}")
        if channel_type in self._channel_map:
            raise ValueError(f"{channel_type} is already connected")

        all_agents = self.cfg.get("agents", {})
        first_agent = next(iter(all_agents), "")
        webhook_routers: list = []

        def list_agents_fn():
            return dict(self.agent_registry)

        common = {
            "dispatch_fn": self.dispatch_fn,
            "list_agents_fn": list_agents_fn,
            **self.callbacks,
        }

        if channel_type == "telegram":
            token = tokens.get("token", "")
            if not token:
                raise ValueError("token is required for Telegram")
            _ensure_pairing_code(PROJECT_ROOT / "config" / "telegram_paired.json")
            from src.channels.telegram import TelegramChannel
            ch = TelegramChannel(token=token, default_agent=first_agent, **common)
            self._start_async_channel(ch)
            self._channel_map["telegram"] = ch

        elif channel_type == "discord":
            token = tokens.get("token", "")
            if not token:
                raise ValueError("token is required for Discord")
            _ensure_pairing_code(PROJECT_ROOT / "config" / "discord_paired.json")
            from src.channels.discord import DiscordChannel
            ch = DiscordChannel(token=token, default_agent=first_agent, **common)
            self._start_async_channel(ch)
            self._channel_map["discord"] = ch

        elif channel_type == "slack":
            bot_token = tokens.get("bot_token", "")
            app_token = tokens.get("app_token", "")
            if not bot_token or not app_token:
                raise ValueError("bot_token and app_token are required for Slack")
            _ensure_pairing_code(PROJECT_ROOT / "config" / "slack_paired.json")
            from src.channels.slack import SlackChannel
            ch = SlackChannel(bot_token=bot_token, app_token=app_token, default_agent=first_agent, **common)
            self._start_async_channel(ch)
            self._channel_map["slack"] = ch

        elif channel_type == "whatsapp":
            access_token = tokens.get("access_token", "")
            phone_number_id = tokens.get("phone_number_id", "")
            if not access_token or not phone_number_id:
                raise ValueError("access_token and phone_number_id are required for WhatsApp")
            _ensure_pairing_code(PROJECT_ROOT / "config" / "whatsapp_paired.json")
            verify_token = tokens.get("verify_token") or secrets.token_hex(16)
            from src.channels.whatsapp import WhatsAppChannel
            ch = WhatsAppChannel(
                access_token=access_token, phone_number_id=phone_number_id,
                verify_token=verify_token, default_agent=first_agent, **common,
            )
            asyncio.run(ch.start())
            webhook_routers.append(ch.create_router())
            self.active.append(ch)
            self._channel_map["whatsapp"] = ch

        return webhook_routers

    def stop_channel(self, channel_type: str) -> None:
        """Stop a single channel."""
        ch = self._channel_map.get(channel_type)
        if ch is None:
            raise ValueError(f"{channel_type} is not connected")
        try:
            loop = getattr(ch, "_channel_loop", None)
            if loop and loop.is_running():
                try:
                    future = asyncio.run_coroutine_threadsafe(ch.stop(), loop)
                    future.result(timeout=10)
                finally:
                    loop.call_soon_threadsafe(loop.stop)
            else:
                asyncio.run(ch.stop())
        except Exception as e:
            logger.debug("Error stopping channel %s: %s", channel_type, e)
        if ch in self.active:
            self.active.remove(ch)
        del self._channel_map[channel_type]

    def _record_pairing(self, label: str, instruction: str, code: str | None) -> None:
        """Record pairing status for a channel (deferred display)."""
        if code:
            self.pairing_instructions.append(f"  {label}: {instruction} {code}")
        else:
            self.channel_status.append((label, True))
