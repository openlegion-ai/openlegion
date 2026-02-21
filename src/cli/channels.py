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

    def __init__(self, cfg: dict, dispatch_fn, agent_registry, **callbacks):
        self.cfg = cfg
        self.dispatch_fn = dispatch_fn
        self.agent_registry = agent_registry
        self.callbacks = callbacks  # status_fn, costs_fn, reset_fn, stream_dispatch_fn, addkey_fn
        self.active: list = []
        self.channel_status: list[tuple[str, bool]] = []  # (message, needs_pairing)
        self.pairing_instructions: list[str] = []

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
            self._record_pairing("Telegram", "send to your bot \u2192  /start", tg_code)

        # Discord
        dc_cfg = channels_cfg.get("discord", {})
        dc_token = (
            dc_cfg.get("token")
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
            self._record_pairing("Discord", "DM your bot \u2192  !start", dc_code)

        # Slack
        sl_cfg = channels_cfg.get("slack", {})
        sl_bot_token = (
            sl_cfg.get("bot_token")
            or os.environ.get("OPENLEGION_CRED_SLACK_BOT_TOKEN", "")
            or os.environ.get("SLACK_BOT_TOKEN", "")
        )
        sl_app_token = (
            sl_cfg.get("app_token")
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
            self._record_pairing("Slack", "message your bot \u2192  !start", sl_code)

        # WhatsApp
        wa_cfg = channels_cfg.get("whatsapp", {})
        wa_token = (
            wa_cfg.get("access_token")
            or os.environ.get("OPENLEGION_CRED_WHATSAPP_ACCESS_TOKEN", "")
            or os.environ.get("WHATSAPP_ACCESS_TOKEN", "")
        )
        wa_phone_id = (
            wa_cfg.get("phone_number_id")
            or os.environ.get("OPENLEGION_CRED_WHATSAPP_PHONE_NUMBER_ID", "")
            or os.environ.get("WHATSAPP_PHONE_NUMBER_ID", "")
        )
        if wa_token and wa_phone_id:
            wa_code = _ensure_pairing_code(PROJECT_ROOT / "config" / "whatsapp_paired.json")
            wa_verify = (
                wa_cfg.get("verify_token")
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
            self._record_pairing("WhatsApp", "send to your number \u2192  !start", wa_code)

        return webhook_routers

    def stop_all(self) -> None:
        """Stop all active messaging channels."""
        for ch in self.active:
            try:
                asyncio.run(ch.stop())
            except Exception as e:
                logger.debug("Error stopping channel %s: %s", type(ch).__name__, e)

    def _start_async_channel(self, channel) -> None:
        """Start an async channel in a background thread."""
        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(channel.start())
            loop.run_forever()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        self.active.append(channel)

    def _record_pairing(self, label: str, instruction: str, code: str | None) -> None:
        """Record pairing status for a channel (deferred display)."""
        if code:
            self.pairing_instructions.append(f"  {label}: {instruction} {code}")
        else:
            self.channel_status.append((label, True))
