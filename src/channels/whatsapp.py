"""WhatsApp channel adapter.

Bridges WhatsApp Business API (Cloud API) to the OpenLegion mesh with the
same UX as the CLI REPL:
  - Per-user active agent tracking (phone number as user key)
  - @agent mentions for routing to specific agents
  - /use, /agents, /status, /broadcast, /costs, /reset, /help commands
  - Agent name labels on all responses: [agent_name] response
  - Push notifications for cron/heartbeat results
  - Pairing: owner must send !start <pairing_code> to claim the bot.
    Code is generated during `openlegion setup`. Others need !allow.

Uses httpx (already a core dependency) to call the Cloud API.
Webhook-based: mounts GET/POST endpoints on the mesh FastAPI app.
Config: WHATSAPP_ACCESS_TOKEN, WHATSAPP_PHONE_NUMBER_ID in .env,
        channels.whatsapp in mesh.yaml.
"""

from __future__ import annotations

import asyncio

import httpx
from fastapi import APIRouter, Query, Request

from src.channels.base import Channel, PairingManager, chunk_text
from src.shared.utils import setup_logging

logger = setup_logging("channels.whatsapp")

MAX_WA_LEN = 4096
_GRAPH_API_BASE = "https://graph.facebook.com/v21.0"


class WhatsAppChannel(Channel):
    """WhatsApp Cloud API adapter for OpenLegion with webhook-based messaging."""

    def __init__(
        self,
        access_token: str,
        phone_number_id: str,
        verify_token: str,
        default_agent: str = "",
        **kwargs,
    ):
        super().__init__(default_agent=default_agent, **kwargs)
        self.access_token = access_token
        self.phone_number_id = phone_number_id
        self.verify_token = verify_token
        self._http: httpx.AsyncClient | None = None
        self._phone_numbers: set[str] = set()
        self._pairing = PairingManager("config/whatsapp_paired.json")

    async def start(self) -> None:
        self._http = httpx.AsyncClient(
            base_url=_GRAPH_API_BASE,
            headers={"Authorization": f"Bearer {self.access_token}"},
            timeout=30,
        )
        owner = self._pairing.owner
        if owner:
            logger.info(f"WhatsApp channel started (owner: {owner})")
        elif self._pairing.pairing_code:
            logger.info("WhatsApp channel started (awaiting pairing code)")
        else:
            logger.info("WhatsApp channel started (no pairing code -- run setup again)")

    async def stop(self) -> None:
        if self._http:
            await self._http.aclose()
            self._http = None
            logger.info("WhatsApp channel stopped")

    async def send_notification(self, text: str) -> None:
        """Push a cron/heartbeat notification to all known phone numbers."""
        if not self._http or not self._phone_numbers:
            return
        for phone in self._phone_numbers:
            try:
                for part in chunk_text(text, MAX_WA_LEN):
                    await self._send_text(phone, part)
            except Exception as e:
                logger.warning(f"Failed to notify {phone}: {e}")

    async def _send_text(self, to: str, text: str) -> None:
        """Send a text message via the WhatsApp Cloud API."""
        if not self._http:
            return
        await self._http.post(
            f"/{self.phone_number_id}/messages",
            json={
                "messaging_product": "whatsapp",
                "to": to,
                "type": "text",
                "text": {"body": text},
            },
        )

    def _is_allowed(self, phone: str) -> bool:
        return self._pairing.is_allowed(phone)

    def _is_owner(self, phone: str) -> bool:
        return self._pairing.is_owner(phone)

    def create_router(self) -> APIRouter:
        """Create a FastAPI router for WhatsApp webhook endpoints."""
        router = APIRouter(prefix="/channels/whatsapp")
        channel_ref = self

        @router.get("/webhook")
        async def verify_webhook(
            hub_mode: str = Query(None, alias="hub.mode"),
            hub_verify_token: str = Query(None, alias="hub.verify_token"),
            hub_challenge: str = Query(None, alias="hub.challenge"),
        ) -> int | str:
            """Verification challenge from Meta."""
            if hub_mode == "subscribe" and hub_verify_token == channel_ref.verify_token:
                logger.info("WhatsApp webhook verified")
                return int(hub_challenge) if hub_challenge else ""
            logger.warning("WhatsApp webhook verification failed")
            return "Verification failed"

        @router.post("/webhook")
        async def receive_webhook(request: Request) -> dict:
            """Receive incoming WhatsApp messages."""
            try:
                body = await request.json()
            except Exception:
                return {"status": "ok"}

            # Extract messages from the webhook payload
            for entry in body.get("entry", []):
                for change in entry.get("changes", []):
                    value = change.get("value", {})
                    for message in value.get("messages", []):
                        asyncio.create_task(
                            channel_ref._process_message(message)
                        )

            # Return 200 immediately (Cloud API retries if >5s)
            return {"status": "ok"}

        return router

    async def _process_message(self, message: dict) -> None:
        """Process a single incoming WhatsApp message."""
        msg_type = message.get("type", "")
        phone = message.get("from", "")

        if not phone:
            return

        # Text-only initially -- log and skip media
        if msg_type != "text":
            logger.info(f"Skipping non-text message type '{msg_type}' from {phone}")
            return

        text = (message.get("text", {}).get("body") or "").strip()
        if not text:
            return

        # Pairing: !start <code>
        if self._pairing.owner is None:
            if text.lower().startswith("!start") or text.lower().startswith("/start"):
                parts = text.split(None, 1)
                code_arg = parts[1].strip() if len(parts) > 1 else ""
                expected = self._pairing.pairing_code
                if not expected or code_arg != expected:
                    await self._send_text(
                        phone,
                        "Pairing required. Send:  !start <pairing_code>\n"
                        "The code was shown during `openlegion setup`.",
                    )
                    logger.warning(
                        f"Rejected !start without valid pairing code from {phone}"
                    )
                    return
                self._pairing.claim_owner(phone)
                logger.info(f"Paired owner via code: {phone}")
                self._phone_numbers.add(phone)
                await self._send_text(
                    phone,
                    f"Paired as owner. Your phone: {phone}\n"
                    f"Only you can use this bot. Send !allow <phone> to grant access.",
                )
                return
            else:
                await self._send_text(
                    phone,
                    "This bot requires pairing. Send !start <pairing_code> to begin.",
                )
                return

        if not self._is_allowed(phone):
            if text.lower().startswith("!start") or text.lower().startswith("/start"):
                await self._send_text(
                    phone,
                    f"Access denied. This bot is paired to its owner.\n"
                    f"Your phone: {phone}\n"
                    f"Ask the owner to send !allow {phone} to grant you access.",
                )
            return

        # Owner-only commands
        if text.startswith("!allow ") or text.startswith("/allow "):
            if not self._is_owner(phone):
                await self._send_text(phone, "Only the owner can use !allow.")
                return
            parts = text.split(None, 1)
            if len(parts) < 2 or not parts[1].strip():
                await self._send_text(phone, "Usage: !allow <phone_number>")
                return
            target = parts[1].strip()
            self._pairing.allow(target)
            await self._send_text(phone, f"User {target} is now allowed.")
            logger.info(f"Owner allowed user {target}")
            return

        if text.startswith("!revoke ") or text.startswith("/revoke "):
            if not self._is_owner(phone):
                await self._send_text(phone, "Only the owner can use !revoke.")
                return
            parts = text.split(None, 1)
            if len(parts) < 2 or not parts[1].strip():
                await self._send_text(phone, "Usage: !revoke <phone_number>")
                return
            target = parts[1].strip()
            if self._pairing.revoke(target):
                await self._send_text(phone, f"User {target} access revoked.")
            else:
                await self._send_text(
                    phone, f"User {target} was not in the allowed list."
                )
            return

        if text.lower() in ("!paired", "/paired"):
            if not self._is_owner(phone):
                await self._send_text(phone, "Only the owner can view pairing info.")
                return
            owner = self._pairing.owner
            allowed = self._pairing.allowed_list()
            lines = [f"Owner: {owner}"]
            if allowed:
                lines.append(f"Allowed users: {', '.join(allowed)}")
            else:
                lines.append("No additional users allowed.")
            await self._send_text(phone, "\n".join(lines))
            return

        self._phone_numbers.add(phone)

        # Translate ! commands to / for base class handling
        if text.startswith("!"):
            text = "/" + text[1:]

        try:
            response = await self.handle_message(phone, text)
        except Exception as e:
            logger.error(f"Dispatch failed for {phone}: {e}")
            response = f"Error: {e}"

        if response:
            for part in chunk_text(response, MAX_WA_LEN):
                try:
                    await self._send_text(phone, part)
                except Exception as e:
                    logger.warning(f"Failed to send response to {phone}: {e}")
