"""Tests for WhatsApp channel adapter.

Verifies webhook verification, incoming message processing, pairing flow,
send_text with mocked httpx, and non-text message skipping.
Uses FastAPI TestClient for webhook endpoint tests.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.channels.whatsapp import WhatsAppChannel


# ── helpers ──────────────────────────────────────────────────────

def _make_channel(
    paired: dict | None = None,
    agents: list[str] | None = None,
    **overrides,
) -> WhatsAppChannel:
    agents = agents or ["alpha", "beta"]

    async def dispatch_fn(agent: str, message: str) -> str:
        return f"reply from {agent}"

    def list_agents_fn():
        return {a: {} for a in agents}

    def status_fn(name: str):
        return {"state": "running", "tasks_completed": 5}

    def costs_fn():
        return [{"agent": "alpha", "tokens": 100, "cost": 0.01}]

    defaults = {
        "access_token": "test-token",
        "phone_number_id": "123456",
        "verify_token": "my-verify-token",
        "dispatch_fn": dispatch_fn,
        "default_agent": "alpha",
        "list_agents_fn": list_agents_fn,
        "status_fn": status_fn,
        "costs_fn": costs_fn,
    }
    defaults.update(overrides)
    ch = WhatsAppChannel(**defaults)
    if paired is not None:
        ch._paired = paired
    return ch


def _make_app(channel: WhatsAppChannel) -> FastAPI:
    """Create a test FastAPI app with the WhatsApp webhook router."""
    app = FastAPI()
    app.include_router(channel.create_router())
    return app


def _webhook_payload(from_phone: str, text: str) -> dict:
    """Create a standard WhatsApp webhook payload for a text message."""
    return {
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [{
                        "from": from_phone,
                        "type": "text",
                        "text": {"body": text},
                    }],
                },
            }],
        }],
    }


# ── webhook verification ────────────────────────────────────────

class TestWebhookVerification:
    def test_correct_token_returns_challenge(self):
        ch = _make_channel()
        app = _make_app(ch)
        client = TestClient(app)
        resp = client.get(
            "/channels/whatsapp/webhook",
            params={
                "hub.mode": "subscribe",
                "hub.verify_token": "my-verify-token",
                "hub.challenge": "12345",
            },
        )
        assert resp.status_code == 200
        assert resp.json() == 12345

    def test_wrong_token_fails(self):
        ch = _make_channel()
        app = _make_app(ch)
        client = TestClient(app)
        resp = client.get(
            "/channels/whatsapp/webhook",
            params={
                "hub.mode": "subscribe",
                "hub.verify_token": "wrong-token",
                "hub.challenge": "12345",
            },
        )
        assert resp.status_code == 200
        assert resp.json() == "Verification failed"

    def test_missing_mode_fails(self):
        ch = _make_channel()
        app = _make_app(ch)
        client = TestClient(app)
        resp = client.get(
            "/channels/whatsapp/webhook",
            params={"hub.verify_token": "my-verify-token", "hub.challenge": "12345"},
        )
        assert resp.json() == "Verification failed"


# ── incoming message processing ──────────────────────────────────

class TestIncomingMessages:
    @pytest.mark.asyncio
    async def test_process_text_message(self):
        ch = _make_channel(paired={"owner": "+1234", "allowed": []})
        ch._http = AsyncMock()
        ch._http.post = AsyncMock()

        msg = {"from": "+1234", "type": "text", "text": {"body": "hello"}}
        await ch._process_message(msg)

        ch._http.post.assert_called()
        # Check the response includes the agent label
        call_args = ch._http.post.call_args
        body = call_args[1]["json"]["text"]["body"]
        assert "[alpha]" in body

    @pytest.mark.asyncio
    async def test_non_text_message_skipped(self):
        ch = _make_channel(paired={"owner": "+1234", "allowed": []})
        ch._http = AsyncMock()
        ch._http.post = AsyncMock()

        msg = {"from": "+1234", "type": "image", "image": {"id": "img123"}}
        await ch._process_message(msg)

        ch._http.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_text_skipped(self):
        ch = _make_channel(paired={"owner": "+1234", "allowed": []})
        ch._http = AsyncMock()
        ch._http.post = AsyncMock()

        msg = {"from": "+1234", "type": "text", "text": {"body": "   "}}
        await ch._process_message(msg)

        ch._http.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_from_skipped(self):
        ch = _make_channel(paired={"owner": "+1234", "allowed": []})
        ch._http = AsyncMock()
        ch._http.post = AsyncMock()

        msg = {"type": "text", "text": {"body": "hello"}}
        await ch._process_message(msg)

        ch._http.post.assert_not_called()


# ── webhook POST endpoint ───────────────────────────────────────

class TestWebhookPost:
    def test_webhook_post_returns_ok(self):
        ch = _make_channel(paired={"owner": "+1234", "allowed": []})
        app = _make_app(ch)
        client = TestClient(app)
        payload = _webhook_payload("+1234", "hello")
        resp = client.post("/channels/whatsapp/webhook", json=payload)
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_webhook_post_empty_body_ok(self):
        ch = _make_channel(paired={"owner": "+1234", "allowed": []})
        app = _make_app(ch)
        client = TestClient(app)
        resp = client.post(
            "/channels/whatsapp/webhook",
            content=b"not json",
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 200


# ── pairing flow ────────────────────────────────────────────────

class TestPairing:
    @pytest.mark.asyncio
    async def test_pairing_with_correct_code(self):
        ch = _make_channel(
            paired={"owner": None, "allowed": [], "pairing_code": "abc123"},
        )
        ch._http = AsyncMock()
        ch._http.post = AsyncMock()

        msg = {"from": "+1234", "type": "text", "text": {"body": "!start abc123"}}
        await ch._process_message(msg)
        assert ch._paired["owner"] == "+1234"

    @pytest.mark.asyncio
    async def test_pairing_with_wrong_code(self):
        ch = _make_channel(
            paired={"owner": None, "allowed": [], "pairing_code": "abc123"},
        )
        ch._http = AsyncMock()
        ch._http.post = AsyncMock()

        msg = {"from": "+1234", "type": "text", "text": {"body": "!start wrong"}}
        await ch._process_message(msg)
        assert ch._paired["owner"] is None

    @pytest.mark.asyncio
    async def test_disallowed_user_rejected(self):
        ch = _make_channel(paired={"owner": "+0000", "allowed": []})
        ch._http = AsyncMock()
        ch._http.post = AsyncMock()

        msg = {"from": "+9999", "type": "text", "text": {"body": "!start code"}}
        await ch._process_message(msg)
        body = ch._http.post.call_args[1]["json"]["text"]["body"]
        assert "denied" in body.lower()

    @pytest.mark.asyncio
    async def test_allowed_user_can_chat(self):
        ch = _make_channel(paired={"owner": "+0000", "allowed": ["+1111"]})
        ch._http = AsyncMock()
        ch._http.post = AsyncMock()

        msg = {"from": "+1111", "type": "text", "text": {"body": "hello"}}
        await ch._process_message(msg)
        body = ch._http.post.call_args[1]["json"]["text"]["body"]
        assert "[alpha]" in body

    @pytest.mark.asyncio
    async def test_owner_allow_command(self):
        ch = _make_channel(paired={"owner": "+0000", "allowed": []})
        ch._http = AsyncMock()
        ch._http.post = AsyncMock()

        msg = {"from": "+0000", "type": "text", "text": {"body": "!allow +1111"}}
        await ch._process_message(msg)
        assert "+1111" in ch._paired["allowed"]

    @pytest.mark.asyncio
    async def test_owner_revoke_command(self):
        ch = _make_channel(paired={"owner": "+0000", "allowed": ["+1111"]})
        ch._http = AsyncMock()
        ch._http.post = AsyncMock()

        msg = {"from": "+0000", "type": "text", "text": {"body": "!revoke +1111"}}
        await ch._process_message(msg)
        assert "+1111" not in ch._paired["allowed"]


# ── _send_text ──────────────────────────────────────────────────

class TestSendText:
    @pytest.mark.asyncio
    async def test_send_text_calls_api(self):
        ch = _make_channel()
        ch._http = AsyncMock()
        ch._http.post = AsyncMock()

        await ch._send_text("+1234", "hello")
        ch._http.post.assert_called_once()
        call_args = ch._http.post.call_args
        assert call_args[0][0] == "/123456/messages"
        payload = call_args[1]["json"]
        assert payload["messaging_product"] == "whatsapp"
        assert payload["to"] == "+1234"
        assert payload["text"]["body"] == "hello"

    @pytest.mark.asyncio
    async def test_send_text_no_client_is_noop(self):
        ch = _make_channel()
        ch._http = None
        await ch._send_text("+1234", "hello")  # should not raise


# ── send_notification ────────────────────────────────────────────

class TestSendNotification:
    @pytest.mark.asyncio
    async def test_notification_sends_to_all_phones(self):
        ch = _make_channel()
        ch._phone_numbers = {"+1111", "+2222"}
        ch._http = AsyncMock()
        ch._http.post = AsyncMock()

        await ch.send_notification("test notification")
        assert ch._http.post.call_count == 2

    @pytest.mark.asyncio
    async def test_notification_no_phones_is_noop(self):
        ch = _make_channel()
        ch._phone_numbers = set()
        ch._http = AsyncMock()
        await ch.send_notification("test")

    @pytest.mark.asyncio
    async def test_notification_no_client_is_noop(self):
        ch = _make_channel()
        ch._http = None
        await ch.send_notification("test")


# ── command translation ─────────────────────────────────────────

class TestCommandTranslation:
    @pytest.mark.asyncio
    async def test_exclamation_commands_translated(self):
        ch = _make_channel(paired={"owner": "+1234", "allowed": []})
        ch._http = AsyncMock()
        ch._http.post = AsyncMock()

        msg = {"from": "+1234", "type": "text", "text": {"body": "!agents"}}
        await ch._process_message(msg)
        body = ch._http.post.call_args[1]["json"]["text"]["body"]
        assert "alpha" in body
        assert "beta" in body
