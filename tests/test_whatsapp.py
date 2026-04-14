"""Tests for WhatsApp channel adapter.

Verifies webhook verification, incoming message processing, pairing flow,
send_text with mocked httpx, and non-text message skipping.
Uses FastAPI TestClient for webhook endpoint tests.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

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
        ch._pairing._data = paired
    return ch


def _make_app(channel: WhatsAppChannel) -> FastAPI:
    """Create a test FastAPI app with the WhatsApp webhook router."""
    app = FastAPI()
    app.include_router(channel.create_router())
    return app


def _mock_ok_response():
    """Return a mock httpx response with status_code=200."""
    resp = MagicMock()
    resp.status_code = 200
    return resp


def _make_http_mock():
    """Return a mock httpx.AsyncClient whose post() returns a 200 response."""
    http = AsyncMock()
    http.post = AsyncMock(return_value=_mock_ok_response())
    return http


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
        ch._http = _make_http_mock()

        msg = {"from": "+1234", "type": "text", "text": {"body": "hello"}}
        await ch._process_message(msg)

        ch._http.post.assert_called()
        # Check the response includes the agent label
        call_args = ch._http.post.call_args
        body = call_args[1]["json"]["text"]["body"]
        assert "[alpha]" in body

    @pytest.mark.asyncio
    async def test_non_text_message_replies_to_allowed_user(self):
        ch = _make_channel(paired={"owner": "+1234", "allowed": []})
        ch._http = _make_http_mock()

        msg = {"from": "+1234", "type": "image", "image": {"id": "img123"}}
        await ch._process_message(msg)

        ch._http.post.assert_called_once()
        body = ch._http.post.call_args[1]["json"]["text"]["body"]
        assert "text messages" in body.lower()

    @pytest.mark.asyncio
    async def test_non_text_message_skipped_for_unknown_user(self):
        ch = _make_channel(paired={"owner": "+1234", "allowed": []})
        ch._http = _make_http_mock()

        msg = {"from": "+9999", "type": "image", "image": {"id": "img123"}}
        await ch._process_message(msg)

        ch._http.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_text_skipped(self):
        ch = _make_channel(paired={"owner": "+1234", "allowed": []})
        ch._http = _make_http_mock()

        msg = {"from": "+1234", "type": "text", "text": {"body": "   "}}
        await ch._process_message(msg)

        ch._http.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_from_skipped(self):
        ch = _make_channel(paired={"owner": "+1234", "allowed": []})
        ch._http = _make_http_mock()

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
        ch._http = _make_http_mock()

        msg = {"from": "+1234", "type": "text", "text": {"body": "!start abc123"}}
        await ch._process_message(msg)
        assert ch._pairing.owner == "+1234"

    @pytest.mark.asyncio
    async def test_pairing_with_wrong_code(self):
        ch = _make_channel(
            paired={"owner": None, "allowed": [], "pairing_code": "abc123"},
        )
        ch._http = _make_http_mock()

        msg = {"from": "+1234", "type": "text", "text": {"body": "!start wrong"}}
        await ch._process_message(msg)
        assert ch._pairing.owner is None

    @pytest.mark.asyncio
    async def test_disallowed_user_rejected(self):
        ch = _make_channel(paired={"owner": "+0000", "allowed": []})
        ch._http = _make_http_mock()

        msg = {"from": "+9999", "type": "text", "text": {"body": "!start code"}}
        await ch._process_message(msg)
        body = ch._http.post.call_args[1]["json"]["text"]["body"]
        assert "denied" in body.lower()

    @pytest.mark.asyncio
    async def test_allowed_user_can_chat(self):
        ch = _make_channel(paired={"owner": "+0000", "allowed": ["+1111"]})
        ch._http = _make_http_mock()

        msg = {"from": "+1111", "type": "text", "text": {"body": "hello"}}
        await ch._process_message(msg)
        body = ch._http.post.call_args[1]["json"]["text"]["body"]
        assert "[alpha]" in body

    @pytest.mark.asyncio
    async def test_owner_allow_command(self):
        ch = _make_channel(paired={"owner": "+0000", "allowed": []})
        ch._http = _make_http_mock()

        msg = {"from": "+0000", "type": "text", "text": {"body": "!allow +1111"}}
        await ch._process_message(msg)
        assert ch._pairing.is_allowed("+1111")

    @pytest.mark.asyncio
    async def test_owner_revoke_command(self):
        ch = _make_channel(paired={"owner": "+0000", "allowed": ["+1111"]})
        ch._http = _make_http_mock()

        msg = {"from": "+0000", "type": "text", "text": {"body": "!revoke +1111"}}
        await ch._process_message(msg)
        assert not ch._pairing.is_allowed("+1111")


# ── _send_text ──────────────────────────────────────────────────

class TestSendText:
    @pytest.mark.asyncio
    async def test_send_text_calls_api(self):
        ch = _make_channel()
        ch._http = _make_http_mock()

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
        ch._http = _make_http_mock()

        await ch.send_notification("test notification")
        assert ch._http.post.call_count == 2

    @pytest.mark.asyncio
    async def test_notification_no_phones_is_noop(self):
        ch = _make_channel()
        ch._phone_numbers = set()
        ch._http = _make_http_mock()
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
        ch._http = _make_http_mock()

        msg = {"from": "+1234", "type": "text", "text": {"body": "!agents"}}
        await ch._process_message(msg)
        body = ch._http.post.call_args[1]["json"]["text"]["body"]
        assert "alpha" in body
        assert "beta" in body


# ── Fix 3: _send_text error logging ────────────────────────────────

class TestSendTextErrorLogging:
    @pytest.mark.asyncio
    async def test_200_response_no_warning(self, caplog):
        ch = _make_channel()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        ch._http = AsyncMock()
        ch._http.post = AsyncMock(return_value=mock_resp)

        with caplog.at_level(logging.WARNING, logger="channels.whatsapp"):
            await ch._send_text("+1234", "hello")

        assert not caplog.records

    @pytest.mark.asyncio
    async def test_400_response_logs_warning_with_body(self, caplog):
        ch = _make_channel()
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.json.return_value = {"error": {"message": "Invalid phone number"}}
        ch._http = AsyncMock()
        ch._http.post = AsyncMock(return_value=mock_resp)

        with caplog.at_level(logging.WARNING, logger="channels.whatsapp"):
            await ch._send_text("+1234", "hello")

        assert any("HTTP 400" in r.message for r in caplog.records)
        assert any("Invalid phone number" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_network_error_logs_warning(self, caplog):
        import httpx

        ch = _make_channel()
        ch._http = AsyncMock()
        ch._http.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

        with caplog.at_level(logging.WARNING, logger="channels.whatsapp"):
            await ch._send_text("+1234", "hello")  # should not raise

        assert any("network" in r.message.lower() for r in caplog.records)


# ── Fix 1: start_channel is async ──────────────────────────────────

class TestStartChannelAsync:
    @pytest.mark.asyncio
    async def test_start_channel_whatsapp_can_be_awaited(self, tmp_path):
        """start_channel must be awaitable (fixes asyncio.run crash in dashboard)."""
        import os

        from src.cli.channels import ChannelManager

        async def dispatch_fn(agent, msg, **kwargs):
            return "ok"

        cfg = {"agents": {"alpha": {}}, "channels": {}}
        manager = ChannelManager(cfg, dispatch_fn, {"alpha": {}})

        # Mock WhatsAppChannel.start to avoid real HTTP setup
        async def fake_start(self_inner):
            pass

        import src.channels.whatsapp as wa_mod
        original_start = wa_mod.WhatsAppChannel.start

        wa_mod.WhatsAppChannel.start = fake_start
        # Patch MESH_AUTH_TOKEN to be absent so start() doesn't raise
        old_env = os.environ.pop("MESH_AUTH_TOKEN", None)
        try:
            tokens = {
                "access_token": "tok",
                "phone_number_id": "phone123",
                "verify_token": "vtoken",
            }
            # This must not raise "cannot be called from a running event loop"
            routers = await manager.start_channel("whatsapp", tokens)
            assert isinstance(routers, list)
        finally:
            wa_mod.WhatsAppChannel.start = original_start
            if old_env is not None:
                os.environ["MESH_AUTH_TOKEN"] = old_env


# ── send_to_user ──────────────────────────────────────────────────

class TestSendToUser:
    @pytest.mark.asyncio
    async def test_send_to_user_calls_send_text_for_each_chunk(self):
        ch = _make_channel()
        ch._http = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        ch._http.post = AsyncMock(return_value=mock_resp)

        await ch.send_to_user("+1234", "hello world")
        ch._http.post.assert_called_once()
        payload = ch._http.post.call_args[1]["json"]
        assert payload["to"] == "+1234"

    @pytest.mark.asyncio
    async def test_send_to_user_no_client_is_noop(self):
        ch = _make_channel()
        ch._http = None
        await ch.send_to_user("+1234", "hello")  # should not raise
