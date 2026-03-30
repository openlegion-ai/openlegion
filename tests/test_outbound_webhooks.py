"""Tests for OutboundWebhookManager: CRUD, delivery, HMAC, filtering, SSRF, retry."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import shutil
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.host.outbound_webhooks import (
    VALID_EVENT_TYPES,
    OutboundWebhookManager,
    _check_url_ssrf,
    _is_blocked_ip,
)


class TestSubscriptionCRUD:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.config_path = f"{self._tmpdir}/outbound_webhooks.json"

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_add_subscription(self):
        mgr = OutboundWebhookManager(config_path=self.config_path)
        sub = mgr.add_subscription(
            "My Webhook", "https://example.com/hook", ["notification"],
        )
        assert sub["name"] == "My Webhook"
        assert sub["url"] == "https://example.com/hook"
        assert sub["events"] == ["notification"]
        assert sub["enabled"] is True
        assert len(sub["secret"]) == 64  # hex of 32 bytes
        assert sub["delivery_count"] == 0
        assert sub["id"].startswith("owh_")

    def test_add_subscription_with_agent_filter(self):
        mgr = OutboundWebhookManager(config_path=self.config_path)
        sub = mgr.add_subscription(
            "Filtered", "https://example.com/hook",
            ["task_complete"], agent_filter=["agent-1", "agent-2"],
        )
        assert sub["agent_filter"] == ["agent-1", "agent-2"]

    def test_add_subscription_invalid_url(self):
        mgr = OutboundWebhookManager(config_path=self.config_path)
        with pytest.raises(ValueError, match="https"):
            mgr.add_subscription("test", "http://example.com/hook", ["notification"])

    def test_add_subscription_invalid_events(self):
        mgr = OutboundWebhookManager(config_path=self.config_path)
        with pytest.raises(ValueError, match="Invalid event types"):
            mgr.add_subscription("test", "https://example.com", ["bogus_event"])

    def test_add_subscription_empty_events(self):
        mgr = OutboundWebhookManager(config_path=self.config_path)
        with pytest.raises(ValueError, match="At least one event"):
            mgr.add_subscription("test", "https://example.com", [])

    def test_add_subscription_empty_name(self):
        mgr = OutboundWebhookManager(config_path=self.config_path)
        with pytest.raises(ValueError, match="name must not be empty"):
            mgr.add_subscription("", "https://example.com", ["notification"])

    def test_add_subscription_empty_url(self):
        mgr = OutboundWebhookManager(config_path=self.config_path)
        with pytest.raises(ValueError, match="url must not be empty"):
            mgr.add_subscription("test", "", ["notification"])

    def test_remove_subscription(self):
        mgr = OutboundWebhookManager(config_path=self.config_path)
        sub = mgr.add_subscription("test", "https://example.com", ["notification"])
        assert mgr.remove_subscription(sub["id"]) is True
        assert mgr.remove_subscription(sub["id"]) is False

    def test_update_subscription(self):
        mgr = OutboundWebhookManager(config_path=self.config_path)
        sub = mgr.add_subscription("test", "https://example.com", ["notification"])
        updated = mgr.update_subscription(
            sub["id"], name="Updated Name", enabled=False,
        )
        assert updated["name"] == "Updated Name"
        assert updated["enabled"] is False
        assert updated["url"] == "https://example.com"  # unchanged

    def test_update_subscription_validation(self):
        mgr = OutboundWebhookManager(config_path=self.config_path)
        sub = mgr.add_subscription("test", "https://example.com", ["notification"])
        with pytest.raises(ValueError, match="https"):
            mgr.update_subscription(sub["id"], url="http://insecure.com")
        with pytest.raises(ValueError, match="Invalid event"):
            mgr.update_subscription(sub["id"], events=["not_real"])
        with pytest.raises(ValueError, match="name must not be empty"):
            mgr.update_subscription(sub["id"], name="")

    def test_update_nonexistent(self):
        mgr = OutboundWebhookManager(config_path=self.config_path)
        assert mgr.update_subscription("owh_nonexistent") is None

    def test_list_subscriptions(self):
        mgr = OutboundWebhookManager(config_path=self.config_path)
        mgr.add_subscription("A", "https://a.com", ["notification"])
        mgr.add_subscription("B", "https://b.com", ["task_complete"])
        subs = mgr.list_subscriptions()
        assert len(subs) == 2
        names = {s["name"] for s in subs}
        assert names == {"A", "B"}


class TestConfigPersistence:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.config_path = f"{self._tmpdir}/outbound_webhooks.json"

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_save_and_load(self):
        mgr1 = OutboundWebhookManager(config_path=self.config_path)
        sub = mgr1.add_subscription("persist", "https://example.com", ["notification"])

        mgr2 = OutboundWebhookManager(config_path=self.config_path)
        assert len(mgr2.subscriptions) == 1
        loaded = list(mgr2.subscriptions.values())[0]
        assert loaded["name"] == "persist"
        assert loaded["secret"] == sub["secret"]

    def test_load_missing_file(self):
        mgr = OutboundWebhookManager(config_path=f"{self._tmpdir}/nonexistent.json")
        assert len(mgr.subscriptions) == 0


class TestHMACSignature:
    def test_signature_matches(self):
        """Verify the HMAC signature format matches expectations."""
        import secrets
        secret = secrets.token_hex(32)
        payload = {"event": "test", "agent": "a1", "data": {}, "timestamp": "T", "delivery_id": "dlv_1"}
        body_bytes = json.dumps(payload).encode()
        expected = hmac.new(secret.encode(), body_bytes, hashlib.sha256).hexdigest()
        # Verify the format is sha256=<hex>
        header_value = f"sha256={expected}"
        assert header_value.startswith("sha256=")
        assert len(expected) == 64


class TestEventFiltering:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.config_path = f"{self._tmpdir}/outbound_webhooks.json"

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_enqueue_matching_event(self):
        mgr = OutboundWebhookManager(config_path=self.config_path)
        mgr.add_subscription("test", "https://example.com", ["notification"])
        mgr.enqueue("notification", "agent-1", {"msg": "hello"})
        assert mgr._queue.qsize() == 1

    def test_enqueue_non_matching_event(self):
        mgr = OutboundWebhookManager(config_path=self.config_path)
        mgr.add_subscription("test", "https://example.com", ["notification"])
        mgr.enqueue("task_complete", "agent-1", {"task": "done"})
        assert mgr._queue.qsize() == 0

    def test_enqueue_agent_filter(self):
        mgr = OutboundWebhookManager(config_path=self.config_path)
        mgr.add_subscription(
            "filtered", "https://example.com", ["notification"],
            agent_filter=["agent-1"],
        )
        mgr.enqueue("notification", "agent-1", {"msg": "yes"})
        assert mgr._queue.qsize() == 1
        mgr.enqueue("notification", "agent-2", {"msg": "no"})
        assert mgr._queue.qsize() == 1  # agent-2 filtered out

    def test_enqueue_disabled_subscription(self):
        mgr = OutboundWebhookManager(config_path=self.config_path)
        sub = mgr.add_subscription("test", "https://example.com", ["notification"])
        mgr.update_subscription(sub["id"], enabled=False)
        mgr.enqueue("notification", "agent-1", {"msg": "hello"})
        assert mgr._queue.qsize() == 0

    def test_enqueue_empty_agent_filter_matches_all(self):
        mgr = OutboundWebhookManager(config_path=self.config_path)
        mgr.add_subscription("test", "https://example.com", ["notification"])
        mgr.enqueue("notification", "any-agent", {"msg": "hi"})
        assert mgr._queue.qsize() == 1


@pytest.mark.asyncio
class TestDelivery:
    async def test_successful_delivery(self, tmp_path):
        config_path = str(tmp_path / "webhooks.json")
        mgr = OutboundWebhookManager(config_path=config_path)
        sub = mgr.add_subscription("test", "https://example.com/hook", ["notification"])

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(mgr, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value = mock_client

            payload = {
                "event": "notification",
                "agent": "agent-1",
                "data": {"msg": "hello"},
                "timestamp": "2026-01-01T00:00:00Z",
                "delivery_id": "dlv_test123",
            }
            with patch("src.host.outbound_webhooks._check_url_ssrf", return_value=None):
                result = await mgr._deliver(sub, payload, "dlv_test123")

        assert result["status"] == "delivered"
        assert result["response_code"] == 200
        # Verify headers
        call_kwargs = mock_client.post.call_args
        headers = call_kwargs.kwargs.get("headers", {})
        assert "X-OpenLegion-Signature" in headers
        assert headers["X-OpenLegion-Signature"].startswith("sha256=")
        assert headers["X-OpenLegion-Event"] == "notification"
        assert headers["User-Agent"] == "OpenLegion/1.0"

    async def test_delivery_correct_hmac(self, tmp_path):
        """Verify delivered HMAC matches manual computation."""
        config_path = str(tmp_path / "webhooks.json")
        mgr = OutboundWebhookManager(config_path=config_path)
        sub = mgr.add_subscription("test", "https://example.com/hook", ["notification"])
        secret = sub["secret"]

        mock_response = MagicMock()
        mock_response.status_code = 200
        captured_body = None
        captured_headers = None

        async def capture_post(url, content=None, headers=None, timeout=None):
            nonlocal captured_body, captured_headers
            captured_body = content
            captured_headers = headers
            return mock_response

        with patch.object(mgr, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = capture_post
            mock_get_client.return_value = mock_client

            payload = {"event": "test", "agent": "a1", "data": {}, "timestamp": "T", "delivery_id": "d1"}
            with patch("src.host.outbound_webhooks._check_url_ssrf", return_value=None):
                await mgr._deliver(sub, payload, "d1")

        expected = hmac.new(secret.encode(), captured_body, hashlib.sha256).hexdigest()
        assert captured_headers["X-OpenLegion-Signature"] == f"sha256={expected}"


@pytest.mark.asyncio
class TestRetry:
    async def test_retry_on_failure(self, tmp_path):
        """Verify retry attempts with exponential backoff."""
        config_path = str(tmp_path / "webhooks.json")
        mgr = OutboundWebhookManager(config_path=config_path)
        sub = mgr.add_subscription("test", "https://example.com/hook", ["notification"])

        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch.object(mgr, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value = mock_client

            payload = {"event": "test", "agent": "a1", "data": {}, "timestamp": "T", "delivery_id": "d1"}
            # Patch sleep to avoid actual delays, and patch SSRF check
            with patch("src.host.outbound_webhooks.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                with patch("src.host.outbound_webhooks._check_url_ssrf", return_value=None):
                    result = await mgr._deliver(sub, payload, "d1")

        assert result["status"] == "failed"
        assert "Failed after 4 attempts" in result["error"]
        # Should have retried 3 times (with 3 sleep calls)
        assert mock_sleep.call_count == 3
        assert mock_sleep.call_args_list[0].args[0] == 5
        assert mock_sleep.call_args_list[1].args[0] == 30
        assert mock_sleep.call_args_list[2].args[0] == 120

    async def test_retry_succeeds_on_second_attempt(self, tmp_path):
        config_path = str(tmp_path / "webhooks.json")
        mgr = OutboundWebhookManager(config_path=config_path)
        sub = mgr.add_subscription("test", "https://example.com/hook", ["notification"])

        fail_response = MagicMock()
        fail_response.status_code = 502
        ok_response = MagicMock()
        ok_response.status_code = 200

        with patch.object(mgr, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.side_effect = [fail_response, ok_response]
            mock_get_client.return_value = mock_client

            payload = {"event": "test", "agent": "a1", "data": {}, "timestamp": "T", "delivery_id": "d1"}
            with patch("src.host.outbound_webhooks.asyncio.sleep", new_callable=AsyncMock):
                with patch("src.host.outbound_webhooks._check_url_ssrf", return_value=None):
                    result = await mgr._deliver(sub, payload, "d1")

        assert result["status"] == "delivered"


class TestSSRFBlocking:
    def test_blocked_private_ip(self):
        import ipaddress
        assert _is_blocked_ip(ipaddress.ip_address("10.0.0.1")) is True
        assert _is_blocked_ip(ipaddress.ip_address("192.168.1.1")) is True
        assert _is_blocked_ip(ipaddress.ip_address("127.0.0.1")) is True
        assert _is_blocked_ip(ipaddress.ip_address("0.0.0.0")) is True

    def test_blocked_cgnat(self):
        import ipaddress
        assert _is_blocked_ip(ipaddress.ip_address("100.64.0.1")) is True
        assert _is_blocked_ip(ipaddress.ip_address("100.127.255.254")) is True

    def test_blocked_link_local(self):
        import ipaddress
        assert _is_blocked_ip(ipaddress.ip_address("169.254.1.1")) is True

    def test_allowed_public_ip(self):
        import ipaddress
        assert _is_blocked_ip(ipaddress.ip_address("8.8.8.8")) is False
        assert _is_blocked_ip(ipaddress.ip_address("93.184.216.34")) is False

    def test_blocked_ipv4_mapped_ipv6(self):
        import ipaddress
        assert _is_blocked_ip(ipaddress.ip_address("::ffff:127.0.0.1")) is True

    @pytest.mark.asyncio
    async def test_ssrf_blocks_delivery(self, tmp_path):
        config_path = str(tmp_path / "webhooks.json")
        mgr = OutboundWebhookManager(config_path=config_path)
        sub = mgr.add_subscription("test", "https://example.com/hook", ["notification"])

        payload = {"event": "test", "agent": "a1", "data": {}, "timestamp": "T", "delivery_id": "d1"}
        with patch("src.host.outbound_webhooks._check_url_ssrf", return_value="Blocked IP: 127.0.0.1"):
            result = await mgr._deliver(sub, payload, "d1")

        assert result["status"] == "failed"
        assert "SSRF blocked" in result["error"]

    def test_check_url_ssrf_private(self):
        """check_url_ssrf should detect private IPs."""
        with patch("socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [(2, 1, 0, '', ('127.0.0.1', 0))]
            result = _check_url_ssrf("https://internal.example.com/hook")
            assert result is not None
            assert "Blocked" in result

    def test_check_url_ssrf_public(self):
        """check_url_ssrf should allow public IPs."""
        with patch("socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [(2, 1, 0, '', ('93.184.216.34', 0))]
            result = _check_url_ssrf("https://example.com/hook")
            assert result is None


class TestRingBuffer:
    @pytest.mark.asyncio
    async def test_recent_deliveries(self, tmp_path):
        config_path = str(tmp_path / "webhooks.json")
        mgr = OutboundWebhookManager(config_path=config_path)
        sub = mgr.add_subscription("test", "https://example.com/hook", ["notification"])

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(mgr, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value = mock_client

            for i in range(3):
                payload = {"event": "test", "agent": "a1", "data": {}, "timestamp": "T", "delivery_id": f"d{i}"}
                with patch("src.host.outbound_webhooks._check_url_ssrf", return_value=None):
                    await mgr._deliver(sub, payload, f"d{i}")

        deliveries = mgr.recent_deliveries()
        assert len(deliveries) == 3
        assert all(d["status"] == "delivered" for d in deliveries)

    @pytest.mark.asyncio
    async def test_ring_buffer_maxlen(self, tmp_path):
        config_path = str(tmp_path / "webhooks.json")
        mgr = OutboundWebhookManager(config_path=config_path)
        # Manually fill the deque beyond maxlen
        for i in range(250):
            mgr._deliveries.append({"delivery_id": f"d{i}"})
        assert len(mgr.recent_deliveries()) == 200  # maxlen


class TestEventBusIntegration:
    def test_listener_integration(self, tmp_path):
        """EventBus.add_listener + emit should call enqueue."""
        from src.dashboard.events import EventBus

        config_path = str(tmp_path / "webhooks.json")
        mgr = OutboundWebhookManager(config_path=config_path)
        mgr.add_subscription("test", "https://example.com/hook", ["notification"])

        bus = EventBus()
        bus.add_listener(mgr.enqueue)

        bus.emit("notification", agent="agent-1", data={"msg": "hello"})
        assert mgr._queue.qsize() == 1

    def test_listener_non_matching_event(self, tmp_path):
        from src.dashboard.events import EventBus

        config_path = str(tmp_path / "webhooks.json")
        mgr = OutboundWebhookManager(config_path=config_path)
        mgr.add_subscription("test", "https://example.com/hook", ["notification"])

        bus = EventBus()
        bus.add_listener(mgr.enqueue)

        bus.emit("llm_call", agent="agent-1", data={"model": "gpt-4"})
        assert mgr._queue.qsize() == 0


@pytest.mark.asyncio
class TestDeliveryWorker:
    async def test_worker_processes_queue(self, tmp_path):
        config_path = str(tmp_path / "webhooks.json")
        mgr = OutboundWebhookManager(config_path=config_path)
        mgr.add_subscription("test", "https://example.com/hook", ["notification"])

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(mgr, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value = mock_client

            with patch("src.host.outbound_webhooks._check_url_ssrf", return_value=None):
                mgr.enqueue("notification", "agent-1", {"msg": "hi"})
                mgr.start()
                # Give worker time to process
                await asyncio.sleep(0.1)
                await mgr.stop()

        assert len(mgr.recent_deliveries()) == 1
        assert mgr.recent_deliveries()[0]["status"] == "delivered"


@pytest.mark.asyncio
class TestTestSubscription:
    async def test_test_delivery(self, tmp_path):
        config_path = str(tmp_path / "webhooks.json")
        mgr = OutboundWebhookManager(config_path=config_path)
        sub = mgr.add_subscription("test", "https://example.com/hook", ["notification"])

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(mgr, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value = mock_client

            with patch("src.host.outbound_webhooks._check_url_ssrf", return_value=None):
                result = await mgr.test_subscription(sub["id"])

        assert result["status"] == "delivered"

    async def test_test_nonexistent(self, tmp_path):
        config_path = str(tmp_path / "webhooks.json")
        mgr = OutboundWebhookManager(config_path=config_path)
        result = await mgr.test_subscription("owh_nonexistent")
        assert result is None


class TestValidEventTypes:
    def test_all_event_types_defined(self):
        expected = {"notification", "task_complete", "task_failed", "agent_state", "cron_complete", "custom"}
        assert VALID_EVENT_TYPES == expected
