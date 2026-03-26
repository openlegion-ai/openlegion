"""Outbound webhook delivery — POST events to registered third-party URLs.

Users register subscriptions (URL + event types). When matching events
occur, the manager queues delivery and POSTs signed payloads with retry.

State persisted to config/outbound_webhooks.json.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import ipaddress
import json
import secrets
import socket
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx

from src.shared.utils import generate_id, setup_logging

logger = setup_logging("host.outbound_webhooks")

VALID_EVENT_TYPES = frozenset({
    "notification",    # anything through notify_user
    "task_complete",   # agent task finished successfully
    "task_failed",     # agent task errored/timed out
    "agent_state",     # agent state changes
    "cron_complete",   # scheduled cron job finished
    "custom",          # agent-emitted custom events
})

_CGNAT_NETWORK = ipaddress.IPv4Network("100.64.0.0/10")

_RETRY_DELAYS = (5, 30, 120)  # seconds between retry attempts
_DELIVERY_TIMEOUT = 10  # seconds per attempt
_RING_BUFFER_SIZE = 200


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True if the IP is in a range that should be blocked for SSRF."""
    if (ip.is_private or ip.is_loopback or ip.is_link_local
            or ip.is_reserved or ip.is_unspecified or ip.is_multicast):
        return True
    if isinstance(ip, ipaddress.IPv4Address) and ip in _CGNAT_NETWORK:
        return True
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        mapped = ip.ipv4_mapped
        if (mapped.is_private or mapped.is_loopback or mapped.is_link_local
                or mapped.is_reserved or mapped.is_unspecified or mapped.is_multicast):
            return True
        if mapped in _CGNAT_NETWORK:
            return True
    if isinstance(ip, ipaddress.IPv6Address) and ip.sixtofour:
        embedded = ip.sixtofour
        if (embedded.is_private or embedded.is_loopback or embedded.is_link_local
                or embedded.is_reserved or embedded.is_unspecified or embedded.is_multicast):
            return True
        if embedded in _CGNAT_NETWORK:
            return True
    if isinstance(ip, ipaddress.IPv6Address) and ip.teredo:
        _, teredo_client = ip.teredo
        if (teredo_client.is_private or teredo_client.is_loopback or teredo_client.is_link_local
                or teredo_client.is_reserved or teredo_client.is_unspecified or teredo_client.is_multicast):
            return True
        if teredo_client in _CGNAT_NETWORK:
            return True
    return False


def _check_url_ssrf(url: str) -> str | None:
    """Resolve URL hostname and check for SSRF. Returns error string or None."""
    parsed = urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        return "Malformed URL"
    # If hostname is an IP literal, validate directly
    try:
        ip = ipaddress.ip_address(hostname)
        if _is_blocked_ip(ip):
            return f"Blocked IP: {ip}"
        return None
    except ValueError:
        pass
    # DNS resolution
    try:
        infos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror:
        return f"DNS resolution failed for {hostname}"
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
            if _is_blocked_ip(ip):
                return f"Blocked IP: {ip} (resolved from {hostname})"
        except ValueError:
            continue
    return None


class OutboundWebhookManager:
    """Manages outbound webhook subscriptions and event delivery."""

    def __init__(self, config_path: str = "config/outbound_webhooks.json"):
        self.config_path = Path(config_path)
        self.subscriptions: dict[str, dict] = {}
        self._queue: asyncio.Queue = asyncio.Queue()
        self._client: httpx.AsyncClient | None = None
        self._worker_task: asyncio.Task | None = None
        self._stopping = False
        self._deliveries: deque[dict] = deque(maxlen=_RING_BUFFER_SIZE)
        self._load()

    def _load(self) -> None:
        if not self.config_path.exists():
            return
        try:
            data = json.loads(self.config_path.read_text())
            for sub in data.get("subscriptions", []):
                self.subscriptions[sub["id"]] = sub
        except Exception as e:
            logger.warning("Failed to load outbound webhook config: %s", e)

    def _save(self) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"subscriptions": list(self.subscriptions.values())}
        self.config_path.write_text(json.dumps(data, indent=2) + "\n")

    def add_subscription(
        self,
        name: str,
        url: str,
        events: list[str],
        agent_filter: list[str] | None = None,
    ) -> dict:
        """Register a new webhook subscription. Returns the subscription dict."""
        if not name or not name.strip():
            raise ValueError("name must not be empty")
        if not url or not url.strip():
            raise ValueError("url must not be empty")
        parsed = urlparse(url.strip())
        if parsed.scheme != "https":
            raise ValueError("URL must use https")
        invalid_events = set(events) - VALID_EVENT_TYPES
        if invalid_events:
            raise ValueError(f"Invalid event types: {', '.join(sorted(invalid_events))}")
        if not events:
            raise ValueError("At least one event type is required")

        sub_id = generate_id("owh")[:16]
        sub: dict = {
            "id": sub_id,
            "name": name.strip(),
            "url": url.strip(),
            "events": list(events),
            "agent_filter": list(agent_filter or []),
            "secret": secrets.token_hex(32),
            "enabled": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "delivery_count": 0,
            "last_delivered_at": None,
            "last_error": None,
        }
        self.subscriptions[sub_id] = sub
        self._save()
        logger.info("Added outbound webhook %s: name=%s url=%s", sub_id, name, url)
        return sub

    def remove_subscription(self, sub_id: str) -> bool:
        if sub_id not in self.subscriptions:
            return False
        del self.subscriptions[sub_id]
        self._save()
        logger.info("Removed outbound webhook %s", sub_id)
        return True

    def update_subscription(
        self,
        sub_id: str,
        *,
        name: str | None = None,
        url: str | None = None,
        events: list[str] | None = None,
        agent_filter: list[str] | None = None,
        enabled: bool | None = None,
    ) -> dict | None:
        """Partial update with validation-before-mutation."""
        sub = self.subscriptions.get(sub_id)
        if sub is None:
            return None

        # Validate before mutating
        if name is not None and not name.strip():
            raise ValueError("name must not be empty")
        if url is not None:
            parsed = urlparse(url.strip())
            if parsed.scheme != "https":
                raise ValueError("URL must use https")
        if events is not None:
            invalid = set(events) - VALID_EVENT_TYPES
            if invalid:
                raise ValueError(f"Invalid event types: {', '.join(sorted(invalid))}")
            if not events:
                raise ValueError("At least one event type is required")

        # Apply mutations
        if name is not None:
            sub["name"] = name.strip()
        if url is not None:
            sub["url"] = url.strip()
        if events is not None:
            sub["events"] = list(events)
        if agent_filter is not None:
            sub["agent_filter"] = list(agent_filter)
        if enabled is not None:
            sub["enabled"] = enabled

        self._save()
        logger.info("Updated outbound webhook %s", sub_id)
        return dict(sub)

    def list_subscriptions(self) -> list[dict]:
        return [dict(s) for s in self.subscriptions.values()]

    async def test_subscription(self, sub_id: str) -> dict | None:
        """Immediately deliver a test event (bypasses queue)."""
        sub = self.subscriptions.get(sub_id)
        if sub is None:
            return None

        delivery_id = generate_id("dlv")[:20]
        payload = {
            "event": "test",
            "agent": "system",
            "data": {"test": True, "message": "Test delivery from OpenLegion"},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "delivery_id": delivery_id,
        }
        result = await self._deliver(sub, payload, delivery_id)
        return result

    def enqueue(self, event_type: str, agent: str, data: dict) -> None:
        """Called by EventBus listener; filters and queues matching deliveries."""
        for sub in self.subscriptions.values():
            if not sub.get("enabled", True):
                continue
            if event_type not in sub.get("events", []):
                continue
            agent_filter = sub.get("agent_filter", [])
            if agent_filter and agent not in agent_filter:
                continue
            delivery_id = generate_id("dlv")[:20]
            item = {
                "subscription": sub,
                "payload": {
                    "event": event_type,
                    "agent": agent,
                    "data": data,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "delivery_id": delivery_id,
                },
                "delivery_id": delivery_id,
            }
            self._queue.put_nowait(item)

    def start(self) -> None:
        """Create the background delivery worker task."""
        self._stopping = False
        self._worker_task = asyncio.ensure_future(self._delivery_worker())
        logger.info("Outbound webhook delivery worker started")

    async def stop(self) -> None:
        """Signal worker to stop and wait for drain."""
        self._stopping = True
        # Put a sentinel to unblock the worker if it's waiting on get()
        self._queue.put_nowait(None)
        if self._worker_task is not None:
            try:
                await asyncio.wait_for(self._worker_task, timeout=30)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._worker_task.cancel()
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def recent_deliveries(self) -> list[dict]:
        return list(self._deliveries)

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )
        return self._client

    async def _delivery_worker(self) -> None:
        """Background worker that processes the delivery queue."""
        while not self._stopping:
            try:
                item = await self._queue.get()
            except asyncio.CancelledError:
                break
            if item is None:  # sentinel
                break
            sub = item["subscription"]
            payload = item["payload"]
            delivery_id = item["delivery_id"]
            await self._deliver(sub, payload, delivery_id)

    async def _deliver(
        self, sub: dict, payload: dict, delivery_id: str,
    ) -> dict:
        """POST payload to subscription URL with retry."""
        url = sub["url"]
        secret = sub["secret"]

        # SSRF check
        ssrf_error = _check_url_ssrf(url)
        if ssrf_error:
            error_msg = f"SSRF blocked: {ssrf_error}"
            logger.warning("Outbound webhook %s blocked: %s", sub["id"], error_msg)
            record = {
                "delivery_id": delivery_id,
                "subscription_id": sub["id"],
                "event": payload.get("event", ""),
                "agent": payload.get("agent", ""),
                "status": "failed",
                "response_code": None,
                "error": error_msg,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._deliveries.append(record)
            sub["last_error"] = error_msg
            self._save()
            return record

        body_bytes = json.dumps(payload).encode()
        signature = hmac.new(
            secret.encode(), body_bytes, hashlib.sha256,
        ).hexdigest()
        headers = {
            "Content-Type": "application/json",
            "X-OpenLegion-Signature": f"sha256={signature}",
            "X-OpenLegion-Event": payload.get("event", ""),
            "User-Agent": "OpenLegion/1.0",
        }

        last_error = None
        response_code = None
        for attempt in range(len(_RETRY_DELAYS) + 1):
            try:
                client = await self._get_client()
                resp = await client.post(
                    url, content=body_bytes, headers=headers,
                    timeout=_DELIVERY_TIMEOUT,
                )
                response_code = resp.status_code
                if 200 <= resp.status_code < 300:
                    # Success
                    sub["delivery_count"] = sub.get("delivery_count", 0) + 1
                    sub["last_delivered_at"] = datetime.now(timezone.utc).isoformat()
                    sub["last_error"] = None
                    self._save()
                    record = {
                        "delivery_id": delivery_id,
                        "subscription_id": sub["id"],
                        "event": payload.get("event", ""),
                        "agent": payload.get("agent", ""),
                        "status": "delivered",
                        "response_code": response_code,
                        "error": None,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    self._deliveries.append(record)
                    return record
                last_error = f"HTTP {resp.status_code}"
            except Exception as e:
                last_error = str(e)

            # Retry with backoff if we haven't exhausted attempts
            if attempt < len(_RETRY_DELAYS):
                await asyncio.sleep(_RETRY_DELAYS[attempt])

        # All retries exhausted
        error_msg = f"Failed after {len(_RETRY_DELAYS) + 1} attempts: {last_error}"
        sub["last_error"] = error_msg
        self._save()
        record = {
            "delivery_id": delivery_id,
            "subscription_id": sub["id"],
            "event": payload.get("event", ""),
            "agent": payload.get("agent", ""),
            "status": "failed",
            "response_code": response_code,
            "error": error_msg,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._deliveries.append(record)
        logger.warning(
            "Outbound webhook %s delivery failed: %s", sub["id"], error_msg,
        )
        return record
