"""In-memory event bus for real-time dashboard observability.

Broadcasts DashboardEvent instances to connected WebSocket clients.
Uses a ring buffer (deque) for ephemeral storage — no persistence needed.
Thread-safe emit() supports cross-thread calls from HealthMonitor/CronScheduler.
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.shared.types import DashboardEvent
from src.shared.utils import setup_logging

if TYPE_CHECKING:
    from fastapi import WebSocket

logger = setup_logging("dashboard.events")

BUFFER_SIZE = 500


@dataclass
class _Subscription:
    """Holds a WebSocket and optional filters for event matching."""

    ws: WebSocket
    agents: set[str] = field(default_factory=set)
    types: set[str] = field(default_factory=set)

    def matches(self, evt: dict) -> bool:
        if self.types and evt.get("type") not in self.types:
            return False
        if self.agents:
            evt_agent = evt.get("agent", "")
            if evt_agent and evt_agent not in self.agents:
                return False
        return True


class EventBus:
    """Broadcasts system events to WebSocket subscribers."""

    def __init__(self) -> None:
        self._buffer: deque[dict] = deque(maxlen=BUFFER_SIZE)
        self._clients: list[_Subscription] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._seq: int = 0  # monotonic sequence counter

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind to the mesh server's event loop. Idempotent."""
        self._loop = loop

    def emit(self, event_type: str, agent: str = "", data: dict[str, Any] | None = None) -> None:
        """Thread-safe event emission.

        Builds a DashboardEvent, appends to the ring buffer, and schedules
        broadcast to all matching WebSocket subscribers.

        Safe to call from any thread: uses call_soon_threadsafe when the
        caller is not on the target event loop.
        """
        evt = DashboardEvent(type=event_type, agent=agent, data=data or {})
        evt_dict = evt.model_dump(mode="json")
        self._seq += 1
        evt_dict["_seq"] = self._seq
        self._buffer.append(evt_dict)

        if not self._clients:
            return

        # Auto-bind loop on first emit from an async context
        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                return  # no loop available — skip broadcast

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        coro = self._broadcast(evt_dict)
        if running_loop is self._loop:
            asyncio.ensure_future(coro)
        else:
            self._loop.call_soon_threadsafe(asyncio.ensure_future, coro)

    def subscribe(
        self,
        ws: WebSocket,
        agents_filter: set[str] | None = None,
        types_filter: set[str] | None = None,
    ) -> None:
        self._clients.append(_Subscription(
            ws=ws,
            agents=agents_filter or set(),
            types=types_filter or set(),
        ))

    def unsubscribe(self, ws: WebSocket) -> None:
        self._clients = [c for c in self._clients if c.ws is not ws]

    @property
    def current_seq(self) -> int:
        """Return the current sequence number (for subscribe-before-replay)."""
        return self._seq

    def recent_events(
        self,
        agents_filter: set[str] | None = None,
        types_filter: set[str] | None = None,
        before_seq: int | None = None,
    ) -> list[dict]:
        """Return buffered events, optionally filtered.

        If *before_seq* is given, only return events with _seq <= before_seq.
        This prevents replaying events that will also arrive via the live feed.
        """
        sub = _Subscription(ws=None, agents=agents_filter or set(), types=types_filter or set())  # type: ignore[arg-type]
        result = []
        for e in self._buffer:
            if before_seq is not None and e.get("_seq", 0) > before_seq:
                break
            if agents_filter or types_filter:
                if not sub.matches(e):
                    continue
            result.append(e)
        return result

    async def _broadcast(self, evt_dict: dict) -> None:
        """Send event to all matching subscribers. Remove dead connections."""
        dead: list[_Subscription] = []
        payload = json.dumps(evt_dict, default=str)
        for sub in self._clients:
            if not sub.matches(evt_dict):
                continue
            try:
                await sub.ws.send_text(payload)
            except Exception:
                dead.append(sub)
        for d in dead:
            self._clients.remove(d)
