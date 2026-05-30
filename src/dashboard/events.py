"""In-memory event bus for real-time dashboard observability.

Broadcasts DashboardEvent instances to connected WebSocket clients.
Uses a ring buffer (deque) for ephemeral storage — no persistence needed.
Thread-safe emit() supports cross-thread calls from HealthMonitor/CronScheduler.
"""

from __future__ import annotations

import asyncio
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.shared.types import DashboardEvent
from src.shared.utils import dumps_safe, setup_logging

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
    # Serializes sends to this one WebSocket. Concurrent _broadcast tasks
    # (one per emitted event) must not interleave frames on the same
    # connection — Starlette's send is not safe under concurrent callers.
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

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
        self._lock = threading.Lock()
        # In-process listeners — invoked synchronously from emit() so the
        # dashboard's own aggregators (e.g. per-platform success rollup) can
        # observe events without going through a WebSocket. Listeners must
        # be cheap and non-blocking; exceptions are swallowed to keep emit
        # robust against a buggy aggregator.
        self._listeners: list[Callable[[dict], None]] = []
        # Strong refs to in-flight fire-and-forget broadcast tasks. Without
        # this, asyncio only keeps a weak reference and the task may be
        # garbage-collected mid-send. Each task discards itself on done.
        self._broadcast_tasks: set[asyncio.Task] = set()

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
        with self._lock:
            self._seq += 1
            evt_dict["_seq"] = self._seq
            self._buffer.append(evt_dict)
            listeners = list(self._listeners)
            has_clients = bool(self._clients)

        # In-process listeners run synchronously on the caller's stack —
        # the dashboard aggregators are O(1) per event so this is cheap.
        # Catch every exception so a single misbehaving listener cannot
        # break the broadcast path or starve other listeners.
        for cb in listeners:
            try:
                cb(evt_dict)
            except Exception as e:
                logger.debug("EventBus listener raised: %s", e)

        if not has_clients:
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

        if running_loop is self._loop:
            self._spawn_broadcast(evt_dict)
        else:
            self._loop.call_soon_threadsafe(self._spawn_broadcast, evt_dict)

    def _spawn_broadcast(self, evt_dict: dict) -> None:
        """Schedule a broadcast task and keep a strong ref to it.

        Runs on the bound event loop. Tracking the task in a set (and
        discarding on completion) prevents the GC from collecting an
        in-flight broadcast — asyncio only holds a weak reference.
        """
        task = asyncio.ensure_future(self._broadcast(evt_dict))
        self._broadcast_tasks.add(task)
        task.add_done_callback(self._broadcast_tasks.discard)

    def subscribe(
        self,
        ws: WebSocket,
        agents_filter: set[str] | None = None,
        types_filter: set[str] | None = None,
    ) -> None:
        sub = _Subscription(
            ws=ws,
            agents=agents_filter or set(),
            types=types_filter or set(),
        )
        with self._lock:
            self._clients.append(sub)

    def unsubscribe(self, ws: WebSocket) -> None:
        with self._lock:
            self._clients = [c for c in self._clients if c.ws is not ws]

    def add_listener(self, cb: Callable[[dict], None]) -> None:
        """Register an in-process callback invoked synchronously from emit().

        Used by the dashboard's per-platform success aggregator (and any
        future module-level rollup) so it can observe events without
        masquerading as a WebSocket subscriber. Idempotent — adding the
        same callback twice will only call it twice (no dedupe).
        """
        with self._lock:
            self._listeners.append(cb)

    def remove_listener(self, cb: Callable[[dict], None]) -> None:
        """Remove a previously-registered listener.  No-op if absent.

        Uses equality (``==``) rather than identity so a bound-method
        round-trip (``agg.handle_event`` evaluates to a fresh
        ``MethodType`` each access) still matches the registered entry.
        """
        with self._lock:
            self._listeners = [c for c in self._listeners if c != cb]

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
            if (agents_filter or types_filter) and not sub.matches(e):
                continue
            result.append(e)
        return result

    async def _broadcast(self, evt_dict: dict) -> None:
        """Send event to all matching subscribers. Remove dead connections."""
        dead: list[_Subscription] = []
        payload = dumps_safe(evt_dict)
        # Snapshot under the lock so a concurrent subscribe/unsubscribe (or
        # another _broadcast's dead-removal) can't mutate the list mid-iter
        # and silently drop a live client.
        with self._lock:
            subscribers = list(self._clients)
        for sub in subscribers:
            if not sub.matches(evt_dict):
                continue
            try:
                # Per-connection lock: never interleave two events' frames
                # on the same WebSocket.
                async with sub.send_lock:
                    await sub.ws.send_text(payload)
            except Exception as e:
                logger.debug("Event send to WebSocket failed (client disconnected?): %s", e)
                dead.append(sub)
        if dead:
            with self._lock:
                self._clients = [c for c in self._clients if c not in dead]
