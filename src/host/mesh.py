"""Mesh layer: Blackboard, PubSub, and MessageRouter.

The mesh is the central nervous system of OpenLegion.
It routes messages, manages shared state, runs pub/sub,
and enforces permissions.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from collections import deque
from typing import TYPE_CHECKING, Any, Optional

import httpx

from src.shared.types import AgentMessage, BlackboardEntry
from src.shared.utils import setup_logging

if TYPE_CHECKING:
    from src.host.permissions import PermissionMatrix

logger = setup_logging("host.mesh")


class Blackboard:
    """Shared state store accessed by agents through the mesh.

    Hierarchical key structure:
      tasks/     -- Active task states
      context/   -- Shared knowledge (prospect profiles, campaign data)
      signals/   -- Cross-agent flags and alerts
      history/   -- Completed results (append-only, no deletes)

    Uses SQLite WAL mode for concurrent reads.
    """

    def __init__(self, db_path: str = "blackboard.db", event_bus=None):
        import threading
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=30000")
        self._write_lock = threading.Lock()
        self._event_bus = event_bus
        self._last_ttl_gc: float = 0.0
        self._init_schema()

    def _init_schema(self) -> None:
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS entries (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                written_by TEXT NOT NULL,
                workflow_id TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                ttl INTEGER,
                version INTEGER DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS event_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                key TEXT,
                agent_id TEXT,
                data TEXT,
                timestamp TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_entries_prefix ON entries(key);
            CREATE INDEX IF NOT EXISTS idx_event_log_timestamp ON event_log(timestamp);
        """)
        self.db.commit()

    def write(
        self,
        key: str,
        value: dict,
        written_by: str,
        workflow_id: str | None = None,
        ttl: int | None = None,
    ) -> BlackboardEntry:
        """Write or update a blackboard entry (atomic upsert)."""
        value_json = json.dumps(value, default=str)

        with self._write_lock:
            cursor = self.db.execute(
                "INSERT INTO entries (key, value, written_by, workflow_id, ttl, version) "
                "VALUES (?, ?, ?, ?, ?, 1) "
                "ON CONFLICT(key) DO UPDATE SET "
                "value = excluded.value, written_by = excluded.written_by, "
                "workflow_id = excluded.workflow_id, updated_at = datetime('now'), "
                "ttl = excluded.ttl, version = version + 1 "
                "RETURNING version",
                (key, value_json, written_by, workflow_id, ttl),
            )
            row = cursor.fetchone()
            new_version = row[0] if row else 1

            self._log_event("write", key, written_by, value_json)
            self.db.commit()
        self._maybe_gc_event_log()
        self._maybe_gc_ttl()

        if self._event_bus:
            # Truncate value preview for dashboard display
            preview = value_json[:200] if len(value_json) > 200 else value_json
            self._event_bus.emit("blackboard_write", agent=written_by,
                data={"key": key, "version": new_version, "value_preview": preview,
                      "written_by": written_by})

        return BlackboardEntry(
            key=key,
            value=value,
            written_by=written_by,
            workflow_id=workflow_id,
            ttl=ttl,
            version=new_version,
        )

    def read(self, key: str) -> Optional[BlackboardEntry]:
        """Read a single entry by exact key."""
        row = self.db.execute(
            "SELECT key, value, written_by, workflow_id, "
            "created_at, updated_at, ttl, version FROM entries WHERE key = ?",
            (key,),
        ).fetchone()
        if not row:
            return None
        return BlackboardEntry(
            key=row[0],
            value=json.loads(row[1]),
            written_by=row[2],
            workflow_id=row[3],
            created_at=row[4],
            updated_at=row[5],
            ttl=row[6],
            version=row[7],
        )

    def list_by_prefix(self, prefix: str) -> list[BlackboardEntry]:
        """List all entries matching a key prefix."""
        # Escape LIKE wildcards so literal '%' and '_' in keys match exactly
        escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        rows = self.db.execute(
            "SELECT key, value, written_by, workflow_id, "
            "created_at, updated_at, ttl, version FROM entries "
            "WHERE key LIKE ? ESCAPE '\\' ORDER BY key",
            (escaped + "%",),
        ).fetchall()
        return [
            BlackboardEntry(
                key=r[0],
                value=json.loads(r[1]),
                written_by=r[2],
                workflow_id=r[3],
                created_at=r[4],
                updated_at=r[5],
                ttl=r[6],
                version=r[7],
            )
            for r in rows
        ]

    def delete(self, key: str, deleted_by: str) -> None:
        """Delete an entry (NOT allowed for history/ namespace)."""
        if key.startswith("history/"):
            raise ValueError("Cannot delete from history namespace")
        with self._write_lock:
            self.db.execute("DELETE FROM entries WHERE key = ?", (key,))
            self._log_event("delete", key, deleted_by)
            self.db.commit()
        self._maybe_gc_event_log()
        self._maybe_gc_ttl()

    def gc_expired(self) -> int:
        """Garbage-collect entries that have exceeded their TTL. Returns count deleted."""
        with self._write_lock:
            cursor = self.db.execute(
                "DELETE FROM entries WHERE ttl IS NOT NULL AND "
                "datetime(updated_at, '+' || ttl || ' seconds') < datetime('now') "
                "AND key NOT LIKE 'history/%'"
            )
            self.db.commit()
        return cursor.rowcount

    _EVENT_LOG_GC_THRESHOLD = 10_000
    _EVENT_LOG_GC_KEEP = 5_000
    _TTL_GC_INTERVAL = 60  # seconds between TTL garbage collection runs

    def _log_event(self, event_type: str, key: str, agent_id: str, data: str | None = None) -> None:
        self.db.execute(
            "INSERT INTO event_log (event_type, key, agent_id, data) VALUES (?, ?, ?, ?)",
            (event_type, key, agent_id, data),
        )

    def _maybe_gc_event_log(self) -> None:
        """Garbage-collect old event_log rows when exceeding threshold.

        Must be called AFTER the caller's commit() so the GC DELETE
        doesn't accidentally commit unrelated pending writes.
        """
        with self._write_lock:
            count = self.db.execute("SELECT COUNT(*) FROM event_log").fetchone()[0]
            if count > self._EVENT_LOG_GC_THRESHOLD:
                self.db.execute(
                    "DELETE FROM event_log WHERE id NOT IN "
                    "(SELECT id FROM event_log ORDER BY id DESC LIMIT ?)",
                    (self._EVENT_LOG_GC_KEEP,),
                )
                self.db.commit()

    def _maybe_gc_ttl(self) -> None:
        """Garbage-collect TTL-expired entries, throttled to once per 60s.

        Must be called AFTER the caller's commit() so the GC DELETE
        doesn't accidentally commit unrelated pending writes.
        """
        now = time.monotonic()
        with self._write_lock:
            if now - self._last_ttl_gc < self._TTL_GC_INTERVAL:
                return
            self._last_ttl_gc = now
        self.gc_expired()

    def close(self) -> None:
        self.db.close()


class PubSub:
    """In-process pub/sub for agent events with optional SQLite persistence.

    Agents subscribe to topics. When an event is published,
    all subscribers are notified via their container HTTP API.

    When ``db_path`` is provided, subscriptions and events are persisted
    to SQLite so they survive restarts.  Without ``db_path``, behaviour
    is identical to the original in-memory-only implementation.
    """

    _EVENT_GC_THRESHOLD = 10_000
    _EVENT_GC_KEEP = 5_000

    def __init__(self, db_path: str | None = None) -> None:
        import threading
        self.subscriptions: dict[str, list[str]] = {}
        self.event_log: deque[dict] = deque(maxlen=self._EVENT_GC_KEEP)
        self._db: sqlite3.Connection | None = None
        self._lock = threading.Lock()

        if db_path is not None:
            self._db = sqlite3.connect(db_path, check_same_thread=False)
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA busy_timeout=30000")
            self._init_schema()
            self._load_subscriptions()
            self._load_events()

    def _init_schema(self) -> None:
        assert self._db is not None
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS subscriptions (
                topic TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                PRIMARY KEY (topic, agent_id)
            );
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                data TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_events_topic ON events(topic);
        """)
        self._db.commit()

    def _load_subscriptions(self) -> None:
        assert self._db is not None
        rows = self._db.execute("SELECT topic, agent_id FROM subscriptions").fetchall()
        for topic, agent_id in rows:
            self.subscriptions.setdefault(topic, [])
            if agent_id not in self.subscriptions[topic]:
                self.subscriptions[topic].append(agent_id)

    def _load_events(self) -> None:
        assert self._db is not None
        rows = self._db.execute("SELECT topic, data FROM events ORDER BY id").fetchall()
        for topic, data in rows:
            try:
                event = json.loads(data)
            except (json.JSONDecodeError, TypeError):
                event = data
            self.event_log.append({"topic": topic, "event": event})

    def subscribe(self, topic: str, agent_id: str) -> None:
        with self._lock:
            if topic not in self.subscriptions:
                self.subscriptions[topic] = []
            if agent_id not in self.subscriptions[topic]:
                self.subscriptions[topic].append(agent_id)
                if self._db is not None:
                    self._db.execute(
                        "INSERT OR IGNORE INTO subscriptions (topic, agent_id) VALUES (?, ?)",
                        (topic, agent_id),
                    )
                    self._db.commit()

    def unsubscribe(self, topic: str, agent_id: str) -> None:
        with self._lock:
            if topic in self.subscriptions:
                self.subscriptions[topic] = [a for a in self.subscriptions[topic] if a != agent_id]
                if self._db is not None:
                    self._db.execute(
                        "DELETE FROM subscriptions WHERE topic = ? AND agent_id = ?",
                        (topic, agent_id),
                    )
                    self._db.commit()

    def unsubscribe_agent(self, agent_id: str) -> None:
        """Remove an agent from ALL topic subscriptions."""
        with self._lock:
            for topic in list(self.subscriptions):
                self.subscriptions[topic] = [a for a in self.subscriptions[topic] if a != agent_id]
                if not self.subscriptions[topic]:
                    del self.subscriptions[topic]
            if self._db is not None:
                self._db.execute("DELETE FROM subscriptions WHERE agent_id = ?", (agent_id,))
                self._db.commit()

    def get_subscribers(self, topic: str) -> list[str]:
        with self._lock:
            return list(self.subscriptions.get(topic, []))

    def publish(self, topic: str, event: Any) -> list[str]:
        """Record an event and return the list of subscribers for it."""
        with self._lock:
            self.event_log.append({"topic": topic, "event": event})
            if self._db is not None:
                self._db.execute(
                    "INSERT INTO events (topic, data) VALUES (?, ?)",
                    (topic, json.dumps(event, default=str)),
                )
                self._db.commit()
                self._maybe_gc_events()
            subscribers = list(self.subscriptions.get(topic, []))
        return subscribers

    def _maybe_gc_events(self) -> None:
        """Garbage-collect old events when the table exceeds the threshold.

        Caller must hold ``self._lock``.
        """
        if self._db is None:
            return
        count = self._db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        if count > self._EVENT_GC_THRESHOLD:
            self._db.execute(
                "DELETE FROM events WHERE id NOT IN "
                "(SELECT id FROM events ORDER BY id DESC LIMIT ?)",
                (self._EVENT_GC_KEEP,),
            )
            self._db.commit()

    def close(self) -> None:
        if self._db is not None:
            self._db.close()


class MessageRouter:
    """Routes messages between agents with permission enforcement."""

    def __init__(
        self,
        permissions: PermissionMatrix,
        agent_registry: dict[str, str],
        trace_store: Any = None,
    ):
        import threading
        self.permissions = permissions
        self.agent_registry: dict[str, str] = agent_registry
        self.agent_roles: dict[str, str] = {}
        self.message_log: deque[dict] = deque(maxlen=10_000)
        self._capabilities_cache: dict[str, list[str]] = {}
        self._registry_lock = threading.Lock()
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()
        self._trace_store = trace_store

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is not None and not self._client.is_closed:
            return self._client
        async with self._client_lock:
            if self._client is None or self._client.is_closed:
                self._client = httpx.AsyncClient(timeout=30)
            return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    async def route(self, message: AgentMessage) -> dict:
        """Route a message. Resolves capability-based addressing. Enforces permissions."""
        if not self.permissions.can_message(message.from_agent, message.to):
            logger.warning(f"Permission denied: {message.from_agent} -> {message.to}")
            return {"error": f"{message.from_agent} cannot message {message.to}"}

        target_url = self._resolve_target(message.to)
        if not target_url:
            return {"error": f"No agent found for target: {message.to}"}

        self.message_log.append({
            "id": message.id,
            "from": message.from_agent,
            "to": message.to,
            "type": message.type,
            "timestamp": message.timestamp.isoformat(),
        })
        if self._trace_store:
            from src.shared.trace import current_trace_id
            tid = current_trace_id.get()
            if tid:
                self._trace_store.record(
                    trace_id=tid, source="router", agent=message.from_agent,
                    event_type="message_route",
                    detail=f"{message.from_agent}->{message.to} ({message.type})",
                )
        payload = message.model_dump(mode="json")
        last_err: Exception | None = None
        for attempt in range(2):  # 1 attempt + 1 retry
            try:
                client = await self._get_client()
                response = await client.post(
                    f"{target_url}/message",
                    json=payload,
                    timeout=message.ttl,
                )
                response.raise_for_status()
                return response.json()
            except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                last_err = e
                if attempt == 0:
                    logger.warning(
                        f"Transient error routing to {message.to}, retrying in 1s: {e}"
                    )
                    await asyncio.sleep(1)
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (502, 503, 504) and attempt == 0:
                    last_err = e
                    logger.warning(
                        f"Agent {message.to} returned HTTP {e.response.status_code}, retrying in 1s"
                    )
                    await asyncio.sleep(1)
                    continue
                logger.error(f"Failed to route message to {message.to}: HTTP {e.response.status_code}")
                return {"error": f"Delivery failed: HTTP {e.response.status_code}"}
            except httpx.HTTPError as e:
                logger.error(f"Failed to route message to {message.to}: {e}")
                return {"error": f"Delivery failed: {e}"}
        logger.error(f"Failed to route message to {message.to} after retry: {last_err}")
        return {"error": f"Delivery failed after retry: {last_err}"}

    def _resolve_target(self, target: str) -> Optional[str]:
        """Resolve agent ID or capability to a container URL."""
        with self._registry_lock:
            if target in self.agent_registry:
                return self.agent_registry[target]

            if target.startswith("capability:"):
                capability = target.split(":", 1)[1]
                for agent_id, url in self.agent_registry.items():
                    if capability in self._capabilities_cache.get(agent_id, []):
                        return url

        return None

    def register_agent(
        self, agent_id: str, url: str,
        capabilities: list[str] | None = None,
        role: str = "",
    ) -> None:
        """Register an agent and its capabilities."""
        with self._registry_lock:
            self.agent_registry[agent_id] = url
            if role:
                self.agent_roles[agent_id] = role
            if capabilities:
                self._capabilities_cache[agent_id] = capabilities
            else:
                self._capabilities_cache.pop(agent_id, None)

    def unregister_agent(self, agent_id: str) -> None:
        """Remove an agent from the registry."""
        with self._registry_lock:
            self.agent_registry.pop(agent_id, None)
            self._capabilities_cache.pop(agent_id, None)
            self.agent_roles.pop(agent_id, None)
