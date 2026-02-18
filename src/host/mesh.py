"""Mesh layer: Blackboard, PubSub, and MessageRouter.

The mesh is the central nervous system of OpenLegion.
It routes messages, manages shared state, runs pub/sub,
and enforces permissions.
"""

from __future__ import annotations

import json
import sqlite3
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

    def __init__(self, db_path: str = "blackboard.db"):
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=5000")
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

        self.db.execute(
            "INSERT INTO entries (key, value, written_by, workflow_id, ttl, version) "
            "VALUES (?, ?, ?, ?, ?, 1) "
            "ON CONFLICT(key) DO UPDATE SET "
            "value = excluded.value, written_by = excluded.written_by, "
            "workflow_id = excluded.workflow_id, updated_at = datetime('now'), "
            "ttl = excluded.ttl, version = version + 1",
            (key, value_json, written_by, workflow_id, ttl),
        )

        row = self.db.execute("SELECT version FROM entries WHERE key = ?", (key,)).fetchone()
        new_version = row[0] if row else 1

        self._log_event("write", key, written_by, value_json)
        self.db.commit()

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
        rows = self.db.execute(
            "SELECT key, value, written_by, workflow_id, "
            "created_at, updated_at, ttl, version FROM entries "
            "WHERE key LIKE ? ORDER BY key",
            (prefix + "%",),
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
        self.db.execute("DELETE FROM entries WHERE key = ?", (key,))
        self._log_event("delete", key, deleted_by)
        self.db.commit()

    def gc_expired(self) -> int:
        """Garbage-collect entries that have exceeded their TTL. Returns count deleted."""
        cursor = self.db.execute(
            "DELETE FROM entries WHERE ttl IS NOT NULL AND "
            "datetime(updated_at, '+' || ttl || ' seconds') < datetime('now') "
            "AND key NOT LIKE 'history/%'"
        )
        self.db.commit()
        return cursor.rowcount

    def _log_event(self, event_type: str, key: str, agent_id: str, data: str | None = None) -> None:
        self.db.execute(
            "INSERT INTO event_log (event_type, key, agent_id, data) VALUES (?, ?, ?, ?)",
            (event_type, key, agent_id, data),
        )

    def close(self) -> None:
        self.db.close()


class PubSub:
    """Simple in-process pub/sub for agent events.

    Agents subscribe to topics. When an event is published,
    all subscribers are notified via their container HTTP API.
    """

    def __init__(self) -> None:
        self.subscriptions: dict[str, list[str]] = {}
        self.event_log: list[dict] = []

    def subscribe(self, topic: str, agent_id: str) -> None:
        if topic not in self.subscriptions:
            self.subscriptions[topic] = []
        if agent_id not in self.subscriptions[topic]:
            self.subscriptions[topic].append(agent_id)

    def unsubscribe(self, topic: str, agent_id: str) -> None:
        if topic in self.subscriptions:
            self.subscriptions[topic] = [a for a in self.subscriptions[topic] if a != agent_id]

    def get_subscribers(self, topic: str) -> list[str]:
        return self.subscriptions.get(topic, [])

    def publish(self, topic: str, event: Any) -> list[str]:
        """Record an event and return the list of subscribers for it."""
        self.event_log.append({"topic": topic, "event": event})
        return self.get_subscribers(topic)


class MessageRouter:
    """Routes messages between agents with permission enforcement."""

    def __init__(self, permissions: PermissionMatrix, agent_registry: dict[str, str]):
        self.permissions = permissions
        self.agent_registry: dict[str, str] = agent_registry
        self.message_log: list[dict] = []
        self._capabilities_cache: dict[str, list[str]] = {}

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
        if len(self.message_log) > 10_000:
            self.message_log = self.message_log[-5_000:]

        try:
            async with httpx.AsyncClient(timeout=message.ttl) as client:
                response = await client.post(
                    f"{target_url}/message",
                    json=message.model_dump(mode="json"),
                )
                return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Failed to route message to {message.to}: {e}")
            return {"error": f"Delivery failed: {e}"}

    def _resolve_target(self, target: str) -> Optional[str]:
        """Resolve agent ID or capability to a container URL."""
        if target in self.agent_registry:
            return self.agent_registry[target]

        if target.startswith("capability:"):
            capability = target.split(":", 1)[1]
            for agent_id, url in self.agent_registry.items():
                if capability in self._capabilities_cache.get(agent_id, []):
                    return url

        return None

    def register_agent(self, agent_id: str, url: str, capabilities: list[str] | None = None) -> None:
        """Register an agent and its capabilities."""
        self.agent_registry[agent_id] = url
        if capabilities:
            self._capabilities_cache[agent_id] = capabilities

    def unregister_agent(self, agent_id: str) -> None:
        """Remove an agent from the registry."""
        self.agent_registry.pop(agent_id, None)
        self._capabilities_cache.pop(agent_id, None)
