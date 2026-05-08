"""Persistent dashboard notifications.

Phase 2 of the Board UX overhaul (`docs/plans/2026-05-08-board-ux-overhaul.md`).
The notifications bell in the top-right corner of the dashboard pulls from a
persistent SQLite-backed store so past events (delivered work, approvals,
alerts) survive page reloads and cross-device viewing.

Distinct from transient toasts (which are in-memory queues for "I just did X")
and from the Needs-You badge (which surfaces *currently-actionable* items).
A notification represents a past event the user should know about.

Schema::

    dashboard_notifications(
        id          INTEGER PK,
        agent_id    TEXT,                -- optional originating agent (NULL = system)
        ts          REAL,                -- Unix epoch seconds (when event occurred)
        kind        TEXT NOT NULL,       -- short tag (delivered / approval / alert / info)
        title       TEXT NOT NULL,       -- one-line headline
        body        TEXT,                -- optional longer body
        read_at     REAL,                -- Unix epoch when read (NULL = unread)
        payload_json TEXT                -- optional JSON blob for click-through targets
    )

Reads return up to 10 rows ordered ``unread first, ts DESC``. Reads paginate
via offset; the dashboard only ever reads the first page.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from src.shared.utils import setup_logging

logger = setup_logging("dashboard.notifications")


_DEFAULT_LIMIT = 10
_MAX_LIMIT = 100
# Frozen kind allowlist — keeps the wire schema stable. Add a new value here
# rather than coining ad-hoc kinds; the bell renders an icon per kind on the
# client side.
_KNOWN_KINDS = frozenset({"delivered", "approval", "alert", "info", "blocker", "credential"})


class NotificationStore:
    """Persistent notifications backed by a small SQLite table.

    Single connection, WAL mode, ``check_same_thread=False`` so the FastAPI
    request handlers (which run on the asyncio loop's default executor) can
    share it. Mirror the conventions used by ``CostTracker`` and
    ``PendingActions``.
    """

    def __init__(self, db_path: str = "data/dashboard_notifications.db") -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=30000")
        self._init_schema()

    def _init_schema(self) -> None:
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS dashboard_notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT,
                ts REAL NOT NULL,
                kind TEXT NOT NULL,
                title TEXT NOT NULL,
                body TEXT,
                read_at REAL,
                payload_json TEXT
            );
            -- Composite index optimises the "unread first, then by ts DESC" read.
            CREATE INDEX IF NOT EXISTS idx_notifications_unread_ts
                ON dashboard_notifications(read_at, ts DESC);
            """,
        )
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    # ── Writes ───────────────────────────────────────────────

    def add(
        self,
        *,
        kind: str,
        title: str,
        body: str | None = None,
        agent_id: str | None = None,
        payload: dict[str, Any] | None = None,
        ts: float | None = None,
    ) -> int:
        """Insert a notification. Returns the new row id.

        Unknown kinds are accepted but logged so the bell still renders;
        coining a new kind on the fly is cheaper than dropping a real event.
        """
        if not title:
            raise ValueError("title must not be empty")
        if not kind:
            raise ValueError("kind must not be empty")
        if kind not in _KNOWN_KINDS:
            logger.info("Notification with unknown kind %r — accepting but icon may be generic", kind)
        ts = ts if ts is not None else time.time()
        payload_json = json.dumps(payload) if payload else None
        cursor = self.db.execute(
            """
            INSERT INTO dashboard_notifications
                (agent_id, ts, kind, title, body, read_at, payload_json)
            VALUES (?, ?, ?, ?, ?, NULL, ?)
            """,
            (agent_id, ts, kind, title, body, payload_json),
        )
        self.db.commit()
        return int(cursor.lastrowid or 0)

    def mark_read(self, notification_id: int) -> bool:
        """Mark a single notification as read. Returns True if a row changed."""
        now = time.time()
        cursor = self.db.execute(
            "UPDATE dashboard_notifications SET read_at = ? "
            "WHERE id = ? AND read_at IS NULL",
            (now, notification_id),
        )
        self.db.commit()
        return cursor.rowcount > 0

    def mark_all_read(self) -> int:
        """Mark every unread notification as read. Returns count updated."""
        now = time.time()
        cursor = self.db.execute(
            "UPDATE dashboard_notifications SET read_at = ? WHERE read_at IS NULL",
            (now,),
        )
        self.db.commit()
        return int(cursor.rowcount or 0)

    # ── Reads ────────────────────────────────────────────────

    def list_recent(self, limit: int = _DEFAULT_LIMIT) -> list[dict[str, Any]]:
        """Return up to ``limit`` notifications, unread first then by ts DESC.

        Returns dicts shaped for direct JSON response; callers should not
        mutate. Limit is clamped to ``[1, 100]``.
        """
        if limit < 1:
            limit = 1
        if limit > _MAX_LIMIT:
            limit = _MAX_LIMIT
        rows = self.db.execute(
            """
            SELECT id, agent_id, ts, kind, title, body, read_at, payload_json
            FROM dashboard_notifications
            ORDER BY (read_at IS NOT NULL) ASC, ts DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            payload = None
            if row[7]:
                try:
                    payload = json.loads(row[7])
                except json.JSONDecodeError:
                    payload = None
            out.append(
                {
                    "id": int(row[0]),
                    "agent_id": row[1],
                    "ts": float(row[2]) if row[2] is not None else None,
                    "kind": row[3],
                    "title": row[4],
                    "body": row[5],
                    "read_at": float(row[6]) if row[6] is not None else None,
                    "payload": payload,
                },
            )
        return out

    def unread_count(self) -> int:
        row = self.db.execute(
            "SELECT COUNT(1) FROM dashboard_notifications WHERE read_at IS NULL",
        ).fetchone()
        return int(row[0]) if row else 0
