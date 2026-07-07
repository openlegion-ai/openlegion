"""User-notification observation log — durable record of agent→user pushes.

When an agent calls ``notify_user(message)`` the mesh fans the message
out to the human's chat/channels and emits a live dashboard event, but
historically persisted nothing. So when the human later asks the
operator agent "what's blocking?", the operator was blind to what its
workers had already told the human.

This module fixes that with an OBSERVATION LOG — emphatically *not* a
message channel:

- Agents are UNTRUSTED. The operator is TRUSTED (a delegated user) and
  can edit/spawn/route agents. Agents must have NO channel to address
  the operator directly — that would invert the trust hierarchy and
  open a prompt-injection path into the privileged agent. So agents
  never address the operator here; they call ``notify_user`` (intent:
  tell the human) and the mesh *incidentally* logs the push.
- PULL, not push. Writing a row NEVER wakes the operator, NEVER creates
  an inbox item, NEVER creates a task. The operator only sees entries
  when it explicitly calls its ``read_user_notifications`` tool.
- This is distinct from ``check_inbox`` — that is the operator's
  addressable task lane. These rows are observed agent→user traffic,
  surfaced ``display_only`` and sanitized at the read-tool boundary.

The store keeps RAW messages (no sanitization here); sanitization
happens at the operator tool boundary so the endpoint stays a thin
data layer. Rows older than 7 days are reaped opportunistically on
write, mirroring the retention pattern in ``summaries.py``.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from src.shared.utils import setup_logging

logger = setup_logging("host.user_notifications")

# Retention window — rows older than this are reaped on the next write.
# 7 days matches ``threads.EVENT_ACTIONABLE_WINDOW_SECONDS`` — the
# back-edge event serving window (the other "diagnostic breadcrumb"
# surface the operator reads).
RETENTION_SECONDS = 7 * 86400


class UserNotificationLog:
    """SQLite-backed observation log of agent→user notifications.

    Schema-managed at construction. WAL + 30s busy timeout matches the
    mesh's other SQLite stores (``WorkSummariesStore`` etc.). Reaping is
    opportunistic on :meth:`record` so the table stays bounded without a
    dedicated sweeper task.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        # In-memory mode keeps a single shared connection so the schema
        # doesn't disappear between calls. Disk-backed mode opens a
        # fresh connection per operation (matches WorkSummariesStore).
        self._shared_conn: sqlite3.Connection | None = None
        if db_path == ":memory:":
            self._shared_conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()

    def close(self) -> None:
        """Release the in-memory connection (no-op for disk-backed)."""
        if self._shared_conn is not None:
            self._shared_conn.close()
            self._shared_conn = None

    def _conn(self) -> sqlite3.Connection:
        """Yield a connection. In-memory uses the shared one; disk opens
        fresh and the caller closes it (matches WorkSummariesStore).
        """
        if self._shared_conn is not None:
            return self._shared_conn
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self) -> None:
        conn = self._conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS user_notification_log (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts        REAL NOT NULL,
                    agent_id  TEXT NOT NULL,
                    message   TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_user_notification_log_ts
                    ON user_notification_log(ts);
            """)
            conn.commit()
        finally:
            if self._shared_conn is None:
                conn.close()

    # ----------------------------------------------------------------- RECORD
    def record(self, agent_id: str, message: str, *, now: float | None = None) -> None:
        """Append one observed agent→user notification.

        Stores the RAW message — sanitization happens at the read-tool
        boundary, not here. Reaps rows older than ``RETENTION_SECONDS``
        on each write so the table stays bounded.
        """
        ts = time.time() if now is None else now
        conn = self._conn()
        try:
            conn.execute(
                "INSERT INTO user_notification_log (ts, agent_id, message) "
                "VALUES (?, ?, ?)",
                (ts, agent_id, message),
            )
            # Opportunistic reap of expired rows on the write path.
            conn.execute(
                "DELETE FROM user_notification_log WHERE ts < ?",
                (ts - RETENTION_SECONDS,),
            )
            conn.commit()
        finally:
            if self._shared_conn is None:
                conn.close()

    # ----------------------------------------------------------------- RECENT
    def recent(self, hours: float = 24, limit: int = 50) -> list[dict]:
        """Return recent notifications, newest first.

        Filters to rows with ``ts >= now - hours*3600``. Returns RAW
        messages — the caller (operator tool) sanitizes at its boundary.
        """
        hours = max(0.0, float(hours))
        limit = max(1, min(int(limit), 500))
        cutoff = time.time() - hours * 3600
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT agent_id, message, ts FROM user_notification_log "
                "WHERE ts >= ? ORDER BY ts DESC LIMIT ?",
                (cutoff, limit),
            ).fetchall()
        finally:
            if self._shared_conn is None:
                conn.close()
        return [
            {"agent_id": agent_id, "message": message, "ts": ts}
            for agent_id, message, ts in rows
        ]
