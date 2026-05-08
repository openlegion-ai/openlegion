"""Persistent per-session "opened conversations" store.

Backs the Phase 1 unified-messenger sidebar. The dashboard tracks which
worker conversations a user has explicitly opened (Operator is always
implicit / pinned and does NOT live here). Two design constraints drive
the SQLite-backed model:

1. **Session-scoped, not process-scoped.** In multi-user deployments
   (CLAUDE.md describes the SSO + Caddy posture where multiple users hit
   one engine via per-user ``ol_session`` cookies), opened-state must
   not leak between concurrent users. The previous implementation used
   a module-level ``set[str]`` shared across every session in the same
   Python process — opening a chat exposed that worker to every other
   user. This store keys every row on a per-session identifier (a hash
   of the ``ol_session`` cookie value, derived inside the dashboard
   router) so two sessions never see each other's opened workers.

2. **Persistent across process restarts.** The old in-memory set was
   wiped on container restart. SQLite persistence preserves opened
   state across ``systemctl restart openlegion`` / Docker bounces.

Schema::

    dashboard_opened_conversations(
        session_id  TEXT NOT NULL,
        agent_id    TEXT NOT NULL,
        opened_at   REAL NOT NULL,
        PRIMARY KEY (session_id, agent_id)
    )

The composite primary key gives idempotent ``open()`` (re-opening the
same conversation refreshes ``opened_at`` via ``ON CONFLICT``) and an
implicit index on ``session_id`` that ``list_for_session()`` rides.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from src.shared.utils import setup_logging

logger = setup_logging("dashboard.conversations")


class OpenedConversationsStore:
    """Persistent (session_id, agent_id) registry for opened workers.

    Single connection, WAL mode, ``check_same_thread=False`` so the
    FastAPI request handlers (which run on the asyncio loop's default
    executor) can share it. Mirrors the conventions used by
    :class:`NotificationStore` and :class:`DashboardTelemetry`.
    """

    def __init__(self, db_path: str | Path = "data/dashboard_conversations.db") -> None:
        Path(str(db_path)).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=30000")
        self._init_schema()

    def _init_schema(self) -> None:
        # Idempotent — safe across redeploys + first-deploy installs.
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS dashboard_opened_conversations (
                session_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                opened_at REAL NOT NULL,
                PRIMARY KEY (session_id, agent_id)
            );
            """,
        )
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    # ── Writes ───────────────────────────────────────────────

    def open(self, session_id: str, agent_id: str) -> None:
        """Mark ``agent_id`` as opened for ``session_id``.

        Idempotent — re-opening the same conversation refreshes
        ``opened_at`` rather than raising on the PK collision.
        """
        if not session_id or not agent_id:
            raise ValueError("session_id and agent_id must be non-empty")
        self.db.execute(
            """
            INSERT INTO dashboard_opened_conversations (session_id, agent_id, opened_at)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id, agent_id) DO UPDATE SET opened_at = excluded.opened_at
            """,
            (session_id, agent_id, time.time()),
        )
        self.db.commit()

    def close_conversation(self, session_id: str, agent_id: str) -> None:
        """Remove ``agent_id`` from ``session_id``'s opened list.

        No-op if the row does not exist (matches the previous
        ``set.discard`` semantics used by the in-memory implementation).
        """
        if not session_id or not agent_id:
            return
        self.db.execute(
            "DELETE FROM dashboard_opened_conversations "
            "WHERE session_id = ? AND agent_id = ?",
            (session_id, agent_id),
        )
        self.db.commit()

    # ── Reads ────────────────────────────────────────────────

    def list_for_session(self, session_id: str) -> list[str]:
        """Return the agent_ids opened by ``session_id``, sorted ascending.

        Sorted so the messenger's worker order is deterministic across
        reloads (matches the prior ``sorted(_opened_conversations)`` call
        site — UI tests rely on this ordering).
        """
        if not session_id:
            return []
        rows = self.db.execute(
            "SELECT agent_id FROM dashboard_opened_conversations "
            "WHERE session_id = ? ORDER BY agent_id ASC",
            (session_id,),
        ).fetchall()
        return [row[0] for row in rows]
