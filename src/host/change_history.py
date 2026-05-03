"""Persistent log of soft edits the operator can revert within a TTL window.

Soft edits (instructions / soul / heartbeat / interface / role) apply
immediately from the operator and surface to the user as receipts with a
single ``[Undo]`` button. This store backs the undo button: each row holds
the ``old_value`` (so the reverse write is reproducible), the ``new_value``
(so the dashboard can render the diff), and a 5-minute expiry that mirrors
the ``PendingActions`` TTL pattern.

External contract:

* :meth:`record` — persist a fresh edit, return the row dict (caller is
  expected to surface ``undo_token`` + ``expires_at`` to the user).
* :meth:`peek` — read without consuming. Returns ``None`` for unknown,
  expired, or already-consumed tokens.
* :meth:`consume_for_undo` — atomic single-shot consume gated on the row
  being unexpired and not already-consumed. Returns the row on success
  (caller does the actual reverse-write), or ``None`` otherwise.

Reaping is opportunistic on the read/write paths just like
``PendingActions``; consumed rows are kept around as audit trail with a
``consumed_at`` timestamp so a future activity feed (PR 5) can render
"Reverted X" entries.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from src.shared.utils import setup_logging

logger = setup_logging("host.change_history")

_DEFAULT_TTL_SEC = 300


class ChangeHistory:
    """SQLite-backed log of revertible soft edits.

    Rows live forever for audit (consumed rows just flip ``consumed=1``);
    only the *undo eligibility* expires after ``ttl``. Future maintenance
    can drop ancient rows with a janitor — for now the table is bounded
    naturally by soft-edit volume, which is low.
    """

    def __init__(self, db_path: str, *, event_bus=None):
        self.db_path = db_path
        self._event_bus = event_bus
        # ``:memory:`` SQLite databases are per-connection -- closing
        # and reopening loses everything. Keep a single long-lived
        # connection for in-memory mode (used in tests + the legacy
        # shim path) so schema and rows survive across calls.
        self._shared_conn: sqlite3.Connection | None = None
        if db_path == ":memory:":
            self._shared_conn = sqlite3.connect(
                db_path, isolation_level=None, check_same_thread=False,
            )
            self._shared_conn.execute("PRAGMA busy_timeout=30000")
        self._init_schema()

    def set_event_bus(self, bus) -> None:
        """Attach (or replace) the EventBus used for dashboard events."""
        self._event_bus = bus

    def _safe_emit(self, event_type: str, agent: str, data: dict) -> None:
        """Emit a dashboard event, swallowing failures."""
        bus = self._event_bus
        if bus is None:
            return
        try:
            bus.emit(event_type, agent=agent, data=data)
        except Exception as e:
            logger.debug("ChangeHistory event emit failed (%s): %s", event_type, e)

    @contextmanager
    def _conn(self):
        if self._shared_conn is not None:
            yield self._shared_conn
            return
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            yield conn
        finally:
            conn.close()

    def close(self) -> None:
        """Close the in-memory connection if any. Tests use this."""
        if self._shared_conn is not None:
            self._shared_conn.close()
            self._shared_conn = None

    def _init_schema(self) -> None:
        """Create the change_history table if it doesn't exist."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS change_history (
                    undo_token TEXT PRIMARY KEY,
                    actor TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    field TEXT NOT NULL,
                    old_value_json TEXT NOT NULL,
                    new_value_json TEXT NOT NULL,
                    summary TEXT,
                    reason TEXT,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    consumed INTEGER NOT NULL DEFAULT 0,
                    consumed_at REAL
                );
                CREATE INDEX IF NOT EXISTS idx_change_history_agent
                    ON change_history (agent_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_change_history_expires
                    ON change_history (expires_at);
            """)

    # ── Core API ────────────────────────────────────────────────────

    def record(
        self,
        *,
        undo_token: str,
        actor: str,
        agent_id: str,
        field: str,
        old_value: Any,
        new_value: Any,
        summary: str = "",
        reason: str = "",
        ttl: int = _DEFAULT_TTL_SEC,
    ) -> dict:
        """Persist a fresh soft-edit. Returns the row dict.

        ``old_value`` / ``new_value`` are JSON-encoded so non-string
        fields (permissions dict, budget dict) round-trip safely. The
        caller is the soft-edit endpoint, which has already written the
        new value to YAML before recording — the row is purely the audit
        trail + undo handle.
        """
        now = time.time()
        expires_at = now + ttl
        # Opportunistic reap of fully-expired-and-consumed rows could go
        # here, but for now we keep them as audit trail. Only failed
        # writes get pruned.
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO change_history "
                "(undo_token, actor, agent_id, field, old_value_json, "
                "new_value_json, summary, reason, created_at, expires_at, "
                "consumed, consumed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL)",
                (
                    undo_token, actor, agent_id, field,
                    json.dumps(old_value, default=str),
                    json.dumps(new_value, default=str),
                    summary, reason, now, expires_at,
                ),
            )
        return {
            "undo_token": undo_token,
            "actor": actor,
            "agent_id": agent_id,
            "field": field,
            "old_value": old_value,
            "new_value": new_value,
            "summary": summary,
            "reason": reason,
            "created_at": now,
            "expires_at": expires_at,
            "consumed": False,
            "consumed_at": None,
        }

    def peek(self, undo_token: str) -> dict | None:
        """Read a change-history row without consuming.

        Returns ``None`` for unknown or already-consumed tokens (consumed
        rows are still in the table but no longer revertible). Expired
        but unconsumed rows are also returned as ``None`` from the
        public peek path — callers want "is this token usable?".
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT actor, agent_id, field, old_value_json, "
                "new_value_json, summary, reason, created_at, "
                "expires_at, consumed, consumed_at "
                "FROM change_history WHERE undo_token=?",
                (undo_token,),
            ).fetchone()
        if not row:
            return None
        if int(row[9]) == 1:
            return None
        if time.time() > row[8]:
            return None
        return {
            "undo_token": undo_token,
            "actor": row[0],
            "agent_id": row[1],
            "field": row[2],
            "old_value": json.loads(row[3]),
            "new_value": json.loads(row[4]),
            "summary": row[5] or "",
            "reason": row[6] or "",
            "created_at": row[7],
            "expires_at": row[8],
            "consumed": False,
            "consumed_at": row[10],
        }

    def consume_for_undo(self, undo_token: str) -> dict | None:
        """Atomically claim a token for undo. Returns the row or None.

        Returns ``None`` when:
          * unknown token
          * already consumed (double-undo prevention)
          * expired (TTL window passed)

        On success, marks the row consumed in the SAME transaction so
        a concurrent undo can't double-apply the reverse-write.
        """
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT actor, agent_id, field, old_value_json, "
                    "new_value_json, summary, reason, created_at, "
                    "expires_at, consumed FROM change_history "
                    "WHERE undo_token=?",
                    (undo_token,),
                ).fetchone()
                if not row:
                    conn.execute("COMMIT")
                    return None
                if int(row[9]) == 1:
                    # Already consumed — double-undo blocked.
                    conn.execute("COMMIT")
                    return None
                now = time.time()
                if now > row[8]:
                    # Expired — leave the row in place as audit but
                    # refuse to apply the reverse-write.
                    conn.execute("COMMIT")
                    return None
                conn.execute(
                    "UPDATE change_history SET consumed=1, consumed_at=? "
                    "WHERE undo_token=?",
                    (now, undo_token),
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return {
            "undo_token": undo_token,
            "actor": row[0],
            "agent_id": row[1],
            "field": row[2],
            "old_value": json.loads(row[3]),
            "new_value": json.loads(row[4]),
            "summary": row[5] or "",
            "reason": row[6] or "",
            "created_at": row[7],
            "expires_at": row[8],
            "consumed": True,
            "consumed_at": now,
        }

    def list_unconsumed_for_field(
        self, agent_id: str, field: str, *, before: float | None = None,
    ) -> list[dict]:
        """Return unconsumed (and unexpired) receipts for ``agent_id``+``field``.

        Used to detect "superseded" receipts: when a new soft-edit lands
        on the same field, the prior receipt's undo button still works
        but rolling back from the latest value would silently lose the
        intervening edits. The endpoint uses this to flag affected
        receipts so the UI can warn the operator.

        ``before`` is an optional epoch cut-off — pass the new edit's
        ``created_at`` to exclude itself when called immediately after
        ``record``.
        """
        now = time.time()
        clauses = [
            "agent_id = ?",
            "field = ?",
            "consumed = 0",
            "expires_at > ?",
        ]
        params: list[Any] = [agent_id, field, now]
        if before is not None:
            clauses.append("created_at < ?")
            params.append(before)
        sql = (
            "SELECT undo_token, actor, agent_id, field, "
            "old_value_json, new_value_json, summary, reason, "
            "created_at, expires_at, consumed, consumed_at "
            "FROM change_history WHERE " + " AND ".join(clauses)
            + " ORDER BY created_at ASC"
        )
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        out = []
        for r in rows:
            out.append({
                "undo_token": r[0],
                "actor": r[1],
                "agent_id": r[2],
                "field": r[3],
                "old_value": json.loads(r[4]),
                "new_value": json.loads(r[5]),
                "summary": r[6] or "",
                "reason": r[7] or "",
                "created_at": r[8],
                "expires_at": r[9],
                "consumed": bool(r[10]),
                "consumed_at": r[11],
            })
        return out

    # ── Maintenance ────────────────────────────────────────────────

    def reap_expired(self) -> int:
        """Mark every expired-and-unconsumed row as consumed.

        We don't delete: history is useful for audit and the future
        activity feed. ``consumed=1`` simply takes the row out of the
        ``peek``-eligible set. Returns count of rows touched.
        """
        now = time.time()
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE change_history SET consumed=1, consumed_at=? "
                "WHERE consumed=0 AND expires_at < ?",
                (now, now),
            )
            return cur.rowcount

    def list_recent(self, agent_id: str | None = None, limit: int = 20) -> list[dict]:
        """Return recent change-history rows, newest first.

        Useful for the dashboard activity feed (PR 5) and for diagnostic
        endpoints. Includes consumed rows so the feed can show "Reverted"
        entries; callers filter on ``consumed`` if they only want live
        ones.
        """
        with self._conn() as conn:
            if agent_id:
                rows = conn.execute(
                    "SELECT undo_token, actor, agent_id, field, "
                    "old_value_json, new_value_json, summary, reason, "
                    "created_at, expires_at, consumed, consumed_at "
                    "FROM change_history WHERE agent_id=? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (agent_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT undo_token, actor, agent_id, field, "
                    "old_value_json, new_value_json, summary, reason, "
                    "created_at, expires_at, consumed, consumed_at "
                    "FROM change_history ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        out = []
        for r in rows:
            out.append({
                "undo_token": r[0],
                "actor": r[1],
                "agent_id": r[2],
                "field": r[3],
                "old_value": json.loads(r[4]),
                "new_value": json.loads(r[5]),
                "summary": r[6] or "",
                "reason": r[7] or "",
                "created_at": r[8],
                "expires_at": r[9],
                "consumed": bool(r[10]),
                "consumed_at": r[11],
            })
        return out
