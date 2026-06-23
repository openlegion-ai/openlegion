"""IntentStore — SQLite append-only store for verbatim inbound messages.

Persists the FULL verbatim inbound message centrally, keyed by ``trace_id``
plus the originating ``MessageOrigin`` (kind/channel/user), so that *intent*
— what the human actually typed — survives container wipes, ``/chat/reset``,
and deploys. Today the user's words live only in container-local
``chat_transcript.jsonl`` (``src/agent/workspace.py``), which rotates (drops
the oldest half) and dies with the container; central stores keep only a
normalized task title and a short redacted preview. This store is the durable
intent layer that makes hosted-VPS reconstruction trustworthy after a deploy
(Phase 2 of docs/plans/2026-06-18-session-observability.md).

Follows the same SQLite-WAL pattern as ``TraceStore`` (``src/host/traces.py``).
Intent is append-only like traces — there is no per-row lifecycle terminal to
anchor a ``retention_until`` against (unlike ``tasks``, where reaping keys off a
terminal status), so retention is a TraceStore-style throttled time-based GC,
not a tasks-style ``retention_until`` reaper. Default window is 90 days, which
matches the session/forensic semantics (vs. traces' shorter rolling window).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from src.shared.redaction import deep_redact, redact_text_with_urls
from src.shared.sqlite_helpers import open_db
from src.shared.utils import dumps_safe, setup_logging

logger = setup_logging("host.intent")


class IntentStore:
    """SQLite-backed append-only store for verbatim inbound messages."""

    def __init__(self, db_path: str = "data/intent.db", max_age_hours: int | None = 24 * 90):
        self.max_age_hours = max_age_hours
        # ensure first GC runs regardless of monotonic() epoch
        self._last_age_gc: float = -300.0
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = open_db(db_path, busy_timeout_ms=5000)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS intent (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id       TEXT NOT NULL,
                timestamp      REAL NOT NULL,
                origin_kind    TEXT NOT NULL DEFAULT '',
                origin_channel TEXT NOT NULL DEFAULT '',
                origin_user    TEXT NOT NULL DEFAULT '',
                agent          TEXT NOT NULL DEFAULT '',
                message        TEXT NOT NULL DEFAULT '',
                meta_json      TEXT NOT NULL DEFAULT ''
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_intent_trace_id ON intent (trace_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_intent_timestamp ON intent (timestamp)"
        )
        self._conn.commit()

    def record(
        self,
        *,
        trace_id: str,
        origin_kind: str = "",
        origin_channel: str = "",
        origin_user: str = "",
        agent: str = "",
        message: str = "",
        meta: dict | None = None,
    ) -> None:
        """Insert a verbatim-intent row.

        The verbatim ``message`` and every string inside ``meta`` are redacted
        at capture so a credential or URL with sensitive query params pasted in
        chat is never persisted in plaintext (H16). Redaction is centralized
        here rather than at the dispatch callsite so no path can regress.
        ``sanitize_for_prompt`` is already applied upstream at every surface;
        this is the additional at-storage layer.
        """
        message = redact_text_with_urls(message or "")
        meta_json = dumps_safe(deep_redact(meta)) if meta else ""
        self._conn.execute(
            "INSERT INTO intent "
            "(trace_id, timestamp, origin_kind, origin_channel, origin_user, agent, message, meta_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                trace_id, time.time(), origin_kind, origin_channel,
                origin_user, agent, message, meta_json,
            ),
        )
        self._conn.commit()
        self._maybe_gc_old()

    def _maybe_gc_old(self) -> None:
        """Remove rows older than max_age_hours, throttled to once per 5 minutes.

        Time-based GC (mirrors ``TraceStore._maybe_gc_old``) rather than a
        tasks-style ``retention_until`` + reaper: intent is append-only with no
        terminal lifecycle status to anchor a per-row retention deadline.
        """
        if self.max_age_hours is None:
            return
        now = time.monotonic()
        if now - self._last_age_gc < 300:
            return
        self._last_age_gc = now
        cutoff = time.time() - (self.max_age_hours * 3600)
        self._conn.execute("DELETE FROM intent WHERE timestamp < ?", (cutoff,))
        self._conn.commit()

    def _row_to_dict(self, row: tuple) -> dict:
        """Convert a query row to an intent dict."""
        meta_json = row[7] if len(row) > 7 else ""
        meta = json.loads(meta_json) if meta_json else {}
        return {
            "trace_id": row[0],
            "timestamp": row[1],
            "origin_kind": row[2],
            "origin_channel": row[3],
            "origin_user": row[4],
            "agent": row[5],
            "message": row[6],
            "meta": meta,
        }

    _INTENT_COLS = (
        "trace_id, timestamp, origin_kind, origin_channel, "
        "origin_user, agent, message, meta_json"
    )

    def get_by_trace(self, trace_id: str) -> list[dict]:
        """Return all intent rows for a trace_id, ordered by time."""
        cur = self._conn.execute(
            f"SELECT {self._INTENT_COLS} FROM intent WHERE trace_id = ? ORDER BY id",
            (trace_id,),
        )
        return [self._row_to_dict(row) for row in cur.fetchall()]

    def list_recent(
        self,
        *,
        since: float | None = None,
        user: str | None = None,
        agent: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Return recent intent rows (newest first), with optional filters.

        Built for Phase 3's ``sessions --since`` reader.
        """
        conditions = []
        params: list = []
        if since is not None:
            conditions.append("timestamp >= ?")
            params.append(since)
        if user is not None:
            conditions.append("origin_user = ?")
            params.append(user)
        if agent is not None:
            conditions.append("agent = ?")
            params.append(agent)
        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        params.append(max(1, min(limit, 1000)))
        cur = self._conn.execute(
            f"SELECT {self._INTENT_COLS} FROM intent{where} ORDER BY id DESC LIMIT ?",
            params,
        )
        return [self._row_to_dict(row) for row in cur.fetchall()]

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
