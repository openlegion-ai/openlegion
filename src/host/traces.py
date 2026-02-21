"""TraceStore — SQLite ring buffer for request trace events.

Stores trace events (dispatch, LLM call, transport hop, etc.) keyed by
trace_id.  Capped at *max_events* rows — oldest events are evicted after
each insert.

Follows the same SQLite-WAL pattern as ``CostTracker`` and ``Blackboard``.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from src.shared.utils import setup_logging

logger = setup_logging("host.traces")


class TraceStore:
    """SQLite-backed ring buffer for request trace events."""

    def __init__(self, db_path: str = "data/traces.db", max_events: int = 10_000):
        self.max_events = max_events
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS traces (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id   TEXT NOT NULL,
                timestamp  REAL NOT NULL,
                source     TEXT NOT NULL,
                agent      TEXT NOT NULL DEFAULT '',
                event_type TEXT NOT NULL,
                detail     TEXT NOT NULL DEFAULT '',
                duration_ms INTEGER NOT NULL DEFAULT 0,
                status     TEXT NOT NULL DEFAULT '',
                error      TEXT NOT NULL DEFAULT '',
                meta_json  TEXT NOT NULL DEFAULT ''
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_traces_trace_id ON traces (trace_id)"
        )
        # Add columns if upgrading from older schema
        for col, typedef in [
            ("status", "TEXT NOT NULL DEFAULT ''"),
            ("error", "TEXT NOT NULL DEFAULT ''"),
            ("meta_json", "TEXT NOT NULL DEFAULT ''"),
        ]:
            try:
                self._conn.execute(f"ALTER TABLE traces ADD COLUMN {col} {typedef}")
            except sqlite3.OperationalError:
                pass  # column already exists
        self._conn.commit()

    def record(
        self,
        trace_id: str,
        source: str,
        agent: str,
        event_type: str,
        detail: str = "",
        duration_ms: int = 0,
        status: str = "",
        error: str = "",
        meta: dict | None = None,
    ) -> None:
        """Insert a trace event and evict overflow."""
        import json as _json
        meta_json = _json.dumps(meta, default=str) if meta else ""
        self._conn.execute(
            "INSERT INTO traces (trace_id, timestamp, source, agent, event_type, detail, duration_ms, status, error, meta_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (trace_id, time.time(), source, agent, event_type, detail, duration_ms, status, error, meta_json),
        )
        # Ring-buffer eviction: keep newest N rows regardless of ID gaps
        self._conn.execute(
            "DELETE FROM traces WHERE id NOT IN ("
            "  SELECT id FROM traces ORDER BY id DESC LIMIT ?"
            ")",
            (self.max_events,),
        )
        self._conn.commit()

    def _row_to_dict(self, row: tuple) -> dict:
        """Convert a query row to a trace event dict."""
        import json as _json
        meta_json = row[9] if len(row) > 9 else ""
        meta = _json.loads(meta_json) if meta_json else {}
        return {
            "trace_id": row[0],
            "timestamp": row[1],
            "source": row[2],
            "agent": row[3],
            "event_type": row[4],
            "detail": row[5],
            "duration_ms": row[6],
            "status": row[7] if len(row) > 7 else "",
            "error": row[8] if len(row) > 8 else "",
            "meta": meta,
        }

    _TRACE_COLS = (
        "trace_id, timestamp, source, agent, event_type, detail, "
        "duration_ms, status, error, meta_json"
    )

    def get_trace(self, trace_id: str) -> list[dict]:
        """Return all events for a given trace_id, ordered by time."""
        cur = self._conn.execute(
            f"SELECT {self._TRACE_COLS} FROM traces WHERE trace_id = ? ORDER BY id",
            (trace_id,),
        )
        return [self._row_to_dict(row) for row in cur.fetchall()]

    def list_recent(self, limit: int = 50) -> list[dict]:
        """Return the most recent trace events (newest first)."""
        cur = self._conn.execute(
            f"SELECT {self._TRACE_COLS} FROM traces ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return [self._row_to_dict(row) for row in cur.fetchall()]

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
