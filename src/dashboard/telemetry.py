"""Dashboard telemetry sink (Phase -1 onboarding wizard).

Persists frontend telemetry events (e.g. wizard step transitions) to a
SQLite table so we can answer the activation hypothesis: does the
empty-fleet onboarding wizard move first-visit activation?

This is a hypothesis-test surface — keep the schema minimal:
``(id, ts, session_id, event_name, props_json)`` is enough to answer
"how many sessions started the wizard?", "where did they drop off?",
"how long did completion take?".

Process-local SQLite, WAL mode, capped retention. No external pipeline
yet — operators read the table directly via ``sqlite3 data/telemetry.db
'select ...'`` until we wire a real BI export.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from src.shared.utils import setup_logging

logger = setup_logging("dashboard.telemetry")

# Retention cap — wizard runs are short-lived; keep ~30 days of events.
# A row is ~250 bytes, so 30 days at ~1k events/day fits in a few MB.
_MAX_EVENTS = 100_000

# Per-event-name length caps (keep payload sane; the schema is JSON so
# operators can add fields without migration, but we don't accept blobs).
_MAX_EVENT_NAME = 64
_MAX_PROPS_BYTES = 4096

# Per-session rate limit for the HTTP endpoint. 60/min matches the spec
# in docs/plans/2026-05-08-board-ux-overhaul.md Phase -1.
RATE_LIMIT_EVENTS_PER_MIN = 60
_RATE_LIMIT_WINDOW_S = 60.0


class DashboardTelemetry:
    """SQLite-backed telemetry event sink with a per-session rate limiter."""

    def __init__(self, db_path: str = "data/telemetry.db") -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=30000")
        self._lock = threading.Lock()
        self._init_schema()
        # Per-session rate buckets — sliding window of recent timestamps.
        self._rate_buckets: dict[str, deque[float]] = defaultdict(deque)

    def _init_schema(self) -> None:
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS dashboard_telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                session_id TEXT NOT NULL,
                event_name TEXT NOT NULL,
                props_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_telemetry_event_ts
                ON dashboard_telemetry(event_name, ts DESC);
            CREATE INDEX IF NOT EXISTS idx_telemetry_session_ts
                ON dashboard_telemetry(session_id, ts DESC);
            """,
        )
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def check_rate_limit(self, session_id: str) -> tuple[bool, int]:
        """Sliding-window rate limit per session.

        Returns ``(allowed, retry_after_ms)``. ``retry_after_ms`` is 0
        when allowed; otherwise the time until the oldest event in the
        window expires.
        """
        now = time.time()
        cutoff = now - _RATE_LIMIT_WINDOW_S
        with self._lock:
            bucket = self._rate_buckets[session_id]
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= RATE_LIMIT_EVENTS_PER_MIN:
                retry = int((bucket[0] + _RATE_LIMIT_WINDOW_S - now) * 1000)
                return False, max(1, retry)
            bucket.append(now)
            # Sweep stale sessions occasionally to prevent unbounded growth.
            if len(self._rate_buckets) > 1024:
                stale = [
                    k for k, b in self._rate_buckets.items()
                    if not b or b[-1] <= cutoff
                ]
                for k in stale:
                    self._rate_buckets.pop(k, None)
            return True, 0

    def record(
        self,
        *,
        event_name: str,
        session_id: str,
        props: dict[str, Any] | None = None,
        ts: float | None = None,
    ) -> int:
        """Persist one telemetry event. Returns the row id."""
        if not event_name or len(event_name) > _MAX_EVENT_NAME:
            raise ValueError(
                f"event_name must be 1..{_MAX_EVENT_NAME} chars",
            )
        try:
            payload = json.dumps(props or {}, separators=(",", ":"))
        except (TypeError, ValueError) as e:
            raise ValueError(f"props must be JSON-serializable: {e}")
        if len(payload) > _MAX_PROPS_BYTES:
            raise ValueError(
                f"props_json exceeds {_MAX_PROPS_BYTES} bytes",
            )
        if ts is None:
            ts = time.time()
        with self._lock:
            cur = self.db.execute(
                "INSERT INTO dashboard_telemetry "
                "(ts, session_id, event_name, props_json) VALUES (?, ?, ?, ?)",
                (float(ts), session_id, event_name, payload),
            )
            row_id = int(cur.lastrowid or 0)
            self._maybe_trim()
            self.db.commit()
        return row_id

    def _maybe_trim(self) -> None:
        """Keep the table bounded. Cheap when called frequently."""
        cur = self.db.execute("SELECT COUNT(*) FROM dashboard_telemetry")
        (count,) = cur.fetchone()
        if count <= _MAX_EVENTS:
            return
        excess = count - _MAX_EVENTS
        self.db.execute(
            "DELETE FROM dashboard_telemetry WHERE id IN ("
            "  SELECT id FROM dashboard_telemetry ORDER BY id ASC LIMIT ?"
            ")",
            (excess,),
        )

    def recent(
        self,
        *,
        event_name: str | None = None,
        session_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return the most recent events (debug / inspection helper)."""
        limit = max(1, min(int(limit), 1000))
        clauses = []
        params: list[Any] = []
        if event_name:
            clauses.append("event_name = ?")
            params.append(event_name)
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        cur = self.db.execute(
            f"SELECT id, ts, session_id, event_name, props_json "
            f"FROM dashboard_telemetry {where} ORDER BY id DESC LIMIT ?",
            params,
        )
        rows = []
        for row in cur.fetchall():
            try:
                props = json.loads(row[4])
            except (TypeError, ValueError):
                props = {}
            rows.append({
                "id": row[0],
                "ts": row[1],
                "session_id": row[2],
                "event_name": row[3],
                "props": props,
            })
        return rows
