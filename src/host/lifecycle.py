"""LifecycleStore — SQLite append-only store for external infra-event markers.

Records out-of-band lifecycle events that the engine itself cannot observe but
which are essential context when reconstructing what happened during a session:
the provisioner restarting the host, a deploy/`systemctl restart openlegion`, a
Docker daemon bounce, an OOM kill, etc. These markers are emitted by EXTERNAL
actors (the provisioner over SSH, an operator runbook) via the internal-only
``POST /mesh/system/lifecycle_event`` endpoint and interleaved by wall-clock into
the ``openlegion session`` timeline.

Motivation (real incident): an in-flight multi-agent workflow died because the
provisioner restarted the host mid-run. Nothing in ``intent.db`` / ``traces.db``
/ ``tasks.db`` recorded the restart, so the timeline showed an unexplained gap.
This store gives the reader a place to surface that external cause.

Unlike ``trace_id``-keyed stores, lifecycle markers are NOT tied to a single
session — a host restart affects every in-flight trace at once. They are keyed
only by wall-clock ``timestamp`` and interleaved into a session timeline by time
overlap, not by ``trace_id`` join.

Follows the same SQLite-WAL + throttled time-based-GC pattern as ``IntentStore``
(``src/host/intent.py``). Append-only, default 90-day retention.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from src.shared.redaction import deep_redact, redact_text_with_urls
from src.shared.sqlite_helpers import open_db
from src.shared.utils import dumps_safe, setup_logging

logger = setup_logging("host.lifecycle")

# Cap the free-form fields so a hostile/buggy emitter can't bloat the DB.
_MAX_KIND_LEN = 64
_MAX_DETAIL_LEN = 2000


class LifecycleStore:
    """SQLite-backed append-only store for external infra-event markers."""

    def __init__(self, db_path: str = "data/lifecycle.db", max_age_hours: int | None = 24 * 90):
        self.max_age_hours = max_age_hours
        # ensure first GC runs regardless of monotonic() epoch
        self._last_age_gc: float = -300.0
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = open_db(db_path, busy_timeout_ms=5000)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS lifecycle_events (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp  REAL NOT NULL,
                kind       TEXT NOT NULL DEFAULT '',
                detail     TEXT NOT NULL DEFAULT '',
                meta_json  TEXT NOT NULL DEFAULT ''
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_lifecycle_timestamp ON lifecycle_events (timestamp)")
        self._conn.commit()

    def record(
        self,
        *,
        kind: str,
        detail: str = "",
        timestamp: float | None = None,
        meta: dict | None = None,
    ) -> dict:
        """Insert a lifecycle-event marker. Returns the stored row as a dict.

        ``timestamp`` defaults to now but may be supplied by the emitter (an
        external actor often records the marker slightly after the event it
        describes — e.g. the provisioner logging a restart once SSH reconnects).

        ``detail`` and every string inside ``meta`` are redacted at storage so a
        stray credential in an emitter's free-form text is never persisted in
        plaintext (matches the IntentStore / TraceStore H16 posture).
        """
        kind = (kind or "").strip()[:_MAX_KIND_LEN]
        detail = redact_text_with_urls((detail or "")[:_MAX_DETAIL_LEN])
        ts = float(timestamp) if timestamp is not None else time.time()
        meta_json = dumps_safe(deep_redact(meta)) if meta else ""
        cur = self._conn.execute(
            "INSERT INTO lifecycle_events (timestamp, kind, detail, meta_json) VALUES (?, ?, ?, ?)",
            (ts, kind, detail, meta_json),
        )
        self._conn.commit()
        self._maybe_gc_old()
        return {
            "id": cur.lastrowid,
            "timestamp": ts,
            "kind": kind,
            "detail": detail,
            "meta": json.loads(meta_json) if meta_json else {},
        }

    def _maybe_gc_old(self) -> None:
        """Remove rows older than max_age_hours, throttled to once per 5 minutes.

        Time-based GC (mirrors ``IntentStore._maybe_gc_old``): lifecycle markers
        are append-only with no terminal lifecycle status to anchor a per-row
        retention deadline against.
        """
        if self.max_age_hours is None:
            return
        now = time.monotonic()
        if now - self._last_age_gc < 300:
            return
        self._last_age_gc = now
        cutoff = time.time() - (self.max_age_hours * 3600)
        self._conn.execute("DELETE FROM lifecycle_events WHERE timestamp < ?", (cutoff,))
        self._conn.commit()

    def _row_to_dict(self, row: tuple) -> dict:
        """Convert a query row to a lifecycle-event dict."""
        meta_json = row[3] if len(row) > 3 else ""
        meta = json.loads(meta_json) if meta_json else {}
        return {
            "timestamp": row[0],
            "kind": row[1],
            "detail": row[2],
            "meta": meta,
        }

    _COLS = "timestamp, kind, detail, meta_json"

    def list_between(self, start: float, end: float, limit: int = 1000) -> list[dict]:
        """Return markers with ``start <= timestamp <= end``, ordered by time."""
        cur = self._conn.execute(
            f"SELECT {self._COLS} FROM lifecycle_events "
            "WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp LIMIT ?",
            (start, end, max(1, min(limit, 10000))),
        )
        return [self._row_to_dict(row) for row in cur.fetchall()]

    def list_recent(self, *, since: float | None = None, limit: int = 100) -> list[dict]:
        """Return recent markers (newest first), optionally floored at ``since``."""
        conditions = []
        params: list = []
        if since is not None:
            conditions.append("timestamp >= ?")
            params.append(since)
        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        params.append(max(1, min(limit, 1000)))
        cur = self._conn.execute(
            f"SELECT {self._COLS} FROM lifecycle_events{where} ORDER BY id DESC LIMIT ?",
            params,
        )
        return [self._row_to_dict(row) for row in cur.fetchall()]

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
