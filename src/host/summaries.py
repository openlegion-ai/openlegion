"""Work summaries — durable storage for operator-generated team summaries.

The Work tab's clutter problem at 30-agent scale isn't a UI bug — it's
that the operator's mental unit is the *task*, but the user's mental
unit at scale is the *team*. This module persists periodic team-level
summaries (one per team per period, default daily) that the user rates.
The user's rating + feedback flows back into the next summary's
composition prompt, closing the loop.

Schema invariants:

- One summary per ``(scope_kind, scope_id, period_start)`` triple. The
  UNIQUE constraint dedupes concurrent cron firings.
- Summaries are immutable after creation EXCEPT for ``rating`` /
  ``feedback`` / ``rated_at`` which the user can write once. A 24h
  edit window after first rating lets the user revise their judgement
  before the row locks (mirrors the per-task outcome pattern).
- ``retention_until`` is stamped at creation (default 30 days).
  Opportunistic reaping on hot paths drops expired rows so the table
  doesn't grow unbounded.

The store is decoupled from the mesh's permission tier — visibility
filtering happens at the endpoint layer (operator sees all, workers
see only their team's). The store accepts whatever scope+id the caller
specifies and trusts the caller-layer gate.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from src.shared.utils import dumps_safe, setup_logging

logger = setup_logging("host.summaries")


# Frozen set of valid scope kinds. ``team`` covers project-attached
# work; ``solo`` is a synthetic scope for agents that aren't members
# of any team (so the Work tab can still surface a single "Solo agents"
# card without inventing a fake team).
VALID_SCOPE_KINDS: frozenset[str] = frozenset({"team", "solo"})

# Frozen set of valid ratings. Mirrors the per-task outcome scale so
# the same UI affordance can drive both surfaces:
#   - ``accepted``     — 👍 "good job, keep doing what you're doing"
#   - ``acknowledged`` — ➖ "neutral, no strong signal"
#   - ``rework``       — 👎 with feedback → operator adjusts approach
VALID_RATINGS: frozenset[str] = frozenset({"accepted", "acknowledged", "rework"})

# Default TTL — summaries older than this are eligible for reaping on
# the next opportunistic sweep. 30 days matches the dashboard's history
# pane window in PR-B.
DEFAULT_RETENTION_SECONDS = 30 * 86400

# Hard cap on feedback length so a runaway prompt can't bloat the DB.
MAX_FEEDBACK_CHARS = 4000

# After the first rating lands, the user has this long to revise it
# before the row locks. 24h is the standard "you slept on it" window.
RATING_EDIT_WINDOW_SECONDS = 24 * 3600

# Minimum seconds between opportunistic ``_safe_reap()`` invocations.
# The list endpoint calls reap on every fire; without this guard a
# sustained list-traffic pattern (e.g. dashboard tab open + polling)
# would run an unbounded ``DELETE`` repeatedly. 60s is short enough
# that expired rows clear within a minute of becoming reapable but
# long enough that the DELETE doesn't dominate the hot path.
_SAFE_REAP_MIN_INTERVAL_SECONDS = 60


class SummaryNotFound(KeyError):
    """Raised when a summary id doesn't exist."""


class RatingLocked(ValueError):
    """Raised when a rating edit lands past the edit window."""


class InvalidScope(ValueError):
    """Raised on an unknown ``scope_kind``."""


class WorkSummariesStore:
    """SQLite-backed work-summaries store.

    Schema-managed at construction. WAL + 30s busy timeout matches the
    mesh's other SQLite stores. Opportunistic reaping is exposed via
    :meth:`reap_expired` — call from hot paths (list / get) so the
    table stays bounded without a dedicated sweeper task.
    """

    _SELECT_COLS = (
        "id, scope_kind, scope_id, period_start, period_end, "
        "narrative_md, metrics_json, recommendations_json, "
        "rating, feedback, generated_by, generated_at, "
        "rated_at, rated_by, retention_until"
    )

    def __init__(
        self,
        db_path: str,
        *,
        retention_seconds: int = DEFAULT_RETENTION_SECONDS,
        event_bus: Any = None,
    ) -> None:
        self.db_path = db_path
        self.retention_seconds = retention_seconds
        self._event_bus = event_bus
        # Last opportunistic-reap wall-clock time. The list endpoint
        # calls ``_safe_reap()`` on every fire; this stamp gates how
        # often the unbounded DELETE actually runs.
        self._last_reap_ts: float = 0.0
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        # In-memory mode keeps a single shared connection so the schema
        # doesn't disappear between calls. Disk-backed mode opens a
        # fresh connection per operation (matches Tasks/CostTracker).
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
        """Yield a connection. In-memory uses the shared one; disk
        opens fresh and the caller is responsible for closing via the
        context manager idiom.
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
                CREATE TABLE IF NOT EXISTS work_summaries (
                    id                   TEXT PRIMARY KEY,
                    scope_kind           TEXT NOT NULL,
                    scope_id             TEXT NOT NULL,
                    period_start         REAL NOT NULL,
                    period_end           REAL NOT NULL,
                    narrative_md         TEXT NOT NULL,
                    metrics_json         TEXT NOT NULL,
                    recommendations_json TEXT NOT NULL DEFAULT '[]',
                    rating               TEXT,
                    feedback             TEXT,
                    generated_by         TEXT NOT NULL,
                    generated_at         REAL NOT NULL,
                    rated_at             REAL,
                    rated_by             TEXT,
                    retention_until      REAL NOT NULL,
                    UNIQUE(scope_kind, scope_id, period_start)
                );
                CREATE INDEX IF NOT EXISTS idx_work_summaries_scope
                    ON work_summaries(scope_kind, scope_id, period_start DESC);
                CREATE INDEX IF NOT EXISTS idx_work_summaries_retention
                    ON work_summaries(retention_until);
                CREATE INDEX IF NOT EXISTS idx_work_summaries_unrated
                    ON work_summaries(rating, generated_at DESC)
                    WHERE rating IS NULL;
            """)
            conn.commit()
        finally:
            if self._shared_conn is None:
                conn.close()

    @staticmethod
    def _row_to_dict(row: tuple) -> dict:
        """Convert a row from ``_SELECT_COLS`` order into a JSON-friendly dict."""
        (
            sid, scope_kind, scope_id, period_start, period_end,
            narrative_md, metrics_json, recommendations_json,
            rating, feedback, generated_by, generated_at,
            rated_at, rated_by, retention_until,
        ) = row
        try:
            metrics = json.loads(metrics_json) if metrics_json else {}
        except (json.JSONDecodeError, TypeError):
            metrics = {"_decode_error": True}
        try:
            recommendations = (
                json.loads(recommendations_json) if recommendations_json else []
            )
        except (json.JSONDecodeError, TypeError):
            recommendations = []
        return {
            "id": sid,
            "scope_kind": scope_kind,
            "scope_id": scope_id,
            "period_start": period_start,
            "period_end": period_end,
            "narrative_md": narrative_md,
            "metrics": metrics,
            "recommendations": recommendations,
            "rating": rating,
            "feedback": feedback,
            "generated_by": generated_by,
            "generated_at": generated_at,
            "rated_at": rated_at,
            "rated_by": rated_by,
            "retention_until": retention_until,
        }

    def _safe_emit(self, event_type: str, agent: str, data: dict) -> None:
        """Emit a dashboard event, swallowing failures.

        Mirrors the orchestration store's pattern — event emission
        must never sink a successful DB write. Matches the
        ``DashboardEvent.type`` literal added in ``src/shared/types.py``.
        """
        bus = self._event_bus
        if bus is None:
            return
        try:
            bus.emit(event_type, agent=agent, data=data)
        except Exception as e:
            logger.debug("work_summary event emit failed (%s): %s", event_type, e)

    # ----------------------------------------------------------------- CREATE
    def create(
        self,
        *,
        scope_kind: str,
        scope_id: str,
        period_start: float,
        period_end: float,
        narrative_md: str,
        metrics: dict,
        recommendations: list[str] | None = None,
        generated_by: str,
        summary_id: str | None = None,
    ) -> dict:
        """Insert a new summary row. Returns the JSON dict.

        Idempotent on ``(scope_kind, scope_id, period_start)`` — a
        second create with the same triple raises ``ValueError`` so
        callers can decide whether to noop or overwrite. Crons can
        catch the error to handle the "already generated this period"
        case cleanly.
        """
        if scope_kind not in VALID_SCOPE_KINDS:
            raise InvalidScope(
                f"scope_kind must be one of {sorted(VALID_SCOPE_KINDS)}, "
                f"got {scope_kind!r}"
            )
        if not scope_id:
            raise ValueError("scope_id is required")
        if period_end < period_start:
            raise ValueError("period_end must be >= period_start")
        from src.shared.utils import generate_id
        sid = summary_id or generate_id("ws")
        now = time.time()
        retention_until = now + self.retention_seconds
        recs_json = dumps_safe(recommendations or [])
        metrics_json = dumps_safe(metrics or {})
        conn = self._conn()
        try:
            conn.execute(
                "INSERT INTO work_summaries ("
                "id, scope_kind, scope_id, period_start, period_end, "
                "narrative_md, metrics_json, recommendations_json, "
                "generated_by, generated_at, retention_until"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    sid, scope_kind, scope_id, period_start, period_end,
                    narrative_md, metrics_json, recs_json,
                    generated_by, now, retention_until,
                ),
            )
            conn.commit()
        except sqlite3.IntegrityError as e:
            # UNIQUE(scope_kind, scope_id, period_start) violation.
            raise ValueError(
                f"summary already exists for {scope_kind}={scope_id!r} "
                f"period_start={period_start}"
            ) from e
        finally:
            if self._shared_conn is None:
                conn.close()
        row = self.get(sid)
        self._safe_emit(
            "work_summary_created",
            agent=generated_by,
            data={
                "summary_id": sid,
                "scope_kind": scope_kind,
                "scope_id": scope_id,
                "period_start": period_start,
                "period_end": period_end,
                "metrics": row["metrics"] if row else {},
            },
        )
        return row  # type: ignore[return-value]

    # -------------------------------------------------------------------- GET
    def get(self, summary_id: str) -> dict | None:
        conn = self._conn()
        try:
            cur = conn.execute(
                f"SELECT {self._SELECT_COLS} FROM work_summaries WHERE id = ?",
                (summary_id,),
            )
            row = cur.fetchone()
        finally:
            if self._shared_conn is None:
                conn.close()
        if row is None:
            return None
        return self._row_to_dict(row)

    # ------------------------------------------------------------------- LIST
    def list_recent(
        self,
        *,
        scope_kind: str | None = None,
        scope_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """Most-recent summaries first, filtered by scope if provided."""
        limit = max(1, min(int(limit), 500))
        offset = max(0, int(offset))
        clauses = []
        params: list[Any] = []
        if scope_kind is not None:
            if scope_kind not in VALID_SCOPE_KINDS:
                raise InvalidScope(f"unknown scope_kind {scope_kind!r}")
            clauses.append("scope_kind = ?")
            params.append(scope_kind)
        if scope_id is not None:
            clauses.append("scope_id = ?")
            params.append(scope_id)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = (
            f"SELECT {self._SELECT_COLS} FROM work_summaries "
            f"{where} ORDER BY generated_at DESC LIMIT ? OFFSET ?"
        )
        params.extend([limit, offset])
        conn = self._conn()
        try:
            rows = conn.execute(sql, params).fetchall()
        finally:
            if self._shared_conn is None:
                conn.close()
        return [self._row_to_dict(r) for r in rows]

    # ----------------------------------------------------------------- RATING
    def set_rating(
        self,
        summary_id: str,
        rating: str,
        feedback: str | None = None,
        *,
        actor: str = "user",
    ) -> dict:
        """Apply a user rating + optional feedback.

        Writeable for ``RATING_EDIT_WINDOW_SECONDS`` after first rating;
        attempts after that raise :class:`RatingLocked`. The first
        ``rated_at`` is preserved across edits (``COALESCE``) so the
        UI knows "rated 12h ago" rather than "rated 2 minutes ago" on
        every revision.

        TOCTOU-free: the edit-window check lives in the SQL UPDATE
        predicate, not a separate SELECT. Two concurrent writers
        racing on the same row can't both succeed past the lock —
        whichever runs second sees ``rowcount == 0`` and a follow-up
        SELECT decides whether the row was missing (404) or locked
        (409).
        """
        if rating not in VALID_RATINGS:
            raise ValueError(
                f"rating must be one of {sorted(VALID_RATINGS)}, got {rating!r}"
            )
        feedback_clean = (feedback or "").strip() or None
        if feedback_clean is not None and len(feedback_clean) > MAX_FEEDBACK_CHARS:
            raise ValueError(
                f"feedback exceeds {MAX_FEEDBACK_CHARS} chars "
                f"(got {len(feedback_clean)})"
            )
        now = time.time()
        conn = self._conn()
        try:
            cur = conn.execute(
                "UPDATE work_summaries SET "
                "rating = ?, feedback = ?, "
                "rated_at = COALESCE(rated_at, ?), "
                "rated_by = ? "
                "WHERE id = ? AND ("
                "  rated_at IS NULL "
                "  OR (? - rated_at) <= ?"
                ")",
                (
                    rating, feedback_clean, now, actor,
                    summary_id, now, RATING_EDIT_WINDOW_SECONDS,
                ),
            )
            conn.commit()
            if cur.rowcount == 0:
                # Update matched nothing — either the row doesn't exist
                # or it exists but the edit window has expired.
                existing = conn.execute(
                    "SELECT rated_at FROM work_summaries WHERE id = ?",
                    (summary_id,),
                ).fetchone()
                if existing is None:
                    raise SummaryNotFound(summary_id)
                (existing_rated_at,) = existing
                raise RatingLocked(
                    f"summary {summary_id} rating locked "
                    f"({int(now - (existing_rated_at or now))}s past "
                    f"first rating; edit window is "
                    f"{RATING_EDIT_WINDOW_SECONDS}s)"
                )
        finally:
            if self._shared_conn is None:
                conn.close()
        result = self.get(summary_id)
        self._safe_emit(
            "work_summary_rated",
            agent=actor,
            data={
                "summary_id": summary_id,
                "scope_kind": (result or {}).get("scope_kind"),
                "scope_id": (result or {}).get("scope_id"),
                "rating": rating,
                "feedback": feedback_clean,
                "actor": actor,
                "ts": now,
            },
        )
        return result  # type: ignore[return-value]

    # ----------------------------------------------------------------- REAP
    def reap_expired(self) -> int:
        """Delete rows whose ``retention_until`` is past. Returns count."""
        now = time.time()
        conn = self._conn()
        try:
            cur = conn.execute(
                "DELETE FROM work_summaries WHERE retention_until < ?",
                (now,),
            )
            conn.commit()
            return cur.rowcount or 0
        finally:
            if self._shared_conn is None:
                conn.close()

    def _safe_reap(self) -> None:
        """Reap-expired wrapper that never raises on the hot path.

        Rate-limited to once per ``_SAFE_REAP_MIN_INTERVAL_SECONDS``
        so a sustained list-traffic pattern (dashboard polling) can't
        run the unbounded DELETE on every fire. Last-reap timestamp
        is best-effort instance state — multi-process deployments
        will reap up to once per process per interval, which is fine.
        """
        now = time.time()
        if now - self._last_reap_ts < _SAFE_REAP_MIN_INTERVAL_SECONDS:
            return
        try:
            self.reap_expired()
            self._last_reap_ts = now
        except Exception as e:
            logger.debug("opportunistic reap_expired failed: %s", e)
            # Stamp even on failure to avoid hammering a broken DB on
            # the hot path. Worst case the next attempt is delayed by
            # the rate-limit interval.
            self._last_reap_ts = now

    # ------------------------------------------------------------ FEEDBACK FETCH
    def recent_feedback(
        self, *, scope_kind: str, scope_id: str, limit: int = 5,
    ) -> list[dict]:
        """Return the last ``limit`` rated summaries for a scope.

        Used by the compose-summary path to inject prior feedback into
        the next summary's recommendation block — the user's 👎 reasons
        shape what operator focuses on next time.
        """
        if scope_kind not in VALID_SCOPE_KINDS:
            raise InvalidScope(f"unknown scope_kind {scope_kind!r}")
        limit = max(1, min(int(limit), 20))
        conn = self._conn()
        try:
            rows = conn.execute(
                f"SELECT {self._SELECT_COLS} FROM work_summaries "
                "WHERE scope_kind = ? AND scope_id = ? AND rating IS NOT NULL "
                "ORDER BY rated_at DESC LIMIT ?",
                (scope_kind, scope_id, limit),
            ).fetchall()
        finally:
            if self._shared_conn is None:
                conn.close()
        return [self._row_to_dict(r) for r in rows]
