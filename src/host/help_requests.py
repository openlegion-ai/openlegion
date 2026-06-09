"""Persistent registry of open "agent asks the user for help" requests.

Covers the three user-actionable asks that drive the dashboard "Needs you"
panel: ``credential_request``, ``browser_login_request``,
``browser_captcha_help_request``. Each row is one open ask, keyed by a minted
``request_id`` (uuid).

Why this exists (replaces a process-local dict): the panel is the
authoritative list of "what currently needs the user". If the backing state
lived only in memory, a mesh restart would blank the panel while requests are
still open and agents still blocked — an empty panel would falsely read as
"nothing needs you". So the registry is persisted (SQLite WAL), exactly like
``PendingActions`` / ``WorkSummariesStore`` / the blackboard.

Lifecycle: ``record`` on the ask, ``resolve`` when it's satisfied OR cancelled
(both DELETE the row — the chat transcript keeps the human-readable history;
the registry only tracks what's still OPEN). ``list_open`` feeds
``GET /mesh/help-requests``. ``resolve`` is atomic (``BEGIN IMMEDIATE``) so a
save racing a cancel can't double-fire side effects: exactly one caller wins
the claim, the loser gets ``None``.

There is intentionally NO silent size-cap eviction (the old in-memory dict
popped the oldest at 256, which could drop a real open ask with no signal).
A generous max-age reap bounds growth from abandoned asks instead.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid as _uuid
from contextlib import contextmanager
from pathlib import Path

from src.shared.utils import dumps_safe, setup_logging

logger = setup_logging("host.help_requests")

# Abandoned asks (never resolved/cancelled) are reaped past this age so the
# table can't grow without bound. Long enough that a genuinely-open ask is
# never swept out from under a user who stepped away for the weekend.
_MAX_AGE_SEC = 14 * 24 * 3600

_KINDS = (
    "credential_request",
    "browser_login_request",
    "browser_captcha_help_request",
)


class HelpRequests:
    """SQLite-backed registry of open help requests. One row per request_id."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        # ``:memory:`` databases are per-connection — keep one long-lived
        # connection so schema + rows survive across calls (tests use this).
        self._shared_conn: sqlite3.Connection | None = None
        if db_path == ":memory:":
            self._shared_conn = sqlite3.connect(
                db_path, isolation_level=None, check_same_thread=False,
            )
            self._shared_conn.execute("PRAGMA busy_timeout=30000")
        self._init_schema()

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
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS help_requests (
                    request_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    service TEXT,
                    name TEXT,
                    description TEXT,
                    url TEXT,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open'
                );
                CREATE INDEX IF NOT EXISTS idx_help_requests_created
                    ON help_requests (created_at);
            """)

    def _row_to_dict(self, r) -> dict:
        return {
            "request_id": r[0],
            "kind": r[1],
            "agent_id": r[2],
            "service": r[3],
            "name": r[4],
            "description": r[5],
            "url": r[6],
            "payload": json.loads(r[7]) if r[7] else {},
            "created_at": r[8],
            "status": r[9],
        }

    _COLS = (
        "request_id, kind, agent_id, service, name, description, url, "
        "payload_json, created_at, status"
    )

    # ── Core API ────────────────────────────────────────────────────

    def record(self, kind: str, agent_id: str, payload: dict) -> str:
        """Register an open help request; return its minted request_id.

        ``payload`` is stored verbatim (jsonb) and its well-known fields
        (service / name / description / url) are also hoisted into columns so
        the feed can render what + why without re-parsing.
        """
        if kind not in _KINDS:
            # Defensive: unknown kinds would never render in the panel.
            logger.warning("recording help request with unknown kind: %s", kind)
        request_id = str(_uuid.uuid4())
        now = time.time()
        payload = payload or {}
        self._safe_reap()
        with self._conn() as conn:
            conn.execute(
                f"INSERT INTO help_requests ({self._COLS}) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')",
                (
                    request_id, kind, agent_id,
                    payload.get("service"),
                    payload.get("name"),
                    payload.get("description"),
                    payload.get("url"),
                    dumps_safe(payload), now,
                ),
            )
        return request_id

    def get(self, request_id: str) -> dict | None:
        """Read an open request by id, or None if unknown/resolved."""
        with self._conn() as conn:
            row = conn.execute(
                f"SELECT {self._COLS} FROM help_requests WHERE request_id=?",
                (request_id,),
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def resolve(
        self, request_id: str, *, expected_kind: str | None = None,
        status: str = "resolved",
    ) -> dict | None:
        """Atomically claim and remove an open request.

        Serves BOTH the satisfied path (credential saved / login completed) and
        the cancelled path — the row is deleted either way; the caller decides
        which steer/event to emit. ``BEGIN IMMEDIATE`` makes the claim atomic,
        so a save racing a cancel resolves once: the winner gets the record,
        the loser gets ``None`` (and must NOT fire its side effect).

        Returns the claimed record (with ``status`` set to the caller's intent)
        or ``None`` if the id is unknown, already resolved, or — when
        ``expected_kind`` is given — of a different kind.
        """
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    f"SELECT {self._COLS} FROM help_requests WHERE request_id=?",
                    (request_id,),
                ).fetchone()
                if not row or (expected_kind and row[1] != expected_kind):
                    conn.execute("COMMIT")
                    return None
                conn.execute(
                    "DELETE FROM help_requests WHERE request_id=?", (request_id,),
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        record = self._row_to_dict(row)
        record["status"] = status
        return record

    def list_open(self) -> list[dict]:
        """Every still-open request, oldest first. Feeds the panel."""
        self._safe_reap()
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT {self._COLS} FROM help_requests ORDER BY created_at ASC",
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # ── Maintenance ────────────────────────────────────────────────

    def reap_old(self, max_age_sec: float = _MAX_AGE_SEC) -> int:
        """Delete asks older than ``max_age_sec``. Returns count deleted."""
        cutoff = time.time() - max_age_sec
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM help_requests WHERE created_at < ?", (cutoff,),
            )
            return cur.rowcount

    def _safe_reap(self) -> None:
        try:
            self.reap_old()
        except Exception as e:
            logger.debug("opportunistic help-request reap failed: %s", e)
