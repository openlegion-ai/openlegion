"""Persistent storage for operator-proposed actions awaiting human confirmation.

Each pending action has a nonce (idempotency key + propose -> confirm pairing),
a payload digest (replay protection), an origin kind (so confirm-side gates
can require ``origin_kind="human"``), and an expiry (default 300s -- same as
the legacy in-memory TTL, configurable per call).

Schema mirrors the Blackboard pattern in ``src/host/mesh.py``: SQLite WAL,
``busy_timeout=30000``, schema migration via ``executescript()``.

External contract: ``store(nonce, actor, target_kind, target_id, action_kind,
payload, origin_kind, ttl)`` returns a record dict; ``consume(nonce, *,
require_origin_kind, expected_payload_digest, confirmer)`` returns the record
or None (expired/missing/wrong-digest/wrong-origin/wrong-actor) and atomically
deletes on success.

Reaping is opportunistic: ``store`` and ``consume`` call ``reap_expired``
themselves so callers do not need to wire a periodic task. A background
reaper could be added later if pressure rises, but the opportunistic path
is enough for correctness given pending records are short-lived and bounded.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from src.shared.utils import dumps_safe, setup_logging

logger = setup_logging("host.pending_actions")

_DEFAULT_TTL_SEC = 300


def _payload_digest(payload: Any) -> str:
    """Stable digest of the proposed payload for replay protection.

    Uses ``sort_keys=True`` so structurally equivalent payloads produce
    matching digests regardless of dict iteration order, and
    ``default=str`` so datetime / Decimal / etc. don't blow up serialization.
    """
    blob = dumps_safe(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class PendingActions:
    """SQLite-backed pending-action store.

    One row per nonce. Same nonce stored twice replaces the row
    (``INSERT OR REPLACE``) -- intentional: the propose endpoint generates
    fresh UUIDs so a collision indicates the caller wants to re-propose
    the same change; replacing the prior payload is the correct semantic.

    Connections are short-lived (one per call, opened lazily via
    :meth:`_conn`). All write paths use a single SQL statement -- the only
    explicit transaction is in :meth:`consume`, which uses ``BEGIN IMMEDIATE``
    so two simultaneous consumers serialize and only one wins.
    """

    def __init__(self, db_path: str, *, event_bus=None):
        self.db_path = db_path
        # Task 9 — optional EventBus for surfacing pending-action
        # lifecycle to the dashboard. Wired in by ``create_mesh_app``
        # via :meth:`set_event_bus` (legacy callers in ``server.py``
        # construct the store before the app builds the bus).
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
        """Attach (or replace) the EventBus used for dashboard events.

        Mirrors the ``Tasks``/``Blackboard`` integration pattern.
        Idempotent — pass ``None`` to detach.
        """
        self._event_bus = bus

    def _safe_emit(self, event_type: str, agent: str, data: dict) -> None:
        """Emit a dashboard event, swallowing failures.

        Pending-action mutations are durable in SQLite; the bus emit is a
        best-effort decoration on top of the commit, so a missing bus or
        an emit failure must not propagate.
        """
        bus = self._event_bus
        if bus is None:
            return
        try:
            bus.emit(event_type, agent=agent, data=data)
        except Exception as e:
            logger.debug("PendingActions event emit failed (%s): %s", event_type, e)

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
        """Create the pending_actions table if it doesn't exist.

        Idempotent -- safe to call repeatedly (used in __init__ but also
        survives schema-evolution callers that init multiple times).

        ``summary`` and ``preview_diff`` were added so the dashboard's
        inline pending-action card can render a human-readable headline
        and a diff preview without a follow-up round-trip. They are
        nullable so older rows (and callers that don't compute them)
        remain valid.
        """
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS pending_actions (
                    nonce TEXT PRIMARY KEY,
                    actor TEXT NOT NULL,
                    target_kind TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    action_kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_digest TEXT NOT NULL,
                    origin_kind TEXT,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    summary TEXT,
                    preview_diff TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_pending_expires
                    ON pending_actions (expires_at);
                CREATE INDEX IF NOT EXISTS idx_pending_target
                    ON pending_actions (target_kind, target_id);
            """)
            # Migrate legacy databases that pre-date the summary/preview_diff
            # columns. SQLite's ``ALTER TABLE ... ADD COLUMN`` is cheap and
            # idempotent enough — wrap in try/except to swallow the
            # "duplicate column" error on already-migrated DBs.
            for col in ("summary", "preview_diff"):
                try:
                    conn.execute(
                        f"ALTER TABLE pending_actions ADD COLUMN {col} TEXT",
                    )
                except sqlite3.OperationalError:
                    pass

    # ── Core API ────────────────────────────────────────────────────

    def store(
        self,
        *,
        nonce: str,
        actor: str,
        target_kind: str,
        target_id: str,
        action_kind: str,
        payload: Any,
        origin_kind: str | None = None,
        ttl: int = _DEFAULT_TTL_SEC,
        summary: str | None = None,
        preview_diff: str | None = None,
    ) -> dict:
        """Persist a pending action. Returns the record dict.

        ``payload_digest`` is computed from a stable serialization of
        ``payload``. Confirm-side replay protection: callers can pass
        the digest of the payload they hold in their request body to
        ``consume`` via ``expected_payload_digest`` to catch tampering.

        ``summary`` is a short human-readable headline for the action
        (e.g. ``"Switch alpha's model from gpt-4o to claude-opus"``) and
        ``preview_diff`` is the multi-line unified diff (config edits
        only). Both are surfaced through ``pending_action_created`` so
        the dashboard's inline chat card can render them without a
        follow-up round-trip.
        """
        now = time.time()
        digest = _payload_digest(payload)
        expires_at = now + ttl
        # Opportunistic reap on the write path keeps the table bounded
        # without a separate scheduler. Failures are logged and swallowed
        # so a corrupt-row sweep never blocks a legitimate store.
        self._safe_reap()
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO pending_actions "
                "(nonce, actor, target_kind, target_id, action_kind, "
                "payload_json, payload_digest, origin_kind, created_at, "
                "expires_at, status, summary, preview_diff) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)",
                (
                    nonce, actor, target_kind, target_id, action_kind,
                    dumps_safe(payload), digest, origin_kind,
                    now, expires_at, summary, preview_diff,
                ),
            )
        # Task 9 — surface to the dashboard so the System > Operator
        # panel and the inline chat bubble render the new pending nonce.
        self._safe_emit(
            "pending_action_created",
            agent=actor,
            data={
                "nonce": nonce,
                "actor": actor,
                "target_kind": target_kind,
                "target_id": target_id,
                "action_kind": action_kind,
                "expires_at": expires_at,
                "summary": summary,
                "preview_diff": preview_diff,
            },
        )
        return {
            "nonce": nonce,
            "actor": actor,
            "target_kind": target_kind,
            "target_id": target_id,
            "action_kind": action_kind,
            "payload": payload,
            "payload_digest": digest,
            "origin_kind": origin_kind,
            "created_at": now,
            "expires_at": expires_at,
            "status": "pending",
            "summary": summary,
            "preview_diff": preview_diff,
        }

    def peek(self, nonce: str) -> dict | None:
        """Read a pending action without consuming it.

        Returns None for unknown or expired nonces. Does NOT delete
        expired rows -- use :meth:`reap_expired` for cleanup. Useful for
        previews / dashboard surfaces that want to display a pending
        change without committing to it.
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT actor, target_kind, target_id, action_kind, "
                "payload_json, payload_digest, origin_kind, created_at, "
                "expires_at, status, summary, preview_diff "
                "FROM pending_actions WHERE nonce=?",
                (nonce,),
            ).fetchone()
        if not row:
            return None
        if time.time() > row[8]:  # expires_at
            return None
        return {
            "nonce": nonce,
            "actor": row[0],
            "target_kind": row[1],
            "target_id": row[2],
            "action_kind": row[3],
            "payload": json.loads(row[4]),
            "payload_digest": row[5],
            "origin_kind": row[6],
            "created_at": row[7],
            "expires_at": row[8],
            "status": row[9],
            "summary": row[10],
            "preview_diff": row[11],
        }

    def consume(
        self,
        nonce: str,
        *,
        confirmer: str | None = None,
        require_origin_kind: str | None = None,
        expected_payload_digest: str | None = None,
    ) -> dict | None:
        """Atomically consume a pending action.

        Returns the record on success and deletes the row.

        Returns None on:
          * unknown nonce
          * expired (also deletes the expired row inside the same txn)
          * payload digest mismatch (replay protection -- row preserved)
          * confirmer != actor (only the proposer can confirm -- row preserved)
          * origin kind below required level (e.g. ``require="human"`` but
            row has ``"agent"`` -- row preserved)

        On a successful consume, the row is deleted in the same
        transaction so the same nonce cannot be replayed.
        """
        # Opportunistic reap on the read path. Cheap, swallowed on error.
        self._safe_reap()
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT actor, target_kind, target_id, action_kind, "
                    "payload_json, payload_digest, origin_kind, created_at, "
                    "expires_at, status, summary, preview_diff "
                    "FROM pending_actions WHERE nonce=?",
                    (nonce,),
                ).fetchone()
                if not row:
                    conn.execute("COMMIT")
                    return None
                now = time.time()
                if now > row[8]:
                    # Expired -- delete and report None
                    conn.execute(
                        "DELETE FROM pending_actions WHERE nonce=?", (nonce,),
                    )
                    conn.execute("COMMIT")
                    return None
                if expected_payload_digest and row[5] != expected_payload_digest:
                    # Replay-protection mismatch -- preserve the row so the
                    # legitimate confirmer can still consume it.
                    conn.execute("COMMIT")
                    return None
                if confirmer and row[0] != confirmer:
                    # Wrong confirmer -- preserve the row.
                    conn.execute("COMMIT")
                    return None
                if require_origin_kind and row[6] != require_origin_kind:
                    # Origin not strong enough -- preserve the row so
                    # nothing prevents a legitimate retry from the right
                    # caller, but refuse to apply.
                    conn.execute("COMMIT")
                    return None
                conn.execute(
                    "DELETE FROM pending_actions WHERE nonce=?", (nonce,),
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        # Task 9 — emit only on successful consume (the row is gone). The
        # ``status="confirmed"`` payload field distinguishes this from the
        # ``cancelled`` path emitted by :meth:`cancel`.
        resolver = confirmer or row[0]
        self._safe_emit(
            "pending_action_resolved",
            agent=resolver,
            data={
                "nonce": nonce,
                "target_kind": row[1],
                "target_id": row[2],
                "action_kind": row[3],
                "status": "confirmed",
                "resolver": resolver,
                "ts": time.time(),
            },
        )
        return {
            "nonce": nonce,
            "actor": row[0],
            "target_kind": row[1],
            "target_id": row[2],
            "action_kind": row[3],
            "payload": json.loads(row[4]),
            "payload_digest": row[5],
            "origin_kind": row[6],
            "created_at": row[7],
            "expires_at": row[8],
            "status": "consumed",
            "summary": row[10],
            "preview_diff": row[11],
        }

    def cancel(
        self,
        nonce: str,
        *,
        actor: str | None = None,
    ) -> dict | None:
        """Atomically delete a pending action without applying it.

        Used by the Task 9 ``/mesh/pending/{nonce}/cancel`` endpoint and
        the dashboard's "Cancel" button. Mirrors :meth:`consume` but
        without the digest / origin / actor gates: cancelling is the
        operator's escape hatch — they're abandoning the proposed change.

        Returns the record that was cancelled (so the caller can log
        what was abandoned), or None if the nonce was unknown / already
        expired. Emits ``pending_action_resolved`` with
        ``status="cancelled"`` on success so the dashboard can clear
        the panel and the inline chat bubble.
        """
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT actor, target_kind, target_id, action_kind, "
                    "payload_json, payload_digest, origin_kind, created_at, "
                    "expires_at, status, summary, preview_diff "
                    "FROM pending_actions WHERE nonce=?",
                    (nonce,),
                ).fetchone()
                if not row:
                    conn.execute("COMMIT")
                    return None
                if time.time() > row[8]:
                    # Expired — drop the row and report as None so the
                    # caller can distinguish "still pending" from "gone".
                    conn.execute(
                        "DELETE FROM pending_actions WHERE nonce=?", (nonce,),
                    )
                    conn.execute("COMMIT")
                    return None
                conn.execute(
                    "DELETE FROM pending_actions WHERE nonce=?", (nonce,),
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        resolver = actor or row[0]
        self._safe_emit(
            "pending_action_resolved",
            agent=resolver,
            data={
                "nonce": nonce,
                "target_kind": row[1],
                "target_id": row[2],
                "action_kind": row[3],
                "status": "cancelled",
                "resolver": resolver,
                "ts": time.time(),
            },
        )
        return {
            "nonce": nonce,
            "actor": row[0],
            "target_kind": row[1],
            "target_id": row[2],
            "action_kind": row[3],
            "payload": json.loads(row[4]),
            "payload_digest": row[5],
            "origin_kind": row[6],
            "created_at": row[7],
            "expires_at": row[8],
            "status": "cancelled",
            "summary": row[10],
            "preview_diff": row[11],
        }

    # ── Maintenance / surfacing ────────────────────────────────────

    def reap_expired(self) -> int:
        """Delete every row past its expiry. Returns count deleted.

        Task 9 — capture (target_kind, target_id, action_kind, nonce)
        of each row being dropped so the dashboard can render an
        ``pending_action_expired`` event per nonce. We snapshot before
        the DELETE so the emit list reflects what actually went away.
        """
        now = time.time()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT nonce, target_kind, target_id, action_kind "
                "FROM pending_actions WHERE expires_at < ?",
                (now,),
            ).fetchall()
            cur = conn.execute(
                "DELETE FROM pending_actions WHERE expires_at < ?",
                (now,),
            )
            count = cur.rowcount
        # Emit outside the SQLite txn so a slow listener never holds the
        # connection. ``_safe_emit`` swallows individual failures.
        for nonce, target_kind, target_id, action_kind in rows:
            self._safe_emit(
                "pending_action_expired",
                agent="",
                data={
                    "nonce": nonce,
                    "target_kind": target_kind,
                    "target_id": target_id,
                    "action_kind": action_kind,
                    "expired_at": now,
                },
            )
        return count

    def _safe_reap(self) -> None:
        """Reap-expired wrapper that never raises in a request path."""
        try:
            self.reap_expired()
        except Exception as e:
            logger.debug("opportunistic reap_expired failed: %s", e)

    def list_pending(self) -> list[dict]:
        """Return every non-expired pending action.

        Used by future dashboard / CLI surfaces to display review queues.
        Not a consume-style call -- rows are left in place.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT nonce, actor, target_kind, target_id, action_kind, "
                "payload_json, payload_digest, origin_kind, created_at, "
                "expires_at, status, summary, preview_diff "
                "FROM pending_actions "
                "WHERE expires_at >= ? ORDER BY created_at ASC",
                (time.time(),),
            ).fetchall()
        return [
            {
                "nonce": r[0],
                "actor": r[1],
                "target_kind": r[2],
                "target_id": r[3],
                "action_kind": r[4],
                "payload": json.loads(r[5]),
                "payload_digest": r[6],
                "origin_kind": r[7],
                "created_at": r[8],
                "expires_at": r[9],
                "status": r[10],
                "summary": r[11],
                "preview_diff": r[12],
            }
            for r in rows
        ]
