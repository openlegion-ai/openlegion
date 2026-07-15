"""ThreadStore — durable Team Threads conversation store (SQLite).

Phase-2 unit 2 of the agent-employee platform plan (docs/plans/
2026-07-04-agent-employee-platform-architecture.md §6 Phase 2, ratified
decision #8 / C.3-a: Team Threads REPLACE two legacy surfaces rather
than adding a third):

  * the ``MessageRouter.message_log`` in-memory deque (C.1 row 4) —
    router traffic now lands as durable ``dm`` thread messages;
  * the blackboard ``inbox/{agent}/task_event/`` back-edge feed
    (C.1 row 3) — terminal-status task events now land as ``event``
    rows on a per-task thread and are served to ``check_inbox`` via
    ``GET /mesh/agents/{id}/task-events``.

Thread kinds this unit:

  * ``channel`` — one per team, created at team create (the team's
    ``thread_ref`` points at it) and backfilled at boot;
  * ``task``    — lazy, one per task, created on the first back-edge
    event for that task;
  * ``dm``      — lazy, one per unordered agent pair
    (id ``dm:{a}:{b}`` with the pair sorted).

``scope_id`` is the EFFECTIVE team scope: a real team id, or — for solo
flows — the agent id itself (the same team-of-one convention as the
``teams/{scope}/`` blackboard prefixes and the summaries store).

The former back-edge TTL split (7-day actionable / 24h informational)
is now a pair of QUERY windows in :meth:`list_events_for` — rows are
kept until the 90-day reaper, but informational kinds stop being served
to ``check_inbox`` after 24h so they can't flood the operator's LLM
context. Plain ``message`` rows are durable (no reap this phase).

Storage follows the canonical-v1 pattern: one executescript, no lazy
ALTER chains, ``PRAGMA user_version = 1``. Disk-backed access opens a
fresh WAL connection per operation; ``:memory:`` keeps a single shared
connection behind a lock (mirrors ``TeamStore``).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from src.shared.utils import dumps_safe, setup_logging

logger = setup_logging("host.threads")

VALID_THREAD_KINDS: frozenset[str] = frozenset({"channel", "task", "dm"})
VALID_MESSAGE_KINDS: frozenset[str] = frozenset({"message", "event"})

# Back-edge event kinds the recipient must act on. Mirrors the mesh's
# ``_BACK_EDGE_WAKE_KINDS`` — these get the long (7-day) serving window
# in ``list_events_for`` so an originator can recover after a gap.
#
# Team signal ledger (Item 2): the two review outcomes that require the
# review AUTHOR to act — a rejection (rework the branch) and a
# merge-preflight failure (branch changed/deleted/conflicted, resubmit)
# — join the actionable set so Item 1's heartbeat backstop force-surfaces
# them if the direct author wake was ever missed. The other review
# outcomes (``review_merged`` / ``review_approved`` / ``review_superseded``)
# stay INFORMATIONAL on purpose: promptness for those rides the direct
# low-priority wake, not a forced heartbeat plate.
ACTIONABLE_EVENT_KINDS: frozenset[str] = frozenset(
    {"task_failed", "task_blocked", "review_rejected", "review_merge_failed"}
)

# Query windows for ``list_events_for``. These replace the old
# blackboard TTLs byte-for-byte in effect: actionable events surface
# for 7 days, informational (task_completed / task_cancelled) for 24h.
EVENT_ACTIONABLE_WINDOW_SECONDS = 604_800  # 7 days
EVENT_INFO_WINDOW_SECONDS = 86_400  # 24 hours

# Retention for event rows: reaped after 90 days (observability window —
# well past the 7-day serving window). Plain messages are never reaped
# this phase.
EVENT_RETENTION_SECONDS = 90 * 86_400

# Hard bound for one ``list_events_for`` call: the SQL LIMIT caps the
# rows scanned (newest-first) BEFORE the Python dedupe pass, and the
# returned list is sliced to the same bound. Well past anything a real
# fleet produces inside the serving windows.
EVENT_QUERY_LIMIT = 500

# Caps. Body text is truncated (with a visible notice); an oversized
# payload is REPLACED with a truncation marker because a truncated JSON
# document is worse than no document.
MAX_BODY_CHARS = 16_000
MAX_PAYLOAD_BYTES = 32 * 1024
_TRUNCATION_NOTICE = "\n…[truncated]"

# Minimum seconds between opportunistic reap runs (mirrors
# ``summaries._SAFE_REAP_MIN_INTERVAL_SECONDS``).
_SAFE_REAP_MIN_INTERVAL_SECONDS = 60


class ThreadNotFound(LookupError):
    """Raised when an operation references a thread id with no row."""


class ThreadStore:
    """SQLite-backed store for threads and thread messages."""

    def __init__(
        self,
        db_path: str = "data/threads.db",
        *,
        event_bus: Any = None,
    ) -> None:
        self.db_path = db_path
        self._event_bus = event_bus
        self._shared_conn: sqlite3.Connection | None = None
        self._mem_lock = threading.Lock()
        self._last_reap_ts: float = 0.0
        if db_path == ":memory:":
            self._shared_conn = sqlite3.connect(":memory:", isolation_level=None, check_same_thread=False)
            self._shared_conn.execute("PRAGMA busy_timeout=30000")
        else:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def close(self) -> None:
        if self._shared_conn is not None:
            self._shared_conn.close()
            self._shared_conn = None

    # ── connections / schema ─────────────────────────────────────

    @contextmanager
    def _conn(self):
        if self._shared_conn is not None:
            with self._mem_lock:
                yield self._shared_conn
            return
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        try:
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
        finally:
            conn.close()

    @contextmanager
    def _txn(self):
        """A connection wrapped in an immediate transaction (mirrors
        ``TeamStore._txn``) for multi-statement mutators."""
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")

    def _init_schema(self) -> None:
        # Canonical schema v1 — exactly one shape, no lazy ALTER chains.
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS threads (
                    id         TEXT PRIMARY KEY,
                    scope_id   TEXT NOT NULL,
                    kind       TEXT NOT NULL,
                    title      TEXT,
                    task_id    TEXT,
                    created_by TEXT,
                    created_at REAL,
                    updated_at REAL,
                    archived   INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_threads_scope
                    ON threads(scope_id, kind, updated_at DESC);

                CREATE TABLE IF NOT EXISTS thread_messages (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id  TEXT NOT NULL,
                    sender     TEXT NOT NULL,
                    recipient  TEXT,
                    kind       TEXT NOT NULL DEFAULT 'message',
                    body       TEXT,
                    payload    TEXT,
                    created_at REAL
                );
                CREATE INDEX IF NOT EXISTS idx_thread_messages_thread
                    ON thread_messages(thread_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_thread_messages_recipient
                    ON thread_messages(recipient, kind, created_at);

                -- Durable per-recipient read cursor (team signal ledger,
                -- Item 1). ``last_seen_id`` is a monotonic high-water mark
                -- over ``thread_messages.id``: everything at or below it has
                -- already been surfaced to the recipient (via the heartbeat
                -- backstop or a proactive check_inbox), so the "does this
                -- agent have UNSEEN actionable events?" probe can answer
                -- cheaply without re-firing an LLM turn for the same events
                -- on every heartbeat. Canonical v1 — created here, no ALTER.
                CREATE TABLE IF NOT EXISTS event_cursors (
                    recipient    TEXT PRIMARY KEY,
                    last_seen_id INTEGER NOT NULL DEFAULT 0,
                    updated_at   REAL
                );

                PRAGMA user_version = 1;
                """
            )

    # ── row helpers ──────────────────────────────────────────────

    _THREAD_COLS = "id, scope_id, kind, title, task_id, created_by, created_at, updated_at, archived"
    _MESSAGE_COLS = "id, thread_id, sender, recipient, kind, body, payload, created_at"

    @staticmethod
    def _row_to_thread(row: tuple) -> dict:
        return {
            "id": row[0],
            "scope_id": row[1],
            "kind": row[2],
            "title": row[3],
            "task_id": row[4],
            "created_by": row[5],
            "created_at": row[6],
            "updated_at": row[7],
            "archived": bool(row[8]),
        }

    @staticmethod
    def _row_to_message(row: tuple) -> dict:
        payload = None
        if row[6]:
            try:
                payload = json.loads(row[6])
            except (ValueError, TypeError):
                payload = {"_decode_error": True}
        return {
            "id": row[0],
            "thread_id": row[1],
            "sender": row[2],
            "recipient": row[3],
            "kind": row[4],
            "body": row[5],
            "payload": payload,
            "created_at": row[7],
        }

    def _safe_emit(self, event_type: str, agent: str, data: dict) -> None:
        """Emit a dashboard event, swallowing failures (mirrors
        ``WorkSummariesStore._safe_emit`` — an emit must never sink a
        successful DB write)."""
        bus = self._event_bus
        if bus is None:
            return
        try:
            bus.emit(event_type, agent=agent, data=data)
        except Exception as e:
            logger.debug("thread event emit failed (%s): %s", event_type, e)

    # ── thread CRUD ──────────────────────────────────────────────

    def create_thread(
        self,
        scope_id: str,
        kind: str,
        *,
        title: str | None = None,
        task_id: str | None = None,
        created_by: str | None = None,
        thread_id: str | None = None,
    ) -> dict:
        """Insert a thread row (idempotent on ``thread_id``) and return it."""
        if kind not in VALID_THREAD_KINDS:
            raise ValueError(f"kind must be one of {sorted(VALID_THREAD_KINDS)}, got {kind!r}")
        if not scope_id:
            raise ValueError("scope_id is required")
        from src.shared.utils import generate_id

        tid = thread_id or generate_id("th")
        now = time.time()
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO threads "
                "(id, scope_id, kind, title, task_id, created_by, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (tid, scope_id, kind, title, task_id, created_by, now, now),
            )
            # Revive an archived row: ensure_* against a recreated team
            # (or a DM pair re-messaging after a team delete) must
            # resurface the live thread, not leave callers pointing at
            # archived history.
            conn.execute(
                "UPDATE threads SET archived = 0 WHERE id = ? AND archived = 1",
                (tid,),
            )
        return self.get_thread(tid)  # type: ignore[return-value]

    def get_thread(self, thread_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                f"SELECT {self._THREAD_COLS} FROM threads WHERE id = ?",
                (thread_id,),
            ).fetchone()
        return self._row_to_thread(row) if row else None

    def ensure_task_thread(self, scope_id: str, task_id: str, title: str | None = None) -> dict:
        """Get-or-create the per-task thread (id ``task:{task_id}``)."""
        if not task_id:
            raise ValueError("task_id is required")
        return self.create_thread(
            scope_id, "task", title=title, task_id=task_id, thread_id=f"task:{task_id}",
        )

    def ensure_dm_thread(self, scope_id: str, a: str, b: str) -> dict:
        """Get-or-create the DM thread for an unordered agent pair."""
        if not a or not b:
            raise ValueError("both agent ids are required")
        lo, hi = sorted((a, b))
        return self.create_thread(
            scope_id, "dm", title=f"{lo} ↔ {hi}", thread_id=f"dm:{lo}:{hi}",
        )

    def ensure_channel(self, team_id: str) -> dict:
        """Get-or-create the team's channel thread (id ``channel:{team_id}``)."""
        if not team_id:
            raise ValueError("team_id is required")
        return self.create_thread(
            team_id, "channel", title=f"#{team_id}", thread_id=f"channel:{team_id}",
        )

    def list_threads(
        self,
        scope_id: str | None = None,
        kind: str | None = None,
        *,
        include_archived: bool = False,
        limit: int = 100,
    ) -> list[dict]:
        """Most-recently-updated threads first, filtered by scope/kind."""
        limit = max(1, min(int(limit), 500))
        clauses: list[str] = []
        params: list[Any] = []
        if scope_id is not None:
            clauses.append("scope_id = ?")
            params.append(scope_id)
        if kind is not None:
            if kind not in VALID_THREAD_KINDS:
                raise ValueError(f"unknown thread kind {kind!r}")
            clauses.append("kind = ?")
            params.append(kind)
        if not include_archived:
            clauses.append("archived = 0")
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT {self._THREAD_COLS} FROM threads {where} ORDER BY updated_at DESC LIMIT ?",
                params,
            ).fetchall()
        return [self._row_to_thread(r) for r in rows]

    def archive_scope(self, scope_id: str) -> int:
        """Archive (NOT delete — audit trail) every thread in a scope.

        Called on team delete: the team row is gone but its conversation
        history stays readable via ``include_archived=True``.
        """
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE threads SET archived = 1 WHERE scope_id = ? AND archived = 0",
                (scope_id,),
            )
            return cur.rowcount or 0

    # ── messages ─────────────────────────────────────────────────

    def post_message(
        self,
        thread_id: str,
        sender: str,
        *,
        recipient: str | None = None,
        kind: str = "message",
        body: str | None = None,
        payload: dict | None = None,
    ) -> dict:
        """Append a message/event row and bump the thread's updated_at.

        Emits a ``thread_message`` dashboard event (best-effort). Raises
        :class:`ThreadNotFound` for an unknown thread id so writers
        can't orphan rows.
        """
        if kind not in VALID_MESSAGE_KINDS:
            raise ValueError(f"kind must be one of {sorted(VALID_MESSAGE_KINDS)}, got {kind!r}")
        if not sender:
            raise ValueError("sender is required")
        if body is not None and len(body) > MAX_BODY_CHARS:
            body = body[: MAX_BODY_CHARS - len(_TRUNCATION_NOTICE)] + _TRUNCATION_NOTICE
        payload_json: str | None = None
        stored_payload = payload
        if payload is not None:
            payload_json = dumps_safe(payload)
            payload_bytes = len(payload_json.encode("utf-8", errors="replace"))
            if payload_bytes > MAX_PAYLOAD_BYTES:
                # A truncated JSON document is worse than no document —
                # replace the whole payload with a marker.
                stored_payload = {"truncated": True, "original_bytes": payload_bytes}
                payload_json = dumps_safe(stored_payload)
        now = time.time()
        with self._txn() as conn:
            exists = conn.execute("SELECT 1 FROM threads WHERE id = ?", (thread_id,)).fetchone()
            if exists is None:
                raise ThreadNotFound(f"Thread '{thread_id}' not found")
            cur = conn.execute(
                "INSERT INTO thread_messages "
                "(thread_id, sender, recipient, kind, body, payload, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (thread_id, sender, recipient, kind, body, payload_json, now),
            )
            message_id = cur.lastrowid
            # New traffic un-archives the thread: a DM row landing after
            # a team delete makes the conversation visible again.
            conn.execute(
                "UPDATE threads SET updated_at = ?, archived = 0 WHERE id = ?",
                (now, thread_id),
            )
        self._safe_emit(
            "thread_message",
            agent=sender,
            data={
                "thread_id": thread_id,
                "message_id": message_id,
                "sender": sender,
                "recipient": recipient,
                "kind": kind,
            },
        )
        return {
            "id": message_id,
            "thread_id": thread_id,
            "sender": sender,
            "recipient": recipient,
            "kind": kind,
            "body": body,
            "payload": stored_payload,
            "created_at": now,
        }

    def list_messages(
        self,
        thread_id: str,
        *,
        before: float | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Newest-last page of a thread's messages (``before`` pages back)."""
        limit = max(1, min(int(limit), 200))
        params: list[Any] = [thread_id]
        where = "thread_id = ?"
        if before is not None:
            where += " AND created_at < ?"
            params.append(float(before))
        params.append(limit)
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT {self._MESSAGE_COLS} FROM thread_messages "
                f"WHERE {where} ORDER BY created_at DESC, id DESC LIMIT ?",
                params,
            ).fetchall()
        return [self._row_to_message(r) for r in reversed(rows)]

    def recent_messages(self, *, kind: str = "message", limit: int = 100) -> list[dict]:
        """Most-recent rows of a kind across ALL threads (dashboard
        recent-traffic feed — replaces the router's message_log deque)."""
        limit = max(1, min(int(limit), 500))
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT {self._MESSAGE_COLS} FROM thread_messages "
                "WHERE kind = ? ORDER BY id DESC LIMIT ?",
                (kind, limit),
            ).fetchall()
        return [self._row_to_message(r) for r in reversed(rows)]

    # ── back-edge event feed ─────────────────────────────────────

    def list_events_for(
        self,
        recipient: str,
        *,
        actionable_window: float = EVENT_ACTIONABLE_WINDOW_SECONDS,
        info_window: float = EVENT_INFO_WINDOW_SECONDS,
    ) -> list[dict]:
        """Serve the back-edge event envelopes addressed to ``recipient``.

        The old blackboard back-edge was an UPSERT per (recipient,
        task): one event per task, latest transition wins, and the
        surviving row's window was re-classified on every overwrite.
        The append model stores every transition, so this read restores
        overwrite semantics: only the NEWEST row per task thread
        survives (dedupe key = the task-scoped thread id; non-task
        threads fall back to the message id, i.e. no dedupe), and the
        window classification applies to that surviving row only — a
        later informational transition (``task_completed`` after
        ``task_blocked``) silences the stale actionable event and ages
        out at 24h.

        The former TTL split is applied as query windows: actionable
        payload kinds (``task_failed`` / ``task_blocked``) surface for
        ``actionable_window`` (7 days); everything else
        (``task_completed`` / ``task_cancelled``) for ``info_window``
        (24 hours). Returns newest-first envelope dicts (the stored
        payload + ``id`` / ``thread_id`` / ``created_at``), hard-capped
        at :data:`EVENT_QUERY_LIMIT`.
        """
        self._safe_reap()
        now = time.time()
        cutoff = now - max(actionable_window, info_window)
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT {self._MESSAGE_COLS} FROM thread_messages "
                "WHERE recipient = ? AND kind = 'event' AND created_at >= ? "
                "ORDER BY created_at DESC, id DESC LIMIT ?",
                (recipient, cutoff, EVENT_QUERY_LIMIT),
            ).fetchall()
        events: list[dict] = []
        seen: set[str] = set()
        info_cutoff = now - info_window
        for r in rows:  # newest-first — the first row per key wins
            msg = self._row_to_message(r)
            thread_id = str(msg["thread_id"] or "")
            key = thread_id if thread_id.startswith("task:") else f"msg:{msg['id']}"
            if key in seen:
                continue
            # Mark BEFORE the window check: an aged-out newest row must
            # still shadow the task's older transitions (overwrite
            # semantics), not let a stale actionable event resurface.
            seen.add(key)
            payload = msg["payload"] if isinstance(msg["payload"], dict) else {}
            if payload.get("kind") not in ACTIONABLE_EVENT_KINDS and (msg["created_at"] or 0) < info_cutoff:
                continue
            envelope = dict(payload)
            envelope["id"] = msg["id"]
            envelope["thread_id"] = msg["thread_id"]
            envelope["created_at"] = msg["created_at"]
            events.append(envelope)
        return events[:EVENT_QUERY_LIMIT]

    # ── unseen-actionable cursor (team signal ledger, Item 1) ────
    #
    # ``list_events_for`` above is a WINDOW read (7d/24h) with no memory of
    # what a recipient has already seen — so the heartbeat backstop can't
    # cheaply ask "does this agent have UNSEEN actionable events?" without
    # re-firing an LLM turn for the same events every interval. The
    # ``event_cursors`` table adds that memory: ``count_unseen_actionable``
    # /``unseen_actionable_events`` read strictly ABOVE the cursor, and
    # ``mark_events_seen`` advances it (mark-on-surface). The window read
    # stays cursor-free — a proactive ``check_inbox`` still sees every event
    # in its serving window regardless of the cursor; only the heartbeat
    # re-trigger is gated.

    def _event_cursor(self, conn: sqlite3.Connection, recipient: str) -> int:
        row = conn.execute(
            "SELECT last_seen_id FROM event_cursors WHERE recipient = ?",
            (recipient,),
        ).fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    def _scan_unseen_actionable(self, recipient: str) -> list[dict]:
        """Deduped, actionable, UNSEEN event message rows for ``recipient``.

        The single shared pass behind ``count_unseen_actionable`` and
        ``unseen_actionable_events`` so the two agree exactly (count ==
        number of returned envelopes, up to the hard bound). Applies, in
        order: the cursor filter (``id > last_seen_id`` — the "unseen"
        clause), the 7-day actionable serving window, the SAME newest-per-
        task overwrite dedupe as ``list_events_for`` (a later
        ``task_completed`` shadows an earlier ``task_blocked`` for the same
        task, so a resolved blocker never nags), and finally the
        actionable-kind filter. Newest-first (by ``created_at``), bounded
        by :data:`EVENT_QUERY_LIMIT`. Mesh-side DB read only — never a
        container call, so it never cold-wakes a hibernated agent.
        """
        now = time.time()
        cutoff = now - EVENT_ACTIONABLE_WINDOW_SECONDS
        with self._conn() as conn:
            cursor = self._event_cursor(conn, recipient)
            rows = conn.execute(
                f"SELECT {self._MESSAGE_COLS} FROM thread_messages "
                "WHERE recipient = ? AND kind = 'event' AND id > ? AND created_at >= ? "
                "ORDER BY created_at DESC, id DESC LIMIT ?",
                (recipient, cursor, cutoff, EVENT_QUERY_LIMIT),
            ).fetchall()
        survivors: list[dict] = []
        seen: set[str] = set()
        for r in rows:  # newest-first — the first row per task key wins
            msg = self._row_to_message(r)
            thread_id = str(msg["thread_id"] or "")
            key = thread_id if thread_id.startswith("task:") else f"msg:{msg['id']}"
            if key in seen:
                continue
            # Mark BEFORE the actionable check (same as ``list_events_for``):
            # an informational newest row must still shadow the task's older
            # actionable transitions rather than let a stale one resurface.
            seen.add(key)
            payload = msg["payload"] if isinstance(msg["payload"], dict) else {}
            if payload.get("kind") not in ACTIONABLE_EVENT_KINDS:
                continue
            survivors.append(msg)
        return survivors

    def count_unseen_actionable(self, recipient: str) -> int:
        """Cheap count of UNSEEN actionable events for ``recipient``.

        The backstop probe's short-circuit: 0 means the heartbeat need not
        escalate on this account, so a non-lead/idle agent pays only for
        this bounded, indexed read. Consistent with
        :meth:`unseen_actionable_events` by construction (both share
        :meth:`_scan_unseen_actionable`), so ``count > 0`` iff that method
        returns a non-empty list.
        """
        return len(self._scan_unseen_actionable(recipient))

    def unseen_actionable_events(
        self, recipient: str, *, limit: int = EVENT_QUERY_LIMIT,
    ) -> list[dict]:
        """UNSEEN actionable event envelopes for ``recipient``, newest-first.

        Same envelope shape as :meth:`list_events_for` (stored payload +
        ``id`` / ``thread_id`` / ``created_at``). The caller learns the max
        id to mark as seen via ``max(e["id"] for e in <result>)`` — the
        list is ordered by ``created_at`` (not id), so use ``max`` rather
        than the first element: a backdated row can carry a higher id than a
        newer-timestamped one, and the cursor must jump past ALL surfaced
        rows to prevent a re-trigger.
        """
        limit = max(1, min(int(limit), EVENT_QUERY_LIMIT))
        events: list[dict] = []
        for msg in self._scan_unseen_actionable(recipient)[:limit]:
            payload = msg["payload"] if isinstance(msg["payload"], dict) else {}
            envelope = dict(payload)
            envelope["id"] = msg["id"]
            envelope["thread_id"] = msg["thread_id"]
            envelope["created_at"] = msg["created_at"]
            events.append(envelope)
        return events

    def mark_events_seen(self, recipient: str, up_to_id: int) -> None:
        """Advance ``recipient``'s read cursor to ``max(existing, up_to_id)``.

        Monotonic UPSERT — an out-of-order or stale mark can never regress
        the high-water mark. A non-positive ``up_to_id`` is a no-op (nothing
        to advance past). This is the storm guard: once events are surfaced
        (heartbeat backstop or proactive ``check_inbox``), they don't
        re-trigger.
        """
        try:
            up_to = int(up_to_id)
        except (TypeError, ValueError):
            return
        if up_to <= 0:
            return
        now = time.time()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO event_cursors (recipient, last_seen_id, updated_at) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(recipient) DO UPDATE SET "
                "last_seen_id = MAX(last_seen_id, excluded.last_seen_id), "
                "updated_at = excluded.updated_at",
                (recipient, up_to, now),
            )

    # ── reaper ───────────────────────────────────────────────────

    def reap_expired(self) -> int:
        """Delete EVENT rows older than the 90-day retention window.

        Plain ``message`` rows are durable — no reap this phase.
        """
        cutoff = time.time() - EVENT_RETENTION_SECONDS
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM thread_messages WHERE kind = 'event' AND created_at < ?",
                (cutoff,),
            )
            return cur.rowcount or 0

    def _safe_reap(self) -> None:
        """Opportunistic reap that never raises on the hot path
        (rate-limited; mirrors ``WorkSummariesStore._safe_reap``)."""
        now = time.time()
        if now - self._last_reap_ts < _SAFE_REAP_MIN_INTERVAL_SECONDS:
            return
        try:
            self.reap_expired()
        except Exception as e:
            logger.debug("opportunistic thread reap failed: %s", e)
        finally:
            # Stamp even on failure so a broken DB isn't hammered.
            self._last_reap_ts = now
