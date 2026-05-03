"""Durable orchestration task records (Task 6).

Replaces the legacy blackboard-dict handoff format
(``tasks/{agent_id}/{handoff_id}`` and ``global/tasks/operator/{handoff_id}``)
with a typed SQLite table. Behind ``OPENLEGION_ORCHESTRATION_TASKS_V2`` —
when enabled, ``coordination_tool`` reads/writes here; when disabled, the
existing blackboard path runs unchanged. **No dual-write.**

Schema mirrors the Blackboard pattern in ``src/host/mesh.py`` and
``src/host/pending_actions.py``: SQLite WAL, ``busy_timeout=30000``,
schema migration via ``executescript()``.

The storage layer enforces status transition validity (so direct callers
cannot corrupt state) and bounds the table by stamping
``retention_until`` on terminal transitions (default 90 days). Reaping
is opportunistic — callers (typically the inbox / project list endpoints)
invoke ``reap_expired`` on their hot path so no separate scheduler is
needed at this slice.

Audit: every state-changing call writes to the companion ``task_events``
table. Operators read this for the per-task audit history.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from src.shared.utils import setup_logging

logger = setup_logging("host.orchestration")


# ── Status machine ────────────────────────────────────────────────

VALID_STATUSES: frozenset[str] = frozenset({
    "pending", "accepted", "working", "blocked", "done", "failed", "cancelled",
})

TERMINAL_STATUSES: frozenset[str] = frozenset({"done", "failed", "cancelled"})

# Outcome enum (Task 9 PR 4) — operator-supplied judgement on a
# completed task. ``None`` means "not yet rated". Outcomes are
# write-many: every submission appends a ``task_outcome`` audit event
# and the ``tasks.outcome`` column reflects the LATEST rating, so an
# operator who clicks the wrong button can re-rate without admin
# intervention.
VALID_OUTCOMES: frozenset[str] = frozenset({"accepted", "rework", "rejected"})

# Feedback length cap (chars). Bounded so the SQLite row stays small
# and the UI textarea doesn't smuggle a multi-MB blob into the table.
MAX_FEEDBACK_CHARS: int = 2000

# Allowed status transitions. Keys are FROM, values are sets of valid TOs.
# Terminal states (done / failed / cancelled) appear here with empty sets
# so the validator's lookup never KeyErrors.
_VALID_TRANSITIONS: dict[str, frozenset[str]] = {
    "pending":   frozenset({"accepted", "working", "cancelled", "failed"}),
    "accepted":  frozenset({"working", "cancelled", "failed"}),
    "working":   frozenset({"blocked", "done", "failed", "cancelled"}),
    "blocked":   frozenset({"working", "cancelled", "failed", "done"}),
    "done":      frozenset(),
    "failed":    frozenset(),
    "cancelled": frozenset(),
}

# Default retention window for terminal rows. Operators can raise / lower
# this per-project later (see plan Task 6 §3); this slice ships the
# fleet default only.
DEFAULT_RETENTION_SECONDS: int = 90 * 24 * 60 * 60  # 90 days


class InvalidStatusTransition(ValueError):
    """Raised when a status transition is not allowed by the state machine."""


class TaskNotFound(LookupError):
    """Raised when a referenced task does not exist."""


# ── Storage layer ─────────────────────────────────────────────────


class Tasks:
    """SQLite-backed durable task record store.

    ``:memory:`` databases share a single connection (per-connection
    storage in SQLite — closing loses everything) so tests and the
    coordination_tool integration both keep state across calls. Disk-
    backed instances open and close a fresh connection per call, matching
    the ``PendingActions`` pattern.
    """

    def __init__(
        self,
        db_path: str,
        *,
        retention_seconds: int = DEFAULT_RETENTION_SECONDS,
        event_bus=None,
    ):
        self.db_path = db_path
        self.retention_seconds = retention_seconds
        self._shared_conn: sqlite3.Connection | None = None
        # Task 9 — optional EventBus for surfacing task lifecycle to the
        # dashboard. Wired in by ``create_mesh_app`` after construction
        # via :meth:`set_event_bus` so callers that build the store
        # eagerly (tests, the legacy in-memory shim path) don't need to
        # pass a bus.
        self._event_bus = event_bus
        # In-memory mode keeps a long-lived connection so schema and rows
        # survive across calls inside a single process (mirrors
        # ``PendingActions``).
        if db_path == ":memory:":
            self._shared_conn = sqlite3.connect(
                db_path, isolation_level=None, check_same_thread=False,
            )
            self._shared_conn.execute("PRAGMA busy_timeout=30000")
        # Module-level lock for the in-memory mode so multi-threaded
        # tests don't trip ``database is locked`` on the shared
        # connection.
        self._mem_lock = threading.Lock()
        self._init_schema()

    def set_event_bus(self, bus) -> None:
        """Attach (or replace) the EventBus used for dashboard events.

        Mirrors the ``Blackboard``/``HealthMonitor`` integration pattern.
        Idempotent — safe to call multiple times. Pass ``None`` to detach.
        """
        self._event_bus = bus

    def _safe_emit(self, event_type: str, agent: str, data: dict) -> None:
        """Emit a dashboard event, swallowing failures.

        The bus is best-effort decoration on top of the durable SQLite
        write — if emission fails (no bus, bus raises, dashboard is
        offline) the underlying task transaction has already committed
        and we must not propagate the error.
        """
        bus = self._event_bus
        if bus is None:
            return
        try:
            bus.emit(event_type, agent=agent, data=data)
        except Exception as e:
            logger.debug("Tasks event emit failed (%s): %s", event_type, e)

    @contextmanager
    def _conn(self):
        if self._shared_conn is not None:
            with self._mem_lock:
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
        if self._shared_conn is not None:
            self._shared_conn.close()
            self._shared_conn = None

    def _init_schema(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True) if self.db_path != ":memory:" else None
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    project_id TEXT,
                    parent_task_id TEXT,
                    title TEXT NOT NULL,
                    description TEXT,
                    creator TEXT NOT NULL,
                    assignee TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    priority INTEGER NOT NULL DEFAULT 0,
                    dependencies_json TEXT,
                    artifact_refs_json TEXT,
                    blocker_note TEXT,
                    origin_kind TEXT,
                    origin_channel TEXT,
                    origin_user TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    completed_at REAL,
                    retention_until REAL
                );
                CREATE INDEX IF NOT EXISTS idx_tasks_project_status
                    ON tasks (project_id, status);
                CREATE INDEX IF NOT EXISTS idx_tasks_assignee_status
                    ON tasks (assignee, status);
                CREATE INDEX IF NOT EXISTS idx_tasks_created
                    ON tasks (created_at);
                CREATE INDEX IF NOT EXISTS idx_tasks_retention
                    ON tasks (retention_until);

                CREATE TABLE IF NOT EXISTS task_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    event_kind TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    payload_json TEXT,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (task_id) REFERENCES tasks(id)
                );
                CREATE INDEX IF NOT EXISTS idx_task_events_task
                    ON task_events (task_id, created_at);
            """)
            # Task 9 PR 4 — outcome capture migration. Existing
            # databases predate the outcome / feedback columns; ALTER
            # TABLE ... ADD COLUMN is a metadata-only op in SQLite so
            # this is fast on populated DBs and safe to re-run after
            # the column already exists (we filter known names below).
            existing = {
                row[1]
                for row in conn.execute("PRAGMA table_info(tasks)").fetchall()
            }
            outcome_columns = (
                ("outcome", "TEXT"),
                ("feedback_text", "TEXT"),
                ("previous_task_id", "TEXT"),
            )
            for col_name, col_type in outcome_columns:
                if col_name not in existing:
                    conn.execute(
                        f"ALTER TABLE tasks ADD COLUMN {col_name} {col_type}"
                    )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_previous_task "
                "ON tasks (previous_task_id)"
            )

    # ── Helpers ─────────────────────────────────────────────────

    @staticmethod
    def _row_to_dict(row: tuple) -> dict:
        """Map a SELECT-* row to a public-facing task dict."""
        return {
            "id": row[0],
            "project_id": row[1],
            "parent_task_id": row[2],
            "title": row[3],
            "description": row[4],
            "creator": row[5],
            "assignee": row[6],
            "status": row[7],
            "priority": row[8],
            "dependencies": json.loads(row[9]) if row[9] else [],
            "artifact_refs": json.loads(row[10]) if row[10] else [],
            "blocker_note": row[11],
            "origin": (
                {
                    "kind": row[12],
                    "channel": row[13] or "",
                    "user": row[14] or "",
                }
                if row[12]
                else None
            ),
            "created_at": row[15],
            "updated_at": row[16],
            "completed_at": row[17],
            "retention_until": row[18],
            "outcome": row[19],
            "feedback_text": row[20],
            "previous_task_id": row[21],
        }

    _SELECT_COLS = (
        "id, project_id, parent_task_id, title, description, creator, "
        "assignee, status, priority, dependencies_json, artifact_refs_json, "
        "blocker_note, origin_kind, origin_channel, origin_user, "
        "created_at, updated_at, completed_at, retention_until, "
        "outcome, feedback_text, previous_task_id"
    )

    def _emit_event(
        self, conn: sqlite3.Connection, task_id: str, event_kind: str,
        actor: str, payload: dict | None,
    ) -> None:
        conn.execute(
            "INSERT INTO task_events "
            "(task_id, event_kind, actor, payload_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                task_id, event_kind, actor,
                json.dumps(payload, default=str) if payload else None,
                time.time(),
            ),
        )

    # ── Core API ────────────────────────────────────────────────

    def create(
        self,
        *,
        creator: str,
        assignee: str,
        title: str,
        description: str | None = None,
        project_id: str | None = None,
        parent_task_id: str | None = None,
        priority: int = 0,
        dependencies: list[str] | None = None,
        artifact_refs: list[str] | None = None,
        origin: dict | None = None,
        task_id: str | None = None,
    ) -> dict:
        """Insert a task. Returns the public record dict.

        ``task_id`` is generated when omitted (UUID-style ``task_<hex12>``).
        ``origin`` is a 3-key dict (``kind``, ``channel``, ``user``) — the
        same shape :class:`MessageOrigin` produces; passing the typed
        Pydantic model is the caller's responsibility (use
        ``model_dump()`` or unpack as needed).
        """
        if not title:
            raise ValueError("title is required")
        if not creator or not assignee:
            raise ValueError("creator and assignee are required")

        tid = task_id or f"task_{uuid.uuid4().hex[:12]}"
        now = time.time()
        kind = origin.get("kind") if isinstance(origin, dict) else None
        channel = origin.get("channel") if isinstance(origin, dict) else None
        user = origin.get("user") if isinstance(origin, dict) else None

        with self._conn() as conn:
            conn.execute(
                "INSERT INTO tasks "
                "(id, project_id, parent_task_id, title, description, "
                "creator, assignee, status, priority, dependencies_json, "
                "artifact_refs_json, blocker_note, origin_kind, "
                "origin_channel, origin_user, created_at, updated_at, "
                "completed_at, retention_until) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, NULL, "
                "?, ?, ?, ?, ?, NULL, NULL)",
                (
                    tid, project_id, parent_task_id, title, description,
                    creator, assignee, priority,
                    json.dumps(dependencies) if dependencies else None,
                    json.dumps(artifact_refs) if artifact_refs else None,
                    kind, channel, user,
                    now, now,
                ),
            )
            self._emit_event(
                conn, tid, "created", creator,
                {
                    "title": title,
                    "assignee": assignee,
                    "project_id": project_id,
                },
            )
        # Task 9 — surface to the dashboard. Compact payload (title, no
        # description) so WS frames stay small.
        self._safe_emit(
            "task_created",
            agent=creator,
            data={
                "task_id": tid,
                "project_id": project_id,
                "creator": creator,
                "assignee": assignee,
                "title": title,
                "status": "pending",
                "created_at": now,
            },
        )
        return self.get(tid)  # type: ignore[return-value]

    def get(self, task_id: str) -> dict | None:
        """Read a task by id. Returns None when not found."""
        with self._conn() as conn:
            row = conn.execute(
                f"SELECT {self._SELECT_COLS} FROM tasks WHERE id=?",
                (task_id,),
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def list_inbox(
        self,
        assignee: str,
        *,
        project_id: str | None = None,
        include_terminal: bool = False,
    ) -> list[dict]:
        """List tasks assigned to ``assignee``.

        By default, terminal tasks (done / failed / cancelled) are excluded
        — matches the blackboard ``check_inbox`` semantics. Pass
        ``include_terminal=True`` to see history. ``project_id`` further
        narrows to a single project.
        """
        clauses = ["assignee = ?"]
        params: list[Any] = [assignee]
        if project_id is not None:
            clauses.append("project_id = ?")
            params.append(project_id)
        if not include_terminal:
            placeholders = ",".join("?" * len(TERMINAL_STATUSES))
            clauses.append(f"status NOT IN ({placeholders})")
            params.extend(sorted(TERMINAL_STATUSES))
        sql = (
            f"SELECT {self._SELECT_COLS} FROM tasks "
            f"WHERE {' AND '.join(clauses)} ORDER BY created_at ASC"
        )
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def list_project(
        self,
        project_id: str,
        *,
        statuses: list[str] | None = None,
    ) -> list[dict]:
        """List tasks in a project. Optional ``statuses`` filter."""
        clauses = ["project_id = ?"]
        params: list[Any] = [project_id]
        if statuses:
            invalid = [s for s in statuses if s not in VALID_STATUSES]
            if invalid:
                raise ValueError(f"unknown status filter: {invalid}")
            placeholders = ",".join("?" * len(statuses))
            clauses.append(f"status IN ({placeholders})")
            params.extend(statuses)
        sql = (
            f"SELECT {self._SELECT_COLS} FROM tasks "
            f"WHERE {' AND '.join(clauses)} ORDER BY created_at ASC"
        )
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def list_by_creator(
        self,
        creator: str,
        *,
        include_terminal: bool = True,
    ) -> list[dict]:
        """List tasks created by ``creator``. Used by sender-side completion checks."""
        clauses = ["creator = ?"]
        params: list[Any] = [creator]
        if not include_terminal:
            placeholders = ",".join("?" * len(TERMINAL_STATUSES))
            clauses.append(f"status NOT IN ({placeholders})")
            params.extend(sorted(TERMINAL_STATUSES))
        sql = (
            f"SELECT {self._SELECT_COLS} FROM tasks "
            f"WHERE {' AND '.join(clauses)} ORDER BY created_at ASC"
        )
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def update_status(
        self,
        task_id: str,
        status: str,
        *,
        actor: str,
        blocker_note: str | None = None,
    ) -> dict:
        """Atomically transition a task to a new status.

        Validates ``status`` against the state machine. Sets
        ``completed_at`` and ``retention_until`` on terminal transitions.
        Emits a ``status_changed`` event row.

        Raises :class:`TaskNotFound` for unknown ids,
        :class:`InvalidStatusTransition` for forbidden transitions, and
        :class:`ValueError` for unknown ``status`` values.
        """
        if status not in VALID_STATUSES:
            raise ValueError(f"unknown status: {status!r}")
        now = time.time()
        # Captured outside the txn so the event-emit (which goes through
        # the bus / asyncio loop) doesn't run while we hold BEGIN IMMEDIATE.
        emitted_change: tuple[str, str, str | None, str | None] | None = None
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT status, project_id, assignee FROM tasks WHERE id=?",
                    (task_id,),
                ).fetchone()
                if not row:
                    conn.execute("ROLLBACK")
                    raise TaskNotFound(task_id)
                current, project_id, assignee = row[0], row[1], row[2]
                if current == status:
                    # No-op transition. Record the event so the audit
                    # trail still shows the call, but skip the row update.
                    self._emit_event(
                        conn, task_id, "status_unchanged", actor,
                        {"status": status},
                    )
                    conn.execute("COMMIT")
                    return self.get(task_id)  # type: ignore[return-value]
                allowed = _VALID_TRANSITIONS.get(current, frozenset())
                if status not in allowed:
                    conn.execute("ROLLBACK")
                    raise InvalidStatusTransition(
                        f"cannot move {task_id} from {current!r} to {status!r}"
                    )
                completed_at = now if status in TERMINAL_STATUSES else None
                retention_until = (
                    now + self.retention_seconds
                    if status in TERMINAL_STATUSES
                    else None
                )
                conn.execute(
                    "UPDATE tasks SET status=?, updated_at=?, "
                    "completed_at=COALESCE(?, completed_at), "
                    "retention_until=COALESCE(?, retention_until), "
                    "blocker_note=? WHERE id=?",
                    (
                        status, now, completed_at, retention_until,
                        blocker_note if status == "blocked" else None,
                        task_id,
                    ),
                )
                self._emit_event(
                    conn, task_id, "status_changed", actor,
                    {
                        "from": current,
                        "to": status,
                        "blocker_note": blocker_note,
                    },
                )
                conn.execute("COMMIT")
                emitted_change = (current, status, project_id, assignee)
            except (TaskNotFound, InvalidStatusTransition):
                raise
            except Exception:
                conn.execute("ROLLBACK")
                raise
        if emitted_change is not None:
            old_status, new_status, project_id, assignee = emitted_change
            self._safe_emit(
                "task_status_changed",
                agent=actor,
                data={
                    "task_id": task_id,
                    "project_id": project_id,
                    "assignee": assignee,
                    "old_status": old_status,
                    "new_status": new_status,
                    "actor": actor,
                    "ts": now,
                },
            )
        return self.get(task_id)  # type: ignore[return-value]

    def reroute(
        self, task_id: str, new_assignee: str, *, actor: str, reason: str = "",
    ) -> dict:
        """Reassign a task to ``new_assignee``. Emits a ``rerouted`` event."""
        if not new_assignee:
            raise ValueError("new_assignee is required")
        now = time.time()
        emitted: tuple[str, str | None] | None = None
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT assignee, status, project_id FROM tasks WHERE id=?",
                    (task_id,),
                ).fetchone()
                if not row:
                    conn.execute("ROLLBACK")
                    raise TaskNotFound(task_id)
                old_assignee, current_status, project_id = row
                if current_status in TERMINAL_STATUSES:
                    conn.execute("ROLLBACK")
                    raise InvalidStatusTransition(
                        f"cannot reroute terminal task {task_id} (status={current_status!r})"
                    )
                conn.execute(
                    "UPDATE tasks SET assignee=?, updated_at=? WHERE id=?",
                    (new_assignee, now, task_id),
                )
                self._emit_event(
                    conn, task_id, "rerouted", actor,
                    {"from": old_assignee, "to": new_assignee, "reason": reason},
                )
                conn.execute("COMMIT")
                emitted = (current_status, project_id)
            except (TaskNotFound, InvalidStatusTransition):
                raise
            except Exception:
                conn.execute("ROLLBACK")
                raise
        if emitted is not None:
            current_status, project_id = emitted
            # Task 9 — reroute is a status_changed event with old==new
            # status; the assignee field carries the *new* assignee so
            # the dashboard sees who picked up the work. The audit row
            # already records ``rerouted`` separately for full history.
            self._safe_emit(
                "task_status_changed",
                agent=actor,
                data={
                    "task_id": task_id,
                    "project_id": project_id,
                    "assignee": new_assignee,
                    "old_status": current_status,
                    "new_status": current_status,
                    "actor": actor,
                    "ts": now,
                },
            )
        return self.get(task_id)  # type: ignore[return-value]

    def cancel(
        self, task_id: str, *, actor: str, reason: str = "",
    ) -> dict:
        """Cancel a task. Convenience wrapper over ``update_status('cancelled')``.

        Carries the cancel ``reason`` on the audit event payload alongside
        the status_changed event.
        """
        result = self.update_status(task_id, "cancelled", actor=actor)
        # Also record the explicit cancel event so the reason is preserved.
        with self._conn() as conn:
            self._emit_event(
                conn, task_id, "cancelled", actor, {"reason": reason},
            )
        return result

    def add_artifact(
        self, task_id: str, ref: str, *, actor: str,
    ) -> dict:
        """Append an artifact ref to a task's ``artifact_refs`` list."""
        if not ref:
            raise ValueError("ref is required")
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT artifact_refs_json FROM tasks WHERE id=?",
                    (task_id,),
                ).fetchone()
                if not row:
                    conn.execute("ROLLBACK")
                    raise TaskNotFound(task_id)
                refs = json.loads(row[0]) if row[0] else []
                if ref not in refs:
                    refs.append(ref)
                conn.execute(
                    "UPDATE tasks SET artifact_refs_json=?, updated_at=? "
                    "WHERE id=?",
                    (json.dumps(refs), time.time(), task_id),
                )
                self._emit_event(
                    conn, task_id, "artifact_added", actor, {"ref": ref},
                )
                conn.execute("COMMIT")
            except TaskNotFound:
                raise
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return self.get(task_id)  # type: ignore[return-value]

    # ── Outcome capture (Task 9 PR 4) ───────────────────────────

    def set_outcome(
        self,
        task_id: str,
        outcome: str,
        feedback_text: str | None = None,
        *,
        actor: str = "operator",
    ) -> dict:
        """Record an operator outcome rating for a completed task.

        ``outcome`` must be one of :data:`VALID_OUTCOMES`. The task must
        already be in a terminal status (``done`` / ``failed`` /
        ``cancelled``) — outcome ratings only make sense for finished
        work.

        Outcomes are write-many: re-rating overwrites ``tasks.outcome``
        and ``tasks.feedback_text`` with the latest values and appends
        a fresh ``task_outcome`` event row so the audit trail records
        every submission. The emitted bus payload includes
        ``previous_outcome`` so dashboards can render "re-rated"
        affordances.

        Emits a ``task_outcome`` audit event row + a ``task_outcome``
        EventBus notification so the dashboard can update the modal /
        kanban without a full reload.
        """
        if outcome not in VALID_OUTCOMES:
            raise ValueError(
                f"unknown outcome: {outcome!r} "
                f"(expected one of {sorted(VALID_OUTCOMES)})"
            )
        feedback = (feedback_text or "").strip() or None
        if feedback is not None and len(feedback) > MAX_FEEDBACK_CHARS:
            raise ValueError(
                f"feedback_text exceeds {MAX_FEEDBACK_CHARS} chars"
            )
        now = time.time()
        emitted: tuple[str, str | None, str | None, str | None] | None = None
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT status, outcome, project_id, assignee "
                    "FROM tasks WHERE id=?",
                    (task_id,),
                ).fetchone()
                if not row:
                    conn.execute("ROLLBACK")
                    raise TaskNotFound(task_id)
                current_status, current_outcome, project_id, assignee = row
                if current_status not in TERMINAL_STATUSES:
                    conn.execute("ROLLBACK")
                    raise InvalidStatusTransition(
                        f"cannot set outcome on non-terminal task {task_id} "
                        f"(status={current_status!r})"
                    )
                conn.execute(
                    "UPDATE tasks SET outcome=?, feedback_text=?, "
                    "updated_at=? WHERE id=?",
                    (outcome, feedback, now, task_id),
                )
                self._emit_event(
                    conn, task_id, "task_outcome", actor,
                    {
                        "outcome": outcome,
                        "feedback": feedback,
                        "previous_outcome": current_outcome,
                    },
                )
                conn.execute("COMMIT")
                emitted = (current_status, project_id, assignee, current_outcome)
            except (TaskNotFound, InvalidStatusTransition):
                raise
            except Exception:
                conn.execute("ROLLBACK")
                raise
        if emitted is not None:
            current_status, project_id, assignee, previous_outcome = emitted
            self._safe_emit(
                "task_outcome",
                agent=actor,
                data={
                    "task_id": task_id,
                    "project_id": project_id,
                    "assignee": assignee,
                    "status": current_status,
                    "outcome": outcome,
                    "previous_outcome": previous_outcome,
                    "feedback": feedback,
                    "actor": actor,
                    "ts": now,
                },
            )
        return self.get(task_id)  # type: ignore[return-value]

    def create_rework_task(
        self,
        previous_task_id: str,
        feedback_text: str,
        *,
        actor: str = "operator",
    ) -> dict:
        """Spawn a new task from a "needs rework" outcome.

        Inherits ``assignee`` and ``project_id`` from ``previous_task_id``
        so the same agent picks up the redo. The new task's title is
        ``"Rework: {previous_title}"`` and its description (the brief
        the agent reads) is the operator's feedback. ``previous_task_id``
        is set on the new row so the lineage is queryable.

        Raises :class:`TaskNotFound` if the source task does not exist
        and :class:`ValueError` if the feedback is empty / oversized.
        """
        feedback = (feedback_text or "").strip()
        if not feedback:
            raise ValueError("feedback_text is required for rework")
        if len(feedback) > MAX_FEEDBACK_CHARS:
            raise ValueError(
                f"feedback_text exceeds {MAX_FEEDBACK_CHARS} chars"
            )
        previous = self.get(previous_task_id)
        if previous is None:
            raise TaskNotFound(previous_task_id)
        new_title = f"Rework: {previous['title']}"[:200]
        new_id = f"task_{uuid.uuid4().hex[:12]}"
        now = time.time()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO tasks "
                "(id, project_id, parent_task_id, title, description, "
                "creator, assignee, status, priority, dependencies_json, "
                "artifact_refs_json, blocker_note, origin_kind, "
                "origin_channel, origin_user, created_at, updated_at, "
                "completed_at, retention_until, outcome, feedback_text, "
                "previous_task_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, NULL, NULL, "
                "NULL, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, ?)",
                (
                    new_id,
                    previous.get("project_id"),
                    previous.get("parent_task_id"),
                    new_title,
                    feedback,
                    actor,
                    previous["assignee"],
                    previous.get("priority", 0),
                    (previous.get("origin") or {}).get("kind"),
                    (previous.get("origin") or {}).get("channel"),
                    (previous.get("origin") or {}).get("user"),
                    now, now,
                    previous_task_id,
                ),
            )
            self._emit_event(
                conn, new_id, "created", actor,
                {
                    "title": new_title,
                    "assignee": previous["assignee"],
                    "project_id": previous.get("project_id"),
                    "previous_task_id": previous_task_id,
                    "kind": "rework",
                },
            )
        # Surface to the dashboard so the kanban / activity feed
        # picks up the new card without polling.
        self._safe_emit(
            "task_created",
            agent=actor,
            data={
                "task_id": new_id,
                "project_id": previous.get("project_id"),
                "creator": actor,
                "assignee": previous["assignee"],
                "title": new_title,
                "status": "pending",
                "previous_task_id": previous_task_id,
                "created_at": now,
            },
        )
        return self.get(new_id)  # type: ignore[return-value]

    def list_events(self, task_id: str) -> list[dict]:
        """Return audit events for ``task_id`` ordered oldest-first."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT event_id, task_id, event_kind, actor, payload_json, "
                "created_at FROM task_events "
                "WHERE task_id=? ORDER BY created_at ASC, event_id ASC",
                (task_id,),
            ).fetchall()
        return [
            {
                "event_id": r[0],
                "task_id": r[1],
                "event_kind": r[2],
                "actor": r[3],
                "payload": json.loads(r[4]) if r[4] else None,
                "created_at": r[5],
            }
            for r in rows
        ]

    def reap_expired(self) -> int:
        """Delete tasks whose ``retention_until`` is past. Returns count.

        Also drops orphaned ``task_events`` rows for the deleted ids in
        the same transaction so the audit table doesn't grow forever
        after retention drops the parent task.
        """
        now = time.time()
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                ids = [
                    r[0] for r in conn.execute(
                        "SELECT id FROM tasks "
                        "WHERE retention_until IS NOT NULL "
                        "AND retention_until < ?",
                        (now,),
                    ).fetchall()
                ]
                if not ids:
                    conn.execute("COMMIT")
                    return 0
                placeholders = ",".join("?" * len(ids))
                conn.execute(
                    f"DELETE FROM task_events WHERE task_id IN ({placeholders})",
                    ids,
                )
                conn.execute(
                    f"DELETE FROM tasks WHERE id IN ({placeholders})", ids,
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return len(ids)

    def _safe_reap(self) -> None:
        """Reap-expired wrapper that never raises on a hot path."""
        try:
            self.reap_expired()
        except Exception as e:
            logger.debug("opportunistic reap_expired failed: %s", e)
