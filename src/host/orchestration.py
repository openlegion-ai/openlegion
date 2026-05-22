"""Durable orchestration task records.

Replaces the legacy blackboard-dict handoff format
(``tasks/{agent_id}/{handoff_id}`` and ``global/tasks/operator/{handoff_id}``)
with a typed SQLite table. ``coordination_tool`` reads/writes here for
all of hand_off / check_inbox / update_status / complete_task. The
legacy blackboard path was sunset after the v2 rollout — see
``orchestration_migration.py`` for the one-shot data conversion that
runs at mesh startup.

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.shared.task_titles import normalize_title_and_description
from src.shared.utils import dumps_safe, setup_logging

logger = setup_logging("host.orchestration")


# ── Status machine ────────────────────────────────────────────────

VALID_STATUSES: frozenset[str] = frozenset({
    "pending", "accepted", "working", "blocked", "done", "failed", "cancelled",
})

TERMINAL_STATUSES: frozenset[str] = frozenset({"done", "failed", "cancelled"})

# Outcome enum — operator-supplied judgement on a completed task.
# ``None`` means "not yet rated". Outcomes are write-many: every
# submission appends a ``task_outcome`` audit event and the
# ``tasks.outcome`` column reflects the LATEST rating, so a re-rate just
# overwrites without admin intervention.
#
# - ``accepted``: positive signal. Writes a reinforcement memory entry.
# - ``rework``: negative signal with a fix-it brief. Auto-spawns a
#   rework task AND writes the feedback into the rated agent's memory
#   so the agent recalls it on the next task.
# - ``rejected``: terminal negative. Writes feedback to memory; does
#   NOT auto-spawn a rework.
# - ``acknowledged``: neutral, "reviewed without judgement". Does not
#   write to memory and does not spawn rework.
VALID_OUTCOMES: frozenset[str] = frozenset(
    {"accepted", "rework", "rejected", "acknowledged"}
)

# Feedback length cap (chars). Bounded so the SQLite row stays small
# and the UI textarea doesn't smuggle a multi-MB blob into the table.
MAX_FEEDBACK_CHARS: int = 2000

# Title length policy lives in ``src.shared.task_titles`` so the agent
# coordination tool can import it without crossing the trust boundary.
# See ``normalize_title_and_description`` (imported above) — applied in
# ``Tasks.create`` as the authoritative server-side check.

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

# Defense-in-depth cap on the recursive walk inside
# ``Tasks.workflow_snapshot``. Caps the inner CTE size before the outer
# LIMIT trims, so a pathological wide chain can't materialize a huge
# intermediate result set. Cycles are impossible by construction
# (parent_task_id is set at creation, never updated); this is purely a
# memory bound on legitimate workloads. 200 is well above any plausible
# real workflow (typical chains are 3–8 stages) but well below the
# point at which the temporary chain table starts to matter.
MAX_WORKFLOW_CHAIN_DEPTH: int = 200


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
        # Active name of the team-membership column on the ``tasks``
        # table. PR 2 created the column as ``project_id``; PR 3
        # renames it to ``team_id`` via :mod:`src.host.team_migration`
        # at startup (now default-on). When the rename hasn't run yet
        # (e.g. a customer with ``OPENLEGION_TEAM_MIGRATION_RENAME_DB=0``,
        # or a stale DB picked up before the migration fired) this
        # attribute drops back to ``project_id`` so existing reads
        # / writes keep working. SQL string builders below splice
        # ``self._team_col`` into the query — never hard-code the
        # column name.
        self._team_col = "team_id"
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
            # Fresh schemas land with ``team_id`` (post-rename name). The
            # ``CREATE TABLE IF NOT EXISTS`` is a no-op on pre-rename
            # databases that still have ``project_id`` — the actual
            # column-name resolution happens below via the PRAGMA
            # introspection step.
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    team_id TEXT,
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
                CREATE INDEX IF NOT EXISTS idx_tasks_assignee_status
                    ON tasks (assignee, status);
                CREATE INDEX IF NOT EXISTS idx_tasks_created
                    ON tasks (created_at);
                CREATE INDEX IF NOT EXISTS idx_tasks_retention
                    ON tasks (retention_until);
                CREATE INDEX IF NOT EXISTS idx_tasks_parent_task_id
                    ON tasks (parent_task_id);

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
            # Resolve the active column name. ``team_id`` is canonical;
            # ``project_id`` is the pre-rename column kept readable so
            # downgrades / opted-out instances still work. The composite
            # status index is rebuilt against whichever column is live.
            existing_cols = {
                row[1]
                for row in conn.execute("PRAGMA table_info(tasks)").fetchall()
            }
            if "team_id" in existing_cols:
                self._team_col = "team_id"
                conn.execute("DROP INDEX IF EXISTS idx_tasks_project_status")
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_tasks_team_status "
                    "ON tasks (team_id, status)"
                )
            elif "project_id" in existing_cols:
                self._team_col = "project_id"
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_tasks_project_status "
                    "ON tasks (project_id, status)"
                )
            else:  # pragma: no cover — defensive: schema in a broken state
                self._team_col = "team_id"
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
            # PR-U — outcome_set_at column. Tracks when set_outcome was
            # called, separately from completed_at, so the heartbeat
            # query (count_outcomes_since) doesn't undercount lagged
            # operator reviews. See _ensure_outcome_set_at_column for
            # migration / backfill semantics.
            self._ensure_outcome_set_at_column(conn)
            # Partial index on (outcome, outcome_set_at). Most rows have
            # outcome IS NULL (work isn't rated yet), so the partial
            # predicate keeps the index small and fast for the
            # heartbeat's per-outcome scan.
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_outcome_set_at "
                "ON tasks (outcome, outcome_set_at) "
                "WHERE outcome IS NOT NULL"
            )

    @staticmethod
    def _ensure_outcome_set_at_column(conn: sqlite3.Connection) -> None:
        """PR-U: add outcome_set_at column if missing. Idempotent.

        Both the ALTER and the backfill UPDATE are safe to run repeatedly:
        the ALTER is gated by PRAGMA inspection (with a duplicate-column
        catch for concurrent-init races, mirroring the pattern in
        ``mesh.py``); the UPDATE is WHERE-guarded so already-stamped rows
        are no-ops. This shape recovers from a mid-init crash where the
        ALTER committed but the backfill did not — re-init still walks
        the backfill UPDATE instead of short-circuiting on the column
        being present. Rows with both ``outcome`` and ``completed_at``
        NULL are left alone (anomalous; better to not invent data).
        """
        cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(tasks)").fetchall()
        }
        if "outcome_set_at" not in cols:
            try:
                conn.execute(
                    "ALTER TABLE tasks ADD COLUMN outcome_set_at REAL"
                )
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise
                # Another process won the race; column exists — proceed
                # to backfill.
        # Always run backfill — WHERE clause makes it idempotent on
        # already-stamped rows. Recovers from prior partial migrations
        # (ALTER committed, UPDATE didn't) on a subsequent re-init.
        conn.execute(
            "UPDATE tasks SET outcome_set_at = completed_at "
            "WHERE outcome IS NOT NULL AND outcome_set_at IS NULL "
            "AND completed_at IS NOT NULL"
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
            "outcome_set_at": row[22],
        }

    # ``{team_col}`` is filled in at query time from ``self._team_col``
    # (``team_id`` post-rename, ``project_id`` on pre-rename DBs). The
    # public dict key from :meth:`_row_to_dict` stays ``"project_id"``
    # for API back-compat — callers and tests still read that key.
    _SELECT_COLS_TEMPLATE = (
        "id, {team_col}, parent_task_id, title, description, creator, "
        "assignee, status, priority, dependencies_json, artifact_refs_json, "
        "blocker_note, origin_kind, origin_channel, origin_user, "
        "created_at, updated_at, completed_at, retention_until, "
        "outcome, feedback_text, previous_task_id, outcome_set_at"
    )

    @property
    def _SELECT_COLS(self) -> str:
        return self._SELECT_COLS_TEMPLATE.format(team_col=self._team_col)

    def _assert_stored_matches(
        self, record: dict, *, expected: dict, context: str,
    ) -> None:
        """Post-write integrity check: stored values must equal ``expected``.

        Centralised so every task-creation path (``create`` /
        ``create_rework_task`` / the retry endpoint's direct ``create``
        call) gets the same defence against a row landing with mistyped
        ``assignee`` / ``team_id`` / ``parent_task_id`` / ``status``.
        Corruption at this layer is exactly what made Bug 1's silent
        handoff drop possible — ``list_task_inbox`` does a byte-exact
        SQLite ``=`` match, so a single divergence here breaks the
        recipient's whole inbox flow without the caller ever knowing.

        Raises ``RuntimeError`` on any divergence so the surrounding
        endpoint converts to 500 + the caller's ``hand_off`` envelope
        fires. ``context`` is a short label spliced into log + error
        for cross-log correlation.
        """
        mismatches: list[str] = []
        for field, expected_value in expected.items():
            stored = record.get(field)
            if stored != expected_value:
                mismatches.append(
                    f"{field}: stored={stored!r} expected={expected_value!r}"
                )
        if mismatches:
            tid = record.get("id") or "<no id>"
            msg = (
                f"task {tid!r} post-write verify failed in {context}: "
                + "; ".join(mismatches)
            )
            logger.error(msg)
            raise RuntimeError(msg)

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
                dumps_safe(payload) if payload else None,
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
        # Strip leading/trailing whitespace before any further checks so
        # a whitespace-only title (audit edge case: 200 spaces) is
        # rejected the same way as an empty string. Without this the
        # normalizer's fallback path could produce a degenerate title
        # like "…" or a useless single-character header.
        title = (title or "").strip()
        if not title:
            raise ValueError("title is required")
        if not creator or not assignee:
            raise ValueError("creator and assignee are required")

        # Defensive title normalization. Agents have shipped 250-char
        # "titles" containing full instructions, which renders as wall-
        # of-text in the dashboard kanban. Apply the title-length policy
        # here so every task in the system stays bounded regardless of
        # which call path created it.
        title, description = normalize_title_and_description(title, description)

        tid = task_id or f"task_{uuid.uuid4().hex[:12]}"
        now = time.time()
        kind = origin.get("kind") if isinstance(origin, dict) else None
        channel = origin.get("channel") if isinstance(origin, dict) else None
        user = origin.get("user") if isinstance(origin, dict) else None

        with self._conn() as conn:
            conn.execute(
                f"INSERT INTO tasks "
                f"(id, {self._team_col}, parent_task_id, title, description, "
                f"creator, assignee, status, priority, dependencies_json, "
                f"artifact_refs_json, blocker_note, origin_kind, "
                f"origin_channel, origin_user, created_at, updated_at, "
                f"completed_at, retention_until) "
                f"VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, NULL, "
                f"?, ?, ?, ?, ?, NULL, NULL)",
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
        # Post-write assert. Under ``isolation_level=None`` (autocommit)
        # the INSERT above is durable before we re-read. If ``get(tid)``
        # returns ``None`` here something genuinely surprising happened —
        # a file-handle race, a mid-flight schema migration, a downgrade
        # path that dropped the row. Don't return ``None`` to the caller
        # (which then surfaces as ``null`` JSON and an ``AttributeError``
        # inside the agent's ``hand_off``). Raise loudly so the endpoint
        # returns 5xx and the agent's ``create_failed`` envelope fires.
        record = self.get(tid)
        if record is None:
            raise RuntimeError(
                f"task {tid!r} INSERT committed but post-read returned no "
                "row — possible storage corruption or mid-migration race"
            )
        # Centralised post-write integrity check (Bug 1 R2). All task-
        # creation paths route through this assert so a row landing with
        # mistyped assignee / team_id / parent_task_id / status raises
        # loudly instead of silently breaking ``list_task_inbox``.
        self._assert_stored_matches(
            record,
            expected={
                "assignee": assignee,
                "creator": creator,
                "project_id": project_id,
                "parent_task_id": parent_task_id,
                "status": "pending",
            },
            context="create",
        )
        # Bug 1 post-mortem evidence: log the canonical stored values so
        # the next repro of a silent handoff drop has authoritative data
        # on what landed in the DB vs what the caller intended. Tagged
        # with ``tid`` for cross-log correlation.
        logger.info(
            "tasks.create stored task=%s creator=%s assignee=%s "
            "team_id=%s parent_task_id=%s status=%s title=%r",
            tid, creator, assignee,
            record.get("project_id"), parent_task_id,
            record.get("status"), (title or "")[:80],
        )
        return record

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
            clauses.append(f"{self._team_col} = ?")
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
        result = [self._row_to_dict(r) for r in rows]
        # Round-4 forensic logging: pairs with ``tasks.create stored`` so
        # an operator running an E2E test can see exactly what the
        # recipient's inbox lookup compared against. ``rows=N`` over the
        # same lookup paired with the create log gives the silent-drop
        # bisect: if create stored ``assignee=X`` and list_inbox queries
        # ``assignee=Y`` the mismatch falls out instantly.
        logger.info(
            "tasks.list_inbox assignee=%r project_id=%r "
            "include_terminal=%s rows=%d team_col=%s",
            assignee, project_id, include_terminal,
            len(result), self._team_col,
        )
        return result

    def list_project(
        self,
        project_id: str,
        *,
        statuses: list[str] | None = None,
    ) -> list[dict]:
        """List tasks in a project. Optional ``statuses`` filter."""
        clauses = [f"{self._team_col} = ?"]
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

    # ── Per-agent aggregates for the operator heartbeat (PR-J') ─────
    #
    # The system_metrics endpoint surfaces per-agent counts so the
    # operator's heartbeat playbook can drill in only when something
    # actually broke. Counts (not rates) so single-task agents don't
    # produce noisy 100% denominators. Operator exclusion is handled
    # by the caller — these helpers run unfiltered and the caller
    # drops ``operator`` from the dict before serialising.

    def count_outcomes_since(
        self,
        outcome: str,
        *,
        since_seconds: float,
    ) -> dict[str, int]:
        """Count tasks per assignee whose latest ``outcome`` was set within ``since_seconds``.

        PR-U: filters on ``outcome_set_at`` (when the operator clicked
        the rating button) rather than ``completed_at`` (when the agent
        finished). The two diverge whenever review is delayed — a task
        completed Monday and rated rejected on Wednesday must show up
        in the Wednesday heartbeat, not Monday's. Tasks rated before
        the cutoff (or never rated) are excluded by the >= filter.
        """
        cutoff = time.time() - since_seconds
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT assignee, COUNT(*) FROM tasks "
                "WHERE outcome = ? AND outcome_set_at IS NOT NULL "
                "AND outcome_set_at >= ? GROUP BY assignee",
                (outcome, cutoff),
            ).fetchall()
        return {row[0]: int(row[1] or 0) for row in rows if row[0]}

    def count_failed_status_since(
        self,
        *,
        since_seconds: float,
    ) -> dict[str, int]:
        """Count tasks per assignee that landed in ``failed`` within ``since_seconds``.

        ``status='failed'`` always sets ``completed_at`` (terminal
        transitions stamp both), so we filter on ``completed_at`` rather
        than ``updated_at`` — keeps the metric stable even if an
        operator no-op-edits a failed task later.
        """
        cutoff = time.time() - since_seconds
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT assignee, COUNT(*) FROM tasks "
                "WHERE status = 'failed' AND completed_at IS NOT NULL "
                "AND completed_at >= ? GROUP BY assignee",
                (cutoff,),
            ).fetchall()
        return {row[0]: int(row[1] or 0) for row in rows if row[0]}

    def count_stale_since(
        self,
        *,
        threshold_seconds: float,
    ) -> dict[str, int]:
        """Count non-terminal tasks per assignee created more than ``threshold_seconds`` ago.

        Stale = open work that's been sitting around longer than the
        operator's patience window. Done / failed / cancelled rows are
        excluded so the count means "still owed".
        """
        cutoff = time.time() - threshold_seconds
        placeholders = ",".join("?" * len(TERMINAL_STATUSES))
        params: list[Any] = sorted(TERMINAL_STATUSES)
        params.append(cutoff)
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT assignee, COUNT(*) FROM tasks "
                f"WHERE status NOT IN ({placeholders}) "
                "AND created_at < ? GROUP BY assignee",
                params,
            ).fetchall()
        return {row[0]: int(row[1] or 0) for row in rows if row[0]}

    def last_event_ts_for_agent(self, agent_id: str) -> float | None:
        """Return the most recent ``task_events.created_at`` for ``agent_id``.

        Joins ``task_events`` against ``tasks`` and matches against the
        task's ``creator``, ``assignee``, OR the event's ``actor`` —
        any of those three roles count as "this agent participated".
        Returns ``None`` when the agent has no events at all (fresh
        fleet, or store disabled). Used by the dashboard agent card to
        render a "Last task" timestamp alongside the health-derived
        "Last seen" (PR-L').
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT MAX(te.created_at) FROM task_events te "
                "LEFT JOIN tasks t ON te.task_id = t.id "
                "WHERE te.actor = ? OR t.creator = ? OR t.assignee = ?",
                (agent_id, agent_id, agent_id),
            ).fetchone()
        if not row or row[0] is None:
            return None
        return float(row[0])

    def list_stale_for_assignee(
        self,
        assignee: str,
        *,
        threshold_seconds: float,
        limit: int = 5,
    ) -> list[str]:
        """Return up to ``limit`` stale task IDs for ``assignee``, oldest first.

        Used by ``inspect_agents(stale_threshold_hours=N)`` so the
        operator drills into a small set of representative IDs without
        fetching the full row payloads.
        """
        cutoff = time.time() - threshold_seconds
        placeholders = ",".join("?" * len(TERMINAL_STATUSES))
        params: list[Any] = [assignee]
        params.extend(sorted(TERMINAL_STATUSES))
        params.append(cutoff)
        params.append(int(limit))
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id FROM tasks "
                "WHERE assignee = ? "
                f"AND status NOT IN ({placeholders}) "
                "AND created_at < ? "
                "ORDER BY created_at ASC LIMIT ?",
                params,
            ).fetchall()
        return [row[0] for row in rows]

    def workflow_snapshot(
        self, root_task_id: str, *, limit: int = 50,
    ) -> dict | None:
        """Return a snapshot of the workflow chain rooted at ``root_task_id``.

        Walks the ``parent_task_id`` graph downstream from the root (the
        operator's kickoff task) via ``WITH RECURSIVE``. Each descendant
        task is one stage of the workflow. Returns ``None`` when the root
        does not exist. Capped at ``limit`` rows (default 50) so a
        runaway chain can't OOM the caller.

        Output shape:

        ::

            {
              "root": "task_abc",
              "stages": [
                {"task_id": "...", "parent_task_id": "...",
                 "assignee": "...", "status": "...",
                 "age_in_state_seconds": int, "title": "..."},
                ...  # ordered by created_at ASC so the chain reads as
                     # kickoff → downstream
              ],
              "summary": {"done": N, "working": N, "pending": N,
                          "failed": N, "blocked": N, "cancelled": N,
                          "total": N},
            }

        ``age_in_state_seconds`` is ``int(now - max(updated_at, created_at))``
        — the wall-clock age since the last status mutation (or
        creation, for never-transitioned rows).
        """
        now = time.time()
        # Inner recursion is bounded by ``MAX_WORKFLOW_CHAIN_DEPTH`` via
        # a depth counter so a pathological wide chain can't materialize
        # millions of intermediate rows before the outer LIMIT trims.
        # Cycles are impossible by construction (parent_task_id is set
        # once at creation, never updated), but the depth guard is
        # cheap defense-in-depth.
        with self._conn() as conn:
            rows = conn.execute(
                "WITH RECURSIVE chain(id, depth) AS ("
                "  SELECT id, 0 FROM tasks WHERE id = ?"
                "  UNION ALL"
                "  SELECT t.id, c.depth + 1 FROM tasks t "
                "    JOIN chain c ON t.parent_task_id = c.id "
                "    WHERE c.depth < ?"
                ") "
                "SELECT t.id, t.parent_task_id, t.assignee, t.status, "
                "  t.title, t.blocker_note, t.created_at, t.updated_at "
                "FROM tasks t JOIN chain c ON t.id = c.id "
                "ORDER BY t.created_at ASC LIMIT ?",
                (root_task_id, MAX_WORKFLOW_CHAIN_DEPTH, limit),
            ).fetchall()
        if not rows:
            return None
        # Build a child-set from the CTE result itself so we don't need a
        # second SQL query — every descendant in the chain is already in
        # ``rows`` (capped at ``limit`` for safety, but the chain-break
        # signal is correct relative to the snapshot the operator sees).
        # NOTE: a leaf inside the chain whose only children fell off the
        # ``limit`` cap would falsely look terminal. That's the same
        # tradeoff the existing snapshot already makes for status counts.
        parent_ids_with_children: set[str] = set()
        for row in rows:
            parent = row[1]
            if parent:
                parent_ids_with_children.add(parent)
        stages: list[dict] = []
        # Derive buckets from ``VALID_STATUSES`` so a future state added
        # to the enum can't silently drift out of the summary math.
        summary = {status: 0 for status in VALID_STATUSES}
        summary["total"] = 0
        for row in rows:
            (
                tid, parent, assignee, status, title, blocker_note,
                created_at, updated_at,
            ) = row
            age_basis = updated_at or created_at or 0.0
            age = int(now - age_basis) if age_basis > 0 else 0
            stage: dict = {
                "task_id": tid,
                "parent_task_id": parent,
                "assignee": assignee,
                "status": status,
                "age_in_state_seconds": age,
                "title": title or "",
            }
            # Surface ``blocker_note`` only when set — saves the operator
            # a follow-up get_task call for failed/blocked stages and
            # keeps the snapshot quiet for nominal stages.
            if blocker_note:
                stage["blocker_note"] = blocker_note
            # Chain-break observability flag — True when this stage is
            # terminal (``done``) AND nothing in the snapshot references
            # it as its parent. Drives the dashboard chain-break marker
            # without a follow-up query. Observability-only.
            stage["terminal_without_children"] = (
                status == "done" and tid not in parent_ids_with_children
            )
            stages.append(stage)
            if status in summary:
                summary[status] += 1
            summary["total"] += 1
        return {"root": root_task_id, "stages": stages, "summary": summary}

    def update_status(
        self,
        task_id: str,
        status: str,
        *,
        actor: str,
        blocker_note: str | None = None,
        extra_payload: dict | None = None,
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
        # Chain-break signal — populated inside the txn (under the same
        # BEGIN IMMEDIATE that flipped status to ``done``) so the child-
        # count query reads consistent state. Emitted after commit so the
        # bus / asyncio loop never runs while the write lock is held.
        chain_break_payload: dict | None = None
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    f"SELECT status, {self._team_col}, assignee, title, "
                    f"outcome, parent_task_id "
                    f"FROM tasks WHERE id=?",
                    (task_id,),
                ).fetchone()
                if not row:
                    conn.execute("ROLLBACK")
                    raise TaskNotFound(task_id)
                (
                    current, project_id, assignee, task_title, task_outcome,
                    task_parent_id,
                ) = (
                    row[0], row[1], row[2], row[3], row[4], row[5],
                )
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
                # Chain-break detection — only on the ``done`` transition.
                # Counts child tasks (where ``parent_task_id`` = this id)
                # under the same connection so the read sees the same
                # snapshot as the UPDATE we just committed. The actual
                # event emit fires AFTER ``COMMIT`` to avoid running the
                # bus inside the write txn.
                if status == "done":
                    child_count_row = conn.execute(
                        "SELECT COUNT(*) FROM tasks WHERE parent_task_id=?",
                        (task_id,),
                    ).fetchone()
                    child_count = child_count_row[0] if child_count_row else 0
                    if child_count == 0:
                        chain_break_payload = {
                            "task_id": task_id,
                            "agent_id": assignee,
                            # No ``role`` column on tasks — left as ``None``
                            # so the payload shape stays stable for future
                            # consumers when an agent-role lookup is wired
                            # in. The dashboard derives role from agent_id
                            # via its own fleet metadata.
                            "role": None,
                            "parent_task_id": task_parent_id,
                            "project": project_id,
                            "title": task_title,
                            "ts": datetime.now(timezone.utc).isoformat(),
                        }
                conn.execute("COMMIT")
                emitted_change = (
                    current, status, project_id, assignee,
                    task_title, task_outcome,
                )
            except (TaskNotFound, InvalidStatusTransition):
                raise
            except Exception:
                conn.execute("ROLLBACK")
                raise
        if emitted_change is not None:
            (
                old_status, new_status, project_id, assignee,
                task_title, task_outcome,
            ) = emitted_change
            payload: dict = {
                "task_id": task_id,
                "project_id": project_id,
                "assignee": assignee,
                "old_status": old_status,
                "new_status": new_status,
                "actor": actor,
                "ts": now,
                # ``title`` and ``outcome`` carried on every transition so
                # downstream consumers (delivery-notification producer,
                # activity feed) don't need a follow-up task lookup. The
                # producer short-circuits on ``outcome`` to avoid pinging
                # the bell for already-rated (e.g. auto-graded) work.
                "title": task_title,
                "outcome": task_outcome,
            }
            # Merge caller-supplied context (e.g. cancel ``reason``) so
            # the dashboard activity feed can render rich status_changed
            # bubbles without a follow-up audit-log read. ``extra_payload``
            # values shadow the canonical keys above only when explicitly
            # passed — typical callers pass only new metadata.
            if extra_payload:
                for k, v in extra_payload.items():
                    if v is not None:
                        payload[k] = v
            self._safe_emit(
                "task_status_changed",
                agent=actor,
                data=payload,
            )
        # Chain-break observability signal — fires when a task transitioned
        # to ``done`` and no child tasks reference it via ``parent_task_id``
        # (i.e. the agent finished work without handing off). Emitted AFTER
        # the canonical ``task_status_changed`` event so consumers that
        # filter on the status change see the canonical event first.
        # Notification-only; ``_safe_emit`` swallows failures so an absent
        # bus / dashboard never blocks the underlying task transition.
        if chain_break_payload is not None:
            self._safe_emit(
                "task_completed_without_handoff",
                agent=actor,
                data=chain_break_payload,
            )
        return self.get(task_id)  # type: ignore[return-value]

    def reroute(
        self, task_id: str, new_assignee: str, *, actor: str, reason: str = "",
    ) -> dict:
        """Reassign a task to ``new_assignee``. Emits a ``rerouted`` event."""
        if not new_assignee:
            raise ValueError("new_assignee is required")
        now = time.time()
        emitted: tuple[str, str | None, str | None, str | None] | None = None
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    f"SELECT assignee, status, {self._team_col}, title, outcome "
                    f"FROM tasks WHERE id=?",
                    (task_id,),
                ).fetchone()
                if not row:
                    conn.execute("ROLLBACK")
                    raise TaskNotFound(task_id)
                (
                    old_assignee, current_status, project_id,
                    task_title, task_outcome,
                ) = row
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
                emitted = (current_status, project_id, task_title, task_outcome)
            except (TaskNotFound, InvalidStatusTransition):
                raise
            except Exception:
                conn.execute("ROLLBACK")
                raise
        if emitted is not None:
            current_status, project_id, task_title, task_outcome = emitted
            # Reroute is a status_changed event with old==new status; the
            # assignee field carries the *new* assignee so the dashboard
            # sees who picked up the work. The audit row already records
            # ``rerouted`` separately for full history. ``title`` and
            # ``outcome`` ride on the payload to match the shape emitted
            # by ``update_status`` — downstream consumers (notification
            # producer, activity feed) get a uniform schema regardless
            # of which transition path fired.
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
                    "title": task_title,
                    "outcome": task_outcome,
                },
            )
        return self.get(task_id)  # type: ignore[return-value]

    def cancel(
        self, task_id: str, *, actor: str, reason: str = "",
    ) -> dict:
        """Cancel a task. Convenience wrapper over ``update_status('cancelled')``.

        Carries the cancel ``reason`` on the ``task_status_changed``
        EventBus payload (so the dashboard activity feed renders the
        reason inline) and also writes an explicit ``cancelled`` audit
        event row so the reason is preserved in the durable history.
        """
        result = self.update_status(
            task_id, "cancelled", actor=actor,
            extra_payload={"reason": reason} if reason else None,
        )
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
        # Captured outside the txn so the EventBus emit (which can hop
        # across the asyncio loop) doesn't run while we hold BEGIN
        # IMMEDIATE — same pattern as ``update_status`` above.
        emitted_project: str | None = None
        emitted_committed = False
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    f"SELECT artifact_refs_json, {self._team_col} FROM tasks WHERE id=?",
                    (task_id,),
                ).fetchone()
                if not row:
                    conn.execute("ROLLBACK")
                    raise TaskNotFound(task_id)
                refs = json.loads(row[0]) if row[0] else []
                emitted_project = row[1]
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
                emitted_committed = True
            except TaskNotFound:
                raise
            except Exception:
                conn.execute("ROLLBACK")
                raise
        if emitted_committed:
            # Dashboard refreshes the task drawer when this lands. The
            # bus emit is best-effort — durable state is the row above.
            self._safe_emit(
                "task_artifact_added",
                agent=actor,
                data={
                    "task_id": task_id,
                    "project_id": emitted_project,
                    "ref": ref,
                    "actor": actor,
                },
            )
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
                    f"SELECT status, outcome, {self._team_col}, assignee "
                    f"FROM tasks WHERE id=?",
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
                # PR-U — stamp outcome_set_at on first rating only.
                # COALESCE preserves the FIRST set_outcome call's timestamp
                # so downstream consumers (heartbeat, audit) see a stable
                # "first rated at" answer regardless of subsequent
                # re-rates. The audit event row below captures every
                # submission with its own created_at, so re-rating
                # history isn't lost.
                conn.execute(
                    "UPDATE tasks SET outcome=?, feedback_text=?, "
                    "outcome_set_at=COALESCE(outcome_set_at, ?), "
                    "updated_at=? WHERE id=?",
                    (outcome, feedback, now, now, task_id),
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
                f"INSERT INTO tasks "
                f"(id, {self._team_col}, parent_task_id, title, description, "
                f"creator, assignee, status, priority, dependencies_json, "
                f"artifact_refs_json, blocker_note, origin_kind, "
                f"origin_channel, origin_user, created_at, updated_at, "
                f"completed_at, retention_until, outcome, feedback_text, "
                f"previous_task_id) "
                f"VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, NULL, NULL, "
                f"NULL, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, ?)",
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
        record = self.get(new_id)
        if record is None:
            raise RuntimeError(
                f"rework task {new_id!r} INSERT committed but post-read "
                "returned no row — possible storage corruption"
            )
        # Same centralised integrity check ``create`` uses — Bug 1 R2
        # closed the bypass gap for this path so a rework row landing
        # with a mistyped assignee can't silently break the recipient.
        self._assert_stored_matches(
            record,
            expected={
                "assignee": previous["assignee"],
                "creator": actor,
                "project_id": previous.get("project_id"),
                "parent_task_id": previous.get("parent_task_id"),
                "status": "pending",
            },
            context="create_rework_task",
        )
        return record

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
