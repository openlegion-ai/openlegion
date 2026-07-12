"""Durable orchestration task records.

A typed SQLite table backing the coordination verbs —
``coordination_tool`` reads/writes here for all of hand_off /
check_inbox / update_status / complete_task.

Schema mirrors the Blackboard pattern in ``src/host/mesh.py`` and
``src/host/pending_actions.py``: SQLite WAL, ``busy_timeout=30000``,
one canonical schema created via ``executescript()``.

The storage layer enforces status transition validity (so direct callers
cannot corrupt state) and bounds the table by stamping
``retention_until`` on terminal transitions (default 90 days). Reaping
is opportunistic — callers (typically the inbox / team list endpoints)
invoke ``reap_expired`` on their hot path so no separate scheduler is
needed at this slice.

Audit: every state-changing call writes to the companion ``task_events``
table. Operators read this for the per-task audit history.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.shared.redaction import normalize_blocker_note
from src.shared.task_titles import normalize_title_and_description
from src.shared.utils import dumps_safe, setup_logging

logger = setup_logging("host.orchestration")


def _env_int(name: str, default: int) -> int:
    """Read a non-negative int from the environment, falling back to
    ``default`` on missing / unparseable values. A negative parsed value
    is also treated as ``default`` (callers use ``<= 0`` to mean
    "disabled", so we preserve 0 but reject malformed negatives)."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        val = int(raw.strip())
    except (TypeError, ValueError):
        logger.warning(
            "Ignoring non-integer %s=%r — using default %d", name, raw, default,
        )
        return default
    if val < 0:
        logger.warning(
            "Ignoring negative %s=%d — using default %d", name, val, default,
        )
        return default
    return val


# ── Status machine ────────────────────────────────────────────────

VALID_STATUSES: frozenset[str] = frozenset({
    "pending", "accepted", "working", "blocked", "done", "failed", "cancelled",
})

TERMINAL_STATUSES: frozenset[str] = frozenset({"done", "failed", "cancelled"})

# Non-terminal, not-yet-being-worked statuses. The per-assignee pending
# cap (H5) counts ONLY these so a backlog of un-started work can't grow
# unbounded — tasks an agent is actively progressing (``working`` /
# ``blocked`` / ``accepted``) are deliberately excluded so a busy agent
# is never blocked from receiving the handoff it's mid-flight on, and
# completed work (``done`` / ``failed`` / ``cancelled``) never counts.
PENDING_STATUSES: frozenset[str] = frozenset({"pending"})

# Outcome enum — operator-supplied judgement on a completed task.
# ``None`` means "not yet rated". Outcomes are write-many: every
# submission appends a ``task_outcome`` audit event and the
# ``tasks.outcome`` column reflects the LATEST rating, so a re-rate just
# overwrites without admin intervention.
#
# - ``accepted``: positive signal. Rating only — nothing is written to
#   the agent (praise in the corrections file would dilute the signal).
# - ``rework``: negative signal with a fix-it brief. Auto-spawns a
#   rework task AND pushes the feedback into the rated agent's
#   learnings (corrections file, via the mesh ``feedback_push`` helper)
#   so the agent sees it in every future prompt.
# - ``rejected``: terminal negative. Pushes feedback to the agent's
#   learnings; does NOT auto-spawn a rework.
# - ``acknowledged``: neutral, "reviewed without judgement". No push,
#   no rework spawn.
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
# this per-team later (see plan Task 6 §3); this slice ships the
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

# H5 — runaway parent-chain guard. ``parent_task_id`` forms a chain
# (each task points at one parent); legitimate workflows are 3–8 stages
# deep, so 25 is generous headroom while still cutting off a self-
# replicating loop that keeps spawning "do the next step" children
# forever. This bounds the parent CHAIN length only — it does NOT bound
# fan-out (one parent may legitimately have dozens of sibling subtasks).
# Self-parent cycles (parent_task_id == own id, or a cycle reached while
# walking up) are rejected regardless of depth. Env override:
# ``OPENLEGION_MAX_TASK_CHAIN_DEPTH``.
DEFAULT_MAX_TASK_CHAIN_DEPTH: int = 25

# H5 — per-assignee backlog cap. Counts only ``PENDING_STATUSES`` tasks
# (un-started work) so a runaway producer can't bury an assignee under an
# unbounded queue. 200 is far above any plausible real backlog (agents
# rarely carry more than a handful of un-started tasks at once) while
# still catching a loop minting thousands. Env override:
# ``OPENLEGION_MAX_PENDING_TASKS_PER_AGENT``.
DEFAULT_MAX_PENDING_TASKS_PER_AGENT: int = 200


class TaskLimitExceeded(ValueError):
    """Raised when a task-creation resource cap (chain depth / pending
    backlog) would be exceeded. The mesh endpoint maps this to HTTP 400.

    Distinct from ``InvalidStatusTransition`` so callers can surface a
    cap-specific ``create_failed`` envelope to the agent.
    """


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
        max_pending_per_agent: int | None = None,
        max_chain_depth: int | None = None,
    ):
        self.db_path = db_path
        self.retention_seconds = retention_seconds
        # H5 resource caps. Resolved once at construction: explicit kwarg
        # wins, then env override, then the generous module default. A
        # non-positive value disables the cap entirely (operator escape
        # hatch — set ``OPENLEGION_MAX_PENDING_TASKS_PER_AGENT=0`` to turn
        # off the backlog cap).
        self.max_pending_per_agent = (
            max_pending_per_agent
            if max_pending_per_agent is not None
            else _env_int(
                "OPENLEGION_MAX_PENDING_TASKS_PER_AGENT",
                DEFAULT_MAX_PENDING_TASKS_PER_AGENT,
            )
        )
        self.max_chain_depth = (
            max_chain_depth
            if max_chain_depth is not None
            else _env_int(
                "OPENLEGION_MAX_TASK_CHAIN_DEPTH",
                DEFAULT_MAX_TASK_CHAIN_DEPTH,
            )
        )
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
            # Canonical schema v1 (``PRAGMA user_version = 1``). There is
            # exactly one shape — no lazy ALTER chains, no legacy column
            # detection. A database created by any earlier scheme is not
            # supported (pre-release: no users, no data to migrate).
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
                    retention_until REAL,
                    outcome TEXT,
                    feedback_text TEXT,
                    previous_task_id TEXT,
                    result_summary TEXT,
                    -- Per-task reasoning depth ("off"/"low"/"medium"/"high",
                    -- NULL = use the assignee's configured default).
                    thinking TEXT,
                    -- Per-turn correlation id (``tr_<hex12>``) that minted
                    -- this task, so a session can be reconstructed by
                    -- JOINing tasks / usage / traces on one key. Nullable:
                    -- paths with no active trace keep NULL.
                    trace_id TEXT,
                    -- When set_outcome was called, separately from
                    -- completed_at, so count_outcomes_since doesn't
                    -- undercount lagged operator reviews.
                    outcome_set_at REAL
                );
                CREATE INDEX IF NOT EXISTS idx_tasks_assignee_status
                    ON tasks (assignee, status);
                CREATE INDEX IF NOT EXISTS idx_tasks_created
                    ON tasks (created_at);
                CREATE INDEX IF NOT EXISTS idx_tasks_retention
                    ON tasks (retention_until);
                CREATE INDEX IF NOT EXISTS idx_tasks_parent_task_id
                    ON tasks (parent_task_id);
                CREATE INDEX IF NOT EXISTS idx_tasks_team_status
                    ON tasks (team_id, status);
                CREATE INDEX IF NOT EXISTS idx_tasks_previous_task
                    ON tasks (previous_task_id);
                -- Partial index: most rows have outcome IS NULL (work isn't
                -- rated yet), so the predicate keeps the index small for
                -- the heartbeat's per-outcome scan.
                CREATE INDEX IF NOT EXISTS idx_tasks_outcome_set_at
                    ON tasks (outcome, outcome_set_at)
                    WHERE outcome IS NOT NULL;

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

                CREATE TABLE IF NOT EXISTS chain_deliveries (
                    root_task_id TEXT PRIMARY KEY,
                    terminal_kind TEXT NOT NULL,
                    delivered_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chain_stall_notices (
                    root_task_id TEXT PRIMARY KEY,
                    notified_at REAL NOT NULL
                );

                -- Blocked-task escalation ladder (plan §8 #22). One row per
                -- currently-escalating blocked task: the rung reached and
                -- when it was last climbed. Cleared by the ladder sweep when
                -- the task leaves ``blocked`` (a re-block restarts at rung 0).
                CREATE TABLE IF NOT EXISTS blocked_escalations (
                    task_id TEXT PRIMARY KEY,
                    blocked_since REAL NOT NULL,
                    rung INTEGER NOT NULL DEFAULT 0,
                    last_climb_at REAL NOT NULL
                );

                -- Rung-4 (human) claim ledger — mirrors chain_stall_notices:
                -- INSERT OR IGNORE on the task id makes the durable Needs-you
                -- entry at-most-once PER TASK EVER, surviving sweeps, mesh
                -- restarts, and unblock→re-block cycles.
                CREATE TABLE IF NOT EXISTS blocked_human_notices (
                    task_id TEXT PRIMARY KEY,
                    notified_at REAL NOT NULL
                );

                PRAGMA user_version = 1;
            """)

    # ── Helpers ─────────────────────────────────────────────────

    # ── Chain-watch support (delegate-and-subscribe) ─────────────
    #
    # A "chain" is the parent_task_id-linked tree rooted at the FIRST
    # task of a user request (``parent_task_id IS NULL``). The operator's
    # initial hand_off creates that root as a trusted caller, so its
    # ``origin_kind == "human"`` is first-party; every later hop is
    # created by an untrusted worker whose ``kind="human"`` claim is
    # downgraded to ``agent`` by ``_validated_origin`` at the mesh edge.
    # The chain watcher therefore delivers ONLY to the root's stored
    # human origin (never a mid-chain, possibly-forged origin), which
    # keeps the existing origin-trust boundary fully intact.

    def list_watchable_human_roots(self, *, since: float) -> list[dict]:
        """Root tasks of user-originated chains awaiting a terminal push.

        A row qualifies when it is a chain root (``parent_task_id IS
        NULL``), carries a first-party human origin, was created within
        the watch window (``created_at >= since``), and has not already
        had a terminal notification claimed in ``chain_deliveries``.

        **Coverage is intentionally limited to OPERATOR-rooted chains.**
        ``origin_kind = 'human'`` is only first-party when a *trusted*
        caller created the root: the operator's initial hand_off keeps
        ``kind="human"``, but a non-operator worker's hand_off has its
        claim downgraded to ``agent`` by ``_validated_origin`` (``dashboard``
        is not a paired channel). A *direct* user→worker→sub-agent chat
        therefore produces an ``agent``-origin root that is (correctly) NOT
        watched — delivering on it would mean trusting a forgeable origin.
        Covering that topology safely needs a trusted server-side chain
        registration at the dashboard ``/chat`` entry (follow-up).

        Projects ``_SELECT_COLS`` (never ``SELECT *``) so ``_row_to_dict``'s
        positional mapping stays correct on DBs with differing ALTER-column
        histories.
        """
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT {self._SELECT_COLS} FROM tasks "
                "WHERE parent_task_id IS NULL "
                "  AND origin_kind = 'human' "
                "  AND created_at >= ? "
                "  AND id NOT IN (SELECT root_task_id FROM chain_deliveries) "
                "ORDER BY created_at ASC",
                (since,),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def chain_terminal_verdict(self, root_task_id: str) -> tuple[str, str] | None:
        """Return ``(terminal_kind, summary)`` iff the WHOLE chain is terminal.

        Walks the same ``parent_task_id``/``previous_task_id`` recursion
        as :meth:`workflow_snapshot`. Returns ``None`` while ANY task in
        the chain is non-terminal (including a still-being-created next
        hop, which is why the watcher additionally debounces). When every
        task is terminal, ``terminal_kind`` is ``failed`` if any task
        failed, else ``cancelled`` if any was cancelled, else ``done``.
        ``summary`` is a best-effort human string: the failed task's
        blocker_note, or the most-recently-completed done leaf's
        result_summary.

        Cancellation is judged by the ROOT task (the user's request itself):
        a chain whose root completed but had a *cancelled* downstream branch
        resolves to ``done`` (don't swallow real completion); only a chain
        whose root was cancelled is treated as a silent manual cancellation
        (don't claim "complete" just because a sub-stage finished before the
        user cancelled). This keeps the "every user pipeline ends in a
        user-facing outcome" guarantee without false "complete" pings.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "WITH RECURSIVE chain(id, depth) AS ("
                "  SELECT id, 0 FROM tasks WHERE id = ?"
                "  UNION ALL"
                "  SELECT t.id, c.depth + 1 FROM tasks t "
                "    JOIN chain c ON (t.parent_task_id = c.id "
                "                  OR t.previous_task_id = c.id) "
                "    WHERE c.depth < ?"
                ") "
                "SELECT t.id, t.status, t.result_summary, t.blocker_note, "
                "  t.completed_at, t.updated_at, "
                "  (SELECT COUNT(*) FROM tasks x "
                "     WHERE (x.parent_task_id IN (SELECT id FROM chain) "
                "         OR x.previous_task_id IN (SELECT id FROM chain)) "
                "       AND x.id NOT IN (SELECT id FROM chain)) "
                "FROM tasks t "
                "WHERE t.id IN (SELECT DISTINCT id FROM chain)",
                (root_task_id, MAX_WORKFLOW_CHAIN_DEPTH),
            ).fetchall()
        if not rows:
            return None
        # Truncation guard: count any task that is a child (by parent_task_id
        # or previous_task_id) of a chain node yet falls OUTSIDE the recursion
        # — i.e. the depth cap cut off real descendants we cannot see. When
        # that happens we can't prove the WHOLE chain is terminal, so treat it
        # as still-active (return None) rather than risk a false "done"/"failed"
        # delivery while a deep branch is still running. A chain that naturally
        # ENDS at the cap has no such children (count 0) and still delivers —
        # only genuine truncation is suppressed. Practically unreachable below
        # the creation guard's default ceiling (25 << 200).
        if (rows[0][6] or 0) > 0:
            logger.warning(
                "chain_terminal_verdict: chain %s has descendants beyond depth "
                "cap %d — treating as active (cannot prove terminal)",
                root_task_id, MAX_WORKFLOW_CHAIN_DEPTH,
            )
            return None
        if any(r[1] not in TERMINAL_STATUSES for r in rows):
            return None
        statuses = {r[1] for r in rows}
        if "failed" in statuses:
            failed = [r for r in rows if r[1] == "failed"]
            failed.sort(key=lambda r: r[4] or r[5] or 0.0)
            return "failed", (failed[-1][3] or "").strip()
        # No failure: the chain is wholly done/cancelled. Judge by the ROOT.
        # The chain is proven terminal above, so the root is itself done or
        # cancelled — gate the silent path on the root, not on "any cancelled".
        root_status = next(
            (r[1] for r in rows if r[0] == root_task_id), None
        )
        if root_status == "cancelled":
            # The user's request task itself was cancelled — a manual action,
            # not a surprise to surface. Don't claim "complete" even if a
            # sub-stage finished first. The watcher claims it silently.
            return "cancelled", ""
        # The root completed — deliver the done outcome (a cancelled downstream
        # branch must not swallow real completion). The root itself is a done
        # row, so done_leaves is non-empty here.
        done_leaves = [r for r in rows if r[1] == "done"]
        done_leaves.sort(key=lambda r: r[4] or r[5] or 0.0)
        summary = (done_leaves[-1][2] or "").strip()
        return "done", summary

    def chain_stall_state(self, root_task_id: str) -> float | None:
        """Last-progress timestamp iff the chain is a STALL candidate, else None.

        A chain is a stall candidate when it is non-terminal but nothing is
        actively progressing it — every task is terminal or *waiting*
        (``pending`` / ``accepted`` / ``blocked``) and none is ``working``.
        Returns the most recent ``updated_at`` across the chain (the last time
        anything moved) so the caller can decide whether it has been quiet
        long enough to nudge the user.

        Returns ``None`` when the chain doesn't exist, is wholly terminal (the
        :meth:`chain_terminal_verdict` path handles that), or has a ``working``
        task — a ``working`` task is making progress, and a genuinely *hung*
        one is the lane watchdog's job to time out into ``failed`` (which then
        flows through the terminal path). This watcher only covers the
        watchdog's blind spot: chains parked in a waiting state.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "WITH RECURSIVE chain(id, depth) AS ("
                "  SELECT id, 0 FROM tasks WHERE id = ?"
                "  UNION ALL"
                "  SELECT t.id, c.depth + 1 FROM tasks t "
                "    JOIN chain c ON (t.parent_task_id = c.id "
                "                  OR t.previous_task_id = c.id) "
                "    WHERE c.depth < ?"
                ") "
                "SELECT t.status, t.updated_at, "
                "  (SELECT COUNT(*) FROM tasks x "
                "     WHERE (x.parent_task_id IN (SELECT id FROM chain) "
                "         OR x.previous_task_id IN (SELECT id FROM chain)) "
                "       AND x.id NOT IN (SELECT id FROM chain)) "
                "FROM tasks t "
                "WHERE t.id IN (SELECT DISTINCT id FROM chain)",
                (root_task_id, MAX_WORKFLOW_CHAIN_DEPTH),
            ).fetchall()
        if not rows:
            return None
        # Truncation guard (mirrors chain_terminal_verdict): if real descendants
        # were cut off by the depth cap, a deep ``working`` node would be
        # invisible — so don't emit a (possibly wrong) stall nudge.
        if (rows[0][2] or 0) > 0:
            return None
        statuses = {r[0] for r in rows}
        if all(s in TERMINAL_STATUSES for s in statuses):
            return None  # wholly terminal — not a stall
        if "working" in statuses:
            return None  # something is actively progressing
        return max((r[1] or 0.0) for r in rows)

    def claim_chain_stall(self, root_task_id: str) -> bool:
        """Atomically claim the one stall nudge for a chain (at-most-once).

        Separate ledger from :meth:`claim_chain_delivery`: a chain can get a
        stall nudge while parked AND, later, a terminal delivery when it
        finally resolves. One nudge per chain (no re-nudging on a re-stall).
        """
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO chain_stall_notices "
                "(root_task_id, notified_at) VALUES (?, ?)",
                (root_task_id, time.time()),
            )
            return cur.rowcount == 1

    def claim_chain_delivery(self, root_task_id: str, terminal_kind: str) -> bool:
        """Atomically claim the one terminal notification for a chain.

        Returns ``True`` for the first caller and ``False`` forever after
        (``INSERT OR IGNORE`` on the ``root_task_id`` primary key). This
        is the exactly-once guard: it is safe against repeated terminal
        transitions, a restart replay, and the startup re-scan all racing
        to deliver the same chain.
        """
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO chain_deliveries "
                "(root_task_id, terminal_kind, delivered_at) VALUES (?, ?, ?)",
                (root_task_id, terminal_kind, time.time()),
            )
            return cur.rowcount == 1

    # ── Blocked-task escalation ladder (plan §8 #22) ──────────────
    #
    # Durable per-task ladder state for the ChainWatcher's blocked-task
    # sweep. Same persistence posture as the chain-delivery/stall claims
    # above: sibling tables in this store, claim-style writes, restart-
    # safe. The ladder is INFLUENCE ONLY — nothing here (or in the
    # sweep) ever transitions a task, reassigns it, or touches goals.

    def list_blocked(self) -> list[dict]:
        """All ``blocked`` tasks, oldest-updated first.

        Feeds the escalation-ladder sweep. ``blocked`` is non-terminal so
        rows here are never reaped mid-escalation; a task that leaves
        ``blocked`` simply drops out of the scan (and
        :meth:`ladder_reset_unblocked` clears its ladder state).
        """
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT {self._SELECT_COLS} FROM tasks "
                "WHERE status = 'blocked' ORDER BY updated_at ASC",
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def ladder_state(self, task_id: str) -> dict | None:
        """The task's ladder row (``blocked_since`` / ``rung`` /
        ``last_climb_at``), or ``None`` before first observation."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT task_id, blocked_since, rung, last_climb_at "
                "FROM blocked_escalations WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "task_id": row[0],
            "blocked_since": row[1],
            "rung": row[2],
            "last_climb_at": row[3],
        }

    def ladder_observe_blocked(self, task_id: str, *, now: float | None = None) -> dict:
        """Register first observation of a blocked task (rung 0).

        ``INSERT OR IGNORE`` so a concurrent sweep can't double-create;
        returns the (possibly pre-existing) row. ``blocked_since`` is the
        observation time — within one sweep interval of the actual
        transition, which is close enough for interval math measured in
        tens of minutes.
        """
        ts = time.time() if now is None else now
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO blocked_escalations "
                "(task_id, blocked_since, rung, last_climb_at) "
                "VALUES (?, ?, 0, ?)",
                (task_id, ts, ts),
            )
        return self.ladder_state(task_id)  # type: ignore[return-value]

    def ladder_climb(
        self, task_id: str, from_rung: int, to_rung: int, *, now: float | None = None,
    ) -> bool:
        """Atomically climb ``from_rung`` → ``to_rung`` (CAS on the rung).

        Returns ``True`` only for the caller that actually moved the row —
        the claim discipline the chain watcher's stall nudge uses, so two
        racing sweeps (or a restart replay) can't both dispatch the same
        rung's nudge. The loser sees ``False`` and sends nothing.
        """
        ts = time.time() if now is None else now
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE blocked_escalations SET rung = ?, last_climb_at = ? "
                "WHERE task_id = ? AND rung = ?",
                (to_rung, ts, task_id, from_rung),
            )
            return cur.rowcount == 1

    def ladder_reset_unblocked(self) -> int:
        """Drop ladder state for tasks no longer ``blocked``. Returns count.

        Run at the top of each ladder sweep: an unblocked (or reaped)
        task's ladder resets, so a later re-block restarts at rung 0.
        The rung-4 human claim (``blocked_human_notices``) deliberately
        survives — that entry is at-most-once per task ever.
        """
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM blocked_escalations WHERE task_id NOT IN "
                "(SELECT id FROM tasks WHERE status = 'blocked')",
            )
            return cur.rowcount

    def claim_blocked_human_notice(self, task_id: str) -> bool:
        """Atomically claim the ONE rung-4 human escalation for a task.

        Mirrors :meth:`claim_chain_stall` exactly: ``INSERT OR IGNORE`` on
        the primary key — first caller ``True``, everyone after ``False``
        forever (across sweeps, restarts, and re-block cycles).
        """
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO blocked_human_notices "
                "(task_id, notified_at) VALUES (?, ?)",
                (task_id, time.time()),
            )
            return cur.rowcount == 1

    def escalated_blocked_for_team(
        self, team_id: str, *, min_rung: int = 3, sample_limit: int = 3,
    ) -> tuple[int, list[str]]:
        """Count + small id sample of a team's blocked tasks at rung ≥ ``min_rung``.

        Feeds the lead-plate ``lead_blocked_tasks_fn`` cron probe (plan §8
        #22 rung 3): only tasks the ladder has escalated onto the lead's
        plate count — a freshly-blocked task (rung < 3) stays off it.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT t.id FROM tasks t "
                "JOIN blocked_escalations b ON b.task_id = t.id "
                "WHERE t.team_id = ? AND t.status = 'blocked' AND b.rung >= ? "
                "ORDER BY b.blocked_since ASC",
                (team_id, min_rung),
            ).fetchall()
        ids = [r[0] for r in rows]
        return len(ids), ids[: max(0, int(sample_limit))]

    @staticmethod
    def _row_to_dict(row: tuple) -> dict:
        """Map a SELECT-* row to a public-facing task dict."""
        return {
            "id": row[0],
            "team_id": row[1],
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
            "result_summary": row[23],
            "thinking": row[24],
            "trace_id": row[25],
        }

    _SELECT_COLS = (
        "id, team_id, parent_task_id, title, description, creator, "
        "assignee, status, priority, dependencies_json, artifact_refs_json, "
        "blocker_note, origin_kind, origin_channel, origin_user, "
        "created_at, updated_at, completed_at, retention_until, "
        "outcome, feedback_text, previous_task_id, outcome_set_at, "
        "result_summary, thinking, trace_id"
    )

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
        team_id: str | None = None,
        parent_task_id: str | None = None,
        priority: int = 0,
        dependencies: list[str] | None = None,
        artifact_refs: list[str] | None = None,
        origin: dict | None = None,
        task_id: str | None = None,
        thinking: str | None = None,
    ) -> dict:
        """Insert a task. Returns the public record dict.

        ``task_id`` is generated when omitted (UUID-style ``task_<hex12>``).
        ``origin`` is a 3-key dict (``kind``, ``channel``, ``user``) — the
        same shape :class:`MessageOrigin` produces; passing the typed
        Pydantic model is the caller's responsibility (use
        ``model_dump()`` or unpack as needed).
        ``thinking`` (B4) pins a per-task reasoning depth for the
        assignee's LLM calls while executing this task; NULL means the
        assignee's configured default applies.
        """
        if thinking is not None and thinking not in (
            "off", "low", "medium", "high",
        ):
            raise ValueError(
                f"thinking must be off/low/medium/high, got {thinking!r}"
            )
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

        # ── H5 resource caps ────────────────────────────────────────
        # Enforced here (the single create choke-point) so EVERY path —
        # the mesh endpoint, coordination_tool hand_off, migration shims —
        # is bounded identically. Both raise ``TaskLimitExceeded`` which
        # the endpoint maps to HTTP 400 and the agent surfaces as a
        # ``create_failed`` envelope.
        #
        # 1. Parent-chain depth + cycle guard. Walks UP the parent chain;
        #    rejects self-parenting, cycles, and chains deeper than the
        #    cap. Fan-out (many children of one parent) is unaffected —
        #    only chain LENGTH is bounded.
        if parent_task_id:
            self._guard_parent_chain(parent_task_id, proposed_id=task_id)

        # 2. Per-assignee pending backlog cap. Counts only un-started
        #    (PENDING_STATUSES) tasks so an actively-working agent is
        #    never blocked from receiving the next handoff, and terminal
        #    rows never count.
        if self.max_pending_per_agent > 0:
            pending = self.count_pending_for_assignee(assignee)
            if pending >= self.max_pending_per_agent:
                raise TaskLimitExceeded(
                    f"assignee {assignee!r} has {pending} pending tasks "
                    f"(cap {self.max_pending_per_agent}); complete or "
                    "cancel un-started work before creating more"
                )

        tid = task_id or f"task_{uuid.uuid4().hex[:12]}"
        now = time.time()
        kind = origin.get("kind") if isinstance(origin, dict) else None
        channel = origin.get("channel") if isinstance(origin, dict) else None
        user = origin.get("user") if isinstance(origin, dict) else None
        # Session observability (Phase 1) — stamp the active per-turn
        # correlation id so this task JOINs to its usage/trace rows.
        # Read from the contextvar at write time: it is seeded from the
        # inbound X-Trace-Id header on the lane / mesh paths. NULL when no
        # trace is active (purely internal task creation).
        from src.shared.trace import current_trace_id
        trace_id = current_trace_id.get()

        with self._conn() as conn:
            conn.execute(
                "INSERT INTO tasks "
                "(id, team_id, parent_task_id, title, description, "
                "creator, assignee, status, priority, dependencies_json, "
                "artifact_refs_json, blocker_note, origin_kind, "
                "origin_channel, origin_user, created_at, updated_at, "
                "completed_at, retention_until, thinking, trace_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, NULL, "
                "?, ?, ?, ?, ?, NULL, NULL, ?, ?)",
                (
                    tid, team_id, parent_task_id, title, description,
                    creator, assignee, priority,
                    json.dumps(dependencies) if dependencies else None,
                    json.dumps(artifact_refs) if artifact_refs else None,
                    kind, channel, user,
                    now, now, thinking, trace_id,
                ),
            )
            self._emit_event(
                conn, tid, "created", creator,
                {
                    "title": title,
                    "assignee": assignee,
                    "team_id": team_id,
                },
            )
        # Task 9 — surface to the dashboard. Compact payload (title, no
        # description) so WS frames stay small.
        self._safe_emit(
            "task_created",
            agent=creator,
            data={
                "task_id": tid,
                "team_id": team_id,
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
                "team_id": team_id,
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
            record.get("team_id"), parent_task_id,
            record.get("status"), (title or "")[:80],
        )
        return record

    # ── H5 cap helpers ──────────────────────────────────────────────

    def count_pending_for_assignee(self, assignee: str) -> int:
        """Count un-started (``PENDING_STATUSES``) tasks for ``assignee``.

        Used by the per-assignee backlog cap. Deliberately excludes
        ``accepted`` / ``working`` / ``blocked`` (in-flight) and terminal
        rows — only genuinely queued, not-yet-started work counts."""
        placeholders = ",".join("?" * len(PENDING_STATUSES))
        params: list[Any] = [assignee, *sorted(PENDING_STATUSES)]
        with self._conn() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) FROM tasks "
                f"WHERE assignee = ? AND status IN ({placeholders})",
                params,
            ).fetchone()
        return int(row[0]) if row else 0

    def has_working_task(self, assignee: str) -> bool:
        """Whether ``assignee`` currently has any ``working`` task.

        Used by the hibernation sweep (plan §8 #24) to refuse hibernating
        an agent mid-task even when its lane is otherwise idle (a task can
        be ``working`` via a path that never touches the lane's busy
        flag). Also backs the manual hibernate endpoint's 409 refusal.
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM tasks WHERE assignee = ? AND status = 'working' LIMIT 1",
                (assignee,),
            ).fetchone()
        return row is not None

    def _guard_parent_chain(
        self, parent_task_id: str, *, proposed_id: str | None = None,
    ) -> None:
        """Reject runaway / cyclic ``parent_task_id`` chains.

        Walks UP from ``parent_task_id`` following each row's own parent.
        Raises :class:`TaskLimitExceeded` on:
          * self-parenting (``proposed_id == parent_task_id``),
          * a cycle reached while walking (a parent already seen),
          * a chain deeper than ``self.max_chain_depth``.

        A non-positive ``max_chain_depth`` disables the depth cap, but the
        self-parent + cycle checks always run (they are correctness, not
        capacity, guards). Bounds only chain LENGTH — sibling fan-out is
        never affected. Unknown parents are allowed through (the endpoint
        does not require the parent to pre-exist; a dangling reference is
        a separate concern)."""
        # Self-parent: the new task names itself as parent.
        if proposed_id is not None and proposed_id == parent_task_id:
            raise TaskLimitExceeded(
                f"task {proposed_id!r} cannot be its own parent "
                "(self-parent cycle)"
            )
        seen: set[str] = set()
        if proposed_id is not None:
            seen.add(proposed_id)
        depth = 1  # the link from the new task to ``parent_task_id``
        current: str | None = parent_task_id
        cap = self.max_chain_depth
        while current is not None:
            if current in seen:
                raise TaskLimitExceeded(
                    f"parent chain through {current!r} forms a cycle"
                )
            seen.add(current)
            if cap > 0 and depth > cap:
                raise TaskLimitExceeded(
                    f"parent chain exceeds max depth {cap} "
                    "(runaway task chain)"
                )
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT parent_task_id FROM tasks WHERE id = ?",
                    (current,),
                ).fetchone()
            if row is None:
                # Dangling / unknown parent — stop walking, allow.
                break
            current = row[0]
            depth += 1

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
        team_id: str | None = None,
        include_terminal: bool = False,
    ) -> list[dict]:
        """List tasks assigned to ``assignee``.

        By default, terminal tasks (done / failed / cancelled) are excluded
        — matches the blackboard ``check_inbox`` semantics. Pass
        ``include_terminal=True`` to see history. ``team_id`` further
        narrows to a single team.
        """
        clauses = ["assignee = ?"]
        params: list[Any] = [assignee]
        if team_id is not None:
            clauses.append("team_id = ?")
            params.append(team_id)
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
            "tasks.list_inbox assignee=%r team_id=%r "
            "include_terminal=%s rows=%d",
            assignee, team_id, include_terminal, len(result),
        )
        return result

    def list_pending(self) -> list[dict]:
        """All un-started (``PENDING_STATUSES``) tasks with an assignee.

        Used by lane rehydration on mesh boot: the in-memory lane queues are
        lost on a restart, but the durable task rows survive. Only ``pending``
        tasks are returned — a task already ``working`` / ``accepted`` may be
        mid-flight, so re-dispatching it would risk double execution; those are
        deliberately excluded and left for the assignee (or a stall reaper) to
        resolve. Unassigned rows are skipped (nothing to dispatch to).
        Ordered ``created_at ASC`` so recovery preserves arrival order.
        """
        placeholders = ",".join("?" * len(PENDING_STATUSES))
        sql = (
            f"SELECT {self._SELECT_COLS} FROM tasks "
            f"WHERE status IN ({placeholders}) AND assignee != '' "
            f"ORDER BY created_at ASC"
        )
        with self._conn() as conn:
            rows = conn.execute(sql, sorted(PENDING_STATUSES)).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def list_team(
        self,
        team_id: str,
        *,
        statuses: list[str] | None = None,
    ) -> list[dict]:
        """List tasks in a team. Optional ``statuses`` filter."""
        clauses = ["team_id = ?"]
        params: list[Any] = [team_id]
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

    def count_team_outcomes_since(
        self,
        team_id: str,
        *,
        since_seconds: float,
    ) -> dict[str, int]:
        """Count rated tasks per OUTCOME for one team within a window.

        P2 — feeds the work-summary metrics block so daily summaries
        reflect the user's rating history (accepted/rework/rejected),
        not just status counts. Same ``outcome_set_at`` filter rationale
        as :meth:`count_outcomes_since`.
        """
        cutoff = time.time() - since_seconds
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT outcome, COUNT(*) FROM tasks "
                "WHERE team_id = ? AND outcome IS NOT NULL "
                "AND outcome_set_at IS NOT NULL AND outcome_set_at >= ? "
                "GROUP BY outcome",
                (team_id, cutoff),
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

    def chain_breaks_24h(self, since: float) -> dict[str, int]:
        """Count chain-break tasks per assignee since the cutoff timestamp.

        A "chain-break" is a ``done`` task whose work didn't get handed
        off to a successor — operationally, a row that:

        * has ``status='done'``
        * has ``completed_at >= since`` (a Unix timestamp; callers
          typically pass ``time.time() - 86400`` for a trailing 24h
          window). ``completed_at`` is stamped once at the terminal
          transition (``done``/``failed``/``cancelled``) and is NOT
          touched by later metadata writes — outcome updates, blocker
          notes, artifact appends all bump ``updated_at`` but never
          ``completed_at``. Filtering on ``completed_at`` keeps a task
          completed >24h ago out of the window even if an unrelated
          metadata write happens later. Mirrors the
          :meth:`count_failed_status_since` pattern.
        * has no row in ``tasks`` whose ``parent_task_id`` references it.
          ``previous_task_id`` (rework) descendants are DELIBERATELY
          NOT considered here — a rework is a retry of the same work,
          not the downstream handoff the chain was waiting for. See
          ``test_previous_task_id_chain_does_not_suppress_chain_break``
          for the pinning test on this semantic.
        * has ``outcome IS NULL`` — the operator hasn't actioned this
          delivery yet via rate / rework. Outcome-set rows drop out of
          the count because the operator already saw them.

        Returns ``{assignee: count}`` for assignees with at least one
        chain-break in the window. The ``NOT EXISTS`` subquery uses
        ``idx_tasks_parent_task_id`` so this is O(#done-rows-in-window)
        index lookups. Surfaced as ``chain_breaks_24h_count`` on the
        ``/mesh/system/metrics`` payload so the operator heartbeat
        playbook can drill into silent-handoff agents via its existing
        ``get_system_status`` call — no need for a separate
        ``workflow_snapshot`` poll.

        Observability-only — NO enforcement. Pairs with the
        ``task_completed_without_handoff`` DashboardEvent emitted by
        :meth:`update_status` at the ``done`` transition.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT assignee, COUNT(*) FROM tasks t1 "
                "WHERE t1.status = 'done' "
                "  AND t1.completed_at IS NOT NULL "
                "  AND t1.completed_at >= ? "
                "  AND t1.outcome IS NULL "
                "  AND NOT EXISTS ("
                "    SELECT 1 FROM tasks t2 WHERE t2.parent_task_id = t1.id"
                "  ) "
                "GROUP BY assignee",
                (since,),
            ).fetchall()
        return {row[0]: int(row[1] or 0) for row in rows if row[0]}

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

        Walks BOTH the ``parent_task_id`` graph (normal handoff chain)
        AND the ``previous_task_id`` graph (rework lineage from
        :meth:`create_rework_task`) downstream from the root via
        ``WITH RECURSIVE``. Each descendant task is one stage of the
        workflow. Returns ``None`` when the root does not exist. Capped
        at ``limit`` rows (default 50) so a runaway chain can't OOM the
        caller.

        Walking both edges matters because rework tasks inherit the
        failed task's ``parent_task_id`` (so the kanban renders them as
        siblings of the failed validator) but their lineage to the
        failed task is captured only via ``previous_task_id``. Without
        traversing that edge:

        * a snapshot rooted at a failed task would miss its rework
          descendants (the parent inheritance points them sideways
          rather than down)
        * an orphan rework (parent outside the chain) would be
          invisible even from the chain root

        Output shape:

        ::

            {
              "root": "task_abc",
              "stages": [
                {"task_id": "...", "parent_task_id": "...",
                 "previous_task_id": "...",  # rework stages only
                 "assignee": "...", "status": "...",
                 "age_in_state_seconds": int, "title": "..."},
                ...  # ordered by created_at ASC so the chain reads as
                     # kickoff → downstream
              ],
              "summary": {"done": N, "working": N, "pending": N,
                          "failed": N, "blocked": N, "cancelled": N,
                          "total": N},
            }

        ``age_in_state_seconds`` is ``int(now - (updated_at or created_at))``
        — the wall-clock age since the last status mutation (or
        creation, for never-transitioned rows).
        """
        now = time.time()
        # Inner recursion is bounded by ``MAX_WORKFLOW_CHAIN_DEPTH`` via
        # a depth counter so a pathological wide chain can't materialize
        # millions of intermediate rows before the outer LIMIT trims.
        # Cycles are impossible by construction (both ``parent_task_id``
        # and ``previous_task_id`` are set once at creation, never
        # updated, and reference tasks that already exist) but the
        # depth guard is cheap defense-in-depth. A task reachable via
        # both edges (e.g. a rework spawned from the chain root) lands
        # in ``chain`` twice at potentially different depths — the
        # outer ``IN (SELECT DISTINCT id FROM chain)`` dedupes so the
        # task surfaces once in the snapshot.
        with self._conn() as conn:
            rows = conn.execute(
                "WITH RECURSIVE chain(id, depth) AS ("
                "  SELECT id, 0 FROM tasks WHERE id = ?"
                "  UNION ALL"
                "  SELECT t.id, c.depth + 1 FROM tasks t "
                "    JOIN chain c ON (t.parent_task_id = c.id "
                "                  OR t.previous_task_id = c.id) "
                "    WHERE c.depth < ?"
                ") "
                "SELECT t.id, t.parent_task_id, t.previous_task_id, "
                "  t.assignee, t.status, t.title, t.blocker_note, "
                "  t.created_at, t.updated_at "
                "FROM tasks t "
                "WHERE t.id IN (SELECT DISTINCT id FROM chain) "
                "ORDER BY t.created_at ASC LIMIT ?",
                (root_task_id, MAX_WORKFLOW_CHAIN_DEPTH, limit),
            ).fetchall()
            if not rows:
                return None
            # Independent child-existence check — derive
            # ``terminal_without_children`` from the full tasks table, not
            # just the snapshot rows. Otherwise a non-leaf at the ``limit``
            # boundary whose only children fell off the cap would falsely
            # report ``terminal_without_children=True``. Uses the
            # ``idx_tasks_parent_task_id`` index so this is O(#done-stages)
            # lookups inside the same connection block.
            #
            # ``previous_task_id`` descendants (rework) are DELIBERATELY
            # NOT considered here — a rework is a retry of the same
            # stage, not a downstream handoff. A done task whose only
            # follow-up is a rework is still a chain break (workflow
            # stalled). The rework appears as its own stage in the
            # snapshot via the CTE's ``previous_task_id`` edge walk, but
            # it doesn't suppress its predecessor's chain-break flag.
            # Mirrors :meth:`chain_breaks_24h` and is pinned by
            # ``test_previous_task_id_chain_does_not_suppress_chain_break``.
            done_task_ids = [row[0] for row in rows if row[4] == "done"]
            parent_ids_with_children: set[str] = set()
            if done_task_ids:
                placeholders = ",".join("?" * len(done_task_ids))
                child_rows = conn.execute(
                    f"SELECT DISTINCT parent_task_id FROM tasks "
                    f"WHERE parent_task_id IN ({placeholders})",
                    done_task_ids,
                ).fetchall()
                parent_ids_with_children = {
                    r[0] for r in child_rows if r[0]
                }
        stages: list[dict] = []
        # Derive buckets from ``VALID_STATUSES`` so a future state added
        # to the enum can't silently drift out of the summary math.
        summary = {status: 0 for status in VALID_STATUSES}
        summary["total"] = 0
        for row in rows:
            (
                tid, parent, previous, assignee, status, title,
                blocker_note, created_at, updated_at,
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
            # ``previous_task_id`` surfaced only when set so the operator
            # can spot rework stages (and which task triggered them)
            # without an extra ``get_task`` call. Nominal handoff stages
            # have ``previous_task_id IS NULL`` and stay quiet.
            if previous:
                stage["previous_task_id"] = previous
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
        result_summary: str | None = None,
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
        # Single choke point for the failure-reason column: redact secrets,
        # collapse noisy raw dumps (truncated tool-call payloads, empty
        # exceptions) to stable codes, and bound length — BEFORE the value
        # is persisted at the UPDATE below or echoed into the
        # ``status_changed`` event payload. Every write path funnels here.
        blocker_note = normalize_blocker_note(blocker_note)
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
                    "SELECT status, team_id, assignee, title, "
                    "outcome, parent_task_id "
                    "FROM tasks WHERE id=?",
                    (task_id,),
                ).fetchone()
                if not row:
                    conn.execute("ROLLBACK")
                    raise TaskNotFound(task_id)
                (
                    current, team_id, assignee, task_title, task_outcome,
                    task_parent_id,
                ) = (
                    row[0], row[1], row[2], row[3], row[4], row[5],
                )
                if current == status:
                    # No-op transition. Record the event so the audit
                    # trail still shows the call, but skip the row update.
                    # Exception: a same-status re-transition may still carry
                    # a worker ``result_summary`` (e.g. the loop's post-turn
                    # auto-close fires ``done`` after the worker already
                    # self-closed ``done`` via complete_task) — persist it so
                    # the deliverable isn't lost to the no-op path.
                    if result_summary is not None:
                        conn.execute(
                            "UPDATE tasks SET "
                            "result_summary=COALESCE(?, result_summary) "
                            "WHERE id=?",
                            (result_summary, task_id),
                        )
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
                    "blocker_note=?, "
                    "result_summary=COALESCE(?, result_summary) WHERE id=?",
                    (
                        status, now, completed_at, retention_until,
                        # ``blocker_note`` is the canonical non-success
                        # status-reason column. ``blocked`` (recoverable —
                        # the original semantic) and ``failed`` (terminal —
                        # Bug 3 "silent model rejection") both populate it.
                        # ``cancelled`` deliberately stays None: a manual
                        # cancellation isn't an error, the user has the
                        # context. ``done`` clears whatever was there.
                        blocker_note if status in ("blocked", "failed") else None,
                        # ``result_summary`` is the worker's deliverable.
                        # COALESCE preserves a previously-captured summary
                        # when a later (non-summary) transition lands.
                        result_summary,
                        task_id,
                    ),
                )
                self._emit_event(
                    conn, task_id, "status_changed", actor,
                    {
                        "from": current,
                        "to": status,
                        # Mirror the column + bus-payload gating: only
                        # blocked/failed carry a reason. A note on any other
                        # transition is stale context that must not be echoed
                        # into the audit event row (exposed via
                        # /mesh/tasks/{id}/events).
                        "blocker_note": (
                            blocker_note
                            if status in ("blocked", "failed")
                            else None
                        ),
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
                            "parent_task_id": task_parent_id,
                            "team_id": team_id,
                            "title": task_title,
                            "ts": datetime.now(timezone.utc).isoformat(),
                        }
                conn.execute("COMMIT")
                emitted_change = (
                    current, status, team_id, assignee,
                    task_title, task_outcome,
                )
            except (TaskNotFound, InvalidStatusTransition):
                raise
            except Exception:
                conn.execute("ROLLBACK")
                raise
        if emitted_change is not None:
            (
                old_status, new_status, team_id, assignee,
                task_title, task_outcome,
            ) = emitted_change
            payload: dict = {
                "task_id": task_id,
                "team_id": team_id,
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
                # Carry the (already-normalized) failure reason so the SPA can
                # live-patch it and bucket by audience without a full reload.
                # Mirror the persistence rule: only blocked/failed keep a note;
                # other transitions explicitly send null so a stale note clears.
                "blocker_note": (
                    blocker_note if new_status in ("blocked", "failed") else None
                ),
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
        #
        # **Race window**: between COMMIT and this emit, another transaction
        # COULD insert a child task with parent_task_id=this task. That
        # child would cause a false-positive chain_break event. Acceptable
        # for an observability signal — the operator can re-query
        # workflow_snapshot to see current state. The alternative
        # (re-check inside another BEGIN IMMEDIATE before emit) trades a
        # tighter race for two additional lock acquisitions on every done
        # transition; not worth it for a signal the operator can trivially
        # verify by re-reading.
        #
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
                    "SELECT assignee, status, team_id, title, outcome "
                    "FROM tasks WHERE id=?",
                    (task_id,),
                ).fetchone()
                if not row:
                    conn.execute("ROLLBACK")
                    raise TaskNotFound(task_id)
                (
                    old_assignee, current_status, team_id,
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
                emitted = (current_status, team_id, task_title, task_outcome)
            except (TaskNotFound, InvalidStatusTransition):
                raise
            except Exception:
                conn.execute("ROLLBACK")
                raise
        if emitted is not None:
            current_status, team_id, task_title, task_outcome = emitted
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
                    "team_id": team_id,
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
        emitted_team: str | None = None
        emitted_committed = False
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT artifact_refs_json, team_id FROM tasks WHERE id=?",
                    (task_id,),
                ).fetchone()
                if not row:
                    conn.execute("ROLLBACK")
                    raise TaskNotFound(task_id)
                refs = json.loads(row[0]) if row[0] else []
                emitted_team = row[1]
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
                    "team_id": emitted_team,
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
                    "SELECT status, outcome, team_id, assignee "
                    "FROM tasks WHERE id=?",
                    (task_id,),
                ).fetchone()
                if not row:
                    conn.execute("ROLLBACK")
                    raise TaskNotFound(task_id)
                current_status, current_outcome, team_id, assignee = row
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
                emitted = (current_status, team_id, assignee, current_outcome)
            except (TaskNotFound, InvalidStatusTransition):
                raise
            except Exception:
                conn.execute("ROLLBACK")
                raise
        if emitted is not None:
            current_status, team_id, assignee, previous_outcome = emitted
            self._safe_emit(
                "task_outcome",
                agent=actor,
                data={
                    "task_id": task_id,
                    "team_id": team_id,
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

        Inherits ``assignee`` and ``team_id`` from ``previous_task_id``
        so the same agent picks up the redo. The new task's title is
        ``"Rework: {previous_title}"`` and its description (the brief
        the agent reads) is the operator's feedback followed by an
        ``## Original brief`` section carrying the previous description
        (A3 — feedback alone left the agent guessing what the original
        ask was). ``previous_task_id`` is set on the new row so the
        lineage is queryable.

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
        # A3 — the rework brief used to be the feedback ALONE; the agent
        # picking it up had no idea what the original ask was (lineage
        # existed only as the previous_task_id column, never surfaced in
        # the prompt). Append the original brief, feedback first so a
        # length cut never trims the actionable part.
        new_description = feedback
        prev_desc = (previous.get("description") or "").strip()
        if prev_desc and prev_desc != previous.get("title"):
            new_description = (
                f"{feedback}\n\n## Original brief\n{prev_desc[:1000]}"
            )
        new_id = f"task_{uuid.uuid4().hex[:12]}"
        now = time.time()
        # Session observability (Phase 1) — a rework is a continuation of
        # the same human-rooted session, so inherit the original task's
        # trace_id (NULL if the source predates trace stamping).
        inherited_trace_id = previous.get("trace_id")
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO tasks "
                "(id, team_id, parent_task_id, title, description, "
                "creator, assignee, status, priority, dependencies_json, "
                "artifact_refs_json, blocker_note, origin_kind, "
                "origin_channel, origin_user, created_at, updated_at, "
                "completed_at, retention_until, outcome, feedback_text, "
                "previous_task_id, trace_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, NULL, NULL, "
                "NULL, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, ?, ?)",
                (
                    new_id,
                    previous.get("team_id"),
                    previous.get("parent_task_id"),
                    new_title,
                    new_description,
                    actor,
                    previous["assignee"],
                    previous.get("priority", 0),
                    (previous.get("origin") or {}).get("kind"),
                    (previous.get("origin") or {}).get("channel"),
                    (previous.get("origin") or {}).get("user"),
                    now, now,
                    previous_task_id,
                    inherited_trace_id,
                ),
            )
            self._emit_event(
                conn, new_id, "created", actor,
                {
                    "title": new_title,
                    "assignee": previous["assignee"],
                    "team_id": previous.get("team_id"),
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
                "team_id": previous.get("team_id"),
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
                "team_id": previous.get("team_id"),
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
