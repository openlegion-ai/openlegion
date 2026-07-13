"""TrackRecordStore — durable, append-only per-agent outcome ledger.

Plan §8 #18 (docs/plans/2026-07-04-agent-employee-platform-architecture.md):
a recon correction on §6's "raw material already collected, just
composed" — that claim is wrong on DURABILITY. Rated ``tasks`` rows
reap at 90 days (``orchestration.py``) and ``work_summaries`` at 30
(``summaries.py``); a live-query composition over either source would
silently lose history the moment a row ages out. This store is the
durable Layer-3 ledger those two (and Team-Drive-review resolution)
feed AT RATING TIME: one ``outcome_events`` row appended host-side, in
the same code paths that already call ``feedback_push`` (task
``set_outcome``, summary ``set_rating``) plus the drive-review
merge/reject paths. APPEND-ONLY, NEVER REAPED — no ``retention_until``
column, no reap method — that durability is precisely what tells it
apart from the two reaped sources it's assembled from.

Rating-trust rule (load-bearing, pinned by test): earned-autonomy
scoring (plan §8 #19) must use only HUMAN ratings (plus objective
``system`` signals) at full weight. Operator-*agent* ratings — the
internal mesh paths an agent's own heartbeat/tool call can reach —
are EXCLUDED from autonomy scoring (agents grading agents must not
feed the trust ladder) while still counted here and still feeding
``feedback_push`` learning. Callers that need the autonomy-safe view
pass ``rater_kinds=AUTONOMY_RATER_KINDS`` to :meth:`counts_for_agent`.

Enum values are NEVER unified across sources — task outcomes
(accepted/rework/rejected/acknowledged), summary ratings
(accepted/acknowledged/rework), and drive-review resolutions
(merged/rejected, plus ``auto_merged``/``auto_merge_flagged``/
``auto_merge_reverted`` from §8 #20's kernel-executed auto-merge) each
keep their own raw vocabulary. Counts are reported keyed by
``(source, outcome)`` — no invented numeric score (none exists
anywhere else in the codebase either).

Self-reinforcement guard (§8 #20, load-bearing, pinned by test):
:meth:`pair_trust`'s auto-merge trust floor counts ``rater_kind="human"``
merged events ONLY — a kernel-executed auto-merge is written with
``rater_kind="system"`` and outcome ``auto_merged`` (never ``merged``)
precisely so it can never feed the very floor that gates further
auto-merges for the pair. ``system`` events are still counted (as
``auto_merged``, read by the sampling-decay schedule), just never
folded into the floor's ``merged`` count.

Storage follows the canonical-v1 pattern (mirrors ``ThreadStore``): one
``executescript``, no lazy ``ALTER`` chains, ``PRAGMA user_version = 1``,
WAL + ``busy_timeout``, env override ``OPENLEGION_TRACK_RECORD_DB``.
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

logger = setup_logging("host.track_record")

VALID_SOURCES: frozenset[str] = frozenset({"task_outcome", "summary_rating", "drive_review"})

# Rater-kind vocabulary. ``human`` = a real user, via the dashboard's
# human-driven surfaces; ``operator_agent`` = the same rating recorded
# through a mesh-reachable (agent-callable) path — the operator trust
# tier is still an agent identity that can act from its own heartbeat;
# ``system`` = a deterministic mesh-computed signal (reserved for a
# future objective-signal writer). See the module docstring's
# rating-trust rule for why the split exists and matters.
VALID_RATER_KINDS: frozenset[str] = frozenset({"human", "operator_agent", "system"})

# The rating-trust rule, pinned: only these rater kinds count toward
# earned-autonomy scoring (plan §8 #19 reads counts filtered to this).
AUTONOMY_RATER_KINDS: tuple[str, ...] = ("human", "system")

# Hard cap on details_json size so a runaway payload can't bloat the
# durable ledger (mirrors the summaries/threads store caps).
MAX_DETAILS_BYTES = 8192


class TrackRecordStore:
    """SQLite-backed, append-only outcome ledger. Never reaped.

    Disk-backed access opens a fresh WAL connection per operation;
    ``:memory:`` keeps a single shared connection behind a lock
    (mirrors ``ThreadStore``).
    """

    _EVENT_COLS = (
        "id, agent_id, team_id, source, ref_id, outcome, "
        "rater_kind, rated_by, details_json, created_at"
    )

    def __init__(self, db_path: str = "data/track_record.db") -> None:
        self.db_path = db_path
        self._shared_conn: sqlite3.Connection | None = None
        self._mem_lock = threading.Lock()
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

    def _init_schema(self) -> None:
        # Canonical schema v1 — exactly one shape, no lazy ALTER chains.
        # Deliberately NO retention_until column and NO reap method: this
        # table is append-only for the life of the deployment (see module
        # docstring — that durability is the whole reason this store
        # exists alongside the two reaped sources it's assembled from).
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS outcome_events (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id     TEXT,
                    team_id      TEXT,
                    source       TEXT NOT NULL,
                    ref_id       TEXT NOT NULL,
                    outcome      TEXT NOT NULL,
                    rater_kind   TEXT NOT NULL,
                    rated_by     TEXT,
                    details_json TEXT,
                    created_at   REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_outcome_events_agent
                    ON outcome_events(agent_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_outcome_events_ref
                    ON outcome_events(source, ref_id);

                PRAGMA user_version = 1;
                """
            )

    @staticmethod
    def _row_to_dict(row: tuple) -> dict:
        details = None
        if row[8]:
            try:
                details = json.loads(row[8])
            except (ValueError, TypeError):
                details = {"_decode_error": True}
        return {
            "id": row[0],
            "agent_id": row[1],
            "team_id": row[2],
            "source": row[3],
            "ref_id": row[4],
            "outcome": row[5],
            "rater_kind": row[6],
            "rated_by": row[7],
            "details": details,
            "created_at": row[9],
        }

    # ── append ───────────────────────────────────────────────────

    def record(
        self,
        *,
        source: str,
        ref_id: str,
        outcome: str,
        rater_kind: str,
        agent_id: str | None = None,
        team_id: str | None = None,
        rated_by: str | None = None,
        details: dict | None = None,
    ) -> dict:
        """Append one outcome event and return it.

        Raises ``ValueError`` on a malformed call — write-site callers
        are expected to go through :func:`record_best_effort` (or an
        equivalent broad try/except) since a ledger write must NEVER
        fail the source operation (task outcome / summary rating /
        drive resolution) that triggered it.
        """
        if source not in VALID_SOURCES:
            raise ValueError(f"source must be one of {sorted(VALID_SOURCES)}, got {source!r}")
        if rater_kind not in VALID_RATER_KINDS:
            raise ValueError(f"rater_kind must be one of {sorted(VALID_RATER_KINDS)}, got {rater_kind!r}")
        if not ref_id:
            raise ValueError("ref_id is required")
        if not outcome:
            raise ValueError("outcome is required")
        if agent_id is None and team_id is None:
            raise ValueError("at least one of agent_id/team_id is required")
        details_json = None
        if details is not None:
            details_json = dumps_safe(details)
            if len(details_json.encode("utf-8", errors="replace")) > MAX_DETAILS_BYTES:
                # A truncated JSON document is worse than no document —
                # replace the whole payload with a marker (mirrors
                # ThreadStore.post_message's oversized-payload handling).
                details_json = dumps_safe({"truncated": True})
        now = time.time()
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO outcome_events "
                "(agent_id, team_id, source, ref_id, outcome, rater_kind, rated_by, details_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (agent_id, team_id, source, ref_id, outcome, rater_kind, rated_by, details_json, now),
            )
            event_id = cur.lastrowid
        return self.get(event_id)  # type: ignore[return-value]

    # ── reads ────────────────────────────────────────────────────

    def get(self, event_id: int) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                f"SELECT {self._EVENT_COLS} FROM outcome_events WHERE id = ?",
                (event_id,),
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def counts_for_agent(
        self,
        agent_id: str,
        *,
        rater_kinds: tuple[str, ...] | None = None,
    ) -> dict[str, dict[str, int]]:
        """Counts keyed ``source -> outcome -> count`` for one agent.

        ``rater_kinds`` restricts to those rater kinds when given — the
        rating-trust rule: pass :data:`AUTONOMY_RATER_KINDS` for the
        autonomy-safe view. ``None`` (default) counts every rater kind.
        """
        clauses = ["agent_id = ?"]
        params: list[Any] = [agent_id]
        if rater_kinds is not None:
            placeholders = ",".join("?" for _ in rater_kinds)
            clauses.append(f"rater_kind IN ({placeholders})")
            params.extend(rater_kinds)
        where = " AND ".join(clauses)
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT source, outcome, COUNT(*) FROM outcome_events "
                f"WHERE {where} GROUP BY source, outcome",
                params,
            ).fetchall()
        counts: dict[str, dict[str, int]] = {}
        for source, outcome, n in rows:
            counts.setdefault(source, {})[outcome] = n
        return counts

    def distinct_accepted_count(
        self,
        agent_id: str,
        *,
        rater_kinds: tuple[str, ...] = AUTONOMY_RATER_KINDS,
        outcome: str = "accepted",
    ) -> int:
        """Count DISTINCT deliverables whose LATEST outcome is ``outcome``.

        Unlike :meth:`counts_for_agent` (raw append-only row counts — kept
        for display/learning), this collapses the ledger to ONE event per
        ``(source, ref_id)`` — the newest by ``(created_at, id)`` — before
        counting. So re-rating the same task N times counts as ONE
        deliverable, and a later ``accepted -> rejected`` correction drops
        the deliverable from the count entirely (the latest event wins).

        Restricted to ``rater_kinds`` (the rating-trust rule — defaults to
        :data:`AUTONOMY_RATER_KINDS`) so an operator-*agent* re-rate can
        neither inflate the count nor retract a human acceptance: the
        autonomy-relevant state of a deliverable is decided only by the
        rater kinds that feed the trust ladder. Used by the probation
        preset's "after N accepted deliverables" release gate (plan §8
        #19) — the raw ledger stays the display/learning surface.

        Only ``task_outcome`` / ``summary_rating`` sources ever carry an
        ``accepted`` outcome (drive-review vocabulary is merged/rejected/
        auto_merged), so the default matches the pre-fix
        task_outcome+summary_rating restriction on WHICH sources count,
        while fixing the re-rate/retract miscount.
        """
        if not rater_kinds:
            return 0
        placeholders = ",".join("?" for _ in rater_kinds)
        params: list[Any] = [agent_id, *rater_kinds]
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT source, ref_id, outcome FROM outcome_events "
                f"WHERE agent_id = ? AND rater_kind IN ({placeholders}) "
                "ORDER BY created_at DESC, id DESC",
                params,
            ).fetchall()
        seen: set[tuple[str, str]] = set()
        count = 0
        for source, ref_id, ev_outcome in rows:
            key = (source, ref_id)
            if key in seen:
                # Older event for a deliverable whose latest was already
                # seen (rows are newest-first) — the latest already decided
                # this deliverable's state.
                continue
            seen.add(key)
            if ev_outcome == outcome:
                count += 1
        return count

    def recent_events(self, agent_id: str, limit: int = 20) -> list[dict]:
        """Newest-first events for one agent, hard-capped at 200."""
        limit = max(1, min(int(limit), 200))
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT {self._EVENT_COLS} FROM outcome_events "
                "WHERE agent_id = ? ORDER BY created_at DESC, id DESC LIMIT ?",
                (agent_id, limit),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def count_events(
        self,
        *,
        source: str,
        outcome: str,
        rater_kind: str | None = None,
        since: float | None = None,
    ) -> int:
        """Count ``outcome_events`` matching source/outcome (optionally
        rater_kind/since). Read-only helper for host-side kill-switch /
        rate-cap checks — e.g. plan §8 #20's per-day auto-merge cap
        (``source="drive_review", outcome="auto_merged",
        rater_kind="system", since=<midnight UTC epoch>``). A plain
        ``COUNT(*)`` over the append-only ledger, no new table.
        """
        clauses = ["source = ?", "outcome = ?"]
        params: list[Any] = [source, outcome]
        if rater_kind is not None:
            clauses.append("rater_kind = ?")
            params.append(rater_kind)
        if since is not None:
            clauses.append("created_at >= ?")
            params.append(since)
        where = " AND ".join(clauses)
        with self._conn() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) FROM outcome_events WHERE {where}", params
            ).fetchone()
        return int(row[0]) if row else 0

    def pair_trust(self, lead_agent_id: str, submitter_agent_id: str) -> dict:
        """Auto-merge trust signal for a (lead, submitter) pair (§8 #19/#20).

        Scans this agent's ``drive_review`` events, split by
        ``rater_kind``:

        * ``human`` events whose ``details_json`` carries
          ``lead_agent_id`` == ``lead_agent_id`` and
          ``lead_verdict == "approve"`` count toward ``merged`` /
          ``rejected_after_approve`` depending on ``outcome`` —
          ``lead_agent_id`` in details carries the EXACT verdict author
          for reviews resolved after §8 #20 landed
          (``_record_drive_review_outcome`` prefers the review's
          ``lead_verdict_by``, falling back to the team's then-current
          lead only for rows recorded before that column existed — the
          pre-U4 approximation, preserved for old data only).
        * ``human`` events with outcome ``auto_merge_flagged`` /
          ``auto_merge_reverted`` (§8 #20 steps 6/7 — operator trust-decay
          signals) count toward ``flagged``, matched the same way but
          WITHOUT requiring ``lead_verdict == "approve"`` in details (a
          decay signal about an already-auto-merged review, whose
          original verdict was necessarily an approve).
        * ``system`` events (the kernel's own ``auto_merged`` writes) are
          NEVER folded into ``merged`` — the self-reinforcement guard: an
          auto-merge must not feed the very trust floor that gates
          further auto-merges. Counted separately as ``auto_merged`` —
          read by the sampling-rate decay schedule (§8 #20 step 5), not
          the trust floor.

        Floor policy (read by the auto-merge consumer, not enforced
        here): eligible iff ``merged >= trust_floor`` AND
        ``rejected_after_approve == 0`` AND ``flagged == 0``.
        """
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT {self._EVENT_COLS} FROM outcome_events "
                "WHERE source = 'drive_review' AND agent_id = ? "
                "ORDER BY created_at DESC, id DESC",
                (submitter_agent_id,),
            ).fetchall()
        merged = 0
        rejected_after_approve = 0
        flagged = 0
        auto_merged = 0
        last_event_at: float | None = None
        for r in rows:
            event = self._row_to_dict(r)
            details = event.get("details") or {}
            if details.get("lead_agent_id") != lead_agent_id:
                continue
            outcome = event["outcome"]
            if event["rater_kind"] == "system":
                # Self-reinforcement guard (§8 #20) — tracked separately,
                # never counted toward `merged`.
                if outcome == "auto_merged":
                    auto_merged += 1
                continue
            if outcome in ("auto_merge_flagged", "auto_merge_reverted"):
                flagged += 1
                if last_event_at is None:
                    last_event_at = event["created_at"]
                continue
            if details.get("lead_verdict") != "approve":
                continue
            if last_event_at is None:
                # Rows are newest-first — the first qualifying row IS
                # the most recent one for this pair.
                last_event_at = event["created_at"]
            if outcome == "merged":
                merged += 1
            elif outcome == "rejected":
                rejected_after_approve += 1
        return {
            "lead_agent_id": lead_agent_id,
            "submitter_agent_id": submitter_agent_id,
            "merged": merged,
            "rejected_after_approve": rejected_after_approve,
            "flagged": flagged,
            "auto_merged": auto_merged,
            "last_event_at": last_event_at,
        }


def record_best_effort(store: "TrackRecordStore | None", **kwargs: Any) -> None:
    """Append an outcome event, swallowing every error.

    Call-site contract (plan §8 #18): a ledger write failure must NEVER
    fail the source operation (task outcome / summary rating / drive
    resolution) that triggered it. Every write point in ``server.py`` /
    ``dashboard/server.py`` calls this instead of ``store.record`` so
    that contract holds without repeating a try/except at each site.
    A ``None`` store (standalone constructions that didn't wire one) is
    silently a no-op — the ledger is best-effort infrastructure, not a
    hard dependency of the operations it observes.
    """
    if store is None:
        return
    try:
        store.record(**kwargs)
    except Exception as e:
        logger.warning(
            "track record event write failed (source=%s ref_id=%s): %s",
            kwargs.get("source"), kwargs.get("ref_id"), e,
        )
