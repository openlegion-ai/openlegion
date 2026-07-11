"""TeamStore — the first-class team entity store (SQLite).

Phase-1 of the agent-employee platform plan (docs/plans/
2026-07-04-agent-employee-platform-architecture.md §6): `team` graduates
from a bolt-on (YAML dirs globbed per-request + an in-memory reverse
map) to a real store. This module is THE single authority for:

  * team identity + metadata (description, status, created_at, settings)
  * team goals (north_star / success_criteria — ratified decision #7,
    C.3-b: goals live here, not on the blackboard or metadata.yaml)
  * membership (strictly one team per agent — enforced by the
    ``team_members.agent_id`` PRIMARY KEY, not by caller discipline)
  * the team budget envelope columns (enforced pre-flight at the LLM
    proxy; unset/NULL/0 = UNLIMITED per plan B4 — the opposite of the
    per-agent ledger's "0 blocks everything" arithmetic)
  * per-agent standing goals (``agent_goals`` — keyed by agent alone so
    goals follow the agent across team moves)
  * drive/thread pointers reserved for Phase 2 (Team Drive / Threads)

The on-disk artifacts under ``config/teams/{id}/`` (``team.md`` shared
brief + ``workflows/``) remain plain files — ``team.md`` is bind-mounted
into member containers — but their create/delete lifecycle is owned here
so "team exists" has exactly one owner.

Storage follows the canonical-v1 pattern (PR #1185): one executescript,
no lazy ALTER chains, ``PRAGMA user_version = 1``. Disk-backed access
opens a fresh WAL connection per operation; ``:memory:`` keeps a single
shared connection behind a lock for tests.
"""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from src.shared.types import RESERVED_AGENT_IDS
from src.shared.utils import generate_id, setup_logging

logger = setup_logging("host.teams")

# Same character rules as agent names (see cli/config._validate_agent_name):
# 1-64 chars, alphanumeric/hyphen/underscore, starts with a letter or digit.
# Teams and agents share namespaces downstream (blackboard prefixes, envs),
# so the identifier grammar must stay aligned.
_TEAM_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")

VALID_TEAM_STATUSES = ("active", "archived")


class TeamNotFound(LookupError):
    """Raised when an operation references a team id with no row."""


class TeamExists(ValueError):
    """Raised when creating a team whose id is already taken."""


def validate_team_id(team_id: str) -> str:
    """Validate and return a safe team id (same rules as agent names).

    Reserved internal agent ids double as reserved team ids: teams and
    agents share downstream namespaces (blackboard prefixes, ``TEAM_NAME``
    env, ``_caller_teams`` sentinels for operator/mesh), so a team named
    ``mesh`` or ``operator`` would shadow system identities.
    """
    if not isinstance(team_id, str) or not _TEAM_ID_RE.fullmatch(team_id):
        raise ValueError(
            f"Invalid team name '{team_id}': must be 1–64 alphanumeric chars, "
            "hyphens, or underscores (must start with a letter or digit)."
        )
    if team_id in RESERVED_AGENT_IDS:
        raise ValueError(f"Team name '{team_id}' is reserved for internal use")
    return team_id


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TeamStore:
    """SQLite-backed store for teams, membership, and standing goals.

    ``teams_dir`` points at the on-disk home of per-team files
    (``team.md`` + ``workflows/``). When ``None`` no file scaffolding is
    performed (pure-DB mode for tests that don't touch containers).
    """

    def __init__(
        self,
        db_path: str = "data/teams.db",
        *,
        teams_dir: str | Path | None = None,
    ) -> None:
        self.db_path = db_path
        self.teams_dir = Path(teams_dir) if teams_dir is not None else None
        # Team Drive provisioner (Phase-2 unit 1, plan A.3 #3): the store
        # owns the drive LIFECYCLE, the runtime backend owns the STORAGE.
        # Wired via set_drive_provisioner in cli/runtime; pure-DB mode
        # (tests / browser tenant lookup) leaves both None → no-op.
        self._drive_ensure = None
        self._drive_remove = None
        self._shared_conn: sqlite3.Connection | None = None
        self._mem_lock = threading.Lock()
        if db_path == ":memory:":
            self._shared_conn = sqlite3.connect(":memory:", isolation_level=None, check_same_thread=False)
            self._shared_conn.execute("PRAGMA busy_timeout=30000")
        else:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

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
        """A connection wrapped in an immediate transaction.

        Multi-statement mutators (read-then-write) go through this so a
        concurrent writer — including another process via the CLI's
        on-demand store — can't interleave between the read and the
        write (e.g. add_member's old-team read vs delete_team's sweep).
        """
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")

    def _init_schema(self) -> None:
        # Canonical schema v1 — exactly one shape, no lazy ALTER chains,
        # no legacy column detection (plan §5 "Remove", mirrors Tasks).
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS teams (
                    id                 TEXT PRIMARY KEY,
                    description        TEXT NOT NULL DEFAULT '',
                    status             TEXT NOT NULL DEFAULT 'active',
                    created_at         TEXT,
                    north_star         TEXT,
                    success_criteria   TEXT,
                    settings           TEXT NOT NULL DEFAULT '{}',
                    budget_daily_usd   REAL,
                    budget_monthly_usd REAL,
                    drive_ref          TEXT,
                    thread_ref         TEXT,
                    lead_agent_id      TEXT
                );

                -- One team per agent, enforced by the schema: the agent id
                -- is the PRIMARY KEY, so a second membership REPLACEs the
                -- first instead of accumulating.
                CREATE TABLE IF NOT EXISTS team_members (
                    agent_id  TEXT PRIMARY KEY,
                    team_id   TEXT NOT NULL,
                    joined_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_team_members_team
                    ON team_members(team_id);

                -- Per-agent standing goals (ratified #7 / C.3-b): keyed by
                -- agent alone so goals follow the agent across team moves.
                CREATE TABLE IF NOT EXISTS agent_goals (
                    agent_id   TEXT PRIMARY KEY,
                    goals      TEXT NOT NULL,
                    set_by     TEXT,
                    updated_at TEXT
                );

                -- Team Drive review queue (Phase-2 unit 1). One row per
                -- submitted branch review; resubmitting the same branch
                -- supersedes the older open row. status:
                -- open | merging | merged | rejected | superseded.
                -- head_sha pins the branch tip reviewed, so the merge path
                -- integrates that EXACT commit (approval TOCTOU guard).
                CREATE TABLE IF NOT EXISTS drive_reviews (
                    id                TEXT PRIMARY KEY,
                    team_id           TEXT NOT NULL,
                    branch            TEXT NOT NULL,
                    author            TEXT NOT NULL,
                    title             TEXT NOT NULL DEFAULT '',
                    summary           TEXT NOT NULL DEFAULT '',
                    status            TEXT NOT NULL DEFAULT 'open',
                    reviewer          TEXT,
                    merge_commit      TEXT,
                    head_sha          TEXT,
                    created_at        TEXT,
                    resolved_at       TEXT,
                    lead_verdict      TEXT,
                    lead_verdict_note TEXT,
                    lead_verdict_at   TEXT,
                    lead_verdict_by   TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_drive_reviews_team
                    ON drive_reviews(team_id, status);

                PRAGMA user_version = 1;
                """
            )

    # ── row helpers ──────────────────────────────────────────────

    _TEAM_COLS = (
        "id, description, status, created_at, north_star, success_criteria, "
        "settings, budget_daily_usd, budget_monthly_usd, drive_ref, thread_ref, "
        "lead_agent_id"
    )

    @staticmethod
    def _row_to_team(row: tuple) -> dict:
        def _json(field, default):
            if not field:
                return default
            try:
                return json.loads(field)
            except (ValueError, TypeError):
                return default

        return {
            "id": row[0],
            "name": row[0],  # id IS the name — one identity, no divergence
            "description": row[1],
            "status": row[2] or "active",
            "created_at": row[3],
            "north_star": row[4],
            "success_criteria": _json(row[5], None),
            "settings": _json(row[6], {}),
            "budget_daily_usd": row[7],
            "budget_monthly_usd": row[8],
            "drive_ref": row[9],
            "thread_ref": row[10],
            "lead_agent_id": row[11],
        }

    def _members_unlocked(self, conn: sqlite3.Connection, team_id: str) -> list[str]:
        rows = conn.execute(
            "SELECT agent_id FROM team_members WHERE team_id = ? ORDER BY rowid",
            (team_id,),
        ).fetchall()
        return [r[0] for r in rows]

    # ── team CRUD ────────────────────────────────────────────────

    def create_team(self, team_id: str, description: str = "") -> dict:
        """Create a team row + on-disk scaffold. Raises TeamExists."""
        team_id = validate_team_id(team_id)
        with self._conn() as conn:
            try:
                conn.execute(
                    "INSERT INTO teams (id, description, created_at) VALUES (?, ?, ?)",
                    (team_id, description, _now()),
                )
            except sqlite3.IntegrityError:
                raise TeamExists(f"Team '{team_id}' already exists")
        self._scaffold_files(team_id, description)
        self._provision_drive(team_id, wipe_stale=True)
        return self.get_team(team_id)  # type: ignore[return-value]

    def _scaffold_files(self, team_id: str, description: str) -> None:
        if self.teams_dir is None:
            return
        team_dir = self.teams_dir / team_id
        try:
            team_dir.mkdir(parents=True, exist_ok=True)
            (team_dir / "workflows").mkdir(exist_ok=True)
            # Always overwrite: create_team only runs for a NEW team row, so
            # any team.md already on disk is a stale leftover from a deleted
            # team of the same name — carrying it into member prompts would
            # leak the old team's context.
            (team_dir / "team.md").write_text(
                f"# {team_id}\n\n{description}\n\n<!-- Shared context for all agents in this team -->\n"
            )
        except OSError:
            logger.exception("Failed to scaffold files for team %s", team_id)

    def get_team(self, team_id: str) -> dict | None:
        """Return the team dict (with ``members``) or None."""
        with self._conn() as conn:
            row = conn.execute(f"SELECT {self._TEAM_COLS} FROM teams WHERE id = ?", (team_id,)).fetchone()
            if row is None:
                return None
            team = self._row_to_team(row)
            team["members"] = self._members_unlocked(conn, team_id)
            return team

    def team_exists(self, team_id: str) -> bool:
        with self._conn() as conn:
            return conn.execute("SELECT 1 FROM teams WHERE id = ?", (team_id,)).fetchone() is not None

    def list_teams(self, include_archived: bool = True) -> dict[str, dict]:
        """Return ``{team_id: team_dict}`` (members included), sorted by id."""
        with self._conn() as conn:
            rows = conn.execute(f"SELECT {self._TEAM_COLS} FROM teams ORDER BY id").fetchall()
            member_rows = conn.execute("SELECT agent_id, team_id FROM team_members ORDER BY rowid").fetchall()
        members_by_team: dict[str, list[str]] = {}
        for agent_id, team_id in member_rows:
            members_by_team.setdefault(team_id, []).append(agent_id)
        teams: dict[str, dict] = {}
        for row in rows:
            team = self._row_to_team(row)
            if not include_archived and team["status"] == "archived":
                continue
            team["members"] = members_by_team.get(team["id"], [])
            teams[team["id"]] = team
        return teams

    def count_teams(self, include_archived: bool = True) -> int:
        with self._conn() as conn:
            if include_archived:
                row = conn.execute("SELECT COUNT(*) FROM teams").fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM teams WHERE status != 'archived'").fetchone()
            return int(row[0])

    def set_status(self, team_id: str, status: str) -> None:
        """Archive/unarchive. Status is free-string at the storage layer;
        callers restrict to VALID_TEAM_STATUSES."""
        with self._conn() as conn:
            cur = conn.execute("UPDATE teams SET status = ? WHERE id = ?", (status, team_id))
            if cur.rowcount == 0:
                raise TeamNotFound(f"Team '{team_id}' not found")

    def get_status(self, team_id: str) -> str:
        with self._conn() as conn:
            row = conn.execute("SELECT status FROM teams WHERE id = ?", (team_id,)).fetchone()
        if row is None:
            raise TeamNotFound(f"Team '{team_id}' not found")
        return row[0] or "active"

    def delete_team(self, team_id: str) -> list[str]:
        """Delete the team + membership rows + on-disk dir.

        Returns the (former) member list so the caller can strip their
        team blackboard permissions — permissions.json wiring stays with
        the permission layer, not the store. ``lead_agent_id`` needs no
        separate clear (plan §8 #14 integrity rule) — it dies with the
        team row itself.
        """
        with self._txn() as conn:
            members = self._members_unlocked(conn, team_id)
            cur = conn.execute("DELETE FROM teams WHERE id = ?", (team_id,))
            if cur.rowcount == 0:
                raise TeamNotFound(f"Team '{team_id}' not found")
            conn.execute("DELETE FROM team_members WHERE team_id = ?", (team_id,))
            conn.execute("DELETE FROM drive_reviews WHERE team_id = ?", (team_id,))
        if self._drive_remove is not None:
            try:
                self._drive_remove(team_id)
            except Exception:
                logger.exception("Drive removal for team %s failed", team_id)
        if self.teams_dir is not None:
            team_dir = self.teams_dir / validate_team_id(team_id)
            if team_dir.exists():
                try:
                    shutil.rmtree(team_dir)
                except OSError:
                    logger.exception("Failed to remove team dir for %s", team_id)
        return members

    # ── metadata / goals ─────────────────────────────────────────

    def set_description(self, team_id: str, description: str) -> None:
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE teams SET description = ? WHERE id = ?",
                (description, team_id),
            )
            if cur.rowcount == 0:
                raise TeamNotFound(f"Team '{team_id}' not found")

    def set_goal(
        self,
        team_id: str,
        north_star: str | None,
        success_criteria: list[str] | None = None,
    ) -> None:
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE teams SET north_star = ?, success_criteria = ? WHERE id = ?",
                (
                    north_star,
                    json.dumps(success_criteria) if success_criteria is not None else None,
                    team_id,
                ),
            )
            if cur.rowcount == 0:
                raise TeamNotFound(f"Team '{team_id}' not found")

    def set_settings(self, team_id: str, settings: dict) -> None:
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE teams SET settings = ? WHERE id = ?",
                (json.dumps(settings or {}), team_id),
            )
            if cur.rowcount == 0:
                raise TeamNotFound(f"Team '{team_id}' not found")

    def set_budget(
        self,
        team_id: str,
        daily_usd: float | None,
        monthly_usd: float | None,
    ) -> None:
        """Set the team budget envelope. NULL/0 = unlimited (plan B4)."""
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE teams SET budget_daily_usd = ?, budget_monthly_usd = ? WHERE id = ?",
                (daily_usd, monthly_usd, team_id),
            )
            if cur.rowcount == 0:
                raise TeamNotFound(f"Team '{team_id}' not found")

    def set_thread_ref(self, team_id: str, thread_id: str | None) -> None:
        """Point the team at its channel thread (Phase-2 Team Threads).

        Written at team create (and boot backfill) once the channel
        thread exists in the ThreadStore.
        """
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE teams SET thread_ref = ? WHERE id = ?",
                (thread_id, team_id),
            )
            if cur.rowcount == 0:
                raise TeamNotFound(f"Team '{team_id}' not found")

    # ── lead designation (plan §8 #14) ────────────────────────────
    #
    # The lead is TEAM DATA, not an identity tier: a nullable pointer
    # column, same shape as ``drive_ref`` / ``thread_ref``. Assignment
    # is operator-or-internal only (enforced at the endpoint layer, see
    # ``server.py``'s ``mesh_set_team_lead``); this store only enforces
    # the membership invariant — a lead MUST be a real team member.

    def set_lead(self, team_id: str, agent_id: str | None) -> dict:
        """Assign (``agent_id``) or clear (``None``) the team's lead.

        Validates real membership: ``agent_id`` must satisfy
        ``team_of(agent_id) == team_id`` — a lead who isn't on the team
        is a broken invariant. ``operator`` is rejected outright (a
        system agent, never team data). Raises ``TeamNotFound`` for an
        unknown team, ``ValueError`` for an invalid/non-member agent.
        """
        with self._txn() as conn:
            exists = conn.execute("SELECT 1 FROM teams WHERE id = ?", (team_id,)).fetchone()
            if exists is None:
                raise TeamNotFound(f"Team '{team_id}' not found")
            if agent_id is not None:
                if agent_id == "operator":
                    raise ValueError("Operator is a system agent and cannot be a team lead")
                member_row = conn.execute(
                    "SELECT team_id FROM team_members WHERE agent_id = ?", (agent_id,),
                ).fetchone()
                if member_row is None or member_row[0] != team_id:
                    raise ValueError(
                        f"Agent '{agent_id}' is not a member of team '{team_id}' — "
                        "the lead must be a real team member."
                    )
            conn.execute("UPDATE teams SET lead_agent_id = ? WHERE id = ?", (agent_id, team_id))
        return self.get_team(team_id)  # type: ignore[return-value]

    def led_team(self, agent_id: str) -> str | None:
        """The id of the team this agent leads, or None.

        A single indexless lookup — cheap enough to run on every
        heartbeat tick (the cron lead-duty probe uses this as its "am I
        a lead at all" fast path before paying for a reviews query).
        """
        with self._conn() as conn:
            row = conn.execute("SELECT id FROM teams WHERE lead_agent_id = ?", (agent_id,)).fetchone()
            return row[0] if row else None

    # ── membership ───────────────────────────────────────────────

    def add_member(self, team_id: str, agent_id: str) -> str | None:
        """Assign an agent to a team, evicting any previous membership.

        Returns the agent's previous team id (or None) so the caller can
        rewire blackboard permissions for the move.
        """
        if agent_id == "operator":
            raise ValueError("Operator is a system agent and cannot be assigned to teams")
        with self._txn() as conn:
            exists = conn.execute("SELECT 1 FROM teams WHERE id = ?", (team_id,)).fetchone()
            if exists is None:
                raise TeamNotFound(f"Team '{team_id}' not found")
            row = conn.execute("SELECT team_id FROM team_members WHERE agent_id = ?", (agent_id,)).fetchone()
            old_team = row[0] if row else None
            if old_team == team_id:
                return old_team
            if old_team is not None:
                # The agent is moving OFF its old team — if it was that
                # team's lead, the pointer must not dangle (integrity
                # rule, plan §8 #14: lead is REAL membership). Cleared in
                # the SAME transaction as the membership move.
                conn.execute(
                    "UPDATE teams SET lead_agent_id = NULL WHERE id = ? AND lead_agent_id = ?",
                    (old_team, agent_id),
                )
            conn.execute(
                "INSERT OR REPLACE INTO team_members (agent_id, team_id, joined_at) VALUES (?, ?, ?)",
                (agent_id, team_id, _now()),
            )
        return old_team

    def remove_member(self, team_id: str, agent_id: str) -> bool:
        """Remove an agent from a team. Returns True if a row was removed.

        Same-transaction integrity (plan §8 #14): if the removed agent
        was the team's lead, the pointer is cleared here — a lead who
        is no longer a member is a broken invariant.
        """
        with self._txn() as conn:
            cur = conn.execute(
                "DELETE FROM team_members WHERE agent_id = ? AND team_id = ?",
                (agent_id, team_id),
            )
            removed = cur.rowcount > 0
            if removed:
                conn.execute(
                    "UPDATE teams SET lead_agent_id = NULL WHERE id = ? AND lead_agent_id = ?",
                    (team_id, agent_id),
                )
            return removed

    def remove_agent(self, agent_id: str) -> str | None:
        """Drop an agent's membership + standing goals (agent deletion).

        Returns the team it was removed from, if any. Same-transaction
        integrity (plan §8 #14): clears ``lead_agent_id`` on any team
        that pointed at this agent (there can only be one, since lead
        implies real membership, but the clear is unconditional on
        agent_id — not scoped to the removed team_id — for defense in
        depth against a prior inconsistency).
        """
        with self._txn() as conn:
            row = conn.execute("SELECT team_id FROM team_members WHERE agent_id = ?", (agent_id,)).fetchone()
            conn.execute("DELETE FROM team_members WHERE agent_id = ?", (agent_id,))
            conn.execute("DELETE FROM agent_goals WHERE agent_id = ?", (agent_id,))
            conn.execute("UPDATE teams SET lead_agent_id = NULL WHERE lead_agent_id = ?", (agent_id,))
            return row[0] if row else None

    def team_of(self, agent_id: str) -> str | None:
        """The agent's team id, or None (standalone)."""
        with self._conn() as conn:
            row = conn.execute("SELECT team_id FROM team_members WHERE agent_id = ?", (agent_id,)).fetchone()
            return row[0] if row else None

    def members(self, team_id: str) -> list[str]:
        with self._conn() as conn:
            return self._members_unlocked(conn, team_id)

    def agent_team_map(self) -> dict[str, str]:
        """Full ``{agent_id: team_id}`` snapshot (boot listings, broadcasts)."""
        with self._conn() as conn:
            rows = conn.execute("SELECT agent_id, team_id FROM team_members").fetchall()
            return {r[0]: r[1] for r in rows}

    # ── per-agent standing goals (ratified #7 / C.3-b) ───────────

    def get_agent_goals(self, agent_id: str) -> dict | None:
        """Return ``{goals, set_by, updated_at}`` or None if unset."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT goals, set_by, updated_at FROM agent_goals WHERE agent_id = ?",
                (agent_id,),
            ).fetchone()
        if row is None:
            return None
        try:
            goals = json.loads(row[0])
        except (ValueError, TypeError):
            goals = []
        return {"goals": goals, "set_by": row[1], "updated_at": row[2]}

    def set_agent_goals(self, agent_id: str, goals: list[str], set_by: str = "operator") -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO agent_goals (agent_id, goals, set_by, updated_at) VALUES (?, ?, ?, ?)",
                (agent_id, json.dumps(list(goals)), set_by, _now()),
            )

    def clear_agent_goals(self, agent_id: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM agent_goals WHERE agent_id = ?", (agent_id,))
            return cur.rowcount > 0

    # ── Team Drive lifecycle (Phase-2 unit 1, plan A.3 #3) ────────

    def set_drive_provisioner(self, ensure_fn, remove_fn) -> None:
        """Wire the drive storage backend (``ensure(team_id) -> ref``,
        ``remove(team_id)``) — normally ``RuntimeBackend.ensure_team_volume``
        / ``remove_team_volume``. Until wired, drive provisioning is a
        silent no-op (pure-DB mode)."""
        self._drive_ensure = ensure_fn
        self._drive_remove = remove_fn

    def _set_drive_ref(self, team_id: str, ref: str | None) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE teams SET drive_ref = ? WHERE id = ?", (ref, team_id))

    def _provision_drive(self, team_id: str, *, wipe_stale: bool = False) -> str | None:
        """Provision the team's drive and persist ``drive_ref``.

        ``wipe_stale=True`` (the create_team path) removes any leftover
        repo dir first — create only runs for a NEW team row, so an
        existing dir is a stale remnant of a deleted team of the same
        name (Phase-1 finding #4 precedent: never adopt stale state).
        Failure is logged and leaves ``drive_ref`` NULL — a drive
        governor must never take down team management; the boot
        backfill retries on the next start.
        """
        if self._drive_ensure is None:
            return None
        try:
            if wipe_stale and self._drive_remove is not None:
                self._drive_remove(team_id)
            ref = self._drive_ensure(team_id)
        except Exception:
            logger.exception("Drive provisioning for team %s failed (drive_ref stays NULL)", team_id)
            return None
        self._set_drive_ref(team_id, ref)
        return ref

    def ensure_drive(self, team_id: str) -> str | None:
        """Return the team's drive_ref, provisioning/repairing on demand.

        Self-healing read path for the mesh drive endpoints: a NULL ref
        (create-time provision failure) or a ref whose directory is gone
        (disk wipe) re-provisions when a provisioner is wired. Returns
        None when the team has no drive and none can be created.
        """
        team = self.get_team(team_id)
        if team is None:
            raise TeamNotFound(f"Team '{team_id}' not found")
        ref = team.get("drive_ref")
        if ref and Path(ref).exists():
            return ref
        return self._provision_drive(team_id)

    def backfill_drives(self) -> list[str]:
        """Provision drives for teams missing one (boot backfill, mirrors
        the solo-ACL backfill pattern). Non-destructive: a team whose
        repo dir already exists on disk adopts it via the idempotent
        ensure — the wipe path is exclusive to create_team."""
        if self._drive_ensure is None:
            return []
        with self._conn() as conn:
            rows = conn.execute("SELECT id, drive_ref FROM teams").fetchall()
        provisioned: list[str] = []
        for team_id, ref in rows:
            if ref and Path(ref).exists():
                continue
            if self._provision_drive(team_id) is not None:
                provisioned.append(team_id)
        return provisioned

    # ── Team Drive reviews (review-before-integrate) ─────────────

    _REVIEW_COLS = (
        "id, team_id, branch, author, title, summary, status, "
        "reviewer, merge_commit, head_sha, created_at, resolved_at, "
        "lead_verdict, lead_verdict_note, lead_verdict_at, lead_verdict_by"
    )

    @staticmethod
    def _row_to_review(row: tuple) -> dict:
        head_sha = row[9]
        return {
            "id": row[0],
            "team_id": row[1],
            "branch": row[2],
            "author": row[3],
            "title": row[4],
            "summary": row[5],
            "status": row[6],
            "reviewer": row[7],
            "merge_commit": row[8],
            "head_sha": head_sha,
            # Short form for the operator's approval view (what they merge).
            "head_sha_short": (head_sha[:10] if head_sha else None),
            "created_at": row[10],
            "resolved_at": row[11],
            # Lead advisory verdict (plan §8 #13) — ZERO enforcement effect;
            # purely informational for the operator's merge/reject decision.
            "lead_verdict": row[12],
            "lead_verdict_note": row[13],
            "lead_verdict_at": row[14],
            # Verified reviewer identity that recorded the verdict (plan §8
            # #20, U4) — durable, exact (lead) attribution for kernel-
            # executed auto-merge's pair_trust, fixing U1's approximation
            # (which read the team's CURRENT lead at resolution time and
            # mis-attributed a pair across a lead swap). NULL for reviews
            # verdicted before this column existed.
            "lead_verdict_by": row[15],
        }

    def create_review(
        self,
        team_id: str,
        branch: str,
        author: str,
        title: str,
        summary: str = "",
        head_sha: str | None = None,
    ) -> dict:
        """Open a review for ``branch``. Any older OPEN review for the
        same (team, branch) is marked ``superseded`` in the same
        transaction — one live review per branch. ``head_sha`` pins the
        reviewed branch tip (the merge path integrates that exact commit)."""
        review_id = generate_id("rev")
        now = _now()
        with self._txn() as conn:
            if conn.execute("SELECT 1 FROM teams WHERE id = ?", (team_id,)).fetchone() is None:
                raise TeamNotFound(f"Team '{team_id}' not found")
            conn.execute(
                "UPDATE drive_reviews SET status = 'superseded', resolved_at = ? "
                "WHERE team_id = ? AND branch = ? AND status = 'open'",
                (now, team_id, branch),
            )
            conn.execute(
                "INSERT INTO drive_reviews "
                "(id, team_id, branch, author, title, summary, status, head_sha, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 'open', ?, ?)",
                (review_id, team_id, branch, author, title, summary, head_sha, now),
            )
        return self.get_review(review_id)  # type: ignore[return-value]

    def get_review(self, review_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                f"SELECT {self._REVIEW_COLS} FROM drive_reviews WHERE id = ?", (review_id,)
            ).fetchone()
        return self._row_to_review(row) if row else None

    def list_reviews(self, team_id: str, status: str | None = None) -> list[dict]:
        with self._conn() as conn:
            if status:
                rows = conn.execute(
                    f"SELECT {self._REVIEW_COLS} FROM drive_reviews "
                    "WHERE team_id = ? AND status = ? ORDER BY created_at DESC",
                    (team_id, status),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"SELECT {self._REVIEW_COLS} FROM drive_reviews WHERE team_id = ? ORDER BY created_at DESC",
                    (team_id,),
                ).fetchall()
        return [self._row_to_review(r) for r in rows]

    def record_lead_verdict(
        self,
        review_id: str,
        verdict: str,
        note: str | None,
        reviewer: str,
    ) -> dict:
        """Record the team lead's advisory approve/reject verdict.

        Plan §8 #13: this has ZERO enforcement effect — the merge/reject
        gates and their operator-or-internal check are untouched and
        keep governing integration. Allowed only while the review is
        still ``open`` (a verdict on an already-resolved review can't
        change anything and would just be confusing state); raises
        ``ValueError`` otherwise. ``reviewer`` is the verified lead
        identity the endpoint already checked against
        ``teams.lead_agent_id`` — stamped durably into ``lead_verdict_by``
        (plan §8 #20, U4) so kernel-executed auto-merge's ``pair_trust``
        can attribute the EXACT (lead, submitter) pair rather than
        approximating via the team's lead at resolution time.
        """
        if verdict not in ("approve", "reject"):
            raise ValueError(f"Invalid verdict '{verdict}'")
        with self._txn() as conn:
            row = conn.execute("SELECT status FROM drive_reviews WHERE id = ?", (review_id,)).fetchone()
            if row is None:
                raise ValueError(f"Review '{review_id}' not found")
            if row[0] != "open":
                raise ValueError(f"Review '{review_id}' is already {row[0]}")
            conn.execute(
                "UPDATE drive_reviews SET lead_verdict = ?, lead_verdict_note = ?, lead_verdict_at = ?, "
                "lead_verdict_by = ? WHERE id = ?",
                (verdict, note, _now(), reviewer, review_id),
            )
        logger.debug("lead verdict recorded on review %s by %s: %s", review_id, reviewer, verdict)
        return self.get_review(review_id)  # type: ignore[return-value]

    def resolve_review(
        self,
        review_id: str,
        status: str,
        reviewer: str,
        merge_commit: str | None = None,
    ) -> dict:
        """Transition an OPEN review to ``merged``/``rejected``. Raises
        ValueError if the review is missing or already resolved (the
        merge endpoint must not double-integrate)."""
        if status not in ("merged", "rejected"):
            raise ValueError(f"Invalid review resolution '{status}'")
        with self._txn() as conn:
            row = conn.execute("SELECT status FROM drive_reviews WHERE id = ?", (review_id,)).fetchone()
            if row is None:
                raise ValueError(f"Review '{review_id}' not found")
            if row[0] != "open":
                raise ValueError(f"Review '{review_id}' is already {row[0]}")
            conn.execute(
                "UPDATE drive_reviews SET status = ?, reviewer = ?, merge_commit = ?, resolved_at = ? WHERE id = ?",
                (status, reviewer, merge_commit, _now(), review_id),
            )
        return self.get_review(review_id)  # type: ignore[return-value]

    # ── atomic merge claim (open → merging → merged/open) ────────
    #
    # The merge endpoint is claim-first: it transitions open→merging in a
    # BEGIN IMMEDIATE transaction BEFORE any git side effect. A lost claim
    # (already merged/rejected/merging, or a racing second merge) 409s and
    # runs no git — so two merges can't both push, and a stray empty merge
    # commit can't land. reject acts on ``open`` only (never on a merging
    # row it does not own), so a concurrent reject can't flip the row to
    # ``rejected`` while a merge is integrating on main.

    def claim_review_for_merge(self, review_id: str) -> dict:
        """Atomically transition an OPEN review to ``merging``.

        Raises ValueError if the review is missing or not open — the caller
        MUST run no git side effect without a successful claim."""
        with self._txn() as conn:
            row = conn.execute("SELECT status FROM drive_reviews WHERE id = ?", (review_id,)).fetchone()
            if row is None:
                raise ValueError(f"Review '{review_id}' not found")
            if row[0] != "open":
                raise ValueError(f"Review '{review_id}' is already {row[0]}")
            conn.execute("UPDATE drive_reviews SET status = 'merging' WHERE id = ?", (review_id,))
        return self.get_review(review_id)  # type: ignore[return-value]

    def finalize_merge(self, review_id: str, merge_commit: str, reviewer: str) -> dict:
        """Transition a ``merging`` review to ``merged`` once main is
        integrated. Raises ValueError if the row is no longer ``merging``
        — but the merge commit is already on main, so the caller must
        surface the divergence rather than silently swallow it."""
        with self._txn() as conn:
            row = conn.execute("SELECT status FROM drive_reviews WHERE id = ?", (review_id,)).fetchone()
            if row is None:
                raise ValueError(f"Review '{review_id}' not found")
            if row[0] != "merging":
                raise ValueError(f"Review '{review_id}' is {row[0]}, expected merging")
            conn.execute(
                "UPDATE drive_reviews SET status = 'merged', reviewer = ?, merge_commit = ?, resolved_at = ? "
                "WHERE id = ?",
                (reviewer, merge_commit, _now(), review_id),
            )
        return self.get_review(review_id)  # type: ignore[return-value]

    def revert_merge_claim(self, review_id: str) -> None:
        """Roll a ``merging`` review back to ``open`` after a failed git
        merge (nothing reached main). No-op if the row already moved."""
        with self._txn() as conn:
            conn.execute(
                "UPDATE drive_reviews SET status = 'open' WHERE id = ? AND status = 'merging'",
                (review_id,),
            )

    # ── files ────────────────────────────────────────────────────

    def team_md_path(self, team_id: str) -> Path | None:
        """Absolute path of the team's shared brief, or None in pure-DB mode."""
        if self.teams_dir is None:
            return None
        return self.teams_dir / validate_team_id(team_id) / "team.md"
