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

from src.shared.utils import setup_logging

logger = setup_logging("host.teams")

# Same character rules as agent names (see cli/config._validate_agent_name):
# 1-64 chars, alphanumeric/hyphen/underscore, starts with a letter or digit.
# Teams and agents share namespaces downstream (blackboard prefixes, envs),
# so the identifier grammar must stay aligned.
_TEAM_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

VALID_TEAM_STATUSES = ("active", "archived")


class TeamNotFound(LookupError):
    """Raised when an operation references a team id with no row."""


class TeamExists(ValueError):
    """Raised when creating a team whose id is already taken."""


def validate_team_id(team_id: str) -> str:
    """Validate and return a safe team id (same rules as agent names)."""
    if not isinstance(team_id, str) or not _TEAM_ID_RE.match(team_id):
        raise ValueError(
            f"Invalid team name '{team_id}': must be 1–64 alphanumeric chars, "
            "hyphens, or underscores (must start with a letter or digit)."
        )
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
                    thread_ref         TEXT
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

                PRAGMA user_version = 1;
                """
            )

    # ── row helpers ──────────────────────────────────────────────

    _TEAM_COLS = (
        "id, description, status, created_at, north_star, success_criteria, "
        "settings, budget_daily_usd, budget_monthly_usd, drive_ref, thread_ref"
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
        return self.get_team(team_id)  # type: ignore[return-value]

    def _scaffold_files(self, team_id: str, description: str) -> None:
        if self.teams_dir is None:
            return
        team_dir = self.teams_dir / team_id
        try:
            team_dir.mkdir(parents=True, exist_ok=True)
            (team_dir / "workflows").mkdir(exist_ok=True)
            team_md = team_dir / "team.md"
            if not team_md.exists():
                team_md.write_text(
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
        the permission layer, not the store.
        """
        with self._conn() as conn:
            members = self._members_unlocked(conn, team_id)
            cur = conn.execute("DELETE FROM teams WHERE id = ?", (team_id,))
            if cur.rowcount == 0:
                raise TeamNotFound(f"Team '{team_id}' not found")
            conn.execute("DELETE FROM team_members WHERE team_id = ?", (team_id,))
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

    # ── membership ───────────────────────────────────────────────

    def add_member(self, team_id: str, agent_id: str) -> str | None:
        """Assign an agent to a team, evicting any previous membership.

        Returns the agent's previous team id (or None) so the caller can
        rewire blackboard permissions for the move.
        """
        if agent_id == "operator":
            raise ValueError("Operator is a system agent and cannot be assigned to teams")
        with self._conn() as conn:
            exists = conn.execute("SELECT 1 FROM teams WHERE id = ?", (team_id,)).fetchone()
            if exists is None:
                raise TeamNotFound(f"Team '{team_id}' not found")
            row = conn.execute("SELECT team_id FROM team_members WHERE agent_id = ?", (agent_id,)).fetchone()
            old_team = row[0] if row else None
            if old_team == team_id:
                return old_team
            conn.execute(
                "INSERT OR REPLACE INTO team_members (agent_id, team_id, joined_at) VALUES (?, ?, ?)",
                (agent_id, team_id, _now()),
            )
        return old_team

    def remove_member(self, team_id: str, agent_id: str) -> bool:
        """Remove an agent from a team. Returns True if a row was removed."""
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM team_members WHERE agent_id = ? AND team_id = ?",
                (agent_id, team_id),
            )
            return cur.rowcount > 0

    def remove_agent(self, agent_id: str) -> str | None:
        """Drop an agent's membership + standing goals (agent deletion).

        Returns the team it was removed from, if any.
        """
        with self._conn() as conn:
            row = conn.execute("SELECT team_id FROM team_members WHERE agent_id = ?", (agent_id,)).fetchone()
            conn.execute("DELETE FROM team_members WHERE agent_id = ?", (agent_id,))
            conn.execute("DELETE FROM agent_goals WHERE agent_id = ?", (agent_id,))
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

    # ── files ────────────────────────────────────────────────────

    def team_md_path(self, team_id: str) -> Path | None:
        """Absolute path of the team's shared brief, or None in pure-DB mode."""
        if self.teams_dir is None:
            return None
        return self.teams_dir / validate_team_id(team_id) / "team.md"
