"""Idempotent startup migrator for the project → team rename.

Runs once on each mesh startup and is a no-op on subsequent invocations.
Three steps, each independently safe:

1. **Filesystem**: rename ``config/projects/`` → ``config/teams/`` and drop
   a ``config/projects`` symlink pointing at the new directory so any
   pre-rename code path that still reads ``config/projects/`` keeps
   working (downgrade safety).
2. **Workspaces**: for every agent workspace under
   ``data/agents/*/workspace/`` that has ``PROJECT.md`` but no
   ``TEAM.md``, copy ``PROJECT.md`` → ``TEAM.md``. The original is
   preserved — the workspace bootstrap reads ``TEAM.md`` first then
   falls back to ``PROJECT.md`` (see :mod:`src.agent.workspace`).
3. **SQLite tasks column**: rename ``tasks.project_id`` →
   ``tasks.team_id``. PR 3 flipped this step to **default-on** —
   :func:`migrate_project_to_team` runs the rename unless an operator
   opts out via ``OPENLEGION_TEAM_MIGRATION_RENAME_DB=0``. The
   orchestration layer is column-name aware (introspects the live
   column via PRAGMA at init) so an opted-out instance still reads
   and writes via the legacy column without code changes.

Failure of any step logs and continues — the caller is expected to
wrap the whole entrypoint in try/except so a migration error doesn't
crash mesh startup. Back-compat aliases throughout the codebase keep
the previous-version data path working.

Escape hatch: set ``OPENLEGION_DISABLE_TEAM_MIGRATION=1`` to skip the
migration entirely (useful when an operator needs to pin a downgrade).
"""

from __future__ import annotations

import os
import shutil
import sqlite3
from pathlib import Path
from typing import Any

from src.shared.utils import setup_logging

logger = setup_logging("host.team_migration")


def _migrate_filesystem(repo_root: Path) -> bool:
    """Rename ``config/projects/`` → ``config/teams/`` if needed.

    Returns True when a rename was performed this call. Idempotent —
    when ``config/teams/`` already exists as a real directory and
    ``config/projects`` is already the symlink, nothing happens.

    Edge cases:

    * Both directories exist as real dirs (unmerged state from a prior
      partial migration): WARN and skip. Operator must intervene.
    * Only ``config/projects/`` exists: rename it, symlink the legacy
      name to it.
    * Only ``config/teams/`` exists: drop a back-compat symlink at
      ``config/projects`` if one isn't already there.
    * Neither exists: nothing to do.
    """
    projects_dir = repo_root / "config" / "projects"
    teams_dir = repo_root / "config" / "teams"

    projects_is_link = projects_dir.is_symlink()
    projects_is_dir = projects_dir.exists() and not projects_is_link and projects_dir.is_dir()
    teams_exists = teams_dir.exists()
    teams_is_dir = teams_exists and not teams_dir.is_symlink() and teams_dir.is_dir()

    if projects_is_dir and teams_is_dir:
        logger.warning(
            "team_migration: BOTH config/projects/ and config/teams/ exist as real "
            "directories — refusing to merge. Operator must reconcile manually.",
        )
        return False

    # Case: legacy real dir exists, new dir doesn't — do the rename.
    if projects_is_dir and not teams_exists:
        teams_dir.parent.mkdir(parents=True, exist_ok=True)
        projects_dir.rename(teams_dir)
        # Best-effort downgrade symlink. If the symlink can't be made
        # (e.g. filesystem doesn't support symlinks on Windows), log
        # and continue — the new directory is enough for the new code
        # path and the legacy fallback won't fire.
        try:
            projects_dir.symlink_to(teams_dir.name)
        except OSError as e:
            logger.warning(
                "team_migration: rename succeeded but symlink failed: %s — "
                "downgrade safety is reduced",
                e,
            )
        return True

    # Case: new dir exists, legacy symlink missing — add it for
    # downgrade safety. This covers a fresh install on the new code
    # plus the case where a previous migration completed but the
    # symlink was removed.
    if teams_is_dir and not projects_dir.exists() and not projects_is_link:
        try:
            projects_dir.symlink_to(teams_dir.name)
            logger.info("team_migration: added downgrade symlink config/projects")
            return True
        except OSError as e:
            logger.warning("team_migration: downgrade symlink creation failed: %s", e)
            return False

    return False


def _migrate_workspaces(repo_root: Path) -> int:
    """Copy ``PROJECT.md`` → ``TEAM.md`` in every agent workspace.

    Handles both case variants independently — workspaces may have only
    the uppercase or only the lowercase file (workspace bootstrap reads
    in order ``["TEAM.md", "team.md", "PROJECT.md", "project.md"]``).
    Idempotent: skips workspaces that already have the corresponding
    ``TEAM.md`` / ``team.md`` file. Returns the count of copies
    performed this call (each case-pair counted independently — a
    workspace with both uppercase and lowercase pairs contributes up
    to 2).
    """
    agents_root = repo_root / "data" / "agents"
    if not agents_root.exists():
        return 0
    # Use ``is_file()`` (rejects symlinks-to-non-files) for the legacy
    # source check. Use ``exists()`` for the canonical target check so
    # we don't clobber an existing symlink the operator may have set up.
    pairs: tuple[tuple[str, str], ...] = (
        ("PROJECT.md", "TEAM.md"),
        ("project.md", "team.md"),
    )
    count = 0
    for agent_dir in agents_root.iterdir():
        if not agent_dir.is_dir():
            continue
        workspace = agent_dir / "workspace"
        if not workspace.is_dir():
            continue
        for legacy_name, canonical_name in pairs:
            legacy = workspace / legacy_name
            canonical = workspace / canonical_name
            if legacy.is_file() and not canonical.exists():
                try:
                    shutil.copy2(legacy, canonical)
                    count += 1
                except OSError as e:
                    logger.warning(
                        "team_migration: failed to copy %s → %s: %s",
                        legacy, canonical, e,
                    )
    return count


def _column_names(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [row[1] for row in rows]


def _migrate_tasks_db(db_path: Path) -> bool:
    """Rename ``tasks.project_id`` → ``tasks.team_id`` on an existing DB.

    Returns True when a rename was performed this call. No-op when the
    DB doesn't exist, the ``tasks`` table is absent, or the column has
    already been renamed.
    """
    if not db_path.exists():
        return False
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA busy_timeout=30000")
        # Confirm the table exists; treat its absence as nothing to do.
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='tasks'",
        ).fetchall()
        if not tables:
            return False
        cols = _column_names(conn, "tasks")
        if "team_id" in cols:
            # Already migrated — drop the legacy index if it lingered
            # and ensure the new index exists, then bail out.
            conn.execute("DROP INDEX IF EXISTS idx_tasks_project_status")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_team_status "
                "ON tasks (team_id, status)",
            )
            conn.commit()
            return False
        if "project_id" not in cols:
            # Fresh schema that never had the legacy column — nothing
            # to rename. Caller is responsible for creating the new
            # column at schema-init time.
            return False
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute("BEGIN")
        try:
            conn.execute("ALTER TABLE tasks RENAME COLUMN project_id TO team_id")
            conn.execute("DROP INDEX IF EXISTS idx_tasks_project_status")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_team_status "
                "ON tasks (team_id, status)",
            )
            conn.execute("COMMIT")
        except sqlite3.Error:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.execute("PRAGMA foreign_keys=ON")
        return True
    finally:
        conn.close()


def migrate_project_to_team(
    *,
    dry_run: bool = False,
    repo_root: Path | None = None,
    tasks_db: Path | None = None,
) -> dict[str, Any]:
    """Run all three migration steps. Idempotent.

    ``dry_run=True`` returns the same shape but performs no writes — the
    counts reflect what *would* have been done. ``repo_root`` defaults
    to the engine's :data:`src.cli.config.PROJECT_ROOT`; tests pass
    ``tmp_path`` and a synthesized ``tasks_db`` to keep the migration
    self-contained.

    Returns ``{"filesystem": bool, "workspaces": int, "db_column": bool}``.
    Honors ``OPENLEGION_DISABLE_TEAM_MIGRATION=1`` — when set, returns a
    skipped result without touching disk or DB.
    """
    if os.environ.get("OPENLEGION_DISABLE_TEAM_MIGRATION") == "1":
        logger.info("team_migration: skipped via OPENLEGION_DISABLE_TEAM_MIGRATION=1")
        return {
            "filesystem": False,
            "workspaces": 0,
            "db_column": False,
            "skipped": True,
        }

    if repo_root is None:
        from src.cli.config import PROJECT_ROOT
        repo_root = PROJECT_ROOT
    if tasks_db is None:
        tasks_db = repo_root / "data" / "tasks.db"

    if dry_run:
        # Inspect-only: don't touch disk or DB. We can still compute
        # the would-be counts cheaply.
        projects_dir = repo_root / "config" / "projects"
        teams_dir = repo_root / "config" / "teams"
        fs_change = (
            projects_dir.exists()
            and not projects_dir.is_symlink()
            and not teams_dir.exists()
        )
        ws_count = 0
        agents_root = repo_root / "data" / "agents"
        if agents_root.exists():
            for agent_dir in agents_root.iterdir():
                ws = agent_dir / "workspace"
                # Mirror the runtime copy step's case-pair semantics so
                # dry-run counts agree with the actual run. On
                # case-insensitive filesystems (macOS HFS+/APFS default)
                # the two pairs alias onto the same inode — dedupe by
                # (device, inode) to avoid double counting.
                seen: set[tuple[int, int]] = set()
                for legacy_name, canonical_name in (
                    ("PROJECT.md", "TEAM.md"),
                    ("project.md", "team.md"),
                ):
                    legacy = ws / legacy_name
                    canonical = ws / canonical_name
                    if not legacy.is_file() or canonical.exists():
                        continue
                    try:
                        st = legacy.stat()
                        key = (st.st_dev, st.st_ino)
                    except OSError:
                        continue
                    if key in seen:
                        continue
                    seen.add(key)
                    ws_count += 1
        db_change = False
        if tasks_db.exists():
            conn = sqlite3.connect(str(tasks_db))
            try:
                tables = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='tasks'",
                ).fetchall()
                if tables:
                    cols = _column_names(conn, "tasks")
                    db_change = "project_id" in cols and "team_id" not in cols
            finally:
                conn.close()
        return {
            "filesystem": fs_change,
            "workspaces": ws_count,
            "db_column": db_change,
            "dry_run": True,
        }

    result: dict[str, Any] = {
        "filesystem": False,
        "workspaces": 0,
        "db_column": False,
    }
    try:
        result["filesystem"] = _migrate_filesystem(repo_root)
    except Exception as e:
        logger.warning("team_migration: filesystem step failed: %s", e)
    try:
        result["workspaces"] = _migrate_workspaces(repo_root)
    except Exception as e:
        logger.warning("team_migration: workspace step failed: %s", e)
    # DB column rename is **default-on** as of PR 3. Operators can
    # opt out with ``OPENLEGION_TEAM_MIGRATION_RENAME_DB=0`` to keep
    # the legacy ``project_id`` column for emergency rollback — the
    # orchestration layer auto-detects which name is live via PRAGMA
    # introspection at init, so either column shape works without
    # source changes.
    if os.environ.get("OPENLEGION_TEAM_MIGRATION_RENAME_DB", "1") != "0":
        try:
            result["db_column"] = _migrate_tasks_db(tasks_db)
        except Exception as e:
            logger.warning("team_migration: db column step failed: %s", e)

    logger.info(
        "teams_migration_complete: filesystem=%s workspaces=%d db_column=%s",
        result["filesystem"], result["workspaces"], result["db_column"],
    )
    return result
