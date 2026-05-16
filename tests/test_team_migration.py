"""Coverage for the idempotent project → team startup migrator.

See :mod:`src.host.team_migration` for the production code. The
migrator runs once on each mesh startup, and the back-compat
invariants on the rest of the codebase depend on its filesystem +
workspace + DB steps being idempotent and safe under partial-state
recovery. Each step here is tested in isolation plus end-to-end
through :func:`migrate_project_to_team`.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from src.host.team_migration import (
    _migrate_filesystem,
    _migrate_tasks_db,
    _migrate_workspaces,
    migrate_project_to_team,
)


def _make_legacy_project_dir(root: Path, name: str = "alpha") -> Path:
    pdir = root / "config" / "projects" / name
    pdir.mkdir(parents=True)
    (pdir / "metadata.yaml").write_text(f"name: {name}\n")
    return pdir


def _make_legacy_workspace(root: Path, agent_id: str, body: str = "# Fleet ctx\n") -> Path:
    ws = root / "data" / "agents" / agent_id / "workspace"
    ws.mkdir(parents=True)
    (ws / "PROJECT.md").write_text(body)
    return ws


def _make_legacy_tasks_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(
            """
            CREATE TABLE tasks (
                id TEXT PRIMARY KEY,
                project_id TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                title TEXT NOT NULL
            );
            CREATE INDEX idx_tasks_project_status ON tasks (project_id, status);
            """
        )
        conn.execute(
            "INSERT INTO tasks(id, project_id, status, title) VALUES (?, ?, ?, ?)",
            ("t1", "alpha", "pending", "first"),
        )
        conn.commit()
    finally:
        conn.close()


# ── Filesystem step ───────────────────────────────────────────────


def test_filesystem_renames_projects_to_teams_and_drops_symlink(tmp_path):
    _make_legacy_project_dir(tmp_path)
    changed = _migrate_filesystem(tmp_path)
    assert changed is True
    teams = tmp_path / "config" / "teams"
    projects = tmp_path / "config" / "projects"
    assert teams.is_dir()
    assert (teams / "alpha" / "metadata.yaml").is_file()
    # Back-compat symlink so legacy code paths still find the data.
    assert projects.is_symlink()
    assert projects.resolve() == teams.resolve()


def test_filesystem_idempotent(tmp_path):
    _make_legacy_project_dir(tmp_path)
    assert _migrate_filesystem(tmp_path) is True
    # Subsequent calls are no-ops.
    assert _migrate_filesystem(tmp_path) is False
    assert _migrate_filesystem(tmp_path) is False


def test_filesystem_both_dirs_present_warns_and_skips(tmp_path, caplog):
    # Pre-existing real ``config/teams/`` AND real ``config/projects/``
    # is the unmerged state — we refuse to touch anything.
    _make_legacy_project_dir(tmp_path, "alpha")
    teams = tmp_path / "config" / "teams"
    teams.mkdir()
    (teams / "beta").mkdir()
    with caplog.at_level("WARNING"):
        changed = _migrate_filesystem(tmp_path)
    assert changed is False
    assert (tmp_path / "config" / "projects" / "alpha").is_dir()
    assert (teams / "beta").is_dir()


def test_filesystem_no_dirs_is_noop(tmp_path):
    (tmp_path / "config").mkdir()
    assert _migrate_filesystem(tmp_path) is False
    assert not (tmp_path / "config" / "teams").exists()
    assert not (tmp_path / "config" / "projects").exists()


def test_filesystem_adds_symlink_when_only_teams_present(tmp_path):
    # Fresh deploy: ``config/teams/`` exists, no legacy symlink yet.
    teams = tmp_path / "config" / "teams"
    teams.mkdir(parents=True)
    (teams / "alpha").mkdir()
    changed = _migrate_filesystem(tmp_path)
    assert changed is True
    projects = tmp_path / "config" / "projects"
    assert projects.is_symlink()
    assert projects.resolve() == teams.resolve()


# ── Workspace step ────────────────────────────────────────────────


def test_workspace_copies_project_md_to_team_md(tmp_path):
    _make_legacy_workspace(tmp_path, "agent-a", body="# alpha\n")
    _make_legacy_workspace(tmp_path, "agent-b", body="# beta\n")
    copied = _migrate_workspaces(tmp_path)
    assert copied == 2
    a = (tmp_path / "data" / "agents" / "agent-a" / "workspace")
    assert (a / "TEAM.md").read_text() == "# alpha\n"
    # Legacy file preserved.
    assert (a / "PROJECT.md").read_text() == "# alpha\n"


def test_workspace_skips_agents_with_existing_team_md(tmp_path):
    ws = _make_legacy_workspace(tmp_path, "agent-a", body="# legacy\n")
    (ws / "TEAM.md").write_text("# already\n")
    copied = _migrate_workspaces(tmp_path)
    assert copied == 0
    assert (ws / "TEAM.md").read_text() == "# already\n"


def test_workspace_idempotent(tmp_path):
    _make_legacy_workspace(tmp_path, "agent-a")
    assert _migrate_workspaces(tmp_path) == 1
    assert _migrate_workspaces(tmp_path) == 0


def test_workspace_no_agents_dir_is_noop(tmp_path):
    assert _migrate_workspaces(tmp_path) == 0


def test_workspace_copies_lowercase_project_md_to_team_md(tmp_path):
    # Rare but possible: a workspace has only the lowercase
    # ``project.md`` (not the uppercase ``PROJECT.md``). The migrator
    # must produce the matching lowercase ``team.md`` so the bootstrap
    # read order in ``src/agent/workspace.py``
    # (``["TEAM.md", "team.md", "PROJECT.md", "project.md"]``) finds
    # the new file first instead of falling back to the legacy lowercase
    # name forever. (Cross-case-sensitivity-of-FS sentinels are
    # intentionally omitted — macOS HFS+/APFS defaults conflate
    # ``project.md`` and ``PROJECT.md``; the load-bearing contract is
    # only that the matching-case canonical was produced.)
    ws = tmp_path / "data" / "agents" / "agent-lower" / "workspace"
    ws.mkdir(parents=True)
    (ws / "project.md").write_text("# lower\n")
    copied = _migrate_workspaces(tmp_path)
    assert copied == 1
    assert (ws / "team.md").read_text() == "# lower\n"
    # Legacy lowercase file preserved (not moved).
    assert (ws / "project.md").read_text() == "# lower\n"


def test_workspace_copies_both_case_pairs_independently(tmp_path):
    # A workspace with BOTH legacy variants gets both canonicals
    # produced — each pair is handled independently. On case-insensitive
    # filesystems (macOS default) the two pairs collapse onto the same
    # inode, so we only assert this where the FS actually distinguishes
    # the cases.
    ws = tmp_path / "data" / "agents" / "agent-mixed" / "workspace"
    ws.mkdir(parents=True)
    (ws / "PROJECT.md").write_text("# upper\n")
    # Detect a case-insensitive filesystem by re-reading the lowercase
    # name after writing the uppercase file. If they match, the FS is
    # case-insensitive — skip the second-pair assertion.
    try:
        case_insensitive = (ws / "project.md").read_text() == "# upper\n"
    except OSError:
        case_insensitive = False
    if case_insensitive:
        pytest.skip("case-insensitive filesystem; pair semantics covered elsewhere")
    (ws / "project.md").write_text("# lower\n")
    copied = _migrate_workspaces(tmp_path)
    assert copied == 2
    assert (ws / "TEAM.md").read_text() == "# upper\n"
    assert (ws / "team.md").read_text() == "# lower\n"


# ── DB column step ────────────────────────────────────────────────


def test_db_renames_project_id_column(tmp_path):
    db = tmp_path / "tasks.db"
    _make_legacy_tasks_db(db)
    changed = _migrate_tasks_db(db)
    assert changed is True
    conn = sqlite3.connect(str(db))
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(tasks)").fetchall()]
        assert "team_id" in cols
        assert "project_id" not in cols
        idx_names = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'",
            ).fetchall()
        }
        assert "idx_tasks_team_status" in idx_names
        assert "idx_tasks_project_status" not in idx_names
        # Row data preserved through the rename.
        row = conn.execute(
            "SELECT id, team_id, status, title FROM tasks WHERE id=?", ("t1",),
        ).fetchone()
        assert row == ("t1", "alpha", "pending", "first")
    finally:
        conn.close()


def test_db_idempotent(tmp_path):
    db = tmp_path / "tasks.db"
    _make_legacy_tasks_db(db)
    assert _migrate_tasks_db(db) is True
    assert _migrate_tasks_db(db) is False


def test_db_no_file_is_noop(tmp_path):
    assert _migrate_tasks_db(tmp_path / "absent.db") is False


def test_db_no_tasks_table_is_noop(tmp_path):
    db = tmp_path / "empty.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE other (id TEXT)")
    conn.commit()
    conn.close()
    assert _migrate_tasks_db(db) is False


def test_db_already_migrated_drops_legacy_index(tmp_path):
    # Half-migrated schema: column already team_id but a legacy
    # project-named index lingered. The migrator should drop the
    # legacy index and add the canonical one.
    db = tmp_path / "tasks.db"
    conn = sqlite3.connect(str(db))
    try:
        conn.executescript(
            """
            CREATE TABLE tasks (
                id TEXT PRIMARY KEY,
                team_id TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                title TEXT
            );
            CREATE INDEX idx_tasks_project_status ON tasks (team_id, status);
            """
        )
        conn.commit()
    finally:
        conn.close()
    # No rename needed but the cleanup step runs and returns False.
    assert _migrate_tasks_db(db) is False
    conn = sqlite3.connect(str(db))
    try:
        idx_names = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'",
            ).fetchall()
        }
        assert "idx_tasks_project_status" not in idx_names
        assert "idx_tasks_team_status" in idx_names
    finally:
        conn.close()


# ── End-to-end ────────────────────────────────────────────────────


def test_end_to_end_migration(tmp_path):
    _make_legacy_project_dir(tmp_path, "alpha")
    _make_legacy_workspace(tmp_path, "agent-a", body="# fleet\n")
    db = tmp_path / "tasks.db"
    _make_legacy_tasks_db(db)
    # PR 3 flipped the DB rename to default-on. No env var needed.
    result = migrate_project_to_team(repo_root=tmp_path, tasks_db=db)
    assert result == {"filesystem": True, "workspaces": 1, "db_column": True}
    # Second run is fully idempotent.
    result2 = migrate_project_to_team(repo_root=tmp_path, tasks_db=db)
    assert result2 == {"filesystem": False, "workspaces": 0, "db_column": False}


def test_disable_flag_honored(tmp_path, monkeypatch):
    _make_legacy_project_dir(tmp_path)
    db = tmp_path / "tasks.db"
    _make_legacy_tasks_db(db)
    monkeypatch.setenv("OPENLEGION_DISABLE_TEAM_MIGRATION", "1")
    result = migrate_project_to_team(repo_root=tmp_path, tasks_db=db)
    assert result.get("skipped") is True
    # Nothing on disk should have moved.
    assert (tmp_path / "config" / "projects" / "alpha").is_dir()
    assert not (tmp_path / "config" / "teams").exists()


def test_db_step_opt_out_keeps_project_id(tmp_path, monkeypatch):
    """``OPENLEGION_TEAM_MIGRATION_RENAME_DB=0`` skips the column rename.

    The orchestration layer's PRAGMA-driven column resolution keeps
    legacy-column instances working without source changes, so the
    opt-out is a safe emergency-rollback escape hatch.
    """
    _make_legacy_project_dir(tmp_path)
    db = tmp_path / "tasks.db"
    _make_legacy_tasks_db(db)
    monkeypatch.setenv("OPENLEGION_TEAM_MIGRATION_RENAME_DB", "0")
    result = migrate_project_to_team(repo_root=tmp_path, tasks_db=db)
    assert result["filesystem"] is True
    assert result["db_column"] is False
    # Column should still be project_id since the rename was opted out.
    conn = sqlite3.connect(str(db))
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(tasks)").fetchall()]
        assert "project_id" in cols
        assert "team_id" not in cols
    finally:
        conn.close()


def test_db_step_runs_by_default(tmp_path):
    """The DB column rename is default-on after PR 3."""
    _make_legacy_project_dir(tmp_path)
    db = tmp_path / "tasks.db"
    _make_legacy_tasks_db(db)
    result = migrate_project_to_team(repo_root=tmp_path, tasks_db=db)
    assert result["filesystem"] is True
    assert result["db_column"] is True
    conn = sqlite3.connect(str(db))
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(tasks)").fetchall()]
        assert "team_id" in cols
        assert "project_id" not in cols
    finally:
        conn.close()


def test_dry_run_reports_pending_changes(tmp_path):
    _make_legacy_project_dir(tmp_path)
    _make_legacy_workspace(tmp_path, "agent-a")
    db = tmp_path / "tasks.db"
    _make_legacy_tasks_db(db)
    result = migrate_project_to_team(repo_root=tmp_path, tasks_db=db, dry_run=True)
    assert result["dry_run"] is True
    assert result["filesystem"] is True
    assert result["workspaces"] == 1
    assert result["db_column"] is True
    # Nothing actually mutated.
    assert (tmp_path / "config" / "projects" / "agent-a" / "workspace").exists() is False
    assert (tmp_path / "config" / "teams").exists() is False
    assert not (tmp_path / "data" / "agents" / "agent-a" / "workspace" / "TEAM.md").exists()


def test_downgrade_symlink_keeps_legacy_path_readable(tmp_path):
    """Old code that still reads config/projects/ keeps working after migrate."""
    _make_legacy_project_dir(tmp_path, "alpha")
    migrate_project_to_team(repo_root=tmp_path, tasks_db=tmp_path / "tasks.db")
    # Read via the legacy path — symlink resolves to the new dir.
    legacy_meta = tmp_path / "config" / "projects" / "alpha" / "metadata.yaml"
    assert legacy_meta.exists()
    assert "name: alpha" in legacy_meta.read_text()


@pytest.mark.skipif(
    os.name == "nt", reason="symlink semantics differ on Windows"
)
def test_symlink_failure_logs_and_continues(tmp_path, monkeypatch, caplog):
    _make_legacy_project_dir(tmp_path)

    real_symlink_to = Path.symlink_to

    def boom(self, target):  # pragma: no cover - tested through caplog
        raise OSError("symlink unsupported")

    monkeypatch.setattr(Path, "symlink_to", boom)
    try:
        with caplog.at_level("WARNING"):
            changed = _migrate_filesystem(tmp_path)
    finally:
        monkeypatch.setattr(Path, "symlink_to", real_symlink_to)
    assert changed is True
    # Filesystem rename still happened even though the symlink couldn't.
    assert (tmp_path / "config" / "teams" / "alpha").is_dir()
    assert any("symlink failed" in rec.getMessage() for rec in caplog.records)
