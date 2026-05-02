"""One-shot blackboard → tasks migration tests (Task 6).

Includes auto-run-at-startup coverage for the rollout PR — the
``create_mesh_app`` factory invokes ``migrate_blackboard_to_tasks`` once
when the v2 flag is on so existing fleets transition seamlessly.
"""

from __future__ import annotations

import importlib
import logging

import pytest

from src.host.mesh import Blackboard, MessageRouter, PubSub
from src.host.orchestration import Tasks
from src.host.orchestration_migration import migrate_blackboard_to_tasks
from src.host.permissions import PermissionMatrix


def _make_blackboard(tmp_path) -> Blackboard:
    return Blackboard(db_path=str(tmp_path / "bb.db"))


def _make_tasks(tmp_path) -> Tasks:
    return Tasks(db_path=str(tmp_path / "tasks.db"))


def test_empty_blackboard_no_migration(tmp_path):
    bb = _make_blackboard(tmp_path)
    t = _make_tasks(tmp_path)
    summary = migrate_blackboard_to_tasks(bb, t)
    assert summary["migrated"] == 0
    assert summary["skipped"] == 0
    assert summary["deleted"] == 0
    assert summary["errors"] == []


def test_legacy_per_agent_task_migrated(tmp_path):
    bb = _make_blackboard(tmp_path)
    t = _make_tasks(tmp_path)
    bb.write(
        "tasks/analyst/ho_abc123",
        {
            "from": "scout",
            "summary": "research handoff",
            "status": "pending",
            "ts": 12345.0,
            "output_key": "output/scout/ho_abc123",
        },
        written_by="scout",
    )
    summary = migrate_blackboard_to_tasks(bb, t)
    assert summary["migrated"] == 1
    assert summary["deleted"] == 1
    assert summary["errors"] == []
    rec = t.get("ho_abc123")
    assert rec is not None
    assert rec["assignee"] == "analyst"
    assert rec["creator"] == "scout"
    assert rec["title"] == "research handoff"
    assert rec["artifact_refs"] == ["output/scout/ho_abc123"]
    # Legacy key gone.
    assert bb.read("tasks/analyst/ho_abc123") is None


def test_project_scoped_legacy_task_carries_project(tmp_path):
    bb = _make_blackboard(tmp_path)
    t = _make_tasks(tmp_path)
    bb.write(
        "projects/research/tasks/analyst/ho_xyz",
        {"from": "scout", "summary": "scoped"},
        written_by="scout",
    )
    summary = migrate_blackboard_to_tasks(bb, t)
    assert summary["migrated"] == 1
    rec = t.get("ho_xyz")
    assert rec is not None
    assert rec["project_id"] == "research"
    assert rec["assignee"] == "analyst"


def test_operator_inbox_task_migrates(tmp_path):
    bb = _make_blackboard(tmp_path)
    t = _make_tasks(tmp_path)
    bb.write(
        "global/tasks/operator/ho_op1",
        {"from": "analyst", "summary": "ping operator"},
        written_by="analyst",
    )
    summary = migrate_blackboard_to_tasks(bb, t)
    assert summary["migrated"] == 1
    rec = t.get("ho_op1")
    assert rec is not None
    assert rec["assignee"] == "operator"
    assert rec["project_id"] is None
    assert rec["creator"] == "analyst"


def test_migration_is_idempotent(tmp_path):
    bb = _make_blackboard(tmp_path)
    t = _make_tasks(tmp_path)
    bb.write(
        "tasks/analyst/ho_abc",
        {"from": "scout", "summary": "x"},
        written_by="scout",
    )
    first = migrate_blackboard_to_tasks(bb, t)
    assert first["migrated"] == 1
    # Second run: legacy keys are gone, new task already exists. Both
    # branches must be no-ops.
    second = migrate_blackboard_to_tasks(bb, t)
    assert second["migrated"] == 0
    assert second["skipped"] == 0
    assert t.get("ho_abc") is not None


def test_migration_preserves_status_when_legacy_was_working(tmp_path):
    bb = _make_blackboard(tmp_path)
    t = _make_tasks(tmp_path)
    bb.write(
        "tasks/analyst/ho_running",
        {"from": "scout", "summary": "in flight", "status": "working"},
        written_by="scout",
    )
    migrate_blackboard_to_tasks(bb, t)
    rec = t.get("ho_running")
    assert rec is not None
    assert rec["status"] == "working"


def test_migration_skips_already_present_then_cleans_legacy_key(tmp_path):
    """If a task row exists for the same handoff_id, count as skipped
    and still delete the now-redundant legacy key."""
    bb = _make_blackboard(tmp_path)
    t = _make_tasks(tmp_path)
    # Pre-seed the new table with a row whose id matches the legacy
    # handoff_id we're about to migrate.
    t.create(
        creator="scout",
        assignee="analyst",
        title="already migrated",
        task_id="ho_dupe",
    )
    bb.write(
        "tasks/analyst/ho_dupe",
        {"from": "scout", "summary": "stale legacy"},
        written_by="scout",
    )
    summary = migrate_blackboard_to_tasks(bb, t)
    assert summary["skipped"] == 1
    assert summary["deleted"] == 1
    # Pre-existing row's title is preserved (no overwrite).
    assert t.get("ho_dupe")["title"] == "already migrated"
    assert bb.read("tasks/analyst/ho_dupe") is None


def test_migration_handles_corrupt_value_gracefully(tmp_path):
    """A blackboard row whose value is not a dict should still migrate
    using fallback ``{"text": ...}`` semantics, not crash the pass."""
    bb = _make_blackboard(tmp_path)
    t = _make_tasks(tmp_path)
    bb.write(
        "tasks/analyst/ho_corrupt",
        # Blackboard.write requires a dict, so simulate the legacy-
        # corrupt case by writing under a key whose value happens to
        # have only a "text" key — the migration coerces these into
        # the "(migrated handoff)" placeholder.
        {"text": "not a real handoff dict"},
        written_by="scout",
    )
    summary = migrate_blackboard_to_tasks(bb, t)
    assert summary["migrated"] == 1
    rec = t.get("ho_corrupt")
    # Title falls back to "(migrated handoff)" since no summary key.
    assert rec is not None


# ── Auto-run at mesh startup (rollout PR) ─────────────────────────


def _reload_server(monkeypatch, *, v2_on: bool, tasks_db: str):
    """Reload ``src.host.server`` after pinning the env vars so the
    module-level ``_ORCHESTRATION_TASKS_V2`` constant picks up the new
    value.
    """
    if v2_on:
        monkeypatch.setenv("OPENLEGION_ORCHESTRATION_TASKS_V2", "1")
        monkeypatch.setenv("OPENLEGION_ORCHESTRATION_TASKS_DB", tasks_db)
    else:
        monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_V2", raising=False)
        monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_DB", raising=False)
    import src.host.server as server_module
    importlib.reload(server_module)
    return server_module


def _build_app(server_module, blackboard):
    """Build a minimal mesh app reusing the supplied blackboard so the
    auto-run migration sees its rows.
    """
    pubsub = PubSub()
    permissions = PermissionMatrix()
    router = MessageRouter(permissions, {})
    return server_module.create_mesh_app(
        blackboard=blackboard,
        pubsub=pubsub,
        router=router,
        permissions=permissions,
    )


def test_migration_auto_runs_at_startup(tmp_path, monkeypatch):
    """Building a mesh app with v2 on migrates legacy task rows once."""
    bb_path = str(tmp_path / "bb.db")
    bb = Blackboard(db_path=bb_path)
    bb.write(
        "tasks/scout/ho_xxx",
        {"from": "operator", "summary": "auto-migrate me", "status": "pending"},
        written_by="operator",
    )

    try:
        server_module = _reload_server(
            monkeypatch, v2_on=True, tasks_db=str(tmp_path / "tasks.db"),
        )
        app = _build_app(server_module, bb)

        # Auto-run lands the row in the tasks store.
        ts: Tasks | None = app.tasks_store
        assert ts is not None
        rec = ts.get("ho_xxx")
        assert rec is not None
        assert rec["assignee"] == "scout"
        assert rec["title"] == "auto-migrate me"
        # Legacy key was deleted.
        assert bb.read("tasks/scout/ho_xxx") is None
    finally:
        bb.close()
        # Restore the suite's idea of "default" by reloading once more
        # without the env vars set.
        monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_V2", raising=False)
        monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_DB", raising=False)
        import src.host.server as server_module
        importlib.reload(server_module)


def test_migration_auto_runs_idempotent(tmp_path, monkeypatch):
    """Building the mesh app twice over the same blackboard is a no-op
    on the second pass — idempotence keyed on the legacy ``handoff_id``.
    """
    bb_path = str(tmp_path / "bb.db")
    bb = Blackboard(db_path=bb_path)
    bb.write(
        "tasks/scout/ho_idem",
        {"from": "operator", "summary": "first pass migrates me"},
        written_by="operator",
    )

    try:
        tasks_db = str(tmp_path / "tasks.db")
        server_module = _reload_server(
            monkeypatch, v2_on=True, tasks_db=tasks_db,
        )

        # First pass.
        app1 = _build_app(server_module, bb)
        ts1: Tasks | None = app1.tasks_store
        assert ts1 is not None
        assert ts1.get("ho_idem") is not None
        # Close the first store handle so the second pass reopens
        # cleanly against the same on-disk file.
        try:
            ts1.close()
        except Exception:
            pass

        # Second pass — same on-disk tasks db, should find nothing to do.
        app2 = _build_app(server_module, bb)
        ts2: Tasks | None = app2.tasks_store
        assert ts2 is not None
        rec = ts2.get("ho_idem")
        # Row is still present (persisted across reopen), the second
        # auto-run did NOT re-create or duplicate it.
        assert rec is not None
        # No legacy key resurrected.
        assert bb.read("tasks/scout/ho_idem") is None
    finally:
        bb.close()
        monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_V2", raising=False)
        monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_DB", raising=False)
        import src.host.server as server_module
        importlib.reload(server_module)


def test_migration_failure_does_not_crash_startup(tmp_path, monkeypatch, caplog):
    """A migration helper that raises must not bubble out of
    ``create_mesh_app``. The error is logged and the app still
    constructs.
    """
    bb_path = str(tmp_path / "bb.db")
    bb = Blackboard(db_path=bb_path)

    def _boom(*args, **kwargs):
        raise RuntimeError("synthetic migration failure")

    try:
        server_module = _reload_server(
            monkeypatch, v2_on=True, tasks_db=str(tmp_path / "tasks.db"),
        )
        # Patch the helper inside the migration module the server
        # imports lazily. The lazy import means we patch the source
        # module, not the server's local binding.
        monkeypatch.setattr(
            "src.host.orchestration_migration.migrate_blackboard_to_tasks",
            _boom,
        )

        with caplog.at_level(logging.ERROR, logger="host.server"):
            app = _build_app(server_module, bb)

        # App still constructed.
        assert app is not None
        # The tasks store is still attached — the failure was the
        # migration, not the store init.
        assert app.tasks_store is not None
        # Error was logged.
        err_lines = [
            r.message for r in caplog.records
            if "orchestration migration failed" in r.message
        ]
        assert err_lines, (
            f"expected an error log line, got {[r.message for r in caplog.records]}"
        )
    finally:
        bb.close()
        monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_V2", raising=False)
        monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_DB", raising=False)
        import src.host.server as server_module
        importlib.reload(server_module)


def test_migration_skipped_when_v2_disabled(tmp_path, monkeypatch):
    """When v2 is OFF, the auto-run is skipped — legacy keys remain in
    place so an operator who deliberately disabled v2 doesn't have
    their data silently moved.
    """
    bb_path = str(tmp_path / "bb.db")
    bb = Blackboard(db_path=bb_path)
    bb.write(
        "tasks/scout/ho_skip",
        {"from": "operator", "summary": "do not auto-migrate"},
        written_by="operator",
    )

    try:
        server_module = _reload_server(monkeypatch, v2_on=False, tasks_db="")
        # Set v2 explicitly off.
        monkeypatch.setenv("OPENLEGION_ORCHESTRATION_TASKS_V2", "0")
        importlib.reload(server_module)

        app = _build_app(server_module, bb)

        # No tasks store was constructed.
        assert app.tasks_store is None
        # Legacy key is still present.
        entry = bb.read("tasks/scout/ho_skip")
        assert entry is not None
    finally:
        bb.close()
        monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_V2", raising=False)
        monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_DB", raising=False)
        import src.host.server as server_module
        importlib.reload(server_module)


# Reference an unused symbol so ruff's F401 is happy after the import
# block was extended for the new tests.
_ = pytest

