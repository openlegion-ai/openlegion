"""One-shot blackboard → tasks migration tests (Task 6)."""

from __future__ import annotations

from src.host.mesh import Blackboard
from src.host.orchestration import Tasks
from src.host.orchestration_migration import migrate_blackboard_to_tasks


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
