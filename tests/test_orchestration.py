"""Storage-layer tests for the durable orchestration tasks store (Task 6).

Covers:

* create + get round-trip
* status transition rules (each valid + invalid pair)
* reroute + cancel + add_artifact
* list_inbox excludes terminal by default
* list_project filter
* list_events ordered chronologically
* reap_expired removes only past-retention rows
* schema migration idempotent (init twice)
* concurrent status updates serialize
* persistence across reopen
"""

from __future__ import annotations

import threading
import time

import pytest

from src.host.orchestration import (
    DEFAULT_RETENTION_SECONDS,
    TERMINAL_STATUSES,
    VALID_STATUSES,
    InvalidStatusTransition,
    TaskNotFound,
    Tasks,
)


def _make_store(tmp_path) -> Tasks:
    return Tasks(db_path=str(tmp_path / "tasks.db"))


# ── Basic CRUD ────────────────────────────────────────────────────


def test_create_and_get_round_trip(tmp_path):
    t = _make_store(tmp_path)
    rec = t.create(
        creator="scout",
        assignee="analyst",
        title="research handoff",
        description="please dig into X",
        project_id="research",
        priority=5,
        dependencies=["dep1", "dep2"],
    )
    assert rec["id"].startswith("task_")
    assert rec["creator"] == "scout"
    assert rec["assignee"] == "analyst"
    assert rec["status"] == "pending"
    assert rec["project_id"] == "research"
    assert rec["priority"] == 5
    assert rec["dependencies"] == ["dep1", "dep2"]
    assert rec["completed_at"] is None
    assert rec["retention_until"] is None

    fetched = t.get(rec["id"])
    assert fetched is not None
    assert fetched["title"] == "research handoff"
    assert fetched["description"] == "please dig into X"


def test_create_requires_title(tmp_path):
    t = _make_store(tmp_path)
    with pytest.raises(ValueError, match="title"):
        t.create(creator="scout", assignee="analyst", title="")


def test_create_with_origin_persists_kind_channel_user(tmp_path):
    t = _make_store(tmp_path)
    rec = t.create(
        creator="scout",
        assignee="analyst",
        title="t",
        origin={"kind": "human", "channel": "telegram", "user": "12345"},
    )
    assert rec["origin"] == {
        "kind": "human", "channel": "telegram", "user": "12345",
    }


def test_get_missing_returns_none(tmp_path):
    t = _make_store(tmp_path)
    assert t.get("does-not-exist") is None


# ── Status transitions ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "from_status,to_status,allowed",
    [
        ("pending", "accepted", True),
        ("pending", "working", True),
        ("pending", "cancelled", True),
        ("pending", "failed", True),
        ("pending", "blocked", False),
        ("pending", "done", False),
        ("accepted", "working", True),
        ("accepted", "cancelled", True),
        ("accepted", "failed", True),
        ("accepted", "pending", False),
        ("accepted", "done", False),
        ("working", "blocked", True),
        ("working", "done", True),
        ("working", "failed", True),
        ("working", "cancelled", True),
        ("working", "pending", False),
        ("blocked", "working", True),
        ("blocked", "cancelled", True),
        ("blocked", "failed", True),
        ("blocked", "done", True),
        ("blocked", "pending", False),
        ("done", "working", False),
        ("done", "pending", False),
        ("done", "cancelled", False),
        ("failed", "working", False),
        ("failed", "pending", False),
        ("cancelled", "working", False),
        ("cancelled", "pending", False),
    ],
)
def test_status_transitions(tmp_path, from_status, to_status, allowed):
    t = _make_store(tmp_path)
    rec = t.create(creator="c", assignee="a", title="t")
    # Walk to the from_status. ``pending`` is the starting state.
    walk_path = {
        "pending": [],
        "accepted": ["accepted"],
        "working": ["working"],
        "blocked": ["working", "blocked"],
        "done": ["working", "done"],
        "failed": ["failed"],
        "cancelled": ["cancelled"],
    }[from_status]
    for step in walk_path:
        t.update_status(rec["id"], step, actor="actor")
    if allowed:
        result = t.update_status(rec["id"], to_status, actor="actor")
        assert result["status"] == to_status
    else:
        with pytest.raises(InvalidStatusTransition):
            t.update_status(rec["id"], to_status, actor="actor")


def test_unknown_status_raises_value_error(tmp_path):
    t = _make_store(tmp_path)
    rec = t.create(creator="c", assignee="a", title="t")
    with pytest.raises(ValueError):
        t.update_status(rec["id"], "bogus", actor="x")


def test_update_status_unknown_id_raises_task_not_found(tmp_path):
    t = _make_store(tmp_path)
    with pytest.raises(TaskNotFound):
        t.update_status("not-a-task", "working", actor="x")


def test_terminal_transition_sets_completed_and_retention(tmp_path):
    t = _make_store(tmp_path)
    rec = t.create(creator="c", assignee="a", title="t")
    t.update_status(rec["id"], "working", actor="a")
    result = t.update_status(rec["id"], "done", actor="a")
    assert result["completed_at"] is not None
    assert result["retention_until"] is not None
    # Default retention window
    assert (
        result["retention_until"] - result["completed_at"]
        == pytest.approx(DEFAULT_RETENTION_SECONDS, abs=1)
    )


def test_blocker_note_attaches_only_on_blocked(tmp_path):
    t = _make_store(tmp_path)
    rec = t.create(creator="c", assignee="a", title="t")
    t.update_status(rec["id"], "working", actor="a")
    result = t.update_status(
        rec["id"], "blocked", actor="a", blocker_note="waiting on auth",
    )
    assert result["blocker_note"] == "waiting on auth"
    # Subsequent transitions clear the blocker note.
    result = t.update_status(rec["id"], "working", actor="a")
    assert result["blocker_note"] is None


def test_repeat_status_is_recorded_as_event_but_no_op(tmp_path):
    """Calling update_status with the current state is a documented no-op."""
    t = _make_store(tmp_path)
    rec = t.create(creator="c", assignee="a", title="t")
    result = t.update_status(rec["id"], "pending", actor="a")
    assert result["status"] == "pending"
    events = t.list_events(rec["id"])
    kinds = [e["event_kind"] for e in events]
    assert kinds == ["created", "status_unchanged"]


# ── Reroute / cancel / artifacts ───────────────────────────────────


def test_reroute_changes_assignee_and_emits_event(tmp_path):
    t = _make_store(tmp_path)
    rec = t.create(creator="c", assignee="a", title="t")
    result = t.reroute(
        rec["id"], "b", actor="operator", reason="overloaded",
    )
    assert result["assignee"] == "b"
    events = t.list_events(rec["id"])
    rerouted = [e for e in events if e["event_kind"] == "rerouted"]
    assert len(rerouted) == 1
    assert rerouted[0]["payload"]["from"] == "a"
    assert rerouted[0]["payload"]["to"] == "b"
    assert rerouted[0]["payload"]["reason"] == "overloaded"


def test_reroute_terminal_task_rejected(tmp_path):
    t = _make_store(tmp_path)
    rec = t.create(creator="c", assignee="a", title="t")
    t.update_status(rec["id"], "cancelled", actor="x")
    with pytest.raises(InvalidStatusTransition):
        t.reroute(rec["id"], "b", actor="op")


def test_cancel_sets_terminal_and_records_reason(tmp_path):
    t = _make_store(tmp_path)
    rec = t.create(creator="c", assignee="a", title="t")
    result = t.cancel(rec["id"], actor="operator", reason="user changed mind")
    assert result["status"] == "cancelled"
    assert result["completed_at"] is not None
    assert result["retention_until"] is not None
    events = t.list_events(rec["id"])
    cancel_event = [e for e in events if e["event_kind"] == "cancelled"]
    assert len(cancel_event) == 1
    assert cancel_event[0]["payload"]["reason"] == "user changed mind"


def test_add_artifact_appends_unique_refs(tmp_path):
    t = _make_store(tmp_path)
    rec = t.create(creator="c", assignee="a", title="t")
    t.add_artifact(rec["id"], "/data/output.json", actor="a")
    t.add_artifact(rec["id"], "/data/output.json", actor="a")  # dupe
    t.add_artifact(rec["id"], "/data/notes.md", actor="a")
    fetched = t.get(rec["id"])
    assert fetched["artifact_refs"] == ["/data/output.json", "/data/notes.md"]


# ── Listing ─────────────────────────────────────────────────────────


def test_list_inbox_excludes_terminal_by_default(tmp_path):
    t = _make_store(tmp_path)
    t.create(creator="c", assignee="a", title="pending")
    t.create(creator="c", assignee="a", title="working")  # will become working
    done = t.create(creator="c", assignee="a", title="done")
    t.update_status(done["id"], "working", actor="a")
    t.update_status(done["id"], "done", actor="a")

    rows = t.list_inbox("a")
    titles = {r["title"] for r in rows}
    assert titles == {"pending", "working"}
    assert "done" not in titles
    assert all(r["status"] not in TERMINAL_STATUSES for r in rows)


def test_list_inbox_include_terminal_returns_all(tmp_path):
    t = _make_store(tmp_path)
    rec = t.create(creator="c", assignee="a", title="t")
    t.cancel(rec["id"], actor="op")
    rows = t.list_inbox("a", include_terminal=True)
    assert len(rows) == 1


def test_list_inbox_filters_by_project(tmp_path):
    t = _make_store(tmp_path)
    t.create(creator="c", assignee="a", title="x", project_id="p1")
    t.create(creator="c", assignee="a", title="y", project_id="p2")
    rows = t.list_inbox("a", project_id="p1")
    assert len(rows) == 1
    assert rows[0]["project_id"] == "p1"


def test_list_project_filter(tmp_path):
    t = _make_store(tmp_path)
    t.create(creator="c", assignee="a", title="x", project_id="p1")
    t.create(creator="c", assignee="b", title="y", project_id="p1")
    t.create(creator="c", assignee="a", title="z", project_id="p2")
    rows = t.list_project("p1")
    assert len(rows) == 2
    assert all(r["project_id"] == "p1" for r in rows)


def test_list_project_status_filter(tmp_path):
    t = _make_store(tmp_path)
    a = t.create(creator="c", assignee="a", title="x", project_id="p1")
    t.create(creator="c", assignee="b", title="y", project_id="p1")
    t.update_status(a["id"], "working", actor="a")
    rows = t.list_project("p1", statuses=["working"])
    assert len(rows) == 1
    assert rows[0]["status"] == "working"


# ── Events ────────────────────────────────────────────────────────


def test_list_events_ordered_chronologically(tmp_path):
    t = _make_store(tmp_path)
    rec = t.create(creator="c", assignee="a", title="t")
    time.sleep(0.005)
    t.update_status(rec["id"], "working", actor="a")
    time.sleep(0.005)
    t.update_status(rec["id"], "blocked", actor="a", blocker_note="x")
    events = t.list_events(rec["id"])
    kinds = [e["event_kind"] for e in events]
    assert kinds == ["created", "status_changed", "status_changed"]
    timestamps = [e["created_at"] for e in events]
    assert timestamps == sorted(timestamps)


# ── Retention ─────────────────────────────────────────────────────


def test_reap_expired_drops_only_past_retention(tmp_path):
    t = Tasks(db_path=str(tmp_path / "tasks.db"), retention_seconds=0)
    keep = t.create(creator="c", assignee="a", title="alive")
    drop = t.create(creator="c", assignee="a", title="terminal")
    t.update_status(drop["id"], "cancelled", actor="x")
    time.sleep(0.01)
    deleted = t.reap_expired()
    assert deleted == 1
    assert t.get(keep["id"]) is not None
    assert t.get(drop["id"]) is None


def test_reap_expired_drops_orphaned_events(tmp_path):
    t = Tasks(db_path=str(tmp_path / "tasks.db"), retention_seconds=0)
    rec = t.create(creator="c", assignee="a", title="t")
    t.update_status(rec["id"], "cancelled", actor="x")
    time.sleep(0.01)
    deleted = t.reap_expired()
    assert deleted == 1
    # Event rows should be gone too.
    assert t.list_events(rec["id"]) == []


# ── Schema / persistence ──────────────────────────────────────────


def test_init_schema_idempotent(tmp_path):
    db = str(tmp_path / "tasks.db")
    t = Tasks(db_path=db)
    t.create(creator="c", assignee="a", title="x")
    t._init_schema()  # should not blow up
    t._init_schema()
    assert len(t.list_inbox("a")) == 1


def test_persistence_across_reopen(tmp_path):
    db = str(tmp_path / "tasks.db")
    t1 = Tasks(db_path=db)
    rec = t1.create(creator="c", assignee="a", title="durable")
    del t1
    t2 = Tasks(db_path=db)
    assert t2.get(rec["id"])["title"] == "durable"


# ── Concurrency ───────────────────────────────────────────────────


def test_concurrent_status_updates_serialize(tmp_path):
    """Two threads racing to transition the same task: both succeed,
    the second sees the post-first-update state and validates against
    that — so e.g. two ``working`` updates from ``pending`` will both
    succeed (transitions are valid). The point being exercised is that
    BEGIN IMMEDIATE serializes the writes — neither corrupts the row.
    """
    t = _make_store(tmp_path)
    rec = t.create(creator="c", assignee="a", title="t")
    barrier = threading.Barrier(2)
    results = []

    def worker():
        barrier.wait()
        try:
            results.append(t.update_status(rec["id"], "working", actor="a"))
        except Exception as e:
            results.append(e)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert all(isinstance(r, dict) for r in results)
    assert all(r["status"] == "working" for r in results)


def test_concurrent_invalid_transition_one_wins(tmp_path):
    """Two threads race to move pending → cancelled and pending → done.
    The cancel succeeds (valid). The done is invalid from pending so it
    must raise InvalidStatusTransition either way, AND if cancel ran
    first, the second thread sees ``cancelled`` (terminal) and also fails.
    Either order leaves exactly one terminal row.
    """
    t = _make_store(tmp_path)
    rec = t.create(creator="c", assignee="a", title="t")
    barrier = threading.Barrier(2)
    results = []

    def cancel_worker():
        barrier.wait()
        try:
            results.append(("cancel", t.update_status(rec["id"], "cancelled", actor="x")))
        except Exception as e:
            results.append(("cancel", e))

    def done_worker():
        barrier.wait()
        try:
            results.append(("done", t.update_status(rec["id"], "done", actor="x")))
        except Exception as e:
            results.append(("done", e))

    th1 = threading.Thread(target=cancel_worker)
    th2 = threading.Thread(target=done_worker)
    th1.start()
    th2.start()
    th1.join()
    th2.join()
    final = t.get(rec["id"])
    assert final["status"] == "cancelled"
    # Done attempt must have raised — pending → done is invalid, and so
    # is cancelled → done.
    done_results = [r for label, r in results if label == "done"]
    assert all(isinstance(r, Exception) for r in done_results)


# ── Sanity: VALID_STATUSES exposed for endpoint validation ────────


def test_valid_statuses_constant():
    assert "pending" in VALID_STATUSES
    assert "done" in VALID_STATUSES
    assert "bogus" not in VALID_STATUSES


# ── Task 9 — EventBus integration ─────────────────────────────────


def _attach_event_bus(store):
    """Attach a fresh ``EventBus`` and return ``(bus, captured)`` where
    ``captured`` is a list of (type, agent, data) tuples for assertion.
    """
    from src.dashboard.events import EventBus
    bus = EventBus()
    captured: list[tuple[str, str, dict]] = []

    class _Recorder:
        def emit(self, event_type, agent="", data=None):
            captured.append((event_type, agent, dict(data or {})))
            bus.emit(event_type, agent=agent, data=data or {})

    store.set_event_bus(_Recorder())
    return bus, captured


def test_event_bus_emits_task_created(tmp_path):
    """``Tasks.create`` emits ``task_created`` with the documented payload."""
    t = _make_store(tmp_path)
    _bus, captured = _attach_event_bus(t)
    rec = t.create(creator="scout", assignee="analyst", title="dig")
    types = [c[0] for c in captured]
    assert "task_created" in types
    evt = next(c for c in captured if c[0] == "task_created")
    assert evt[1] == "scout"  # agent = creator
    assert evt[2]["task_id"] == rec["id"]
    assert evt[2]["creator"] == "scout"
    assert evt[2]["assignee"] == "analyst"
    assert evt[2]["title"] == "dig"
    assert evt[2]["status"] == "pending"


def test_event_bus_emits_task_status_changed(tmp_path):
    """``Tasks.update_status`` emits ``task_status_changed``."""
    t = _make_store(tmp_path)
    _bus, captured = _attach_event_bus(t)
    rec = t.create(creator="scout", assignee="analyst", title="dig")
    captured.clear()
    t.update_status(rec["id"], "accepted", actor="analyst")
    types = [c[0] for c in captured]
    assert types == ["task_status_changed"]
    evt = captured[0]
    assert evt[2]["task_id"] == rec["id"]
    assert evt[2]["old_status"] == "pending"
    assert evt[2]["new_status"] == "accepted"
    assert evt[2]["actor"] == "analyst"


def test_event_bus_no_emit_on_no_op_status_transition(tmp_path):
    """Repeating a status doesn't emit (the no-op branch)."""
    t = _make_store(tmp_path)
    _bus, captured = _attach_event_bus(t)
    rec = t.create(creator="scout", assignee="analyst", title="dig")
    captured.clear()
    t.update_status(rec["id"], "pending", actor="scout")
    assert not [c for c in captured if c[0] == "task_status_changed"]


def test_event_bus_emits_on_reroute(tmp_path):
    t = _make_store(tmp_path)
    _bus, captured = _attach_event_bus(t)
    rec = t.create(creator="scout", assignee="analyst", title="dig")
    captured.clear()
    t.reroute(rec["id"], "writer", actor="operator")
    types = [c[0] for c in captured]
    assert "task_status_changed" in types
    evt = next(c for c in captured if c[0] == "task_status_changed")
    assert evt[2]["assignee"] == "writer"
    assert evt[2]["actor"] == "operator"


def test_event_bus_emits_on_cancel(tmp_path):
    """``cancel`` chains through ``update_status('cancelled')`` and emits."""
    t = _make_store(tmp_path)
    _bus, captured = _attach_event_bus(t)
    rec = t.create(creator="scout", assignee="analyst", title="dig")
    t.update_status(rec["id"], "working", actor="analyst")
    captured.clear()
    t.cancel(rec["id"], actor="operator", reason="not needed")
    types = [c[0] for c in captured]
    assert "task_status_changed" in types
    evt = next(c for c in captured if c[0] == "task_status_changed")
    assert evt[2]["new_status"] == "cancelled"


def test_event_bus_unset_disables_emit(tmp_path):
    """Detaching the bus stops emits (without breaking the txn)."""
    t = _make_store(tmp_path)
    _bus, captured = _attach_event_bus(t)
    t.set_event_bus(None)
    t.create(creator="scout", assignee="analyst", title="dig")
    assert captured == []


# ── PR 4 — outcome capture + rework spawning ──────────────────────


def test_outcome_columns_exist_after_init(tmp_path):
    """Migration adds outcome / feedback_text / previous_task_id columns."""
    t = _make_store(tmp_path)
    with t._conn() as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(tasks)").fetchall()}
    assert "outcome" in cols
    assert "feedback_text" in cols
    assert "previous_task_id" in cols


def test_init_schema_idempotent_with_outcome_migration(tmp_path):
    """Re-instantiating against the same DB does not re-add columns or error."""
    db_path = str(tmp_path / "tasks.db")
    t1 = Tasks(db_path=db_path)
    t1.close()
    t2 = Tasks(db_path=db_path)  # second init must succeed
    with t2._conn() as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(tasks)").fetchall()}
    assert "outcome" in cols and "feedback_text" in cols and "previous_task_id" in cols
    t2.close()


def test_row_to_dict_exposes_new_fields_default_null(tmp_path):
    """Newly created tasks have outcome / feedback_text / previous_task_id as None."""
    t = _make_store(tmp_path)
    rec = t.create(creator="scout", assignee="analyst", title="dig")
    assert rec["outcome"] is None
    assert rec["feedback_text"] is None
    assert rec["previous_task_id"] is None


def test_set_outcome_happy_path_done(tmp_path):
    t = _make_store(tmp_path)
    rec = t.create(creator="scout", assignee="analyst", title="dig")
    t.update_status(rec["id"], "working", actor="analyst")
    t.update_status(rec["id"], "done", actor="analyst")
    updated = t.set_outcome(
        rec["id"], "accepted", "great work", actor="operator",
    )
    assert updated["outcome"] == "accepted"
    assert updated["feedback_text"] == "great work"
    events = t.list_events(rec["id"])
    kinds = [e["event_kind"] for e in events]
    assert kinds.count("task_outcome") == 1
    payload = next(e["payload"] for e in events if e["event_kind"] == "task_outcome")
    assert payload["outcome"] == "accepted"
    assert payload["feedback"] == "great work"


def test_set_outcome_rework_with_feedback(tmp_path):
    t = _make_store(tmp_path)
    rec = t.create(creator="scout", assignee="analyst", title="dig")
    t.update_status(rec["id"], "working", actor="analyst")
    t.update_status(rec["id"], "done", actor="analyst")
    updated = t.set_outcome(rec["id"], "rework", "missed the deadline angle")
    assert updated["outcome"] == "rework"
    assert updated["feedback_text"] == "missed the deadline angle"


def test_set_outcome_rejected_persists(tmp_path):
    t = _make_store(tmp_path)
    rec = t.create(creator="scout", assignee="analyst", title="dig")
    t.update_status(rec["id"], "working", actor="analyst")
    t.update_status(rec["id"], "failed", actor="analyst")
    updated = t.set_outcome(rec["id"], "rejected", "irrelevant")
    assert updated["outcome"] == "rejected"


def test_set_outcome_rejects_non_terminal_status(tmp_path):
    t = _make_store(tmp_path)
    rec = t.create(creator="scout", assignee="analyst", title="dig")
    t.update_status(rec["id"], "working", actor="analyst")
    with pytest.raises(InvalidStatusTransition, match="non-terminal"):
        t.set_outcome(rec["id"], "accepted", "")


def test_set_outcome_allows_re_rating(tmp_path):
    """Outcomes are write-many — operators can re-rate after a misclick.

    Each submission appends a fresh ``task_outcome`` audit event so the
    full re-rating history stays queryable, but ``tasks.outcome`` and
    ``tasks.feedback_text`` reflect only the latest value.
    """
    t = _make_store(tmp_path)
    rec = t.create(creator="scout", assignee="analyst", title="dig")
    t.update_status(rec["id"], "working", actor="analyst")
    t.update_status(rec["id"], "done", actor="analyst")
    first = t.set_outcome(rec["id"], "rejected", "wrong angle")
    assert first["outcome"] == "rejected"
    second = t.set_outcome(rec["id"], "accepted", "actually fine")
    assert second["outcome"] == "accepted"
    assert second["feedback_text"] == "actually fine"
    # Both submissions produce a task_outcome audit row.
    events = t.list_events(rec["id"])
    outcome_events = [e for e in events if e["event_kind"] == "task_outcome"]
    assert len(outcome_events) == 2
    # The newer event records the prior outcome so audit/analytics can
    # detect re-ratings without scanning the whole history.
    assert outcome_events[0]["payload"]["outcome"] == "rejected"
    assert outcome_events[0]["payload"]["previous_outcome"] is None
    assert outcome_events[1]["payload"]["outcome"] == "accepted"
    assert outcome_events[1]["payload"]["previous_outcome"] == "rejected"


def test_set_outcome_rejects_unknown_outcome(tmp_path):
    t = _make_store(tmp_path)
    rec = t.create(creator="scout", assignee="analyst", title="dig")
    t.update_status(rec["id"], "working", actor="analyst")
    t.update_status(rec["id"], "done", actor="analyst")
    with pytest.raises(ValueError, match="unknown outcome"):
        t.set_outcome(rec["id"], "meh", "")


def test_set_outcome_rejects_oversized_feedback(tmp_path):
    from src.host.orchestration import MAX_FEEDBACK_CHARS
    t = _make_store(tmp_path)
    rec = t.create(creator="scout", assignee="analyst", title="dig")
    t.update_status(rec["id"], "working", actor="analyst")
    t.update_status(rec["id"], "done", actor="analyst")
    huge = "x" * (MAX_FEEDBACK_CHARS + 1)
    with pytest.raises(ValueError, match="exceeds"):
        t.set_outcome(rec["id"], "accepted", huge)


def test_set_outcome_unknown_task(tmp_path):
    t = _make_store(tmp_path)
    with pytest.raises(TaskNotFound):
        t.set_outcome("task_missing", "accepted", "")


def test_set_outcome_emits_event_bus(tmp_path):
    t = _make_store(tmp_path)
    _bus, captured = _attach_event_bus(t)
    rec = t.create(creator="scout", assignee="analyst", title="dig")
    t.update_status(rec["id"], "working", actor="analyst")
    t.update_status(rec["id"], "done", actor="analyst")
    captured.clear()
    t.set_outcome(rec["id"], "accepted", "good")
    types = [c[0] for c in captured]
    assert "task_outcome" in types
    payload = next(c[2] for c in captured if c[0] == "task_outcome")
    assert payload["task_id"] == rec["id"]
    assert payload["outcome"] == "accepted"
    assert payload["feedback"] == "good"


def test_create_rework_task_links_to_previous(tmp_path):
    t = _make_store(tmp_path)
    rec = t.create(
        creator="op", assignee="analyst", title="research X",
        project_id="research", priority=3,
        origin={"kind": "human", "channel": "telegram", "user": "u1"},
    )
    new = t.create_rework_task(rec["id"], "go deeper on the legal angle")
    assert new["previous_task_id"] == rec["id"]
    assert new["title"] == "Rework: research X"
    assert new["assignee"] == "analyst"
    assert new["project_id"] == "research"
    assert new["description"] == "go deeper on the legal angle"
    assert new["status"] == "pending"
    assert new["origin"] == {"kind": "human", "channel": "telegram", "user": "u1"}
    assert new["creator"] == "operator"  # the actor (default)


def test_create_rework_task_requires_feedback(tmp_path):
    t = _make_store(tmp_path)
    rec = t.create(creator="op", assignee="analyst", title="t")
    with pytest.raises(ValueError, match="required"):
        t.create_rework_task(rec["id"], "")


def test_create_rework_task_unknown_previous(tmp_path):
    t = _make_store(tmp_path)
    with pytest.raises(TaskNotFound):
        t.create_rework_task("task_missing", "do better")


def test_create_rework_task_emits_task_created(tmp_path):
    t = _make_store(tmp_path)
    _bus, captured = _attach_event_bus(t)
    rec = t.create(creator="op", assignee="analyst", title="t")
    captured.clear()
    new = t.create_rework_task(rec["id"], "redo")
    types = [c[0] for c in captured]
    assert "task_created" in types
    evt = next(c for c in captured if c[0] == "task_created")
    assert evt[2]["task_id"] == new["id"]
    assert evt[2]["previous_task_id"] == rec["id"]


def test_create_rework_task_records_creation_event(tmp_path):
    """Audit trail records ``created`` event with kind=rework + previous link."""
    t = _make_store(tmp_path)
    rec = t.create(creator="op", assignee="analyst", title="t")
    new = t.create_rework_task(rec["id"], "redo")
    events = t.list_events(new["id"])
    assert any(
        e["event_kind"] == "created"
        and e["payload"].get("previous_task_id") == rec["id"]
        and e["payload"].get("kind") == "rework"
        for e in events
    )


# ── PR-J' per-agent aggregates for the operator heartbeat ──────────


def _seed_done_with_outcome(store, *, assignee, outcome):
    """Helper — create a task, complete it, set the outcome rating."""
    rec = store.create(creator="op", assignee=assignee, title="t")
    store.update_status(rec["id"], "working", actor=assignee)
    store.update_status(rec["id"], "done", actor=assignee)
    store.set_outcome(rec["id"], outcome, actor="op")
    return rec


def test_count_outcomes_since_groups_by_assignee(tmp_path):
    t = _make_store(tmp_path)
    _seed_done_with_outcome(t, assignee="alpha", outcome="rejected")
    _seed_done_with_outcome(t, assignee="alpha", outcome="rejected")
    _seed_done_with_outcome(t, assignee="beta", outcome="rejected")
    _seed_done_with_outcome(t, assignee="beta", outcome="accepted")  # filtered
    counts = t.count_outcomes_since("rejected", since_seconds=24 * 3600)
    assert counts == {"alpha": 2, "beta": 1}


def test_count_outcomes_since_excludes_old(tmp_path):
    t = _make_store(tmp_path)
    rec = _seed_done_with_outcome(t, assignee="alpha", outcome="rejected")
    # Manually rewind completed_at to 2 days ago.
    with t._conn() as conn:
        conn.execute(
            "UPDATE tasks SET completed_at=? WHERE id=?",
            (time.time() - 2 * 24 * 3600, rec["id"]),
        )
    counts = t.count_outcomes_since("rejected", since_seconds=24 * 3600)
    assert counts == {}


def test_count_failed_status_since_excludes_done(tmp_path):
    t = _make_store(tmp_path)
    failed = t.create(creator="op", assignee="alpha", title="x")
    t.update_status(failed["id"], "failed", actor="alpha")
    done = t.create(creator="op", assignee="alpha", title="y")
    t.update_status(done["id"], "working", actor="alpha")
    t.update_status(done["id"], "done", actor="alpha")
    counts = t.count_failed_status_since(since_seconds=24 * 3600)
    assert counts == {"alpha": 1}


def test_count_failed_status_since_excludes_old(tmp_path):
    t = _make_store(tmp_path)
    rec = t.create(creator="op", assignee="alpha", title="x")
    t.update_status(rec["id"], "failed", actor="alpha")
    with t._conn() as conn:
        conn.execute(
            "UPDATE tasks SET completed_at=? WHERE id=?",
            (time.time() - 2 * 24 * 3600, rec["id"]),
        )
    counts = t.count_failed_status_since(since_seconds=24 * 3600)
    assert counts == {}


def test_count_stale_since_excludes_terminal(tmp_path):
    t = _make_store(tmp_path)
    # alpha: pending task created 2 days ago — stale.
    stale = t.create(creator="op", assignee="alpha", title="stale")
    with t._conn() as conn:
        conn.execute(
            "UPDATE tasks SET created_at=? WHERE id=?",
            (time.time() - 2 * 24 * 3600, stale["id"]),
        )
    # alpha: done task created 2 days ago — NOT stale (terminal).
    done = t.create(creator="op", assignee="alpha", title="done")
    with t._conn() as conn:
        conn.execute(
            "UPDATE tasks SET created_at=? WHERE id=?",
            (time.time() - 2 * 24 * 3600, done["id"]),
        )
    t.update_status(done["id"], "working", actor="alpha")
    t.update_status(done["id"], "done", actor="alpha")
    # alpha: fresh pending task — NOT stale (created within window).
    t.create(creator="op", assignee="alpha", title="fresh")
    counts = t.count_stale_since(threshold_seconds=24 * 3600)
    assert counts == {"alpha": 1}


def test_count_stale_since_groups_by_assignee(tmp_path):
    t = _make_store(tmp_path)
    for assignee, n in [("alpha", 2), ("beta", 1)]:
        for _ in range(n):
            rec = t.create(creator="op", assignee=assignee, title="x")
            with t._conn() as conn:
                conn.execute(
                    "UPDATE tasks SET created_at=? WHERE id=?",
                    (time.time() - 2 * 24 * 3600, rec["id"]),
                )
    counts = t.count_stale_since(threshold_seconds=24 * 3600)
    assert counts == {"alpha": 2, "beta": 1}


def test_list_stale_for_assignee_returns_oldest_first_capped(tmp_path):
    t = _make_store(tmp_path)
    ids = []
    base = time.time() - 2 * 24 * 3600
    for i in range(7):
        rec = t.create(creator="op", assignee="alpha", title=f"t{i}")
        with t._conn() as conn:
            conn.execute(
                "UPDATE tasks SET created_at=? WHERE id=?",
                (base + i, rec["id"]),
            )
        ids.append(rec["id"])
    rows = t.list_stale_for_assignee(
        "alpha", threshold_seconds=24 * 3600, limit=5,
    )
    # Oldest 5 — created in insertion order, so first 5 ids.
    assert rows == ids[:5]


def test_list_stale_for_assignee_excludes_terminal(tmp_path):
    t = _make_store(tmp_path)
    rec = t.create(creator="op", assignee="alpha", title="cancelled")
    with t._conn() as conn:
        conn.execute(
            "UPDATE tasks SET created_at=? WHERE id=?",
            (time.time() - 2 * 24 * 3600, rec["id"]),
        )
    t.cancel(rec["id"], actor="op")
    rows = t.list_stale_for_assignee(
        "alpha", threshold_seconds=24 * 3600, limit=5,
    )
    assert rows == []


def test_list_stale_for_assignee_excludes_fresh(tmp_path):
    t = _make_store(tmp_path)
    t.create(creator="op", assignee="alpha", title="fresh")
    rows = t.list_stale_for_assignee(
        "alpha", threshold_seconds=24 * 3600, limit=5,
    )
    assert rows == []


# ── PR-L' — recent activity surface ───────────────────────────────


def test_last_event_ts_for_agent_returns_none_when_no_events(tmp_path):
    t = _make_store(tmp_path)
    assert t.last_event_ts_for_agent("nobody") is None


def test_last_event_ts_for_agent_picks_up_creator_role(tmp_path):
    t = _make_store(tmp_path)
    rec = t.create(creator="alpha", assignee="beta", title="hi")
    ts = t.last_event_ts_for_agent("alpha")
    assert ts is not None
    # Within a couple of seconds of when the row was created.
    assert abs(ts - rec["created_at"]) < 5


def test_last_event_ts_for_agent_picks_up_assignee_role(tmp_path):
    t = _make_store(tmp_path)
    t.create(creator="alpha", assignee="beta", title="hi")
    ts = t.last_event_ts_for_agent("beta")
    assert ts is not None


def test_last_event_ts_for_agent_picks_up_actor_role(tmp_path):
    """An actor who is neither creator nor assignee still counts."""
    t = _make_store(tmp_path)
    rec = t.create(creator="alpha", assignee="beta", title="hi")
    # Operator-grade emits an event with actor="operator".
    t.update_status(rec["id"], "working", actor="operator")
    ts = t.last_event_ts_for_agent("operator")
    assert ts is not None


def test_last_event_ts_for_agent_returns_most_recent(tmp_path):
    t = _make_store(tmp_path)
    rec_a = t.create(creator="alpha", assignee="beta", title="early")
    time.sleep(0.05)
    rec_b = t.create(creator="alpha", assignee="beta", title="later")
    ts = t.last_event_ts_for_agent("alpha")
    assert ts is not None
    # The later row's event should win.
    assert ts >= rec_b["created_at"]
    assert ts >= rec_a["created_at"]
