"""Tests for the SQLite-backed ChangeHistory store (PR 1).

Covers:

* basic record / peek / consume_for_undo semantics
* atomic single-shot consume (double-undo prevention)
* expiry behaviour (consumed but not deleted; peek returns None)
* schema is idempotent across reopen
* persistence across reopen
* event_bus emit happens via the standard hook (not our concern here —
  ``record`` does NOT emit; the server endpoint emits the
  ``operator_action_receipt`` event with a richer payload)
* list_recent shape
"""

from __future__ import annotations

import time
import uuid

from src.host.change_history import ChangeHistory


def _make_store(tmp_path) -> ChangeHistory:
    return ChangeHistory(db_path=str(tmp_path / "change_history.db"))


# ── Basic API ─────────────────────────────────────────────────────


def test_record_and_peek_returns_row(tmp_path):
    ch = _make_store(tmp_path)
    token = "tok-1"
    rec = ch.record(
        undo_token=token,
        actor="operator",
        agent_id="writer",
        field="instructions",
        old_value="be brief",
        new_value="be punchy",
        summary="Updated writer's instructions",
        reason="user_asked",
    )
    assert rec["undo_token"] == token
    assert rec["consumed"] is False

    peek = ch.peek(token)
    assert peek is not None
    assert peek["agent_id"] == "writer"
    assert peek["field"] == "instructions"
    assert peek["old_value"] == "be brief"
    assert peek["new_value"] == "be punchy"
    assert peek["consumed"] is False


def test_peek_unknown_returns_none(tmp_path):
    ch = _make_store(tmp_path)
    assert ch.peek("does-not-exist") is None


def test_record_handles_dict_values(tmp_path):
    """Permissions/budget dicts must round-trip via JSON cleanly."""
    ch = _make_store(tmp_path)
    rec = ch.record(
        undo_token="t",
        actor="operator",
        agent_id="writer",
        field="permissions",
        old_value={"can_use_browser": False},
        new_value={"can_use_browser": True},
    )
    peek = ch.peek("t")
    assert peek["old_value"] == {"can_use_browser": False}
    assert peek["new_value"] == {"can_use_browser": True}
    assert rec["new_value"] == {"can_use_browser": True}


# ── Consume / undo ─────────────────────────────────────────────


def test_consume_for_undo_returns_row_then_blocks_double(tmp_path):
    ch = _make_store(tmp_path)
    ch.record(
        undo_token="t1", actor="operator", agent_id="writer",
        field="instructions", old_value="A", new_value="B",
    )
    first = ch.consume_for_undo("t1")
    second = ch.consume_for_undo("t1")
    assert first is not None
    assert first["consumed"] is True
    assert second is None  # double-undo blocked


def test_consume_unknown_returns_none(tmp_path):
    ch = _make_store(tmp_path)
    assert ch.consume_for_undo("nope") is None


def test_peek_after_consume_returns_none(tmp_path):
    ch = _make_store(tmp_path)
    ch.record(
        undo_token="t1", actor="operator", agent_id="writer",
        field="instructions", old_value="A", new_value="B",
    )
    ch.consume_for_undo("t1")
    # Even though the row is still in the table for audit, peek refuses
    # to surface consumed rows for undo eligibility.
    assert ch.peek("t1") is None


# ── Expiry ─────────────────────────────────────────────────────


def test_expired_peek_returns_none(tmp_path):
    ch = _make_store(tmp_path)
    ch.record(
        undo_token="t", actor="operator", agent_id="writer",
        field="instructions", old_value="A", new_value="B",
        ttl=0,
    )
    time.sleep(0.01)
    assert ch.peek("t") is None


def test_expired_consume_returns_none(tmp_path):
    ch = _make_store(tmp_path)
    ch.record(
        undo_token="t", actor="operator", agent_id="writer",
        field="instructions", old_value="A", new_value="B",
        ttl=0,
    )
    time.sleep(0.01)
    assert ch.consume_for_undo("t") is None


def test_reap_expired_marks_consumed(tmp_path):
    ch = _make_store(tmp_path)
    ch.record(
        undo_token="t1", actor="operator", agent_id="w",
        field="instructions", old_value="A", new_value="B",
        ttl=0,
    )
    ch.record(
        undo_token="t2", actor="operator", agent_id="w",
        field="instructions", old_value="C", new_value="D",
        ttl=300,
    )
    time.sleep(0.01)
    n = ch.reap_expired()
    assert n == 1
    # After reap, the expired one is no longer eligible for undo.
    assert ch.consume_for_undo("t1") is None
    # The fresh one still works.
    assert ch.consume_for_undo("t2") is not None


# ── Persistence ────────────────────────────────────────────────


def test_persistence_across_reopen(tmp_path):
    db = str(tmp_path / "ch.db")
    ch1 = ChangeHistory(db_path=db)
    ch1.record(
        undo_token="persist", actor="operator", agent_id="w",
        field="role", old_value="x", new_value="y",
    )
    # Reopen — same db file.
    ch2 = ChangeHistory(db_path=db)
    rec = ch2.peek("persist")
    assert rec is not None
    assert rec["field"] == "role"


def test_init_schema_is_idempotent(tmp_path):
    """Constructing twice on the same file must not raise."""
    db = str(tmp_path / "ch.db")
    ChangeHistory(db_path=db)
    ChangeHistory(db_path=db)  # no error


# ── list_recent ───────────────────────────────────────────────


def test_list_recent_filters_by_agent(tmp_path):
    ch = _make_store(tmp_path)
    for aid in ("writer", "writer", "researcher"):
        ch.record(
            undo_token=str(uuid.uuid4()), actor="operator",
            agent_id=aid, field="instructions",
            old_value="x", new_value="y",
        )
    rows = ch.list_recent(agent_id="writer")
    assert len(rows) == 2
    assert all(r["agent_id"] == "writer" for r in rows)
    rows_all = ch.list_recent()
    assert len(rows_all) == 3


def test_list_recent_includes_consumed(tmp_path):
    """Audit feed must include reverted rows so the activity feed can render them."""
    ch = _make_store(tmp_path)
    ch.record(
        undo_token="a", actor="operator", agent_id="w",
        field="instructions", old_value="x", new_value="y",
    )
    ch.consume_for_undo("a")
    rows = ch.list_recent()
    assert len(rows) == 1
    assert rows[0]["consumed"] is True


# ── list_unconsumed_for_field (supersede detection) ────────────


def test_list_unconsumed_for_field_returns_only_matching(tmp_path):
    """Used by the soft-edit endpoint to find receipts the new edit makes stale."""
    ch = _make_store(tmp_path)
    ch.record(
        undo_token="a1", actor="operator", agent_id="writer",
        field="instructions", old_value="x", new_value="y",
    )
    ch.record(
        undo_token="a2", actor="operator", agent_id="writer",
        field="soul", old_value="x", new_value="y",
    )
    ch.record(
        undo_token="a3", actor="operator", agent_id="researcher",
        field="instructions", old_value="x", new_value="y",
    )
    rows = ch.list_unconsumed_for_field("writer", "instructions")
    assert [r["undo_token"] for r in rows] == ["a1"]


def test_list_unconsumed_for_field_excludes_consumed_and_expired(tmp_path):
    ch = _make_store(tmp_path)
    ch.record(
        undo_token="live", actor="operator", agent_id="w",
        field="instructions", old_value="x", new_value="y",
    )
    ch.record(
        undo_token="dead", actor="operator", agent_id="w",
        field="instructions", old_value="x", new_value="y",
    )
    ch.consume_for_undo("dead")
    ch.record(
        undo_token="expired", actor="operator", agent_id="w",
        field="instructions", old_value="x", new_value="y",
        ttl=0,
    )
    time.sleep(0.01)
    rows = ch.list_unconsumed_for_field("w", "instructions")
    assert [r["undo_token"] for r in rows] == ["live"]
