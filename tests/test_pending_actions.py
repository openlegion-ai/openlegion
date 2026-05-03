"""Tests for the SQLite-backed PendingActions store.

These tests cover:

* basic store / peek / consume semantics
* atomic single-shot consume (replay protection)
* expiry cleanup
* origin / actor / payload-digest gates on consume
* list_pending excludes expired
* schema migration is idempotent
* persistence across reopen
* concurrent consumes serialize via ``BEGIN IMMEDIATE``
"""

from __future__ import annotations

import threading
import time

from src.host.pending_actions import PendingActions, _payload_digest


def _make_store(tmp_path) -> PendingActions:
    return PendingActions(db_path=str(tmp_path / "pending.db"))


# ── Basic API ─────────────────────────────────────────────────────


def test_store_and_peek_returns_record_without_consuming(tmp_path):
    pa = _make_store(tmp_path)
    rec = pa.store(
        nonce="n1",
        actor="operator",
        target_kind="agent",
        target_id="alpha",
        action_kind="model",
        payload={"old_value": "gpt-4o", "new_value": "claude"},
        origin_kind="human",
    )
    assert rec["nonce"] == "n1"
    assert rec["payload_digest"]
    assert rec["origin_kind"] == "human"

    # Peek twice -- both should return the record (no consumption).
    p1 = pa.peek("n1")
    p2 = pa.peek("n1")
    assert p1 is not None and p2 is not None
    assert p1["target_id"] == "alpha"
    assert p1["action_kind"] == "model"
    assert p1["payload"] == {"old_value": "gpt-4o", "new_value": "claude"}


def test_store_then_consume_returns_record_then_none(tmp_path):
    pa = _make_store(tmp_path)
    pa.store(
        nonce="n1", actor="operator", target_kind="agent",
        target_id="alpha", action_kind="model", payload={"x": 1},
    )
    first = pa.consume("n1")
    second = pa.consume("n1")
    assert first is not None
    assert first["target_id"] == "alpha"
    assert second is None  # already consumed


def test_consume_unknown_nonce_returns_none(tmp_path):
    pa = _make_store(tmp_path)
    assert pa.consume("does-not-exist") is None


def test_peek_unknown_nonce_returns_none(tmp_path):
    pa = _make_store(tmp_path)
    assert pa.peek("does-not-exist") is None


# ── Expiry ────────────────────────────────────────────────────────


def test_expired_peek_returns_none(tmp_path):
    pa = _make_store(tmp_path)
    pa.store(
        nonce="n1", actor="operator", target_kind="agent",
        target_id="alpha", action_kind="model", payload={"x": 1},
        ttl=0,  # already expired
    )
    # Force the boundary by sleeping a hair to ensure now > expires_at.
    time.sleep(0.01)
    assert pa.peek("n1") is None


def test_expired_consume_returns_none_and_deletes_row(tmp_path):
    pa = _make_store(tmp_path)
    pa.store(
        nonce="n1", actor="operator", target_kind="agent",
        target_id="alpha", action_kind="model", payload={"x": 1},
        ttl=0,
    )
    time.sleep(0.01)
    assert pa.consume("n1") is None
    # Row should be gone -- a fresh store with the same nonce works.
    pa.store(
        nonce="n1", actor="operator", target_kind="agent",
        target_id="beta", action_kind="model", payload={"y": 2},
    )
    assert pa.peek("n1")["target_id"] == "beta"


def test_reap_expired_only_drops_expired(tmp_path):
    pa = _make_store(tmp_path)
    pa.store(
        nonce="alive", actor="operator", target_kind="agent",
        target_id="alpha", action_kind="model", payload={"x": 1},
        ttl=300,
    )
    pa.store(
        nonce="dead", actor="operator", target_kind="agent",
        target_id="beta", action_kind="model", payload={"y": 2},
        ttl=0,
    )
    time.sleep(0.01)
    deleted = pa.reap_expired()
    assert deleted == 1
    assert pa.peek("alive") is not None
    assert pa.peek("dead") is None


def test_list_pending_excludes_expired(tmp_path):
    pa = _make_store(tmp_path)
    pa.store(
        nonce="a", actor="operator", target_kind="agent",
        target_id="alpha", action_kind="model", payload={}, ttl=300,
    )
    pa.store(
        nonce="b", actor="operator", target_kind="agent",
        target_id="beta", action_kind="model", payload={}, ttl=0,
    )
    time.sleep(0.01)
    rows = pa.list_pending()
    assert [r["nonce"] for r in rows] == ["a"]


# ── Confirm-side gates ────────────────────────────────────────────


def test_consume_wrong_confirmer_preserves_row(tmp_path):
    pa = _make_store(tmp_path)
    pa.store(
        nonce="n1", actor="operator", target_kind="agent",
        target_id="alpha", action_kind="model", payload={"x": 1},
    )
    # Wrong confirmer -- returns None and does NOT delete.
    assert pa.consume("n1", confirmer="someone-else") is None
    assert pa.peek("n1") is not None
    # Right confirmer -- succeeds.
    assert pa.consume("n1", confirmer="operator") is not None


def test_consume_wrong_payload_digest_preserves_row(tmp_path):
    pa = _make_store(tmp_path)
    rec = pa.store(
        nonce="n1", actor="operator", target_kind="agent",
        target_id="alpha", action_kind="model", payload={"x": 1},
    )
    bogus_digest = "0" * 64
    assert pa.consume("n1", expected_payload_digest=bogus_digest) is None
    assert pa.peek("n1") is not None
    # Real digest -- succeeds.
    assert pa.consume("n1", expected_payload_digest=rec["payload_digest"]) is not None


def test_consume_origin_kind_mismatch_preserves_row(tmp_path):
    pa = _make_store(tmp_path)
    pa.store(
        nonce="n1", actor="operator", target_kind="agent",
        target_id="alpha", action_kind="model", payload={"x": 1},
        origin_kind="agent",
    )
    # Require human, row says agent -- refuse.
    assert pa.consume("n1", require_origin_kind="human") is None
    # Row preserved.
    assert pa.peek("n1") is not None


def test_consume_origin_kind_match_succeeds(tmp_path):
    pa = _make_store(tmp_path)
    pa.store(
        nonce="n1", actor="operator", target_kind="agent",
        target_id="alpha", action_kind="model", payload={"x": 1},
        origin_kind="human",
    )
    assert pa.consume("n1", require_origin_kind="human") is not None


def test_consume_origin_kind_required_but_missing(tmp_path):
    """A row without origin_kind cannot satisfy require_origin_kind="human"."""
    pa = _make_store(tmp_path)
    pa.store(
        nonce="n1", actor="operator", target_kind="agent",
        target_id="alpha", action_kind="model", payload={"x": 1},
        origin_kind=None,
    )
    assert pa.consume("n1", require_origin_kind="human") is None
    assert pa.peek("n1") is not None  # still there


def test_payload_digest_is_stable_across_dict_orderings():
    """Equivalent payloads with reordered keys must hash identically."""
    a = _payload_digest({"a": 1, "b": 2})
    b = _payload_digest({"b": 2, "a": 1})
    c = _payload_digest({"a": 1, "b": 3})
    assert a == b
    assert a != c


# ── Schema / migration ────────────────────────────────────────────


def test_init_schema_is_idempotent(tmp_path):
    pa = _make_store(tmp_path)
    # Calling twice must not blow up.
    pa._init_schema()
    pa._init_schema()
    # Sanity: store still works after re-init.
    pa.store(
        nonce="n1", actor="operator", target_kind="agent",
        target_id="alpha", action_kind="model", payload={"x": 1},
    )
    assert pa.peek("n1") is not None


def test_persistence_across_reopen(tmp_path):
    """Records survive reopening the database (mesh-restart scenario)."""
    db_path = str(tmp_path / "pending.db")
    pa1 = PendingActions(db_path=db_path)
    pa1.store(
        nonce="n1", actor="operator", target_kind="agent",
        target_id="alpha", action_kind="model",
        payload={"old_value": "gpt-4o", "new_value": "claude"},
        origin_kind="human", ttl=300,
    )
    # Drop the first instance entirely.
    del pa1
    # Reopen.
    pa2 = PendingActions(db_path=db_path)
    rec = pa2.peek("n1")
    assert rec is not None
    assert rec["target_id"] == "alpha"
    assert rec["origin_kind"] == "human"
    assert rec["payload"] == {"old_value": "gpt-4o", "new_value": "claude"}


# ── Concurrency ──────────────────────────────────────────────────


def test_concurrent_consume_serializes(tmp_path):
    """BEGIN IMMEDIATE in consume serializes two threads on the same nonce.

    Two threads race to consume the same nonce. Exactly one wins; the
    other returns None. (The losing thread's BEGIN IMMEDIATE waits on
    SQLite's busy_timeout for the winner to commit; once that happens,
    the loser sees an empty row and returns None.)
    """
    pa = _make_store(tmp_path)
    pa.store(
        nonce="n1", actor="operator", target_kind="agent",
        target_id="alpha", action_kind="model", payload={"x": 1},
    )
    results: list = []
    barrier = threading.Barrier(2)

    def worker():
        barrier.wait()
        results.append(pa.consume("n1"))

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    successes = [r for r in results if r is not None]
    assert len(successes) == 1, f"expected exactly one success, got {results}"


# ── Replace-on-duplicate behavior ─────────────────────────────────


def test_duplicate_nonce_replaces_row(tmp_path):
    """INSERT OR REPLACE: storing the same nonce twice replaces the row.

    Rationale (documented in PendingActions docstring): the propose
    endpoint generates fresh UUIDs, so a collision in production
    indicates a deliberate re-propose. Replacing the prior payload is
    the correct semantic.
    """
    pa = _make_store(tmp_path)
    pa.store(
        nonce="n1", actor="operator", target_kind="agent",
        target_id="alpha", action_kind="model",
        payload={"old_value": "x", "new_value": "y"},
    )
    pa.store(
        nonce="n1", actor="operator", target_kind="agent",
        target_id="alpha", action_kind="model",
        payload={"old_value": "x", "new_value": "z"},
    )
    rec = pa.peek("n1")
    # New payload won.
    assert rec["payload"]["new_value"] == "z"
    # And only one row exists.
    assert len(pa.list_pending()) == 1


# ── Reap on consume / store ──────────────────────────────────────


def test_opportunistic_reap_runs_on_store(tmp_path):
    pa = _make_store(tmp_path)
    pa.store(
        nonce="dead", actor="operator", target_kind="agent",
        target_id="alpha", action_kind="model", payload={}, ttl=0,
    )
    time.sleep(0.01)
    # Storing a fresh row triggers opportunistic reap.
    pa.store(
        nonce="alive", actor="operator", target_kind="agent",
        target_id="beta", action_kind="model", payload={}, ttl=300,
    )
    assert pa.peek("dead") is None
    assert pa.peek("alive") is not None


def test_safe_reap_swallows_errors(tmp_path, monkeypatch):
    """_safe_reap must never raise even when reap_expired raises."""
    pa = _make_store(tmp_path)

    def boom():
        raise RuntimeError("simulated database failure")

    monkeypatch.setattr(pa, "reap_expired", boom)
    # Must not raise.
    pa._safe_reap()


# ── Optional edge: list_pending returns ordered ──────────────────


def test_list_pending_ordered_by_created_at(tmp_path):
    pa = _make_store(tmp_path)
    pa.store(
        nonce="first", actor="operator", target_kind="agent",
        target_id="alpha", action_kind="model", payload={}, ttl=300,
    )
    time.sleep(0.005)
    pa.store(
        nonce="second", actor="operator", target_kind="agent",
        target_id="beta", action_kind="model", payload={}, ttl=300,
    )
    rows = pa.list_pending()
    assert [r["nonce"] for r in rows] == ["first", "second"]


# ── Task 9 — EventBus integration ─────────────────────────────────


def _attach_recording_bus(pa):
    """Attach a fresh recorder bus to ``pa`` and return the captured list."""
    captured: list[tuple[str, str, dict]] = []

    class _Recorder:
        def emit(self, event_type, agent="", data=None):
            captured.append((event_type, agent, dict(data or {})))

    pa.set_event_bus(_Recorder())
    return captured


def test_event_bus_emits_pending_action_created_on_store(tmp_path):
    pa = _make_store(tmp_path)
    captured = _attach_recording_bus(pa)
    pa.store(
        nonce="n1", actor="operator", target_kind="agent",
        target_id="alpha", action_kind="model", payload={"x": 1},
        origin_kind="human",
    )
    types = [c[0] for c in captured]
    assert "pending_action_created" in types
    evt = next(c for c in captured if c[0] == "pending_action_created")
    assert evt[1] == "operator"  # agent = actor
    assert evt[2]["nonce"] == "n1"
    assert evt[2]["target_kind"] == "agent"
    assert evt[2]["target_id"] == "alpha"
    assert evt[2]["action_kind"] == "model"
    # expires_at is included so the dashboard can render the countdown
    assert "expires_at" in evt[2]


def test_event_bus_emits_pending_action_resolved_on_consume_success(tmp_path):
    pa = _make_store(tmp_path)
    captured = _attach_recording_bus(pa)
    pa.store(
        nonce="n1", actor="operator", target_kind="agent",
        target_id="alpha", action_kind="model", payload={"x": 1},
        origin_kind="human",
    )
    captured.clear()
    rec = pa.consume("n1", confirmer="operator", require_origin_kind="human")
    assert rec is not None
    types = [c[0] for c in captured]
    assert "pending_action_resolved" in types
    evt = next(c for c in captured if c[0] == "pending_action_resolved")
    assert evt[2]["nonce"] == "n1"
    assert evt[2]["status"] == "confirmed"
    assert evt[2]["resolver"] == "operator"


def test_event_bus_does_not_emit_resolved_on_failed_consume(tmp_path):
    """Wrong digest / wrong actor / origin mismatch must NOT emit resolved."""
    pa = _make_store(tmp_path)
    captured = _attach_recording_bus(pa)
    pa.store(
        nonce="n1", actor="operator", target_kind="agent",
        target_id="alpha", action_kind="model", payload={"x": 1},
        origin_kind="human",
    )
    captured.clear()
    pa.consume("n1", expected_payload_digest="wrong")
    assert not [c for c in captured if c[0] == "pending_action_resolved"]


def test_event_bus_emits_pending_action_expired_on_reap(tmp_path):
    pa = _make_store(tmp_path)
    # Stage rows BEFORE attaching the recorder so the opportunistic
    # reap that fires on the second store() doesn't drop n1 inside the
    # bus's history (where it would be a confounder for the assertion
    # on counts post-clear).
    pa.store(
        nonce="n1", actor="operator", target_kind="agent",
        target_id="alpha", action_kind="model", payload={}, ttl=0,
    )
    captured = _attach_recording_bus(pa)
    time.sleep(0.01)
    n = pa.reap_expired()
    assert n == 1
    expired_events = [c for c in captured if c[0] == "pending_action_expired"]
    assert len(expired_events) == 1
    evt = expired_events[0]
    assert evt[2]["nonce"] == "n1"
    assert evt[2]["target_kind"] == "agent"
    assert evt[2]["target_id"] == "alpha"
    assert evt[2]["action_kind"] == "model"
    assert "expired_at" in evt[2]


def test_event_bus_emits_per_row_on_multi_reap(tmp_path):
    """When multiple rows expire in one reap pass, one event per row fires."""
    pa = _make_store(tmp_path)
    pa.store(
        nonce="n1", actor="operator", target_kind="agent",
        target_id="alpha", action_kind="model", payload={}, ttl=60,
    )
    pa.store(
        nonce="n2", actor="operator", target_kind="agent",
        target_id="beta", action_kind="model", payload={}, ttl=60,
    )
    captured = _attach_recording_bus(pa)
    # Force expiry by hand so the opportunistic ``_safe_reap`` doesn't
    # drop one row early.
    with pa._conn() as conn:
        conn.execute("UPDATE pending_actions SET expires_at = 0 WHERE nonce IN ('n1', 'n2')")
    n = pa.reap_expired()
    assert n == 2
    expired_events = [c for c in captured if c[0] == "pending_action_expired"]
    assert len(expired_events) == 2
    nonces = {e[2]["nonce"] for e in expired_events}
    assert nonces == {"n1", "n2"}


def test_event_bus_emits_pending_action_resolved_on_cancel(tmp_path):
    """``cancel(nonce)`` deletes the row and emits resolved with status='cancelled'."""
    pa = _make_store(tmp_path)
    captured = _attach_recording_bus(pa)
    pa.store(
        nonce="n1", actor="operator", target_kind="agent",
        target_id="alpha", action_kind="model", payload={"x": 1},
    )
    captured.clear()
    rec = pa.cancel("n1", actor="operator")
    assert rec is not None
    assert rec["status"] == "cancelled"
    # Row is gone — second cancel returns None (no double-emit).
    assert pa.cancel("n1") is None
    resolved = [c for c in captured if c[0] == "pending_action_resolved"]
    assert len(resolved) == 1
    assert resolved[0][2]["status"] == "cancelled"
    assert resolved[0][2]["nonce"] == "n1"


def test_cancel_unknown_nonce_returns_none(tmp_path):
    pa = _make_store(tmp_path)
    assert pa.cancel("does-not-exist") is None


def test_event_bus_unset_disables_emit(tmp_path):
    pa = _make_store(tmp_path)
    captured = _attach_recording_bus(pa)
    pa.set_event_bus(None)
    pa.store(
        nonce="n1", actor="operator", target_kind="agent",
        target_id="alpha", action_kind="model", payload={},
    )
    assert captured == []


# ── Inline pending-action card metadata (PR #2) ──────────────────


def test_store_persists_summary_and_preview_diff(tmp_path):
    """The dashboard's inline pending-action card needs both the
    summary and the preview_diff to render without a follow-up
    round-trip. ``store`` accepts both, persists them to the row,
    and ``peek`` / ``list_pending`` / ``consume`` / ``cancel`` all
    return them."""
    pa = _make_store(tmp_path)
    rec = pa.store(
        nonce="n1", actor="operator", target_kind="agent",
        target_id="alpha", action_kind="model", payload={"x": 1},
        summary="Switch alpha's model from gpt-4o to claude-opus",
        preview_diff="--- model\n+++ model\n- gpt-4o\n+ claude-opus\n",
    )
    assert rec["summary"].startswith("Switch alpha")
    assert "+ claude-opus" in rec["preview_diff"]

    peeked = pa.peek("n1")
    assert peeked is not None
    assert peeked["summary"] == rec["summary"]
    assert peeked["preview_diff"] == rec["preview_diff"]

    listed = pa.list_pending()
    assert len(listed) == 1
    assert listed[0]["summary"] == rec["summary"]
    assert listed[0]["preview_diff"] == rec["preview_diff"]

    consumed = pa.consume("n1", confirmer="operator")
    assert consumed is not None
    assert consumed["summary"] == rec["summary"]
    assert consumed["preview_diff"] == rec["preview_diff"]


def test_store_summary_and_preview_diff_default_to_none(tmp_path):
    """Existing call sites that don't pass the new args must keep
    working — the columns are nullable and the dict carries None."""
    pa = _make_store(tmp_path)
    rec = pa.store(
        nonce="n1", actor="operator", target_kind="agent",
        target_id="alpha", action_kind="model", payload={"x": 1},
    )
    assert rec["summary"] is None
    assert rec["preview_diff"] is None
    assert pa.peek("n1")["summary"] is None


def test_pending_action_created_event_carries_summary_and_diff(tmp_path):
    """The event payload must carry summary + preview_diff so the
    dashboard can render the inline card from the event alone."""
    pa = _make_store(tmp_path)
    captured = _attach_recording_bus(pa)
    pa.store(
        nonce="n1", actor="operator", target_kind="agent",
        target_id="alpha", action_kind="model", payload={"x": 1},
        summary="Switch alpha's model from gpt-4o to claude",
        preview_diff="diff goes here",
    )
    evt = next(c for c in captured if c[0] == "pending_action_created")
    assert evt[2]["summary"] == "Switch alpha's model from gpt-4o to claude"
    assert evt[2]["preview_diff"] == "diff goes here"
