"""Tests for IntentStore + verbatim-intent capture wiring at _direct_dispatch.

Phase 2 of docs/plans/2026-06-18-session-observability.md: the verbatim
inbound message is persisted centrally (keyed by trace_id + MessageOrigin),
redacted at storage, so intent survives container wipes / resets / deploys.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.host.intent import IntentStore
from src.shared.types import MessageOrigin

# ── IntentStore ──────────────────────────────────────────────


class TestIntentStore:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self._tmpdir, "intent.db")
        self.store = IntentStore(db_path=self.db_path)

    def teardown_method(self):
        self.store.close()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_record_and_get_by_trace(self):
        self.store.record(
            trace_id="tr_abc", origin_kind="human", origin_channel="dashboard",
            origin_user="u1", agent="alpha", message="build me a report",
        )
        self.store.record(
            trace_id="tr_abc", origin_kind="human", origin_channel="dashboard",
            origin_user="u1", agent="alpha", message="and email it",
        )
        rows = self.store.get_by_trace("tr_abc")
        assert len(rows) == 2
        assert rows[0]["trace_id"] == "tr_abc"
        assert rows[0]["origin_kind"] == "human"
        assert rows[0]["origin_channel"] == "dashboard"
        assert rows[0]["origin_user"] == "u1"
        assert rows[0]["agent"] == "alpha"
        assert rows[0]["message"] == "build me a report"
        # ordered by time
        assert rows[1]["message"] == "and email it"

    def test_get_by_trace_empty(self):
        assert self.store.get_by_trace("tr_nope") == []

    def test_full_message_not_truncated(self):
        # Realistic prose (no long unbroken alnum blob that redaction would
        # collapse) — assert the FULL message is stored, not a preview.
        long_msg = ("please build a detailed quarterly report. " * 200).strip()
        assert len(long_msg) > 4000
        self.store.record(trace_id="tr_long", message=long_msg, agent="alpha")
        rows = self.store.get_by_trace("tr_long")
        assert rows[0]["message"] == long_msg

    def test_list_recent_newest_first(self):
        for i in range(3):
            self.store.record(trace_id=f"tr_{i}", message=f"m{i}", agent="alpha")
        rows = self.store.list_recent(limit=10)
        assert [r["trace_id"] for r in rows] == ["tr_2", "tr_1", "tr_0"]

    def test_list_recent_filter_since(self):
        self.store.record(trace_id="tr_old", message="old", agent="alpha")
        cut = time.time()
        time.sleep(0.01)
        self.store.record(trace_id="tr_new", message="new", agent="alpha")
        rows = self.store.list_recent(since=cut)
        assert [r["trace_id"] for r in rows] == ["tr_new"]

    def test_list_recent_filter_user(self):
        self.store.record(trace_id="tr_a", message="a", agent="alpha", origin_user="alice")
        self.store.record(trace_id="tr_b", message="b", agent="alpha", origin_user="bob")
        rows = self.store.list_recent(user="alice")
        assert [r["trace_id"] for r in rows] == ["tr_a"]

    def test_list_recent_filter_agent(self):
        self.store.record(trace_id="tr_a", message="a", agent="alpha")
        self.store.record(trace_id="tr_b", message="b", agent="beta")
        rows = self.store.list_recent(agent="beta")
        assert [r["trace_id"] for r in rows] == ["tr_b"]

    def test_redaction_at_storage(self):
        secret = "sk-ant-api03ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        self.store.record(trace_id="tr_sec", message=f"my key is {secret}", agent="alpha")
        # raw SQLite bytes must not contain the secret
        raw = self.store._conn.execute(
            "SELECT message FROM intent WHERE trace_id = ?", ("tr_sec",)
        ).fetchone()[0]
        assert secret not in raw
        assert "[REDACTED]" in raw
        rows = self.store.get_by_trace("tr_sec")
        assert secret not in rows[0]["message"]

    def test_meta_redacted_at_storage(self):
        secret = "sk-ant-api03ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        self.store.record(
            trace_id="tr_meta", message="hi", agent="alpha",
            meta={"note": f"creds {secret}", "system_note": False},
        )
        raw = self.store._conn.execute(
            "SELECT meta_json FROM intent WHERE trace_id = ?", ("tr_meta",)
        ).fetchone()[0]
        assert secret not in raw
        rows = self.store.get_by_trace("tr_meta")
        assert secret not in rows[0]["meta"]["note"]

    def test_time_based_gc_drops_old_rows(self):
        store = IntentStore(
            db_path=os.path.join(self._tmpdir, "gc.db"), max_age_hours=1,
        )
        try:
            # Insert an ancient row directly, bypassing the GC throttle.
            store._conn.execute(
                "INSERT INTO intent (trace_id, timestamp, message) VALUES (?, ?, ?)",
                ("tr_ancient", time.time() - 7200, "old"),
            )
            store._conn.commit()
            assert store.get_by_trace("tr_ancient")
            # Force GC to run (reset throttle), then a fresh record triggers it.
            store._last_age_gc = -300.0
            store.record(trace_id="tr_fresh", message="new", agent="alpha")
            assert store.get_by_trace("tr_ancient") == []
            assert store.get_by_trace("tr_fresh")
        finally:
            store.close()

    def test_idempotent_schema_init(self):
        # Re-open the same DB; CREATE TABLE/INDEX IF NOT EXISTS must not raise.
        self.store.record(trace_id="tr_x", message="hi", agent="alpha")
        self.store.close()
        store2 = IntentStore(db_path=self.db_path)
        try:
            assert store2.get_by_trace("tr_x")
            store2.record(trace_id="tr_y", message="yo", agent="alpha")
            assert store2.get_by_trace("tr_y")
        finally:
            self.store = store2  # so teardown closes it


# ── Capture wiring at _direct_dispatch ───────────────────────


def _dispatch_with_intent(monkeypatch, intent_store, *, response="done"):
    """Build the real ``_direct_dispatch`` closure via test_runtime's proven
    harness, then attach our intent-store mock and a stubbed transport reply.
    Returns ``(dispatch_fn, ctx, transport_mock)``."""
    from tests.test_runtime import _capture_dispatch_fn

    dispatch_fn, ctx, transport = _capture_dispatch_fn(monkeypatch)
    ctx.intent_store = intent_store
    transport.request = AsyncMock(return_value={"response": response})
    return dispatch_fn, ctx, transport


@pytest.mark.asyncio
async def test_dispatch_captures_full_verbatim_intent(monkeypatch):
    intent_store = MagicMock()
    dispatch_fn, _ctx, _t = _dispatch_with_intent(monkeypatch, intent_store)

    origin = MessageOrigin(kind="human", channel="dashboard", user="u9")
    long_message = "please build a detailed quarterly report. " * 50
    await dispatch_fn("alpha", long_message, origin=origin)

    assert intent_store.record.called
    kwargs = intent_store.record.call_args.kwargs
    # FULL message, not truncated, with origin + agent stamped.
    assert kwargs["message"] == long_message
    assert kwargs["agent"] == "alpha"
    assert kwargs["origin_kind"] == "human"
    assert kwargs["origin_channel"] == "dashboard"
    assert kwargs["origin_user"] == "u9"
    assert kwargs["trace_id"]


@pytest.mark.asyncio
async def test_dispatch_capture_marks_system_note(monkeypatch):
    intent_store = MagicMock()
    dispatch_fn, _ctx, _t = _dispatch_with_intent(
        monkeypatch, intent_store, response="ok",
    )

    # Webhook path: machine origin, no MessageOrigin object.
    await dispatch_fn("alpha", "webhook fired", system_note=True)

    kwargs = intent_store.record.call_args.kwargs
    assert kwargs["meta"]["system_note"] is True
    assert kwargs["origin_kind"] == ""  # no origin object
    assert kwargs["origin_channel"] == ""
    assert kwargs["origin_user"] == ""


@pytest.mark.asyncio
async def test_dispatch_survives_intent_store_failure(monkeypatch):
    intent_store = MagicMock()
    intent_store.record.side_effect = RuntimeError("disk full")
    dispatch_fn, _ctx, _t = _dispatch_with_intent(
        monkeypatch, intent_store, response="still works",
    )

    result = await dispatch_fn("alpha", "hi", origin=None)

    # A failing intent store must NEVER break dispatch.
    assert result == "still works"
    assert intent_store.record.called
