"""Tests for ``WorkSummariesStore`` (src/host/summaries.py)."""

from __future__ import annotations

import time

import pytest

from src.host.summaries import (
    DEFAULT_RETENTION_SECONDS,
    MAX_FEEDBACK_CHARS,
    RATING_EDIT_WINDOW_SECONDS,
    InvalidScope,
    RatingLocked,
    SummaryNotFound,
    WorkSummariesStore,
)


@pytest.fixture
def store():
    s = WorkSummariesStore(":memory:")
    yield s
    s.close()


@pytest.fixture
def now():
    return time.time()


def _make(store, *, scope_id="content-seo", scope_kind="team", now=None):
    """Helper: create a baseline summary for the given scope."""
    t = now if now is not None else time.time()
    return store.create(
        scope_kind=scope_kind,
        scope_id=scope_id,
        period_start=t - 86400,
        period_end=t,
        narrative_md="Daily summary for content-seo",
        metrics={"created": 5, "delivered": 3, "blocked": 1, "stuck": 0},
        recommendations=["Investigate stage-4 stall"],
        generated_by="operator",
    )


# =============================================================================
# CRUD basics
# =============================================================================


def test_create_returns_full_record_with_generated_id(store):
    r = _make(store)
    assert r["id"].startswith("ws_")
    assert r["scope_kind"] == "team"
    assert r["scope_id"] == "content-seo"
    assert r["rating"] is None
    assert r["rated_at"] is None
    assert r["metrics"] == {"created": 5, "delivered": 3, "blocked": 1, "stuck": 0}
    assert r["recommendations"] == ["Investigate stage-4 stall"]
    # retention_until = generated_at + DEFAULT_RETENTION_SECONDS
    assert r["retention_until"] - r["generated_at"] == pytest.approx(
        DEFAULT_RETENTION_SECONDS, abs=2,
    )


def test_get_returns_none_for_missing_id(store):
    assert store.get("ws_doesnotexist") is None


def test_create_accepts_explicit_summary_id(store):
    r = store.create(
        scope_kind="team", scope_id="x",
        period_start=0, period_end=1,
        narrative_md="n", metrics={}, generated_by="operator",
        summary_id="ws_fixed",
    )
    assert r["id"] == "ws_fixed"


# =============================================================================
# UNIQUE constraint — one summary per (scope_kind, scope_id, period_start)
# =============================================================================


def test_create_dedupes_same_scope_and_period(store, now):
    _make(store, now=now)
    with pytest.raises(ValueError, match="already exists"):
        _make(store, now=now)  # same period_start


def test_create_different_period_allowed(store, now):
    _make(store, now=now)
    later = store.create(
        scope_kind="team", scope_id="content-seo",
        period_start=now,  # different period_start
        period_end=now + 86400,
        narrative_md="next day", metrics={}, generated_by="operator",
    )
    assert later["id"] != _make.__name__


def test_create_different_scope_id_allowed(store, now):
    _make(store, scope_id="content-seo", now=now)
    other = _make(store, scope_id="growth", now=now)
    assert other["scope_id"] == "growth"


# =============================================================================
# Validation
# =============================================================================


def test_create_rejects_unknown_scope_kind(store):
    with pytest.raises(InvalidScope):
        store.create(
            scope_kind="bogus", scope_id="x",
            period_start=0, period_end=1,
            narrative_md="n", metrics={}, generated_by="operator",
        )


def test_create_rejects_empty_scope_id(store):
    with pytest.raises(ValueError, match="scope_id"):
        store.create(
            scope_kind="team", scope_id="",
            period_start=0, period_end=1,
            narrative_md="n", metrics={}, generated_by="operator",
        )


def test_create_rejects_inverted_period(store):
    with pytest.raises(ValueError, match="period_end"):
        store.create(
            scope_kind="team", scope_id="x",
            period_start=100, period_end=50,
            narrative_md="n", metrics={}, generated_by="operator",
        )


# =============================================================================
# Listing
# =============================================================================


def test_list_recent_orders_newest_first(store):
    a = store.create(
        scope_kind="team", scope_id="x",
        period_start=0, period_end=1, narrative_md="a",
        metrics={}, generated_by="operator",
    )
    time.sleep(0.001)  # ensure distinct generated_at
    b = store.create(
        scope_kind="team", scope_id="x",
        period_start=2, period_end=3, narrative_md="b",
        metrics={}, generated_by="operator",
    )
    rows = store.list_recent()
    assert [r["id"] for r in rows] == [b["id"], a["id"]]


def test_list_recent_filters_by_scope(store):
    _make(store, scope_id="x", now=time.time())
    _make(store, scope_id="y", now=time.time())
    rows = store.list_recent(scope_kind="team", scope_id="x")
    assert len(rows) == 1
    assert rows[0]["scope_id"] == "x"


def test_list_recent_filter_solo_only(store):
    _make(store, scope_kind="team", scope_id="x", now=time.time())
    _make(store, scope_kind="solo", scope_id="standalone-agent", now=time.time())
    solos = store.list_recent(scope_kind="solo")
    assert len(solos) == 1
    assert solos[0]["scope_kind"] == "solo"


def test_list_recent_limit_clamped(store):
    for i in range(10):
        store.create(
            scope_kind="team", scope_id=f"t{i}",
            period_start=i, period_end=i + 1,
            narrative_md="n", metrics={}, generated_by="operator",
        )
    rows = store.list_recent(limit=3)
    assert len(rows) == 3


def test_list_recent_limit_above_max_clamped_to_500(store):
    rows = store.list_recent(limit=99999)
    # No rows yet → empty, but the clamp doesn't error.
    assert rows == []


# =============================================================================
# Rating
# =============================================================================


def test_set_rating_accepts_thumbs_up(store):
    r = _make(store)
    out = store.set_rating(r["id"], "accepted")
    assert out["rating"] == "accepted"
    assert out["feedback"] is None
    assert out["rated_at"] is not None
    assert out["rated_by"] == "user"


def test_set_rating_records_actor(store):
    r = _make(store)
    out = store.set_rating(r["id"], "accepted", actor="admin@curiouscake.com")
    assert out["rated_by"] == "admin@curiouscake.com"


def test_set_rating_accepts_feedback_with_rework(store):
    r = _make(store)
    out = store.set_rating(
        r["id"], "rework",
        feedback="Focus more on stage-4 publishing throughput",
    )
    assert out["rating"] == "rework"
    assert "stage-4" in out["feedback"]


def test_set_rating_rejects_unknown_rating(store):
    r = _make(store)
    with pytest.raises(ValueError, match="rating must"):
        store.set_rating(r["id"], "kinda-good")


def test_set_rating_rejects_oversize_feedback(store):
    r = _make(store)
    with pytest.raises(ValueError, match="exceeds"):
        store.set_rating(r["id"], "rework", feedback="x" * (MAX_FEEDBACK_CHARS + 1))


def test_set_rating_missing_summary_raises(store):
    with pytest.raises(SummaryNotFound):
        store.set_rating("ws_ghost", "accepted")


def test_set_rating_within_edit_window_overwrites(store):
    r = _make(store)
    first = store.set_rating(r["id"], "accepted")
    edited = store.set_rating(r["id"], "rework", feedback="changed my mind")
    assert edited["rating"] == "rework"
    assert edited["feedback"] == "changed my mind"
    # rated_at preserved from first rating (COALESCE)
    assert edited["rated_at"] == first["rated_at"]


def test_set_rating_past_edit_window_raises(monkeypatch, store):
    r = _make(store)
    store.set_rating(r["id"], "accepted")
    # Simulate clock moving past the edit window.
    real_time = time.time
    fake_now = real_time() + RATING_EDIT_WINDOW_SECONDS + 60
    monkeypatch.setattr(time, "time", lambda: fake_now)
    with pytest.raises(RatingLocked):
        store.set_rating(r["id"], "rework", feedback="too late")


# =============================================================================
# TTL / reaping
# =============================================================================


def test_reap_expired_drops_old_rows(monkeypatch, store):
    r = _make(store)
    # Fast-forward past retention.
    real_time = time.time
    fake_now = real_time() + DEFAULT_RETENTION_SECONDS + 10
    monkeypatch.setattr(time, "time", lambda: fake_now)
    reaped = store.reap_expired()
    assert reaped == 1
    assert store.get(r["id"]) is None


def test_reap_expired_preserves_fresh_rows(store):
    r = _make(store)
    reaped = store.reap_expired()
    assert reaped == 0
    assert store.get(r["id"]) is not None


def test_safe_reap_swallows_errors(store, monkeypatch):
    def boom():
        raise RuntimeError("simulated reap failure")
    monkeypatch.setattr(store, "reap_expired", boom)
    # Must not raise.
    store._safe_reap()


# =============================================================================
# Feedback fetch (for next-summary prompt injection)
# =============================================================================


def test_recent_feedback_returns_only_rated_in_scope(store):
    r1 = _make(store, scope_id="x", now=time.time())
    time.sleep(0.001)
    r2 = _make(store, scope_id="y", now=time.time())
    # Only r1 gets rated.
    store.set_rating(r1["id"], "rework", feedback="improve velocity")
    fb_x = store.recent_feedback(scope_kind="team", scope_id="x")
    fb_y = store.recent_feedback(scope_kind="team", scope_id="y")
    assert len(fb_x) == 1
    assert fb_x[0]["feedback"] == "improve velocity"
    assert fb_y == []


def test_recent_feedback_orders_newest_first(store):
    r1 = _make(store, scope_id="x", now=time.time())
    time.sleep(0.001)
    r2 = store.create(
        scope_kind="team", scope_id="x",
        period_start=time.time(), period_end=time.time() + 100,
        narrative_md="later", metrics={}, generated_by="operator",
    )
    store.set_rating(r1["id"], "accepted")
    time.sleep(0.001)
    store.set_rating(r2["id"], "rework", feedback="newer")
    fb = store.recent_feedback(scope_kind="team", scope_id="x")
    assert fb[0]["id"] == r2["id"]
    assert fb[1]["id"] == r1["id"]


def test_recent_feedback_rejects_bad_scope(store):
    with pytest.raises(InvalidScope):
        store.recent_feedback(scope_kind="garbage", scope_id="x")


# =============================================================================
# Event emission
# =============================================================================


class _RecordingBus:
    def __init__(self):
        self.events: list[tuple[str, str, dict]] = []

    def emit(self, event_type, *, agent, data):
        self.events.append((event_type, agent, data))


def test_create_emits_work_summary_created():
    bus = _RecordingBus()
    s = WorkSummariesStore(":memory:", event_bus=bus)
    try:
        _make(s)
    finally:
        s.close()
    assert len(bus.events) == 1
    event_type, agent, data = bus.events[0]
    assert event_type == "work_summary_created"
    assert agent == "operator"
    assert data["scope_kind"] == "team"
    assert data["scope_id"] == "content-seo"
    assert "metrics" in data


def test_set_rating_emits_work_summary_rated():
    bus = _RecordingBus()
    s = WorkSummariesStore(":memory:", event_bus=bus)
    try:
        r = _make(s)
        bus.events.clear()
        s.set_rating(r["id"], "accepted", actor="admin@curiouscake.com")
    finally:
        s.close()
    assert len(bus.events) == 1
    event_type, agent, data = bus.events[0]
    assert event_type == "work_summary_rated"
    assert agent == "admin@curiouscake.com"
    assert data["rating"] == "accepted"
    assert data["summary_id"] == r["id"]


def test_emit_failure_does_not_sink_db_write():
    class _Broken:
        def emit(self, *_a, **_kw):
            raise RuntimeError("broken bus")
    s = WorkSummariesStore(":memory:", event_bus=_Broken())
    try:
        # Create must succeed despite the bus raising.
        r = _make(s)
        assert r["id"] is not None
    finally:
        s.close()
