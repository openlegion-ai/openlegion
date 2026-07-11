"""Tests for ``TrackRecordStore`` (src/host/track_record.py).

Covers append/counts/rater-filtering/pair_trust/persistence-across-
reopen and the append-only invariant (no reap method, no
``retention_until`` column) — plan §8 #18's durability property, the
whole reason this store exists alongside the reaped tasks/summaries
tables it's assembled from.
"""

from __future__ import annotations

import sqlite3
import time

import pytest

from src.host.track_record import (
    AUTONOMY_RATER_KINDS,
    TrackRecordStore,
    record_best_effort,
)


@pytest.fixture
def store():
    s = TrackRecordStore(":memory:")
    yield s
    s.close()


# =============================================================================
# append / basic shape
# =============================================================================


def test_record_returns_full_event(store):
    event = store.record(
        source="task_outcome",
        ref_id="task_1",
        outcome="accepted",
        rater_kind="human",
        agent_id="writer",
        team_id="content-seo",
        rated_by="operator",
    )
    assert event["id"] is not None
    assert event["source"] == "task_outcome"
    assert event["ref_id"] == "task_1"
    assert event["outcome"] == "accepted"
    assert event["rater_kind"] == "human"
    assert event["agent_id"] == "writer"
    assert event["team_id"] == "content-seo"
    assert event["rated_by"] == "operator"
    assert event["details"] is None
    assert event["created_at"] <= time.time()


def test_record_stores_details_json(store):
    event = store.record(
        source="drive_review",
        ref_id="rev_1",
        outcome="merged",
        rater_kind="human",
        agent_id="writer",
        team_id="content-seo",
        rated_by="operator",
        details={"branch": "feat-x", "lead_agent_id": "lead1", "lead_verdict": "approve"},
    )
    assert event["details"] == {"branch": "feat-x", "lead_agent_id": "lead1", "lead_verdict": "approve"}


def test_record_team_scoped_summary_has_no_agent_id(store):
    """Team-scoped summary ratings have no single rated agent (spec)."""
    event = store.record(
        source="summary_rating",
        ref_id="ws_1",
        outcome="accepted",
        rater_kind="human",
        team_id="content-seo",
        rated_by="operator",
    )
    assert event["agent_id"] is None
    assert event["team_id"] == "content-seo"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source": "bogus", "ref_id": "x", "outcome": "accepted", "rater_kind": "human", "agent_id": "a"},
        {"source": "task_outcome", "ref_id": "x", "outcome": "accepted", "rater_kind": "bogus", "agent_id": "a"},
        {"source": "task_outcome", "ref_id": "", "outcome": "accepted", "rater_kind": "human", "agent_id": "a"},
        {"source": "task_outcome", "ref_id": "x", "outcome": "", "rater_kind": "human", "agent_id": "a"},
        {"source": "task_outcome", "ref_id": "x", "outcome": "accepted", "rater_kind": "human"},
    ],
)
def test_record_rejects_invalid_input(store, kwargs):
    with pytest.raises(ValueError):
        store.record(**kwargs)


def test_record_oversized_details_replaced_with_marker(store):
    huge = {"blob": "x" * 20_000}
    event = store.record(
        source="task_outcome",
        ref_id="task_big",
        outcome="accepted",
        rater_kind="human",
        agent_id="writer",
        details=huge,
    )
    assert event["details"] == {"truncated": True}


# =============================================================================
# counts_for_agent + rater filtering (rating-trust rule)
# =============================================================================


def test_counts_for_agent_keyed_by_source_and_outcome(store):
    store.record(source="task_outcome", ref_id="t1", outcome="accepted", rater_kind="human", agent_id="a")
    store.record(source="task_outcome", ref_id="t2", outcome="accepted", rater_kind="human", agent_id="a")
    store.record(source="task_outcome", ref_id="t3", outcome="rework", rater_kind="human", agent_id="a")
    store.record(source="summary_rating", ref_id="s1", outcome="accepted", rater_kind="human", agent_id="a")
    # A different agent's events must not bleed into "a"'s counts.
    store.record(source="task_outcome", ref_id="t4", outcome="accepted", rater_kind="human", agent_id="b")

    counts = store.counts_for_agent("a")
    assert counts == {
        "task_outcome": {"accepted": 2, "rework": 1},
        "summary_rating": {"accepted": 1},
    }


def test_counts_for_agent_never_double_unifies_enum_values(store):
    """Drive-review outcomes (merged/rejected) never collide with task
    outcomes (accepted/rework/rejected/acknowledged) — keyed by source
    first, so the raw per-source vocabulary is preserved verbatim."""
    store.record(source="task_outcome", ref_id="t1", outcome="rejected", rater_kind="human", agent_id="a")
    store.record(source="drive_review", ref_id="r1", outcome="rejected", rater_kind="human", agent_id="a")
    counts = store.counts_for_agent("a")
    assert counts["task_outcome"]["rejected"] == 1
    assert counts["drive_review"]["rejected"] == 1


def test_rater_kinds_filter_restricts_counts(store):
    store.record(source="task_outcome", ref_id="t1", outcome="accepted", rater_kind="human", agent_id="a")
    store.record(source="task_outcome", ref_id="t2", outcome="accepted", rater_kind="operator_agent", agent_id="a")
    store.record(source="task_outcome", ref_id="t3", outcome="accepted", rater_kind="system", agent_id="a")

    all_counts = store.counts_for_agent("a")
    assert all_counts["task_outcome"]["accepted"] == 3

    autonomy_counts = store.counts_for_agent("a", rater_kinds=AUTONOMY_RATER_KINDS)
    assert autonomy_counts["task_outcome"]["accepted"] == 2  # human + system, operator_agent excluded


def test_operator_agent_rated_events_absent_from_autonomy_counts_present_in_counts(store):
    """Pins the rating-trust rule (§8 #18): an operator-agent-rated
    event is counted in ``counts`` but excluded from the autonomy-safe
    view — agents grading agents must not feed the trust ladder."""
    store.record(
        source="task_outcome", ref_id="t1", outcome="accepted",
        rater_kind="operator_agent", agent_id="a", rated_by="operator",
    )
    counts = store.counts_for_agent("a")
    autonomy_counts = store.counts_for_agent("a", rater_kinds=AUTONOMY_RATER_KINDS)
    assert counts.get("task_outcome", {}).get("accepted") == 1
    assert autonomy_counts.get("task_outcome", {}).get("accepted", 0) == 0


def test_counts_for_agent_unknown_agent_is_empty(store):
    assert store.counts_for_agent("nobody") == {}


# =============================================================================
# recent_events
# =============================================================================


def test_recent_events_newest_first_and_limit_clamped(store):
    for i in range(5):
        store.record(source="task_outcome", ref_id=f"t{i}", outcome="accepted", rater_kind="human", agent_id="a")
    recent = store.recent_events("a", limit=3)
    assert len(recent) == 3
    assert [e["ref_id"] for e in recent] == ["t4", "t3", "t2"]


def test_recent_events_scoped_to_agent(store):
    store.record(source="task_outcome", ref_id="t1", outcome="accepted", rater_kind="human", agent_id="a")
    store.record(source="task_outcome", ref_id="t2", outcome="accepted", rater_kind="human", agent_id="b")
    recent = store.recent_events("a", limit=20)
    assert [e["ref_id"] for e in recent] == ["t1"]


# =============================================================================
# pair_trust (§8 #20 auto-merge trust signal)
# =============================================================================


def _drive_event(store, *, submitter, lead, verdict, resolution, ts_offset=0.0):
    store.record(
        source="drive_review",
        ref_id=f"rev_{submitter}_{resolution}_{ts_offset}",
        outcome=resolution,
        rater_kind="human",
        agent_id=submitter,
        team_id="team-x",
        rated_by="operator",
        details={
            "branch": "feat",
            "lead_agent_id": lead,
            "lead_verdict": verdict,
            "lead_verdict_at": "2026-07-11T00:00:00Z",
            "resolution": resolution,
            "resolved_by": "operator",
        },
    )


def test_pair_trust_counts_merged_and_rejected_after_approve(store):
    _drive_event(store, submitter="writer", lead="lead1", verdict="approve", resolution="merged")
    _drive_event(store, submitter="writer", lead="lead1", verdict="approve", resolution="merged")
    _drive_event(store, submitter="writer", lead="lead1", verdict="approve", resolution="rejected")

    result = store.pair_trust("lead1", "writer")
    assert result["merged"] == 2
    assert result["rejected_after_approve"] == 1
    assert result["last_event_at"] is not None


def test_pair_trust_ignores_non_approve_verdicts(store):
    _drive_event(store, submitter="writer", lead="lead1", verdict="reject", resolution="rejected")
    _drive_event(store, submitter="writer", lead="lead1", verdict=None, resolution="merged")

    result = store.pair_trust("lead1", "writer")
    assert result["merged"] == 0
    assert result["rejected_after_approve"] == 0
    assert result["last_event_at"] is None


def test_pair_trust_scoped_to_exact_lead_submitter_pair(store):
    # Same submitter, different lead — must not count toward lead1's trust.
    _drive_event(store, submitter="writer", lead="lead2", verdict="approve", resolution="merged")
    # Same lead, different submitter — must not count toward writer's trust.
    _drive_event(store, submitter="other", lead="lead1", verdict="approve", resolution="merged")

    result = store.pair_trust("lead1", "writer")
    assert result["merged"] == 0
    assert result["rejected_after_approve"] == 0


def test_pair_trust_last_event_at_is_most_recent_qualifying_event(store):
    older = store.record(
        source="drive_review", ref_id="rev_a", outcome="merged", rater_kind="human",
        agent_id="writer", team_id="team-x",
        details={"lead_agent_id": "lead1", "lead_verdict": "approve"},
    )
    newer = store.record(
        source="drive_review", ref_id="rev_b", outcome="merged", rater_kind="human",
        agent_id="writer", team_id="team-x",
        details={"lead_agent_id": "lead1", "lead_verdict": "approve"},
    )
    assert newer["created_at"] >= older["created_at"]
    result = store.pair_trust("lead1", "writer")
    assert result["last_event_at"] == newer["created_at"]


def test_pair_trust_no_events_returns_zeroed_result(store):
    result = store.pair_trust("lead1", "nobody")
    assert result == {
        "lead_agent_id": "lead1",
        "submitter_agent_id": "nobody",
        "merged": 0,
        "rejected_after_approve": 0,
        "flagged": 0,
        "auto_merged": 0,
        "last_event_at": None,
    }


def test_pair_trust_excludes_system_rated_auto_merges_from_floor_count(store):
    """The self-reinforcement pin (§8 #20): a kernel-executed auto-merge
    is rater_kind='system' + outcome='auto_merged' — it must be counted
    ONLY as `auto_merged` (the sampling-decay input), NEVER folded into
    `merged` (the trust-floor input). Otherwise auto-merges would feed
    the very floor that gates further auto-merges."""
    _drive_event(store, submitter="writer", lead="lead1", verdict="approve", resolution="merged")
    store.record(
        source="drive_review", ref_id="rev_auto_1", outcome="auto_merged", rater_kind="system",
        agent_id="writer", team_id="team-x", rated_by="policy_engine",
        details={"lead_agent_id": "lead1", "lead_verdict": "approve", "resolution": "auto_merged"},
    )
    store.record(
        source="drive_review", ref_id="rev_auto_2", outcome="auto_merged", rater_kind="system",
        agent_id="writer", team_id="team-x", rated_by="policy_engine",
        details={"lead_agent_id": "lead1", "lead_verdict": "approve", "resolution": "auto_merged"},
    )

    result = store.pair_trust("lead1", "writer")
    assert result["merged"] == 1, "only the human-rated merge counts toward the trust floor"
    assert result["auto_merged"] == 2
    assert result["rejected_after_approve"] == 0


def test_pair_trust_flagged_counts_decay_events_and_zeroes_eligibility(store):
    _drive_event(store, submitter="writer", lead="lead1", verdict="approve", resolution="merged")
    store.record(
        source="drive_review", ref_id="rev_flag", outcome="auto_merge_flagged", rater_kind="human",
        agent_id="writer", team_id="team-x", rated_by="operator",
        details={"lead_agent_id": "lead1", "resolution": "auto_merge_flagged"},
    )

    result = store.pair_trust("lead1", "writer")
    assert result["merged"] == 1
    assert result["flagged"] == 1


def test_pair_trust_reverted_also_counts_as_flagged(store):
    store.record(
        source="drive_review", ref_id="rev_revert", outcome="auto_merge_reverted", rater_kind="human",
        agent_id="writer", team_id="team-x", rated_by="operator",
        details={"lead_agent_id": "lead1", "resolution": "auto_merge_reverted"},
    )
    result = store.pair_trust("lead1", "writer")
    assert result["flagged"] == 1


# =============================================================================
# count_events (§8 #20 auto-merge daily rate cap)
# =============================================================================


def test_count_events_filters_source_outcome_rater_kind_and_since(store):
    store.record(source="drive_review", ref_id="r1", outcome="auto_merged", rater_kind="system", agent_id="a")
    store.record(source="drive_review", ref_id="r2", outcome="auto_merged", rater_kind="system", agent_id="b")
    store.record(source="drive_review", ref_id="r3", outcome="merged", rater_kind="human", agent_id="a")

    assert store.count_events(source="drive_review", outcome="auto_merged") == 2
    assert store.count_events(source="drive_review", outcome="auto_merged", rater_kind="system") == 2
    assert store.count_events(source="drive_review", outcome="auto_merged", rater_kind="human") == 0
    assert store.count_events(source="drive_review", outcome="merged") == 1
    # since=far future excludes everything already written.
    assert store.count_events(source="drive_review", outcome="auto_merged", since=time.time() + 3600) == 0


# =============================================================================
# persistence across reopen (disk-backed)
# =============================================================================


def test_persists_across_reopen(tmp_path):
    db_path = str(tmp_path / "track_record.db")
    s1 = TrackRecordStore(db_path)
    s1.record(source="task_outcome", ref_id="t1", outcome="accepted", rater_kind="human", agent_id="a")
    s1.close()

    s2 = TrackRecordStore(db_path)
    try:
        counts = s2.counts_for_agent("a")
        assert counts == {"task_outcome": {"accepted": 1}}
        recent = s2.recent_events("a")
        assert len(recent) == 1
        assert recent[0]["ref_id"] == "t1"
    finally:
        s2.close()


def test_disk_backed_creates_parent_dir(tmp_path):
    db_path = tmp_path / "nested" / "dir" / "track_record.db"
    s = TrackRecordStore(str(db_path))
    try:
        assert db_path.parent.exists()
    finally:
        s.close()


# =============================================================================
# append-only invariant (§8 #18: NEVER reaped)
# =============================================================================


def test_store_has_no_reap_method(store):
    """The whole point of this store vs. tasks/work_summaries is that it
    is NEVER reaped — pin the absence of any reap-shaped method so a
    future edit can't quietly reintroduce one."""
    assert not hasattr(store, "reap_expired")
    assert not hasattr(store, "_safe_reap")


def test_schema_has_no_retention_column(tmp_path):
    db_path = str(tmp_path / "track_record.db")
    s = TrackRecordStore(db_path)
    try:
        conn = sqlite3.connect(db_path)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(outcome_events)").fetchall()}
        conn.close()
        assert "retention_until" not in cols
    finally:
        s.close()


def test_schema_user_version_is_1(tmp_path):
    db_path = str(tmp_path / "track_record.db")
    s = TrackRecordStore(db_path)
    try:
        conn = sqlite3.connect(db_path)
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        conn.close()
        assert version == 1
    finally:
        s.close()


# =============================================================================
# record_best_effort — never raises, never fails the caller
# =============================================================================


def test_record_best_effort_noop_when_store_is_none():
    # Must not raise even though there's nothing to write to.
    record_best_effort(None, source="task_outcome", ref_id="t1", outcome="accepted", rater_kind="human", agent_id="a")


def test_record_best_effort_swallows_store_errors(store, caplog):
    # Missing agent_id/team_id raises ValueError inside store.record —
    # record_best_effort must swallow it, not propagate.
    record_best_effort(store, source="task_outcome", ref_id="t1", outcome="accepted", rater_kind="human")
    assert store.counts_for_agent("t1") == {}


def test_record_best_effort_writes_through_on_success(store):
    record_best_effort(
        store, source="task_outcome", ref_id="t1", outcome="accepted",
        rater_kind="human", agent_id="a", rated_by="operator",
    )
    assert store.counts_for_agent("a") == {"task_outcome": {"accepted": 1}}
