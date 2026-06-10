"""Tests for the chain watcher (delegate-and-subscribe terminal delivery).

Covers the Tasks store helpers (chain_deliveries dedup, human-root listing,
whole-chain terminal verdict) and the ChainWatcher sweep semantics
(settle/debounce against the in-flight hand-off race, exactly-once delivery,
deliver-then-claim retry, cancelled handling, and human-origin gating).
"""

from __future__ import annotations

import time

import pytest

from src.host.chain_watcher import ChainWatcher
from src.host.orchestration import Tasks

HUMAN = {"kind": "human", "channel": "dashboard", "user": "u1"}


def _store() -> Tasks:
    return Tasks(db_path=":memory:")


def _human_root(store: Tasks, *, assignee: str = "scout") -> dict:
    return store.create(
        creator="operator", assignee=assignee, title="do research",
        origin=HUMAN,
    )


def _finish(store: Tasks, task_id: str, status: str = "done", **kw) -> None:
    store.update_status(task_id, "working", actor="x")
    store.update_status(task_id, status, actor="x", **kw)


# ── Store: chain_deliveries dedup ────────────────────────────────


def test_claim_chain_delivery_is_exactly_once():
    s = _store()
    assert s.claim_chain_delivery("root-1", "done") is True
    assert s.claim_chain_delivery("root-1", "done") is False
    assert s.claim_chain_delivery("root-1", "failed") is False


# ── Store: list_watchable_human_roots ────────────────────────────


def test_list_watchable_only_human_roots():
    s = _store()
    human = _human_root(s)
    # Non-human root (worker-origin / no origin) — excluded.
    s.create(creator="a", assignee="b", title="internal")
    # A child hop of the human root — not a root, excluded.
    s.create(creator="scout", assignee="writer", title="next",
             parent_task_id=human["id"])
    roots = s.list_watchable_human_roots(since=0.0)
    assert [r["id"] for r in roots] == [human["id"]]


def test_list_watchable_maps_columns_correctly():
    """The projected row must map back to the right fields (guards against a
    SELECT * positional drift vs _row_to_dict / _SELECT_COLS)."""
    s = _store()
    r = _human_root(s)
    _finish(s, r["id"], "done", result_summary="the deliverable")
    [row] = s.list_watchable_human_roots(since=0.0)
    assert row["id"] == r["id"]
    assert row["origin"] == HUMAN
    assert row["assignee"] == "scout"
    assert row["result_summary"] == "the deliverable"
    assert row["status"] == "done"


def test_list_watchable_excludes_claimed_and_old():
    s = _store()
    r = _human_root(s)
    # since in the future → outside window → excluded.
    assert s.list_watchable_human_roots(since=r["created_at"] + 1000) == []
    # claimed → excluded.
    s.claim_chain_delivery(r["id"], "done")
    assert s.list_watchable_human_roots(since=0.0) == []


# ── Store: chain_terminal_verdict ────────────────────────────────


def test_verdict_none_while_active():
    s = _store()
    r = _human_root(s)
    assert s.chain_terminal_verdict(r["id"]) is None  # pending
    s.update_status(r["id"], "working", actor="x")
    assert s.chain_terminal_verdict(r["id"]) is None  # working


def test_verdict_done_single_task():
    s = _store()
    r = _human_root(s)
    _finish(s, r["id"], "done", result_summary="found 5 trends")
    assert s.chain_terminal_verdict(r["id"]) == ("done", "found 5 trends")


def test_verdict_multi_hop_all_done():
    s = _store()
    r = _human_root(s)
    _finish(s, r["id"], "done", result_summary="stage1")
    child = s.create(creator="scout", assignee="writer", title="draft",
                     parent_task_id=r["id"])
    # Chain not terminal until the child finishes.
    assert s.chain_terminal_verdict(r["id"]) is None
    _finish(s, child["id"], "done", result_summary="final draft")
    kind, summary = s.chain_terminal_verdict(r["id"])
    assert kind == "done"
    assert summary == "final draft"  # most-recent done leaf


def test_verdict_failed_takes_precedence():
    s = _store()
    r = _human_root(s)
    _finish(s, r["id"], "done", result_summary="ok")
    child = s.create(creator="scout", assignee="writer", title="draft",
                     parent_task_id=r["id"])
    _finish(s, child["id"], "failed", blocker_note="model rejected")
    assert s.chain_terminal_verdict(r["id"]) == ("failed", "model rejected")


def test_verdict_cancelled():
    s = _store()
    r = _human_root(s)
    s.update_status(r["id"], "cancelled", actor="x")
    assert s.chain_terminal_verdict(r["id"]) == ("cancelled", "")


def test_verdict_partial_cancel_delivers_done():
    """A chain that substantively completed but has one cancelled branch must
    resolve to 'done' (not a silent 'cancelled') — every user pipeline ends in
    a user-facing outcome."""
    s = _store()
    r = _human_root(s)
    _finish(s, r["id"], "done", result_summary="main work done")
    branch = s.create(creator="scout", assignee="helper", title="optional",
                      parent_task_id=r["id"])
    s.update_status(branch["id"], "cancelled", actor="x")
    assert s.chain_terminal_verdict(r["id"]) == ("done", "main work done")


def test_verdict_all_cancelled_stays_silent():
    """A chain whose root was cancelled is a genuine manual cancellation →
    ('cancelled', ''), even with downstream cancelled branches."""
    s = _store()
    r = _human_root(s)
    s.update_status(r["id"], "cancelled", actor="x")
    branch = s.create(creator="scout", assignee="helper", title="optional",
                      parent_task_id=r["id"])
    s.update_status(branch["id"], "cancelled", actor="x")
    assert s.chain_terminal_verdict(r["id"]) == ("cancelled", "")


def test_verdict_root_cancelled_after_partial_progress_stays_silent():
    """Cancellation is judged by the ROOT, not "any done task": if the user
    cancels their request after a sub-stage already finished, we must NOT claim
    "✅ complete" — the root was cancelled, so stay silent."""
    s = _store()
    r = _human_root(s)
    child = s.create(creator="scout", assignee="helper", title="sub-stage",
                     parent_task_id=r["id"])
    _finish(s, child["id"], "done", result_summary="a stage finished")
    s.update_status(r["id"], "cancelled", actor="x")  # user cancels the request
    assert s.chain_terminal_verdict(r["id"]) == ("cancelled", "")


def test_verdict_depth_cap_not_false_terminal(monkeypatch):
    """A chain deeper than the recursion cap must NOT be declared terminal
    while a node beyond the cap is still non-terminal — otherwise the watcher
    delivers a false 'done' while a deep branch is still running."""
    monkeypatch.setattr("src.host.orchestration.MAX_WORKFLOW_CHAIN_DEPTH", 3)
    s = _store()
    r = _human_root(s)
    _finish(s, r["id"], "done", result_summary="root")
    prev = r["id"]
    for i in range(1, 4):  # depths 1..3 — all done, all visible to the CTE
        rec = s.create(creator="scout", assignee="w", title=f"s{i}",
                       parent_task_id=prev)
        _finish(s, rec["id"], "done")
        prev = rec["id"]
    # depth 4 — still working, beyond the cap, so invisible to the CTE.
    deep = s.create(creator="scout", assignee="w", title="deep",
                    parent_task_id=prev)
    s.update_status(deep["id"], "working", actor="x")
    assert s.chain_terminal_verdict(r["id"]) is None


def test_verdict_chain_ending_exactly_at_cap_still_delivers(monkeypatch):
    """A chain that NATURALLY ends at the depth cap (no descendants beyond it)
    is NOT truncated — it must still deliver, not be silenced forever. Guards
    against a depth-heuristic that can't tell a cap-deep leaf from truncation."""
    monkeypatch.setattr("src.host.orchestration.MAX_WORKFLOW_CHAIN_DEPTH", 3)
    s = _store()
    r = _human_root(s)
    _finish(s, r["id"], "done", result_summary="root")
    prev = r["id"]
    for i in range(1, 4):  # depths 1..3 — the deepest (3) is a genuine leaf
        rec = s.create(creator="scout", assignee="w", title=f"s{i}",
                       parent_task_id=prev)
        _finish(s, rec["id"], "done", result_summary=f"leaf{i}")
        prev = rec["id"]
    assert s.chain_terminal_verdict(r["id"]) == ("done", "leaf3")


# ── Watcher: a recording deliver double ──────────────────────────


class _Deliver:
    def __init__(self, returns=True):
        self.calls: list[tuple] = []
        self._returns = returns

    def __call__(self, root, kind, summary):
        self.calls.append((root["id"], kind, summary))
        if callable(self._returns):
            return self._returns(len(self.calls))
        return self._returns


@pytest.mark.asyncio
async def test_settle_debounce_then_deliver_once():
    s = _store()
    r = _human_root(s)
    _finish(s, r["id"], "done", result_summary="r")
    clk = [1000.0]
    d = _Deliver()
    w = ChainWatcher(s, d, settle_s=10, clock=lambda: clk[0])

    await w.sweep_once()           # first observe terminal → start settle
    assert d.calls == []
    clk[0] += 5
    await w.sweep_once()           # within settle → still nothing
    assert d.calls == []
    clk[0] += 6                    # 11s > 10s settle
    await w.sweep_once()           # settled → deliver
    assert [c[1] for c in d.calls] == ["done"]
    await w.sweep_once()           # claimed → never again
    assert len(d.calls) == 1


@pytest.mark.asyncio
async def test_in_flight_handoff_resets_settle():
    """A successor hop appearing before settle must cancel a premature done."""
    s = _store()
    r = _human_root(s)
    _finish(s, r["id"], "done", result_summary="stage1")
    clk = [0.0]
    d = _Deliver()
    w = ChainWatcher(s, d, settle_s=10, clock=lambda: clk[0])

    await w.sweep_once()           # root looks terminal → settle armed
    clk[0] += 5
    # Hand-off lands: a non-terminal child appears.
    child = s.create(creator="scout", assignee="writer", title="draft",
                     parent_task_id=r["id"])
    s.update_status(child["id"], "working", actor="x")
    await w.sweep_once()           # chain no longer terminal → no deliver
    clk[0] += 20
    await w.sweep_once()           # still working → no deliver
    assert d.calls == []
    # Child finishes — chain terminal again, settle re-arms from scratch.
    s.update_status(child["id"], "done", actor="x", result_summary="final")
    await w.sweep_once()           # re-arm
    assert d.calls == []
    clk[0] += 11
    await w.sweep_once()           # settled → deliver exactly once
    assert [c[1] for c in d.calls] == ["done"]
    assert d.calls[0][2] == "final"


@pytest.mark.asyncio
async def test_failed_delivery_is_retried_not_lost():
    s = _store()
    r = _human_root(s)
    _finish(s, r["id"], "done", result_summary="r")
    clk = [0.0]
    # Fail the first delivery, succeed the second.
    d = _Deliver(returns=lambda n: n >= 2)
    w = ChainWatcher(s, d, settle_s=0, clock=lambda: clk[0])

    await w.sweep_once()           # arm settle (settle_s=0 still needs 2 passes)
    await w.sweep_once()           # settled → deliver #1 returns False
    await w.sweep_once()           # retry → deliver #2 returns True → claim
    assert len(d.calls) == 2
    await w.sweep_once()           # claimed → no more
    assert len(d.calls) == 2


@pytest.mark.asyncio
async def test_cancelled_chain_claimed_not_delivered():
    s = _store()
    r = _human_root(s)
    s.update_status(r["id"], "cancelled", actor="x")
    d = _Deliver()
    w = ChainWatcher(s, d, settle_s=0)

    await w.sweep_once()
    assert d.calls == []
    # Claimed, so it drops out of the watch set.
    assert s.list_watchable_human_roots(since=0.0) == []


@pytest.mark.asyncio
async def test_non_human_root_never_delivered():
    s = _store()
    # Worker-origin root (the kind a downgraded mid-chain create would yield).
    r = s.create(creator="a", assignee="b", title="internal",
                 origin={"kind": "agent", "channel": "", "user": "b"})
    _finish(s, r["id"], "done", result_summary="x")
    d = _Deliver()
    w = ChainWatcher(s, d, settle_s=0)
    await w.sweep_once()
    await w.sweep_once()
    assert d.calls == []


@pytest.mark.asyncio
async def test_poison_root_does_not_abort_sweep():
    """A store error deriving the verdict for ONE root must not starve the
    others — the terminal path is isolated per-root like stall/milestone."""
    s = _store()
    bad = _human_root(s, assignee="bad")    # created first → swept first
    good = _human_root(s, assignee="good")
    _finish(s, bad["id"], "done", result_summary="bad")
    _finish(s, good["id"], "done", result_summary="good")
    # The bad root's verdict raises every sweep; the good one is fine.
    real_verdict = s.chain_terminal_verdict

    def flaky(root_id):
        if root_id == bad["id"]:
            raise RuntimeError("boom")
        return real_verdict(root_id)

    s.chain_terminal_verdict = flaky
    d = _Deliver()
    w = ChainWatcher(s, d, settle_s=0)
    await w.sweep_once()   # arm settle for good (bad raises, isolated)
    await w.sweep_once()   # settled → good delivered despite bad raising
    assert [c[0] for c in d.calls] == [good["id"]]


# ── Store: stall state + claim (Phase 3a) ────────────────────────


def test_chain_stall_state_none_when_terminal():
    s = _store()
    r = _human_root(s)
    _finish(s, r["id"], "done")
    assert s.chain_stall_state(r["id"]) is None


def test_chain_stall_state_none_when_working():
    s = _store()
    r = _human_root(s)
    s.update_status(r["id"], "working", actor="x")
    # Actively progressing — not a stall, the lane watchdog owns hung working.
    assert s.chain_stall_state(r["id"]) is None


def test_chain_stall_state_ts_when_blocked():
    s = _store()
    r = _human_root(s)
    s.update_status(r["id"], "working", actor="x")
    s.update_status(r["id"], "blocked", actor="x", blocker_note="need creds")
    ts = s.chain_stall_state(r["id"])
    assert ts is not None and ts > 0


def test_chain_stall_state_ts_when_child_pending():
    # root done, successor created but never dispatched — chain is parked.
    s = _store()
    r = _human_root(s)
    _finish(s, r["id"], "done")
    s.create(creator="scout", assignee="writer", title="next",
             parent_task_id=r["id"])  # pending, nothing working
    assert s.chain_stall_state(r["id"]) is not None


def test_chain_stall_state_depth_cap_returns_none(monkeypatch):
    """A truncated chain isn't assessed for stall: the visible prefix reads as
    parked (all pending) but a deep ``working`` node beyond the cap is making
    progress invisibly — the guard suppresses a wrong nudge."""
    monkeypatch.setattr("src.host.orchestration.MAX_WORKFLOW_CHAIN_DEPTH", 3)
    s = _store()
    r = _human_root(s)  # pending
    prev = r["id"]
    for i in range(1, 4):  # depths 1..3 — pending, visible to the CTE
        rec = s.create(creator="scout", assignee="w", title=f"s{i}",
                       parent_task_id=prev)
        prev = rec["id"]
    # depth 4 — working, beyond the cap, so invisible. Without the guard the
    # visible prefix would read as a stall candidate and return a timestamp.
    deep = s.create(creator="scout", assignee="w", title="deep",
                    parent_task_id=prev)
    s.update_status(deep["id"], "working", actor="x")
    assert s.chain_stall_state(r["id"]) is None


def test_claim_chain_stall_exactly_once():
    s = _store()
    assert s.claim_chain_stall("root-x") is True
    assert s.claim_chain_stall("root-x") is False


# ── Watcher: stall nudge (Phase 3a) ──────────────────────────────


@pytest.mark.asyncio
async def test_watcher_nudges_parked_chain_once():
    s = _store()
    r = _human_root(s)
    s.update_status(r["id"], "working", actor="x")
    s.update_status(r["id"], "blocked", actor="x", blocker_note="stuck")
    d = _Deliver()
    # wall_clock far ahead so the parked chain reads as long-quiet.
    w = ChainWatcher(s, d, stall_after_s=600,
                     wall_clock=lambda: time.time() + 10_000)
    await w.sweep_once()
    assert [c[1] for c in d.calls] == ["stall"]
    await w.sweep_once()           # claimed → no second nudge
    assert len(d.calls) == 1


@pytest.mark.asyncio
async def test_watcher_no_nudge_while_working():
    s = _store()
    r = _human_root(s)
    s.update_status(r["id"], "working", actor="x")
    d = _Deliver()
    w = ChainWatcher(s, d, stall_after_s=0,
                     wall_clock=lambda: time.time() + 10_000)
    await w.sweep_once()
    assert d.calls == []           # working = progressing, never a stall


@pytest.mark.asyncio
async def test_watcher_no_nudge_before_threshold():
    s = _store()
    r = _human_root(s)
    s.update_status(r["id"], "working", actor="x")
    s.update_status(r["id"], "blocked", actor="x")
    d = _Deliver()
    # Parked, but only just — wall clock ≈ now, big threshold.
    w = ChainWatcher(s, d, stall_after_s=10_000, wall_clock=time.time)
    await w.sweep_once()
    assert d.calls == []


@pytest.mark.asyncio
async def test_stall_then_terminal_both_delivered():
    """A parked chain gets a stall nudge; when it later completes it still
    gets its terminal delivery (independent ledgers)."""
    s = _store()
    r = _human_root(s)
    s.update_status(r["id"], "working", actor="x")
    s.update_status(r["id"], "blocked", actor="x")
    d = _Deliver()
    w = ChainWatcher(s, d, settle_s=0, stall_after_s=600,
                     wall_clock=lambda: time.time() + 10_000)
    await w.sweep_once()           # stall nudge
    assert [c[1] for c in d.calls] == ["stall"]
    # Chain unsticks and finishes.
    s.update_status(r["id"], "working", actor="x")
    s.update_status(r["id"], "done", actor="x", result_summary="done now")
    await w.sweep_once()           # settle armed
    await w.sweep_once()           # terminal delivery
    assert [c[1] for c in d.calls] == ["stall", "done"]


# ── Milestone pings (Phase 2, opt-in) ────────────────────────────


def test_claim_milestone_ping_exactly_once():
    s = _store()
    assert s.claim_milestone_ping("t1") is True
    assert s.claim_milestone_ping("t1") is False


@pytest.mark.asyncio
async def test_milestone_ping_when_enabled():
    s = _store()
    r = _human_root(s)                       # root assigned to "scout"
    _finish(s, r["id"], "done", result_summary="stage1")
    child = s.create(creator="scout", assignee="writer", title="draft",
                     parent_task_id=r["id"])
    s.update_status(child["id"], "working", actor="writer")  # chain in-flight
    d = _Deliver()
    w = ChainWatcher(s, d, milestones_enabled=lambda: True)
    await w.sweep_once()
    # The recently-done root stage is pinged; the working child is not.
    assert [c[1] for c in d.calls] == ["milestone"]
    await w.sweep_once()                     # deduped — one ping per stage
    assert [c[1] for c in d.calls] == ["milestone"]


@pytest.mark.asyncio
async def test_no_milestone_when_disabled():
    s = _store()
    r = _human_root(s)
    _finish(s, r["id"], "done")
    child = s.create(creator="scout", assignee="writer", title="draft",
                     parent_task_id=r["id"])
    s.update_status(child["id"], "working", actor="writer")
    d = _Deliver()
    w = ChainWatcher(s, d, milestones_enabled=lambda: False)  # default off
    await w.sweep_once()
    assert d.calls == []


@pytest.mark.asyncio
async def test_milestone_skips_old_completions():
    s = _store()
    r = _human_root(s)
    _finish(s, r["id"], "done")
    child = s.create(creator="scout", assignee="writer", title="d",
                     parent_task_id=r["id"])
    s.update_status(child["id"], "working", actor="writer")
    # Backdate the root's completion well outside the recent window.
    with s._conn() as conn:
        conn.execute("UPDATE tasks SET updated_at=1.0 WHERE id=?", (r["id"],))
    d = _Deliver()
    w = ChainWatcher(s, d, milestones_enabled=lambda: True)
    await w.sweep_once()
    assert d.calls == []  # toggling on mid-pipeline doesn't retro-ping


@pytest.mark.asyncio
async def test_milestone_at_most_once_even_if_delivery_fails():
    # claim-then-deliver: a failing bell write does NOT cause a re-ping next
    # sweep (advisory at-most-once, deliberately NOT at-least-once).
    s = _store()
    r = _human_root(s)
    _finish(s, r["id"], "done")
    child = s.create(creator="scout", assignee="writer", title="d",
                     parent_task_id=r["id"])
    s.update_status(child["id"], "working", actor="writer")
    d = _Deliver(returns=False)              # delivery always "fails"
    w = ChainWatcher(s, d, milestones_enabled=lambda: True)
    await w.sweep_once()
    assert len(d.calls) == 1                 # delivered once (failed)
    await w.sweep_once()
    assert len(d.calls) == 1                 # claimed → not retried despite failure


@pytest.mark.asyncio
async def test_milestone_skipped_when_chain_already_terminal():
    # Fast-chain contract: a chain wholly terminal at sweep time yields the
    # terminal delivery only — intermediate stages get no milestone.
    s = _store()
    r = _human_root(s)
    _finish(s, r["id"], "done")
    child = s.create(creator="scout", assignee="writer", title="d",
                     parent_task_id=r["id"])
    _finish(s, child["id"], "done")          # both done before any sweep
    d = _Deliver()
    w = ChainWatcher(s, d, settle_s=0, milestones_enabled=lambda: True)
    await w.sweep_once()                      # arm terminal settle
    await w.sweep_once()                      # terminal delivery
    kinds = [c[1] for c in d.calls]
    assert "milestone" not in kinds
    assert kinds.count("done") == 1
