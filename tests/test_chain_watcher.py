"""Tests for the chain watcher (delegate-and-subscribe terminal delivery).

Covers the Tasks store helpers (chain_deliveries dedup, human-root listing,
whole-chain terminal verdict) and the ChainWatcher sweep semantics
(settle/debounce against the in-flight hand-off race, exactly-once delivery,
deliver-then-claim retry, cancelled handling, and human-origin gating).
"""

from __future__ import annotations

import time

import pytest

from src.host.chain_watcher import BlockedTaskLadder, ChainWatcher
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
    others — the terminal path is isolated per-root like stall."""
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


# ── Blocked-task escalation ladder (plan §8 #22) ─────────────────
#
# Store-level ladder state + the BlockedTaskLadder sweep semantics.
# The ladder is INFLUENCE ONLY — a pinned test asserts the task row is
# byte-identical after a full climb to rung 4.

INTERVAL = 600.0
FALLBACK = 48 * 3600.0


def _blocked_task(
    store: Tasks, *, creator: str = "ops", assignee: str = "scout",
    team_id: str | None = None, note: str = "stuck on X",
    title: str = "do the thing",
) -> dict:
    t = store.create(
        creator=creator, assignee=assignee, title=title, team_id=team_id,
    )
    store.update_status(t["id"], "working", actor=assignee)
    store.update_status(t["id"], "blocked", actor=assignee, blocker_note=note)
    return store.get(t["id"])


class _Sender:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def __call__(self, agent, message):
        self.calls.append((agent, message))


class _HelpRegistry:
    def __init__(self):
        self.records: list[dict] = []

    def record(self, kind, agent_id, payload):
        self.records.append(
            {"kind": kind, "agent_id": agent_id, "payload": payload},
        )
        return f"req-{len(self.records)}"


class _Audit:
    def __init__(self):
        self.rows: list[tuple[str, int, dict]] = []

    def __call__(self, task_id, rung, detail):
        self.rows.append((task_id, rung, detail))


def _ladder(store: Tasks, clk: list, **kw) -> BlockedTaskLadder:
    defaults: dict = dict(
        interval_s=INTERVAL,
        fallback_s=FALLBACK,
        wall_clock=lambda: clk[0],
    )
    defaults.update(kw)
    return BlockedTaskLadder(store, **defaults)


# ── Store: ladder state / claims ─────────────────────────────────


def test_ladder_observe_and_climb_cas():
    s = _store()
    t = _blocked_task(s)
    assert s.ladder_state(t["id"]) is None
    st = s.ladder_observe_blocked(t["id"], now=100.0)
    assert st == {
        "task_id": t["id"], "blocked_since": 100.0,
        "rung": 0, "last_climb_at": 100.0,
    }
    # Re-observe is idempotent (INSERT OR IGNORE) — state untouched.
    assert s.ladder_observe_blocked(t["id"], now=999.0)["blocked_since"] == 100.0
    # CAS climb: the right from_rung wins once; a stale replay loses.
    assert s.ladder_climb(t["id"], 0, 1, now=200.0) is True
    assert s.ladder_climb(t["id"], 0, 1, now=200.0) is False
    st = s.ladder_state(t["id"])
    assert (st["rung"], st["last_climb_at"]) == (1, 200.0)


def test_ladder_reset_clears_only_unblocked():
    s = _store()
    still = _blocked_task(s)
    gone = _blocked_task(s, assignee="other")
    s.ladder_observe_blocked(still["id"], now=1.0)
    s.ladder_observe_blocked(gone["id"], now=1.0)
    s.update_status(gone["id"], "working", actor="x")   # unblocked
    assert s.ladder_reset_unblocked() == 1
    assert s.ladder_state(still["id"]) is not None
    assert s.ladder_state(gone["id"]) is None


def test_claim_blocked_human_notice_exactly_once():
    s = _store()
    assert s.claim_blocked_human_notice("task-1") is True
    assert s.claim_blocked_human_notice("task-1") is False


def test_list_blocked_returns_only_blocked():
    s = _store()
    b = _blocked_task(s)
    w = s.create(creator="ops", assignee="scout", title="working one")
    s.update_status(w["id"], "working", actor="x")
    assert [t["id"] for t in s.list_blocked()] == [b["id"]]


def test_escalated_blocked_for_team_filters_rung_team_and_status():
    s = _store()
    hot = _blocked_task(s, team_id="alpha")            # rung 3 — counts
    cold = _blocked_task(s, team_id="alpha")           # rung 1 — too low
    other = _blocked_task(s, team_id="beta")           # rung 3, other team
    fixed = _blocked_task(s, team_id="alpha")          # rung 3 but unblocked
    for task, rung in ((hot, 3), (cold, 1), (other, 3), (fixed, 3)):
        s.ladder_observe_blocked(task["id"], now=1.0)
        for step in range(1, rung + 1):
            s.ladder_climb(task["id"], step - 1, step, now=float(step))
    s.update_status(fixed["id"], "working", actor="x")
    count, sample = s.escalated_blocked_for_team("alpha")
    assert (count, sample) == (1, [hot["id"]])
    # Rung 4 still counts as "on the lead's plate" (>= min_rung).
    s.ladder_climb(hot["id"], 3, 4, now=9.0)
    assert s.escalated_blocked_for_team("alpha")[0] == 1


# ── Ladder: rung progression ─────────────────────────────────────


@pytest.mark.asyncio
async def test_ladder_rung_progression_at_intervals():
    s = _store()
    task = _blocked_task(s, team_id="alpha")
    clk = [10_000.0]
    assignee, creator = _Sender(), _Sender()
    hp, audit = _HelpRegistry(), _Audit()
    lad = _ladder(
        s, clk, deliver_assignee=assignee, deliver_creator=creator,
        lead_of_fn=lambda team: "lead-1", help_requests=hp, audit_fn=audit,
    )

    await lad.sweep_once()          # first observation → rung 0, no sends
    assert s.ladder_state(task["id"])["rung"] == 0
    assert assignee.calls == [] and creator.calls == []

    clk[0] += INTERVAL              # rung 1 due → assignee re-driven
    await lad.sweep_once()
    assert s.ladder_state(task["id"])["rung"] == 1
    assert [a for a, _ in assignee.calls] == ["scout"]
    assert creator.calls == []

    clk[0] += INTERVAL              # rung 2 → creator followup
    await lad.sweep_once()
    assert s.ladder_state(task["id"])["rung"] == 2
    assert [a for a, _ in creator.calls] == ["ops"]

    clk[0] += INTERVAL              # rung 3 → lead plate, NO message
    await lad.sweep_once()
    assert s.ladder_state(task["id"])["rung"] == 3
    assert len(assignee.calls) == 1 and len(creator.calls) == 1
    assert s.escalated_blocked_for_team("alpha") == (1, [task["id"]])

    clk[0] += INTERVAL              # parked at rung 3 — no interval climb
    await lad.sweep_once()
    assert s.ladder_state(task["id"])["rung"] == 3
    assert hp.records == []
    # One audit row per climb, carrying the rung + task id.
    assert [(tid, rung) for tid, rung, _ in audit.rows] == [
        (task["id"], 1), (task["id"], 2), (task["id"], 3),
    ]


@pytest.mark.asyncio
async def test_ladder_climbs_one_rung_per_sweep():
    """A long-overdue task climbs one rung per sweep — never skips
    straight past a rung's nudge (rung 4's own triggers excepted)."""
    s = _store()
    task = _blocked_task(s)
    clk = [0.0]
    assignee = _Sender()
    lad = _ladder(s, clk, deliver_assignee=assignee)
    await lad.sweep_once()          # observe
    clk[0] += INTERVAL * 3          # three intervals overdue
    await lad.sweep_once()
    assert s.ladder_state(task["id"])["rung"] == 1
    assert len(assignee.calls) == 1


@pytest.mark.asyncio
async def test_rung1_message_carries_blocker_note_via_deliver_seam():
    s = _store()
    task = _blocked_task(s, note="waiting on prod schema decision")
    clk = [0.0]
    assignee = _Sender()
    lad = _ladder(s, clk, deliver_assignee=assignee)
    await lad.sweep_once()
    clk[0] += INTERVAL
    await lad.sweep_once()
    [(agent, message)] = assignee.calls
    assert agent == "scout"
    assert task["id"] in message
    assert "waiting on prod schema decision" in message
    assert "update the task status" in message


@pytest.mark.asyncio
async def test_rung2_skips_creator_equals_assignee():
    s = _store()
    task = _blocked_task(s, creator="scout", assignee="scout")
    clk = [0.0]
    creator = _Sender()
    audit = _Audit()
    lad = _ladder(s, clk, deliver_creator=creator, audit_fn=audit)
    await lad.sweep_once()
    clk[0] += INTERVAL
    await lad.sweep_once()          # rung 1
    clk[0] += INTERVAL
    await lad.sweep_once()          # rung 2 — climb happens, message skipped
    assert s.ladder_state(task["id"])["rung"] == 2
    assert creator.calls == []
    assert audit.rows[-1] == (task["id"], 2, {"skipped": "creator_is_assignee"})


@pytest.mark.asyncio
async def test_rung2_skips_operator_creator():
    """The operator path is rung 4's job — rung 2 never pings the human's
    agent about its own hand-offs."""
    s = _store()
    task = _blocked_task(s, creator="operator")
    clk = [0.0]
    creator = _Sender()
    audit = _Audit()
    lad = _ladder(s, clk, deliver_creator=creator, audit_fn=audit)
    await lad.sweep_once()
    clk[0] += INTERVAL
    await lad.sweep_once()
    clk[0] += INTERVAL
    await lad.sweep_once()
    assert s.ladder_state(task["id"])["rung"] == 2
    assert creator.calls == []
    assert audit.rows[-1] == (task["id"], 2, {"skipped": "creator_is_operator"})


@pytest.mark.asyncio
async def test_rung3_teamless_climbs_straight_past():
    """Teamless (or leaderless) tasks climb rung 3 on schedule but nothing
    surfaces them — no lead plate exists; rung 4 waits for its own
    triggers (budget / max age)."""
    s = _store()
    task = _blocked_task(s, team_id=None)
    clk = [0.0]
    audit = _Audit()
    hp = _HelpRegistry()
    lad = _ladder(s, clk, lead_of_fn=lambda team: None, audit_fn=audit, help_requests=hp)
    await lad.sweep_once()
    for _ in range(3):
        clk[0] += INTERVAL
        await lad.sweep_once()
    assert s.ladder_state(task["id"])["rung"] == 3
    assert audit.rows[-1] == (task["id"], 3, {"skipped": "no_lead"})
    # No team ⇒ no lead-plate probe surface anywhere; rung 4 not yet due.
    assert hp.records == []


# ── Ladder: rung 4 (human) ───────────────────────────────────────


@pytest.mark.parametrize("note", [
    "Budget exceeded: $2.00/$2.00 daily, $10.00/$50.00 monthly",
    "Team budget exceeded for team 'alpha': $9.00/$9.00 daily",
    "Coordination budget exceeded: $2.00/$2.00 daily",
])
@pytest.mark.asyncio
async def test_rung4_budget_fast_path_fires_immediately(note):
    """The verified budget-error family escalates straight to the human
    on the FIRST sweep — no interval wait, from rung 0."""
    s = _store()
    task = _blocked_task(s, note=note)
    clk = [0.0]
    assignee = _Sender()
    hp, audit = _HelpRegistry(), _Audit()
    lad = _ladder(
        s, clk, deliver_assignee=assignee, help_requests=hp, audit_fn=audit,
    )
    await lad.sweep_once()
    assert s.ladder_state(task["id"])["rung"] == 4
    assert assignee.calls == []     # rungs 1-3 never fired
    [rec] = hp.records
    assert rec["kind"] == "blocked_task_escalation"
    assert rec["agent_id"] == "scout"
    assert rec["payload"]["name"] == task["id"]
    assert "budget" in rec["payload"]["description"].lower()
    assert audit.rows == [
        (task["id"], 4, {"reason": "budget_exhausted", "help_request_filed": True}),
    ]


@pytest.mark.asyncio
async def test_rung4_max_age_fallback_fires():
    s = _store()
    task = _blocked_task(s, note="waiting on upstream API contract")
    clk = [50_000.0]
    hp, audit = _HelpRegistry(), _Audit()
    lad = _ladder(s, clk, help_requests=hp, audit_fn=audit)
    await lad.sweep_once()          # observe at rung 0
    clk[0] += FALLBACK              # 48h blocked → human, from any rung
    await lad.sweep_once()
    assert s.ladder_state(task["id"])["rung"] == 4
    [rec] = hp.records
    assert rec["kind"] == "blocked_task_escalation"
    assert audit.rows[-1] == (
        task["id"], 4, {"reason": "max_age", "help_request_filed": True},
    )


@pytest.mark.asyncio
async def test_rung4_fires_once_across_sweeps_and_restart():
    s = _store()
    task = _blocked_task(s, note="Budget exceeded: $2.00/$2.00 daily")
    clk = [0.0]
    hp = _HelpRegistry()
    lad = _ladder(s, clk, help_requests=hp)
    await lad.sweep_once()
    await lad.sweep_once()          # rung already 4 → no re-file
    assert len(hp.records) == 1
    # "Restart": a fresh ladder over the same durable store — the rung-4
    # state AND the human-notice claim both survive.
    lad2 = _ladder(s, clk, help_requests=hp)
    await lad2.sweep_once()
    assert len(hp.records) == 1
    assert s.ladder_state(task["id"])["rung"] == 4


@pytest.mark.asyncio
async def test_unblock_reblock_resets_rungs_but_never_refiles_human():
    s = _store()
    task = _blocked_task(s, note="Budget exceeded: $2.00/$2.00 daily")
    clk = [0.0]
    assignee = _Sender()
    hp = _HelpRegistry()
    lad = _ladder(s, clk, deliver_assignee=assignee, help_requests=hp)
    await lad.sweep_once()          # budget fast path → rung 4, one entry
    assert len(hp.records) == 1
    # Unblock, sweep (state clears), re-block with an ordinary note.
    s.update_status(task["id"], "working", actor="scout")
    await lad.sweep_once()
    assert s.ladder_state(task["id"]) is None
    s.update_status(task["id"], "blocked", actor="scout", blocker_note="stuck again")
    await lad.sweep_once()          # fresh episode → rung 0
    assert s.ladder_state(task["id"])["rung"] == 0
    clk[0] += INTERVAL
    await lad.sweep_once()          # the ladder re-runs: assignee re-driven
    assert s.ladder_state(task["id"])["rung"] == 1
    assert len(assignee.calls) == 1
    # But the human entry is at-most-once PER TASK EVER: age it to the
    # fallback — rung 4 climbs again, files nothing.
    clk[0] += FALLBACK
    await lad.sweep_once()
    assert s.ladder_state(task["id"])["rung"] == 4
    assert len(hp.records) == 1


@pytest.mark.asyncio
async def test_ladder_never_mutates_task_rows():
    """Influence only — a full climb to rung 4 leaves the task row
    byte-identical (status, assignee, blocker_note, updated_at: only
    real actors move tasks)."""
    s = _store()
    task = _blocked_task(s, team_id="alpha")
    before = s.get(task["id"])
    clk = [0.0]
    hp = _HelpRegistry()
    lad = _ladder(
        s, clk, deliver_assignee=_Sender(), deliver_creator=_Sender(),
        lead_of_fn=lambda team: "lead-1", help_requests=hp,
    )
    await lad.sweep_once()
    for _ in range(3):
        clk[0] += INTERVAL
        await lad.sweep_once()
    clk[0] += FALLBACK
    await lad.sweep_once()          # rung 4 via max age
    assert s.ladder_state(task["id"])["rung"] == 4
    assert len(hp.records) == 1
    assert s.get(task["id"]) == before


@pytest.mark.asyncio
async def test_interval_zero_disables_whole_ladder():
    """``ladder_rung_interval_minutes=0`` turns the ENTIRE ladder off —
    no state rows, no nudges, and no rung-4 fast path (B4-style 0-valid
    kill switch, pinned here)."""
    s = _store()
    task = _blocked_task(s, note="Budget exceeded: $2.00/$2.00 daily")
    clk = [0.0]
    assignee = _Sender()
    hp = _HelpRegistry()
    lad = _ladder(
        s, clk, interval_s=0.0,
        deliver_assignee=assignee, help_requests=hp,
    )
    for _ in range(3):
        await lad.sweep_once()
        clk[0] += FALLBACK
    assert s.ladder_state(task["id"]) is None
    assert assignee.calls == []
    assert hp.records == []


@pytest.mark.asyncio
async def test_ladder_send_failure_does_not_stall_the_climb():
    """A raising delivery seam is logged and isolated — the rung climb
    (already claimed) stands and later rungs still fire."""
    s = _store()
    task = _blocked_task(s)
    clk = [0.0]

    def _boom(agent, message):
        raise RuntimeError("transport down")

    creator = _Sender()
    lad = _ladder(s, clk, deliver_assignee=_boom, deliver_creator=creator)
    await lad.sweep_once()
    clk[0] += INTERVAL
    await lad.sweep_once()          # rung 1 send raises — isolated
    assert s.ladder_state(task["id"])["rung"] == 1
    clk[0] += INTERVAL
    await lad.sweep_once()          # rung 2 proceeds normally
    assert [a for a, _ in creator.calls] == ["ops"]


# ── Ladder: chain-watcher piggyback ──────────────────────────────


@pytest.mark.asyncio
async def test_chain_watcher_runs_ladder_on_its_cadence():
    s = _store()
    task = _blocked_task(s)
    clk = [0.0]
    lad = _ladder(s, clk)
    w = ChainWatcher(s, _Deliver(), settle_s=0, ladder=lad)
    await w.sweep_once()
    assert s.ladder_state(task["id"]) is not None   # ladder swept


@pytest.mark.asyncio
async def test_ladder_failure_isolated_from_terminal_delivery():
    """A ladder blow-up must never starve the watcher's terminal-outcome
    guarantee."""
    s = _store()
    r = _human_root(s)
    _finish(s, r["id"], "done", result_summary="r")

    class _BoomLadder:
        async def sweep_once(self):
            raise RuntimeError("ladder boom")

    d = _Deliver()
    w = ChainWatcher(s, d, settle_s=0, ladder=_BoomLadder())
    await w.sweep_once()            # arm settle (ladder raises, isolated)
    await w.sweep_once()            # settled → deliver
    assert [c[1] for c in d.calls] == ["done"]
