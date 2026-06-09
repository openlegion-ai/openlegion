"""Tests for the chain watcher (delegate-and-subscribe terminal delivery).

Covers the Tasks store helpers (chain_deliveries dedup, human-root listing,
whole-chain terminal verdict) and the ChainWatcher sweep semantics
(settle/debounce against the in-flight hand-off race, exactly-once delivery,
deliver-then-claim retry, cancelled handling, and human-origin gating).
"""

from __future__ import annotations

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
