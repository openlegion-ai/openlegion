"""Regression tests for M9 + C5 + C6 (Phase 0-5 integration review).

The blocked-task ladder files ONE durable Needs-you ``help_requests`` row per
task ever (rung 4). Three seam defects around that row:

M9 — the 14-day ``reap_old`` age-swept the ``blocked_task_escalation`` row while
the eternal ``blocked_human_notices`` claim guaranteed it could never be re-filed
— evaporating a still-stuck task's sole human surface.

C5 — unblock/complete cleared the ladder state but not the open ``help_requests``
row, so the authoritative Needs-you feed kept asserting "human action required".

C6 — rung 4 consumed the forever-claim BEFORE inserting the row; a transient
``help_requests.db`` failure burned the claim with no row filed, permanently
denying the task its only human escalation.
"""

from __future__ import annotations

import pytest

from src.host.chain_watcher import BlockedTaskLadder
from src.host.help_requests import HelpRequests
from src.host.orchestration import Tasks

INTERVAL = 600.0
FALLBACK = 48 * 3600.0
_BUDGET_NOTE = "Budget exceeded: $2.00/$2.00 daily"


def _store() -> Tasks:
    return Tasks(db_path=":memory:")


def _blocked_task(store: Tasks, *, note: str = _BUDGET_NOTE) -> dict:
    t = store.create(creator="ops", assignee="scout", title="do the thing")
    store.update_status(t["id"], "working", actor="scout")
    store.update_status(t["id"], "blocked", actor="scout", blocker_note=note)
    return store.get(t["id"])


def _ladder(store: Tasks, clk: list, **kw) -> BlockedTaskLadder:
    defaults: dict = dict(interval_s=INTERVAL, fallback_s=FALLBACK, wall_clock=lambda: clk[0])
    defaults.update(kw)
    return BlockedTaskLadder(store, **defaults)


# ── M9: age reap exempts escalation rows ─────────────────────────────


def test_reap_old_exempts_blocked_task_escalation():
    hp = HelpRequests(":memory:")
    hp.record("blocked_task_escalation", "scout", {"name": "task-1", "description": "d"})
    hp.record("credential_request", "scout", {"name": "GITHUB_TOKEN"})
    # Force both rows well past the reap cutoff.
    with hp._conn() as c:
        c.execute("UPDATE help_requests SET created_at = 0")
    reaped = hp.reap_old(max_age_sec=1.0)
    assert reaped == 1  # only the credential_request row
    open_kinds = {r["kind"] for r in hp.list_open()}
    assert open_kinds == {"blocked_task_escalation"}
    hp.close()


# ── C5: resolve the Needs-you row when the task leaves blocked ────────


@pytest.mark.asyncio
async def test_unblock_resolves_open_escalation_row():
    s = _store()
    hp = HelpRequests(":memory:")
    task = _blocked_task(s)
    clk = [0.0]
    lad = _ladder(s, clk, help_requests=hp)

    await lad.sweep_once()  # budget fast path → rung 4 → real row filed
    assert task["id"] in hp.open_escalation_task_ids()
    assert {r["kind"] for r in hp.list_open()} == {"blocked_task_escalation"}

    # Task leaves `blocked` → the next sweep reconciles the stale row away.
    s.update_status(task["id"], "working", actor="scout")
    await lad.sweep_once()
    assert hp.list_open() == []
    # ...but the once-ever claim is deliberately preserved (no re-file on re-block).
    assert s.blocked_human_notice_claimed(task["id"]) is True
    hp.close()


# ── C6: record BEFORE consuming the forever-claim ────────────────────


class _FlakyHelp:
    """A help-request store whose first ``record`` raises (transient DB
    failure), then succeeds — plus the reconcile read surface."""

    def __init__(self):
        self.fail_next = True
        self.records: list[tuple] = []

    def record(self, kind, agent_id, payload):
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("help_requests.db momentarily locked")
        self.records.append((kind, agent_id, payload))
        return f"req-{len(self.records)}"

    def open_escalation_task_ids(self):
        return set()

    def resolve_for_task(self, task_id):
        return 0


@pytest.mark.asyncio
async def test_record_failure_preserves_claim_for_retry():
    s = _store()
    task = _blocked_task(s)
    clk = [0.0]
    hp = _FlakyHelp()
    lad = _ladder(s, clk, help_requests=hp)

    # First sweep: climb to rung 4, then record() FAILS. The forever-claim
    # must NOT be consumed (pre-fix it was consumed first → burned forever).
    await lad.sweep_once()
    assert hp.records == []
    assert s.blocked_human_notice_claimed(task["id"]) is False

    # Re-block → a fresh episode retries and now files successfully.
    s.update_status(task["id"], "working", actor="scout")
    await lad.sweep_once()
    s.update_status(task["id"], "blocked", actor="scout", blocker_note=_BUDGET_NOTE)
    await lad.sweep_once()
    assert len(hp.records) == 1
    assert s.blocked_human_notice_claimed(task["id"]) is True
