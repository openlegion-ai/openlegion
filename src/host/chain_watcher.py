"""Chain watcher — delegate-and-subscribe terminal delivery.

The operator hands user work off to a team and *releases* its turn instead
of block-watching a multi-hop pipeline (which dies at the browser's 120s
idle abort and strands the operator). This watcher makes "watching" a
durable, system-side behavior: it periodically sweeps user-originated task
chains and pushes exactly ONE terminal outcome (done / failed) to the human
who started the chain.

Design (see docs/plans/2026-06-10-operator-delegate-and-subscribe.md):

- **Periodic sweep, not an event listener.** A sweep re-derives everything
  from the durable tasks table on each pass, so it is inherently
  restart-safe and free of cross-thread event-affinity concerns. The
  ``chain_deliveries`` table is the exactly-once ledger.
- **Settle/debounce.** A root reaching ``done`` a beat before its successor
  task row exists reads as "whole chain terminal" momentarily. We require a
  chain to stay terminal for ``settle_s`` before delivering, so an in-flight
  hand-off resets the timer instead of triggering a premature "done".
- **Deliver-then-claim.** The durable surface (the dashboard bell) is
  written first; only on success is the delivery claimed. A chain is thus
  never silently lost (it retries until the durable write succeeds); the
  worst case is a rare duplicate if the process dies between write and
  claim, which is strictly preferable to silence.
- **Security.** Targeting comes from the chain ROOT's first-party human
  origin only (the store filters ``origin_kind='human'`` roots); mid-chain
  worker origins — which ``_validated_origin`` downgrades and the L9 binding
  treats as forgeable — are never used. No origin-trust boundary is touched.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable

from src.shared.utils import setup_logging

logger = setup_logging("host.chain_watcher")

# A chain must look terminal continuously for this long before we deliver —
# covers the in-flight hand-off window (root `done` before the next hop's
# row exists).
_DEFAULT_SETTLE_S = 30.0
# How often to sweep watchable chains.
_DEFAULT_SWEEP_S = 10.0
# Only chains created within this window are swept; abandoned, never-terminal
# chains age out of the scan instead of being re-examined forever.
_DEFAULT_WATCH_WINDOW_S = 7 * 24 * 3600.0
# A chain parked in a waiting state (blocked / pending / accepted, nothing
# working) with no progress for this long earns one stall nudge to the user.
_DEFAULT_STALL_AFTER_S = 600.0
# Opt-in milestone pings fire only for stages that completed within this
# window — so toggling the feature on mid-pipeline doesn't retro-ping every
# already-done stage, and a sweep only ever pings a just-finished one.
_MILESTONE_RECENT_S = 120.0


class ChainWatcher:
    """Sweeps user chains and delivers one terminal outcome each.

    ``deliver(root: dict, kind: str, summary: str) -> bool`` must write the
    durable user-facing surface and return truthy on success. ``kind`` is one
    of ``done`` / ``failed`` (terminal outcomes) or ``stall`` (a parked-chain
    nudge). For terminal outcomes the watcher claims only on a truthy return
    (deliver-then-claim, no silent loss); the stall nudge is advisory and uses
    claim-then-deliver (at-most-once — the terminal delivery is the real
    guarantee). It may be sync or async.
    """

    def __init__(
        self,
        tasks_store: Any,
        deliver: Callable[[dict, str, str], Any],
        *,
        settle_s: float = _DEFAULT_SETTLE_S,
        sweep_s: float = _DEFAULT_SWEEP_S,
        watch_window_s: float = _DEFAULT_WATCH_WINDOW_S,
        stall_after_s: float = _DEFAULT_STALL_AFTER_S,
        milestones_enabled: Callable[[], bool] = lambda: False,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self._tasks = tasks_store
        self._deliver = deliver
        self._settle_s = settle_s
        self._sweep_s = sweep_s
        self._watch_window_s = watch_window_s
        self._stall_after_s = stall_after_s
        # Read each sweep — an opt-in toggle (default off); only when on do we
        # do the extra per-stage snapshot + milestone pings.
        self._milestones_enabled = milestones_enabled
        self._clock = clock          # monotonic — for the terminal settle debounce
        self._wall_clock = wall_clock  # wall-clock — for stall age vs stored updated_at
        # root_task_id -> monotonic timestamp first observed terminal.
        self._settling: dict[str, float] = {}
        self._running = False

    def stop(self) -> None:
        self._running = False

    async def start(self) -> None:
        self._running = True
        logger.info(
            "chain watcher started (settle=%ss sweep=%ss window=%ss)",
            self._settle_s, self._sweep_s, self._watch_window_s,
        )
        while self._running:
            try:
                await self.sweep_once()
            except Exception as e:
                logger.warning("chain watcher sweep failed: %s", e)
            await asyncio.sleep(self._sweep_s)

    async def sweep_once(self) -> None:
        """One pass: deliver terminal outcomes for settled human chains."""
        since = time.time() - self._watch_window_s
        try:
            roots = self._tasks.list_watchable_human_roots(since=since)
        except Exception as e:
            logger.warning("chain watcher: listing roots failed: %s", e)
            return

        # Read the opt-in milestone toggle ONCE per sweep, not per chain —
        # otherwise the default-off common case re-reads + parses
        # config/settings.json for every in-flight chain on every sweep.
        try:
            milestones_on = bool(self._milestones_enabled())
        except Exception:
            milestones_on = False

        live_ids: set[str] = set()
        for root in roots:
            root_id = root.get("id")
            if not root_id:
                continue
            live_ids.add(root_id)

            # Per-root isolation: a failure deriving the verdict or claiming a
            # delivery for ONE root must not abort the rest of the sweep (the
            # stall/milestone helpers already self-isolate; this guards the
            # terminal path the same way). A poison root self-heals next sweep.
            try:
                await self._process_root(root, root_id, milestones_on)
            except Exception as e:
                logger.warning(
                    "chain watcher: processing root %s failed: %s", root_id, e,
                )

        # Bound memory: forget settle timers for roots no longer watchable.
        for stale in [r for r in self._settling if r not in live_ids]:
            self._settling.pop(stale, None)

    async def _process_root(
        self, root: dict, root_id: str, milestones_on: bool,
    ) -> None:
        """Deliver/settle/nudge a single watchable root. Raises on store
        errors so :meth:`sweep_once` can isolate one bad root from the rest."""
        verdict = self._tasks.chain_terminal_verdict(root_id)
        if verdict is None:
            # Still active (or a new hop just appeared) — reset settle.
            self._settling.pop(root_id, None)
            await self._maybe_nudge_stall(root, root_id)
            if milestones_on:
                await self._maybe_ping_milestones(root, root_id)
            return

        kind, summary = verdict
        if kind == "cancelled":
            # Manual cancellation is not a surprise to surface. Claim it
            # so the chain stops being re-scanned, but deliver nothing.
            self._tasks.claim_chain_delivery(root_id, "cancelled")
            self._settling.pop(root_id, None)
            return

        first = self._settling.get(root_id)
        now_mono = self._clock()
        if first is None:
            self._settling[root_id] = now_mono
            return
        if now_mono - first < self._settle_s:
            return

        # Settled and still terminal — deliver, then claim on success.
        delivered = await self._run_deliver(root, kind, summary)
        if delivered:
            self._tasks.claim_chain_delivery(root_id, kind)
            self._settling.pop(root_id, None)
        # If delivery failed, leave the settle timestamp in place so the
        # next sweep retries immediately (no silent loss).

    async def _maybe_nudge_stall(self, root: dict, root_id: str) -> None:
        """Nudge the user once if a non-terminal chain is parked + quiet.

        ``chain_stall_state`` returns the last-progress timestamp only when the
        chain is stuck in a waiting state (nothing ``working``); ``None`` means
        it's progressing or terminal, so there's nothing to nudge about.
        Claim-then-deliver (advisory, at-most-once).
        """
        try:
            last_progress = self._tasks.chain_stall_state(root_id)
        except Exception as e:
            logger.warning("chain watcher: stall check failed for %s: %s", root_id, e)
            return
        if last_progress is None:
            return
        if self._wall_clock() - last_progress <= self._stall_after_s:
            return
        if self._tasks.claim_chain_stall(root_id):
            await self._run_deliver(root, "stall", "")

    async def _maybe_ping_milestones(self, root: dict, root_id: str) -> None:
        """Ping the user once per *recently-completed* stage of an in-flight
        chain (play-by-play progress). Only called when the opt-in toggle is on
        (checked once per sweep by the caller), so the extra per-stage snapshot
        runs only then. Claim-then-deliver (advisory, at-most-once per stage).

        Contract: milestones fire only while the chain is non-terminal, so the
        FINAL stage is never a milestone (it's the terminal delivery), and a
        chain that completes *between* sweeps (e.g. two fast stages inside one
        ~10s window) surfaces only as the terminal outcome — intermediate pings
        for such a chain may be skipped. Acceptable for an advisory feature; the
        guaranteed result/failure/stall delivery is unaffected."""
        try:
            snap = self._tasks.workflow_snapshot(root_id)
        except Exception as e:
            logger.warning(
                "chain watcher: milestone snapshot failed for %s: %s",
                root_id, e,
            )
            return
        if not snap:
            return
        for st in snap.get("stages", []):
            if st.get("status") != "done":
                continue
            # Only just-finished stages — flipping the toggle on mid-pipeline
            # must not retro-ping every already-done stage.
            if (st.get("age_in_state_seconds") or 0) > _MILESTONE_RECENT_S:
                continue
            tid = st.get("task_id")
            if not tid:
                continue
            if self._tasks.claim_milestone_ping(tid):
                assignee = st.get("assignee") or "an agent"
                stage_title = (st.get("title") or "").strip()
                msg = (
                    f"{assignee} finished: {stage_title}"
                    if stage_title else f"{assignee} finished a stage"
                )
                await self._run_deliver(root, "milestone", msg)

    async def _run_deliver(self, root: dict, kind: str, summary: str) -> bool:
        try:
            result = self._deliver(root, kind, summary)
            if asyncio.iscoroutine(result):
                result = await result
            return bool(result)
        except Exception as e:
            logger.warning(
                "chain watcher delivery raised for root %s: %s",
                root.get("id"), e,
            )
            return False
