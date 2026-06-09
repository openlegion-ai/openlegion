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
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self._tasks = tasks_store
        self._deliver = deliver
        self._settle_s = settle_s
        self._sweep_s = sweep_s
        self._watch_window_s = watch_window_s
        self._stall_after_s = stall_after_s
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

        live_ids: set[str] = set()
        for root in roots:
            root_id = root.get("id")
            if not root_id:
                continue
            live_ids.add(root_id)

            verdict = self._tasks.chain_terminal_verdict(root_id)
            if verdict is None:
                # Still active (or a new hop just appeared) — reset settle.
                self._settling.pop(root_id, None)
                await self._maybe_nudge_stall(root, root_id)
                continue

            kind, summary = verdict
            if kind == "cancelled":
                # Manual cancellation is not a surprise to surface. Claim it
                # so the chain stops being re-scanned, but deliver nothing.
                self._tasks.claim_chain_delivery(root_id, "cancelled")
                self._settling.pop(root_id, None)
                continue

            first = self._settling.get(root_id)
            now_mono = self._clock()
            if first is None:
                self._settling[root_id] = now_mono
                continue
            if now_mono - first < self._settle_s:
                continue

            # Settled and still terminal — deliver, then claim on success.
            delivered = await self._run_deliver(root, kind, summary)
            if delivered:
                self._tasks.claim_chain_delivery(root_id, kind)
                self._settling.pop(root_id, None)
            # If delivery failed, leave the settle timestamp in place so the
            # next sweep retries immediately (no silent loss).

        # Bound memory: forget settle timers for roots no longer watchable.
        for stale in [r for r in self._settling if r not in live_ids]:
            self._settling.pop(stale, None)

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
