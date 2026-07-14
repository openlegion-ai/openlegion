"""Chain watcher — delegate-and-subscribe terminal delivery.

The operator hands user work off to a team and *releases* its turn instead
of block-watching a multi-hop pipeline (which dies at the browser's 120s
idle abort and strands the operator). This watcher makes "watching" a
durable, system-side behavior: it periodically sweeps user-originated task
chains and pushes exactly ONE terminal outcome (done / failed) to the human
who started the chain.

Design (see docs/plans/archive/2026-06-10-operator-delegate-and-subscribe.md):

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

The watcher also hosts the **blocked-task escalation ladder** (plan §8 #22,
:class:`BlockedTaskLadder`) — a sibling sweep on the same cadence that climbs
``blocked`` tasks through influence-only rungs (assignee re-drive → creator
→ the lead's plate → ONE durable human Needs-you entry). Same persistence
posture: durable claim tables in the tasks store, at-most-once per rung.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any, Callable

from src.shared.utils import sanitize_for_prompt, setup_logging

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

# Rung-4 budget fast path (plan §8 #22). The verified in-tree budget-
# exhaustion error family the LLM proxy produces (src/host/credentials.py):
# "Budget exceeded: …", "Team budget exceeded for team …", and
# "Coordination budget exceeded: …" — all share this stem, matched
# case-insensitively against the (already-normalized) ``blocker_note``.
# ``cred:``-style blocker codes have NO backend producer (recon-verified,
# plan §8 #22) and are deliberately NOT matched; credential needs already
# reach the Needs-you registry via ``credential_request`` events.
_BUDGET_BLOCKER_RE = re.compile(r"budget exceeded", re.IGNORECASE)


def _is_budget_blocker(note: str) -> bool:
    return bool(note and _BUDGET_BLOCKER_RE.search(note))


class BlockedTaskLadder:
    """Blocked-task escalation ladder (plan §8 #22) — INFLUENCE ONLY.

    A periodic sweep over ``status='blocked'`` tasks that climbs each one
    through four rungs, one per ``ladder_rung_interval_minutes`` of blocked
    time (0 disables the whole ladder, rung-4 fast path included):

    - **Rung 1** — re-drive the ASSIGNEE: ``deliver_assignee(agent, msg)``
      with the sanitized ``blocker_note`` (runtime wires it to
      ``lane_manager.deliver_chat``, which forks busy→steer / idle→followup;
      the turn bills the nudged agent's normal work ledger).
    - **Rung 2** — escalate to the task CREATOR (a followup turn); skipped
      (rung still climbs) when creator == assignee or creator is the
      operator — the operator path is rung 4's job.
    - **Rung 3** — the LEAD's plate: no message is sent; the cron
      ``lead_blocked_tasks_fn`` probe surfaces rung ≥ 3 blocked tasks on
      the lead's heartbeat plate. Teamless/leaderless tasks climb straight
      past (nothing surfaces them until rung 4's own trigger).
    - **Rung 4** — ONE durable human Needs-you entry via the
      ``help_requests`` registry, at-most-once PER TASK EVER
      (``claim_blocked_human_notice``). Fires immediately for the verified
      budget-exhausted blocker family (see ``_BUDGET_BLOCKER_RE``) or at
      ``ladder_human_fallback_hours`` blocked, whichever first — from any
      rung. The chain watcher's existing stall nudge is untouched.

    The ladder NEVER changes task status, reassigns, auto-cancels, or
    writes goals — the nudged actors act through their own already-legal
    verbs. State (rung + last climb) is durable in the tasks store and
    resets when a task leaves ``blocked`` (cleared by the sweep; a flip
    faster than one sweep interval keeps its prior rung — bounded, benign).
    Every climb is CAS-claimed (``ladder_climb``) so racing sweeps or a
    restart replay can't double-send, and writes one audit row via
    ``audit_fn`` (``action="blocked_task_escalation"`` at the wiring site).

    Delivery seams may be sync (the runtime passes fire-and-forget
    wrappers so a sweep never blocks on an agent turn) or async (tests).
    """

    def __init__(
        self,
        tasks_store: Any,
        *,
        deliver_assignee: Callable[[str, str], Any] | None = None,
        deliver_creator: Callable[[str, str], Any] | None = None,
        lead_of_fn: Callable[[str | None], str | None] | None = None,
        help_requests: Any = None,
        audit_fn: Callable[[str, int, dict], Any] | None = None,
        interval_s: float | None = None,
        fallback_s: float | None = None,
        operator_id: str = "operator",
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self._tasks = tasks_store
        self._deliver_assignee = deliver_assignee
        self._deliver_creator = deliver_creator
        self._lead_of_fn = lead_of_fn
        self._help_requests = help_requests
        self._audit_fn = audit_fn
        # None ⇒ resolve from limits.py on every sweep (env-adjustable
        # without a restart); an explicit value pins it (tests).
        self._interval_s = interval_s
        self._fallback_s = fallback_s
        self._operator_id = operator_id
        self._wall_clock = wall_clock

    def _resolve_interval_s(self) -> float:
        if self._interval_s is not None:
            return self._interval_s
        from src.shared import limits as limits_mod

        return limits_mod.resolve("ladder_rung_interval_minutes") * 60.0

    def _resolve_fallback_s(self) -> float:
        if self._fallback_s is not None:
            return self._fallback_s
        from src.shared import limits as limits_mod

        return limits_mod.resolve("ladder_human_fallback_hours") * 3600.0

    async def sweep_once(self) -> None:
        """One ladder pass over every blocked task (claim-then-deliver)."""
        interval_s = self._resolve_interval_s()
        if interval_s <= 0:
            # 0 = the whole ladder is OFF (B4-style kill switch): no state
            # rows, no nudges, and no rung-4 fast path.
            return
        fallback_s = self._resolve_fallback_s()
        now = self._wall_clock()
        try:
            self._tasks.ladder_reset_unblocked()
            blocked = self._tasks.list_blocked()
        except Exception as e:
            logger.warning("ladder: listing blocked tasks failed: %s", e)
            return
        # C5: clear stale Needs-you rows for tasks that have left `blocked`.
        # ``ladder_reset_unblocked`` above drops the per-episode ladder
        # state, but the durable rung-4 ``help_requests`` row (M9-exempt
        # from age reap) would otherwise keep the authoritative Needs-you
        # feed asserting "human action required" long after the task
        # unblocked/completed. The once-ever claim is deliberately kept.
        self._reconcile_resolved_escalations(blocked)
        for task in blocked:
            task_id = task.get("id")
            if not task_id:
                continue
            # Per-task isolation, mirroring the chain sweep: one poison
            # task must not starve the rest; it self-heals next sweep.
            try:
                await self._process_blocked(task, task_id, now, interval_s, fallback_s)
            except Exception as e:
                logger.warning("ladder: escalating task %s failed: %s", task_id, e)

    def _reconcile_resolved_escalations(self, blocked: list[dict]) -> None:
        """Resolve open rung-4 Needs-you rows whose task is no longer blocked (C5)."""
        if self._help_requests is None:
            return
        try:
            open_ids = self._help_requests.open_escalation_task_ids()
        except Exception as e:
            logger.debug("ladder: reading open escalations failed: %s", e)
            return
        if not open_ids:
            return
        blocked_ids = {t.get("id") for t in blocked if t.get("id")}
        for task_id in open_ids - blocked_ids:
            try:
                self._help_requests.resolve_for_task(task_id)
            except Exception as e:
                logger.debug(
                    "ladder: resolving stale escalation for task %s failed: %s",
                    task_id, e,
                )

    async def _process_blocked(
        self, task: dict, task_id: str, now: float,
        interval_s: float, fallback_s: float,
    ) -> None:
        state = self._tasks.ladder_state(task_id)
        if state is None:
            state = self._tasks.ladder_observe_blocked(task_id, now=now)
        rung = int(state.get("rung") or 0)
        if rung >= 4:
            return  # the human owns it now — nothing above rung 4
        note = (task.get("blocker_note") or "").strip()
        # Explicit None checks — 0.0 is a legitimate timestamp (frozen
        # test clocks), so `or`-style fallbacks would misread it.
        raw_since = state.get("blocked_since")
        blocked_since = float(raw_since) if raw_since is not None else now
        blocked_age = now - blocked_since

        # Rung 4 first — its triggers (budget family / max age) preempt the
        # interval climb from ANY rung, including a first observation.
        if _is_budget_blocker(note) or blocked_age >= fallback_s:
            if self._tasks.ladder_climb(task_id, rung, 4, now=now):
                reason = "budget_exhausted" if _is_budget_blocker(note) else "max_age"
                filed = self._escalate_human(task, task_id, note, reason)
                self._audit(task_id, 4, {"reason": reason, "help_request_filed": filed})
            return

        if rung >= 3:
            return  # parked on the lead's plate until rung 4's own trigger
        raw_climb = state.get("last_climb_at")
        last_climb = float(raw_climb) if raw_climb is not None else blocked_since
        if now - last_climb < interval_s:
            return
        target = rung + 1
        if not self._tasks.ladder_climb(task_id, rung, target, now=now):
            return  # lost the claim to a concurrent sweep — it sends, we don't
        detail: dict = {}
        if target == 1:
            await self._send(
                self._deliver_assignee, task.get("assignee") or "",
                self._rung1_message(task, task_id, note), "assignee", task_id,
            )
        elif target == 2:
            creator = task.get("creator") or ""
            assignee = task.get("assignee") or ""
            if not creator or creator == assignee:
                detail["skipped"] = "creator_is_assignee"
            elif creator == self._operator_id:
                detail["skipped"] = "creator_is_operator"
            else:
                await self._send(
                    self._deliver_creator, creator,
                    self._rung2_message(task, task_id, note), "creator", task_id,
                )
        elif target == 3:
            # No message — the lead-duty plate probe IS the mechanism.
            lead = self._lead_of(task.get("team_id"))
            if not lead:
                detail["skipped"] = "no_lead"
        self._audit(task_id, target, detail)

    # ── rung mechanics ───────────────────────────────────────────

    async def _send(
        self, fn: Callable | None, agent: str, message: str,
        label: str, task_id: str,
    ) -> None:
        if fn is None or not agent:
            return
        try:
            result = fn(agent, message)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.warning("ladder: %s nudge for task %s failed: %s", label, task_id, e)

    def _lead_of(self, team_id: str | None) -> str | None:
        if self._lead_of_fn is None or not team_id:
            return None
        try:
            return self._lead_of_fn(team_id) or None
        except Exception as e:
            logger.debug("ladder: lead lookup for team %s failed: %s", team_id, e)
            return None

    def _escalate_human(self, task: dict, task_id: str, note: str, reason: str) -> bool:
        """File the ONE durable Needs-you entry for this task (claim-gated).

        RECORD-then-claim (C6): the once-ever ``blocked_human_notices``
        claim is consumed only AFTER the durable Needs-you row is filed, so
        a transient ``help_requests.db`` failure leaves the forever-claim
        intact and a future re-block can still surface the task (instead of
        the claim being permanently burned with no row). A read-only peek
        preserves at-most-once-per-task-ever without consuming the claim.
        """
        if self._help_requests is None:
            return False
        if self._tasks.blocked_human_notice_claimed(task_id):
            return False  # already surfaced once for this task ever — never re-file
        title = sanitize_for_prompt(task.get("title") or "")[:120]
        description = (
            f"Task '{title}' ({task_id}) is blocked and the escalation "
            f"ladder is exhausted ({reason}). "
            f"Blocker: {sanitize_for_prompt(note) or '(no blocker note recorded)'}"
        )[:500]
        try:
            self._help_requests.record(
                "blocked_task_escalation",
                task.get("assignee") or "",
                {
                    "name": task_id,
                    "service": "tasks",
                    "description": description,
                    "reason": reason,
                    "team_id": task.get("team_id"),
                },
            )
        except Exception as e:
            # Claim NOT consumed — a later re-block re-enters this path and
            # retries the durable file (the escalation is never lost).
            logger.warning("ladder: human escalation for task %s failed: %s", task_id, e)
            return False
        # Durable row landed — now consume the forever-claim.
        self._tasks.claim_blocked_human_notice(task_id)
        return True

    def _audit(self, task_id: str, rung: int, detail: dict) -> None:
        if self._audit_fn is None:
            return
        try:
            self._audit_fn(task_id, rung, detail)
        except Exception as e:
            logger.debug("ladder: audit write for task %s failed: %s", task_id, e)

    # ── nudge text (blocker_note is normalized at persistence; it is
    #    still agent-authored, so it passes sanitize_for_prompt before
    #    reaching another agent's context) ─────────────────────────

    def _rung1_message(self, task: dict, task_id: str, note: str) -> str:
        title = sanitize_for_prompt(task.get("title") or "")[:120]
        reason = sanitize_for_prompt(note) or "(no blocker note recorded)"
        return (
            f"[blocked-task escalation] Your task '{title}' ({task_id}) is "
            f"blocked with no progress.\nBlocker: {reason}\n"
            "Resolve the blocker now if you can, or update the task status "
            "(update_status) with what you need — if someone else must act, "
            "say who and why."
        )

    def _rung2_message(self, task: dict, task_id: str, note: str) -> str:
        title = sanitize_for_prompt(task.get("title") or "")[:120]
        assignee = sanitize_for_prompt(task.get("assignee") or "")[:80]
        reason = sanitize_for_prompt(note) or "(no blocker note recorded)"
        return (
            f"[blocked-task escalation] A task you created — '{title}' "
            f"({task_id}), assigned to {assignee} — is still blocked.\n"
            f"Blocker: {reason}\n"
            "Help unblock it: answer the blocker, adjust or hand_off the "
            "work, or get its status updated."
        )


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
        ladder: BlockedTaskLadder | None = None,
    ) -> None:
        self._tasks = tasks_store
        self._deliver = deliver
        self._settle_s = settle_s
        self._sweep_s = sweep_s
        self._watch_window_s = watch_window_s
        self._stall_after_s = stall_after_s
        self._clock = clock          # monotonic — for the terminal settle debounce
        self._wall_clock = wall_clock  # wall-clock — for stall age vs stored updated_at
        # Blocked-task escalation ladder (plan §8 #22) — piggybacks this
        # watcher's sweep cadence; None in deployments/tests that don't
        # wire it. The ladder resolves its own enable/interval knobs.
        self._ladder = ladder
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

            # Per-root isolation: a failure deriving the verdict or claiming a
            # delivery for ONE root must not abort the rest of the sweep (the
            # stall helper already self-isolates; this guards the
            # terminal path the same way). A poison root self-heals next sweep.
            try:
                await self._process_root(root, root_id)
            except Exception as e:
                logger.warning(
                    "chain watcher: processing root %s failed: %s", root_id, e,
                )

        # Bound memory: forget settle timers for roots no longer watchable.
        for stale in [r for r in self._settling if r not in live_ids]:
            self._settling.pop(stale, None)

        # Blocked-task escalation ladder (plan §8 #22) — same cadence,
        # isolated failure domain: a ladder error never starves the
        # terminal-delivery guarantee above.
        if self._ladder is not None:
            try:
                await self._ladder.sweep_once()
            except Exception as e:
                logger.warning("blocked-task ladder sweep failed: %s", e)

    async def _process_root(self, root: dict, root_id: str) -> None:
        """Deliver/settle/nudge a single watchable root. Raises on store
        errors so :meth:`sweep_once` can isolate one bad root from the rest."""
        verdict = self._tasks.chain_terminal_verdict(root_id)
        if verdict is None:
            # Still active (or a new hop just appeared) — reset settle.
            self._settling.pop(root_id, None)
            await self._maybe_nudge_stall(root, root_id)
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
