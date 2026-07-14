"""Kernel-executed auto-merge consumer (plan §8 #20).

Supersedes the zero-enforcement clause of §8 #13 at the KERNEL layer ONLY:
the lead's Team-Drive-review verdict remains advisory at the permission
layer (the merge/reject endpoints and their `_require_operator_or_internal`
gates in ``src/host/server.py`` are byte-untouched, and the verdict
endpoint's lead-only gate is untouched too) — what changes is that the
governance kernel MAY act on an ``approve`` verdict when earned-autonomy
policy clears the (lead, submitter) pair. No new agent-reachable surface
grants merge ability; a lead records a verdict exactly as before, and this
module — fired host-side, in-process, via ``asyncio.create_task`` from the
verdict endpoint — decides whether to execute it.

Pipeline (:func:`consider_auto_merge`), each step logged and any failure
aborting silently-but-audited (the calling endpoint's response never blocks
on, or fails because of, this module):

  1. Kill switch / daily rate cap (``limits.auto_merge_daily_cap`` —
     0 disables auto-merge entirely, B4-style).
  2. Pair trust floor (:meth:`TrackRecordStore.pair_trust`) — HUMAN-executed
     merges of this pair's lead-approved reviews, zero rejected-after-
     approve, zero flag/revert decay events. Includes the self-approval
     hard block: a lead approving its own submission is never a pair,
     checked before spending a query on a pair that can never qualify.
  3. Claim + re-verify the reviewed tip (``TeamStore.claim_review_for_merge``
     / ``branch_head_sha``) — a deleted/advanced branch reverts the claim
     and skips (a resubmit, not an auto-merge; never consumes the cap).
  4. RESERVE the cap slot: record the ``rater_kind="system"`` /
     ``resolution="auto_merged"`` track-record event BEFORE the irreversible
     merge (C2). The cap counts these rows, so a failed durable write must
     fail CLOSED (revert + skip) — recording after a best-effort append (the
     pre-fix order) let a full/read-only ledger bypass ``auto_merge_daily_cap``
     indefinitely. The self-reinforcement guard lives in ``track_record.py``:
     system-rated events never feed :meth:`pair_trust`'s floor count.
  5. Execute the merge + finalize as ONE cancellation-atomic unit
     (``_merge_and_finalize`` under ``asyncio.shield``, C1) so a consumer
     cancellation (mesh shutdown) can never land between ``update-ref`` and
     finalize — the exact window that would strand a ``merging`` row and an
     UNRECORDED landed commit. The unit reverts ``merging→open`` on any git
     failure and finalizes on success — never leaving a stranded row.
     A concurrent human merge losing/winning the claim is harmless.
  7. Audit (``blackboard.log_audit``).
  8. Notify — an operator-chat note (best-effort, injected via
     ``notify_fn`` so this module has no direct transport dependency),
     with a decaying sampling rate that asks for post-review.

Trust decay (steps 6/7 in the plan; the corresponding endpoints —
``flag-auto-merge`` / ``revert-merge`` — live in ``server.py`` next to the
other drive endpoints, since they need the same operator-or-internal gate
and drive git plumbing those already use) is read back in via
``pair_trust``'s ``flagged`` count, not duplicated here.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from src.host import drive as team_drive
from src.shared import limits as limits_mod
from src.shared.utils import dumps_safe, setup_logging

logger = setup_logging("host.auto_merge")

# The identity every auto-merge / decay write uses in `rated_by` /
# `drive_reviews.reviewer` — never a real agent id, so it reads
# unambiguously as kernel-executed in every listing/audit trail.
AUTO_MERGE_RATER = "policy_engine"

# Per-event-loop serialization lock for the daily-cap window (Phase-5
# review finding). The consumer is fired fire-and-forget, once per
# ``approve`` verdict; a lead clearing a backlog of approvals in one turn
# spawns several consumers concurrently. Each claims a DIFFERENT review, so
# the claim-first merge machinery does not serialize them — and the daily
# cap is a read-decide-then-record-later window, so without this lock every
# consumer reads the same pre-merge count and all proceed, blowing past the
# operator's ``auto_merge_daily_cap``. Holding one lock from the cap read
# through the ``auto_merged`` record makes count→merge→record atomic across
# consumers (auto-merges are rare and the cap is small, so serializing them
# is free and arguably correct — the kernel merges one at a time). Keyed by
# running loop so a fresh test loop never reuses a lock bound to another.
_cap_locks: dict[Any, asyncio.Lock] = {}


def _cap_lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    lock = _cap_locks.get(loop)
    if lock is None:
        lock = asyncio.Lock()
        _cap_locks[loop] = lock
    return lock


def midnight_utc_ts(now: float | None = None) -> float:
    """Epoch seconds for the most recent UTC midnight at or before ``now``."""
    dt = datetime.fromtimestamp(now if now is not None else time.time(), tz=timezone.utc)
    return dt.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


@dataclass(frozen=True)
class TrustGate:
    """Result of evaluating a (lead, submitter) pair's auto-merge eligibility."""

    eligible: bool
    reason: str


def evaluate_pair_trust(pair: dict[str, Any], *, trust_floor: int) -> TrustGate:
    """Apply the §8 #19/#20 floor to a :meth:`TrackRecordStore.pair_trust` result.

    Eligible iff ``merged >= trust_floor`` AND ``rejected_after_approve == 0``
    AND ``flagged == 0`` — any decay event (flag or revert) or any single
    rejected-after-approve zeroes the pair's eligibility outright, regardless
    of how high ``merged`` climbs afterward (the pair must rebuild trust via
    fresh human-executed merges).
    """
    if pair.get("flagged", 0) > 0:
        return TrustGate(False, "pair has a flag/revert decay event — trust reset to zero")
    if pair.get("rejected_after_approve", 0) > 0:
        return TrustGate(False, "pair has a rejected-after-approve event")
    merged = pair.get("merged", 0)
    if merged < trust_floor:
        return TrustGate(False, f"human-merged-approved count {merged} below floor {trust_floor}")
    return TrustGate(True, "trust floor met")


def sample_rate_for(prior_auto_merges: int, *, initial: float, decay_after: int, floor: float) -> float:
    """Sampling rate for a pair's NEXT auto-merge.

    ``prior_auto_merges`` is the pair's count of auto-merges BEFORE this
    one (0-indexed) — the first ``decay_after`` auto-merges (indices
    ``0..decay_after-1``) sample at ``initial``; from the ``decay_after``-th
    one onward, sampling decays to ``floor``.
    """
    return initial if prior_auto_merges < decay_after else floor


def should_sample(rate: float, rng: random.Random) -> bool:
    return rng.random() < rate


async def _merge_and_finalize(
    teams_store: Any,
    repo: Any,
    review_id: str,
    claimed: dict[str, Any],
    recorded_sha: str | None,
    live_sha: str,
    message: str,
    merge_branch_fn: Callable[..., Awaitable[str]],
) -> tuple[str, dict] | None:
    """The cancellation-atomic critical section (C1): merge the reviewed
    commit into main, then finalize the review row.

    Run as its OWN task (``asyncio.ensure_future``) and awaited under
    ``asyncio.shield`` by :func:`consider_auto_merge`, so a cancellation
    of the consumer can never land BETWEEN the git ``update-ref`` and
    ``finalize_merge`` — the exact window that would otherwise advance
    main after the coroutine is gone, leaving an UNRECORDED landed commit
    and a stranded ``merging`` row. This unit fully owns the claim's
    resolution: it reverts ``merging→open`` on ANY git failure and
    finalizes on success, so it NEVER leaves a stranded ``merging`` row.

    Returns ``(commit, resolved)`` on success, ``None`` on a (reverted)
    merge failure. Only ``CancelledError`` can escape — and only when
    THIS task is itself cancelled (loop teardown), never from the
    consumer detaching.
    """
    try:
        commit = await merge_branch_fn(repo, recorded_sha or live_sha, message=message)
    except team_drive.MergeConflict as e:
        teams_store.revert_merge_claim(review_id)
        logger.info("auto-merge reverted claim for review %s: merge conflict: %s", review_id, e)
        return None
    except team_drive.RefMoved as e:
        teams_store.revert_merge_claim(review_id)
        logger.info(
            "auto-merge reverted claim for review %s: ref moved (lost CAS — a human merge likely won): %s",
            review_id, e,
        )
        return None
    except team_drive.DriveError as e:
        teams_store.revert_merge_claim(review_id)
        logger.warning("auto-merge git failure for review %s: %s", review_id, e)
        return None
    except asyncio.CancelledError:
        raise
    except Exception:
        teams_store.revert_merge_claim(review_id)
        logger.exception("auto-merge unexpected git failure for review %s", review_id)
        return None
    try:
        resolved = teams_store.finalize_merge(review_id, commit, reviewer=AUTO_MERGE_RATER)
    except ValueError:
        # The row moved out from under us — should not happen (nothing
        # else touches a 'merging' row) — but the commit IS on main, so
        # surface the divergence honestly rather than pretend it didn't
        # land. Best-effort shape for the steps in the caller.
        logger.exception(
            "auto-merge finalize divergence for review %s (commit %s already on main)",
            review_id, commit,
        )
        resolved = {**claimed, "status": "merged", "merge_commit": commit}
    return commit, resolved


async def consider_auto_merge(
    *,
    team_id: str,
    review: dict[str, Any],
    lead_verdict_by: str,
    teams_store: Any,
    track_record_store: Any,
    repo: Any,
    merge_branch_fn: Callable[..., Awaitable[str]] = team_drive.merge_branch,
    branch_head_sha_fn: Callable[[Any, str], str] = team_drive.branch_head_sha,
    notify_fn: Callable[..., Awaitable[None]] | None = None,
    audit_fn: Callable[..., None] | None = None,
    clock: Callable[[], float] = time.time,
    rng: random.Random | None = None,
) -> None:
    """Run the auto-merge pipeline for one lead-approved review.

    Never raises an ordinary exception — the outer net absorbs it (the
    verdict endpoint's fire-and-forget ``asyncio.create_task`` contract).
    ``CancelledError`` DOES propagate (C1): a cancelled consumer must
    unwind, but only AFTER the claim is settled — the claim is reverted
    (if the merge had not begun) or the shielded merge/finalize unit is
    let run to completion, so a cancellation never strands a ``merging``
    row or lands an unrecorded commit.

    ``repo`` is the team's already-resolved bare-repo path (the caller uses
    the SAME resolver the human merge endpoint uses — ``server.py``'s
    ``_drive_repo`` — so this function stays git-transport-agnostic).
    ``merge_branch_fn`` / ``branch_head_sha_fn`` default to the real
    ``drive.py`` functions and are only overridden by tests. Every exit path
    logs its reason.
    """
    review_id = review.get("id", "")
    author = review.get("author", "")
    rng = rng if rng is not None else random.Random()

    def _skip(reason: str) -> None:
        logger.info("auto-merge skipped for review %s (team %s): %s", review_id, team_id, reason)

    lock: asyncio.Lock | None = None
    lock_held = False
    claimed_review = False   # C1: a claim exists (review row is `merging`)…
    claim_settled = False    # …that a later step has resolved (merged or reverted).
    try:
        if track_record_store is None:
            _skip("no track record store wired")
            return

        # Serialize the daily-cap window (Phase-5 review finding): held from
        # the cap read through the ``auto_merged`` record so concurrent
        # consumers (a lead approving a backlog in one turn) can't all read the
        # same pre-merge count and each proceed past ``auto_merge_daily_cap``.
        # Released before the best-effort notify below (which need not be
        # serialized) and, on any early return/exception, by the ``finally``.
        lock = _cap_lock()
        await lock.acquire()
        lock_held = True

        # Step 1: kill switch / daily rate cap.
        cap = limits_mod.resolve("auto_merge_daily_cap")
        if cap <= 0:
            _skip("auto-merge disabled (auto_merge_daily_cap=0)")
            return
        since = midnight_utc_ts(clock())
        today_count = track_record_store.count_events(
            source="drive_review", outcome="auto_merged", rater_kind="system", since=since,
        )
        if today_count >= cap:
            _skip(f"daily auto-merge cap reached ({today_count}/{cap})")
            return

        # Step 2: pair trust floor, including the self-approval hard block —
        # a lead approving its own submission is not a pair, no matter how
        # trusted the lead is, so this is checked before spending a query on
        # `pair_trust` for a pair that can never be eligible anyway.
        if lead_verdict_by == author:
            _skip(f"self-approval by {lead_verdict_by} — never auto-merged")
            return
        pair = track_record_store.pair_trust(lead_verdict_by, author)
        floor = limits_mod.resolve("auto_merge_trust_floor")
        gate = evaluate_pair_trust(pair, trust_floor=floor)
        if not gate.eligible:
            _skip(gate.reason)
            return

        # Step 3: execute through the SAME claim-first machinery the human
        # merge endpoint uses. A lost claim means a human (or a racing
        # second auto-merge attempt) already resolved the review — harmless.
        try:
            claimed = teams_store.claim_review_for_merge(review_id)
        except ValueError as e:
            _skip(f"claim lost — already resolved: {e}")
            return
        claimed_review = True
        branch = claimed.get("branch")
        recorded_sha = claimed.get("head_sha")
        message = (
            f"Auto-merge review {review_id}: {claimed.get('title', '')} "
            f"(branch {branch}, by {author}, lead-approved by {lead_verdict_by})"
        )

        # Re-verify the reviewed tip is still live BEFORE reserving a cap
        # slot — a deleted/advanced branch is a resubmit, not an auto-merge,
        # and must not consume the daily cap.
        try:
            loop = asyncio.get_running_loop()
            live_sha = await loop.run_in_executor(None, branch_head_sha_fn, repo, branch)
        except asyncio.CancelledError:
            # Cancelled before the merge began — nothing landed on main;
            # revert the claim so we never strand a `merging` row (C1).
            with contextlib.suppress(Exception):
                teams_store.revert_merge_claim(review_id)
            claim_settled = True
            raise
        except Exception:
            teams_store.revert_merge_claim(review_id)
            claim_settled = True
            logger.exception("auto-merge branch head check failed for review %s", review_id)
            return
        if not live_sha:
            teams_store.revert_merge_claim(review_id)
            claim_settled = True
            _skip("branch deleted since approval")
            return
        if recorded_sha and live_sha != recorded_sha:
            teams_store.revert_merge_claim(review_id)
            claim_settled = True
            _skip("branch advanced since approval — resubmit required")
            return

        # Step 4a: RESERVE the daily-cap slot BEFORE the irreversible merge
        # (C2). The cap counts `auto_merged` rows, but the merge is a git
        # side effect on main — so if this durable increment can't be
        # written (full / read-only track_record.db) the merge must NOT
        # proceed. A best-effort append AFTER the merge (the pre-fix order)
        # let every approval merge while the count stayed flat, silently
        # bypassing `auto_merge_daily_cap`. Recording first is fail-CLOSED:
        # a rare merge failure after a successful reserve over-counts the
        # cap (conservative — never under-counts). Sampling decay reads the
        # pair's PRIOR auto-merge count, available now.
        rate = sample_rate_for(
            pair.get("auto_merged", 0),
            initial=limits_mod.auto_merge_sample_rate_initial(),
            decay_after=limits_mod.resolve("auto_merge_sample_decay_after"),
            floor=limits_mod.auto_merge_sample_rate_floor(),
        )
        sampled = should_sample(rate, rng)
        try:
            track_record_store.record(
                source="drive_review",
                ref_id=review_id,
                outcome="auto_merged",
                rater_kind="system",
                agent_id=author,
                team_id=team_id,
                rated_by=AUTO_MERGE_RATER,
                details={
                    "branch": branch,
                    "lead_agent_id": lead_verdict_by,
                    "lead_verdict_by": lead_verdict_by,
                    "lead_verdict": "approve",
                    "resolution": "auto_merged",
                    "resolved_by": AUTO_MERGE_RATER,
                    "sampled": sampled,
                },
            )
        except Exception:
            # Fail closed: the cap slot could not be durably reserved, so do
            # NOT merge — a bypassed cap is worse than a skipped auto-merge.
            teams_store.revert_merge_claim(review_id)
            claim_settled = True
            logger.exception(
                "auto-merge cap-record failed for review %s — failing closed (no merge)", review_id,
            )
            return

        # Step 4b: merge + finalize as ONE cancellation-atomic unit (C1),
        # run as its own task and awaited under `shield` so a consumer
        # cancellation can never land between `update-ref` and finalize —
        # the unit runs to completion (finalize or revert) regardless.
        merge_task = asyncio.ensure_future(
            _merge_and_finalize(
                teams_store, repo, review_id, claimed, recorded_sha, live_sha,
                message, merge_branch_fn,
            )
        )
        try:
            outcome = await asyncio.shield(merge_task)
        except asyncio.CancelledError:
            # Let the shielded unit finish (finalize the landed commit or
            # revert a failed merge) so we never strand a `merging` row or
            # land an unrecorded commit, then propagate the cancellation —
            # detaching only this consumer, never the merge itself.
            with contextlib.suppress(BaseException):
                await asyncio.shield(merge_task)
            claim_settled = True
            raise
        claim_settled = True  # the merge task resolved the claim (merged or reverted)
        if outcome is None:
            return  # merge failed; `_merge_and_finalize` already reverted + logged
        commit, resolved = outcome

        if audit_fn is not None:
            try:
                # ``after_value`` carries a small JSON blob (not just the
                # bare commit sha) so the plan §8 #19 autonomy-log dashboard
                # view can show the submitting agent + sampled flag without
                # a second lookup — the same "structured after_value"
                # convention ``policy.py``'s ``policy_decision`` rows
                # already use.
                audit_fn(
                    action="drive_review_auto_merged",
                    actor=AUTO_MERGE_RATER,
                    target=team_id,
                    field=branch,
                    after_value=dumps_safe({
                        "commit": commit,
                        "author": author,
                        "sampled": sampled,
                    }),
                    provenance="system",
                )
            except Exception:
                logger.exception("auto-merge audit write failed for review %s", review_id)

        logger.info(
            "auto-merged review %s (team %s, branch %s) -> %s (lead=%s author=%s sampled=%s)",
            review_id, team_id, branch, commit, lead_verdict_by, author, sampled,
        )

        # The cap slot is now durably recorded — release before the notify so
        # a slow operator-chat post doesn't serialize the next consumer.
        if lock_held:
            lock.release()
            lock_held = False

        # Step 5: notify (best-effort — never blocks/fails the pipeline
        # outcome, the merge already landed).
        if notify_fn is not None:
            try:
                await notify_fn(
                    team_id=team_id,
                    review=resolved,
                    branch=branch,
                    commit=commit,
                    lead_verdict_by=lead_verdict_by,
                    author=author,
                    sampled=sampled,
                )
            except Exception:
                logger.exception("auto-merge operator-chat note failed for review %s", review_id)
    except Exception:
        # Outermost net: the verdict endpoint's response must never block
        # on, or fail because of, this consumer.
        logger.exception("auto-merge consumer raised for review %s (team %s)", review_id, team_id)
        # C1 safety net: an unexpected error after the claim but before the
        # merge task resolved it would otherwise strand a `merging` row.
        if claimed_review and not claim_settled:
            with contextlib.suppress(Exception):
                teams_store.revert_merge_claim(review_id)
    finally:
        # Release the cap lock on any early return (cap reached, trust floor,
        # lost claim, git failure) or exception before the pre-notify release.
        if lock_held and lock is not None:
            lock.release()
