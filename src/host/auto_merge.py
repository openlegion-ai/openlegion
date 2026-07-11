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
  3. Execute through the SAME claim-first machinery the human merge
     endpoint uses (``TeamStore.claim_review_for_merge`` /
     ``drive.merge_branch`` / ``finalize_merge`` / ``revert_merge_claim``) —
     no git logic is reimplemented here. A concurrent human merge
     losing/winning the claim is harmless (the claim 409-equivalent path
     just aborts).
  4. Record a ``rater_kind="system"`` track-record event
     (``resolution="auto_merged"``) + one ``blackboard.log_audit`` row.
     The self-reinforcement guard lives in ``track_record.py``:
     system-rated events never feed :meth:`pair_trust`'s floor count.
  5. Notify — an operator-chat note (best-effort, injected via
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
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from src.host import drive as team_drive
from src.host.track_record import record_best_effort
from src.shared import limits as limits_mod
from src.shared.utils import setup_logging

logger = setup_logging("host.auto_merge")

# The identity every auto-merge / decay write uses in `rated_by` /
# `drive_reviews.reviewer` — never a real agent id, so it reads
# unambiguously as kernel-executed in every listing/audit trail.
AUTO_MERGE_RATER = "policy_engine"


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
    """Run the auto-merge pipeline for one lead-approved review. Never raises.

    ``repo`` is the team's already-resolved bare-repo path (the caller uses
    the SAME resolver the human merge endpoint uses — ``server.py``'s
    ``_drive_repo`` — so this function stays git-transport-agnostic).
    ``merge_branch_fn`` / ``branch_head_sha_fn`` default to the real
    ``drive.py`` functions and are only overridden by tests. Every exit path
    logs its reason; nothing here ever propagates to the caller (the verdict
    endpoint's fire-and-forget ``asyncio.create_task`` contract).
    """
    review_id = review.get("id", "")
    author = review.get("author", "")
    rng = rng if rng is not None else random.Random()

    def _skip(reason: str) -> None:
        logger.info("auto-merge skipped for review %s (team %s): %s", review_id, team_id, reason)

    try:
        if track_record_store is None:
            _skip("no track record store wired")
            return

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
        branch = claimed.get("branch")
        recorded_sha = claimed.get("head_sha")
        message = (
            f"Auto-merge review {review_id}: {claimed.get('title', '')} "
            f"(branch {branch}, by {author}, lead-approved by {lead_verdict_by})"
        )
        try:
            loop = asyncio.get_running_loop()
            live_sha = await loop.run_in_executor(None, branch_head_sha_fn, repo, branch)
            if not live_sha:
                teams_store.revert_merge_claim(review_id)
                _skip("branch deleted since approval")
                return
            if recorded_sha and live_sha != recorded_sha:
                teams_store.revert_merge_claim(review_id)
                _skip("branch advanced since approval — resubmit required")
                return
            commit = await merge_branch_fn(repo, recorded_sha or live_sha, message=message)
        except team_drive.MergeConflict as e:
            teams_store.revert_merge_claim(review_id)
            _skip(f"merge conflict: {e}")
            return
        except team_drive.RefMoved as e:
            teams_store.revert_merge_claim(review_id)
            _skip(f"ref moved (lost CAS — a human merge likely won the race): {e}")
            return
        except team_drive.DriveError as e:
            teams_store.revert_merge_claim(review_id)
            logger.warning("auto-merge git failure for review %s: %s", review_id, e)
            return
        except Exception:
            teams_store.revert_merge_claim(review_id)
            logger.exception("auto-merge unexpected git failure for review %s", review_id)
            return

        try:
            resolved = teams_store.finalize_merge(review_id, commit, reviewer=AUTO_MERGE_RATER)
        except ValueError:
            # The row moved out from under us — should not happen (nothing
            # else touches a 'merging' row) — but the commit IS on main, so
            # surface the divergence honestly rather than pretend it didn't
            # land. Best-effort shape for the steps below.
            logger.exception(
                "auto-merge finalize divergence for review %s (commit %s already on main)",
                review_id, commit,
            )
            resolved = {**claimed, "status": "merged", "merge_commit": commit}

        # Step 4: record. Sampling decay reads the PRIOR auto-merge count
        # for this pair (before this one), so the pair's first
        # `auto_merge_sample_decay_after` auto-merges sample at the initial
        # rate and later ones decay to the floor rate.
        rate = sample_rate_for(
            pair.get("auto_merged", 0),
            initial=limits_mod.auto_merge_sample_rate_initial(),
            decay_after=limits_mod.resolve("auto_merge_sample_decay_after"),
            floor=limits_mod.auto_merge_sample_rate_floor(),
        )
        sampled = should_sample(rate, rng)
        record_best_effort(
            track_record_store,
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
        if audit_fn is not None:
            try:
                audit_fn(
                    action="drive_review_auto_merged",
                    actor=AUTO_MERGE_RATER,
                    target=team_id,
                    field=branch,
                    after_value=commit,
                    provenance="system",
                )
            except Exception:
                logger.exception("auto-merge audit write failed for review %s", review_id)

        logger.info(
            "auto-merged review %s (team %s, branch %s) -> %s (lead=%s author=%s sampled=%s)",
            review_id, team_id, branch, commit, lead_verdict_by, author, sampled,
        )

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
