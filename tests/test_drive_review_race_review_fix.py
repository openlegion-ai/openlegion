"""M14 regression — Team Drive review-queue race: a same-branch resubmit
racing an in-flight merge claim must not break the one-live-review-per-
branch invariant.

``create_review`` only supersedes ``open`` rows (it misses a ``merging``
one), so a resubmit that lands during the merge-claim window inserts a
NEW open review alongside the claimed one. Two failure modes follow:

  (A) zombie — the merge aborts (live-sha check fails) and
      ``revert_merge_claim`` blindly flips the claimed row back to
      ``open`` → TWO open reviews for one branch, one a stale zombie.
  (B) double-merge — a SAME-sha resubmit while the first review merges
      pins the already-merged sha; approving it later re-merges
      integrated content.

Store-only fix (no endpoint change): ``revert_merge_claim`` supersedes
the reverted row when a newer open row already owns the branch;
``finalize_merge`` supersedes any open same-branch row on the same
head_sha it just merged.
"""

import pytest

from src.host.teams import TeamStore


@pytest.fixture
def store():
    s = TeamStore(db_path=":memory:")
    s.create_team("team-x")
    return s


def _open_reviews(store: TeamStore, branch: str) -> list[dict]:
    return [r for r in store.list_reviews("team-x", status="open") if r["branch"] == branch]


def test_revert_after_resubmit_supersedes_zombie(store):
    """Mode A: resubmit races the merge claim → revert must supersede the
    reverted row, leaving exactly ONE open review for the branch."""
    r1 = store.create_review("team-x", "feat-a", "member1", "first", head_sha="aaa")
    store.claim_review_for_merge(r1["id"])  # open -> merging

    # Resubmit lands during the claim window. create_review only supersedes
    # OPEN rows, so R1 (merging) survives and R2 is inserted open.
    r2 = store.create_review("team-x", "feat-a", "member1", "second", head_sha="bbb")
    assert store.get_review(r1["id"])["status"] == "merging"
    assert store.get_review(r2["id"])["status"] == "open"

    # Git merge aborts (live-sha advanced) → revert.
    store.revert_merge_claim(r1["id"])

    assert store.get_review(r1["id"])["status"] == "superseded"
    assert store.get_review(r2["id"])["status"] == "open"
    # The invariant holds: exactly one live review for the branch.
    assert [r["id"] for r in _open_reviews(store, "feat-a")] == [r2["id"]]


def test_revert_without_resubmit_still_reopens(store):
    """No resubmit raced → revert restores the row to open (unchanged
    pre-M14 behavior, so a genuine transient git failure is re-mergeable)."""
    r = store.create_review("team-x", "feat-b", "member1", "t", head_sha="ccc")
    store.claim_review_for_merge(r["id"])
    store.revert_merge_claim(r["id"])
    assert store.get_review(r["id"])["status"] == "open"


def test_revert_noop_when_not_merging(store):
    """Revert is a no-op on a row that already moved off ``merging``
    (resolve_review only acts on ``open``, so reach ``rejected`` via a
    clean revert-to-open first)."""
    r = store.create_review("team-x", "feat-b2", "member1", "t")
    store.claim_review_for_merge(r["id"])
    store.revert_merge_claim(r["id"])  # merging -> open (no resubmit raced)
    store.resolve_review(r["id"], "rejected", reviewer="operator")  # open -> rejected
    store.revert_merge_claim(r["id"])  # no-op: row is not merging
    assert store.get_review(r["id"])["status"] == "rejected"


def test_finalize_supersedes_same_sha_resubmit(store):
    """Mode B: a same-sha resubmit racing the merge must be superseded when
    the first review finalizes, so it can't later re-merge merged content."""
    r1 = store.create_review("team-x", "feat-c", "member1", "first", head_sha="deadbeef")
    store.claim_review_for_merge(r1["id"])  # open -> merging

    # Same-sha resubmit during the claim window (open, same head_sha).
    r2 = store.create_review("team-x", "feat-c", "member1", "resubmit", head_sha="deadbeef")
    assert store.get_review(r2["id"])["status"] == "open"

    store.finalize_merge(r1["id"], "mergecommit", reviewer="operator")

    assert store.get_review(r1["id"])["status"] == "merged"
    # The redundant same-sha open review is dropped.
    assert store.get_review(r2["id"])["status"] == "superseded"
    assert _open_reviews(store, "feat-c") == []


def test_finalize_leaves_different_sha_resubmit_open(store):
    """A resubmit with a DIFFERENT sha is genuinely new work — finalizing
    the first review must leave it open (it still needs its own merge)."""
    r1 = store.create_review("team-x", "feat-d", "member1", "first", head_sha="1111")
    store.claim_review_for_merge(r1["id"])
    r2 = store.create_review("team-x", "feat-d", "member1", "newer", head_sha="2222")

    store.finalize_merge(r1["id"], "mc", reviewer="operator")

    assert store.get_review(r1["id"])["status"] == "merged"
    assert store.get_review(r2["id"])["status"] == "open"


def test_finalize_null_sha_does_not_supersede_other_nulls(store):
    """A NULL head_sha must not sweep up unrelated NULL-sha open reviews
    (SQL NULL never equals NULL; the guard makes that explicit)."""
    r1 = store.create_review("team-x", "feat-e", "member1", "first")  # head_sha None
    store.claim_review_for_merge(r1["id"])
    r2 = store.create_review("team-x", "feat-e", "member1", "other")  # head_sha None

    store.finalize_merge(r1["id"], "mc", reviewer="operator")

    assert store.get_review(r1["id"])["status"] == "merged"
    assert store.get_review(r2["id"])["status"] == "open"
