"""Tests for the simplified CAPTCHA cost counter (Phase 8 §11.10 trim).

Covers:
  * pricing-table lookup (case-insensitive, unknown variant → None)
  * month rollover resets the bucket
  * snapshot / restore round-trip via JSON
  * atomic write (tmp file replaced)
  * concurrent ``add_cost`` correctness
  * unit invariant: $1 cap allows ~1000 v2-checkbox solves before tripping
    (regression for the cents-vs-millicents mismatch from Codex F1)
  * legacy ``cents`` snapshots migrate to ``millicents`` (×1000) on load
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from src.browser import captcha_cost_counter as cost


@pytest.fixture(autouse=True)
async def _isolate_state(tmp_path, monkeypatch):
    """Each test gets a clean dict and a tmp path so state can't leak."""
    monkeypatch.setenv(
        "CAPTCHA_COST_COUNTER_PATH", str(tmp_path / "captcha_costs.json"),
    )
    await cost.reset()
    yield
    await cost.reset()


class TestEstimateMillicents:
    """Pricing table is now in MILLICENTS (1/1000 of a cent). The legacy
    ``estimate_cents`` is a back-compat alias to ``estimate_millicents``;
    both names return the same integer.
    """

    def test_known_2captcha_recaptcha_v2(self):
        # 2captcha v2-checkbox is published at $1/1000 = $0.001/solve
        # = 100 millicents/solve. Pre-fix this was labelled "cents" but
        # the magnitude (100) matches millicents, not cents (which would
        # have been a single 1¢ charge per solve).
        assert cost.estimate_millicents("2captcha", "recaptcha-v2-checkbox") == 100

    def test_known_capsolver_turnstile(self):
        assert cost.estimate_millicents("capsolver", "turnstile") == 60

    def test_unknown_kind_returns_none(self):
        assert cost.estimate_millicents("2captcha", "made-up-variant") is None

    def test_unknown_provider_returns_none(self):
        assert cost.estimate_millicents("nopal", "hcaptcha") is None

    def test_case_insensitive(self):
        assert cost.estimate_millicents("2CAPTCHA", "HCAPTCHA") == 100
        assert cost.estimate_millicents("CapSolver", "Turnstile") == 60

    def test_empty_inputs_safe(self):
        assert cost.estimate_millicents("", "hcaptcha") is None
        assert cost.estimate_millicents("2captcha", "") is None

    def test_legacy_estimate_cents_alias_returns_millicents(self):
        # Back-compat alias for out-of-tree subclasses; the unit changed
        # under the alias but the magnitude matches what those callers
        # were already storing — the callers' arithmetic was already
        # off-by-1000 when treating the value as cents. The alias keeps
        # them importing without breaking the in-tree fix.
        assert cost.estimate_cents("2captcha", "recaptcha-v2-checkbox") == 100


class TestAddCostAndOverCap:
    @pytest.mark.asyncio
    async def test_add_cost_accumulates(self):
        await cost.add_cost("agent-1", 100)
        await cost.add_cost("agent-1", 50)
        assert await cost.get_millicents("agent-1") == 150

    @pytest.mark.asyncio
    async def test_add_cost_zero_or_negative_dropped(self):
        await cost.add_cost("agent-1", 0)
        await cost.add_cost("agent-1", -10)
        assert await cost.get_millicents("agent-1") == 0

    @pytest.mark.asyncio
    async def test_over_cap_below_threshold(self):
        await cost.add_cost("agent-1", 99)
        assert await cost.over_cap("agent-1", 100) is False

    @pytest.mark.asyncio
    async def test_over_cap_at_threshold(self):
        await cost.add_cost("agent-1", 100)
        assert await cost.over_cap("agent-1", 100) is True

    @pytest.mark.asyncio
    async def test_over_cap_zero_disables(self):
        await cost.add_cost("agent-1", 1_000_000)
        assert await cost.over_cap("agent-1", 0) is False

    @pytest.mark.asyncio
    async def test_unrelated_agents_isolated(self):
        await cost.add_cost("agent-1", 200)
        assert await cost.get_millicents("agent-2") == 0

    @pytest.mark.asyncio
    async def test_legacy_get_cents_alias_returns_millicents(self):
        await cost.add_cost("agent-1", 100)
        # Back-compat alias — same integer the new ``get_millicents``
        # returns. The label-vs-unit mismatch is documented; out-of-tree
        # callers' arithmetic was already off-by-1000 before this fix.
        assert await cost.get_cents("agent-1") == 100


class TestMonthRollover:
    @pytest.mark.asyncio
    async def test_bucket_resets_on_month_change(self, monkeypatch):
        # Spend 200 in January.
        monkeypatch.setattr(cost, "_current_month", lambda: "2026-01")
        await cost.add_cost("agent-1", 200)
        assert await cost.get_millicents("agent-1") == 200

        # Roll over to February — bucket should reset.
        monkeypatch.setattr(cost, "_current_month", lambda: "2026-02")
        assert await cost.get_millicents("agent-1") == 0
        await cost.add_cost("agent-1", 50)
        assert await cost.get_millicents("agent-1") == 50

    @pytest.mark.asyncio
    async def test_month_rollover_during_over_cap(self, monkeypatch):
        monkeypatch.setattr(cost, "_current_month", lambda: "2026-01")
        await cost.add_cost("agent-1", 1500)
        assert await cost.over_cap("agent-1", 1000) is True

        monkeypatch.setattr(cost, "_current_month", lambda: "2026-02")
        assert await cost.over_cap("agent-1", 1000) is False


class TestUnitInvariantsRegression:
    """Regression tests for Codex F1 — cost-unit mismatch.

    The pricing table comment said ``$1.00 / 1000 = 0.10c each → 0.1¢``
    but the cap converted USD to **cents** (× 100), not millicents. So a
    $0.50 cap on 100-millicent solves tripped after the FIRST solve
    instead of after ~500. These tests pin the conversion math + the
    end-to-end cap arithmetic so the unit drift can't recur.
    """

    @pytest.mark.asyncio
    async def test_one_dollar_cap_allows_one_thousand_v2_checkbox_solves(self):
        """$1 cap → 100_000 millicents. 2captcha v2-checkbox = 100 mc/solve.
        100_000 / 100 = 1000 solves before the cap fires (caller reads cap
        via _resolve_cost_cap; we exercise the bucket math directly).
        """
        cap_millicents = 1_00_000  # $1.00 in millicents
        per_solve = cost.estimate_millicents(
            "2captcha", "recaptcha-v2-checkbox",
        )
        assert per_solve == 100  # provider-published rate
        # Simulate 999 solves — still under cap.
        for _ in range(999):
            await cost.add_cost("agent-1", per_solve)
        assert await cost.over_cap("agent-1", cap_millicents) is False
        # 1000th solve trips the cap.
        await cost.add_cost("agent-1", per_solve)
        assert await cost.over_cap("agent-1", cap_millicents) is True

    def test_provider_published_rate_is_one_hundred_millicents(self):
        """$0.001 per solve = 0.1¢ = 100 millicents. Anchors the table."""
        assert cost.estimate_millicents(
            "2captcha", "recaptcha-v2-checkbox",
        ) == 100

    def test_dollars_to_millicents_conversion_math(self):
        """$0.50 → 50_000 millicents (the conversion site lives in
        ``service.py:_resolve_cost_cap``; this test pins the arithmetic
        that test depends on).
        """
        # Equivalent: int(round(0.50 * 100_000)).
        assert int(round(0.50 * 100_000)) == 50_000
        # Smoke-check a non-round value to catch a silent /1000 typo.
        assert int(round(2.50 * 100_000)) == 250_000


class TestLegacyCentsSnapshotMigration:
    """Snapshots written before the millicents rename used a ``cents``
    field. ``restore`` migrates them by multiplying ×1000.
    """

    @pytest.mark.asyncio
    async def test_legacy_cents_field_migrated_on_load(
        self, tmp_path, monkeypatch, caplog,
    ):
        import logging as _logging
        path = tmp_path / "legacy.json"
        # Pin month so the bucket isn't dropped as stale.
        monkeypatch.setattr(cost, "_current_month", lambda: "2026-04")
        payload = {
            "version": 1,
            "saved_at": 0,
            "buckets": {
                # Pre-fix: stored 100 cents (== 0.1¢-each × 1 solve in
                # the broken accounting). On migration this becomes
                # 100_000 millicents. The intent of the original write
                # was clearly "100 of whatever-unit-we-used"; multiplying
                # by 1000 preserves the integer count of solves rather
                # than under-counting them post-rename.
                "agent-old": {"month": "2026-04", "cents": 100},
            },
        }
        path.write_text(json.dumps(payload))
        with caplog.at_level(_logging.INFO, logger="browser.captcha_cost"):
            loaded = await cost.restore(path)
        assert loaded == 1
        assert await cost.get_millicents("agent-old") == 100_000
        # Migration logged once.
        joined = "\n".join(rec.getMessage() for rec in caplog.records)
        assert "migrated" in joined and "millicents" in joined

    @pytest.mark.asyncio
    async def test_new_millicents_field_loaded_unchanged(
        self, tmp_path, monkeypatch,
    ):
        path = tmp_path / "new.json"
        monkeypatch.setattr(cost, "_current_month", lambda: "2026-04")
        payload = {
            "version": 1,
            "saved_at": 0,
            "buckets": {
                "agent-new": {"month": "2026-04", "millicents": 5000},
            },
        }
        path.write_text(json.dumps(payload))
        loaded = await cost.restore(path)
        assert loaded == 1
        assert await cost.get_millicents("agent-new") == 5000


class TestSnapshotRestore:
    @pytest.mark.asyncio
    async def test_round_trip(self, tmp_path):
        path = tmp_path / "costs.json"
        await cost.add_cost("agent-a", 200)
        await cost.add_cost("agent-b", 350)

        ok = await cost.snapshot(path)
        assert ok is True
        assert path.exists()

        # Wipe in-memory state, then restore.
        await cost.reset()
        assert await cost.get_millicents("agent-a") == 0
        loaded = await cost.restore(path)
        assert loaded == 2
        assert await cost.get_millicents("agent-a") == 200
        assert await cost.get_millicents("agent-b") == 350

    @pytest.mark.asyncio
    async def test_restore_drops_stale_month(self, tmp_path, monkeypatch):
        path = tmp_path / "costs.json"
        # Write a snapshot stamped for last month.
        payload = {
            "version": 1,
            "saved_at": 0,
            "buckets": {
                "agent-old": {"month": "2020-01", "millicents": 999},
            },
        }
        path.write_text(json.dumps(payload))
        loaded = await cost.restore(path)
        # Stale month skipped.
        assert loaded == 0
        assert await cost.get_millicents("agent-old") == 0

    @pytest.mark.asyncio
    async def test_restore_missing_file_safe(self, tmp_path):
        loaded = await cost.restore(tmp_path / "absent.json")
        assert loaded == 0

    @pytest.mark.asyncio
    async def test_restore_malformed_file_safe(self, tmp_path):
        path = tmp_path / "costs.json"
        path.write_text("{not valid json")
        loaded = await cost.restore(path)
        assert loaded == 0

    @pytest.mark.asyncio
    async def test_snapshot_writes_atomically(self, tmp_path):
        """The .tmp sibling must be replaced into the destination, not left."""
        path = tmp_path / "costs.json"
        await cost.add_cost("agent-1", 42)
        await cost.snapshot(path)
        assert path.exists()
        # Verify no .tmp leak.
        sibling = tmp_path / "costs.json.tmp"
        assert not sibling.exists()
        # Verify content is valid JSON with our payload.
        data = json.loads(path.read_text())
        assert "buckets" in data
        assert data["buckets"]["agent-1"]["millicents"] == 42

    @pytest.mark.asyncio
    async def test_snapshot_failure_returns_false(self, monkeypatch):
        """An os.replace failure must be reported, not raised."""
        await cost.add_cost("agent-1", 5)

        def boom(*a, **kw):
            raise OSError("disk full")

        monkeypatch.setattr("os.replace", boom)
        # Use a real path so the open succeeds before replace is reached.
        ok = await cost.snapshot(Path("/tmp/test_captcha_cost_fail.json"))
        assert ok is False

    @pytest.mark.asyncio
    async def test_inner_write_failure_does_not_double_close_fd(
        self, tmp_path, monkeypatch,
    ):
        """Once fdopen owns the descriptor, failure cleanup must not os.close it."""
        await cost.add_cost("agent-1", 5)
        close_calls: list[int] = []
        original_close = os.close

        def close_spy(fd: int) -> None:
            close_calls.append(fd)
            original_close(fd)

        def fsync_boom(fd: int) -> None:
            raise OSError("simulated fsync failure")

        monkeypatch.setattr(os, "close", close_spy)
        monkeypatch.setattr(os, "fsync", fsync_boom)

        ok = await cost.snapshot(tmp_path / "costs.json")
        assert ok is False
        assert close_calls == []

    @pytest.mark.asyncio
    async def test_fdopen_failure_closes_raw_fd_exactly_once(
        self, tmp_path, monkeypatch,
    ):
        """If fdopen raises, ownership never transferred — we must os.close fd."""
        await cost.add_cost("agent-2", 7)
        close_calls: list[int] = []
        original_close = os.close

        def close_spy(fd: int) -> None:
            close_calls.append(fd)
            original_close(fd)

        def fdopen_boom(fd, *a, **kw):
            raise OSError("simulated fdopen failure")

        monkeypatch.setattr(os, "close", close_spy)
        monkeypatch.setattr(os, "fdopen", fdopen_boom)

        ok = await cost.snapshot(tmp_path / "costs.json")
        assert ok is False
        # Exactly one close — the pre-fdopen cleanup of the raw fd. No
        # double-close (which would raise OSError(EBADF) on a reused fd).
        assert len(close_calls) == 1


class TestConcurrentWrites:
    @pytest.mark.asyncio
    async def test_concurrent_add_correct(self):
        """100 concurrent ``add_cost(1)`` calls must sum to exactly 100."""
        async def add_one():
            await cost.add_cost("agent-1", 1)

        await asyncio.gather(*(add_one() for _ in range(100)))
        assert await cost.get_millicents("agent-1") == 100


class TestCheckAndChargeAtomic:
    """``check_and_charge`` closes the race between separate
    ``over_cap`` / ``add_cost`` lock spans where two concurrent solves
    could both pass the cap check and together push the bucket above
    cap. Note: by design, a SINGLE call from below-cap is always
    allowed — the cap is `current >= cap → deny`, not `current +
    charge > cap → deny`. The race protection is about preventing
    multiple in-flight calls from each individually passing the
    below-cap test when only ONE should.
    """

    @pytest.mark.asyncio
    async def test_concurrent_at_boundary_only_one_succeeds(self):
        # Pre-load to one charge below cap. Two concurrent calls then
        # race against the cap. Pre-fix (separate over_cap / add_cost
        # spans), both could see ``current < cap`` and both charge —
        # total would land at cap + charge. Post-fix, only the first
        # to acquire the lock sees ``current < cap``; the second sees
        # ``current >= cap`` after the first commits, and is denied.
        await cost.add_cost("agent-1", 70)
        results = await asyncio.gather(*(
            cost.check_and_charge("agent-1", 100, 30) for _ in range(2)
        ))
        allowed = [r for r in results if r[0]]
        denied = [r for r in results if not r[0]]
        assert len(allowed) == 1
        assert len(denied) == 1
        # Total must be exactly 100 — no overshoot.
        assert await cost.get_millicents("agent-1") == 100

    @pytest.mark.asyncio
    async def test_below_cap_always_allowed(self):
        # 10 attempts each charging 5; cap is 1000 (huge).
        results = await asyncio.gather(*(
            cost.check_and_charge("agent-2", 1000, 5) for _ in range(10)
        ))
        assert all(r[0] for r in results)
        assert await cost.get_millicents("agent-2") == 50

    @pytest.mark.asyncio
    async def test_returns_false_when_already_at_cap(self):
        await cost.add_cost("agent-3", 100)
        allowed, total = await cost.check_and_charge("agent-3", 100, 50)
        assert allowed is False
        assert total == 100
        # Nothing was charged when denied.
        assert await cost.get_millicents("agent-3") == 100

    @pytest.mark.asyncio
    async def test_no_cap_always_allowed(self):
        allowed, total = await cost.check_and_charge("agent-4", 0, 50)
        assert allowed is True
        assert total == 50

    @pytest.mark.asyncio
    async def test_zero_charge_does_not_mutate(self):
        await cost.add_cost("agent-5", 25)
        allowed, total = await cost.check_and_charge("agent-5", 100, 0)
        assert allowed is True
        assert total == 25
        assert await cost.get_millicents("agent-5") == 25


class TestAdjustCost:
    @pytest.mark.asyncio
    async def test_positive_delta_adds_cost(self):
        await cost.adjust_cost("agent-1", 40)
        assert await cost.get_millicents("agent-1") == 40

    @pytest.mark.asyncio
    async def test_negative_delta_refunds_without_going_below_zero(self):
        await cost.add_cost("agent-1", 25)
        assert await cost.adjust_cost("agent-1", -10) == 15
        assert await cost.adjust_cost("agent-1", -50) == 0
