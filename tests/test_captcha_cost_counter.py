"""Tests for the simplified CAPTCHA cost counter (Phase 8 §11.10 trim).

Covers:
  * pricing-table lookup (case-insensitive, unknown variant → None)
  * month rollover resets the bucket
  * snapshot / restore round-trip via JSON
  * atomic write (tmp file replaced)
  * concurrent ``add_cost`` correctness
"""

from __future__ import annotations

import asyncio
import json
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


class TestEstimateCents:
    def test_known_2captcha_recaptcha_v2(self):
        assert cost.estimate_cents("2captcha", "recaptcha-v2-checkbox") == 100

    def test_known_capsolver_turnstile(self):
        assert cost.estimate_cents("capsolver", "turnstile") == 60

    def test_unknown_kind_returns_none(self):
        assert cost.estimate_cents("2captcha", "made-up-variant") is None

    def test_unknown_provider_returns_none(self):
        assert cost.estimate_cents("nopal", "hcaptcha") is None

    def test_case_insensitive(self):
        assert cost.estimate_cents("2CAPTCHA", "HCAPTCHA") == 100
        assert cost.estimate_cents("CapSolver", "Turnstile") == 60

    def test_empty_inputs_safe(self):
        assert cost.estimate_cents("", "hcaptcha") is None
        assert cost.estimate_cents("2captcha", "") is None


class TestAddCostAndOverCap:
    @pytest.mark.asyncio
    async def test_add_cost_accumulates(self):
        await cost.add_cost("agent-1", 100)
        await cost.add_cost("agent-1", 50)
        assert await cost.get_cents("agent-1") == 150

    @pytest.mark.asyncio
    async def test_add_cost_zero_or_negative_dropped(self):
        await cost.add_cost("agent-1", 0)
        await cost.add_cost("agent-1", -10)
        assert await cost.get_cents("agent-1") == 0

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
        assert await cost.get_cents("agent-2") == 0


class TestMonthRollover:
    @pytest.mark.asyncio
    async def test_bucket_resets_on_month_change(self, monkeypatch):
        # Spend 200 in January.
        monkeypatch.setattr(cost, "_current_month", lambda: "2026-01")
        await cost.add_cost("agent-1", 200)
        assert await cost.get_cents("agent-1") == 200

        # Roll over to February — bucket should reset.
        monkeypatch.setattr(cost, "_current_month", lambda: "2026-02")
        assert await cost.get_cents("agent-1") == 0
        await cost.add_cost("agent-1", 50)
        assert await cost.get_cents("agent-1") == 50

    @pytest.mark.asyncio
    async def test_month_rollover_during_over_cap(self, monkeypatch):
        monkeypatch.setattr(cost, "_current_month", lambda: "2026-01")
        await cost.add_cost("agent-1", 1500)
        assert await cost.over_cap("agent-1", 1000) is True

        monkeypatch.setattr(cost, "_current_month", lambda: "2026-02")
        assert await cost.over_cap("agent-1", 1000) is False


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
        assert await cost.get_cents("agent-a") == 0
        loaded = await cost.restore(path)
        assert loaded == 2
        assert await cost.get_cents("agent-a") == 200
        assert await cost.get_cents("agent-b") == 350

    @pytest.mark.asyncio
    async def test_restore_drops_stale_month(self, tmp_path, monkeypatch):
        path = tmp_path / "costs.json"
        # Write a snapshot stamped for last month.
        payload = {
            "version": 1,
            "saved_at": 0,
            "buckets": {
                "agent-old": {"month": "2020-01", "cents": 999},
            },
        }
        path.write_text(json.dumps(payload))
        loaded = await cost.restore(path)
        # Stale month skipped.
        assert loaded == 0
        assert await cost.get_cents("agent-old") == 0

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
        assert data["buckets"]["agent-1"]["cents"] == 42

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


class TestConcurrentWrites:
    @pytest.mark.asyncio
    async def test_concurrent_add_correct(self):
        """100 concurrent ``add_cost(1)`` calls must sum to exactly 100."""
        async def add_one():
            await cost.add_cost("agent-1", 1)

        await asyncio.gather(*(add_one() for _ in range(100)))
        assert await cost.get_cents("agent-1") == 100
