"""Tests for CostTracker: recording, budgets, spend queries."""

from __future__ import annotations

import os
import shutil
import tempfile

from src.host.costs import CostTracker


class TestCostTracking:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self._tmpdir, "costs.db")
        self.tracker = CostTracker(db_path=self.db_path)

    def teardown_method(self):
        self.tracker.close()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_track_records_usage(self):
        cost = self.tracker.track("agent1", "openai/gpt-4o", 1000, 500)
        assert cost > 0
        spend = self.tracker.get_spend("agent1", "today")
        assert spend["total_tokens"] == 1500
        assert spend["total_cost"] > 0

    def test_track_multiple_calls(self):
        self.tracker.track("agent1", "openai/gpt-4o-mini", 100, 50)
        self.tracker.track("agent1", "openai/gpt-4o-mini", 200, 100)
        spend = self.tracker.get_spend("agent1", "today")
        assert spend["total_tokens"] == 450
        assert spend["total_cost"] > 0

    def test_track_different_agents(self):
        self.tracker.track("agent1", "openai/gpt-4o-mini", 100, 50)
        self.tracker.track("agent2", "openai/gpt-4o-mini", 200, 100)
        s1 = self.tracker.get_spend("agent1", "today")
        s2 = self.tracker.get_spend("agent2", "today")
        assert s1["total_tokens"] == 150
        assert s2["total_tokens"] == 300

    def test_track_different_models(self):
        self.tracker.track("agent1", "openai/gpt-4o-mini", 100, 50)
        self.tracker.track("agent1", "openai/gpt-4o", 100, 50)
        spend = self.tracker.get_spend("agent1", "today")
        assert len(spend["by_model"]) == 2
        assert "openai/gpt-4o-mini" in spend["by_model"]
        assert "openai/gpt-4o" in spend["by_model"]

    def test_get_all_agents_spend(self):
        self.tracker.track("agent1", "openai/gpt-4o-mini", 100, 50)
        self.tracker.track("agent2", "openai/gpt-4o-mini", 200, 100)
        agents = self.tracker.get_all_agents_spend("today")
        assert len(agents) == 2
        names = {a["agent"] for a in agents}
        assert names == {"agent1", "agent2"}


class TestBudgetEnforcement:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self._tmpdir, "costs.db")
        self.tracker = CostTracker(db_path=self.db_path)

    def teardown_method(self):
        self.tracker.close()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_within_budget(self):
        self.tracker.set_budget("agent1", daily_usd=10.0, monthly_usd=200.0)
        self.tracker.track("agent1", "openai/gpt-4o-mini", 100, 50)
        budget = self.tracker.check_budget("agent1")
        assert budget["allowed"] is True
        assert budget["daily_used"] < budget["daily_limit"]

    def test_exceeds_daily_budget(self):
        self.tracker.set_budget("agent1", daily_usd=0.001, monthly_usd=200.0)
        self.tracker.track("agent1", "openai/gpt-4o", 10000, 5000)
        budget = self.tracker.check_budget("agent1")
        assert budget["allowed"] is False

    def test_default_budget(self):
        budget = self.tracker.check_budget("unknown_agent")
        assert budget["allowed"] is True
        assert budget["daily_limit"] == 10.0
        assert budget["monthly_limit"] == 200.0

    def test_no_spend_is_allowed(self):
        self.tracker.set_budget("agent1", daily_usd=5.0, monthly_usd=100.0)
        budget = self.tracker.check_budget("agent1")
        assert budget["allowed"] is True
        assert budget["daily_used"] == 0


class TestCostTrackerWithVault:
    """Verify CostTracker integrates with CredentialVault."""

    def test_vault_accepts_cost_tracker(self):
        tmpdir = tempfile.mkdtemp()
        try:
            tracker = CostTracker(db_path=os.path.join(tmpdir, "costs.db"))
            from src.host.credentials import CredentialVault
            vault = CredentialVault(cost_tracker=tracker)
            assert vault.cost_tracker is tracker
            tracker.close()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
