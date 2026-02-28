"""Tests for CostTracker: recording, budgets, spend queries."""

from __future__ import annotations

import os
import shutil
import tempfile
from unittest.mock import patch

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
        result = self.tracker.track("agent1", "openai/gpt-4o", 1000, 500)
        assert result["cost"] > 0
        assert result["over_budget"] is False
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


class TestBudgetOverrunWarning:
    """Post-hoc budget warning when track() pushes spend over budget."""

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self._tmpdir, "costs.db")
        self.tracker = CostTracker(db_path=self.db_path)

    def teardown_method(self):
        self.tracker.close()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_warning_logged_on_overrun(self):
        """track() logs a warning when the call pushes spend over daily budget."""
        self.tracker.set_budget("agent1", daily_usd=0.001, monthly_usd=200.0)
        with patch("src.host.costs.logger") as mock_logger:
            self.tracker.track("agent1", "openai/gpt-4o", 10000, 5000)
            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args
            assert "exceeded daily budget" in call_args[0][0]
            assert "agent1" in call_args[0][1]

    def test_no_warning_within_budget(self):
        """track() does NOT warn when spend stays under budget."""
        self.tracker.set_budget("agent1", daily_usd=100.0, monthly_usd=2000.0)
        with patch("src.host.costs.logger") as mock_logger:
            self.tracker.track("agent1", "openai/gpt-4o-mini", 100, 50)
            mock_logger.warning.assert_not_called()

    def test_no_warning_without_budget(self):
        """track() does NOT warn when no explicit budget is set."""
        with patch("src.host.costs.logger") as mock_logger:
            self.tracker.track("agent1", "openai/gpt-4o-mini", 100, 50)
            mock_logger.warning.assert_not_called()

    def test_track_returns_over_budget_true(self):
        """track() returns over_budget=True when daily budget exceeded."""
        self.tracker.set_budget("agent1", daily_usd=0.001, monthly_usd=200.0)
        result = self.tracker.track("agent1", "openai/gpt-4o", 10000, 5000)
        assert result["over_budget"] is True
        assert result["cost"] > 0

    def test_track_returns_over_budget_false_within_budget(self):
        """track() returns over_budget=False when within budget."""
        self.tracker.set_budget("agent1", daily_usd=100.0, monthly_usd=2000.0)
        result = self.tracker.track("agent1", "openai/gpt-4o-mini", 100, 50)
        assert result["over_budget"] is False

    def test_track_returns_over_budget_true_monthly(self):
        """track() returns over_budget=True when monthly budget exceeded."""
        self.tracker.set_budget("agent1", daily_usd=100.0, monthly_usd=0.001)
        result = self.tracker.track("agent1", "openai/gpt-4o", 10000, 5000)
        assert result["over_budget"] is True

    def test_monthly_warning_logged(self):
        """track() logs a warning when monthly budget exceeded."""
        self.tracker.set_budget("agent1", daily_usd=100.0, monthly_usd=0.001)
        with patch("src.host.costs.logger") as mock_logger:
            self.tracker.track("agent1", "openai/gpt-4o", 10000, 5000)
            # Should warn about monthly, not daily
            calls = mock_logger.warning.call_args_list
            assert any("exceeded monthly budget" in str(c) for c in calls)


class TestProjectCostAggregation:
    """Project-level cost tracking and budget enforcement."""

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self._tmpdir, "costs.db")
        self.tracker = CostTracker(db_path=self.db_path)

    def teardown_method(self):
        self.tracker.close()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_project_spend_aggregates_members(self):
        """get_project_spend sums spend across all member agents."""
        self.tracker.set_project_budget(
            "teamA", members=["alice", "bob"], daily_usd=50.0, monthly_usd=500.0,
        )
        self.tracker.track("alice", "openai/gpt-4o-mini", 1000, 500)
        self.tracker.track("bob", "openai/gpt-4o-mini", 2000, 1000)

        result = self.tracker.get_project_spend("teamA", "today")
        assert result["project"] == "teamA"
        assert result["total_tokens"] == 4500
        assert result["total_cost"] > 0
        assert len(result["agents"]) == 2
        alice_spend = next(a for a in result["agents"] if a["agent"] == "alice")
        bob_spend = next(a for a in result["agents"] if a["agent"] == "bob")
        assert alice_spend["tokens"] == 1500
        assert bob_spend["tokens"] == 3000

    def test_project_spend_no_budget_configured(self):
        """get_project_spend returns error when no budget is set."""
        result = self.tracker.get_project_spend("unknown", "today")
        assert "error" in result

    def test_project_budget_limits(self):
        """set_project_budget stores limits correctly."""
        self.tracker.set_project_budget(
            "proj", members=["a1"], daily_usd=100.0, monthly_usd=2000.0,
        )
        result = self.tracker.get_project_spend("proj", "today")
        assert result["daily_limit"] == 100.0
        assert result["monthly_limit"] == 2000.0


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
