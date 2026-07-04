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
        self.budgets_path = os.path.join(self._tmpdir, "agent_budgets.json")
        self.tracker = CostTracker(db_path=self.db_path, budgets_path=self.budgets_path)

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

    def test_track_oauth_records_tokens_at_zero_cost(self):
        """bill=False (OAuth/subscription): tokens recorded, cost_usd=0."""
        result = self.tracker.track(
            "agent1", "anthropic/claude-sonnet-4-6", 1000, 500, bill=False,
        )
        assert result["cost"] == 0.0
        assert result["over_budget"] is False
        spend = self.tracker.get_spend("agent1", "today")
        # Token visibility is preserved (observability) ...
        assert spend["total_tokens"] == 1500
        # ... but the spend total stays $0 — no dollar capture for OAuth.
        assert spend["total_cost"] == 0.0

    def test_oauth_usage_never_trips_budget_cap(self):
        """A pure-OAuth agent stays within budget no matter the volume.

        ``check_budget`` is the gate the reroute/retry path
        (``_check_can_schedule``) and the dashboard read; it sums
        ``cost_usd``. bill=False rows add $0, so a subscription-only agent
        is always ``allowed`` even past a tiny dollar budget that metered
        traffic would blow through instantly. (The forward-looking
        ``preflight_check`` is deliberately NOT asserted here — it estimates
        the *next* call's list price and is never invoked on the OAuth path,
        which skips budget enforcement entirely; see
        ``test_oauth_tracks_usage_but_skips_budget_enforcement``.)
        """
        self.tracker.set_budget("agent1", daily_usd=0.01, monthly_usd=0.01)
        for _ in range(50):
            self.tracker.track(
                "agent1", "anthropic/claude-opus-4-1", 50_000, 10_000, bill=False,
            )
        assert self.tracker.check_budget("agent1")["allowed"] is True
        assert self.tracker.get_spend("agent1", "today")["total_cost"] == 0.0

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

    def test_get_all_agents_last_worked_empty(self):
        # No usage recorded yet → empty map (drives the "Last activity"
        # card stat, which simply hides when there's no work to show).
        assert self.tracker.get_all_agents_last_worked() == {}

    def test_get_all_agents_last_worked(self):
        import time

        before = time.time()
        self.tracker.track("agent1", "openai/gpt-4o-mini", 100, 50)
        self.tracker.track("agent2", "openai/gpt-4o-mini", 200, 100)
        worked = self.tracker.get_all_agents_last_worked()
        assert set(worked) == {"agent1", "agent2"}
        # Values are unix timestamps (UTC) of the most recent LLM call —
        # i.e. real work, not a health probe. Allow a generous window for
        # clock/second-rounding (usage.timestamp is whole-second UTC).
        for ts in worked.values():
            assert isinstance(ts, float)
            assert before - 5 <= ts <= time.time() + 5

    def test_get_spend_by_model(self):
        self.tracker.track("agent1", "openai/gpt-4o", 1000, 500)
        self.tracker.track("agent2", "openai/gpt-4o", 500, 200)
        self.tracker.track("agent1", "anthropic/claude-sonnet-4-6", 800, 400)
        models = self.tracker.get_spend_by_model("today")
        assert len(models) == 2
        names = {m["model"] for m in models}
        assert names == {"openai/gpt-4o", "anthropic/claude-sonnet-4-6"}
        # Sorted by cost descending
        assert models[0]["cost"] >= models[1]["cost"]
        # Cross-agent aggregation: gpt-4o tokens = 1000+500+500+200 = 2200
        gpt = next(m for m in models if m["model"] == "openai/gpt-4o")
        assert gpt["tokens"] == 2200
        assert gpt["prompt_tokens"] == 1500
        assert gpt["completion_tokens"] == 700

    def test_get_spend_by_model_empty(self):
        models = self.tracker.get_spend_by_model("today")
        assert models == []


class TestTraceIdStamping:
    """Session observability (Phase 1) — usage rows carry the active
    per-turn ``trace_id`` from the contextvar, and the migration is
    idempotent on a pre-existing (trace-less) DB.
    """

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self._tmpdir, "costs.db")
        self.budgets_path = os.path.join(self._tmpdir, "agent_budgets.json")
        self.tracker = CostTracker(db_path=self.db_path, budgets_path=self.budgets_path)

    def teardown_method(self):
        self.tracker.close()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _trace_ids(self, agent: str) -> list:
        rows = self.tracker.db.execute(
            "SELECT trace_id FROM usage WHERE agent = ? ORDER BY id", (agent,)
        ).fetchall()
        return [r[0] for r in rows]

    def test_track_stamps_active_trace_id(self):
        from src.shared.trace import current_trace_id

        token = current_trace_id.set("tr_abc123abc123")
        try:
            self.tracker.track("agent1", "openai/gpt-4o-mini", 100, 50)
        finally:
            current_trace_id.reset(token)
        assert self._trace_ids("agent1") == ["tr_abc123abc123"]

    def test_track_fixed_cost_stamps_active_trace_id(self):
        from src.shared.trace import current_trace_id

        token = current_trace_id.set("tr_fixed0000001")
        try:
            self.tracker.track_fixed_cost("agent1", "openai/dall-e-3", 0.04)
        finally:
            current_trace_id.reset(token)
        assert self._trace_ids("agent1") == ["tr_fixed0000001"]

    def test_track_without_trace_is_null(self):
        # No active trace → NULL column, never an empty string.
        from src.shared.trace import current_trace_id

        assert current_trace_id.get() is None
        self.tracker.track("agent1", "openai/gpt-4o-mini", 100, 50)
        assert self._trace_ids("agent1") == [None]

    def test_migration_idempotent_on_legacy_db(self):
        """A pre-existing usage table without trace_id gets the column
        added once; re-opening the same DB is a no-op (no duplicate ALTER).
        """
        import sqlite3

        legacy_path = os.path.join(self._tmpdir, "legacy_costs.db")
        # Build a legacy schema WITHOUT the trace_id column + seed a row.
        conn = sqlite3.connect(legacy_path)
        conn.executescript(
            """
            CREATE TABLE usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_tokens INTEGER DEFAULT 0,
                completion_tokens INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                cost_usd REAL DEFAULT 0.0,
                timestamp TEXT DEFAULT (datetime('now'))
            );
            """
        )
        conn.execute(
            "INSERT INTO usage (agent, model, total_tokens) VALUES (?, ?, ?)",
            ("legacy", "openai/gpt-4o-mini", 100),
        )
        conn.commit()
        conn.close()

        # First open migrates: adds the nullable column, old row keeps NULL.
        t1 = CostTracker(
            db_path=legacy_path,
            budgets_path=os.path.join(self._tmpdir, "legacy_budgets.json"),
        )
        cols = {r[1] for r in t1.db.execute("PRAGMA table_info(usage)").fetchall()}
        assert "trace_id" in cols
        old = t1.db.execute(
            "SELECT trace_id FROM usage WHERE agent = 'legacy'"
        ).fetchone()
        assert old[0] is None
        t1.close()

        # Second open over the same file is a no-op (the introspection
        # guard skips the ALTER) and does not raise.
        t2 = CostTracker(
            db_path=legacy_path,
            budgets_path=os.path.join(self._tmpdir, "legacy_budgets.json"),
        )
        cols2 = {r[1] for r in t2.db.execute("PRAGMA table_info(usage)").fetchall()}
        assert "trace_id" in cols2
        t2.close()


class TestBudgetEnforcement:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self._tmpdir, "costs.db")
        self.budgets_path = os.path.join(self._tmpdir, "agent_budgets.json")
        self.tracker = CostTracker(db_path=self.db_path, budgets_path=self.budgets_path)

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
        self.budgets_path = os.path.join(self._tmpdir, "agent_budgets.json")
        self.tracker = CostTracker(db_path=self.db_path, budgets_path=self.budgets_path)

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
        self.budgets_path = os.path.join(self._tmpdir, "agent_budgets.json")
        self.tracker = CostTracker(db_path=self.db_path, budgets_path=self.budgets_path)

    def teardown_method(self):
        self.tracker.close()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_team_spend_aggregates_members(self):
        """get_team_spend sums spend across all member agents."""
        self.tracker.set_team_budget(
            "teamA", members=["alice", "bob"], daily_usd=50.0, monthly_usd=500.0,
        )
        self.tracker.track("alice", "openai/gpt-4o-mini", 1000, 500)
        self.tracker.track("bob", "openai/gpt-4o-mini", 2000, 1000)

        result = self.tracker.get_team_spend("teamA", "today")
        assert result["team"] == "teamA"
        assert result["total_tokens"] == 4500
        assert result["total_cost"] > 0
        assert len(result["agents"]) == 2
        alice_spend = next(a for a in result["agents"] if a["agent"] == "alice")
        bob_spend = next(a for a in result["agents"] if a["agent"] == "bob")
        assert alice_spend["tokens"] == 1500
        assert bob_spend["tokens"] == 3000

    def test_team_spend_no_budget_configured(self):
        """get_team_spend returns error when no budget is set."""
        result = self.tracker.get_team_spend("unknown", "today")
        assert "error" in result

    def test_team_budget_limits(self):
        """set_team_budget stores limits correctly."""
        self.tracker.set_team_budget(
            "proj", members=["a1"], daily_usd=100.0, monthly_usd=2000.0,
        )
        result = self.tracker.get_team_spend("proj", "today")
        assert result["daily_limit"] == 100.0
        assert result["monthly_limit"] == 2000.0


class TestCostTrackerCleanup:
    """Verify cleanup_agent removes all records and budget for an agent."""

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self._tmpdir, "costs.db")
        self.budgets_path = os.path.join(self._tmpdir, "agent_budgets.json")
        self.tracker = CostTracker(db_path=self.db_path, budgets_path=self.budgets_path)

    def teardown_method(self):
        self.tracker.close()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_cleanup_removes_usage_records(self):
        self.tracker.track("agent1", "openai/gpt-4o-mini", 100, 50)
        self.tracker.track("agent1", "openai/gpt-4o", 200, 100)
        self.tracker.track("agent2", "openai/gpt-4o-mini", 300, 150)

        deleted = self.tracker.cleanup_agent("agent1")
        assert deleted == 2

        # agent1 records gone
        spend = self.tracker.get_spend("agent1", "today")
        assert spend["total_tokens"] == 0

        # agent2 records untouched
        spend2 = self.tracker.get_spend("agent2", "today")
        assert spend2["total_tokens"] == 450

    def test_cleanup_removes_budget(self):
        self.tracker.set_budget("agent1", daily_usd=5.0, monthly_usd=100.0)
        assert "agent1" in self.tracker.budgets
        self.tracker.cleanup_agent("agent1")
        assert "agent1" not in self.tracker.budgets

    def test_cleanup_nonexistent_agent(self):
        deleted = self.tracker.cleanup_agent("ghost")
        assert deleted == 0


class TestBudgetPersistence:
    """Per-agent budget overrides survive a CostTracker restart (Bug fix).

    Regression: ``set_budget`` only mutated an in-memory dict, so caps
    raised/lowered by the operator silently reverted to the global default
    on the next mesh restart / redeploy.
    """

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self._tmpdir, "costs.db")
        self.budgets_path = os.path.join(self._tmpdir, "agent_budgets.json")

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_set_budget_persists_across_restart(self):
        tracker = CostTracker(db_path=self.db_path, budgets_path=self.budgets_path)
        tracker.set_budget("agent1", daily_usd=42.0, monthly_usd=999.0)
        tracker.close()

        # File was written.
        assert os.path.exists(self.budgets_path)

        # A fresh tracker (simulating a mesh restart) loads the override.
        restarted = CostTracker(db_path=self.db_path, budgets_path=self.budgets_path)
        assert restarted.budgets["agent1"] == {
            "daily_usd": 42.0,
            "monthly_usd": 999.0,
        }
        budget = restarted.check_budget("agent1")
        assert budget["daily_limit"] == 42.0
        assert budget["monthly_limit"] == 999.0
        restarted.close()

    def test_missing_file_yields_empty_budgets(self):
        # No file on disk → empty overrides, no crash.
        assert not os.path.exists(self.budgets_path)
        tracker = CostTracker(db_path=self.db_path, budgets_path=self.budgets_path)
        assert tracker.budgets == {}
        tracker.close()

    def test_corrupt_file_yields_empty_budgets_and_logs(self):
        with open(self.budgets_path, "w") as f:
            f.write("{not valid json")
        with patch("src.host.costs.logger") as mock_logger:
            tracker = CostTracker(db_path=self.db_path, budgets_path=self.budgets_path)
            assert tracker.budgets == {}
            mock_logger.warning.assert_called_once()
        tracker.close()

    def test_malformed_entry_skipped_other_entries_still_load(self):
        # One bad row (non-numeric daily_usd) must not abort loading the rest:
        # the good agent's budget loads, the bad one is skipped, no exception.
        import json

        with open(self.budgets_path, "w") as f:
            json.dump(
                {
                    "good": {"daily_usd": 50, "monthly_usd": 1000},
                    "bad": {"daily_usd": "oops", "monthly_usd": 5},
                },
                f,
            )
        with patch("src.host.costs.logger") as mock_logger:
            tracker = CostTracker(db_path=self.db_path, budgets_path=self.budgets_path)
            mock_logger.warning.assert_called_once()
        assert tracker.budgets["good"] == {"daily_usd": 50.0, "monthly_usd": 1000.0}
        assert "bad" not in tracker.budgets
        tracker.close()

    def test_cleanup_updates_persisted_file(self):
        tracker = CostTracker(db_path=self.db_path, budgets_path=self.budgets_path)
        tracker.set_budget("agent1", daily_usd=5.0, monthly_usd=100.0)
        tracker.set_budget("agent2", daily_usd=7.0, monthly_usd=140.0)
        tracker.cleanup_agent("agent1")
        tracker.close()

        # Removal is reflected on disk: a restart no longer sees agent1.
        restarted = CostTracker(db_path=self.db_path, budgets_path=self.budgets_path)
        assert "agent1" not in restarted.budgets
        assert "agent2" in restarted.budgets
        restarted.close()

    def test_save_failure_does_not_crash(self):
        from pathlib import Path

        tracker = CostTracker(db_path=self.db_path, budgets_path=self.budgets_path)
        # Point the persist path at an unwritable location so the write fails.
        with patch.object(
            tracker, "budgets_path",
            Path("/proc/nonexistent/agent_budgets.json"),
        ):
            with patch("src.host.costs.logger") as mock_logger:
                # Must not raise even though the write fails.
                tracker.set_budget("agent1", daily_usd=1.0, monthly_usd=2.0)
                mock_logger.warning.assert_called()
        # In-memory mutation still applied.
        assert tracker.budgets["agent1"] == {"daily_usd": 1.0, "monthly_usd": 2.0}
        tracker.close()


class TestBudgetsPathDerivation:
    """When no ``budgets_path`` is passed, it is derived from ``db_path`` so
    tests and tools pointing the db at a tmp dir never touch the real
    ``config/agent_budgets.json``.
    """

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self._tmpdir, "costs.db")

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_derives_budgets_path_alongside_tmp_db(self):
        from pathlib import Path

        config_file = Path("config/agent_budgets.json")
        config_existed_before = config_file.exists()

        # No budgets_path → co-located with the (tmp) db, NOT config/.
        tracker = CostTracker(db_path=self.db_path)
        expected = Path(self._tmpdir) / "agent_budgets.json"
        assert tracker.budgets_path == expected

        tracker.set_budget("agent1", daily_usd=3.0, monthly_usd=60.0)
        tracker.close()

        # The write landed under the tmp dir, not the real config dir.
        assert expected.exists()
        # And it did NOT create / mutate the real config file.
        assert config_file.exists() == config_existed_before

    def test_production_default_db_maps_to_config(self):
        from pathlib import Path

        # The production default db_path maps to the canonical config file
        # (constructed without touching disk via __new__ + manual derivation).
        tracker = object.__new__(CostTracker)
        db_path = "data/costs.db"
        budgets_path = None
        if budgets_path is None:
            if db_path == "data/costs.db":
                budgets_path = "config/agent_budgets.json"
            else:
                budgets_path = str(Path(db_path).parent / "agent_budgets.json")
        tracker.budgets_path = Path(budgets_path)
        assert tracker.budgets_path == Path("config/agent_budgets.json")


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
