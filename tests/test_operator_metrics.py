"""Tests for pre-computed metrics endpoints used by the operator heartbeat."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.host.costs import CostTracker
from src.host.health import HealthMonitor
from src.host.mesh import Blackboard, MessageRouter, PubSub
from src.host.permissions import PermissionMatrix
from src.shared.types import AgentPermissions


def _make_perms(*agent_ids: str) -> PermissionMatrix:
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {
        aid: AgentPermissions(
            agent_id=aid,
            can_message=["*"],
            blackboard_read=["*"],
            blackboard_write=["*"],
            allowed_apis=[],
        )
        for aid in agent_ids
    }
    return perms


@pytest.fixture
def metrics_app(tmp_path, monkeypatch):
    """Build a mesh app with health monitor, cost tracker, and lane manager."""
    # PR-J' — point tasks_v2 at a per-test sqlite file so the stale /
    # outcome / failure counts surface in ``system_metrics`` without
    # touching ``data/tasks.db`` on the developer's machine.
    monkeypatch.setenv("OPENLEGION_ORCHESTRATION_TASKS_V2", "1")
    monkeypatch.setenv(
        "OPENLEGION_ORCHESTRATION_TASKS_DB", str(tmp_path / "tasks.db"),
    )
    import importlib

    import src.host.server as server_module
    importlib.reload(server_module)

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = _make_perms("alpha", "beta")
    router = MessageRouter(permissions=perms, agent_registry={})

    # Register agents
    router.register_agent("alpha", "http://localhost:8401")
    router.register_agent("beta", "http://localhost:8402")

    # Health monitor
    runtime_mock = MagicMock()
    transport_mock = MagicMock()
    health = HealthMonitor(
        runtime=runtime_mock, transport=transport_mock, router=router,
    )
    health.register("alpha")
    health.register("beta")
    health.agents["alpha"].status = "healthy"
    health.agents["beta"].status = "failed"
    health.agents["beta"].consecutive_failures = 3
    health.agents["beta"].restart_count = 2

    # Cost tracker
    cost_tracker = CostTracker(db_path=str(tmp_path / "costs.db"))

    # Lane manager mock
    lane_manager = MagicMock()
    lane_manager.get_status.return_value = {
        "alpha": {"queued": 2, "busy": True, "pending": 0, "collected": 0},
        "beta": {"queued": 0, "busy": False, "pending": 0, "collected": 0},
    }

    app = server_module.create_mesh_app(
        bb, pubsub, router, perms,
        health_monitor=health,
        cost_tracker=cost_tracker,
        lane_manager=lane_manager,
    )
    client = TestClient(app)

    yield {
        "client": client,
        "bb": bb,
        "cost_tracker": cost_tracker,
        "health": health,
        "router": router,
        "lane_manager": lane_manager,
        "tasks_store": getattr(app, "tasks_store", None),
    }

    cost_tracker.close()
    bb.close()
    # Reload the module so the env var change doesn't leak into other
    # tests that run later in the session.
    monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_V2", raising=False)
    monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_DB", raising=False)
    importlib.reload(server_module)


class TestSystemMetrics:
    """Tests for GET /mesh/system/metrics."""

    def test_basic_response(self, metrics_app):
        client = metrics_app["client"]
        resp = client.get("/mesh/system/metrics")
        assert resp.status_code == 200

        data = resp.json()
        assert data["total_agents"] == 2
        assert data["healthy"] == 1
        assert data["failed"] == 1
        assert data["busy"] == 1  # alpha is busy
        assert "total_cost_today_usd" in data
        assert "cost_vs_yesterday_ratio" in data
        assert "failure_rate_by_agent" in data
        assert "agents_needing_attention" in data
        assert "plan_limits" in data

    def test_agents_needing_attention(self, metrics_app):
        client = metrics_app["client"]
        resp = client.get("/mesh/system/metrics")
        data = resp.json()

        attention = data["agents_needing_attention"]
        assert len(attention) == 1
        assert attention[0]["agent_id"] == "beta"
        assert attention[0]["issue"] == "failed"
        assert attention[0]["failures"] == 3
        assert attention[0]["restarts"] == 2

    def test_plan_limits(self, metrics_app):
        client = metrics_app["client"]
        # Set env vars for plan limits
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("OPENLEGION_MAX_AGENTS", "10")
            mp.setenv("OPENLEGION_MAX_PROJECTS", "5")
            resp = client.get("/mesh/system/metrics")

        data = resp.json()
        assert data["plan_limits"]["max_agents"] == 10
        assert data["plan_limits"]["max_projects"] == 5
        assert data["plan_limits"]["current_agents"] == 2

    def test_cost_tracking(self, metrics_app):
        client = metrics_app["client"]
        cost_tracker = metrics_app["cost_tracker"]

        # Record some cost today
        cost_tracker.track("alpha", "claude-3-5-sonnet-20241022", 1000, 500)

        resp = client.get("/mesh/system/metrics")
        data = resp.json()

        # Should have non-zero cost
        assert data["total_cost_today_usd"] > 0

    def test_all_healthy_no_attention(self, metrics_app):
        health = metrics_app["health"]
        # Make both agents healthy
        health.agents["beta"].status = "healthy"
        health.agents["beta"].consecutive_failures = 0

        client = metrics_app["client"]
        resp = client.get("/mesh/system/metrics")
        data = resp.json()

        assert data["healthy"] == 2
        assert data["failed"] == 0
        assert len(data["agents_needing_attention"]) == 0


class TestSystemMetricsPRJ:
    """PR-J' — per-agent cost + outcome + stale-task fields on system_metrics."""

    def test_new_fields_present_with_correct_shape(self, metrics_app):
        client = metrics_app["client"]
        resp = client.get("/mesh/system/metrics")
        data = resp.json()
        # All five new fields exist and are dicts.
        for field in (
            "per_agent_cost_today_usd",
            "per_agent_cost_vs_yesterday_ratio",
            "outcome_rejected_24h_count",
            "execution_failures_24h_count",
            "stale_tasks_24h_count",
        ):
            assert field in data, f"missing field {field!r}"
            assert isinstance(data[field], dict), f"{field} should be a dict"

    def test_per_agent_cost_populated_from_costs_store(self, metrics_app):
        cost_tracker = metrics_app["cost_tracker"]
        cost_tracker.track("alpha", "claude-3-5-sonnet-20241022", 1000, 500)

        client = metrics_app["client"]
        resp = client.get("/mesh/system/metrics")
        data = resp.json()
        # alpha got tracked spend; beta did not.
        assert data["per_agent_cost_today_usd"]["alpha"] > 0
        assert data["per_agent_cost_today_usd"]["beta"] == 0
        # PR-Q: when no yesterday baseline exists the ratio is ``None``
        # (not ``0.0``) so the heartbeat playbook can distinguish "agent
        # stopped spending today" (ratio == 0.0) from "no baseline".
        assert data["per_agent_cost_vs_yesterday_ratio"]["alpha"] is None
        assert data["per_agent_cost_vs_yesterday_ratio"]["beta"] is None

    def test_per_agent_cost_excludes_operator(self, metrics_app):
        # Register operator on the router so it shows up in agents.
        router = metrics_app["router"]
        router.register_agent("operator", "http://localhost:8400")
        cost_tracker = metrics_app["cost_tracker"]
        cost_tracker.track("operator", "claude-3-5-sonnet-20241022", 500, 200)

        client = metrics_app["client"]
        resp = client.get("/mesh/system/metrics")
        data = resp.json()
        # Operator must NOT appear in any per-agent dict (mirrors the
        # exclusion pattern used by total_agents / healthy / failed).
        assert "operator" not in data["per_agent_cost_today_usd"]
        assert "operator" not in data["per_agent_cost_vs_yesterday_ratio"]
        assert "operator" not in data["outcome_rejected_24h_count"]
        assert "operator" not in data["execution_failures_24h_count"]
        assert "operator" not in data["stale_tasks_24h_count"]

    def test_outcome_rejected_attribution(self, metrics_app):
        store = metrics_app["tasks_store"]
        assert store is not None, "tasks_store should be wired in PR-J' fixture"
        # alpha: 2 rejected, beta: 1 rejected, alpha: 1 accepted (filtered)
        for assignee, outcome in [
            ("alpha", "rejected"),
            ("alpha", "rejected"),
            ("beta", "rejected"),
            ("alpha", "accepted"),
        ]:
            rec = store.create(creator="op", assignee=assignee, title="t")
            store.update_status(rec["id"], "working", actor=assignee)
            store.update_status(rec["id"], "done", actor=assignee)
            store.set_outcome(rec["id"], outcome, actor="op")

        client = metrics_app["client"]
        resp = client.get("/mesh/system/metrics")
        data = resp.json()
        assert data["outcome_rejected_24h_count"] == {"alpha": 2, "beta": 1}

    def test_execution_failures_distinct_from_rejected(self, metrics_app):
        store = metrics_app["tasks_store"]
        assert store is not None
        # alpha: 1 failed; beta: 1 done+rejected (NOT execution failure)
        failed = store.create(creator="op", assignee="alpha", title="x")
        store.update_status(failed["id"], "failed", actor="alpha")

        rejected = store.create(creator="op", assignee="beta", title="y")
        store.update_status(rejected["id"], "working", actor="beta")
        store.update_status(rejected["id"], "done", actor="beta")
        store.set_outcome(rejected["id"], "rejected", actor="op")

        client = metrics_app["client"]
        resp = client.get("/mesh/system/metrics")
        data = resp.json()
        assert data["execution_failures_24h_count"] == {"alpha": 1}
        assert data["outcome_rejected_24h_count"] == {"beta": 1}

    def test_stale_tasks_excludes_terminal_and_fresh(self, metrics_app):
        import time as _time
        store = metrics_app["tasks_store"]
        assert store is not None
        # alpha: stale pending (created 2 days ago)
        stale = store.create(creator="op", assignee="alpha", title="stale")
        with store._conn() as conn:
            conn.execute(
                "UPDATE tasks SET created_at=? WHERE id=?",
                (_time.time() - 2 * 24 * 3600, stale["id"]),
            )
        # beta: stale done (created 2 days ago, then completed) — excluded
        done = store.create(creator="op", assignee="beta", title="done")
        with store._conn() as conn:
            conn.execute(
                "UPDATE tasks SET created_at=? WHERE id=?",
                (_time.time() - 2 * 24 * 3600, done["id"]),
            )
        store.update_status(done["id"], "working", actor="beta")
        store.update_status(done["id"], "done", actor="beta")
        # alpha: fresh pending — excluded
        store.create(creator="op", assignee="alpha", title="fresh")

        client = metrics_app["client"]
        resp = client.get("/mesh/system/metrics")
        data = resp.json()
        assert data["stale_tasks_24h_count"] == {"alpha": 1}

    def test_failure_rate_placeholder_still_present(self, metrics_app):
        """Legacy placeholder stays on the contract as ``{}`` for back-compat."""
        client = metrics_app["client"]
        resp = client.get("/mesh/system/metrics")
        data = resp.json()
        assert "failure_rate_by_agent" in data
        assert data["failure_rate_by_agent"] == {}


class TestStaleTasksEndpoint:
    """PR-J' — /mesh/agents/{agent_id}/stale-tasks for inspect_agents drill-in."""

    def test_returns_empty_when_no_stale(self, metrics_app):
        client = metrics_app["client"]
        resp = client.get("/mesh/agents/alpha/stale-tasks?threshold_hours=24")
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent_id"] == "alpha"
        assert data["threshold_hours"] == 24
        assert data["count"] == 0
        assert data["task_ids"] == []

    def test_returns_oldest_first_capped_at_5(self, metrics_app):
        import time as _time
        store = metrics_app["tasks_store"]
        assert store is not None
        ids = []
        base = _time.time() - 2 * 24 * 3600
        for i in range(7):
            rec = store.create(creator="op", assignee="alpha", title=f"t{i}")
            with store._conn() as conn:
                conn.execute(
                    "UPDATE tasks SET created_at=? WHERE id=?",
                    (base + i, rec["id"]),
                )
            ids.append(rec["id"])

        client = metrics_app["client"]
        resp = client.get("/mesh/agents/alpha/stale-tasks?threshold_hours=24")
        data = resp.json()
        assert data["count"] == 5
        assert data["task_ids"] == ids[:5]

    def test_invalid_threshold_400(self, metrics_app):
        client = metrics_app["client"]
        resp = client.get("/mesh/agents/alpha/stale-tasks?threshold_hours=0")
        assert resp.status_code == 400
        resp = client.get("/mesh/agents/alpha/stale-tasks?threshold_hours=999")
        assert resp.status_code == 400

    def test_invalid_agent_id_400(self, metrics_app):
        client = metrics_app["client"]
        resp = client.get("/mesh/agents/!!!bad!!!/stale-tasks?threshold_hours=24")
        assert resp.status_code == 400


class TestAgentMetrics:
    """Tests for GET /mesh/agents/{agent_id}/metrics."""

    def test_healthy_agent(self, metrics_app):
        client = metrics_app["client"]
        resp = client.get("/mesh/agents/alpha/metrics")
        assert resp.status_code == 200

        data = resp.json()
        assert data["agent_id"] == "alpha"
        assert data["health_status"] == "healthy"
        assert data["consecutive_failures"] == 0
        assert data["queued_tasks"] == 2
        assert data["busy"] is True
        assert "cost_today_usd" in data
        assert "budget" in data

    def test_failed_agent(self, metrics_app):
        client = metrics_app["client"]
        resp = client.get("/mesh/agents/beta/metrics")
        assert resp.status_code == 200

        data = resp.json()
        assert data["agent_id"] == "beta"
        assert data["health_status"] == "failed"
        assert data["consecutive_failures"] == 3
        assert data["restart_count"] == 2
        assert data["queued_tasks"] == 0
        assert data["busy"] is False

    def test_unknown_agent_404(self, metrics_app):
        client = metrics_app["client"]
        resp = client.get("/mesh/agents/nonexistent/metrics")
        assert resp.status_code == 404

    def test_invalid_agent_id_400(self, metrics_app):
        client = metrics_app["client"]
        resp = client.get("/mesh/agents/!!!invalid!!!/metrics")
        assert resp.status_code == 400

    def test_cost_data(self, metrics_app):
        client = metrics_app["client"]
        cost_tracker = metrics_app["cost_tracker"]

        # Record cost for alpha
        cost_tracker.track("alpha", "claude-3-5-sonnet-20241022", 500, 200)

        resp = client.get("/mesh/agents/alpha/metrics")
        data = resp.json()
        assert data["cost_today_usd"] > 0

    def test_budget_included(self, metrics_app):
        client = metrics_app["client"]
        cost_tracker = metrics_app["cost_tracker"]
        cost_tracker.set_budget("alpha", daily_usd=5.0, monthly_usd=100.0)

        resp = client.get("/mesh/agents/alpha/metrics")
        data = resp.json()
        assert data["budget"]["daily_limit"] == 5.0
        assert data["budget"]["monthly_limit"] == 100.0
        assert data["budget"]["allowed"] is True


class TestMeshClientMethods:
    """Test that MeshClient methods call the right endpoints."""

    @pytest.mark.asyncio
    async def test_get_system_metrics(self):
        from unittest.mock import AsyncMock, patch

        from src.agent.mesh_client import MeshClient

        client = MeshClient("http://localhost:8420", "test-agent")

        mock_response = MagicMock()
        mock_response.json.return_value = {"total_agents": 3}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_with_retry", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            result = await client.get_system_metrics()

        mock_get.assert_called_once_with("http://localhost:8420/mesh/system/metrics")
        assert result == {"total_agents": 3}

    @pytest.mark.asyncio
    async def test_get_agent_metrics(self):
        from unittest.mock import AsyncMock, patch

        from src.agent.mesh_client import MeshClient

        client = MeshClient("http://localhost:8420", "test-agent")

        mock_response = MagicMock()
        mock_response.json.return_value = {"agent_id": "alpha", "health_status": "healthy"}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_with_retry", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            result = await client.get_agent_metrics("alpha")

        mock_get.assert_called_once_with("http://localhost:8420/mesh/agents/alpha/metrics")
        assert result["agent_id"] == "alpha"


class TestCostPeriodYesterday:
    """Test that the 'yesterday' period works in _period_to_since."""

    def test_yesterday_period(self):
        from datetime import datetime, timezone

        from src.host.costs import _period_to_since

        result = _period_to_since("yesterday")
        # Should be a valid datetime string from yesterday
        parsed = datetime.strptime(result, "%Y-%m-%d %H:%M:%S")
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        # Yesterday's midnight should be between 24-48 hours ago
        delta = now - parsed
        assert 0.5 < delta.days <= 1 or (delta.days == 1 and delta.seconds >= 0)
