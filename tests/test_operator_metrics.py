"""Tests for pre-computed metrics endpoints used by the operator heartbeat."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.host.costs import CostTracker
from src.host.health import HealthMonitor
from src.host.mesh import Blackboard, MessageRouter, PubSub
from src.host.permissions import PermissionMatrix
from src.host.server import create_mesh_app
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
def metrics_app(tmp_path):
    """Build a mesh app with health monitor, cost tracker, and lane manager."""
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

    app = create_mesh_app(
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
    }

    cost_tracker.close()
    bb.close()


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
