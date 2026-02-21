"""Tests for the dashboard API router."""

from __future__ import annotations

import os
import shutil
import tempfile

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.dashboard.events import EventBus
from src.host.costs import CostTracker
from src.host.health import HealthMonitor
from src.host.mesh import Blackboard
from src.host.traces import TraceStore


def _make_components(tmp_path: str) -> dict:
    """Create all dashboard dependencies with tmp-dir databases."""
    bb = Blackboard(db_path=os.path.join(tmp_path, "bb.db"))
    cost_tracker = CostTracker(db_path=os.path.join(tmp_path, "costs.db"))
    trace_store = TraceStore(db_path=os.path.join(tmp_path, "traces.db"))
    event_bus = EventBus()
    agent_registry: dict[str, str] = {
        "alpha": "http://localhost:8401",
        "beta": "http://localhost:8402",
    }

    # Minimal health monitor with mocked runtime/transport/router
    from unittest.mock import MagicMock

    runtime_mock = MagicMock()
    transport_mock = MagicMock()
    router_mock = MagicMock()
    health_monitor = HealthMonitor(
        runtime=runtime_mock, transport=transport_mock, router=router_mock,
    )
    health_monitor.register("alpha")
    health_monitor.register("beta")
    health_monitor.agents["alpha"].status = "healthy"
    health_monitor.agents["beta"].status = "unknown"

    return {
        "blackboard": bb,
        "health_monitor": health_monitor,
        "cost_tracker": cost_tracker,
        "trace_store": trace_store,
        "event_bus": event_bus,
        "agent_registry": agent_registry,
    }


def _make_client(components: dict) -> TestClient:
    """Build a TestClient with the dashboard router mounted."""
    from src.dashboard.server import create_dashboard_router

    router = create_dashboard_router(**components, mesh_port=8420)
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestDashboardIndex:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.components = _make_components(self._tmpdir)
        self.client = _make_client(self.components)

    def teardown_method(self):
        self.components["cost_tracker"].close()
        self.components["trace_store"].close()
        self.components["blackboard"].close()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_dashboard_index_serves_html(self):
        resp = self.client.get("/dashboard/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "window.__config" in resp.text
        # WebSocket URL is derived from window.location at runtime;
        # the template injects only the path.
        assert "/ws/events" in resp.text
        assert "/dashboard/api" in resp.text

    def test_static_css_served(self):
        resp = self.client.get("/dashboard/static/css/dashboard.css")
        assert resp.status_code == 200
        assert "pulse-border" in resp.text

    def test_static_js_served(self):
        resp = self.client.get("/dashboard/static/js/app.js")
        assert resp.status_code == 200
        assert "dashboard" in resp.text

    def test_static_websocket_js_served(self):
        resp = self.client.get("/dashboard/static/js/websocket.js")
        assert resp.status_code == 200
        assert "DashboardWebSocket" in resp.text

    def test_static_nonexistent_returns_404(self):
        resp = self.client.get("/dashboard/static/js/nonexistent.js")
        assert resp.status_code == 404

    def test_static_path_traversal_blocked(self):
        resp = self.client.get("/dashboard/static/../../server.py")
        assert resp.status_code == 404


class TestDashboardAgentsAPI:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.components = _make_components(self._tmpdir)
        self.client = _make_client(self.components)

    def teardown_method(self):
        self.components["cost_tracker"].close()
        self.components["trace_store"].close()
        self.components["blackboard"].close()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_api_agents_returns_fleet(self):
        resp = self.client.get("/dashboard/api/agents")
        assert resp.status_code == 200
        data = resp.json()
        assert "agents" in data
        assert len(data["agents"]) == 2
        ids = {a["id"] for a in data["agents"]}
        assert ids == {"alpha", "beta"}

    def test_api_agents_includes_all_fields(self):
        resp = self.client.get("/dashboard/api/agents")
        data = resp.json()
        agent = data["agents"][0]
        expected_fields = {
            "id", "url", "health_status", "failures", "restarts",
            "last_check", "last_healthy", "daily_cost", "daily_tokens",
        }
        assert expected_fields.issubset(agent.keys())

    def test_api_agents_includes_health(self):
        resp = self.client.get("/dashboard/api/agents")
        data = resp.json()
        alpha = next(a for a in data["agents"] if a["id"] == "alpha")
        assert alpha["health_status"] == "healthy"
        beta = next(a for a in data["agents"] if a["id"] == "beta")
        assert beta["health_status"] == "unknown"

    def test_api_agents_includes_costs(self):
        self.components["cost_tracker"].track("alpha", "openai/gpt-4o-mini", 500, 100)
        resp = self.client.get("/dashboard/api/agents")
        data = resp.json()
        alpha = next(a for a in data["agents"] if a["id"] == "alpha")
        assert alpha["daily_cost"] > 0
        assert alpha["daily_tokens"] > 0

    def test_api_agents_empty_registry(self):
        self.components["agent_registry"].clear()
        resp = self.client.get("/dashboard/api/agents")
        data = resp.json()
        assert data["agents"] == []


class TestDashboardAgentDetail:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.components = _make_components(self._tmpdir)
        self.client = _make_client(self.components)

    def teardown_method(self):
        self.components["cost_tracker"].close()
        self.components["trace_store"].close()
        self.components["blackboard"].close()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_api_agent_detail_found(self):
        resp = self.client.get("/dashboard/api/agents/alpha")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "alpha"
        assert data["url"] == "http://localhost:8401"
        assert "health" in data
        assert "spend_today" in data
        assert "spend_week" in data
        assert "budget" in data

    def test_api_agent_detail_health_data(self):
        resp = self.client.get("/dashboard/api/agents/alpha")
        data = resp.json()
        assert data["health"]["status"] == "healthy"
        assert "failures" in data["health"]
        assert "restarts" in data["health"]

    def test_api_agent_detail_budget_data(self):
        self.components["cost_tracker"].set_budget("alpha", daily_usd=5.0, monthly_usd=100.0)
        resp = self.client.get("/dashboard/api/agents/alpha")
        data = resp.json()
        assert data["budget"]["daily_limit"] == 5.0
        assert data["budget"]["monthly_limit"] == 100.0
        assert data["budget"]["allowed"] is True

    def test_api_agent_detail_not_found(self):
        resp = self.client.get("/dashboard/api/agents/nonexistent")
        assert resp.status_code == 404


class TestDashboardCosts:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.components = _make_components(self._tmpdir)
        self.client = _make_client(self.components)

    def teardown_method(self):
        self.components["cost_tracker"].close()
        self.components["trace_store"].close()
        self.components["blackboard"].close()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_api_costs_today(self):
        self.components["cost_tracker"].track("alpha", "openai/gpt-4o-mini", 1000, 200)
        resp = self.client.get("/dashboard/api/costs?period=today")
        assert resp.status_code == 200
        data = resp.json()
        assert data["period"] == "today"
        assert len(data["agents"]) >= 1
        assert data["agents"][0]["agent"] == "alpha"
        assert data["agents"][0]["cost"] > 0
        assert data["agents"][0]["tokens"] > 0

    def test_api_costs_week(self):
        self.components["cost_tracker"].track("beta", "openai/gpt-4o", 500, 100)
        resp = self.client.get("/dashboard/api/costs?period=week")
        assert resp.status_code == 200
        data = resp.json()
        assert data["period"] == "week"
        assert len(data["agents"]) >= 1

    def test_api_costs_with_budgets(self):
        self.components["cost_tracker"].set_budget("alpha", daily_usd=5.0, monthly_usd=100.0)
        self.components["cost_tracker"].track("alpha", "openai/gpt-4o-mini", 1000, 200)
        resp = self.client.get("/dashboard/api/costs?period=today")
        data = resp.json()
        assert "budgets" in data
        assert "alpha" in data["budgets"]
        assert data["budgets"]["alpha"]["daily_limit"] == 5.0
        assert data["budgets"]["alpha"]["allowed"] is True

    def test_api_costs_invalid_period_defaults_to_today(self):
        """Invalid period parameter silently defaults to 'today'."""
        self.components["cost_tracker"].track("alpha", "openai/gpt-4o-mini", 500, 100)
        resp = self.client.get("/dashboard/api/costs?period=invalid")
        assert resp.status_code == 200
        data = resp.json()
        assert data["period"] == "today"

    def test_api_costs_empty(self):
        resp = self.client.get("/dashboard/api/costs?period=today")
        assert resp.status_code == 200
        data = resp.json()
        assert data["agents"] == []


class TestDashboardBlackboard:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.components = _make_components(self._tmpdir)
        self.client = _make_client(self.components)

    def teardown_method(self):
        self.components["cost_tracker"].close()
        self.components["trace_store"].close()
        self.components["blackboard"].close()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_api_blackboard_with_entries(self):
        self.components["blackboard"].write("test/key1", {"val": 1}, written_by="alpha")
        self.components["blackboard"].write("test/key2", {"val": 2}, written_by="beta")
        resp = self.client.get("/dashboard/api/blackboard?prefix=test/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["prefix"] == "test/"
        assert len(data["entries"]) == 2

    def test_api_blackboard_entry_fields(self):
        self.components["blackboard"].write("test/key1", {"val": 42}, written_by="alpha")
        resp = self.client.get("/dashboard/api/blackboard?prefix=test/")
        data = resp.json()
        entry = data["entries"][0]
        assert entry["key"] == "test/key1"
        assert entry["value"] == {"val": 42}
        assert entry["written_by"] == "alpha"
        assert "updated_at" in entry
        assert entry["version"] == 1

    def test_api_blackboard_empty(self):
        resp = self.client.get("/dashboard/api/blackboard?prefix=nothing/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["entries"] == []

    def test_api_blackboard_all_entries(self):
        """Empty prefix returns all entries."""
        self.components["blackboard"].write("a/1", {"v": 1}, written_by="alpha")
        self.components["blackboard"].write("b/2", {"v": 2}, written_by="beta")
        resp = self.client.get("/dashboard/api/blackboard?prefix=")
        data = resp.json()
        assert len(data["entries"]) == 2


class TestDashboardTraces:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.components = _make_components(self._tmpdir)
        self.client = _make_client(self.components)

    def teardown_method(self):
        self.components["cost_tracker"].close()
        self.components["trace_store"].close()
        self.components["blackboard"].close()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_api_traces_list(self):
        ts = self.components["trace_store"]
        ts.record(trace_id="tr_abc", source="dispatch", agent="alpha", event_type="chat", detail="hello")
        ts.record(trace_id="tr_abc", source="llm", agent="alpha", event_type="llm_call", duration_ms=150)
        resp = self.client.get("/dashboard/api/traces?limit=50")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["traces"]) == 2

    def test_api_traces_limit_respected(self):
        ts = self.components["trace_store"]
        for i in range(10):
            ts.record(trace_id=f"tr_{i}", source="test", agent="alpha", event_type="chat")
        resp = self.client.get("/dashboard/api/traces?limit=3")
        data = resp.json()
        assert len(data["traces"]) == 3

    def test_api_traces_limit_clamped(self):
        """Negative limit gets clamped to 1."""
        ts = self.components["trace_store"]
        ts.record(trace_id="tr_x", source="test", agent="alpha", event_type="chat")
        resp = self.client.get("/dashboard/api/traces?limit=-5")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["traces"]) >= 1

    def test_api_traces_detail(self):
        ts = self.components["trace_store"]
        ts.record(trace_id="tr_abc", source="dispatch", agent="alpha", event_type="chat", detail="hello")
        ts.record(trace_id="tr_abc", source="llm", agent="alpha", event_type="llm_call", duration_ms=150)
        resp = self.client.get("/dashboard/api/traces/tr_abc")
        assert resp.status_code == 200
        data = resp.json()
        assert data["trace_id"] == "tr_abc"
        assert len(data["events"]) == 2
        # Verify event fields
        evt = data["events"][0]
        assert "trace_id" in evt
        assert "timestamp" in evt
        assert "source" in evt
        assert "agent" in evt
        assert "event_type" in evt
        assert "detail" in evt
        assert "duration_ms" in evt

    def test_api_traces_not_found(self):
        resp = self.client.get("/dashboard/api/traces/nonexistent")
        assert resp.status_code == 404
