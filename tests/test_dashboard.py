"""Tests for the dashboard API router."""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.dashboard.events import EventBus
from src.host.costs import CostTracker
from src.host.health import HealthMonitor
from src.host.mesh import Blackboard
from src.host.traces import TraceStore


def _make_components(tmp_path: str, *, include_v2: bool = False) -> dict:
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

    result = {
        "blackboard": bb,
        "health_monitor": health_monitor,
        "cost_tracker": cost_tracker,
        "trace_store": trace_store,
        "event_bus": event_bus,
        "agent_registry": agent_registry,
    }

    if include_v2:
        lane_manager = MagicMock()
        lane_manager.get_status.return_value = {}

        cron_scheduler = MagicMock()
        cron_scheduler.list_jobs.return_value = []
        cron_scheduler.jobs = {}

        orchestrator = MagicMock()
        orchestrator.workflows = {}
        orchestrator.active_executions = {}

        pubsub = MagicMock()
        pubsub.subscriptions = {}

        permissions_mock = MagicMock()
        credential_vault = MagicMock()
        credential_vault.list_credential_names.return_value = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"]

        msg_router = MagicMock()
        msg_router.message_log = []

        result.update({
            "lane_manager": lane_manager,
            "cron_scheduler": cron_scheduler,
            "orchestrator": orchestrator,
            "pubsub": pubsub,
            "permissions": permissions_mock,
            "credential_vault": credential_vault,
            "transport": transport_mock,
            "runtime": runtime_mock,
            "router": msg_router,
        })

    return result


def _make_client(components: dict) -> TestClient:
    """Build a TestClient with the dashboard router mounted."""
    from src.dashboard.server import create_dashboard_router

    router = create_dashboard_router(**components, mesh_port=8420)
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _teardown(components: dict) -> None:
    components["cost_tracker"].close()
    components["trace_store"].close()
    components["blackboard"].close()
    if "pubsub" in components and hasattr(components["pubsub"], "close"):
        components["pubsub"].close()


class TestDashboardIndex:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.components = _make_components(self._tmpdir)
        self.client = _make_client(self.components)

    def teardown_method(self):
        _teardown(self.components)
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
        _teardown(self.components)
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
            "role", "model",
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
        self.components["cost_tracker"].track("alpha", "openai/gpt-4.1-mini", 500, 100)
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


class TestDashboardAgentCRUD:
    """Tests for POST /api/agents and DELETE /api/agents/{id}."""

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.components = _make_components(self._tmpdir, include_v2=True)
        self.client = _make_client(self.components)

    def teardown_method(self):
        _teardown(self.components)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    @patch("src.cli.config._create_agent")
    @patch("src.cli.config._load_config")
    def test_post_agent_success(self, mock_load, mock_create):
        mock_load.return_value = {
            "llm": {"default_model": "openai/gpt-4o-mini"},
            "agents": {"new_agent": {"role": "tester", "system_prompt": "test", "skills_dir": "", "model": "openai/gpt-4o-mini", "browser_backend": ""}},
        }
        self.components["runtime"].start_agent.return_value = "http://localhost:8403"
        self.components["runtime"].wait_for_agent = AsyncMock(return_value=True)
        self.components["permissions"].reload = MagicMock()

        resp = self.client.post(
            "/dashboard/api/agents",
            json={"name": "new_agent", "role": "tester"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["created"] is True
        assert data["agent"] == "new_agent"
        assert data["ready"] is True
        mock_create.assert_called_once()

    def test_post_agent_missing_name(self):
        resp = self.client.post("/dashboard/api/agents", json={"role": "tester"})
        assert resp.status_code == 400
        assert "name" in resp.json()["detail"].lower()

    def test_post_agent_invalid_name(self):
        resp = self.client.post("/dashboard/api/agents", json={"name": "Bad-Name!"})
        assert resp.status_code == 400

    def test_post_agent_duplicate(self):
        resp = self.client.post("/dashboard/api/agents", json={"name": "alpha"})
        assert resp.status_code == 409

    @patch("src.cli.config._save_permissions")
    @patch("src.cli.config._load_permissions", return_value={"permissions": {}})
    @patch("src.cli.config.AGENTS_FILE")
    def test_delete_agent_success(self, mock_agents_file, mock_lp, mock_sp):
        mock_agents_file.exists.return_value = False
        resp = self.client.delete("/dashboard/api/agents/alpha")
        assert resp.status_code == 200
        data = resp.json()
        assert data["removed"] is True
        assert data["agent"] == "alpha"
        # Verify agent is removed from registry
        assert "alpha" not in self.components["agent_registry"]
        # Verify health monitor cleanup
        assert "alpha" not in self.components["health_monitor"].agents
        # Verify PubSub, cron, and lane cleanup
        self.components["pubsub"].unsubscribe_agent.assert_called_once_with("alpha")
        self.components["cron_scheduler"].remove_agent_jobs.assert_called_once_with("alpha")
        self.components["lane_manager"].remove_lane.assert_called_once_with("alpha")

    def test_delete_agent_not_found(self):
        resp = self.client.delete("/dashboard/api/agents/nonexistent")
        assert resp.status_code == 404


class TestDashboardAgentDetail:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.components = _make_components(self._tmpdir)
        self.client = _make_client(self.components)

    def teardown_method(self):
        _teardown(self.components)
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
        _teardown(self.components)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_api_costs_today(self):
        self.components["cost_tracker"].track("alpha", "openai/gpt-4.1-mini", 1000, 200)
        resp = self.client.get("/dashboard/api/costs?period=today")
        assert resp.status_code == 200
        data = resp.json()
        assert data["period"] == "today"
        assert len(data["agents"]) >= 1
        assert data["agents"][0]["agent"] == "alpha"
        assert data["agents"][0]["cost"] > 0
        assert data["agents"][0]["tokens"] > 0

    def test_api_costs_week(self):
        self.components["cost_tracker"].track("beta", "openai/gpt-4.1", 500, 100)
        resp = self.client.get("/dashboard/api/costs?period=week")
        assert resp.status_code == 200
        data = resp.json()
        assert data["period"] == "week"
        assert len(data["agents"]) >= 1

    def test_api_costs_with_budgets(self):
        self.components["cost_tracker"].set_budget("alpha", daily_usd=5.0, monthly_usd=100.0)
        self.components["cost_tracker"].track("alpha", "openai/gpt-4.1-mini", 1000, 200)
        resp = self.client.get("/dashboard/api/costs?period=today")
        data = resp.json()
        assert "budgets" in data
        assert "alpha" in data["budgets"]
        assert data["budgets"]["alpha"]["daily_limit"] == 5.0
        assert data["budgets"]["alpha"]["allowed"] is True

    def test_api_costs_invalid_period_defaults_to_today(self):
        """Invalid period parameter silently defaults to 'today'."""
        self.components["cost_tracker"].track("alpha", "openai/gpt-4.1-mini", 500, 100)
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
        _teardown(self.components)
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
        _teardown(self.components)
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


# ── V2 Tests: Agent Config ──────────────────────────────────


class TestDashboardAgentConfig:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.components = _make_components(self._tmpdir, include_v2=True)
        self.client = _make_client(self.components)

    def teardown_method(self):
        _teardown(self.components)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    @patch("src.cli.config._load_config")
    def test_get_config(self, mock_load):
        mock_load.return_value = {
            "llm": {"default_model": "openai/gpt-4.1-mini"},
            "agents": {
                "alpha": {
                    "model": "openai/gpt-4.1",
                    "role": "researcher",
                    "system_prompt": "You research.",
                    "budget": {"daily_usd": 5.0},
                    "browser_backend": "stealth",
                },
            },
        }
        resp = self.client.get("/dashboard/api/agents/alpha/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "alpha"
        assert data["model"] == "openai/gpt-4.1"
        assert data["role"] == "researcher"
        assert data["browser_backend"] == "stealth"

    def test_get_config_not_found(self):
        resp = self.client.get("/dashboard/api/agents/nonexistent/config")
        assert resp.status_code == 404

    @patch("src.cli.config._update_agent_field")
    @patch("src.cli.config._load_config")
    def test_put_config_update_model(self, mock_load, mock_update):
        mock_load.return_value = {
            "llm": {"default_model": "openai/gpt-4.1-mini"},
            "agents": {"alpha": {"model": "openai/gpt-4.1-mini"}},
        }
        resp = self.client.put(
            "/dashboard/api/agents/alpha/config",
            json={"model": "openai/gpt-4.1"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "model" in data["updated"]
        assert data["restart_required"] is True
        mock_update.assert_called_with("alpha", "model", "openai/gpt-4.1")

    def test_put_config_invalid_model(self):
        with patch("src.cli.config._load_config") as mock_load:
            mock_load.return_value = {
                "llm": {"default_model": "openai/gpt-4.1-mini"},
                "agents": {"alpha": {"model": "openai/gpt-4.1-mini"}},
            }
            resp = self.client.put(
                "/dashboard/api/agents/alpha/config",
                json={"model": "invalid/model"},
            )
            assert resp.status_code == 400

    @patch("src.cli.config._update_agent_field")
    @patch("src.cli.config._load_config")
    def test_put_config_sanitizes_prompt(self, mock_load, mock_update):
        mock_load.return_value = {
            "llm": {"default_model": "openai/gpt-4.1-mini"},
            "agents": {"alpha": {}},
        }
        # U+200B zero-width space should be stripped
        resp = self.client.put(
            "/dashboard/api/agents/alpha/config",
            json={"system_prompt": "Hello\u200B world"},
        )
        assert resp.status_code == 200
        assert "system_prompt" in resp.json()["updated"]
        # The sanitized prompt was passed to _update_agent_field
        call_args = mock_update.call_args_list
        prompt_call = [c for c in call_args if c[0][1] == "system_prompt"]
        assert len(prompt_call) == 1
        assert "\u200b" not in prompt_call[0][0][2]

    @patch("src.cli.config._update_agent_field")
    @patch("src.cli.config._load_config")
    def test_put_config_browser_validation(self, mock_load, mock_update):
        mock_load.return_value = {
            "llm": {"default_model": "openai/gpt-4.1-mini"},
            "agents": {"alpha": {"browser_backend": "basic"}},
        }
        resp = self.client.put(
            "/dashboard/api/agents/alpha/config",
            json={"browser_backend": "stealth"},
        )
        assert resp.status_code == 200
        assert "browser_backend" in resp.json()["updated"]

    def test_put_config_invalid_browser(self):
        with patch("src.cli.config._load_config") as mock_load:
            mock_load.return_value = {
                "llm": {"default_model": "openai/gpt-4.1-mini"},
                "agents": {"alpha": {}},
            }
            resp = self.client.put(
                "/dashboard/api/agents/alpha/config",
                json={"browser_backend": "invalid_browser"},
            )
            assert resp.status_code == 400

    def test_put_budget_quick(self):
        resp = self.client.put(
            "/dashboard/api/agents/alpha/budget",
            json={"daily_usd": 7.5},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["updated"] is True
        assert data["daily_usd"] == 7.5
        # Verify cost_tracker budget was updated
        budget = self.components["cost_tracker"].check_budget("alpha")
        assert budget["daily_limit"] == 7.5

    def test_put_budget_invalid(self):
        resp = self.client.put(
            "/dashboard/api/agents/alpha/budget",
            json={"daily_usd": -5},
        )
        assert resp.status_code == 400


# ── V2 Tests: Queues ────────────────────────────────────────


class TestDashboardQueues:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.components = _make_components(self._tmpdir, include_v2=True)
        self.client = _make_client(self.components)

    def teardown_method(self):
        _teardown(self.components)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_queues_empty(self):
        resp = self.client.get("/dashboard/api/queues")
        assert resp.status_code == 200
        queues = resp.json()["queues"]
        # All agents appear with idle status (even without active lanes)
        assert "alpha" in queues
        assert "beta" in queues
        assert queues["alpha"]["busy"] is False
        assert queues["alpha"]["queued"] == 0

    def test_queues_with_data(self):
        self.components["lane_manager"].get_status.return_value = {
            "alpha": {"queued": 2, "pending": 1, "collected": 0, "busy": True},
        }
        resp = self.client.get("/dashboard/api/queues")
        data = resp.json()
        assert "alpha" in data["queues"]
        assert data["queues"]["alpha"]["busy"] is True
        assert data["queues"]["alpha"]["queued"] == 2

    def test_queues_no_lane_manager(self):
        self.components["lane_manager"] = None
        self.client = _make_client(self.components)
        resp = self.client.get("/dashboard/api/queues")
        assert resp.status_code == 200
        queues = resp.json()["queues"]
        # All agents still appear with idle defaults
        assert "alpha" in queues
        assert queues["alpha"]["busy"] is False


# ── V2 Tests: Cron ──────────────────────────────────────────


class TestDashboardCron:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.components = _make_components(self._tmpdir, include_v2=True)
        self.client = _make_client(self.components)

    def teardown_method(self):
        _teardown(self.components)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_cron_list_empty(self):
        resp = self.client.get("/dashboard/api/cron")
        assert resp.status_code == 200
        assert resp.json()["jobs"] == []

    def test_cron_list_with_jobs(self):
        self.components["cron_scheduler"].list_jobs.return_value = [
            {"id": "cron_abc", "agent": "alpha", "schedule": "every 1h", "enabled": True},
        ]
        resp = self.client.get("/dashboard/api/cron")
        data = resp.json()
        assert len(data["jobs"]) == 1
        assert data["jobs"][0]["agent"] == "alpha"

    def test_cron_run_job(self):
        self.components["cron_scheduler"].run_job = AsyncMock(return_value="done")
        self.components["cron_scheduler"].jobs = {"cron_abc": True}
        resp = self.client.post("/dashboard/api/cron/cron_abc/run")
        assert resp.status_code == 200
        assert resp.json()["executed"] is True

    def test_cron_pause_resume(self):
        self.components["cron_scheduler"].pause_job.return_value = True
        resp = self.client.post("/dashboard/api/cron/cron_abc/pause")
        assert resp.status_code == 200
        assert resp.json()["paused"] is True

        self.components["cron_scheduler"].resume_job.return_value = True
        resp = self.client.post("/dashboard/api/cron/cron_abc/resume")
        assert resp.status_code == 200
        assert resp.json()["resumed"] is True

    def test_cron_not_available(self):
        self.components["cron_scheduler"] = None
        self.client = _make_client(self.components)
        resp = self.client.post("/dashboard/api/cron/abc/run")
        assert resp.status_code == 503


# ── V2 Tests: Blackboard Write/Delete ────────────────────────


class TestDashboardBlackboardWrite:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.components = _make_components(self._tmpdir, include_v2=True)
        self.client = _make_client(self.components)

    def teardown_method(self):
        _teardown(self.components)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_write_creates_entry(self):
        resp = self.client.put(
            "/dashboard/api/blackboard/test/new_key",
            json={"value": {"hello": "world"}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["key"] == "test/new_key"
        assert data["value"] == {"hello": "world"}
        assert data["written_by"] == "dashboard"

    def test_delete_removes_entry(self):
        self.components["blackboard"].write("test/del", {"v": 1}, written_by="alpha")
        resp = self.client.delete("/dashboard/api/blackboard/test/del")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True
        # Verify it's gone
        assert self.components["blackboard"].read("test/del") is None

    def test_delete_history_blocked(self):
        resp = self.client.delete("/dashboard/api/blackboard/history/old_entry")
        assert resp.status_code == 400

    def test_write_with_custom_author(self):
        resp = self.client.put(
            "/dashboard/api/blackboard/test/authored",
            json={"value": {"data": 1}, "written_by": "admin"},
        )
        assert resp.status_code == 200
        assert resp.json()["written_by"] == "admin"


# ── V2 Tests: Settings ──────────────────────────────────────


class TestDashboardSettings:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.components = _make_components(self._tmpdir, include_v2=True)
        self.client = _make_client(self.components)

    def teardown_method(self):
        _teardown(self.components)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_returns_all_sections(self):
        resp = self.client.get("/dashboard/api/settings")
        assert resp.status_code == 200
        data = resp.json()
        assert "credentials" in data
        assert "pubsub_subscriptions" in data
        assert "model_costs" in data
        assert "provider_models" in data
        assert "browser_backends" in data

    def test_credentials_no_values(self):
        resp = self.client.get("/dashboard/api/settings")
        data = resp.json()
        assert data["credentials"]["count"] == 2
        assert "ANTHROPIC_API_KEY" in data["credentials"]["names"]
        # No actual credential values exposed
        for name in data["credentials"]["names"]:
            assert len(name) < 100  # Names, not values

    def test_includes_models(self):
        resp = self.client.get("/dashboard/api/settings")
        data = resp.json()
        assert "openai" in data["provider_models"]
        assert "anthropic" in data["provider_models"]
        assert len(data["model_costs"]) > 0


# ── V2 Tests: Messages ──────────────────────────────────────


class TestDashboardMessages:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.components = _make_components(self._tmpdir, include_v2=True)
        self.client = _make_client(self.components)

    def teardown_method(self):
        _teardown(self.components)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_returns_log(self):
        self.components["router"].message_log = [
            {"id": "m1", "from": "alpha", "to": "beta", "type": "chat"},
            {"id": "m2", "from": "beta", "to": "alpha", "type": "response"},
        ]
        resp = self.client.get("/dashboard/api/messages")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["messages"]) == 2

    def test_empty_log(self):
        resp = self.client.get("/dashboard/api/messages")
        assert resp.status_code == 200
        assert resp.json()["messages"] == []


# ── V2 Tests: Workflows ─────────────────────────────────────


class TestDashboardWorkflows:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.components = _make_components(self._tmpdir, include_v2=True)
        self.client = _make_client(self.components)

    def teardown_method(self):
        _teardown(self.components)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_list_empty(self):
        resp = self.client.get("/dashboard/api/workflows")
        assert resp.status_code == 200
        data = resp.json()
        assert data["workflows"] == []
        assert data["active"] == []

    def test_list_with_definitions(self):
        mock_wf = MagicMock()
        mock_wf.name = "test_wf"
        mock_wf.steps = [1, 2, 3]
        mock_wf.trigger = "manual"
        mock_wf.timeout = 300
        self.components["orchestrator"].workflows = {"test_wf": mock_wf}
        resp = self.client.get("/dashboard/api/workflows")
        data = resp.json()
        assert len(data["workflows"]) == 1
        assert data["workflows"][0]["name"] == "test_wf"
        assert data["workflows"][0]["steps"] == 3
