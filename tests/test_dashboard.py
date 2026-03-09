"""Tests for the dashboard API router."""

from __future__ import annotations

import os
import shutil
import tempfile
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
    runtime_mock.browser_vnc_url = None
    runtime_mock.browser_service_url = None
    runtime_mock.browser_auth_token = ""
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
        credential_vault.list_agent_credential_names.return_value = []
        credential_vault.list_system_credential_names.return_value = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"]
        credential_vault.system_credentials = {"anthropic_api_key": "sk-test", "openai_api_key": "sk-openai"}
        credential_vault.get_providers_with_credentials.return_value = {"anthropic", "openai"}

        msg_router = MagicMock()
        msg_router.agent_registry = agent_registry
        msg_router.register_agent.side_effect = lambda aid, url, **kw: agent_registry.__setitem__(aid, url)
        msg_router.unregister_agent.side_effect = lambda aid: agent_registry.pop(aid, None)
        msg_router.message_log = []

        channel_manager = MagicMock()
        channel_manager.get_channel_status.return_value = [
            {"type": t, "connected": False, "paired": False, "pairing_code": None}
            for t in ("telegram", "discord", "slack", "whatsapp")
        ]
        channel_manager.start_channel.return_value = []

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
            "channel_manager": channel_manager,
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
            "role", "model", "avatar", "project",
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

    def test_api_agents_avatar_default(self):
        resp = self.client.get("/dashboard/api/agents")
        data = resp.json()
        alpha = next(a for a in data["agents"] if a["id"] == "alpha")
        assert alpha["avatar"] == 1

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
            "agents": {
                "new_agent": {
                    "role": "tester",
                    "skills_dir": "", "model": "openai/gpt-4o-mini",
                },
            },
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

    @patch("src.cli.config._create_agent")
    @patch("src.cli.config._load_config")
    def test_post_agent_creates_heartbeat(self, mock_load, mock_create):
        mock_load.return_value = {
            "llm": {"default_model": "openai/gpt-4o-mini"},
            "mesh": {"heartbeat_schedule": "every 30m"},
            "agents": {
                "new_agent": {
                    "role": "tester",
                    "skills_dir": "", "model": "openai/gpt-4o-mini",
                },
            },
        }
        self.components["runtime"].start_agent.return_value = "http://localhost:8403"
        self.components["runtime"].wait_for_agent = AsyncMock(return_value=True)
        self.components["permissions"].reload = MagicMock()
        resp = self.client.post(
            "/dashboard/api/agents",
            json={"name": "new_agent", "role": "tester"},
        )
        assert resp.status_code == 200
        self.components["cron_scheduler"].ensure_heartbeat.assert_called_once_with(
            "new_agent", "every 30m",
        )

    @patch("src.cli.config._update_agent_field")
    @patch("src.cli.config._create_agent")
    @patch("src.cli.config._load_config")
    def test_post_agent_with_avatar(self, mock_load, mock_create, mock_update):
        mock_load.return_value = {
            "llm": {"default_model": "openai/gpt-4o-mini"},
            "agents": {
                "new_agent": {
                    "role": "tester",
                    "skills_dir": "", "model": "openai/gpt-4o-mini",
                },
            },
        }
        self.components["runtime"].start_agent.return_value = "http://localhost:8403"
        self.components["runtime"].wait_for_agent = AsyncMock(return_value=True)
        self.components["permissions"].reload = MagicMock()

        resp = self.client.post(
            "/dashboard/api/agents",
            json={"name": "new_agent", "role": "tester", "avatar": 7},
        )
        assert resp.status_code == 200
        mock_update.assert_called_with("new_agent", "avatar", 7)

    def test_post_agent_invalid_avatar(self):
        resp = self.client.post(
            "/dashboard/api/agents",
            json={"name": "new_agent", "role": "tester", "avatar": 99},
        )
        assert resp.status_code == 400
        assert "avatar" in resp.json()["detail"].lower()

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

    @patch("src.cli.config._create_agent")
    @patch("src.cli.config._load_config")
    def test_post_agent_default_model_uses_credentialed_provider(
        self, mock_load, mock_create,
    ):
        """When no model is specified, pick from a provider that has credentials."""
        # Config says default is deepseek, but only anthropic has credentials
        mock_load.return_value = {
            "llm": {"default_model": "deepseek/deepseek-chat"},
            "agents": {
                "new_agent": {
                    "role": "tester",
                    "skills_dir": "", "model": "anthropic/claude-opus-4-6",
                },
            },
        }
        self.components["credential_vault"].get_providers_with_credentials.return_value = {
            "anthropic",
        }
        self.components["runtime"].start_agent.return_value = "http://localhost:8403"
        self.components["runtime"].wait_for_agent = AsyncMock(return_value=True)
        self.components["permissions"].reload = MagicMock()

        resp = self.client.post(
            "/dashboard/api/agents",
            json={"name": "new_agent", "role": "tester"},  # no model
        )
        assert resp.status_code == 200
        # _create_agent should be called with a model from anthropic, not deepseek
        call_args = mock_create.call_args[0]
        assert call_args[2].startswith("anthropic/"), (
            f"Expected anthropic model, got: {call_args[2]}"
        )

    def test_delete_agent_not_found(self):
        resp = self.client.delete("/dashboard/api/agents/nonexistent")
        assert resp.status_code == 404

    def test_get_agent_templates(self):
        resp = self.client.get("/dashboard/api/agent-templates")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0
        # Check structure
        tpl = data[0]
        assert "id" in tpl
        assert "name" in tpl
        assert "source" in tpl
        assert "role" in tpl
        assert "/" in tpl["id"]

    @patch("src.cli.config._create_agent_from_template")
    @patch("src.cli.config._load_config")
    def test_post_agent_with_template(self, mock_load, mock_create_tpl):
        mock_load.return_value = {
            "llm": {"default_model": "openai/gpt-4o-mini"},
            "agents": {
                "my_engineer": {
                    "role": "Software engineer",
                    "skills_dir": "", "model": "openai/gpt-4o-mini",
                },
            },
        }
        self.components["runtime"].start_agent.return_value = "http://localhost:8403"
        self.components["runtime"].wait_for_agent = AsyncMock(return_value=True)
        self.components["permissions"].reload = MagicMock()

        resp = self.client.post(
            "/dashboard/api/agents",
            json={"name": "my_engineer", "template": "devteam/engineer"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["created"] is True
        assert data["agent"] == "my_engineer"
        mock_create_tpl.assert_called_once_with("my_engineer", "devteam/engineer", "openai/gpt-4o-mini")

    @patch("src.cli.config._create_agent_from_template")
    @patch("src.cli.config._load_config")
    def test_post_agent_template_passes_initial_env(self, mock_load, mock_create_tpl):
        """Verify that initial_soul/instructions/heartbeat reach runtime.extra_env."""
        mock_load.return_value = {
            "llm": {"default_model": "openai/gpt-4o-mini"},
            "agents": {
                "my_pm": {
                    "role": "Product manager",
                    "skills_dir": "", "model": "openai/gpt-4o-mini",
                    "initial_instructions": "You are a PM.",
                    "initial_soul": "Precise and outcome-focused.",
                    "initial_heartbeat": "Check tasks board.",
                },
            },
        }
        runtime = self.components["runtime"]
        runtime.start_agent.return_value = "http://localhost:8403"
        runtime.wait_for_agent = AsyncMock(return_value=True)
        runtime.extra_env = {}
        self.components["permissions"].reload = MagicMock()

        # Capture extra_env at the moment start_agent is called
        captured_env: dict = {}

        def _capture(*a, **kw):
            captured_env.update(runtime.extra_env)
            return "http://localhost:8403"

        runtime.start_agent.side_effect = _capture

        resp = self.client.post(
            "/dashboard/api/agents",
            json={"name": "my_pm", "template": "devteam/pm"},
        )
        assert resp.status_code == 200
        assert captured_env["INITIAL_INSTRUCTIONS"] == "You are a PM."
        assert captured_env["INITIAL_SOUL"] == "Precise and outcome-focused."
        assert captured_env["INITIAL_HEARTBEAT"] == "Check tasks board."
        # Cleaned up after start
        assert "INITIAL_INSTRUCTIONS" not in runtime.extra_env
        assert "INITIAL_SOUL" not in runtime.extra_env
        assert "INITIAL_HEARTBEAT" not in runtime.extra_env

    @patch("src.cli.config._create_agent_from_template")
    @patch("src.cli.config._load_config")
    def test_post_agent_invalid_template(self, mock_load, mock_create_tpl):
        mock_load.return_value = {
            "llm": {"default_model": "openai/gpt-4o-mini"},
            "agents": {},
        }
        mock_create_tpl.side_effect = ValueError("Template not found")
        self.components["permissions"].reload = MagicMock()

        resp = self.client.post(
            "/dashboard/api/agents",
            json={"name": "my_agent", "template": "nonexistent/template"},
        )
        assert resp.status_code == 400
        assert "not found" in resp.json()["detail"].lower()

    def test_post_agent_malformed_template_id(self):
        resp = self.client.post(
            "/dashboard/api/agents",
            json={"name": "my_agent", "template": "no-slash"},
        )
        assert resp.status_code == 400
        assert "template" in resp.json()["detail"].lower()

    def test_post_agent_template_path_traversal(self):
        resp = self.client.post(
            "/dashboard/api/agents",
            json={"name": "my_agent", "template": "../../../etc/passwd"},
        )
        assert resp.status_code == 400


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
        ts.record(trace_id="tr_def", source="dispatch", agent="beta", event_type="chat", detail="world")
        resp = self.client.get("/dashboard/api/traces?limit=50")
        assert resp.status_code == 200
        data = resp.json()
        # Grouped by trace_id: tr_abc (2 events) + tr_def (1 event) = 2 summaries
        assert len(data["traces"]) == 2
        # Newest first
        assert data["traces"][0]["trace_id"] == "tr_def"
        assert data["traces"][0]["event_count"] == 1
        assert data["traces"][1]["trace_id"] == "tr_abc"
        assert data["traces"][1]["event_count"] == 2

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
                    "budget": {"daily_usd": 5.0},
                },
            },
        }
        resp = self.client.get("/dashboard/api/agents/alpha/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "alpha"
        assert data["model"] == "openai/gpt-4.1"
        assert data["role"] == "researcher"
        assert "system_prompt" not in data

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

    @patch("src.cli.config._load_config")
    def test_get_config_includes_avatar_default(self, mock_load):
        mock_load.return_value = {
            "llm": {"default_model": "openai/gpt-4.1-mini"},
            "agents": {"alpha": {"model": "openai/gpt-4.1"}},
        }
        resp = self.client.get("/dashboard/api/agents/alpha/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["avatar"] == 1

    @patch("src.cli.config._load_config")
    def test_get_config_includes_avatar_custom(self, mock_load):
        mock_load.return_value = {
            "llm": {"default_model": "openai/gpt-4.1-mini"},
            "agents": {"alpha": {"model": "openai/gpt-4.1", "avatar": 7}},
        }
        resp = self.client.get("/dashboard/api/agents/alpha/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["avatar"] == 7

    @patch("src.cli.config._update_agent_field")
    @patch("src.cli.config._load_config")
    def test_put_config_update_avatar(self, mock_load, mock_update):
        mock_load.return_value = {
            "llm": {"default_model": "openai/gpt-4.1-mini"},
            "agents": {"alpha": {"model": "openai/gpt-4.1-mini"}},
        }
        resp = self.client.put(
            "/dashboard/api/agents/alpha/config",
            json={"avatar": 15},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "avatar" in data["updated"]
        assert data["restart_required"] is False
        mock_update.assert_called_with("alpha", "avatar", 15)

    @patch("src.cli.config._load_config")
    def test_put_config_avatar_invalid_range(self, mock_load):
        mock_load.return_value = {
            "llm": {"default_model": "openai/gpt-4.1-mini"},
            "agents": {"alpha": {"model": "openai/gpt-4.1-mini"}},
        }
        resp = self.client.put(
            "/dashboard/api/agents/alpha/config",
            json={"avatar": 99},
        )
        assert resp.status_code == 400

    @patch("src.cli.config._load_config")
    def test_put_config_avatar_invalid_type(self, mock_load):
        mock_load.return_value = {
            "llm": {"default_model": "openai/gpt-4.1-mini"},
            "agents": {"alpha": {"model": "openai/gpt-4.1-mini"}},
        }
        resp = self.client.put(
            "/dashboard/api/agents/alpha/config",
            json={"avatar": "not_a_number"},
        )
        assert resp.status_code == 400

    # ── Thinking & MCP servers ────────────────────────────

    @patch("src.cli.config._load_config")
    def test_get_config_includes_thinking_and_mcp(self, mock_load):
        mock_load.return_value = {
            "llm": {"default_model": "openai/gpt-4.1-mini"},
            "agents": {
                "alpha": {
                    "model": "openai/gpt-4.1",
                    "thinking": "medium",
                    "mcp_servers": [{"name": "brave", "command": "npx"}],
                },
            },
        }
        resp = self.client.get("/dashboard/api/agents/alpha/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["thinking"] == "medium"
        assert len(data["mcp_servers"]) == 1
        assert data["mcp_servers"][0]["name"] == "brave"

    @patch("src.cli.config._load_config")
    def test_get_config_thinking_defaults_off(self, mock_load):
        mock_load.return_value = {
            "llm": {"default_model": "openai/gpt-4.1-mini"},
            "agents": {"alpha": {"model": "openai/gpt-4.1"}},
        }
        resp = self.client.get("/dashboard/api/agents/alpha/config")
        data = resp.json()
        assert data["thinking"] == "off"
        assert data["mcp_servers"] == []

    @patch("src.cli.config._update_agent_field")
    @patch("src.cli.config._load_config")
    def test_put_config_thinking(self, mock_load, mock_update):
        mock_load.return_value = {
            "llm": {"default_model": "openai/gpt-4.1-mini"},
            "agents": {"alpha": {"model": "openai/gpt-4.1-mini"}},
        }
        resp = self.client.put(
            "/dashboard/api/agents/alpha/config",
            json={"thinking": "high"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "thinking" in data["updated"]
        assert data["restart_required"] is True
        mock_update.assert_called_with("alpha", "thinking", "high")

    @patch("src.cli.config._load_config")
    def test_put_config_thinking_invalid(self, mock_load):
        mock_load.return_value = {
            "llm": {"default_model": "openai/gpt-4.1-mini"},
            "agents": {"alpha": {"model": "openai/gpt-4.1-mini"}},
        }
        resp = self.client.put(
            "/dashboard/api/agents/alpha/config",
            json={"thinking": "extreme"},
        )
        assert resp.status_code == 400

    @patch("src.cli.config._update_agent_field")
    @patch("src.cli.config._load_config")
    def test_put_config_mcp_servers(self, mock_load, mock_update):
        mock_load.return_value = {
            "llm": {"default_model": "openai/gpt-4.1-mini"},
            "agents": {"alpha": {"model": "openai/gpt-4.1-mini"}},
        }
        servers = [{"name": "fs", "command": "npx", "args": ["-y", "fs-server"]}]
        resp = self.client.put(
            "/dashboard/api/agents/alpha/config",
            json={"mcp_servers": servers},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "mcp_servers" in data["updated"]
        assert data["restart_required"] is True

    @patch("src.cli.config._load_config")
    def test_put_config_mcp_servers_invalid(self, mock_load):
        mock_load.return_value = {
            "llm": {"default_model": "openai/gpt-4.1-mini"},
            "agents": {"alpha": {"model": "openai/gpt-4.1-mini"}},
        }
        # Missing 'command' key
        resp = self.client.put(
            "/dashboard/api/agents/alpha/config",
            json={"mcp_servers": [{"name": "bad"}]},
        )
        assert resp.status_code == 400

    @patch("src.cli.config._load_config")
    def test_put_config_mcp_servers_not_a_list(self, mock_load):
        mock_load.return_value = {
            "llm": {"default_model": "openai/gpt-4.1-mini"},
            "agents": {"alpha": {"model": "openai/gpt-4.1-mini"}},
        }
        resp = self.client.put(
            "/dashboard/api/agents/alpha/config",
            json={"mcp_servers": "not_a_list"},
        )
        assert resp.status_code == 400

    @patch("src.cli.config._update_agent_field")
    @patch("src.cli.config._load_config")
    def test_put_config_mcp_servers_empty_clears(self, mock_load, mock_update):
        mock_load.return_value = {
            "llm": {"default_model": "openai/gpt-4.1-mini"},
            "agents": {"alpha": {"model": "openai/gpt-4.1-mini"}},
        }
        resp = self.client.put(
            "/dashboard/api/agents/alpha/config",
            json={"mcp_servers": []},
        )
        assert resp.status_code == 200
        # Empty list should store None to clean up the YAML
        mock_update.assert_called_with("alpha", "mcp_servers", None)


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
        assert resp.json()["triggered"] is True

    def test_cron_run_job_not_found(self):
        self.components["cron_scheduler"].jobs = {}
        resp = self.client.post("/dashboard/api/cron/nonexistent/run")
        assert resp.status_code == 404

    def test_cron_pause_resume(self):
        self.components["cron_scheduler"].pause_job = AsyncMock(return_value=True)
        resp = self.client.post("/dashboard/api/cron/cron_abc/pause")
        assert resp.status_code == 200
        assert resp.json()["paused"] is True

        self.components["cron_scheduler"].resume_job = AsyncMock(return_value=True)
        resp = self.client.post("/dashboard/api/cron/cron_abc/resume")
        assert resp.status_code == 200
        assert resp.json()["resumed"] is True

    def test_cron_delete_success(self):
        from src.host.cron import CronJob
        regular_job = CronJob(id="cron_abc", agent="alpha", schedule="every 1h", message="test")
        self.components["cron_scheduler"].jobs = {"cron_abc": regular_job}
        self.components["cron_scheduler"].remove_job.return_value = True
        resp = self.client.delete("/dashboard/api/cron/cron_abc")
        assert resp.status_code == 200
        data = resp.json()
        assert data["deleted"] is True
        assert data["job_id"] == "cron_abc"
        self.components["cron_scheduler"].remove_job.assert_called_once_with("cron_abc")

    def test_cron_delete_not_found(self):
        self.components["cron_scheduler"].jobs = {}
        resp = self.client.delete("/dashboard/api/cron/nonexistent")
        assert resp.status_code == 404

    def test_cron_delete_not_available(self):
        self.components["cron_scheduler"] = None
        self.client = _make_client(self.components)
        resp = self.client.delete("/dashboard/api/cron/abc")
        assert resp.status_code == 503

    def test_cron_not_available(self):
        self.components["cron_scheduler"] = None
        self.client = _make_client(self.components)
        resp = self.client.post("/dashboard/api/cron/abc/run")
        assert resp.status_code == 503


# ── Heartbeat info in agents API + heartbeat delete guard ────


class TestHeartbeatInAgentsAPI:
    """GET /api/agents includes heartbeat fields when a heartbeat job exists."""

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.components = _make_components(self._tmpdir, include_v2=True)
        self.client = _make_client(self.components)

    def teardown_method(self):
        _teardown(self.components)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    @patch("src.cli.config._load_config")
    def test_agents_include_heartbeat_fields(self, mock_load):
        from src.host.cron import CronJob
        mock_load.return_value = {
            "llm": {"default_model": "openai/gpt-4o-mini"},
            "agents": {},
            "_agent_projects": {},
        }
        hb_job = CronJob(
            id="hb_alpha", agent="alpha", schedule="every 15m",
            message="heartbeat", heartbeat=True, enabled=True,
            next_run="2026-03-09T12:00:00+00:00",
        )
        self.components["cron_scheduler"].find_heartbeat_job.side_effect = (
            lambda aid: hb_job if aid == "alpha" else None
        )
        resp = self.client.get("/dashboard/api/agents")
        assert resp.status_code == 200
        agents = resp.json()["agents"]
        alpha = next(a for a in agents if a["id"] == "alpha")
        assert alpha["heartbeat_schedule"] == "every 15m"
        assert alpha["heartbeat_enabled"] is True
        assert alpha["heartbeat_next_run"] == "2026-03-09T12:00:00+00:00"

    @patch("src.cli.config._load_config")
    def test_agents_no_heartbeat_fields_when_absent(self, mock_load):
        mock_load.return_value = {
            "llm": {"default_model": "openai/gpt-4o-mini"},
            "agents": {},
            "_agent_projects": {},
        }
        self.components["cron_scheduler"].find_heartbeat_job.return_value = None
        resp = self.client.get("/dashboard/api/agents")
        assert resp.status_code == 200
        agents = resp.json()["agents"]
        alpha = next(a for a in agents if a["id"] == "alpha")
        assert "heartbeat_schedule" not in alpha
        assert "heartbeat_enabled" not in alpha
        assert "heartbeat_next_run" not in alpha


class TestHeartbeatDeleteGuard:
    """DELETE /api/cron/{id} blocks deletion of heartbeat jobs."""

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.components = _make_components(self._tmpdir, include_v2=True)
        self.client = _make_client(self.components)

    def teardown_method(self):
        _teardown(self.components)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_delete_heartbeat_job_returns_403(self):
        from src.host.cron import CronJob
        hb_job = CronJob(
            id="hb_alpha", agent="alpha", schedule="every 15m",
            message="heartbeat", heartbeat=True,
        )
        self.components["cron_scheduler"].jobs = {"hb_alpha": hb_job}
        resp = self.client.delete("/dashboard/api/cron/hb_alpha")
        assert resp.status_code == 403
        assert "Heartbeat" in resp.json()["detail"]

    def test_delete_regular_job_succeeds(self):
        from src.host.cron import CronJob
        regular_job = CronJob(
            id="cron_abc", agent="alpha", schedule="every 1h",
            message="regular task", heartbeat=False,
        )
        self.components["cron_scheduler"].jobs = {"cron_abc": regular_job}
        self.components["cron_scheduler"].remove_job.return_value = True
        resp = self.client.delete("/dashboard/api/cron/cron_abc")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True


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
        """S2 fix: dashboard always attributes writes to 'dashboard' regardless of
        client-supplied written_by to prevent spoofing."""
        resp = self.client.put(
            "/dashboard/api/blackboard/test/authored",
            json={"value": {"data": 1}, "written_by": "admin"},
        )
        assert resp.status_code == 200
        assert resp.json()["written_by"] == "dashboard"

    def test_write_key_too_long(self):
        """Dashboard rejects keys longer than 512 chars."""
        long_key = "x" * 520
        resp = self.client.put(
            f"/dashboard/api/blackboard/{long_key}",
            json={"value": {"data": 1}},
        )
        assert resp.status_code == 400
        assert "Key too long" in resp.json()["detail"]

    def test_write_value_too_large(self):
        """Dashboard rejects values larger than 256 KB."""
        resp = self.client.put(
            "/dashboard/api/blackboard/test/big",
            json={"value": {"data": "x" * 300_000}},
        )
        assert resp.status_code == 413
        assert "Value too large" in resp.json()["detail"]


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

    def test_has_llm_credentials_true(self):
        """has_llm_credentials is True when a known LLM key exists."""
        self.components["credential_vault"].list_credential_names.return_value = [
            "anthropic_api_key", "brave_search_api_key",
        ]
        self.client = _make_client(self.components)
        resp = self.client.get("/dashboard/api/settings")
        data = resp.json()
        assert data["has_llm_credentials"] is True

    def test_has_llm_credentials_false(self):
        """has_llm_credentials is False when no known LLM keys exist."""
        self.components["credential_vault"].list_credential_names.return_value = [
            "brave_search_api_key",
        ]
        self.client = _make_client(self.components)
        resp = self.client.get("/dashboard/api/settings")
        data = resp.json()
        assert data["has_llm_credentials"] is False

    def test_has_llm_credentials_empty(self):
        """has_llm_credentials is False when no credentials at all."""
        self.components["credential_vault"].list_credential_names.return_value = []
        self.client = _make_client(self.components)
        resp = self.client.get("/dashboard/api/settings")
        data = resp.json()
        assert data["has_llm_credentials"] is False


# ── V2 Tests: Credential Base URL ────────────────────────────


class TestDashboardCredentialBaseUrl:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.components = _make_components(self._tmpdir, include_v2=True)
        self.client = _make_client(self.components)

    def teardown_method(self):
        _teardown(self.components)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_post_credential_with_base_url(self):
        """POST /api/credentials with base_url stores both key and base."""
        resp = self.client.post(
            "/dashboard/api/credentials",
            json={"service": "openai", "key": "sk-test", "base_url": "https://gateway.example.com/v1"},
        )
        assert resp.status_code == 200
        vault = self.components["credential_vault"]
        calls = vault.add_credential.call_args_list
        # First call: the API key itself
        assert calls[0][0] == ("openai_api_key", "sk-test")
        # Second call: the base URL
        assert calls[1][0] == ("openai_api_base", "https://gateway.example.com/v1")

    def test_post_credential_without_base_url(self):
        """POST /api/credentials without base_url stores only the key."""
        resp = self.client.post(
            "/dashboard/api/credentials",
            json={"service": "openai", "key": "sk-test"},
        )
        assert resp.status_code == 200
        vault = self.components["credential_vault"]
        assert vault.add_credential.call_count == 1
        assert vault.add_credential.call_args[0] == ("openai_api_key", "sk-test")

    def test_post_credential_empty_base_url_ignored(self):
        """Empty base_url string is ignored."""
        resp = self.client.post(
            "/dashboard/api/credentials",
            json={"service": "openai", "key": "sk-test", "base_url": "  "},
        )
        assert resp.status_code == 200
        vault = self.components["credential_vault"]
        assert vault.add_credential.call_count == 1


class TestDashboardCredentialRemove:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.components = _make_components(self._tmpdir, include_v2=True)
        self.client = _make_client(self.components)

    def teardown_method(self):
        _teardown(self.components)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_delete_credential(self):
        """DELETE /api/credentials/{name} removes the credential."""
        vault = self.components["credential_vault"]
        vault.remove_credential = MagicMock(return_value=True)
        resp = self.client.delete("/dashboard/api/credentials/anthropic_api_key")
        assert resp.status_code == 200
        data = resp.json()
        assert data["removed"] is True
        assert data["service"] == "anthropic_api_key"
        vault.remove_credential.assert_called_once_with("anthropic_api_key")

    def test_delete_credential_not_found(self):
        """DELETE /api/credentials/{name} returns 404 when credential doesn't exist."""
        vault = self.components["credential_vault"]
        vault.remove_credential = MagicMock(return_value=False)
        resp = self.client.delete("/dashboard/api/credentials/nonexistent")
        assert resp.status_code == 404


class TestDashboardCredentialValidate:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.components = _make_components(self._tmpdir, include_v2=True)
        self.client = _make_client(self.components)

    def teardown_method(self):
        _teardown(self.components)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_validate_valid_key(self):
        """Valid API key returns valid=True."""
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=MagicMock()):
            resp = self.client.post("/dashboard/api/credentials/validate", json={
                "service": "anthropic", "key": "sk-valid-key",
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is True
        assert data["skipped"] is False

    def test_validate_invalid_key(self):
        """Invalid API key returns valid=False."""
        import litellm
        with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=litellm.AuthenticationError(
            message="Invalid API Key", llm_provider="anthropic",
            model="anthropic/claude-haiku-4-5-20251001",
        )):
            resp = self.client.post("/dashboard/api/credentials/validate", json={
                "service": "anthropic", "key": "sk-bad-key",
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is False

    def test_validate_unknown_provider_skipped(self):
        """Unknown provider skips validation and returns valid=True."""
        resp = self.client.post("/dashboard/api/credentials/validate", json={
            "service": "unknown_provider", "key": "some-key",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is True
        assert data["skipped"] is True

    def test_validate_missing_fields(self):
        """Missing service or key returns 400."""
        resp = self.client.post("/dashboard/api/credentials/validate", json={
            "service": "", "key": "",
        })
        assert resp.status_code == 400

    def test_validate_with_api_key_suffix(self):
        """Service name with _api_key suffix still validates correctly."""
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=MagicMock()):
            resp = self.client.post("/dashboard/api/credentials/validate", json={
                "service": "anthropic_api_key", "key": "sk-valid-key",
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is True
        assert data["skipped"] is False

    def test_validate_network_error_allows_save(self):
        """Network errors don't block — returns valid=True, skipped=True."""
        import litellm
        with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=litellm.Timeout(
            message="Connection timed out",
            llm_provider="anthropic",
            model="anthropic/claude-haiku-4-5-20251001",
        )):
            resp = self.client.post("/dashboard/api/credentials/validate", json={
                "service": "anthropic", "key": "sk-some-key",
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is True
        assert data["skipped"] is True


# ── V2 Tests: Credential Tier ───────────────────────────────


class TestDashboardCredentialTier:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.components = _make_components(self._tmpdir, include_v2=True)
        self.client = _make_client(self.components)

    def teardown_method(self):
        _teardown(self.components)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_llm_provider_auto_promoted_to_system(self):
        """LLM provider keys auto-detect as system tier."""
        resp = self.client.post(
            "/dashboard/api/credentials",
            json={"service": "openai", "key": "sk-test"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["tier"] == "system"
        vault = self.components["credential_vault"]
        vault.add_credential.assert_called_once_with(
            "openai_api_key", "sk-test", system=True,
        )

    def test_custom_credential_defaults_to_agent(self):
        """Non-LLM credentials default to agent tier."""
        resp = self.client.post(
            "/dashboard/api/credentials",
            json={"service": "brave_search_api_key", "key": "bsk-test"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["tier"] == "agent"
        vault = self.components["credential_vault"]
        vault.add_credential.assert_called_once_with(
            "brave_search_api_key", "bsk-test", system=False,
        )

    def test_explicit_tier_system_overrides_default(self):
        """Explicit tier='system' forces system tier for custom credentials."""
        resp = self.client.post(
            "/dashboard/api/credentials",
            json={"service": "my_custom_svc", "key": "abc", "tier": "system"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["tier"] == "system"
        vault = self.components["credential_vault"]
        vault.add_credential.assert_called_once_with(
            "my_custom_svc", "abc", system=True,
        )

    def test_invalid_tier_falls_through_to_auto_detect(self):
        """Invalid tier values are ignored; auto-detect applies."""
        resp = self.client.post(
            "/dashboard/api/credentials",
            json={"service": "my_svc", "key": "abc", "tier": "garbage"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["tier"] == "agent"


class TestDashboardSettingsProviderModels:
    """Tests for the settings endpoint provider model filtering."""

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.components = _make_components(self._tmpdir, include_v2=True)

    def teardown_method(self):
        _teardown(self.components)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_settings_returns_available_provider_models(self):
        """Settings includes available_provider_models filtered by credentials."""
        vault = self.components["credential_vault"]
        vault.get_providers_with_credentials.return_value = {"anthropic"}
        vault.system_credentials = {"anthropic_api_key": "sk-test"}
        client = _make_client(self.components)
        resp = client.get("/dashboard/api/settings")
        data = resp.json()
        assert "anthropic" in data["available_provider_models"]
        assert "openai" not in data["available_provider_models"]

    def test_settings_available_provider_models_empty_when_no_creds(self):
        """available_provider_models is empty when no credentials configured."""
        vault = self.components["credential_vault"]
        vault.get_providers_with_credentials.return_value = set()
        vault.system_credentials = {}
        client = _make_client(self.components)
        resp = client.get("/dashboard/api/settings")
        data = resp.json()
        assert data["available_provider_models"] == {}
        # Full model list is still available as fallback
        assert len(data["provider_models"]) > 0


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


# ── V2 Tests: Streaming Broadcast ────────────────────────────


def _parse_sse(resp_text: str) -> list[dict]:
    """Parse SSE response text into a list of event dicts."""
    import json

    return [
        json.loads(line[6:])
        for line in resp_text.strip().split("\n")
        if line.startswith("data: ")
    ]


class TestDashboardBroadcastStream:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.components = _make_components(self._tmpdir, include_v2=True)
        self.client = _make_client(self.components)

    def teardown_method(self):
        _teardown(self.components)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_broadcast_stream_returns_sse(self):
        """Streaming broadcast yields agent_start, agent_done, and all_done SSE events."""
        async def _mock_stream(aid, method, path, **kwargs):
            yield {"type": "text_delta", "content": f"Hello from {aid}"}

        self.components["transport"].stream_request = _mock_stream

        resp = self.client.post(
            "/dashboard/api/broadcast/stream",
            json={"message": "Hello all"},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        events = _parse_sse(resp.text)

        # Should have: agent_start(alpha), text_delta(alpha), agent_done(alpha),
        #              agent_start(beta), text_delta(beta), agent_done(beta), all_done
        types = [e["type"] for e in events]
        assert types.count("agent_start") == 2
        assert types.count("agent_done") == 2
        assert types[-1] == "all_done"

        # Each text_delta should carry agent info
        deltas = [e for e in events if e["type"] == "text_delta"]
        assert len(deltas) == 2
        delta_agents = {e["agent"] for e in deltas}
        assert delta_agents == {"alpha", "beta"}

    def test_broadcast_stream_empty_message_rejected(self):
        resp = self.client.post(
            "/dashboard/api/broadcast/stream",
            json={"message": ""},
        )
        assert resp.status_code == 400

    def test_broadcast_stream_no_transport(self):
        self.components["transport"] = None
        self.client = _make_client(self.components)
        resp = self.client.post(
            "/dashboard/api/broadcast/stream",
            json={"message": "Hello"},
        )
        assert resp.status_code == 503

    def test_broadcast_stream_handles_agent_error(self):
        """If one agent's stream raises, an error event is emitted for that agent."""
        async def _mock_stream(aid, method, path, **kwargs):
            if aid == "alpha":
                raise ConnectionError("agent down")
            yield {"type": "text_delta", "content": "ok"}

        self.components["transport"].stream_request = _mock_stream

        resp = self.client.post(
            "/dashboard/api/broadcast/stream",
            json={"message": "Hello"},
        )
        assert resp.status_code == 200
        events = _parse_sse(resp.text)

        error_events = [e for e in events if e["type"] == "error"]
        assert len(error_events) == 1
        assert error_events[0]["agent"] == "alpha"
        assert "agent down" in error_events[0]["message"]

        # all_done should still be emitted
        assert events[-1]["type"] == "all_done"

    def test_broadcast_stream_no_agents(self):
        """Streaming broadcast with no agents returns empty JSON response."""
        self.components["agent_registry"].clear()
        self.client = _make_client(self.components)
        resp = self.client.post(
            "/dashboard/api/broadcast/stream",
            json={"message": "Hello"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["responses"] == {}

    def test_broadcast_stream_does_not_mutate_source_events(self):
        """Streaming broadcast copies event dicts instead of mutating them."""
        captured_events = []

        async def _mock_stream(aid, method, path, **kwargs):
            evt = {"type": "text_delta", "content": "reply"}
            captured_events.append(evt)
            yield evt

        self.components["transport"].stream_request = _mock_stream

        resp = self.client.post(
            "/dashboard/api/broadcast/stream",
            json={"message": "test"},
        )
        assert resp.status_code == 200
        # Source events should not have "agent" key injected
        for evt in captured_events:
            assert "agent" not in evt


# ── Credential Scoping Tests ────────────────────────────────


class TestDashboardPermissionsEndpoint:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.components = _make_components(self._tmpdir, include_v2=True)
        # Set up permissions mock with get_permissions and get_allowed_credentials
        from src.shared.types import AgentPermissions
        self.components["permissions"].get_permissions.return_value = AgentPermissions(
            agent_id="alpha",
            allowed_credentials=["brightdata_*"],
            allowed_apis=["llm"],
        )
        self.components["permissions"].get_allowed_credentials.return_value = ["brightdata_*"]
        self.components["credential_vault"].list_agent_credential_names.return_value = [
            "brightdata_cdp_url", "myapp_password",
        ]
        self.client = _make_client(self.components)

    def teardown_method(self):
        _teardown(self.components)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_get_agent_permissions(self):
        resp = self.client.get("/dashboard/api/agents/alpha/permissions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent_id"] == "alpha"
        assert data["allowed_credentials"] == ["brightdata_*"]
        assert data["allowed_apis"] == ["llm"]
        assert "brightdata_cdp_url" in data["available_credentials"]
        assert "myapp_password" in data["available_credentials"]

    def test_get_permissions_not_found(self):
        resp = self.client.get("/dashboard/api/agents/nonexistent/permissions")
        assert resp.status_code == 404

    def test_put_agent_permissions(self):
        # Patch config loading/saving to avoid filesystem side effects
        with patch("src.cli.config._load_permissions", return_value={"permissions": {"alpha": {}}}), \
             patch("src.cli.config._save_permissions"):
            resp = self.client.put(
                "/dashboard/api/agents/alpha/permissions",
                json={"allowed_credentials": ["*"], "allowed_apis": ["llm", "brave_search"]},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "allowed_credentials" in data["updated"]
        assert "allowed_apis" in data["updated"]
        # Verify permissions.reload was called
        self.components["permissions"].reload.assert_called()

    def test_put_permissions_invalid_type(self):
        with patch("src.cli.config._load_permissions", return_value={"permissions": {"alpha": {}}}), \
             patch("src.cli.config._save_permissions"):
            resp = self.client.put(
                "/dashboard/api/agents/alpha/permissions",
                json={"allowed_credentials": "not_a_list"},
            )
        assert resp.status_code == 400


class TestDashboardSettingsAgentCredentials:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.components = _make_components(self._tmpdir, include_v2=True)
        self.components["credential_vault"].list_agent_credential_names.return_value = [
            "brightdata_cdp_url", "myapp_password",
        ]
        self.components["credential_vault"].list_system_credential_names.return_value = []
        self.client = _make_client(self.components)

    def teardown_method(self):
        _teardown(self.components)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_settings_includes_agent_credentials(self):
        resp = self.client.get("/dashboard/api/settings")
        assert resp.status_code == 200
        data = resp.json()
        assert "agent_credentials" in data
        assert "brightdata_cdp_url" in data["agent_credentials"]
        assert "myapp_password" in data["agent_credentials"]


class TestDashboardAgentConfigAllowedCredentials:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.components = _make_components(self._tmpdir, include_v2=True)
        self.components["permissions"].get_allowed_credentials.return_value = ["brightdata_*"]
        self.components["credential_vault"].list_credential_names.return_value = [
            "anthropic_api_key", "openai_api_key", "brightdata_cdp_url", "myapp_password",
        ]
        self.components["credential_vault"].list_agent_credential_names.return_value = [
            "brightdata_cdp_url", "myapp_password",
        ]
        self.components["credential_vault"].list_system_credential_names.return_value = [
            "anthropic_api_key", "openai_api_key",
        ]
        self.client = _make_client(self.components)

    def teardown_method(self):
        _teardown(self.components)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_config_includes_allowed_credentials(self):
        resp = self.client.get("/dashboard/api/agents/alpha/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "allowed_credentials" in data
        assert data["allowed_credentials"] == ["brightdata_*"]

    def test_config_includes_credential_visibility(self):
        """Config endpoint returns available, system, and resolved credentials."""
        resp = self.client.get("/dashboard/api/agents/alpha/config")
        assert resp.status_code == 200
        data = resp.json()
        # Agent-tier credentials available for assignment
        assert data["available_credentials"] == ["brightdata_cdp_url", "myapp_password"]
        # System credentials (always blocked)
        assert data["system_credentials"] == ["anthropic_api_key", "openai_api_key"]
        # Resolved: brightdata_* matches brightdata_cdp_url but not myapp_password
        assert data["resolved_credentials"] == ["brightdata_cdp_url"]

    def test_config_resolved_credentials_wildcard(self):
        """Wildcard pattern resolves to all agent-tier credentials."""
        self.components["permissions"].get_allowed_credentials.return_value = ["*"]
        self.client = _make_client(self.components)
        resp = self.client.get("/dashboard/api/agents/alpha/config")
        data = resp.json()
        assert data["resolved_credentials"] == ["brightdata_cdp_url", "myapp_password"]

    def test_config_resolved_credentials_none(self):
        """Empty allowed_credentials resolves to nothing."""
        self.components["permissions"].get_allowed_credentials.return_value = []
        self.client = _make_client(self.components)
        resp = self.client.get("/dashboard/api/agents/alpha/config")
        data = resp.json()
        assert data["resolved_credentials"] == []


def _make_full_client(components: dict) -> TestClient:
    """Build a TestClient with dashboard router + root-level SPA catch-all."""
    from src.dashboard.server import create_dashboard_router, create_spa_catchall_router

    router = create_dashboard_router(**components, mesh_port=8420)
    app = FastAPI()
    app.include_router(router)
    app.include_router(create_spa_catchall_router())  # Must be last
    return TestClient(app)


class TestDashboardSPACatchall:
    """Tests for the root-level SPA catch-all route that enables deep linking."""

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.components = _make_components(self._tmpdir)
        self.client = _make_full_client(self.components)

    def teardown_method(self):
        _teardown(self.components)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_catchall_serves_html_for_agent_path(self):
        resp = self.client.get("/agents/alice")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "window.__config" in resp.text

    def test_catchall_serves_html_for_activity(self):
        resp = self.client.get("/activity")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_catchall_serves_html_for_system(self):
        resp = self.client.get("/system")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_catchall_serves_html_for_activity_events(self):
        resp = self.client.get("/activity/events")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_catchall_serves_html_for_activity_logs(self):
        resp = self.client.get("/activity/logs")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_catchall_serves_html_for_nested_agent_path(self):
        resp = self.client.get("/agents/alice/memory")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_catchall_404_for_dashboard_paths(self):
        resp = self.client.get("/dashboard/agents/alice")
        assert resp.status_code == 404

    def test_catchall_404_for_mesh_paths(self):
        resp = self.client.get("/mesh/agents")
        assert resp.status_code == 404

    def test_existing_api_routes_not_shadowed(self):
        resp = self.client.get("/dashboard/api/agents")
        assert resp.status_code == 200
        data = resp.json()
        assert "agents" in data


class TestDashboardCacheBusting:
    """Tests for asset versioning and cache-control headers."""

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.components = _make_components(self._tmpdir)
        self.client = _make_full_client(self.components)

    def teardown_method(self):
        _teardown(self.components)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_asset_version_is_deterministic(self):
        from src.dashboard.server import ASSET_VERSION, _compute_asset_version
        assert ASSET_VERSION == _compute_asset_version()
        assert len(ASSET_VERSION) == 12

    def test_html_includes_versioned_static_urls(self):
        from src.dashboard.server import ASSET_VERSION
        resp = self.client.get("/dashboard/")
        assert resp.status_code == 200
        assert f"app.js?v={ASSET_VERSION}" in resp.text
        assert f"websocket.js?v={ASSET_VERSION}" in resp.text
        assert f"dashboard.css?v={ASSET_VERSION}" in resp.text

    def test_html_has_no_store_cache_control(self):
        resp = self.client.get("/dashboard/")
        assert resp.headers["cache-control"] == "no-store"

    def test_spa_catchall_has_no_store_cache_control(self):
        resp = self.client.get("/agents/alice")
        assert resp.headers["cache-control"] == "no-store"

    def test_static_with_version_param_is_cacheable(self):
        resp = self.client.get("/dashboard/static/js/app.js?v=abc123")
        assert resp.status_code == 200
        assert "max-age=86400" in resp.headers["cache-control"]
        assert "immutable" in resp.headers["cache-control"]

    def test_static_without_version_param_is_no_store(self):
        resp = self.client.get("/dashboard/static/js/app.js")
        assert resp.status_code == 200
        assert resp.headers["cache-control"] == "no-store"


class TestDashboardProjectAPI:
    """Tests for /api/projects and /api/project endpoints."""

    def setup_method(self):
        import yaml
        self._tmpdir = tempfile.mkdtemp()
        self.components = _make_components(self._tmpdir, include_v2=True)
        # Set up runtime mock with project_root
        self.components["runtime"].project_root = MagicMock()
        self.components["runtime"].project_root.__truediv__ = MagicMock(
            return_value=MagicMock(exists=MagicMock(return_value=False))
        )
        self.client = _make_client(self.components)

        # Create a project directory for tests
        self._projects_dir = os.path.join(self._tmpdir, "projects")
        proj_dir = os.path.join(self._projects_dir, "alpha")
        os.makedirs(proj_dir)
        with open(os.path.join(proj_dir, "metadata.yaml"), "w") as f:
            yaml.dump({"name": "alpha", "description": "Test", "members": ["bot1"]}, f)
        with open(os.path.join(proj_dir, "project.md"), "w") as f:
            f.write("# Alpha Project\nShared context here.")

    def teardown_method(self):
        _teardown(self.components)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_list_projects(self):
        with patch("src.cli.config.PROJECTS_DIR", MagicMock(exists=MagicMock(return_value=True),
                    glob=MagicMock(return_value=[]))):
            resp = self.client.get("/dashboard/api/projects")
        assert resp.status_code == 200
        data = resp.json()
        assert "projects" in data

    def test_project_read_path_traversal_blocked(self):
        """Path traversal in project name is rejected."""
        resp = self.client.get("/dashboard/api/project", params={"project": "../../etc"})
        assert resp.status_code == 400
        assert "Invalid project name" in resp.json()["detail"]

    def test_project_write_path_traversal_blocked(self):
        """Path traversal in project name is rejected for writes."""
        resp = self.client.put(
            "/dashboard/api/project",
            params={"project": "../../../tmp"},
            json={"content": "pwned"},
        )
        assert resp.status_code == 400
        assert "Invalid project name" in resp.json()["detail"]

    def test_project_read_valid_name(self):
        """Valid project name is accepted and reads project.md."""
        from pathlib import Path
        with patch("src.cli.config.PROJECTS_DIR", Path(self._projects_dir)):
            resp = self.client.get("/dashboard/api/project", params={"project": "alpha"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["project"] == "alpha"
        assert "Alpha Project" in data["content"]

    def test_project_read_requires_project_param(self):
        """GET /api/project without project param returns 400."""
        resp = self.client.get("/dashboard/api/project")
        assert resp.status_code == 400
        assert "required" in resp.json()["detail"]

    def test_project_write_requires_project_param(self):
        """PUT /api/project without project param returns 400."""
        resp = self.client.put("/dashboard/api/project", json={"content": "hello"})
        assert resp.status_code == 400
        assert "required" in resp.json()["detail"]


class TestDashboardProjectCRUD:
    """Tests for project CRUD endpoints."""

    def setup_method(self):
        import yaml
        self._tmpdir = tempfile.mkdtemp()
        self.components = _make_components(self._tmpdir, include_v2=True)
        self.components["runtime"].project_root = MagicMock()
        self.client = _make_client(self.components)

        # Create projects dir and a config with known agents
        self._projects_dir = os.path.join(self._tmpdir, "projects")
        os.makedirs(self._projects_dir, exist_ok=True)

        self._config_dir = os.path.join(self._tmpdir, "config")
        os.makedirs(self._config_dir, exist_ok=True)
        self._config_file = os.path.join(self._config_dir, "mesh.yaml")
        with open(self._config_file, "w") as f:
            yaml.dump({"agents": {"alpha": {"role": "test"}, "beta": {"role": "test"}}}, f)

        self._agents_file = os.path.join(self._config_dir, "agents.yaml")
        with open(self._agents_file, "w") as f:
            yaml.dump({"alpha": {"role": "test"}, "beta": {"role": "test"}}, f)

    def teardown_method(self):
        _teardown(self.components)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_create_project(self):
        """POST /api/projects creates a project directory."""
        from pathlib import Path
        with patch("src.cli.config.PROJECTS_DIR", Path(self._projects_dir)), \
             patch("src.cli.config.CONFIG_FILE", Path(self._config_file)), \
             patch("src.cli.config.AGENTS_FILE", Path(self._agents_file)), \
             patch("src.cli.config.PERMISSIONS_FILE", Path(self._tmpdir) / "perms.json"):
            resp = self.client.post("/dashboard/api/projects", json={
                "name": "myproject",
                "description": "A test project",
                "members": [],
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["created"] is True
        assert data["name"] == "myproject"
        assert os.path.isdir(os.path.join(self._projects_dir, "myproject"))

    def test_create_project_empty_name(self):
        """POST /api/projects with empty name returns 400."""
        resp = self.client.post("/dashboard/api/projects", json={
            "name": "",
            "description": "test",
        })
        assert resp.status_code == 400

    def test_create_project_unknown_members(self):
        """POST /api/projects with unknown member agents returns 400."""
        from pathlib import Path
        with patch("src.cli.config.CONFIG_FILE", Path(self._config_file)), \
             patch("src.cli.config.AGENTS_FILE", Path(self._agents_file)):
            resp = self.client.post("/dashboard/api/projects", json={
                "name": "proj",
                "members": ["nonexistent"],
            })
        assert resp.status_code == 400
        assert "Unknown agents" in resp.json()["detail"]

    def test_delete_project(self):
        """DELETE /api/projects/{name} removes the project."""
        from pathlib import Path

        import yaml
        # Create a project first
        proj_dir = os.path.join(self._projects_dir, "doomed")
        os.makedirs(proj_dir)
        with open(os.path.join(proj_dir, "metadata.yaml"), "w") as f:
            yaml.dump({"name": "doomed", "members": []}, f)

        with patch("src.cli.config.PROJECTS_DIR", Path(self._projects_dir)), \
             patch("src.cli.config.PERMISSIONS_FILE", Path(self._tmpdir) / "perms.json"):
            resp = self.client.delete("/dashboard/api/projects/doomed")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True
        assert not os.path.exists(proj_dir)

    def test_delete_project_not_found(self):
        """DELETE /api/projects/{name} for nonexistent project returns 404."""
        from pathlib import Path
        with patch("src.cli.config.PROJECTS_DIR", Path(self._projects_dir)):
            resp = self.client.delete("/dashboard/api/projects/ghost")
        assert resp.status_code == 404

    def test_add_member(self):
        """POST /api/projects/{name}/members adds an agent to the project."""
        from pathlib import Path

        import yaml
        proj_dir = os.path.join(self._projects_dir, "team")
        os.makedirs(proj_dir)
        with open(os.path.join(proj_dir, "metadata.yaml"), "w") as f:
            yaml.dump({"name": "team", "members": []}, f)

        with patch("src.cli.config.PROJECTS_DIR", Path(self._projects_dir)), \
             patch("src.cli.config.PERMISSIONS_FILE", Path(self._tmpdir) / "perms.json"):
            resp = self.client.post("/dashboard/api/projects/team/members", json={"agent": "alpha"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["added"] is True
        assert data["agent"] == "alpha"
        assert "restarted" in data
        assert isinstance(data["restarted"], bool)

    def test_add_member_missing_agent(self):
        """POST /api/projects/{name}/members without agent returns 400."""
        resp = self.client.post("/dashboard/api/projects/team/members", json={})
        assert resp.status_code == 400

    def test_remove_member(self):
        """DELETE /api/projects/{name}/members/{agent} removes the agent."""
        from pathlib import Path

        import yaml
        proj_dir = os.path.join(self._projects_dir, "team")
        os.makedirs(proj_dir, exist_ok=True)
        with open(os.path.join(proj_dir, "metadata.yaml"), "w") as f:
            yaml.dump({"name": "team", "members": ["alpha"]}, f)

        with patch("src.cli.config.PROJECTS_DIR", Path(self._projects_dir)), \
             patch("src.cli.config.PERMISSIONS_FILE", Path(self._tmpdir) / "perms.json"):
            resp = self.client.delete("/dashboard/api/projects/team/members/alpha")
        assert resp.status_code == 200
        data = resp.json()
        assert data["removed"] is True
        assert "restarted" in data
        assert isinstance(data["restarted"], bool)

    def test_remove_member_not_found(self):
        """DELETE /api/projects/{name}/members/{agent} for nonexistent project returns 400."""
        from pathlib import Path
        with patch("src.cli.config.PROJECTS_DIR", Path(self._projects_dir)):
            resp = self.client.delete("/dashboard/api/projects/ghost/members/alpha")
        assert resp.status_code == 400


class TestDashboardAgentProjectField:
    """Tests for project field in /api/agents response."""

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.components = _make_components(self._tmpdir)
        self.client = _make_client(self.components)

    def teardown_method(self):
        _teardown(self.components)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_agents_include_project_field(self):
        """Agent entries include a project field (None when no project assigned)."""
        with patch("src.cli.config._load_config", return_value={
            "agents": {"alpha": {"role": "coder"}, "beta": {"role": "writer"}},
            "llm": {"default_model": "openai/gpt-4o-mini"},
            "_agent_projects": {"alpha": "myproject"},
        }):
            resp = self.client.get("/dashboard/api/agents")
        assert resp.status_code == 200
        data = resp.json()
        alpha = next(a for a in data["agents"] if a["id"] == "alpha")
        beta = next(a for a in data["agents"] if a["id"] == "beta")
        assert alpha["project"] == "myproject"
        assert beta["project"] is None

    def test_agents_project_field_absent_when_no_projects(self):
        """When no projects configured, project field is None for all agents."""
        with patch("src.cli.config._load_config", return_value={
            "agents": {"alpha": {}, "beta": {}},
            "llm": {"default_model": "openai/gpt-4o-mini"},
            "_agent_projects": {},
        }):
            resp = self.client.get("/dashboard/api/agents")
        assert resp.status_code == 200
        data = resp.json()
        for agent in data["agents"]:
            assert agent["project"] is None

    def test_agents_project_field_when_agent_projects_key_missing(self):
        """When _agent_projects key is absent from config, project is None."""
        with patch("src.cli.config._load_config", return_value={
            "agents": {"alpha": {}, "beta": {}},
            "llm": {"default_model": "openai/gpt-4o-mini"},
        }):
            resp = self.client.get("/dashboard/api/agents")
        assert resp.status_code == 200
        data = resp.json()
        for agent in data["agents"]:
            assert agent["project"] is None


class TestDashboardBroadcastProjectScoping:
    """Tests for project-scoped broadcast endpoints."""

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.components = _make_components(self._tmpdir, include_v2=True)
        # Add a third agent for richer filtering tests
        self.components["agent_registry"]["gamma"] = "http://localhost:8403"
        self.client = _make_client(self.components)

    def teardown_method(self):
        _teardown(self.components)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_broadcast_stream_scoped_to_project(self):
        """When project is set, only project members receive the broadcast."""
        streamed_agents = []

        async def _mock_stream(aid, method, path, **kwargs):
            streamed_agents.append(aid)
            yield {"type": "text_delta", "content": f"Hello from {aid}"}

        self.components["transport"].stream_request = _mock_stream

        with patch("src.cli.config._load_projects", return_value={
            "myproject": {"members": ["alpha", "gamma"]},
        }):
            resp = self.client.post(
                "/dashboard/api/broadcast/stream",
                json={"message": "Hello project", "project": "myproject"},
            )
        assert resp.status_code == 200
        events = _parse_sse(resp.text)

        # Only alpha and gamma should be streamed (beta excluded)
        agent_starts = [e["agent"] for e in events if e["type"] == "agent_start"]
        assert set(agent_starts) == {"alpha", "gamma"}
        assert "beta" not in streamed_agents

    def test_broadcast_stream_without_project_sends_to_all(self):
        """Without project field, broadcast goes to all agents."""
        async def _mock_stream(aid, method, path, **kwargs):
            yield {"type": "text_delta", "content": "ok"}

        self.components["transport"].stream_request = _mock_stream

        resp = self.client.post(
            "/dashboard/api/broadcast/stream",
            json={"message": "Hello all"},
        )
        assert resp.status_code == 200
        events = _parse_sse(resp.text)

        agent_starts = [e["agent"] for e in events if e["type"] == "agent_start"]
        assert set(agent_starts) == {"alpha", "beta", "gamma"}

    def test_broadcast_non_stream_scoped_to_project(self):
        """Non-streaming broadcast with project only sends to members."""
        sent_to = []

        async def _mock_request(aid, method, path, **kwargs):
            sent_to.append(aid)
            return {"response": f"Reply from {aid}"}

        self.components["transport"].request = AsyncMock(side_effect=_mock_request)

        with patch("src.cli.config._load_projects", return_value={
            "proj1": {"members": ["beta"]},
        }):
            resp = self.client.post(
                "/dashboard/api/broadcast",
                json={"message": "Hello project", "project": "proj1"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "beta" in data["responses"]
        assert "alpha" not in data["responses"]
        assert "gamma" not in data["responses"]

    def test_broadcast_non_stream_without_project_sends_to_all(self):
        """Non-streaming broadcast without project sends to all agents."""
        async def _mock_request(aid, method, path, **kwargs):
            return {"response": f"Reply from {aid}"}

        self.components["transport"].request = AsyncMock(side_effect=_mock_request)

        resp = self.client.post(
            "/dashboard/api/broadcast",
            json={"message": "Hello all"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert set(data["responses"].keys()) == {"alpha", "beta", "gamma"}

    def test_broadcast_stream_project_no_matching_agents(self):
        """Streaming broadcast with project that has no running members."""
        with patch("src.cli.config._load_projects", return_value={
            "empty_proj": {"members": ["nonexistent"]},
        }):
            resp = self.client.post(
                "/dashboard/api/broadcast/stream",
                json={"message": "Hello", "project": "empty_proj"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["responses"] == {}
        assert "No agents" in data.get("message", "")

    def test_broadcast_non_stream_project_no_matching_agents(self):
        """Non-streaming broadcast with project that has no running members."""
        with patch("src.cli.config._load_projects", return_value={
            "empty_proj": {"members": ["nonexistent"]},
        }):
            resp = self.client.post(
                "/dashboard/api/broadcast",
                json={"message": "Hello", "project": "empty_proj"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["responses"] == {}
        assert "No matching agents" in data.get("message", "")

    def test_broadcast_empty_string_project_sends_to_all(self):
        """Empty string project is treated as no project filter."""
        async def _mock_stream(aid, method, path, **kwargs):
            yield {"type": "text_delta", "content": "ok"}

        self.components["transport"].stream_request = _mock_stream

        resp = self.client.post(
            "/dashboard/api/broadcast/stream",
            json={"message": "Hello", "project": ""},
        )
        assert resp.status_code == 200
        events = _parse_sse(resp.text)
        agent_starts = [e["agent"] for e in events if e["type"] == "agent_start"]
        assert set(agent_starts) == {"alpha", "beta", "gamma"}

    def test_broadcast_stream_all_agents_are_members(self):
        """Streaming broadcast where all running agents are project members."""
        async def _mock_stream(aid, method, path, **kwargs):
            yield {"type": "text_delta", "content": f"ok from {aid}"}

        self.components["transport"].stream_request = _mock_stream

        with patch("src.cli.config._load_projects", return_value={
            "full_proj": {"members": ["alpha", "beta", "gamma"]},
        }):
            resp = self.client.post(
                "/dashboard/api/broadcast/stream",
                json={"message": "Hello", "project": "full_proj"},
            )
        assert resp.status_code == 200
        events = _parse_sse(resp.text)
        agent_starts = [e["agent"] for e in events if e["type"] == "agent_start"]
        assert set(agent_starts) == {"alpha", "beta", "gamma"}

    def test_broadcast_project_type_validation(self):
        """Non-string project value returns 400."""
        resp = self.client.post(
            "/dashboard/api/broadcast/stream",
            json={"message": "Hello", "project": 123},
        )
        assert resp.status_code == 400
        assert "string" in resp.json()["detail"]

        resp = self.client.post(
            "/dashboard/api/broadcast",
            json={"message": "Hello", "project": ["bad"]},
        )
        assert resp.status_code == 400
        assert "string" in resp.json()["detail"]

    def test_broadcast_stream_standalone_only(self):
        """Streaming broadcast with standalone=true targets only unassigned agents."""
        with patch("src.cli.config._load_projects", return_value={
            "proj1": {"members": ["alpha", "beta"]},
        }):
            resp = self.client.post(
                "/dashboard/api/broadcast/stream",
                json={"message": "Hello standalone", "standalone": True},
            )
        assert resp.status_code == 200
        lines = resp.text.strip().split("\n")
        agent_starts = [line for line in lines if "agent_start" in line]
        # Only gamma is unassigned
        assert len(agent_starts) == 1
        assert "gamma" in agent_starts[0]

    def test_broadcast_non_stream_standalone_only(self):
        """Non-streaming broadcast with standalone=true targets only unassigned agents."""
        async def _mock_request(aid, method, path, **kwargs):
            return {"response": f"Reply from {aid}"}

        self.components["transport"].request = AsyncMock(side_effect=_mock_request)

        with patch("src.cli.config._load_projects", return_value={
            "proj1": {"members": ["alpha", "beta"]},
        }):
            resp = self.client.post(
                "/dashboard/api/broadcast",
                json={"message": "Hello standalone", "standalone": True},
            )
        assert resp.status_code == 200
        data = resp.json()
        # Only gamma is unassigned
        assert set(data["responses"].keys()) == {"gamma"}


class TestLogsEndpoint:
    def test_logs_endpoint(self, tmp_path):
        """GET /dashboard/api/logs returns log lines."""
        log_path = tmp_path / ".openlegion.log"
        log_path.write_text(
            '{"level":"INFO","msg":"test1"}\n{"level":"ERROR","msg":"test2"}\n'
        )

        components = _make_components(str(tmp_path))
        client = _make_client(components)

        with patch("src.cli.config.PROJECT_ROOT", tmp_path):
            resp = client.get("/dashboard/api/logs")
            assert resp.status_code == 200
            data = resp.json()
            assert "lines" in data
            assert len(data["lines"]) == 2
        _teardown(components)

    def test_logs_level_filter(self, tmp_path):
        """GET /dashboard/api/logs?level=error filters by level."""
        log_path = tmp_path / ".openlegion.log"
        log_path.write_text(
            '{"level":"INFO","msg":"test1"}\n{"level":"ERROR","msg":"test2"}\n'
        )

        components = _make_components(str(tmp_path))
        client = _make_client(components)

        with patch("src.cli.config.PROJECT_ROOT", tmp_path):
            resp = client.get("/dashboard/api/logs?level=error")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["lines"]) == 1
            assert "ERROR" in data["lines"][0]
        _teardown(components)

    def test_logs_missing_file(self, tmp_path):
        """GET /dashboard/api/logs returns empty when log file missing."""
        components = _make_components(str(tmp_path))
        client = _make_client(components)

        with patch("src.cli.config.PROJECT_ROOT", tmp_path):
            resp = client.get("/dashboard/api/logs")
            assert resp.status_code == 200
            data = resp.json()
            assert data["lines"] == []
            assert data["total"] == 0
        _teardown(components)


class TestDashboardChannels:
    """Tests for channel management endpoints."""

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.components = _make_components(self._tmpdir, include_v2=True)
        self.client = _make_client(self.components)

    def teardown_method(self):
        _teardown(self.components)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_channels_list(self):
        resp = self.client.get("/dashboard/api/channels")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["channels"]) == 4
        types = {ch["type"] for ch in data["channels"]}
        assert types == {"telegram", "discord", "slack", "whatsapp"}
        assert all(not ch["connected"] for ch in data["channels"])

    def test_channel_connect_invalid_type(self):
        resp = self.client.post(
            "/dashboard/api/channels/irc/connect",
            json={"tokens": {"token": "abc"}},
        )
        assert resp.status_code == 400
        assert "Unknown" in resp.json()["detail"]

    def test_channel_connect_missing_tokens(self):
        resp = self.client.post(
            "/dashboard/api/channels/telegram/connect",
            json={"tokens": {}},
        )
        assert resp.status_code == 400
        assert "Missing" in resp.json()["detail"]

    def test_channel_connect_success(self):
        resp = self.client.post(
            "/dashboard/api/channels/telegram/connect",
            json={"tokens": {"token": "123:ABC"}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["connected"] is True
        assert data["type"] == "telegram"
        self.components["channel_manager"].start_channel.assert_called_once_with(
            "telegram", {"token": "123:ABC"},
        )

    def test_channel_connect_already_connected(self):
        self.components["channel_manager"].start_channel.side_effect = ValueError("telegram is already connected")
        resp = self.client.post(
            "/dashboard/api/channels/telegram/connect",
            json={"tokens": {"token": "123:ABC"}},
        )
        assert resp.status_code == 400
        assert "already connected" in resp.json()["detail"]

    def test_channel_connect_persists_credentials(self):
        resp = self.client.post(
            "/dashboard/api/channels/telegram/connect",
            json={"tokens": {"token": "123:ABC"}},
        )
        assert resp.status_code == 200
        self.components["credential_vault"].add_credential.assert_called_once_with(
            "TELEGRAM_BOT_TOKEN", "123:ABC", system=True,
        )

    def test_channel_connect_slack_multi_token(self):
        resp = self.client.post(
            "/dashboard/api/channels/slack/connect",
            json={"tokens": {"bot_token": "xoxb-123"}},
        )
        assert resp.status_code == 400
        assert "app_token" in resp.json()["detail"]

    def test_channel_connect_rollback_on_failure(self):
        self.components["channel_manager"].start_channel.side_effect = ValueError("bad token")
        resp = self.client.post(
            "/dashboard/api/channels/telegram/connect",
            json={"tokens": {"token": "123:ABC"}},
        )
        assert resp.status_code == 400
        # Credential should have been added then rolled back
        self.components["credential_vault"].add_credential.assert_called_once()
        self.components["credential_vault"].remove_credential.assert_called_once_with(
            "TELEGRAM_BOT_TOKEN",
        )

    def test_channel_disconnect_invalid_type(self):
        resp = self.client.post(
            "/dashboard/api/channels/irc/disconnect",
            json={},
        )
        assert resp.status_code == 400
        assert "Unknown" in resp.json()["detail"]

    def test_channel_disconnect_removes_credentials(self):
        resp = self.client.post(
            "/dashboard/api/channels/telegram/disconnect",
            json={},
        )
        assert resp.status_code == 200
        self.components["credential_vault"].remove_credential.assert_called_once_with(
            "TELEGRAM_BOT_TOKEN",
        )

    def test_channel_disconnect_not_connected(self):
        self.components["channel_manager"].stop_channel.side_effect = ValueError("telegram is not connected")
        resp = self.client.post(
            "/dashboard/api/channels/telegram/disconnect",
            json={},
        )
        assert resp.status_code == 400
        assert "not connected" in resp.json()["detail"]

    def test_channel_disconnect_success(self):
        resp = self.client.post(
            "/dashboard/api/channels/telegram/disconnect",
            json={},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["disconnected"] is True
        self.components["channel_manager"].stop_channel.assert_called_once_with("telegram")


class TestCredentialValidation:
    """Tests for POST /api/credentials/validate — exception handling."""

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.components = _make_components(self._tmpdir, include_v2=True)
        self.client = _make_client(self.components)

    def teardown_method(self):
        _teardown(self.components)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_validate_auth_error_returns_invalid(self):
        """AuthenticationError → valid: False."""
        import litellm
        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            side_effect=litellm.AuthenticationError(
                message="Invalid API key",
                llm_provider="anthropic",
                model="anthropic/claude-haiku-4-5-20251001",
            ),
        ):
            resp = self.client.post(
                "/dashboard/api/credentials/validate",
                json={"service": "anthropic_api_key", "key": "sk-bad"},
            )
        data = resp.json()
        assert data["valid"] is False
        assert data["skipped"] is False

    def test_validate_permission_denied_returns_invalid(self):
        """PermissionDeniedError → valid: False."""
        import httpx
        import litellm
        mock_resp = httpx.Response(status_code=403, request=httpx.Request("POST", "https://api.anthropic.com"))
        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            side_effect=litellm.PermissionDeniedError(
                message="Permission denied",
                llm_provider="anthropic",
                model="anthropic/claude-haiku-4-5-20251001",
                response=mock_resp,
            ),
        ):
            resp = self.client.post(
                "/dashboard/api/credentials/validate",
                json={"service": "anthropic_api_key", "key": "sk-bad"},
            )
        data = resp.json()
        assert data["valid"] is False
        assert data["skipped"] is False
        assert "Permission denied" in data["reason"]

    def test_validate_rate_limit_allows_save(self):
        """RateLimitError → valid: True, skipped: True (transient)."""
        import litellm
        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            side_effect=litellm.RateLimitError(
                message="Rate limit exceeded",
                llm_provider="anthropic",
                model="anthropic/claude-haiku-4-5-20251001",
            ),
        ):
            resp = self.client.post(
                "/dashboard/api/credentials/validate",
                json={"service": "anthropic_api_key", "key": "sk-test"},
            )
        data = resp.json()
        assert data["valid"] is True
        assert data["skipped"] is True

    def test_validate_unknown_error_returns_invalid(self):
        """Unknown Exception → valid: False (safe default)."""
        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Something unexpected"),
        ):
            resp = self.client.post(
                "/dashboard/api/credentials/validate",
                json={"service": "anthropic_api_key", "key": "sk-test"},
            )
        data = resp.json()
        assert data["valid"] is False
        assert "Validation failed" in data["reason"]

    def test_validate_success(self):
        """Successful completion → valid: True."""
        mock_response = MagicMock()
        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            resp = self.client.post(
                "/dashboard/api/credentials/validate",
                json={"service": "anthropic_api_key", "key": "sk-valid"},
            )
        data = resp.json()
        assert data["valid"] is True
        assert data["skipped"] is False
