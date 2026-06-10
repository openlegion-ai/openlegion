"""Dashboard endpoint tests for the embedding / semantic-memory selector.

Covers ``POST /dashboard/api/embedding-model`` (writes ``llm.embedding_model``
to ``config/mesh.yaml``) and the ``embedding`` block added to
``GET /dashboard/api/system-settings``.

Harness mirrors ``tests/test_dashboard_ui.py`` (``create_dashboard_router`` →
``FastAPI`` → ``TestClient``). The router-level ``_csrf_check`` requires the
``X-Requested-With`` header on state-changing requests, so POSTs set it
explicitly (the browser fetch monkey-patch that auto-adds it isn't in play
under ``TestClient``).

The POST handler writes a *relative* ``config/mesh.yaml`` (cwd-anchored) while
``GET`` reads it through ``cli.config._load_config`` → ``CONFIG_FILE`` (an
absolute path). We point both at the same temp file by ``chdir``-ing into a
tmp dir AND patching ``CONFIG_FILE``.
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.cli.config as cli_config

_CSRF = {"X-Requested-With": "XMLHttpRequest"}


class TestEmbeddingModelEndpoint:
    def setup_method(self):
        from unittest.mock import MagicMock

        from src.dashboard.events import EventBus
        from src.dashboard.server import create_dashboard_router
        from src.dashboard.telemetry import DashboardTelemetry
        from src.host.costs import CostTracker
        from src.host.health import HealthMonitor
        from src.host.mesh import Blackboard
        from src.host.traces import TraceStore

        self._tmpdir = tempfile.mkdtemp()
        bb = Blackboard(db_path=os.path.join(self._tmpdir, "bb.db"))
        cost_tracker = CostTracker(db_path=os.path.join(self._tmpdir, "costs.db"))
        trace_store = TraceStore(db_path=os.path.join(self._tmpdir, "traces.db"))
        runtime_mock = MagicMock()
        runtime_mock.browser_vnc_url = None
        runtime_mock.browser_service_url = None
        runtime_mock.browser_auth_token = ""
        # Real dict so the POST handler's extra_env refresh is observable
        # (containers read EMBEDDING_MODEL/DIM from extra_env at start).
        runtime_mock.extra_env = {}
        self.runtime_mock = runtime_mock
        health_monitor = HealthMonitor(
            runtime=runtime_mock,
            transport=MagicMock(),
            router=MagicMock(),
        )
        self.components = {
            "blackboard": bb,
            "health_monitor": health_monitor,
            "cost_tracker": cost_tracker,
            "trace_store": trace_store,
            "event_bus": EventBus(),
            "agent_registry": {},
        }
        self.telemetry = DashboardTelemetry(
            db_path=os.path.join(self._tmpdir, "telemetry.db"),
        )
        router = create_dashboard_router(
            **self.components, mesh_port=8420, telemetry=self.telemetry,
            runtime=runtime_mock,
        )
        app = FastAPI()
        app.include_router(router)
        self.client = TestClient(app)

    def teardown_method(self):
        self.telemetry.close()
        self.components["cost_tracker"].close()
        self.components["trace_store"].close()
        self.components["blackboard"].close()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    @pytest.fixture
    def mesh_yaml(self, tmp_path, monkeypatch):
        """Anchor both the POST write and the GET read at one temp mesh.yaml."""
        cfg_path = tmp_path / "config" / "mesh.yaml"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        monkeypatch.chdir(tmp_path)  # POST writes relative config/mesh.yaml
        monkeypatch.setattr(cli_config, "CONFIG_FILE", cfg_path)  # GET reads it
        return cfg_path

    def _read_embedding(self, cfg_path: Path):
        data = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
        return (data or {}).get("llm", {}).get("embedding_model", "__ABSENT__")

    # ── POST /api/embedding-model ───────────────────────────────────────

    def test_provider_writes_model_string(self, mesh_yaml, monkeypatch):
        monkeypatch.setenv("OPENLEGION_SYSTEM_VOYAGE_API_KEY", "vk-test")
        resp = self.client.post(
            "/dashboard/api/embedding-model",
            json={"value": "voyage"},
            headers=_CSRF,
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body == {"value": "voyage", "embedding_model": "voyage/voyage-3.5"}
        assert self._read_embedding(mesh_yaml) == "voyage/voyage-3.5"

    def test_none_stores_none(self, mesh_yaml):
        resp = self.client.post(
            "/dashboard/api/embedding-model",
            json={"value": "none"},
            headers=_CSRF,
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["embedding_model"] == "none"
        assert self._read_embedding(mesh_yaml) == "none"

    def test_auto_removes_key(self, mesh_yaml):
        # Seed an explicit value, then "auto" must remove it (and leave
        # other llm keys intact).
        mesh_yaml.write_text(
            yaml.dump(
                {"llm": {"default_model": "openai/gpt-4o-mini",
                         "embedding_model": "voyage/voyage-3.5"}},
            )
        )
        resp = self.client.post(
            "/dashboard/api/embedding-model",
            json={"value": "auto"},
            headers=_CSRF,
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["embedding_model"] is None
        assert self._read_embedding(mesh_yaml) == "__ABSENT__"
        # Sibling llm key preserved.
        data = yaml.safe_load(mesh_yaml.read_text())
        assert data["llm"]["default_model"] == "openai/gpt-4o-mini"

    def test_bogus_value_400(self, mesh_yaml):
        resp = self.client.post(
            "/dashboard/api/embedding-model",
            json={"value": "bogus"},
            headers=_CSRF,
        )
        assert resp.status_code == 400

    def test_empty_value_400(self, mesh_yaml):
        resp = self.client.post(
            "/dashboard/api/embedding-model",
            json={"value": ""},
            headers=_CSRF,
        )
        assert resp.status_code == 400

    def test_provider_post_refreshes_runtime_extra_env(
        self, mesh_yaml, monkeypatch,
    ):
        # Agent containers read EMBEDDING_MODEL/DIM from runtime.extra_env at
        # container start — extra_env is otherwise populated only at engine
        # boot. Without the write-time refresh, "save + restart agents" would
        # restart containers with the OLD embedding env.
        monkeypatch.setenv("OPENLEGION_SYSTEM_VOYAGE_API_KEY", "vk-test")
        resp = self.client.post(
            "/dashboard/api/embedding-model",
            json={"value": "voyage"},
            headers=_CSRF,
        )
        assert resp.status_code == 200, resp.text
        assert self.runtime_mock.extra_env["EMBEDDING_MODEL"] == "voyage/voyage-3.5"
        assert self.runtime_mock.extra_env["EMBEDDING_DIM"] == "1024"

    def test_none_post_refreshes_runtime_extra_env(self, mesh_yaml, monkeypatch):
        # "none" must propagate too — keyword-only memory on next restart.
        for p in ("OPENAI", "VOYAGE", "GEMINI", "COHERE"):
            monkeypatch.delenv(f"OPENLEGION_SYSTEM_{p}_API_KEY", raising=False)
        resp = self.client.post(
            "/dashboard/api/embedding-model",
            json={"value": "none"},
            headers=_CSRF,
        )
        assert resp.status_code == 200, resp.text
        assert self.runtime_mock.extra_env["EMBEDDING_MODEL"] == "none"

    def test_provider_without_key_rejected(self, mesh_yaml, monkeypatch):
        # A provider with no configured API key is rejected — persisting it
        # would mint a dead embedding model agents restart into.
        monkeypatch.delenv("OPENLEGION_SYSTEM_VOYAGE_API_KEY", raising=False)
        resp = self.client.post(
            "/dashboard/api/embedding-model",
            json={"value": "voyage"},
            headers=_CSRF,
        )
        assert resp.status_code == 400
        assert self._read_embedding(mesh_yaml) == "__ABSENT__"

    # ── GET /api/system-settings embedding block ────────────────────────

    def test_system_settings_embedding_block_off(self, mesh_yaml, monkeypatch):
        # No embedding-capable keys → effective "none" → off.
        for p in ("OPENAI", "VOYAGE", "GEMINI", "COHERE"):
            monkeypatch.delenv(f"OPENLEGION_SYSTEM_{p}_API_KEY", raising=False)
        resp = self.client.get("/dashboard/api/system-settings")
        assert resp.status_code == 200, resp.text
        emb = resp.json()["embedding"]
        assert emb["configured_provider"] == "auto"  # absent key → auto
        assert emb["on"] is False
        assert emb["effective_model"] == "none"
        assert emb["available_providers"] == []

    def test_system_settings_embedding_block_on(self, mesh_yaml, monkeypatch):
        # A Voyage key present, embedding_model absent → resolver auto-picks
        # the first laddered keyed provider.
        for p in ("OPENAI", "VOYAGE", "GEMINI", "COHERE"):
            monkeypatch.delenv(f"OPENLEGION_SYSTEM_{p}_API_KEY", raising=False)
        monkeypatch.setenv("OPENLEGION_SYSTEM_VOYAGE_API_KEY", "vk-test")
        resp = self.client.get("/dashboard/api/system-settings")
        assert resp.status_code == 200, resp.text
        emb = resp.json()["embedding"]
        assert emb["configured_provider"] == "auto"
        assert emb["on"] is True
        assert emb["effective_model"] == "voyage/voyage-3.5"
        assert "voyage" in emb["available_providers"]

    def test_system_settings_embedding_explicit_provider(
        self, mesh_yaml, monkeypatch,
    ):
        monkeypatch.setenv("OPENLEGION_SYSTEM_OPENAI_API_KEY", "sk-test")
        mesh_yaml.write_text(
            yaml.dump({"llm": {"embedding_model": "text-embedding-3-small"}})
        )
        resp = self.client.get("/dashboard/api/system-settings")
        assert resp.status_code == 200, resp.text
        emb = resp.json()["embedding"]
        assert emb["configured_provider"] == "openai"
        assert emb["configured"] == "text-embedding-3-small"
        assert emb["on"] is True

    def test_system_settings_explicit_provider_no_key_is_off(
        self, mesh_yaml, monkeypatch,
    ):
        # Explicit model whose provider key was removed → honest OFF status.
        for p in ("OPENAI", "VOYAGE", "GEMINI", "COHERE"):
            monkeypatch.delenv(f"OPENLEGION_SYSTEM_{p}_API_KEY", raising=False)
        mesh_yaml.write_text(
            yaml.dump({"llm": {"embedding_model": "voyage/voyage-3.5"}})
        )
        resp = self.client.get("/dashboard/api/system-settings")
        assert resp.status_code == 200, resp.text
        emb = resp.json()["embedding"]
        assert emb["configured_provider"] == "voyage"
        assert emb["on"] is False  # provider has no key → not actually on

    def test_system_settings_embedding_custom(self, mesh_yaml, monkeypatch):
        # A model not in the ladder → "custom".
        mesh_yaml.write_text(
            yaml.dump({"llm": {"embedding_model": "some/exotic-embed-v9"}})
        )
        resp = self.client.get("/dashboard/api/system-settings")
        assert resp.status_code == 200, resp.text
        emb = resp.json()["embedding"]
        assert emb["configured_provider"] == "custom"
        assert emb["configured"] == "some/exotic-embed-v9"
