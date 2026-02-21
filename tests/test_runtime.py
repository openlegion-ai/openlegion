"""Tests for RuntimeBackend (DockerBackend and SandboxBackend)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from src.cli.runtime import _default_embedding_model
from src.host.runtime import (
    DockerBackend,
    RuntimeBackend,
    SandboxBackend,
    sandbox_available,
    select_backend,
)

# ── RuntimeBackend interface ──────────────────────────────────

class TestRuntimeBackendInterface:
    def test_containers_alias(self):
        """The `containers` property is a backward-compat alias for `agents`."""
        class StubBackend(RuntimeBackend):
            def start_agent(self, *a, **kw): ...
            def stop_agent(self, *a, **kw): ...
            def health_check(self, *a, **kw): ...
            def get_logs(self, *a, **kw): ...
            async def wait_for_agent(self, *a, **kw): ...

        b = StubBackend.__new__(StubBackend)
        b.agents = {"x": {"url": "http://localhost:1", "role": "test"}}
        assert b.containers is b.agents

    def test_get_container_logs_alias(self):
        class StubBackend(RuntimeBackend):
            def start_agent(self, *a, **kw): ...
            def stop_agent(self, *a, **kw): ...
            def health_check(self, *a, **kw): ...
            def get_logs(self, agent_id, tail=40):
                return f"logs for {agent_id} tail={tail}"
            async def wait_for_agent(self, *a, **kw): ...

        b = StubBackend.__new__(StubBackend)
        b.agents = {}
        assert b.get_container_logs("x", tail=10) == "logs for x tail=10"

    def test_list_agents(self):
        class StubBackend(RuntimeBackend):
            def start_agent(self, *a, **kw): ...
            def stop_agent(self, *a, **kw): ...
            def health_check(self, *a, **kw): ...
            def get_logs(self, *a, **kw): ...
            async def wait_for_agent(self, *a, **kw): ...

        b = StubBackend.__new__(StubBackend)
        b.agents = {
            "a": {"url": "http://localhost:1", "role": "r1"},
            "b": {"url": "http://localhost:2", "role": "r2"},
        }
        result = b.list_agents()
        assert set(result.keys()) == {"a", "b"}
        assert result["a"]["role"] == "r1"


# ── DockerBackend ─────────────────────────────────────────────

class TestDockerBackend:
    def test_backend_name(self):
        assert DockerBackend.backend_name() == "docker"


# ── SandboxBackend ────────────────────────────────────────────

class TestSandboxBackend:
    def test_backend_name(self):
        assert SandboxBackend.backend_name() == "sandbox"

    def test_prepare_workspace(self, tmp_path):
        """Workspace directory is created with expected structure."""
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / "PROJECT.md").write_text("# Test Project")

        skills_src = project_root / "skills" / "alpha"
        skills_src.mkdir(parents=True)
        (skills_src / "my_skill.py").write_text("# skill")

        backend = SandboxBackend.__new__(SandboxBackend)
        backend.project_root = project_root
        backend.mesh_host_port = 8420
        backend.agents = {}
        backend.auth_tokens = {}
        backend.extra_env = {}
        backend._workspace_root = tmp_path / ".openlegion" / "agents"
        backend._workspace_root.mkdir(parents=True)

        ws = backend._prepare_workspace(
            agent_id="alpha",
            role="test",
            skills_dir=str(skills_src),
            system_prompt="You are test",
            model="openai/gpt-4o-mini",
        )

        assert ws.exists()
        assert (ws / "data" / "workspace").is_dir()
        assert (ws / "PROJECT.md").read_text() == "# Test Project"
        assert (ws / "skills" / "my_skill.py").read_text() == "# skill"
        assert (ws / ".agent.env").exists()

        env_content = (ws / ".agent.env").read_text()
        assert "AGENT_ID=alpha" in env_content
        assert "AGENT_ROLE=test" in env_content
        assert "LLM_MODEL=openai/gpt-4o-mini" in env_content
        assert "MESH_AUTH_TOKEN=" in env_content
        assert "alpha" in backend.auth_tokens

    def test_prepare_workspace_browser_backend(self, tmp_path):
        """browser_backend config is passed as BROWSER_BACKEND env var."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        backend = SandboxBackend.__new__(SandboxBackend)
        backend.project_root = project_root
        backend.mesh_host_port = 8420
        backend.agents = {}
        backend.auth_tokens = {}
        backend.extra_env = {}
        backend._workspace_root = tmp_path / ".openlegion" / "agents"
        backend._workspace_root.mkdir(parents=True)

        ws = backend._prepare_workspace(
            agent_id="beta",
            role="scraper",
            skills_dir="",
            system_prompt="You scrape",
            model="openai/gpt-4o-mini",
            browser_backend="stealth",
        )

        env_content = (ws / ".agent.env").read_text()
        assert "BROWSER_BACKEND=stealth" in env_content

    def test_prepare_workspace_no_browser_backend(self, tmp_path):
        """Without browser_backend, BROWSER_BACKEND is not in env."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        backend = SandboxBackend.__new__(SandboxBackend)
        backend.project_root = project_root
        backend.mesh_host_port = 8420
        backend.agents = {}
        backend.auth_tokens = {}
        backend.extra_env = {}
        backend._workspace_root = tmp_path / ".openlegion" / "agents"
        backend._workspace_root.mkdir(parents=True)

        ws = backend._prepare_workspace(
            agent_id="gamma",
            role="helper",
            skills_dir="",
            system_prompt="You help",
            model="openai/gpt-4o-mini",
        )

        env_content = (ws / ".agent.env").read_text()
        assert "BROWSER_BACKEND" not in env_content

    def test_prepare_workspace_extra_env(self, tmp_path):
        """extra_env dict values appear in the generated .agent.env file."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        backend = SandboxBackend.__new__(SandboxBackend)
        backend.project_root = project_root
        backend.mesh_host_port = 8420
        backend.agents = {}
        backend.auth_tokens = {}
        backend.extra_env = {"EMBEDDING_MODEL": "custom/embed-v2"}
        backend._workspace_root = tmp_path / ".openlegion" / "agents"
        backend._workspace_root.mkdir(parents=True)

        ws = backend._prepare_workspace(
            agent_id="delta",
            role="helper",
            skills_dir="",
            system_prompt="You help",
            model="openai/gpt-4o-mini",
        )

        env_content = (ws / ".agent.env").read_text()
        assert "EMBEDDING_MODEL=custom/embed-v2" in env_content

    def test_stop_agent_removes_from_registry(self):
        backend = SandboxBackend.__new__(SandboxBackend)
        backend.agents = {
            "alpha": {"sandbox_name": "openlegion_alpha", "url": "sandbox://x", "role": "test"},
        }
        with patch("subprocess.run"):
            backend.stop_agent("alpha")
        assert "alpha" not in backend.agents

    def test_stop_agent_nonexistent_is_safe(self):
        backend = SandboxBackend.__new__(SandboxBackend)
        backend.agents = {}
        backend.stop_agent("nonexistent")  # Should not raise

    def test_health_check_sandbox_inspect(self):
        backend = SandboxBackend.__new__(SandboxBackend)
        backend.agents = {
            "alpha": {"sandbox_name": "openlegion_alpha"},
        }
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"Status": "Running"})

        with patch("subprocess.run", return_value=mock_result):
            assert backend.health_check("alpha") is True

    def test_health_check_not_running(self):
        backend = SandboxBackend.__new__(SandboxBackend)
        backend.agents = {
            "alpha": {"sandbox_name": "openlegion_alpha"},
        }
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"Status": "Stopped"})

        with patch("subprocess.run", return_value=mock_result):
            assert backend.health_check("alpha") is False

    def test_health_check_missing_agent(self):
        backend = SandboxBackend.__new__(SandboxBackend)
        backend.agents = {}
        assert backend.health_check("alpha") is False

    def test_get_logs_via_exec(self):
        backend = SandboxBackend.__new__(SandboxBackend)
        backend.agents = {
            "alpha": {"sandbox_name": "openlegion_alpha"},
        }
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "line1\nline2\n"

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            logs = backend.get_logs("alpha", tail=20)

        assert logs == "line1\nline2\n"
        cmd = mock_run.call_args[0][0]
        assert "tail" in cmd
        assert "-20" in cmd


# ── sandbox_available detection ───────────────────────────────

class TestSandboxDetection:
    def test_available_when_command_succeeds(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result):
            assert sandbox_available() is True

    def test_not_available_when_command_fails(self):
        mock_result = MagicMock()
        mock_result.returncode = 1
        with patch("subprocess.run", return_value=mock_result):
            assert sandbox_available() is False

    def test_not_available_when_command_not_found(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert sandbox_available() is False


# ── select_backend ────────────────────────────────────────────

class TestSelectBackend:
    def test_default_is_docker(self):
        """Default (no flags) should always use DockerBackend."""
        with patch("src.host.runtime.DockerBackend") as MockDocker:
            mock_instance = MagicMock()
            MockDocker.return_value = mock_instance
            result = select_backend()
            assert result is mock_instance

    def test_sandbox_when_opted_in_and_available(self):
        with (
            patch("src.host.runtime.sandbox_available", return_value=True),
            patch("src.host.runtime.SandboxBackend") as MockSandbox,
        ):
            mock_instance = MagicMock()
            MockSandbox.return_value = mock_instance
            result = select_backend(use_sandbox=True)
            assert result is mock_instance

    def test_sandbox_requested_but_unavailable_falls_back(self):
        with (
            patch("src.host.runtime.sandbox_available", return_value=False),
            patch("src.host.runtime.DockerBackend") as MockDocker,
        ):
            mock_instance = MagicMock()
            MockDocker.return_value = mock_instance
            result = select_backend(use_sandbox=True)
            assert result is mock_instance


# ── _default_embedding_model ─────────────────────────────────

class TestDefaultEmbeddingModel:
    def test_openai_provider(self):
        assert _default_embedding_model("openai/gpt-4o-mini") == "text-embedding-3-small"

    def test_gpt_prefix(self):
        assert _default_embedding_model("gpt-4o") == "text-embedding-3-small"

    def test_gemini_provider(self):
        assert _default_embedding_model("gemini/gemini-2.0-flash") == "gemini/text-embedding-004"

    def test_anthropic_no_embeddings(self):
        assert _default_embedding_model("anthropic/claude-haiku-4-5-20251001") == "none"

    def test_unknown_provider_no_embeddings(self):
        assert _default_embedding_model("deepseek/deepseek-chat") == "none"
