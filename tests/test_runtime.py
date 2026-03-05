"""Tests for RuntimeBackend (DockerBackend and SandboxBackend)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from src.cli.runtime import _default_embedding_model
from src.host.runtime import (
    DockerBackend,
    RuntimeBackend,
    SandboxBackend,
    _docker_safe_name,
    _should_use_host_network,
    sandbox_available,
    select_backend,
)

# ── RuntimeBackend interface ──────────────────────────────────

class TestRuntimeBackendInterface:
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
        # Create a project-specific PROJECT.md (not global)
        project_md = tmp_path / "project_context.md"
        project_md.write_text("# Test Project")

        skills_src = project_root / "skills" / "alpha"
        skills_src.mkdir(parents=True)
        (skills_src / "my_skill.py").write_text("# skill")

        backend = SandboxBackend.__new__(SandboxBackend)
        backend.project_root = project_root
        backend.mesh_host_port = 8420
        backend.agents = {}
        backend.auth_tokens = {}
        backend.extra_env = {"PROJECT_MD_PATH": str(project_md)}
        backend._workspace_root = tmp_path / ".openlegion" / "agents"
        backend._workspace_root.mkdir(parents=True)

        ws = backend._prepare_workspace(
            agent_id="alpha",
            role="test",
            skills_dir=str(skills_src),
            system_prompt="You are a test agent.",
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

    def test_prepare_workspace_standalone_no_project_md(self, tmp_path):
        """Standalone agents (no PROJECT_MD_PATH) get no PROJECT.md."""
        project_root = tmp_path / "project"
        project_root.mkdir()
        # Even if global PROJECT.md exists, standalone agents don't get it
        (project_root / "PROJECT.md").write_text("# Global")

        backend = SandboxBackend.__new__(SandboxBackend)
        backend.project_root = project_root
        backend.mesh_host_port = 8420
        backend.agents = {}
        backend.auth_tokens = {}
        backend.extra_env = {}
        backend._workspace_root = tmp_path / ".openlegion" / "agents"
        backend._workspace_root.mkdir(parents=True)

        ws = backend._prepare_workspace(
            agent_id="solo", role="test", skills_dir="",
            system_prompt="", model="",
        )
        assert not (ws / "PROJECT.md").exists()

    def test_prepare_workspace_env_no_browser_vars(self, tmp_path):
        """BROWSER_BACKEND is not written to the env config file."""
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
            system_prompt="",
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
            system_prompt="",
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

    def test_default_uses_bridge_networking(self):
        """Default select_backend() uses bridge networking (use_host_network=False)."""
        with (
            patch.dict("os.environ", {}, clear=False),
            patch("src.host.runtime.DockerBackend") as MockDocker,
        ):
            # Ensure OPENLEGION_HOST_NETWORK is not set
            import os
            os.environ.pop("OPENLEGION_HOST_NETWORK", None)
            mock_instance = MagicMock()
            MockDocker.return_value = mock_instance
            select_backend()
            MockDocker.assert_called_once()
            assert MockDocker.call_args.kwargs["use_host_network"] is False

    def test_host_network_env_override(self):
        """OPENLEGION_HOST_NETWORK=1 enables host networking."""
        with (
            patch.dict("os.environ", {"OPENLEGION_HOST_NETWORK": "1"}),
            patch("src.host.runtime.DockerBackend") as MockDocker,
        ):
            mock_instance = MagicMock()
            MockDocker.return_value = mock_instance
            select_backend()
            MockDocker.assert_called_once()
            assert MockDocker.call_args.kwargs["use_host_network"] is True


# ── _should_use_host_network ─────────────────────────────────

class TestShouldUseHostNetwork:
    def test_default_is_false(self):
        """Without env var, bridge networking is used."""
        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("OPENLEGION_HOST_NETWORK", None)
            assert _should_use_host_network() is False

    def test_enabled_with_1(self):
        with patch.dict("os.environ", {"OPENLEGION_HOST_NETWORK": "1"}):
            assert _should_use_host_network() is True

    def test_enabled_with_true(self):
        with patch.dict("os.environ", {"OPENLEGION_HOST_NETWORK": "true"}):
            assert _should_use_host_network() is True

    def test_disabled_with_0(self):
        with patch.dict("os.environ", {"OPENLEGION_HOST_NETWORK": "0"}):
            assert _should_use_host_network() is False

    def test_disabled_with_empty(self):
        with patch.dict("os.environ", {"OPENLEGION_HOST_NETWORK": ""}):
            assert _should_use_host_network() is False


# ── _default_embedding_model ─────────────────────────────────

class TestDefaultEmbeddingModel:
    def test_openai_provider(self):
        assert _default_embedding_model("openai/gpt-4o-mini") == "text-embedding-3-small"

    def test_gpt_prefix(self):
        assert _default_embedding_model("gpt-4o") == "text-embedding-3-small"

    def test_gemini_no_compatible_embeddings(self):
        """Gemini embedding models are 768-dim, incompatible with EMBEDDING_DIM=1536."""
        assert _default_embedding_model("gemini/gemini-2.0-flash") == "none"

    def test_anthropic_no_embeddings(self):
        assert _default_embedding_model("anthropic/claude-haiku-4-5-20251001") == "none"

    def test_unknown_provider_no_embeddings(self):
        assert _default_embedding_model("deepseek/deepseek-chat") == "none"


# ── _docker_safe_name ────────────────────────────────────────

class TestDockerSafeName:
    def test_spaces_replaced(self):
        assert _docker_safe_name("Signal Watcher") == "Signal_Watcher"

    def test_already_valid(self):
        assert _docker_safe_name("alpha") == "alpha"

    def test_dots_and_dashes_preserved(self):
        assert _docker_safe_name("my-agent.v2") == "my-agent.v2"

    def test_special_chars(self):
        assert _docker_safe_name("agent@home!") == "agent_home_"

    def test_unicode(self):
        result = _docker_safe_name("agente_espanol")
        assert result == "agente_espanol"


# ── DockerBackend slim agent resources & browser service ─────


def _make_docker_backend(**overrides):
    """Create a DockerBackend without calling __init__ (avoids Docker)."""
    import threading
    backend = DockerBackend.__new__(DockerBackend)
    backend.agents = {}
    backend.auth_tokens = {}
    backend.extra_env = {}
    backend.mesh_host_port = 8420
    backend.use_host_network = False
    backend._next_port = 8401
    backend._port_lock = threading.Lock()
    backend._relay_lock = threading.Lock()
    backend.project_root = __import__("pathlib").Path("/tmp")
    backend.browser_service_url = None
    backend.browser_vnc_url = None
    backend.browser_auth_token = ""
    backend._browser_container = None
    backend._network_name = "openlegion_agents"
    backend._network = MagicMock()
    # Pretend relay is already running (status check passes)
    relay_mock = MagicMock()
    relay_mock.status = "running"
    backend._mesh_relay = relay_mock
    for k, v in overrides.items():
        setattr(backend, k, v)
    return backend


class TestDockerBackendSlimResources:
    def _make_backend(self):
        return _make_docker_backend()

    def test_slim_agent_resources(self):
        """Agent containers use slim resources (384m, 0.15 CPU, no shm)."""
        import docker as _docker

        backend = self._make_backend()
        mock_container = MagicMock()
        mock_client = MagicMock()
        mock_client.containers.run.return_value = mock_container
        mock_client.containers.get.side_effect = _docker.errors.NotFound("not found")
        backend.client = mock_client

        backend.start_agent(
            agent_id="test-agent",
            role="test",
            skills_dir="",
        )

        run_call = mock_client.containers.run.call_args
        assert run_call.kwargs.get("mem_limit") == "384m"
        assert run_call.kwargs.get("cpu_quota") == 15000
        assert "shm_size" not in run_call.kwargs

    def test_agent_gets_single_port(self):
        """Agents get only one port (API), no VNC port."""
        import docker as _docker

        backend = self._make_backend()
        mock_container = MagicMock()
        mock_client = MagicMock()
        mock_client.containers.run.return_value = mock_container
        mock_client.containers.get.side_effect = _docker.errors.NotFound("not found")
        backend.client = mock_client

        backend.start_agent(agent_id="test-agent", role="test", skills_dir="")

        agent_info = backend.agents["test-agent"]
        assert "vnc_port" not in agent_info
        assert "vnc_url" not in agent_info

        run_call = mock_client.containers.run.call_args
        ports = run_call.kwargs.get("ports", {})
        assert "8400/tcp" in ports
        assert "6080/tcp" not in ports

    def test_browser_service_lifecycle(self):
        """start_browser_service creates container, stop removes it."""
        import docker as _docker

        backend = self._make_backend()
        mock_container = MagicMock()
        mock_client = MagicMock()
        mock_client.containers.run.return_value = mock_container
        mock_client.containers.get.side_effect = _docker.errors.NotFound("not found")
        backend.client = mock_client

        backend.start_browser_service()

        assert backend.browser_service_url is not None
        assert backend.browser_vnc_url is not None
        assert backend.browser_auth_token != ""
        assert backend._browser_container is mock_container

        # Verify browser service resource limits
        run_call = mock_client.containers.run.call_args
        assert run_call.kwargs.get("mem_limit") == "2g"
        assert run_call.kwargs.get("cpu_quota") == 100000
        assert run_call.kwargs.get("shm_size") == "512m"

        backend.stop_browser_service()
        mock_container.stop.assert_called_once()
        mock_container.remove.assert_called_once()
        assert backend.browser_service_url is None

    def test_containers_no_docker_init(self):
        """Docker init=True must NOT be set — Dockerfile ENTRYPOINT tini handles it.

        Using both causes tini to run as a non-PID-1 child of docker-init,
        where it cannot reap zombies.
        """
        import docker as _docker

        backend = self._make_backend()
        mock_container = MagicMock()
        mock_client = MagicMock()
        mock_client.containers.run.return_value = mock_container
        mock_client.containers.get.side_effect = _docker.errors.NotFound("not found")
        backend.client = mock_client

        backend.start_agent(
            agent_id="test-init",
            role="test",
            skills_dir="",
        )

        run_call = mock_client.containers.run.call_args
        assert "init" not in run_call.kwargs

    def test_agent_mesh_url_uses_relay(self):
        """Agent MESH_URL points to the relay container, not host.docker.internal."""
        import docker as _docker

        backend = self._make_backend()
        mock_container = MagicMock()
        mock_client = MagicMock()
        mock_client.containers.run.return_value = mock_container
        mock_client.containers.get.side_effect = _docker.errors.NotFound("not found")
        backend.client = mock_client

        backend.start_agent(agent_id="test-agent", role="test", skills_dir="")

        run_call = mock_client.containers.run.call_args
        env = run_call.kwargs.get("environment", {})
        assert env["MESH_URL"] == f"http://{DockerBackend.MESH_RELAY_NAME}:8420"
        assert "host.docker.internal" not in env["MESH_URL"]

    def test_agent_mesh_url_localhost_on_host_network(self):
        """On host networking, MESH_URL uses 127.0.0.1 (no relay)."""
        import docker as _docker

        backend = _make_docker_backend(use_host_network=True, _mesh_relay=None)
        mock_container = MagicMock()
        mock_client = MagicMock()
        mock_client.containers.run.return_value = mock_container
        mock_client.containers.get.side_effect = _docker.errors.NotFound("not found")
        backend.client = mock_client

        backend.start_agent(agent_id="test-agent", role="test", skills_dir="")

        run_call = mock_client.containers.run.call_args
        env = run_call.kwargs.get("environment", {})
        assert env["MESH_URL"] == "http://127.0.0.1:8420"

    def test_agent_on_internal_network(self):
        """Agent containers are placed on the internal agent network."""
        import docker as _docker

        backend = self._make_backend()
        mock_container = MagicMock()
        mock_client = MagicMock()
        mock_client.containers.run.return_value = mock_container
        mock_client.containers.get.side_effect = _docker.errors.NotFound("not found")
        backend.client = mock_client

        backend.start_agent(agent_id="test-agent", role="test", skills_dir="")

        run_call = mock_client.containers.run.call_args
        assert run_call.kwargs.get("network") == "openlegion_agents"


class TestMeshRelay:
    """Tests for the mesh relay container that bridges internal network to mesh."""

    def test_ensure_relay_starts_container(self):
        """_ensure_mesh_relay creates a relay on first call."""
        import docker as _docker

        backend = _make_docker_backend(_mesh_relay=None)
        mock_relay = MagicMock()
        mock_relay.status = "running"
        mock_client = MagicMock()
        mock_client.containers.run.return_value = mock_relay
        mock_client.containers.get.side_effect = _docker.errors.NotFound("not found")
        backend.client = mock_client

        backend._ensure_mesh_relay()

        assert backend._mesh_relay is mock_relay
        mock_client.containers.run.assert_called_once()
        run_call = mock_client.containers.run.call_args
        assert run_call.kwargs["name"] == DockerBackend.MESH_RELAY_NAME
        # Relay connected to internal network
        backend._network.connect.assert_called_once_with(mock_relay)

    def test_ensure_relay_idempotent(self):
        """_ensure_mesh_relay skips if relay is already running."""
        existing_relay = MagicMock()
        existing_relay.status = "running"
        backend = _make_docker_backend(_mesh_relay=existing_relay)
        mock_client = MagicMock()
        backend.client = mock_client

        backend._ensure_mesh_relay()

        mock_client.containers.run.assert_not_called()
        assert backend._mesh_relay is existing_relay

    def test_ensure_relay_skips_host_network(self):
        """No relay needed when using host networking."""
        backend = _make_docker_backend(use_host_network=True, _mesh_relay=None)
        mock_client = MagicMock()
        backend.client = mock_client

        backend._ensure_mesh_relay()

        mock_client.containers.run.assert_not_called()
        assert backend._mesh_relay is None

    def test_ensure_relay_restarts_crashed(self):
        """_ensure_mesh_relay restarts a crashed relay container."""
        import docker as _docker

        dead_relay = MagicMock()
        dead_relay.status = "exited"
        backend = _make_docker_backend(_mesh_relay=dead_relay)

        new_relay = MagicMock()
        new_relay.status = "running"
        mock_client = MagicMock()
        mock_client.containers.run.return_value = new_relay
        mock_client.containers.get.side_effect = _docker.errors.NotFound("not found")
        backend.client = mock_client

        backend._ensure_mesh_relay()

        dead_relay.remove.assert_called_once_with(force=True)
        assert backend._mesh_relay is new_relay

    def test_relay_hardening(self):
        """Relay container has security hardening and auto-restart."""
        import docker as _docker

        backend = _make_docker_backend(_mesh_relay=None)
        mock_relay = MagicMock()
        mock_relay.status = "running"
        mock_client = MagicMock()
        mock_client.containers.run.return_value = mock_relay
        mock_client.containers.get.side_effect = _docker.errors.NotFound("not found")
        backend.client = mock_client

        backend._ensure_mesh_relay()

        run_call = mock_client.containers.run.call_args
        assert run_call.kwargs["mem_limit"] == "128m"
        assert run_call.kwargs["read_only"] is True
        assert run_call.kwargs["cap_drop"] == ["ALL"]
        assert run_call.kwargs["security_opt"] == ["no-new-privileges"]
        assert run_call.kwargs["restart_policy"] == {"Name": "unless-stopped"}
        # No unnecessary capabilities
        assert "cap_add" not in run_call.kwargs

    def test_relay_cleans_stale_on_start(self):
        """_start_mesh_relay removes a stale relay container before starting."""
        backend = _make_docker_backend(_mesh_relay=None)
        stale = MagicMock()
        new_relay = MagicMock()
        new_relay.status = "running"
        mock_client = MagicMock()
        mock_client.containers.get.return_value = stale  # stale exists
        mock_client.containers.run.return_value = new_relay
        backend.client = mock_client

        backend._ensure_mesh_relay()

        stale.remove.assert_called_once_with(force=True)

    def test_relay_network_connect_failure_cleans_up(self):
        """If connecting relay to internal network fails, relay is removed."""
        import docker as _docker

        backend = _make_docker_backend(_mesh_relay=None)
        relay = MagicMock()
        relay.status = "running"
        mock_client = MagicMock()
        mock_client.containers.run.return_value = relay
        mock_client.containers.get.side_effect = _docker.errors.NotFound("not found")
        backend.client = mock_client
        backend._network = MagicMock()
        backend._network.connect.side_effect = _docker.errors.APIError("network error")

        backend._ensure_mesh_relay()

        # Relay should be cleaned up, not left as orphan
        relay.stop.assert_called_once()
        relay.remove.assert_called_once()
        assert backend._mesh_relay is None

    def test_relay_startup_failure_cleans_up(self):
        """If relay container exits immediately, it is cleaned up."""
        import docker as _docker

        backend = _make_docker_backend(_mesh_relay=None)
        relay = MagicMock()
        relay.status = "exited"  # never reaches "running"
        relay.logs.return_value = b"ImportError: no module named asyncio"
        mock_client = MagicMock()
        mock_client.containers.run.return_value = relay
        mock_client.containers.get.side_effect = _docker.errors.NotFound("not found")
        backend.client = mock_client

        backend._ensure_mesh_relay()

        relay.remove.assert_called_once_with(force=True)
        assert backend._mesh_relay is None

    def test_relay_reload_exception_triggers_restart(self):
        """If relay.reload() raises, relay is recreated."""
        import docker as _docker

        dead_relay = MagicMock()
        dead_relay.reload.side_effect = _docker.errors.NotFound("removed externally")
        backend = _make_docker_backend(_mesh_relay=dead_relay)

        new_relay = MagicMock()
        new_relay.status = "running"
        mock_client = MagicMock()
        mock_client.containers.run.return_value = new_relay
        mock_client.containers.get.side_effect = _docker.errors.NotFound("not found")
        backend.client = mock_client

        backend._ensure_mesh_relay()

        assert backend._mesh_relay is new_relay


class TestStopAll:
    """Tests for stop_all cleanup of relay and network."""

    def test_stop_all_cleans_relay(self):
        """stop_all stops and removes the mesh relay."""
        relay = MagicMock()
        backend = _make_docker_backend(_mesh_relay=relay)
        backend.client = MagicMock()

        backend.stop_all()

        relay.stop.assert_called_once_with(timeout=5)
        relay.remove.assert_called_once()
        assert backend._mesh_relay is None

    def test_stop_all_cleans_network(self):
        """stop_all removes the agent network."""
        network = MagicMock()
        backend = _make_docker_backend(_network=network)
        backend.client = MagicMock()

        backend.stop_all()

        network.remove.assert_called_once()
        assert backend._network is None

    def test_stop_all_relay_before_network(self):
        """Relay is stopped before network is removed (relay is connected)."""
        call_order = []
        relay = MagicMock()
        relay.stop.side_effect = lambda **kw: call_order.append("relay_stop")
        relay.remove.side_effect = lambda: call_order.append("relay_remove")
        network = MagicMock()
        network.remove.side_effect = lambda: call_order.append("network_remove")

        backend = _make_docker_backend(_mesh_relay=relay, _network=network)
        backend.client = MagicMock()

        backend.stop_all()

        assert call_order == ["relay_stop", "relay_remove", "network_remove"]

    def test_stop_all_no_relay(self):
        """stop_all handles missing relay gracefully."""
        backend = _make_docker_backend(_mesh_relay=None)
        backend.client = MagicMock()

        backend.stop_all()  # should not raise


class TestNetworkInternalFlag:
    """Tests for _ensure_internal_network validation."""

    def test_reuses_internal_network(self):
        """Existing internal network is reused without recreation."""
        existing = MagicMock()
        existing.attrs = {"Internal": True}

        backend = _make_docker_backend()
        backend.client = MagicMock()
        backend.client.networks.get.return_value = existing

        result = backend._ensure_internal_network()

        assert result is existing
        backend.client.networks.create.assert_not_called()

    def test_replaces_non_internal_network(self):
        """Non-internal network is removed and recreated with internal=True."""
        non_internal = MagicMock()
        non_internal.attrs = {"Internal": False}

        new_network = MagicMock()

        backend = _make_docker_backend()
        backend.client = MagicMock()
        backend.client.networks.get.return_value = non_internal
        backend.client.networks.create.return_value = new_network

        result = backend._ensure_internal_network()

        non_internal.remove.assert_called_once()
        backend.client.networks.create.assert_called_once_with(
            backend._network_name, driver="bridge", internal=True,
        )
        assert result is new_network

    def test_creates_when_not_found(self):
        """Creates a new internal network when none exists."""
        import docker

        new_network = MagicMock()

        backend = _make_docker_backend()
        backend.client = MagicMock()
        backend.client.networks.get.side_effect = docker.errors.NotFound("not found")
        backend.client.networks.create.return_value = new_network

        result = backend._ensure_internal_network()

        backend.client.networks.create.assert_called_once_with(
            backend._network_name, driver="bridge", internal=True,
        )
        assert result is new_network

    def test_keeps_non_internal_if_remove_fails(self):
        """Falls back to non-internal network if removal fails."""
        import docker

        non_internal = MagicMock()
        non_internal.attrs = {"Internal": False}
        non_internal.remove.side_effect = docker.errors.APIError("has active endpoints")

        backend = _make_docker_backend()
        backend.client = MagicMock()
        backend.client.networks.get.return_value = non_internal

        result = backend._ensure_internal_network()

        assert result is non_internal
        backend.client.networks.create.assert_not_called()

