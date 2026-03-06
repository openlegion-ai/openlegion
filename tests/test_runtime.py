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
    backend.project_root = __import__("pathlib").Path("/tmp")
    backend.browser_service_url = None
    backend.browser_vnc_url = None
    backend.browser_auth_token = ""
    backend._browser_container = None
    backend._network_name = "openlegion_agents"
    backend._network = MagicMock()
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

        # Mock httpx so API + VNC health checks pass
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("httpx.get", return_value=mock_resp):
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

    def test_browser_service_vnc_health_fail(self):
        """browser_vnc_url stays None when KasmVNC is unreachable."""
        import docker as _docker

        backend = self._make_backend()
        mock_container = MagicMock()
        mock_client = MagicMock()
        mock_client.containers.run.return_value = mock_container
        mock_client.containers.get.side_effect = _docker.errors.NotFound("not found")
        backend.client = mock_client

        call_count = 0

        def _fake_get(url, **kwargs):
            nonlocal call_count
            call_count += 1
            # API health check succeeds
            if "/browser/status" in url:
                resp = MagicMock()
                resp.status_code = 200
                return resp
            # VNC health check fails
            import httpx
            raise httpx.ConnectError("refused")

        with patch("httpx.get", side_effect=_fake_get), \
             patch("time.sleep"):
            backend.start_browser_service()

        assert backend.browser_service_url is not None
        assert backend.browser_vnc_url is None

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

    def test_agent_mesh_url_uses_host_docker_internal(self):
        """Agent MESH_URL points to host.docker.internal on bridge networking."""
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
        assert env["MESH_URL"] == "http://host.docker.internal:8420"

    def test_agent_mesh_url_localhost_on_host_network(self):
        """On host networking, MESH_URL uses 127.0.0.1."""
        import docker as _docker

        backend = _make_docker_backend(use_host_network=True)
        mock_container = MagicMock()
        mock_client = MagicMock()
        mock_client.containers.run.return_value = mock_container
        mock_client.containers.get.side_effect = _docker.errors.NotFound("not found")
        backend.client = mock_client

        backend.start_agent(agent_id="test-agent", role="test", skills_dir="")

        run_call = mock_client.containers.run.call_args
        env = run_call.kwargs.get("environment", {})
        assert env["MESH_URL"] == "http://127.0.0.1:8420"

    def test_agent_on_bridge_network(self):
        """Agent containers are placed on the agent bridge network."""
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


class TestStopAll:
    """Tests for stop_all cleanup."""

    def test_stop_all_cleans_network(self):
        """stop_all removes the agent network."""
        network = MagicMock()
        backend = _make_docker_backend(_network=network)
        backend.client = MagicMock()

        backend.stop_all()

        network.remove.assert_called_once()
        assert backend._network is None

    def test_stop_all_no_network(self):
        """stop_all handles missing network gracefully."""
        backend = _make_docker_backend(_network=None)
        backend.client = MagicMock()

        backend.stop_all()  # should not raise

        assert backend._network is None


class TestAgentNetwork:
    """Tests for _ensure_agent_network — plain bridge with stale internal replacement."""

    def _make_backend(self):
        backend = _make_docker_backend()
        backend.client = MagicMock()
        return backend

    def test_reuses_normal_bridge(self):
        """Existing non-internal bridge network is reused."""
        backend = self._make_backend()
        existing = MagicMock()
        existing.attrs = {"Internal": False}
        backend.client.networks.get.return_value = existing

        result = backend._ensure_agent_network()

        assert result is existing
        backend.client.networks.create.assert_not_called()

    def test_replaces_internal_network(self):
        """Old internal=True networks are replaced with a plain bridge."""
        backend = self._make_backend()
        old_internal = MagicMock()
        old_internal.attrs = {"Internal": True}
        backend.client.networks.get.return_value = old_internal

        new_net = MagicMock()
        backend.client.networks.create.return_value = new_net

        result = backend._ensure_agent_network()

        old_internal.remove.assert_called_once()
        backend.client.networks.create.assert_called_once_with(
            "openlegion_agents", driver="bridge",
        )
        assert result is new_net

    def test_creates_when_not_found(self):
        """Creates a new bridge network when none exists."""
        import docker as _docker

        backend = self._make_backend()
        backend.client.networks.get.side_effect = _docker.errors.NotFound("nope")

        new_net = MagicMock()
        backend.client.networks.create.return_value = new_net

        result = backend._ensure_agent_network()

        backend.client.networks.create.assert_called_once_with(
            "openlegion_agents", driver="bridge",
        )
        assert result is new_net

    def test_keeps_stale_if_remove_fails(self):
        """Falls back to stale internal network if removal fails."""
        backend = self._make_backend()
        old_internal = MagicMock()
        old_internal.attrs = {"Internal": True}
        old_internal.remove.side_effect = Exception("in use")
        backend.client.networks.get.return_value = old_internal

        result = backend._ensure_agent_network()

        assert result is old_internal
        backend.client.networks.create.assert_not_called()


class TestContainerHardening:
    """Tests for security hardening on agent containers."""

    def test_pids_limit(self):
        """Agent containers have a PID limit to prevent fork bombs."""
        import docker as _docker

        backend = _make_docker_backend()
        mock_client = MagicMock()
        mock_client.containers.run.return_value = MagicMock()
        mock_client.containers.get.side_effect = _docker.errors.NotFound("not found")
        backend.client = mock_client

        backend.start_agent(agent_id="test-agent", role="test", skills_dir="")

        run_call = mock_client.containers.run.call_args
        assert run_call.kwargs.get("pids_limit") == 256

    def test_no_unnecessary_capabilities(self):
        """Agent containers drop ALL caps and don't add any back."""
        import docker as _docker

        backend = _make_docker_backend()
        mock_client = MagicMock()
        mock_client.containers.run.return_value = MagicMock()
        mock_client.containers.get.side_effect = _docker.errors.NotFound("not found")
        backend.client = mock_client

        backend.start_agent(agent_id="test-agent", role="test", skills_dir="")

        run_call = mock_client.containers.run.call_args
        assert run_call.kwargs.get("cap_drop") == ["ALL"]
        assert "cap_add" not in run_call.kwargs

    def test_read_only_rootfs(self):
        """Agent containers have read-only root filesystem."""
        import docker as _docker

        backend = _make_docker_backend()
        mock_client = MagicMock()
        mock_client.containers.run.return_value = MagicMock()
        mock_client.containers.get.side_effect = _docker.errors.NotFound("not found")
        backend.client = mock_client

        backend.start_agent(agent_id="test-agent", role="test", skills_dir="")

        run_call = mock_client.containers.run.call_args
        assert run_call.kwargs.get("read_only") is True
        assert run_call.kwargs.get("security_opt") == ["no-new-privileges"]

