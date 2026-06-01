"""Tests for RuntimeBackend (DockerBackend and SandboxBackend)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

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


# ── Agent lifecycle: delete vs archive token/volume semantics ──
#
# Security findings H11 (token revocation on delete), H12 (data volume
# wipe on delete) and M16 (token leak on failed mid-creation).

class TestDockerBackendLifecycle:
    def _make_backend(self) -> DockerBackend:
        backend = DockerBackend.__new__(DockerBackend)
        backend.agents = {}
        backend.auth_tokens = {}
        backend.client = MagicMock()
        return backend

    def test_delete_pops_token_and_removes_volume(self):
        """H11+H12: stop_agent(remove_data=True) revokes token + wipes volume."""
        backend = self._make_backend()
        backend.agents["alpha"] = {"container": MagicMock()}
        backend.auth_tokens["alpha"] = "secret-token"
        vol = MagicMock()
        backend.client.volumes.get.return_value = vol

        backend.stop_agent("alpha", remove_data=True)

        # H11: token revoked.
        assert "alpha" not in backend.auth_tokens
        # H12: the agent's private named volume was removed.
        backend.client.volumes.get.assert_called_once_with(
            f"openlegion_data_{_docker_safe_name('alpha')}",
        )
        vol.remove.assert_called_once_with(force=True)
        assert "alpha" not in backend.agents

    def test_delete_wipes_volume_for_deregistered_agent(self):
        """H12: archive→delete is the only delete path, and archive already
        deregistered the agent, so deleting a NOT-in-registry agent must STILL
        wipe its /data volume (the wipe is independent of live registration)."""
        backend = self._make_backend()
        # "ghost" is NOT in backend.agents — simulates an already-archived /
        # deregistered agent being deleted.
        assert "ghost" not in backend.agents
        vol = MagicMock()
        backend.client.volumes.get.return_value = vol

        backend.stop_agent("ghost", remove_data=True)

        # The volume is wiped by name even though the agent was deregistered.
        backend.client.volumes.get.assert_called_once_with(
            f"openlegion_data_{_docker_safe_name('ghost')}",
        )
        vol.remove.assert_called_once_with(force=True)

    def test_archive_keeps_volume_but_pops_token(self):
        """Archive uses remove_data=False (default): volume RETAINED for unarchive."""
        backend = self._make_backend()
        backend.agents["beta"] = {"container": MagicMock()}
        backend.auth_tokens["beta"] = "secret-token"

        # Default remove_data=False — the archive contract.
        backend.stop_agent("beta")

        # Token still gets popped (archive already revoked it; that's fine).
        assert "beta" not in backend.auth_tokens
        # CRITICAL: volume must NOT be touched on archive.
        backend.client.volumes.get.assert_not_called()
        assert "beta" not in backend.agents

    def test_start_agent_failure_pops_token(self):
        """M16: a failed containers.run must not leak a registered auth token."""
        backend = self._make_backend()
        backend.auth_tokens = {}
        backend.agents = {}
        backend._port_lock = __import__("threading").Lock()
        backend._next_port = 8400
        backend.use_host_network = True
        backend.mesh_host_port = 8420
        backend.extra_env = {}
        backend.uploads_dir = __import__("pathlib").Path("/tmp/uploads")
        backend.project_root = __import__("pathlib").Path("/tmp/proj")
        backend.BASE_IMAGE = "test-image"
        backend._network_name = "openlegion_agents"
        # containers.run raises mid-creation, after the token is registered.
        backend.client.containers.get.side_effect = __import__(
            "docker",
        ).errors.NotFound("no stale")
        backend.client.containers.run.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            backend.start_agent("gamma", "worker", "")

        # Token must have been popped on the failure path.
        assert "gamma" not in backend.auth_tokens
        assert "gamma" not in backend.agents


# ── SandboxBackend ────────────────────────────────────────────

class TestSandboxBackend:
    def test_backend_name(self):
        assert SandboxBackend.backend_name() == "sandbox"

    def test_prepare_workspace(self, tmp_path):
        """Workspace directory is created with expected structure."""
        project_root = tmp_path / "project"
        project_root.mkdir()
        # Create a team-specific context file (not global). Test the
        # legacy ``PROJECT_MD_PATH`` env name still works as an alias for
        # ``TEAM_MD_PATH`` so existing deploys don't break.
        team_md = tmp_path / "team_context.md"
        team_md.write_text("# Test Team")

        tools_src = project_root / "agent_tools" / "alpha"
        tools_src.mkdir(parents=True)
        (tools_src / "my_tool.py").write_text("# tool")

        backend = SandboxBackend.__new__(SandboxBackend)
        backend.project_root = project_root
        backend.mesh_host_port = 8420
        backend.agents = {}
        backend.auth_tokens = {}
        backend.extra_env = {"PROJECT_MD_PATH": str(team_md)}
        backend._workspace_root = tmp_path / ".openlegion" / "agents"
        backend._workspace_root.mkdir(parents=True)

        ws = backend._prepare_workspace(
            agent_id="alpha",
            role="test",
            tools_dir=str(tools_src),
            system_prompt="You are a test agent.",
            model="openai/gpt-4o-mini",
        )

        assert ws.exists()
        assert (ws / "data" / "workspace").is_dir()
        assert (ws / "TEAM.md").read_text() == "# Test Team"
        # PR 3 dropped the legacy PROJECT.md write — only TEAM.md ships.
        assert not (ws / "PROJECT.md").exists()
        assert (ws / "tools" / "my_tool.py").read_text() == "# tool"
        assert (ws / ".agent.env").exists()

        env_content = (ws / ".agent.env").read_text()
        assert "AGENT_ID=alpha" in env_content
        assert "AGENT_ROLE=test" in env_content
        assert "LLM_MODEL=openai/gpt-4o-mini" in env_content
        assert "MESH_AUTH_TOKEN=" in env_content
        assert "alpha" in backend.auth_tokens

    def test_prepare_workspace_solo_no_team_md(self, tmp_path):
        """Solo agents (no TEAM_MD_PATH) get no TEAM.md."""
        project_root = tmp_path / "project"
        project_root.mkdir()
        # Even if a stray TEAM.md exists at project root, solo agents
        # don't get it (the path is selected via env var, not by
        # walking the project tree).
        (project_root / "TEAM.md").write_text("# Global")

        backend = SandboxBackend.__new__(SandboxBackend)
        backend.project_root = project_root
        backend.mesh_host_port = 8420
        backend.agents = {}
        backend.auth_tokens = {}
        backend.extra_env = {}
        backend._workspace_root = tmp_path / ".openlegion" / "agents"
        backend._workspace_root.mkdir(parents=True)

        ws = backend._prepare_workspace(
            agent_id="solo", role="test", tools_dir="",
            system_prompt="", model="",
        )
        assert not (ws / "TEAM.md").exists()
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
            tools_dir="",
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
            tools_dir="",
            system_prompt="",
            model="openai/gpt-4o-mini",
        )

        env_content = (ws / ".agent.env").read_text()
        assert "EMBEDDING_MODEL=custom/embed-v2" in env_content

    def test_prepare_workspace_newline_in_value(self, tmp_path):
        """Newlines in env values are escaped to prevent env-file injection."""
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
            agent_id="nl-test",
            role="test",
            tools_dir="",
            system_prompt="line1\nline2\r\nline3",
            model="openai/gpt-4o-mini",
        )

        env_content = (ws / ".agent.env").read_text()
        # Newlines must be escaped — not raw — to prevent env-file injection
        assert "INITIAL_INSTRUCTIONS=line1\\nline2\\nline3" in env_content
        # Each real line in the file should be a single KEY=VALUE entry
        for line in env_content.strip().split("\n"):
            assert "=" in line, f"Malformed env line: {line}"

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
    backend.uploads_dir = __import__("pathlib").Path("/tmp/openlegion_uploads")
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
            tools_dir="",
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

        backend.start_agent(agent_id="test-agent", role="test", tools_dir="")

        agent_info = backend.agents["test-agent"]
        assert "vnc_port" not in agent_info
        assert "vnc_url" not in agent_info

        run_call = mock_client.containers.run.call_args
        ports = run_call.kwargs.get("ports", {})
        assert "8400/tcp" in ports
        assert "6080/tcp" not in ports

    def test_agent_port_binds_loopback_only(self):
        """Agent's published :8400 binds to 127.0.0.1 only — the mesh reaches it
        via loopback, so it must not be exposed on all host interfaces."""
        import docker as _docker

        backend = self._make_backend()
        mock_client = MagicMock()
        mock_client.containers.run.return_value = MagicMock()
        mock_client.containers.get.side_effect = _docker.errors.NotFound("not found")
        backend.client = mock_client

        backend.start_agent(agent_id="test-agent", role="test", tools_dir="")

        run_call = mock_client.containers.run.call_args
        ports = run_call.kwargs.get("ports", {})
        host_binding = ports["8400/tcp"]
        assert isinstance(host_binding, tuple)
        assert host_binding[0] == "127.0.0.1"

    def test_browser_port_binds_loopback_only(self):
        """Browser service's published :8500 binds to 127.0.0.1 only."""
        import docker as _docker

        backend = self._make_backend()
        mock_client = MagicMock()
        mock_client.containers.run.return_value = MagicMock()
        mock_client.containers.get.side_effect = _docker.errors.NotFound("not found")
        backend.client = mock_client

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("httpx.get", return_value=mock_resp):
            backend.start_browser_service()

        run_call = mock_client.containers.run.call_args
        ports = run_call.kwargs.get("ports", {})
        host_binding = ports["8500/tcp"]
        assert isinstance(host_binding, tuple)
        assert host_binding[0] == "127.0.0.1"

    def test_browser_service_lifecycle(self):
        """start_browser_service creates container, stop removes it."""
        import docker as _docker

        backend = self._make_backend()
        mock_container = MagicMock()
        mock_client = MagicMock()
        mock_client.containers.run.return_value = mock_container
        mock_client.containers.get.side_effect = _docker.errors.NotFound("not found")
        backend.client = mock_client

        # Mock httpx so the API health check passes (no global VNC to
        # health-check anymore — per-agent KasmVNCs spawn lazily).
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("httpx.get", return_value=mock_resp):
            backend.start_browser_service()

        assert backend.browser_service_url is not None
        assert backend.browser_auth_token != ""
        assert backend._browser_container is mock_container

        # Verify browser service resource limits
        run_call = mock_client.containers.run.call_args
        assert run_call.kwargs.get("mem_limit") == "2g"
        assert run_call.kwargs.get("cpu_quota") == 100000  # Basic plan: 1 core
        assert run_call.kwargs.get("shm_size") == "512m"  # Basic plan: 512m

        backend.stop_browser_service()
        mock_container.stop.assert_called_once()
        mock_container.remove.assert_called_once()
        assert backend.browser_service_url is None

    @pytest.mark.parametrize(
        "max_agents,exp_mem,exp_shm,exp_cpu,exp_max_browsers",
        [
            # Basic — cax11 (default when env unset == 0)
            (0, "2g", "512m", 100000, 1),
            (1, "2g", "512m", 100000, 1),
            # Growth — cax21 (max_browsers == max_agents in this band)
            (2, "4g", "1g", 150000, 2),
            (5, "4g", "1g", 150000, 5),
            # Pro — cax31 (mem-bound, capped at 10 browsers even with 15 agents)
            (6, "8g", "2g", 200000, 6),
            (10, "8g", "2g", 200000, 10),
            (15, "8g", "2g", 200000, 10),
            # Pro Max — cax41 (32GB ARM, 30 agents / 30 browsers / 16GB / 4.0 CPU)
            (16, "16g", "4g", 400000, 16),
            (30, "16g", "4g", 400000, 30),
            (50, "16g", "4g", 400000, 30),  # capped at 30
        ],
    )
    def test_browser_tier_sizing(self, max_agents, exp_mem, exp_shm, exp_cpu, exp_max_browsers):
        """Browser container resources scale across the 4-tier plan table
        (Basic / Growth / Pro / Pro Max) driven by OPENLEGION_MAX_AGENTS."""
        import os as _os

        import docker as _docker

        backend = self._make_backend()
        mock_client = MagicMock()
        mock_client.containers.run.return_value = MagicMock()
        mock_client.containers.get.side_effect = _docker.errors.NotFound("nope")
        backend.client = mock_client

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("httpx.get", return_value=mock_resp), \
             patch.dict(_os.environ, {"OPENLEGION_MAX_AGENTS": str(max_agents)}):
            backend.start_browser_service()

        run_call = mock_client.containers.run.call_args
        assert run_call.kwargs.get("mem_limit") == exp_mem
        assert run_call.kwargs.get("shm_size") == exp_shm
        assert run_call.kwargs.get("cpu_quota") == exp_cpu
        env = run_call.kwargs.get("environment", {})
        assert env.get("OPENLEGION_BROWSER_MAX_CONCURRENT") == str(exp_max_browsers)

    def test_browser_max_concurrent_env_forwarded(self):
        """OPENLEGION_BROWSER_MAX_CONCURRENT (provisioner-set per-instance scale
        cap) must be forwarded into the browser container — otherwise the
        provisioner's scale_instance write to /opt/openlegion/.env is a no-op."""
        import os as _os

        import docker as _docker

        backend = self._make_backend()
        mock_client = MagicMock()
        mock_client.containers.run.return_value = MagicMock()
        mock_client.containers.get.side_effect = _docker.errors.NotFound("nope")
        backend.client = mock_client

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("httpx.get", return_value=mock_resp), \
             patch.dict(_os.environ, {"OPENLEGION_BROWSER_MAX_CONCURRENT": "7"}):
            backend.start_browser_service()

        run_call = mock_client.containers.run.call_args
        env = run_call.kwargs.get("environment", {})
        assert env.get("OPENLEGION_BROWSER_MAX_CONCURRENT") == "7"

    def test_browser_max_concurrent_env_set_from_tier_when_host_env_unset(self):
        """When OPENLEGION_BROWSER_MAX_CONCURRENT is not in the host env, the
        runtime still seeds the container's env with the tier-derived
        max_browsers value (the legacy MAX_BROWSERS alias was retired in
        the Phase 1 back-compat cleanup, so the canonical name carries
        the value in either path)."""
        import os as _os

        import docker as _docker

        backend = self._make_backend()
        mock_client = MagicMock()
        mock_client.containers.run.return_value = MagicMock()
        mock_client.containers.get.side_effect = _docker.errors.NotFound("nope")
        backend.client = mock_client

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("httpx.get", return_value=mock_resp), \
             patch.dict(_os.environ, {}, clear=False):
            _os.environ.pop("OPENLEGION_BROWSER_MAX_CONCURRENT", None)
            backend.start_browser_service()

        run_call = mock_client.containers.run.call_args
        env = run_call.kwargs.get("environment", {})
        # Canonical env var is always set to the tier value, even if the
        # host's own env doesn't define it (the value comes from the
        # OPENLEGION_MAX_AGENTS-driven tier table).
        assert "OPENLEGION_BROWSER_MAX_CONCURRENT" in env
        assert env["OPENLEGION_BROWSER_MAX_CONCURRENT"].isdigit()

    def test_browser_has_net_admin_in_bridge_mode(self):
        """Browser container gets the minimal cap set its entrypoint needs in bridge mode."""
        import docker as _docker

        backend = self._make_backend()
        mock_client = MagicMock()
        mock_client.containers.run.return_value = MagicMock()
        mock_client.containers.get.side_effect = _docker.errors.NotFound("nope")
        backend.client = mock_client

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("httpx.get", return_value=mock_resp):
            backend.start_browser_service()

        run_call = mock_client.containers.run.call_args
        # NET_ADMIN for iptables-restore; SETUID + SETGID so gosu can drop to UID 1000.
        assert run_call.kwargs.get("cap_add") == ["NET_ADMIN", "SETUID", "SETGID"]
        env = run_call.kwargs.get("environment", {})
        # Filter must be active (not disabled) in the default bridge mode
        assert "BROWSER_EGRESS_DISABLE" not in env

    def test_browser_egress_filter_disabled_in_host_network(self):
        """Host network mode shares host netns — iptables would mutate the host, so disable."""
        import os as _os

        import docker as _docker

        backend = _make_docker_backend(use_host_network=True)
        mock_client = MagicMock()
        mock_client.containers.run.return_value = MagicMock()
        mock_client.containers.get.side_effect = _docker.errors.NotFound("nope")
        backend.client = mock_client

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("httpx.get", return_value=mock_resp), \
             patch.dict(_os.environ, {"OPENLEGION_BROWSER_ALLOW_HOST_NETWORK": "1"}):
            backend.start_browser_service()

        run_call = mock_client.containers.run.call_args
        # No NET_ADMIN in host network mode
        assert "cap_add" not in run_call.kwargs
        # Explicit opt-out signal so the entrypoint skips iptables entirely
        env = run_call.kwargs.get("environment", {})
        assert env.get("BROWSER_EGRESS_DISABLE") == "1"

    def test_browser_egress_allowlist_forwarded(self):
        """Operator-supplied BROWSER_EGRESS_ALLOWLIST is forwarded to the container."""
        import os as _os

        import docker as _docker

        backend = self._make_backend()
        mock_client = MagicMock()
        mock_client.containers.run.return_value = MagicMock()
        mock_client.containers.get.side_effect = _docker.errors.NotFound("nope")
        backend.client = mock_client

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("httpx.get", return_value=mock_resp), \
             patch.dict(_os.environ, {"BROWSER_EGRESS_ALLOWLIST": "10.0.0.0/24,192.168.5.5/32"}):
            backend.start_browser_service()

        run_call = mock_client.containers.run.call_args
        env = run_call.kwargs.get("environment", {})
        assert env.get("BROWSER_EGRESS_ALLOWLIST") == "10.0.0.0/24,192.168.5.5/32"

    def test_browser_egress_disable_env_forwarded(self):
        """BROWSER_EGRESS_DISABLE from host env is forwarded to the container."""
        import os as _os

        import docker as _docker

        backend = self._make_backend()
        mock_client = MagicMock()
        mock_client.containers.run.return_value = MagicMock()
        mock_client.containers.get.side_effect = _docker.errors.NotFound("nope")
        backend.client = mock_client

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("httpx.get", return_value=mock_resp), \
             patch.dict(_os.environ, {"BROWSER_EGRESS_DISABLE": "1"}):
            backend.start_browser_service()

        run_call = mock_client.containers.run.call_args
        env = run_call.kwargs.get("environment", {})
        assert env.get("BROWSER_EGRESS_DISABLE") == "1"
        # NET_ADMIN (+ SETUID/SETGID for gosu) is still added in bridge mode —
        # operator opt-out is an entrypoint-level signal, not a cap-level signal.
        assert run_call.kwargs.get("cap_add") == ["NET_ADMIN", "SETUID", "SETGID"]

    def test_browser_host_network_emits_warning(self, caplog):
        """Host network mode logs a loud warning about disabled SSRF filter."""
        import logging
        import os as _os

        import docker as _docker

        backend = _make_docker_backend(use_host_network=True)
        mock_client = MagicMock()
        mock_client.containers.run.return_value = MagicMock()
        mock_client.containers.get.side_effect = _docker.errors.NotFound("nope")
        backend.client = mock_client

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("httpx.get", return_value=mock_resp), \
             patch.dict(_os.environ, {"OPENLEGION_BROWSER_ALLOW_HOST_NETWORK": "1"}), \
             caplog.at_level(logging.WARNING, logger="host.runtime"):
            backend.start_browser_service()

        # At least one WARNING record mentioning "egress filter" should be present
        matches = [r for r in caplog.records
                   if r.levelno >= logging.WARNING and "egress filter" in r.message.lower()]
        assert matches, f"Expected warning about egress filter in host mode, got: {[r.message for r in caplog.records]}"

    def test_browser_cap_drop_all_with_minimal_adds(self):
        """Browser container drops Docker default caps and adds back only the minimum."""
        import docker as _docker

        backend = self._make_backend()
        mock_client = MagicMock()
        mock_client.containers.run.return_value = MagicMock()
        mock_client.containers.get.side_effect = _docker.errors.NotFound("nope")
        backend.client = mock_client

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("httpx.get", return_value=mock_resp):
            backend.start_browser_service()

        run_call = mock_client.containers.run.call_args
        assert run_call.kwargs.get("cap_drop") == ["ALL"]
        assert run_call.kwargs.get("cap_add") == ["NET_ADMIN", "SETUID", "SETGID"]

    def test_browser_host_network_hard_refuses_without_ack(self):
        """Host network mode must raise unless OPENLEGION_BROWSER_ALLOW_HOST_NETWORK is set."""
        import os as _os

        import pytest

        import docker as _docker

        backend = _make_docker_backend(use_host_network=True)
        mock_client = MagicMock()
        mock_client.containers.run.return_value = MagicMock()
        mock_client.containers.get.side_effect = _docker.errors.NotFound("nope")
        backend.client = mock_client

        # Ensure the ack var is unset for this test
        with patch.dict(_os.environ, {}, clear=False):
            _os.environ.pop("OPENLEGION_BROWSER_ALLOW_HOST_NETWORK", None)
            with pytest.raises(RuntimeError, match="host network mode"):
                backend.start_browser_service()

        # Container.run should not have been called
        assert not mock_client.containers.run.called

    def test_browser_host_network_allowed_with_ack(self):
        """With the explicit ack env var, host-network browser startup proceeds."""
        import os as _os

        import docker as _docker

        backend = _make_docker_backend(use_host_network=True)
        mock_client = MagicMock()
        mock_client.containers.run.return_value = MagicMock()
        mock_client.containers.get.side_effect = _docker.errors.NotFound("nope")
        backend.client = mock_client

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("httpx.get", return_value=mock_resp), \
             patch.dict(_os.environ, {"OPENLEGION_BROWSER_ALLOW_HOST_NETWORK": "1"}):
            backend.start_browser_service()

        run_call = mock_client.containers.run.call_args
        assert run_call.kwargs.get("network_mode") == "host"
        # No cap_add in host mode (host netns shares with host, NET_ADMIN would be dangerous)
        assert "cap_add" not in run_call.kwargs
        env = run_call.kwargs.get("environment", {})
        assert env.get("BROWSER_EGRESS_DISABLE") == "1"

    def test_browser_private_ip_proxy_refused_without_allowlist(self):
        """A private-IP literal proxy URL must refuse startup unless allowlisted."""
        import os as _os

        import pytest

        import docker as _docker

        backend = self._make_backend()
        mock_client = MagicMock()
        mock_client.containers.run.return_value = MagicMock()
        mock_client.containers.get.side_effect = _docker.errors.NotFound("nope")
        backend.client = mock_client

        with patch.dict(_os.environ, {"BROWSER_PROXY_URL": "http://10.0.0.5:3128"}, clear=False):
            _os.environ.pop("BROWSER_EGRESS_ALLOWLIST", None)
            with pytest.raises(RuntimeError, match="private IP literal"):
                backend.start_browser_service()

        assert not mock_client.containers.run.called

    def test_browser_private_ip_proxy_allowed_with_explicit_allowlist(self):
        """Private-IP proxy is fine as long as operator explicitly allowlists it."""
        import os as _os

        import docker as _docker

        backend = self._make_backend()
        mock_client = MagicMock()
        mock_client.containers.run.return_value = MagicMock()
        mock_client.containers.get.side_effect = _docker.errors.NotFound("nope")
        backend.client = mock_client

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("httpx.get", return_value=mock_resp), \
             patch.dict(_os.environ, {
                 "BROWSER_PROXY_URL": "http://10.0.0.5:3128",
                 "BROWSER_EGRESS_ALLOWLIST": "10.0.0.5/32",
             }):
            backend.start_browser_service()

        run_call = mock_client.containers.run.call_args
        env = run_call.kwargs.get("environment", {})
        assert env.get("BROWSER_PROXY_URL") == "http://10.0.0.5:3128"
        assert env.get("BROWSER_EGRESS_ALLOWLIST") == "10.0.0.5/32"

    def test_browser_private_ip_proxy_refused_when_allowlist_does_not_cover(self):
        """Private-IP proxy must refuse if BROWSER_EGRESS_ALLOWLIST is set but does not cover the proxy IP."""
        import os as _os

        import pytest

        import docker as _docker

        backend = self._make_backend()
        mock_client = MagicMock()
        mock_client.containers.run.return_value = MagicMock()
        mock_client.containers.get.side_effect = _docker.errors.NotFound("nope")
        backend.client = mock_client

        # Allowlist covers 192.168.x but proxy is on 10.0.0.5 — misconfiguration.
        with patch.dict(_os.environ, {
            "BROWSER_PROXY_URL": "http://10.0.0.5:3128",
            "BROWSER_EGRESS_ALLOWLIST": "192.168.1.0/24",
        }):
            with pytest.raises(RuntimeError, match="does not cover"):
                backend.start_browser_service()

        assert not mock_client.containers.run.called

    def test_browser_private_ip_proxy_allowed_when_allowlist_covers_via_cidr(self):
        """A CIDR covering the proxy IP (not just /32) should satisfy the coverage check."""
        import os as _os

        import docker as _docker

        backend = self._make_backend()
        mock_client = MagicMock()
        mock_client.containers.run.return_value = MagicMock()
        mock_client.containers.get.side_effect = _docker.errors.NotFound("nope")
        backend.client = mock_client

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        # 10.0.0.5 is inside 10.0.0.0/24 — should pass.
        with patch("httpx.get", return_value=mock_resp), \
             patch.dict(_os.environ, {
                 "BROWSER_PROXY_URL": "http://10.0.0.5:3128",
                 "BROWSER_EGRESS_ALLOWLIST": "10.0.0.0/24",
             }):
            backend.start_browser_service()

        run_call = mock_client.containers.run.call_args
        env = run_call.kwargs.get("environment", {})
        assert env.get("BROWSER_EGRESS_ALLOWLIST") == "10.0.0.0/24"

    def test_browser_private_ip_proxy_allowed_with_mixed_allowlist(self):
        """Multi-entry allowlist where one entry covers the proxy should pass."""
        import os as _os

        import docker as _docker

        backend = self._make_backend()
        mock_client = MagicMock()
        mock_client.containers.run.return_value = MagicMock()
        mock_client.containers.get.side_effect = _docker.errors.NotFound("nope")
        backend.client = mock_client

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        # First entry is irrelevant, second covers 10.0.0.5.
        with patch("httpx.get", return_value=mock_resp), \
             patch.dict(_os.environ, {
                 "BROWSER_PROXY_URL": "http://10.0.0.5:3128",
                 "BROWSER_EGRESS_ALLOWLIST": "172.16.0.0/12, 10.0.0.0/8",
             }):
            backend.start_browser_service()

        run_call = mock_client.containers.run.call_args
        assert mock_client.containers.run.called
        env = run_call.kwargs.get("environment", {})
        assert env.get("BROWSER_EGRESS_ALLOWLIST") == "172.16.0.0/12, 10.0.0.0/8"

    def test_browser_public_proxy_no_refusal(self):
        """A public-IP proxy URL proceeds normally without any allowlist."""
        import os as _os

        import docker as _docker

        backend = self._make_backend()
        mock_client = MagicMock()
        mock_client.containers.run.return_value = MagicMock()
        mock_client.containers.get.side_effect = _docker.errors.NotFound("nope")
        backend.client = mock_client

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        # 1.1.1.1 (Cloudflare) is genuinely public — not classified as private,
        # loopback, link-local, or reserved by Python's ipaddress module, so the
        # private-IP guard should not trip. BROWSER_EGRESS_ALLOWLIST stays unset.
        with patch("httpx.get", return_value=mock_resp), \
             patch.dict(_os.environ, {"BROWSER_PROXY_URL": "http://1.1.1.1:3128"}, clear=False):
            _os.environ.pop("BROWSER_EGRESS_ALLOWLIST", None)
            backend.start_browser_service()

        run_call = mock_client.containers.run.call_args
        env = run_call.kwargs.get("environment", {})
        assert env.get("BROWSER_PROXY_URL") == "http://1.1.1.1:3128"
        assert "BROWSER_EGRESS_ALLOWLIST" not in env
        assert mock_client.containers.run.called

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
            tools_dir="",
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

        backend.start_agent(agent_id="test-agent", role="test", tools_dir="")

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

        backend.start_agent(agent_id="test-agent", role="test", tools_dir="")

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

        backend.start_agent(agent_id="test-agent", role="test", tools_dir="")

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
            "openlegion_agents",
            driver="bridge",
            options={"com.docker.network.bridge.enable_icc": "false"},
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
            "openlegion_agents",
            driver="bridge",
            options={"com.docker.network.bridge.enable_icc": "false"},
        )
        assert result is new_net

    def test_create_disables_inter_container_communication(self):
        """New network is created with enable_icc=false so a compromised agent
        cannot reach a peer agent's container directly on the shared bridge."""
        import docker as _docker

        backend = self._make_backend()
        backend.client.networks.get.side_effect = _docker.errors.NotFound("nope")
        backend.client.networks.create.return_value = MagicMock()

        backend._ensure_agent_network()

        create_kwargs = backend.client.networks.create.call_args.kwargs
        assert create_kwargs["options"] == {
            "com.docker.network.bridge.enable_icc": "false"
        }

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

        backend.start_agent(agent_id="test-agent", role="test", tools_dir="")

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

        backend.start_agent(agent_id="test-agent", role="test", tools_dir="")

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

        backend.start_agent(agent_id="test-agent", role="test", tools_dir="")

        run_call = mock_client.containers.run.call_args
        assert run_call.kwargs.get("read_only") is True
        assert run_call.kwargs.get("security_opt") == ["no-new-privileges"]


# ── env_overrides ──────────────────────────────────────────────


class TestEnvOverrides:
    """Tests for the env_overrides parameter on start_agent."""

    def test_docker_env_overrides_applied(self):
        """env_overrides are merged into container environment."""
        import docker as _docker

        backend = _make_docker_backend(extra_env={"EMBEDDING_MODEL": "text-embedding-3-small"})
        mock_client = MagicMock()
        mock_client.containers.run.return_value = MagicMock()
        mock_client.containers.get.side_effect = _docker.errors.NotFound("not found")
        backend.client = mock_client

        backend.start_agent(
            agent_id="test-agent",
            role="test",
            tools_dir="",
            env_overrides={"INITIAL_INSTRUCTIONS": "Be helpful.", "PROJECT_NAME": "myproj"},
        )

        run_call = mock_client.containers.run.call_args
        env = run_call.kwargs.get("environment", {})
        assert env["INITIAL_INSTRUCTIONS"] == "Be helpful."
        assert env["PROJECT_NAME"] == "myproj"
        assert env["EMBEDDING_MODEL"] == "text-embedding-3-small"

    def test_docker_env_overrides_do_not_mutate_extra_env(self):
        """Passing env_overrides must not modify the shared extra_env dict."""
        import docker as _docker

        backend = _make_docker_backend(extra_env={"EMBEDDING_MODEL": "text-embedding-3-small"})
        mock_client = MagicMock()
        mock_client.containers.run.return_value = MagicMock()
        mock_client.containers.get.side_effect = _docker.errors.NotFound("not found")
        backend.client = mock_client

        original_extra_env = dict(backend.extra_env)

        backend.start_agent(
            agent_id="agent-a",
            role="test",
            tools_dir="",
            env_overrides={"INITIAL_INSTRUCTIONS": "Agent A instructions", "INITIAL_SOUL": "A soul"},
        )

        # extra_env must be unchanged
        assert backend.extra_env == original_extra_env
        assert "INITIAL_INSTRUCTIONS" not in backend.extra_env
        assert "INITIAL_SOUL" not in backend.extra_env

    def test_docker_env_overrides_take_precedence(self):
        """env_overrides win over extra_env for the same key."""
        import docker as _docker

        backend = _make_docker_backend(extra_env={"LLM_MODEL": "system-default"})
        mock_client = MagicMock()
        mock_client.containers.run.return_value = MagicMock()
        mock_client.containers.get.side_effect = _docker.errors.NotFound("not found")
        backend.client = mock_client

        backend.start_agent(
            agent_id="test-agent",
            role="test",
            tools_dir="",
            env_overrides={"LLM_MODEL": "per-agent-model"},
        )

        run_call = mock_client.containers.run.call_args
        env = run_call.kwargs.get("environment", {})
        assert env["LLM_MODEL"] == "per-agent-model"

    def test_sandbox_env_overrides_in_env_file(self, tmp_path):
        """SandboxBackend merges env_overrides into the .agent.env file."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        backend = SandboxBackend.__new__(SandboxBackend)
        backend.project_root = project_root
        backend.mesh_host_port = 8420
        backend.agents = {}
        backend.auth_tokens = {}
        backend.extra_env = {"EMBEDDING_MODEL": "text-embedding-3-small"}
        backend._workspace_root = tmp_path / ".openlegion" / "agents"
        backend._workspace_root.mkdir(parents=True)

        ws = backend._prepare_workspace(
            agent_id="test-agent",
            role="test",
            tools_dir="",
            system_prompt="",
            model="openai/gpt-4o-mini",
            env_overrides={"INITIAL_INSTRUCTIONS": "Override instruction", "PROJECT_NAME": "proj1"},
        )

        env_content = (ws / ".agent.env").read_text()
        assert "INITIAL_INSTRUCTIONS=Override instruction" in env_content
        assert "PROJECT_NAME=proj1" in env_content
        assert "EMBEDDING_MODEL=text-embedding-3-small" in env_content

        # extra_env must be unchanged
        assert "INITIAL_INSTRUCTIONS" not in backend.extra_env
        assert "PROJECT_NAME" not in backend.extra_env

    def test_sandbox_env_overrides_do_not_mutate_extra_env(self, tmp_path):
        """Passing env_overrides to _prepare_workspace must not modify extra_env."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        backend = SandboxBackend.__new__(SandboxBackend)
        backend.project_root = project_root
        backend.mesh_host_port = 8420
        backend.agents = {}
        backend.auth_tokens = {}
        backend.extra_env = {"EMBEDDING_MODEL": "text-embedding-3-small"}
        backend._workspace_root = tmp_path / ".openlegion" / "agents"
        backend._workspace_root.mkdir(parents=True)

        original_extra_env = dict(backend.extra_env)

        backend._prepare_workspace(
            agent_id="agent-b",
            role="test",
            tools_dir="",
            env_overrides={"INITIAL_SOUL": "B soul", "HTTP_PROXY": "http://proxy:8080"},
        )

        assert backend.extra_env == original_extra_env

    def test_docker_none_env_overrides_is_safe(self):
        """Passing None for env_overrides works fine (no crash)."""
        import docker as _docker

        backend = _make_docker_backend()
        mock_client = MagicMock()
        mock_client.containers.run.return_value = MagicMock()
        mock_client.containers.get.side_effect = _docker.errors.NotFound("not found")
        backend.client = mock_client

        # Should not raise
        backend.start_agent(
            agent_id="test-agent",
            role="test",
            tools_dir="",
            env_overrides=None,
        )


class TestEntrypointHelpers:
    """Unit tests for the bash validation helpers in docker/browser-entrypoint.sh.

    The entrypoint script guards its install/exec section so that sourcing it
    from a subshell only loads function definitions, allowing us to test
    is_valid_ipv4 and is_valid_ipv4_cidr without Docker or root privileges.
    """

    @staticmethod
    def _call(helper: str, arg: str) -> bool:
        """Source the entrypoint and invoke a helper with the given arg.

        Returns True if the helper returned 0 (valid), False otherwise.
        Uses shlex.quote to prevent shell injection from test inputs.
        """
        import shlex
        import subprocess
        from pathlib import Path

        repo_root = Path(__file__).resolve().parent.parent
        script = repo_root / "docker" / "browser-entrypoint.sh"
        if not script.exists():
            import pytest
            pytest.skip(f"entrypoint script not found at {script}")
        # shellcheck-clean: source then call helper with quoted arg.
        cmd = f"source {shlex.quote(str(script))} && {helper} {shlex.quote(arg)}"
        result = subprocess.run(
            ["bash", "-c", cmd],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0

    @staticmethod
    def _run_entrypoint_function(function_name: str, *, env: dict[str, str]):
        import shlex
        import subprocess
        from pathlib import Path

        repo_root = Path(__file__).resolve().parent.parent
        script = repo_root / "docker" / "browser-entrypoint.sh"
        if not script.exists():
            import pytest
            pytest.skip(f"entrypoint script not found at {script}")
        cmd = f"source {shlex.quote(str(script))} && {function_name}"
        return subprocess.run(
            ["/bin/bash", "-c", cmd],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )

    # ── is_valid_ipv4 ─────────────────────────────────────────

    def test_ipv4_accepts_typical_addresses(self):
        for ip in ("10.0.0.5", "192.168.1.1", "1.2.3.4", "172.16.0.1", "8.8.8.8"):
            assert self._call("is_valid_ipv4", ip), ip

    def test_ipv4_accepts_boundary_octets(self):
        for ip in ("0.0.0.0", "255.255.255.255", "0.1.2.255", "255.0.0.0"):
            assert self._call("is_valid_ipv4", ip), ip

    def test_ipv4_rejects_out_of_range_octets(self):
        for ip in ("256.0.0.0", "999.999.999.999", "10.0.0.256", "300.1.2.3"):
            assert not self._call("is_valid_ipv4", ip), ip

    def test_ipv4_rejects_wrong_shape(self):
        for ip in ("10.0.0", "10.0.0.5.5", "", "10..0.5", "10.0.0.", ".10.0.0.5"):
            assert not self._call("is_valid_ipv4", ip), ip

    def test_ipv4_rejects_non_numeric(self):
        for ip in ("10.0.0.a", "localhost", "example.com", "10.0.0.5 ", " 10.0.0.5"):
            assert not self._call("is_valid_ipv4", ip), ip

    def test_ipv4_rejects_cidr_notation(self):
        # The strict IP validator does not accept /cidr — that is is_valid_ipv4_cidr's job.
        assert not self._call("is_valid_ipv4", "10.0.0.0/24")

    def test_ipv4_rejects_negative_and_hex(self):
        for ip in ("-1.0.0.0", "0x10.0.0.0", "10.0.0.-1"):
            assert not self._call("is_valid_ipv4", ip), ip

    # ── is_valid_ipv4_cidr ────────────────────────────────────

    def test_cidr_accepts_valid(self):
        for c in ("10.0.0.0/24", "192.168.1.1/32", "0.0.0.0/0",
                  "255.255.255.255/32", "10.0.0.5"):  # bare IP defaults to /32
            assert self._call("is_valid_ipv4_cidr", c), c

    def test_cidr_accepts_boundary_prefix(self):
        for c in ("10.0.0.0/0", "10.0.0.0/32", "10.0.0.0/1", "10.0.0.0/31"):
            assert self._call("is_valid_ipv4_cidr", c), c

    def test_cidr_rejects_out_of_range_prefix(self):
        for c in ("10.0.0.0/33", "10.0.0.0/99", "10.0.0.0/-1"):
            assert not self._call("is_valid_ipv4_cidr", c), c

    def test_cidr_rejects_bad_prefix_format(self):
        for c in ("10.0.0.0/abc", "10.0.0.0/", "10.0.0.0//24", "10.0.0.0/ 24"):
            assert not self._call("is_valid_ipv4_cidr", c), c

    def test_cidr_rejects_invalid_ip_portion(self):
        for c in ("999.999.999.999/24", "10.0.0/24", "10.0.0.a/24", "/24"):
            assert not self._call("is_valid_ipv4_cidr", c), c

    def test_cidr_rejects_empty(self):
        assert not self._call("is_valid_ipv4_cidr", "")

    def test_egress_filter_fails_closed_without_iptables_restore(self, tmp_path):
        result = self._run_entrypoint_function(
            "install_egress_filter",
            env={"PATH": str(tmp_path)},
        )
        assert result.returncode == 1
        assert "iptables-restore not installed" in result.stderr
        assert "refusing to start" in result.stderr

    def test_egress_filter_explicit_disable_still_skips(self, tmp_path):
        result = self._run_entrypoint_function(
            "install_egress_filter",
            env={"PATH": str(tmp_path), "BROWSER_EGRESS_DISABLE": "1"},
        )
        assert result.returncode == 0
        assert "skipping firewall setup" in result.stderr


# ── _build_mcp_servers_env: $CRED{name} resolution at agent start ──


class TestBuildMcpServersEnv:
    """The mesh resolves $CRED handles in mcp_servers env values and args
    just before serializing MCP_SERVERS for the agent container. The
    command field is left literal (handles there are rejected at config
    validation time). When the vault is not wired, configs with handles
    must fail loudly so a misconfigured deploy doesn't ship literal
    ``$CRED{...}`` to subprocesses.
    """

    def _backend(self, vault=None, permissions=None):
        b = _make_docker_backend()
        if vault is not None:
            b._vault = vault
        if permissions is not None:
            b._permissions = permissions
        return b

    def _vault_with(self, monkeypatch, **creds):
        from src.host.credentials import CredentialVault
        for name, value in creds.items():
            monkeypatch.setenv(f"OPENLEGION_CRED_{name.upper()}", value)
        return CredentialVault()

    def _perms_allow_all(self):
        p = MagicMock()
        p.can_access_credential.return_value = True
        return p

    def _perms_deny_all(self):
        p = MagicMock()
        p.can_access_credential.return_value = False
        return p

    def test_empty_servers_returns_empty_json_array(self):
        b = self._backend()
        assert b._build_mcp_servers_env(None, agent_id="a1") == "[]"
        assert b._build_mcp_servers_env([], agent_id="a1") == "[]"

    def test_plain_config_no_handles_passes_through(self):
        b = self._backend()
        servers = [{
            "name": "fs", "command": "mcp-server-fs", "args": ["/data"],
            "env": {"DEBUG": "1"},
        }]
        result = json.loads(b._build_mcp_servers_env(servers, agent_id="a1"))
        assert result == servers

    def test_handle_in_env_resolves_when_vault_wired(self, monkeypatch):
        b = self._backend(
            vault=self._vault_with(monkeypatch, linear_token="LINEAR-SECRET"),
            permissions=self._perms_allow_all(),
        )
        servers = [{
            "name": "linear", "command": "mcp-server-linear",
            "env": {"API_KEY": "$CRED{linear_token}"},
        }]
        result = json.loads(b._build_mcp_servers_env(servers, agent_id="a1"))
        assert result[0]["env"]["API_KEY"] == "LINEAR-SECRET"

    def test_handle_in_args_resolves_when_vault_wired(self, monkeypatch):
        b = self._backend(
            vault=self._vault_with(monkeypatch, linear_token="LINEAR-SECRET"),
            permissions=self._perms_allow_all(),
        )
        servers = [{
            "name": "linear", "command": "mcp-server-linear",
            "args": ["--token", "$CRED{linear_token}"],
        }]
        result = json.loads(b._build_mcp_servers_env(servers, agent_id="a1"))
        assert result[0]["args"] == ["--token", "LINEAR-SECRET"]

    def test_handle_in_env_without_vault_raises_clearly(self):
        # vault and permissions still None (class defaults).
        b = self._backend()
        servers = [{
            "name": "x", "command": "y",
            "env": {"KEY": "$CRED{anything}"},
        }]
        with pytest.raises(ValueError) as excinfo:
            b._build_mcp_servers_env(servers, agent_id="a1")
        msg = str(excinfo.value)
        assert "$CRED" in msg
        assert "a1" in msg
        assert "set_credential_resolver" in msg

    def test_handle_in_args_without_vault_raises_clearly(self):
        b = self._backend()
        servers = [{
            "name": "x", "command": "y",
            "args": ["--token", "$CRED{anything}"],
        }]
        with pytest.raises(ValueError):
            b._build_mcp_servers_env(servers, agent_id="a1")

    def test_clean_config_without_vault_does_not_raise(self):
        # No $CRED references → vault wiring not required.
        b = self._backend()
        servers = [{"name": "x", "command": "y", "args": ["a"], "env": {"K": "v"}}]
        # Should not raise.
        result = json.loads(b._build_mcp_servers_env(servers, agent_id="a1"))
        assert result == servers

    def test_missing_credential_propagates_value_error(self, monkeypatch):
        # Vault has linear_token; config asks for github_token.
        b = self._backend(
            vault=self._vault_with(monkeypatch, linear_token="x"),
            permissions=self._perms_allow_all(),
        )
        servers = [{
            "name": "x", "command": "y",
            "env": {"K": "$CRED{github_token}"},
        }]
        with pytest.raises(ValueError) as excinfo:
            b._build_mcp_servers_env(servers, agent_id="a1")
        assert "github_token" in str(excinfo.value)

    def test_permission_denied_propagates_value_error(self, monkeypatch):
        b = self._backend(
            vault=self._vault_with(monkeypatch, linear_token="x"),
            permissions=self._perms_deny_all(),
        )
        servers = [{
            "name": "x", "command": "y",
            "env": {"K": "$CRED{linear_token}"},
        }]
        with pytest.raises(ValueError) as excinfo:
            b._build_mcp_servers_env(servers, agent_id="a1")
        assert "permission" in str(excinfo.value).lower()

    def test_setter_wires_vault_and_permissions(self, monkeypatch):
        b = self._backend()
        v = self._vault_with(monkeypatch, linear_token="ABC")
        p = self._perms_allow_all()
        b.set_credential_resolver(vault=v, permissions=p)
        servers = [{
            "name": "x", "command": "y",
            "env": {"K": "$CRED{linear_token}"},
        }]
        result = json.loads(b._build_mcp_servers_env(servers, agent_id="a1"))
        assert result[0]["env"]["K"] == "ABC"

    def test_setter_with_none_resets_to_unwired(self):
        # Wiring None re-creates the "no resolver" state — refs would re-raise.
        b = self._backend()
        b._vault = MagicMock()  # pretend wired
        b._permissions = MagicMock()
        b.set_credential_resolver(vault=None, permissions=None)
        servers = [{
            "name": "x", "command": "y",
            "env": {"K": "$CRED{anything}"},
        }]
        with pytest.raises(ValueError):
            b._build_mcp_servers_env(servers, agent_id="a1")
