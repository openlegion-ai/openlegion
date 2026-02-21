"""Runtime backend abstraction for agent isolation.

Two backends:
  - DockerBackend: runs agents in Docker containers (shared kernel)
  - SandboxBackend: runs agents in Docker Sandbox microVMs (own kernel)

Both implement the same RuntimeBackend interface so the rest of the
system (health monitor, REPL, server) is isolation-agnostic.
"""

from __future__ import annotations

import abc
import asyncio
import json
import platform
import secrets
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from src.shared.utils import setup_logging

logger = setup_logging("host.runtime")


class RuntimeBackend(abc.ABC):
    """Abstract backend for starting, stopping, and monitoring agents."""

    def __init__(self, mesh_host_port: int = 8420, project_root: str | None = None):
        self.mesh_host_port = mesh_host_port
        self.project_root = (
            Path(project_root) if project_root
            else Path(__file__).resolve().parent.parent.parent
        )
        self.agents: dict[str, dict] = {}
        self.auth_tokens: dict[str, str] = {}

    @abc.abstractmethod
    def start_agent(
        self,
        agent_id: str,
        role: str,
        skills_dir: str,
        system_prompt: str = "",
        model: str = "",
        mcp_servers: list[dict] | None = None,
        browser_backend: str = "",
    ) -> str:
        """Start an agent. Returns a URL or identifier for reaching it."""

    @abc.abstractmethod
    def stop_agent(self, agent_id: str) -> None:
        """Stop and clean up an agent."""

    @abc.abstractmethod
    def health_check(self, agent_id: str) -> bool:
        """Check if the agent's runtime is alive (container/VM running)."""

    @abc.abstractmethod
    def get_logs(self, agent_id: str, tail: int = 40) -> str:
        """Retrieve recent logs from the agent's runtime."""

    @abc.abstractmethod
    async def wait_for_agent(self, agent_id: str, timeout: int = 30) -> bool:
        """Wait for the agent to become healthy. Returns True if ready."""

    def spawn_agent(
        self,
        agent_id: str,
        role: str,
        system_prompt: str = "",
        model: str = "",
        ttl: int = 3600,
        mcp_servers: list[dict] | None = None,
        browser_backend: str = "",
    ) -> str:
        """Spawn an ephemeral agent with a TTL for auto-cleanup."""
        url = self.start_agent(
            agent_id=agent_id, role=role, skills_dir="",
            system_prompt=system_prompt, model=model,
            mcp_servers=mcp_servers,
            browser_backend=browser_backend,
        )
        self.agents[agent_id]["ephemeral"] = True
        self.agents[agent_id]["ttl"] = ttl
        self.agents[agent_id]["spawned_at"] = time.time()
        return url

    def get_agent_url(self, agent_id: str) -> str | None:
        info = self.agents.get(agent_id)
        return info["url"] if info else None

    def list_agents(self) -> dict:
        return {
            aid: {"url": info["url"], "role": info["role"]}
            for aid, info in self.agents.items()
        }

    def stop_all(self) -> None:
        for agent_id in list(self.agents.keys()):
            self.stop_agent(agent_id)

    def get_container_logs(self, agent_id: str, tail: int = 40) -> str:
        """Backward-compatible alias for get_logs."""
        return self.get_logs(agent_id, tail=tail)

    @property
    def containers(self) -> dict[str, dict]:
        """Backward-compatible alias used by health monitor for restart info."""
        return self.agents

    @staticmethod
    def backend_name() -> str:
        return "unknown"


# ── Docker Container Backend ─────────────────────────────────


class DockerBackend(RuntimeBackend):
    """Runs agents in standard Docker containers (shared host kernel)."""

    BASE_IMAGE = "openlegion-agent:latest"

    def __init__(
        self,
        mesh_host_port: int = 8420,
        use_host_network: bool = False,
        project_root: str | None = None,
    ):
        super().__init__(mesh_host_port=mesh_host_port, project_root=project_root)
        import docker
        self.client = docker.from_env()
        self.use_host_network = use_host_network
        self._next_port = 8401
        self._cleanup_stale()

    @staticmethod
    def backend_name() -> str:
        return "docker"

    def _cleanup_stale(self) -> None:
        try:
            stale = self.client.containers.list(all=True, filters={"name": "openlegion_"})
            for c in stale:
                try:
                    c.remove(force=True)
                except Exception as e:
                    logger.debug("Could not remove stale container %s: %s", c.name, e)
        except Exception as e:
            logger.debug("Could not list stale containers: %s", e)

    def start_agent(
        self,
        agent_id: str,
        role: str,
        skills_dir: str,
        system_prompt: str = "",
        model: str = "",
        mcp_servers: list[dict] | None = None,
        browser_backend: str = "",
    ) -> str:
        import docker as _docker

        port = self._next_port
        self._next_port += 1

        # Generate per-agent auth token for mesh request verification
        auth_token = secrets.token_urlsafe(32)
        self.auth_tokens[agent_id] = auth_token

        mesh_host = "127.0.0.1" if self.use_host_network else "host.docker.internal"
        environment: dict[str, str] = {
            "AGENT_ID": agent_id,
            "AGENT_ROLE": role,
            "MESH_URL": f"http://{mesh_host}:{self.mesh_host_port}",
            "SKILLS_DIR": "/app/skills",
            "SYSTEM_PROMPT": system_prompt,
            "MESH_AUTH_TOKEN": auth_token,
        }
        if model:
            environment["LLM_MODEL"] = model
        if mcp_servers:
            environment["MCP_SERVERS"] = json.dumps(mcp_servers)
        if browser_backend:
            environment["BROWSER_BACKEND"] = browser_backend
        if self.use_host_network:
            environment["AGENT_PORT"] = str(port)

        volumes: dict[str, Any] = {
            f"openlegion_data_{agent_id}": {"bind": "/data", "mode": "rw"},
        }
        if skills_dir:
            # docker-py on Windows needs forward-slash or POSIX paths for bind mounts
            volumes[str(Path(skills_dir).as_posix() if platform.system() == "Windows" else skills_dir)] = {
                "bind": "/app/skills", "mode": "ro",
            }
        project_md = self.project_root / "PROJECT.md"
        if project_md.exists():
            host_path = str(project_md)
            if platform.system() == "Windows":
                host_path = project_md.as_posix()
            volumes[host_path] = {"bind": "/app/PROJECT.md", "mode": "ro"}

        soul_md = self.project_root / "SOUL.md"
        if soul_md.exists():
            host_path = str(soul_md)
            if platform.system() == "Windows":
                host_path = soul_md.as_posix()
            volumes[host_path] = {"bind": "/app/SOUL.md", "mode": "ro"}

        run_kwargs: dict[str, Any] = {
            "detach": True,
            "name": f"openlegion_{agent_id}",
            "environment": environment,
            "volumes": volumes,
            "mem_limit": "512m",
            "cpu_quota": 50000,
            "security_opt": ["no-new-privileges"],
        }

        if self.use_host_network:
            run_kwargs["network_mode"] = "host"
        else:
            run_kwargs["ports"] = {"8400/tcp": port}
            # On Linux Docker Engine, host.docker.internal isn't automatic
            if platform.system() == "Linux":
                run_kwargs["extra_hosts"] = {
                    "host.docker.internal": "host-gateway",
                }

        container_name = f"openlegion_{agent_id}"
        try:
            stale = self.client.containers.get(container_name)
            stale.remove(force=True)
        except _docker.errors.NotFound:
            pass

        container = self.client.containers.run(self.BASE_IMAGE, **run_kwargs)
        url = f"http://127.0.0.1:{port}"
        self.agents[agent_id] = {
            "container": container,
            "url": url,
            "port": port,
            "role": role,
            "skills_dir": skills_dir,
            "system_prompt": system_prompt,
            "model": model,
            "mcp_servers": mcp_servers,
        }
        logger.info(f"Started agent '{agent_id}' (role={role}) at {url}")
        return url

    def stop_agent(self, agent_id: str) -> None:
        if agent_id in self.agents:
            try:
                self.agents[agent_id]["container"].stop(timeout=10)
                self.agents[agent_id]["container"].remove()
                logger.info(f"Stopped agent '{agent_id}'")
            except Exception as e:
                logger.warning(f"Error stopping agent '{agent_id}': {e}")
            del self.agents[agent_id]

    def health_check(self, agent_id: str) -> bool:
        if agent_id not in self.agents:
            return False
        try:
            container = self.agents[agent_id]["container"]
            container.reload()
            return container.status == "running"
        except Exception as e:
            logger.debug("Health check failed for '%s': %s", agent_id, e)
            return False

    def get_logs(self, agent_id: str, tail: int = 40) -> str:
        if agent_id not in self.agents:
            return ""
        try:
            container = self.agents[agent_id]["container"]
            container.reload()
            return container.logs(tail=tail).decode("utf-8", errors="replace")
        except Exception as e:
            logger.debug("Could not get logs for '%s': %s", agent_id, e)
            return ""

    async def wait_for_agent(self, agent_id: str, timeout: int = 30) -> bool:
        import httpx
        url = self.get_agent_url(agent_id)
        if not url:
            return False
        start = time.time()
        last_error = ""
        while time.time() - start < timeout:
            if not self.health_check(agent_id):
                logs = self.get_logs(agent_id, tail=20)
                logger.warning(
                    f"Agent '{agent_id}' container exited during startup. "
                    f"Logs:\n{logs}"
                )
                return False
            try:
                async with httpx.AsyncClient(timeout=3) as client:
                    resp = await client.get(f"{url}/status")
                    if resp.status_code == 200:
                        return True
            except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError) as e:
                last_error = str(e)
            except Exception as e:
                last_error = str(e)
                logger.debug(f"Unexpected error polling agent '{agent_id}': {e}")
            await asyncio.sleep(0.5)
        logger.warning(
            f"Agent '{agent_id}' did not respond within {timeout}s. "
            f"URL: {url}, last error: {last_error}"
        )
        return False


# ── Docker Sandbox (MicroVM) Backend ─────────────────────────


class SandboxBackend(RuntimeBackend):
    """Runs agents in Docker Sandbox microVMs (hypervisor isolation).

    Each agent gets its own VM with a private kernel. Communication
    happens via ``docker sandbox exec`` (no direct network access
    from the host into the VM).
    """

    AGENT_PORT = 8400

    def __init__(
        self,
        mesh_host_port: int = 8420,
        project_root: str | None = None,
    ):
        super().__init__(mesh_host_port=mesh_host_port, project_root=project_root)
        self._workspace_root = self.project_root / ".openlegion" / "agents"
        self._workspace_root.mkdir(parents=True, exist_ok=True)
        self._cleanup_stale()

    @staticmethod
    def backend_name() -> str:
        return "sandbox"

    def _cleanup_stale(self) -> None:
        """Remove leftover sandboxes from previous runs."""
        try:
            result = subprocess.run(
                ["docker", "sandbox", "ls", "--format", "json"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return
            for line in result.stdout.strip().splitlines():
                try:
                    info = json.loads(line)
                    name = info.get("Name", "")
                    if name.startswith("openlegion_"):
                        subprocess.run(
                            ["docker", "sandbox", "rm", "-f", name],
                            capture_output=True, timeout=15,
                        )
                        logger.debug(f"Removed stale sandbox: {name}")
                except (json.JSONDecodeError, KeyError):
                    pass
        except Exception as e:
            logger.debug("Could not list stale sandboxes: %s", e)

    def _prepare_workspace(
        self,
        agent_id: str,
        role: str,
        skills_dir: str,
        system_prompt: str,
        model: str,
        mcp_servers: list[dict] | None = None,
        browser_backend: str = "",
    ) -> Path:
        """Create the per-agent host directory that will sync into the sandbox."""
        ws = self._workspace_root / agent_id
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "data" / "workspace").mkdir(parents=True, exist_ok=True)

        # Copy PROJECT.md
        project_md = self.project_root / "PROJECT.md"
        if project_md.exists():
            shutil.copy2(project_md, ws / "PROJECT.md")

        # Copy SOUL.md (project-level persona)
        soul_md = self.project_root / "SOUL.md"
        if soul_md.exists():
            shutil.copy2(soul_md, ws / "SOUL.md")

        # Copy skills
        skills_dest = ws / "skills"
        if skills_dir and Path(skills_dir).is_dir():
            if skills_dest.exists():
                shutil.rmtree(skills_dest)
            shutil.copytree(skills_dir, skills_dest)
        else:
            skills_dest.mkdir(exist_ok=True)

        # Generate per-agent auth token for mesh request verification
        auth_token = secrets.token_urlsafe(32)
        self.auth_tokens[agent_id] = auth_token

        # Write agent env config
        env_cfg = {
            "AGENT_ID": agent_id,
            "AGENT_ROLE": role,
            "MESH_URL": f"http://host.docker.internal:{self.mesh_host_port}",
            "SKILLS_DIR": "/app/skills",
            "SYSTEM_PROMPT": system_prompt,
            "AGENT_PORT": str(self.AGENT_PORT),
            "MESH_AUTH_TOKEN": auth_token,
        }
        if model:
            env_cfg["LLM_MODEL"] = model
        if mcp_servers:
            env_cfg["MCP_SERVERS"] = json.dumps(mcp_servers)
        if browser_backend:
            env_cfg["BROWSER_BACKEND"] = browser_backend

        env_file = ws / ".agent.env"
        env_file.write_text(
            "\n".join(f"{k}={v}" for k, v in env_cfg.items()) + "\n"
        )
        return ws

    def start_agent(
        self,
        agent_id: str,
        role: str,
        skills_dir: str,
        system_prompt: str = "",
        model: str = "",
        mcp_servers: list[dict] | None = None,
        browser_backend: str = "",
    ) -> str:
        sandbox_name = f"openlegion_{agent_id}"
        ws = self._prepare_workspace(agent_id, role, skills_dir, system_prompt, model, mcp_servers=mcp_servers, browser_backend=browser_backend)

        # Create sandbox with the shell agent type and workspace
        # First creation can be slow (microVM init), allow up to 120s
        create_cmd = [
            "docker", "sandbox", "create",
            "--name", sandbox_name,
            "shell", str(ws),
        ]
        result = subprocess.run(
            create_cmd, capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to create sandbox for '{agent_id}': {result.stderr.strip()}"
            )
        logger.info(f"Created sandbox '{sandbox_name}'")

        # Start the agent process inside the sandbox (detached)
        env_file = ws / ".agent.env"
        start_cmd = [
            "docker", "sandbox", "exec", "-d",
            "--env-file", str(env_file),
            sandbox_name, "--",
            "python", "-m", "src.agent",
        ]
        result = subprocess.run(
            start_cmd, capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            logger.warning(
                f"Agent process start may have failed for '{agent_id}': "
                f"{result.stderr.strip()}"
            )

        # No direct URL -- communication via sandbox exec transport
        url = f"sandbox://{sandbox_name}"
        self.agents[agent_id] = {
            "sandbox_name": sandbox_name,
            "workspace": str(ws),
            "url": url,
            "role": role,
            "skills_dir": skills_dir,
            "system_prompt": system_prompt,
            "model": model,
            "mcp_servers": mcp_servers,
        }
        logger.info(f"Started agent '{agent_id}' in sandbox '{sandbox_name}'")
        return url

    def stop_agent(self, agent_id: str) -> None:
        if agent_id not in self.agents:
            return
        sandbox_name = self.agents[agent_id]["sandbox_name"]
        try:
            subprocess.run(
                ["docker", "sandbox", "rm", "-f", sandbox_name],
                capture_output=True, timeout=15,
            )
            logger.info(f"Removed sandbox '{sandbox_name}'")
        except Exception as e:
            logger.warning(f"Error removing sandbox '{sandbox_name}': {e}")
        del self.agents[agent_id]

    def health_check(self, agent_id: str) -> bool:
        if agent_id not in self.agents:
            return False
        sandbox_name = self.agents[agent_id]["sandbox_name"]
        try:
            result = subprocess.run(
                ["docker", "sandbox", "inspect", sandbox_name],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return False
            info = json.loads(result.stdout)
            status = info.get("Status", "") if isinstance(info, dict) else ""
            return status.lower() in ("running", "ready")
        except Exception as e:
            logger.debug("Health check failed for sandbox '%s': %s", agent_id, e)
            return False

    def get_logs(self, agent_id: str, tail: int = 40) -> str:
        if agent_id not in self.agents:
            return ""
        sandbox_name = self.agents[agent_id]["sandbox_name"]
        try:
            result = subprocess.run(
                [
                    "docker", "sandbox", "exec", sandbox_name, "--",
                    "tail", f"-{tail}", "/tmp/agent.log",
                ],
                capture_output=True, text=True, timeout=10,
            )
            return result.stdout if result.returncode == 0 else ""
        except Exception as e:
            logger.debug("Could not get logs for sandbox '%s': %s", agent_id, e)
            return ""

    async def wait_for_agent(self, agent_id: str, timeout: int = 30) -> bool:
        from src.host.transport import SandboxTransport
        transport = SandboxTransport()
        start = time.time()
        while time.time() - start < timeout:
            if not self.health_check(agent_id):
                logger.warning(f"Agent '{agent_id}' sandbox not running during startup")
                return False
            if await transport.is_reachable(agent_id, timeout=5):
                return True
            await asyncio.sleep(1.0)
        return False


# ── Detection ─────────────────────────────────────────────────


def sandbox_available() -> bool:
    """Check if ``docker sandbox`` CLI is available on this system."""
    try:
        result = subprocess.run(
            ["docker", "sandbox", "version"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def select_backend(
    mesh_host_port: int = 8420,
    project_root: str | None = None,
    use_sandbox: bool = False,
) -> RuntimeBackend:
    """Select the runtime backend.

    Defaults to DockerBackend (container isolation) which works everywhere.
    Use use_sandbox=True to opt in to SandboxBackend (microVM isolation),
    which requires Docker Desktop 4.58+ with sandbox support.
    """
    if use_sandbox:
        if sandbox_available():
            logger.info("Docker Sandbox available -- using microVM isolation")
            return SandboxBackend(
                mesh_host_port=mesh_host_port,
                project_root=project_root,
            )
        logger.warning(
            "Docker Sandbox requested but not available. "
            "Requires Docker Desktop 4.58+. Falling back to containers."
        )
    # Host networking only works reliably on Linux. On macOS/Windows,
    # Docker Desktop runs in a VM so --network=host doesn't expose ports
    # to the actual host. Use port mapping (bridge network) instead.
    use_host_net = platform.system() == "Linux"
    logger.info("Using Docker container isolation (host_network=%s)", use_host_net)
    return DockerBackend(
        mesh_host_port=mesh_host_port,
        use_host_network=use_host_net,
        project_root=project_root,
    )
