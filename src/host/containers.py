"""Docker container lifecycle management for agent instances.

Security model:
  - Each agent runs in its own container
  - Containers are on a Docker bridge network
  - No credentials mounted inside containers
  - Filesystem isolation: each agent gets its own /data volume
  - Resource limits (memory, CPU, no-new-privileges)

Note: When mesh runs on the host, the network must NOT be internal
so agents can reach host.docker.internal. For full isolation, deploy
the mesh as a container on the same network (see docker-compose.yml).
"""

from __future__ import annotations

import asyncio
import platform
import time
from typing import Optional

import docker
import httpx

from src.shared.utils import setup_logging

logger = setup_logging("host.containers")


class ContainerManager:
    """Manages Docker containers for agent instances."""

    NETWORK_NAME = "openlegion_internal"
    BASE_IMAGE = "openlegion-agent:latest"

    def __init__(self, mesh_host_port: int = 8420, use_host_network: bool = False):
        self.client = docker.from_env()
        self.containers: dict[str, dict] = {}
        self.mesh_host_port = mesh_host_port
        self.use_host_network = use_host_network
        self._next_port = 8401
        if not use_host_network:
            self._ensure_network()

    def _ensure_network(self) -> None:
        """Create Docker bridge network if it doesn't exist."""
        try:
            self.client.networks.get(self.NETWORK_NAME)
        except docker.errors.NotFound:
            self.client.networks.create(
                self.NETWORK_NAME,
                driver="bridge",
            )
            logger.info(f"Created network: {self.NETWORK_NAME}")

    def start_agent(
        self,
        agent_id: str,
        role: str,
        skills_dir: str,
        system_prompt: str = "",
        model: str = "",
        env_extra: dict | None = None,
    ) -> str:
        """Start a new agent container. Returns the container's URL."""
        port = self._next_port
        self._next_port += 1

        mesh_host = "127.0.0.1" if self.use_host_network else "host.docker.internal"
        environment = {
            "AGENT_ID": agent_id,
            "AGENT_ROLE": role,
            "MESH_URL": f"http://{mesh_host}:{self.mesh_host_port}",
            "SKILLS_DIR": "/app/skills",
            "SYSTEM_PROMPT": system_prompt,
        }
        if model:
            environment["LLM_MODEL"] = model
        if env_extra:
            environment.update(env_extra)

        volumes = {
            f"openlegion_data_{agent_id}": {"bind": "/data", "mode": "rw"},
        }
        if skills_dir:
            volumes[skills_dir] = {"bind": "/app/skills", "mode": "ro"}

        if self.use_host_network:
            environment["AGENT_PORT"] = str(port)

        run_kwargs: dict = {
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
            extra_hosts = {}
            if platform.system() == "Linux":
                extra_hosts["host.docker.internal"] = "host-gateway"
            run_kwargs["network"] = self.NETWORK_NAME
            run_kwargs["ports"] = {"8400/tcp": port}
            run_kwargs["extra_hosts"] = extra_hosts

        container = self.client.containers.run(self.BASE_IMAGE, **run_kwargs)

        # With host networking, the agent listens directly on the assigned port
        url = f"http://localhost:{port}"
        self.containers[agent_id] = {
            "container": container,
            "url": url,
            "port": port,
            "role": role,
        }

        logger.info(f"Started agent '{agent_id}' (role={role}) at {url}")
        return url

    def stop_agent(self, agent_id: str) -> None:
        """Stop and remove an agent container."""
        if agent_id in self.containers:
            try:
                self.containers[agent_id]["container"].stop(timeout=10)
                self.containers[agent_id]["container"].remove()
                logger.info(f"Stopped agent '{agent_id}'")
            except Exception as e:
                logger.warning(f"Error stopping agent '{agent_id}': {e}")
            del self.containers[agent_id]

    def get_agent_url(self, agent_id: str) -> Optional[str]:
        """Get the URL for an agent's container."""
        if agent_id in self.containers:
            return self.containers[agent_id]["url"]
        return None

    def list_agents(self) -> dict:
        """List all running agent containers."""
        return {aid: {"url": info["url"], "role": info["role"]} for aid, info in self.containers.items()}

    def stop_all(self) -> None:
        """Stop all agent containers."""
        for agent_id in list(self.containers.keys()):
            self.stop_agent(agent_id)

    def health_check(self, agent_id: str) -> bool:
        """Check if an agent container is healthy."""
        if agent_id not in self.containers:
            return False
        try:
            container = self.containers[agent_id]["container"]
            container.reload()
            return container.status == "running"
        except Exception:
            return False

    async def wait_for_agent(self, agent_id: str, timeout: int = 30) -> bool:
        """Wait for an agent to become healthy. Returns True if ready."""
        url = self.get_agent_url(agent_id)
        if not url:
            return False
        start = time.time()
        while time.time() - start < timeout:
            try:
                async with httpx.AsyncClient(timeout=2) as client:
                    response = await client.get(f"{url}/status")
                    if response.status_code == 200:
                        return True
            except (httpx.ConnectError, httpx.TimeoutException):
                pass
            await asyncio.sleep(0.5)
        return False
