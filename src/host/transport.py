"""Transport layer for reaching agent HTTP APIs.

Two implementations:
  - HttpTransport: direct HTTP calls (Docker containers on host network)
  - SandboxTransport: docker sandbox exec + curl (microVM isolation)

Both present the same interface so callers (health monitor, REPL, server)
don't need to know which isolation backend is active.
"""

from __future__ import annotations

import abc
import asyncio
import json as json_module

import httpx

from src.shared.utils import setup_logging

logger = setup_logging("host.transport")


class Transport(abc.ABC):
    """Abstract transport for reaching an agent's HTTP API."""

    @abc.abstractmethod
    async def request(
        self,
        agent_id: str,
        method: str,
        path: str,
        json: dict | None = None,
        timeout: int = 120,
    ) -> dict:
        """Send an HTTP request to an agent. Returns parsed JSON response."""

    @abc.abstractmethod
    async def is_reachable(self, agent_id: str, timeout: int = 5) -> bool:
        """Quick check whether the agent responds to /status."""

    @abc.abstractmethod
    def request_sync(
        self,
        agent_id: str,
        method: str,
        path: str,
        json: dict | None = None,
        timeout: int = 120,
    ) -> dict:
        """Synchronous variant for use in non-async contexts (REPL, callbacks)."""


class HttpTransport(Transport):
    """Direct HTTP transport -- for agents running in Docker containers."""

    def __init__(self) -> None:
        self._urls: dict[str, str] = {}

    def register(self, agent_id: str, url: str) -> None:
        self._urls[agent_id] = url

    def get_url(self, agent_id: str) -> str | None:
        return self._urls.get(agent_id)

    async def request(
        self,
        agent_id: str,
        method: str,
        path: str,
        json: dict | None = None,
        timeout: int = 120,
    ) -> dict:
        url = self._urls.get(agent_id)
        if not url:
            return {"error": f"Agent '{agent_id}' not registered in transport"}
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.request(method, f"{url}{path}", json=json)
            resp.raise_for_status()
            return resp.json()

    async def is_reachable(self, agent_id: str, timeout: int = 5) -> bool:
        url = self._urls.get(agent_id)
        if not url:
            return False
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(f"{url}/status")
                return resp.status_code == 200
        except Exception:
            return False

    def request_sync(
        self,
        agent_id: str,
        method: str,
        path: str,
        json: dict | None = None,
        timeout: int = 120,
    ) -> dict:
        url = self._urls.get(agent_id)
        if not url:
            return {"error": f"Agent '{agent_id}' not registered in transport"}
        resp = httpx.request(method, f"{url}{path}", json=json, timeout=timeout)
        resp.raise_for_status()
        return resp.json()


class SandboxTransport(Transport):
    """docker sandbox exec transport -- for agents running in microVMs.

    Reaches the agent's FastAPI server (listening on localhost:8400 inside
    the VM) by shelling out to ``docker sandbox exec ... curl``.
    """

    AGENT_PORT = 8400

    async def request(
        self,
        agent_id: str,
        method: str,
        path: str,
        json: dict | None = None,
        timeout: int = 120,
    ) -> dict:
        sandbox_name = f"openlegion_{agent_id}"
        url = f"http://localhost:{self.AGENT_PORT}{path}"
        cmd: list[str] = [
            "docker", "sandbox", "exec", sandbox_name, "--",
            "curl", "-s", "-X", method, url,
            "-H", "Content-Type: application/json",
        ]
        if json is not None:
            cmd.extend(["-d", json_module.dumps(json)])

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout,
            )
            if proc.returncode != 0:
                err = stderr.decode(errors="replace").strip()
                logger.warning(f"sandbox exec failed for '{agent_id}': {err}")
                return {"error": err or f"exit code {proc.returncode}"}
            return json_module.loads(stdout.decode(errors="replace"))
        except asyncio.TimeoutError:
            logger.warning(f"sandbox exec timed out for '{agent_id}' ({path})")
            return {"error": f"Timeout after {timeout}s"}
        except json_module.JSONDecodeError as e:
            logger.warning(f"Non-JSON response from '{agent_id}' ({path}): {e}")
            return {"error": "Non-JSON response from agent"}
        except Exception as e:
            logger.warning(f"sandbox exec error for '{agent_id}': {e}")
            return {"error": str(e)}

    async def is_reachable(self, agent_id: str, timeout: int = 5) -> bool:
        result = await self.request(agent_id, "GET", "/status", timeout=timeout)
        return "error" not in result

    def request_sync(
        self,
        agent_id: str,
        method: str,
        path: str,
        json: dict | None = None,
        timeout: int = 120,
    ) -> dict:
        import subprocess

        sandbox_name = f"openlegion_{agent_id}"
        url = f"http://localhost:{self.AGENT_PORT}{path}"
        cmd: list[str] = [
            "docker", "sandbox", "exec", sandbox_name, "--",
            "curl", "-s", "-X", method, url,
            "-H", "Content-Type: application/json",
        ]
        if json is not None:
            cmd.extend(["-d", json_module.dumps(json)])

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
            )
            if result.returncode != 0:
                return {"error": result.stderr.strip() or f"exit code {result.returncode}"}
            return json_module.loads(result.stdout)
        except subprocess.TimeoutExpired:
            return {"error": f"Timeout after {timeout}s"}
        except Exception as e:
            return {"error": str(e)}


def resolve_url(agent_registry: dict, agent_id: str) -> str | None:
    """Extract a URL from the router's agent_registry entry."""
    info = agent_registry.get(agent_id)
    if info is None:
        return None
    return info.get("url", info) if isinstance(info, dict) else info
