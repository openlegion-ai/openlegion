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

    def _resolve_headers(self, headers: dict[str, str] | None) -> dict[str, str]:
        """Return *headers* with mesh-internal marker and trace context.

        Always injects X-Mesh-Internal so agent endpoints can distinguish
        mesh/dashboard requests from agent self-calls (http_request tool).
        """
        if headers is None:
            from src.shared.trace import trace_headers
            headers = trace_headers()
        headers.setdefault("x-mesh-internal", "1")
        return headers

    @abc.abstractmethod
    async def request(
        self,
        agent_id: str,
        method: str,
        path: str,
        json: dict | None = None,
        timeout: int = 120,
        headers: dict[str, str] | None = None,
    ) -> dict:
        """Send an HTTP request to an agent. Returns parsed JSON response."""

    @abc.abstractmethod
    async def is_reachable(self, agent_id: str, timeout: int = 5) -> bool:
        """Quick check whether the agent responds to /status."""

    async def stream_request(
        self,
        agent_id: str,
        method: str,
        path: str,
        json: dict | None = None,
        timeout: int = 120,
        headers: dict[str, str] | None = None,
    ):
        """Streaming HTTP request. Yields SSE lines as they arrive.

        Default implementation falls back to non-streaming request.
        """
        result = await self.request(agent_id, method, path, json=json, timeout=timeout, headers=headers)
        yield result

    @abc.abstractmethod
    def request_sync(
        self,
        agent_id: str,
        method: str,
        path: str,
        json: dict | None = None,
        timeout: int = 120,
        headers: dict[str, str] | None = None,
    ) -> dict:
        """Synchronous variant for use in non-async contexts (REPL, callbacks)."""


class HttpTransport(Transport):
    """Direct HTTP transport -- for agents running in Docker containers."""

    def __init__(self) -> None:
        self._urls: dict[str, str] = {}
        # One httpx.AsyncClient per event loop — each loop's client has
        # asyncio internals bound to that loop.  Keyed by id(loop).
        self._clients: dict[int, httpx.AsyncClient] = {}

    async def _get_client(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        key = id(loop)
        client = self._clients.get(key)
        if client is None or client.is_closed:
            client = httpx.AsyncClient(timeout=120)
            self._clients[key] = client
        return client

    async def close(self) -> None:
        loop = asyncio.get_running_loop()
        key = id(loop)
        client = self._clients.pop(key, None)
        if client and not client.is_closed:
            await client.aclose()

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
        headers: dict[str, str] | None = None,
    ) -> dict:
        url = self._urls.get(agent_id)
        if not url:
            return {"error": f"Agent '{agent_id}' not registered in transport"}
        client = await self._get_client()
        resp = await client.request(
            method, f"{url}{path}", json=json, timeout=timeout,
            headers=self._resolve_headers(headers),
        )
        resp.raise_for_status()
        return resp.json()

    async def is_reachable(self, agent_id: str, timeout: int = 5) -> bool:
        url = self._urls.get(agent_id)
        if not url:
            return False
        try:
            client = await self._get_client()
            resp = await client.get(f"{url}/status", timeout=timeout)
            return resp.status_code == 200
        except Exception as e:
            logger.debug("Reachability check failed for '%s': %s", agent_id, e)
            return False

    async def stream_request(
        self,
        agent_id: str,
        method: str,
        path: str,
        json: dict | None = None,
        timeout: int = 120,
        headers: dict[str, str] | None = None,
    ):
        """Streaming HTTP request. Yields parsed SSE data lines."""
        url = self._urls.get(agent_id)
        if not url:
            yield {"error": f"Agent '{agent_id}' not registered in transport"}
            return
        client = await self._get_client()
        async with client.stream(
            method, f"{url}{path}", json=json, timeout=timeout,
            headers=self._resolve_headers(headers),
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    try:
                        yield json_module.loads(line[6:])
                    except json_module.JSONDecodeError:
                        yield {"raw": line[6:]}

    def request_sync(
        self,
        agent_id: str,
        method: str,
        path: str,
        json: dict | None = None,
        timeout: int = 120,
        headers: dict[str, str] | None = None,
    ) -> dict:
        url = self._urls.get(agent_id)
        if not url:
            return {"error": f"Agent '{agent_id}' not registered in transport"}
        resp = httpx.request(
            method, f"{url}{path}", json=json, timeout=timeout,
            headers=self._resolve_headers(headers),
        )
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
        headers: dict[str, str] | None = None,
    ) -> dict:
        sandbox_name = f"openlegion_{agent_id}"
        url = f"http://localhost:{self.AGENT_PORT}{path}"
        cmd: list[str] = [
            "docker", "sandbox", "exec", sandbox_name, "--",
            "curl", "-s", "-X", method, url,
            "-H", "Content-Type: application/json",
        ]
        for hk, hv in self._resolve_headers(headers).items():
            cmd.extend(["-H", f"{hk}: {hv}"])
        if json is not None:
            cmd.extend(["-d", json_module.dumps(json)])

        proc = None
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
            if proc is not None:
                try:
                    proc.kill()
                    await proc.wait()
                except ProcessLookupError:
                    pass
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
        headers: dict[str, str] | None = None,
    ) -> dict:
        import subprocess

        sandbox_name = f"openlegion_{agent_id}"
        url = f"http://localhost:{self.AGENT_PORT}{path}"
        cmd: list[str] = [
            "docker", "sandbox", "exec", sandbox_name, "--",
            "curl", "-s", "-X", method, url,
            "-H", "Content-Type: application/json",
        ]
        for hk, hv in self._resolve_headers(headers).items():
            cmd.extend(["-H", f"{hk}: {hv}"])
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
