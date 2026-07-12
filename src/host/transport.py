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
import re
from collections.abc import Awaitable, Callable

import httpx

from src.shared.types import AGENT_ID_RE_PATTERN
from src.shared.utils import friendly_streaming_error, setup_logging

logger = setup_logging("host.transport")


class Transport(abc.ABC):
    """Abstract transport for reaching an agent's HTTP API."""

    def __init__(self) -> None:
        # agent_id -> mesh→agent bearer token. The agent server (B7) requires
        # ``Authorization: Bearer <MESH_AUTH_TOKEN>`` on every request except
        # ``GET /status``; this mapping supplies the per-agent token. Wired
        # via ``bind_tokens`` (production: the runtime's LIVE ``auth_tokens``
        # dict, so restart-rotated tokens are picked up automatically) or
        # ``register_token`` (granular, mirrors ``register(agent_id, url)``).
        self._agent_tokens: dict[str, str] = {}
        # Cold-wake seam (plan §8 #24 leg 3). ``None`` (the default, every
        # test construction) means hibernation isn't wired at all — every
        # request path below is then a pure no-op branch. Production
        # wiring (``cli/runtime.py``) sets this to the mesh app's
        # ``ensure_agent_running`` closure AFTER ``create_mesh_app`` runs
        # (mirrors ``bind_tokens``'s post-construction wiring). Deliberately
        # NOT called from ``is_reachable`` — health polls must never wake a
        # hibernated agent (it is health-unregistered anyway; belt and
        # suspenders).
        self._ensure_running_fn: Callable[[str], Awaitable[bool]] | None = None

    def register_token(self, agent_id: str, token: str) -> None:
        """Record the mesh→agent bearer token for *agent_id*."""
        self._agent_tokens[agent_id] = token

    def set_ensure_running_fn(
        self, fn: Callable[[str], Awaitable[bool]] | None,
    ) -> None:
        """Wire the cold-wake seam (plan §8 #24) after construction.

        ``fn(agent_id) -> bool`` is a cheap no-op (a cached status check,
        no I/O) when the agent isn't hibernated, and cold-wakes + waits
        for readiness when it is. Called at the top of ``request`` /
        ``stream_request`` / ``request_sync`` — never from
        ``is_reachable``. Pass ``None`` to disable (the default).
        """
        self._ensure_running_fn = fn

    async def _ensure_running(self, agent_id: str) -> None:
        """Best-effort cold-wake check before forwarding a request.

        Never raises and never blocks the caller on a wake failure — a
        hibernated agent that fails to wake just stays unreachable, which
        the caller's existing connect-failure handling already covers.
        """
        fn = self._ensure_running_fn
        if fn is None or not agent_id:
            return
        try:
            await fn(agent_id)
        except Exception as e:
            logger.warning("ensure_running_fn failed for '%s': %s", agent_id, e)

    def bind_tokens(self, tokens: dict[str, str]) -> None:
        """Bind a LIVE agent_id→token mapping (kept by reference, not copied).

        Production wiring passes ``runtime.auth_tokens`` — the same dict the
        backend mutates on every ``start_agent``/``remove_agent`` — so every
        registration path (initial start, REPL /restart, health-monitor
        restart, dashboard restart, /mesh/register) is covered by a single
        bind at transport construction, and token rotation on container
        restart needs no per-call-site re-sync.
        """
        self._agent_tokens = tokens

    def _resolve_headers(
        self, headers: dict[str, str] | None, agent_id: str | None = None,
    ) -> dict[str, str]:
        """Return *headers* with mesh-internal marker, trace context, and auth.

        Always injects X-Mesh-Internal so agent endpoints can distinguish
        mesh/dashboard requests from agent self-calls (http_request tool).
        When the target agent's mesh→agent bearer token is known, attaches
        it as ``Authorization`` (the agent server enforces it — audit C1/B7).
        """
        if headers is None:
            from src.shared.trace import trace_headers
            headers = trace_headers()
        headers.setdefault("x-mesh-internal", "1")
        # Advertise the wire-protocol version so the agent server can reject a
        # version-skewed mesh (rolling upgrade) instead of mis-decoding JSON.
        from src.shared.trace import PROTOCOL_VERSION, PROTOCOL_VERSION_HEADER
        headers.setdefault(PROTOCOL_VERSION_HEADER, PROTOCOL_VERSION)
        token = self._agent_tokens.get(agent_id) if agent_id else None
        if token:
            headers.setdefault("Authorization", f"Bearer {token}")
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
        super().__init__()
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
        await self._ensure_running(agent_id)
        url = self._urls.get(agent_id)
        if not url:
            return {"error": f"Agent '{agent_id}' not registered in transport"}
        try:
            client = await self._get_client()
            resp = await client.request(
                method, f"{url}{path}", json=json, timeout=timeout,
                headers=self._resolve_headers(headers, agent_id=agent_id),
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            logger.warning("HTTP %d from agent '%s' %s: %s", e.response.status_code, agent_id, path, e)
            return {"error": f"HTTP {e.response.status_code}", "status_code": e.response.status_code}
        except httpx.TimeoutException:
            logger.warning("Timeout reaching agent '%s' %s", agent_id, path)
            return {"error": f"Timeout after {timeout}s"}
        except httpx.ConnectError as e:
            logger.debug("Connection failed for agent '%s' %s: %s", agent_id, path, e)
            return {"error": f"Connection failed: {e}"}

    async def is_reachable(self, agent_id: str, timeout: int = 5) -> bool:
        # Deliberately does NOT call ``_ensure_running`` (plan §8 #24) —
        # health polls must never wake a hibernated agent. A hibernated
        # agent is health-unregistered anyway (belt and suspenders): the
        # health monitor never calls this for one in the first place.
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
        await self._ensure_running(agent_id)
        url = self._urls.get(agent_id)
        if not url:
            yield {"type": "error", "message": f"Agent '{agent_id}' not registered in transport"}
            return
        try:
            client = await self._get_client()
            async with client.stream(
                method, f"{url}{path}", json=json, timeout=timeout,
                headers=self._resolve_headers(headers, agent_id=agent_id),
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            yield json_module.loads(line[6:])
                        except json_module.JSONDecodeError:
                            yield {"raw": line[6:]}
                    elif line.startswith(":"):
                        # SSE comment. The agent emits ": keepalive" every
                        # ~15s while a tool runs silently (agent/server.py
                        # chat_stream). Receiving the line here already resets
                        # THIS hop's read-idle clock, but historically the
                        # comment was dropped — so a downstream streaming hop
                        # (dashboard -> browser) saw zero bytes during a long
                        # quiet tool call and its own idle-abort fired at 120s,
                        # cancelling the whole turn. Forward it as a liveness
                        # sentinel so each hop can reset its timer. It is a true
                        # end-to-end liveness signal: if the agent's loop wedges
                        # the keepalive stops and downstream correctly times out.
                        yield {"type": "keepalive"}
        except httpx.HTTPStatusError as e:
            logger.warning("Stream HTTP %d from agent '%s' %s", e.response.status_code, agent_id, path)
            yield {"type": "error", "message": f"HTTP {e.response.status_code}"}
        except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as e:
            logger.warning("Stream connection failed for agent '%s' %s: %s", agent_id, path, e)
            yield {"type": "error", "message": friendly_streaming_error(e)}

    def request_sync(
        self,
        agent_id: str,
        method: str,
        path: str,
        json: dict | None = None,
        timeout: int = 120,
        headers: dict[str, str] | None = None,
    ) -> dict:
        self._ensure_running_sync(agent_id)
        url = self._urls.get(agent_id)
        if not url:
            return {"error": f"Agent '{agent_id}' not registered in transport"}
        try:
            resp = httpx.request(
                method, f"{url}{path}", json=json, timeout=timeout,
                headers=self._resolve_headers(headers, agent_id=agent_id),
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            logger.warning("Sync HTTP %d from agent '%s' %s", e.response.status_code, agent_id, path)
            return {"error": f"HTTP {e.response.status_code}", "status_code": e.response.status_code}
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            logger.warning("Sync request failed for agent '%s' %s: %s", agent_id, path, e)
            return {"error": str(e)}

    def _ensure_running_sync(self, agent_id: str) -> None:
        """Best-effort cold-wake check for the sync request path.

        ``request_sync`` is documented as "for use in non-async contexts"
        (CLI REPL, callbacks) — there is normally no running loop to hop
        onto, so a fresh one is spun up via ``asyncio.run`` for the
        one-shot wake check. RESIDUAL: if this is somehow called from
        inside an already-running loop, the wake is skipped (logged, not
        raised) rather than crashing on ``asyncio.run()``'s "cannot be
        called from a running event loop" — a missed wake here degrades
        to the pre-existing "agent unreachable" behavior, never a hard
        failure. Every current call site (CLI REPL, the channel-manager
        status/reset callbacks) runs synchronously with no loop, so the
        residual is not believed to be reachable in practice.
        """
        fn = self._ensure_running_fn
        if fn is None or not agent_id:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            logger.debug(
                "request_sync ensure-running skipped for '%s': called from "
                "a running event loop",
                agent_id,
            )
            return
        try:
            asyncio.run(fn(agent_id))
        except Exception as e:
            logger.debug("request_sync ensure-running failed for '%s': %s", agent_id, e)


class SandboxTransport(Transport):
    """docker sandbox exec transport -- for agents running in microVMs.

    Reaches the agent's FastAPI server (listening on localhost:8400 inside
    the VM) by shelling out to ``docker sandbox exec ... curl``.
    """

    AGENT_PORT = 8400
    _AGENT_ID_RE = re.compile(AGENT_ID_RE_PATTERN)

    @classmethod
    def _validate_agent_id(cls, agent_id: str) -> None:
        """Reject agent_ids that could escape the sandbox container name."""
        if not cls._AGENT_ID_RE.match(agent_id):
            raise ValueError(f"Invalid agent_id format: {agent_id}")

    async def request(
        self,
        agent_id: str,
        method: str,
        path: str,
        json: dict | None = None,
        timeout: int = 120,
        headers: dict[str, str] | None = None,
    ) -> dict:
        self._validate_agent_id(agent_id)
        sandbox_name = f"openlegion_{agent_id}"
        url = f"http://localhost:{self.AGENT_PORT}{path}"
        cmd: list[str] = [
            "docker", "sandbox", "exec", sandbox_name, "--",
            "curl", "-s", "-X", method, url,
            "-H", "Content-Type: application/json",
        ]
        for hk, hv in self._resolve_headers(headers, agent_id=agent_id).items():
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
            if proc is not None and proc.returncode is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass  # Process already exited between returncode check and kill
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5)
                except (asyncio.TimeoutError, ProcessLookupError):
                    logger.warning("Failed to reap subprocess for '%s' after kill", agent_id)
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

        self._validate_agent_id(agent_id)
        sandbox_name = f"openlegion_{agent_id}"
        url = f"http://localhost:{self.AGENT_PORT}{path}"
        cmd: list[str] = [
            "docker", "sandbox", "exec", sandbox_name, "--",
            "curl", "-s", "-X", method, url,
            "-H", "Content-Type: application/json",
        ]
        for hk, hv in self._resolve_headers(headers, agent_id=agent_id).items():
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
