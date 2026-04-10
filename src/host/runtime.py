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
import os
import platform
import re
import secrets
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from src.shared.utils import setup_logging

logger = setup_logging("host.runtime")

# Docker container/volume names only allow [a-zA-Z0-9][a-zA-Z0-9_.-]
_DOCKER_NAME_RE = re.compile(r"[^a-zA-Z0-9_.-]")


def _docker_safe_name(agent_id: str) -> str:
    """Sanitize an agent ID for use in Docker container/volume names."""
    return _DOCKER_NAME_RE.sub("_", agent_id)


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
        self.extra_env: dict[str, str] = {}

    @abc.abstractmethod
    def start_agent(
        self,
        agent_id: str,
        role: str,
        skills_dir: str,
        system_prompt: str = "",
        model: str = "",
        mcp_servers: list[dict] | None = None,
        thinking: str = "",
        env_overrides: dict[str, str] | None = None,
    ) -> str:
        """Start an agent. Returns a URL or identifier for reaching it.

        ``env_overrides`` are per-agent environment variables that are merged
        on top of the shared ``extra_env`` dict for this call only, without
        mutating ``extra_env``.
        """

    @abc.abstractmethod
    def stop_agent(self, agent_id: str, *, remove_data: bool = False) -> None:
        """Stop and clean up an agent. When remove_data is True, also remove persistent storage."""

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
        thinking: str = "",
        env_overrides: dict[str, str] | None = None,
    ) -> str:
        """Spawn an ephemeral agent with a TTL for auto-cleanup."""
        url = self.start_agent(
            agent_id=agent_id, role=role, skills_dir="",
            system_prompt=system_prompt, model=model,
            mcp_servers=mcp_servers,
            thinking=thinking,
            env_overrides=env_overrides,
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

    @staticmethod
    def backend_name() -> str:
        return "unknown"


# ── Docker Container Backend ─────────────────────────────────


class DockerBackend(RuntimeBackend):
    """Runs agents in standard Docker containers (shared host kernel)."""

    BASE_IMAGE = "openlegion-agent:latest"
    BROWSER_IMAGE = "openlegion-browser:latest"

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
        self._port_lock = threading.Lock()
        self.browser_service_url: str | None = None
        self.browser_vnc_url: str | None = None
        self.browser_auth_token: str = ""
        self._browser_container = None
        # User-managed uploads directory.  Mounted read-only in every agent
        # container at /data/uploads and in the browser container at /app/uploads.
        # The browser service serves /app/uploads at GET /uploads/{path} so the
        # VNC browser can navigate to http://localhost:8500/uploads/<file>.
        self.uploads_dir = self.project_root / ".openlegion" / "uploads"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self._cleanup_stale()

        # User-defined bridge network for agent containers.
        # Provides DNS-based service discovery between containers and isolates
        # agents from the host network (NAT only).  Port publishing allows the
        # mesh host to reach agents.  Agents reach the mesh via
        # host.docker.internal.
        #
        # NOTE: internal=True would block all egress but also breaks port
        # publishing on Docker Desktop (macOS/Windows).  Since the mesh proxy
        # already controls API access and agents have no credentials, a
        # regular bridge provides sufficient isolation.
        self._network_name = "openlegion_agents"
        self._network = None
        if not use_host_network:
            self._network = self._ensure_agent_network()

    def _ensure_agent_network(self):
        """Get or create the bridge network for agent containers."""
        import docker
        try:
            network = self.client.networks.get(self._network_name)
            # If the network was previously created with internal=True (which
            # breaks port publishing on Docker Desktop), replace it.
            if (network.attrs or {}).get("Internal", False):
                try:
                    network.remove()
                except Exception:
                    logger.warning(
                        "Could not replace internal network '%s' — "
                        "port publishing may not work",
                        self._network_name,
                    )
                    return network
            else:
                return network
        except docker.errors.NotFound:
            pass
        return self.client.networks.create(
            self._network_name, driver="bridge",
        )

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
            logger.debug("Stale container cleanup failed: %s", e)

    def start_agent(
        self,
        agent_id: str,
        role: str,
        skills_dir: str,
        system_prompt: str = "",
        model: str = "",
        mcp_servers: list[dict] | None = None,
        thinking: str = "",
        env_overrides: dict[str, str] | None = None,
    ) -> str:
        import docker as _docker

        with self._port_lock:
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
            "MESH_AUTH_TOKEN": auth_token,
        }
        # Route system_prompt through INITIAL_INSTRUCTIONS for spawn compat
        if system_prompt:
            environment["INITIAL_INSTRUCTIONS"] = system_prompt
        if model:
            environment["LLM_MODEL"] = model
        if mcp_servers:
            environment["MCP_SERVERS"] = json.dumps(mcp_servers)
        if thinking:
            environment["THINKING"] = thinking
        environment.update(self.extra_env)
        if env_overrides:
            environment.update(env_overrides)
        if self.use_host_network:
            environment["AGENT_PORT"] = str(port)

        safe_name = _docker_safe_name(agent_id)
        volumes: dict[str, Any] = {
            f"openlegion_data_{safe_name}": {"bind": "/data", "mode": "rw"},
        }
        if skills_dir:
            # docker-py on Windows needs forward-slash or POSIX paths for bind mounts
            volumes[str(Path(skills_dir).as_posix() if platform.system() == "Windows" else skills_dir)] = {
                "bind": "/app/skills", "mode": "ro",
            }
        # Mount project-specific PROJECT.md (standalone agents get none)
        # Check env_overrides first (per-agent), then fall back to extra_env (system-wide)
        project_md_path = (env_overrides or {}).get(
            "PROJECT_MD_PATH", self.extra_env.get("PROJECT_MD_PATH", ""),
        )
        if project_md_path and Path(project_md_path).exists():
            host_path = project_md_path
            if platform.system() == "Windows":
                host_path = Path(project_md_path).as_posix()
            volumes[host_path] = {"bind": "/app/PROJECT.md", "mode": "ro"}
        elif project_md_path:
            logger.warning("Project file not found: %s", project_md_path)

        marketplace_dir = self.project_root / "skills" / "_marketplace"
        if marketplace_dir.is_dir():
            mp_path = str(marketplace_dir)
            if platform.system() == "Windows":
                mp_path = marketplace_dir.as_posix()
            volumes[mp_path] = {"bind": "/app/marketplace_skills", "mode": "ro"}

        # Read-only uploads: user-managed files agents can read but never write.
        uploads_path = str(self.uploads_dir.as_posix() if platform.system() == "Windows" else self.uploads_dir)
        volumes[uploads_path] = {"bind": "/data/uploads", "mode": "ro"}

        # Slim agent containers (no browser). 384MB / 0.15 CPU.
        # Agents are mostly I/O-bound (waiting on LLM APIs).
        # Browser ops are handled by the shared browser service container.
        # Operator gets reduced limits (128MB / 0.05 CPU) — it only does
        # LLM chat and mesh API calls, no browser/shell/file processing.
        is_operator = bool(env_overrides and env_overrides.get("ALLOWED_TOOLS"))
        run_kwargs: dict[str, Any] = {
            "detach": True,
            "name": f"openlegion_{safe_name}",
            "environment": environment,
            "volumes": volumes,
            "mem_limit": "128m" if is_operator else "384m",
            "cpu_quota": 5000 if is_operator else 15000,
            "security_opt": ["no-new-privileges"],
            "cap_drop": ["ALL"],
            "read_only": True,
            "tmpfs": {"/tmp": "size=100m,noexec,nosuid"},
            "pids_limit": 256,
        }

        if self.use_host_network:
            run_kwargs["network_mode"] = "host"
        else:
            run_kwargs["network"] = self._network_name
            run_kwargs["ports"] = {"8400/tcp": port}
            # On Linux, host.docker.internal requires explicit mapping
            if platform.system() == "Linux":
                run_kwargs["extra_hosts"] = {
                    "host.docker.internal": "host-gateway",
                }

        container_name = f"openlegion_{safe_name}"
        try:
            stale = self.client.containers.get(container_name)
            stale.remove(force=True)
        except _docker.errors.NotFound:
            pass

        container = self.client.containers.run(self.BASE_IMAGE, **run_kwargs)
        url = f"http://127.0.0.1:{port}"
        agent_info: dict[str, Any] = {
            "container": container,
            "url": url,
            "port": port,
            "role": role,
            "skills_dir": skills_dir,
            "model": model,
            "mcp_servers": mcp_servers,
            "thinking": thinking,
        }
        self.agents[agent_id] = agent_info
        logger.info(f"Started agent '{agent_id}' (role={role}) at {url}")
        return url

    def start_browser_service(self) -> None:
        """Start the shared browser service container."""
        import docker as _docker

        # Host network mode disables the SSRF egress filter (iptables rules
        # would mutate the host's network namespace instead of the container's).
        # That is a real and silent security regression, so we require a second
        # explicit opt-in from the operator to run the browser in host mode.
        _host_net_ack = os.environ.get("OPENLEGION_BROWSER_ALLOW_HOST_NETWORK", "").strip()
        if self.use_host_network and _host_net_ack not in ("1", "true", "yes"):
            raise RuntimeError(
                "Refusing to start the browser container in host network mode: "
                "SSRF egress filter cannot be installed when the container shares "
                "the host's network namespace. To proceed anyway (INSECURE — the "
                "browser will have unrestricted access to the host's private "
                "networks), set OPENLEGION_BROWSER_ALLOW_HOST_NETWORK=1. The "
                "recommended fix is to unset OPENLEGION_HOST_NETWORK so the "
                "browser runs on a bridge network with the filter active."
            )

        self.browser_auth_token = secrets.token_urlsafe(32)
        mesh_host = "127.0.0.1" if self.use_host_network else "host.docker.internal"

        # Scale browser container resources based on plan size.
        # Each Camoufox instance uses ~200-400 MB RAM.  We size the
        # container memory and max concurrent browsers to support as
        # many agents browsing simultaneously as the VPS can handle.
        # shm_size is for Firefox compositor IPC — too small causes VNC freezes.
        #
        #   Plan    Server         Agents  Mem   SHM   CPU   Max Browsers
        #   Basic   cax11  4GB 2c    1    2GB  512m  1.0c     1
        #   Growth  cax21  8GB 4c    5    4GB    1g  1.5c     5
        #   Pro     cax31 16GB 8c   15    8GB    2g  2.0c    10
        max_agents = int(os.environ.get("OPENLEGION_MAX_AGENTS", "0"))
        if max_agents <= 1:
            max_browsers, browser_mem, browser_shm, browser_cpu = 1, "2g", "512m", 100000
        elif max_agents <= 5:
            max_browsers, browser_mem, browser_shm, browser_cpu = max_agents, "4g", "1g", 150000
        else:
            max_browsers, browser_mem, browser_shm, browser_cpu = min(max_agents, 10), "8g", "2g", 200000

        # Override browser idle timeout from dashboard config
        idle_timeout_minutes = 30
        settings_path = self.project_root / "config" / "settings.json"
        if settings_path.exists():
            try:
                bsettings = json.loads(settings_path.read_text())
                if "browser_idle_timeout" in bsettings:
                    idle_timeout_minutes = int(bsettings["browser_idle_timeout"])
            except (json.JSONDecodeError, OSError, ValueError):
                pass

        environment = {
            "BROWSER_AUTH_TOKEN": self.browser_auth_token,
            "MESH_URL": f"http://{mesh_host}:{self.mesh_host_port}",
            "MAX_BROWSERS": str(max_browsers),
            "IDLE_TIMEOUT_MINUTES": str(idle_timeout_minutes),
        }

        for var in ("BROWSER_PROXY_URL", "BROWSER_PROXY_USER", "BROWSER_PROXY_PASS",
                    "BROWSER_EGRESS_ALLOWLIST", "BROWSER_EGRESS_DISABLE"):
            if os.environ.get(var):
                environment[var] = os.environ[var]

        # If the operator configured a proxy whose host is a literal private IP,
        # the egress filter will block the browser from reaching it unless the
        # proxy CIDR is explicitly allowlisted. Detect misconfiguration at
        # startup — BOTH the "no allowlist at all" case AND the "allowlist set
        # but does not cover the proxy IP" case — instead of surfacing as a
        # cryptic "browser cannot reach proxy" error after the container is
        # already running. Hostname-based proxies are left alone (resolving
        # them at startup would be brittle and racy).
        proxy_url = os.environ.get("BROWSER_PROXY_URL", "").strip()
        proxy_allowlist = os.environ.get("BROWSER_EGRESS_ALLOWLIST", "").strip()
        if proxy_url:
            try:
                import ipaddress
                from urllib.parse import urlparse
                parsed = urlparse(proxy_url if "://" in proxy_url else f"http://{proxy_url}")
                host = parsed.hostname or ""
                # Only act on IP literals — hostnames are intentionally untouched.
                try:
                    ip_obj = ipaddress.ip_address(host)
                except ValueError:
                    ip_obj = None
                _is_private = ip_obj is not None and (
                    ip_obj.is_private
                    or ip_obj.is_loopback
                    or ip_obj.is_link_local
                    or ip_obj.is_reserved
                )
                if _is_private:
                    # Parse the allowlist (if any) and verify the proxy IP is
                    # actually covered by at least one entry. Malformed entries
                    # are skipped silently here — the entrypoint will warn on
                    # them at container-start time.
                    covered = False
                    if proxy_allowlist:
                        for cidr_str in proxy_allowlist.split(","):
                            cidr_str = cidr_str.strip()
                            if not cidr_str:
                                continue
                            try:
                                network = ipaddress.ip_network(cidr_str, strict=False)
                            except ValueError:
                                continue
                            if ip_obj in network:
                                covered = True
                                break
                    if not covered:
                        if proxy_allowlist:
                            raise RuntimeError(
                                f"BROWSER_PROXY_URL host {host} is a private IP "
                                f"literal, but BROWSER_EGRESS_ALLOWLIST does not "
                                f"cover it (current value: {proxy_allowlist!r}). "
                                f"The browser container's egress filter will "
                                f"block connections to {host}. Add {host}/32 or "
                                f"a containing CIDR to BROWSER_EGRESS_ALLOWLIST."
                            )
                        raise RuntimeError(
                            f"BROWSER_PROXY_URL host {host} is a private IP "
                            f"literal, but BROWSER_EGRESS_ALLOWLIST is not set. "
                            f"The browser container's egress filter will block "
                            f"connections to {host}. Set "
                            f"BROWSER_EGRESS_ALLOWLIST={host}/32 (or the "
                            f"appropriate CIDR) to allow the proxy through."
                        )
            except RuntimeError:
                raise
            except Exception as e:
                logger.warning("Could not parse BROWSER_PROXY_URL for validation: %s", e)

        with self._port_lock:
            api_port = self._next_port
            self._next_port += 1
            vnc_port = self._next_port
            self._next_port += 1

        uploads_path = str(self.uploads_dir.as_posix() if platform.system() == "Windows" else self.uploads_dir)
        run_kwargs: dict[str, Any] = {
            "detach": True,
            "name": "openlegion_browser",
            "environment": environment,
            "volumes": {
                "openlegion_browser_data": {"bind": "/data", "mode": "rw"},
                uploads_path: {"bind": "/app/uploads", "mode": "ro"},
            },
            "mem_limit": browser_mem,
            "cpu_quota": browser_cpu,
            "shm_size": browser_shm,
            "security_opt": ["no-new-privileges"],
            # Drop Docker's default cap set. The browser container holds only the
            # capabilities it actually needs. The init phase (running as root via
            # the entrypoint) uses NET_ADMIN to install iptables rules, then calls
            # gosu(1) which needs SETUID + SETGID to drop to UID 1000 before
            # handing control to Firefox. After gosu, the long-running browser
            # process runs with no effective capabilities (non-root users do not
            # inherit caps), and no-new-privileges prevents re-acquisition via
            # setuid binaries or file capabilities.
            "cap_drop": ["ALL"],
        }

        if self.use_host_network:
            run_kwargs["network_mode"] = "host"
            environment["API_PORT"] = str(api_port)
            environment["VNC_PORT"] = str(vnc_port)
            # Host network mode shares the host's network namespace. Installing
            # iptables rules inside the container would mutate host networking,
            # so we explicitly disable the egress filter and warn loudly. SSRF
            # protection is off in this mode — use bridge networking in prod.
            environment["BROWSER_EGRESS_DISABLE"] = "1"
            logger.warning(
                "Browser container is running in host network mode — SSRF "
                "egress filter is DISABLED. Browser has unrestricted access "
                "to the host's private networks. Use bridge networking "
                "(use_host_network=False) for production deployments."
            )
        else:
            run_kwargs["ports"] = {"8500/tcp": api_port, "6080/tcp": vnc_port}
            if platform.system() == "Linux":
                run_kwargs["extra_hosts"] = {"host.docker.internal": "host-gateway"}
            # Bridge network mode: grant minimal caps needed by the entrypoint.
            # NET_ADMIN lets the root init phase install the iptables egress
            # filter; SETUID + SETGID let gosu(1) drop to the non-root browser
            # user before execing Firefox. All three are in the container's
            # bounding set but unreachable after gosu (UID 1000, no ambient).
            run_kwargs["cap_add"] = ["NET_ADMIN", "SETUID", "SETGID"]

        # Remove stale browser container
        try:
            stale = self.client.containers.get("openlegion_browser")
            stale.remove(force=True)
        except _docker.errors.NotFound:
            pass

        self._browser_container = self.client.containers.run(self.BROWSER_IMAGE, **run_kwargs)
        self.browser_service_url = f"http://127.0.0.1:{api_port}"

        # Wait for browser service API to be ready
        import httpx as _httpx
        browser_ready = False
        for attempt in range(15):
            try:
                resp = _httpx.get(
                    f"{self.browser_service_url}/browser/status",
                    headers={"Authorization": f"Bearer {self.browser_auth_token}"},
                    timeout=2,
                )
                if resp.status_code == 200:
                    browser_ready = True
                    break
            except Exception as e:
                logger.debug("Browser service not ready (attempt %d): %s", attempt + 1, e)
            time.sleep(1)

        if not browser_ready:
            logger.warning("Browser service API failed to become ready after 15 attempts")
            return

        # Verify KasmVNC is also reachable — it starts independently of the
        # FastAPI service and can fail even when the API is healthy.
        vnc_url = f"http://127.0.0.1:{vnc_port}"
        vnc_ready = False
        for attempt in range(10):
            try:
                resp = _httpx.get(f"{vnc_url}/index.html", timeout=2)
                if resp.status_code == 200:
                    vnc_ready = True
                    break
            except Exception as e:
                logger.debug("KasmVNC not ready (attempt %d): %s", attempt + 1, e)
            time.sleep(1)

        if not vnc_ready:
            logger.warning("KasmVNC failed to become reachable on port %d", vnc_port)
            return

        self.browser_vnc_url = f"http://127.0.0.1:{vnc_port}/index.html?autoconnect=true&path=&resize=scale"

        # Push saved browser settings (speed) so they survive container restarts
        try:
            _settings_path = Path("config/settings.json")
            if _settings_path.exists():
                _saved = json.loads(_settings_path.read_text())
                _speed = _saved.get("browser_speed")
                if _speed is not None:
                    _httpx.post(
                        f"{self.browser_service_url}/browser/settings",
                        json={"speed": _speed},
                        headers={"Authorization": f"Bearer {self.browser_auth_token}"},
                        timeout=5,
                    )
                    logger.info("Pushed browser speed=%.2f from saved settings", _speed)
        except Exception as e:
            logger.debug("Browser settings push skipped: %s", e)

        logger.info("Started browser service at %s (VNC: %s)", self.browser_service_url, self.browser_vnc_url)

    def stop_browser_service(self) -> None:
        """Stop the browser service container."""
        if self._browser_container:
            try:
                self._browser_container.stop(timeout=10)
                self._browser_container.remove()
                logger.info("Stopped browser service")
            except Exception as e:
                logger.warning("Error stopping browser service: %s", e)
            self._browser_container = None
            self.browser_service_url = None
            self.browser_vnc_url = None

    def stop_agent(self, agent_id: str, *, remove_data: bool = False) -> None:
        if agent_id in self.agents:
            safe_name = _docker_safe_name(agent_id)
            try:
                self.agents[agent_id]["container"].stop(timeout=10)
                self.agents[agent_id]["container"].remove()
                logger.info(f"Stopped agent '{agent_id}'")
            except Exception as e:
                logger.warning(f"Error stopping agent '{agent_id}': {e}")
            if remove_data:
                try:
                    vol = self.client.volumes.get(f"openlegion_data_{safe_name}")
                    vol.remove(force=True)
                    logger.info(f"Removed data volume for agent '{agent_id}'")
                except Exception as e:
                    logger.debug(f"Volume cleanup for '{agent_id}': {e}")
            del self.agents[agent_id]
            if hasattr(self, "auth_tokens"):
                self.auth_tokens.pop(agent_id, None)

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
        loop = asyncio.get_running_loop()
        async with httpx.AsyncClient(timeout=3) as client:
            while time.time() - start < timeout:
                # health_check does blocking Docker API calls — run off event loop
                is_healthy = await loop.run_in_executor(None, self.health_check, agent_id)
                if not is_healthy:
                    logs = await loop.run_in_executor(None, self.get_logs, agent_id, 20)
                    logger.warning(
                        f"Agent '{agent_id}' container exited during startup. "
                        f"Logs:\n{logs}"
                    )
                    return False
                try:
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

    def stop_all(self) -> None:
        super().stop_all()
        if self._network is not None:
            try:
                self._network.remove()
            except Exception as e:
                logger.debug("Network removal failed: %s", e)
            self._network = None


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
        system_prompt: str = "",
        model: str = "",
        mcp_servers: list[dict] | None = None,
        thinking: str = "",
        env_overrides: dict[str, str] | None = None,
    ) -> Path:
        """Create the per-agent host directory that will sync into the sandbox."""
        ws = self._workspace_root / agent_id
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "data" / "workspace").mkdir(parents=True, exist_ok=True)

        # Copy project-specific PROJECT.md (standalone agents get none)
        # Check env_overrides first (per-agent), then fall back to extra_env (system-wide)
        project_md_path = (env_overrides or {}).get(
            "PROJECT_MD_PATH", self.extra_env.get("PROJECT_MD_PATH", ""),
        )
        if project_md_path and Path(project_md_path).exists():
            shutil.copy2(project_md_path, ws / "PROJECT.md")
        elif project_md_path:
            logger.warning("Project file not found: %s", project_md_path)

        # Copy skills
        skills_dest = ws / "skills"
        if skills_dir and Path(skills_dir).is_dir():
            if skills_dest.exists():
                shutil.rmtree(skills_dest)
            shutil.copytree(skills_dir, skills_dest, symlinks=True)
        else:
            skills_dest.mkdir(exist_ok=True)

        # Copy marketplace skills
        marketplace_src = self.project_root / "skills" / "_marketplace"
        marketplace_dest = ws / "marketplace_skills"
        if marketplace_src.is_dir():
            if marketplace_dest.exists():
                shutil.rmtree(marketplace_dest)
            shutil.copytree(marketplace_src, marketplace_dest, symlinks=True)
        else:
            marketplace_dest.mkdir(exist_ok=True)

        # Generate per-agent auth token for mesh request verification
        auth_token = secrets.token_urlsafe(32)
        self.auth_tokens[agent_id] = auth_token

        # Write agent env config
        env_cfg = {
            "AGENT_ID": agent_id,
            "AGENT_ROLE": role,
            "MESH_URL": f"http://host.docker.internal:{self.mesh_host_port}",
            "SKILLS_DIR": "/app/skills",
            "AGENT_PORT": str(self.AGENT_PORT),
            "MESH_AUTH_TOKEN": auth_token,
        }
        # Route system_prompt through INITIAL_INSTRUCTIONS for spawn compat
        if system_prompt:
            env_cfg["INITIAL_INSTRUCTIONS"] = system_prompt
        if model:
            env_cfg["LLM_MODEL"] = model
        if mcp_servers:
            env_cfg["MCP_SERVERS"] = json.dumps(mcp_servers)
        if thinking:
            env_cfg["THINKING"] = thinking
        env_cfg.update(self.extra_env)
        if env_overrides:
            env_cfg.update(env_overrides)

        def _sanitize_env_value(v: str) -> str:
            """Sanitize a value for Docker --env-file format.

            Docker reads values literally after '=' — no quote stripping
            or shell expansion.  Only newlines need escaping since each
            line is a separate entry.
            """
            return v.replace("\r", "").replace("\n", "\\n")

        env_file = ws / ".agent.env"
        env_file.write_text(
            "\n".join(f"{k}={_sanitize_env_value(v)}" for k, v in env_cfg.items()) + "\n"
        )
        env_file.chmod(0o600)
        return ws

    def start_agent(
        self,
        agent_id: str,
        role: str,
        skills_dir: str,
        system_prompt: str = "",
        model: str = "",
        mcp_servers: list[dict] | None = None,
        thinking: str = "",
        env_overrides: dict[str, str] | None = None,
    ) -> str:
        sandbox_name = f"openlegion_{_docker_safe_name(agent_id)}"
        ws = self._prepare_workspace(
            agent_id, role, skills_dir, system_prompt, model,
            mcp_servers=mcp_servers, thinking=thinking,
            env_overrides=env_overrides,
        )

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
            "model": model,
            "mcp_servers": mcp_servers,
            "thinking": thinking,
        }
        logger.info(f"Started agent '{agent_id}' in sandbox '{sandbox_name}'")
        return url

    def stop_agent(self, agent_id: str, *, remove_data: bool = False) -> None:
        if agent_id not in self.agents:
            return
        sandbox_name = self.agents[agent_id]["sandbox_name"]
        workspace = self.agents[agent_id].get("workspace")
        try:
            subprocess.run(
                ["docker", "sandbox", "rm", "-f", sandbox_name],
                capture_output=True, timeout=15,
            )
            logger.info(f"Removed sandbox '{sandbox_name}'")
        except Exception as e:
            logger.warning(f"Error removing sandbox '{sandbox_name}': {e}")
        if remove_data and workspace:
            try:
                shutil.rmtree(workspace)
                logger.info(f"Removed workspace for agent '{agent_id}'")
            except Exception as e:
                logger.debug(f"Workspace cleanup for '{agent_id}': {e}")
        del self.agents[agent_id]
        if hasattr(self, "auth_tokens"):
            self.auth_tokens.pop(agent_id, None)

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
        loop = asyncio.get_running_loop()
        while time.time() - start < timeout:
            # health_check does blocking subprocess.run — run off event loop
            is_healthy = await loop.run_in_executor(None, self.health_check, agent_id)
            if not is_healthy:
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


def _should_use_host_network() -> bool:
    """Check if host networking is explicitly opted in via env var.

    Defaults to False (bridge networking) for security — bridge mode prevents
    agents from accessing the host network, mesh, and cloud metadata endpoints.
    Set OPENLEGION_HOST_NETWORK=1 to opt in to --network=host (e.g. for
    debugging or legacy setups that require it).
    """
    enabled = os.environ.get("OPENLEGION_HOST_NETWORK", "").strip()
    if enabled in ("1", "true", "yes"):
        logger.warning(
            "Host networking enabled via OPENLEGION_HOST_NETWORK — "
            "agents will have full access to the host network. "
            "This is insecure in production."
        )
        return True
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
    use_host_net = _should_use_host_network()
    logger.info("Using Docker container isolation (host_network=%s)", use_host_net)
    return DockerBackend(
        mesh_host_port=mesh_host_port,
        use_host_network=use_host_net,
        project_root=project_root,
    )
