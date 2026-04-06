"""RuntimeContext: manages the full OpenLegion runtime lifecycle."""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import click

from src.cli.channels import ChannelManager
from src.cli.config import (
    ENV_FILE,
    PROJECT_ROOT,
    PROJECTS_DIR,
    _check_docker_running,
    _ensure_docker_image,
    _load_config,
)
from src.cli.formatting import echo_fail, echo_header, echo_ok
from src.cli.proxy import build_proxy_env_vars, resolve_agent_proxy
from src.shared.types import RESERVED_AGENT_IDS

logger = logging.getLogger("cli")

# Provider prefix → default embedding model.  Providers without embedding
# APIs (Anthropic, xAI, Groq, …) map to "none" which disables vector search.
# Only map providers whose embedding models output 1536-dim vectors,
# matching EMBEDDING_DIM in memory.py.  Other providers default to "none".
_PROVIDER_EMBEDDING_DEFAULTS: list[tuple[str, str]] = [
    ("openai/",  "text-embedding-3-small"),
    ("gpt-",     "text-embedding-3-small"),
    ("o1",       "text-embedding-3-small"),
    ("o3",       "text-embedding-3-small"),
    ("o4",       "text-embedding-3-small"),
]


def _default_embedding_model(llm_model: str) -> str:
    """Pick a sensible embedding model default based on the LLM provider."""
    lower = llm_model.lower()
    for prefix, embed_model in _PROVIDER_EMBEDDING_DEFAULTS:
        if lower.startswith(prefix):
            return embed_model
    return "none"


class RuntimeContext:
    """Manages the full OpenLegion runtime lifecycle."""

    def __init__(self, config_path: str, use_sandbox: bool = False, port_override: int | None = None):
        self.cfg = _load_config(Path(config_path))
        if port_override is not None:
            self.cfg["mesh"]["port"] = port_override
        self.use_sandbox = use_sandbox
        self.runtime = None
        self.transport = None
        self.blackboard = None
        self.pubsub = None
        self.permissions = None
        self.cost_tracker = None
        self.credential_vault = None
        self.router = None
        self.health_monitor = None
        self.cron_scheduler = None
        self.lane_manager = None
        self.channel_manager = None
        self.event_bus = None
        self.trace_store = None
        self.agent_urls: dict[str, str] = {}
        self._dispatch_loop = None
        self._server = None
        self._active_channels: list = []
        self._start_time: float | None = None
        self._agent_results: list = []
        self._cron_job_count: int = 0

    def start(self) -> None:
        """Initialize and start all components. Called once."""
        self._start_time = time.time()
        # Match detached mode behavior so relative paths (pubsub.db, skills/, config/)
        # resolve under the OpenLegion project root even when launched elsewhere.
        os.chdir(PROJECT_ROOT)
        self._validate_prereqs()
        self._select_backend()
        self._create_components()
        self._start_browser_service()
        self._start_agents()
        self._setup_dispatch()
        self._create_cron_scheduler()
        self._start_mesh_server()
        self._wait_for_readiness()
        self._start_background()
        self._start_channels()
        self._print_ready()

    def shutdown(self) -> None:
        """Tear down all components in reverse order."""
        click.echo("  Stopping OpenLegion...", nl=False)
        if self.channel_manager:
            self.channel_manager.stop_all()
        if self.health_monitor:
            self.health_monitor.stop()
        if self.cron_scheduler:
            self.cron_scheduler.stop()
        if self.runtime:
            self.runtime.stop_all()
            if hasattr(self.runtime, 'stop_browser_service'):
                self.runtime.stop_browser_service()
        if self.cost_tracker:
            self.cost_tracker.close()
        if self.trace_store:
            self.trace_store.close()
        if self.pubsub:
            self.pubsub.close()
        if self.blackboard:
            self.blackboard.close()

        # Close shared httpx clients on the dispatch loop — close all
        # concurrently so one slow close doesn't block the others.
        if self._dispatch_loop:
            async def _close_clients():
                async def _close_one(name, closeable):
                    if closeable is None or not hasattr(closeable, 'close'):
                        return
                    try:
                        result = closeable.close()
                        if hasattr(result, '__await__'):
                            await asyncio.wait_for(result, timeout=3)
                    except Exception as e:
                        logger.debug("Error closing %s: %s", name, e)
                closeables = {
                    "transport": self.transport,
                    "router": self.router,
                    "credential_vault": self.credential_vault,
                }
                await asyncio.gather(
                    *(_close_one(n, c) for n, c in closeables.items()),
                    return_exceptions=True,
                )
            try:
                future = asyncio.run_coroutine_threadsafe(_close_clients(), self._dispatch_loop)
                future.result(timeout=10)
            except Exception as e:
                logger.debug("Shutdown cleanup error: %s", e)
            self._dispatch_loop.call_soon_threadsafe(self._dispatch_loop.stop)

        if self._server:
            self._server.should_exit = True

        click.echo(" done.")

    def dispatch(
        self, agent: str, message: str, mode: str = "followup", trace_id: str | None = None,
    ) -> str:
        """Thread-safe synchronous message dispatch.

        Schedules the coroutine on the dedicated dispatch loop and blocks
        until the result is ready.  For async callers, use async_dispatch().
        """
        future = asyncio.run_coroutine_threadsafe(
            self.lane_manager.enqueue(agent, message, mode=mode, trace_id=trace_id),
            self._dispatch_loop,
        )
        return future.result()

    async def async_dispatch(
        self, agent: str, message: str, mode: str = "followup", trace_id: str | None = None,
    ) -> str:
        """Async dispatch: schedules onto the dedicated dispatch loop."""
        future = asyncio.run_coroutine_threadsafe(
            self.lane_manager.enqueue(agent, message, mode=mode, trace_id=trace_id),
            self._dispatch_loop,
        )
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is not None:
            return await running_loop.run_in_executor(None, future.result)
        return future.result()

    @property
    def agents(self) -> dict[str, str]:
        return self.agent_urls

    @property
    def dispatch_loop(self):
        return self._dispatch_loop

    # ── Private lifecycle steps ─────────────────────────────

    def _validate_prereqs(self) -> None:
        if not _check_docker_running():
            click.echo("Docker is not running. Please start Docker first.", err=True)
            sys.exit(1)
        # Warn about missing credentials (non-fatal — setup wizard handles it)
        if ENV_FILE.exists():
            env_content = ENV_FILE.read_text()
            if "OPENLEGION_SYSTEM_" not in env_content and "OPENLEGION_CRED_" not in env_content:
                click.echo("Warning: No API credentials found in .env", err=True)
        else:
            click.echo("Warning: No .env file found", err=True)

    def _select_backend(self) -> None:
        from src.host.runtime import SandboxBackend, select_backend
        from src.host.transport import HttpTransport, SandboxTransport

        mesh_port = self.cfg["mesh"]["port"]
        self.runtime = select_backend(
            mesh_host_port=mesh_port, project_root=str(PROJECT_ROOT),
            use_sandbox=self.use_sandbox,
        )
        self._is_sandbox = isinstance(self.runtime, SandboxBackend)
        self._backend_label = self.runtime.backend_name()

        if self._is_sandbox:
            self.transport = SandboxTransport()
        else:
            self.transport = HttpTransport()
            _ensure_docker_image()

    def _create_components(self) -> None:
        from src.cli.config import _ensure_all_agent_permissions
        from src.dashboard.events import EventBus
        from src.host.costs import CostTracker
        from src.host.credentials import CredentialVault
        from src.host.mesh import Blackboard, MessageRouter, PubSub
        from src.host.permissions import PermissionMatrix
        from src.host.traces import TraceStore

        # Backfill permissions for agents missing from permissions.json
        _ensure_all_agent_permissions()

        # Ensure collaborative permissions are up to date before loading
        if self.cfg.get("collaboration", False):
            from src.cli.config import _set_collaborative_permissions
            _set_collaborative_permissions()

        self.event_bus = EventBus()
        self.trace_store = TraceStore()
        self.blackboard = Blackboard(event_bus=self.event_bus)
        self.pubsub = PubSub(db_path="pubsub.db")
        self.permissions = PermissionMatrix()
        self.cost_tracker = CostTracker()
        failover_config = self.cfg.get("llm", {}).get("failover", {})
        self.credential_vault = CredentialVault(
            cost_tracker=self.cost_tracker,
            failover_config=failover_config or None,
        )
        self.router = MessageRouter(
            self.permissions, self.agent_urls,
            trace_store=self.trace_store,
            agent_projects=self.cfg.get("_agent_projects", {}),
        )
        # Create HealthMonitor early so the dashboard router can reference it.
        # Only register() is called here; start() happens in _start_background().
        from src.host.health import HealthMonitor

        self.health_monitor = HealthMonitor(
            runtime=self.runtime, transport=self.transport, router=self.router,
            event_bus=self.event_bus,
            blackboard=self.blackboard,
        )

    def _start_browser_service(self) -> None:
        """Start the shared browser service container."""
        from src.host.runtime import DockerBackend
        if isinstance(self.runtime, DockerBackend):
            try:
                self.runtime.start_browser_service()
            except Exception as e:
                logger.warning("Failed to start browser service: %s", e)

    def _start_agents(self) -> None:
        from src.cli.config import _OPERATOR_AGENT_ID, _OPERATOR_ALLOWED_TOOLS, _ensure_operator_agent
        from src.host.runtime import DockerBackend, SandboxBackend
        from src.host.transport import HttpTransport

        # Auto-create operator if it doesn't exist
        default_model = self.cfg.get("llm", {}).get("default_model", "openai/gpt-4o-mini")
        _ensure_operator_agent(default_model=default_model)
        # Reload permissions so the operator's allowed_apis (e.g. "llm") are
        # present in the in-memory PermissionMatrix — without this the operator
        # gets 403 on /mesh/api because its permissions were written to disk
        # after PermissionMatrix was constructed in __init__.
        self.permissions.reload()
        # Reload config after possible operator creation
        self.cfg = _load_config()
        agents_cfg = self.cfg.get("agents", {})

        # Reorder agents so operator starts first
        if _OPERATOR_AGENT_ID in agents_cfg:
            ordered = {_OPERATOR_AGENT_ID: agents_cfg[_OPERATOR_AGENT_ID]}
            ordered.update({k: v for k, v in agents_cfg.items() if k != _OPERATOR_AGENT_ID})
            agents_cfg = ordered

        embedding_model = self.cfg.get("llm", {}).get(
            "embedding_model", _default_embedding_model(default_model),
        )
        mesh_port = self.cfg["mesh"]["port"]
        agent_projects = self.cfg.get("_agent_projects", {})

        # Respect plan limits on startup — only start up to max_agents.
        # Prevents OOM on downsized servers after a plan downgrade.
        # Operator is excluded from the count — it's always allowed.
        max_agents = int(os.environ.get("OPENLEGION_MAX_AGENTS", "0"))
        non_operator = {k: v for k, v in agents_cfg.items() if k != _OPERATOR_AGENT_ID}
        if max_agents > 0 and len(non_operator) > max_agents:
            logger.warning(
                "Agent limit is %d but %d agents configured — only starting first %d",
                max_agents, len(non_operator), max_agents,
            )
            trimmed = dict(list(non_operator.items())[:max_agents])
            if _OPERATOR_AGENT_ID in agents_cfg:
                trimmed[_OPERATOR_AGENT_ID] = agents_cfg[_OPERATOR_AGENT_ID]
            agents_cfg = trimmed

        self.runtime.extra_env["EMBEDDING_MODEL"] = embedding_model

        # Inject dashboard system settings as env vars for agent containers
        settings_path = Path("config/settings.json")
        if settings_path.exists():
            try:
                import json
                sys_settings = json.loads(settings_path.read_text())
                for env_key, cfg_key in {
                    "OPENLEGION_MAX_ITERATIONS": "max_iterations",
                    "OPENLEGION_CHAT_MAX_TOOL_ROUNDS": "chat_max_tool_rounds",
                    "OPENLEGION_CHAT_MAX_TOTAL_ROUNDS": "chat_max_total_rounds",
                    "OPENLEGION_TOOL_TIMEOUT": "tool_timeout",
                }.items():
                    if cfg_key in sys_settings:
                        self.runtime.extra_env[env_key] = str(sys_settings[cfg_key])
            except (ValueError, OSError):
                pass

        for agent_id, agent_cfg in agents_cfg.items():
            if agent_id in RESERVED_AGENT_IDS and agent_id != _OPERATOR_AGENT_ID:
                raise click.ClickException(
                    f"Agent ID '{agent_id}' is reserved for internal use"
                )
            budget = agent_cfg.get("budget", {})
            if budget:
                self.cost_tracker.set_budget(
                    agent_id,
                    daily_usd=budget.get("daily_usd", 10.0),
                    monthly_usd=budget.get("monthly_usd", 200.0),
                )
            skills_dir = os.path.abspath(agent_cfg.get("skills_dir", ""))
            agent_model = agent_cfg.get("model", default_model)
            agent_mcp_servers = agent_cfg.get("mcp_servers") or None
            agent_thinking = agent_cfg.get("thinking", "")

            # Build per-agent env overrides (no shared extra_env mutation)
            agent_env: dict[str, str] = {}
            initial_instructions = agent_cfg.get("initial_instructions", "")
            if initial_instructions:
                agent_env["INITIAL_INSTRUCTIONS"] = initial_instructions
            initial_soul = agent_cfg.get("initial_soul", "")
            if initial_soul:
                agent_env["INITIAL_SOUL"] = initial_soul
            initial_heartbeat = agent_cfg.get("initial_heartbeat", "")
            if initial_heartbeat:
                agent_env["INITIAL_HEARTBEAT"] = initial_heartbeat
            if agent_id == _OPERATOR_AGENT_ID:
                agent_env["ALLOWED_TOOLS"] = ",".join(_OPERATOR_ALLOWED_TOOLS)

            # Project env vars
            project_name = agent_projects.get(agent_id)
            if project_name:
                project_md = PROJECTS_DIR / project_name / "project.md"
                agent_env["PROJECT_MD_PATH"] = str(project_md)
                agent_env["PROJECT_NAME"] = project_name

            # Proxy env injection
            proxy_url = resolve_agent_proxy(
                agent_id,
                self.cfg.get("agents", {}),
                self.cfg.get("network", {}),
            )
            proxy_env = build_proxy_env_vars(
                proxy_url,
                no_proxy_user=self.cfg.get("network", {}).get("no_proxy", ""),
            )
            self.runtime.extra_env.update(proxy_env)

            try:
                url = self.runtime.start_agent(
                    agent_id=agent_id,
                    role=agent_cfg["role"],
                    skills_dir=skills_dir,
                    model=agent_model,
                    mcp_servers=agent_mcp_servers,
                    thinking=agent_thinking,
                    env_overrides=agent_env,
                )
            except (subprocess.TimeoutExpired, RuntimeError) as exc:
                if isinstance(self.runtime, SandboxBackend):
                    click.echo(
                        f"\n  Sandbox failed for '{agent_id}': {exc}\n"
                        "  Falling back to Docker container isolation...\n",
                        err=True,
                    )
                    saved_extra_env = self.runtime.extra_env
                    self.runtime.stop_all()
                    from src.host.runtime import _should_use_host_network
                    self.runtime = DockerBackend(
                        mesh_host_port=mesh_port,
                        use_host_network=_should_use_host_network(),
                        project_root=str(PROJECT_ROOT),
                    )
                    self.runtime.extra_env = saved_extra_env
                    self.transport = HttpTransport()
                    _ensure_docker_image()
                    url = self.runtime.start_agent(
                        agent_id=agent_id,
                        role=agent_cfg["role"],
                        skills_dir=skills_dir,
                        model=agent_model,
                        mcp_servers=agent_mcp_servers,
                        thinking=agent_thinking,
                        env_overrides=agent_env,
                    )
                else:
                    raise
            finally:
                # Clean up per-agent env vars so they don't leak to the next agent
                self.runtime.extra_env.pop("INITIAL_INSTRUCTIONS", None)
                self.runtime.extra_env.pop("INITIAL_SOUL", None)
                self.runtime.extra_env.pop("INITIAL_HEARTBEAT", None)
                self.runtime.extra_env.pop("PROJECT_MD_PATH", None)
                self.runtime.extra_env.pop("PROJECT_NAME", None)
                self.runtime.extra_env.pop("HTTP_PROXY", None)
                self.runtime.extra_env.pop("HTTPS_PROXY", None)
                self.runtime.extra_env.pop("NO_PROXY", None)
            self.router.register_agent(agent_id, url, role=agent_cfg.get("role", ""))
            if isinstance(self.transport, HttpTransport):
                self.transport.register(agent_id, url)
            if self.health_monitor:
                self.health_monitor.register(agent_id)

    def _setup_dispatch(self) -> None:
        from src.host.lanes import LaneManager

        async def _direct_dispatch(agent_name: str, message: str) -> str:
            from src.shared.trace import current_trace_id

            tid = current_trace_id.get()
            if tid and self.trace_store:
                self.trace_store.record(
                    trace_id=tid, source="dispatch", agent=agent_name,
                    event_type="chat", detail=message[:200],
                    meta={"message_length": len(message)},
                )
            import time as _time
            t0 = _time.time()
            try:
                result = await self.transport.request(
                    agent_name, "POST", "/chat", json={"message": message},
                )
                response = result.get("response", "(no response)")
                duration_ms = int((_time.time() - t0) * 1000)
                if tid and self.trace_store:
                    self.trace_store.record(
                        trace_id=tid, source="dispatch", agent=agent_name,
                        event_type="chat_response", duration_ms=duration_ms,
                        status="ok",
                        meta={"response_length": len(response),
                              "response_preview": response[:200]},
                    )
                if self.event_bus:
                    self.event_bus.emit("message_sent", agent=agent_name,
                        data={"message": message[:200], "response_length": len(response),
                              "source": "dispatch"})
                return response
            except Exception as e:
                duration_ms = int((_time.time() - t0) * 1000)
                if tid and self.trace_store:
                    self.trace_store.record(
                        trace_id=tid, source="dispatch", agent=agent_name,
                        event_type="chat_response", duration_ms=duration_ms,
                        status="error", error=str(e),
                    )
                return f"Error: {e}"

        async def _direct_steer(agent_name: str, message: str) -> dict:
            try:
                return await self.transport.request(
                    agent_name, "POST", "/chat/steer", json={"message": message},
                )
            except Exception as e:
                return {"injected": False, "error": str(e)}

        self.lane_manager = LaneManager(
            dispatch_fn=_direct_dispatch, steer_fn=_direct_steer,
            trace_store=self.trace_store,
        )

        self._dispatch_loop = asyncio.new_event_loop()

        def _run_dispatch_loop():
            asyncio.set_event_loop(self._dispatch_loop)
            self._dispatch_loop.run_forever()

        _dispatch_thread = threading.Thread(target=_run_dispatch_loop, daemon=True)
        _dispatch_thread.start()

    async def _handle_notify(self, agent_name: str, message: str) -> None:
        """Push an agent notification to REPL and all active channels."""
        notification = f"[{agent_name}] {message}"
        sys.stdout.write(f"\n{notification}\n")
        sys.stdout.flush()
        if self._active_channels:
            await asyncio.gather(*(
                ch.send_notification(notification)
                for ch in self._active_channels
            ), return_exceptions=True)

    def _start_mesh_server(self) -> None:
        import uvicorn

        from src.host.server import create_mesh_app
        from src.host.webhooks import WebhookManager

        mesh_port = self.cfg["mesh"]["port"]

        webhook_manager = WebhookManager(dispatch_fn=self.async_dispatch)

        # Wallet signing service (only if master seed is configured).
        # Wrapped in a single-item list so both mesh and dashboard closures
        # share a mutable reference — allows hot-loading after dashboard init.
        wallet_service = None
        if os.environ.get("OPENLEGION_SYSTEM_WALLET_MASTER_SEED"):
            from src.host.wallet import WalletService

            wallet_service = WalletService(event_bus=self.event_bus)
            logger.info(
                "Wallet service initialized (%d chains configured)",
                len(wallet_service.chains),
            )
        wallet_ref = [wallet_service]
        self._wallet_ref = wallet_ref

        from src.host.api_keys import ApiKeyManager
        self._api_key_manager = ApiKeyManager()

        app = create_mesh_app(
            self.blackboard, self.pubsub, self.router, self.permissions,
            self.credential_vault, self.cron_scheduler, self.runtime,
            self.transport,
            auth_tokens=self.runtime.auth_tokens,
            trace_store=self.trace_store,
            event_bus=self.event_bus,
            health_monitor=self.health_monitor,
            cost_tracker=self.cost_tracker,
            notify_fn=self._handle_notify,
            agent_projects=self.cfg.get("_agent_projects", {}),
            lane_manager=self.lane_manager,
            dispatch_loop=self._dispatch_loop,
            wallet_service_ref=wallet_ref,
            api_key_manager=self._api_key_manager,
            cfg=self.cfg,
        )
        app.include_router(webhook_manager.create_router())
        self.health_monitor._cleanup_agent = app.cleanup_agent  # type: ignore[attr-defined]

        self._init_channel_manager()

        from src.dashboard.server import create_dashboard_router, create_spa_catchall_router

        dashboard_router = create_dashboard_router(
            blackboard=self.blackboard,
            health_monitor=self.health_monitor,
            cost_tracker=self.cost_tracker,
            trace_store=self.trace_store,
            event_bus=self.event_bus,
            agent_registry=self.router.agent_registry,
            mesh_port=mesh_port,
            lane_manager=self.lane_manager,
            cron_scheduler=self.cron_scheduler,
            pubsub=self.pubsub,
            permissions=self.permissions,
            credential_vault=self.credential_vault,
            transport=self.transport,
            runtime=self.runtime,
            router=self.router,
            webhook_manager=webhook_manager,
            channel_manager=self.channel_manager,
            wallet_service_ref=wallet_ref,
            api_key_manager=self._api_key_manager,
        )
        app.include_router(dashboard_router)
        app.include_router(create_spa_catchall_router())  # Must be last — SPA deep linking
        self._app = app

        server_config = uvicorn.Config(app, host="0.0.0.0", port=mesh_port, log_level="warning")
        self._server = uvicorn.Server(server_config)
        mesh_thread = threading.Thread(target=self._server.run, daemon=True)
        mesh_thread.start()

    def _wait_for_readiness(self) -> None:
        import httpx

        mesh_port = self.cfg["mesh"]["port"]

        # Wait for mesh
        mesh_ready = False
        for _ in range(30):
            try:
                httpx.get(f"http://localhost:{mesh_port}/mesh/agents", timeout=1)
                mesh_ready = True
                break
            except Exception as e:
                logger.debug("Mesh not ready yet: %s", e)
                time.sleep(0.5)

        if not mesh_ready:
            echo_fail(
                f"Mesh server failed to start on port {mesh_port}. "
                f"Port may be in use. Try: openlegion stop"
            )
            self.runtime.stop_all()
            sys.exit(1)

        # Wait for agents (results displayed later in _print_ready)
        agents_cfg = self.cfg.get("agents", {})
        if agents_cfg:
            async def _wait_all_agents():
                async def _wait_one(aid):
                    ready = await self.runtime.wait_for_agent(aid, timeout=60)
                    return aid, ready
                return await asyncio.gather(*[_wait_one(aid) for aid in agents_cfg])

            self._agent_results = asyncio.run(_wait_all_agents())

    def _create_cron_scheduler(self) -> None:
        from src.host.cron import CronScheduler

        async def cron_dispatch(agent_name: str, message: str) -> str:
            from src.shared.trace import new_trace_id
            result = await self.async_dispatch(agent_name, message, trace_id=new_trace_id())
            return result

        async def fetch_heartbeat_context(agent_name: str) -> dict:
            try:
                return await self.transport.request(
                    agent_name, "GET", "/heartbeat-context", timeout=10,
                )
            except Exception as e:
                logger.debug("Failed to fetch heartbeat context for '%s': %s", agent_name, e)
                return {}

        async def invoke_tool(agent_name: str, tool_name: str, params: dict) -> dict:
            try:
                return await self.transport.request(
                    agent_name, "POST", "/invoke",
                    json={"tool": tool_name, "params": params},
                    timeout=30,
                )
            except Exception as e:
                logger.warning("Failed to invoke tool '%s' on '%s': %s", tool_name, agent_name, e)
                return {"error": str(e)}

        async def heartbeat_dispatch(agent_name: str, message: str) -> dict:
            """Dispatch heartbeat via dedicated /heartbeat endpoint."""
            try:
                return await self.transport.request(
                    agent_name, "POST", "/heartbeat",
                    json={"message": message},
                    timeout=120,
                )
            except Exception as e:
                logger.warning("Heartbeat dispatch failed for '%s': %s", agent_name, e)
                return {"response": f"Error: {e}", "outcome": "error", "skipped": False}

        self.cron_scheduler = CronScheduler(
            dispatch_fn=cron_dispatch,
            invoke_fn=invoke_tool,
            blackboard=self.blackboard,
            trace_store=self.trace_store,
            context_fn=fetch_heartbeat_context,
            heartbeat_dispatch_fn=heartbeat_dispatch,
            event_bus=self.event_bus,
        )
        self._cron_job_count = len(self.cron_scheduler.jobs)

    def _reconcile_heartbeats(self) -> None:
        """Ensure every agent in config has a heartbeat cron job."""
        if not self.cron_scheduler:
            return
        from src.cli.config import _OPERATOR_AGENT_ID
        from src.host.cron import CronScheduler
        agents_cfg = self.cfg.get("agents", {})
        schedule = self.cfg.get("mesh", {}).get(
            "heartbeat_schedule", CronScheduler.DEFAULT_HEARTBEAT_SCHEDULE,
        )
        for agent_id in agents_cfg:
            agent_schedule = "every 1h" if agent_id == _OPERATOR_AGENT_ID else schedule
            self.cron_scheduler.ensure_heartbeat(agent_id, agent_schedule)

    def _start_background(self) -> None:
        self._reconcile_heartbeats()

        # Start cron
        def run_cron():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.cron_scheduler.start())

        cron_thread = threading.Thread(target=run_cron, daemon=True)
        cron_thread.start()

        # Start health monitor
        def run_health():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.health_monitor.start())

        health_thread = threading.Thread(target=run_health, daemon=True)
        health_thread.start()

    def _init_channel_manager(self) -> None:
        """Create the ChannelManager with callbacks (but don't start channels yet)."""
        # Channel callback helpers
        def _channel_status(agent_name: str) -> dict | None:
            try:
                return self.transport.request_sync(agent_name, "GET", "/status", timeout=3)
            except Exception as e:
                logger.debug("Channel status check failed for '%s': %s", agent_name, e)
                return None

        def _channel_costs() -> list[dict]:
            return self.cost_tracker.get_all_agents_spend("today")

        def _channel_reset(agent_name: str) -> bool:
            try:
                self.transport.request_sync(agent_name, "POST", "/chat/reset", timeout=5)
                return True
            except Exception as e:
                logger.debug("Channel reset failed for '%s': %s", agent_name, e)
                return False

        async def stream_dispatch_to_agent(agent_name: str, message: str):
            if self.event_bus:
                self.event_bus.emit("message_received", agent=agent_name,
                    data={"message": message[:200], "source": "channel_stream"})
            if self.trace_store:
                from src.shared.trace import new_trace_id
                self.trace_store.record(
                    trace_id=new_trace_id(), source="channel_stream",
                    agent=agent_name, event_type="chat",
                    detail=message[:120],
                )
            async for event in self.transport.stream_request(
                agent_name, "POST", "/chat/stream",
                json={"message": message}, timeout=120,
            ):
                if self.event_bus and isinstance(event, dict):
                    etype = event.get("type", "")
                    if etype in ("tool_start", "tool_result"):
                        self.event_bus.emit(etype, agent=agent_name,
                            data={k: v for k, v in event.items() if k != "type"})
                yield event

        def _channel_addkey(service: str, key: str) -> None:
            from src.host.credentials import (
                SYSTEM_CREDENTIAL_PROVIDERS,
                is_system_credential,
            )
            # Normalize bare provider names (defense-in-depth, caller may also normalize)
            if service.lower() in SYSTEM_CREDENTIAL_PROVIDERS and not service.lower().endswith("_api_key"):
                service = f"{service}_api_key"
            self.credential_vault.add_credential(
                service, key, system=is_system_credential(service),
            )

        def _channel_steer(agent: str, msg: str) -> None:
            if self.lane_manager:
                asyncio.run_coroutine_threadsafe(
                    self.lane_manager.enqueue(agent, msg, mode="steer"), self._dispatch_loop
                ).result(timeout=5)

        def _channel_debug(trace_id: str | None = None) -> list[dict]:
            if not self.trace_store:
                return []
            if trace_id:
                return self.trace_store.get_trace(trace_id)
            return self.trace_store.list_recent(10)

        self.channel_manager = ChannelManager(
            self.cfg, self.async_dispatch, self.router.agent_registry,
            status_fn=_channel_status,
            costs_fn=_channel_costs,
            reset_fn=_channel_reset,
            stream_dispatch_fn=stream_dispatch_to_agent,
            addkey_fn=_channel_addkey,
            steer_fn=_channel_steer,
            debug_fn=_channel_debug,
        )

    def _start_channels(self) -> None:
        channel_routers = self.channel_manager.start_all()
        for ch_router in channel_routers:
            self._app.include_router(ch_router)
        self._active_channels = self.channel_manager.active

    def _print_ready(self) -> None:
        agents_cfg = self.cfg.get("agents", {})
        active_agents = list(agents_cfg.keys())
        agent_projects = self.cfg.get("_agent_projects", {})
        projects = self.cfg.get("projects", {})
        mesh_port = self.cfg["mesh"]["port"]

        # ── Services ──
        echo_header("OpenLegion")
        echo_ok(f"Dashboard: http://localhost:{mesh_port}")
        if hasattr(self.runtime, 'browser_vnc_url') and self.runtime.browser_vnc_url:
            echo_ok(f"Browser VNC: {self.runtime.browser_vnc_url}")
        if self._cron_job_count:
            echo_ok(f"Cron: {self._cron_job_count} job{'s' if self._cron_job_count != 1 else ''}")
        if self.channel_manager and self.channel_manager.active:
            for label, paired in self.channel_manager.channel_status:
                if paired:
                    echo_ok(f"{label} (paired)")
            for instruction in self.channel_manager.pairing_instructions:
                click.echo(click.style("  \u26a0 ", fg="yellow") + instruction.strip())

        # ── Agents ──
        if self._agent_results:
            echo_header("Agents")
            default_model = self.cfg.get("llm", {}).get("default_model", "openai/gpt-4o-mini")
            for agent_id, ready in self._agent_results:
                agent_cfg = agents_cfg.get(agent_id, {})
                model = agent_cfg.get("model", default_model)
                if ready:
                    echo_ok(f"{agent_id:<20} {model}")
                else:
                    logs = self.runtime.get_logs(agent_id, tail=15)
                    echo_fail(f"{agent_id:<20} failed to start")
                    if logs:
                        click.echo(logs, err=True)

        # ── Projects ──
        if projects:
            assigned = set(agent_projects.keys())
            standalone = [a for a in active_agents if a not in assigned]

            for pname in sorted(projects.keys()):
                members = [a for a in active_agents if agent_projects.get(a) == pname]
                if members:
                    click.echo(f"\n  Project [{pname}]: {', '.join(members)}")

            if standalone:
                click.echo(f"\n  Standalone: {', '.join(standalone)}")

        # ── Footer ──
        if active_agents:
            active_agent = active_agents[0]
            click.echo(f"\nChatting with '{active_agent}'.", nl=False)
            if len(active_agents) > 1:
                click.echo(" Use @agent to direct messages. /help for commands.")
            else:
                click.echo(" /help for commands.")
        else:
            click.echo("\nNo agents running. Add one with /add or via the dashboard:")
            click.echo(f"  http://localhost:{mesh_port}")

        if self._start_time:
            elapsed = time.time() - self._start_time
            click.echo(f"  Started in {elapsed:.1f}s\n")
        else:
            click.echo()
