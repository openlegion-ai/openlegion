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
from src.cli.config import PROJECT_ROOT, _check_docker_running, _ensure_docker_image, _load_config
from src.cli.formatting import echo_fail, echo_header, echo_ok

logger = logging.getLogger("cli")


class RuntimeContext:
    """Manages the full OpenLegion runtime lifecycle."""

    def __init__(self, config_path: str, use_sandbox: bool = False):
        self.cfg = _load_config(Path(config_path))
        self.use_sandbox = use_sandbox
        self.runtime = None
        self.transport = None
        self.blackboard = None
        self.pubsub = None
        self.permissions = None
        self.cost_tracker = None
        self.credential_vault = None
        self.router = None
        self.orchestrator = None
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

    def start(self) -> None:
        """Initialize and start all components. Called once."""
        self._validate_prereqs()
        self._select_backend()
        self._create_components()
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
        if self.cost_tracker:
            self.cost_tracker.close()
        if self.trace_store:
            self.trace_store.close()
        if self.pubsub:
            self.pubsub.close()
        if self.blackboard:
            self.blackboard.close()

        # Close shared httpx clients on the dispatch loop
        if self._dispatch_loop:
            async def _close_clients():
                for closeable in [self.transport, self.router, self.orchestrator, self.credential_vault]:
                    if hasattr(closeable, 'close') and callable(closeable.close):
                        try:
                            result = closeable.close()
                            if hasattr(result, '__await__'):
                                await result
                        except Exception as e:
                            logger.debug("Error closing %s: %s", type(closeable).__name__, e)
            try:
                future = asyncio.run_coroutine_threadsafe(_close_clients(), self._dispatch_loop)
                future.result(timeout=5)
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
        agents_cfg = self.cfg.get("agents", {})
        if not agents_cfg:
            click.echo("No agents configured. Run: openlegion setup", err=True)
            sys.exit(1)

        if not _check_docker_running():
            click.echo("Docker is not running. Please start Docker first.", err=True)
            sys.exit(1)

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
        from src.dashboard.events import EventBus
        from src.host.costs import CostTracker
        from src.host.credentials import CredentialVault
        from src.host.mesh import Blackboard, MessageRouter, PubSub
        from src.host.orchestrator import Orchestrator
        from src.host.permissions import PermissionMatrix
        from src.host.traces import TraceStore

        # Ensure collaborative permissions are up to date before loading
        if self.cfg.get("collaboration", False):
            from src.cli.config import _set_collaborative_permissions
            _set_collaborative_permissions()

        mesh_port = self.cfg["mesh"]["port"]

        self.event_bus = EventBus()
        self.blackboard = Blackboard(event_bus=self.event_bus)
        self.pubsub = PubSub(db_path="pubsub.db")
        self.permissions = PermissionMatrix()
        self.cost_tracker = CostTracker(event_bus=self.event_bus)
        failover_config = self.cfg.get("llm", {}).get("failover", {})
        self.credential_vault = CredentialVault(
            cost_tracker=self.cost_tracker,
            failover_config=failover_config or None,
        )
        self.router = MessageRouter(self.permissions, {})
        self.orchestrator = Orchestrator(
            mesh_url=f"http://localhost:{mesh_port}",
            blackboard=self.blackboard,
            pubsub=self.pubsub,
            container_manager=self.runtime,
        )
        self.trace_store = TraceStore()

        # Create HealthMonitor early so the dashboard router can reference it.
        # Only register() is called here; start() happens in _start_background().
        from src.host.health import HealthMonitor

        self.health_monitor = HealthMonitor(
            runtime=self.runtime, transport=self.transport, router=self.router,
            event_bus=self.event_bus,
        )

    def _start_agents(self) -> None:
        from src.host.runtime import DockerBackend, SandboxBackend
        from src.host.transport import HttpTransport

        agents_cfg = self.cfg.get("agents", {})
        default_model = self.cfg.get("llm", {}).get("default_model", "openai/gpt-4o-mini")
        mesh_port = self.cfg["mesh"]["port"]

        for agent_id, agent_cfg in agents_cfg.items():
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
            agent_browser_backend = agent_cfg.get("browser_backend", "")
            try:
                url = self.runtime.start_agent(
                    agent_id=agent_id,
                    role=agent_cfg["role"],
                    skills_dir=skills_dir,
                    system_prompt=agent_cfg.get("system_prompt", ""),
                    model=agent_model,
                    mcp_servers=agent_mcp_servers,
                    browser_backend=agent_browser_backend,
                )
            except (subprocess.TimeoutExpired, RuntimeError) as exc:
                if isinstance(self.runtime, SandboxBackend):
                    click.echo(
                        f"\n  Sandbox failed for '{agent_id}': {exc}\n"
                        "  Falling back to Docker container isolation...\n",
                        err=True,
                    )
                    self.runtime.stop_all()
                    import platform as _platform
                    self.runtime = DockerBackend(
                        mesh_host_port=mesh_port,
                        use_host_network=_platform.system() == "Linux",
                        project_root=str(PROJECT_ROOT),
                    )
                    self.transport = HttpTransport()
                    _ensure_docker_image()
                    url = self.runtime.start_agent(
                        agent_id=agent_id,
                        role=agent_cfg["role"],
                        skills_dir=skills_dir,
                        system_prompt=agent_cfg.get("system_prompt", ""),
                        model=agent_model,
                        mcp_servers=agent_mcp_servers,
                        browser_backend=agent_browser_backend,
                    )
                else:
                    raise
            self.router.register_agent(agent_id, url)
            self.agent_urls[agent_id] = url
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
                    event_type="chat", detail=message[:120],
                )
            try:
                result = await self.transport.request(
                    agent_name, "POST", "/chat", json={"message": message},
                )
                response = result.get("response", "(no response)")
                if self.event_bus:
                    self.event_bus.emit("message_sent", agent=agent_name,
                        data={"message": message[:200], "response_length": len(response),
                              "source": "dispatch"})
                return response
            except Exception as e:
                return f"Error: {e}"

        async def _direct_steer(agent_name: str, message: str) -> dict:
            try:
                return await self.transport.request(
                    agent_name, "POST", "/chat/steer", json={"message": message},
                )
            except Exception as e:
                return {"injected": False, "error": str(e)}

        self.lane_manager = LaneManager(dispatch_fn=_direct_dispatch, steer_fn=_direct_steer)

        self._dispatch_loop = asyncio.new_event_loop()

        def _run_dispatch_loop():
            asyncio.set_event_loop(self._dispatch_loop)
            self._dispatch_loop.run_forever()

        _dispatch_thread = threading.Thread(target=_run_dispatch_loop, daemon=True)
        _dispatch_thread.start()

    def _start_mesh_server(self) -> None:
        import uvicorn

        from src.channels.webhook import create_webhook_router
        from src.host.server import create_mesh_app
        from src.host.webhooks import WebhookManager

        mesh_port = self.cfg["mesh"]["port"]

        webhook_manager = WebhookManager(dispatch_fn=self.async_dispatch)

        app = create_mesh_app(
            self.blackboard, self.pubsub, self.router, self.permissions,
            self.credential_vault, self.cron_scheduler, self.runtime,
            self.transport, self.orchestrator,
            auth_tokens=self.runtime.auth_tokens,
            trace_store=self.trace_store,
            event_bus=self.event_bus,
        )
        app.include_router(create_webhook_router(self.orchestrator))
        app.include_router(webhook_manager.create_router())

        from src.dashboard.server import create_dashboard_router

        dashboard_router = create_dashboard_router(
            blackboard=self.blackboard,
            health_monitor=self.health_monitor,
            cost_tracker=self.cost_tracker,
            trace_store=self.trace_store,
            event_bus=self.event_bus,
            agent_registry=self.router.agent_registry,
            mesh_port=mesh_port,
        )
        app.include_router(dashboard_router)
        self._app = app

        server_config = uvicorn.Config(app, host="0.0.0.0", port=mesh_port, log_level="warning")
        self._server = uvicorn.Server(server_config)
        mesh_thread = threading.Thread(target=self._server.run, daemon=True)
        mesh_thread.start()

    def _wait_for_readiness(self) -> None:
        import httpx

        mesh_port = self.cfg["mesh"]["port"]
        agents_cfg = self.cfg.get("agents", {})

        # Wait for mesh
        mesh_ready = False
        for _ in range(30):
            try:
                httpx.get(f"http://localhost:{mesh_port}/mesh/agents", timeout=1)
                mesh_ready = True
                break
            except Exception:
                time.sleep(0.5)

        if not mesh_ready:
            echo_fail(
                f"Mesh server failed to start on port {mesh_port}. "
                f"Port may be in use. Try: openlegion stop"
            )
            self.runtime.stop_all()
            sys.exit(1)

        echo_header("OpenLegion")
        click.echo()
        echo_ok(f"Mesh host on port {mesh_port}")
        echo_ok(f"Isolation: {self._backend_label}")
        echo_ok(f"Dashboard: http://localhost:{mesh_port}/dashboard")

        # Wait for agents
        async def _wait_all_agents():
            async def _wait_one(aid):
                ready = await self.runtime.wait_for_agent(aid, timeout=60)
                return aid, ready
            return await asyncio.gather(*[_wait_one(aid) for aid in agents_cfg])

        agent_results = asyncio.run(_wait_all_agents())

        echo_header("Agents")
        default_model = self.cfg.get("llm", {}).get("default_model", "openai/gpt-4o-mini")
        for agent_id, ready in agent_results:
            agent_cfg = agents_cfg.get(agent_id, {})
            model = agent_cfg.get("model", default_model)
            browser = agent_cfg.get("browser_backend", "basic") or "basic"
            if ready:
                echo_ok(f"{agent_id:<20} ready     model: {model:<20} browser: {browser}")
            else:
                logs = self.runtime.get_logs(agent_id, tail=15)
                echo_fail(f"{agent_id:<20} failed to start")
                if logs:
                    click.echo(logs, err=True)

    def _create_cron_scheduler(self) -> None:
        from src.host.cron import CronScheduler

        async def cron_dispatch(agent_name: str, message: str) -> str:
            from src.shared.trace import new_trace_id
            result = await self.async_dispatch(agent_name, message, trace_id=new_trace_id())
            if result and result.strip():
                notification = f"[cron -> {agent_name}] {result}"
                sys.stdout.write(f"\n{notification}\n")
                sys.stdout.flush()
                for ch in self._active_channels:
                    try:
                        await ch.send_notification(notification)
                    except Exception as e:
                        logger.debug("Cron notification to %s failed: %s", type(ch).__name__, e)
            return result

        async def trigger_workflow(workflow_name: str, payload: dict) -> str:
            exec_id = await self.orchestrator.trigger_workflow(workflow_name, payload)
            return f"workflow:{exec_id}"

        self.cron_scheduler = CronScheduler(
            dispatch_fn=cron_dispatch,
            workflow_trigger_fn=trigger_workflow,
            blackboard=self.blackboard,
        )
        if self.cron_scheduler.jobs:
            echo_ok(f"Cron scheduler: {len(self.cron_scheduler.jobs)} jobs loaded")

    def _start_background(self) -> None:
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

    def _start_channels(self) -> None:
        # Channel callback helpers
        def _channel_status(agent_name: str) -> dict | None:
            try:
                return self.transport.request_sync(agent_name, "GET", "/status", timeout=3)
            except Exception:
                return None

        def _channel_costs() -> list[dict]:
            return self.cost_tracker.get_all_agents_spend("today")

        def _channel_reset(agent_name: str) -> bool:
            try:
                self.transport.request_sync(agent_name, "POST", "/chat/reset", timeout=5)
                return True
            except Exception:
                return False

        async def stream_dispatch_to_agent(agent_name: str, message: str):
            if self.event_bus:
                self.event_bus.emit("message_received", agent=agent_name,
                    data={"message": message[:200], "source": "channel_stream"})
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
            self.credential_vault.add_credential(service, key)

        self.channel_manager = ChannelManager(
            self.cfg, self.async_dispatch, self.router.agent_registry,
            status_fn=_channel_status,
            costs_fn=_channel_costs,
            reset_fn=_channel_reset,
            stream_dispatch_fn=stream_dispatch_to_agent,
            addkey_fn=_channel_addkey,
        )
        channel_routers = self.channel_manager.start_all()
        for ch_router in channel_routers:
            self._app.include_router(ch_router)
        self._active_channels = self.channel_manager.active

    def _print_ready(self) -> None:
        agents_cfg = self.cfg.get("agents", {})
        active_agents = list(agents_cfg.keys())
        active_agent = active_agents[0]

        # Show channel status
        if self.channel_manager and self.channel_manager.active:
            echo_header("Channels")
            for label, paired in self.channel_manager.channel_status:
                if paired:
                    echo_ok(f"{label} (paired)")
            for instruction in self.channel_manager.pairing_instructions:
                click.echo(click.style("  \u26a0 ", fg="yellow") + instruction.strip())

        click.echo(f"\nChatting with '{active_agent}'.", nl=False)
        if len(active_agents) > 1:
            click.echo(" Use @agent to direct messages. /help for commands.")
        else:
            click.echo(" /help for commands.")
        click.echo("")
