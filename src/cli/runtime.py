"""RuntimeContext: manages the full OpenLegion runtime lifecycle."""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import click

from src.cli.channels import ChannelManager
from src.cli.config import (
    ENV_FILE,
    PROJECT_ROOT,
    TEAMS_DIR,
    _check_docker_running,
    _ensure_docker_image,
    _load_config,
)
from src.cli.formatting import echo_fail, echo_header, echo_ok
from src.cli.proxy import build_proxy_env_vars, resolve_agent_proxy
from src.shared.types import RESERVED_AGENT_IDS
from src.shared.utils import set_llm_max_tokens_env

if TYPE_CHECKING:
    from src.shared.types import MessageOrigin

logger = logging.getLogger("cli")

# Embedding-provider ladder for auto-selecting a default embedding model
# when the operator hasn't set one. Each entry: (provider, litellm model,
# output dim), ordered by preference. Only providers that actually offer an
# embeddings API appear — Anthropic/xAI/Groq have none, so an Anthropic-only
# deployment correctly falls through to keyword-only memory. Non-OpenAI
# models are fully qualified (``voyage/…``) so the mesh provider resolver
# maps them to the right SYSTEM API key with no extra config.
_EMBEDDING_PROVIDER_LADDER: list[tuple[str, str, int]] = [
    ("openai", "text-embedding-3-small", 1536),
    ("voyage", "voyage/voyage-3.5", 1024),  # Anthropic's recommended embedder
    ("gemini", "gemini/text-embedding-004", 768),
    ("cohere", "cohere/embed-english-v3.0", 1024),
]

# Known embedding model → output dimension. Sets EMBEDDING_DIM for an
# explicit operator override. Unknown models fall back to the 1536 default;
# the agent self-heals a wrong dim at runtime (memory.py
# _reconcile_embedding_dim + the _store_embedding length guard), so this map
# is an optimization, not a correctness dependency.
_EMBEDDING_MODEL_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "voyage/voyage-3.5": 1024,
    "voyage/voyage-3": 1024,
    "gemini/text-embedding-004": 768,
    "cohere/embed-english-v3.0": 1024,
    "cohere/embed-multilingual-v3.0": 1024,
}
_DEFAULT_EMBEDDING_DIM = 1536


def _embedding_providers_with_keys() -> set[str]:
    """Embedding-capable providers (from the ladder) that have a SYSTEM API
    key configured in the environment.

    Deliberately checks the API key directly rather than reusing
    ``get_available_providers()``: the mesh embed proxy authenticates with
    an API key only (no OAuth path), so an OAuth-only provider must NOT be
    treated as embedding-capable — it would select embeddings that then fail.
    """
    return {
        provider
        for provider, _model, _dim in _EMBEDDING_PROVIDER_LADDER
        if os.environ.get(f"OPENLEGION_SYSTEM_{provider.upper()}_API_KEY")
    }


def _resolve_embedding(
    cfg_embedding_model: str | None,
    keyed_providers: set[str],
) -> tuple[str, int]:
    """Resolve ``(embedding_model, output_dim)`` to run with.

    Priority:
      1. Explicit operator choice (``llm.embedding_model`` in config),
         including the literal ``"none"`` to force keyword-only memory.
      2. First embedding-capable provider with a configured API key, walked
         in ``_EMBEDDING_PROVIDER_LADDER`` order — this is what lets an
         Anthropic-chat deployment light up semantic memory via a Voyage or
         OpenAI key.
      3. ``"none"`` — no embedding-capable key; memory runs on BM25/FTS5
         keyword search only (fully supported, never an error).

    The returned ``dim`` is advisory; the agent corrects a wrong dimension
    at runtime without data loss.
    """
    if cfg_embedding_model:
        model = str(cfg_embedding_model).strip()
        return model, _EMBEDDING_MODEL_DIMS.get(model, _DEFAULT_EMBEDDING_DIM)
    for provider, model, dim in _EMBEDDING_PROVIDER_LADDER:
        if provider in keyed_providers:
            return model, dim
    return "none", _DEFAULT_EMBEDDING_DIM


# Cap on the operator-visible ``blocker_note`` written when a mesh→agent
# ``/chat`` dispatch raises. Mirrors the lane watchdog's ``[:500]`` cap and
# the mesh ``/mesh/tasks/{id}/status`` endpoint's 500-char ``error`` promotion.
_DISPATCH_ERROR_NOTE_MAX = 500


def _build_dispatch_error_note(exc: BaseException) -> str:
    """Render a transport/agent dispatch error into a SAFE ``blocker_note``.

    The exception string is UNTRUSTED — the ``/chat`` body it came from may
    echo agent-authored content, secrets, or injection. It MUST NOT reach the
    operator-visible ``blocker_note`` raw. We:

    1. Run it through the central free-form redactor
       (:func:`redact_text_with_urls`), which strips both token-shaped secrets
       AND credential-keyed URL query-param values, and
    2. Truncate to :data:`_DISPATCH_ERROR_NOTE_MAX` chars.

    Returns a ``"dispatch_error: <redacted reason>"`` string. XSS in the note
    is handled downstream by the dashboard's Jinja autoescape — this layer is
    purely about redaction + truncation.
    """
    from src.shared.redaction import redact_text_with_urls

    raw = str(exc).strip() or exc.__class__.__name__
    safe = redact_text_with_urls(raw)
    note = f"dispatch_error: {safe}"
    return note[:_DISPATCH_ERROR_NOTE_MAX]


# Honest, operator-visible ``blocker_note`` written when a dispatch is killed
# by an INFRA shutdown / connection-drop rather than a real task error. The
# task is left ``blocked`` (recoverable) — NOT terminal ``failed`` — so its
# work is not silently lost. (Prod incident: a host restart under running
# tasks killed three in-flight subtasks as terminal ``failed`` with
# ``dispatch_error: Server disconnected without sending a response``; they had
# to be manually re-dispatched.)
_INFRA_INTERRUPT_NOTE = "interrupted by host restart — recoverable"

# Substrings (lower-cased) that mark a transport/connection-drop signature —
# i.e. the host (or the agent container) went away mid-request, the hallmark
# of a graceful-shutdown / restart tear-down rather than a genuine task error.
# Conservative on purpose: only the connection-drop class is reclassified to
# the recoverable ``blocked`` state; everything else stays terminal ``failed``.
_INFRA_DISCONNECT_MARKERS: tuple[str, ...] = (
    "server disconnected",
    "disconnected without sending",
    "connection reset",
    "connection refused",
    "connection aborted",
    "remote protocol error",
    "peer closed connection",
    "incomplete chunked read",
    "all connection attempts failed",
)


def _is_infra_disconnect(exc: BaseException) -> bool:
    """True when ``exc`` is a transport-level connection drop — the signature
    of a host/agent shutdown or restart tearing down an in-flight ``/chat``
    dispatch, as opposed to a genuine task error.

    Detection is conservative and twofold:

    1. **Type-based** — httpx transport-error subclasses that only arise when
       the connection itself fails (``ConnectError``, ``ReadError``,
       ``WriteError``, ``RemoteProtocolError``, ``ConnectTimeout``,
       ``PoolTimeout``). httpx is an optional-at-import dependency here, so the
       check degrades gracefully if it can't be imported.
    2. **String-based** — a fallback that matches the connection-drop markers
       in the (already untrusted, but here only substring-tested) message, so
       a non-httpx wrapper carrying the same signature still classifies right.

    A ``read`` *timeout while the connection stays up* is deliberately NOT
    matched — that is the agent being slow, governed by the lane watchdog, not
    an infra tear-down.
    """
    try:
        import httpx

        if isinstance(
            exc,
            (
                httpx.ConnectError,
                httpx.ReadError,
                httpx.WriteError,
                httpx.RemoteProtocolError,
                httpx.ConnectTimeout,
                httpx.PoolTimeout,
            ),
        ):
            return True
    except Exception:  # pragma: no cover - httpx import shouldn't fail
        pass

    text = str(exc).lower()
    return any(marker in text for marker in _INFRA_DISCONNECT_MARKERS)


def _bind_mesh_socket(host: str, port: int) -> socket.socket:
    """Bind and return the mesh's primary listening socket — FATAL on failure.

    Production incident (70-min silent outage): a ``systemctl restart`` raced
    the old mesh's teardown, so the new process found port 8420 still held.
    uvicorn's own ``startup()`` catches the bind ``OSError``, logs the
    ``[Errno 98] address already in use`` line, and calls ``sys.exit(1)`` — but
    it runs the server in a daemon thread, so that ``SystemExit`` only kills the
    *thread*. The main process kept running bound to NO port while systemd
    reported ``active (running)`` and the provisioner health check (which hit
    the *stale* listener still answering on 8420) believed the mesh was up. The
    mesh served 404s for ~70 minutes.

    We bind the socket ourselves on the MAIN thread *before* launching uvicorn
    and hand it the open socket. A bind failure now raises here, on the caller's
    thread, where it tears the whole process down (caller exits non-zero) — so
    systemd restarts it and the health check sees the truth. The success banner
    is never printed when the bind fails.

    ``SO_REUSEADDR`` matches uvicorn's default and lets a genuinely-released
    port be re-bound through TIME_WAIT; it does NOT mask a live listener
    (a second ``bind()`` to an actively-LISTENing socket still raises
    ``EADDRINUSE``), which is exactly the failure we must surface.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.listen(2048)
        sock.set_inheritable(True)
    except OSError:
        sock.close()
        raise
    return sock


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
        self.intent_store = None
        self.lifecycle_store = None
        self.teams_store = None
        self.thread_store = None
        self.agent_urls: dict[str, str] = {}
        self._dispatch_loop = None
        # Post-completion verification wakes (success path) — sliding
        # window so a burst of completing chains can't turn the operator
        # interrupt-driven. Suppressed chains are still rated on the
        # next heartbeat.
        self._verify_wake_times: deque = deque()
        self._server = None
        self._active_channels: list = []
        self._start_time: float | None = None
        self._agent_results: list = []
        self._cron_job_count: int = 0
        # Set True as the FIRST act of shutdown() so an in-flight dispatch that
        # drops *because we are tearing down* is attributable. A connection-drop
        # is reclassified to recoverable ``blocked`` ONLY while this is True
        # (see _close_task_on_dispatch_error); a drop while still running stays
        # terminal ``failed`` — a genuine agent-container crash must not hide.
        self._shutting_down = False

    def start(self) -> None:
        """Initialize and start all components. Called once."""
        self._start_time = time.time()
        # Match detached mode behavior so relative paths (pubsub.db, agent_tools/, config/)
        # resolve under the OpenLegion repo root even when launched elsewhere.
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
        # FIRST, before any teardown: mark the mesh as shutting down so any
        # in-flight dispatch that drops during teardown is attributable to the
        # restart (and thus reclassified to recoverable ``blocked``).
        self._shutting_down = True
        click.echo("  Stopping OpenLegion...", nl=False)
        if self.chain_watcher:
            self.chain_watcher.stop()
        if self.channel_manager:
            self.channel_manager.stop_all()
        if self.health_monitor:
            self.health_monitor.stop()
        if self.cron_scheduler:
            self.cron_scheduler.stop()
        if self.runtime:
            self.runtime.stop_all()
            if hasattr(self.runtime, "stop_browser_service"):
                self.runtime.stop_browser_service()
        if self.cost_tracker:
            self.cost_tracker.close()
        if self.trace_store:
            self.trace_store.close()
        if self.intent_store:
            self.intent_store.close()
        if self.lifecycle_store:
            self.lifecycle_store.close()
        if self.pubsub:
            self.pubsub.close()
        if self.blackboard:
            self.blackboard.close()

        # Close shared httpx clients on the dispatch loop — close all
        # concurrently so one slow close doesn't block the others.
        if self._dispatch_loop:

            async def _close_clients():
                async def _close_one(name, closeable):
                    if closeable is None or not hasattr(closeable, "close"):
                        return
                    try:
                        result = closeable.close()
                        if hasattr(result, "__await__"):
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

    @staticmethod
    def _resolve_dispatch_trace_id(trace_id: str | None) -> str:
        """Resolve the per-turn trace_id for a dispatch (Phase 1).

        Order of precedence — never clobber an existing trace (per the
        session-observability plan):
        1. An explicit ``trace_id`` argument (cron / CLI mint it upstream).
        2. The active ``current_trace_id`` contextvar — set by
           ``Channel.dispatch`` for channel-originated messages, so the
           lane worker reuses it rather than losing it.
        3. A freshly minted id — covers callers with neither (e.g. the
           webhook manager dispatching directly), so every human/external
           -rooted message correlates end-to-end.
        """
        if trace_id:
            return trace_id
        from src.shared.trace import current_trace_id, new_trace_id

        return current_trace_id.get() or new_trace_id()

    def dispatch(
        self,
        agent: str,
        message: str,
        mode: str = "followup",
        trace_id: str | None = None,
        origin: "MessageOrigin | None" = None,
        auto_notify: bool = False,
        system_note: bool = False,
    ) -> str:
        """Thread-safe synchronous message dispatch.

        Schedules the coroutine on the dedicated dispatch loop and blocks
        until the result is ready.  For async callers, use async_dispatch().
        """
        trace_id = self._resolve_dispatch_trace_id(trace_id)
        future = asyncio.run_coroutine_threadsafe(
            self.lane_manager.enqueue(
                agent,
                message,
                mode=mode,
                trace_id=trace_id,
                origin=origin,
                auto_notify=auto_notify,
                system_note=system_note,
            ),
            self._dispatch_loop,
        )
        return future.result()

    async def async_dispatch(
        self,
        agent: str,
        message: str,
        mode: str = "followup",
        trace_id: str | None = None,
        origin: "MessageOrigin | None" = None,
        auto_notify: bool = False,
        system_note: bool = False,
    ) -> str:
        """Async dispatch: schedules onto the dedicated dispatch loop."""
        trace_id = self._resolve_dispatch_trace_id(trace_id)
        future = asyncio.run_coroutine_threadsafe(
            self.lane_manager.enqueue(
                agent,
                message,
                mode=mode,
                trace_id=trace_id,
                origin=origin,
                auto_notify=auto_notify,
                system_note=system_note,
            ),
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
            mesh_host_port=mesh_port,
            project_root=str(PROJECT_ROOT),
            use_sandbox=self.use_sandbox,
        )
        self._is_sandbox = isinstance(self.runtime, SandboxBackend)
        self._backend_label = self.runtime.backend_name()

        if self._is_sandbox:
            self.transport = SandboxTransport()
        else:
            self.transport = HttpTransport()
            _ensure_docker_image()
        # Mesh→agent bearer auth (B7): bind the runtime's LIVE token dict so
        # every URL-registration path (initial start, /restart, health-monitor
        # restart, dashboard restart, /mesh/register) automatically sends the
        # current per-agent token — including tokens re-minted on restart.
        self.transport.bind_tokens(self.runtime.auth_tokens)

    def _create_components(self) -> None:
        from src.cli.config import (
            _backfill_capabilities_for_existing_agents,
            _ensure_all_agent_permissions,
        )
        from src.dashboard.events import EventBus
        from src.host.costs import CostTracker
        from src.host.credentials import CredentialVault
        from src.host.mesh import Blackboard, MessageRouter, PubSub
        from src.host.permissions import PermissionMatrix
        from src.host.traces import TraceStore

        # Backfill permissions for agents missing from permissions.json
        _ensure_all_agent_permissions()
        # Task 8 — derive structured routing fields from INTERFACE.md for
        # agents that pre-date the structured schema. Idempotent: only
        # runs for agents whose ``capabilities`` field is empty.
        _backfill_capabilities_for_existing_agents()

        # Ensure collaborative permissions are up to date before loading
        if self.cfg.get("collaboration", False):
            from src.cli.config import _set_collaborative_permissions

            _set_collaborative_permissions()

        self.event_bus = EventBus()
        # Trace retention GC on by default (7 days). Override via
        # OPENLEGION_TRACE_RETENTION_HOURS; <=0 disables time-based GC.
        try:
            _trace_retention = int(os.environ.get("OPENLEGION_TRACE_RETENTION_HOURS", "168"))
        except ValueError:
            _trace_retention = 168
        self.trace_store = TraceStore(max_age_hours=_trace_retention if _trace_retention > 0 else None)
        # Durable verbatim-intent store (Phase 2, session observability).
        # Captures the FULL inbound message centrally, keyed by trace_id +
        # origin, so intent survives container wipes / resets / deploys
        # (container-local chat_transcript.jsonl rotates and dies with the
        # container). Append-only with a 90-day time-based GC.
        from src.host.intent import IntentStore

        self.intent_store = IntentStore(db_path=os.environ.get("OPENLEGION_INTENT_DB", "data/intent.db"))
        # External infra-event markers (host restart / deploy / OOM). Emitted
        # out-of-band by the provisioner or an operator runbook via the
        # internal-only POST /mesh/system/lifecycle_event endpoint and
        # interleaved by wall-clock into the session-reader timeline. Append-
        # only with a 90-day time-based GC (mirrors the intent store).
        from src.host.lifecycle import LifecycleStore

        self.lifecycle_store = LifecycleStore(db_path=os.environ.get("OPENLEGION_LIFECYCLE_DB", "data/lifecycle.db"))
        self.blackboard = Blackboard(event_bus=self.event_bus)
        self.pubsub = PubSub(db_path="pubsub.db")
        self.permissions = PermissionMatrix()
        self.cost_tracker = CostTracker()
        failover_config = self.cfg.get("llm", {}).get("failover", {})
        default_model = self.cfg.get("llm", {}).get("default_model", "openai/gpt-4o-mini")
        self.credential_vault = CredentialVault(
            cost_tracker=self.cost_tracker,
            failover_config=failover_config or None,
            default_model=default_model,
        )
        # Plumb the credential resolver into the already-constructed runtime
        # backend so $CRED{name} handles in MCP server configs (env values
        # and args) resolve at agent start. Idempotent; safe if the runtime
        # is later replaced (the sandbox→docker fallback path in
        # _start_agents also calls set_credential_resolver explicitly).
        self.runtime.set_credential_resolver(
            vault=self.credential_vault,
            permissions=self.permissions,
        )
        # Fleet MCP connector catalog — the single source of truth for
        # which agents run which MCP servers. The runtime reads it at
        # every agent (re)start; the dashboard Connectors page manages it.
        from src.host.connectors import ConnectorStore

        self.connector_store = ConnectorStore()
        self.runtime.set_connector_store(self.connector_store)
        # THE team authority (identity, metadata, membership, standing
        # goals). Disk-backed SQLite (WAL) + the config/teams/{id}/
        # scaffold; every consumer — mesh app, dashboard, router,
        # agent-start env — reads through this instance.
        from src.host.teams import TeamStore

        self.teams_store = TeamStore(
            db_path=os.environ.get("OPENLEGION_TEAMS_DB", "data/teams.db"),
            teams_dir=TEAMS_DIR,
        )
        # Team budget envelope (plan B4): the cost tracker resolves each
        # caller's team + envelope through the store at pre-flight time.
        self.cost_tracker.set_team_store(self.teams_store)

        # Team env resolver, injected at the RUNTIME-BACKEND level (like
        # LLM_UTILITY_MODEL) so EVERY start path — initial boot, REPL
        # /restart, health-monitor restart, dashboard restart, ephemeral
        # spawn — resolves the agent's effective team without per-caller
        # plumbing. Workers always get TEAM_NAME (real team, or their own
        # id as the private team-of-one namespace, ratified #5);
        # TEAM_MD_PATH only for real teams; the operator stays unscoped.
        def _team_env_for(agent_id: str) -> dict[str, str]:
            from src.cli.config import _OPERATOR_AGENT_ID

            if agent_id == _OPERATOR_AGENT_ID:
                return {}
            team = self.teams_store.team_of(agent_id)
            env = {"TEAM_NAME": team or agent_id}
            if team:
                env["TEAM_MD_PATH"] = str(TEAMS_DIR / team / "team.md")
            return env

        self.runtime.set_team_env_provider(_team_env_for)
        # Team Drive lifecycle (Phase-2 unit 1, plan A.3 #3): the store
        # owns WHEN drives exist; the runtime backend owns WHERE (bare
        # git repos under data/team_drives/). Backfill provisions any
        # team created before the drive landed (or whose create-time
        # provision failed) — mirrors the solo-ACL boot backfill.
        self.teams_store.set_drive_provisioner(
            self.runtime.ensure_team_volume,
            self.runtime.remove_team_volume,
        )
        try:
            backfilled = self.teams_store.backfill_drives()
            if backfilled:
                logger.info("Provisioned team drives at boot: %s", ", ".join(backfilled))
        except Exception:
            logger.exception("Team drive boot backfill failed (teams keep working without drives)")
        # Durable Team Threads store (Phase-2 unit 2): channel/task/dm
        # conversations. Replaces the router's in-memory message_log
        # deque AND the blackboard back-edge feed — the router records
        # DM traffic here; the mesh records task back-edge events.
        from src.host.threads import ThreadStore

        self.thread_store = ThreadStore(
            db_path=os.environ.get("OPENLEGION_THREADS_DB", "data/threads.db"),
            event_bus=self.event_bus,
        )
        self.router = MessageRouter(
            self.permissions,
            self.agent_urls,
            trace_store=self.trace_store,
            team_resolver=self.teams_store.team_of,
            thread_store=self.thread_store,
        )
        # Create HealthMonitor early so the dashboard router can reference it.
        # Only register() is called here; start() happens in _start_background().
        from src.host.health import HealthMonitor

        # System-signal reroute (bell removal): the two formerly
        # bell-only signals (dead OAuth connection, agent quarantine)
        # now land in the operator chat thread as durable notes + live
        # notification events.
        self.event_bus.add_listener(self._system_signal_producer)

        # Chain watcher (delegate-and-subscribe terminal delivery). Wired in
        # _start_background once the mesh app's tasks_store is available.
        self.chain_watcher = None
        self._tasks_store_ref = None
        # Strong refs for best-effort channel pushes so the event loop
        # doesn't GC an in-flight asyncio.Task before it sends.
        self._pending_chain_pushes: set = set()

        self.health_monitor = HealthMonitor(
            runtime=self.runtime,
            transport=self.transport,
            router=self.router,
            event_bus=self.event_bus,
            blackboard=self.blackboard,
        )

        # Fix 4 (seam follow-up): wire the credential vault's auth-failure
        # recorder to the health monitor. This is the load-bearing path
        # for quarantine — the agent-side report channel can also fire,
        # but the mesh-proxy boundary erases exception types so the
        # mesh-side recording must be the authoritative trigger.
        if self.credential_vault is not None:

            def _record_auth(agent_id: str, provider: str, model: str, http_status: int) -> None:
                try:
                    self.health_monitor.record_auth_failure(
                        agent_id,
                        provider=provider,
                        model=model,
                        http_status=http_status,
                    )
                except Exception as e:
                    logger.warning(
                        "HealthMonitor.record_auth_failure failed for agent='%s': %s",
                        agent_id,
                        e,
                    )

            self.credential_vault.set_auth_failure_recorder(_record_auth)

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
        # Reload config and permissions after possible operator creation
        self.cfg = _load_config()
        self.permissions.reload()
        agents_cfg = self.cfg.get("agents", {})

        # Reorder agents so operator starts first
        if _OPERATOR_AGENT_ID in agents_cfg:
            ordered = {_OPERATOR_AGENT_ID: agents_cfg[_OPERATOR_AGENT_ID]}
            ordered.update({k: v for k, v in agents_cfg.items() if k != _OPERATOR_AGENT_ID})
            agents_cfg = ordered

        # Resolve embedding model + dimension. Auto-selects from configured
        # provider keys when the operator hasn't set one (an Anthropic-chat
        # deployment lights up semantic memory via a Voyage/OpenAI key);
        # falls back to keyword-only memory when no embedding-capable key
        # exists — never an error.
        embedding_model, embedding_dim = _resolve_embedding(
            self.cfg.get("llm", {}).get("embedding_model"),
            _embedding_providers_with_keys(),
        )
        mesh_port = self.cfg["mesh"]["port"]

        # Respect plan limits on startup — only start up to max_agents.
        # Prevents OOM on downsized servers after a plan downgrade.
        # Operator is excluded from the count — it's always allowed.
        max_agents = int(os.environ.get("OPENLEGION_MAX_AGENTS", "0"))
        non_operator = {k: v for k, v in agents_cfg.items() if k != _OPERATOR_AGENT_ID}
        if max_agents > 0 and len(non_operator) > max_agents:
            logger.warning(
                "Agent limit is %d but %d agents configured — only starting first %d",
                max_agents,
                len(non_operator),
                max_agents,
            )
            trimmed = dict(list(non_operator.items())[:max_agents])
            if _OPERATOR_AGENT_ID in agents_cfg:
                trimmed[_OPERATOR_AGENT_ID] = agents_cfg[_OPERATOR_AGENT_ID]
            agents_cfg = trimmed

        self.runtime.extra_env["EMBEDDING_MODEL"] = embedding_model
        self.runtime.extra_env["EMBEDDING_DIM"] = str(embedding_dim)
        if embedding_model and embedding_model.lower() != "none":
            logger.info(
                "Semantic memory: ON — embedding model %s (dim %d)",
                embedding_model,
                embedding_dim,
            )
        else:
            logger.info(
                "Semantic memory: OFF (keyword-only) — no embedding-capable provider key configured",
            )

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
                raise click.ClickException(f"Agent ID '{agent_id}' is reserved for internal use")
            budget = agent_cfg.get("budget", {})
            if budget:
                from src.host.costs import DEFAULT_DAILY_BUDGET_USD, DEFAULT_MONTHLY_BUDGET_USD

                self.cost_tracker.set_budget(
                    agent_id,
                    daily_usd=budget.get("daily_usd", DEFAULT_DAILY_BUDGET_USD),
                    monthly_usd=budget.get("monthly_usd", DEFAULT_MONTHLY_BUDGET_USD),
                )
            # Guard the empty case: os.path.abspath("") resolves to the CWD
            # (the repo root), which would bind-mount the whole tree — .venv,
            # .git, src — into the agent. Empty stays empty (no tools mount).
            _td = agent_cfg.get("tools_dir", "")
            tools_dir = os.path.abspath(_td) if _td else ""
            agent_model = agent_cfg.get("model", default_model)
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
            initial_interface = agent_cfg.get("initial_interface", "")
            if initial_interface:
                agent_env["INITIAL_INTERFACE"] = initial_interface
            # Per-agent task-loop iteration cap. Overrides the global
            # OPENLEGION_MAX_ITERATIONS (from dashboard system settings,
            # injected into extra_env above) because env_overrides is
            # applied after extra_env in DockerBackend (see
            # src/host/runtime.py line ~262). Clamping to 1-100 happens
            # agent-side in _clamp_env (src/agent/loop.py).
            agent_max_iters = agent_cfg.get("max_iterations")
            if agent_max_iters is not None:
                agent_env["OPENLEGION_MAX_ITERATIONS"] = str(agent_max_iters)
            # Per-agent output-token cap → LLM_MAX_TOKENS (survives restart).
            set_llm_max_tokens_env(agent_env, agent_cfg)
            from src.shared.limits import set_llm_limits_env

            set_llm_limits_env(agent_env, agent_cfg)
            if agent_id == _OPERATOR_AGENT_ID:
                agent_env["ALLOWED_TOOLS"] = ",".join(_OPERATOR_ALLOWED_TOOLS)
                # NOTE: the operator no longer seeds a boot greeting into
                # its chat transcript. The onboarding modal + in-chat
                # starting-point card cover the first-run experience; a
                # seeded assistant message stacked confusingly underneath
                # that card.
                # Seed the runtime internet/browser access state from the
                # operator's stored permissions so a restart while a
                # toggle is OFF doesn't briefly re-expose the gated tools.
                # ``_load_permissions`` is cheap (JSON read); default-True
                # matches the operator-by-default UX.
                try:
                    from src.cli.config import _load_permissions

                    _op_perms = (
                        _load_permissions()
                        .get(
                            "permissions",
                            {},
                        )
                        .get(_OPERATOR_AGENT_ID, {})
                    )
                    agent_env["OL_INTERNET_ACCESS_ENABLED"] = (
                        "true" if _op_perms.get("can_use_internet", True) else "false"
                    )
                    agent_env["OL_BROWSER_ACCESS_ENABLED"] = (
                        "true" if _op_perms.get("can_use_browser", True) else "false"
                    )
                except Exception:
                    agent_env["OL_INTERNET_ACCESS_ENABLED"] = "true"
                    agent_env["OL_BROWSER_ACCESS_ENABLED"] = "true"

            # Team env vars (TEAM_NAME/TEAM_MD_PATH) are injected by the
            # runtime backend's team env provider (set_team_env_provider)
            # so restart/spawn paths resolve them identically — no
            # per-start plumbing here.

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
                    tools_dir=tools_dir,
                    model=agent_model,
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
                    # Re-plumb the credential resolver + connector catalog
                    # into the fresh backend.
                    self.runtime.set_credential_resolver(
                        vault=self.credential_vault,
                        permissions=self.permissions,
                    )
                    self.runtime.set_connector_store(self.connector_store)
                    self.transport = HttpTransport()
                    # Fresh backend → fresh auth_tokens dict; re-bind so the
                    # replacement transport keeps sending mesh→agent bearers.
                    self.transport.bind_tokens(self.runtime.auth_tokens)
                    _ensure_docker_image()
                    url = self.runtime.start_agent(
                        agent_id=agent_id,
                        role=agent_cfg["role"],
                        tools_dir=tools_dir,
                        model=agent_model,
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
                self.runtime.extra_env.pop("INITIAL_INTERFACE", None)
                self.runtime.extra_env.pop("INITIAL_GREETING", None)
                self.runtime.extra_env.pop("TEAM_MD_PATH", None)
                self.runtime.extra_env.pop("TEAM_NAME", None)
                self.runtime.extra_env.pop("HTTP_PROXY", None)
                self.runtime.extra_env.pop("HTTPS_PROXY", None)
                self.runtime.extra_env.pop("NO_PROXY", None)
            self.router.register_agent(agent_id, url, role=agent_cfg.get("role", ""))
            if isinstance(self.transport, HttpTransport):
                self.transport.register(agent_id, url)
            if self.health_monitor:
                self.health_monitor.register(agent_id)

    async def _close_task_on_dispatch_error(
        self,
        task_id: str,
        exc: BaseException,
    ) -> None:
        """Close a durable task after its mesh→agent dispatch raised.

        Without this, ``_direct_dispatch`` used to return ``f"Error: {e}"`` —
        an opaque string the lane recorded as a *successful* turn, so the
        durable task never got the lane's structured failure close and the
        operator saw a useless ``exception: Error:`` note. We mirror the lane
        watchdog's terminal-status write here: pre-check for an
        already-terminal status (so a real status written by the assignee
        during a race isn't clobbered), then write a REDACTED + truncated
        ``blocker_note``. Best-effort: a failure to close must never mask the
        original error from the lane.

        **Classification (prod incident):** an INFRA shutdown / connection-drop
        (host restart tearing down an in-flight ``/chat``) is NOT a real task
        error — its work is recoverable. Those land as ``blocked`` (recoverable)
        with an honest ``interrupted by host restart`` note, NOT terminal
        ``failed``, so the task is not silently lost. A connection-drop signature
        is **necessary but NOT sufficient** for that reclassification: the exact
        same transport errors also fire when an agent container genuinely
        CRASHES with no host restart. So we additionally gate on the mesh
        actually shutting down (``self._shutting_down``) — only a dispatch that
        drops *because we are restarting* becomes ``blocked``; a drop while NOT
        shutting down stays terminal ``failed`` (see :func:`_is_infra_disconnect`).
        """
        from src.host.orchestration import InvalidStatusTransition

        tasks_store = getattr(getattr(self, "_app", None), "tasks_store", None)
        if tasks_store is None:
            return
        # Reclassify to recoverable ``blocked`` ONLY when BOTH the error is a
        # transport-drop signature AND the mesh is shutting down. A drop while
        # still running is a real failure (e.g. agent container crash) and must
        # stay terminal ``failed`` so operators see it. ``getattr`` is defensive
        # in case a construction path bypasses ``__init__``.
        if _is_infra_disconnect(exc) and getattr(self, "_shutting_down", False):
            target_status = "blocked"
            note = _INFRA_INTERRUPT_NOTE
        else:
            target_status = "failed"
            note = _build_dispatch_error_note(exc)
        loop = asyncio.get_running_loop()
        try:
            pre = await loop.run_in_executor(None, lambda: tasks_store.get(task_id))
        except Exception as pre_err:
            logger.debug(
                "Dispatch-error pre-check for task %s failed: %s",
                task_id,
                pre_err,
            )
            pre = None
        if pre is not None and pre.get("status") in ("done", "failed", "cancelled", "blocked"):
            # Already terminal (assignee or lane watchdog beat us) — do not
            # clobber the real status/note. ``blocked`` is included because
            # ``blocked → failed`` is a *valid* transition: a dispatch
            # transport error must not overwrite an assignee-authored
            # ``blocked`` status (and its meaningful blocker_note) with a
            # generic dispatch note.
            logger.info(
                "Dispatch error: task %s already terminal (status=%s) — skipping close",
                task_id,
                pre.get("status"),
            )
            return
        # Infra interrupts ride the recoverable error code so the dashboard /
        # observers can distinguish "host restart, will recover" from a real
        # dispatch failure.
        error_code = "infra_interrupt" if target_status == "blocked" else "dispatch_error"
        try:
            await loop.run_in_executor(
                None,
                lambda: tasks_store.update_status(
                    task_id,
                    target_status,
                    actor="dispatch",
                    blocker_note=note,
                    extra_payload={"error": error_code},
                ),
            )
        except InvalidStatusTransition as race_err:
            # Benign race: task went terminal between pre-check and UPDATE.
            logger.info(
                "Dispatch-error close race: task %s went terminal (%s) — skipping",
                task_id,
                race_err,
            )
        except Exception as close_err:
            logger.warning(
                "Dispatch-error failed to close task %s: %s",
                task_id,
                close_err,
            )

    def _setup_dispatch(self) -> None:
        from src.host.lanes import _DEFAULT_LANE_TIMEOUT_SECONDS, LaneManager

        # Long-run dispatch drop (prod incident): the mesh→agent ``/chat``
        # call previously inherited transport's 120s default while a task
        # run legitimately takes many minutes (loop MAX_ITERATIONS × 300s
        # tool cap). The inner 120s timeout fired long before the lane's
        # 900s wall-clock cap, returned ``"(no response)"`` as a *success*,
        # and the lane treated it as a completed turn — silently dropping
        # in-flight output. The inner ``/chat`` timeout is set 60s ABOVE the
        # agent's EFFECTIVE lane cap — which may be a per-agent
        # ``watchdog_ttl_seconds`` override (set via
        # ``LaneManager.set_agent_timeout``), not just the module default —
        # computed per dispatch below. That keeps the lane's ``wait_for``
        # watchdog the sole governor of long runs for EVERY agent: on a long
        # run the lane fires first, cancels the dispatch coroutine
        # (cancelling the in-flight httpx request), and runs its existing
        # failed-marking + check_inbox back-edge. The inner timeout is only a
        # backstop. ``_DEFAULT_LANE_TIMEOUT_SECONDS`` is the fallback when no
        # lane manager is wired.

        async def _direct_dispatch(
            agent_name: str,
            message: str,
            origin: "MessageOrigin | None" = None,
            task_id: str | None = None,
            system_note: bool = False,
            **_kwargs,
        ) -> str:
            from src.shared.trace import (
                current_trace_id,
                new_trace_id,
                origin_header,
            )

            # Synthesize a trace id when the dispatch context has none.
            # Lane-dispatched turns (hand_off wakes, recovery wakes) run
            # in the lane worker's bare context, so ``current_trace_id``
            # is unset — the x-trace-id header was never sent, the
            # recipient's LLM calls carried no trace id, and the mesh
            # proxy (whose trace writes are gated on ``req_trace_id``)
            # recorded NO llm rows for exactly the handoff runs that
            # ``inspect_task_run`` exists to diagnose: its execution
            # summary read 0 calls / 0 tokens in production. (Caught
            # live on cake; unit tests seed trace rows manually and
            # could not surface this.)
            tid = current_trace_id.get() or new_trace_id()
            if self.trace_store:
                self.trace_store.record(
                    trace_id=tid,
                    source="dispatch",
                    agent=agent_name,
                    event_type="chat",
                    detail=message[:200],
                    meta={"message_length": len(message)},
                )
            # Phase 2 (session observability): capture the FULL verbatim
            # message centrally and durably, keyed by trace_id + origin, so
            # intent survives container wipes / resets / deploys. Best-effort
            # — a store failure must NEVER break dispatch. Redaction happens
            # at storage inside IntentStore.record. Webhooks are machine
            # origin (system_note / kind=system) — still captured, stamped
            # with origin_kind so the reader can tell human from machine.
            if self.intent_store is not None:
                try:
                    self.intent_store.record(
                        trace_id=tid,
                        origin_kind=(origin.kind if origin else ""),
                        origin_channel=(origin.channel if origin else ""),
                        origin_user=(origin.user if origin else ""),
                        agent=agent_name,
                        message=message,
                        meta={"system_note": system_note},
                    )
                except Exception as _intent_err:
                    logger.debug("intent capture failed: %s", _intent_err)
            import time as _time

            t0 = _time.time()
            extra_headers: dict[str, str] = {"x-trace-id": tid}
            extra_headers.update(origin_header(origin))
            # Bug 2/3 fix: when the wake carried an originating task_id,
            # forward it so the agent's /chat handler can auto-close that
            # specific task once its loop returns.
            if task_id:
                extra_headers["x-task-id"] = task_id
            # System-composed message (wake/cron/webhook — no human typed
            # it): the agent persists it with transcript role ``system``
            # so the dashboard never renders it as the user's own bubble.
            if system_note:
                extra_headers["x-system-wake"] = "1"
            effective_cap = (
                self.lane_manager.timeout_for(agent_name)
                if self.lane_manager is not None
                else _DEFAULT_LANE_TIMEOUT_SECONDS
            )
            try:
                result = await self.transport.request(
                    agent_name,
                    "POST",
                    "/chat",
                    json={"message": message},
                    headers=extra_headers or None,
                    timeout=effective_cap + 60,
                )
                # ``HttpTransport.request`` reports failures (unreachable
                # container, HTTP error, 426 protocol skew) as an error dict
                # instead of raising. Treating that as a successful turn
                # would record an "ok" trace and auto-notify the literal
                # string "(no response)" to the originating human channel.
                # Return the silent token instead: the worker skips the
                # notify, and a task-carrying wake stays ``pending`` for the
                # at-least-once recovery paths.
                if isinstance(result, dict) and result.get("error"):
                    err = str(result.get("error"))
                    duration_ms = int((_time.time() - t0) * 1000)
                    logger.warning(
                        "Dispatch to '%s' failed: %s (status_code=%s)",
                        agent_name,
                        err,
                        result.get("status_code"),
                    )
                    if tid and self.trace_store:
                        self.trace_store.record(
                            trace_id=tid,
                            source="dispatch",
                            agent=agent_name,
                            event_type="chat_response",
                            duration_ms=duration_ms,
                            status="error",
                            meta={"error": err[:200], "status_code": result.get("status_code")},
                        )
                    from src.shared.types import SILENT_REPLY_TOKEN

                    return SILENT_REPLY_TOKEN
                response = result.get("response", "(no response)")
                duration_ms = int((_time.time() - t0) * 1000)
                if tid and self.trace_store:
                    self.trace_store.record(
                        trace_id=tid,
                        source="dispatch",
                        agent=agent_name,
                        event_type="chat_response",
                        duration_ms=duration_ms,
                        status="ok",
                        meta={"response_length": len(response), "response_preview": response[:200]},
                    )
                if self.event_bus:
                    self.event_bus.emit(
                        "message_sent",
                        agent=agent_name,
                        data={"message": message[:200], "response_length": len(response), "source": "dispatch"},
                    )
                    # Lane-dispatched turn (wake/cron/webhook) finished
                    # OUTSIDE any dashboard stream — the reply is already
                    # in the transcript but no open chat view knows.
                    # source="dispatch" tells the JS to do a debounced
                    # history reload only (skip the remote-stream
                    # finalize logic; never bypass the 5s debounce — a
                    # wake burst must not stampede full refetches).
                    self.event_bus.emit(
                        "chat_done",
                        agent=agent_name,
                        data={"source": "dispatch"},
                    )
                return response
            # NOTE: ``except Exception`` (NOT ``BaseException``/bare) is
            # deliberate — ``asyncio.CancelledError`` MUST propagate so the
            # lane's 900s ``wait_for`` watchdog can still cancel a long run and
            # run its structured timeout close. Swallowing it here would wedge
            # the lane.
            except Exception as e:
                duration_ms = int((_time.time() - t0) * 1000)
                if tid and self.trace_store:
                    self.trace_store.record(
                        trace_id=tid,
                        source="dispatch",
                        agent=agent_name,
                        event_type="chat_response",
                        duration_ms=duration_ms,
                        status="error",
                        error=str(e),
                    )
                # RC-2 fix: a transport/agent error used to be returned as a
                # bare ``f"Error: {e}"`` string. The lane recorded that as a
                # *successful* turn, so the durable task never went through a
                # structured failure close — the failure surfaced only as an
                # opaque ``exception: Error:`` note with no actionable signal,
                # AND the raw (untrusted) error string could leak secrets.
                #
                # Now: when the wake carried an originating ``task_id``, close
                # that durable task to ``failed`` with a REDACTED + truncated
                # ``blocker_note`` (mirrors the lane watchdog's terminal write,
                # with an already-terminal pre-check to avoid clobbering a real
                # status on a race). The string we return is also redacted so
                # the no-task path (manual chat / heartbeat) stays non-leaky.
                note = _build_dispatch_error_note(e)
                if task_id:
                    await self._close_task_on_dispatch_error(task_id, e)
                return note

        async def _direct_steer(
            agent_name: str,
            message: str,
            system_note: bool = False,
        ) -> dict:
            try:
                # Same marker as _direct_dispatch: a system-composed steer
                # (e.g. blackboard watch hitting a BUSY agent) must drain
                # from the steer queue as a ``system`` transcript row, not
                # a "[steer]" user bubble.
                headers = {"x-system-wake": "1"} if system_note else None
                return await self.transport.request(
                    agent_name,
                    "POST",
                    "/chat/steer",
                    json={"message": message},
                    headers=headers,
                )
            except Exception as e:
                return {"injected": False, "error": str(e)}

        self.lane_manager = LaneManager(
            dispatch_fn=_direct_dispatch,
            steer_fn=_direct_steer,
            trace_store=self.trace_store,
            notify_fn=self._handle_notify_origin,
            quarantine_check=(self.health_monitor.is_quarantined if self.health_monitor is not None else None),
        )
        # Bug 1: hand the lane queue-depth lookup to the health monitor so
        # the staleness check (reachable+busy+no-tick → unhealthy) has
        # the data it needs. Health monitor was built before the lane.
        if self.health_monitor is not None:
            self.health_monitor.set_queue_depth_fn(self.lane_manager.get_queue_depth)

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
            await asyncio.gather(
                *(ch.send_notification(notification) for ch in self._active_channels), return_exceptions=True
            )

    async def _handle_notify_origin(
        self,
        origin: "MessageOrigin",
        message: str,
        agent_name: str = "",
    ) -> bool:
        """Route a completed task result back to the originating channel+user.

        Called by the lane worker after a hand-off task completes with
        auto_notify=True.  Delivers the result to the specific user on the
        specific channel that originally dispatched the work.

        Prefixes the message with ``[agent_name]`` for parity with the direct
        dispatch path (``Channel.handle_message`` labels responses the same
        way), so users can see which agent produced the reply.

        Telegram/Discord/Slack adapters each run on their own event loop in a
        daemon thread.  The lane worker calling this is on the dispatch loop,
        so we hop onto the channel's own loop via ``run_coroutine_threadsafe``
        to avoid cross-loop client reuse.  WhatsApp has no dedicated loop
        (it's webhook-driven), so we call directly.

        Returns ``True`` iff the send succeeded — the chain-outcome retry path
        gates on this. The lane caller ignores the return (backward compatible).
        """
        if not origin or not self.channel_manager:
            return False
        channel_type = origin.channel
        user = origin.user
        if not channel_type or not user:
            return False
        ch = self.channel_manager._channel_map.get(channel_type)
        if ch is None:
            logger.debug(
                "Origin channel %s not connected — dropping notification to %s",
                channel_type,
                user,
            )
            return False

        labelled = f"[{agent_name}] {message}" if agent_name else message

        channel_loop = getattr(ch, "_channel_loop", None)
        try:
            if channel_loop is not None and channel_loop.is_running():
                concurrent_fut = asyncio.run_coroutine_threadsafe(
                    ch.send_to_user(user, labelled),
                    channel_loop,
                )
                await asyncio.wrap_future(concurrent_fut)
            else:
                await ch.send_to_user(user, labelled)
            return True
        except Exception as e:
            logger.warning(
                "send_to_user(%s, %s) failed: %s",
                channel_type,
                user,
                e,
            )
            return False

    async def _deliver_chain_outcome(
        self,
        root: dict,
        kind: str,
        summary: str,
    ) -> bool:
        """Deliver one terminal outcome for a user chain to its originator.

        The durable surface is a ``notification``-role row in the
        operator's chat transcript (written via the agent's ``/chat/note``
        endpoint) — the watcher only claims the delivery when this returns
        True, so a failed write is retried rather than silently lost. The
        live surface is the existing ``notification`` DashboardEvent (amber
        bubble + toast in the operator thread). A paired chat channel
        (telegram/discord/slack/whatsapp) additionally gets a best-effort
        push so a non-dashboard originator hears back in their own channel.
        Targeting uses ONLY the root's first-party human origin.

        Async on purpose: it runs on the ChainWatcher's own event loop
        (``_run_deliver`` awaits coroutine delivers) and the transport
        keeps one httpx client per loop, so awaiting here is safe.
        """
        origin = root.get("origin") or {}
        channel = origin.get("channel") or "dashboard"
        user = origin.get("user") or ""
        root_id = root.get("id") or ""
        assignee = root.get("assignee") or ""
        # Raw worker output — bound it so a large result_summary can't
        # bloat the transcript note or the channel message. (Display
        # surfaces are autoescaped; this is a size guard, not a safety
        # one.)
        summary = (summary or "").strip()[:1500]
        root_title = (root.get("title") or "your request").strip()[:80]
        if kind == "done":
            title = "✅ Task complete"
            body = summary or "Your request finished."
        elif kind == "stall":
            title = "⏳ Taking longer than expected"
            body = f"No progress on '{root_title}' for a while — it may be stuck. Want me to check in?"
        else:
            title = "⚠️ Task failed"
            body = summary or "Your request hit a failure and stopped."
        # 1. Durable surface — a notification-role row in the operator's
        #    chat transcript. This IS the guarantee: the watcher claims
        #    the delivery only on a positive ack, so any failure here
        #    returns False and the next sweep retries. The transport
        #    NEVER raises for the failures that matter (HTTP errors,
        #    timeouts, connect failures come back as {"error": ...}
        #    dicts), so success is judged on the returned dict + the
        #    endpoint's explicit {"ok": true}.
        if self.transport is None:
            return False
        note_text = f"{title}\n{body}"
        try:
            result = await self.transport.request(
                "operator",
                "POST",
                "/chat/note",
                json={"message": note_text},
                timeout=15,
            )
        except Exception as e:
            logger.warning(
                "chain outcome note write failed for %s: %s",
                root_id,
                e,
            )
            return False
        if not isinstance(result, dict) or result.get("error") or not result.get("ok"):
            status_code = result.get("status_code") if isinstance(result, dict) else None
            if status_code == 404:
                # The deploy trap: host updated, agent image not rebuilt —
                # /chat/note doesn't exist in the running container and
                # delivery would retry silently forever. Make it loud.
                logger.error(
                    "chain outcome delivery for %s got 404 from /chat/note "
                    "— the operator agent image predates this endpoint. "
                    "Rebuild it (docker build -t openlegion-agent:latest "
                    "-f Dockerfile.agent .) and restart; delivery retries "
                    "until then.",
                    root_id,
                )
            else:
                logger.warning(
                    "chain outcome note rejected for %s: %s",
                    root_id,
                    result,
                )
            return False
        # Live surface — the chat UI renders this as the amber
        # notification bubble + toast in the operator thread (same
        # handler notify_user uses). Best-effort: the durable row above
        # already guarantees the user sees it on next history load.
        if self.event_bus is not None:
            try:
                self.event_bus.emit(
                    "notification",
                    agent="operator",
                    data={
                        "message": note_text,
                        "root_task_id": root_id,
                        "outcome": kind,
                    },
                )
            except Exception as e:
                logger.debug("chain outcome notification emit failed: %s", e)

        # 1b. Post-completion verification wake (success path only). The
        #     user already has the result (bell above); the operator gets
        #     ONE wake per completed chain to verify side effects are
        #     REAL (artifacts/files/PRs, not just 'done' statuses), rate
        #     the stages, and speak up only if something is wrong.
        #     failed/blocked chains are excluded — the mesh recovery wake
        #     already fires per failed task. Best-effort: a wake failure
        #     never flips this delivery to False (the bell is the
        #     guarantee), and the rate cap keeps a completion burst from
        #     monopolizing the operator's lane.
        if kind == "done":
            self._maybe_wake_operator_verification(root, root_id, root_title)

        # 2. Push to a paired chat channel for non-dashboard originators, with
        #    bounded retry so a transient channel failure (network blip, rate
        #    limit) doesn't silently drop it. Runs as a background task so it
        #    never blocks the watcher sweep; the durable transcript note
        #    (above) remains the cross-restart guarantee, so a channel down
        #    past the retries still leaves the outcome in the operator
        #    thread.
        if channel and channel != "dashboard" and user:
            try:
                from src.shared.types import MessageOrigin

                origin_obj = MessageOrigin(
                    kind="human",
                    channel=channel,
                    user=user,
                )
                push = asyncio.create_task(
                    self._channel_push_with_retry(
                        origin_obj,
                        f"{title}\n{body}",
                        assignee,
                    )
                )
                # Retain a strong ref until it completes (asyncio only holds a
                # weak ref; without this the task can be GC'd mid-send).
                self._pending_chain_pushes.add(push)
                push.add_done_callback(self._pending_chain_pushes.discard)
            except Exception as e:
                logger.debug(
                    "chain outcome channel push schedule failed: %s",
                    e,
                )
        return True

    def _system_signal_producer(self, evt: dict) -> None:
        """Reroute formerly bell-only system signals into the operator
        thread (durable /chat/note row + live notification event).

        Two signals qualify (everything else the bell carried has its own
        surface — chat cards, Needs-You panel, fleet badges):

        - ``connection_refresh_failed`` — a third-party OAuth connection's
          refresh was hard-rejected (user revoked access).
        - ``health_change`` → ``quarantined`` — repeated auth failures
          stopped the lane; carries the remediation text.

        EventBus listeners run synchronously on the emitter's thread, so
        the note POST is marshalled onto the dispatch loop fire-and-forget.
        Unlike chain delivery there is no claim/retry sweep behind this —
        after the bounded retries the signal is accepted-lost to the
        transcript (the Connectors page / fleet badges remain).
        """
        try:
            etype = evt.get("type")
            data = evt.get("data") or {}
            if etype == "connection_refresh_failed":
                conn = (data.get("connection") or "connection").strip()
                provider = (data.get("provider") or "").strip()
                label = provider.capitalize() if provider else "OAuth"
                err = (data.get("error") or "").strip()
                text = (
                    f"🔌 {label} connection '{conn}' needs reconnecting\n"
                    "Token refresh was rejected by the provider — open the "
                    "Connectors page and reconnect." + (f" ({err[:200]})" if err else "")
                )
            elif etype == "health_change":
                if (data.get("current") or "") != "quarantined":
                    return
                agent = (evt.get("agent") or "agent").strip()
                reason = (data.get("reason") or "").strip()
                text = (
                    f"⛔ Agent '{agent}' quarantined: credential broken\n"
                    + (f"{reason}. " if reason else "")
                    + "The lane has stopped dispatching new work. Rotate "
                    "the credential or run edit_agent to pick a compatible "
                    "model, then restart the agent."
                )
            else:
                return
            if self._dispatch_loop is None or self.transport is None:
                return
            asyncio.run_coroutine_threadsafe(
                self._operator_note_with_retry(text),
                self._dispatch_loop,
            )
        except Exception as e:
            logger.debug("system signal reroute failed: %s", e)

    async def _operator_note_with_retry(
        self,
        text: str,
        *,
        attempts: int = 3,
        backoff_s: float = 3.0,
    ) -> None:
        """Write a system-signal note to the operator transcript, retrying
        transient failures, then emit the live notification event. Same
        success contract as chain delivery: error-dict-aware + explicit
        {ok: true} ack (the transport never raises for HTTP failures)."""
        for i in range(attempts):
            try:
                result = await asyncio.wait_for(
                    self.transport.request(
                        "operator",
                        "POST",
                        "/chat/note",
                        json={"message": text},
                        timeout=15,
                    ),
                    timeout=30,
                )
                if isinstance(result, dict) and result.get("ok") and not result.get("error"):
                    if self.event_bus is not None:
                        try:
                            self.event_bus.emit(
                                "notification",
                                agent="operator",
                                data={"message": text},
                            )
                        except Exception as e:
                            logger.debug("system note emit failed: %s", e)
                    return
                logger.debug("system note write rejected: %s", result)
            except Exception as e:
                logger.debug("system note attempt failed: %s", e)
            if i < attempts - 1:
                await asyncio.sleep(backoff_s * (i + 1))
        logger.warning(
            "operator note for system signal gave up after %d attempts: %s",
            attempts,
            text.splitlines()[0][:120],
        )

    async def _channel_push_with_retry(
        self,
        origin: "MessageOrigin",
        message: str,
        agent_name: str,
        *,
        attempts: int = 3,
        backoff_s: float = 3.0,
        timeout_s: float = 30.0,
    ) -> None:
        """Deliver a chain-outcome push to a chat channel, retrying transient
        failures. Best-effort: the operator-thread transcript note is the
        durable guarantee, so after the attempts are exhausted we log and
        stop (no hot loop).

        Each attempt is bounded by ``timeout_s`` (mirrors the lane's
        ``_NOTIFY_FORWARD_TIMEOUT``) so a wedged channel adapter can't hang the
        loop forever — without it the retry count is meaningless and this
        background task (held by a strong ref) would leak for the process life.
        """
        for i in range(attempts):
            try:
                ok = await asyncio.wait_for(
                    self._handle_notify_origin(origin, message, agent_name),
                    timeout=timeout_s,
                )
                if ok:
                    return
            except asyncio.TimeoutError:
                logger.debug("chain outcome channel push attempt timed out")
            except Exception as e:
                logger.debug("chain outcome channel push attempt failed: %s", e)
            if i < attempts - 1:
                await asyncio.sleep(backoff_s * (i + 1))
        logger.warning(
            "chain outcome channel push to %s/%s gave up after %d attempts "
            "(operator-thread note delivered as fallback)",
            origin.channel,
            origin.user,
            attempts,
        )

    # Verification-wake rate cap: at most N wakes per sliding window.
    # Mirrors the mesh-side recovery-wake storm guard (same numbers) —
    # one operator turn per completed chain is the point, but a backlog
    # of chains completing together must not queue an interrupt storm.
    _VERIFY_WAKE_MAX = 5
    _VERIFY_WAKE_WINDOW_S = 600.0

    def _maybe_wake_operator_verification(
        self,
        root: dict,
        root_id: str,
        root_title: str,
    ) -> None:
        """Wake the operator once to verify a COMPLETED user chain.

        Fire-and-forget onto the dispatch loop (this runs on the chain
        watcher's own loop — never block its sweep on a busy operator
        lane). ``auto_notify=False``: the user already got the terminal
        delivery; if verification finds a problem the operator decides
        to speak via notify_user. ``task_id`` is deliberately NOT
        threaded — a verification turn must never auto-close anything.
        """
        if self.lane_manager is None or self._dispatch_loop is None:
            return
        now = time.time()
        while self._verify_wake_times and now - self._verify_wake_times[0] > self._VERIFY_WAKE_WINDOW_S:
            self._verify_wake_times.popleft()
        if len(self._verify_wake_times) >= self._VERIFY_WAKE_MAX:
            logger.warning(
                "verification wake for chain %s suppressed: cap (%d/%ds) reached — the heartbeat rating step covers it",
                root_id,
                self._VERIFY_WAKE_MAX,
                int(self._VERIFY_WAKE_WINDOW_S),
            )
            return
        self._verify_wake_times.append(now)
        from src.shared.utils import sanitize_for_prompt

        msg = (
            f"User chain {root_id} ('{sanitize_for_prompt(root_title)}') "
            "completed and the user was ALREADY notified of the result — "
            "do NOT re-announce it. Verify it now: confirm the promised "
            "side effects really exist (artifacts, files, merged PRs — "
            "check externally visible effects with your tools, not just "
            f"task statuses), use workflow_snapshot('{root_id}') and "
            "inspect_task_run on any stage that looks shallow, then "
            "rate_delivery the done stages. Message the user ONLY if "
            "verification fails or adds something material."
        )
        origin_dict = root.get("origin") or {}
        try:
            from src.shared.types import MessageOrigin

            wake_origin = MessageOrigin(
                kind="human",
                channel=str(origin_dict.get("channel") or ""),
                user=str(origin_dict.get("user") or ""),
            )
            asyncio.run_coroutine_threadsafe(
                self.lane_manager.enqueue(
                    "operator",
                    msg,
                    mode="followup",
                    origin=wake_origin,
                    auto_notify=False,
                    system_note=True,
                ),
                self._dispatch_loop,
            )
        except Exception as e:
            logger.warning(
                "verification wake for chain %s failed: %s",
                root_id,
                e,
            )

    def _start_mesh_server(self) -> None:
        import uvicorn

        from src.host.server import create_mesh_app
        from src.host.webhooks import WebhookManager

        mesh_port = self.cfg["mesh"]["port"]

        # Bind the primary port on THIS (main) thread BEFORE building the app
        # or launching uvicorn. A bind failure must be fatal to the whole
        # process — see _bind_mesh_socket for the outage this prevents. uvicorn
        # would otherwise swallow the OSError inside its daemon thread, leaving
        # the process "running" but bound to nothing while the health check
        # answered off a stale listener. Binding first also avoids wiring the
        # whole app only to throw it away.
        try:
            mesh_socket = _bind_mesh_socket("0.0.0.0", mesh_port)
        except OSError as exc:
            echo_fail(
                f"Mesh server could not bind port {mesh_port}: {exc}. "
                f"Another process may already own it. Try: openlegion stop"
            )
            logger.error("FATAL: mesh failed to bind port %d: %s", mesh_port, exc)
            # Tear down any agent containers we started so a failed-to-bind
            # boot doesn't leave orphans behind for systemd to restart over.
            try:
                self.runtime.stop_all()
            except Exception as cleanup_err:
                logger.debug("Cleanup after bind failure errored: %s", cleanup_err)
            sys.exit(1)

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

        # Mesh-side gateway for remote (http) MCP connectors — holds no
        # state beyond a discovery cache; auth resolves from the vault
        # per call (tokens never enter containers).
        from src.host.mcp_gateway import MCPGateway

        self.mcp_gateway = MCPGateway(
            self.connector_store,
            self.credential_vault,
        )

        app = create_mesh_app(
            self.blackboard,
            self.pubsub,
            self.router,
            self.permissions,
            self.credential_vault,
            self.cron_scheduler,
            self.runtime,
            self.transport,
            auth_tokens=self.runtime.auth_tokens,
            trace_store=self.trace_store,
            intent_store=self.intent_store,
            lifecycle_store=self.lifecycle_store,
            event_bus=self.event_bus,
            health_monitor=self.health_monitor,
            cost_tracker=self.cost_tracker,
            notify_fn=self._handle_notify,
            teams_store=self.teams_store,
            thread_store=self.thread_store,
            lane_manager=self.lane_manager,
            dispatch_loop=self._dispatch_loop,
            wallet_service_ref=wallet_ref,
            api_key_manager=self._api_key_manager,
            cfg=self.cfg,
            connector_store=self.connector_store,
            mcp_gateway=self.mcp_gateway,
        )
        app.include_router(webhook_manager.create_router())
        self.health_monitor._cleanup_agent = app.cleanup_agent  # type: ignore[attr-defined]

        # Bug 4: lane watchdog needs the durable task store so a per-task
        # timeout can mark the row ``failed`` (back-edge inbox event fires)
        # instead of leaving the originator waiting forever. The store is
        # created inside ``create_mesh_app`` so we wire it here after the
        # fact rather than threading it through the lane constructor.
        _tasks_store_ref = getattr(app, "tasks_store", None)
        if _tasks_store_ref is not None and self.lane_manager is not None:
            self.lane_manager.set_tasks_store(_tasks_store_ref)
        # Hold the same store instance for the chain watcher (started in
        # _start_background) so terminal-chain delivery reads/writes the
        # very tasks DB the mesh transitions tasks in.
        self._tasks_store_ref = _tasks_store_ref
        # Wire the mesh's back-edge writer into the lane watchdog so a
        # lane-timeout failure produces a ``task_failed`` inbox event for
        # the originator AND triggers the wake-on-event chain. Without
        # this the durable status update lands but the originating agent
        # never learns until its next heartbeat.
        _back_edge_fn = getattr(app, "_write_task_event_back_edge", None)
        if _back_edge_fn is not None and self.lane_manager is not None:
            self.lane_manager.set_back_edge_fn(_back_edge_fn)

        # Per-agent watchdog override. Workflow-stage agents that run a
        # bounded fast loop can opt into a tighter cap than the 15-min
        # default via ``settings.watchdog_ttl_seconds`` in agents.yaml.
        # Workers without the field stay on the module default.
        if self.lane_manager is not None:
            try:
                from src.cli.config import _load_config as _load_cfg_for_lanes

                _cfg = _load_cfg_for_lanes()
                _agents_cfg = _cfg.get("agents", {}) or {}
                for aid, entry in _agents_cfg.items():
                    if not isinstance(entry, dict):
                        continue
                    settings = entry.get("settings") or {}
                    ttl = settings.get("watchdog_ttl_seconds")
                    if ttl is None:
                        continue
                    try:
                        self.lane_manager.set_agent_timeout(aid, int(ttl))
                    except (TypeError, ValueError):
                        logger.warning(
                            "Invalid watchdog_ttl_seconds for %s: %r — ignored",
                            aid,
                            ttl,
                        )
            except Exception as e:
                logger.warning(
                    "Per-agent watchdog override wiring failed: %s",
                    e,
                )

        self._init_channel_manager()

        from src.dashboard.server import create_dashboard_router, create_spa_catchall_router

        # Task 9 — pass the mesh's pending-action and tasks stores into
        # the dashboard so the new ``/api/workplace/*`` endpoints can
        # render the Board tab without a second HTTP hop.
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
            pending_actions=getattr(app, "pending_actions", None),
            tasks_store=getattr(app, "tasks_store", None),
            help_requests_store=getattr(app, "help_requests_store", None),
            summaries_store=getattr(app, "summaries_store", None),
            connector_store=self.connector_store,
            mcp_gateway=self.mcp_gateway,
            intent_store=self.intent_store,
            teams_store=self.teams_store,
            thread_store=self.thread_store,
        )
        app.include_router(dashboard_router)
        app.include_router(create_spa_catchall_router())  # Must be last — SPA deep linking
        self._app = app

        server_config = uvicorn.Config(app, host="0.0.0.0", port=mesh_port, log_level="warning")
        self._server = uvicorn.Server(server_config)
        # Hand uvicorn the already-bound socket so it never re-binds (and so
        # the bind result was already asserted, fatally, above).
        mesh_thread = threading.Thread(
            target=self._server.run,
            kwargs={"sockets": [mesh_socket]},
            daemon=True,
        )
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
            echo_fail(f"Mesh server failed to start on port {mesh_port}. Port may be in use. Try: openlegion stop")
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
            # Task 2b: stamp cron origin so downstream gates can
            # distinguish a scheduled tick from a human-initiated wake.
            from src.shared.trace import new_trace_id
            from src.shared.types import MessageOrigin

            origin = MessageOrigin(kind="cron", channel="cron", user="")
            result = await self.async_dispatch(
                agent_name,
                message,
                trace_id=new_trace_id(),
                origin=origin,
                system_note=True,
            )
            return result

        async def fetch_heartbeat_context(agent_name: str) -> dict:
            try:
                return await self.transport.request(
                    agent_name,
                    "GET",
                    "/heartbeat-context",
                    timeout=10,
                )
            except Exception as e:
                logger.debug("Failed to fetch heartbeat context for '%s': %s", agent_name, e)
                return {}

        async def invoke_tool(agent_name: str, tool_name: str, params: dict) -> dict:
            try:
                return await self.transport.request(
                    agent_name,
                    "POST",
                    "/invoke",
                    json={"tool": tool_name, "params": params},
                    timeout=30,
                )
            except Exception as e:
                logger.warning("Failed to invoke tool '%s' on '%s': %s", tool_name, agent_name, e)
                return {"error": str(e)}

        async def heartbeat_dispatch(
            agent_name: str,
            message: str,
            *,
            force_llm: bool = False,
        ) -> dict:
            """Dispatch heartbeat via dedicated /heartbeat endpoint.

            Task 2b: stamp ``kind="heartbeat"`` origin so the agent's
            tools (and downstream gates) can identify self-triggered
            heartbeat work versus a human or cron wake.

            Bug 6 (codex P2 r2): ``force_llm`` is forwarded to the
            agent via the ``x-force-llm`` header so
            ``AgentLoop.execute_heartbeat`` skips its own ``empty
            HEARTBEAT.md → no_heartbeat_rules`` short-circuit. Without
            this, bypassing only the cron-side skip leaves
            pipeline-kicker agents silent because the agent-side check
            still fires.
            """
            from src.shared.trace import origin_header, trace_headers
            from src.shared.types import MessageOrigin

            origin = MessageOrigin(kind="heartbeat", channel="heartbeat", user="")
            headers = trace_headers()
            headers.update(origin_header(origin))
            if force_llm:
                headers["x-force-llm"] = "true"
            try:
                return await self.transport.request(
                    agent_name,
                    "POST",
                    "/heartbeat",
                    json={"message": message},
                    timeout=120,
                    headers=headers,
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
            health_monitor=self.health_monitor,
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
            "heartbeat_schedule",
            CronScheduler.DEFAULT_HEARTBEAT_SCHEDULE,
        )
        for agent_id in agents_cfg:
            # Operator heartbeat is faster than the fleet default so
            # fleet-health regressions surface within minutes rather
            # than hours. The previous 1h cadence was set when the
            # heartbeat path was new; with the inspect/observation
            # surface stable, 15m is the right tradeoff between
            # responsiveness and LLM cost.
            agent_schedule = "every 15m" if agent_id == _OPERATOR_AGENT_ID else schedule
            self.cron_scheduler.ensure_heartbeat(agent_id, agent_schedule)

    def _reconcile_work_summary_jobs(self) -> None:
        """Ensure each active team has a daily ``compose_work_summary``
        cron job, and prune jobs for teams that no longer exist or are
        archived. Idempotent — runs at every mesh startup so the cron
        set converges with the on-disk team metadata.

        Per-team cadence override: ``settings.summary_schedule`` on
        the team's stored ``settings`` dict (TeamStore). Defaults to
        ``DEFAULT_SUMMARY_SCHEDULE`` (daily at 9am) when absent.

        IMPORTANT — runtime settings-edit contract: the mesh team
        endpoints only mutate the team's *description* (PUT
        ``/mesh/teams/{name}/context``) and *goal* (POST
        ``/mesh/teams/{name}/goal``) at runtime; there is no live
        endpoint for editing ``settings.summary_schedule``. Operators
        who need to change a cron cadence today edit the stored
        settings + restart the mesh — this reconcile catches the
        drift and reschedules via ``_validate_schedule`` +
        ``_compute_next_run`` below. If a future PR adds a live
        settings-edit endpoint, it MUST also call
        ``ensure_summary_job`` / mutate the existing job's schedule
        + ``_compute_next_run`` to keep runtime behavior aligned.
        """
        if not self.cron_scheduler:
            return
        try:
            teams = self.teams_store.list_teams()
        except Exception as e:
            logger.warning("work-summary cron bootstrap failed to load teams: %s", e)
            return
        # Build the desired set + ensure each. Per-team cadence
        # overrides live in ``team.settings.summary_schedule`` so the
        # extension point is the canonical settings dict, not a
        # special top-level metadata field. The default schedule is
        # baked into ``CronScheduler.ensure_summary_job`` (daily 9am).
        #
        # ``ensure_summary_job`` returns an EXISTING job unchanged
        # when one already exists — so a metadata edit that changes
        # the schedule wouldn't propagate without help. Compare the
        # existing job's schedule to the desired value and reschedule
        # via ``update_job`` when they drift.
        active_team_names: set[str] = set()
        for name, meta in teams.items():
            if (meta.get("status") or "active") == "archived":
                continue
            active_team_names.add(name)
            settings = meta.get("settings") or {}
            schedule = settings.get("summary_schedule")
            try:
                existing_before = self.cron_scheduler.find_summary_job(
                    "team",
                    name,
                )
                job = self.cron_scheduler.ensure_summary_job(
                    scope_kind="team",
                    scope_id=name,
                    schedule=schedule,
                )
                # Schedule-drift sync. ``ensure_summary_job`` only
                # creates when absent; if a per-team metadata edit
                # changed the cadence between boots, push the new
                # schedule onto the existing job. Safe to mutate
                # synchronously here — reconcile runs in
                # ``_start_background`` BEFORE the cron loop thread
                # starts, so the async per-job lock isn't yet
                # contended (only the loop's tick coroutine acquires
                # it). Validate the schedule first to surface a bad
                # team-metadata value as a warning, not a crash on
                # the next tick.
                desired = schedule or self.cron_scheduler.DEFAULT_SUMMARY_SCHEDULE
                if existing_before is not None and job.schedule != desired:
                    validation_error = self.cron_scheduler._validate_schedule(
                        desired,
                    )
                    if validation_error:
                        logger.warning(
                            "team %s has invalid summary_schedule %r: %s — keeping existing schedule %r",
                            name,
                            desired,
                            validation_error,
                            job.schedule,
                        )
                    else:
                        previous = job.schedule
                        job.schedule = desired
                        self.cron_scheduler._compute_next_run(job)
                        self.cron_scheduler._save()
                        logger.info(
                            "rescheduled work-summary cron for team %s: %s → %s",
                            name,
                            previous,
                            desired,
                        )
            except Exception as e:
                logger.warning(
                    "ensure_summary_job for team %s failed: %s",
                    name,
                    e,
                )
        # Prune orphan summary jobs for teams that no longer exist
        # (renamed, deleted, archived). Tool jobs whose tool_params we
        # can't parse are left alone — they're not ours to manage.
        import json as _json

        for job in list(self.cron_scheduler.jobs.values()):
            if job.tool_name != "compose_work_summary":
                continue
            try:
                params = _json.loads(job.tool_params or "{}")
            except (_json.JSONDecodeError, TypeError):
                continue
            if params.get("scope_kind") != "team":
                continue
            scope_id = params.get("scope_id")
            if scope_id and scope_id not in active_team_names:
                try:
                    self.cron_scheduler.remove_job(job.id)
                    logger.info(
                        "pruned work-summary cron for archived/deleted team %s",
                        scope_id,
                    )
                except Exception as e:
                    logger.debug(
                        "failed to prune orphan summary cron %s: %s",
                        job.id,
                        e,
                    )

    def _start_background(self) -> None:
        self._reconcile_heartbeats()
        self._reconcile_work_summary_jobs()

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

        # Start chain watcher — delivers a guaranteed terminal outcome for
        # user-originated task chains so the operator can hand off and
        # release instead of block-watching a multi-hop pipeline.
        if self._tasks_store_ref is not None and self.transport is not None:
            from src.host.chain_watcher import ChainWatcher

            self.chain_watcher = ChainWatcher(
                self._tasks_store_ref,
                self._deliver_chain_outcome,
            )

            def run_chain_watcher():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.chain_watcher.start())

            threading.Thread(target=run_chain_watcher, daemon=True).start()

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
                self.event_bus.emit(
                    "message_received", agent=agent_name, data={"message": message[:200], "source": "channel_stream"}
                )
            if self.trace_store:
                from src.shared.trace import new_trace_id

                self.trace_store.record(
                    trace_id=new_trace_id(),
                    source="channel_stream",
                    agent=agent_name,
                    event_type="chat",
                    detail=message[:120],
                )
            async for event in self.transport.stream_request(
                agent_name,
                "POST",
                "/chat/stream",
                json={"message": message},
                timeout=120,
            ):
                # Liveness sentinel from the transport keepalive-forwarding —
                # drop it so it never lands in a channel's assembled response.
                if isinstance(event, dict) and event.get("type") == "keepalive":
                    continue
                if self.event_bus and isinstance(event, dict):
                    etype = event.get("type", "")
                    if etype in ("tool_start", "tool_result"):
                        self.event_bus.emit(
                            etype, agent=agent_name, data={k: v for k, v in event.items() if k != "type"}
                        )
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
                service,
                key,
                system=is_system_credential(service),
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
            self.cfg,
            self.async_dispatch,
            self.router.agent_registry,
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
        try:
            agent_teams = self.teams_store.agent_team_map()
            teams_cfg = self.teams_store.list_teams()
        except Exception:
            agent_teams, teams_cfg = {}, {}
        mesh_port = self.cfg["mesh"]["port"]

        # ── Services ──
        echo_header("OpenLegion")
        echo_ok(f"Dashboard: http://localhost:{mesh_port}")
        if hasattr(self.runtime, "browser_service_url") and self.runtime.browser_service_url:
            echo_ok(f"Browser service: {self.runtime.browser_service_url}")
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

        # ── Teams ──
        if teams_cfg:
            assigned = set(agent_teams.keys())
            standalone = [a for a in active_agents if a not in assigned]

            for pname in sorted(teams_cfg.keys()):
                members = [a for a in active_agents if agent_teams.get(a) == pname]
                if members:
                    click.echo(f"\n  Team [{pname}]: {', '.join(members)}")

            if standalone:
                click.echo(f"\n  Solo: {', '.join(standalone)}")

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
