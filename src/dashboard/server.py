"""Dashboard API router: fleet overview, costs, comms (blackboard + pubsub), traces, management.

Serves the SPA template and static files, plus JSON API endpoints
consumed by the Alpine.js frontend.  All data comes from live Python
objects — no HTTP round-trips through mesh endpoints.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import sqlite3
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from jinja2 import Environment, FileSystemLoader

from src.cli.proxy import build_proxy_env_vars, resolve_agent_proxy
from src.dashboard.auth import verify_session_cookie
from src.shared.utils import friendly_streaming_error, sanitize_for_prompt, setup_logging

if TYPE_CHECKING:
    from src.dashboard.events import EventBus
    from src.host.costs import CostTracker
    from src.host.health import HealthMonitor
    from src.host.mesh import Blackboard
    from src.host.traces import TraceStore

logger = setup_logging("dashboard.server")

_HERE = Path(__file__).resolve().parent
_TEMPLATES_DIR = _HERE / "templates"
_STATIC_DIR = _HERE / "static"
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _get_builtin_tool_names() -> frozenset[str]:
    """Return the names of all built-in agent tools by scanning the builtins package.

    Uses a regex over source files rather than executing code, so it is safe to
    call from the dashboard host process without any side effects.  Result is
    cached in a module-level variable after the first call.
    """
    if _get_builtin_tool_names._cache is not None:
        return _get_builtin_tool_names._cache
    builtins_dir = Path(__file__).parent.parent / "agent" / "builtins"
    names: set[str] = set()
    if builtins_dir.exists():
        for py_file in builtins_dir.glob("**/*.py"):
            if py_file.name.startswith("_"):
                continue
            try:
                text = py_file.read_text()
                names.update(re.findall(r'@skill\s*\(\s*name\s*=\s*["\']([^"\']+)["\']', text))
            except OSError:
                pass
    _get_builtin_tool_names._cache = frozenset(names)
    return _get_builtin_tool_names._cache


_get_builtin_tool_names._cache = None  # type: ignore[attr-defined]


def _compute_asset_version() -> str:
    """Hash all static files + the template to produce a cache-bust version string.

    Changes to ANY dashboard file (JS, CSS, HTML template) produce a new hash,
    which changes the query parameter on all static file URLs, forcing browsers
    to fetch fresh copies.  Computed once at import time.
    """
    h = hashlib.sha256()
    for pattern in ("static/**/*", "templates/**/*"):
        for f in sorted(_HERE.glob(pattern)):
            if f.is_file():
                h.update(f.read_bytes())
    return h.hexdigest()[:12]


ASSET_VERSION = _compute_asset_version()


def _verify_dashboard_auth(request: Request) -> None:
    """Verify the ol_session cookie on dashboard API requests."""
    error = verify_session_cookie(request.cookies.get("ol_session", ""))
    if error is not None:
        raise HTTPException(401, error)


def _parse_positive_float(value: Any, field: str, fallback: float) -> float:
    """Validate *value* as a positive float, returning *fallback* if None."""
    if value is None:
        return fallback
    try:
        result = float(value)
        if result <= 0:
            raise ValueError
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail=f"{field} must be a positive number")
    return result


def _log_cron_task_exception(task: object) -> None:
    """Log unhandled exceptions from fire-and-forget cron tasks."""
    import asyncio
    t = task if isinstance(task, asyncio.Task) else None
    if t is None or t.cancelled():
        return
    exc = t.exception()
    if exc:
        logger.error("Background cron job failed: %s", exc, exc_info=exc)


def _wallet_chain_label(chain_id: str, cfg: dict) -> str:
    """Human-friendly chain label for the dashboard UI."""
    name = chain_id.split(":", 1)[-1].replace("-", " ").title()
    eco = cfg.get("ecosystem", "").upper()
    symbol = cfg.get("symbol", "")
    if "devnet" in chain_id or "sepolia" in chain_id:
        return f"{name} ({eco} Testnet)"
    return f"{name} ({eco} · {symbol})"


def create_dashboard_router(
    blackboard: Blackboard,
    health_monitor: HealthMonitor | None,
    cost_tracker: CostTracker,
    trace_store: TraceStore | None,
    event_bus: EventBus | None,
    agent_registry: dict[str, str],
    mesh_port: int = 8420,
    # Optional subsystem dependencies (not all deployments include all subsystems)
    lane_manager: Any = None,
    cron_scheduler: Any = None,
    pubsub: Any = None,
    permissions: Any = None,
    credential_vault: Any = None,
    transport: Any = None,
    runtime: Any = None,
    router: Any = None,
    webhook_manager: Any = None,
    channel_manager: Any = None,
    wallet_service_ref: list | None = None,
    api_key_manager: Any = None,
) -> APIRouter:
    """Create the dashboard FastAPI router."""
    # Plan limits — read once at startup; provisioner restarts engine after updating .env
    # 0 = unlimited (self-hosted / open-source) unless env var is explicitly set to 0
    _max_agents = int(os.environ.get("OPENLEGION_MAX_AGENTS", "0"))
    _max_projects = int(os.environ.get("OPENLEGION_MAX_PROJECTS", "0"))
    _projects_disabled = _max_projects == 0 and "OPENLEGION_MAX_PROJECTS" in os.environ

    async def _csrf_check(request: Request) -> None:
        """Require X-Requested-With header on state-changing requests.

        Browsers block custom headers on cross-origin requests (CORS preflight),
        so this prevents CSRF attacks on cookie-authenticated endpoints.
        GET/HEAD/OPTIONS are exempt (safe methods).
        """
        if request.method in ("GET", "HEAD", "OPTIONS"):
            return
        if not request.headers.get("X-Requested-With"):
            raise HTTPException(403, "Missing X-Requested-With header")

    def _mask_proxy_url(url: str) -> str:
        """Mask credentials in a proxy URL for display."""
        if not url:
            return ""
        from urllib.parse import urlparse, urlunparse
        try:
            parsed = urlparse(url)
            if parsed.username:
                masked_netloc = f"{parsed.username[:2]}***@{parsed.hostname}:{parsed.port}"
                return urlunparse((parsed.scheme, masked_netloc, "", "", "", ""))
            return f"{parsed.scheme}://{parsed.hostname}:{parsed.port}" if parsed.port else url
        except Exception:
            return "***"

    async def _push_browser_proxy_for_agent(agent_id: str) -> None:
        """Push proxy config to browser service for an agent after restart."""
        if not runtime or not hasattr(runtime, "browser_service_url") or not runtime.browser_service_url:
            return
        try:
            from src.cli.config import _load_config
            from src.cli.proxy import parse_proxy_url, resolve_agent_proxy
            _cfg = _load_config()
            proxy_url = resolve_agent_proxy(agent_id, _cfg.get("agents", {}), _cfg.get("network", {}))
            if proxy_url:
                parsed = parse_proxy_url(proxy_url)
                if parsed:
                    body = {"url": parsed["url"], "username": parsed["username"], "password": parsed["password"]}
                else:
                    body = {}  # explicit no-proxy
            else:
                body = {}  # explicit no-proxy (direct mode or no system proxy)
            headers: dict = {}
            svc_token = getattr(runtime, "browser_auth_token", "")
            if svc_token:
                headers["Authorization"] = f"Bearer {svc_token}"
            resp = await _dashboard_browser_client.put(
                f"{runtime.browser_service_url}/browser/{agent_id}/proxy",
                json=body, headers=headers,
            )
            if resp.status_code >= 400:
                logger.warning("Browser proxy push for %s returned %d", agent_id, resp.status_code)
        except Exception as e:
            logger.warning("Failed to push browser proxy for %s: %s", agent_id, e)

    api_router = APIRouter(
        prefix="/dashboard",
        dependencies=[Depends(_verify_dashboard_auth), Depends(_csrf_check)],
    )

    jinja_env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=True,
    )

    # Build valid-models set for validation (dynamic from litellm)
    from src.cli.config import _PROVIDER_MODELS
    from src.shared.models import get_provider_models

    def _is_valid_model(model: str) -> bool:
        """Check if a model is known (from litellm or featured lists).

        Ollama models are always accepted since they're user-installed locally.
        Custom LLM providers registered via settings.json are also accepted.
        """
        provider = model.split("/")[0] if "/" in model else ""
        if provider in ("ollama", "ollama_chat"):
            return True
        settings = _load_settings()
        custom_providers = settings.get("custom_llm_providers", {})
        provider_lower = provider.lower()
        if provider_lower in custom_providers:
            custom_models = custom_providers[provider_lower].get("models", [])
            return model.lower() in [m.lower() for m in custom_models]
        if provider:
            return model in get_provider_models(provider)
        return any(model in models for models in _PROVIDER_MODELS.values())

    # ── SPA entry point ──────────────────────────────────────

    @api_router.get("/", response_class=HTMLResponse)
    async def dashboard_index() -> HTMLResponse:
        from src.shared.models import KEYLESS_PROVIDERS, get_all_providers
        all_providers = get_all_providers()
        template = jinja_env.get_template("index.html")
        html = template.render(
            ws_path="/ws/events",
            api_base="/dashboard/api",
            v=ASSET_VERSION,
            providers=[p for p in all_providers if p["name"] not in KEYLESS_PROVIDERS],
            all_providers=all_providers,
        )
        return HTMLResponse(html, headers={
            "Cache-Control": "no-store",
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.tailwindcss.com https://cdn.jsdelivr.net; "
                "style-src 'self' 'unsafe-inline'; "
                "connect-src 'self'; "
                "frame-src 'self'; "
                "object-src 'none'"
            ),
        })

    def _browser_vnc_url_for_request(request: Request) -> str | None:
        """Build shared browser VNC URL using the request host.

        Supports two modes:
        - **Behind reverse proxy** (Caddy/nginx): detected via x-forwarded-proto
          header. Routes VNC through /vnc/ path on the same origin so it works
          through HTTPS without exposing extra ports.
        - **Direct access** (local dev): uses the VNC port directly.
        """
        if not runtime or not hasattr(runtime, 'browser_vnc_url') or not runtime.browser_vnc_url:
            return None

        forwarded_proto = request.headers.get("x-forwarded-proto")
        if forwarded_proto:
            # Behind a reverse proxy (Caddy, nginx, etc.)
            # Route through /vnc/ path — proxy must forward to the VNC port
            scheme = forwarded_proto
            host = request.headers.get("host", "127.0.0.1:8420")
            return f"{scheme}://{host}/vnc/index.html?autoconnect=true&reconnect=true&reconnect_delay=2000&path=vnc/websockify&resize=scale&quality=7&enable_perf_stats=0"

        # Direct access (local dev) — route through mesh proxy
        host = request.headers.get("host", "127.0.0.1:8420")
        return f"http://{host}/vnc/index.html?autoconnect=true&reconnect=true&reconnect_delay=2000&path=vnc/websockify&resize=scale&quality=7&enable_perf_stats=0"

    # ── Fleet overview ───────────────────────────────────────

    @api_router.get("/api/agents")
    async def api_agents(request: Request) -> dict:
        from src.cli.config import _load_config
        cfg = _load_config()
        agents_cfg = cfg.get("agents", {})
        default_model = cfg.get("llm", {}).get("default_model", "openai/gpt-4o-mini")

        health_list = health_monitor.get_status() if health_monitor else []
        health_map = {h["agent"]: h for h in health_list}
        cost_list = cost_tracker.get_all_agents_spend("today")
        cost_map = {c["agent"]: c for c in cost_list}

        agent_projects = cfg.get("_agent_projects", {})

        agents = []
        for agent_id, url in agent_registry.items():
            h = health_map.get(agent_id, {})
            c = cost_map.get(agent_id, {})
            acfg = agents_cfg.get(agent_id, {})
            entry = {
                "id": agent_id,
                "url": url,
                "running": True,
                "over_limit": False,
                "health_status": h.get("status", "unknown"),
                "failures": h.get("failures", 0),
                "restarts": h.get("restarts", 0),
                "last_check": h.get("last_check", 0),
                "last_healthy": h.get("last_healthy", 0),
                "daily_cost": c.get("cost", 0),
                "daily_tokens": c.get("tokens", 0),
                "role": acfg.get("role", ""),
                "model": acfg.get("model", default_model),
                "avatar": acfg.get("avatar", 1),
                "color": acfg.get("color"),
                "project": agent_projects.get(agent_id),
            }
            if cron_scheduler is not None:
                hb = cron_scheduler.find_heartbeat_job(agent_id)
                if hb:
                    entry["heartbeat_job_id"] = hb.id
                    entry["heartbeat_schedule"] = hb.schedule
                    entry["heartbeat_enabled"] = hb.enabled
                    entry["heartbeat_next_run"] = hb.next_run
            vnc_url = _browser_vnc_url_for_request(request)
            if vnc_url:
                entry["vnc_url"] = vnc_url
            agents.append(entry)

        # Append over-limit agents from config that are not running
        for agent_id, acfg in agents_cfg.items():
            if agent_id not in agent_registry:
                entry = {
                    "id": agent_id,
                    "url": None,
                    "running": False,
                    "over_limit": True,
                    "health_status": "stopped",
                    "failures": 0,
                    "restarts": 0,
                    "last_check": 0,
                    "last_healthy": 0,
                    "daily_cost": 0,
                    "daily_tokens": 0,
                    "role": acfg.get("role", ""),
                    "model": acfg.get("model", default_model),
                    "avatar": acfg.get("avatar", 1),
                    "color": acfg.get("color"),
                    "project": agent_projects.get(agent_id),
                }
                agents.append(entry)

        return {"agents": agents}

    import httpx as _httpx
    _dashboard_browser_client = _httpx.AsyncClient(timeout=10)

    async def _push_browser_settings() -> None:
        """Push saved browser speed/delay to the browser service.

        Called after browser service (re)start so persisted settings
        survive container restarts.  Failures are silently logged —
        the browser service will simply use its defaults.
        """
        if not runtime or not getattr(runtime, 'browser_service_url', ''):
            return
        settings = _load_settings()
        payload: dict = {}
        speed = settings.get("browser_speed")
        if speed is not None:
            payload["speed"] = speed
        delay = settings.get("browser_delay")
        if delay is not None:
            payload["delay"] = delay
        if not payload:
            return
        try:
            browser_auth = getattr(runtime, 'browser_auth_token', '')
            headers = {}
            if browser_auth:
                headers["Authorization"] = f"Bearer {browser_auth}"
            await _dashboard_browser_client.post(
                f"{runtime.browser_service_url}/browser/settings",
                json=payload,
                headers=headers,
            )
        except Exception as e:
            logger.debug("Failed to push browser settings on startup: %s", e)

    @api_router.post("/api/browser/{agent_id}/focus")
    async def api_browser_focus(agent_id: str, request: Request) -> dict:
        """Tell the browser service to bring this agent's browser to foreground."""
        if not runtime or not hasattr(runtime, 'browser_service_url') or not runtime.browser_service_url:
            raise HTTPException(503, "Browser service not available")
        try:
            browser_auth = getattr(runtime, 'browser_auth_token', '')
            headers = {}
            if browser_auth:
                headers["Authorization"] = f"Bearer {browser_auth}"
            resp = await _dashboard_browser_client.post(
                f"{runtime.browser_service_url}/browser/{agent_id}/focus",
                json={},
                headers=headers,
                timeout=60,  # Cold-start can take 20-30s (Camoufox + geoip)
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

    @api_router.post("/api/browser/{agent_id}/reset")
    async def api_browser_reset(agent_id: str) -> dict:
        """Reset an agent's browser session (close and relaunch with current config)."""
        if agent_id not in agent_registry:
            raise HTTPException(404, "Agent not found")
        if not runtime or not hasattr(runtime, 'browser_service_url') or not runtime.browser_service_url:
            raise HTTPException(503, "Browser service not available")
        try:
            browser_auth = getattr(runtime, 'browser_auth_token', '')
            headers = {}
            if browser_auth:
                headers["Authorization"] = f"Bearer {browser_auth}"
            resp = await _dashboard_browser_client.post(
                f"{runtime.browser_service_url}/browser/{agent_id}/reset",
                json={},
                headers=headers,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning("Browser reset failed for '%s': %s", agent_id, e)
            raise HTTPException(500, "Browser reset failed")

    @api_router.get("/api/agent-templates")
    async def api_agent_templates() -> list:
        """Return available skill templates for creating new agents."""
        from src.cli.config import _load_skill_templates
        return _load_skill_templates()

    @api_router.get("/api/fleet/templates")
    async def api_fleet_templates(request: Request) -> dict:
        """Return available fleet templates."""
        from src.cli.config import _load_templates
        templates = _load_templates()
        result = []
        for name, tpl in templates.items():
            result.append({
                "name": name,
                "description": tpl.get("description", ""),
                "agent_count": len(tpl.get("agents", {})),
                "agents": list(tpl.get("agents", {}).keys()),
            })
        return {"templates": result}

    @api_router.post("/api/agents")
    async def api_add_agent(request: Request) -> dict:
        """Add a new agent: create config, start container, register."""
        body = await request.json()
        name = body.get("name", "").strip()
        role = body.get("role", "").strip()
        model = body.get("model", "").strip()
        avatar = body.get("avatar", 1)
        color = body.get("color")
        template = body.get("template", "").strip()
        project = body.get("project", "").strip()

        if not name:
            raise HTTPException(status_code=400, detail="name is required")
        if not re.match(r"^[a-z][a-z0-9_]{0,29}$", name):
            raise HTTPException(status_code=400, detail="name must match ^[a-z][a-z0-9_]{0,29}$")
        if name in agent_registry:
            raise HTTPException(status_code=409, detail=f"Agent '{name}' already exists")
        # Limit based on running agents (resource usage), not config definitions.
        # A stopped agent frees a slot. Operator is excluded from the count.
        from src.cli.config import _OPERATOR_AGENT_ID
        non_operator_count = sum(1 for a in agent_registry if a != _OPERATOR_AGENT_ID)
        if _max_agents > 0 and non_operator_count >= _max_agents:
            raise HTTPException(
                status_code=403,
                detail=f"Agent limit reached ({_max_agents}). Upgrade your plan for more agents.",
            )

        if model and not _is_valid_model(model):
            raise HTTPException(status_code=400, detail=f"Invalid model: {model}")

        try:
            avatar = int(avatar)
            if avatar < 1 or avatar > 50:
                raise HTTPException(status_code=400, detail="Avatar must be between 1 and 50")
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="Avatar must be an integer between 1 and 50")

        if color is not None:
            try:
                color = int(color)
                if color < 0 or color > 15:
                    raise HTTPException(status_code=400, detail="Color must be between 0 and 15")
            except (ValueError, TypeError):
                raise HTTPException(status_code=400, detail="Color must be an integer between 0 and 15")

        if template and not re.match(r"^[a-z][a-z0-9_-]*/[a-z][a-z0-9_-]*$", template):
            raise HTTPException(status_code=400, detail="Invalid template id format")

        if not model:
            from src.cli.config import _load_config
            default = _load_config().get("llm", {}).get("default_model", "openai/gpt-4o-mini")
            # Only use the configured default if its provider has credentials
            default_provider = default.split("/")[0] if "/" in default else ""
            active = credential_vault.get_providers_with_credentials() if credential_vault else set()
            if active and default_provider not in active:
                # Pick the first model from the first provider that has a key
                model = ""
                for p, models in _PROVIDER_MODELS.items():
                    if p in active and models:
                        model = models[0]
                        break
                if not model:
                    model = default  # No credentials at all — use config default
            else:
                model = default
        if not role:
            role = "assistant"

        if runtime is None:
            raise HTTPException(status_code=503, detail="Runtime not available")

        try:
            from src.cli.config import (
                _create_agent,
                _create_agent_from_template,
                _load_config,
                _update_agent_field,
            )
            if template:
                try:
                    _create_agent_from_template(name, template, model)
                except ValueError as e:
                    raise HTTPException(status_code=400, detail=str(e))
            else:
                _create_agent(name, role, model)
            _update_agent_field(name, "avatar", avatar)
            if color is not None:
                _update_agent_field(name, "color", color)
            if permissions is not None:
                permissions.reload()

            cfg = _load_config()
            acfg = cfg.get("agents", {}).get(name, {})
            if template:
                role = acfg.get("role", role)
            skills_dir = os.path.abspath(acfg.get("skills_dir", ""))
            # Build per-agent env overrides (no shared extra_env mutation)
            agent_env: dict[str, str] = {}
            for env_key, cfg_key in (
                ("INITIAL_INSTRUCTIONS", "initial_instructions"),
                ("INITIAL_SOUL", "initial_soul"),
                ("INITIAL_HEARTBEAT", "initial_heartbeat"),
            ):
                val = acfg.get(cfg_key, "")
                if val:
                    agent_env[env_key] = val
            url = runtime.start_agent(
                agent_id=name,
                role=role,
                skills_dir=skills_dir,
                model=acfg.get("model", model),
                thinking=acfg.get("thinking", ""),
                env_overrides=agent_env,
            )
            if router is not None:
                router.register_agent(name, url, role=role)
            else:
                agent_registry[name] = url
            if transport is not None:
                from src.host.transport import HttpTransport
                if isinstance(transport, HttpTransport):
                    transport.register(name, url)
            if health_monitor is not None:
                health_monitor.register(name)
            if cron_scheduler is not None:
                hb_schedule = cfg.get("mesh", {}).get("heartbeat_schedule")
                cron_scheduler.ensure_heartbeat(name, hb_schedule)
            ready = await runtime.wait_for_agent(name, timeout=60)
            if project:
                if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$", project):
                    logger.warning("Skipping invalid project name '%s' for agent '%s'", project, name)
                    project = ""
                else:
                    from src.cli.config import _add_agent_to_project
                    try:
                        _add_agent_to_project(project, name)
                    except ValueError:
                        logger.warning("Project '%s' not found; agent '%s' created standalone", project, name)
                        project = ""
            if event_bus is not None:
                event_bus.emit("agent_state", agent=name,
                    data={"state": "added", "role": role, "ready": ready})
            return {"created": True, "agent": name, "ready": ready, "project": project or None}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to add agent {name}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @api_router.delete("/api/agents/{agent_id}")
    async def api_remove_agent(agent_id: str) -> dict:
        """Remove an agent: stop container, unregister, remove config."""
        if agent_id == "operator":
            raise HTTPException(status_code=403, detail="The operator is a system agent and cannot be deleted")
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")

        # Stop container and remove data volume (best-effort — agent may already be gone)
        if runtime is not None:
            try:
                runtime.stop_agent(agent_id, remove_data=True)
            except Exception as e:
                logger.debug("Runtime cleanup for '%s' failed: %s", agent_id, e)

        # Unregister from router, transport, and health monitor
        if router is not None:
            router.unregister_agent(agent_id)
        else:
            agent_registry.pop(agent_id, None)
        if transport is not None:
            from src.host.transport import HttpTransport
            if isinstance(transport, HttpTransport):
                transport._urls.pop(agent_id, None)
        if health_monitor is not None:
            health_monitor.unregister(agent_id)

        # Clean up PubSub subscriptions, cron jobs, and lane state
        if pubsub is not None:
            pubsub.unsubscribe_agent(agent_id)
        if cron_scheduler is not None:
            removed = cron_scheduler.remove_agent_jobs(agent_id)
            if removed:
                logger.info(f"Removed {removed} cron job(s) for agent {agent_id}")
        if lane_manager is not None:
            lane_manager.remove_lane(agent_id)

        # Clean up per-agent data: blackboard, costs, traces, wallet
        try:
            blackboard.cleanup_agent_data(agent_id)
        except Exception as e:
            logger.warning("Blackboard cleanup for '%s' failed: %s", agent_id, e)
        if cost_tracker is not None:
            try:
                cost_tracker.cleanup_agent(agent_id)
            except Exception as e:
                logger.warning("Cost cleanup for '%s' failed: %s", agent_id, e)
        if trace_store is not None:
            try:
                trace_store.cleanup_agent(agent_id)
            except Exception as e:
                logger.warning("Trace cleanup for '%s' failed: %s", agent_id, e)
        _ws_ref_local = wallet_service_ref or [None]
        _wallet_svc = _ws_ref_local[0]
        if _wallet_svc is not None:
            try:
                _wallet_svc.cleanup_agent(agent_id)
            except Exception as e:
                logger.warning("Wallet cleanup for '%s' failed: %s", agent_id, e)

        # Clean up proxy credential if exists
        try:
            from src.cli.config import _load_config as _load_cfg_for_delete
            _del_cfg = _load_cfg_for_delete()
            _del_agent_cfg = _del_cfg.get("agents", {}).get(agent_id, {})
            _del_proxy = _del_agent_cfg.get("proxy", {})
            if _del_proxy.get("credential"):
                from src.host.credentials import _remove_from_env
                _cred_env_key = f"OPENLEGION_CRED_{_del_proxy['credential']}"
                _remove_from_env(_cred_env_key)
                os.environ.pop(_cred_env_key, None)
        except Exception as e:
            logger.warning("Proxy credential cleanup for '%s' failed: %s", agent_id, e)

        # Remove from config and permissions (best-effort — don't fail if files are missing)
        try:
            import yaml

            from src.cli.config import AGENTS_FILE, _load_permissions, _save_permissions

            if AGENTS_FILE.exists():
                with open(AGENTS_FILE) as f:
                    agents_data = yaml.safe_load(f) or {}
                agents_data.get("agents", {}).pop(agent_id, None)
                with open(AGENTS_FILE, "w") as f:
                    yaml.dump(agents_data, f, default_flow_style=False, sort_keys=False)

            perms = _load_permissions()
            perms.get("permissions", {}).pop(agent_id, None)
            _save_permissions(perms)
            if permissions is not None:
                permissions.reload()
        except Exception as e:
            logger.warning(f"Failed to clean config for {agent_id}: {e}")

        if event_bus is not None:
            event_bus.emit("agent_state", agent=agent_id,
                data={"state": "removed"})

        return {"removed": True, "agent": agent_id}

    # ── Agent detail ─────────────────────────────────────────

    @api_router.get("/api/agents/{agent_id}")
    async def api_agent_detail(agent_id: str, request: Request) -> dict:
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")

        url = agent_registry[agent_id]
        health_list = health_monitor.get_status() if health_monitor else []
        health = next((h for h in health_list if h["agent"] == agent_id), {})
        spend_today = cost_tracker.get_spend(agent_id, "today")
        spend_week = cost_tracker.get_spend(agent_id, "week")
        budget = cost_tracker.check_budget(agent_id)

        result = {
            "id": agent_id,
            "url": url,
            "health": health or {"status": "unknown"},
            "spend_today": spend_today,
            "spend_week": spend_week,
            "budget": budget,
        }
        # Include shared browser VNC info
        vnc_url = _browser_vnc_url_for_request(request)
        if vnc_url:
            result["vnc_url"] = vnc_url
        return result

    # ── Agent config CRUD ────────────────────────────────────

    @api_router.get("/api/agents/{agent_id}/config")
    async def api_agent_config(agent_id: str, request: Request) -> dict:
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        from fnmatch import fnmatch

        from src.cli.config import _load_config
        cfg = _load_config()
        agent_cfg = cfg.get("agents", {}).get(agent_id, {})
        default_model = cfg.get("llm", {}).get("default_model", "openai/gpt-4o-mini")
        allowed_creds: list[str] = []
        if permissions is not None:
            allowed_creds = permissions.get_allowed_credentials(agent_id)

        # Compute credential visibility for the dashboard UI
        agent_cred_names = credential_vault.list_agent_credential_names() if credential_vault else []
        system_cred_names = sorted(
            credential_vault.list_system_credential_names()
        ) if credential_vault else []
        resolved = sorted(
            c for c in agent_cred_names
            if any(fnmatch(c, p) for p in allowed_creds)
        ) if allowed_creds else []

        # Capability flags from permissions
        agent_perms = permissions.get_permissions(agent_id) if permissions else None
        cfg_result = {
            "id": agent_id,
            "model": agent_cfg.get("model", default_model),
            "role": agent_cfg.get("role", ""),
            "avatar": agent_cfg.get("avatar", 1),
            "color": agent_cfg.get("color"),
            "budget": agent_cfg.get("budget", {}),
            "thinking": agent_cfg.get("thinking", "off") or "off",
            "mcp_servers": agent_cfg.get("mcp_servers") or [],
            "allowed_credentials": allowed_creds,
            "available_credentials": sorted(agent_cred_names),
            "system_credentials": system_cred_names,
            "resolved_credentials": resolved,
            "can_use_browser": agent_perms.can_use_browser if agent_perms else False,
            "can_spawn": agent_perms.can_spawn if agent_perms else False,
            "can_manage_cron": agent_perms.can_manage_cron if agent_perms else False,
            "can_use_wallet": agent_perms.can_use_wallet if agent_perms else False,
            "wallet_allowed_chains": (
                agent_perms.wallet_allowed_chains if agent_perms else []
            ),
        }
        # Wallet: available chains + derived addresses
        _ws_ref_local = wallet_service_ref or [None]
        ws = _ws_ref_local[0]
        cfg_result["wallet_configured"] = ws is not None
        if ws is not None:
            cfg_result["wallet_available_chains"] = [
                {"id": cid, "label": _wallet_chain_label(cid, ccfg), "ecosystem": ccfg["ecosystem"]}
                for cid, ccfg in ws.chains.items()
            ]
        else:
            cfg_result["wallet_available_chains"] = []
        if ws is not None and cfg_result["can_use_wallet"]:
            try:
                evm_addr = await ws.get_address(agent_id, "evm:ethereum")
                sol_addr = await ws.get_address(agent_id, "solana:mainnet")
                cfg_result["wallet_addresses"] = {
                    "evm": evm_addr, "solana": sol_addr,
                }
            except Exception:
                cfg_result["wallet_addresses"] = None
        else:
            cfg_result["wallet_addresses"] = None

        # Proxy configuration
        proxy_cfg = agent_cfg.get("proxy", {})
        proxy_mode = proxy_cfg.get("mode", "inherit")
        proxy_info: dict[str, Any] = {"mode": proxy_mode}
        if proxy_mode == "custom":
            cred = proxy_cfg.get("credential", "")
            raw = os.environ.get(f"OPENLEGION_CRED_{cred}", "")
            proxy_info["url"] = _mask_proxy_url(raw) if raw else ""
            if raw:
                from urllib.parse import urlparse as _urlparse
                _p = _urlparse(raw)
                proxy_info["host"] = f"{_p.hostname}:{_p.port}" if _p.hostname and _p.port else (_p.hostname or "")
                proxy_info["scheme"] = _p.scheme or "http"
                proxy_info["has_credential"] = bool(_p.username)
            else:
                proxy_info["has_credential"] = False
        cfg_result["proxy"] = proxy_info

        vnc_url = _browser_vnc_url_for_request(request)
        if vnc_url:
            cfg_result["vnc_url"] = vnc_url
        return cfg_result

    @api_router.put("/api/agents/{agent_id}/config")
    async def api_update_agent_config(agent_id: str, request: Request) -> dict:
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        body = await request.json()
        from src.cli.config import _load_config, _update_agent_field
        cfg = _load_config()
        agent_cfg = cfg.get("agents", {}).get(agent_id, {})
        default_model = cfg.get("llm", {}).get("default_model", "openai/gpt-4o-mini")

        updated = []
        restart_required = False

        if "model" in body:
            new_model = body["model"]
            if not _is_valid_model(new_model):
                raise HTTPException(status_code=400, detail=f"Invalid model: {new_model}")
            old_model = agent_cfg.get("model", default_model)
            if new_model != old_model:
                _update_agent_field(agent_id, "model", new_model)
                updated.append("model")
                restart_required = True

        if "role" in body:
            _update_agent_field(agent_id, "role", body["role"])
            updated.append("role")

        if "avatar" in body:
            try:
                av = int(body["avatar"])
                if av < 1 or av > 50:
                    raise HTTPException(status_code=400, detail="Avatar must be between 1 and 50")
                _update_agent_field(agent_id, "avatar", av)
                updated.append("avatar")
            except (ValueError, TypeError):
                raise HTTPException(status_code=400, detail="Avatar must be an integer between 1 and 50")

        if "color" in body:
            raw_color = body["color"]
            if raw_color is None:
                _update_agent_field(agent_id, "color", None)
                updated.append("color")
            else:
                try:
                    cv = int(raw_color)
                    if cv < 0 or cv > 15:
                        raise HTTPException(status_code=400, detail="Color must be between 0 and 15")
                    _update_agent_field(agent_id, "color", cv)
                    updated.append("color")
                except (ValueError, TypeError):
                    raise HTTPException(status_code=400, detail="Color must be an integer between 0 and 15")

        if "budget" in body:
            budget_val = body["budget"]
            if isinstance(budget_val, dict):
                raw_daily = budget_val.get("daily_usd")
                raw_monthly = budget_val.get("monthly_usd")
                if raw_daily is not None or raw_monthly is not None:
                    current = cost_tracker.check_budget(agent_id)
                    daily = _parse_positive_float(raw_daily, "daily_usd", current.get("daily_limit", 10.0))
                    monthly = _parse_positive_float(raw_monthly, "monthly_usd", current.get("monthly_limit", 200.0))
                    _update_agent_field(agent_id, "budget", {"daily_usd": daily, "monthly_usd": monthly})
                    cost_tracker.set_budget(agent_id, daily_usd=daily, monthly_usd=monthly)
                    updated.append("budget")

        if "thinking" in body:
            thinking_val = body["thinking"]
            if thinking_val not in ("off", "low", "medium", "high"):
                raise HTTPException(
                    status_code=400,
                    detail="thinking must be one of: off, low, medium, high",
                )
            _update_agent_field(agent_id, "thinking", thinking_val)
            updated.append("thinking")
            restart_required = True

        if "mcp_servers" in body:
            mcp_val = body["mcp_servers"]
            if not isinstance(mcp_val, list):
                raise HTTPException(status_code=400, detail="mcp_servers must be a list")
            for srv in mcp_val:
                if not isinstance(srv, dict) or "name" not in srv or "command" not in srv:
                    raise HTTPException(
                        status_code=400,
                        detail="Each MCP server must have 'name' and 'command' keys",
                    )
            _update_agent_field(agent_id, "mcp_servers", mcp_val if mcp_val else None)
            updated.append("mcp_servers")
            restart_required = True

        return {"updated": updated, "restart_required": restart_required}

    @api_router.post("/api/agents/{agent_id}/restart")
    async def api_restart_agent(agent_id: str) -> dict:
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        if runtime is None:
            raise HTTPException(status_code=503, detail="Runtime not available")
        try:
            from src.cli.config import _load_config
            cfg = _load_config()
            agent_cfg = cfg.get("agents", {}).get(agent_id, {})
            default_model = cfg.get("llm", {}).get("default_model", "openai/gpt-4o-mini")
            runtime.stop_agent(agent_id)
            skills_dir = agent_cfg.get("skills_dir", "")
            if skills_dir:
                skills_dir = str(Path(skills_dir).resolve())
            # Preserve operator's ALLOWED_TOOLS on restart
            from src.cli.config import _OPERATOR_AGENT_ID, _OPERATOR_ALLOWED_TOOLS
            restart_env: dict[str, str] = {}
            if agent_id == _OPERATOR_AGENT_ID:
                restart_env["ALLOWED_TOOLS"] = ",".join(_OPERATOR_ALLOWED_TOOLS)
            # Proxy goes in env_overrides (not runtime.extra_env) so
            # concurrent single-agent restarts don't stomp each other.
            _proxy_url = resolve_agent_proxy(
                agent_id, cfg.get("agents", {}), cfg.get("network", {}),
            )
            _proxy_env = build_proxy_env_vars(
                _proxy_url, cfg.get("network", {}).get("no_proxy", ""),
            )
            restart_env.update(_proxy_env)
            url = runtime.start_agent(
                agent_id=agent_id,
                role=agent_cfg.get("role", "assistant"),
                skills_dir=skills_dir,
                model=agent_cfg.get("model", default_model),
                mcp_servers=agent_cfg.get("mcp_servers") or None,
                thinking=agent_cfg.get("thinking", ""),
                env_overrides=restart_env,
            )
            if router is not None:
                router.register_agent(agent_id, url, role=agent_cfg.get("role", ""))
            else:
                agent_registry[agent_id] = url
            if transport is not None:
                from src.host.transport import HttpTransport
                if isinstance(transport, HttpTransport):
                    transport.register(agent_id, url)
            ready = await runtime.wait_for_agent(agent_id, timeout=60)
            # Push proxy config to browser service
            await _push_browser_proxy_for_agent(agent_id)
            return {"restarted": True, "ready": ready}
        except Exception as e:
            logger.error(f"Failed to restart agent {agent_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @api_router.put("/api/agents/{agent_id}/budget")
    async def api_update_budget(agent_id: str, request: Request) -> dict:
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        body = await request.json()
        raw_daily = body.get("daily_usd")
        raw_monthly = body.get("monthly_usd")
        if raw_daily is None and raw_monthly is None:
            raise HTTPException(status_code=400, detail="Provide daily_usd and/or monthly_usd")
        current = cost_tracker.check_budget(agent_id)
        daily_usd = _parse_positive_float(raw_daily, "daily_usd", current.get("daily_limit", 10.0))
        monthly_usd = _parse_positive_float(raw_monthly, "monthly_usd", current.get("monthly_limit", 200.0))
        cost_tracker.set_budget(agent_id, daily_usd=daily_usd, monthly_usd=monthly_usd)
        from src.cli.config import _update_agent_field
        _update_agent_field(agent_id, "budget", {"daily_usd": daily_usd, "monthly_usd": monthly_usd})
        return {"updated": True, "agent": agent_id, "daily_usd": daily_usd, "monthly_usd": monthly_usd}

    # ── Network / Proxy ─────────────────────────────────────

    @api_router.get("/api/network/proxy")
    async def api_get_network_proxy(request: Request):
        """Return system proxy info (masked), NO_PROXY, and per-agent proxy summary."""
        from src.cli.config import _load_config
        from src.cli.proxy import _assemble_proxy_url

        cfg = _load_config()
        browser_proxy_url = os.environ.get("BROWSER_PROXY_URL", "")
        system_proxy = os.environ.get("OPENLEGION_SYSTEM_PROXY", "")
        is_managed = bool(browser_proxy_url)

        # Managed URL (always compute when managed, for display)
        managed_masked = ""
        if browser_proxy_url:
            full = _assemble_proxy_url(
                browser_proxy_url,
                os.environ.get("BROWSER_PROXY_USER", ""),
                os.environ.get("BROWSER_PROXY_PASS", ""),
            )
            managed_masked = _mask_proxy_url(full)

        # Active URL follows resolution order: user override > managed
        if system_proxy:
            active_masked = _mask_proxy_url(system_proxy)
        elif managed_masked:
            active_masked = managed_masked
        else:
            active_masked = ""

        is_overridden = is_managed and bool(system_proxy)

        network_cfg = cfg.get("network", {})
        no_proxy = network_cfg.get("no_proxy", "")

        agents_cfg = cfg.get("agents", {})
        agent_summary = []
        for aid, acfg in agents_cfg.items():
            proxy = acfg.get("proxy", {})
            mode = proxy.get("mode", "inherit")
            agent_proxy_url = ""
            if mode == "custom":
                cred = proxy.get("credential", "")
                raw = os.environ.get(f"OPENLEGION_CRED_{cred}", "")
                agent_proxy_url = _mask_proxy_url(raw) if raw else "(credential missing)"
            agent_summary.append({"agent_id": aid, "mode": mode, "proxy_url": agent_proxy_url})

        return {
            "system_proxy": {
                "configured": bool(active_masked),
                "managed": is_managed,
                "managed_url": managed_masked,
                "overridden": is_overridden,
                "url": active_masked,
            },
            "no_proxy": no_proxy,
            "agents": agent_summary,
        }

    @api_router.put("/api/network/proxy")
    async def api_put_network_proxy(request: Request):
        """Update system proxy and/or NO_PROXY."""
        body = await request.json()
        updated = []

        if "no_proxy" in body:
            from src.cli.config import _update_network_config
            _update_network_config("no_proxy", body["no_proxy"])
            updated.append("no_proxy")

        if "system_proxy" in body:
            from src.cli.proxy import _assemble_proxy_url, validate_proxy_url
            from src.host.credentials import _persist_to_env, _remove_from_env

            sp = body["system_proxy"]
            if sp is None or sp == "":
                _remove_from_env("OPENLEGION_SYSTEM_PROXY")
                os.environ.pop("OPENLEGION_SYSTEM_PROXY", None)
                updated.append("system_proxy_removed")
            else:
                url = sp.get("url", "")
                if url and url.strip().lower().startswith("socks"):
                    raise HTTPException(400, "SOCKS5 proxies are not supported — please use an HTTP/HTTPS proxy")
                username = sp.get("username", "")
                password = sp.get("password", "")
                full_url = _assemble_proxy_url(url, username, password) if username else url
                if not validate_proxy_url(full_url):
                    raise HTTPException(400, "Invalid proxy URL")
                _persist_to_env("OPENLEGION_SYSTEM_PROXY", full_url)
                os.environ["OPENLEGION_SYSTEM_PROXY"] = full_url
                updated.append("system_proxy")

        return {"updated": updated, "restart_required": bool(updated)}

    @api_router.put("/api/agents/{agent_id}/proxy")
    async def api_put_agent_proxy(agent_id: str, request: Request):
        """Set per-agent proxy config. Works for stopped agents (checks config, not registry)."""
        from src.cli.config import _load_config, _update_agent_field
        from src.cli.proxy import _assemble_proxy_url, sanitize_agent_id_for_env, validate_proxy_url
        from src.host.credentials import _persist_to_env, _remove_from_env

        cfg = _load_config()
        agents_cfg = cfg.get("agents", {})
        if agent_id not in agents_cfg:
            raise HTTPException(404, "Agent not found in config")

        body = await request.json()
        mode = body.get("mode", "inherit")
        if mode not in ("inherit", "custom", "direct"):
            raise HTTPException(400, f"Invalid proxy mode: {mode}")

        proxy_yaml: dict[str, str] = {"mode": mode}
        safe_id = sanitize_agent_id_for_env(agent_id)

        if mode == "custom":
            url = body.get("url", "")
            if url and url.strip().lower().startswith("socks"):
                raise HTTPException(400, "SOCKS5 proxies are not supported — please use an HTTP/HTTPS proxy")
            username = body.get("username", "")
            password = body.get("password", "")
            full_url = _assemble_proxy_url(url, username, password) if username else url
            if not validate_proxy_url(full_url):
                raise HTTPException(400, "Invalid proxy URL")

            cred_name = f"agent_{safe_id}_proxy"
            env_key = f"OPENLEGION_CRED_{cred_name}"
            _persist_to_env(env_key, full_url)
            os.environ[env_key] = full_url
            proxy_yaml["credential"] = cred_name

        elif mode in ("inherit", "direct"):
            # Clean up any existing custom credential
            old_proxy = agents_cfg.get(agent_id, {}).get("proxy", {})
            old_cred = old_proxy.get("credential", "")
            if old_cred:
                env_key = f"OPENLEGION_CRED_{old_cred}"
                _remove_from_env(env_key)
                os.environ.pop(env_key, None)

        _update_agent_field(agent_id, "proxy", proxy_yaml)

        # Push new proxy config to browser service immediately so manual
        # browser resets pick up the change without a full agent restart.
        await _push_browser_proxy_for_agent(agent_id)

        return {"updated": ["proxy"], "restart_required": True}

    @api_router.get("/api/agents/{agent_id}/permissions")
    async def api_agent_permissions(agent_id: str) -> dict:
        """Return agent permissions and available agent-tier credentials."""
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        if permissions is None:
            raise HTTPException(status_code=503, detail="Permissions not available")
        perms = permissions.get_permissions(agent_id)
        available_creds = []
        if credential_vault is not None:
            available_creds = credential_vault.list_agent_credential_names()
        return {
            "agent_id": agent_id,
            "allowed_credentials": perms.allowed_credentials,
            "allowed_apis": perms.allowed_apis,
            "available_credentials": available_creds,
            "can_use_browser": perms.can_use_browser,
            "can_spawn": perms.can_spawn,
            "can_manage_cron": perms.can_manage_cron,
        }

    @api_router.put("/api/agents/{agent_id}/permissions")
    async def api_update_agent_permissions(agent_id: str, request: Request) -> dict:
        """Update allowed_credentials and/or allowed_apis for an agent."""
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        if permissions is None:
            raise HTTPException(status_code=503, detail="Permissions not available")
        body = await request.json()
        from src.cli.config import _load_permissions, _save_permissions
        perms_data = _load_permissions()
        agent_perms = perms_data.get("permissions", {}).get(agent_id, {})

        updated = []
        if "allowed_credentials" in body:
            val = body["allowed_credentials"]
            if not isinstance(val, list) or not all(isinstance(v, str) for v in val):
                raise HTTPException(status_code=400, detail="allowed_credentials must be a list of strings")
            agent_perms["allowed_credentials"] = val
            updated.append("allowed_credentials")
        if "allowed_apis" in body:
            val = body["allowed_apis"]
            if not isinstance(val, list) or not all(isinstance(v, str) for v in val):
                raise HTTPException(status_code=400, detail="allowed_apis must be a list of strings")
            agent_perms["allowed_apis"] = val
            updated.append("allowed_apis")
        for flag in ("can_use_browser", "can_spawn", "can_manage_cron", "can_use_wallet"):
            if flag in body:
                agent_perms[flag] = bool(body[flag])
                updated.append(flag)
        if "wallet_allowed_chains" in body:
            val = body["wallet_allowed_chains"]
            if not isinstance(val, list) or not all(isinstance(v, str) for v in val):
                raise HTTPException(
                    status_code=400,
                    detail="wallet_allowed_chains must be a list of strings",
                )
            agent_perms["wallet_allowed_chains"] = val
            updated.append("wallet_allowed_chains")

        perms_data.setdefault("permissions", {})[agent_id] = agent_perms
        _save_permissions(perms_data)
        permissions.reload()
        return {"updated": updated, "agent_id": agent_id}

    @api_router.get("/api/agents/{agent_id}/status")
    async def api_agent_live_status(agent_id: str) -> dict:
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        if transport is None:
            raise HTTPException(status_code=503, detail="Transport not available")
        try:
            return await transport.request(agent_id, "GET", "/status", timeout=10)
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    @api_router.get("/api/agents/{agent_id}/capabilities")
    async def api_agent_capabilities(agent_id: str) -> dict:
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        if transport is None:
            raise HTTPException(status_code=503, detail="Transport not available")
        try:
            data = await transport.request(agent_id, "GET", "/capabilities", timeout=10)
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))
        # Backfill tool_sources for agents running pre-badge code (no container
        # restart required).  The host process has access to the builtins package
        # and can classify tools without executing agent-side code.
        if "tool_sources" not in data:
            builtins = _get_builtin_tool_names()
            sources: dict[str, str] = {}
            for tool in data.get("tool_definitions", []):
                name = (tool.get("function") or {}).get("name") or tool.get("name")
                if not name:
                    continue
                if tool.get("function") == "mcp":
                    sources[name] = "mcp"
                elif name in builtins:
                    sources[name] = "builtin"
                else:
                    sources[name] = "custom"
            data["tool_sources"] = sources
        return data

    # ── Chat with agent ────────────────────────────────────

    @api_router.post("/api/agents/{agent_id}/chat")
    async def api_chat(agent_id: str, request: Request) -> dict:
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        if transport is None:
            raise HTTPException(status_code=503, detail="Transport not available")
        body = await request.json()
        message = body.get("message", "").strip()
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        from src.shared.utils import sanitize_for_prompt
        message = sanitize_for_prompt(message)
        chat_session = request.headers.get("x-chat-session", "")
        if event_bus:
            event_bus.emit("chat_user_message", agent=agent_id,
                data={"message": message, "session": chat_session})
        try:
            result = await transport.request(
                agent_id, "POST", "/chat", json={"message": message}, timeout=120,
            )
            response = result.get("response", "(no response)")
            if event_bus:
                event_bus.emit("chat_done", agent=agent_id,
                    data={"response": response, "session": chat_session})
            return {"response": response}
        except Exception as e:
            if event_bus:
                event_bus.emit("chat_done", agent=agent_id,
                    data={"response": "", "session": chat_session})
            raise HTTPException(status_code=502, detail=str(e))

    @api_router.post("/api/agents/{agent_id}/chat/stream")
    async def api_chat_stream(agent_id: str, request: Request):
        """SSE streaming chat with an agent."""
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        if transport is None:
            raise HTTPException(status_code=503, detail="Transport not available")
        body = await request.json()
        message = body.get("message", "").strip()
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        from src.shared.utils import sanitize_for_prompt
        message = sanitize_for_prompt(message)
        chat_session = request.headers.get("x-chat-session", "")

        # Broadcast user message so other tabs/devices see it immediately
        if event_bus:
            event_bus.emit("chat_user_message", agent=agent_id,
                data={"message": message, "session": chat_session})

        async def event_generator():
            final_response = ""
            try:
                async for event in transport.stream_request(
                    agent_id, "POST", "/chat/stream",
                    json={"message": message}, timeout=120,
                ):
                    if isinstance(event, dict):
                        yield f"data: {json.dumps(event, default=str)}\n\n"
                        etype = event.get("type", "")
                        if event_bus:
                            if etype in ("tool_start", "tool_result", "text_delta"):
                                event_bus.emit(etype, agent=agent_id,
                                    data={k: v for k, v in event.items()
                                          if k != "type"} | {"session": chat_session})
                            if etype == "done":
                                final_response = event.get("response", "")
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': friendly_streaming_error(e)})}\n\n"
            # Notify other sessions that the response is complete
            if event_bus:
                event_bus.emit("chat_done", agent=agent_id,
                    data={"response": final_response, "session": chat_session})

        from starlette.responses import StreamingResponse
        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @api_router.post("/api/broadcast")
    async def api_broadcast(request: Request) -> dict:
        if transport is None:
            raise HTTPException(status_code=503, detail="Transport not available")
        body = await request.json()
        message = body.get("message", "").strip()
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        message = sanitize_for_prompt(message)
        import asyncio

        targets = list(agent_registry.keys())
        project = body.get("project") or ""
        if not isinstance(project, str):
            raise HTTPException(status_code=400, detail="project must be a string")
        if project:
            from src.cli.config import _load_projects
            members = set(_load_projects().get(project, {}).get("members", []))
            targets = [a for a in targets if a in members]
        elif body.get("standalone"):
            from src.cli.config import _load_projects
            assigned = set()
            for pdata in _load_projects().values():
                assigned.update(pdata.get("members", []))
            targets = [a for a in targets if a not in assigned]
        if not targets:
            return {"responses": {}, "message": "No matching agents"}

        results = {}
        async def _send(aid: str) -> tuple[str, str]:
            try:
                data = await transport.request(
                    aid, "POST", "/chat", json={"message": message}, timeout=120,
                )
                return aid, data.get("response", "(no response)")
            except Exception as e:
                return aid, f"Error: {e}"
        tasks = [_send(aid) for aid in targets]
        for coro in asyncio.as_completed(tasks):
            aid, resp = await coro
            results[aid] = resp
        return {"responses": results}

    @api_router.post("/api/broadcast/stream")
    async def api_broadcast_stream(request: Request):
        """SSE streaming broadcast — streams per-agent responses as they arrive."""
        if transport is None:
            raise HTTPException(status_code=503, detail="Transport not available")
        body = await request.json()
        message = body.get("message", "").strip()
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        message = sanitize_for_prompt(message)

        import asyncio

        agents = list(agent_registry.keys())
        project = body.get("project") or ""
        if not isinstance(project, str):
            raise HTTPException(status_code=400, detail="project must be a string")
        if project:
            from src.cli.config import _load_projects
            members = set(_load_projects().get(project, {}).get("members", []))
            agents = [a for a in agents if a in members]
        elif body.get("standalone"):
            from src.cli.config import _load_projects
            assigned = set()
            for pdata in _load_projects().values():
                assigned.update(pdata.get("members", []))
            agents = [a for a in agents if a not in assigned]
        if not agents:
            return {"responses": {}, "message": "No agents registered"}

        queue: asyncio.Queue = asyncio.Queue()

        async def _stream_agent(aid: str) -> None:
            await queue.put({"type": "agent_start", "agent": aid})
            try:
                async for event in transport.stream_request(
                    aid, "POST", "/chat/stream",
                    json={"message": message}, timeout=120,
                ):
                    if isinstance(event, dict):
                        tagged = {**event, "agent": aid}
                        await queue.put(tagged)
                        if event_bus:
                            etype = event.get("type", "")
                            if etype in ("tool_start", "tool_result"):
                                event_bus.emit(etype, agent=aid,
                                    data={k: v for k, v in tagged.items() if k != "type"})
            except Exception as e:
                await queue.put({"type": "error", "agent": aid, "message": friendly_streaming_error(e)})
            await queue.put({"type": "agent_done", "agent": aid})

        async def event_generator():
            tasks = [asyncio.create_task(_stream_agent(aid)) for aid in agents]
            done_count = 0
            try:
                while done_count < len(agents):
                    event = await queue.get()
                    if event.get("type") == "agent_done":
                        done_count += 1
                    yield f"data: {json.dumps(event, default=str)}\n\n"
            except asyncio.CancelledError:
                for t in tasks:
                    t.cancel()
                raise
            yield f"data: {json.dumps({'type': 'all_done'})}\n\n"

        from starlette.responses import StreamingResponse
        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @api_router.post("/api/agents/{agent_id}/steer")
    async def api_steer(agent_id: str, request: Request) -> dict:
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        if lane_manager is None:
            raise HTTPException(status_code=503, detail="Lane manager not available")
        body = await request.json()
        message = body.get("message", "").strip()
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        from src.shared.utils import sanitize_for_prompt
        message = sanitize_for_prompt(message)
        chat_session = request.headers.get("x-chat-session", "")
        if event_bus:
            event_bus.emit("chat_user_message", agent=agent_id,
                data={"message": f"[steer] {message}", "session": chat_session})
        from src.shared.trace import new_trace_id
        result = await lane_manager.enqueue(agent_id, message, mode="steer", trace_id=new_trace_id())
        return {"result": result}

    @api_router.post("/api/agents/{agent_id}/reset")
    async def api_reset(agent_id: str) -> dict:
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        if transport is None:
            raise HTTPException(status_code=503, detail="Transport not available")
        try:
            await transport.request(agent_id, "POST", "/chat/reset", timeout=10)
            if event_bus:
                event_bus.emit("chat_reset", agent=agent_id)
            return {"reset": True, "agent": agent_id}
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    @api_router.get("/api/agents/{agent_id}/chat/history")
    async def api_chat_history(agent_id: str) -> dict:
        """Return the agent's current in-memory chat conversation."""
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        if transport is None:
            raise HTTPException(status_code=503, detail="Transport not available")
        try:
            result = await transport.request(agent_id, "GET", "/chat/history", timeout=10)
            return result
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    @api_router.get("/api/agents/{agent_id}/artifacts")
    async def api_list_artifacts(agent_id: str) -> dict:
        """List artifact files in an agent's workspace."""
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        if transport is None:
            raise HTTPException(status_code=503, detail="Transport not available")
        try:
            return await transport.request(agent_id, "GET", "/artifacts", timeout=10)
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    @api_router.get("/api/agents/{agent_id}/artifacts/{name:path}")
    async def api_get_artifact(agent_id: str, name: str) -> dict:
        """Fetch artifact content from an agent's workspace."""
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        if transport is None:
            raise HTTPException(status_code=503, detail="Transport not available")
        try:
            result = await transport.request(agent_id, "GET", f"/artifacts/{name}", timeout=30)
            if "error" in result:
                status = result.get("status_code", 502)
                raise HTTPException(status_code=status, detail=result["error"])
            return result
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    @api_router.delete("/api/agents/{agent_id}/artifacts/{name:path}")
    async def api_delete_artifact(agent_id: str, name: str) -> dict:
        """Delete an artifact file from an agent's workspace."""
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        if transport is None:
            raise HTTPException(status_code=503, detail="Transport not available")
        try:
            result = await transport.request(agent_id, "DELETE", f"/artifacts/{name}", timeout=10)
            if "error" in result:
                status = result.get("status_code", 502)
                raise HTTPException(status_code=status, detail=result["error"])
            return result
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    @api_router.get("/api/agents/{agent_id}/files")
    async def api_list_files(
        agent_id: str,
        path: str = ".",
        recursive: bool = False,
        pattern: str = "*",
    ) -> dict:
        """List files under the agent's /data volume."""
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        if transport is None:
            raise HTTPException(status_code=503, detail="Transport not available")
        try:
            qs = f"path={path}&recursive={'true' if recursive else 'false'}&pattern={pattern}"
            return await transport.request(agent_id, "GET", f"/files?{qs}", timeout=10)
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    @api_router.get("/api/agents/{agent_id}/files/{path:path}")
    async def api_read_file(agent_id: str, path: str) -> dict:
        """Read a file from the agent's /data volume."""
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        if transport is None:
            raise HTTPException(status_code=503, detail="Transport not available")
        try:
            result = await transport.request(agent_id, "GET", f"/files/{path}", timeout=30)
            if "error" in result:
                status = result.get("status_code", 404)
                raise HTTPException(status_code=status, detail=result["error"])
            return result
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    @api_router.post("/api/credentials/validate")
    async def api_validate_credential(request: Request) -> dict:
        """Validate an API key by making a minimal LLM call."""
        body = await request.json()
        service = body.get("service", "").strip().lower()
        key = "".join(body.get("key", "").split())
        base_url = body.get("base_url", "").strip() or None
        if not service or not key:
            raise HTTPException(status_code=400, detail="service and key are required")

        # Anthropic OAuth setup-token: validate directly against provider API
        from src.host.credentials import is_oauth_token
        if is_oauth_token(key):
            from src.setup_wizard import SetupWizard
            fmt_error = SetupWizard._validate_oauth_token_format(key)
            if fmt_error:
                return {"valid": False, "skipped": False, "reason": fmt_error}
            import asyncio
            valid = await asyncio.get_running_loop().run_in_executor(
                None, SetupWizard._validate_oauth_token_live, key,
            )
            if valid:
                return {"valid": True, "skipped": False, "oauth": True}
            return {"valid": False, "skipped": False, "reason": "Invalid or expired setup-token"}

        # OAuth JSON blob detection (Anthropic or OpenAI)
        import json as _json

        from src.host.credentials import CredentialVault as _CV
        try:
            parsed = _json.loads(key)
            if isinstance(parsed, dict):
                # Anthropic OAuth (access_token starts with sk-ant-oat)
                if parsed.get("access_token", "").startswith("sk-ant-oat"):
                    return {"valid": True, "skipped": False, "oauth": True}
                # OpenAI OAuth (flat or nested Codex CLI format)
                if _CV.normalize_openai_oauth(parsed) is not None:
                    return {"valid": True, "skipped": False, "oauth": True}
        except (_json.JSONDecodeError, ValueError):
            pass

        # Strip _api_key suffix to get provider name
        provider = service.replace("_api_key", "")
        from src.setup_wizard import _VALIDATION_MODELS
        validation_model = _VALIDATION_MODELS.get(provider)
        if not validation_model:
            if base_url:
                # Custom provider with base URL — attempt OpenAI-compatible validation
                try:
                    import litellm
                    custom_kwargs: dict = {
                        "model": "openai/test",
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 1,
                        "api_key": key,
                        "api_base": base_url,
                    }
                    await litellm.acompletion(**custom_kwargs)
                    return {"valid": True, "skipped": False}
                except ImportError:
                    return {"valid": True, "skipped": True, "reason": "litellm not installed"}
                except Exception as e:
                    if isinstance(e, litellm.AuthenticationError):
                        return {"valid": False, "skipped": False, "reason": "Invalid API key"}
                    if isinstance(e, getattr(litellm, "PermissionDeniedError", type(None))):
                        return {"valid": False, "skipped": False, "reason": "Permission denied — check API key"}
                    emsg = str(e).lower()
                    _auth_kw = ("invalid api key", "invalid key", "unauthorized", "authentication fail")
                    if any(k in emsg for k in _auth_kw):
                        return {"valid": False, "skipped": False, "reason": "Invalid API key"}
                    # Non-auth errors (model not found, etc.) suggest key is probably valid
                    return {"valid": True, "skipped": False}
            return {"valid": True, "skipped": True, "reason": "unknown provider"}
        try:
            import litellm
            kwargs: dict = {
                "model": validation_model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
                "api_key": key,
            }
            if base_url:
                kwargs["api_base"] = base_url
            await litellm.acompletion(**kwargs)
            return {"valid": True, "skipped": False}
        except ImportError:
            return {"valid": True, "skipped": True, "reason": "litellm not installed"}
        except Exception as e:
            if isinstance(e, litellm.AuthenticationError):
                return {"valid": False, "skipped": False, "reason": "Invalid API key"}
            if isinstance(e, getattr(litellm, "PermissionDeniedError", type(None))):
                return {"valid": False, "skipped": False, "reason": "Permission denied — check API key"}
            # Some providers wrap auth errors as BadRequest/APIConnection —
            # check message before treating as transient
            emsg = str(e).lower()
            _auth_keywords = ("invalid api key", "invalid key", "invalid x-api-key",
                              "authentication fail", "login fail", "unauthorized",
                              "api key", "api_key", "secret key")
            if any(kw in emsg for kw in _auth_keywords):
                return {"valid": False, "skipped": False, "reason": "Invalid API key"}
            if isinstance(e, (litellm.Timeout, litellm.RateLimitError,
                              litellm.ServiceUnavailableError)):
                return {"valid": True, "skipped": True, "reason": str(e)[:200]}
            if isinstance(e, litellm.APIConnectionError):
                return {"valid": True, "skipped": True, "reason": str(e)[:200]}
            return {"valid": False, "skipped": False, "reason": f"Validation failed: {str(e)[:200]}"}

    @api_router.post("/api/credentials")
    async def api_add_credential(request: Request) -> dict:
        if credential_vault is None:
            raise HTTPException(status_code=503, detail="Credential vault not available")
        body = await request.json()
        service = body.get("service", "").strip()
        key = "".join(body.get("key", "").split())
        if not service or not key:
            raise HTTPException(status_code=400, detail="service and key are required")
        if not re.match(r"^[a-zA-Z0-9_.-]{1,128}$", service):
            raise HTTPException(
                status_code=400,
                detail="Invalid service name (alphanumeric, _, ., - only, max 128 chars)",
            )
        if len(key) > 10_000:
            raise HTTPException(status_code=400, detail="Key value too long (max 10000 chars)")
        # Detect OAuth JSON blobs (compact JSON after whitespace strip is still valid)
        import json as _json

        from src.host.credentials import CredentialVault as _CV
        try:
            parsed = _json.loads(key)
            if isinstance(parsed, dict):
                # Check for Anthropic OAuth (access_token starts with sk-ant-oat)
                if parsed.get("access_token", "").startswith("sk-ant-oat"):
                    credential_vault.store_anthropic_oauth(parsed)
                    credential_vault.remove_credential("anthropic_api_key")
                    return {"stored": True, "service": "anthropic_oauth", "tier": "system"}
                # Check for OpenAI OAuth (flat or nested Codex CLI format)
                normalized = _CV.normalize_openai_oauth(parsed)
                if normalized is not None:
                    credential_vault.store_openai_oauth(normalized)
                    return {"stored": True, "service": "openai_oauth", "tier": "system"}
        except (_json.JSONDecodeError, ValueError):
            pass
        # Detect bare Anthropic OAuth setup tokens (sk-ant-oat01-...)
        # Store as structured OAuth so they use the primary OAuth path
        from src.host.credentials import is_oauth_token
        if is_oauth_token(key):
            credential_vault.store_anthropic_oauth({"access_token": key})
            # Clear any stale api_key credential to avoid confusion
            credential_vault.remove_credential("anthropic_api_key")
            return {"stored": True, "service": "anthropic_oauth", "tier": "system"}
        # Normalize bare provider names
        from src.host.credentials import (
            SYSTEM_CREDENTIAL_PROVIDERS,
            is_system_credential,
        )
        if service.lower() in SYSTEM_CREDENTIAL_PROVIDERS and not service.lower().endswith("_api_key"):
            service = f"{service}_api_key"
        # Explicit tier override from request body, then auto-detect
        tier_field = body.get("tier", "").strip().lower()
        is_system = tier_field == "system" or is_system_credential(service)
        # Normalize custom LLM provider names to end with _api_key so
        # credential resolution and provider detection work correctly
        if body.get("custom_llm_models", "").strip() and not service.lower().endswith("_api_key"):
            service = f"{service.lower()}_api_key"
            is_system = True  # LLM provider keys are always system-tier
        credential_vault.add_credential(service, key, system=is_system)
        # If storing a regular Anthropic API key, clear any stale OAuth
        # credential so the API key path is used at runtime
        if service.lower() == "anthropic_api_key" and not is_oauth_token(key):
            if credential_vault._has_anthropic_oauth():
                credential_vault.remove_credential("anthropic_oauth")
        # Store optional custom API base URL alongside the key
        base_url = body.get("base_url", "").strip()
        if base_url:
            provider = service.replace("_api_key", "")
            credential_vault.add_credential(f"{provider}_api_base", base_url, system=is_system)
        # Store custom LLM provider models alongside the credential
        custom_models_raw = body.get("custom_llm_models", "").strip()
        if custom_models_raw and is_system:
            provider_name = service.replace("_api_key", "")
            models = [m.strip() for m in custom_models_raw.split(",") if m.strip()]
            models = [m if "/" in m else f"{provider_name}/{m}" for m in models]
            if models:
                with _settings_lock:
                    settings = _load_settings()
                    custom_providers = settings.setdefault("custom_llm_providers", {})
                    custom_providers[provider_name] = {
                        "label": body.get("custom_llm_label", "").strip() or provider_name.replace("_", " ").title(),
                        "models": models,
                    }
                    _save_settings(settings)
        tier = "system" if is_system else "agent"
        return {"stored": True, "service": service, "tier": tier}

    @api_router.post("/api/credentials/agent")
    async def api_add_agent_credential(request: Request) -> dict:
        """Store an agent-tier credential from a chat credential-request card.

        Unlike POST /api/credentials, this endpoint:
        - Always stores as agent-tier (never promotes to system credentials)
        - Rejects system credential names
        - Preserves the exact submitted value (only trims leading/trailing whitespace)
        """
        if credential_vault is None:
            raise HTTPException(status_code=503, detail="Credential vault not available")
        body = await request.json()
        service = body.get("service", "").strip()
        key = body.get("key", "").strip()
        if not service or not key:
            raise HTTPException(status_code=400, detail="service and key are required")
        if not re.match(r"^[a-zA-Z0-9_.-]{1,128}$", service):
            raise HTTPException(
                status_code=400,
                detail="Invalid credential name (alphanumeric, _, ., - only, max 128 chars)",
            )
        if len(key) > 10_000:
            raise HTTPException(status_code=400, detail="Key value too long (max 10000 chars)")
        from src.host.credentials import is_system_credential
        if is_system_credential(service):
            raise HTTPException(
                status_code=403,
                detail=f"Cannot store system credential via chat card: {service}",
            )
        credential_vault.add_credential(service, key)
        return {"stored": True, "service": service, "tier": "agent"}

    @api_router.post("/api/browser-login/complete")
    async def api_browser_login_complete(request: Request) -> dict:
        """User completed browser login — notify the requesting agent."""
        body = await request.json()
        agent_id = body.get("agent_id", "").strip()
        service = body.get("service", "").strip()[:128]
        if not agent_id or not service:
            raise HTTPException(status_code=400, detail="agent_id and service are required")
        # Notify the agent that login is complete
        if agent_id in agent_registry and lane_manager is not None:
            from src.shared.trace import new_trace_id
            from src.shared.utils import sanitize_for_prompt
            try:
                msg = sanitize_for_prompt(
                    f"The user has completed the browser login for {service}. "
                    f"The session (cookies, localStorage) is now saved in your browser profile. "
                    f"You can resume using browser tools to interact with {service}."
                )
                await lane_manager.enqueue(
                    agent_id, msg, mode="steer", trace_id=new_trace_id(),
                )
            except Exception:
                pass
        if event_bus:
            event_bus.emit("browser_login_completed", agent=agent_id, data={"service": service})
        return {"completed": True, "agent_id": agent_id, "service": service}

    @api_router.post("/api/browser-login/cancel")
    async def api_browser_login_cancel(request: Request) -> dict:
        """User cancelled browser login — notify the requesting agent."""
        body = await request.json()
        agent_id = body.get("agent_id", "").strip()
        service = body.get("service", "").strip()[:128]
        if not agent_id or not service:
            raise HTTPException(status_code=400, detail="agent_id and service are required")
        if agent_id in agent_registry and lane_manager is not None:
            from src.shared.trace import new_trace_id
            from src.shared.utils import sanitize_for_prompt
            try:
                msg = sanitize_for_prompt(
                    f"The user cancelled the browser login for {service}. "
                    f"You may need to find an alternative approach or ask again later."
                )
                await lane_manager.enqueue(
                    agent_id, msg, mode="steer", trace_id=new_trace_id(),
                )
            except Exception:
                pass
        if event_bus:
            event_bus.emit("browser_login_cancelled", agent=agent_id, data={"service": service})
        return {"cancelled": True, "agent_id": agent_id, "service": service}

    @api_router.post("/api/credentials/upload-env")
    async def api_upload_env(request: Request, file: UploadFile = File(...)) -> dict:
        """Bulk-import credentials from an uploaded .env file.

        Parses KEY=VALUE pairs (skips comments and blank lines) and stores each
        as a credential in the vault.  Values are never logged or returned.
        Returns the count of credentials loaded and the list of key names only.
        """
        if credential_vault is None:
            raise HTTPException(status_code=503, detail="Credential vault not available")

        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > 64 * 1024:
                    raise HTTPException(status_code=413, detail="File too large (max 64KB)")
            except ValueError:
                pass  # Malformed header; body-length check below is authoritative

        content = await file.read(65537)  # 64*1024 + 1
        if len(content) > 64 * 1024:
            raise HTTPException(status_code=413, detail="File too large (max 64KB)")
        if not content.strip():
            raise HTTPException(status_code=400, detail="Empty file")

        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="File must be UTF-8 encoded")

        from src.host.credentials import is_system_credential

        loaded_keys: list[str] = []
        parse_errors: list[str] = []

        for line_num, raw_line in enumerate(text.splitlines(), 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:]
            if "=" not in line:
                parse_errors.append(f"Line {line_num}: missing '=' separator")
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            if not _ENV_KEY_RE.match(key):
                parse_errors.append(f"Line {line_num}: invalid key name '{key}'")
                continue
            if not value:
                parse_errors.append(f"Line {line_num}: empty value for key '{key}'")
                continue
            is_system = is_system_credential(key)
            try:
                credential_vault.add_credential(key, value, system=is_system)
                loaded_keys.append(key)
            except Exception as exc:
                parse_errors.append(f"Line {line_num}: failed to store '{key}': {exc}")

        if not loaded_keys and parse_errors:
            raise HTTPException(
                status_code=400,
                detail=f"No valid credentials found. Errors: {'; '.join(parse_errors)}",
            )

        return {"count": len(loaded_keys), "keys": loaded_keys, "errors": parse_errors}

    @api_router.delete("/api/credentials/{name}")
    async def api_remove_credential(name: str) -> dict:
        if credential_vault is None:
            raise HTTPException(status_code=503, detail="Credential vault not available")
        existed = credential_vault.remove_credential(name)
        if not existed:
            raise HTTPException(status_code=404, detail=f"Credential '{name}' not found")
        # Clean up custom LLM provider config and paired api_base
        if name.endswith("_api_key"):
            provider_name = name[: -len("_api_key")]
            credential_vault.remove_credential(f"{provider_name}_api_base")
            with _settings_lock:
                settings = _load_settings()
                custom_providers = settings.get("custom_llm_providers", {})
                if provider_name in custom_providers:
                    del custom_providers[provider_name]
                    _save_settings(settings)
        return {"removed": True, "service": name}

    @api_router.get("/api/credentials/{name}/value")
    async def api_credential_value(name: str, request: Request) -> dict:
        """Reveal a masked credential value. Dashboard-auth gated."""
        _verify_dashboard_auth(request)
        if credential_vault is None:
            raise HTTPException(status_code=503, detail="Credential vault not available")
        # Check agent tier first, then system tier
        value = credential_vault.resolve_credential(name)
        if value is None:
            value = credential_vault.system_credentials.get(name.lower())
        if value is None:
            value = credential_vault.api_bases.get(name.lower())
        if value is None:
            raise HTTPException(status_code=404, detail=f"Credential '{name}' not found")
        masked = value[-4:].rjust(len(value), "*") if len(value) > 4 else "****"
        return {"name": name, "value": masked}

    # ── External API key management ─────────────────────────

    @api_router.get("/api/external-api-keys")
    async def api_list_external_keys(request: Request) -> dict:
        """List all named API keys (metadata only, never raw keys)."""
        _verify_dashboard_auth(request)
        if api_key_manager is None:
            return {"keys": [], "legacy": False}
        keys = api_key_manager.list_keys()
        legacy = bool(os.environ.get("OPENLEGION_API_KEY", ""))
        return {"keys": keys, "legacy": legacy}

    @api_router.post("/api/external-api-keys")
    async def api_create_external_key(request: Request) -> dict:
        """Create a named API key. Returns the raw key once."""
        _verify_dashboard_auth(request)
        if api_key_manager is None:
            raise HTTPException(status_code=503, detail="API key manager not available")
        body = await request.json()
        name = (body.get("name") or "").strip()
        if not name:
            raise HTTPException(status_code=400, detail="name is required")
        if len(name) > 128:
            raise HTTPException(status_code=400, detail="name must be 128 characters or fewer")
        try:
            key_id, raw_key = api_key_manager.create_key(name)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"id": key_id, "name": name, "key": raw_key}

    @api_router.delete("/api/external-api-keys/{key_id}")
    async def api_revoke_external_key(key_id: str, request: Request) -> dict:
        """Revoke an API key by ID."""
        _verify_dashboard_auth(request)
        if api_key_manager is None:
            raise HTTPException(status_code=503, detail="API key manager not available")
        if not api_key_manager.revoke_key(key_id):
            raise HTTPException(status_code=404, detail=f"API key not found: {key_id}")
        return {"revoked": True, "id": key_id}

    # ── Wallet management ────────────────────────────────────

    @api_router.post("/api/wallet/init")
    async def api_wallet_init(request: Request) -> dict:
        """Generate a master wallet seed and store it in .env."""
        _verify_dashboard_auth(request)
        if os.environ.get("OPENLEGION_SYSTEM_WALLET_MASTER_SEED"):
            raise HTTPException(
                status_code=409,
                detail="Master seed already configured. Remove "
                "OPENLEGION_SYSTEM_WALLET_MASTER_SEED from .env to reset.",
            )
        try:
            from mnemonic import Mnemonic
        except ImportError:
            raise HTTPException(
                status_code=503,
                detail="mnemonic package not installed. Run: pip install mnemonic",
            )
        from src.host.credentials import _persist_to_env

        mnemo = Mnemonic("english")
        words = mnemo.generate(strength=256)
        _persist_to_env("OPENLEGION_SYSTEM_WALLET_MASTER_SEED", words)
        os.environ["OPENLEGION_SYSTEM_WALLET_MASTER_SEED"] = words

        # Hot-load WalletService into the shared ref so mesh endpoints
        # work immediately — no restart needed.
        addresses = {}
        _ws_ref = wallet_service_ref or [None]
        try:
            from src.host.wallet import WalletService

            ws = WalletService()
            _ws_ref[0] = ws
            addresses["evm"] = ws._derive_evm_account(0).address
            addresses["solana"] = str(ws._derive_solana_keypair(0).pubkey())
        except Exception as e:
            logger.warning("WalletService init failed: %s", e)

        from starlette.responses import JSONResponse
        return JSONResponse(
            content={"initialized": True, "seed": words, "sample_addresses": addresses},
            headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
        )

    @api_router.get("/api/wallet/seed")
    async def api_wallet_seed(request: Request) -> dict:
        """Seed reveal removed for security.

        The seed is shown once at /api/wallet/init time. After that,
        revealing it requires re-provisioning the wallet. This prevents
        exfiltration via XSS or CSRF on the dashboard.
        """
        raise HTTPException(
            status_code=410,
            detail="Seed reveal disabled. The seed was shown at wallet init time. "
            "Re-initialize the wallet to generate a new seed.",
        )

    @api_router.get("/api/wallet/addresses")
    async def api_wallet_addresses(request: Request) -> dict:
        """List all agent wallet addresses."""
        _verify_dashboard_auth(request)
        seed = os.environ.get("OPENLEGION_SYSTEM_WALLET_MASTER_SEED")
        if not seed:
            return {"configured": False, "agents": []}

        _ws_ref = wallet_service_ref or [None]
        ws = _ws_ref[0]
        temp_ws = None
        if ws is None:
            try:
                from src.host.wallet import WalletService
                temp_ws = WalletService()
                ws = temp_ws
            except Exception as e:
                return {"configured": True, "agents": [], "error": str(e)}

        try:
            # Build agent list from the live registry (not stale DB entries).
            # Derive addresses for agents that have wallet enabled.
            live_agents = set(agent_registry.keys())
            agents = []
            all_agent_wallet_status = []

            for aid in sorted(live_agents):
                enabled = permissions.can_use_wallet(aid) if permissions else False
                chains = (
                    permissions.get_permissions(aid).wallet_allowed_chains
                    if permissions else []
                )
                status_entry: dict = {
                    "agent_id": aid,
                    "wallet_enabled": enabled,
                    "wallet_chains": chains,
                }
                if enabled:
                    try:
                        evm = await ws.get_address(aid, "evm:ethereum")
                        sol = await ws.get_address(aid, "solana:mainnet")
                        agents.append({
                            "agent_id": aid,
                            "evm_address": evm,
                            "solana_address": sol,
                        })
                        status_entry["has_addresses"] = True
                    except Exception:
                        status_entry["has_addresses"] = False
                else:
                    status_entry["has_addresses"] = False
                all_agent_wallet_status.append(status_entry)

            return {
                "configured": True,
                "agents": agents,
                "all_agents": all_agent_wallet_status,
            }
        except Exception as e:
            return {"configured": True, "agents": [], "error": str(e)}
        finally:
            if temp_ws is not None:
                temp_ws.close()

    @api_router.post("/api/wallet/enable/{agent_id}")
    async def api_wallet_enable_agent(agent_id: str, request: Request) -> dict:
        """Quick-enable wallet for an agent with all chains."""
        _verify_dashboard_auth(request)
        if permissions is None:
            raise HTTPException(status_code=503, detail="Permissions not available")
        from src.cli.config import _load_permissions, _save_permissions

        perms_data = _load_permissions()
        agent_perms = perms_data.get("permissions", {}).get(agent_id, {})
        agent_perms["can_use_wallet"] = True
        if not agent_perms.get("wallet_allowed_chains"):
            # Default to all known chains (not wildcard "*")
            _ws_local = (wallet_service_ref or [None])[0]
            if _ws_local:
                agent_perms["wallet_allowed_chains"] = list(_ws_local.chains.keys())
            else:
                agent_perms["wallet_allowed_chains"] = ["*"]
        perms_data.setdefault("permissions", {})[agent_id] = agent_perms
        _save_permissions(perms_data)
        permissions.reload()
        return {"enabled": True, "agent_id": agent_id}

    @api_router.get("/api/wallet/rpc")
    async def api_wallet_rpc(request: Request) -> dict:
        """List RPC URLs for all chains (current + default)."""
        _verify_dashboard_auth(request)
        from src.host.wallet import _CHAINS

        chains = []
        for chain_id, cfg in _CHAINS.items():
            env_key = cfg["rpc_env"]
            custom = os.environ.get(env_key, "")
            chains.append({
                "chain_id": chain_id,
                "label": _wallet_chain_label(chain_id, cfg),
                "rpc_env": env_key,
                "rpc_default": cfg["rpc_default"],
                "rpc_current": custom or cfg["rpc_default"],
                "is_custom": bool(custom),
            })
        return {"chains": chains}

    @api_router.put("/api/wallet/rpc")
    async def api_wallet_rpc_update(request: Request) -> dict:
        """Set or clear a custom RPC URL for a chain."""
        _verify_dashboard_auth(request)
        body = await request.json()
        chain_id = body.get("chain_id", "")
        rpc_url = body.get("rpc_url", "").strip()

        from src.host.wallet import _CHAINS

        if chain_id not in _CHAINS:
            raise HTTPException(status_code=400, detail=f"Unknown chain: {chain_id}")

        env_key = _CHAINS[chain_id]["rpc_env"]
        from src.host.credentials import _persist_to_env, _remove_from_env

        if rpc_url:
            # Validate URL format
            if not rpc_url.startswith(("http://", "https://")):
                raise HTTPException(
                    status_code=400, detail="RPC URL must start with http:// or https://",
                )
            _persist_to_env(env_key, rpc_url)
            os.environ[env_key] = rpc_url
        else:
            # Clear custom → revert to default
            _remove_from_env(env_key)
            os.environ.pop(env_key, None)

        # Hot-reload chains in the wallet service
        _ws_local = (wallet_service_ref or [None])[0]
        if _ws_local is not None:
            _ws_local._chains = _ws_local._load_chains()
            # Clear cached providers so they reconnect with new URLs
            _ws_local._evm_providers.pop(chain_id, None)
            _ws_local._solana_clients.pop(chain_id, None)

        return {"updated": True, "chain_id": chain_id}

    # ── Cost detail per agent ────────────────────────────────

    @api_router.get("/api/costs/{agent_id}")
    async def api_agent_costs(agent_id: str, period: str = "today") -> dict:
        if period not in {"today", "week", "month"}:
            period = "today"
        return cost_tracker.get_spend(agent_id, period)

    # ── Cost dashboard ───────────────────────────────────────

    _VALID_PERIODS = {"today", "week", "month"}

    @api_router.get("/api/costs")
    async def api_costs(period: str = "today") -> dict:
        if period not in _VALID_PERIODS:
            period = "today"
        agents_spend = cost_tracker.get_all_agents_spend(period)
        budgets = {}
        for item in agents_spend:
            budgets[item["agent"]] = cost_tracker.check_budget(item["agent"])
        # Include budgets for registered agents with zero spend in this period
        for aid in agent_registry:
            if aid not in budgets:
                budgets[aid] = cost_tracker.check_budget(aid)
        by_model = cost_tracker.get_spend_by_model(period)
        # Always include month-to-date totals for the stat card
        if period != "month":
            month = cost_tracker.get_spend(period="month")
            month_total = month["total_cost"]
            month_tokens = month["total_tokens"]
        else:
            month_total = sum(a["cost"] for a in agents_spend)
            month_tokens = sum(a["tokens"] for a in agents_spend)
        return {
            "period": period,
            "agents": agents_spend,
            "budgets": budgets,
            "by_model": by_model,
            "month_total": month_total,
            "month_tokens": month_tokens,
        }

    # ── Projects ──────────────────────────────────────────────

    @api_router.get("/api/projects")
    async def api_projects_list() -> dict:
        """List all projects with members."""
        from src.cli.config import _load_projects
        projects = _load_projects()
        result = []
        sorted_projects = sorted(projects.items(), key=lambda x: x[1].get("created_at") or "")
        for i, (pname, pdata) in enumerate(sorted_projects):
            is_over = _projects_disabled or (_max_projects > 0 and i >= _max_projects)
            result.append({
                "name": pname,
                "description": pdata.get("description", ""),
                "members": pdata.get("members", []),
                "created_at": pdata.get("created_at", ""),
                "over_limit": is_over,
            })
        return {"projects": result}

    @api_router.post("/api/projects")
    async def api_projects_create(request: Request) -> dict:
        """Create a new project."""
        if _projects_disabled:
            raise HTTPException(
                status_code=403,
                detail="Projects are not available on your current plan. Upgrade to enable projects.",
            )
        if _max_projects > 0:
            from src.cli.config import _load_projects
            current_count = len(_load_projects())
            if current_count >= _max_projects:
                raise HTTPException(
                    status_code=403,
                    detail=f"Project limit reached ({_max_projects}). Upgrade your plan for more projects.",
                )
        from src.cli.config import _create_project, _load_config
        body = await request.json()
        name = body.get("name", "").strip()
        description = sanitize_for_prompt(body.get("description", "")).strip()
        members = body.get("members", [])
        if not name:
            raise HTTPException(status_code=400, detail="name is required")
        if not isinstance(members, list):
            raise HTTPException(status_code=400, detail="members must be a list")
        # Validate that member agents exist in the config
        cfg = _load_config()
        known_agents = set(cfg.get("agents", {}).keys())
        unknown = [m for m in members if m not in known_agents]
        if unknown:
            raise HTTPException(status_code=400, detail=f"Unknown agents: {', '.join(unknown)}")
        try:
            _create_project(name, description=description, members=members)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"created": True, "name": name}

    @api_router.delete("/api/projects/{name}")
    async def api_projects_delete(name: str) -> dict:
        """Delete a project and release its members."""
        from src.cli.config import _delete_project
        try:
            _delete_project(name)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        return {"deleted": True, "name": name}

    @api_router.post("/api/projects/{name}/members")
    async def api_projects_add_member(name: str, request: Request) -> dict:
        """Add a member agent to a project."""
        from src.cli.config import _add_agent_to_project
        body = await request.json()
        agent = body.get("agent", "").strip()
        if not agent:
            raise HTTPException(status_code=400, detail="agent is required")
        try:
            _add_agent_to_project(name, agent)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        # Auto-restart the agent so new scope takes effect
        restarted = False
        if transport is not None and agent in agent_registry:
            try:
                await transport.request(agent, "POST", "/restart", timeout=10)
                restarted = True
            except Exception as e:
                logger.warning("Failed to restart agent %s after project change: %s", agent, e)
        return {"added": True, "project": name, "agent": agent, "restarted": restarted}

    @api_router.delete("/api/projects/{name}/members/{agent}")
    async def api_projects_remove_member(name: str, agent: str) -> dict:
        """Remove a member agent from a project."""
        from src.cli.config import _remove_agent_from_project
        try:
            _remove_agent_from_project(name, agent)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        # Auto-restart the agent so new scope takes effect
        restarted = False
        if transport is not None and agent in agent_registry:
            try:
                await transport.request(agent, "POST", "/restart", timeout=10)
                restarted = True
            except Exception as e:
                logger.warning("Failed to restart agent %s after project change: %s", agent, e)
        return {"removed": True, "project": name, "agent": agent, "restarted": restarted}

    # ── Project PROJECT.md ─────────────────────────────────

    def _resolve_project_path(project: str) -> Path:
        """Validate project name and return path to its project.md."""
        from src.cli.config import PROJECTS_DIR, _validate_project_name
        try:
            _validate_project_name(project)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid project name")
        return PROJECTS_DIR / project / "project.md"

    @api_router.get("/api/project")
    async def api_project_read(project: str = "") -> dict:
        """Read a project's project.md. Requires a project name."""
        if not project:
            raise HTTPException(status_code=400, detail="project parameter is required")
        if runtime is None:
            raise HTTPException(status_code=503, detail="Runtime not available")
        project_path = _resolve_project_path(project)
        if not project_path.parent.exists():
            raise HTTPException(status_code=404, detail=f"Project '{project}' not found")
        exists = project_path.exists()
        content = project_path.read_text(errors="replace")[:200_000] if exists else ""
        return {"content": content, "exists": exists, "project": project}

    @api_router.put("/api/project")
    async def api_project_write(request: Request, project: str = "") -> dict:
        """Write project.md to host and push to running agents."""
        if not project:
            raise HTTPException(status_code=400, detail="project parameter is required")
        if runtime is None:
            raise HTTPException(status_code=503, detail="Runtime not available")
        body = await request.json()
        content = body.get("content", "")
        if not isinstance(content, str):
            raise HTTPException(status_code=400, detail="content must be a string")
        content = sanitize_for_prompt(content)

        project_path = _resolve_project_path(project)
        if not project_path.parent.exists():
            raise HTTPException(status_code=404, detail=f"Project '{project}' not found")
        project_path.write_text(content)

        # Push to project members only
        from src.cli.config import _load_projects
        projects_data = _load_projects()
        pdata = projects_data.get(project, {})
        members = set(pdata.get("members", []))
        push_targets = [a for a in agent_registry.keys() if a in members]

        push_results = {}
        if transport is not None and push_targets:
            import asyncio as _asyncio

            async def _push(aid: str) -> tuple[str, bool]:
                try:
                    await transport.request(
                        aid, "PUT", "/project",
                        json={"content": content}, timeout=10,
                    )
                    return aid, True
                except Exception as e:
                    logger.warning("Failed to push PROJECT.md to %s: %s", aid, e)
                    return aid, False

            tasks = [_push(aid) for aid in push_targets]
            for coro in _asyncio.as_completed(tasks):
                aid, ok = await coro
                push_results[aid] = ok

        return {
            "saved": True,
            "size": project_path.stat().st_size,
            "pushed": push_results,
        }

    # ── Blackboard viewer + write/delete ─────────────────────

    @api_router.get("/api/blackboard")
    async def api_blackboard(prefix: str = "") -> dict:
        entries = blackboard.list_by_prefix(prefix)
        return {
            "prefix": prefix,
            "entries": [e.model_dump(mode="json") for e in entries],
        }

    def _parse_event_preview(data: str) -> dict:
        """Parse event data JSON and extract a human-readable preview."""
        result: dict = {}
        try:
            parsed = json.loads(data)
            if isinstance(parsed, dict):
                result["agent"] = parsed.get("source", parsed.get("agent", ""))
                for f in ("text", "summary", "status", "message",
                          "description", "name", "result"):
                    if f in parsed and isinstance(parsed[f], str):
                        result["preview"] = parsed[f][:200]
                        break
                if "preview" not in result:
                    result["preview"] = json.dumps(parsed, default=str)[:200]
            else:
                result["preview"] = str(parsed)[:200]
        except (ValueError, TypeError):
            result["preview"] = str(data)[:200]
        return result

    @api_router.get("/api/comms/activity")
    async def api_comms_activity(limit: int = 100, project: str = "") -> dict:
        """Recent inter-agent communication: blackboard writes/deletes + pubsub events."""

        limit = max(1, min(limit, 500))
        project_prefix = f"projects/{project}/" if project else ""
        activity: list[dict] = []

        # 1. Blackboard event_log (persisted in SQLite)
        try:
            if project_prefix:
                rows = blackboard.db.execute(
                    "SELECT event_type, key, agent_id, data, timestamp "
                    "FROM event_log WHERE key LIKE ? "
                    "ORDER BY id DESC LIMIT ?",
                    (project_prefix + "%", limit),
                ).fetchall()
            else:
                rows = blackboard.db.execute(
                    "SELECT event_type, key, agent_id, data, timestamp "
                    "FROM event_log ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            for event_type, key, agent_id, data, ts in rows:
                entry: dict = {
                    "source": "blackboard",
                    "action": event_type,
                    "key": key,
                    "agent": agent_id,
                    "timestamp": ts,
                }
                if data:
                    preview_info = _parse_event_preview(data)
                    # Don't overwrite agent from the database row
                    preview_info.pop("agent", None)
                    entry.update(preview_info)
                activity.append(entry)
        except Exception:
            pass  # event_log may not exist yet

        # 2. PubSub events (persisted in SQLite when db_path is set)
        if pubsub and getattr(pubsub, "_db", None) is not None:
            try:
                if project_prefix:
                    rows = pubsub._db.execute(
                        "SELECT topic, data, created_at "
                        "FROM events WHERE topic LIKE ? "
                        "ORDER BY id DESC LIMIT ?",
                        (project_prefix + "%", limit),
                    ).fetchall()
                else:
                    rows = pubsub._db.execute(
                        "SELECT topic, data, created_at "
                        "FROM events ORDER BY id DESC LIMIT ?",
                        (limit,),
                    ).fetchall()
                for topic, data, ts in rows:
                    entry = {
                        "source": "pubsub",
                        "action": "publish",
                        "topic": topic,
                        "timestamp": ts,
                    }
                    if data:
                        preview_info = _parse_event_preview(data)
                        entry.update(preview_info)
                    activity.append(entry)
            except Exception:
                pass

        # Sort merged activity by timestamp descending, then trim
        activity.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
        activity = activity[:limit]

        # Also return current pubsub subscriptions for context
        subs: dict[str, list[str]] = {}
        if pubsub:
            with pubsub._lock:
                for t, agents in pubsub.subscriptions.items():
                    if project_prefix and not t.startswith(project_prefix):
                        continue
                    subs[t] = list(agents)

        return {"activity": activity, "subscriptions": subs}

    _MAX_BB_KEY_LEN = 512
    _MAX_BB_VALUE_BYTES = 262_144  # 256 KB

    @api_router.put("/api/blackboard/{key:path}")
    async def api_blackboard_write(key: str, request: Request) -> dict:

        if len(key) > _MAX_BB_KEY_LEN:
            raise HTTPException(status_code=400, detail=f"Key too long ({len(key)} chars, max {_MAX_BB_KEY_LEN})")
        body = await request.json()
        value = body.get("value", {})
        if not isinstance(value, dict):
            raise HTTPException(status_code=400, detail="value must be a JSON object")
        value_size = len(json.dumps(value, default=str))
        if value_size > _MAX_BB_VALUE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Value too large ({value_size} bytes, max {_MAX_BB_VALUE_BYTES})",
            )
        # Always attribute to "dashboard" — never trust client-supplied written_by
        # to prevent impersonation of agents via the dashboard API.
        entry = blackboard.write(key, value, written_by="dashboard")
        return entry.model_dump(mode="json")

    @api_router.delete("/api/blackboard/{key:path}")
    async def api_blackboard_delete(key: str) -> dict:
        if key.startswith("history/"):
            raise HTTPException(status_code=400, detail="Cannot delete from history namespace")
        blackboard.delete(key, deleted_by="dashboard")
        return {"deleted": True, "key": key}

    # ── Trace inspector ──────────────────────────────────────

    @api_router.get("/api/traces")
    async def api_traces(limit: int = 50) -> dict:
        if trace_store is None:
            return {"traces": []}
        limit = max(1, min(limit, 200))
        return {"traces": trace_store.list_trace_summaries(limit)}

    @api_router.get("/api/traces/{trace_id}")
    async def api_trace_detail(trace_id: str) -> dict:
        if trace_store is None:
            raise HTTPException(status_code=404, detail="Trace store not configured")
        events = trace_store.get_trace(trace_id)
        if not events:
            raise HTTPException(status_code=404, detail="Trace not found")
        return {"trace_id": trace_id, "events": events}

    @api_router.get("/api/audit")
    async def api_audit(
        agent: str | None = None,
        event_type: str | None = None,
        since: float | None = None,
        until: float | None = None,
        limit: int = 100,
    ) -> dict:
        """Query audit trail with filters."""
        if trace_store is None:
            return {"events": [], "total": 0}
        limit = max(1, min(limit, 1000))
        events = trace_store.query(
            agent=agent, event_type=event_type,
            since=since, until=until, limit=limit,
        )
        return {"events": events, "total": len(events)}

    @api_router.get("/api/operator-audit")
    async def api_operator_audit(request: Request) -> dict:
        """Operator audit log backed by blackboard."""
        page = int(request.query_params.get("page", "1"))
        per_page = int(request.query_params.get("per_page", "20"))
        agent_id = request.query_params.get("agent_id", "")
        action_filter = request.query_params.get("action", "")
        since = request.query_params.get("since", "")
        if blackboard is None:
            return {"entries": [], "total": 0, "page": page, "per_page": per_page}
        return blackboard.get_audit_log(
            page=page, per_page=per_page, agent_id=agent_id,
            action=action_filter, since=since,
        )

    # ── Queue status ─────────────────────────────────────────

    @api_router.get("/api/queues")
    async def api_queues() -> dict:
        lane_status = lane_manager.get_status() if lane_manager else {}
        # Merge with agent registry so all agents appear (even idle ones)
        queues = {}
        for agent_id in agent_registry:
            queues[agent_id] = lane_status.get(agent_id, {
                "queued": 0, "pending": 0, "collected": 0, "busy": False,
            })
        # Include any lanes for agents not in registry (shouldn't happen, but safe)
        for agent_id, status in lane_status.items():
            if agent_id not in queues:
                queues[agent_id] = status
        return {"queues": queues}

    # ── Model health ──────────────────────────────────────

    @api_router.get("/api/model-health")
    async def api_model_health() -> dict:
        if credential_vault is None:
            return {"models": []}
        return {"models": credential_vault.get_model_health()}

    # ── Cron management ──────────────────────────────────

    @api_router.get("/api/cron")
    async def api_cron() -> dict:
        if cron_scheduler is None:
            return {"jobs": []}
        return {"jobs": cron_scheduler.list_jobs()}

    @api_router.post("/api/cron")
    async def api_cron_create(request: Request) -> dict:
        if cron_scheduler is None:
            raise HTTPException(status_code=503, detail="Cron scheduler not available")
        body = await request.json()
        agent = body.get("agent", "").strip()
        schedule = body.get("schedule", "").strip()
        message = body.get("message", "").strip()
        if not agent or not schedule or not message:
            raise HTTPException(status_code=400, detail="agent, schedule, and message are required")
        if agent not in agent_registry:
            raise HTTPException(status_code=400, detail=f"Agent '{agent}' not found")
        try:
            job = cron_scheduler.add_job(agent=agent, schedule=schedule, message=message)
            return {"created": True, "job_id": job.id}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @api_router.post("/api/cron/{job_id}/run")
    async def api_cron_run(job_id: str) -> dict:
        if cron_scheduler is None:
            raise HTTPException(status_code=503, detail="Cron scheduler not available")
        if job_id not in cron_scheduler.jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        # Fire-and-forget: dispatch in background so the HTTP response returns
        # immediately.  Agent execution can take minutes; blocking the request
        # made the dashboard Run button appear stuck.
        import asyncio
        task = asyncio.create_task(cron_scheduler.run_job(job_id))
        task.add_done_callback(_log_cron_task_exception)
        return {"triggered": True, "job_id": job_id}

    @api_router.put("/api/cron/{job_id}")
    async def api_cron_update(job_id: str, request: Request) -> dict:
        if cron_scheduler is None:
            raise HTTPException(status_code=503, detail="Cron scheduler not available")
        body = await request.json()
        if "schedule" in body:
            error = cron_scheduler._validate_schedule(body["schedule"])
            if error:
                raise HTTPException(status_code=400, detail=error)
        job = await cron_scheduler.update_job(job_id, **body)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return {"status": "updated", "job_id": job_id}

    @api_router.post("/api/cron/{job_id}/pause")
    async def api_cron_pause(job_id: str) -> dict:
        if cron_scheduler is None:
            raise HTTPException(status_code=503, detail="Cron scheduler not available")
        if not await cron_scheduler.pause_job(job_id):
            raise HTTPException(status_code=404, detail="Job not found")
        return {"paused": True, "job_id": job_id}

    @api_router.post("/api/cron/{job_id}/resume")
    async def api_cron_resume(job_id: str) -> dict:
        if cron_scheduler is None:
            raise HTTPException(status_code=503, detail="Cron scheduler not available")
        if not await cron_scheduler.resume_job(job_id):
            raise HTTPException(status_code=404, detail="Job not found")
        return {"resumed": True, "job_id": job_id}

    @api_router.delete("/api/cron/{job_id}")
    async def api_cron_delete(job_id: str) -> dict:
        if cron_scheduler is None:
            raise HTTPException(status_code=503, detail="Cron scheduler not available")
        job = cron_scheduler.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.heartbeat:
            raise HTTPException(status_code=403, detail="Heartbeat jobs cannot be deleted")
        if not cron_scheduler.remove_job(job_id):
            raise HTTPException(status_code=404, detail="Job not found")
        return {"deleted": True, "job_id": job_id}

    # ── Settings / environment ───────────────────────────────

    @api_router.get("/api/settings")
    async def api_settings() -> dict:
        from src.host.credentials import SYSTEM_CREDENTIAL_PROVIDERS
        from src.shared.models import get_all_model_costs

        app_url = os.environ.get("OPENLEGION_APP_URL", "")
        cred_names = credential_vault.list_credential_names() if credential_vault else []
        agent_cred_names = credential_vault.list_agent_credential_names() if credential_vault else []
        _llm_key_names = {f"{p}_api_key" for p in SYSTEM_CREDENTIAL_PROVIDERS}
        has_llm = bool(set(cred_names) & _llm_key_names)
        if not has_llm and "openlegion_api_key" in cred_names:
            has_llm = True
        if not has_llm and credential_vault:
            if credential_vault._has_anthropic_oauth() or credential_vault._has_openai_oauth():
                has_llm = True

        # Credit-awareness: distinguish BYOK keys from credit proxy
        credit_proxy_configured = "openlegion_api_key" in cred_names
        _curated_llm_names = set(cred_names) & _llm_key_names  # curated provider keys present
        has_oauth = (
            (credential_vault._has_anthropic_oauth() or credential_vault._has_openai_oauth())
            if credential_vault else False
        )
        has_byok_keys = bool(_curated_llm_names) or has_oauth

        pubsub_subs = pubsub.subscriptions if pubsub else {}

        # Filtered models: only providers with credentials
        available_provider_models: dict[str, list[str]] = {}
        if credential_vault:
            active_providers = credential_vault.get_providers_with_credentials()
            available_provider_models = {
                p: models for p, models in _PROVIDER_MODELS.items()
                if p in active_providers
            }

            # Merge custom LLM providers from settings
            settings = _load_settings()
            for prov, info in settings.get("custom_llm_providers", {}).items():
                if prov in active_providers:
                    available_provider_models[prov] = info.get("models", [])
                    has_llm = True

            # Discover locally-installed Ollama models and merge them in.
            # Only adds Ollama to the dropdown if it's actually reachable.
            try:
                discovered = await credential_vault.discover_ollama_models()
                if discovered:
                    featured = available_provider_models.get("ollama", [])
                    discovered_set = set(discovered)
                    merged = discovered + [
                        m for m in featured if m not in discovered_set
                    ]
                    available_provider_models["ollama"] = merged
                    has_llm = True
            except Exception:
                pass  # Keep whatever's already there (if any)

            # Discover models available through the OpenLegion credit proxy.
            # Similar to Ollama discovery — merges gateway catalog into dropdown.
            try:
                ol_models, ol_pricing = await credential_vault.discover_openlegion_models()
                if ol_models:
                    from src.shared.models import set_gateway_pricing
                    set_gateway_pricing(ol_pricing)
                    featured = available_provider_models.get("openlegion", [])
                    featured_set = set(featured)
                    merged = featured + [m for m in ol_models if m not in featured_set]
                    available_provider_models["openlegion"] = merged
                    has_llm = True
            except Exception:
                pass

        all_costs = get_all_model_costs()

        # Include gateway pricing for openlegion models.  Gateway pricing
        # overwrites litellm/fallback costs so the dashboard displays the
        # same prices that the credit proxy actually charges.
        if "openlegion" in available_provider_models:
            from src.shared.models import get_gateway_pricing
            for gw_model, cost in get_gateway_pricing().items():
                all_costs[f"openlegion/{gw_model}"] = cost

        return {
            "credentials": {"names": cred_names, "count": len(cred_names)},
            "agent_credentials": agent_cred_names,
            "has_llm_credentials": has_llm,
            "pubsub_subscriptions": pubsub_subs,
            "model_costs": {k: {"input_per_1k": v[0], "output_per_1k": v[1]} for k, v in all_costs.items()},
            "provider_models": dict(_PROVIDER_MODELS.items()),
            "available_provider_models": available_provider_models,
            "credit_proxy_configured": credit_proxy_configured,
            "has_byok_keys": has_byok_keys,
            "app_url": app_url,
            "plan_limits": {
                "max_agents": _max_agents,
                "max_projects": _max_projects,
                "projects_enabled": not _projects_disabled,
            },
        }

    # ── Browser settings ─────────────────────────────────────────

    _SETTINGS_PATH = Path("config/settings.json")
    _settings_lock = threading.Lock()

    def _load_settings() -> dict:
        """Load persisted settings from config/settings.json."""
        if _SETTINGS_PATH.exists():
            try:
                return json.loads(_SETTINGS_PATH.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_settings(settings: dict) -> None:
        """Persist settings to config/settings.json (atomic write)."""
        import tempfile
        _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(settings, indent=2) + "\n"
        fd, tmp_path = tempfile.mkstemp(
            dir=str(_SETTINGS_PATH.parent), suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w") as f:
                f.write(content)
        except BaseException:
            with contextlib.suppress(OSError):
                os.close(fd)
            Path(tmp_path).unlink(missing_ok=True)
            raise
        try:
            Path(tmp_path).replace(_SETTINGS_PATH)
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise

    @api_router.get("/api/browser-settings")
    async def api_get_browser_settings() -> dict:
        """Return saved browser speed and delay settings."""
        settings = _load_settings()
        return {
            "speed": settings.get("browser_speed", 1.0),
            "delay": settings.get("browser_delay", 0.0),
        }

    @api_router.post("/api/browser-settings")
    async def api_set_browser_settings(request: Request) -> dict:
        """Save browser speed/delay settings and push to browser service."""
        body = await request.json()
        speed = body.get("speed")
        delay = body.get("delay")

        if speed is None and delay is None:
            raise HTTPException(400, "speed or delay is required")

        payload: dict = {}

        if speed is not None:
            try:
                speed = float(speed)
            except (ValueError, TypeError):
                raise HTTPException(400, "speed must be a number")
            if speed < 0.25 or speed > 4.0:
                raise HTTPException(400, "speed must be between 0.25 and 4.0")
            payload["speed"] = speed

        if delay is not None:
            try:
                delay = float(delay)
            except (ValueError, TypeError):
                raise HTTPException(400, "delay must be a number")
            if delay < 0.0 or delay > 10.0:
                raise HTTPException(400, "delay must be between 0.0 and 10.0")
            payload["delay"] = delay

        # Persist to config file
        with _settings_lock:
            settings = _load_settings()
            if "speed" in payload:
                settings["browser_speed"] = payload["speed"]
            if "delay" in payload:
                settings["browser_delay"] = payload["delay"]
            _save_settings(settings)

        # Push to browser service immediately
        if runtime and hasattr(runtime, 'browser_service_url') and runtime.browser_service_url:
            try:
                browser_auth = getattr(runtime, 'browser_auth_token', '')
                headers = {}
                if browser_auth:
                    headers["Authorization"] = f"Bearer {browser_auth}"
                await _dashboard_browser_client.post(
                    f"{runtime.browser_service_url}/browser/settings",
                    json=payload,
                    headers=headers,
                )
            except Exception as e:
                logger.debug("Failed to push browser settings: %s", e)

        settings = _load_settings()
        return {
            "speed": settings.get("browser_speed", 1.0),
            "delay": settings.get("browser_delay", 0.0),
        }

    # ── System settings (consolidated) ────────────────────────

    _SYSTEM_SETTINGS_VALIDATORS: dict[str, tuple[type, float, float]] = {
        "default_daily_budget": (float, 0.01, 10000),
        "default_monthly_budget": (float, 0.01, 100000),
        "max_iterations": (int, 1, 100),
        "chat_max_tool_rounds": (int, 1, 200),
        "chat_max_total_rounds": (int, 10, 1000),
        "tool_timeout": (int, 10, 3600),
        "browser_idle_timeout": (int, 5, 120),
        "health_poll_interval": (int, 5, 300),
        "health_max_failures": (int, 1, 20),
        "health_restart_limit": (int, 0, 20),
        "health_restart_window": (int, 60, 86400),
    }

    _SYSTEM_SETTINGS_DEFAULTS: dict[str, float | int] = {
        "default_daily_budget": 10.0,
        "default_monthly_budget": 200.0,
        "max_iterations": 20,
        "chat_max_tool_rounds": 30,
        "chat_max_total_rounds": 200,
        "tool_timeout": 300,
        "browser_idle_timeout": 30,
        "health_poll_interval": 30,
        "health_max_failures": 3,
        "health_restart_limit": 3,
        "health_restart_window": 3600,
    }

    @api_router.get("/api/system-settings")
    async def api_get_system_settings() -> dict:
        """Return all system settings with defaults."""
        from src.cli.config import _load_config
        settings = _load_settings()
        result = {}
        for key, default in _SYSTEM_SETTINGS_DEFAULTS.items():
            result[key] = settings.get(key, default)
        # Include default_model from mesh.yaml
        cfg = _load_config()
        result["default_model"] = cfg.get("llm", {}).get("default_model", "openai/gpt-4o-mini")
        return result

    @api_router.post("/api/system-settings")
    async def api_set_system_settings(request: Request) -> dict:
        """Update system settings. Accepts a partial dict of settings."""
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(400, "Request body must be a JSON object")

        updated = []

        with _settings_lock:
            settings = _load_settings()

            for key, value in body.items():
                if key not in _SYSTEM_SETTINGS_VALIDATORS:
                    continue
                typ, min_val, max_val = _SYSTEM_SETTINGS_VALIDATORS[key]
                try:
                    coerced = typ(value)
                except (ValueError, TypeError):
                    raise HTTPException(400, f"{key} must be a {typ.__name__}")
                if coerced < min_val or coerced > max_val:
                    raise HTTPException(400, f"{key} must be between {min_val} and {max_val}")
                settings[key] = coerced
                updated.append(key)

            if updated:
                _save_settings(settings)

        # Apply health settings at runtime
        if health_monitor:
            _health_keys = {
                "health_poll_interval": "POLL_INTERVAL",
                "health_max_failures": "MAX_FAILURES",
                "health_restart_limit": "RESTART_LIMIT",
                "health_restart_window": "RESTART_WINDOW",
            }
            for cfg_key, attr in _health_keys.items():
                if cfg_key in updated:
                    setattr(health_monitor, attr, settings[cfg_key])

        return {"updated": updated}

    @api_router.post("/api/default-model")
    async def api_set_default_model(request: Request) -> dict:
        """Update the default LLM model in mesh.yaml."""
        import yaml
        body = await request.json()
        model = body.get("model", "").strip()
        if not model:
            raise HTTPException(400, "model is required")
        if not _is_valid_model(model):
            raise HTTPException(400, f"Unknown model: {model}")

        config_path = Path("config/mesh.yaml")
        mesh_cfg: dict = {}
        if config_path.exists():
            with open(config_path) as f:
                mesh_cfg = yaml.safe_load(f) or {}
        mesh_cfg.setdefault("llm", {})["default_model"] = model
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(mesh_cfg, f, default_flow_style=False, sort_keys=False)

        return {"model": model}

    # ── Restart agents ────────────────────────────────────────

    @api_router.post("/api/restart-agents")
    async def api_restart_agents() -> dict:
        """Restart all agent containers and the browser service.

        Re-reads config/settings.json and mesh.yaml so env-var-based
        settings (execution limits, browser idle timeout) take effect.
        """
        import asyncio as _asyncio

        if runtime is None:
            raise HTTPException(status_code=503, detail="Runtime not available")
        from src.cli.config import _load_config
        from src.host.runtime import DockerBackend

        # Refresh env vars from settings
        settings_path = Path("config/settings.json")
        if settings_path.exists():
            try:
                sys_settings = json.loads(settings_path.read_text())
                for env_key, cfg_key in {
                    "OPENLEGION_MAX_ITERATIONS": "max_iterations",
                    "OPENLEGION_CHAT_MAX_TOOL_ROUNDS": "chat_max_tool_rounds",
                    "OPENLEGION_CHAT_MAX_TOTAL_ROUNDS": "chat_max_total_rounds",
                    "OPENLEGION_TOOL_TIMEOUT": "tool_timeout",
                }.items():
                    if cfg_key in sys_settings:
                        runtime.extra_env[env_key] = str(sys_settings[cfg_key])
            except (ValueError, OSError):
                pass

        cfg = _load_config()
        agents_cfg = cfg.get("agents", {})
        default_model = cfg.get("llm", {}).get("default_model", "openai/gpt-4o-mini")
        loop = _asyncio.get_running_loop()
        results = {}

        # Restart browser service first (picks up idle timeout from settings)
        if isinstance(runtime, DockerBackend) and hasattr(runtime, "stop_browser_service"):
            try:
                await loop.run_in_executor(None, runtime.stop_browser_service)
                await loop.run_in_executor(None, runtime.start_browser_service)
                # Push saved speed/delay to the freshly started browser service
                await _push_browser_settings()
            except Exception as e:
                logger.warning("Browser service restart failed: %s", e)

        # Restart all agents in parallel
        _network_cfg = cfg.get("network", {})
        from src.cli.config import _OPERATOR_AGENT_ID, _OPERATOR_ALLOWED_TOOLS

        async def _restart_one(agent_id: str) -> tuple[str, str]:
            agent_cfg = agents_cfg.get(agent_id, {})
            try:
                await loop.run_in_executor(None, runtime.stop_agent, agent_id)
                skills_dir = agent_cfg.get("skills_dir", "")
                if skills_dir:
                    skills_dir = str(Path(skills_dir).resolve())
                # Per-agent env overrides (proxy + operator tools).
                # Proxy goes in env_overrides instead of runtime.extra_env
                # so parallel restarts don't stomp each other's proxy vars.
                _restart_env: dict[str, str] = {}
                if agent_id == _OPERATOR_AGENT_ID:
                    _restart_env["ALLOWED_TOOLS"] = ",".join(_OPERATOR_ALLOWED_TOOLS)
                _proxy_url = resolve_agent_proxy(agent_id, agents_cfg, _network_cfg)
                _proxy_env = build_proxy_env_vars(
                    _proxy_url, _network_cfg.get("no_proxy", ""),
                )
                _restart_env.update(_proxy_env)
                url = await loop.run_in_executor(
                    None,
                    lambda aid=agent_id, acfg=agent_cfg, sd=skills_dir, re=_restart_env: runtime.start_agent(
                        agent_id=aid,
                        role=acfg.get("role", "assistant"),
                        skills_dir=sd,
                        model=acfg.get("model", default_model),
                        mcp_servers=acfg.get("mcp_servers") or None,
                        thinking=acfg.get("thinking", ""),
                        env_overrides=re,
                    ),
                )
                if router is not None:
                    router.register_agent(agent_id, url, role=agent_cfg.get("role", ""))
                else:
                    agent_registry[agent_id] = url
                if transport is not None:
                    from src.host.transport import HttpTransport
                    if isinstance(transport, HttpTransport):
                        transport.register(agent_id, url)
                ready = await runtime.wait_for_agent(agent_id, timeout=60)
                await _push_browser_proxy_for_agent(agent_id)
                return (agent_id, "ready" if ready else "started")
            except Exception as e:
                logger.error("Failed to restart agent '%s': %s", agent_id, e)
                return (agent_id, f"error: {e}")

        agent_results = await _asyncio.gather(
            *[_restart_one(aid) for aid in list(agent_registry.keys())]
        )
        results = dict(agent_results)

        return {"restarted": results}

    # ── Storage ────────────────────────────────────────────────

    _STORAGE_SKIP_DIRS = {"src", ".git", ".venv", "venv", "node_modules", ".claude"}
    _STORAGE_DB_SUFFIXES = {".db", ".db-wal", ".db-shm"}

    def _scan_storage(root: Path) -> dict:
        """Scan project root for storage breakdown (blocking I/O).

        Uses os.walk with dir pruning to avoid descending into source,
        git, and virtualenv directories.
        """
        # System-wide disk usage
        try:
            disk = shutil.disk_usage(str(root))
            disk_info = {"total": disk.total, "used": disk.used, "free": disk.free}
        except OSError:
            disk_info = {"total": 0, "used": 0, "free": 0}

        db_bytes = 0
        log_bytes = 0
        config_bytes = 0
        agent_bytes = 0
        other_bytes = 0

        root_str = str(root)
        for dirpath, dirnames, filenames in os.walk(root):
            # Compute top-level directory relative to root
            rel = os.path.relpath(dirpath, root_str)
            top = rel.split(os.sep, 1)[0] if rel != "." else ""

            # Prune skipped directories so os.walk doesn't descend
            dirnames[:] = [
                d for d in dirnames
                if (d if rel == "." else top) not in _STORAGE_SKIP_DIRS
            ]

            for name in filenames:
                fpath = os.path.join(dirpath, name)
                try:
                    size = os.lstat(fpath).st_size
                except OSError:
                    continue

                # Categorize: files at root level use their own name/suffix,
                # files in subdirs are categorized by top-level dir
                _, suffix = os.path.splitext(name)
                suffix = suffix.lower()
                if suffix in _STORAGE_DB_SUFFIXES:
                    db_bytes += size
                elif suffix == ".log":
                    log_bytes += size
                elif top == "config":
                    config_bytes += size
                elif top == ".openlegion":
                    agent_bytes += size
                elif top == "data":
                    # data/ contains costs.db, traces.db (caught above by suffix);
                    # any other files in data/ are still engine data
                    other_bytes += size
                elif top:
                    other_bytes += size
                else:
                    # Root-level files that aren't db/log
                    other_bytes += size

        engine_total = db_bytes + log_bytes + config_bytes + agent_bytes + other_bytes
        return {
            "disk": disk_info,
            "engine": {
                "total": engine_total,
                "databases": db_bytes,
                "agent_data": agent_bytes,
                "logs": log_bytes,
                "config": config_bytes,
                "other": other_bytes,
            },
        }

    @api_router.get("/api/storage")
    async def api_storage() -> dict:
        """Return disk usage breakdown for the engine's data directory."""
        import asyncio

        project_root = (
            runtime.project_root if runtime and hasattr(runtime, "project_root")
            else Path(__file__).resolve().parent.parent.parent
        )
        return await asyncio.get_running_loop().run_in_executor(
            None, _scan_storage, project_root,
        )

    # ── Database details ─────────────────────────────────────

    _DB_REGISTRY = [
        {
            "id": "blackboard",
            "label": "Blackboard",
            "description": "Shared agent coordination state",
            "path": "blackboard.db",
            "tables": {
                "entries": {"ts_col": "created_at", "ts_type": "text"},
                "event_log": {"ts_col": "timestamp", "ts_type": "text"},
            },
            "purgeable": True,
        },
        {
            "id": "traces",
            "label": "Traces",
            "description": "Request execution traces and events",
            "path": "data/traces.db",
            "tables": {
                "traces": {"ts_col": "timestamp", "ts_type": "real"},
            },
            "purgeable": True,
        },
        {
            "id": "costs",
            "label": "Cost History",
            "description": "LLM token usage and spend tracking",
            "path": "data/costs.db",
            "tables": {
                "usage": {"ts_col": "timestamp", "ts_type": "text"},
            },
            "purgeable": True,
        },
        {
            "id": "wallet",
            "label": "Wallet",
            "description": "Transaction history and key indexes",
            "path": "data/wallet.db",
            "tables": {
                "transactions": {"ts_col": "timestamp", "ts_type": "text"},
                "agent_index": {"ts_col": "created_at", "ts_type": "text"},
            },
            "purgeable": False,
        },
    ]

    def _scan_database_details(root: Path) -> list[dict]:
        """Scan engine databases for record counts and metadata (blocking I/O)."""
        from datetime import datetime as _dt
        from datetime import timezone as _tz

        results = []
        for entry in _DB_REGISTRY:
            db_path = root / entry["path"]
            info: dict = {
                "id": entry["id"],
                "label": entry["label"],
                "description": entry["description"],
                "purgeable": entry["purgeable"],
                "size_bytes": 0,
                "tables": [],
                "total_records": 0,
                "oldest": None,
            }

            # Sum file sizes (.db + .db-wal + .db-shm)
            for suffix in ("", "-wal", "-shm"):
                p = db_path.parent / (db_path.name + suffix)
                try:
                    info["size_bytes"] += p.stat().st_size
                except OSError:
                    pass

            if not db_path.exists():
                results.append(info)
                continue

            try:
                conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
                try:
                    conn.execute("PRAGMA busy_timeout=2000")
                    oldest_ts = None
                    for table_name, meta in entry["tables"].items():
                        try:
                            row = conn.execute(
                                f"SELECT COUNT(*) FROM [{table_name}]"  # noqa: S608
                            ).fetchone()
                            count = row[0] if row else 0
                        except sqlite3.OperationalError:
                            count = 0
                        info["tables"].append({"name": table_name, "count": count})
                        info["total_records"] += count

                        # Find oldest timestamp
                        if count > 0:
                            ts_col = meta["ts_col"]
                            ts_type = meta["ts_type"]
                            try:
                                row = conn.execute(
                                    f"SELECT MIN([{ts_col}]) FROM [{table_name}]"  # noqa: S608
                                ).fetchone()
                                if row and row[0] is not None:
                                    if ts_type == "real":
                                        val = float(row[0])
                                        if oldest_ts is None or val < oldest_ts:
                                            oldest_ts = val
                                    else:
                                        try:
                                            dt = _dt.fromisoformat(
                                                str(row[0]).replace(" ", "T")
                                            )
                                            val = dt.replace(tzinfo=_tz.utc).timestamp()
                                            if oldest_ts is None or val < oldest_ts:
                                                oldest_ts = val
                                        except (ValueError, TypeError):
                                            pass
                            except sqlite3.OperationalError:
                                pass
                finally:
                    conn.close()

                if oldest_ts is not None:
                    info["oldest"] = oldest_ts
            except (sqlite3.Error, OSError) as exc:
                logger.debug("Failed to scan database %s: %s", db_path, exc)

            results.append(info)
        return results

    @api_router.get("/api/storage/databases")
    async def api_storage_databases() -> dict:
        """Return detailed per-database stats."""
        import asyncio

        project_root = (
            runtime.project_root if runtime and hasattr(runtime, "project_root")
            else Path(__file__).resolve().parent.parent.parent
        )
        databases = await asyncio.get_running_loop().run_in_executor(
            None, _scan_database_details, project_root,
        )
        return {"databases": databases}

    @api_router.post("/api/storage/databases/{db_id}/purge")
    async def api_purge_database(db_id: str, request: Request) -> dict:
        """Purge old records from a database."""
        import asyncio
        import time as _time

        # Find the database entry
        entry = next((e for e in _DB_REGISTRY if e["id"] == db_id), None)
        if entry is None:
            raise HTTPException(404, f"Unknown database: {db_id}")
        if not entry["purgeable"]:
            raise HTTPException(400, f"Database '{db_id}' cannot be purged")

        body = await request.json() if await request.body() else {}
        older_than_days = body.get("older_than_days")  # None means purge all
        if older_than_days is not None:
            if not isinstance(older_than_days, (int, float)) or older_than_days <= 0:
                raise HTTPException(400, "older_than_days must be a positive number")
            older_than_days = int(older_than_days)

        project_root = (
            runtime.project_root if runtime and hasattr(runtime, "project_root")
            else Path(__file__).resolve().parent.parent.parent
        )

        def _do_purge() -> dict:
            db_path = project_root / entry["path"]
            if not db_path.exists():
                return {"purged": True, "deleted_records": 0}

            conn = sqlite3.connect(str(db_path))
            try:
                conn.execute("PRAGMA busy_timeout=5000")
                total_deleted = 0

                for table_name, meta in entry["tables"].items():
                    ts_col = meta["ts_col"]
                    ts_type = meta["ts_type"]

                    try:
                        if older_than_days is None:
                            cur = conn.execute(
                                f"DELETE FROM [{table_name}]"  # noqa: S608
                            )
                        elif ts_type == "real":
                            cutoff = _time.time() - (older_than_days * 86400)
                            cur = conn.execute(
                                f"DELETE FROM [{table_name}] WHERE [{ts_col}] < ?",  # noqa: S608
                                (cutoff,),
                            )
                        else:
                            cur = conn.execute(
                                f"DELETE FROM [{table_name}] WHERE [{ts_col}] < datetime('now', ?)",  # noqa: S608
                                (f"-{older_than_days} days",),
                            )
                        total_deleted += cur.rowcount
                    except sqlite3.OperationalError as exc:
                        logger.warning("Purge %s.%s failed: %s", db_id, table_name, exc)

                conn.commit()

                # Best-effort VACUUM to reclaim disk space
                try:
                    conn.execute("VACUUM")
                except sqlite3.OperationalError:
                    pass
            finally:
                conn.close()

            return {"purged": True, "deleted_records": total_deleted}

        return await asyncio.get_running_loop().run_in_executor(None, _do_purge)

    # ── Messages log ─────────────────────────────────────────

    @api_router.get("/api/messages")
    async def api_messages() -> dict:
        if router is None:
            return {"messages": []}
        return {"messages": router.message_log[-100:]}

    # ── Webhooks ──────────────────────────────────────────────

    @api_router.get("/api/webhooks")
    async def api_webhooks_list(request: Request) -> dict:
        if webhook_manager is None:
            return {"webhooks": []}
        hooks = webhook_manager.list_hooks() if hasattr(webhook_manager, "list_hooks") else []
        base = str(request.base_url).rstrip("/")
        result = []
        for h in hooks:
            entry = {k: v for k, v in h.items() if k != "secret"}
            entry["url"] = f"{base}/webhook/hook/{h['id']}"
            entry["has_secret"] = "secret" in h
            result.append(entry)
        return {"webhooks": result}

    @api_router.post("/api/webhooks")
    async def api_webhooks_create(request: Request) -> dict:
        if webhook_manager is None:
            raise HTTPException(status_code=503, detail="Webhook manager not available")
        body = await request.json()
        name = body.get("name", "")
        agent = body.get("agent", "")
        if not name or not agent:
            raise HTTPException(status_code=400, detail="name and agent are required")
        require_signature = bool(body.get("secret"))
        instructions = body.get("instructions", "")
        hook = webhook_manager.add_hook(
            agent=agent,
            name=name,
            require_signature=require_signature,
            instructions=instructions,
        )
        base = str(request.base_url).rstrip("/")
        # Return a copy so we don't mutate the stored dict; include
        # secret once so the user can copy it at creation time.
        result = dict(hook)
        result["url"] = f"{base}/webhook/hook/{hook['id']}"
        return {"created": True, "hook": result}

    @api_router.delete("/api/webhooks/{hook_id}")
    async def api_webhooks_delete(hook_id: str) -> dict:
        if webhook_manager is None:
            raise HTTPException(status_code=503, detail="Webhook manager not available")
        removed = webhook_manager.remove_hook(hook_id)
        if not removed:
            raise HTTPException(status_code=404, detail=f"Webhook '{hook_id}' not found")
        return {"removed": True, "id": hook_id}

    @api_router.patch("/api/webhooks/{hook_id}")
    async def api_webhooks_update(hook_id: str, request: Request) -> dict:
        if webhook_manager is None:
            raise HTTPException(status_code=503, detail="Webhook manager not available")
        body = await request.json()
        fields: dict = {}
        for key in ("name", "agent", "instructions"):
            if key in body:
                fields[key] = body[key]
        if "require_signature" in body:
            fields["require_signature"] = bool(body["require_signature"])
        if body.get("regenerate_secret"):
            fields["regenerate_secret"] = True
        if not fields:
            raise HTTPException(status_code=400, detail="No valid fields provided")
        try:
            updated = webhook_manager.update_hook(hook_id, **fields)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if updated is None:
            raise HTTPException(status_code=404, detail=f"Webhook '{hook_id}' not found")
        base = str(request.base_url).rstrip("/")
        updated["url"] = f"{base}/webhook/hook/{updated['id']}"
        return {"updated": True, "hook": updated}

    @api_router.post("/api/webhooks/{hook_id}/test")
    async def api_webhooks_test(hook_id: str, request: Request) -> dict:
        if webhook_manager is None:
            raise HTTPException(status_code=503, detail="Webhook manager not available")
        body = await request.json()
        result = await webhook_manager.test_hook(hook_id, payload=body)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Webhook '{hook_id}' not found")
        return {"tested": True, "id": hook_id, "result": result}

    # ── Channels ──────────────────────────────────────────────

    _CHANNEL_TOKEN_KEYS: dict[str, list[tuple[str, str]]] = {
        "telegram": [("token", "TELEGRAM_BOT_TOKEN")],
        "discord": [("token", "DISCORD_BOT_TOKEN")],
        "slack": [("bot_token", "SLACK_BOT_TOKEN"), ("app_token", "SLACK_APP_TOKEN")],
        "whatsapp": [("access_token", "WHATSAPP_ACCESS_TOKEN"), ("phone_number_id", "WHATSAPP_PHONE_NUMBER_ID")],
    }

    @api_router.get("/api/channels")
    async def api_channels_list() -> dict:
        if channel_manager is None:
            return {"channels": []}
        return {"channels": channel_manager.get_channel_status()}

    @api_router.post("/api/channels/{channel_type}/connect")
    async def api_channel_connect(channel_type: str, request: Request) -> dict:
        if channel_manager is None:
            raise HTTPException(status_code=503, detail="Channel manager not available")
        if channel_type not in _CHANNEL_TOKEN_KEYS:
            raise HTTPException(status_code=400, detail=f"Unknown channel type: {channel_type}")
        body = await request.json()
        tokens = body.get("tokens", {})
        # Validate required token fields
        required = _CHANNEL_TOKEN_KEYS[channel_type]
        missing = [key for key, _env in required if not tokens.get(key)]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing required tokens: {', '.join(missing)}")
        # Persist tokens to credential vault before starting
        persisted_env_names: list[str] = []
        if credential_vault is not None:
            for token_key, env_name in required:
                val = tokens.get(token_key, "")
                if val:
                    credential_vault.add_credential(env_name, val, system=True)
                    persisted_env_names.append(env_name)

        def _rollback_credentials() -> None:
            for env_name in persisted_env_names:
                with contextlib.suppress(Exception):
                    credential_vault.remove_credential(env_name)

        try:
            routers = channel_manager.start_channel(channel_type, tokens)
            if routers:
                for ch_router in routers:
                    request.app.include_router(ch_router)
            return {"connected": True, "type": channel_type}
        except ValueError as e:
            _rollback_credentials()
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            _rollback_credentials()
            logger.error("Failed to connect channel %s: %s", channel_type, e)
            raise HTTPException(status_code=500, detail=str(e))

    @api_router.post("/api/channels/{channel_type}/disconnect")
    async def api_channel_disconnect(channel_type: str) -> dict:
        if channel_manager is None:
            raise HTTPException(status_code=503, detail="Channel manager not available")
        if channel_type not in _CHANNEL_TOKEN_KEYS:
            raise HTTPException(status_code=400, detail=f"Unknown channel type: {channel_type}")
        try:
            channel_manager.stop_channel(channel_type)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        # Remove tokens from credential vault
        if credential_vault is not None:
            for _token_key, env_name in _CHANNEL_TOKEN_KEYS[channel_type]:
                with contextlib.suppress(Exception):
                    credential_vault.remove_credential(env_name)
        return {"disconnected": True, "type": channel_type}

    # ── Agent Workspace (proxy to agent) ─────────────────────

    _WORKSPACE_ALLOWLIST = frozenset({
        "SOUL.md", "HEARTBEAT.md", "USER.md", "INSTRUCTIONS.md", "AGENTS.md", "MEMORY.md",
        "INTERFACE.md",
    })

    @api_router.get("/api/agents/{agent_id}/workspace")
    async def api_agent_workspace(agent_id: str) -> dict:
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        if transport is None:
            raise HTTPException(status_code=503, detail="Transport not available")
        try:
            return await transport.request(agent_id, "GET", "/workspace", timeout=10)
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    @api_router.get("/api/agents/{agent_id}/workspace/{filename}")
    async def api_agent_workspace_read(agent_id: str, filename: str) -> dict:
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        if filename not in _WORKSPACE_ALLOWLIST:
            raise HTTPException(status_code=400, detail=f"File not allowed: {filename}")
        if transport is None:
            raise HTTPException(status_code=503, detail="Transport not available")
        try:
            return await transport.request(
                agent_id, "GET", f"/workspace/{filename}", timeout=10,
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    @api_router.put("/api/agents/{agent_id}/workspace/{filename}")
    async def api_agent_workspace_write(agent_id: str, filename: str, request: Request) -> dict:
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        if filename not in _WORKSPACE_ALLOWLIST:
            raise HTTPException(status_code=400, detail=f"File not allowed: {filename}")
        if transport is None:
            raise HTTPException(status_code=503, detail="Transport not available")
        body = await request.json()
        content = body.get("content", "")
        if not isinstance(content, str):
            raise HTTPException(status_code=400, detail="content must be a string")
        content = sanitize_for_prompt(content)
        try:
            result = await transport.request(
                agent_id, "PUT", f"/workspace/{filename}",
                json={"content": content}, timeout=10,
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))
        if event_bus is not None:
            event_bus.emit("workspace_updated", agent=agent_id,
                           data={"message": f"Dashboard updated {filename}"})
        return result

    # ── Agent Workspace Logs + Learnings (proxy to agent) ─────

    @api_router.get("/api/agents/{agent_id}/workspace-logs")
    async def api_agent_workspace_logs(agent_id: str, days: int = 3) -> dict:
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        if transport is None:
            raise HTTPException(status_code=503, detail="Transport not available")
        days = max(1, min(days, 14))
        try:
            return await transport.request(
                agent_id, "GET", f"/workspace-logs?days={days}", timeout=10,
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    @api_router.get("/api/agents/{agent_id}/workspace-learnings")
    async def api_agent_workspace_learnings(agent_id: str) -> dict:
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        if transport is None:
            raise HTTPException(status_code=503, detail="Transport not available")
        try:
            return await transport.request(
                agent_id, "GET", "/workspace-learnings", timeout=10,
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    # ── Agent Activity Log ────────────────────────────────────

    @api_router.get("/api/agents/{agent_id}/activity")
    async def api_agent_activity(agent_id: str, limit: int = 100) -> dict:
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        if transport is None:
            raise HTTPException(status_code=503, detail="Transport not available")
        limit = max(1, min(limit, 500))
        try:
            return await transport.request(
                agent_id, "GET", f"/activity?limit={limit}", timeout=10,
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    # ── Logs ──────────────────────────────────────────────────

    @api_router.get("/api/logs")
    async def api_logs(lines: int = 100, level: str = "") -> dict:
        """Return recent log lines from .openlegion.log."""
        from src.cli.config import PROJECT_ROOT

        log_path = PROJECT_ROOT / ".openlegion.log"
        if not log_path.exists():
            return {"lines": [], "total": 0}

        content = log_path.read_text()
        all_lines = content.splitlines()

        if level:
            level_upper = level.upper()
            level_pat = re.compile(r'\b' + re.escape(level_upper) + r'\b')
            all_lines = [ln for ln in all_lines if level_pat.search(ln.upper())]

        result_lines = all_lines[-lines:]
        return {"lines": result_lines, "total": len(all_lines)}

    # ── Static files ─────────────────────────────────────────

    _MEDIA_TYPES = {
        ".css": "text/css",
        ".js": "application/javascript",
        ".png": "image/png",
        ".svg": "image/svg+xml",
        ".ico": "image/x-icon",
    }

    # ── User Uploads ─────────────────────────────────────────────────────
    # User-managed files that agents can read (read-only) and the VNC browser
    # can navigate to at http://localhost:8500/uploads/<filename>.
    # All endpoints read/write the host uploads dir directly — no transport.

    def _uploads_dir() -> Path:
        root = (
            runtime.project_root if runtime and hasattr(runtime, "project_root")
            else Path(__file__).resolve().parent.parent.parent
        )
        d = root / ".openlegion" / "uploads"
        d.mkdir(parents=True, exist_ok=True)
        return d

    _MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB

    def _safe_upload_path(name: str) -> Path:
        r"""Resolve upload path, blocking traversal, absolute paths, and null bytes.

        Two-stage check:
          1. Structural: reject absolute paths and any '..' component
             using Path.parts (platform-aware, handles both / and \).
          2. Symlink-safe: resolve and verify the final path is inside root.
        """
        try:
            p = Path(name)
        except (ValueError, TypeError):
            # ValueError is raised for embedded null bytes on all platforms.
            raise HTTPException(400, "Invalid path")
        if p.is_absolute() or ".." in p.parts:
            raise HTTPException(400, "Invalid path")
        root = _uploads_dir().resolve()
        candidate = (root / name).resolve()
        if not candidate.is_relative_to(root):
            raise HTTPException(400, "Path traversal not allowed")
        return candidate

    @api_router.get("/api/uploads")
    async def api_list_uploads() -> dict:
        """List all files in the uploads directory."""
        import mimetypes
        root = _uploads_dir()
        entries = []
        for f in sorted(root.rglob("*")):
            if not f.is_file():
                continue
            rel = str(f.relative_to(root))
            stat = f.stat()
            mime = mimetypes.guess_type(rel)[0] or "application/octet-stream"
            entries.append({
                "name": rel,
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "mime_type": mime,
            })
        return {"uploads": entries}

    @api_router.post("/api/uploads/{name:path}")
    async def api_upload_file(name: str, request: Request) -> dict:
        """Upload a file to the uploads directory.

        Accepts raw bytes in the request body.  The caller sets Content-Type
        so the browser download later uses the right MIME type.
        Maximum upload size: 50 MB.
        """
        dest = _safe_upload_path(name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        body = await request.body()
        if not body:
            raise HTTPException(400, "Empty body")
        if len(body) > _MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"File too large (max {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB)")
        dest.write_bytes(body)
        return {"uploaded": True, "name": name, "size": len(body)}

    @api_router.get("/api/uploads/{name:path}/download")
    async def api_download_upload(name: str):
        """Download a file from the uploads directory with correct Content-Type."""
        import mimetypes

        from fastapi.responses import Response
        path = _safe_upload_path(name)
        if not path.exists() or not path.is_file():
            raise HTTPException(404, f"Upload not found: {name}")
        mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
        return Response(
            content=path.read_bytes(),
            media_type=mime,
            headers={"Content-Disposition": f'attachment; filename="{path.name}"'},
        )

    @api_router.delete("/api/uploads/{name:path}")
    async def api_delete_upload(name: str) -> dict:
        """Delete an uploaded file."""
        path = _safe_upload_path(name)
        if not path.exists() or not path.is_file():
            raise HTTPException(404, f"Upload not found: {name}")
        path.unlink()
        # Clean up empty parent dirs up to (but not including) root
        root = _uploads_dir().resolve()
        parent = path.parent
        while parent != root and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent
        return {"deleted": True, "name": name}

    @api_router.get("/static/{file_path:path}")
    async def static_file(file_path: str, v: str | None = None) -> FileResponse:
        full = (_STATIC_DIR / file_path).resolve()
        if not str(full).startswith(str(_STATIC_DIR)) or not full.is_file():
            raise HTTPException(status_code=404, detail="Not found")
        suffix = full.suffix.lower()
        # When served with a versioned URL (?v=<hash>), cache aggressively —
        # the URL changes whenever file content changes.  Without a version
        # param (direct access, bookmarks), prevent caching entirely.
        cache = "public, max-age=86400, immutable" if v else "no-store"
        return FileResponse(
            str(full),
            media_type=_MEDIA_TYPES.get(suffix),
            headers={"Cache-Control": cache},
        )

    return api_router


def create_spa_catchall_router() -> APIRouter:
    """Root-level catch-all for SPA deep linking (no /dashboard/ prefix).

    Must be included LAST on the app so it never shadows mesh/dashboard routes.
    """
    from jinja2 import Environment, FileSystemLoader

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=True,
    )
    catchall = APIRouter(dependencies=[Depends(_verify_dashboard_auth)])

    @catchall.get("/{path:path}", response_class=HTMLResponse)
    async def spa_catchall(path: str) -> HTMLResponse:
        if path.startswith(("mesh/", "dashboard/", "ws/")):
            raise HTTPException(status_code=404, detail="Not found")
        from src.shared.models import KEYLESS_PROVIDERS, get_all_providers
        all_providers = get_all_providers()
        template = env.get_template("index.html")
        html = template.render(
            ws_path="/ws/events", api_base="/dashboard/api", v=ASSET_VERSION,
            providers=[p for p in all_providers if p["name"] not in KEYLESS_PROVIDERS],
            all_providers=all_providers,
        )
        return HTMLResponse(html, headers={
            "Cache-Control": "no-store",
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.tailwindcss.com https://cdn.jsdelivr.net; "
                "style-src 'self' 'unsafe-inline'; "
                "connect-src 'self'; "
                "frame-src 'self'; "
                "object-src 'none'"
            ),
        })

    return catchall
