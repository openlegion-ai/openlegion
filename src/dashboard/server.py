"""Dashboard API router: fleet overview, costs, blackboard, traces, management.

Serves the SPA template and static files, plus JSON API endpoints
consumed by the Alpine.js frontend.  All data comes from live Python
objects — no HTTP round-trips through mesh endpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from jinja2 import Environment, FileSystemLoader

from src.shared.utils import sanitize_for_prompt, setup_logging

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
    orchestrator: Any = None,
    pubsub: Any = None,
    permissions: Any = None,
    credential_vault: Any = None,
    transport: Any = None,
    runtime: Any = None,
    router: Any = None,
) -> APIRouter:
    """Create the dashboard FastAPI router."""
    api_router = APIRouter(prefix="/dashboard")

    jinja_env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=True,
    )

    # Build flat valid-models list and browser backend names for validation
    from src.cli.config import _PROVIDER_MODELS, BROWSER_BACKENDS
    _valid_models = [m for models in _PROVIDER_MODELS.values() for m in models]
    _valid_browsers = [b["name"] for b in BROWSER_BACKENDS]

    # ── SPA entry point ──────────────────────────────────────

    @api_router.get("/", response_class=HTMLResponse)
    async def dashboard_index() -> HTMLResponse:
        template = jinja_env.get_template("index.html")
        html = template.render(
            ws_path="/ws/events",
            api_base="/dashboard/api",
        )
        return HTMLResponse(html)

    def _vnc_url_for_request(request: Request, agent_info: dict) -> str | None:
        """Build VNC URL using the request host so the iframe connects correctly.

        The runtime stores ``vnc_url`` with ``127.0.0.1`` which only works
        from the same machine.  Rewriting with the request's ``Host`` header
        lets the KasmVNC iframe work from any browser that can reach the mesh.
        """
        vnc_port = agent_info.get("vnc_port")
        if not vnc_port:
            return None
        host = request.headers.get("host", "127.0.0.1:8420").split(":")[0]
        return f"http://{host}:{vnc_port}/index.html?autoconnect=true&path=&resize=scale"

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
                "health_status": h.get("status", "unknown"),
                "failures": h.get("failures", 0),
                "restarts": h.get("restarts", 0),
                "last_check": h.get("last_check", 0),
                "last_healthy": h.get("last_healthy", 0),
                "daily_cost": c.get("cost", 0),
                "daily_tokens": c.get("tokens", 0),
                "role": acfg.get("role", ""),
                "model": acfg.get("model", default_model),
                "project": agent_projects.get(agent_id),
            }
            if runtime:
                agent_info = runtime.agents.get(agent_id, {})
                vnc_url = _vnc_url_for_request(request, agent_info)
                if vnc_url:
                    entry["vnc_url"] = vnc_url
            agents.append(entry)
        return {"agents": agents}

    @api_router.post("/api/agents")
    async def api_add_agent(request: Request) -> dict:
        """Add a new agent: create config, start container, register."""
        import re
        body = await request.json()
        name = body.get("name", "").strip()
        role = body.get("role", "").strip()
        model = body.get("model", "").strip()
        browser_backend = body.get("browser_backend", "").strip()

        if not name:
            raise HTTPException(status_code=400, detail="name is required")
        if not re.match(r"^[a-z][a-z0-9_]{0,29}$", name):
            raise HTTPException(status_code=400, detail="name must match ^[a-z][a-z0-9_]{0,29}$")
        if name in agent_registry:
            raise HTTPException(status_code=409, detail=f"Agent '{name}' already exists")

        if model and model not in _valid_models:
            raise HTTPException(status_code=400, detail=f"Invalid model: {model}")
        if browser_backend and browser_backend not in _valid_browsers:
            raise HTTPException(status_code=400, detail=f"Invalid browser: {browser_backend}")

        if not model:
            from src.cli.config import _load_config
            model = _load_config().get("llm", {}).get("default_model", "openai/gpt-4o-mini")
        if not role:
            role = "assistant"

        if runtime is None:
            raise HTTPException(status_code=503, detail="Runtime not available")

        try:
            from src.cli.config import _create_agent, _load_config
            _create_agent(name, role, model, browser_backend=browser_backend)
            if permissions is not None:
                permissions.reload()

            cfg = _load_config()
            acfg = cfg.get("agents", {}).get(name, {})
            import os
            skills_dir = os.path.abspath(acfg.get("skills_dir", ""))
            url = runtime.start_agent(
                agent_id=name,
                role=role,
                skills_dir=skills_dir,
                system_prompt="",
                model=acfg.get("model", model),
                browser_backend=acfg.get("browser_backend", ""),
                thinking=acfg.get("thinking", ""),
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
            if event_bus is not None:
                event_bus.emit("agent_state", agent=name,
                    data={"state": "added", "role": role, "ready": ready})
            return {"created": True, "agent": name, "ready": ready}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to add agent {name}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @api_router.delete("/api/agents/{agent_id}")
    async def api_remove_agent(agent_id: str) -> dict:
        """Remove an agent: stop container, unregister, remove config."""
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")

        # Stop container
        if runtime is not None:
            try:
                runtime.stop_agent(agent_id)
            except Exception:
                pass

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
        # Include VNC info for persistent browser agents
        if runtime:
            agent_info = runtime.agents.get(agent_id, {})
            vnc_url = _vnc_url_for_request(request, agent_info)
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

        cfg_result = {
            "id": agent_id,
            "model": agent_cfg.get("model", default_model),
            "role": agent_cfg.get("role", ""),
            "budget": agent_cfg.get("budget", {}),
            "browser_backend": agent_cfg.get("browser_backend", "persistent") or "persistent",
            "allowed_credentials": allowed_creds,
            "available_credentials": sorted(agent_cred_names),
            "system_credentials": system_cred_names,
            "resolved_credentials": resolved,
        }
        if runtime:
            agent_info = runtime.agents.get(agent_id, {})
            vnc_url = _vnc_url_for_request(request, agent_info)
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
            if new_model not in _valid_models:
                raise HTTPException(status_code=400, detail=f"Invalid model: {new_model}")
            old_model = agent_cfg.get("model", default_model)
            if new_model != old_model:
                _update_agent_field(agent_id, "model", new_model)
                updated.append("model")
                restart_required = True

        if "browser_backend" in body:
            new_browser = body["browser_backend"]
            if new_browser not in _valid_browsers:
                raise HTTPException(status_code=400, detail=f"Invalid browser: {new_browser}")
            old_browser = agent_cfg.get("browser_backend", "basic") or "basic"
            if new_browser != old_browser:
                _update_agent_field(agent_id, "browser_backend", new_browser)
                updated.append("browser_backend")
                restart_required = True

        if "role" in body:
            _update_agent_field(agent_id, "role", body["role"])
            updated.append("role")

        if "budget" in body:
            budget_val = body["budget"]
            if isinstance(budget_val, dict):
                daily = budget_val.get("daily_usd")
                if daily is not None:
                    try:
                        daily = float(daily)
                        if daily <= 0:
                            raise ValueError
                    except (ValueError, TypeError):
                        raise HTTPException(status_code=400, detail="Budget must be a positive number")
                    _update_agent_field(agent_id, "budget", {"daily_usd": daily})
                    cost_tracker.set_budget(agent_id, daily_usd=daily)
                    updated.append("budget")

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
            url = runtime.start_agent(
                agent_id=agent_id,
                role=agent_cfg.get("role", "assistant"),
                skills_dir=skills_dir,
                system_prompt=agent_cfg.get("system_prompt", ""),
                model=agent_cfg.get("model", default_model),
                mcp_servers=agent_cfg.get("mcp_servers") or None,
                browser_backend=agent_cfg.get("browser_backend", ""),
                thinking=agent_cfg.get("thinking", ""),
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
            return {"restarted": True, "ready": ready}
        except Exception as e:
            logger.error(f"Failed to restart agent {agent_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @api_router.put("/api/agents/{agent_id}/budget")
    async def api_update_budget(agent_id: str, request: Request) -> dict:
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        body = await request.json()
        daily_usd = body.get("daily_usd")
        try:
            daily_usd = float(daily_usd)
            if daily_usd <= 0:
                raise ValueError
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="daily_usd must be a positive number")
        cost_tracker.set_budget(agent_id, daily_usd=daily_usd)
        from src.cli.config import _update_agent_field
        _update_agent_field(agent_id, "budget", {"daily_usd": daily_usd})
        return {"updated": True, "agent": agent_id, "daily_usd": daily_usd}

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
            return await transport.request(agent_id, "GET", "/capabilities", timeout=10)
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

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
        try:
            result = await transport.request(
                agent_id, "POST", "/chat", json={"message": message}, timeout=120,
            )
            return {"response": result.get("response", "(no response)")}
        except Exception as e:
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

        import json as _json

        async def event_generator():
            try:
                async for event in transport.stream_request(
                    agent_id, "POST", "/chat/stream",
                    json={"message": message}, timeout=120,
                ):
                    if isinstance(event, dict):
                        yield f"data: {_json.dumps(event, default=str)}\n\n"
                        etype = event.get("type", "")
                        if event_bus and etype in ("tool_start", "tool_result"):
                            event_bus.emit(etype, agent=agent_id,
                                data={k: v for k, v in event.items() if k != "type"})
            except Exception as e:
                yield f"data: {_json.dumps({'type': 'error', 'message': str(e)})}\n\n"

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
        import json as _json

        agents = list(agent_registry.keys())
        project = body.get("project") or ""
        if not isinstance(project, str):
            raise HTTPException(status_code=400, detail="project must be a string")
        if project:
            from src.cli.config import _load_projects
            members = set(_load_projects().get(project, {}).get("members", []))
            agents = [a for a in agents if a in members]
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
                await queue.put({"type": "error", "agent": aid, "message": str(e)})
            await queue.put({"type": "agent_done", "agent": aid})

        async def event_generator():
            tasks = [asyncio.create_task(_stream_agent(aid)) for aid in agents]
            done_count = 0
            try:
                while done_count < len(agents):
                    event = await queue.get()
                    if event.get("type") == "agent_done":
                        done_count += 1
                    yield f"data: {_json.dumps(event, default=str)}\n\n"
            except asyncio.CancelledError:
                for t in tasks:
                    t.cancel()
                raise
            yield f"data: {_json.dumps({'type': 'all_done'})}\n\n"

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
            return {"reset": True, "agent": agent_id}
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    @api_router.post("/api/credentials")
    async def api_add_credential(request: Request) -> dict:
        if credential_vault is None:
            raise HTTPException(status_code=503, detail="Credential vault not available")
        body = await request.json()
        service = body.get("service", "").strip()
        key = body.get("key", "").strip()
        if not service or not key:
            raise HTTPException(status_code=400, detail="service and key are required")
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
        credential_vault.add_credential(service, key, system=is_system)
        # Store optional custom API base URL alongside the key
        base_url = body.get("base_url", "").strip()
        if base_url:
            provider = service.replace("_api_key", "")
            credential_vault.add_credential(f"{provider}_api_base", base_url, system=is_system)
        tier = "system" if is_system else "agent"
        return {"stored": True, "service": service, "tier": tier}

    @api_router.delete("/api/credentials/{name}")
    async def api_remove_credential(name: str) -> dict:
        if credential_vault is None:
            raise HTTPException(status_code=503, detail="Credential vault not available")
        existed = credential_vault.remove_credential(name)
        if not existed:
            raise HTTPException(status_code=404, detail=f"Credential '{name}' not found")
        return {"removed": True, "service": name}

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
        return {"period": period, "agents": agents_spend, "budgets": budgets}

    # ── Projects ──────────────────────────────────────────────

    @api_router.get("/api/projects")
    async def api_projects_list() -> dict:
        """List all projects with members."""
        from src.cli.config import _load_projects
        projects = _load_projects()
        result = []
        for pname, pdata in projects.items():
            result.append({
                "name": pname,
                "description": pdata.get("description", ""),
                "members": pdata.get("members", []),
                "created_at": pdata.get("created_at", ""),
            })
        return {"projects": result}

    @api_router.post("/api/projects")
    async def api_projects_create(request: Request) -> dict:
        """Create a new project."""
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

    @api_router.put("/api/blackboard/{key:path}")
    async def api_blackboard_write(key: str, request: Request) -> dict:
        body = await request.json()
        value = body.get("value", {})
        if not isinstance(value, dict):
            raise HTTPException(status_code=400, detail="value must be a JSON object")
        written_by = body.get("written_by", "dashboard")
        entry = blackboard.write(key, value, written_by=written_by)
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

    # ── Cron management ──────────────────────────────────────

    @api_router.get("/api/cron")
    async def api_cron() -> dict:
        if cron_scheduler is None:
            return {"jobs": []}
        return {"jobs": cron_scheduler.list_jobs()}

    @api_router.post("/api/cron/{job_id}/run")
    async def api_cron_run(job_id: str) -> dict:
        if cron_scheduler is None:
            raise HTTPException(status_code=503, detail="Cron scheduler not available")
        result = await cron_scheduler.run_job(job_id)
        if result is None and job_id not in cron_scheduler.jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        return {"executed": True, "job_id": job_id, "result": result}

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
        if not cron_scheduler.remove_job(job_id):
            raise HTTPException(status_code=404, detail="Job not found")
        return {"deleted": True, "job_id": job_id}

    # ── Settings / environment ───────────────────────────────

    @api_router.get("/api/settings")
    async def api_settings() -> dict:
        from src.host.costs import MODEL_COSTS
        from src.host.credentials import SYSTEM_CREDENTIAL_PROVIDERS

        cred_names = credential_vault.list_credential_names() if credential_vault else []
        agent_cred_names = credential_vault.list_agent_credential_names() if credential_vault else []
        _llm_key_names = {f"{p}_api_key" for p in SYSTEM_CREDENTIAL_PROVIDERS}
        has_llm = bool(set(cred_names) & _llm_key_names)
        pubsub_subs = pubsub.subscriptions if pubsub else {}
        return {
            "credentials": {"names": cred_names, "count": len(cred_names)},
            "agent_credentials": agent_cred_names,
            "has_llm_credentials": has_llm,
            "pubsub_subscriptions": pubsub_subs,
            "model_costs": {k: {"input_per_1k": v[0], "output_per_1k": v[1]} for k, v in MODEL_COSTS.items()},
            "provider_models": _PROVIDER_MODELS,
            "browser_backends": BROWSER_BACKENDS,
        }

    # ── Messages log ─────────────────────────────────────────

    @api_router.get("/api/messages")
    async def api_messages() -> dict:
        if router is None:
            return {"messages": []}
        return {"messages": router.message_log[-100:]}

    # ── Workflows ────────────────────────────────────────────

    @api_router.get("/api/workflows")
    async def api_workflows() -> dict:
        if orchestrator is None:
            return {"workflows": [], "active": []}
        wf_list = [
            {"name": wf.name, "steps": len(wf.steps), "trigger": wf.trigger, "timeout": wf.timeout}
            for wf in orchestrator.workflows.values()
        ]
        active = [
            orchestrator.get_execution_status(eid)
            for eid in orchestrator.active_executions
        ]
        return {"workflows": wf_list, "active": [a for a in active if a]}

    # ── Agent Workspace (proxy to agent) ─────────────────────

    _WORKSPACE_ALLOWLIST = frozenset({"SOUL.md", "HEARTBEAT.md", "USER.md", "AGENTS.md", "MEMORY.md"})

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
            return await transport.request(
                agent_id, "PUT", f"/workspace/{filename}",
                json={"content": content}, timeout=10,
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

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

    # ── Static files ─────────────────────────────────────────

    _MEDIA_TYPES = {
        ".css": "text/css",
        ".js": "application/javascript",
        ".png": "image/png",
        ".svg": "image/svg+xml",
        ".ico": "image/x-icon",
    }

    @api_router.get("/static/{file_path:path}")
    async def static_file(file_path: str) -> FileResponse:
        full = (_STATIC_DIR / file_path).resolve()
        if not str(full).startswith(str(_STATIC_DIR)) or not full.is_file():
            raise HTTPException(status_code=404, detail="Not found")
        suffix = full.suffix.lower()
        return FileResponse(str(full), media_type=_MEDIA_TYPES.get(suffix))

    return api_router


def create_spa_catchall_router() -> APIRouter:
    """Root-level catch-all for SPA deep linking (no /dashboard/ prefix).

    Must be included LAST on the app so it never shadows mesh/dashboard routes.
    """
    from jinja2 import Environment, FileSystemLoader

    env = Environment(loader=FileSystemLoader(str(_TEMPLATES_DIR)))
    catchall = APIRouter()

    @catchall.get("/{path:path}", response_class=HTMLResponse)
    async def spa_catchall(path: str) -> HTMLResponse:
        if path.startswith(("mesh/", "dashboard/", "ws/")):
            raise HTTPException(status_code=404, detail="Not found")
        template = env.get_template("index.html")
        html = template.render(ws_path="/ws/events", api_base="/dashboard/api")
        return HTMLResponse(html)

    return catchall
