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
    # V2: additional dependencies (all optional for backward compat)
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
    from src.cli.config import BROWSER_BACKENDS, _PROVIDER_MODELS
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

    # ── Fleet overview ───────────────────────────────────────

    @api_router.get("/api/agents")
    async def api_agents() -> dict:
        from src.cli.config import _load_config
        cfg = _load_config()
        agents_cfg = cfg.get("agents", {})
        default_model = cfg.get("llm", {}).get("default_model", "openai/gpt-4o-mini")

        health_list = health_monitor.get_status() if health_monitor else []
        health_map = {h["agent"]: h for h in health_list}
        cost_list = cost_tracker.get_all_agents_spend("today")
        cost_map = {c["agent"]: c for c in cost_list}

        agents = []
        for agent_id, url in agent_registry.items():
            h = health_map.get(agent_id, {})
            c = cost_map.get(agent_id, {})
            acfg = agents_cfg.get(agent_id, {})
            agents.append({
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
            })
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
            from src.cli.config import _load_config as _lc
            model = _lc().get("llm", {}).get("default_model", "openai/gpt-4o-mini")
        if not role:
            role = "assistant"

        if runtime is None:
            raise HTTPException(status_code=503, detail="Runtime not available")

        try:
            from src.cli.config import _create_agent, _load_config as _lc2
            _create_agent(name, role, model, browser_backend=browser_backend)
            if permissions is not None:
                permissions.reload()

            acfg = _lc2().get("agents", {}).get(name, {})
            import os
            skills_dir = os.path.abspath(acfg.get("skills_dir", ""))
            url = runtime.start_agent(
                agent_id=name,
                role=role,
                skills_dir=skills_dir,
                system_prompt=acfg.get("system_prompt", ""),
                model=acfg.get("model", model),
                browser_backend=acfg.get("browser_backend", ""),
            )
            agent_registry[name] = url
            if router is not None:
                router.register_agent(name, url)
            if transport is not None:
                from src.host.transport import HttpTransport
                if isinstance(transport, HttpTransport):
                    transport.register(name, url)
            if health_monitor is not None:
                health_monitor.register(name)
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

        # Unregister from router and transport
        if router is not None:
            router.unregister_agent(agent_id)
        agent_registry.pop(agent_id, None)
        if transport is not None:
            from src.host.transport import HttpTransport
            if isinstance(transport, HttpTransport):
                transport._urls.pop(agent_id, None)

        # Remove from config and permissions
        import yaml
        from src.cli.config import AGENTS_FILE
        from src.cli.config import _load_permissions, _save_permissions

        if AGENTS_FILE.exists():
            with open(AGENTS_FILE) as f:
                agents_data = yaml.safe_load(f) or {}
            agents_data.get("agents", {}).pop(agent_id, None)
            with open(AGENTS_FILE, "w") as f:
                yaml.dump(agents_data, f, default_flow_style=False, sort_keys=False)

        perms = _load_permissions()
        perms.get("permissions", {}).pop(agent_id, None)
        _save_permissions(perms)

        if event_bus is not None:
            event_bus.emit("agent_state", agent=agent_id,
                data={"state": "removed"})

        return {"removed": True, "agent": agent_id}

    # ── Agent detail ─────────────────────────────────────────

    @api_router.get("/api/agents/{agent_id}")
    async def api_agent_detail(agent_id: str) -> dict:
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")

        url = agent_registry[agent_id]
        health_list = health_monitor.get_status() if health_monitor else []
        health = next((h for h in health_list if h["agent"] == agent_id), {})
        spend_today = cost_tracker.get_spend(agent_id, "today")
        spend_week = cost_tracker.get_spend(agent_id, "week")
        budget = cost_tracker.check_budget(agent_id)

        return {
            "id": agent_id,
            "url": url,
            "health": health or {"status": "unknown"},
            "spend_today": spend_today,
            "spend_week": spend_week,
            "budget": budget,
        }

    # ── Agent config CRUD ────────────────────────────────────

    @api_router.get("/api/agents/{agent_id}/config")
    async def api_agent_config(agent_id: str) -> dict:
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        from src.cli.config import _load_config
        cfg = _load_config()
        agent_cfg = cfg.get("agents", {}).get(agent_id, {})
        default_model = cfg.get("llm", {}).get("default_model", "openai/gpt-4o-mini")
        return {
            "id": agent_id,
            "model": agent_cfg.get("model", default_model),
            "role": agent_cfg.get("role", ""),
            "system_prompt": agent_cfg.get("system_prompt", ""),
            "budget": agent_cfg.get("budget", {}),
            "browser_backend": agent_cfg.get("browser_backend", "basic") or "basic",
        }

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

        if "system_prompt" in body:
            sanitized = sanitize_for_prompt(body["system_prompt"])
            _update_agent_field(agent_id, "system_prompt", sanitized)
            updated.append("system_prompt")

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
            url = runtime.start_agent(
                agent_id=agent_id,
                role=agent_cfg.get("role", "assistant"),
                skills_dir=agent_cfg.get("skills_dir", ""),
                system_prompt=agent_cfg.get("system_prompt", ""),
                model=agent_cfg.get("model", default_model),
                browser_backend=agent_cfg.get("browser_backend", ""),
            )
            agent_registry[agent_id] = url
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

    @api_router.post("/api/broadcast")
    async def api_broadcast(request: Request) -> dict:
        if transport is None:
            raise HTTPException(status_code=503, detail="Transport not available")
        body = await request.json()
        message = body.get("message", "").strip()
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        from src.shared.utils import sanitize_for_prompt
        message = sanitize_for_prompt(message)
        import asyncio
        results = {}
        async def _send(aid: str) -> tuple[str, str]:
            try:
                data = await transport.request(
                    aid, "POST", "/chat", json={"message": message}, timeout=120,
                )
                return aid, data.get("response", "(no response)")
            except Exception as e:
                return aid, f"Error: {e}"
        tasks = [_send(aid) for aid in agent_registry]
        for coro in asyncio.as_completed(tasks):
            aid, resp = await coro
            results[aid] = resp
        return {"responses": results}

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
        known_providers = {"anthropic", "openai", "gemini", "deepseek", "moonshot", "xai", "groq"}
        if service.lower() in known_providers and not service.lower().endswith("_api_key"):
            service = f"{service}_api_key"
        credential_vault.add_credential(service, key)
        return {"stored": True, "service": service}

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
        limit = max(1, min(limit, 500))
        return {"traces": trace_store.list_recent(limit)}

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
        job = cron_scheduler.update_job(job_id, **body)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return {"status": "updated", "job_id": job_id}

    @api_router.post("/api/cron/{job_id}/pause")
    async def api_cron_pause(job_id: str) -> dict:
        if cron_scheduler is None:
            raise HTTPException(status_code=503, detail="Cron scheduler not available")
        if not cron_scheduler.pause_job(job_id):
            raise HTTPException(status_code=404, detail="Job not found")
        return {"paused": True, "job_id": job_id}

    @api_router.post("/api/cron/{job_id}/resume")
    async def api_cron_resume(job_id: str) -> dict:
        if cron_scheduler is None:
            raise HTTPException(status_code=503, detail="Cron scheduler not available")
        if not cron_scheduler.resume_job(job_id):
            raise HTTPException(status_code=404, detail="Job not found")
        return {"resumed": True, "job_id": job_id}

    # ── Settings / environment ───────────────────────────────

    @api_router.get("/api/settings")
    async def api_settings() -> dict:
        from src.host.costs import MODEL_COSTS

        cred_names = credential_vault.list_credential_names() if credential_vault else []
        pubsub_subs = pubsub.subscriptions if pubsub else {}
        return {
            "credentials": {"names": cred_names, "count": len(cred_names)},
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
