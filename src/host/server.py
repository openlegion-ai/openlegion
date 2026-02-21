"""Mesh HTTP server -- the central API for fleet coordination.

Provides endpoints for:
  - Blackboard CRUD (shared state)
  - Pub/Sub (event signals)
  - API proxy (agents call external services through mesh)
  - Agent registration
  - System messaging (orchestrator-to-agent)
"""

from __future__ import annotations

import hmac
import time
from collections import defaultdict
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from src.shared.types import AgentMessage, APIProxyRequest, APIProxyResponse, MeshEvent
from src.shared.utils import setup_logging

_server_logger = setup_logging("host.server")

if TYPE_CHECKING:
    from src.host.credentials import CredentialVault
    from src.host.cron import CronScheduler
    from src.host.mesh import Blackboard, MessageRouter, PubSub
    from src.host.orchestrator import Orchestrator
    from src.host.permissions import PermissionMatrix
    from src.host.runtime import RuntimeBackend
    from src.host.traces import TraceStore
    from src.host.transport import Transport


def create_mesh_app(
    blackboard: Blackboard,
    pubsub: PubSub,
    router: MessageRouter,
    permissions: PermissionMatrix,
    credential_vault: CredentialVault | None = None,
    cron_scheduler: CronScheduler | None = None,
    container_manager: RuntimeBackend | None = None,
    transport: Transport | None = None,
    orchestrator: Orchestrator | None = None,
    auth_tokens: dict[str, str] | None = None,
    trace_store: TraceStore | None = None,
) -> FastAPI:
    """Create the FastAPI application for the mesh host process."""
    app = FastAPI(title="OpenLegion Mesh")

    _auth_tokens = auth_tokens if auth_tokens is not None else {}

    # Vault resolve rate limiting: max 5 resolves per agent per 60 seconds
    _VAULT_RESOLVE_LIMIT = 5
    _VAULT_RESOLVE_WINDOW = 60
    _vault_resolve_ts: dict[str, list[float]] = defaultdict(list)

    def _verify_auth(agent_id: str, request: Request) -> None:
        """Verify agent identity via auth token. No-op when auth is not configured."""
        if not _auth_tokens or not agent_id or agent_id in ("mesh", "orchestrator"):
            return
        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(401, "Missing authentication token")
        token = auth_header[7:]
        expected = _auth_tokens.get(agent_id)
        if not expected or not hmac.compare_digest(token, expected):
            raise HTTPException(401, "Invalid authentication token")

    # === System Messaging (orchestrator/mesh → agent) ===

    @app.post("/mesh/message")
    async def send_message(msg: AgentMessage, request: Request) -> dict:
        """Route a system message to an agent (task results, orchestrator commands).

        Special case: messages addressed to "orchestrator" with type "task_result"
        are intercepted and resolved against the orchestrator's pending futures
        instead of being routed to an agent container.
        """
        _verify_auth(msg.from_agent, request)
        if msg.to == "orchestrator" and msg.type == "task_result" and orchestrator is not None:
            from src.shared.types import TaskResult
            try:
                result = TaskResult(**msg.payload)
                resolved = orchestrator.resolve_task_result(result.task_id, result)
                return {"delivered": resolved, "target": "orchestrator"}
            except Exception as e:
                return {"error": f"Failed to resolve task result: {e}"}
        return await router.route(msg)

    # === Blackboard ===
    # NOTE: list route must be defined BEFORE the {key:path} route to avoid shadowing

    @app.get("/mesh/blackboard/")
    async def list_blackboard(prefix: str, agent_id: str, request: Request) -> list[dict]:
        """List blackboard entries by prefix."""
        _verify_auth(agent_id, request)
        if not permissions.can_read_blackboard(agent_id, prefix):
            raise HTTPException(403, f"Agent {agent_id} cannot read {prefix}")
        entries = blackboard.list_by_prefix(prefix)
        return [e.model_dump(mode="json") for e in entries]

    @app.get("/mesh/blackboard/{key:path}")
    async def read_blackboard(key: str, agent_id: str, request: Request) -> dict:
        """Read a blackboard entry. Agent must have read permission."""
        _verify_auth(agent_id, request)
        if not permissions.can_read_blackboard(agent_id, key):
            raise HTTPException(403, f"Agent {agent_id} cannot read {key}")
        entry = blackboard.read(key)
        if not entry:
            raise HTTPException(404, f"Key not found: {key}")
        return entry.model_dump(mode="json")

    @app.put("/mesh/blackboard/{key:path}")
    async def write_blackboard(key: str, agent_id: str, value: dict, request: Request) -> dict:
        """Write to blackboard. Agent must have write permission."""
        _verify_auth(agent_id, request)
        if not permissions.can_write_blackboard(agent_id, key):
            raise HTTPException(403, f"Agent {agent_id} cannot write {key}")
        entry = blackboard.write(key, value, written_by=agent_id)
        return entry.model_dump(mode="json")

    # === Pub/Sub ===

    @app.post("/mesh/publish")
    async def publish_event(event: MeshEvent, request: Request) -> dict:
        """Publish an event to a topic."""
        _verify_auth(event.source, request)
        if not permissions.can_publish(event.source, event.topic):
            raise HTTPException(403, f"Agent {event.source} cannot publish to {event.topic}")
        subscribers = pubsub.get_subscribers(event.topic)
        for agent_id in subscribers:
            await router.route(
                AgentMessage(
                    from_agent="mesh",
                    to=agent_id,
                    type="event",
                    payload=event.model_dump(mode="json"),
                )
            )
        return {"subscribers_notified": len(subscribers)}

    @app.post("/mesh/subscribe")
    async def subscribe(topic: str, agent_id: str, request: Request) -> dict:
        """Subscribe an agent to an event topic."""
        _verify_auth(agent_id, request)
        if not permissions.can_subscribe(agent_id, topic):
            raise HTTPException(403, f"Agent {agent_id} cannot subscribe to {topic}")
        pubsub.subscribe(topic, agent_id)
        return {"subscribed": True}

    # === API Proxy ===

    @app.post("/mesh/api", response_model=APIProxyResponse)
    async def proxy_api_call(request: Request, api_request: APIProxyRequest, agent_id: str) -> APIProxyResponse:
        """Proxy external API calls. Agent never sees credentials."""
        _verify_auth(agent_id, request)
        if not permissions.can_use_api(agent_id, api_request.service):
            raise HTTPException(403, f"Agent {agent_id} cannot access {api_request.service}")
        if credential_vault is None:
            return APIProxyResponse(success=False, error="No credential vault configured")

        req_trace_id = request.headers.get("x-trace-id")
        t0 = time.time()
        result = await credential_vault.execute_api_call(api_request, agent_id=agent_id)
        if req_trace_id and trace_store:
            trace_store.record(
                trace_id=req_trace_id,
                source="mesh.api_proxy",
                agent=agent_id,
                event_type="llm_call",
                detail=f"{api_request.service}/{api_request.action}",
                duration_ms=int((time.time() - t0) * 1000),
            )
        return result

    @app.post("/mesh/api/stream")
    async def proxy_api_stream(request: Request, api_request: APIProxyRequest, agent_id: str) -> StreamingResponse:
        """Streaming API proxy. Returns SSE stream for LLM completions."""
        _verify_auth(agent_id, request)
        if not permissions.can_use_api(agent_id, api_request.service):
            raise HTTPException(403, f"Agent {agent_id} cannot access {api_request.service}")
        if credential_vault is None:
            raise HTTPException(503, "No credential vault configured")

        req_trace_id = request.headers.get("x-trace-id")
        if req_trace_id and trace_store:
            trace_store.record(
                trace_id=req_trace_id,
                source="mesh.api_proxy",
                agent=agent_id,
                event_type="llm_stream",
                detail=f"{api_request.service}/{api_request.action}",
            )

        return StreamingResponse(
            credential_vault.stream_llm(api_request, agent_id=agent_id),
            media_type="text/event-stream",
        )

    # === Model Health Diagnostic ===

    @app.get("/mesh/model-health")
    async def model_health() -> list[dict]:
        """Return model failover health status. Mesh-internal diagnostic."""
        if credential_vault is None:
            return []
        return credential_vault.get_model_health()

    # === Vault (credential management) ===

    @app.post("/mesh/vault/store")
    async def vault_store(data: dict, request: Request) -> dict:
        """Store a credential and return an opaque $CRED{name} handle."""
        agent_id = data.get("agent_id", "")
        _verify_auth(agent_id, request)
        if not permissions.can_manage_vault(agent_id):
            raise HTTPException(403, f"Agent {agent_id} cannot manage vault")
        if credential_vault is None:
            raise HTTPException(503, "No credential vault configured")
        name = data.get("name", "")
        value = data.get("value", "")
        if not name or not value:
            raise HTTPException(400, "name and value are required")
        handle = credential_vault.add_credential(name, value)
        return {"stored": True, "handle": handle}

    @app.get("/mesh/vault/list")
    async def vault_list(agent_id: str, request: Request) -> dict:
        """List credential names (never values)."""
        _verify_auth(agent_id, request)
        if not permissions.can_manage_vault(agent_id):
            raise HTTPException(403, f"Agent {agent_id} cannot manage vault")
        if credential_vault is None:
            raise HTTPException(503, "No credential vault configured")
        names = credential_vault.list_credential_names()
        return {"credentials": names, "count": len(names)}

    @app.get("/mesh/vault/status/{name}")
    async def vault_status(name: str, agent_id: str, request: Request) -> dict:
        """Check if a credential exists by name."""
        _verify_auth(agent_id, request)
        if not permissions.can_manage_vault(agent_id):
            raise HTTPException(403, f"Agent {agent_id} cannot manage vault")
        if credential_vault is None:
            raise HTTPException(503, "No credential vault configured")
        return {"name": name, "exists": credential_vault.has_credential(name)}

    @app.post("/mesh/vault/resolve")
    async def vault_resolve(data: dict, request: Request) -> dict:
        """Resolve a credential handle to its value. Internal use only (browser tool)."""
        agent_id = data.get("agent_id", "")
        _verify_auth(agent_id, request)
        if not permissions.can_manage_vault(agent_id):
            raise HTTPException(403, f"Agent {agent_id} cannot manage vault")
        if credential_vault is None:
            raise HTTPException(503, "No credential vault configured")
        name = data.get("name", "")
        if not name:
            raise HTTPException(400, "name is required")

        # Rate limit: max N resolves per agent per window
        now = time.time()
        ts_list = _vault_resolve_ts[agent_id]
        ts_list[:] = [t for t in ts_list if now - t < _VAULT_RESOLVE_WINDOW]
        if len(ts_list) >= _VAULT_RESOLVE_LIMIT:
            _server_logger.warning(
                "Vault resolve rate limit hit",
                extra={"extra_data": {"agent_id": agent_id, "name": name}},
            )
            raise HTTPException(429, "Vault resolve rate limit exceeded")
        ts_list.append(now)

        # Audit log every resolve
        _server_logger.info(
            "Vault credential resolved",
            extra={"extra_data": {"agent_id": agent_id, "credential": name}},
        )

        value = credential_vault.resolve_credential(name)
        if value is None:
            raise HTTPException(404, f"Credential not found: {name}")
        return {"name": name, "value": value}

    # === Agent Registry ===

    @app.post("/mesh/register")
    async def register_agent(data: dict, request: Request) -> dict:
        """Agent registers itself with the mesh on startup."""
        agent_id = data["agent_id"]
        _verify_auth(agent_id, request)
        capabilities = data.get("capabilities", [])
        port = data.get("port", 8400)

        existing = router.agent_registry.get(agent_id)
        if existing:
            url = existing.get("url", existing) if isinstance(existing, dict) else existing
        else:
            url = f"http://localhost:{port}"

        router.register_agent(agent_id, url, capabilities)
        agent_perms = permissions.get_permissions(agent_id)
        for topic in agent_perms.can_subscribe:
            pubsub.subscribe(topic, agent_id)
        return {"registered": True}

    @app.get("/mesh/agents")
    async def list_agents() -> dict:
        """List all registered agents and their URLs."""
        return dict(router.agent_registry)

    # === Cron CRUD ===

    @app.post("/mesh/cron")
    async def create_cron_job(data: dict, request: Request) -> dict:
        """Create a cron job. Body: {agent_id, schedule, message, heartbeat?}."""
        if cron_scheduler is None:
            raise HTTPException(503, "Cron scheduler not available")
        agent_id = data.get("agent_id")
        if agent_id:
            _verify_auth(agent_id, request)
        schedule = data.get("schedule")
        message = data.get("message", "")
        heartbeat = data.get("heartbeat", False)
        if not agent_id or not schedule:
            raise HTTPException(400, "agent_id and schedule are required")
        try:
            job = cron_scheduler.add_job(
                agent=agent_id, schedule=schedule, message=message, heartbeat=heartbeat,
            )
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"id": job.id, "agent": job.agent, "schedule": job.schedule, "heartbeat": job.heartbeat}

    @app.get("/mesh/cron")
    async def list_cron_jobs(agent_id: str | None = None) -> list[dict]:
        """List cron jobs, optionally filtered by agent_id."""
        if cron_scheduler is None:
            raise HTTPException(503, "Cron scheduler not available")
        jobs = cron_scheduler.list_jobs()
        if agent_id:
            jobs = [j for j in jobs if j["agent"] == agent_id]
        return jobs

    @app.delete("/mesh/cron/{job_id}")
    async def delete_cron_job(job_id: str) -> dict:
        """Remove a cron job by ID."""
        if cron_scheduler is None:
            raise HTTPException(503, "Cron scheduler not available")
        if cron_scheduler.remove_job(job_id):
            return {"removed": True, "id": job_id}
        raise HTTPException(404, f"Job not found: {job_id}")

    # === Dynamic Agent Spawning ===

    @app.post("/mesh/spawn")
    async def spawn_agent(data: dict) -> dict:
        """Spawn an ephemeral agent. Body: {role, system_prompt?, model?, ttl?}."""
        if container_manager is None:
            raise HTTPException(503, "Container manager not available")
        role = data.get("role", "assistant")
        spawned_by = data.get("spawned_by", "unknown")
        model = data.get("model", "")
        ttl = data.get("ttl", 3600)
        system_prompt = data.get("system_prompt", f"You are a '{role}' agent.")
        from src.shared.utils import generate_id
        agent_id = generate_id("spawn")
        try:
            url = container_manager.spawn_agent(
                agent_id=agent_id, role=role, system_prompt=system_prompt,
                model=model, ttl=ttl,
            )
            router.register_agent(agent_id, url)
            ready = await container_manager.wait_for_agent(agent_id, timeout=60)
            return {
                "agent_id": agent_id, "url": url, "role": role,
                "ready": ready, "spawned_by": spawned_by, "ttl": ttl,
            }
        except Exception as e:
            raise HTTPException(500, f"Failed to spawn agent: {e}") from e

    # === Agent History Access ===

    @app.get("/mesh/agents/{agent_id}/history")
    async def get_agent_history(agent_id: str, requesting_agent: str = "") -> dict:
        """Retrieve an agent's daily logs. Permission-checked."""
        if requesting_agent and not permissions.can_message(requesting_agent, agent_id):
            raise HTTPException(403, f"Agent {requesting_agent} cannot read history of {agent_id}")
        agent_entry = router.agent_registry.get(agent_id)
        if not agent_entry:
            raise HTTPException(404, f"Agent not found: {agent_id}")
        if transport is not None:
            try:
                return await transport.request(agent_id, "GET", "/history", timeout=10)
            except Exception as e:
                raise HTTPException(502, f"Failed to fetch history from {agent_id}: {e}") from e
        # Fallback: direct HTTP if no transport provided
        agent_url = agent_entry.get("url", agent_entry) if isinstance(agent_entry, dict) else agent_entry
        import httpx
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{agent_url}/history")
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            raise HTTPException(502, f"Failed to fetch history from {agent_id}: {e}") from e

    # === Request Traces ===

    @app.get("/mesh/traces")
    async def list_traces(limit: int = 50) -> list[dict]:
        """Return recent trace events."""
        if trace_store is None:
            return []
        return trace_store.list_recent(limit=limit)

    @app.get("/mesh/traces/{trace_id}")
    async def get_trace(trace_id: str) -> list[dict]:
        """Return all events for a specific trace."""
        if trace_store is None:
            return []
        return trace_store.get_trace(trace_id)

    return app
