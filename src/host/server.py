"""Mesh HTTP server -- the central API for fleet coordination.

Provides endpoints for:
  - Blackboard CRUD (shared state)
  - Pub/Sub (event signals)
  - API proxy (agents call external services through mesh)
  - Agent registration
  - System messaging (orchestrator-to-agent)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from src.shared.types import AgentMessage, APIProxyRequest, APIProxyResponse, MeshEvent

if TYPE_CHECKING:
    from src.host.credentials import CredentialVault
    from src.host.cron import CronScheduler
    from src.host.mesh import Blackboard, MessageRouter, PubSub
    from src.host.orchestrator import Orchestrator
    from src.host.permissions import PermissionMatrix
    from src.host.runtime import RuntimeBackend
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
) -> FastAPI:
    """Create the FastAPI application for the mesh host process."""
    app = FastAPI(title="OpenLegion Mesh")

    # === System Messaging (orchestrator/mesh → agent) ===

    @app.post("/mesh/message")
    async def send_message(msg: AgentMessage) -> dict:
        """Route a system message to an agent (task results, orchestrator commands).

        Special case: messages addressed to "orchestrator" with type "task_result"
        are intercepted and resolved against the orchestrator's pending futures
        instead of being routed to an agent container.
        """
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
    async def list_blackboard(prefix: str, agent_id: str) -> list[dict]:
        """List blackboard entries by prefix."""
        if not permissions.can_read_blackboard(agent_id, prefix):
            raise HTTPException(403, f"Agent {agent_id} cannot read {prefix}")
        entries = blackboard.list_by_prefix(prefix)
        return [e.model_dump(mode="json") for e in entries]

    @app.get("/mesh/blackboard/{key:path}")
    async def read_blackboard(key: str, agent_id: str) -> dict:
        """Read a blackboard entry. Agent must have read permission."""
        if not permissions.can_read_blackboard(agent_id, key):
            raise HTTPException(403, f"Agent {agent_id} cannot read {key}")
        entry = blackboard.read(key)
        if not entry:
            raise HTTPException(404, f"Key not found: {key}")
        return entry.model_dump(mode="json")

    @app.put("/mesh/blackboard/{key:path}")
    async def write_blackboard(key: str, agent_id: str, value: dict) -> dict:
        """Write to blackboard. Agent must have write permission."""
        if not permissions.can_write_blackboard(agent_id, key):
            raise HTTPException(403, f"Agent {agent_id} cannot write {key}")
        entry = blackboard.write(key, value, written_by=agent_id)
        return entry.model_dump(mode="json")

    # === Pub/Sub ===

    @app.post("/mesh/publish")
    async def publish_event(event: MeshEvent) -> dict:
        """Publish an event to a topic."""
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
    async def subscribe(topic: str, agent_id: str) -> dict:
        """Subscribe an agent to an event topic."""
        if not permissions.can_subscribe(agent_id, topic):
            raise HTTPException(403, f"Agent {agent_id} cannot subscribe to {topic}")
        pubsub.subscribe(topic, agent_id)
        return {"subscribed": True}

    # === API Proxy ===

    @app.post("/mesh/api", response_model=APIProxyResponse)
    async def proxy_api_call(request: APIProxyRequest, agent_id: str) -> APIProxyResponse:
        """Proxy external API calls. Agent never sees credentials."""
        if not permissions.can_use_api(agent_id, request.service):
            raise HTTPException(403, f"Agent {agent_id} cannot access {request.service}")
        if credential_vault is None:
            return APIProxyResponse(success=False, error="No credential vault configured")
        return await credential_vault.execute_api_call(request, agent_id=agent_id)

    @app.post("/mesh/api/stream")
    async def proxy_api_stream(request: APIProxyRequest, agent_id: str) -> StreamingResponse:
        """Streaming API proxy. Returns SSE stream for LLM completions."""
        if not permissions.can_use_api(agent_id, request.service):
            raise HTTPException(403, f"Agent {agent_id} cannot access {request.service}")
        if credential_vault is None:
            raise HTTPException(503, "No credential vault configured")
        return StreamingResponse(
            credential_vault.stream_llm(request, agent_id=agent_id),
            media_type="text/event-stream",
        )

    # === Model Health Diagnostic ===

    @app.get("/mesh/model-health")
    async def model_health() -> list[dict]:
        """Return model failover health status. Mesh-internal diagnostic."""
        if credential_vault is None:
            return []
        return credential_vault.get_model_health()

    # === Agent Registry ===

    @app.post("/mesh/register")
    async def register_agent(data: dict) -> dict:
        """Agent registers itself with the mesh on startup."""
        agent_id = data["agent_id"]
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
    async def create_cron_job(data: dict) -> dict:
        """Create a cron job. Body: {agent_id, schedule, message, heartbeat?}."""
        if cron_scheduler is None:
            raise HTTPException(503, "Cron scheduler not available")
        agent_id = data.get("agent_id")
        schedule = data.get("schedule")
        message = data.get("message", "")
        heartbeat = data.get("heartbeat", False)
        if not agent_id or not schedule:
            raise HTTPException(400, "agent_id and schedule are required")
        job = cron_scheduler.add_job(
            agent=agent_id, schedule=schedule, message=message, heartbeat=heartbeat,
        )
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

    return app
