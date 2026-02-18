"""Mesh HTTP server -- the central API for all agent communication.

Provides endpoints for:
  - Agent-to-agent messaging
  - Blackboard CRUD
  - Pub/Sub
  - API proxy (agents call external services through mesh)
  - Agent registration
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException

from src.shared.types import AgentMessage, APIProxyRequest, APIProxyResponse, MeshEvent

if TYPE_CHECKING:
    from src.host.credentials import CredentialVault
    from src.host.mesh import Blackboard, MessageRouter, PubSub
    from src.host.permissions import PermissionMatrix


def create_mesh_app(
    blackboard: Blackboard,
    pubsub: PubSub,
    router: MessageRouter,
    permissions: PermissionMatrix,
    credential_vault: CredentialVault | None = None,
) -> FastAPI:
    """Create the FastAPI application for the mesh host process."""
    app = FastAPI(title="OpenLegion Mesh")

    # === Agent-to-Agent Messaging ===

    @app.post("/mesh/message")
    async def send_message(msg: AgentMessage) -> dict:
        """Route a message from one agent to another."""
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

    # === Agent Registry ===

    @app.post("/mesh/register")
    async def register_agent(data: dict) -> dict:
        """Agent registers itself with the mesh on startup."""
        agent_id = data["agent_id"]
        capabilities = data.get("capabilities", [])
        port = data.get("port", 8400)
        url = f"http://{agent_id}:{port}"
        router.register_agent(agent_id, url, capabilities)
        agent_perms = permissions.get_permissions(agent_id)
        for topic in agent_perms.can_subscribe:
            pubsub.subscribe(topic, agent_id)
        return {"registered": True}

    @app.get("/mesh/agents")
    async def list_agents() -> dict:
        """List all registered agents and their URLs."""
        return dict(router.agent_registry)

    return app
