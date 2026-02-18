"""HTTP client for agent-to-mesh communication.

This is the agent's ONLY interface to the outside world.
All external interaction is mediated by the mesh host process.
"""

from __future__ import annotations

from typing import Optional

import httpx

from src.shared.types import AgentMessage, APIProxyRequest, APIProxyResponse, MeshEvent
from src.shared.utils import setup_logging

logger = setup_logging("agent.mesh_client")


class MeshClient:
    """HTTP client for agent-to-mesh communication.

    Uses a shared httpx.AsyncClient for connection pooling.
    """

    def __init__(self, mesh_url: str, agent_id: str):
        self.mesh_url = mesh_url
        self.agent_id = agent_id
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30)
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def register(self, capabilities: list[str], port: int = 8400) -> None:
        """Register this agent with the mesh on startup."""
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/register",
            json={"agent_id": self.agent_id, "capabilities": capabilities, "port": port},
        )
        response.raise_for_status()

    async def read_blackboard(self, key: str) -> Optional[dict]:
        """Read a value from the shared blackboard."""
        client = await self._get_client()
        response = await client.get(
            f"{self.mesh_url}/mesh/blackboard/{key}",
            params={"agent_id": self.agent_id},
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    async def write_blackboard(self, key: str, value: dict) -> dict:
        """Write a value to the shared blackboard."""
        client = await self._get_client()
        response = await client.put(
            f"{self.mesh_url}/mesh/blackboard/{key}",
            params={"agent_id": self.agent_id},
            json=value,
        )
        response.raise_for_status()
        return response.json()

    async def list_blackboard(self, prefix: str) -> list[dict]:
        """List blackboard entries by key prefix."""
        client = await self._get_client()
        response = await client.get(
            f"{self.mesh_url}/mesh/blackboard/",
            params={"agent_id": self.agent_id, "prefix": prefix},
        )
        response.raise_for_status()
        return response.json()

    async def send_message(self, to: str, msg_type: str, payload: dict) -> dict:
        """Send a message to another agent through the mesh."""
        message = AgentMessage(from_agent=self.agent_id, to=to, type=msg_type, payload=payload)
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/message",
            json=message.model_dump(mode="json"),
        )
        response.raise_for_status()
        return response.json()

    async def publish_event(self, topic: str, payload: dict | None = None) -> dict:
        """Publish an event to the mesh pub/sub."""
        event = MeshEvent(topic=topic, source=self.agent_id, payload=payload or {})
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/publish",
            json=event.model_dump(mode="json"),
        )
        response.raise_for_status()
        return response.json()

    async def api_call(
        self, service: str, action: str, params: dict | None = None, timeout: int = 30
    ) -> APIProxyResponse:
        """Request an external API call through the mesh proxy."""
        request = APIProxyRequest(service=service, action=action, params=params or {}, timeout=timeout)
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/api",
            json=request.model_dump(mode="json"),
            params={"agent_id": self.agent_id},
            timeout=timeout + 5,
        )
        response.raise_for_status()
        return APIProxyResponse(**response.json())
