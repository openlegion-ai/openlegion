"""HTTP client for agent-to-mesh communication.

This is the agent's ONLY interface to the outside world.
All external interaction is mediated by the mesh host process.
"""

from __future__ import annotations

import os
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
        self._auth_token: str = os.environ.get("MESH_AUTH_TOKEN", "")

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers: dict[str, str] = {}
            if self._auth_token:
                headers["Authorization"] = f"Bearer {self._auth_token}"
            self._client = httpx.AsyncClient(timeout=30, headers=headers)
        return self._client

    def _trace_headers(self) -> dict[str, str]:
        """Return current trace headers for per-request injection."""
        from src.shared.trace import trace_headers
        return trace_headers()

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def register(
        self, capabilities: list[str], port: int = 8400, timeout: int = 5,
    ) -> None:
        """Register this agent with the mesh on startup."""
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/register",
            json={"agent_id": self.agent_id, "capabilities": capabilities, "port": port},
            timeout=timeout,
            headers=self._trace_headers(),
        )
        response.raise_for_status()

    async def read_blackboard(self, key: str) -> Optional[dict]:
        """Read a value from the shared blackboard."""
        client = await self._get_client()
        response = await client.get(
            f"{self.mesh_url}/mesh/blackboard/{key}",
            params={"agent_id": self.agent_id},
            headers=self._trace_headers(),
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
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def list_blackboard(self, prefix: str) -> list[dict]:
        """List blackboard entries by key prefix."""
        client = await self._get_client()
        response = await client.get(
            f"{self.mesh_url}/mesh/blackboard/",
            params={"agent_id": self.agent_id, "prefix": prefix},
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def send_system_message(self, to: str, msg_type: str, payload: dict) -> dict:
        """Send a system-level message to the orchestrator/mesh (not for agent-to-agent use)."""
        message = AgentMessage(from_agent=self.agent_id, to=to, type=msg_type, payload=payload)
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/message",
            json=message.model_dump(mode="json"),
            headers=self._trace_headers(),
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
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def list_agents(self) -> dict:
        """List all registered agents and their capabilities."""
        client = await self._get_client()
        response = await client.get(
            f"{self.mesh_url}/mesh/agents",
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def create_cron(
        self, schedule: str, message: str, heartbeat: bool = False,
    ) -> dict:
        """Create a cron job for this agent via the mesh."""
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/cron",
            json={
                "agent_id": self.agent_id,
                "schedule": schedule,
                "message": message,
                "heartbeat": heartbeat,
            },
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def list_cron(self) -> list[dict]:
        """List cron jobs for this agent."""
        client = await self._get_client()
        response = await client.get(
            f"{self.mesh_url}/mesh/cron",
            params={"agent_id": self.agent_id},
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def update_cron(self, job_id: str, **kwargs) -> dict:
        """Update a cron job by ID."""
        client = await self._get_client()
        response = await client.put(
            f"{self.mesh_url}/mesh/cron/{job_id}",
            json=kwargs,
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def remove_cron(self, job_id: str) -> dict:
        """Remove a cron job by ID."""
        client = await self._get_client()
        response = await client.delete(
            f"{self.mesh_url}/mesh/cron/{job_id}",
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def spawn_agent(
        self,
        role: str,
        system_prompt: str = "",
        model: str = "",
        ttl: int = 3600,
    ) -> dict:
        """Request the mesh to spawn an ephemeral agent."""
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/spawn",
            json={
                "role": role,
                "spawned_by": self.agent_id,
                "system_prompt": system_prompt,
                "model": model,
                "ttl": ttl,
            },
            timeout=90,
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def get_agent_history(self, agent_id: str) -> dict:
        """Read another agent's daily logs (permission-checked on server)."""
        client = await self._get_client()
        response = await client.get(
            f"{self.mesh_url}/mesh/agents/{agent_id}/history",
            params={"requesting_agent": self.agent_id},
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    # === Vault (credential management) ===

    async def vault_store(self, name: str, value: str) -> dict:
        """Store a credential in the mesh vault. Returns handle."""
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/vault/store",
            json={"agent_id": self.agent_id, "name": name, "value": value},
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def vault_list(self) -> list[str]:
        """List credential names stored in the vault."""
        client = await self._get_client()
        response = await client.get(
            f"{self.mesh_url}/mesh/vault/list",
            params={"agent_id": self.agent_id},
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json().get("credentials", [])

    async def vault_status(self, name: str) -> dict:
        """Check whether a credential exists."""
        client = await self._get_client()
        response = await client.get(
            f"{self.mesh_url}/mesh/vault/status/{name}",
            params={"agent_id": self.agent_id},
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def vault_resolve(self, name: str) -> str | None:
        """Resolve a credential name to its value.

        WARNING: The return value is a secret. The caller must NEVER
        return it to the LLM or include it in any tool output dict.
        """
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/vault/resolve",
            json={"agent_id": self.agent_id, "name": name},
            headers=self._trace_headers(),
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json().get("value")

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
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return APIProxyResponse(**response.json())
