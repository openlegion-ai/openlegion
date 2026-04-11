"""HTTP client for agent-to-mesh communication.

This is the agent's ONLY interface to the outside world.
All external interaction is mediated by the mesh host process.
"""

from __future__ import annotations

import asyncio
import os

import httpx

from src.shared.types import MeshEvent
from src.shared.utils import setup_logging

logger = setup_logging("agent.mesh_client")


class MeshClient:
    """HTTP client for agent-to-mesh communication.

    Uses a shared httpx.AsyncClient for connection pooling.
    """

    def __init__(self, mesh_url: str, agent_id: str, project_name: str | None = None):
        self.mesh_url = mesh_url
        self.agent_id = agent_id
        self.project_name = project_name
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()
        self._auth_token: str = os.environ.get("MESH_AUTH_TOKEN", "")

    @property
    def is_standalone(self) -> bool:
        """True when this agent is not assigned to any project."""
        return self.project_name is None

    def _scope_key(self, key: str) -> str:
        """Prefix a blackboard key with the project namespace.

        Project agents transparently read/write under ``projects/{name}/``
        so that each project's blackboard data is isolated.  Standalone
        agents (no project) pass keys through unchanged — but they are
        blocked from the blackboard at both the tool and permission layers.
        """
        if self.project_name:
            return f"projects/{self.project_name}/{key}"
        return key

    def _scope_topic(self, topic: str) -> str:
        """Prefix a pub/sub topic with the project namespace.

        Project agents transparently publish/subscribe under
        ``projects/{name}/`` so that each project's events are isolated.
        Standalone agents (no project) use raw topic names.
        """
        if self.project_name:
            return f"projects/{self.project_name}/{topic}"
        return topic

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is not None and not self._client.is_closed:
            return self._client
        async with self._client_lock:
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

    # Retry config for idempotent reads — resilience to transient mesh errors
    _GET_MAX_RETRIES = 2
    _GET_BACKOFFS = (0.5, 1.0)  # seconds per retry
    _GET_RETRYABLE_STATUS = frozenset({502, 503})

    async def _get_with_retry(
        self, url: str, *, params: dict | None = None,
        headers: dict | None = None, timeout: int = 30,
    ) -> httpx.Response:
        """GET with automatic retries on transient errors.

        Retries on ConnectError, TimeoutException, 502, 503.
        Only used for idempotent read-only endpoints.
        """
        hdrs = {**(headers or {}), **self._trace_headers()}
        last_exc: Exception | None = None
        for attempt in range(self._GET_MAX_RETRIES + 1):
            try:
                client = await self._get_client()
                response = await client.get(
                    url, params=params, headers=hdrs, timeout=timeout,
                )
                if response.status_code in self._GET_RETRYABLE_STATUS:
                    if attempt < self._GET_MAX_RETRIES:
                        await asyncio.sleep(self._GET_BACKOFFS[attempt])
                        continue
                return response
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_exc = e
                if attempt < self._GET_MAX_RETRIES:
                    logger.debug(
                        "GET %s failed (%s), retrying in %ss",
                        url, type(e).__name__, self._GET_BACKOFFS[attempt],
                    )
                    await asyncio.sleep(self._GET_BACKOFFS[attempt])
                    continue
                raise
        raise last_exc  # type: ignore[misc]

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

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

    async def read_blackboard(self, key: str) -> dict | None:
        """Read a value from the shared blackboard."""
        scoped = self._scope_key(key)
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/blackboard/{scoped}",
            params={"agent_id": self.agent_id},
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    async def write_blackboard(
        self, key: str, value: dict, ttl: int | None = None,
        *, project: str | None = None,
    ) -> dict:
        """Write a value to the shared blackboard.

        If *project* is given, scope the key to that project instead of
        this agent's own project.  Used by cross-project coordination
        (e.g. operator handing off work to a project-scoped agent).
        """
        if project is not None:
            scoped = f"projects/{project}/{key}"
        else:
            scoped = self._scope_key(key)
        client = await self._get_client()
        params: dict[str, str] = {"agent_id": self.agent_id}
        if ttl is not None:
            params["ttl"] = str(ttl)
        response = await client.put(
            f"{self.mesh_url}/mesh/blackboard/{scoped}",
            params=params,
            json=value,
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def delete_blackboard(self, key: str) -> dict:
        """Delete an entry from the shared blackboard."""
        scoped = self._scope_key(key)
        client = await self._get_client()
        response = await client.delete(
            f"{self.mesh_url}/mesh/blackboard/{scoped}",
            params={"agent_id": self.agent_id},
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def claim_blackboard(
        self, key: str, value: dict, expected_version: int,
    ) -> dict | None:
        """Atomic compare-and-swap write. Returns None on version conflict (409)."""
        scoped = self._scope_key(key)
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/blackboard/claim",
            json={
                "agent_id": self.agent_id,
                "key": scoped,
                "value": value,
                "expected_version": expected_version,
            },
            headers=self._trace_headers(),
        )
        if response.status_code == 409:
            return None
        response.raise_for_status()
        return response.json()

    async def list_blackboard(self, prefix: str) -> list[dict]:
        """List blackboard entries by key prefix."""
        scoped = self._scope_key(prefix)
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/blackboard/",
            params={"agent_id": self.agent_id, "prefix": scoped},
        )
        response.raise_for_status()
        entries = response.json()
        # Strip project prefix from returned keys so agents see natural keys
        if self.project_name:
            scope = f"projects/{self.project_name}/"
            for entry in entries:
                k = entry.get("key", "")
                if k.startswith(scope):
                    entry["key"] = k[len(scope):]
        return entries

    async def notify_user(self, message: str) -> None:
        """Send an unsolicited notification to the user via all channels."""
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/notify",
            json={"agent_id": self.agent_id, "message": message},
            headers=self._trace_headers(),
        )
        response.raise_for_status()

    async def wake_agent(self, target: str, message: str = "") -> dict:
        """Wake a target agent so it processes work immediately."""
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/wake",
            params={"target": target, "message": message},
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def publish_event(self, topic: str, payload: dict | None = None) -> dict:
        """Publish an event to the mesh pub/sub."""
        scoped = self._scope_topic(topic)
        event = MeshEvent(topic=scoped, source=self.agent_id, payload=payload or {})
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/publish",
            json=event.model_dump(mode="json"),
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def subscribe_topic(self, topic: str) -> dict:
        """Subscribe to a pub/sub topic at runtime."""
        scoped = self._scope_topic(topic)
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/subscribe",
            params={"topic": scoped, "agent_id": self.agent_id},
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def watch_blackboard(self, pattern: str) -> dict:
        """Register a glob pattern watch on blackboard keys."""
        scoped = self._scope_key(pattern)
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/blackboard/watch",
            json={"agent_id": self.agent_id, "pattern": scoped},
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def list_agents(self) -> dict:
        """List agents visible to this agent.

        Project-scoped agents see only their project's members.
        Standalone agents see all registered agents so they can
        coordinate cross-project (e.g. operator handing off work).
        """
        params: dict[str, str] = {}
        if self.project_name:
            params["project"] = self.project_name
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/agents",
            params=params,
        )
        response.raise_for_status()
        return response.json()

    async def create_cron(
        self, schedule: str, message: str = "", heartbeat: bool = False,
        tool_name: str | None = None, tool_params: str | None = None,
    ) -> dict:
        """Create a cron job for this agent via the mesh."""
        client = await self._get_client()
        body: dict = {
            "agent_id": self.agent_id,
            "schedule": schedule,
            "message": message,
            "heartbeat": heartbeat,
        }
        if tool_name is not None:
            body["tool_name"] = tool_name
        if tool_params is not None:
            body["tool_params"] = tool_params
        response = await client.post(
            f"{self.mesh_url}/mesh/cron",
            json=body,
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def list_cron(self) -> list[dict]:
        """List cron jobs for this agent."""
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/cron",
            params={"agent_id": self.agent_id},
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

    async def create_custom_agent(
        self, name: str, role: str, model: str = "",
        instructions: str = "", soul: str = "",
    ) -> dict:
        """Request the mesh to create a new custom agent."""
        client = await self._get_client()
        body: dict[str, str] = {"name": name, "role": role}
        if model:
            body["model"] = model
        if instructions:
            body["instructions"] = instructions
        if soul:
            body["soul"] = soul
        response = await client.post(
            f"{self.mesh_url}/mesh/agents/create",
            json=body,
            timeout=120,
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def get_agent_history(self, agent_id: str) -> dict:
        """Read another agent's daily logs (permission-checked on server)."""
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/agents/{agent_id}/history",
            params={"requesting_agent": self.agent_id},
        )
        response.raise_for_status()
        return response.json()

    async def get_agent_profile(self, agent_id: str) -> dict:
        """Read another agent's public profile (permission-checked on server)."""
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/agents/{agent_id}/profile",
            params={"requesting_agent": self.agent_id},
        )
        response.raise_for_status()
        return response.json()

    # === Vault (credential management) ===

    async def request_credential_from_user(
        self, name: str, description: str, service: str = "",
    ) -> dict:
        """Emit a credential request event to the dashboard."""
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/credential-request",
            json={
                "agent_id": self.agent_id,
                "name": name,
                "description": description,
                "service": service,
            },
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def request_browser_login(
        self, url: str, service: str, description: str,
        target_agent_id: str | None = None,
    ) -> dict:
        """Emit a browser login request event to the dashboard.

        When ``target_agent_id`` is set, the login card is routed to the
        target agent's browser profile (orchestration/delegation). The
        mesh enforces that the caller can message the target and the
        target has browser access.
        """
        client = await self._get_client()
        body: dict = {
            "agent_id": self.agent_id,
            "url": url,
            "service": service,
            "description": description,
        }
        if target_agent_id:
            body["target_agent_id"] = target_agent_id
        response = await client.post(
            f"{self.mesh_url}/mesh/browser-login-request",
            json=body,
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

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
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/vault/list",
            params={"agent_id": self.agent_id},
        )
        response.raise_for_status()
        return response.json().get("credentials", [])

    async def vault_status(self, name: str) -> dict:
        """Check whether a credential exists."""
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/vault/status/{name}",
            params={"agent_id": self.agent_id},
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

    async def introspect(self, section: str = "all") -> dict:
        """Query runtime state from the mesh (permissions, budget, fleet, etc.)."""
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/introspect",
            params={"section": section},
            headers={"X-Agent-ID": self.agent_id},
        )
        response.raise_for_status()
        return response.json()

    # === Wallet (blockchain transactions via mesh signing service) ===

    async def wallet_get_address(self, chain: str) -> dict:
        """Get this agent's wallet address for a chain."""
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/wallet/address",
            params={"agent_id": self.agent_id, "chain": chain},
        )
        response.raise_for_status()
        return response.json()

    async def wallet_get_balance(self, chain: str, token: str = "native") -> dict:
        """Get wallet balance for a token on a chain."""
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/wallet/balance",
            params={"agent_id": self.agent_id, "chain": chain, "token": token},
        )
        response.raise_for_status()
        return response.json()

    async def wallet_read_contract(
        self, chain: str, contract: str, function: str, args: list,
    ) -> dict:
        """Read-only contract call (EVM: eth_call, Solana: getAccountInfo)."""
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/wallet/read",
            json={
                "agent_id": self.agent_id, "chain": chain,
                "contract": contract, "function": function, "args": args,
            },
            timeout=30,
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def wallet_transfer(
        self, chain: str, to: str, amount: str, token: str = "native",
    ) -> dict:
        """Request a token transfer.  Signed and broadcast by the mesh."""
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/wallet/transfer",
            json={
                "agent_id": self.agent_id, "chain": chain,
                "to": to, "amount": amount, "token": token,
            },
            timeout=60,
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def wallet_execute(
        self,
        chain: str,
        contract: str = "",
        function: str = "",
        args: list | None = None,
        value: str = "0",
        transaction: str = "",
    ) -> dict:
        """EVM: contract + function + args.  Solana: base64 unsigned tx."""
        client = await self._get_client()
        body: dict = {"agent_id": self.agent_id, "chain": chain}
        if transaction:
            body["transaction"] = transaction
        else:
            body.update({
                "contract": contract, "function": function,
                "args": args or [], "value": value,
            })
        response = await client.post(
            f"{self.mesh_url}/mesh/wallet/execute",
            json=body,
            timeout=60,
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    # === Image generation (via mesh API proxy) ===

    async def image_generate(
        self,
        prompt: str,
        size: str = "square",
        provider: str = "gemini",
        timeout: int = 60,
    ) -> dict:
        """Generate an image via the mesh API proxy."""
        from src.shared.types import APIProxyRequest

        api_request = APIProxyRequest(
            service="image_gen",
            action="generate",
            params={"prompt": prompt, "size": size, "provider": provider},
        )
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/api",
            params={"agent_id": self.agent_id},
            json=api_request.model_dump(mode="json"),
            timeout=timeout,
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    # === Fleet Templates ===

    async def list_fleet_templates(self) -> dict:
        """List available fleet templates from the mesh."""
        response = await self._get_with_retry(f"{self.mesh_url}/mesh/fleet/templates")
        response.raise_for_status()
        return response.json()

    async def apply_fleet_template(self, template: str, model: str = "") -> dict:
        """Apply a fleet template to create a team of agents."""
        client = await self._get_client()
        body: dict = {"template": template}
        if model:
            body["model"] = model
        response = await client.post(
            f"{self.mesh_url}/mesh/fleet/apply",
            json=body,
            timeout=120,
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    # === Browser (shared browser service via mesh proxy) ===

    # === Operator agent config management ===

    async def propose_config_change(self, agent_id: str, field: str, value) -> dict:
        """Propose a config change for an agent. Returns preview + change_id."""
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/agents/{agent_id}/propose",
            json={"field": field, "value": value, "proposed_by": self.agent_id},
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def confirm_config_change(self, change_id: str) -> dict:
        """Confirm and apply a previously proposed config change.

        Uses the dedicated /mesh/config/confirm endpoint which resolves
        the target agent_id from the pending change internally.
        """
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/config/confirm",
            json={"change_id": change_id, "confirmed_by": self.agent_id},
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def get_agent_config(self, agent_id: str) -> dict:
        """Get the current config for an agent."""
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/agents/{agent_id}/config",
            params={"requesting_agent": self.agent_id},
        )
        response.raise_for_status()
        return response.json()

    # === Project management (mesh proxy endpoints) ===

    async def list_projects(self) -> dict:
        """List all projects via mesh proxy."""
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/projects",
        )
        response.raise_for_status()
        return response.json()

    async def create_project(
        self, name: str, description: str, members: list[str] | None = None,
    ) -> dict:
        """Create a new project via mesh proxy."""
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/projects",
            json={
                "name": name,
                "description": description,
                "members": members or [],
            },
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def add_agent_to_project(self, project_name: str, agent_id: str) -> dict:
        """Add an agent to a project via mesh proxy."""
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/projects/{project_name}/members",
            json={"agent": agent_id},
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def remove_agent_from_project(
        self, project_name: str, agent_id: str,
    ) -> dict:
        """Remove an agent from a project via mesh proxy."""
        client = await self._get_client()
        response = await client.delete(
            f"{self.mesh_url}/mesh/projects/{project_name}/members/{agent_id}",
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def update_project_context(
        self, project_name: str, context: str,
    ) -> dict:
        """Update a project's description/context via mesh proxy."""
        client = await self._get_client()
        response = await client.put(
            f"{self.mesh_url}/mesh/projects/{project_name}/context",
            json={"context": context},
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def browser_command(
        self, action: str, params: dict | None = None,
        target_agent_id: str | None = None,
    ) -> dict:
        """Send a browser command through the mesh to the shared browser service.

        When ``target_agent_id`` is set, the command is delegated to the
        target agent's browser profile (used by orchestrators like
        operator). The mesh validates that the caller can message the
        target and the target has browser access. When omitted, the
        command acts on the caller's own browser profile.
        """
        client = await self._get_client()
        body: dict = {
            "agent_id": self.agent_id,
            "action": action,
            "params": params or {},
        }
        if target_agent_id:
            body["target_agent_id"] = target_agent_id
        response = await client.post(
            f"{self.mesh_url}/mesh/browser/command",
            json=body,
            timeout=60,
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    # ── Operator metrics ─────────────────────────────────────────

    async def get_system_metrics(self) -> dict:
        """Get fleet-wide pre-computed metrics from the mesh."""
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/system/metrics",
        )
        response.raise_for_status()
        return response.json()

    async def get_agent_metrics(self, agent_id: str) -> dict:
        """Get per-agent pre-computed metrics from the mesh."""
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/agents/{agent_id}/metrics",
        )
        response.raise_for_status()
        return response.json()


