"""HTTP client for agent-to-mesh communication.

This is the agent's ONLY interface to the outside world.
All external interaction is mediated by the mesh host process.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

import httpx

from src.shared.trace import origin_header
from src.shared.types import MeshEvent
from src.shared.utils import setup_logging

if TYPE_CHECKING:
    from src.shared.types import MessageOrigin

logger = setup_logging("agent.mesh_client")


class MeshClient:
    """HTTP client for agent-to-mesh communication.

    Uses a shared httpx.AsyncClient for connection pooling.
    """

    def __init__(
        self,
        mesh_url: str,
        agent_id: str,
        team_name: str | None = None,
        *,
        project_name: str | None = None,
    ):
        # ``team_name`` is the canonical kwarg; ``project_name`` is kept
        # as a back-compat alias through PR 3. If both are passed,
        # ``team_name`` wins.
        self.mesh_url = mesh_url
        self.agent_id = agent_id
        self.team_name = team_name if team_name is not None else project_name
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()
        self._auth_token: str = os.environ.get("MESH_AUTH_TOKEN", "")

    # Back-compat alias — kept until PR 3.
    @property
    def project_name(self) -> str | None:
        """DEPRECATED: alias for :attr:`team_name`."""
        return self.team_name

    @project_name.setter
    def project_name(self, value: str | None) -> None:
        self.team_name = value

    @property
    def is_standalone(self) -> bool:
        """True when this agent is not assigned to any team."""
        return self.team_name is None

    def _scope_key(self, key: str) -> str:
        """Prefix a blackboard key with the team namespace.

        Team agents transparently read/write under ``projects/{name}/``
        (the blackboard key prefix is left as ``projects/`` for back-
        compat with existing blackboard data; PR 3 may revisit). Solo
        agents (no team) pass keys through unchanged — but they are
        blocked from the blackboard at both the tool and permission
        layers.
        """
        if self.team_name:
            return f"projects/{self.team_name}/{key}"
        return key

    def _scope_topic(self, topic: str) -> str:
        """Prefix a pub/sub topic with the team namespace.

        Team agents transparently publish/subscribe under
        ``projects/{name}/`` (prefix retained for back-compat with
        existing pub/sub topics; PR 3 may revisit). Solo agents (no
        team) use raw topic names.
        """
        if self.team_name:
            return f"projects/{self.team_name}/{topic}"
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

    async def read_blackboard(self, key: str, *, global_scope: bool = False) -> dict | None:
        """Read a value from the shared blackboard.

        If *global_scope* is True, the key is sent as-is, bypassing the
        per-agent project scoping. Used to read fleet-global keys such as
        the operator inbox under ``global/tasks/operator/``.
        """
        scoped = key if global_scope else self._scope_key(key)
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
        *, project: str | None = None, global_scope: bool = False,
    ) -> dict:
        """Write a value to the shared blackboard.

        If *project* is given, scope the key to that project instead of
        this agent's own project.  Used by cross-project coordination
        (e.g. operator handing off work to a project-scoped agent).

        If *global_scope* is True, the key is sent as-is — bypassing both
        the explicit ``project=`` override and the auto project prefix.
        Used for fleet-global namespaces such as the operator inbox.
        """
        if global_scope:
            scoped = key
        elif project is not None:
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

    async def delete_blackboard(self, key: str, *, global_scope: bool = False) -> dict:
        """Delete an entry from the shared blackboard.

        If *global_scope* is True, the key is sent as-is, bypassing project
        scoping. Used for keys in the fleet-global namespace.
        """
        scoped = key if global_scope else self._scope_key(key)
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

    async def list_blackboard(
        self, prefix: str, *, global_scope: bool = False,
    ) -> list[dict]:
        """List blackboard entries by key prefix.

        If *global_scope* is True, the prefix is sent as-is, bypassing the
        per-agent project scoping. Used for fleet-global namespaces such
        as the operator inbox at ``global/tasks/operator/``.
        """
        scoped = prefix if global_scope else self._scope_key(prefix)
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/blackboard/",
            params={"agent_id": self.agent_id, "prefix": scoped},
        )
        response.raise_for_status()
        entries = response.json()
        # Strip project prefix only when we used project scoping
        if not global_scope and self.project_name:
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

    async def wake_agent(
        self, target: str, message: str = "",
        origin: "MessageOrigin | None" = None,
        task_id: str | None = None,
    ) -> dict:
        """Wake a target agent so it processes work immediately.

        ``task_id`` plumbs through as the ``x-task-id`` header so the
        recipient's lane worker can pass it to ``/chat``; the agent then
        auto-closes that task when its loop returns. Omitting ``task_id``
        preserves legacy wake semantics — no auto-close fires.
        """
        client = await self._get_client()
        headers = self._trace_headers()
        headers.update(origin_header(origin))
        if task_id:
            headers["x-task-id"] = task_id
        response = await client.post(
            f"{self.mesh_url}/mesh/wake",
            params={"target": target, "message": message},
            headers=headers,
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

    async def request_captcha_help(
        self, service: str, description: str,
        target_agent_id: str | None = None,
    ) -> dict:
        """Phase 8 §11.14 — emit a CAPTCHA-help handoff event.

        Mirrors :meth:`request_browser_login` shape: the dashboard renders
        a handoff card so the operator can take VNC control and clear
        the captcha manually. ``service`` names the captcha kind/service
        (e.g. ``"Cloudflare Turnstile"`` or ``"PerimeterX"``);
        ``description`` is agent-supplied context for the operator.
        """
        client = await self._get_client()
        body: dict = {
            "agent_id": self.agent_id,
            "service": service,
            "description": description,
        }
        if target_agent_id:
            body["target_agent_id"] = target_agent_id
        response = await client.post(
            f"{self.mesh_url}/mesh/browser-captcha-help-request",
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

    async def apply_fleet_template(
        self,
        template: str,
        model: str = "",
        agent_overrides: dict[str, dict] | None = None,
    ) -> dict:
        """Apply a fleet template to create a team of agents.

        ``agent_overrides`` is an optional mapping of ``agent_name`` →
        per-slot dict with any of: ``model``, ``instructions``, ``soul``,
        ``heartbeat``, ``interface`` (PR-N v2). Validated upfront on the
        mesh side; unknown names / fields / models return HTTP 400, oversized
        string fields return HTTP 413, with the offender named.

        Per-slot creation is non-atomic: a mid-loop failure leaves
        earlier-created agents in place; check the returned ``created`` list.
        """
        client = await self._get_client()
        body: dict = {"template": template}
        if model:
            body["model"] = model
        if agent_overrides:
            body["agent_overrides"] = agent_overrides
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

    async def edit_soft(self, agent_id: str, field: str, value, reason: str) -> dict:
        """Apply an agent-config edit immediately. Returns receipt.

        Backed by ``POST /mesh/agents/{id}/edit-soft`` (path name retained
        for backward compatibility; semantically it is "edit-apply"). All
        fields — soft (instructions/soul/heartbeat/...) and hard (model/
        permissions/budget/thinking) — apply immediately. The undo TTL is
        field-aware: 5 min for soft fields, 30 min for hard fields. The
        response includes ``undo_token``, ``expires_at``, ``ttl_seconds``,
        and ``field_class``.
        """
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/agents/{agent_id}/edit-soft",
            json={"field": field, "value": value, "reason": reason},
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def undo_change(self, undo_token: str) -> dict:
        """Reverse a recent soft edit by undo_token.

        404 if the token is unknown, expired (5min TTL), or already used.
        On success returns the restored value so the operator can echo
        what was reverted to the user.
        """
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/changes/undo/{undo_token}",
            json={},
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def get_agent_config(
        self, agent_id: str, fields: list[str] | None = None,
    ) -> dict:
        """Get the current config for an agent (operator-only).

        Returns ``{agent_id, config: {...}}`` mirroring ``edit_agent``'s
        field surface. Pass ``fields=["instructions", "soul"]`` to scope
        the response.
        """
        params: dict[str, str] = {"requesting_agent": self.agent_id}
        if fields:
            params["fields"] = ",".join(fields)
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/agents/{agent_id}/config",
            params=params,
        )
        response.raise_for_status()
        return response.json()

    async def list_peer_artifacts(self, agent_id: str) -> dict:
        """List a peer agent's artifact files (operator-only).

        Backs the operator's ``list_peer_artifacts`` tool. Returns
        ``{agent_id, artifacts: [{name, size, modified}, ...]}``. 404
        if the agent doesn't exist; 403 if the caller isn't operator.
        """
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/agents/{agent_id}/artifacts",
            params={"requesting_agent": self.agent_id},
        )
        response.raise_for_status()
        return response.json()

    async def read_peer_artifact(self, agent_id: str, name: str) -> dict:
        """Read a single peer artifact's content (operator-only).

        Backs the operator's ``read_peer_artifact`` tool. Returns
        ``{agent_id, name, content, size, encoding}``. 404 if the
        agent or artifact is missing; 413 if the file exceeds the
        mesh-layer cap; 403 if the caller isn't operator.
        """
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/agents/{agent_id}/artifacts/{name}",
            params={"requesting_agent": self.agent_id},
        )
        response.raise_for_status()
        return response.json()

    async def cancel_pending_action(self, nonce: str) -> dict:
        """Cancel a pending action by nonce (operator self-cleanup).

        Backs the operator's ``cancel_pending_action`` tool. Returns the
        cancelled record's ``target_kind`` / ``target_id`` / ``action_kind``
        so the operator can describe what was cancelled. 404 if unknown
        or already expired/consumed.
        """
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/pending/{nonce}/cancel",
            json={},
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def list_pending_actions(self) -> dict:
        """List every non-expired pending action.

        Backs the operator's ``list_pending`` tool. Returns
        ``{pending: [{nonce, action_kind, target_kind, target_id,
        expires_at, actor, summary}, ...]}``. Operator-or-internal only —
        non-operator agents will see HTTP 403.
        """
        client = await self._get_client()
        response = await client.get(
            f"{self.mesh_url}/mesh/pending",
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def archive_audit_before(self, before_date: str) -> dict:
        """Bulk-archive operator audit entries older than ``before_date``.

        Soft-archive: rows are flipped to ``archived=1`` and dropped from
        the default audit-log view. Returns ``{archived_count: N}``.
        """
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/audit/archive",
            json={"before_date": before_date},
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    # === Team management (mesh proxy endpoints) ===
    #
    # The ``*_team`` methods are the canonical names; the legacy
    # ``*_project`` methods are preserved as thin no-cost shims that
    # proxy through the canonical name. Every method hits the
    # ``/mesh/teams/*`` route — PR 3 removed the legacy
    # ``/mesh/projects/*`` mirror endpoints.

    async def list_teams(self) -> dict:
        """List all teams via mesh proxy."""
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/teams",
        )
        response.raise_for_status()
        return response.json()

    async def list_projects(self) -> dict:
        """DEPRECATED: alias for :meth:`list_teams`."""
        return await self.list_teams()

    async def create_team(
        self, name: str, description: str, members: list[str] | None = None,
    ) -> dict:
        """Create a new team via mesh proxy."""
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/teams",
            json={
                "name": name,
                "description": description,
                "members": members or [],
            },
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def create_project(
        self, name: str, description: str, members: list[str] | None = None,
    ) -> dict:
        """DEPRECATED: alias for :meth:`create_team`."""
        return await self.create_team(name, description, members)

    async def add_agent_to_team(self, team_name: str, agent_id: str) -> dict:
        """Add an agent to a team via mesh proxy."""
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/teams/{team_name}/members",
            json={"agent": agent_id},
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def add_agent_to_project(self, project_name: str, agent_id: str) -> dict:
        """DEPRECATED: alias for :meth:`add_agent_to_team`."""
        return await self.add_agent_to_team(project_name, agent_id)

    async def remove_agent_from_team(
        self, team_name: str, agent_id: str,
    ) -> dict:
        """Remove an agent from a team via mesh proxy."""
        client = await self._get_client()
        response = await client.delete(
            f"{self.mesh_url}/mesh/teams/{team_name}/members/{agent_id}",
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def remove_agent_from_project(
        self, project_name: str, agent_id: str,
    ) -> dict:
        """DEPRECATED: alias for :meth:`remove_agent_from_team`."""
        return await self.remove_agent_from_team(project_name, agent_id)

    async def update_team_context(
        self, team_name: str, context: str,
    ) -> dict:
        """Update a team's description/context via mesh proxy."""
        client = await self._get_client()
        response = await client.put(
            f"{self.mesh_url}/mesh/teams/{team_name}/context",
            json={"context": context},
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def update_project_context(
        self, project_name: str, context: str,
    ) -> dict:
        """DEPRECATED: alias for :meth:`update_team_context`."""
        return await self.update_team_context(project_name, context)

    async def set_team_goal(
        self,
        team_name: str,
        north_star: str | None,
        success_criteria: list[str] | None = None,
    ) -> dict:
        """Set a team's north star + success criteria via mesh proxy."""
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/teams/{team_name}/goal",
            json={
                "north_star": north_star,
                "success_criteria": success_criteria,
            },
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def set_project_goal(
        self,
        project_name: str,
        north_star: str | None,
        success_criteria: list[str] | None = None,
    ) -> dict:
        """DEPRECATED: alias for :meth:`set_team_goal`."""
        return await self.set_team_goal(project_name, north_star, success_criteria)

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

    async def browser_upload_stage(
        self, body, idempotency_key: str | None = None,
    ) -> dict:
        """Phase A of the §4.5 file-upload flow.

        Streams ``body`` to ``/mesh/browser/upload-stage`` and returns the
        staging response (``{staged_handle, size_bytes, expires_at}``).
        ``body`` may be ``bytes`` or any object accepted by httpx's
        ``content=`` (file handle, async iterable). Mesh enforces the 50MB
        cap; pass an ``idempotency_key`` to dedupe retried uploads of the
        same bytes.
        """
        client = await self._get_client()
        headers = dict(self._trace_headers())
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        response = await client.post(
            f"{self.mesh_url}/mesh/browser/upload-stage",
            content=body,
            headers=headers,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()

    async def browser_upload_apply(self, body: dict) -> dict:
        """Phase B of the §4.5 file-upload flow.

        Body: ``{ref, staged_handles, target_agent_id?, idempotency_key?}``.
        Mesh resolves the staged handles, streams bytes into the browser
        container, drives the file-chooser, and returns the browser's
        response envelope.
        """
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/browser/upload_file",
            json=body,
            timeout=120,
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def browser_download(
        self, ref: str, timeout_ms: int = 30000,
        target_agent_id: str | None = None,
    ) -> dict:
        """Trigger a browser download via mesh and ingest it as an artifact.

        Mesh orchestrates the click+save in the shared browser, streams
        the bytes into the (target) agent's ``/artifacts/ingest`` endpoint,
        and cleans up the browser-side staging file. Returns
        ``{success, data: {artifact_name, size_bytes, mime_type}}`` (or
        an error envelope passed through from the browser side).
        """
        client = await self._get_client()
        body: dict = {"ref": ref, "timeout_ms": timeout_ms}
        if target_agent_id:
            body["target_agent_id"] = target_agent_id
        response = await client.post(
            f"{self.mesh_url}/mesh/browser/download",
            json=body,
            timeout=300,
            headers=self._trace_headers(),
        )
        try:
            data = response.json()
        except Exception:
            response.raise_for_status()
            raise
        # Pass through structured error envelopes (e.g. operator kill switch
        # returning ``BROWSER_DOWNLOADS_DISABLED``) so the calling skill can
        # surface the reason to the LLM without an exception. FastAPI wraps
        # HTTPException(detail=...) as ``{"detail": <envelope>}``, so unwrap.
        if response.status_code >= 400 and isinstance(data, dict):
            envelope = data.get("detail") if "detail" in data else data
            if (
                isinstance(envelope, dict)
                and envelope.get("success") is False
            ):
                return envelope
        response.raise_for_status()
        if not isinstance(data, dict):
            raise ValueError(
                f"Unexpected browser_download response: {type(data).__name__}",
            )
        return data

    # ── Orchestration tasks v2 (Task 6) ─────────────────────────

    # Cached probe result. Set on first call so the agent doesn't
    # round-trip to /mesh/orchestration/status on every coordination
    # call. ``None`` = not yet probed; ``True``/``False`` once known.
    _orchestration_v2_cache: bool | None = None

    async def orchestration_v2_enabled(self) -> bool:
        """Return True when the mesh has v2 enabled. Fail-closed.

        Cached for the process lifetime — the env var only changes on
        mesh restart, and a restart drops this whole client. Any error
        (network, 503, malformed JSON) is treated as "v2 off" so the
        coordination tool falls back to the legacy blackboard path.
        """
        if self._orchestration_v2_cache is not None:
            return self._orchestration_v2_cache
        try:
            response = await self._get_with_retry(
                f"{self.mesh_url}/mesh/orchestration/status",
                timeout=5,
            )
            if response.status_code == 200:
                data = response.json()
                self._orchestration_v2_cache = bool(data.get("enabled"))
            else:
                self._orchestration_v2_cache = False
        except Exception as e:
            logger.debug("orchestration_v2 probe failed (fail-closed): %s", e)
            self._orchestration_v2_cache = False
        return self._orchestration_v2_cache

    async def create_task(
        self,
        *,
        assignee: str,
        title: str,
        description: str | None = None,
        project: str | None = None,
        parent_task_id: str | None = None,
        priority: int = 0,
        dependencies: list[str] | None = None,
        artifact_refs: list[str] | None = None,
        origin: "MessageOrigin | None" = None,
    ) -> dict:
        """Create a durable task. Returns the new task record.

        ``origin`` propagates the cross-agent provenance header so the
        receiving agent's lane worker can attribute task completion back
        to the originating channel/user. Without this, ``hand_off`` v2
        loses the origin a sibling ``wake_agent`` call still carries.
        """
        client = await self._get_client()
        body: dict = {
            "assignee": assignee,
            "title": title,
            "priority": priority,
        }
        if description is not None:
            body["description"] = description
        if project is not None:
            body["project"] = project
        if parent_task_id is not None:
            body["parent_task_id"] = parent_task_id
        if dependencies is not None:
            body["dependencies"] = dependencies
        if artifact_refs is not None:
            body["artifact_refs"] = artifact_refs
        headers = self._trace_headers()
        if origin is not None:
            headers.update(origin_header(origin))
        response = await client.post(
            f"{self.mesh_url}/mesh/tasks",
            json=body,
            headers=headers,
        )
        response.raise_for_status()
        return response.json()

    async def get_task(self, task_id: str) -> dict | None:
        """Read a task by id. Returns None on 404."""
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/tasks/{task_id}",
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    async def list_task_inbox(self, assignee: str | None = None) -> list[dict]:
        """List tasks assigned to ``assignee`` (defaults to self)."""
        target = assignee or self.agent_id
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/tasks/inbox/{target}",
        )
        response.raise_for_status()
        data = response.json()
        return data.get("tasks", []) if isinstance(data, dict) else []

    async def list_team_tasks(self, team_id: str) -> list[dict]:
        """List tasks scoped to ``team_id``."""
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/tasks/team/{team_id}",
        )
        response.raise_for_status()
        data = response.json()
        return data.get("tasks", []) if isinstance(data, dict) else []

    async def list_project_tasks(self, project_id: str) -> list[dict]:
        """DEPRECATED: alias for :meth:`list_team_tasks`."""
        return await self.list_team_tasks(project_id)

    async def set_task_status(
        self, task_id: str, status: str,
        blocker_note: str | None = None,
        result: dict | None = None,
        error: str | None = None,
    ) -> dict:
        """Transition a task to ``status``.

        ``result`` (typically ``{"summary": "..."}``) and ``error`` are
        forwarded to the mesh status endpoint so terminal transitions can
        carry payload data into the back-edge inbox event written for the
        originating agent.
        """
        client = await self._get_client()
        body: dict = {"status": status}
        if blocker_note is not None:
            body["blocker_note"] = blocker_note
        if result is not None:
            body["result"] = result
        if error is not None:
            body["error"] = error
        response = await client.post(
            f"{self.mesh_url}/mesh/tasks/{task_id}/status",
            json=body,
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def reroute_task(
        self, task_id: str, new_assignee: str, reason: str = "",
    ) -> dict:
        """Reassign a task to ``new_assignee``."""
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/tasks/{task_id}/reroute",
            json={"new_assignee": new_assignee, "reason": reason},
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def cancel_task(self, task_id: str, reason: str = "") -> dict:
        """Cancel a task."""
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/tasks/{task_id}/cancel",
            json={"reason": reason},
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def retry_task(
        self, task_id: str,
        title: str | None = None,
        description: str | None = None,
        assignee: str | None = None,
    ) -> dict:
        """Retry a failed task. Optional patch overrides title/description/assignee."""
        body: dict = {}
        if title is not None:
            body["title"] = title
        if description is not None:
            body["description"] = description
        if assignee is not None:
            body["assignee"] = assignee
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/tasks/{task_id}/retry",
            json=body,
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    # ── Operator product surface (Task 7) ────────────────────────

    async def team_status(self, team_id: str) -> dict:
        """Per-team status counts + recent blockers/completions."""
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/teams/{team_id}/status",
        )
        response.raise_for_status()
        return response.json()

    async def project_status(self, project_id: str) -> dict:
        """DEPRECATED: alias for :meth:`team_status`."""
        return await self.team_status(project_id)

    async def all_teams_status(self) -> dict:
        """Status rollup across every visible team."""
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/teams/status",
        )
        response.raise_for_status()
        return response.json()

    async def all_projects_status(self) -> dict:
        """DEPRECATED: alias for :meth:`all_teams_status`."""
        return await self.all_teams_status()

    async def agent_queue(self, agent_id: str, limit: int = 10) -> dict:
        """Recent tasks for an agent grouped by status."""
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/agents/{agent_id}/queue",
            params={"limit": limit},
        )
        response.raise_for_status()
        return response.json()

    async def team_outputs(self, team_id: str, since: str = "") -> dict:
        """Completed task artifacts for a team in a time window."""
        params = {"since": since} if since else None
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/teams/{team_id}/outputs",
            params=params,
        )
        response.raise_for_status()
        return response.json()

    async def project_outputs(self, project_id: str, since: str = "") -> dict:
        """DEPRECATED: alias for :meth:`team_outputs`."""
        return await self.team_outputs(project_id, since)

    async def team_summary(self, team_id: str) -> dict:
        """Synthesized status summary for a team."""
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/teams/{team_id}/summary",
        )
        response.raise_for_status()
        return response.json()

    async def project_summary(self, project_id: str) -> dict:
        """DEPRECATED: alias for :meth:`team_summary`."""
        return await self.team_summary(project_id)

    async def archive_team(self, name: str) -> dict:
        """Archive a team."""
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/teams/{name}/archive",
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def archive_project(self, name: str) -> dict:
        """DEPRECATED: alias for :meth:`archive_team`."""
        return await self.archive_team(name)

    async def unarchive_team(self, name: str) -> dict:
        """Unarchive (restore) a previously archived team."""
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/teams/{name}/unarchive",
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def unarchive_project(self, name: str) -> dict:
        """DEPRECATED: alias for :meth:`unarchive_team`."""
        return await self.unarchive_team(name)

    async def archive_agent(self, agent_id: str) -> dict:
        """Archive an agent."""
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/agents/{agent_id}/archive",
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def propose_delete_team(self, name: str) -> dict:
        """Propose deletion of an archived team. Returns nonce for human confirm."""
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/teams/{name}/propose-delete",
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def propose_delete_project(self, name: str) -> dict:
        """DEPRECATED: alias for :meth:`propose_delete_team`."""
        return await self.propose_delete_team(name)

    async def propose_delete_agent(self, agent_id: str) -> dict:
        """Propose deletion of an archived agent. Returns nonce for human confirm."""
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/agents/{agent_id}/propose-delete",
            headers=self._trace_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def list_task_events(self, task_id: str) -> list[dict]:
        """Audit history for a task."""
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/tasks/{task_id}/events",
        )
        response.raise_for_status()
        data = response.json()
        return data.get("events", []) if isinstance(data, dict) else []

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

    async def get_agent_stale_tasks(
        self, agent_id: str, threshold_hours: int = 24,
    ) -> dict:
        """Get up to 5 oldest stale task IDs for ``agent_id``.

        Powers ``inspect_agents(stale_threshold_hours=N)`` in the
        operator heartbeat. Returns ``{"agent_id", "threshold_hours",
        "count", "task_ids"}``. Operator-only on the mesh side.
        """
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/agents/{agent_id}/stale-tasks"
            f"?threshold_hours={int(threshold_hours)}",
        )
        response.raise_for_status()
        return response.json()


