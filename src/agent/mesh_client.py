"""HTTP client for agent-to-mesh communication.

This is the agent's ONLY interface to the outside world.
All external interaction is mediated by the mesh host process.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

import httpx

from src.shared.trace import TRACE_HEADER, origin_header
from src.shared.types import MeshEvent
from src.shared.utils import setup_logging

if TYPE_CHECKING:
    from src.shared.types import MessageOrigin

logger = setup_logging("agent.mesh_client")


def _raise_with_body(response: httpx.Response) -> None:
    """``response.raise_for_status`` plus the body's ``detail`` field.

    The mesh server wraps every denial in a descriptive ``HTTPException``
    detail (e.g. ``"Agent X cannot wake Y"``, ``"Caller Z is not a
    member of team 'foo'"``). Plain ``raise_for_status`` discards
    that body — the exception string is just ``"Client error '403
    Forbidden' for url ..."``, leaving the caller to guess which gate
    fired. Wrapping it here means every coordination tool's failure
    envelope (``wake_failed``, ``create_failed``, ``drive_write_failed``,
    ``update_status_failed``, ``complete_task_failed``) surfaces the
    gate reason in its ``error`` field without changes at the callsites.
    """
    if response.is_success:
        return
    body_detail = ""
    try:
        parsed = response.json()
        if isinstance(parsed, dict) and "detail" in parsed:
            body_detail = str(parsed["detail"])[:500]
    except (ValueError, TypeError):
        try:
            body_detail = response.text[:500]
        except Exception:
            body_detail = ""
    # Call httpx's stock raiser directly — referring to the bound method
    # via ``type(response)`` sidesteps the module's ``response.raise_for_status``
    # replace-all that produced this wrapper in the first place. Without
    # this dance the helper would recurse into itself.
    raiser = httpx.Response.raise_for_status
    try:
        raiser(response)
    except httpx.HTTPStatusError as e:
        if body_detail:
            raise httpx.HTTPStatusError(
                f"{e}: {body_detail}",
                request=e.request,
                response=e.response,
            ) from None
        raise


class MeshClient:
    """HTTP client for agent-to-mesh communication.

    Uses a shared httpx.AsyncClient for connection pooling.
    """

    def __init__(
        self,
        mesh_url: str,
        agent_id: str,
        team_name: str | None = None,
    ):
        self.mesh_url = mesh_url
        self.agent_id = agent_id
        self.team_name = team_name
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()
        self._auth_token: str = os.environ.get("MESH_AUTH_TOKEN", "")

    def _scope_key(self, key: str) -> str:
        """Prefix a blackboard key with the team namespace.

        Workers always carry a team name: their real team, or their own
        agent id when solo (team-of-one, ratified decision #5) — so all
        their keys live under ``teams/{name}/``. Only the operator runs
        with ``team_name=None`` and passes keys through unscoped.
        """
        if self.team_name:
            return f"teams/{self.team_name}/{key}"
        return key

    def _scope_topic(self, topic: str) -> str:
        """Prefix a pub/sub topic with the team namespace.

        Workers always publish/subscribe under ``teams/{name}/`` (real
        team, or the private team-of-one namespace when solo). Only the
        operator uses raw topic names.
        """
        if self.team_name:
            return f"teams/{self.team_name}/{topic}"
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
        _raise_with_body(response)

    async def read_blackboard(self, key: str, *, global_scope: bool = False) -> dict | None:
        """Read a value from the shared blackboard.

        If *global_scope* is True, the key is sent as-is, bypassing the
        per-agent team scoping. Used to read fleet-global keys such as
        the operator inbox under ``global/tasks/operator/``.
        """
        scoped = key if global_scope else self._scope_key(key)
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/blackboard/{scoped}",
            params={"agent_id": self.agent_id},
        )
        if response.status_code == 404:
            return None
        _raise_with_body(response)
        return response.json()

    async def write_blackboard(
        self, key: str, value: dict, ttl: int | None = None,
        *, team: str | None = None, global_scope: bool = False,
    ) -> dict:
        """Write a value to the shared blackboard.

        If *team* is given, scope the key to that team instead of
        this agent's own team.  Used by cross-team coordination
        (e.g. operator handing off work to a team-scoped agent).

        If *global_scope* is True, the key is sent as-is — bypassing both
        the explicit ``team=`` override and the auto team prefix.
        Used for fleet-global namespaces such as the operator inbox.
        """
        if global_scope:
            scoped = key
        elif team is not None:
            scoped = f"teams/{team}/{key}"
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
        _raise_with_body(response)
        return response.json()

    async def delete_blackboard(
        self, key: str, *, team: str | None = None, global_scope: bool = False,
    ) -> dict:
        """Delete an entry from the shared blackboard.

        If *team* is given, scope the key to that team instead of
        this agent's own team.  Used by cross-team coordination
        (e.g. operator clearing a team-scoped agent's goals key).

        If *global_scope* is True, the key is sent as-is — bypassing both
        the explicit ``team=`` override and the auto team prefix.
        Used for keys in the fleet-global namespace.
        """
        if global_scope:
            scoped = key
        elif team is not None:
            scoped = f"teams/{team}/{key}"
        else:
            scoped = self._scope_key(key)
        client = await self._get_client()
        response = await client.delete(
            f"{self.mesh_url}/mesh/blackboard/{scoped}",
            params={"agent_id": self.agent_id},
            headers=self._trace_headers(),
        )
        _raise_with_body(response)
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
        _raise_with_body(response)
        return response.json()

    async def list_blackboard(
        self, prefix: str, *, global_scope: bool = False,
    ) -> list[dict]:
        """List blackboard entries by key prefix.

        If *global_scope* is True, the prefix is sent as-is, bypassing the
        per-agent team scoping. Used for fleet-global namespaces such
        as the operator inbox at ``global/tasks/operator/``.
        """
        scoped = prefix if global_scope else self._scope_key(prefix)
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/blackboard/",
            params={"agent_id": self.agent_id, "prefix": scoped},
        )
        _raise_with_body(response)
        entries = response.json()
        # Strip the team prefix only when we used team scoping
        if not global_scope and self.team_name:
            scope = f"teams/{self.team_name}/"
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
        _raise_with_body(response)

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
        _raise_with_body(response)
        return response.json()

    async def ask_teammate(
        self, to: str, question: str, timeout_seconds: int | None = None,
    ) -> dict:
        """Inline teammate question via the mesh ask broker (Phase 2 u3).

        Blocks until the mesh returns an answer or a timeout envelope —
        the SERVER owns the wait; our HTTP timeout only adds headroom on
        top of the (clamped) ask timeout so a mesh that resolves the
        limit differently still gets to answer. Non-200 responses are
        returned as ``{"http_error": True, "status_code", "detail"}``
        instead of raising so the tool can shape Constraint-#10 failure
        envelopes from the structured detail (e.g. the unknown-recipient
        roster).
        """
        from src.shared import limits

        payload: dict = {"to": to, "question": question}
        if timeout_seconds:
            timeout_seconds = limits.clamp(
                "ask_timeout_seconds", int(timeout_seconds),
            )
            payload["timeout_seconds"] = timeout_seconds
        effective = timeout_seconds or limits.resolve("ask_timeout_seconds")
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/ask",
            json=payload,
            headers=self._trace_headers(),
            timeout=effective + 30,
        )
        if response.status_code == 200:
            return response.json()
        return {
            "http_error": True,
            "status_code": response.status_code,
            "detail": self._error_detail(response),
        }

    async def answer_ask(self, ask_id: str, answer: str) -> dict:
        """Deliver an answer for an in-flight ask (single-use)."""
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/ask/{ask_id}/answer",
            json={"answer": answer},
            headers=self._trace_headers(),
            timeout=30,
        )
        if response.status_code == 200:
            return response.json()
        return {
            "http_error": True,
            "status_code": response.status_code,
            "detail": self._error_detail(response),
        }

    @staticmethod
    def _error_detail(response: httpx.Response):
        """Best-effort structured FastAPI ``detail`` from an error body."""
        try:
            parsed = response.json()
            if isinstance(parsed, dict) and "detail" in parsed:
                return parsed["detail"]
            return parsed
        except Exception:
            return response.text[:500]

    async def record_trace(
        self,
        event_type: str,
        *,
        detail: str = "",
        status: str = "ok",
        error: str = "",
        duration_ms: int = 0,
        meta: dict | None = None,
    ) -> None:
        """Record an agent-side trace event on the mesh (Phase 4).

        The agent never writes ``traces.db`` directly — container isolation is
        the whole reason traces are host-only — so it POSTs to ``/mesh/traces``
        and the mesh records on its behalf under the inbound ``x-trace-id``
        (mirroring how ``llm_call`` traces already reach the store from the API
        proxy). Best-effort and non-blocking: callers fire-and-forget this, and
        it swallows every error so tracing can never stall or break the loop. If
        no trace context is active there is nothing to correlate, so it no-ops.
        """
        headers = self._trace_headers()
        if TRACE_HEADER not in headers:
            return
        try:
            client = await self._get_client()
            resp = await client.post(
                f"{self.mesh_url}/mesh/traces",
                json={
                    "agent_id": self.agent_id,
                    "event_type": event_type,
                    "detail": detail,
                    "status": status,
                    "error": error,
                    "duration_ms": duration_ms,
                    "meta": meta or {},
                },
                headers=headers,
                timeout=5,
            )
            # Surface a silently-dropped trace (auth/rate-limit/no-store) at
            # debug so a broken tracer is diagnosable without spamming logs or
            # ever raising — the response is otherwise discarded.
            if resp.status_code >= 400:
                logger.debug(
                    "record_trace(%s) rejected: HTTP %s", event_type, resp.status_code,
                )
            else:
                try:
                    if resp.json().get("recorded") is False:
                        logger.debug("record_trace(%s) not recorded by mesh", event_type)
                except Exception:
                    pass
        except Exception as e:
            logger.debug("record_trace(%s) failed (non-fatal): %s", event_type, e)

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
        _raise_with_body(response)
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
        _raise_with_body(response)
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
        _raise_with_body(response)
        return response.json()

    async def list_agents(self) -> dict:
        """List agents visible to this agent.

        Team-scoped agents see only their team's members (a solo
        worker's team-of-one resolves to itself plus the operator).
        The operator (no team) sees all registered agents so it can
        coordinate cross-team.
        """
        params: dict[str, str] = {}
        if self.team_name:
            params["team"] = self.team_name
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/agents",
            params=params,
        )
        _raise_with_body(response)
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
        _raise_with_body(response)
        return response.json()

    async def list_cron(self) -> list[dict]:
        """List cron jobs for this agent."""
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/cron",
            params={"agent_id": self.agent_id},
        )
        _raise_with_body(response)
        return response.json()

    async def update_cron(self, job_id: str, **kwargs) -> dict:
        """Update a cron job by ID."""
        client = await self._get_client()
        response = await client.put(
            f"{self.mesh_url}/mesh/cron/{job_id}",
            json=kwargs,
            headers=self._trace_headers(),
        )
        _raise_with_body(response)
        return response.json()

    async def remove_cron(self, job_id: str) -> dict:
        """Remove a cron job by ID."""
        client = await self._get_client()
        response = await client.delete(
            f"{self.mesh_url}/mesh/cron/{job_id}",
            headers=self._trace_headers(),
        )
        _raise_with_body(response)
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
        _raise_with_body(response)
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
        _raise_with_body(response)
        return response.json()

    async def get_agent_history(self, agent_id: str) -> dict:
        """Read another agent's daily logs (permission-checked on server)."""
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/agents/{agent_id}/history",
            params={"requesting_agent": self.agent_id},
        )
        _raise_with_body(response)
        return response.json()

    async def get_agent_profile(self, agent_id: str) -> dict:
        """Read another agent's public profile (permission-checked on server)."""
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/agents/{agent_id}/profile",
            params={"requesting_agent": self.agent_id},
        )
        _raise_with_body(response)
        return response.json()

    # === Per-agent standing goals (Team store, ratified #7 / C.3-b) ===

    async def get_my_goals(self) -> dict:
        """Read this agent's own standing goals record from the mesh.

        Returns ``{agent_id, goals: [...], set_by, updated_at}`` with
        ``goals: []`` when unset. Self-read is always allowed on the
        mesh side (goal delivery must never depend on ACL variance).
        The ``X-Agent-ID`` hint mirrors :meth:`introspect` — in dev/
        no-auth mode the mesh derives identity from it; with tokens
        configured the Bearer identity wins.
        """
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/agents/{self.agent_id}/goals",
            headers={"X-Agent-ID": self.agent_id},
        )
        _raise_with_body(response)
        return response.json()

    async def set_agent_goals(
        self, agent_id: str, goals: list[str], set_by: str = "operator",
    ) -> dict:
        """Replace a worker's standing goals (operator-gated on the mesh).

        Empty ``goals`` clears the record — same as :meth:`clear_agent_goals`.
        """
        client = await self._get_client()
        response = await client.put(
            f"{self.mesh_url}/mesh/agents/{agent_id}/goals",
            json={"goals": goals, "set_by": set_by},
            headers=self._trace_headers(),
        )
        _raise_with_body(response)
        return response.json()

    async def clear_agent_goals(self, agent_id: str) -> dict:
        """Clear a worker's standing goals (operator-gated on the mesh)."""
        client = await self._get_client()
        response = await client.delete(
            f"{self.mesh_url}/mesh/agents/{agent_id}/goals",
            headers=self._trace_headers(),
        )
        _raise_with_body(response)
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
        _raise_with_body(response)
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
        _raise_with_body(response)
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
        _raise_with_body(response)
        return response.json()

    async def vault_store(self, name: str, value: str) -> dict:
        """Store a credential in the mesh vault. Returns handle."""
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/vault/store",
            json={"agent_id": self.agent_id, "name": name, "value": value},
            headers=self._trace_headers(),
        )
        _raise_with_body(response)
        return response.json()

    async def vault_list(self) -> list[str]:
        """List credential names stored in the vault."""
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/vault/list",
            params={"agent_id": self.agent_id},
        )
        _raise_with_body(response)
        return response.json().get("credentials", [])

    async def vault_status(self, name: str) -> dict:
        """Check whether a credential exists."""
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/vault/status/{name}",
            params={"agent_id": self.agent_id},
        )
        _raise_with_body(response)
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
        _raise_with_body(response)
        return response.json().get("value")

    async def list_connector_tools(self) -> dict:
        """Sanitized tool schemas for every remote (http) MCP connector
        assigned to this agent — fetched from the mesh gateway at
        startup. Shape: ``{connector_name: {"tools": [...], "error"?}}``.
        """
        # Timeout sits ABOVE the gateway's per-connector DISCOVERY
        # deadline (20s, connectors discovered in parallel) so a slow
        # remote degrades to a per-connector error entry mesh-side
        # instead of this whole request timing out and stripping every
        # healthy connector's tools from the boot.
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/connectors/tools",
            params={"agent_id": self.agent_id},
            timeout=60,
        )
        _raise_with_body(response)
        return response.json().get("connectors", {})

    async def call_connector_tool(
        self, connector: str, tool: str, arguments: dict,
    ) -> dict:
        """Execute one remote-connector tool call through the mesh
        gateway. Returns the stdio-MCP result contract:
        ``{"result": text}`` or ``{"error": text}`` (+ optional
        ``truncated``). Auth lives mesh-side; nothing secret transits
        here."""
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/connectors/call",
            json={
                "agent_id": self.agent_id,
                "connector": connector,
                "tool": tool,
                "arguments": arguments,
            },
            headers=self._trace_headers(),
            timeout=90.0,  # gateway's 60s upstream cap + headroom
        )
        if response.status_code in (403, 404, 429, 502, 503):
            detail = ""
            try:
                detail = response.json().get("detail", "")
            except Exception:
                detail = response.text[:200]
            return {"error": f"Connector call failed: {detail}"}
        _raise_with_body(response)
        return response.json()

    async def introspect(self, section: str = "all") -> dict:
        """Query runtime state from the mesh (permissions, budget, fleet, etc.)."""
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/introspect",
            params={"section": section},
            headers={"X-Agent-ID": self.agent_id},
        )
        _raise_with_body(response)
        return response.json()

    # === Wallet (blockchain transactions via mesh signing service) ===

    async def wallet_get_address(self, chain: str) -> dict:
        """Get this agent's wallet address for a chain."""
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/wallet/address",
            params={"agent_id": self.agent_id, "chain": chain},
        )
        _raise_with_body(response)
        return response.json()

    async def wallet_get_balance(self, chain: str, token: str = "native") -> dict:
        """Get wallet balance for a token on a chain."""
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/wallet/balance",
            params={"agent_id": self.agent_id, "chain": chain, "token": token},
        )
        _raise_with_body(response)
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
        _raise_with_body(response)
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
        _raise_with_body(response)
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
        _raise_with_body(response)
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
        _raise_with_body(response)
        return response.json()

    # === Fleet Templates ===

    async def list_fleet_templates(self) -> dict:
        """List available fleet templates from the mesh."""
        response = await self._get_with_retry(f"{self.mesh_url}/mesh/fleet/templates")
        _raise_with_body(response)
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
        _raise_with_body(response)
        return response.json()

    async def install_skill(self, repo_url: str, ref: str = "") -> dict:
        """Install a SKILL.md skill pack from a git repo (operator-gated)."""
        client = await self._get_client()
        body: dict = {"repo_url": repo_url}
        if ref:
            body["ref"] = ref
        response = await client.post(
            f"{self.mesh_url}/mesh/skills/install",
            json=body,
            timeout=120,
            headers=self._trace_headers(),
        )
        _raise_with_body(response)
        return response.json()

    async def remove_skill(self, name: str) -> dict:
        """Remove an installed skill pack (operator-gated)."""
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/skills/remove",
            json={"name": name},
            timeout=30,
            headers=self._trace_headers(),
        )
        _raise_with_body(response)
        return response.json()

    async def list_my_skills(self) -> list[str]:
        """Skill-pack names assigned to THIS agent (effective: fleet ∪ per-agent).

        The mesh resolves the caller from the request's agent identity, so an
        agent only ever learns its own assignment. Used by skills_list /
        skill_view to scope discovery per agent.
        """
        client = await self._get_client()
        response = await client.get(
            f"{self.mesh_url}/mesh/skills/mine",
            timeout=10,
            headers=self._trace_headers(),
        )
        _raise_with_body(response)
        return response.json().get("skills", [])

    async def get_skill_assignments(self) -> dict:
        """Current fleet + per-agent skill assignment (operator-gated read)."""
        client = await self._get_client()
        response = await client.get(
            f"{self.mesh_url}/mesh/skills/assignments",
            timeout=10,
            headers=self._trace_headers(),
        )
        _raise_with_body(response)
        return response.json()

    async def assign_skills(self, agent_id: str, skills: list[str]) -> dict:
        """Replace an agent's per-agent skill allowlist (operator-gated)."""
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/skills/assign",
            json={"agent_id": agent_id, "skills": skills},
            timeout=30,
            headers=self._trace_headers(),
        )
        _raise_with_body(response)
        return response.json()

    async def set_fleet_skills(self, skills: list[str]) -> dict:
        """Replace the fleet-wide skill allowlist (operator-gated)."""
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/skills/fleet",
            json={"skills": skills},
            timeout=30,
            headers=self._trace_headers(),
        )
        _raise_with_body(response)
        return response.json()

    # === Browser (shared browser service via mesh proxy) ===

    # === Operator agent config management ===

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
        _raise_with_body(response)
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
        _raise_with_body(response)
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
        _raise_with_body(response)
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
        _raise_with_body(response)
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
        _raise_with_body(response)
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
        _raise_with_body(response)
        return response.json()

    async def list_peer_files(self, agent_id: str, path: str = ".",
                              recursive: bool = False) -> dict:
        """List a peer agent's /data files (operator-only).

        Backs the operator's ``list_peer_files`` tool. Reaches beyond
        ``artifacts/`` to the worker's full /data volume so the operator can
        locate a deliverable wherever it was written. Returns
        ``{agent_id, entries: [...], count}``. 403 if the caller isn't
        operator; 404 if the agent doesn't exist.
        """
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/agents/{agent_id}/files",
            params={"requesting_agent": self.agent_id, "path": path,
                    "recursive": str(bool(recursive)).lower()},
        )
        _raise_with_body(response)
        return response.json()

    async def read_peer_file(self, agent_id: str, path: str,
                             offset: int = 0, max_bytes: int = 0) -> dict:
        """Read a peer agent's /data file content (operator-only).

        Backs the operator's ``read_peer_file`` tool. Returns
        ``{agent_id, path, content, size, encoding, offset, next_offset,
        truncated}``. Pass the prior ``next_offset`` back as ``offset`` to
        page a large file. 403 if the caller isn't operator; 404 if the file
        is missing.
        """
        params: dict = {"requesting_agent": self.agent_id}
        if offset:
            params["offset"] = offset
        if max_bytes:
            params["max_bytes"] = max_bytes
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/agents/{agent_id}/files/{path}",
            params=params,
        )
        _raise_with_body(response)
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
        _raise_with_body(response)
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
        _raise_with_body(response)
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
        _raise_with_body(response)
        return response.json()

    # === Team management (mesh proxy endpoints) ===
    #
    # Every method hits the ``/mesh/teams/*`` route — PR 3 removed the

    async def list_teams(self) -> dict:
        """List all teams via mesh proxy."""
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/teams",
        )
        _raise_with_body(response)
        return response.json()

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
        _raise_with_body(response)
        return response.json()

    async def add_agent_to_team(self, team_name: str, agent_id: str) -> dict:
        """Add an agent to a team via mesh proxy."""
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/teams/{team_name}/members",
            json={"agent": agent_id},
            headers=self._trace_headers(),
        )
        _raise_with_body(response)
        return response.json()

    async def remove_agent_from_team(
        self, team_name: str, agent_id: str,
    ) -> dict:
        """Remove an agent from a team via mesh proxy."""
        client = await self._get_client()
        response = await client.delete(
            f"{self.mesh_url}/mesh/teams/{team_name}/members/{agent_id}",
            headers=self._trace_headers(),
        )
        _raise_with_body(response)
        return response.json()

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
        _raise_with_body(response)
        return response.json()

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
        _raise_with_body(response)
        return response.json()

    async def set_task_outcome(
        self,
        task_id: str,
        outcome: str,
        feedback: str = "",
    ) -> dict:
        """Record an operator outcome rating on a completed task.

        Routes through ``POST /mesh/tasks/{task_id}/outcome`` — the
        canonical mesh-tier endpoint that the operator's
        ``rate_delivery`` tool calls. Returns the mesh response dict
        (carries ``rework_task_id`` / ``rework_assignee`` when the
        outcome triggered a follow-up task spawn).
        """
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/tasks/{task_id}/outcome",
            json={"outcome": outcome, "feedback": feedback},
            headers=self._trace_headers(),
        )
        _raise_with_body(response)
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
        _raise_with_body(response)
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
        _raise_with_body(response)
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
        _raise_with_body(response)
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
            _raise_with_body(response)
            raise
        # Pass through structured error envelopes (e.g. operator kill switch
        # returning ``BROWSER_DOWNLOADS_DISABLED``) so the calling tool can
        # surface the reason to the LLM without an exception. FastAPI wraps
        # HTTPException(detail=...) as ``{"detail": <envelope>}``, so unwrap.
        if response.status_code >= 400 and isinstance(data, dict):
            envelope = data.get("detail") if "detail" in data else data
            if (
                isinstance(envelope, dict)
                and envelope.get("success") is False
            ):
                return envelope
        _raise_with_body(response)
        if not isinstance(data, dict):
            raise ValueError(
                f"Unexpected browser_download response: {type(data).__name__}",
            )
        return data

    # ── Orchestration tasks ─────────────────────────────────────

    async def create_task(
        self,
        *,
        assignee: str,
        title: str,
        description: str | None = None,
        team_id: str | None = None,
        parent_task_id: str | None = None,
        priority: int = 0,
        dependencies: list[str] | None = None,
        artifact_refs: list[str] | None = None,
        origin: "MessageOrigin | None" = None,
        thinking: str | None = None,
    ) -> dict:
        """Create a durable task. Returns the new task record.

        ``origin`` propagates the cross-agent provenance header so the
        receiving agent's lane worker can attribute task completion back
        to the originating channel/user. Without this, ``hand_off`` v2
        loses the origin a sibling ``wake_agent`` call still carries.
        ``thinking`` (B4) pins a per-task reasoning depth
        (off/low/medium/high) for the assignee while executing this task.
        """
        client = await self._get_client()
        body: dict = {
            "assignee": assignee,
            "title": title,
            "priority": priority,
        }
        if description is not None:
            body["description"] = description
        if team_id is not None:
            body["team_id"] = team_id
        if parent_task_id is not None:
            body["parent_task_id"] = parent_task_id
        if dependencies is not None:
            body["dependencies"] = dependencies
        if artifact_refs is not None:
            body["artifact_refs"] = artifact_refs
        if thinking is not None:
            body["thinking"] = thinking
        headers = self._trace_headers()
        if origin is not None:
            headers.update(origin_header(origin))
        response = await client.post(
            f"{self.mesh_url}/mesh/tasks",
            json=body,
            headers=headers,
        )
        _raise_with_body(response)
        return response.json()

    async def get_task(self, task_id: str) -> dict | None:
        """Read a task by id. Returns None on 404."""
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/tasks/{task_id}",
        )
        if response.status_code == 404:
            return None
        _raise_with_body(response)
        return response.json()

    async def get_workflow_snapshot(
        self, root_task_id: str,
    ) -> dict | None:
        """Operator-tier read of a workflow chain. Returns ``None`` on 404."""
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/tasks/workflow/{root_task_id}",
        )
        if response.status_code == 404:
            return None
        _raise_with_body(response)
        return response.json()

    async def get_task_run(self, task_id: str) -> dict | None:
        """Operator-tier per-task execution diagnostics. ``None`` on 404."""
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/tasks/{task_id}/run",
        )
        if response.status_code == 404:
            return None
        _raise_with_body(response)
        return response.json()

    async def read_user_notifications(
        self, hours: float = 24, limit: int = 50,
    ) -> dict:
        """Operator-tier read of recent agent→user notifications.

        Backs the operator's ``read_user_notifications`` tool. Returns
        ``{notifications: [{from, message, ts}, ...]}`` with RAW messages
        — the tool sanitizes each at its boundary. 403 if the caller
        isn't operator/internal.
        """
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/user-notifications"
            f"?hours={hours}&limit={int(limit)}",
        )
        _raise_with_body(response)
        return response.json()

    async def list_inbox_events(self) -> list[dict]:
        """Back-edge task events addressed to this agent (Team Threads).

        Reads ``GET /mesh/agents/{self}/task-events`` — the thread-store
        replacement for the old blackboard ``inbox/{agent}/task_event/``
        prefix read. The serving windows (7-day actionable / 24h
        informational) are applied mesh-side. The ``X-Agent-ID`` hint
        mirrors :meth:`get_my_goals` for tokenless dev mode.
        """
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/agents/{self.agent_id}/task-events",
            headers={"X-Agent-ID": self.agent_id},
        )
        _raise_with_body(response)
        data = response.json()
        return data.get("events", []) if isinstance(data, dict) else []

    async def list_task_inbox(self, assignee: str | None = None) -> list[dict]:
        """List tasks assigned to ``assignee`` (defaults to self)."""
        target = assignee or self.agent_id
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/tasks/inbox/{target}",
        )
        _raise_with_body(response)
        data = response.json()
        return data.get("tasks", []) if isinstance(data, dict) else []

    async def list_team_tasks(self, team_id: str) -> list[dict]:
        """List tasks scoped to ``team_id``."""
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/tasks/team/{team_id}",
        )
        _raise_with_body(response)
        data = response.json()
        return data.get("tasks", []) if isinstance(data, dict) else []

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
        _raise_with_body(response)
        return response.json()

    async def report_auth_failure(
        self, *, provider: str, model: str, http_status: int,
    ) -> dict:
        """Self-report a credential failure so the mesh can quarantine.

        Fire-and-forget from the agent's perspective: never raises out of
        the loop. Returns ``{"recorded": False, "error": ...}`` on failure
        so the caller can log without a try/except wrapper.
        """
        try:
            client = await self._get_client()
            response = await client.post(
                f"{self.mesh_url}/mesh/agents/{self.agent_id}/auth-failure",
                json={
                    "provider": provider,
                    "model": model,
                    "http_status": http_status,
                },
                headers=self._trace_headers(),
            )
            _raise_with_body(response)
            return response.json()
        except Exception as e:
            logger.warning("auth-failure self-report failed: %s", e)
            return {"recorded": False, "error": str(e)}

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
        _raise_with_body(response)
        return response.json()

    async def cancel_task(self, task_id: str, reason: str = "") -> dict:
        """Cancel a task."""
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/tasks/{task_id}/cancel",
            json={"reason": reason},
            headers=self._trace_headers(),
        )
        _raise_with_body(response)
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
        _raise_with_body(response)
        return response.json()

    # ── Operator product surface (Task 7) ────────────────────────

    async def team_status(self, team_id: str) -> dict:
        """Per-team status counts + recent blockers/completions."""
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/teams/{team_id}/status",
        )
        _raise_with_body(response)
        return response.json()

    async def all_teams_status(self) -> dict:
        """Status rollup across every visible team."""
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/teams/status",
        )
        _raise_with_body(response)
        return response.json()

    async def agent_queue(self, agent_id: str, limit: int = 10) -> dict:
        """Recent tasks for an agent grouped by status."""
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/agents/{agent_id}/queue",
            params={"limit": limit},
        )
        _raise_with_body(response)
        return response.json()

    async def team_outputs(self, team_id: str, since: str = "") -> dict:
        """Completed task artifacts for a team in a time window."""
        params = {"since": since} if since else None
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/teams/{team_id}/outputs",
            params=params,
        )
        _raise_with_body(response)
        return response.json()

    async def update_team_brief(
        self, team_name: str, section: str, content: str,
    ) -> dict:
        """Section-scoped TEAM.md update + push to running members (P2)."""
        client = await self._get_client()
        response = await client.put(
            f"{self.mesh_url}/mesh/teams/{team_name}/brief",
            json={"section": section, "content": content},
            headers=self._trace_headers(),
        )
        _raise_with_body(response)
        return response.json()

    # ---- Team Drive reviews (Phase-2 unit 1) ----------------------------
    async def submit_drive_review(self, branch: str, title: str, summary: str = "") -> dict:
        """Submit a pushed Team Drive branch for review-before-integrate."""
        if not self.team_name:
            raise RuntimeError("No team drive available: agent has no team scope")
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/teams/{self.team_name}/drive/reviews",
            json={"branch": branch, "title": title, "summary": summary},
            headers=self._trace_headers(),
        )
        _raise_with_body(response)
        return response.json()

    async def list_drive_reviews(self, status: str = "") -> dict:
        """List this team's Team Drive reviews (optionally by status)."""
        if not self.team_name:
            raise RuntimeError("No team drive available: agent has no team scope")
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/teams/{self.team_name}/drive/reviews",
            params={"status": status} if status else None,
        )
        _raise_with_body(response)
        return response.json()

    async def record_drive_verdict(self, review_id: str, verdict: str, note: str = "") -> dict:
        """Record this agent's advisory approve/reject verdict on a review.

        Server-enforced lead-only (plan §8 #13): the mesh 403s any
        caller that isn't this team's ``lead_agent_id``. Purely
        advisory — has zero effect on the merge/reject gates.
        """
        if not self.team_name:
            raise RuntimeError("No team drive available: agent has no team scope")
        client = await self._get_client()
        body: dict = {"verdict": verdict}
        if note:
            body["note"] = note
        response = await client.post(
            f"{self.mesh_url}/mesh/teams/{self.team_name}/drive/reviews/{review_id}/verdict",
            json=body,
            headers=self._trace_headers(),
        )
        _raise_with_body(response)
        return response.json()

    # ---- Held-action lead recommendations (plan §8 #19) -----------------
    async def recommend_pending_action(
        self, nonce: str, recommendation: str, note: str = "",
    ) -> dict:
        """Record this agent's advisory approve/reject recommendation on
        a teammate's held pending action.

        Server-enforced lead-only: the mesh 403s any caller that isn't
        the team lead of the agent who proposed the held action, and
        409s a nonce whose proposer has no team or whose team has no
        lead. Purely advisory -- has ZERO effect on whether the action is
        confirmed, cancelled, or executed.
        """
        client = await self._get_client()
        body: dict = {"recommendation": recommendation}
        if note:
            body["note"] = note
        response = await client.post(
            f"{self.mesh_url}/mesh/pending/{nonce}/recommend",
            json=body,
            headers=self._trace_headers(),
        )
        _raise_with_body(response)
        return response.json()

    async def commit_drive_artifact(
        self, team: str, *, name: str, content: str, kind: str = "artifact",
        encoding: str = "utf8",
    ) -> dict:
        """Commit a deliverable/handoff-data file to a team drive's main.

        Direct-commit registration (Phase-2 unit 4): the mesh records THIS
        agent as the commit author under ``{handoffs|artifacts}/{me}/{name}``
        and returns ``{committed, ref: "drive://{team}/{path}@{sha}", ...}``.
        ``team`` is the SENDER's team scope (the drive that holds the
        payload). Raises on any non-2xx so callers can build a failure
        envelope from the mesh's descriptive detail.
        """
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/teams/{team}/drive/artifacts",
            json={"kind": kind, "name": name, "content": content, "encoding": encoding},
            headers=self._trace_headers(),
        )
        _raise_with_body(response)
        return response.json()

    async def read_drive_file(
        self, team: str, path: str, *, ref: str = "main",
    ) -> dict:
        """Read one file from a team drive without a clone.

        Returns ``{path, ref, content, encoding, size}`` (content is
        base64 when the file is binary). RAW — the caller sanitizes before
        surfacing into an LLM prompt. Returns ``None`` on 404.
        """
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/teams/{team}/drive/file",
            params={"path": path, "ref": ref},
        )
        if response.status_code == 404:
            return None
        _raise_with_body(response)
        return response.json()

    async def team_summary(self, team_id: str, hours: float = 0) -> dict:
        """Synthesized status summary for a team.

        ``hours`` > 0 also returns ``outcomes_window`` — per-outcome
        rating counts within the trailing window (P2).
        """
        url = f"{self.mesh_url}/mesh/teams/{team_id}/summary"
        if hours and hours > 0:
            url += f"?hours={float(hours)}"
        response = await self._get_with_retry(url)
        _raise_with_body(response)
        return response.json()

    # ---- Work summaries (PR-A) -----------------------------------------
    async def create_work_summary(
        self,
        *,
        scope_kind: str,
        scope_id: str,
        period_start: float,
        period_end: float,
        narrative_md: str,
        metrics: dict,
        recommendations: list[str] | None = None,
    ) -> dict:
        """Operator-only: create a new work summary record."""
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/work-summaries",
            json={
                "scope_kind": scope_kind,
                "scope_id": scope_id,
                "period_start": period_start,
                "period_end": period_end,
                "narrative_md": narrative_md,
                "metrics": metrics,
                "recommendations": recommendations or [],
            },
            headers=self._trace_headers(),
        )
        _raise_with_body(response)
        return response.json()

    async def list_work_summaries(
        self,
        *,
        scope_kind: str | None = None,
        scope_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict:
        """List recent summaries, scope-filtered."""
        params: dict = {"limit": limit, "offset": offset}
        if scope_kind:
            params["scope_kind"] = scope_kind
        if scope_id:
            params["scope_id"] = scope_id
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/work-summaries", params=params,
        )
        _raise_with_body(response)
        return response.json()

    async def get_work_summary(self, summary_id: str) -> dict:
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/work-summaries/{summary_id}",
        )
        _raise_with_body(response)
        return response.json()

    async def rate_work_summary(
        self,
        summary_id: str,
        rating: str,
        feedback: str | None = None,
    ) -> dict:
        """Operator-only: rate a summary."""
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/work-summaries/{summary_id}/rating",
            json={"rating": rating, "feedback": feedback},
            headers=self._trace_headers(),
        )
        _raise_with_body(response)
        return response.json()

    async def set_team_budget(
        self, name: str, daily_usd: float | None, monthly_usd: float | None,
    ) -> dict:
        """Set a team's budget envelope. None/0 = unlimited (plan B4)."""
        client = await self._get_client()
        response = await client.put(
            f"{self.mesh_url}/mesh/teams/{name}/budget",
            json={"daily_usd": daily_usd, "monthly_usd": monthly_usd},
            headers=self._trace_headers(),
        )
        _raise_with_body(response)
        return response.json()

    async def archive_team(self, name: str) -> dict:
        """Archive a team."""
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/teams/{name}/archive",
            headers=self._trace_headers(),
        )
        _raise_with_body(response)
        return response.json()

    async def unarchive_team(self, name: str) -> dict:
        """Unarchive (restore) a previously archived team."""
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/teams/{name}/unarchive",
            headers=self._trace_headers(),
        )
        _raise_with_body(response)
        return response.json()

    async def archive_agent(self, agent_id: str) -> dict:
        """Archive an agent."""
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/agents/{agent_id}/archive",
            headers=self._trace_headers(),
        )
        _raise_with_body(response)
        return response.json()

    async def offboard_agent(self, agent_id: str) -> dict:
        """Offboard an agent: handover + Team Drive snapshot, then archive
        (plan §8 #15)."""
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/agents/{agent_id}/offboard",
            headers=self._trace_headers(),
        )
        _raise_with_body(response)
        return response.json()

    async def propose_delete_team(self, name: str) -> dict:
        """Propose deletion of an archived team. Returns nonce for human confirm."""
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/teams/{name}/propose-delete",
            headers=self._trace_headers(),
        )
        _raise_with_body(response)
        return response.json()

    async def propose_delete_agent(self, agent_id: str) -> dict:
        """Propose deletion of an archived agent. Returns nonce for human confirm."""
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/agents/{agent_id}/propose-delete",
            headers=self._trace_headers(),
        )
        _raise_with_body(response)
        return response.json()

    async def list_task_events(self, task_id: str) -> list[dict]:
        """Audit history for a task."""
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/tasks/{task_id}/events",
        )
        _raise_with_body(response)
        data = response.json()
        return data.get("events", []) if isinstance(data, dict) else []

    # ── Operator metrics ─────────────────────────────────────────

    async def get_system_metrics(self) -> dict:
        """Get fleet-wide pre-computed metrics from the mesh."""
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/system/metrics",
        )
        _raise_with_body(response)
        return response.json()

    async def get_agent_metrics(self, agent_id: str) -> dict:
        """Get per-agent pre-computed metrics from the mesh."""
        response = await self._get_with_retry(
            f"{self.mesh_url}/mesh/agents/{agent_id}/metrics",
        )
        _raise_with_body(response)
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
        _raise_with_body(response)
        return response.json()


