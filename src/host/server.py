"""Mesh HTTP server -- the central API for fleet coordination.

Provides endpoints for:
  - Blackboard CRUD (shared state)
  - Pub/Sub (event signals)
  - API proxy (agents call external services through mesh)
  - Agent registration
  - System messaging (mesh-to-agent)
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import re
import time
from collections import defaultdict
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import StreamingResponse

from src.host.credentials import is_system_credential
from src.shared.types import (
    AGENT_ID_RE_PATTERN,
    RESERVED_AGENT_IDS,
    AgentMessage,
    APIProxyRequest,
    APIProxyResponse,
    BlackboardClaimRequest,
    BlackboardWatchRequest,
    MeshEvent,
    NotifyRequest,
)
from src.shared.utils import sanitize_for_prompt, setup_logging

logger = setup_logging("host.server")

_MAX_SYSTEM_PROMPT = 10_000  # chars — caps agent-supplied system prompt to limit context cost
_MAX_BB_KEY_LEN = 512  # chars — prevents abusive key lengths in blackboard
_MAX_BB_VALUE_BYTES = 262_144  # 256 KB — bounds per-key storage to keep SQLite WAL manageable


def _extract_prompt_preview(params: dict, max_len: int = 500) -> str:
    """Extract the last user message content as a short preview string."""
    for msg in reversed(params.get("messages", [])):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content[:max_len]
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return (part.get("text") or "")[:max_len]
            break
    return ""


if TYPE_CHECKING:
    from src.dashboard.events import EventBus
    from src.host.api_keys import ApiKeyManager
    from src.host.costs import CostTracker
    from src.host.credentials import CredentialVault
    from src.host.cron import CronScheduler
    from src.host.health import HealthMonitor
    from src.host.lanes import LaneManager
    from src.host.mesh import Blackboard, MessageRouter, PubSub
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
    auth_tokens: dict[str, str] | None = None,
    trace_store: TraceStore | None = None,
    event_bus: EventBus | None = None,
    health_monitor: HealthMonitor | None = None,
    cost_tracker: CostTracker | None = None,
    notify_fn: Callable[[str, str], Coroutine] | None = None,
    agent_projects: dict[str, str] | None = None,
    lane_manager: LaneManager | None = None,
    dispatch_loop: asyncio.AbstractEventLoop | None = None,
    wallet_service_ref: list | None = None,
    api_key_manager: ApiKeyManager | None = None,
    cfg: dict | None = None,
) -> FastAPI:
    """Create the FastAPI application for the mesh host process."""
    app = FastAPI(title="OpenLegion Mesh")
    # Exposed for external callers (dashboard, health monitor) to clean up
    # agent state when agents are removed.
    app.cleanup_agent = lambda agent_id: None  # replaced below

    _auth_tokens = auth_tokens if auth_tokens is not None else {}
    _agent_projects = agent_projects if agent_projects is not None else {}

    # -- Input validation helpers ------------------------------------------------
    _AGENT_ID_RE = re.compile(AGENT_ID_RE_PATTERN)

    def _validate_agent_id(agent_id: str) -> str:
        if not agent_id or not _AGENT_ID_RE.match(agent_id):
            raise HTTPException(400, "Invalid agent_id: must be 1-64 alphanumeric/hyphen/underscore chars")
        if agent_id in RESERVED_AGENT_IDS:
            raise HTTPException(400, f"Agent ID '{agent_id}' is reserved for internal use")
        return agent_id

    def _validate_port(port: int) -> int:
        if not isinstance(port, int) or port < 1024 or port > 65535:
            raise HTTPException(400, f"Invalid port: must be 1024-65535, got {port}")
        return port

    # -- Per-agent rate limiting --------------------------------------------------
    # Each bucket is keyed by (endpoint_name, agent_id).
    _rate_ts: dict[str, list[float]] = defaultdict(list)
    _rate_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    _RATE_LIMITS: dict[str, tuple[int, int]] = {
        # (max_requests, window_seconds)
        "api_proxy": (30, 60),
        "vault_resolve": (5, 60),
        "vault_store": (10, 3600),
        "blackboard_read": (200, 60),
        "blackboard_write": (100, 60),
        "publish": (200, 60),
        "notify": (10, 60),
        "cron_create": (10, 3600),
        "spawn": (5, 3600),
        "wallet_read": (120, 60),
        "wallet_transfer": (10, 3600),
        "wallet_execute": (10, 3600),
        "image_gen": (10, 60),
        "agent_profile": (30, 60),
    }

    async def _check_rate_limit(endpoint: str, agent_id: str) -> None:
        """Enforce per-agent rate limit. Raises 429 if exceeded."""
        limit, window = _RATE_LIMITS.get(endpoint, (100, 60))
        bucket_key = f"{endpoint}:{agent_id}"
        async with _rate_locks[bucket_key]:
            now = time.time()
            ts_list = _rate_ts[bucket_key]
            ts_list[:] = [t for t in ts_list if now - t < window]
            if len(ts_list) >= limit:
                raise HTTPException(429, f"Rate limit exceeded for {endpoint}")
            ts_list.append(now)

    def _notify_watchers_batch(watcher_ids: list[str], msg: str) -> None:
        """Batch-notify watchers via a single cross-thread call."""
        if not watcher_ids or lane_manager is None or dispatch_loop is None:
            return
        msg = sanitize_for_prompt(msg)

        async def _do_notify():
            results = await asyncio.gather(
                *(lane_manager.enqueue(wid, msg, mode="steer") for wid in watcher_ids),
                return_exceptions=True,
            )
            for wid, result in zip(watcher_ids, results, strict=True):
                if isinstance(result, Exception):
                    logger.warning("Watch notification to %s failed: %s", wid, result)

        try:
            asyncio.run_coroutine_threadsafe(_do_notify(), dispatch_loop)
        except Exception as e:
            logger.warning("Batch watch notification failed: %s", e)

    def _cleanup_agent(agent_id: str) -> None:
        """Clean up all per-agent state when an agent is deregistered.

        Covers: rate-limit buckets, credential vault locks, blackboard
        data, pub/sub subscriptions, lane workers, cron jobs, cost
        records, trace records, and wallet records.
        """
        stale = [k for k in _rate_ts if k.endswith(f":{agent_id}")]
        for k in stale:
            del _rate_ts[k]
        if credential_vault is not None:
            credential_vault.cleanup_agent(agent_id)
        blackboard.cleanup_agent_data(agent_id)
        if pubsub is not None:
            pubsub.unsubscribe_agent(agent_id)
        if lane_manager is not None:
            lane_manager.remove_lane(agent_id)
        if cron_scheduler is not None:
            cron_scheduler.remove_agent_jobs(agent_id)
        if cost_tracker is not None:
            try:
                cost_tracker.cleanup_agent(agent_id)
            except Exception as e:
                logger.warning("Cost cleanup for '%s' failed: %s", agent_id, e)
        if trace_store is not None:
            try:
                trace_store.cleanup_agent(agent_id)
            except Exception as e:
                logger.warning("Trace cleanup for '%s' failed: %s", agent_id, e)
        _ws_ref = wallet_service_ref or [None]
        wallet_service = _ws_ref[0]
        if wallet_service is not None:
            try:
                wallet_service.cleanup_agent(agent_id)
            except Exception as e:
                logger.warning("Wallet cleanup for '%s' failed: %s", agent_id, e)

    app.cleanup_agent = _cleanup_agent  # type: ignore[attr-defined]

    def _extract_verified_agent_id(request: Request) -> str:
        """Extract and verify agent identity from a Bearer token.

        Derives the agent_id from the token itself, preventing
        identity spoofing via headers or query parameters.

        Returns 'unknown' when auth is not configured (dev/test mode).
        """
        if not _auth_tokens:
            # Auth not configured (dev/test mode) — fall back to header hint
            return request.headers.get("X-Agent-ID", "unknown")
        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(401, "Missing authentication token")
        token = auth_header[7:]
        for aid, expected in _auth_tokens.items():
            if hmac.compare_digest(token, expected):
                return aid
        raise HTTPException(401, "Invalid authentication token")

    def _resolve_agent_id(agent_id: str, request: Request) -> str:
        """Verified agent_id when auth active, else trust caller.

        When auth tokens are configured, derives the true agent identity
        from the Bearer token — ignoring the caller-supplied agent_id to
        prevent spoofing.  In dev/test mode (no tokens), trusts the caller.
        """
        if _auth_tokens:
            return _extract_verified_agent_id(request)
        return agent_id

    def _require_any_auth(request: Request) -> None:
        """Require any valid auth token (identity-agnostic).

        Used for endpoints that should be restricted in production but
        don't need a specific agent identity (traces, model-health, etc.).
        No-op in dev/test mode (no auth tokens configured).
        Accepts x-mesh-internal header for localhost callers (health checks).
        """
        if not _auth_tokens:
            return
        if request.headers.get("x-mesh-internal"):
            # Only accept from localhost (Caddy or internal callers)
            client_host = request.client.host if request.client else ""
            try:
                import ipaddress
                if ipaddress.ip_address(client_host).is_loopback:
                    return
            except (ValueError, AttributeError):
                pass  # Not a valid IP — fall through to Bearer token check
        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(401, "Missing authentication token")
        token = auth_header[7:]
        for expected in _auth_tokens.values():
            if hmac.compare_digest(token, expected):
                return
        raise HTTPException(401, "Invalid authentication token")

    # === System Messaging (mesh → agent) ===

    @app.post("/mesh/message")
    async def send_message(msg: AgentMessage, request: Request) -> dict:
        """Route a message to an agent via the mesh router."""
        msg.from_agent = _resolve_agent_id(msg.from_agent, request)
        if not permissions.can_message(msg.from_agent, msg.to):
            raise HTTPException(403, f"Agent {msg.from_agent} cannot message {msg.to}")
        return await router.route(msg)

    # === Blackboard ===
    # NOTE: list route must be defined BEFORE the {key:path} route to avoid shadowing

    @app.get("/mesh/blackboard/")
    async def list_blackboard(prefix: str, agent_id: str, request: Request) -> list[dict]:
        """List blackboard entries by prefix."""
        agent_id = _resolve_agent_id(agent_id, request)
        await _check_rate_limit("blackboard_read", agent_id)
        if not permissions.can_read_blackboard(agent_id, prefix):
            raise HTTPException(403, f"Agent {agent_id} cannot read {prefix}")
        entries = blackboard.list_by_prefix(prefix)
        return [e.model_dump(mode="json") for e in entries]

    @app.get("/mesh/blackboard/{key:path}")
    async def read_blackboard(key: str, agent_id: str, request: Request) -> dict:
        """Read a blackboard entry. Agent must have read permission."""
        agent_id = _resolve_agent_id(agent_id, request)
        await _check_rate_limit("blackboard_read", agent_id)
        if not permissions.can_read_blackboard(agent_id, key):
            raise HTTPException(403, f"Agent {agent_id} cannot read {key}")
        entry = blackboard.read(key)
        if not entry:
            raise HTTPException(404, f"Key not found: {key}")
        return entry.model_dump(mode="json")

    @app.put("/mesh/blackboard/{key:path}")
    async def write_blackboard(
        key: str, agent_id: str, value: dict, request: Request,
        ttl: int | None = None,
    ) -> dict:
        """Write to blackboard. Agent must have write permission."""
        agent_id = _resolve_agent_id(agent_id, request)
        await _check_rate_limit("blackboard_write", agent_id)
        if len(key) > _MAX_BB_KEY_LEN:
            raise HTTPException(400, f"Key too long ({len(key)} chars, max {_MAX_BB_KEY_LEN})")
        value_size = len(json.dumps(value, default=str))
        if value_size > _MAX_BB_VALUE_BYTES:
            raise HTTPException(413, f"Value too large ({value_size} bytes, max {_MAX_BB_VALUE_BYTES})")
        if ttl is not None and ttl <= 0:
            raise HTTPException(400, "TTL must be positive")
        if not permissions.can_write_blackboard(agent_id, key):
            raise HTTPException(403, f"Agent {agent_id} cannot write {key}")
        entry = blackboard.write(key, value, written_by=agent_id, ttl=ttl)
        if trace_store:
            req_trace_id = request.headers.get("x-trace-id")
            if req_trace_id:
                trace_store.record(
                    trace_id=req_trace_id, source="mesh.blackboard", agent=agent_id,
                    event_type="blackboard_write", detail=key,
                )
        # Notify watchers via steer (batched into a single cross-thread call)
        watchers = blackboard.get_watchers_for_key(key, exclude=agent_id)
        if watchers:
            notify_msg = (
                f"[Blackboard: {key}] updated by {agent_id}, v{entry.version}"
            )
            _notify_watchers_batch(watchers, notify_msg)
        return entry.model_dump(mode="json")

    @app.delete("/mesh/blackboard/{key:path}")
    async def delete_blackboard_entry(key: str, agent_id: str, request: Request) -> dict:
        """Delete a blackboard entry. Agent must have write permission."""
        agent_id = _resolve_agent_id(agent_id, request)
        await _check_rate_limit("blackboard_write", agent_id)
        if not permissions.can_write_blackboard(agent_id, key):
            raise HTTPException(403, f"Agent {agent_id} cannot write {key}")
        # Protect history namespace (including project-scoped keys)
        bare = key.split("/", 2)[2] if key.startswith("projects/") and key.count("/") >= 2 else key
        if bare.startswith("history/"):
            raise HTTPException(400, "Cannot delete from history namespace")
        try:
            blackboard.delete(key, deleted_by=agent_id)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"deleted": True, "key": key}

    @app.post("/mesh/blackboard/watch")
    async def watch_blackboard(data: BlackboardWatchRequest, request: Request) -> dict:
        """Register a glob pattern watch on blackboard keys."""
        agent_id = _resolve_agent_id(data.agent_id, request)
        pattern = data.pattern
        if not permissions.can_read_blackboard(agent_id, pattern):
            raise HTTPException(403, f"Agent {agent_id} cannot read pattern '{pattern}'")
        blackboard.add_watch(agent_id, pattern)
        return {"watching": True, "pattern": pattern}

    @app.post("/mesh/blackboard/claim")
    async def claim_blackboard(body: BlackboardClaimRequest, request: Request) -> dict:
        """Atomic compare-and-swap write. Returns 409 on version mismatch."""
        agent_id = _resolve_agent_id(body.agent_id, request)
        key = body.key
        await _check_rate_limit("blackboard_write", agent_id)
        if len(key) > _MAX_BB_KEY_LEN:
            raise HTTPException(400, f"Key too long ({len(key)} chars, max {_MAX_BB_KEY_LEN})")
        value_size = len(json.dumps(body.value, default=str))
        if value_size > _MAX_BB_VALUE_BYTES:
            raise HTTPException(413, f"Value too large ({value_size} bytes, max {_MAX_BB_VALUE_BYTES})")
        if not permissions.can_write_blackboard(agent_id, key):
            raise HTTPException(403, f"Agent {agent_id} cannot write {key}")
        expected_version = body.expected_version
        value = body.value
        entry = blackboard.write_if_version(
            key, value, written_by=agent_id, expected_version=expected_version,
        )
        if entry is None:
            raise HTTPException(409, f"Version conflict on key '{key}'")
        if trace_store:
            req_trace_id = request.headers.get("x-trace-id")
            if req_trace_id:
                trace_store.record(
                    trace_id=req_trace_id, source="mesh.blackboard", agent=agent_id,
                    event_type="blackboard_claim", detail=key,
                )
        # Notify watchers (CAS writes are still writes, batched into single call)
        watchers = blackboard.get_watchers_for_key(key, exclude=agent_id)
        if watchers:
            notify_msg = (
                f"[Blackboard: {key}] claimed by {agent_id}, v{entry.version}"
            )
            _notify_watchers_batch(watchers, notify_msg)
        return entry.model_dump(mode="json")

    # === Pub/Sub ===

    @app.post("/mesh/publish")
    async def publish_event(event: MeshEvent, request: Request) -> dict:
        """Publish an event to a topic."""
        event.source = _resolve_agent_id(event.source, request)
        await _check_rate_limit("publish", event.source)

        # Enforce project isolation: topic must match the publisher's project prefix
        source_project = _agent_projects.get(event.source)
        if source_project:
            expected_prefix = f"projects/{source_project}/"
            if not event.topic.startswith(expected_prefix):
                raise HTTPException(
                    403,
                    f"Agent {event.source} (project={source_project}) cannot publish to topic '{event.topic}'"
                )

        if not permissions.can_publish(event.source, event.topic):
            raise HTTPException(403, f"Agent {event.source} cannot publish to {event.topic}")
        if trace_store:
            req_trace_id = request.headers.get("x-trace-id")
            if req_trace_id:
                trace_store.record(
                    trace_id=req_trace_id, source="mesh.pubsub", agent=event.source,
                    event_type="pubsub_publish", detail=event.topic,
                )
        subscribers = pubsub.get_subscribers(event.topic)
        if subscribers:
            # Prefer steer delivery for real-time reactivity (batched into single call)
            if lane_manager is not None and dispatch_loop is not None:
                formatted_msg = (
                    f"[Event: {event.topic}] from {event.source}: "
                    f"{json.dumps(event.payload, default=str)[:500]}"
                )
                _notify_watchers_batch(subscribers, formatted_msg)
            else:
                await asyncio.gather(*(
                    router.route(AgentMessage(
                        from_agent="mesh",
                        to=agent_id,
                        type="event",
                        payload=event.model_dump(mode="json"),
                    ))
                    for agent_id in subscribers
                ), return_exceptions=True)
        return {"subscribers_notified": len(subscribers)}

    @app.post("/mesh/subscribe")
    async def subscribe(topic: str, agent_id: str, request: Request) -> dict:
        """Subscribe an agent to an event topic."""
        agent_id = _resolve_agent_id(agent_id, request)

        # Enforce project isolation: topic must match the subscriber's project prefix
        sub_project = _agent_projects.get(agent_id)
        if sub_project:
            expected_prefix = f"projects/{sub_project}/"
            if not topic.startswith(expected_prefix):
                raise HTTPException(
                    403,
                    f"Agent {agent_id} (project={sub_project}) cannot subscribe to topic '{topic}'"
                )

        if not permissions.can_subscribe(agent_id, topic):
            raise HTTPException(403, f"Agent {agent_id} cannot subscribe to {topic}")
        pubsub.subscribe(topic, agent_id)
        return {"subscribed": True}

    # === API Proxy ===

    @app.post("/mesh/api", response_model=APIProxyResponse)
    async def proxy_api_call(request: Request, api_request: APIProxyRequest, agent_id: str) -> APIProxyResponse:
        """Proxy external API calls. Agent never sees credentials."""
        agent_id = _resolve_agent_id(agent_id, request)
        await _check_rate_limit("api_proxy", agent_id)
        if api_request.service in _RATE_LIMITS:
            await _check_rate_limit(api_request.service, agent_id)
        if not permissions.can_use_api(agent_id, api_request.service):
            raise HTTPException(403, f"Agent {agent_id} cannot access {api_request.service}")
        if credential_vault is None:
            return APIProxyResponse(success=False, error="No credential vault configured")

        req_trace_id = request.headers.get("x-trace-id")
        prompt_preview = _extract_prompt_preview(api_request.params)
        t0 = time.time()
        result = await credential_vault.execute_api_call(api_request, agent_id=agent_id)
        duration_ms = int((time.time() - t0) * 1000)
        response_preview = ""
        if result.success and result.data:
            resp_content = result.data.get("content", "")
            if isinstance(resp_content, str):
                response_preview = resp_content[:500]
            elif isinstance(resp_content, list):
                for block in resp_content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        response_preview = (block.get("text") or "")[:500]
                        break
        if req_trace_id and trace_store:
            trace_meta = {
                "service": api_request.service,
                "action": api_request.action,
            }
            if prompt_preview:
                trace_meta["prompt_preview"] = prompt_preview
            if response_preview:
                trace_meta["response_preview"] = response_preview
            trace_status = "ok"
            trace_error = ""
            if result.success and result.data:
                trace_meta["model"] = result.data.get("model", "")
                trace_meta["tokens_used"] = result.data.get("tokens_used", 0)
                trace_meta["input_tokens"] = result.data.get("input_tokens", 0)
                trace_meta["output_tokens"] = result.data.get("output_tokens", 0)
            elif not result.success:
                trace_status = "error"
                trace_error = result.error or "Unknown error"
            trace_store.record(
                trace_id=req_trace_id,
                source="mesh.api_proxy",
                agent=agent_id,
                event_type="llm_call",
                detail=f"{api_request.service}/{api_request.action}",
                duration_ms=duration_ms,
                status=trace_status,
                error=trace_error,
                meta=trace_meta,
            )
        if event_bus is not None and not result.success and result.status_code == 402:
            event_bus.emit("credit_exhausted", agent=agent_id, data={
                "error": result.error or "Insufficient credits",
            })
        if event_bus is not None and result.success and result.data:
            model = result.data.get("model", "")
            tokens = result.data.get("tokens_used", 0)
            input_tok = result.data.get("input_tokens", 0)
            output_tok = result.data.get("output_tokens", 0)
            from src.host.costs import estimate_cost
            fixed_cost = result.data.get("fixed_cost_usd", 0)
            event_data = {
                "service": api_request.service, "action": api_request.action,
                "duration_ms": duration_ms,
                "model": model,
                "total_tokens": tokens,
                "input_tokens": input_tok,
                "output_tokens": output_tok,
                "cost_usd": fixed_cost if fixed_cost else estimate_cost(
                    model, input_tokens=input_tok, output_tokens=output_tok, total_tokens=tokens,
                ),
            }
            if prompt_preview:
                event_data["prompt_preview"] = prompt_preview
            if response_preview:
                event_data["response_preview"] = response_preview
            event_bus.emit("llm_call", agent=agent_id, data=event_data)
        return result

    @app.post("/mesh/api/stream")
    async def proxy_api_stream(request: Request, api_request: APIProxyRequest, agent_id: str) -> StreamingResponse:
        """Streaming API proxy. Returns SSE stream for LLM completions."""
        agent_id = _resolve_agent_id(agent_id, request)
        await _check_rate_limit("api_proxy", agent_id)
        if not permissions.can_use_api(agent_id, api_request.service):
            raise HTTPException(403, f"Agent {agent_id} cannot access {api_request.service}")
        if credential_vault is None:
            raise HTTPException(503, "No credential vault configured")

        req_trace_id = request.headers.get("x-trace-id")
        prompt_preview = _extract_prompt_preview(api_request.params)

        if req_trace_id and trace_store:
            stream_meta: dict = {
                "service": api_request.service,
                "action": api_request.action,
            }
            if prompt_preview:
                stream_meta["prompt_preview"] = prompt_preview
            trace_store.record(
                trace_id=req_trace_id,
                source="mesh.api_proxy",
                agent=agent_id,
                event_type="llm_stream",
                detail=f"{api_request.service}/{api_request.action}",
                meta=stream_meta,
            )

        async def _stream_with_events():
            start = time.monotonic()
            done_data: dict = {}
            credit_error = False
            async for chunk in credential_vault.stream_llm(api_request, agent_id=agent_id):
                yield chunk
                if chunk.startswith("data: {"):
                    try:
                        parsed = json.loads(chunk[6:].rstrip("\n"))
                        if not done_data and parsed.get("type") == "done":
                            done_data = parsed
                        if parsed.get("credit_exhausted"):
                            credit_error = True
                    except (json.JSONDecodeError, ValueError):
                        pass
            # Post-stream: emit llm_call event + trace completion
            duration_ms = int((time.monotonic() - start) * 1000)
            tokens = done_data.get("tokens_used", 0)
            model = done_data.get("model", "")
            if event_bus is not None and (tokens or model):
                from src.host.costs import estimate_cost
                ev: dict = {
                    "service": api_request.service,
                    "action": api_request.action,
                    "duration_ms": duration_ms,
                    "model": model,
                    "total_tokens": tokens,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": estimate_cost(model, total_tokens=tokens),
                }
                if prompt_preview:
                    ev["prompt_preview"] = prompt_preview
                response_preview = (done_data.get("content") or "")[:500]
                if response_preview:
                    ev["response_preview"] = response_preview
                event_bus.emit("llm_call", agent=agent_id, data=ev)
            if req_trace_id and trace_store and done_data:
                trace_store.record(
                    trace_id=req_trace_id,
                    source="mesh.api_proxy",
                    agent=agent_id,
                    event_type="llm_call",
                    detail=f"{api_request.service}/{api_request.action}",
                    duration_ms=duration_ms,
                    status="ok",
                    meta={
                        "model": model,
                        "tokens_used": tokens,
                        "streaming": True,
                    },
                )
            if event_bus is not None and credit_error:
                event_bus.emit("credit_exhausted", agent=agent_id, data={
                    "error": "Insufficient credits",
                })

        return StreamingResponse(_stream_with_events(), media_type="text/event-stream")

    # === Model Health Diagnostic ===

    @app.get("/mesh/model-health")
    async def model_health(request: Request) -> list[dict]:
        """Return model failover health status. Mesh-internal diagnostic."""
        _require_any_auth(request)
        if credential_vault is None:
            return []
        return credential_vault.get_model_health()

    # === Vault (credential management) ===

    @app.post("/mesh/vault/store")
    async def vault_store(data: dict, request: Request) -> dict:
        """Store a credential and return an opaque $CRED{name} handle."""
        agent_id = _resolve_agent_id(data.get("agent_id", ""), request)
        await _check_rate_limit("vault_store", agent_id)
        if not permissions.can_manage_vault(agent_id):
            raise HTTPException(403, f"Agent {agent_id} cannot manage vault")
        if credential_vault is None:
            raise HTTPException(503, "No credential vault configured")
        name = data.get("name", "").strip()
        value = data.get("value", "").strip()
        if not name or not value:
            raise HTTPException(400, "name and value are required")
        if not re.match(r"^[a-zA-Z0-9_.-]{1,128}$", name):
            raise HTTPException(400, "Credential name must be 1-128 alphanumeric/underscore/dot/dash chars")
        if len(value) > 10_000:
            raise HTTPException(400, "Credential value exceeds 10KB limit")
        if is_system_credential(name):
            raise HTTPException(403, f"Cannot store system credential: {name}")
        if not permissions.can_access_credential(agent_id, name):
            raise HTTPException(403, f"Agent {agent_id} cannot access credential: {name}")
        handle = credential_vault.add_credential(name, value)
        return {"stored": True, "handle": handle}

    @app.get("/mesh/vault/list")
    async def vault_list(agent_id: str, request: Request) -> dict:
        """List credential names the agent can access (never values)."""
        agent_id = _resolve_agent_id(agent_id, request)
        if not permissions.can_manage_vault(agent_id):
            raise HTTPException(403, f"Agent {agent_id} cannot manage vault")
        if credential_vault is None:
            raise HTTPException(503, "No credential vault configured")
        # Agent-tier only: system credentials are never resolvable by agents
        all_names = credential_vault.list_agent_credential_names()
        names = [n for n in all_names if permissions.can_access_credential(agent_id, n)]
        return {"credentials": names, "count": len(names)}

    @app.get("/mesh/vault/status/{name}")
    async def vault_status(name: str, agent_id: str, request: Request) -> dict:
        """Check if a credential exists by name."""
        agent_id = _resolve_agent_id(agent_id, request)
        if not permissions.can_access_credential(agent_id, name):
            raise HTTPException(403, f"Agent {agent_id} cannot access credential {name}")
        if credential_vault is None:
            raise HTTPException(503, "No credential vault configured")
        return {"name": name, "exists": credential_vault.has_credential(name)}

    @app.post("/mesh/vault/resolve")
    async def vault_resolve(data: dict, request: Request) -> dict:
        """Resolve a credential handle to its value. Internal use only (browser tool)."""
        agent_id = _resolve_agent_id(data.get("agent_id", ""), request)
        name = data.get("name", "")
        if not name:
            raise HTTPException(400, "name is required")
        if not permissions.can_access_credential(agent_id, name):
            raise HTTPException(403, f"Agent {agent_id} cannot access credential {name}")
        if credential_vault is None:
            raise HTTPException(503, "No credential vault configured")

        await _check_rate_limit("vault_resolve", agent_id)

        # Audit log every resolve
        logger.info(
            "Vault credential resolved",
            extra={"extra_data": {"agent_id": agent_id, "credential": name}},
        )

        value = credential_vault.resolve_credential(name)
        if value is None:
            raise HTTPException(404, f"Credential not found: {name}")
        return {"name": name, "value": value}

    # === Wallet Signing Service ===

    _ws_ref = wallet_service_ref or [None]

    @app.get("/mesh/wallet/address")
    async def wallet_address(chain: str, agent_id: str, request: Request) -> dict:
        """Get agent's wallet address for a chain."""
        agent_id = _resolve_agent_id(agent_id, request)
        if not permissions.can_use_wallet(agent_id):
            raise HTTPException(403, "Wallet access denied")
        if not permissions.can_use_wallet_chain(agent_id, chain):
            raise HTTPException(403, f"Chain not allowed: {chain}")
        if _ws_ref[0] is None:
            raise HTTPException(503, "Wallet service not configured")
        await _check_rate_limit("wallet_read", agent_id)
        try:
            address = await _ws_ref[0].get_address(agent_id, chain)
            return {"address": address, "chain": chain}
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.get("/mesh/wallet/balance")
    async def wallet_balance(
        chain: str, agent_id: str, request: Request, token: str = "native",
    ) -> dict:
        """Get wallet balance.  Read-only, no signing."""
        agent_id = _resolve_agent_id(agent_id, request)
        if not permissions.can_use_wallet(agent_id):
            raise HTTPException(403, "Wallet access denied")
        if not permissions.can_use_wallet_chain(agent_id, chain):
            raise HTTPException(403, f"Chain not allowed: {chain}")
        if _ws_ref[0] is None:
            raise HTTPException(503, "Wallet service not configured")
        await _check_rate_limit("wallet_read", agent_id)
        try:
            return await _ws_ref[0].get_balance(agent_id, chain, token)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post("/mesh/wallet/read")
    async def wallet_read(data: dict, request: Request) -> dict:
        """Read-only contract call / account read."""
        agent_id = _resolve_agent_id(data.get("agent_id", ""), request)
        chain = data.get("chain", "")
        if not permissions.can_use_wallet(agent_id):
            raise HTTPException(403, "Wallet access denied")
        if not permissions.can_use_wallet_chain(agent_id, chain):
            raise HTTPException(403, f"Chain not allowed: {chain}")
        if _ws_ref[0] is None:
            raise HTTPException(503, "Wallet service not configured")
        await _check_rate_limit("wallet_read", agent_id)
        try:
            return await _ws_ref[0].read_contract(
                agent_id, chain, data.get("contract", ""),
                data.get("function", ""), data.get("args", []),
            )
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post("/mesh/wallet/transfer")
    async def wallet_transfer_endpoint(data: dict, request: Request) -> dict:
        """Sign and broadcast a token transfer."""
        agent_id = _resolve_agent_id(data.get("agent_id", ""), request)
        chain = data.get("chain", "")
        if not permissions.can_use_wallet(agent_id):
            raise HTTPException(403, "Wallet access denied")
        if not permissions.can_use_wallet_chain(agent_id, chain):
            raise HTTPException(403, f"Chain not allowed: {chain}")
        if _ws_ref[0] is None:
            raise HTTPException(503, "Wallet service not configured")
        await _check_rate_limit("wallet_transfer", agent_id)
        logger.info(
            "Wallet transfer",
            extra={"extra_data": {
                "agent_id": agent_id, "chain": chain,
                "to": data.get("to", ""), "amount": data.get("amount", ""),
            }},
        )
        try:
            return await _ws_ref[0].transfer(
                agent_id, chain,
                data.get("to", ""), data.get("amount", ""),
                data.get("token", "native"), permissions,
            )
        except ValueError as e:
            raise HTTPException(400, str(e))
        except PermissionError as e:
            raise HTTPException(403, str(e))

    @app.post("/mesh/wallet/execute")
    async def wallet_execute_endpoint(data: dict, request: Request) -> dict:
        """Sign and broadcast a contract call or Solana transaction."""
        agent_id = _resolve_agent_id(data.get("agent_id", ""), request)
        chain = data.get("chain", "")
        contract = data.get("contract", "")
        if not permissions.can_use_wallet(agent_id):
            raise HTTPException(403, "Wallet access denied")
        if not permissions.can_use_wallet_chain(agent_id, chain):
            raise HTTPException(403, f"Chain not allowed: {chain}")
        if contract and not permissions.can_access_wallet_contract(agent_id, contract):
            raise HTTPException(403, f"Contract not allowed: {contract}")
        if _ws_ref[0] is None:
            raise HTTPException(503, "Wallet service not configured")
        await _check_rate_limit("wallet_execute", agent_id)
        logger.info(
            "Wallet execute",
            extra={"extra_data": {
                "agent_id": agent_id, "chain": chain,
                "contract": contract, "function": data.get("function", ""),
            }},
        )
        try:
            return await _ws_ref[0].execute_contract(
                agent_id, chain, contract,
                data.get("function", ""), data.get("args", []),
                data.get("value", "0"), data.get("transaction", ""),
                permissions,
            )
        except ValueError as e:
            raise HTTPException(400, str(e))
        except PermissionError as e:
            raise HTTPException(403, str(e))

    # === Agent Registry ===

    @app.post("/mesh/register")
    async def register_agent(data: dict, request: Request) -> dict:
        """Agent registers itself with the mesh on startup."""
        agent_id = _validate_agent_id(data.get("agent_id", ""))
        agent_id = _resolve_agent_id(agent_id, request)
        capabilities = data.get("capabilities", [])
        if not isinstance(capabilities, list) or len(capabilities) > 50:
            raise HTTPException(400, "capabilities must be a list of at most 50 items")
        port = _validate_port(data.get("port", 8400))

        existing = router.agent_registry.get(agent_id)
        if existing:
            url = existing.get("url", existing) if isinstance(existing, dict) else existing
        else:
            url = f"http://localhost:{port}"

        router.register_agent(agent_id, url, capabilities)
        agent_perms = permissions.get_permissions(agent_id)
        reg_project = _agent_projects.get(agent_id)
        for topic in agent_perms.can_subscribe:
            scoped = f"projects/{reg_project}/{topic}" if reg_project else topic
            pubsub.subscribe(scoped, agent_id)
        # Auto-watch task inbox (coordination protocol).
        # Only for agents with blackboard access (skip standalone agents).
        if agent_perms.blackboard_read:
            inbox_pattern = (
                f"projects/{reg_project}/tasks/{agent_id}/*"
                if reg_project
                else f"tasks/{agent_id}/*"
            )
            blackboard.add_watch(agent_id, inbox_pattern)
        if event_bus is not None:
            event_bus.emit("agent_state", agent=agent_id, data={
                "state": "registered", "capabilities": capabilities,
            })
        return {"registered": True}

    # === Agent Notifications ===

    _NOTIFY_MAX_LEN = 2000
    _WS_FILE_NAMES = ("SOUL.md", "INSTRUCTIONS.md", "USER.md", "HEARTBEAT.md", "MEMORY.md")

    @app.post("/mesh/notify")
    async def notify_user(body: NotifyRequest, request: Request) -> dict:
        """Push a notification from an agent to the user across all channels."""
        body.agent_id = _resolve_agent_id(body.agent_id, request)
        await _check_rate_limit("notify", body.agent_id)
        if notify_fn is None:
            raise HTTPException(503, "Notifications not available")
        message = body.message[:_NOTIFY_MAX_LEN]
        # Emit to dashboard first — users should see notifications even if
        # channel delivery (Telegram/Discord/etc.) fails below.
        if event_bus:
            event_bus.emit("notification", agent=body.agent_id,
                           data={"message": message})
            if any(f in message for f in _WS_FILE_NAMES):
                event_bus.emit("workspace_updated", agent=body.agent_id,
                               data={"message": message})
        try:
            await notify_fn(body.agent_id, message)
        except Exception as e:
            logger.warning("notify_user failed: %s", e)
            raise HTTPException(500, f"Notification failed: {e}")
        return {"sent": True}

    @app.get("/mesh/agents")
    async def list_agents(request: Request, project: str = "", agent_id: str = "") -> dict:
        """List registered agents, optionally scoped by project or agent_id.

        - project set: return only that project's members
        - agent_id set (standalone): return only that agent
        - neither (dashboard/internal): return all
        """
        if agent_id:
            agent_id = _resolve_agent_id(agent_id, request)
        else:
            _require_any_auth(request)
        def _agent_entry(aid: str, url: str) -> dict:
            return {"url": url, "role": router.agent_roles.get(aid, "")}

        if project:
            from src.cli.config import _load_projects
            projects = _load_projects()
            pdata = projects.get(project)
            if pdata is None:
                logger.warning("list_agents: unknown project %r", project)
                return {}
            members = set(pdata.get("members", []))
            return {
                aid: _agent_entry(aid, url)
                for aid, url in router.agent_registry.items()
                if aid in members
            }
        if agent_id:
            url = router.agent_registry.get(agent_id)
            if url:
                return {agent_id: _agent_entry(agent_id, url)}
            return {}
        return {
            aid: _agent_entry(aid, url)
            for aid, url in router.agent_registry.items()
        }

    # === Agent Introspection ===

    @app.get("/mesh/introspect")
    async def introspect(section: str = "all", request: Request = ...):
        """Return runtime state for the requesting agent.

        Agents use this to understand their permissions, budget, fleet,
        cron schedule, and health.  No sensitive data is exposed.
        """
        agent_id = _extract_verified_agent_id(request)
        result: dict = {}

        if section in ("permissions", "all"):
            perms = permissions.get_permissions(agent_id)
            result["permissions"] = {
                "blackboard_read": perms.blackboard_read,
                "blackboard_write": perms.blackboard_write,
                "can_message": perms.can_message,
                "can_publish": perms.can_publish,
                "can_subscribe": perms.can_subscribe,
                "allowed_apis": perms.allowed_apis,
                "allowed_credentials": perms.allowed_credentials,
            }

        if section in ("budget", "all") and cost_tracker:
            result["budget"] = cost_tracker.check_budget(agent_id)
            # Include project budget if agent belongs to a project
            agent_proj = _agent_projects.get(agent_id)
            if agent_proj and hasattr(cost_tracker, "get_project_spend"):
                project_spend = cost_tracker.get_project_spend(agent_proj, "today")
                if "error" not in project_spend:
                    result["project_budget"] = project_spend

        if section in ("fleet", "all"):
            # Scope fleet list by project: project agents see only peers,
            # standalone agents see only themselves.
            from src.cli.config import _load_projects
            _projects = _load_projects()
            _agent_project_members: set[str] | None = None
            for _pdata in _projects.values():
                if agent_id in _pdata.get("members", []):
                    _agent_project_members = set(_pdata["members"])
                    break

            if _agent_project_members is not None:
                result["fleet"] = [
                    {"id": aid, "role": router.agent_roles.get(aid, "")}
                    for aid in router.agent_registry
                    if aid in _agent_project_members
                ]
            else:
                result["fleet"] = [
                    {"id": agent_id, "role": router.agent_roles.get(agent_id, "")}
                ]

        if section in ("cron", "all") and cron_scheduler:
            result["cron"] = [
                j for j in cron_scheduler.list_jobs()
                if j.get("agent") == agent_id
            ]

        if section in ("health", "all") and health_monitor:
            statuses = health_monitor.get_status()
            result["health"] = next(
                (s for s in statuses if s["agent"] == agent_id), None
            )

        return result

    # === Project Costs ===

    @app.get("/mesh/costs/project/{project}")
    async def get_project_costs(project: str, request: Request, period: str = "today") -> dict:
        """Return aggregated cost data for a project."""
        _require_any_auth(request)
        if cost_tracker is None:
            raise HTTPException(503, "Cost tracker not available")
        if not hasattr(cost_tracker, "get_project_spend"):
            raise HTTPException(503, "Project cost tracking not available")
        return cost_tracker.get_project_spend(project, period)

    # === Cron CRUD ===

    @app.post("/mesh/cron")
    async def create_cron_job(data: dict, request: Request) -> dict:
        """Create a cron job. Body: {agent_id, schedule, message, heartbeat?}."""
        if cron_scheduler is None:
            raise HTTPException(503, "Cron scheduler not available")
        agent_id = data.get("agent_id", "")
        if agent_id:
            _validate_agent_id(agent_id)
            agent_id = _resolve_agent_id(agent_id, request)
            if not permissions.can_manage_cron(agent_id):
                raise HTTPException(403, f"Agent {agent_id} is not allowed to manage cron jobs")
            await _check_rate_limit("cron_create", agent_id)
        schedule = data.get("schedule")
        message = data.get("message", "")
        heartbeat = data.get("heartbeat", False)
        tool_name = data.get("tool_name") or None
        tool_params = data.get("tool_params") or None
        if not agent_id or not schedule:
            raise HTTPException(400, "agent_id and schedule are required")
        if tool_name and message:
            raise HTTPException(400, "tool_name and message are mutually exclusive — use one or the other")
        if tool_params:
            try:
                json.loads(tool_params)
            except (json.JSONDecodeError, TypeError):
                raise HTTPException(400, "tool_params must be a valid JSON string (e.g. '{\"key\": \"value\"}')")
        try:
            job = cron_scheduler.add_job(
                agent=agent_id, schedule=schedule, message=message, heartbeat=heartbeat,
                tool_name=tool_name, tool_params=tool_params,
            )
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {
            "id": job.id, "agent": job.agent, "schedule": job.schedule,
            "heartbeat": job.heartbeat, "tool_name": job.tool_name,
        }

    @app.get("/mesh/cron")
    async def list_cron_jobs(request: Request, agent_id: str | None = None) -> list[dict]:
        """List cron jobs, optionally filtered by agent_id."""
        _require_any_auth(request)
        if cron_scheduler is None:
            raise HTTPException(503, "Cron scheduler not available")
        jobs = cron_scheduler.list_jobs()
        if agent_id:
            jobs = [j for j in jobs if j["agent"] == agent_id]
        return jobs

    @app.put("/mesh/cron/{job_id}")
    async def update_cron_job(job_id: str, request: Request) -> dict:
        """Update a cron job by ID. Body: fields to update (schedule, enabled, etc)."""
        agent_id = _resolve_agent_id("", request)
        if not permissions.can_manage_cron(agent_id):
            raise HTTPException(403, f"Agent {agent_id} cannot manage cron jobs")
        if cron_scheduler is None:
            raise HTTPException(503, "Cron scheduler not available")
        job = cron_scheduler.jobs.get(job_id)
        if not job:
            raise HTTPException(404, f"Job not found: {job_id}")
        if job.agent != agent_id and not permissions._is_trusted(agent_id):
            raise HTTPException(403, f"Agent {agent_id} does not own job {job_id}")
        body = await request.json()
        if "schedule" in body:
            error = cron_scheduler._validate_schedule(body["schedule"])
            if error:
                raise HTTPException(400, error)
        job = await cron_scheduler.update_job(job_id, **body)
        if not job:
            raise HTTPException(404, f"Job not found: {job_id}")
        from dataclasses import asdict
        return {"status": "updated", "job": asdict(job)}

    @app.delete("/mesh/cron/{job_id}")
    async def delete_cron_job(job_id: str, request: Request) -> dict:
        """Remove a cron job by ID."""
        agent_id = _resolve_agent_id("", request)
        if not permissions.can_manage_cron(agent_id):
            raise HTTPException(403, f"Agent {agent_id} cannot manage cron jobs")
        if cron_scheduler is None:
            raise HTTPException(503, "Cron scheduler not available")
        job = cron_scheduler.jobs.get(job_id)
        if not job:
            raise HTTPException(404, f"Job not found: {job_id}")
        if job.agent != agent_id and not permissions._is_trusted(agent_id):
            raise HTTPException(403, f"Agent {agent_id} does not own job {job_id}")
        if cron_scheduler.remove_job(job_id):
            return {"removed": True, "id": job_id}
        raise HTTPException(404, f"Job not found: {job_id}")

    # === Dynamic Agent Spawning ===

    @app.post("/mesh/spawn")
    async def spawn_agent(data: dict, request: Request) -> dict:
        """Spawn an ephemeral agent. Body: {role, system_prompt?, model?, ttl?}."""
        if container_manager is None:
            raise HTTPException(503, "Container manager not available")
        role = data.get("role", "assistant")
        if not isinstance(role, str) or len(role) > 64:
            raise HTTPException(400, "role must be a string of at most 64 chars")
        spawned_by = _resolve_agent_id(data.get("spawned_by", "unknown"), request)
        await _check_rate_limit("spawn", spawned_by)
        if not permissions.can_spawn(spawned_by):
            raise HTTPException(403, f"Agent {spawned_by} is not allowed to spawn agents")
        model = data.get("model", "")
        ttl = data.get("ttl", 3600)
        if not isinstance(ttl, (int, float)) or ttl < 60 or ttl > 86400:
            raise HTTPException(400, "ttl must be 60-86400 seconds")
        # system_prompt is routed through as initial_instructions (workspace seed)
        system_prompt = data.get("system_prompt", f"You are a '{role}' agent.")
        if len(system_prompt) > _MAX_SYSTEM_PROMPT:
            raise HTTPException(400, f"system_prompt exceeds {_MAX_SYSTEM_PROMPT} char limit")
        from src.shared.utils import generate_id
        agent_id = generate_id("spawn")
        try:
            url = container_manager.spawn_agent(
                agent_id=agent_id, role=role, system_prompt=system_prompt,
                model=model, ttl=ttl,
            )
            router.register_agent(agent_id, url)
            if health_monitor is not None:
                health_monitor.register(agent_id)
            # Store ephemeral metadata for TTL cleanup
            container_manager.agents.setdefault(agent_id, {}).update({
                "ephemeral": True, "ttl": ttl,
                "spawned_at": time.time(), "role": role,
            })
            ready = await container_manager.wait_for_agent(agent_id, timeout=60)
            if trace_store:
                from src.shared.trace import new_trace_id as _new_trace_id
                trace_store.record(
                    trace_id=_new_trace_id(), source="mesh.spawn", agent=agent_id,
                    event_type="agent_spawn",
                    detail=f"role={role} spawned_by={spawned_by}",
                )
            if event_bus is not None:
                event_bus.emit("agent_state", agent=agent_id, data={
                    "state": "spawned", "role": role, "ready": ready,
                })
            return {
                "agent_id": agent_id, "url": url, "role": role,
                "ready": ready, "spawned_by": spawned_by, "ttl": ttl,
            }
        except Exception as e:
            raise HTTPException(500, f"Failed to spawn agent: {e}") from e

    # === Agent History Access ===

    @app.get("/mesh/agents/{agent_id}/history")
    async def get_agent_history(agent_id: str, request: Request, requesting_agent: str = "") -> dict:
        """Retrieve an agent's daily logs. Permission-checked."""
        if requesting_agent:
            requesting_agent = _resolve_agent_id(requesting_agent, request)
            if not permissions.can_message(requesting_agent, agent_id):
                raise HTTPException(403, f"Agent {requesting_agent} cannot read history of {agent_id}")
        else:
            # Internal/dashboard callers: require any valid auth token.
            _require_any_auth(request)
            logger.debug("History access for %s without requesting_agent (mesh-internal)", agent_id)
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

    # === Agent Profile ===

    @app.get("/mesh/agents/{agent_id}/profile")
    async def get_agent_profile(agent_id: str, request: Request, requesting_agent: str = "") -> dict:
        """Return an agent's public profile: mesh-derived metadata + INTERFACE.md.

        Agents use this to understand how to collaborate with a peer —
        what it accepts, produces, subscribes to, and its public contract.
        Permission-checked: requesting agent must be allowed to message the target.
        """
        if requesting_agent:
            requesting_agent = _resolve_agent_id(requesting_agent, request)
            await _check_rate_limit("agent_profile", requesting_agent)
            if not permissions.can_message(requesting_agent, agent_id):
                raise HTTPException(403, f"Agent {requesting_agent} cannot read profile of {agent_id}")
        else:
            _require_any_auth(request)

        if agent_id not in router.agent_registry:
            raise HTTPException(404, f"Agent not found: {agent_id}")

        # -- Mesh-derived metadata (verifiable, always fresh) --
        role = router.agent_roles.get(agent_id, "")
        capabilities = router._capabilities_cache.get(agent_id, [])

        # Health status
        status = "unknown"
        last_active = None
        if health_monitor is not None:
            h = health_monitor.agents.get(agent_id)
            if h is not None:
                status = h.status
                if h.last_healthy:
                    from datetime import datetime, timezone
                    last_active = datetime.fromtimestamp(
                        h.last_healthy, tz=timezone.utc,
                    ).isoformat()

        # Heartbeat schedule
        heartbeat_schedule = None
        if cron_scheduler is not None:
            for job in cron_scheduler.jobs.values():
                if job.agent == agent_id and job.heartbeat:
                    heartbeat_schedule = job.schedule
                    break

        # Subscriptions (strip project prefix for readability)
        raw_subs = pubsub.get_agent_subscriptions(agent_id) if pubsub else []
        project = _agent_projects.get(agent_id)
        prefix = f"projects/{project}/" if project else ""
        subscriptions = [
            t[len(prefix):] if prefix and t.startswith(prefix) else t
            for t in raw_subs
        ]

        # Watches (strip project prefix)
        raw_watches = blackboard.get_agent_watches(agent_id)
        watches = [
            w[len(prefix):] if prefix and w.startswith(prefix) else w
            for w in raw_watches
        ]

        # Recent blackboard writes (strip project prefix, keys only)
        raw_writes = blackboard.recent_keys_by_agent(agent_id)
        recent_writes = [
            k[len(prefix):] if prefix and k.startswith(prefix) else k
            for k in raw_writes
        ]

        # -- Agent-declared interface (INTERFACE.md) --
        interface = None
        if transport is not None:
            try:
                ws_resp = await transport.request(agent_id, "GET", "/workspace/INTERFACE.md", timeout=10)
                content = ws_resp.get("content", "") if isinstance(ws_resp, dict) else ""
                if content and content.strip() not in ("", "# Interface"):
                    interface = sanitize_for_prompt(content)
            except Exception:
                logger.debug("Could not fetch INTERFACE.md from %s", agent_id)
        elif agent_id in router.agent_registry:
            agent_url = router.agent_registry[agent_id]
            if isinstance(agent_url, dict):
                agent_url = agent_url.get("url", agent_url)
            import httpx
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.get(f"{agent_url}/workspace/INTERFACE.md")
                    if resp.status_code == 200:
                        content = resp.json().get("content", "")
                        if content and content.strip() not in ("", "# Interface"):
                            interface = sanitize_for_prompt(content)
            except Exception:
                logger.debug("Could not fetch INTERFACE.md from %s (direct)", agent_id)

        return {
            "agent_id": agent_id,
            "role": role,
            "status": status,
            "last_active": last_active,
            "heartbeat_schedule": heartbeat_schedule,
            "subscriptions": subscriptions,
            "watches": watches,
            "recent_writes": recent_writes,
            "capabilities": capabilities,
            "interface": interface,
        }

    # === Request Traces ===

    @app.get("/mesh/traces")
    async def list_traces(request: Request, limit: int = 50) -> list[dict]:
        """Return recent trace events."""
        _require_any_auth(request)
        if trace_store is None:
            return []
        return trace_store.list_recent(limit=limit)

    @app.get("/mesh/traces/{trace_id}")
    async def get_trace(trace_id: str, request: Request) -> list[dict]:
        """Return all events for a specific trace."""
        _require_any_auth(request)
        if trace_store is None:
            return []
        return trace_store.get_trace(trace_id)

    # === External API (API-key authenticated) ===
    #
    # These endpoints let external systems manage credentials and query
    # agent status without a dashboard session cookie.
    # Authenticated via X-API-Key header against named keys in ApiKeyManager.

    _RATE_LIMITS["ext_credentials"] = (30, 60)
    _RATE_LIMITS["ext_status"] = (60, 60)

    def _require_api_key(request: Request) -> str:
        """Verify the X-API-Key header.  Returns the key ID for rate limiting."""
        if api_key_manager is None or not api_key_manager.has_keys():
            raise HTTPException(503, "External API not configured (create an API key in the dashboard)")
        provided = request.headers.get("x-api-key", "")
        if not provided:
            raise HTTPException(401, "Missing API key (X-API-Key header)")
        result = api_key_manager.authenticate(provided)
        if result is None:
            raise HTTPException(401, "Invalid API key")
        return result["id"]

    _CRED_NAME_RE = re.compile(r"^[a-zA-Z0-9_.\-]{1,128}$")

    @app.post("/mesh/credentials")
    async def ext_store_credential(request: Request) -> dict:
        """Store an agent-tier credential. Returns an opaque $CRED{name} handle.

        External systems use this to inject per-session secrets (e.g. PII)
        that agents reference by handle without seeing raw values.
        """
        kid = _require_api_key(request)
        await _check_rate_limit("ext_credentials", kid)
        if credential_vault is None:
            raise HTTPException(503, "Credential vault not configured")
        body = await request.json()
        name = (body.get("name") or "").strip()
        value = (body.get("value") or "").strip()
        if not name or not value:
            raise HTTPException(400, "name and value are required")
        if not _CRED_NAME_RE.match(name):
            raise HTTPException(400, "name must be 1-128 alphanumeric/underscore/dot/dash chars")
        if len(value) > 10_000:
            raise HTTPException(400, "value exceeds 10KB limit")
        if is_system_credential(name):
            raise HTTPException(403, "Cannot store system credentials via external API")
        handle = credential_vault.add_credential(name, value)
        return {"stored": True, "handle": handle, "name": name}

    @app.delete("/mesh/credentials/{name}")
    async def ext_remove_credential(name: str, request: Request) -> dict:
        """Remove an agent-tier credential by name."""
        kid = _require_api_key(request)
        await _check_rate_limit("ext_credentials", kid)
        if credential_vault is None:
            raise HTTPException(503, "Credential vault not configured")
        if is_system_credential(name):
            raise HTTPException(403, "Cannot remove system credentials via external API")
        existed = credential_vault.remove_credential(name)
        if not existed:
            raise HTTPException(404, f"Credential not found: {name}")
        return {"removed": True, "name": name}

    @app.get("/mesh/credentials")
    async def ext_list_credentials(request: Request) -> dict:
        """List agent-tier credential names (never values)."""
        kid = _require_api_key(request)
        await _check_rate_limit("ext_credentials", kid)
        if credential_vault is None:
            raise HTTPException(503, "Credential vault not configured")
        names = credential_vault.list_agent_credential_names()
        return {"credentials": names, "count": len(names)}

    @app.get("/mesh/credentials/{name}/exists")
    async def ext_credential_exists(name: str, request: Request) -> dict:
        """Check if a credential exists by name (never returns value)."""
        kid = _require_api_key(request)
        await _check_rate_limit("ext_credentials", kid)
        if credential_vault is None:
            raise HTTPException(503, "Credential vault not configured")
        return {"name": name, "exists": credential_vault.has_credential(name)}

    @app.get("/mesh/agents/{agent_id}/ext-status")
    async def ext_agent_status(agent_id: str, request: Request) -> dict:
        """Query agent status from an external system.

        Returns agent state, queue depth, and health — enough for an
        external system to decide whether to trigger a workflow.
        """
        kid = _require_api_key(request)
        await _check_rate_limit("ext_status", kid)
        _validate_agent_id(agent_id)
        result: dict = {"agent_id": agent_id}
        if agent_id not in router.agent_registry:
            raise HTTPException(404, f"Agent not found: {agent_id}")
        if health_monitor is not None:
            statuses = health_monitor.get_status()
            result["health"] = next(
                (s for s in statuses if s["agent"] == agent_id), None,
            )
        if lane_manager is not None:
            lane_status = lane_manager.get_status()
            agent_lane = lane_status.get(agent_id, {})
            result["queue"] = {
                "queued": agent_lane.get("queued", 0),
                "busy": agent_lane.get("busy", False),
            }
        if cost_tracker is not None:
            result["budget"] = cost_tracker.check_budget(agent_id)
        return result

    # === Browser Service Proxy ===

    import httpx as _httpx
    _browser_proxy_client = _httpx.AsyncClient(timeout=60)

    _last_browser_boot_id: str | None = None

    async def _push_browser_proxy(agent_id: str) -> None:
        """Push proxy config for an agent to the browser service."""
        if not container_manager:
            return
        svc_url = getattr(container_manager, "browser_service_url", None)
        svc_token = getattr(container_manager, "browser_auth_token", "")
        if not svc_url:
            return

        from src.cli.config import _load_config
        from src.cli.proxy import parse_proxy_url, resolve_agent_proxy
        _fresh_cfg = _load_config()
        agents_cfg = _fresh_cfg.get("agents", {})
        network_cfg = _fresh_cfg.get("network", {})
        proxy_url = resolve_agent_proxy(agent_id, agents_cfg, network_cfg)

        body = None
        if proxy_url:
            parsed = parse_proxy_url(proxy_url)
            if parsed:
                body = {
                    "url": parsed["url"],
                    "username": parsed["username"],
                    "password": parsed["password"],
                }

        headers: dict = {"X-Mesh-Internal": "1"}
        if svc_token:
            headers["Authorization"] = f"Bearer {svc_token}"
        try:
            await _browser_proxy_client.put(
                f"{svc_url}/browser/{agent_id}/proxy",
                json=body,
                headers=headers,
            )
        except Exception as e:
            logger.warning("Failed to push proxy config for %s: %s", agent_id, e)

    async def _check_browser_boot_id_changed() -> bool:
        """Check if browser service restarted by comparing boot_id."""
        nonlocal _last_browser_boot_id
        if not container_manager:
            return False
        svc_url = getattr(container_manager, "browser_service_url", None)
        svc_token = getattr(container_manager, "browser_auth_token", "")
        if not svc_url:
            return False
        try:
            headers: dict = {}
            if svc_token:
                headers["Authorization"] = f"Bearer {svc_token}"
            resp = await _browser_proxy_client.get(
                f"{svc_url}/browser/status", headers=headers,
            )
            data = resp.json()
            boot_id = data.get("boot_id")
            if _last_browser_boot_id is None:
                _last_browser_boot_id = boot_id
                return False
            if boot_id != _last_browser_boot_id:
                _last_browser_boot_id = boot_id
                return True
        except Exception:
            pass
        return False

    async def _deferred_push_browser_proxies() -> None:
        """Push proxy config for all agents after a delay (non-blocking background task)."""
        await asyncio.sleep(5)  # Wait for agents to register
        if not container_manager:
            return
        svc_url = getattr(container_manager, "browser_service_url", None)
        if not svc_url:
            return
        agents = list(router.agent_registry.keys())
        if not agents:
            return
        for agent_id in agents:
            await _push_browser_proxy(agent_id)
        logger.info("Pushed browser proxy config for %d agents", len(agents))

    @app.on_event("startup")
    async def _schedule_initial_proxy_push() -> None:
        """Schedule initial browser proxy push as a background task (non-blocking)."""
        asyncio.create_task(_deferred_push_browser_proxies())

    _ALLOWED_BROWSER_ACTIONS = frozenset({
        "navigate", "snapshot", "click", "type", "hover",
        "screenshot", "reset", "focus", "status", "solve_captcha", "scroll",
        "wait_for", "press_key", "go_back", "go_forward", "switch_tab",
    })

    @app.post("/mesh/browser/command")
    async def browser_command(request: Request) -> dict:
        """Proxy a browser command to the shared browser service.

        Agents never talk to the browser service directly — the mesh
        enforces authentication and permission checks.
        """
        agent_id = _extract_verified_agent_id(request)
        body = await request.json()
        req_agent_id = body.get("agent_id", agent_id)
        # Use verified identity, not the claimed one
        req_agent_id = _resolve_agent_id(req_agent_id, request)

        if not permissions.can_use_browser(req_agent_id):
            raise HTTPException(403, "Browser access denied")

        action = body.get("action", "")
        params = body.get("params", {})

        if not action:
            raise HTTPException(400, "action is required")

        if action not in _ALLOWED_BROWSER_ACTIONS:
            raise HTTPException(400, f"Unknown browser action: {action}")

        # SSRF protection: validate navigate URLs before forwarding to browser
        if action == "navigate":
            nav_url = params.get("url", "")
            if nav_url:
                from src.agent.builtins.http_tool import _resolve_and_pin
                try:
                    _resolve_and_pin(nav_url)
                except ValueError as e:
                    raise HTTPException(400, str(e))

        # Check for browser service restart — lazy re-push proxy config
        try:
            restarted = await _check_browser_boot_id_changed()
            if restarted:
                logger.info("Browser service restarted, re-pushing proxy for %s", req_agent_id)
                await _push_browser_proxy(req_agent_id)
        except Exception:
            pass  # Non-blocking — don't fail the browser command

        # Proxy to browser service
        browser_service_url = None
        if container_manager:
            browser_service_url = getattr(container_manager, "browser_service_url", None)
        if not browser_service_url:
            raise HTTPException(503, "Browser service not available")

        try:
            browser_auth = getattr(container_manager, "browser_auth_token", "")
            headers = {}
            if browser_auth:
                headers["Authorization"] = f"Bearer {browser_auth}"
            resp = await _browser_proxy_client.post(
                f"{browser_service_url}/browser/{req_agent_id}/{action}",
                json=params,
                headers=headers,
            )
            resp.raise_for_status()
            return resp.json()
        except _httpx.HTTPStatusError as e:
            raise HTTPException(e.response.status_code, e.response.text)
        except Exception as e:
            logger.warning("Browser proxy error: %s", e)
            raise HTTPException(502, f"Browser service error: {e}")

    # === Event Bus ===

    @app.websocket("/ws/events")
    async def ws_events(websocket: WebSocket) -> None:
        """Stream real-time dashboard events to WebSocket clients."""
        if event_bus is None:
            await websocket.close(code=1013, reason="Event bus not configured")
            return

        # Verify session cookie before accepting the WebSocket upgrade.
        # Browsers send cookies with WebSocket upgrade requests, so we can
        # use the same auth as the dashboard HTTP endpoints.
        from src.dashboard.auth import verify_session_cookie
        cookie_value = websocket.cookies.get("ol_session", "")
        auth_error = verify_session_cookie(cookie_value)
        if auth_error is not None:
            await websocket.close(code=1008, reason=auth_error)
            return

        # Lazily bind event loop on first WebSocket connect
        import asyncio
        event_bus.set_loop(asyncio.get_running_loop())

        await websocket.accept()

        # Parse optional filters from query params
        agents_param = websocket.query_params.get("agents", "")
        types_param = websocket.query_params.get("types", "")
        agents_filter = set(agents_param.split(",")) - {""} if agents_param else None
        types_filter = set(types_param.split(",")) - {""} if types_param else None

        # Subscribe first, then replay events that existed before subscribe.
        # This eliminates the race where events emitted between replay and
        # subscribe appear twice (once in replay, once in live feed).
        snapshot_seq = event_bus.current_seq
        event_bus.subscribe(websocket, agents_filter, types_filter)
        for evt in event_bus.recent_events(agents_filter, types_filter, before_seq=snapshot_seq):
            await websocket.send_text(json.dumps(evt, default=str))
        try:
            while True:
                await websocket.receive_text()  # keep-alive
        except Exception as e:
            logger.debug("WebSocket disconnected: %s", e)
        finally:
            event_bus.unsubscribe(websocket)

    # ── VNC reverse proxy ────────────────────────────────────────────────
    # Proxies HTTP (static files) and WebSocket (VNC stream) requests from
    # /vnc/{path} to the browser container's KasmVNC port.  This lets VNC
    # work out-of-the-box behind a reverse proxy without exposing extra
    # ports or dealing with mixed-content issues.

    def _get_vnc_port() -> int | None:
        """Extract KasmVNC port from browser_vnc_url, or None."""
        if container_manager is None:
            return None
        url = getattr(container_manager, "browser_vnc_url", None)
        if not url:
            return None
        try:
            from urllib.parse import urlparse
            return urlparse(url).port
        except Exception:
            return None

    def _reject_agent_tokens(request: Request) -> None:
        """Block agent Bearer tokens from VNC routes.

        Dashboard users are authenticated by Caddy's forward_auth (session
        cookie) — they never send Bearer tokens.  Agent containers always
        send Bearer tokens.  Rejecting known agent tokens here prevents
        untrusted agents from accessing VNC via the mesh port.
        No-op in dev/test mode (no auth tokens configured).
        """
        if not _auth_tokens:
            return
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            for expected in _auth_tokens.values():
                if hmac.compare_digest(token, expected):
                    raise HTTPException(403, "Agent access denied")

    @app.get("/vnc/{path:path}")
    async def vnc_http_proxy(path: str, request: Request):
        """Reverse-proxy HTTP requests to KasmVNC (static files)."""
        _reject_agent_tokens(request)
        from src.dashboard.auth import verify_session_cookie
        cookie_value = request.cookies.get("ol_session", "")
        auth_error = verify_session_cookie(cookie_value)
        if auth_error is not None:
            raise HTTPException(401, auth_error)
        import httpx

        port = _get_vnc_port()
        if port is None:
            raise HTTPException(502, "Browser service not available")
        query = str(request.url.query)
        target = f"http://127.0.0.1:{port}/{path}"
        if query:
            target += f"?{query}"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(target)
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            logger.warning("VNC HTTP proxy failed to reach %s: %s", target, exc)
            raise HTTPException(502, "Browser VNC not reachable")
        headers = {}
        ct = resp.headers.get("content-type")
        if ct:
            headers["content-type"] = ct
        return StreamingResponse(
            iter([resp.content]),
            status_code=resp.status_code,
            headers=headers,
        )

    @app.websocket("/vnc/{path:path}")
    async def vnc_ws_proxy(websocket: WebSocket, path: str):
        """Reverse-proxy WebSocket connections to KasmVNC."""
        # Verify session cookie
        from src.dashboard.auth import verify_session_cookie
        cookie_value = websocket.cookies.get("ol_session", "")
        auth_error = verify_session_cookie(cookie_value)
        if auth_error is not None:
            await websocket.close(code=1008, reason=auth_error)
            return
        # Block agent tokens — could arrive as header or query param.
        if _auth_tokens:
            token = websocket.query_params.get("token", "")
            auth_header = websocket.headers.get("authorization", "")
            if auth_header.startswith("Bearer "):
                token = token or auth_header[7:]
            if token:
                for expected in _auth_tokens.values():
                    if hmac.compare_digest(token, expected):
                        await websocket.close(code=1008, reason="Agent access denied")
                        return
        port = _get_vnc_port()
        if port is None:
            await websocket.close(code=1011, reason="Browser service not available")
            return

        # KasmVNC requires Origin header for /websockify (even with -disableBasicAuth).
        target = f"ws://127.0.0.1:{port}/websockify"
        extra_headers = {
            "Origin": f"http://127.0.0.1:{port}",
        }

        await websocket.accept(subprotocol="binary")
        try:
            import websockets

            async with websockets.connect(
                target,
                subprotocols=["binary"],
                additional_headers=extra_headers,
                compression=None,
                ping_interval=None,  # noVNC handles keepalive; websockets' auto-ping
                                     # drops the connection when KasmVNC doesn't pong
            ) as upstream:

                async def client_to_upstream():
                    try:
                        while True:
                            msg = await websocket.receive()
                            if "bytes" in msg and msg["bytes"]:
                                await upstream.send(msg["bytes"])
                            elif "text" in msg and msg["text"]:
                                await upstream.send(msg["text"])
                    except Exception as e:
                        logger.warning("VNC client→upstream error: %s", e)

                async def upstream_to_client():
                    try:
                        async for msg in upstream:
                            if isinstance(msg, bytes):
                                await websocket.send_bytes(msg)
                            else:
                                await websocket.send_text(msg)
                    except Exception as e:
                        logger.warning("VNC upstream→client error: %s", e)

                async def browser_keepalive():
                    """Touch all browser instances every 5 min while VNC is open.

                    Prevents idle cleanup from killing a browser that a user
                    is actively viewing without any agent operations running.
                    """
                    svc_url = getattr(container_manager, "browser_service_url", None)
                    svc_token = getattr(container_manager, "browser_auth_token", "")
                    if not svc_url:
                        return
                    try:
                        import httpx as _httpx
                        async with _httpx.AsyncClient(timeout=5) as _client:
                            while True:
                                await asyncio.sleep(30)
                                with contextlib.suppress(Exception):
                                    await _client.post(
                                        f"{svc_url}/browser/keepalive",
                                        headers={"Authorization": f"Bearer {svc_token}"},
                                    )
                    except asyncio.CancelledError:
                        pass

                tasks = [
                    asyncio.create_task(client_to_upstream()),
                    asyncio.create_task(upstream_to_client()),
                    asyncio.create_task(browser_keepalive()),
                ]
                _done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
        except Exception as exc:
            logger.warning("VNC WebSocket proxy error: %s", exc)
        finally:
            with contextlib.suppress(Exception):
                await websocket.close()

    return app
