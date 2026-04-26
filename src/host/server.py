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
import uuid as _uuid
from collections import defaultdict
from collections.abc import Callable, Coroutine
from datetime import datetime, timedelta, timezone
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


# ── Pending Config Change Store (in-memory) ──────────────────────

_pending_changes: dict[str, dict] = {}
_MAX_PENDING = 10
_CHANGE_TTL_SECONDS = 300


def _cleanup_expired_changes() -> None:
    now = datetime.now(timezone.utc)
    expired = [k for k, v in _pending_changes.items() if v["expires_at"] < now]
    for k in expired:
        del _pending_changes[k]


def _store_pending_change(agent_id: str, field: str, old_value: object, new_value: object) -> str:
    _cleanup_expired_changes()
    if len(_pending_changes) >= _MAX_PENDING:
        oldest = min(_pending_changes, key=lambda k: _pending_changes[k]["expires_at"])
        del _pending_changes[oldest]
    change_id = str(_uuid.uuid4())
    _pending_changes[change_id] = {
        "agent_id": agent_id, "field": field,
        "old_value": old_value, "new_value": new_value,
        "expires_at": datetime.now(timezone.utc) + timedelta(seconds=_CHANGE_TTL_SECONDS),
    }
    return change_id


def _get_pending_change(change_id: str) -> dict | None:
    _cleanup_expired_changes()
    return _pending_changes.get(change_id)


def _consume_pending_change(change_id: str) -> dict | None:
    _cleanup_expired_changes()
    return _pending_changes.pop(change_id, None)


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
        "upload_stage": (30, 60),
        "upload_apply": (30, 60),
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

    def _resolve_browser_target(caller_id: str, target_claim: object) -> str:
        """Resolve the effective browser-target agent_id for self/delegation paths.

        - ``None``/empty/whitespace target_claim → self path (returns ``caller_id``).
        - target_claim equal to caller → self path (returns ``caller_id``).
        - Otherwise delegation: requires the caller to be permitted to
          message the target AND the target to have ``can_use_browser``.
          Raises ``HTTPException(403)`` on either gate failure.
        - Non-string target_claim (e.g. list, dict, int) → ``HTTPException(400)``.
          Callers pull this value out of the JSON body where it can be any
          type, so we defensively reject non-strings rather than crashing
          with ``AttributeError`` on ``.strip()``.

        Note: ``can_message`` semantically grants "send a chat message".
        Reusing it for browser delegation is intentional but means a
        worker that can message a peer can also navigate that peer's
        browser. Endpoints that call this helper accept that coupling.
        """
        if target_claim is None or target_claim == "":
            return caller_id
        if not isinstance(target_claim, str):
            raise HTTPException(400, "target_agent_id must be a string")
        target = target_claim.strip()
        if not target or target == caller_id:
            return caller_id
        if not permissions.can_message(caller_id, target):
            raise HTTPException(
                403,
                "Cannot delegate browser: target is not in your can_message allowlist",
            )
        if not permissions.can_use_browser(target):
            raise HTTPException(
                403,
                "Cannot delegate browser: target agent has no browser access",
            )
        return target

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

    @app.post("/mesh/wake")
    async def wake_agent(
        request: Request, target: str = "", message: str = "",
    ) -> dict:
        """Wake a target agent by enqueuing a followup message via lanes.

        Used by hand_off to prompt the target agent to check its inbox
        immediately instead of waiting for the next heartbeat.
        """
        caller = _extract_verified_agent_id(request)
        if not target:
            raise HTTPException(400, "target is required")
        if not permissions.can_message(caller, target):
            raise HTTPException(403, f"Agent {caller} cannot wake {target}")
        await _check_rate_limit("blackboard_write", caller)  # reuse bb rate limit
        if target not in router.agent_registry:
            raise HTTPException(404, f"Agent '{target}' not registered")

        if message:
            wake_msg = sanitize_for_prompt(message)
        else:
            wake_msg = f"You have a new task from {caller}. Call check_inbox() to see it."

        from src.shared.trace import parse_origin_header
        origin = parse_origin_header(request.headers.get("x-origin"))

        if lane_manager is not None and dispatch_loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(
                    lane_manager.enqueue(
                        target, wake_msg, mode="followup",
                        origin=origin, auto_notify=origin is not None,
                    ),
                    dispatch_loop,
                )
            except Exception as e:
                logger.warning("Wake enqueue for %s failed: %s", target, e)
                return {"woken": False, "error": str(e)}
            return {"woken": True, "target": target}
        # Fallback: send via router (message-only, no task processing)
        await router.route(AgentMessage(
            from_agent="mesh", to=target, type="coordination",
            payload={"wake": True, "message": sanitize_for_prompt(wake_msg)},
        ))
        return {"woken": True, "target": target, "fallback": True}

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
        try:
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
        except Exception:
            logger.error(
                "Post-processing failed (trace) for %s/%s agent=%s",
                api_request.service, api_request.action, agent_id, exc_info=True,
            )
        try:
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
        except Exception:
            logger.error(
                "Post-processing failed (events) for %s/%s agent=%s",
                api_request.service, api_request.action, agent_id, exc_info=True,
            )
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

        try:
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
        except Exception:
            logger.error(
                "Post-processing failed (trace) for %s/%s agent=%s",
                api_request.service, api_request.action, agent_id, exc_info=True,
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
            try:
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
                if event_bus is not None and credit_error:
                    event_bus.emit("credit_exhausted", agent=agent_id, data={
                        "error": "Insufficient credits",
                    })
            except Exception:
                logger.error(
                    "Post-processing failed (events) for %s/%s agent=%s",
                    api_request.service, api_request.action, agent_id, exc_info=True,
                )
            try:
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
            except Exception:
                logger.error(
                    "Post-processing failed (trace) for %s/%s agent=%s",
                    api_request.service, api_request.action, agent_id, exc_info=True,
                )

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

    @app.post("/mesh/credential-request")
    async def credential_request(data: dict, request: Request) -> dict:
        """Agent requests a credential from the user via dashboard UI.

        Emits a ``credential_request`` event so the dashboard renders a
        secure input card.  The credential value is never part of the
        event — only name, description, and service travel over the wire.
        """
        agent_id = _resolve_agent_id(data.get("agent_id", ""), request)
        await _check_rate_limit("notify", agent_id)

        name = data.get("name", "")
        description = data.get("description", "")
        service = data.get("service", name)

        if not name:
            raise HTTPException(400, "Credential name is required")
        if not re.match(r"^[a-zA-Z0-9_.\-]{1,128}$", name):
            raise HTTPException(400, "Invalid credential name")

        if event_bus:
            event_bus.emit(
                "credential_request",
                agent=agent_id,
                data={
                    "name": name,
                    "description": description[:500],
                    "service": service[:128],
                },
            )

        return {"requested": True, "name": name}

    @app.post("/mesh/browser-login-request")
    async def browser_login_request(data: dict, request: Request) -> dict:
        """Agent requests user login via browser VNC viewer in chat.

        Emits a ``browser_login_request`` event so the dashboard renders
        an interactive browser card with VNC viewer.

        Supports delegation via ``target_agent_id``: when an orchestrator
        (e.g. operator) sets up a login for a worker, the caller passes
        the worker's ID and we emit the event under the worker's identity
        so the dashboard's existing cross-surfacing logic routes it
        correctly. Session cookies must land in the profile of the agent
        that will use them — operator itself owns no browser, so it
        delegates to the worker whose profile the cookies must target.
        """
        caller_id = _resolve_agent_id(data.get("agent_id", ""), request)
        agent_id = _resolve_browser_target(
            caller_id, data.get("target_agent_id") or "",
        )

        # Rate-limit on the caller, not the target — otherwise a noisy
        # caller could exhaust an unrelated worker's notify quota.
        await _check_rate_limit("notify", caller_id)

        url = data.get("url", "").strip()
        service = data.get("service", "").strip()
        description = data.get("description", "").strip()

        if not url:
            raise HTTPException(400, "URL is required")
        if not service:
            raise HTTPException(400, "Service name is required")

        if event_bus:
            event_bus.emit(
                "browser_login_request",
                agent=agent_id,
                data={
                    "url": url[:2048],
                    "service": service[:128],
                    "description": description[:500],
                },
            )

        return {"requested": True, "service": service, "target_agent": agent_id}

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
            entry: dict = {"url": url, "role": router.agent_roles.get(aid, "")}
            proj = _agent_projects.get(aid)
            if proj:
                entry["project"] = proj
            return entry

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
            # Exception: operator sees all agents (it manages the entire fleet).

            # Operator also needs per-agent model so dashboard-initiated
            # model changes don't leave its mental state stale. Load the
            # YAML once; other agents don't see models (noise for peers).
            include_models = agent_id == "operator"
            agent_models: dict[str, str] = {}
            if include_models:
                from src.cli.config import _load_config
                _cfg = _load_config()
                _agents_cfg = _cfg.get("agents", {})
                _default_model = _cfg.get("llm", {}).get(
                    "default_model", "openai/gpt-4o-mini",
                )
                agent_models = {
                    aid: _agents_cfg.get(aid, {}).get("model", _default_model)
                    for aid in router.agent_registry
                }

            def _fleet_entry(aid: str) -> dict:
                entry: dict = {
                    "id": aid, "role": router.agent_roles.get(aid, ""),
                }
                if include_models:
                    entry["model"] = agent_models.get(aid, "")
                return entry

            if agent_id == "operator":
                result["fleet"] = [
                    _fleet_entry(aid) for aid in router.agent_registry
                ]
            else:
                from src.cli.config import _load_projects
                _projects = _load_projects()
                _agent_project_members: set[str] | None = None
                for _pdata in _projects.values():
                    if agent_id in _pdata.get("members", []):
                        _agent_project_members = set(_pdata["members"])
                        break

                if _agent_project_members is not None:
                    result["fleet"] = [
                        _fleet_entry(aid)
                        for aid in router.agent_registry
                        if aid in _agent_project_members
                    ]
                else:
                    result["fleet"] = [_fleet_entry(agent_id)]

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

        if section in ("project", "all"):
            result["project"] = _agent_projects.get(agent_id)

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


    # === Fleet Templates ===

    @app.get("/mesh/fleet/templates")
    async def list_fleet_templates(request: Request) -> dict:
        """List available fleet templates."""
        _require_any_auth(request)
        from src.cli.config import _load_templates

        templates = _load_templates()
        result = []
        for name, tpl in templates.items():
            result.append({
                "name": name,
                "description": tpl.get("description", ""),
                "agent_count": len(tpl.get("agents", {})),
                "agents": list(tpl.get("agents", {}).keys()),
            })
        return {"templates": result}

    @app.post("/mesh/fleet/apply")
    async def apply_fleet_template(data: dict, request: Request) -> dict:
        """Create a team of agents from a fleet template.

        Body: {template: str, model?: str}
        """
        if container_manager is None:
            raise HTTPException(503, "Container manager not available")

        # Auth + permission check
        spawned_by = _resolve_agent_id(data.get("spawned_by", "unknown"), request)
        if _auth_tokens:
            # In authenticated mode, require spawn permission
            if not permissions.can_spawn(spawned_by):
                raise HTTPException(403, f"Agent {spawned_by} is not allowed to spawn agents")

        template_name = data.get("template", "").strip()
        if not template_name:
            raise HTTPException(400, "template is required")

        from src.cli.config import _apply_template, _load_config, _load_templates

        templates = _load_templates()
        tpl = templates.get(template_name)
        if tpl is None:
            raise HTTPException(404, f"Template not found: {template_name}")

        tpl_agents = tpl.get("agents", {})
        if not tpl_agents:
            raise HTTPException(400, f"Template '{template_name}' has no agents")

        # Check plan limits (exclude operator from count)
        import os as _os
        max_agents = int(_os.environ.get("OPENLEGION_MAX_AGENTS", "0"))
        if max_agents > 0:
            current_count = sum(
                1 for aid in router.agent_registry
                if aid != "operator"
            )
            if current_count + len(tpl_agents) > max_agents:
                raise HTTPException(
                    403,
                    f"Would exceed agent limit ({current_count} + {len(tpl_agents)} > {max_agents}). "
                    "Upgrade your plan for more agents.",
                )

        # Optional model override
        model_override = data.get("model", "")

        # Apply template to create config entries
        created_names = _apply_template(template_name, tpl)
        if not created_names:
            return {
                "template": template_name,
                "created": [],
                "skipped": list(tpl_agents.keys()),
                "message": "All agents already exist",
            }

        # Assign random unique avatars to newly created agents
        import random

        from src.cli.config import _update_agent_field
        used_avatars: set[int] = set()
        for agent_name in created_names:
            avatar = random.choice([i for i in range(1, 51) if i not in used_avatars])
            used_avatars.add(avatar)
            _update_agent_field(agent_name, "avatar", avatar)

        # Load config to get per-agent settings
        cfg = _load_config()
        agents_cfg = cfg.get("agents", {})
        default_model = cfg.get("llm", {}).get("default_model", "openai/gpt-4o-mini")
        hb_schedule = cfg.get("mesh", {}).get("heartbeat_schedule")

        created_agents = []
        failed_agents = []

        for agent_name in created_names:
            acfg = agents_cfg.get(agent_name, {})
            agent_model = model_override or acfg.get("model", default_model)
            skills_dir = str(
                (container_manager.project_root / "skills" / agent_name).resolve()
            ) if container_manager.project_root else ""

            # Build per-agent env_overrides (NOT shared extra_env)
            env_overrides: dict[str, str] = {}
            for env_key, cfg_key in (
                ("INITIAL_INSTRUCTIONS", "initial_instructions"),
                ("INITIAL_SOUL", "initial_soul"),
                ("INITIAL_HEARTBEAT", "initial_heartbeat"),
                ("INITIAL_INTERFACE", "initial_interface"),
            ):
                val = acfg.get(cfg_key, "")
                if val:
                    env_overrides[env_key] = val

            try:
                # Start container with per-agent env_overrides (not shared extra_env)
                url = container_manager.start_agent(
                    agent_id=agent_name,
                    role=acfg.get("role", agent_name),
                    skills_dir=skills_dir,
                    model=agent_model,
                    thinking=acfg.get("thinking", ""),
                    env_overrides=env_overrides,
                )

                # Register with router, transport, health, cron
                router.register_agent(agent_name, url, role=acfg.get("role", ""))
                if transport is not None:
                    from src.host.transport import HttpTransport
                    if isinstance(transport, HttpTransport):
                        transport.register(agent_name, url)
                if health_monitor is not None:
                    health_monitor.register(agent_name)
                if cron_scheduler is not None:
                    cron_scheduler.ensure_heartbeat(agent_name, hb_schedule)

                # Wait for readiness
                ready = await container_manager.wait_for_agent(agent_name, timeout=60)

                if event_bus is not None:
                    event_bus.emit("agent_state", agent=agent_name, data={
                        "state": "added", "role": acfg.get("role", ""), "ready": ready,
                    })

                created_agents.append({
                    "agent_id": agent_name,
                    "role": acfg.get("role", agent_name),
                    "ready": ready,
                })
            except Exception as e:
                logger.error("Failed to start agent '%s' from template: %s", agent_name, e)
                failed_agents.append({"agent_id": agent_name, "error": str(e)})

        return {
            "template": template_name,
            "created": created_agents,
            "failed": failed_agents,
            "skipped": [n for n in tpl_agents if n not in created_names],
        }

    # === Create Custom Agent ===

    @app.post("/mesh/agents/create")
    async def create_custom_agent(data: dict, request: Request) -> dict:
        """Create a new custom agent. Used by the operator."""
        if container_manager is None:
            raise HTTPException(503, "Container manager not available")

        # Auth + spawn permission
        agent_id = _resolve_agent_id(data.get("agent_id", ""), request)
        await _check_rate_limit("spawn", agent_id)
        if not permissions.can_spawn(agent_id):
            raise HTTPException(403, f"Agent {agent_id} is not allowed to create agents")

        # Validate inputs
        name = data.get("name", "")
        role = data.get("role", "")
        model = data.get("model", "")
        instructions = data.get("instructions", "")
        soul = data.get("soul", "")

        if not name or not isinstance(name, str):
            raise HTTPException(400, "name is required")

        # Validate agent name format
        from src.cli.config import _validate_agent_name
        try:
            name = _validate_agent_name(name)
        except Exception as e:
            raise HTTPException(400, f"Invalid agent name: {e}") from e

        # Check if agent already exists
        from src.cli.config import _load_config
        config = _load_config()
        if name in config.get("agents", {}):
            raise HTTPException(409, f"Agent '{name}' already exists")

        # Check plan limits (exclude operator)
        import os
        max_agents = int(os.environ.get("OPENLEGION_MAX_AGENTS", "0"))
        if max_agents > 0:
            current = len([a for a in config.get("agents", {}) if a != "operator"])
            if current >= max_agents:
                raise HTTPException(
                    409,
                    f"Plan limit reached ({current}/{max_agents} agents). "
                    "Remove an agent or upgrade your plan.",
                )

        # Default model
        if not model:
            model = config.get("llm", {}).get("default_model", "openai/gpt-4o-mini")

        # Create agent config
        import random

        from src.cli.config import (
            PROJECT_ROOT,
            _add_agent_permissions,
            _add_agent_to_config,
            _update_agent_field,
        )
        _add_agent_to_config(
            name=name, role=role or name, model=model,
            initial_instructions=instructions, initial_soul=soul,
        )
        _update_agent_field(name, "avatar", random.randint(1, 50))
        _add_agent_permissions(name)
        skills_dir = PROJECT_ROOT / "skills" / name
        skills_dir.mkdir(parents=True, exist_ok=True)

        # Start container using env_overrides pattern
        agent_env: dict[str, str] = {}
        if instructions:
            agent_env["INITIAL_INSTRUCTIONS"] = instructions
        if soul:
            agent_env["INITIAL_SOUL"] = soul

        try:
            url = container_manager.start_agent(
                agent_id=name, role=role or name,
                skills_dir=str(skills_dir), model=model,
                env_overrides=agent_env,
            )
        except Exception as e:
            # Roll back: remove config and permissions so the name isn't blocked
            from src.cli.config import _remove_agent
            try:
                _remove_agent(name)
            except Exception:
                pass
            import shutil
            shutil.rmtree(skills_dir, ignore_errors=True)
            raise HTTPException(500, f"Failed to start agent container: {e}") from e

        try:
            router.register_agent(name, url, role=role or name)
            if transport is not None:
                from src.host.transport import HttpTransport
                if isinstance(transport, HttpTransport):
                    transport.register(name, url)
            if health_monitor is not None:
                health_monitor.register(name)
            if cron_scheduler is not None:
                hb_schedule = config.get("mesh", {}).get("heartbeat_schedule")
                cron_scheduler.ensure_heartbeat(name, hb_schedule)

            ready = await container_manager.wait_for_agent(name, timeout=60)

            if event_bus is not None:
                event_bus.emit("agent_state", agent=name,
                    data={"state": "added", "role": role, "ready": ready})

            if trace_store:
                from src.shared.trace import new_trace_id as _new_trace_id
                trace_store.record(
                    trace_id=_new_trace_id(), source="mesh.create_agent",
                    agent=name, event_type="create_agent",
                    detail=f"role={role}, model={model}, created_by={data.get('created_by', 'operator')}",
                )

            return {"agent_id": name, "role": role or name, "ready": ready}
        except Exception as e:
            # Roll back: stop container, remove config so the name isn't blocked
            try:
                container_manager.stop_agent(name)
            except Exception:
                pass
            from src.cli.config import _remove_agent
            try:
                _remove_agent(name)
            except Exception:
                pass
            import shutil
            shutil.rmtree(skills_dir, ignore_errors=True)
            raise HTTPException(500, f"Failed to register agent: {e}") from e


    # === Agent History Access ===

    _PERIOD_TO_DAYS = {"today": 1, "yesterday": 2, "week": 7}

    @app.get("/mesh/agents/{agent_id}/history")
    async def get_agent_history(
        agent_id: str, request: Request,
        requesting_agent: str = "", period: str = "",
    ) -> dict:
        """Retrieve an agent's daily logs. Permission-checked."""
        if requesting_agent:
            requesting_agent = _resolve_agent_id(requesting_agent, request)
            if not permissions.can_message(requesting_agent, agent_id):
                raise HTTPException(403, f"Agent {requesting_agent} cannot read history of {agent_id}")
        else:
            _require_any_auth(request)
            logger.debug("History access for %s without requesting_agent (mesh-internal)", agent_id)
        agent_entry = router.agent_registry.get(agent_id)
        if not agent_entry:
            raise HTTPException(404, f"Agent not found: {agent_id}")
        days = _PERIOD_TO_DAYS.get(period, 3)
        history_qs = f"/history?days={days}"
        if transport is not None:
            try:
                return await transport.request(agent_id, "GET", history_qs, timeout=10)
            except Exception as e:
                raise HTTPException(502, f"Failed to fetch history from {agent_id}: {e}") from e
        agent_url = agent_entry.get("url", agent_entry) if isinstance(agent_entry, dict) else agent_entry
        import httpx
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{agent_url}/history", params={"days": days})
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

    # === Pre-computed Metrics (for operator heartbeat) ===

    @app.get("/mesh/system/metrics")
    async def system_metrics(request: Request) -> dict:
        """Fleet-wide aggregate metrics for operator heartbeat.

        Pre-computes ratios and flags so the operator LLM doesn't need
        to do arithmetic.  Read-only — no mutations.
        """
        _require_any_auth(request)

        # -- Agent counts (exclude operator — it's a system agent, not a user slot) --
        agents = dict(router.agent_registry)
        total = sum(1 for aid in agents if aid != "operator")

        # -- Health breakdown (exclude operator) --
        health_list = health_monitor.get_status() if health_monitor else []
        health_list_user = [s for s in health_list if s.get("agent") != "operator"]
        healthy = sum(1 for s in health_list_user if s.get("status") == "healthy")
        failed = sum(1 for s in health_list_user if s.get("status") == "failed")

        # -- Busy count from lane manager (exclude operator) --
        lane_status = lane_manager.get_status() if lane_manager else {}
        busy = sum(1 for aid, ls in lane_status.items() if ls.get("busy", False) and aid != "operator")

        # -- Cost data --
        cost_today = 0.0
        cost_yesterday = 0.0
        if cost_tracker:
            today_spend = cost_tracker.get_spend(None, "today")
            cost_today = today_spend.get("total_cost", 0.0)
            # "yesterday" returns spend since yesterday midnight (includes today).
            # Subtract today's spend to get yesterday-only spend.
            since_yesterday = cost_tracker.get_spend(None, "yesterday")
            cost_yesterday = max(since_yesterday.get("total_cost", 0.0) - cost_today, 0.0)

        cost_ratio = round(cost_today / cost_yesterday, 2) if cost_yesterday > 0 else 0

        # -- Per-agent failure rates (placeholder — needs task tracking) --
        failure_rates: dict[str, float] = {}

        # -- Agents needing attention (exclude operator) --
        agents_attention: list[dict] = []
        for status_entry in health_list_user:
            agent_status = status_entry.get("status", "unknown")
            if agent_status in ("failed", "unhealthy"):
                agents_attention.append({
                    "agent_id": status_entry["agent"],
                    "issue": agent_status,
                    "failures": status_entry.get("failures", 0),
                    "restarts": status_entry.get("restarts", 0),
                })

        # -- Plan limits from env vars --
        import os
        max_agents = int(os.environ.get("OPENLEGION_MAX_AGENTS", "0"))
        max_projects = int(os.environ.get("OPENLEGION_MAX_PROJECTS", "0"))

        # Count actual projects
        from src.cli.config import _load_projects
        current_projects = len(_load_projects())

        return {
            "total_agents": total,
            "healthy": healthy,
            "failed": failed,
            "busy": busy,
            "total_cost_today_usd": round(cost_today, 4),
            "cost_vs_yesterday_ratio": cost_ratio,
            "failure_rate_by_agent": failure_rates,
            "agents_needing_attention": agents_attention,
            "plan_limits": {
                "max_agents": max_agents,
                "current_agents": total,
                "max_projects": max_projects,
                "current_projects": current_projects,
            },
        }

    @app.get("/mesh/agents/{agent_id}/metrics")
    async def agent_metrics(agent_id: str, request: Request) -> dict:
        """Per-agent pre-computed metrics for operator heartbeat.

        Returns health, cost, and queue data for a single agent.
        Read-only — no mutations.
        """
        _require_any_auth(request)
        _validate_agent_id(agent_id)

        if agent_id not in router.agent_registry:
            raise HTTPException(404, f"Agent not found: {agent_id}")

        # -- Health --
        health_status = "unknown"
        failures = 0
        restarts = 0
        if health_monitor:
            statuses = health_monitor.get_status()
            match = next((s for s in statuses if s["agent"] == agent_id), None)
            if match:
                health_status = match.get("status", "unknown")
                failures = match.get("failures", 0)
                restarts = match.get("restarts", 0)

        # -- Cost --
        cost_today = 0.0
        cost_yesterday = 0.0
        if cost_tracker:
            today_spend = cost_tracker.get_spend(agent_id, "today")
            cost_today = today_spend.get("total_cost", 0.0)
            since_yesterday = cost_tracker.get_spend(agent_id, "yesterday")
            cost_yesterday = max(since_yesterday.get("total_cost", 0.0) - cost_today, 0.0)

        cost_ratio = round(cost_today / cost_yesterday, 2) if cost_yesterday > 0 else 0

        # -- Queue / busy --
        queued = 0
        is_busy = False
        if lane_manager:
            ls = lane_manager.get_status().get(agent_id, {})
            queued = ls.get("queued", 0)
            is_busy = ls.get("busy", False)

        # -- Budget --
        budget = cost_tracker.check_budget(agent_id) if cost_tracker else {}

        return {
            "agent_id": agent_id,
            "health_status": health_status,
            "consecutive_failures": failures,
            "restart_count": restarts,
            "cost_today_usd": round(cost_today, 4),
            "cost_vs_yesterday_ratio": cost_ratio,
            "budget": budget,
            "queued_tasks": queued,
            "busy": is_busy,
            "tasks_completed_24h": 0,
            "tasks_failed_24h": 0,
            "failure_rate": 0.0,
            "avg_task_duration_s": 0,
        }

    # === Mesh Project Proxy Endpoints ===
    # These proxy the dashboard's /api/projects/* endpoints through the mesh
    # so operator agents can manage projects using their mesh auth token.

    @app.get("/mesh/projects")
    async def mesh_list_projects(request: Request) -> dict:
        """List all projects (mesh-authed proxy)."""
        _require_any_auth(request)
        if _resolve_agent_id("", request) != "operator":
            raise HTTPException(403, "Only the operator can manage projects")
        from src.cli.config import _load_projects
        projects = _load_projects()
        result = []
        for pname, pdata in sorted(projects.items(), key=lambda x: x[1].get("created_at") or ""):
            result.append({
                "name": pname,
                "description": pdata.get("description", ""),
                "members": pdata.get("members", []),
                "created_at": pdata.get("created_at", ""),
            })
        return {"projects": result}

    @app.post("/mesh/projects")
    async def mesh_create_project(request: Request) -> dict:
        """Create a new project (mesh-authed proxy)."""
        _require_any_auth(request)
        if _resolve_agent_id("", request) != "operator":
            raise HTTPException(403, "Only the operator can manage projects")
        import os as _os

        from src.cli.config import _create_project, _load_config, _load_projects

        _max_projects_env = _os.environ.get("OPENLEGION_MAX_PROJECTS")
        if _max_projects_env is not None:
            _max_projects = int(_max_projects_env)
            if _max_projects == 0:
                raise HTTPException(
                    403,
                    "Projects are not available on your plan. Upgrade for project support.",
                )
            current_count = len(_load_projects())
            if current_count >= _max_projects:
                raise HTTPException(
                    403,
                    f"Project limit reached ({_max_projects}). Upgrade your plan for more projects.",
                )

        body = await request.json()
        name = body.get("name", "").strip()
        description = sanitize_for_prompt(body.get("description", "")).strip()
        members = body.get("members", [])
        if not name:
            raise HTTPException(400, "name is required")
        if not isinstance(members, list):
            raise HTTPException(400, "members must be a list")
        cfg = _load_config()
        known_agents = set(cfg.get("agents", {}).keys())
        unknown = [m for m in members if m not in known_agents]
        if unknown:
            raise HTTPException(400, f"Unknown agents: {', '.join(unknown)}")
        try:
            _create_project(name, description=description, members=members)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"created": True, "name": name}

    @app.post("/mesh/projects/{name}/members")
    async def mesh_add_project_member(name: str, request: Request) -> dict:
        """Add an agent to a project (mesh-authed proxy)."""
        _require_any_auth(request)
        if _resolve_agent_id("", request) != "operator":
            raise HTTPException(403, "Only the operator can manage projects")
        from src.cli.config import _add_agent_to_project
        body = await request.json()
        agent = body.get("agent", "").strip()
        if not agent:
            raise HTTPException(400, "agent is required")
        try:
            _add_agent_to_project(name, agent)
        except ValueError as e:
            raise HTTPException(400, str(e))
        # Update in-memory project mapping so scoping takes effect immediately
        _agent_projects[agent] = name
        return {"added": True, "project": name, "agent": agent}

    @app.delete("/mesh/projects/{name}/members/{agent}")
    async def mesh_remove_project_member(name: str, agent: str, request: Request) -> dict:
        """Remove an agent from a project (mesh-authed proxy)."""
        _require_any_auth(request)
        if _resolve_agent_id("", request) != "operator":
            raise HTTPException(403, "Only the operator can manage projects")
        from src.cli.config import _remove_agent_from_project
        try:
            _remove_agent_from_project(name, agent)
        except ValueError as e:
            raise HTTPException(400, str(e))
        _agent_projects.pop(agent, None)
        return {"removed": True, "project": name, "agent": agent}

    @app.delete("/mesh/projects/{name}")
    async def mesh_delete_project(name: str, request: Request) -> dict:
        """Delete a project (mesh-authed proxy)."""
        _require_any_auth(request)
        if _resolve_agent_id("", request) != "operator":
            raise HTTPException(403, "Only the operator can manage projects")
        from src.cli.config import _delete_project
        try:
            _delete_project(name)
        except ValueError as e:
            raise HTTPException(404, str(e))
        return {"deleted": True, "name": name}

    @app.put("/mesh/projects/{name}/context")
    async def mesh_update_project_context(name: str, request: Request) -> dict:
        """Update a project's description/context (mesh-authed proxy)."""
        _require_any_auth(request)
        if _resolve_agent_id("", request) != "operator":
            raise HTTPException(403, "Only the operator can manage projects")
        from src.cli.config import PROJECTS_DIR, _load_projects
        body = await request.json()
        context = sanitize_for_prompt(body.get("context", "")).strip()

        projects = _load_projects()
        if name not in projects:
            raise HTTPException(404, f"Project '{name}' not found")

        # Update metadata.yaml description in place (never delete the project)
        import yaml
        meta_file = PROJECTS_DIR / name / "metadata.yaml"
        if meta_file.exists():
            with open(meta_file) as f:
                meta = yaml.safe_load(f) or {}
            meta["description"] = context
            with open(meta_file, "w") as f:
                yaml.dump(meta, f, default_flow_style=False, sort_keys=False)

        # Update project.md in place
        project_md = PROJECTS_DIR / name / "project.md"
        project_md.write_text(f"# {name}\n\n{context}\n")

        return {"updated": True, "project": name}

    # === Operator Config Endpoints ===

    _VALID_CONFIG_FIELDS = {
        "instructions", "soul", "model", "role", "heartbeat",
        "interface", "thinking", "budget", "permissions",
    }
    _CONFIG_FIELD_MAP = {
        "instructions": "initial_instructions", "soul": "initial_soul",
        "heartbeat": "initial_heartbeat",
        "interface": "initial_interface",
        "model": "model", "role": "role",
        "thinking": "thinking", "budget": "budget",
    }

    @app.get("/mesh/agents/{agent_id}/config")
    async def get_agent_config(agent_id: str, request: Request) -> dict:
        """Read agent config from agents.yaml."""
        _require_any_auth(request)
        if _resolve_agent_id("", request) != "operator":
            raise HTTPException(403, "Only the operator can read agent configs")
        from src.cli.config import _load_config
        agent_cfg = _load_config()
        agents = agent_cfg.get("agents", {})
        if agent_id not in agents:
            raise HTTPException(404, f"Agent '{agent_id}' not found")
        return {"agent_id": agent_id, "config": agents[agent_id]}

    @app.post("/mesh/agents/{agent_id}/propose")
    async def propose_agent_config_change(agent_id: str, request: Request) -> dict:
        """Create a pending config change for review."""
        _require_any_auth(request)
        if _resolve_agent_id("", request) != "operator":
            raise HTTPException(403, "Only the operator can propose config changes")
        data = await request.json()

        field = data.get("field", "")
        new_value = data.get("value")

        if not field:
            raise HTTPException(400, "field is required")
        if field not in _VALID_CONFIG_FIELDS:
            raise HTTPException(400, f"Invalid field: {field}. Must be one of: {_VALID_CONFIG_FIELDS}")

        # Validate runtime-critical values before they can be persisted to
        # YAML. The agent-side hot-reload endpoint also rejects these, but
        # catching them here prevents a bad value from being stored as a
        # pending change, confirmed, and surviving across restarts.
        if field == "model":
            if not isinstance(new_value, str) or not new_value:
                raise HTTPException(400, "model must be a non-empty string")
        elif field == "thinking":
            from src.agent.llm import LLMClient
            if new_value not in LLMClient.VALID_THINKING_LEVELS:
                raise HTTPException(
                    400,
                    f"thinking must be one of: {sorted(LLMClient.VALID_THINKING_LEVELS)}",
                )

        # Get current value
        from src.cli.config import _load_config
        agent_cfg = _load_config()
        agents = agent_cfg.get("agents", {})
        if agent_id not in agents:
            raise HTTPException(404, f"Agent '{agent_id}' not found")

        if field == "permissions":
            from src.cli.config import _load_permissions
            perms = _load_permissions()
            old_value = perms.get("permissions", {}).get(agent_id, {})
        else:
            yaml_key = _CONFIG_FIELD_MAP.get(field, field)
            old_value = agents[agent_id].get(yaml_key, "")

        change_id = _store_pending_change(agent_id, field, old_value, new_value)

        # Generate preview diff
        old_str = json.dumps(old_value, indent=2) if not isinstance(old_value, str) else old_value
        new_str = json.dumps(new_value, indent=2) if not isinstance(new_value, str) else new_value

        preview = f"--- {field} (current)\n+++ {field} (proposed)\n"
        if old_str != new_str:
            old_lines = old_str.splitlines()
            new_lines = new_str.splitlines()
            for line in old_lines:
                if line not in new_lines:
                    preview += f"- {line}\n"
            for line in new_lines:
                if line not in old_lines:
                    preview += f"+ {line}\n"

        return {
            "change_id": change_id,
            "preview_diff": preview,
            "expires_at": _pending_changes[change_id]["expires_at"].isoformat(),
        }

    async def _apply_pending_change(change_id: str, change: dict) -> dict:
        """Apply a consumed pending change — shared by both config endpoints."""
        agent_id = change["agent_id"]
        field = change["field"]
        old_value = change["old_value"]
        new_value = change["new_value"]

        import yaml

        from src.cli.config import AGENTS_FILE, _load_config

        agent_cfg = _load_config()
        agents = agent_cfg.get("agents", {})
        if agent_id not in agents:
            raise HTTPException(404, f"Agent '{agent_id}' not found")

        if field == "permissions":
            from src.cli.config import _load_permissions, _save_permissions
            perms = _load_permissions()
            if agent_id in perms.get("permissions", {}):
                perms["permissions"][agent_id].update(new_value)
                _save_permissions(perms)
            # Hot-reload the in-memory permission matrix
            if permissions is not None:
                permissions.reload()
        else:
            yaml_key = _CONFIG_FIELD_MAP.get(field, field)
            agents[agent_id][yaml_key] = new_value
            with open(AGENTS_FILE, "w") as f:
                yaml.dump(agent_cfg, f, default_flow_style=False, sort_keys=False)

        # Budget sync: update the in-memory cost tracker immediately
        if field == "budget" and cost_tracker is not None and isinstance(new_value, dict):
            _daily = new_value.get("daily_usd")
            _monthly = new_value.get("monthly_usd")
            if _daily is not None or _monthly is not None:
                cost_tracker.set_budget(
                    agent_id,
                    daily_usd=_daily,
                    monthly_usd=_monthly,
                )

        # Hot-reload: push to running agent's workspace
        if transport and agent_id in router.agent_registry:
            workspace_map = {
                "instructions": "INSTRUCTIONS.md",
                "soul": "SOUL.md",
                "heartbeat": "HEARTBEAT.md",
                "interface": "INTERFACE.md",
            }
            ws_file = workspace_map.get(field)
            if ws_file and isinstance(new_value, str):
                try:
                    await transport.request(
                        agent_id, "PUT", f"/workspace/{ws_file}",
                        json={"content": f"# {field.title()}\n\n{new_value}"},
                        timeout=10,
                    )
                except Exception:
                    pass  # Agent might not be running

        # Hot-reload: runtime config (model, thinking) — env-var fields that
        # won't get picked up by the YAML write alone.
        if (
            transport
            and agent_id in router.agent_registry
            and field in ("model", "thinking")
            and isinstance(new_value, str)
        ):
            try:
                result = await transport.request(
                    agent_id, "POST", "/config",
                    json={field: new_value}, timeout=10,
                )
                # transport.request returns {"error": ...} dicts for HTTP /
                # timeout / connect failures rather than raising. Surface
                # those so a silent agent-side failure isn't mistaken for
                # success — the YAML write is durable, but runtime state
                # only catches up on restart.
                if isinstance(result, dict) and "error" in result:
                    logger.warning(
                        "Hot-reload %s for '%s' returned error: %s",
                        field, agent_id, result["error"],
                    )
            except Exception as e:
                logger.warning(
                    "Failed to hot-reload %s for '%s': %s",
                    field, agent_id, e,
                )

        # Heartbeat schedule sync: if the value looks like a cron schedule
        # (5-field expression or "every Xm/h/d"), update the cron job too.
        # This handles the common case where the operator edits heartbeat
        # frequency — the cron scheduler schedule must match.
        if field == "heartbeat" and cron_scheduler is not None and isinstance(new_value, str):
            sched = new_value.strip()
            _is_schedule = (
                bool(re.fullmatch(r"every\s+\d+[smhd]", sched, re.IGNORECASE))
                or (len(sched.split()) == 5 and all(
                    re.match(r"^[\d,\-\*/]+$", p) for p in sched.split()
                ))
            )
            if _is_schedule:
                hb_job = cron_scheduler.find_heartbeat_job(agent_id)
                if hb_job:
                    try:
                        await cron_scheduler.update_job(hb_job.id, schedule=sched)
                        logger.info(
                            "Updated heartbeat schedule for '%s': %s",
                            agent_id, sched,
                        )
                    except Exception as e:
                        logger.warning(
                            "Failed to update heartbeat schedule for '%s': %s",
                            agent_id, e,
                        )

        blackboard.log_audit(
            action="edit_agent", target=agent_id, field=field,
            before_value=json.dumps(old_value) if not isinstance(old_value, str) else old_value,
            after_value=json.dumps(new_value) if not isinstance(new_value, str) else new_value,
            change_id=change_id,
        )

        return {"success": True, "agent_id": agent_id, "field": field}

    @app.post("/mesh/agents/{agent_id}/config")
    async def update_agent_config(agent_id: str, request: Request) -> dict:
        """Apply a pending config change. Used by confirm_edit tool."""
        _require_any_auth(request)
        if _resolve_agent_id("", request) != "operator":
            raise HTTPException(403, "Only the operator can apply config changes")
        data = await request.json()
        change_id = data.get("change_id", "")

        change = _get_pending_change(change_id)
        if not change:
            raise HTTPException(404, "Change not found or expired")
        if change["agent_id"] != agent_id:
            raise HTTPException(400, "Agent ID mismatch")
        # Consume only after validation passes
        _consume_pending_change(change_id)

        return await _apply_pending_change(change_id, change)

    @app.post("/mesh/config/confirm")
    async def confirm_config_change(request: Request) -> dict:
        """Apply a pending config change by change_id only.

        Unlike POST /mesh/agents/{agent_id}/config, this endpoint resolves
        the agent_id from the pending change itself so callers don't need
        to know the target agent_id up front.  Used by the operator's
        ``confirm_edit`` tool.
        """
        _require_any_auth(request)
        if _resolve_agent_id("", request) != "operator":
            raise HTTPException(403, "Only the operator can confirm config changes")
        data = await request.json()
        change_id = data.get("change_id", "")

        change = _get_pending_change(change_id)
        if not change:
            raise HTTPException(404, "Change not found or expired")
        # Consume before apply to prevent double-apply on retry. If
        # _apply_pending_change raises, the change is already gone — the
        # caller must propose a new change rather than retry this one.
        _consume_pending_change(change_id)

        return await _apply_pending_change(change_id, change)

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

        if proxy_url:
            parsed = parse_proxy_url(proxy_url)
            if parsed:
                body = {
                    "url": parsed["url"],
                    "username": parsed["username"],
                    "password": parsed["password"],
                }
            else:
                body = {}  # explicit no-proxy (URL failed validation)
        else:
            body = {}  # explicit no-proxy (direct mode or no system proxy)

        headers: dict = {"X-Mesh-Internal": "1"}
        if svc_token:
            headers["Authorization"] = f"Bearer {svc_token}"
        try:
            resp = await _browser_proxy_client.put(
                f"{svc_url}/browser/{agent_id}/proxy",
                json=body,
                headers=headers,
            )
            if resp.status_code >= 400:
                logger.warning("Browser proxy push for %s returned %d", agent_id, resp.status_code)
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

    # § Browser metrics poll loop. The browser service container can't push
    # to the mesh's in-process EventBus, so the mesh pulls. Runs on a 60s
    # cadence — matching BrowserManager._emit_metrics — and fans each new
    # per-agent aggregate out as a ``browser_metrics`` event. High-water
    # seq per boot_id; boot_id change resets the watermark. Poll failures
    # are logged at DEBUG normally; after ``_POLL_WARN_THRESHOLD`` consecutive
    # failures we escalate to WARNING so operators can diagnose persistent
    # issues (auth rotation, browser down, network partition).
    _poll_state: dict = {
        "boot_id": "",
        "last_seen_seq": 0,
        "consecutive_failures": 0,
        "last_success_ts": 0.0,
    }
    _POLL_WARN_THRESHOLD = 5  # ~5 minutes of failures before warning

    async def _poll_browser_metrics_once() -> None:
        if not container_manager or event_bus is None:
            return
        svc_url = getattr(container_manager, "browser_service_url", None)
        svc_token = getattr(container_manager, "browser_auth_token", "")
        if not svc_url:
            return
        headers: dict = {}
        if svc_token:
            headers["Authorization"] = f"Bearer {svc_token}"

        def _record_failure(reason: str) -> None:
            _poll_state["consecutive_failures"] += 1
            n = _poll_state["consecutive_failures"]
            # Quiet when it's the first couple of misses (normal during
            # boot / brief outages); loud once it looks like a real problem,
            # then back off so we don't flood logs indefinitely.
            if n in (_POLL_WARN_THRESHOLD, _POLL_WARN_THRESHOLD * 4,
                     _POLL_WARN_THRESHOLD * 16):
                logger.warning(
                    "Browser metrics poll has failed %d consecutive times "
                    "(%s)", n, reason,
                )
            else:
                logger.debug("Browser metrics poll failed: %s", reason)

        try:
            resp = await _browser_proxy_client.get(
                f"{svc_url}/browser/metrics",
                params={"since": _poll_state["last_seen_seq"]},
                headers=headers,
                timeout=10,
            )
        except Exception as e:
            _record_failure(f"request error: {e}")
            return
        if resp.status_code >= 400:
            _record_failure(f"HTTP {resp.status_code}")
            return
        try:
            data = resp.json()
        except Exception as e:
            _record_failure(f"bad JSON: {e}")
            return

        # Success — reset failure counter and log recovery if we were noisy.
        if _poll_state["consecutive_failures"] >= _POLL_WARN_THRESHOLD:
            logger.info(
                "Browser metrics poll recovered after %d failures",
                _poll_state["consecutive_failures"],
            )
        _poll_state["consecutive_failures"] = 0
        _poll_state["last_success_ts"] = time.time()

        boot_id = data.get("boot_id") or ""
        previous_boot_id = _poll_state["boot_id"]
        since_used = _poll_state["last_seen_seq"]
        is_first_seen = previous_boot_id != boot_id
        # Reset the watermark when the browser service restarts, otherwise
        # we'd starve forever waiting for seqs that never arrive.
        if is_first_seen:
            _poll_state["boot_id"] = boot_id
            _poll_state["last_seen_seq"] = 0
            # If we queried with a non-zero ``since`` (stale high-water
            # from the pre-restart browser) and the browser's ``seq > since``
            # filter dropped everything, we have to re-poll with ``since=0``
            # before those payloads scroll off the browser's history deque.
            # Skip the re-poll when: (a) since was already 0 — nothing got
            # filtered; or (b) the response already contains payloads — the
            # filter wasn't the problem. Both cases would double-emit.
            if since_used > 0 and previous_boot_id and not data.get("metrics"):
                try:
                    resp = await _browser_proxy_client.get(
                        f"{svc_url}/browser/metrics",
                        params={"since": 0},
                        headers=headers,
                        timeout=10,
                    )
                    if resp.status_code < 400:
                        data = resp.json()
                except Exception as e:
                    logger.debug("Post-restart re-poll failed: %s", e)
                    # Keep the original response rather than aborting;
                    # next tick will recover.

        payloads = [
            p for p in (data.get("metrics") or [])
            if int(p.get("seq", 0)) > _poll_state["last_seen_seq"]
        ]
        # On first-seen (fresh mesh or browser restart), only surface the
        # latest payload per (agent, kind). A long-running browser
        # service can return hours of history; flooding the dashboard's
        # 500-event ring buffer with stale entries would evict live
        # events. Collapse keys are ``(agent_id, kind)`` so a drain
        # payload and a §6.3 nav_probe for the same agent are both
        # preserved (each kind has at most one entry per agent).
        if is_first_seen and payloads:
            latest_by_key: dict[tuple[str, str], dict] = {}
            for p in payloads:
                key = (p.get("agent_id", ""), p.get("kind", ""))
                latest_by_key[key] = p
            payloads = sorted(
                latest_by_key.values(), key=lambda p: int(p.get("seq", 0)),
            )

        for payload in payloads:
            seq = int(payload.get("seq", 0))
            if seq > _poll_state["last_seen_seq"]:
                _poll_state["last_seen_seq"] = seq
            agent_id = payload.get("agent_id", "")
            # §6.3 nav_probe events get their own type so the dashboard can
            # render them distinctly (warning toast on mismatch, etc.) and
            # they don't overwrite the per-minute drain in browserMetrics[].
            event_type = (
                "browser_nav_probe"
                if payload.get("kind") == "nav_probe"
                else "browser_metrics"
            )
            event_bus.emit(event_type, agent=agent_id, data=payload)

    async def _browser_metrics_loop() -> None:
        # First pass runs ~5s after boot to give the browser service time
        # to register, then every 60s. Tolerates transient failures — the
        # metric channel is best-effort observability, not correctness.
        await asyncio.sleep(5)
        while True:
            try:
                await _poll_browser_metrics_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.debug("Browser metrics poll loop tick failed: %s", e)
            await asyncio.sleep(60)

    # Expose the poll primitives on app.state so tests (and future admin
    # endpoints) can reach them without walking closure cells.
    app.state.poll_browser_metrics_once = _poll_browser_metrics_once
    app.state.browser_metrics_poll_state = _poll_state

    @app.on_event("startup")
    async def _start_browser_metrics_poll() -> None:
        # No-op if the mesh is running without a browser service configured
        # (early-returned inside _poll_browser_metrics_once).
        asyncio.create_task(_browser_metrics_loop())

    # Mesh-side input validation: reject typo'd action names with a clean 400
    # before proxying to the browser service. Permissions are enforced separately
    # via PermissionMatrix.can_browser_action (default-allow for all known
    # actions when `browser_actions` is unset in the template).
    #
    # New actions must be added to src/host/permissions.KNOWN_BROWSER_ACTIONS
    # so the mesh stops rejecting them.
    from src.host.permissions import KNOWN_BROWSER_ACTIONS as _ALLOWED_BROWSER_ACTIONS

    @app.post("/mesh/browser/command")
    async def browser_command(request: Request) -> dict:
        """Proxy a browser command to the shared browser service.

        Agents never talk to the browser service directly — the mesh
        enforces authentication and permission checks.

        When the body includes ``target_agent_id``, this is a delegation:
        the caller (e.g. operator) wants the command to run against the
        target agent's browser profile. Authorized when the caller can
        message the target and the target has browser access. Used so
        orchestrators can set up browser logins on behalf of workers
        without having their own browser (session cookies must land in
        the worker's profile).
        """
        caller_id = _extract_verified_agent_id(request)
        body = await request.json()
        req_agent_id = _resolve_browser_target(
            caller_id, body.get("target_agent_id") or "",
        )

        action = body.get("action", "")
        params = body.get("params", {})

        if not action:
            raise HTTPException(400, "action is required")

        # Reject unknown actions with 400 before any permission check — we
        # want a distinct error for "action doesn't exist" vs "not authorized."
        if action not in _ALLOWED_BROWSER_ACTIONS:
            raise HTTPException(400, f"Unknown browser action: {action}")

        # Per-action gate applies to the EFFECTIVE target (`req_agent_id`),
        # not the caller. The action runs against the target's browser
        # profile, so the target's `browser_actions` list is the authoritative
        # policy. This closes a delegation bypass where an operator with
        # `browser_actions=["*"]` could otherwise exercise actions the target
        # was never granted. `_resolve_browser_target` already confirmed the
        # caller has delegation rights (can_message + target has browser).
        if not permissions.can_browser_action(req_agent_id, action):
            raise HTTPException(403, f"Browser action '{action}' denied")

        # SSRF protection: early-reject for literal private-IP navigations so
        # agents get a clean 400 rather than a cryptic browser error. The
        # authoritative enforcement happens at the network layer inside the
        # browser container via an iptables egress filter installed by
        # docker/browser-entrypoint.sh — that layer covers DNS rebinding,
        # HTTP redirects, subresource loads (img/script/iframe), XHR/fetch,
        # and WebSockets, none of which can be gated at the Playwright API.
        if action in ("navigate", "open_tab"):
            nav_url = params.get("url", "")
            if nav_url:
                from src.agent.builtins.http_tool import _resolve_and_pin
                try:
                    _resolve_and_pin(nav_url)
                except ValueError as e:
                    raise HTTPException(400, str(e))

        # Check for browser service restart — re-push proxy config for ALL agents
        try:
            restarted = await _check_browser_boot_id_changed()
            if restarted:
                all_agents = list(router.agent_registry.keys())
                logger.info("Browser service restarted, re-pushing proxy for %d agents", len(all_agents))
                for _aid in all_agents:
                    await _push_browser_proxy(_aid)
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
            # §2.5 trace propagation: forward X-Trace-Id so log lines in the
            # browser service correlate with the agent's originating request.
            incoming_trace = request.headers.get("x-trace-id")
            if incoming_trace:
                headers["X-Trace-Id"] = incoming_trace
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

    # ── §4.5 / §8.1 file-upload staging ──────────────────────────────────
    #
    # Two-phase mesh-mediated upload:
    #   A) /mesh/browser/upload-stage stores raw bytes from the agent into
    #      a tmpfs-backed staging dir keyed by an opaque handle.
    #   B) /mesh/browser/upload_file resolves staged_handles → bytes,
    #      streams them into the browser container's receive dir, then
    #      drives /browser/{agent}/upload_file with the resulting paths.
    #
    # Stage files older than _UPLOAD_STAGE_TTL_S are reaped by a periodic
    # garbage-collection loop scheduled at startup.

    import hashlib as _hashlib
    import os as _os
    from pathlib import Path as _Path

    _UPLOAD_STAGE_DIR = _Path(
        _os.environ.get(
            "OPENLEGION_UPLOAD_STAGE_DIR",
            "/tmp/openlegion-upload-stage",
        ),
    )
    try:
        _UPLOAD_STAGE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as _e:
        logger.warning(
            "Could not create upload-stage dir %s: %s", _UPLOAD_STAGE_DIR, _e,
        )

    try:
        _UPLOAD_STAGE_TTL_S = max(
            5, int(_os.environ.get("OPENLEGION_UPLOAD_STAGE_TTL_S", "60")),
        )
    except ValueError:
        _UPLOAD_STAGE_TTL_S = 60

    try:
        _UPLOAD_STAGE_MAX_MB = max(
            1, int(_os.environ.get("OPENLEGION_UPLOAD_STAGE_MAX_MB", "50")),
        )
    except ValueError:
        _UPLOAD_STAGE_MAX_MB = 50
    _UPLOAD_STAGE_MAX_BYTES = _UPLOAD_STAGE_MAX_MB * 1024 * 1024

    _stage_dir_resolved = _UPLOAD_STAGE_DIR.resolve()

    def _stage_paths(handle: str) -> tuple[_Path, _Path]:
        bin_path = (_UPLOAD_STAGE_DIR / f"{handle}.bin").resolve()
        meta_path = (_UPLOAD_STAGE_DIR / f"{handle}.json").resolve()
        if not bin_path.is_relative_to(_stage_dir_resolved):
            raise HTTPException(400, "invalid handle")
        if not meta_path.is_relative_to(_stage_dir_resolved):
            raise HTTPException(400, "invalid handle")
        return bin_path, meta_path

    def _read_stage_meta(meta_path: _Path) -> dict | None:
        try:
            return json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def _idem_handle(caller_id: str, idem_key: str) -> str:
        digest = _hashlib.sha256(f"{caller_id}|{idem_key}".encode()).hexdigest()
        return f"{caller_id}-idem{digest[:24]}"

    def _atomic_write_bytes(path: _Path, data: bytes) -> None:
        partial = path.with_suffix(path.suffix + ".partial")
        try:
            with open(partial, "wb") as fh:
                fh.write(data)
            _os.replace(partial, path)
        except OSError:
            with contextlib.suppress(OSError):
                partial.unlink()
            raise

    # P0.2: Per-handle async lock for stage writes. Two parallel stage
    # requests with the same idempotency_key produce the same handle —
    # without this lock, both would write to the partial file concurrently
    # and corrupt the resulting bytes.
    _stage_locks: dict[str, asyncio.Lock] = {}
    _stage_locks_guard = asyncio.Lock()

    async def _get_stage_lock(handle: str) -> asyncio.Lock:
        async with _stage_locks_guard:
            lock = _stage_locks.get(handle)
            if lock is None:
                lock = asyncio.Lock()
                _stage_locks[handle] = lock
            return lock

    async def _release_stage_lock_if_idle(handle: str) -> None:
        async with _stage_locks_guard:
            lock = _stage_locks.get(handle)
            if lock is not None and not lock.locked():
                _stage_locks.pop(handle, None)

    # P0.3: Track in-flight stage writes so the GC does not unlink a
    # ``.partial`` mid-write for slow uploads. The set is keyed by handle.
    _active_stage_handles: set[str] = set()

    # P0.3 / GC tuning: ``.partial`` files belonging to inactive handles
    # are reaped only when older than this longer threshold (5 × TTL),
    # giving slow uploads room to complete even when in-flight tracking
    # was reset by a process restart.
    _UPLOAD_STAGE_PARTIAL_TTL_S = max(_UPLOAD_STAGE_TTL_S, _UPLOAD_STAGE_TTL_S * 5)

    @app.post("/mesh/browser/upload-stage")
    async def upload_stage(request: Request) -> dict:
        """Phase A: stream raw bytes into the mesh staging dir.

        Returns ``{staged_handle, size_bytes, expires_at}``. Bytes are
        capped at ``_UPLOAD_STAGE_MAX_BYTES``; oversize requests return
        413 and clean up the partial file.

        Idempotency: when ``Idempotency-Key`` is supplied the staged
        handle is derived deterministically from ``sha256(caller_id|key)``
        so a duplicate (caller, key, sha256) within TTL returns the same
        handle without an O(N) directory scan.
        """
        caller_id = _extract_verified_agent_id(request)
        await _check_rate_limit("upload_stage", caller_id)
        if not permissions.can_browser_action(caller_id, "upload_file"):
            raise HTTPException(403, "Browser action 'upload_file' denied")

        idem_key = request.headers.get("idempotency-key", "") or None
        max_bytes = _UPLOAD_STAGE_MAX_BYTES

        if idem_key:
            handle = _idem_handle(caller_id, idem_key)
        else:
            handle = f"{caller_id}-{_uuid.uuid4().hex[:24]}"
        bin_path, meta_path = _stage_paths(handle)
        bin_partial = bin_path.with_suffix(".bin.partial")

        lock = await _get_stage_lock(handle)
        async with lock:
            _active_stage_handles.add(handle)
            try:
                sha = _hashlib.sha256()
                size = 0
                try:
                    with open(bin_partial, "wb") as fh:
                        async for chunk in request.stream():
                            if not chunk:
                                continue
                            size += len(chunk)
                            if size > max_bytes:
                                fh.close()
                                with contextlib.suppress(OSError):
                                    bin_partial.unlink()
                                raise HTTPException(
                                    413,
                                    f"Upload exceeds {_UPLOAD_STAGE_MAX_MB}MB limit",
                                )
                            fh.write(chunk)
                            sha.update(chunk)
                except HTTPException:
                    raise
                except OSError as e:
                    with contextlib.suppress(OSError):
                        bin_partial.unlink()
                    raise HTTPException(500, f"Stage write failed: {e}")

                digest = sha.hexdigest()

                if idem_key and meta_path.is_file() and bin_path.is_file():
                    now = time.time()
                    try:
                        age = now - meta_path.stat().st_mtime
                    except OSError:
                        age = _UPLOAD_STAGE_TTL_S + 1
                    existing_meta = _read_stage_meta(meta_path) if age <= _UPLOAD_STAGE_TTL_S else None
                    if (
                        existing_meta
                        and existing_meta.get("caller_id") == caller_id
                        and existing_meta.get("idempotency_key") == idem_key
                        and existing_meta.get("sha256") == digest
                        and existing_meta.get("status") != "consumed"
                    ):
                        with contextlib.suppress(OSError):
                            bin_partial.unlink()
                        expires_at = (
                            datetime.fromtimestamp(
                                meta_path.stat().st_mtime, timezone.utc,
                            )
                            + timedelta(seconds=_UPLOAD_STAGE_TTL_S)
                        ).isoformat()
                        return {
                            "staged_handle": handle,
                            "size_bytes": int(existing_meta.get("size_bytes", 0)),
                            "expires_at": expires_at,
                        }

                try:
                    _os.replace(bin_partial, bin_path)
                except OSError as e:
                    with contextlib.suppress(OSError):
                        bin_partial.unlink()
                    raise HTTPException(500, f"Stage finalize failed: {e}")

                created_at = datetime.now(timezone.utc)
                expires_at_dt = created_at + timedelta(seconds=_UPLOAD_STAGE_TTL_S)
                meta_payload = {
                    "caller_id": caller_id,
                    "idempotency_key": idem_key,
                    "created_at": created_at.isoformat(),
                    "expires_at": expires_at_dt.isoformat(),
                    "size_bytes": size,
                    "sha256": digest,
                    "status": "staged",
                    "consumed_at": None,
                    "last_result": None,
                }
                try:
                    _atomic_write_bytes(meta_path, json.dumps(meta_payload).encode())
                except OSError as e:
                    with contextlib.suppress(OSError):
                        bin_path.unlink()
                    raise HTTPException(500, f"Stage meta write failed: {e}")

                return {
                    "staged_handle": handle,
                    "size_bytes": size,
                    "expires_at": expires_at_dt.isoformat(),
                }
            finally:
                _active_stage_handles.discard(handle)
                await _release_stage_lock_if_idle(handle)

    @app.post("/mesh/browser/upload_file")
    async def upload_apply(request: Request) -> dict:
        """Phase B: forward staged bytes into the browser, drive upload_file.

        Body: ``{target_agent_id?, ref, staged_handles: [str],
        suggested_filenames?: [str], idempotency_key?}``. Resolves each
        handle to its mesh-side bytes, streams to
        ``/browser/{a}/_stage_upload``, then POSTs the resulting paths to
        ``/browser/{a}/upload_file``.

        Idempotency: when ``idempotency_key`` is supplied, a successful
        apply caches its result envelope on each handle's sidecar. A
        subsequent apply with the same ``(caller, key, handles)`` set
        within the stage TTL returns the cached envelope — the browser
        is NOT driven a second time. Without this, retry-after-success
        would 404 (handles consumed) and double-upload after re-stage.
        """
        caller_id = _extract_verified_agent_id(request)
        await _check_rate_limit("upload_apply", caller_id)
        body = await request.json()
        req_agent_id = _resolve_browser_target(
            caller_id, body.get("target_agent_id") or "",
        )

        if not permissions.can_browser_action(req_agent_id, "upload_file"):
            raise HTTPException(403, "Browser action 'upload_file' denied")

        ref = body.get("ref", "")
        staged_handles = body.get("staged_handles") or []
        suggested_filenames = body.get("suggested_filenames") or []
        idempotency_key = body.get("idempotency_key")
        if idempotency_key is not None and not isinstance(idempotency_key, str):
            raise HTTPException(400, "idempotency_key must be a string")
        if not ref:
            raise HTTPException(400, "ref required")
        if (
            not isinstance(staged_handles, list)
            or not all(isinstance(h, str) and h for h in staged_handles)
        ):
            raise HTTPException(400, "staged_handles must be a non-empty list of strings")
        if not staged_handles:
            raise HTTPException(400, "staged_handles must not be empty")
        if len(staged_handles) > 5:
            raise HTTPException(400, "at most 5 files per upload")
        if suggested_filenames:
            if (
                not isinstance(suggested_filenames, list)
                or not all(isinstance(n, str) for n in suggested_filenames)
            ):
                raise HTTPException(
                    400, "suggested_filenames must be a list of strings",
                )
            if len(suggested_filenames) != len(staged_handles):
                raise HTTPException(
                    400,
                    "suggested_filenames length must match staged_handles",
                )

        # P0.1 — replay check. If the caller supplied an idempotency_key
        # and EVERY referenced handle's sidecar is consumed, owned by the
        # caller, tagged with the same key, within TTL, and carries a
        # cached last_result — return the cached envelope. The browser
        # is not driven again.
        if idempotency_key:
            cached_results: list[dict] = []
            replay_eligible = True
            now = time.time()
            for handle in staged_handles:
                _bin, meta_path = _stage_paths(handle)
                if not meta_path.is_file():
                    replay_eligible = False
                    break
                try:
                    age = now - meta_path.stat().st_mtime
                except OSError:
                    replay_eligible = False
                    break
                if age > _UPLOAD_STAGE_TTL_S:
                    replay_eligible = False
                    break
                meta = _read_stage_meta(meta_path) or {}
                if (
                    meta.get("caller_id") != caller_id
                    or meta.get("status") != "consumed"
                    or meta.get("idempotency_key") != idempotency_key
                    or not isinstance(meta.get("last_result"), dict)
                ):
                    replay_eligible = False
                    break
                cached_results.append(meta["last_result"])
            if replay_eligible and cached_results:
                # All handles share the same apply call → all sidecars
                # carry the same envelope. Return the first.
                return cached_results[0]

        resolved: list[tuple[str, _Path, dict, str]] = []
        for i, handle in enumerate(staged_handles):
            bin_path, meta_path = _stage_paths(handle)
            if not bin_path.is_file() or not meta_path.is_file():
                raise HTTPException(404, f"Unknown staged_handle: {handle}")
            meta = _read_stage_meta(meta_path) or {}
            if meta.get("caller_id") != caller_id:
                raise HTTPException(403, "staged_handle does not belong to caller")
            if meta.get("status") == "consumed":
                # Handle was consumed by a prior apply but the replay
                # check above did not match (key mismatch, missing key,
                # or stale). Treat as unknown.
                raise HTTPException(404, f"Unknown staged_handle: {handle}")
            suggested = (
                suggested_filenames[i] if i < len(suggested_filenames) else ""
            )
            resolved.append((handle, bin_path, meta, suggested))

        browser_service_url = None
        if container_manager:
            browser_service_url = getattr(container_manager, "browser_service_url", None)
        if not browser_service_url:
            raise HTTPException(503, "Browser service not available")

        browser_auth = getattr(container_manager, "browser_auth_token", "")
        ingest_headers: dict = {"X-Mesh-Internal": "1"}
        if browser_auth:
            ingest_headers["Authorization"] = f"Bearer {browser_auth}"
        incoming_trace = request.headers.get("x-trace-id")
        if incoming_trace:
            ingest_headers["X-Trace-Id"] = incoming_trace

        async def _stream_file(path: _Path):
            fh = open(path, "rb")
            try:
                while True:
                    chunk = await asyncio.to_thread(fh.read, 64 * 1024)
                    if not chunk:
                        break
                    yield chunk
            finally:
                fh.close()

        browser_paths: list[str] = []
        try:
            for handle, bin_path, _meta, suggested in resolved:
                ingest_filename = suggested or handle
                try:
                    resp = await _browser_proxy_client.post(
                        f"{browser_service_url}/browser/{req_agent_id}/_stage_upload",
                        params={"suggested_filename": ingest_filename},
                        content=_stream_file(bin_path),
                        headers=ingest_headers,
                        timeout=180,
                    )
                except _httpx.TimeoutException as e:
                    raise HTTPException(503, f"Browser service timeout: {e}")
                except _httpx.ConnectError as e:
                    raise HTTPException(503, f"Browser service unreachable: {e}")
                if resp.status_code >= 400:
                    raise HTTPException(
                        resp.status_code,
                        f"Browser stage ingest failed: {resp.text}",
                    )
                ingest = resp.json()
                ingested_path = ingest.get("path")
                if not ingested_path:
                    raise HTTPException(502, "Browser stage ingest returned no path")
                browser_paths.append(ingested_path)
        except HTTPException:
            raise
        except Exception as e:
            logger.warning("Upload ingest stream error: %s", e)
            raise HTTPException(502, f"Browser ingest error: {e}")

        upload_headers: dict = {}
        if browser_auth:
            upload_headers["Authorization"] = f"Bearer {browser_auth}"
        if incoming_trace:
            upload_headers["X-Trace-Id"] = incoming_trace
        try:
            resp = await _browser_proxy_client.post(
                f"{browser_service_url}/browser/{req_agent_id}/upload_file",
                json={"ref": ref, "paths": browser_paths},
                headers=upload_headers,
            )
            resp.raise_for_status()
            result = resp.json()
        except _httpx.TimeoutException as e:
            raise HTTPException(503, f"Browser service timeout: {e}")
        except _httpx.HTTPStatusError as e:
            raise HTTPException(e.response.status_code, e.response.text)
        except Exception as e:
            logger.warning("Browser upload_file proxy error: %s", e)
            raise HTTPException(502, f"Browser service error: {e}")

        if result.get("success"):
            now_iso = datetime.now(timezone.utc).isoformat()
            for handle, _bin_path, meta, _suggested in resolved:
                _bin, _meta_path = _stage_paths(handle)
                with contextlib.suppress(OSError):
                    _bin.unlink()
                meta_consumed = dict(meta)
                meta_consumed["status"] = "consumed"
                meta_consumed["consumed_at"] = now_iso
                # P0.1 — only cache the result envelope when an
                # idempotency_key was supplied. Without a key, replay
                # protection is impossible so don't pretend we have it.
                if idempotency_key:
                    meta_consumed["idempotency_key"] = idempotency_key
                    meta_consumed["last_result"] = result
                try:
                    _atomic_write_bytes(_meta_path, json.dumps(meta_consumed).encode())
                except OSError:
                    with contextlib.suppress(OSError):
                        _meta_path.unlink()
        return result

    async def _upload_stage_gc_once() -> int:
        """Reap orphan stage files older than TTL. Returns reaped count.

        Iterates by handle (drawn from sidecar metadata): when both ``.bin``
        and ``.json`` exist for a handle they are considered paired; the GC
        reaps the pair only when BOTH are older than TTL. Stray ``.partial``
        files use a longer TTL (``_UPLOAD_STAGE_PARTIAL_TTL_S``) and are
        skipped while the corresponding handle is in ``_active_stage_handles``
        (P0.3 — slow uploads must not have their partial unlinked
        mid-write). Unpaired bare ``.bin`` files are left alone — they may
        belong to an in-progress write.
        """
        if not _UPLOAD_STAGE_DIR.exists():
            return 0
        now = time.time()
        reaped = 0
        try:
            entries = list(_UPLOAD_STAGE_DIR.iterdir())
        except OSError:
            return 0

        for child in entries:
            if not child.name.endswith(".partial"):
                continue
            # ``foo.bin.partial`` → handle is ``foo``.
            handle = child.name[: -len(".bin.partial")] if child.name.endswith(".bin.partial") else child.stem
            if handle in _active_stage_handles:
                continue
            try:
                age = now - child.stat().st_mtime
            except OSError:
                continue
            if age > _UPLOAD_STAGE_PARTIAL_TTL_S:
                try:
                    child.unlink()
                    reaped += 1
                except OSError:
                    continue

        for child in entries:
            if child.suffix != ".json":
                continue
            handle = child.stem
            bin_path, meta_path = _stage_paths(handle)
            try:
                meta_age = now - meta_path.stat().st_mtime
            except OSError:
                meta_age = _UPLOAD_STAGE_TTL_S + 1
            if bin_path.is_file():
                try:
                    bin_age = now - bin_path.stat().st_mtime
                except OSError:
                    bin_age = _UPLOAD_STAGE_TTL_S + 1
                if meta_age > _UPLOAD_STAGE_TTL_S and bin_age > _UPLOAD_STAGE_TTL_S:
                    try:
                        bin_path.unlink()
                        reaped += 1
                    except OSError:
                        pass
                    try:
                        meta_path.unlink()
                        reaped += 1
                    except OSError:
                        pass
            else:
                if meta_age > _UPLOAD_STAGE_TTL_S:
                    try:
                        meta_path.unlink()
                        reaped += 1
                    except OSError:
                        pass
        return reaped

    async def _upload_stage_gc_loop() -> None:
        while True:
            try:
                await _upload_stage_gc_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.debug("upload_stage gc tick failed: %s", e)
            await asyncio.sleep(max(1, _UPLOAD_STAGE_TTL_S // 2))

    app.state.upload_stage_dir = _UPLOAD_STAGE_DIR
    app.state.upload_stage_ttl_s = _UPLOAD_STAGE_TTL_S
    app.state.upload_stage_partial_ttl_s = _UPLOAD_STAGE_PARTIAL_TTL_S
    app.state.upload_stage_gc_once = _upload_stage_gc_once
    app.state.upload_stage_active_handles = _active_stage_handles

    @app.on_event("startup")
    async def _start_upload_stage_gc() -> None:
        asyncio.create_task(_upload_stage_gc_loop())

    _ARTIFACT_NAME_SAFE_RE = re.compile(r"[^\w.\-]+")

    def _sanitize_artifact_name(suggested: str) -> str:
        """Reduce a browser-supplied filename to a safe artifact basename.

        Strips path components, collapses unsafe chars to ``_``, trims
        leading/trailing punctuation, and falls back to ``download.bin``
        when the result would be empty.
        """
        name = (suggested or "").strip()
        if "/" in name or "\\" in name:
            name = name.replace("\\", "/").rsplit("/", 1)[-1]
        name = _ARTIFACT_NAME_SAFE_RE.sub("_", name)
        name = name.strip("._-")
        if not name or len(name) > 180:
            name = name[:180].strip("._-") if name else ""
        if not name:
            name = "download.bin"
        if len(name) < 2:
            name = name + "_"
        return name

    @app.post("/mesh/browser/download")
    async def browser_download(request: Request) -> dict:
        """Trigger a download in the shared browser, stream it into the
        target agent's ``/artifacts/ingest`` endpoint, then clean up the
        browser-side staging file.

        Body: ``{ref, timeout_ms?, target_agent_id?}``. When
        ``target_agent_id`` is set, the download lands in the target's
        artifacts. Operator kill switch: ``BROWSER_DOWNLOADS_DISABLED``
        returns a ``forbidden`` error envelope and never touches the
        browser service.
        """
        from src.browser.flags import get_bool

        caller_id = _extract_verified_agent_id(request)
        body = await request.json()
        req_agent_id = _resolve_browser_target(
            caller_id, body.get("target_agent_id") or "",
        )

        if get_bool("BROWSER_DOWNLOADS_DISABLED", False, agent_id=req_agent_id):
            raise HTTPException(403, detail={
                "success": False,
                "error": {
                    "code": "forbidden",
                    "message": "Downloads disabled by operator",
                },
            })

        ref = body.get("ref", "")
        if not ref:
            raise HTTPException(400, "ref is required")
        timeout_ms = int(body.get("timeout_ms", 30000))

        if not permissions.can_browser_action(req_agent_id, "download"):
            raise HTTPException(403, "Browser action 'download' denied")

        browser_service_url = None
        if container_manager:
            browser_service_url = getattr(container_manager, "browser_service_url", None)
        if not browser_service_url:
            raise HTTPException(503, "Browser service not available")

        agent_entry = router.agent_registry.get(req_agent_id)
        if not agent_entry:
            raise HTTPException(404, f"Agent not registered: {req_agent_id}")
        agent_url = (
            agent_entry.get("url", agent_entry)
            if isinstance(agent_entry, dict) else agent_entry
        )

        browser_auth = getattr(container_manager, "browser_auth_token", "")
        headers: dict = {}
        if browser_auth:
            headers["Authorization"] = f"Bearer {browser_auth}"
        incoming_trace = request.headers.get("x-trace-id")
        if incoming_trace:
            headers["X-Trace-Id"] = incoming_trace

        try:
            trigger_resp = await _browser_proxy_client.post(
                f"{browser_service_url}/browser/{req_agent_id}/download",
                json={"ref": ref, "timeout_ms": timeout_ms},
                headers=headers,
                timeout=180,
            )
        except Exception as e:
            logger.warning("Browser download trigger error: %s", e)
            raise HTTPException(502, f"Browser service error: {e}")

        if trigger_resp.status_code >= 400:
            raise HTTPException(trigger_resp.status_code, trigger_resp.text)

        trigger_data = trigger_resp.json()
        if not trigger_data.get("success"):
            return trigger_data

        data = trigger_data["data"]
        nonce = data.get("nonce", "")
        suggested = data.get("suggested_filename", "")
        mime_type = data.get("mime_type", "application/octet-stream")
        sanitized_name = _sanitize_artifact_name(suggested)
        ingest_url = f"{agent_url}/artifacts/ingest/{sanitized_name}"

        ingest_data: dict = {}
        ingest_error: Exception | None = None
        try:
            async with _browser_proxy_client.stream(
                "GET",
                f"{browser_service_url}/browser/{req_agent_id}/_download_stream",
                params={"nonce": nonce},
                headers=headers,
                timeout=180,
            ) as bsrc:
                if bsrc.status_code >= 400:
                    raise HTTPException(
                        502,
                        f"Browser stream returned {bsrc.status_code}",
                    )
                ingest_headers: dict = {
                    "X-Mesh-Internal": "1",
                    "Content-Type": "application/octet-stream",
                }
                if incoming_trace:
                    ingest_headers["X-Trace-Id"] = incoming_trace
                async with _httpx.AsyncClient(timeout=240) as agent_client:
                    ingest_resp = await agent_client.post(
                        ingest_url,
                        content=bsrc.aiter_bytes(),
                        headers=ingest_headers,
                    )
                    if ingest_resp.status_code >= 400:
                        raise HTTPException(
                            ingest_resp.status_code, ingest_resp.text,
                        )
                    ingest_data = ingest_resp.json()
        except HTTPException as e:
            ingest_error = e
        except Exception as e:
            ingest_error = e
            logger.warning("Browser→agent ingest error: %s", e)
        finally:
            try:
                await _browser_proxy_client.post(
                    f"{browser_service_url}/browser/{req_agent_id}/_download_cleanup",
                    json={"nonce": nonce},
                    headers=headers,
                )
            except Exception as e:
                logger.warning("Browser download cleanup error: %s", e)

        if ingest_error is not None:
            if isinstance(ingest_error, HTTPException):
                raise ingest_error
            raise HTTPException(502, f"Ingest error: {ingest_error}")

        return {
            "success": True,
            "data": {
                "artifact_name": ingest_data.get("artifact_name"),
                "size_bytes": ingest_data.get("size_bytes"),
                "mime_type": mime_type,
            },
        }

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
