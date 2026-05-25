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
import inspect
import json
import os
import re
import time
import uuid as _uuid
from collections import defaultdict, deque
from collections.abc import Callable, Coroutine
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import StreamingResponse

from src.host.change_history import ChangeHistory
from src.host.credentials import is_system_credential
from src.host.orchestration import (
    VALID_STATUSES,
    InvalidStatusTransition,
    TaskNotFound,
    Tasks,
)
from src.host.pending_actions import PendingActions
from src.shared.paths import resolve_under_root
from src.shared.redaction import redact_url
from src.shared.types import (
    AGENT_ID_RE_PATTERN,
    HARD_EDIT_FIELDS,
    RESERVED_AGENT_IDS,
    SOFT_EDIT_FIELDS,
    AgentMessage,
    APIProxyRequest,
    APIProxyResponse,
    BlackboardClaimRequest,
    BlackboardWatchRequest,
    MeshEvent,
    MessageOrigin,
    NotifyRequest,
)
from src.shared.utils import dumps_safe, sanitize_for_prompt, setup_logging

logger = setup_logging("host.server")

_MAX_SYSTEM_PROMPT = 10_000  # chars — caps agent-supplied system prompt to limit context cost
_MAX_BB_KEY_LEN = 512  # chars — prevents abusive key lengths in blackboard
_MAX_BB_VALUE_BYTES = 262_144  # 256 KB — bounds per-key storage to keep SQLite WAL manageable


def _websockets_headers_kw(connect, headers: dict[str, str]) -> dict:
    """Return the header kwarg name supported by the installed websockets."""
    try:
        params = inspect.signature(connect).parameters
    except (TypeError, ValueError):
        return {"additional_headers": headers}
    if "additional_headers" in params:
        return {"additional_headers": headers}
    return {"extra_headers": headers}


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
    from src.shared.types import MessageOrigin


# ── Pending Action Store TTLs ────────────────────────────────────
#
# Pending operator actions (delete confirmations, undo-receipt rows)
# live in ``PendingActions`` (``data/pending_actions.db``) so they
# survive mesh restarts. The store itself is constructed inside
# ``create_mesh_app`` and exposed on the app as ``app.pending_actions``.
# Only the field-aware TTLs live at module scope so the receipt-card
# layer and tests can compute them without instantiating an app.
_MAX_PENDING = 10
# Default TTL for pending-action rows (deletes, soft-edit receipts,
# anything we don't have a more specific budget for). 5 minutes is the
# legacy value and still appropriate for low-stakes acts the user can
# undo.
_CHANGE_TTL_SECONDS = 300
# Riskier config edits — model swaps, permission changes, budget
# tweaks, thinking-level changes — get a longer undo window because
# the user typically wants to read the diff before deciding to revert.
# 30 minutes mirrors the worktree code-review reflex.
_HARD_CHANGE_TTL_SECONDS = 1800


def _ttl_for_field(field: str | None) -> int:
    """Pick the pending-action TTL for a config field.

    Hard fields (model / permissions / budget / thinking) get the
    longer ``_HARD_CHANGE_TTL_SECONDS`` window so the user has time to
    read the diff before the undo receipt expires. Everything else
    (soft fields, non-config actions like team/agent deletes, unknown
    action kinds) falls back to ``_CHANGE_TTL_SECONDS``. The hard set
    is defined once in :data:`src.shared.types.HARD_EDIT_FIELDS` and
    imported here.
    """
    if field and field in HARD_EDIT_FIELDS:
        return _HARD_CHANGE_TTL_SECONDS
    if field and field not in SOFT_EDIT_FIELDS:
        # Unknown field — log so future debugging surfaces typos /
        # newly-added fields that forgot to declare a TTL bucket.
        # Soft TTL is the safer fallback (shorter window favours
        # iteration over delay).
        logger.debug("unknown TTL field %r, defaulting to soft", field)
    return _CHANGE_TTL_SECONDS


# ── Task 2c: Server-side channel origin validation ────────────────
#
# ``X-Origin`` is authorization-bearing only after the mesh has a reason
# to trust the stamping layer. Paired messaging channels are re-checked
# against their on-disk pairing record; non-paired human channels and
# privileged non-human kinds are preserved only for trusted callers such
# as the operator or loopback-internal mesh calls.
_PAIRED_CHANNELS = frozenset({"telegram", "discord", "slack", "whatsapp", "webhook"})
_TRUSTED_ORIGIN_CALLERS = frozenset({"mesh", "operator"})


def _caller_is_operator(caller: str, request: Request) -> bool:
    """Return True iff ``caller`` is the verified operator persona.

    The single source of truth for the operator trust tier introduced
    alongside the structured-denial work: the operator is the user's
    extension at the chat/UI layer, where user-control surfaces (revoke
    credentials, revoke browser, budget caps, destructive-action nonces)
    already gate the things that matter. Treating operator as a worker
    behind the same scope/permission gates produced opaque 403s on
    coordination paths and was the proximate cause of every silent
    pipeline stall in early-2026 incident reports.

    Trust derivation: ``caller`` is the bearer-verified agent id from
    ``_extract_verified_agent_id``. In production (tokens configured)
    the value can only be ``"operator"`` if the HMAC compare against
    ``_auth_tokens["operator"]`` succeeded. In dev mode (no tokens)
    ``X-Agent-ID`` is trusted by the same rules as every other identity
    in the file — the fail-closed startup check refuses to boot under
    ``OPENLEGION_TEAM_SCOPE_MODE=enforce`` without tokens, which
    is the configuration where forgery would matter.

    ``request`` is unused today but kept on the signature so a future
    extension (e.g. promoting loopback ``x-mesh-internal`` to operator-
    equivalent for a specific gate) lands in one place instead of
    re-threading the request argument through every callsite.
    """
    del request  # reserved for future use; see docstring
    return caller == "operator"

# Tiny TTL cache for pairing-record reads. The on-disk file changes only
# on ``/pair`` operations, which are rare. 5s staleness is acceptable for
# a defense-in-depth gate; ``_invalidate_pairing_cache`` lets the pair/
# unpair flow drop a stale entry on demand.
_pairing_cache: dict[str, tuple[float, dict | None]] = {}
_PAIRING_CACHE_TTL = 5.0


def _read_pairing_record(channel: str) -> dict | None:
    """Read the pairing JSON for a channel.

    Returns ``None`` if missing, malformed, or unreadable. Cached for
    ``_PAIRING_CACHE_TTL`` seconds keyed on channel name.
    """
    cached = _pairing_cache.get(channel)
    if cached and time.monotonic() - cached[0] < _PAIRING_CACHE_TTL:
        return cached[1]
    from src.cli.config import PROJECT_ROOT
    path = PROJECT_ROOT / "config" / f"{channel}_paired.json"
    record: dict | None
    if not path.exists():
        record = None
    else:
        try:
            raw = path.read_text()
            parsed = json.loads(raw)
            record = parsed if isinstance(parsed, dict) else None
        except (json.JSONDecodeError, OSError):
            record = None
    _pairing_cache[channel] = (time.monotonic(), record)
    return record


def _invalidate_pairing_cache(channel: str | None = None) -> None:
    """Drop the cached pairing record for ``channel`` (or all channels).

    Pair/unpair endpoints can call this to force a fresh read on the
    next ``/mesh/wake`` (or other validated-origin) request. When called
    with no argument, clears the entire cache — useful in tests.
    """
    if channel is None:
        _pairing_cache.clear()
        return
    _pairing_cache.pop(channel, None)


def _is_paired_user(channel: str, user: str) -> bool:
    """Check if ``user`` is paired for ``channel``.

    Mirrors :meth:`PairingManager.is_allowed` semantics: owner match OR
    membership in the allowed list. Comparisons are stringified so
    Telegram numeric ids and Slack/Discord string ids round-trip the
    same way.
    """
    record = _read_pairing_record(channel)
    if not record:
        return False
    owner = record.get("owner")
    allowed = record.get("allowed", [])
    if not isinstance(allowed, list):
        allowed = []
    if owner is not None and str(owner) == str(user):
        return True
    return str(user) in (str(u) for u in allowed)


def _is_internal_caller(request: Request) -> bool:
    """Return True for trusted in-process callers (loopback + ``x-mesh-internal``).

    The mesh's own dispatcher, the dashboard router, the CLI manager
    process, and health checks all hit ``localhost`` with the
    ``x-mesh-internal: 1`` header. Authorization gates that should let
    those callers through (e.g. Task 2e's worker → operator wake block)
    use this predicate.

    Both conditions must hold: the header AND a loopback peer. The
    header alone is insufficient — a public-internet caller can set any
    header they like — and a loopback peer alone is insufficient too,
    since an agent container can be wired to reach the mesh via
    loopback in some test/dev setups.
    """
    if not request.headers.get("x-mesh-internal"):
        return False
    client_host = request.client.host if request.client else ""
    try:
        import ipaddress

        return ipaddress.ip_address(client_host).is_loopback
    except (ValueError, AttributeError):
        return False


def _downgrade_origin(origin: "MessageOrigin", reason: str) -> "MessageOrigin":
    logger.warning(
        "origin validation failed: %s kind=%s channel=%s user=%s — downgrading to kind=agent",
        reason,
        origin.kind,
        origin.channel,
        origin.user,
    )
    return origin.model_copy(update={"kind": "agent"})


def _validated_origin(
    request: Request, caller: str = "",
) -> "MessageOrigin | None":
    """Parse ``X-Origin`` and downgrade unverifiable channel claims.

    This is the authorization-grade origin getter — endpoints that care
    about an origin's trust level (durable confirms, wakes, pending
    actions) must use this instead of :func:`parse_origin_header`.

    Behaviour:

    * Missing/malformed header → ``None`` (caller decides what default
      to use; ``/mesh/wake`` falls back to ``MessageOrigin(kind="agent")``).
    * Trusted caller (operator/mesh or loopback internal) → returned unchanged.
    * ``kind == "agent"`` → returned unchanged.
    * Privileged non-human kinds from other callers → downgraded.
    * ``kind == "human"`` with a non-paired channel → downgraded.
    * ``kind == "human"`` with a paired channel and a paired user →
      returned unchanged.
    * ``kind == "human"`` with a paired channel and an unpaired/empty
      user → downgraded to ``kind="agent"`` (warning logged).

    The helper never raises: a forged X-Origin header must not crash a
    request. ``MessageOrigin`` is ``frozen=True`` (Task 2a), so the
    downgrade is via :meth:`pydantic.BaseModel.model_copy`, not mutation.
    """
    from src.shared.types import MessageOrigin

    raw = request.headers.get("x-origin")
    # Use ``trust_kind=True`` here because we deliberately want to see
    # the caller-supplied kind so we can validate it. The validation
    # below either preserves it (for verified channel claims) or
    # downgrades it. ``parse_origin_header`` (trust_kind=False) is
    # still the right call for trace-only / non-authorization paths.
    origin = MessageOrigin.from_header_value(raw, trust_kind=True)
    if origin is None:
        return None
    if caller in _TRUSTED_ORIGIN_CALLERS or _is_internal_caller(request):
        return origin
    if origin.kind == "agent":
        return origin
    if origin.kind != "human":
        return _downgrade_origin(origin, "untrusted non-human origin kind")
    if origin.channel not in _PAIRED_CHANNELS:
        return _downgrade_origin(origin, "unverifiable human origin channel")
    if not origin.user:
        return _downgrade_origin(origin, "empty paired-channel user")
    if _is_paired_user(origin.channel, origin.user):
        return origin
    return _downgrade_origin(origin, "unpaired channel user")


# ── Task 5: Team scope isolation (default enforce) ────────────────────
#
# ``OPENLEGION_TEAM_SCOPE_MODE`` is the kill switch for the team
# isolation rollout. Default ``enforce`` — workers see only their own
# team members on ``/mesh/agents`` and only their own team blackboard
# ACLs. ``warn`` is preserved as an emergency rollback that restores the
# legacy fleet-wide visibility while still emitting structured warnings
# on every call that would have been denied under enforce. Read once at
# module import (env vars don't change at runtime); invalid values fall
# back to ``enforce`` with a logged warning.
_TEAM_SCOPE_MODE = os.environ.get("OPENLEGION_TEAM_SCOPE_MODE", "enforce").lower()
if _TEAM_SCOPE_MODE not in {"warn", "enforce"}:
    logger.warning(
        "Invalid OPENLEGION_TEAM_SCOPE_MODE=%r, defaulting to enforce",
        _TEAM_SCOPE_MODE,
    )
    _TEAM_SCOPE_MODE = "enforce"
# Internal alias — kept for the existing scope-gate call sites below.
_PROJECT_SCOPE_MODE = _TEAM_SCOPE_MODE


# Counter surfaced on ``/mesh/system/metrics`` as ``scope_warn_total``.
# Incremented every time a worker's call would have returned a smaller
# response under enforce mode. Lets ops gauge soak-window readiness
# before flipping the env var to ``enforce``.
_scope_warn_count = 0


def _record_scope_warn() -> None:
    """Bump the warn-mode counter (visible on ``/mesh/system/metrics``)."""
    global _scope_warn_count
    _scope_warn_count += 1


# Counter for cross-team blackboard access (read or write where the
# caller's team differs from the existing key's writer-team). Pure
# observability for Phase 3 enforcement design — NOT a denial. Surfaced on
# ``/mesh/system/metrics`` as BOTH ``blackboard_cross_team_total`` and
# the legacy ``blackboard_cross_project_total`` (same value, kept through
# PR 3 for back-compat). Counter is process-lifetime (resets on restart),
# naming follows ``scope_warn_total`` rather than the ``_24h`` mental
# model — restarts are roughly daily-ish in practice and a true 24h
# window would need a separate ledger.
_blackboard_xteam_count: dict[str, int] = {"read": 0, "write": 0}
# Back-compat alias — keep until PR 3.
_blackboard_xproject_count = _blackboard_xteam_count


def _record_blackboard_xteam(kind: str) -> None:
    """Bump the cross-team blackboard counter."""
    if kind not in _blackboard_xteam_count:
        return
    _blackboard_xteam_count[kind] += 1


def _record_blackboard_xproject(kind: str) -> None:
    """DEPRECATED: alias for :func:`_record_blackboard_xteam`."""
    _record_blackboard_xteam(kind)


# Maps the legacy ``project_*`` lifecycle event names (callers still
# pass these for code-reuse during the rename) to the canonical
# ``team_*`` event names that actually fire on the bus. PR 3 stopped
# dual-emitting under the legacy names — subscribers must listen on
# ``team_*``. The five legacy literals stay in ``DashboardEvent.type``
# for type-safety of any historical-record code that parses them.
_PROJECT_TO_TEAM_EVENT = {
    "project_created": "team_created",
    "project_updated": "team_updated",
    "project_deleted": "team_deleted",
    "project_archived": "team_archived",
    "project_unarchived": "team_unarchived",
}


def _emit_team_event(
    event_bus,
    legacy_event: str,
    *,
    agent: str,
    name: str,
    extra: dict | None = None,
    logger=logger,
) -> None:
    """Emit a team lifecycle event on the bus under the canonical name.

    ``legacy_event`` accepts the pre-rename ``project_*`` name purely
    for caller convenience — the actual emit happens under the
    ``team_*`` equivalent looked up via ``_PROJECT_TO_TEAM_EVENT``.
    Payload always contains ``project_id`` / ``team_id`` / ``name`` so
    listeners reading either key keep working. ``extra`` is merged on
    top so callers can pass description / members / etc.
    """
    if event_bus is None:
        return
    payload: dict = {
        "project_id": name,
        "team_id": name,
        "name": name,
    }
    if extra:
        payload.update(extra)
    team_event = _PROJECT_TO_TEAM_EVENT.get(legacy_event, legacy_event)
    try:
        event_bus.emit(team_event, agent=agent, data=payload)
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("%s emit failed: %s", team_event, e)


# Per-category denial counter surfaced on ``/mesh/system/metrics`` as
# ``tool_denials_24h``. Operators previously had no way to see when
# auth/permission denials were happening — silent 401/403/429s only show
# up in HTTP access logs, which the dashboard doesn't render. The counter
# auto-resets at the day boundary so the value answers "how many denials
# fired today". Categories are FROZEN (lower-cardinality, operator-readable):
#
#   * ``auth``       — missing/invalid bearer token
#   * ``scope``      — caller out of project / fleet-roster scope
#   * ``role``       — operator-only or operator-or-internal denied to a worker
#   * ``permission`` — ``permissions.can_*()`` returned False
#   * ``rate``       — rate limiter rejected
_DENIAL_CATEGORIES: frozenset[str] = frozenset(
    {"auth", "scope", "role", "permission", "rate"}
)
_denial_counter: dict[str, int] = defaultdict(int)
# Mutable single-element wrapper so ``_record_denial`` can rotate the
# day-key without juggling a ``global`` declaration on each call. Stored
# as a list to match the established pattern (mutate in place).
_denial_counter_reset_day: list[int] = [int(time.time() // 86400)]


def _record_denial(
    category: str,
    *,
    caller: str | None = None,
    target: str | None = None,
    gate: str | None = None,
    extra: dict | None = None,
) -> None:
    """Bump the per-category 24h denial counter and emit a structured log line.

    Day rollover is detected lazily by comparing the current epoch-day
    against the last reset day — when they differ the counter clears
    before the increment so the ``tool_denials_24h`` field on
    ``/mesh/system/metrics`` reflects the current-day window.

    The keyword fields exist so denials are debuggable from the mesh
    log alone — the previous opaque counter forced operators to cross-
    reference HTTPException bodies with request logs to figure out
    which gate fired. ``caller``/``target``/``gate`` flow through as
    structured ``extra`` on the warning so JSON-mode logs index them.
    """
    if category not in _DENIAL_CATEGORIES:
        return
    today = int(time.time() // 86400)
    if today != _denial_counter_reset_day[0]:
        _denial_counter.clear()
        _denial_counter_reset_day[0] = today
    _denial_counter[category] += 1
    log_payload: dict[str, object] = {
        "denial_category": category,
        "denial_caller": caller,
        "denial_target": target,
        "denial_gate": gate,
    }
    if extra:
        log_payload["denial_extra"] = extra
    logger.warning(
        "mesh denial: category=%s gate=%s caller=%s target=%s",
        category, gate or "?", caller or "?", target or "?",
        extra=log_payload,
    )


def _caller_projects(agent_id: str) -> set[str]:
    """Return the project memberships visible to ``agent_id``.

    Workers see only projects whose ``metadata.yaml`` lists them as
    members. The operator and trusted internal callers (``mesh``) are
    fleet-global by design — they get a sentinel meaning "all projects",
    represented here as an empty set with the caller_is_global flag the
    caller computes separately. Use the helper purely as a lookup of
    *worker* memberships and branch on operator/internal in the caller.
    """
    if agent_id in {"operator", "mesh"}:
        # Operator and the mesh-internal pseudo-id are global; the caller
        # branches on those identities directly. Returning an empty set
        # here forces callers to think about the global path and not
        # silently include "every project" in a worker-style filter.
        return set()
    from src.cli.config import _load_projects

    projects = _load_projects()
    return {
        name
        for name, meta in projects.items()
        if agent_id in meta.get("members", [])
    }


def _is_blackboard_cross_project(
    caller: str, writer: str | None
) -> bool:
    """Return True when caller and writer are workers in disjoint project sets.

    Best-effort detection used purely for telemetry — never gates access.
    Returns False (so the counter is NOT incremented) when:

    - ``writer`` is missing (e.g. unknown / deleted agent or no prior entry)
    - either party is operator / ``mesh`` (fleet-global by design)
    - caller and writer share at least one project membership
    - either party has an empty membership set (standalone agent — not
      meaningfully "cross-project" without two project anchors)

    The intent is to count the case where two distinct *project-bound*
    workers touch the same key, since that is what Phase 3 enforcement
    will gate.
    """
    if not writer or writer == caller:
        return False
    if caller in {"operator", "mesh"} or writer in {"operator", "mesh"}:
        return False
    caller_set = _caller_projects(caller)
    writer_set = _caller_projects(writer)
    if not caller_set or not writer_set:
        return False
    return caller_set.isdisjoint(writer_set)


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

    # Persistent pending-action store. Mirrors the path convention of
    # ``data/costs.db`` / ``data/traces.db``. Backs the
    # ``/mesh/config/confirm`` delete dispatcher and the dashboard's
    # pending-action review surface.
    pending_actions = PendingActions(db_path="data/pending_actions.db")
    app.pending_actions = pending_actions  # exposed for tests/dashboard
    # Task 9 — wire EventBus so store/consume/cancel/reap_expired emit
    # ``pending_action_*`` events to the dashboard.
    if event_bus is not None:
        pending_actions.set_event_bus(event_bus)

    # PR 1 — soft-edit receipts + 5-minute Undo. Mirrors the pending_actions
    # plumbing: SQLite-backed, exposed on the app for tests/dashboard,
    # event_bus wired so the dashboard can render receipt cards live.
    change_history = ChangeHistory(db_path="data/change_history.db")
    app.change_history = change_history
    if event_bus is not None:
        change_history.set_event_bus(event_bus)

    # PR 2 — project → team rename. Idempotent on-startup migrator
    # renames ``config/projects/`` → ``config/teams/`` and copies
    # PROJECT.md → TEAM.md in each agent workspace. The DB column
    # rename is gated behind ``OPENLEGION_TEAM_MIGRATION_RENAME_DB=1``
    # (off by default this PR — see :mod:`src.host.team_migration` for
    # rationale). Failures log and continue — back-compat aliases keep
    # the previous-version data path working. Operators can disable the
    # whole migrator with ``OPENLEGION_DISABLE_TEAM_MIGRATION=1``.
    try:
        from src.host.team_migration import migrate_project_to_team
        migrate_project_to_team()
    except Exception as e:
        logger.error("team_migration failed at startup: %s — continuing", e, exc_info=True)

    # Durable orchestration task records. ``OPENLEGION_ORCHESTRATION_TASKS_DB``
    # overrides the on-disk path — used by tests to keep the db inside
    # ``tmp_path`` instead of polluting cwd.
    _tasks_db_path = os.environ.get(
        "OPENLEGION_ORCHESTRATION_TASKS_DB", "data/tasks.db",
    )
    tasks_store = Tasks(db_path=_tasks_db_path)
    app.tasks_store = tasks_store  # exposed for tests/dashboard
    # Wire EventBus so create / update_status / reroute / cancel emit
    # ``task_*`` events to the dashboard.
    if event_bus is not None:
        tasks_store.set_event_bus(event_bus)

    # Durable work-summaries store. One row per (scope, period_start);
    # operator generates via the ``compose_work_summary`` skill or the
    # per-team cron, user rates via the dashboard. The bus emit drives
    # the Work tab's summary cards live (PR-B).
    from src.host.summaries import WorkSummariesStore
    _summaries_db_path = os.environ.get(
        "OPENLEGION_WORK_SUMMARIES_DB", "data/work_summaries.db",
    )
    summaries_store = WorkSummariesStore(
        db_path=_summaries_db_path, event_bus=event_bus,
    )
    app.summaries_store = summaries_store  # exposed for tests/dashboard

    # In-memory registry of open "agent asks user for help" requests:
    # credential_request, browser_login_request, browser_captcha_help_request.
    # Keyed by request_id (uuid). The dict lets the cancel endpoints
    # address a specific request and lets the dashboard cancel-button
    # path resolve a card without needing to reconstruct (agent_id,
    # service) identity. State is small (a handful of open asks at a
    # time per fleet) and intentionally NOT persisted: a mesh restart
    # already loses the in-flight steer-message contract anyway, so the
    # cards just become stale on the dashboard and gracefully degrade
    # (the Cancel button 404s, which the UI handles).
    help_requests: dict[str, dict] = {}
    _MAX_HELP_REQUESTS = 256  # cap so a noisy agent can't OOM the host
    app.help_requests = help_requests  # exposed for tests + dashboard

    def _record_help_request(
        kind: str, agent_id: str, payload: dict,
    ) -> str:
        """Register an open help request and return its request_id."""
        if len(help_requests) >= _MAX_HELP_REQUESTS:
            # Evict oldest to bound growth.
            oldest = min(
                help_requests.items(),
                key=lambda kv: kv[1].get("created_at", 0),
            )[0]
            help_requests.pop(oldest, None)
        request_id = str(_uuid.uuid4())
        help_requests[request_id] = {
            "kind": kind,
            "agent_id": agent_id,
            "created_at": time.time(),
            "status": "open",
            "payload": payload,
        }
        return request_id

    # Idempotent legacy-data migration. Existing fleets that had
    # blackboard-stored tasks before the v2 rollout (PR #835) get them
    # imported into the durable store on first restart. Subsequent
    # restarts find nothing to migrate and the helper is a no-op.
    # Migration failures are logged loudly but never crash startup —
    # legacy keys stay in place and ops can re-run the helper manually.
    from src.host.orchestration_migration import migrate_blackboard_to_tasks
    try:
        _migration_result = migrate_blackboard_to_tasks(blackboard, tasks_store)
        _migrated = int(_migration_result.get("migrated", 0) or 0)
        _skipped = int(_migration_result.get("skipped", 0) or 0)
        _deleted = int(_migration_result.get("deleted", 0) or 0)
        _errors = _migration_result.get("errors", []) or []
        _error_count = len(_errors) if isinstance(_errors, list) else int(_errors)
        if _migrated > 0 or _deleted > 0:
            logger.info(
                "orchestration migration: migrated=%d skipped=%d deleted=%d errors=%d",
                _migrated, _skipped, _deleted, _error_count,
            )
        else:
            logger.debug("orchestration migration: no legacy tasks found")
    except Exception as e:
        logger.error(
            "orchestration migration failed at startup: %s — legacy tasks preserved",
            e,
            exc_info=True,
        )

    _auth_tokens = auth_tokens if auth_tokens is not None else {}
    _agent_projects = agent_projects if agent_projects is not None else {}

    # Fail-closed startup gate for the operator trust tier.
    #
    # Threat model: under ``_TEAM_SCOPE_MODE=enforce`` the mesh derives
    # caller identity from the bearer token (``_extract_verified_agent_id``).
    # When ``_auth_tokens`` is empty the path falls back to trusting the
    # ``X-Agent-ID`` header, which makes the ``_caller_is_operator``
    # short-circuit forgeable by any caller with network reach.
    #
    # In production the CLI (``src/cli/runtime.py``) builds the
    # ``auth_tokens`` dict during agent setup BEFORE calling
    # ``create_mesh_app`` — so a non-empty dict here is the normal
    # boot state, and an empty dict signals a misconfigured deployment
    # that should refuse to start.
    #
    # The bypass signal is a DEDICATED env var, not an ambient
    # ``"pytest" in sys.modules`` check: production deploys that
    # transitively import pytest (coverage, CI quirks, a tool that
    # depends on ``_pytest.outcomes``) would otherwise silently skip the
    # gate exactly when it should fire. ``tests/conftest.py`` sets the
    # var globally for the in-process test session.
    if (
        _TEAM_SCOPE_MODE == "enforce"
        and not _auth_tokens
        and os.environ.get("OPENLEGION_SKIP_TRUST_TIER_BOOT_GATE") != "1"
    ):
        raise SystemExit(
            "FATAL: OPENLEGION_TEAM_SCOPE_MODE=enforce requires "
            "auth_tokens to be configured. Without tokens the "
            "X-Agent-ID header is unverifiable and the operator trust "
            "tier becomes forgeable. The CLI normally populates auth "
            "tokens before starting the mesh; an empty dict at boot "
            "signals a misconfigured deployment. Set "
            "OPENLEGION_TEAM_SCOPE_MODE=warn for dev use without auth."
        )

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
    # Sliding-window rate-limit buckets keyed by f"{endpoint}:{agent_id}".
    # Deque + popleft makes pruning amortized O(1) — each timestamp is
    # walked off the head exactly once instead of full-scanning on every
    # request. With the post-bump limits (up to 20k/min), the old list
    # comprehension would have spent meaningful CPU inside the lock.
    _rate_ts: dict[str, deque[float]] = defaultdict(deque)
    _rate_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    _RATE_LIMITS: dict[str, tuple[int, int]] = {
        # (max_requests, window_seconds)
        # Self-hosted single-tenant: limits only exist to catch a genuinely
        # runaway loop. Cost budgets (costs.py) and per-tx wallet caps are
        # the real spend guardrails — these buckets should never fire in
        # normal operation.
        "api_proxy": (6000, 60),
        "vault_resolve": (10000, 60),
        "vault_store": (600, 60),
        "blackboard_read": (20000, 60),
        "blackboard_write": (10000, 60),
        "publish": (20000, 60),
        "notify": (3000, 60),
        "cron_create": (1000, 60),
        "spawn": (600, 60),
        "wallet_read": (6000, 60),
        "wallet_transfer": (600, 60),
        "wallet_execute": (600, 60),
        "image_gen": (600, 60),
        "agent_profile": (6000, 60),
        "upload_stage": (3000, 60),
        "upload_apply": (3000, 60),
        # Self-reported LLM auth failures. Quarantine threshold is 3 so
        # legitimate traffic never approaches this — the bucket exists to
        # cap notification-store writes when a runaway agent retries on a
        # broken credential before its lane gate latches.
        "auth_failure": (60, 60),
    }

    async def _check_rate_limit(endpoint: str, agent_id: str) -> None:
        """Enforce per-agent rate limit. Raises 429 if exceeded."""
        limit, window = _RATE_LIMITS.get(endpoint, (10000, 60))
        bucket_key = f"{endpoint}:{agent_id}"
        async with _rate_locks[bucket_key]:
            now = time.time()
            bucket = _rate_ts[bucket_key]
            cutoff = now - window
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= limit:
                _record_denial(
                    "rate", caller=agent_id, gate=f"rate_limit:{endpoint}",
                )
                raise HTTPException(429, f"Rate limit exceeded for {endpoint}")
            bucket.append(now)

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
        suffix = f":{agent_id}"
        stale = [k for k in _rate_ts if k.endswith(suffix)]
        for k in stale:
            _rate_ts.pop(k, None)
            _rate_locks.pop(k, None)
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

    def _extract_bearer(request: Request) -> str | None:
        """Return the raw ``Authorization: Bearer <token>`` value, or ``None``.

        Unlike ``_extract_verified_agent_id``, this does NOT resolve the
        token to an agent identity — it returns the raw token string for
        callers that need to do their own constant-time comparison
        (e.g. the operator-registration gate in ``/mesh/register``).

        Returns ``None`` when the header is missing or malformed; never
        raises. The caller decides how strict to be.
        """
        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            return None
        token = auth_header[7:]
        return token if token else None

    def _extract_verified_agent_id(request: Request) -> str:
        """Extract and verify agent identity.

        Internal callers (loopback + ``x-mesh-internal`` header) are
        trusted based on the network boundary — they don't need a Bearer
        token. Dashboard proxies, the mesh dispatcher, the CLI manager,
        and health checks all reach the mesh this way and don't have
        access to per-agent Bearer tokens (which are server-side
        secrets). For these callers we trust the ``X-Agent-ID`` header
        and default to ``"operator"`` (the dashboard's persona for
        cross-cutting actions).

        Public callers must present a valid Bearer token. The agent_id
        is derived from the token itself, preventing identity spoofing
        via headers or query parameters.

        Returns 'unknown' when auth is not configured (dev/test mode).
        """
        if _is_internal_caller(request):
            return request.headers.get("X-Agent-ID", "operator")
        if not _auth_tokens:
            # Auth not configured (dev/test mode) — fall back to header hint
            return request.headers.get("X-Agent-ID", "unknown")
        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            _record_denial(
                "auth",
                caller=request.headers.get("X-Agent-ID") or None,
                gate="extract_verified_agent_id:missing_bearer",
            )
            raise HTTPException(401, "Missing authentication token")
        token = auth_header[7:]
        for aid, expected in _auth_tokens.items():
            if hmac.compare_digest(token, expected):
                return aid
        _record_denial(
            "auth",
            caller=request.headers.get("X-Agent-ID") or None,
            gate="extract_verified_agent_id:invalid_token",
        )
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

    def _resolve_browser_target(
        caller_id: str, target_claim: object, request: Request,
    ) -> str:
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
        # Operator trust tier: operator coordinates the fleet, so its
        # delegation reach is not gated by the per-agent can_message
        # matrix. The target-side ``can_use_browser`` check below still
        # fires — operator can only delegate to agents whose own grant
        # actually enables browser access.
        if not _caller_is_operator(caller_id, request):
            if not permissions.can_message(caller_id, target):
                _record_denial(
                    "permission", caller=caller_id, target=target,
                    gate="browser_delegate:can_message",
                )
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
            _record_denial(
                "auth",
                caller=request.headers.get("X-Agent-ID") or None,
                gate="require_any_auth:missing_bearer",
            )
            raise HTTPException(401, "Missing authentication token")
        token = auth_header[7:]
        for expected in _auth_tokens.values():
            if hmac.compare_digest(token, expected):
                return
        _record_denial(
            "auth",
            caller=request.headers.get("X-Agent-ID") or None,
            gate="require_any_auth:invalid_token",
        )
        raise HTTPException(401, "Invalid authentication token")

    def _require_operator_or_internal(request: Request) -> None:
        """Restrict to the operator agent or localhost x-mesh-internal.

        Stricter than ``_require_any_auth``: an arbitrary agent's bearer
        token does NOT pass. Used for fleet-wide pre-computed metrics
        endpoints that previously leaked health, costs, attention list,
        and per-agent budgets to every authenticated agent.
        """
        if not _auth_tokens:
            return
        if request.headers.get("x-mesh-internal"):
            client_host = request.client.host if request.client else ""
            try:
                import ipaddress
                if ipaddress.ip_address(client_host).is_loopback:
                    return
            except (ValueError, AttributeError):
                pass
        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(401, "Missing authentication token")
        token = auth_header[7:]
        operator_token = _auth_tokens.get("operator")
        if operator_token and hmac.compare_digest(token, operator_token):
            return
        # Token is valid for SOME agent but not operator — return 403
        # rather than 401 so legitimate-but-wrong-scope callers can
        # distinguish "I'm authenticated, just not allowed here" from
        # "my credential is bad".
        # Find the actual agent the bearer maps to (don't trust the
        # spoofable ``X-Agent-ID`` header for the denial log — a worker
        # could mislabel itself as ``operator`` here, which would make
        # the structured warning name the wrong caller).
        matched_agent_id: str | None = None
        for aid, expected in _auth_tokens.items():
            if hmac.compare_digest(token, expected):
                matched_agent_id = aid
                break
        if matched_agent_id is not None:
            _record_denial(
                "role",
                caller=matched_agent_id,
                gate="require_operator_or_internal:wrong_token",
            )
            raise HTTPException(403, "Operator-only endpoint")
        _record_denial(
            "auth",
            caller=request.headers.get("X-Agent-ID") or None,
            gate="require_operator_or_internal:invalid_token",
        )
        raise HTTPException(401, "Invalid authentication token")

    # === System Messaging (mesh → agent) ===

    @app.post("/mesh/message")
    async def send_message(msg: AgentMessage, request: Request) -> dict:
        """Route a message to an agent via the mesh router."""
        msg.from_agent = _resolve_agent_id(msg.from_agent, request)
        if not _caller_is_operator(msg.from_agent, request):
            if not permissions.can_message(msg.from_agent, msg.to):
                _record_denial(
                    "permission", caller=msg.from_agent, target=msg.to,
                    gate="message:can_message",
                )
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
        # Operator trust tier: operator coordinates the fleet by design,
        # so wake denials based on its per-agent can_message grant are
        # the wrong layer of enforcement (the user-control layer already
        # gates anything reachable from operator). Workers stay subject
        # to the matrix.
        if not _caller_is_operator(caller, request):
            if not permissions.can_message(caller, target):
                _record_denial(
                    "permission", caller=caller, target=target,
                    gate="wake:can_message",
                )
                raise HTTPException(403, f"Agent {caller} cannot wake {target}")
        await _check_rate_limit("blackboard_write", caller)  # reuse bb rate limit
        if target not in router.agent_registry:
            raise HTTPException(404, f"Agent '{target}' not registered")

        if message:
            wake_msg = sanitize_for_prompt(message)
        else:
            wake_msg = f"You have a new task from {caller}. Call check_inbox() to see it."

        from src.shared.types import MessageOrigin
        # Task 2c: ``_validated_origin`` re-checks ``kind="human"``
        # channel claims against the on-disk pairing record. Forged or
        # unverifiable claims are downgraded to ``kind="agent"`` so the
        # lane payload (and any downstream auth gate that will be added
        # in Task 2d/2e) cannot be tricked by a hostile/buggy adapter.
        origin = _validated_origin(request, caller)

        # Task 2e: block synchronous worker → operator wakes.
        #
        # Workers signal operator-bound completion by writing task
        # records (Task 0 hotfix at ``global/tasks/operator/<id>``); the
        # operator polls those records on heartbeat. This block prevents
        # an agent from steering the operator into a privileged action
        # just by being able to message it.
        #
        # Allowed paths (must NOT be blocked):
        #   * ``caller == "operator"`` — the operator can wake itself
        #     for self-tests / self-resume.
        #   * ``_is_internal_caller(request)`` — loopback +
        #     ``x-mesh-internal``: the dashboard / CLI manager process /
        #     mesh internals can still wake the operator.
        #   * ``origin.kind == "human"`` — a real human action that
        #     survived ``_validated_origin``'s pairing recheck.
        if (
            target == "operator"
            and not _caller_is_operator(caller, request)
            and not _is_internal_caller(request)
            and (origin is None or origin.kind != "human")
        ):
            raise HTTPException(
                403,
                "Worker agents cannot synchronously wake the operator. "
                "Hand off via tasks/operator/* instead; the operator "
                "polls on heartbeat.",
            )
        # Task 2b: missing/invalid origin downgrades to ``kind="agent"``
        # (least-trusted) instead of ``None`` so downstream gates always
        # see an explicit kind. Auto-notify still gated on the original
        # origin presence — agent-default wakes have no addressable
        # channel/user, so there is no notification target.
        had_origin = origin is not None
        if origin is None:
            origin = MessageOrigin(kind="agent", channel="", user="")

        # Bug 2/3 fix: thread the originating task_id through the lane so
        # the recipient's /chat call auto-closes the task when its loop
        # returns. Missing header preserves legacy fire-and-forget wakes.
        task_id = request.headers.get("x-task-id") or None

        if lane_manager is not None and dispatch_loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(
                    lane_manager.enqueue(
                        target, wake_msg, mode="followup",
                        origin=origin, auto_notify=had_origin,
                        task_id=task_id,
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
        if not _caller_is_operator(agent_id, request):
            if not permissions.can_read_blackboard(agent_id, prefix):
                _record_denial(
                    "permission", caller=agent_id, target=prefix,
                    gate="blackboard.list:can_read_blackboard",
                )
                raise HTTPException(403, f"Agent {agent_id} cannot read {prefix}")
        entries = blackboard.list_by_prefix(prefix)
        return [e.model_dump(mode="json") for e in entries]

    @app.get("/mesh/blackboard/{key:path}")
    async def read_blackboard(key: str, agent_id: str, request: Request) -> dict:
        """Read a blackboard entry. Agent must have read permission."""
        agent_id = _resolve_agent_id(agent_id, request)
        await _check_rate_limit("blackboard_read", agent_id)
        if not _caller_is_operator(agent_id, request):
            if not permissions.can_read_blackboard(agent_id, key):
                _record_denial(
                    "permission", caller=agent_id, target=key,
                    gate="blackboard.read:can_read_blackboard",
                )
                raise HTTPException(403, f"Agent {agent_id} cannot read {key}")
        entry = blackboard.read(key)
        if not entry:
            raise HTTPException(404, f"Key not found: {key}")
        # Phase 3 Slice 1 telemetry: count cross-project reads. Skip
        # internal/operator callers (fleet-global by design). No
        # enforcement — pure observability informing the design doc.
        if not _is_internal_caller(request) and _is_blackboard_cross_project(
            agent_id, entry.written_by
        ):
            _record_blackboard_xproject("read")
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
        value_size = len(dumps_safe(value))
        if value_size > _MAX_BB_VALUE_BYTES:
            raise HTTPException(413, f"Value too large ({value_size} bytes, max {_MAX_BB_VALUE_BYTES})")
        if ttl is not None and ttl <= 0:
            raise HTTPException(400, "TTL must be positive")
        if not _caller_is_operator(agent_id, request):
            if not permissions.can_write_blackboard(agent_id, key):
                _record_denial(
                    "permission", caller=agent_id, target=key,
                    gate="blackboard.write:can_write_blackboard",
                )
                raise HTTPException(403, f"Agent {agent_id} cannot write {key}")
        # Phase 3 Slice 1 telemetry: count cross-project writes against an
        # EXISTING entry (new keys are by definition not cross-project).
        # Skip internal/operator callers (fleet-global by design).
        if not _is_internal_caller(request):
            existing = blackboard.read(key)
            if existing is not None and _is_blackboard_cross_project(
                agent_id, existing.written_by
            ):
                _record_blackboard_xproject("write")
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
        if not _caller_is_operator(agent_id, request):
            if not permissions.can_write_blackboard(agent_id, key):
                _record_denial(
                    "permission", caller=agent_id, target=key,
                    gate="blackboard.delete:can_write_blackboard",
                )
                raise HTTPException(403, f"Agent {agent_id} cannot write {key}")
        # Protect history namespace (including project-scoped keys)
        bare = key.split("/", 2)[2] if key.startswith("projects/") and key.count("/") >= 2 else key
        if bare.startswith("history/"):
            raise HTTPException(400, "Cannot delete from history namespace")
        # Phase 3 Slice 1 telemetry: count cross-project deletes (a delete
        # is a write that mutates the key). Skip internal/operator callers.
        if not _is_internal_caller(request):
            existing = blackboard.read(key)
            if existing is not None and _is_blackboard_cross_project(
                agent_id, existing.written_by
            ):
                _record_blackboard_xproject("write")
        try:
            blackboard.delete(key, deleted_by=agent_id)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if event_bus is not None:
            try:
                event_bus.emit(
                    "blackboard_delete", agent=agent_id,
                    data={"key": key, "deleted_by": agent_id},
                )
            except Exception as e:
                logger.debug("blackboard_delete emit failed: %s", e)
        return {"deleted": True, "key": key}

    @app.post("/mesh/blackboard/watch")
    async def watch_blackboard(data: BlackboardWatchRequest, request: Request) -> dict:
        """Register a glob pattern watch on blackboard keys."""
        agent_id = _resolve_agent_id(data.agent_id, request)
        pattern = data.pattern
        if not _caller_is_operator(agent_id, request):
            if not permissions.can_read_blackboard(agent_id, pattern):
                _record_denial(
                    "permission", caller=agent_id, target=pattern,
                    gate="blackboard.watch:can_read_blackboard",
                )
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
        value_size = len(dumps_safe(body.value))
        if value_size > _MAX_BB_VALUE_BYTES:
            raise HTTPException(413, f"Value too large ({value_size} bytes, max {_MAX_BB_VALUE_BYTES})")
        if not _caller_is_operator(agent_id, request):
            if not permissions.can_write_blackboard(agent_id, key):
                _record_denial(
                    "permission", caller=agent_id, target=key,
                    gate="blackboard.claim:can_write_blackboard",
                )
                raise HTTPException(403, f"Agent {agent_id} cannot write {key}")
        # Phase 3 Slice 1 telemetry: count cross-project CAS writes against
        # an EXISTING entry. Skip internal/operator callers.
        if not _is_internal_caller(request):
            existing = blackboard.read(key)
            if existing is not None and _is_blackboard_cross_project(
                agent_id, existing.written_by
            ):
                _record_blackboard_xproject("write")
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
                _record_denial(
                    "scope", caller=event.source, target=event.topic,
                    gate="publish:project_prefix",
                    extra={"caller_project": source_project},
                )
                raise HTTPException(
                    403,
                    f"Agent {event.source} (project={source_project}) cannot publish to topic '{event.topic}'"
                )

        if not _caller_is_operator(event.source, request):
            if not permissions.can_publish(event.source, event.topic):
                _record_denial(
                    "permission", caller=event.source, target=event.topic,
                    gate="publish:can_publish",
                )
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
                    f"{dumps_safe(event.payload)[:500]}"
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
                _record_denial(
                    "scope", caller=agent_id, target=topic,
                    gate="subscribe:project_prefix",
                    extra={"caller_project": sub_project},
                )
                raise HTTPException(
                    403,
                    f"Agent {agent_id} (project={sub_project}) cannot subscribe to topic '{topic}'"
                )

        if not _caller_is_operator(agent_id, request):
            if not permissions.can_subscribe(agent_id, topic):
                _record_denial(
                    "permission", caller=agent_id, target=topic,
                    gate="subscribe:can_subscribe",
                )
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
        if not _caller_is_operator(agent_id, request):
            if not permissions.can_use_api(agent_id, api_request.service):
                _record_denial(
                    "permission", caller=agent_id, target=api_request.service,
                    gate="api:can_use_api",
                )
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
        if not _caller_is_operator(agent_id, request):
            if not permissions.can_use_api(agent_id, api_request.service):
                _record_denial(
                    "permission", caller=agent_id, target=api_request.service,
                    gate="api_stream:can_use_api",
                )
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
        """Agent registers itself with the mesh on startup.

        Reserved-ID handling (Task 4):
          * ``mesh`` / ``canary-probe`` — always rejected (system-only).
          * ``operator`` — accepted only when the caller's bearer token
            matches ``auth_tokens["operator"]`` (constant-time compare).
            Fail-closed: if the token pool is empty (no auth configured)
            the operator path is rejected as well — operator identity is
            cryptographic, not positional.
          * everything else — passes the standard
            ``_validate_agent_id`` regex check and is identified by the
            Bearer token via ``_resolve_agent_id``.
        """
        requested_id = data.get("agent_id", "")
        # Format check first (so reserved-ID gating below sees a well-formed
        # claim). Note: we deliberately don't call ``_validate_agent_id``
        # here — that helper rejects the operator outright, but Task 4
        # needs to accept the operator's claim when the bearer matches.
        if not requested_id or not _AGENT_ID_RE.match(requested_id):
            raise HTTPException(
                400,
                "Invalid agent_id: must be 1-64 alphanumeric/hyphen/underscore chars",
            )

        if requested_id == "operator":
            # Cryptographic gate: require a bearer matching the
            # operator-specific token. The error message intentionally
            # does NOT echo the supplied bearer — comparing in
            # constant time still leaks via debug logs if we surface it.
            expected = _auth_tokens.get("operator") if _auth_tokens else None
            bearer = _extract_bearer(request)
            if (
                not expected
                or not bearer
                or not hmac.compare_digest(bearer, expected)
            ):
                raise HTTPException(
                    403,
                    "Reserved agent_id 'operator' requires the operator's bearer token",
                )
            agent_id = "operator"
        elif requested_id in {"mesh", "canary-probe"}:
            # Stay rejected for any caller, including the operator's
            # bearer. ``canary-probe`` continues to use the internal-only
            # registration path (router.register_agent directly).
            raise HTTPException(
                403, f"Reserved agent_id '{requested_id}' cannot register",
            )
        else:
            # Standard path: format-validate and resolve identity from
            # the Bearer token (auth on) or trust the caller (dev/test).
            agent_id = _validate_agent_id(requested_id)
            agent_id = _resolve_agent_id(agent_id, request)

        capabilities = data.get("capabilities", [])
        if not isinstance(capabilities, list) or len(capabilities) > 200:
            raise HTTPException(400, "capabilities must be a list of at most 200 items")
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
        # TODO(Task 4): gate on ``permissions.can_request_user_credentials``.
        # The capability bit is wired in Task 3 (defaults False for workers,
        # True for operator), but enforcing it here today would block every
        # non-operator agent from asking the user for a credential.  Task 4
        # is responsible for populating the field on workers that actually
        # need it (template-driven) before this gate flips to enforced.

        name = data.get("name", "")
        description = data.get("description", "")
        service = data.get("service", name)

        if not name:
            raise HTTPException(400, "Credential name is required")
        if not re.match(r"^[a-zA-Z0-9_.\-]{1,128}$", name):
            raise HTTPException(400, "Invalid credential name")

        request_id = _record_help_request(
            "credential_request",
            agent_id,
            {"name": name, "service": service[:128]},
        )

        if event_bus:
            event_bus.emit(
                "credential_request",
                agent=agent_id,
                data={
                    "name": name,
                    "description": description[:500],
                    "service": service[:128],
                    "request_id": request_id,
                },
            )

        return {"requested": True, "name": name, "request_id": request_id}

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
            caller_id, data.get("target_agent_id") or "", request,
        )

        # Per-action gate applies to the EFFECTIVE target (`agent_id`), same
        # semantics as ``/mesh/browser/command``: an operator with
        # ``browser_actions=["*"]`` cannot exercise actions the target was
        # never granted.  Without this check, an operator who narrows a
        # template to ``browser_actions=["navigate"]`` would see the first
        # call (``browser_command`` → e.g. navigate) gated correctly but
        # the dedicated handoff endpoint would still succeed — permission
        # narrowing wouldn't actually narrow.
        if not permissions.can_browser_action(agent_id, "request_browser_login"):
            raise HTTPException(
                403, "Agent not permitted to perform 'request_browser_login'",
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

        request_id = _record_help_request(
            "browser_login_request",
            agent_id,
            {"service": service[:128]},
        )

        if event_bus:
            # OAuth callback URLs (?code=...&state=...) and other
            # query-string secrets must not leak to the dashboard event
            # history. ``redact_url`` strips sensitive query params
            # while preserving scheme/host/path so the operator still
            # sees what target the agent meant to log into.
            event_bus.emit(
                "browser_login_request",
                agent=agent_id,
                data={
                    "url": redact_url(url)[:2048],
                    "service": service[:128],
                    "description": description[:500],
                    "request_id": request_id,
                },
            )

        return {
            "requested": True,
            "service": service,
            "target_agent": agent_id,
            "request_id": request_id,
        }

    @app.post("/mesh/browser-captcha-help-request")
    async def browser_captcha_help_request(
        data: dict, request: Request,
    ) -> dict:
        """Phase 8 §11.14 — agent requests human help for a CAPTCHA.

        Mirrors :func:`browser_login_request` exactly. Emits a dashboard
        ``browser_captcha_help_request`` event so the operator sees a
        handoff card with the VNC viewer for the target agent's browser.
        """
        caller_id = _resolve_agent_id(data.get("agent_id", ""), request)
        agent_id = _resolve_browser_target(
            caller_id, data.get("target_agent_id") or "", request,
        )

        # Per-action gate (mirrors ``browser_login_request`` and
        # ``/mesh/browser/command``).  Without this, an operator who
        # narrows a template's ``browser_actions`` to exclude the captcha
        # handoff would see the ``browser_command`` route correctly
        # rejected for ``request_captcha_help`` (PR #769) but the dedicated
        # endpoint here would still succeed — defeating permission
        # narrowing.
        if not permissions.can_browser_action(agent_id, "request_captcha_help"):
            raise HTTPException(
                403, "Agent not permitted to perform 'request_captcha_help'",
            )

        await _check_rate_limit("notify", caller_id)

        service = data.get("service", "").strip()
        description = data.get("description", "").strip()

        if not service:
            raise HTTPException(400, "Service name is required")
        if not description:
            raise HTTPException(400, "Description is required")

        request_id = _record_help_request(
            "browser_captcha_help_request",
            agent_id,
            {"service": service[:128]},
        )

        if event_bus:
            event_bus.emit(
                "browser_captcha_help_request",
                agent=agent_id,
                data={
                    "service": service[:128],
                    "description": description[:500],
                    "request_id": request_id,
                },
            )

        return {
            "requested": True,
            "service": service,
            "target_agent": agent_id,
            "request_id": request_id,
        }

    def _cancel_help_request(
        kind: str, request_id: str, reason: str,
    ) -> dict:
        """Resolve an open help request as cancelled.

        Returns the popped record. Raises HTTPException(404) if the id
        is unknown or already resolved. Caller is responsible for
        emitting any follow-up event / steer.
        """
        record = help_requests.get(request_id)
        if record is None:
            raise HTTPException(
                404, f"{kind} request not found or already resolved",
            )
        if record.get("kind") != kind:
            raise HTTPException(
                404, f"{kind} request not found or already resolved",
            )
        if record.get("status") != "open":
            raise HTTPException(
                404, f"{kind} request not found or already resolved",
            )
        record["status"] = "cancelled"
        record["reason"] = reason
        # Pop after mutating so the dict is the source of truth on
        # whether the request is still resolvable.
        help_requests.pop(request_id, None)
        return record

    async def _enqueue_cancel_steer(agent_id: str, message: str) -> None:
        """Push a steer message to the awaiting agent.

        Best-effort: silently swallows if no lane manager is wired
        (mesh-only test setup) or the agent isn't registered.
        """
        if lane_manager is None or not agent_id:
            return
        try:
            from src.shared.trace import new_trace_id
            await lane_manager.enqueue(
                agent_id, sanitize_for_prompt(message),
                mode="steer", trace_id=new_trace_id(),
            )
        except Exception as e:
            logger.warning("cancel-steer enqueue failed for %s: %s", agent_id, e)

    @app.post("/mesh/credential-request/{request_id}/cancel")
    async def credential_request_cancel(
        request_id: str, data: dict, request: Request,
    ) -> dict:
        """User cancelled a pending credential request from the dashboard.

        Pops the open record, emits ``credential_request_cancelled``
        so all card copies update, and pushes a steer message to the
        requesting agent so it can react instead of waiting on a
        credential that will never arrive.
        """
        # Loopback or operator only — same access model as the dashboard
        # cancel proxy that fronts this.
        caller = _extract_verified_agent_id(request)
        if not _caller_is_operator(caller, request) and not _is_internal_caller(request):
            raise HTTPException(
                403, "Only the operator can cancel a credential request",
            )
        reason = (data or {}).get("reason", "user_cancelled")
        record = _cancel_help_request("credential_request", request_id, reason)
        agent_id = record["agent_id"]
        service = record["payload"].get("service") or record["payload"].get("name", "")
        name = record["payload"].get("name", "")
        if event_bus:
            event_bus.emit(
                "credential_request_cancelled",
                agent=agent_id,
                data={
                    "request_id": request_id,
                    "name": name,
                    "service": service,
                    "reason": reason,
                },
            )
        await _enqueue_cancel_steer(
            agent_id,
            f"The user cancelled your request for credential '{name}'. "
            f"They did not provide it. Skip this step or ask "
            f"differently — do not retry the same request immediately.",
        )
        return {
            "ok": True,
            "request_id": request_id,
            "status": "cancelled",
            "reason": reason,
        }

    @app.post("/mesh/browser-login-request/{request_id}/cancel")
    async def browser_login_request_cancel(
        request_id: str, data: dict, request: Request,
    ) -> dict:
        """User cancelled a pending browser login request."""
        caller = _extract_verified_agent_id(request)
        if not _caller_is_operator(caller, request) and not _is_internal_caller(request):
            raise HTTPException(
                403, "Only the operator can cancel a browser login request",
            )
        reason = (data or {}).get("reason", "user_cancelled")
        record = _cancel_help_request(
            "browser_login_request", request_id, reason,
        )
        agent_id = record["agent_id"]
        service = record["payload"].get("service", "")
        if event_bus:
            event_bus.emit(
                "browser_login_cancelled",
                agent=agent_id,
                data={
                    "request_id": request_id,
                    "service": service,
                    "reason": reason,
                },
            )
        await _enqueue_cancel_steer(
            agent_id,
            f"The user cancelled the browser login for {service}. "
            f"You may need to find an alternative approach or ask "
            f"again later. Do not retry the same login immediately.",
        )
        return {
            "ok": True,
            "request_id": request_id,
            "status": "cancelled",
            "reason": reason,
        }

    @app.post("/mesh/browser-captcha-help-request/{request_id}/cancel")
    async def browser_captcha_help_request_cancel(
        request_id: str, data: dict, request: Request,
    ) -> dict:
        """User cancelled a pending CAPTCHA-help request."""
        caller = _extract_verified_agent_id(request)
        if not _caller_is_operator(caller, request) and not _is_internal_caller(request):
            raise HTTPException(
                403, "Only the operator can cancel a CAPTCHA-help request",
            )
        reason = (data or {}).get("reason", "user_cancelled")
        record = _cancel_help_request(
            "browser_captcha_help_request", request_id, reason,
        )
        agent_id = record["agent_id"]
        service = record["payload"].get("service", "")
        if event_bus:
            event_bus.emit(
                "browser_captcha_help_cancelled",
                agent=agent_id,
                data={
                    "request_id": request_id,
                    "service": service,
                    "reason": reason,
                },
            )
        await _enqueue_cancel_steer(
            agent_id,
            f"The user cancelled the CAPTCHA help request for "
            f"{service}. Try a different approach (wait + retry, "
            f"escalate via notify_user). Do not re-request the same "
            f"captcha help immediately.",
        )
        return {
            "ok": True,
            "request_id": request_id,
            "status": "cancelled",
            "reason": reason,
        }

    @app.get("/mesh/agents")
    async def list_agents(request: Request, project: str = "", agent_id: str = "") -> dict:
        """List registered agents, optionally scoped by project or agent_id.

        - project set: return only that project's members
        - agent_id set (standalone): return only that agent
        - neither (dashboard/internal): return all (under enforce mode,
          worker callers see only their own projects + operator)

        Task 5 layered a per-caller filter on the unscoped path. Today's
        legacy behavior is "every authenticated agent sees the full
        fleet" — under ``OPENLEGION_TEAM_SCOPE_MODE=enforce`` (the
        default) workers see only members of their own projects (plus
        the always-global operator). Under ``warn`` the response shape
        stays legacy and a structured ``scope-warn`` log line is
        emitted so operators can soak before flipping.
        """
        if agent_id:
            agent_id = _resolve_agent_id(agent_id, request)
        else:
            _require_any_auth(request)

        # Resolve the caller for scope filtering. ``_extract_verified_agent_id``
        # handles dev/test mode (no auth tokens → ``X-Agent-ID`` header
        # hint or ``"unknown"``) and production (Bearer token → identity).
        # Internal loopback callers are treated as global; their
        # ``X-Agent-ID`` hint is ignored. The endpoint already gated
        # auth above, so caller resolution can't 401 here.
        caller_is_internal = _is_internal_caller(request)
        if caller_is_internal:
            caller = "mesh"
        else:
            try:
                caller = _extract_verified_agent_id(request)
            except HTTPException:
                # Already passed _require_any_auth, but be defensive.
                caller = "unknown"
        caller_is_global = caller_is_internal or _caller_is_operator(caller, request)

        # Task 8 — pull structured routing fields from agents.yaml. The
        # ``capabilities`` key on the entry is the runtime tool list
        # (router.get_capabilities) — keep that name as-is. Surface the
        # human-routing capabilities + the four siblings under explicit
        # keys so callers can distinguish them from tool capabilities.
        from src.cli.config import _load_config as _load_cfg_for_listing
        try:
            _cfg_for_listing = _load_cfg_for_listing()
        except Exception:
            _cfg_for_listing = {"agents": {}}
        _agents_cfg_listing = _cfg_for_listing.get("agents", {}) or {}

        def _agent_entry(aid: str, url: str) -> dict:
            entry: dict = {"url": url, "role": router.agent_roles.get(aid, "")}
            entry["capabilities"] = router.get_capabilities(aid)
            acfg = _agents_cfg_listing.get(aid) or {}
            # Structured routing fields (Task 8). The ``interface_*`` keys
            # are the human-facing routing surface; the bare
            # ``capabilities`` key remains the runtime tool/skill list to
            # avoid breaking back-compat with existing dashboard / CLI
            # consumers.
            entry["interface_capabilities"] = list(acfg.get("capabilities") or [])
            entry["preferred_inputs"] = list(acfg.get("preferred_inputs") or [])
            entry["expected_outputs"] = list(acfg.get("expected_outputs") or [])
            entry["escalation_to"] = acfg.get("escalation_to")
            entry["forbidden"] = list(acfg.get("forbidden") or [])
            proj = _agent_projects.get(aid)
            if proj:
                entry["project"] = proj
            if aid == "operator":
                entry["scope"] = "global"
            return entry

        if project:
            from src.cli.config import _load_projects
            projects = _load_projects()
            pdata = projects.get(project)
            if pdata is None:
                logger.warning("list_agents: unknown project %r", project)
                return {}
            members = set(pdata.get("members", []))

            # Task 5: only members of the requested project (or global
            # callers) may scope by it. Under warn mode, log but allow;
            # under enforce, return empty.
            if not caller_is_global and caller not in members:
                _record_scope_warn()
                logger.warning(
                    "scope-warn: caller=%s requested /mesh/agents?project=%s "
                    "but is not a member; mode=%s",
                    caller, project, _TEAM_SCOPE_MODE,
                )
                if _TEAM_SCOPE_MODE == "enforce":
                    return {}

            result = {
                aid: _agent_entry(aid, url)
                for aid, url in router.agent_registry.items()
                if aid in members
            }
            # Operator is fleet-global by design: project agents must be able to
            # discover and hand off back to it regardless of project membership.
            op_url = router.agent_registry.get("operator")
            if op_url is not None and "operator" not in result:
                result["operator"] = _agent_entry("operator", op_url)
            return result
        if agent_id:
            url = router.agent_registry.get(agent_id)
            if url:
                return {agent_id: _agent_entry(agent_id, url)}
            return {}

        # Unscoped path: full fleet for global callers; per-caller-project
        # filter for workers (warn-logged, enforce-applied).
        full_fleet = {
            aid: _agent_entry(aid, url)
            for aid, url in router.agent_registry.items()
        }
        if caller_is_global:
            return full_fleet

        own_projects = _caller_projects(caller)
        # Visible set: members of any project the caller belongs to,
        # plus the always-global operator, plus the caller itself
        # (a worker should always see its own entry — including
        # standalone agents who belong to no project).
        visible_members: set[str] = {caller}
        if own_projects:
            from src.cli.config import _load_projects
            projects = _load_projects()
            for pname in own_projects:
                visible_members.update(projects.get(pname, {}).get("members", []))
        if "operator" in router.agent_registry:
            visible_members.add("operator")

        filtered = {
            aid: entry for aid, entry in full_fleet.items()
            if aid in visible_members
        }

        # If the filter would have shrunk the response, emit warn
        # telemetry so ops can size the soak before flipping the flag.
        if len(filtered) < len(full_fleet):
            _record_scope_warn()
            logger.warning(
                "scope-warn: caller=%s requested /mesh/agents (no project filter); "
                "would return %d under enforce, returning %d under %s",
                caller, len(filtered), len(full_fleet), _TEAM_SCOPE_MODE,
            )
            if _TEAM_SCOPE_MODE == "enforce":
                return filtered
        return full_fleet

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
            include_models = _caller_is_operator(agent_id, request)
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

            if _caller_is_operator(agent_id, request):
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

        if section in ("llm", "all"):
            # BYOK visibility: surface the set of providers that actually
            # have credentials configured so agents can introspect what
            # models are reachable before they request one. Bug 5 —
            # paired with the mesh-side validation in
            # ``create_custom_agent``.
            available_providers: list[str] = []
            allowed_models: dict[str, list[str]] = {}
            credential_kinds: dict[str, str] = {}
            if credential_vault is not None:
                try:
                    available_providers = sorted(
                        credential_vault.get_providers_with_credentials(),
                    )
                except Exception as e:
                    logger.debug("introspect available_providers failed: %s", e)
                try:
                    allowed_models = credential_vault.get_allowed_models()
                except Exception as e:
                    logger.debug("introspect allowed_models failed: %s", e)
                try:
                    credential_kinds = {
                        p: credential_vault.get_credential_kind(p)
                        for p in available_providers
                    }
                except Exception as e:
                    logger.debug("introspect credential_kinds failed: %s", e)
            result["llm"] = {
                "available_providers": available_providers,
                "allowed_models": allowed_models,
                "credential_kinds": credential_kinds,
            }

        return result

    # === Team Costs ===

    @app.get("/mesh/costs/team/{team}")
    async def get_team_costs(team: str, request: Request, period: str = "today") -> dict:
        """Return aggregated cost data for a team."""
        _require_any_auth(request)
        if cost_tracker is None:
            raise HTTPException(503, "Cost tracker not available")
        if not hasattr(cost_tracker, "get_project_spend"):
            raise HTTPException(503, "Team cost tracking not available")
        return cost_tracker.get_project_spend(team, period)

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
            if not _caller_is_operator(agent_id, request):
                if not permissions.can_manage_cron(agent_id):
                    _record_denial(
                        "permission", caller=agent_id,
                        gate="cron.create:can_manage_cron",
                    )
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
        if not _caller_is_operator(agent_id, request):
            if not permissions.can_manage_cron(agent_id):
                _record_denial(
                    "permission", caller=agent_id,
                    gate="cron.update:can_manage_cron",
                )
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
        if not _caller_is_operator(agent_id, request):
            if not permissions.can_manage_cron(agent_id):
                _record_denial(
                    "permission", caller=agent_id,
                    gate="cron.delete:can_manage_cron",
                )
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
        if not _caller_is_operator(spawned_by, request):
            if not permissions.can_spawn(spawned_by):
                _record_denial(
                    "permission", caller=spawned_by,
                    gate="spawn:can_spawn",
                )
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

        Note: agent creation is per-slot — a mid-loop failure leaves
        earlier-created agents in place. Callers should verify the
        returned ``created`` list matches their intent and inspect the
        on-disk fleet.
        """
        if container_manager is None:
            raise HTTPException(503, "Container manager not available")

        # Auth + permission check
        spawned_by = _resolve_agent_id(data.get("spawned_by", "unknown"), request)
        if _auth_tokens:
            # Applying a fleet template creates DURABLE named agents — Task 3
            # split this off ``can_spawn`` (which is now ephemeral-only) onto
            # the new control-plane permission ``can_manage_fleet``.
            if not _caller_is_operator(spawned_by, request):
                if not permissions.can_manage_fleet(spawned_by):
                    _record_denial(
                        "permission", caller=spawned_by,
                        gate="fleet.apply:can_manage_fleet",
                    )
                    raise HTTPException(
                        403,
                        f"Agent {spawned_by} is not allowed to apply fleet templates "
                        "(requires can_manage_fleet)",
                )

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

        # Optional per-agent overrides (PR-N v2): allowed fields are
        # {model, instructions, soul, heartbeat, interface}. `role` is
        # intentionally NOT overrideable here — templates define role per slot
        # and changing it post-creation is a different operation (rename).
        # Validated UPFRONT — no agent is created if any override is invalid.
        agent_overrides = data.get("agent_overrides") or {}
        if agent_overrides:
            if not isinstance(agent_overrides, dict):
                raise HTTPException(400, "agent_overrides must be an object")
            _ALLOWED_OVERRIDE_FIELDS = {
                "model", "instructions", "soul", "heartbeat", "interface",
            }
            # Per-field length caps mirror src/agent/server.py _FILE_CAPS:
            #   INSTRUCTIONS.md: 12000, SOUL.md: 4000, INTERFACE.md: 4000,
            #   HEARTBEAT.md: None (uncapped).
            _STRING_FIELD_CAPS: dict[str, int | None] = {
                "instructions": 12000,
                "soul": 4000,
                "heartbeat": None,
                "interface": 4000,
            }

            # 1. Unknown agent names
            unknown_agents = [
                name for name in agent_overrides if name not in tpl_agents
            ]
            if unknown_agents:
                raise HTTPException(
                    400,
                    "agent_overrides references unknown agent(s): "
                    f"{sorted(unknown_agents)}. "
                    f"Template '{template_name}' defines: {sorted(tpl_agents.keys())}",
                )

            # 2. Per-override field/value validation
            from src.shared.models import _resolve_litellm_key
            for agent_name, override in agent_overrides.items():
                if not isinstance(override, dict):
                    raise HTTPException(
                        400,
                        f"agent_overrides['{agent_name}'] must be an object, "
                        f"got {type(override).__name__}",
                    )
                bad_fields = [
                    k for k in override if k not in _ALLOWED_OVERRIDE_FIELDS
                ]
                if bad_fields:
                    raise HTTPException(
                        400,
                        f"agent_overrides['{agent_name}'] has unsupported "
                        f"field(s): {sorted(bad_fields)}. "
                        f"Allowed: {sorted(_ALLOWED_OVERRIDE_FIELDS)}",
                    )
                if "model" in override:
                    mv = override["model"]
                    if not isinstance(mv, str) or not mv.strip():
                        raise HTTPException(
                            400,
                            f"agent_overrides['{agent_name}'].model must be a "
                            "non-empty string",
                        )
                    if _resolve_litellm_key(mv) is None:
                        raise HTTPException(
                            400,
                            f"agent_overrides['{agent_name}'].model "
                            f"'{mv}' is not a known model",
                        )
                for _field, _cap in _STRING_FIELD_CAPS.items():
                    if _field not in override:
                        continue
                    val = override[_field]
                    if not isinstance(val, str):
                        raise HTTPException(
                            400,
                            f"agent_overrides['{agent_name}'].{_field} "
                            "must be a string",
                        )
                    if _cap is not None and len(val) > _cap:
                        raise HTTPException(
                            413,
                            f"agent_overrides['{agent_name}'].{_field} "
                            f"exceeds cap ({len(val)} > {_cap} chars)",
                        )

        # Credential-compatibility check on every model the apply will
        # actually use — runs UPFRONT so a bad model rejects the whole
        # call before any agent is created. Mirrors the gate on
        # ``create_custom_agent`` / ``edit-soft`` (Bug 3, "silent model
        # rejection"): without this the slot starts and dies on its
        # first LLM call with no surfaced reason.
        #
        # Validation and the creation site below BOTH route through
        # ``resolve_slot_model`` so the model we validate is exactly the
        # model we end up handing to the container. P1.1 — see helper
        # docstring for the precedence rules and the None-coercion
        # contract.
        from src.shared.models import resolve_slot_model
        _cfg_for_resolve = _load_config()
        _default_model_for_resolve = _cfg_for_resolve.get(
            "llm", {},
        ).get("default_model", "openai/gpt-4o-mini")
        if credential_vault is not None:
            _seen_models: set[str] = set()
            for slot_name, slot_def in tpl_agents.items():
                _slot_model = resolve_slot_model(
                    slot_name,
                    slot_def,
                    agent_overrides,
                    model_override,
                    _default_model_for_resolve,
                )
                if not _slot_model or _slot_model in _seen_models:
                    continue
                _seen_models.add(_slot_model)
                _compatible, _reason = credential_vault.is_model_compatible(
                    _slot_model,
                )
                if not _compatible:
                    raise HTTPException(
                        400,
                        f"Slot '{slot_name}' model '{_slot_model}': "
                        + (_reason or "not compatible with current credentials."),
                    )

        # Apply template to create config entries
        created_names = _apply_template(
            template_name, tpl, agent_overrides=agent_overrides or None,
        )
        # _apply_template calls _add_agent_permissions for each new agent;
        # reload the live matrix so /mesh/register sees the on-disk perms
        # instead of falling through to default/deny-all. Cheap no-op when
        # created_names is empty.
        permissions.reload()
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
            # P1.1 — resolve via the same helper used at validation so
            # there's no precedence drift. ``_apply_template`` already
            # wrote the slot-override-or-template-default model to acfg
            # (no awareness of the top-level model_override), so reading
            # ``slot_def`` straight from ``tpl_agents`` gives us the
            # untouched template view that ``resolve_slot_model`` needs
            # to apply the canonical precedence
            # (slot override > top-level > template default > config default).
            slot_def = tpl_agents.get(agent_name, {})
            agent_model = resolve_slot_model(
                agent_name,
                slot_def,
                agent_overrides,
                model_override,
                default_model,
            )
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

        # Auth + permission check. Creating a durable named agent is a
        # control-plane action — Task 3 split this off ``can_spawn`` onto
        # the dedicated ``can_manage_fleet`` capability.
        agent_id = _resolve_agent_id(data.get("agent_id", ""), request)
        await _check_rate_limit("spawn", agent_id)
        if not _caller_is_operator(agent_id, request):
            if not permissions.can_manage_fleet(agent_id):
                _record_denial(
                    "permission", caller=agent_id,
                    gate="agent.create:can_manage_fleet",
                )
                raise HTTPException(
                    403,
                    f"Agent {agent_id} is not allowed to create agents "
                    "(requires can_manage_fleet)",
                )

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

        # BYOK-safety: validate the chosen model's provider has
        # credentials configured before we spin a container that would
        # otherwise die on its first LLM call. Bug 5 — silent
        # dead-on-arrival agents on deployments that only have one
        # provider key set. Skip when no vault is wired (test
        # harnesses, sandbox transport) — the vault is the only way
        # to enumerate which providers actually have OAuth state.
        if credential_vault is not None:
            from src.shared.models import resolve_provider_for_model
            provider = resolve_provider_for_model(model)
            if provider:
                available = credential_vault.get_providers_with_credentials()
                if provider not in available:
                    available_list = sorted(available) if available else "none"
                    raise HTTPException(
                        400,
                        f"Model '{model}' requires '{provider}' credentials, "
                        f"but no {provider.upper()} key is configured. "
                        f"Available providers: {available_list}. Set "
                        f"OPENLEGION_SYSTEM_{provider.upper()}_API_KEY or "
                        "pick a different model.",
                    )
            # Credential-kind-aware check: OAuth-only providers only accept
            # specific models. Surface the allowed list so the operator
            # doesn't have to guess (see Fix 2 in the seam follow-up).
            compatible, reason = credential_vault.is_model_compatible(model)
            if not compatible:
                raise HTTPException(400, reason or f"Model '{model}' is not compatible.")

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
        # Operator-created agents need the same coordination defaults as
        # template-created agents — empty blackboard_read/write would lock
        # them out of the coordination protocol entirely (and skip the
        # auto-watch setup at /mesh/register, which is gated on
        # blackboard_read being truthy). Mirrors the operator-permission
        # ceiling in operator_tools.py and the starter.yaml template.
        default_perms = {
            "blackboard_read":  ["*"],
            "blackboard_write": ["tasks/*", "context/*", "status/*", "output/*", "artifacts/*"],
            "can_publish":      ["*"],
            "can_subscribe":    ["*"],
            "can_use_browser":  True,
            "can_manage_cron":  True,
        }
        _add_agent_permissions(name, permissions=default_perms)
        # _add_agent_permissions writes to config/permissions.json on disk;
        # the live PermissionMatrix has to reload or the agent's imminent
        # /mesh/register call will fall through to default/deny-all (cf. PR
        # #656 which added the same reload for the no-defaults case).
        permissions.reload()
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
            if not _caller_is_operator(requesting_agent, request):
                if not permissions.can_message(requesting_agent, agent_id):
                    _record_denial(
                        "permission", caller=requesting_agent, target=agent_id,
                        gate="agent.history:can_message",
                    )
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
            if not _caller_is_operator(requesting_agent, request):
                if not permissions.can_message(requesting_agent, agent_id):
                    _record_denial(
                        "permission", caller=requesting_agent, target=agent_id,
                        gate="agent.profile:can_message",
                    )
                    raise HTTPException(403, f"Agent {requesting_agent} cannot read profile of {agent_id}")
        else:
            _require_any_auth(request)

        if agent_id not in router.agent_registry:
            raise HTTPException(404, f"Agent not found: {agent_id}")

        # -- Mesh-derived metadata (verifiable, always fresh) --
        role = router.agent_roles.get(agent_id, "")
        capabilities = router.get_capabilities(agent_id)

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

        # PR-L' — most recent task_events row for this agent. Distinct
        # from ``last_active`` (health probe response) so the dashboard
        # card can render "Last seen 4m ago · Last task 12m ago" — the
        # two diverge when an agent is healthy but hasn't picked up
        # work, OR completed work but is now offline.
        last_task_event_ts: str | None = None
        if tasks_store is not None:
            try:
                ts = tasks_store.last_event_ts_for_agent(agent_id)
                if ts is not None:
                    from datetime import datetime, timezone
                    last_task_event_ts = datetime.fromtimestamp(
                        ts, tz=timezone.utc,
                    ).isoformat()
            except Exception as e:
                logger.debug(
                    "last_event_ts_for_agent failed for '%s': %s",
                    agent_id, e,
                )

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

        # Task 8 — structured routing fields from agents.yaml. ``capabilities``
        # remains the runtime tool/skill list (router.get_capabilities); the
        # human-routing capabilities live under ``interface_capabilities`` to
        # avoid the naming collision. The other four sibling fields keep
        # their natural names (no collision).
        from src.cli.config import _load_config as _load_cfg_for_profile
        try:
            _cfg_for_profile = _load_cfg_for_profile()
        except Exception:
            _cfg_for_profile = {"agents": {}}
        _acfg_profile = (_cfg_for_profile.get("agents", {}) or {}).get(agent_id, {}) or {}
        interface_capabilities = list(_acfg_profile.get("capabilities") or [])
        preferred_inputs = list(_acfg_profile.get("preferred_inputs") or [])
        expected_outputs = list(_acfg_profile.get("expected_outputs") or [])
        escalation_to = _acfg_profile.get("escalation_to")
        forbidden = list(_acfg_profile.get("forbidden") or [])

        # Runtime debugging fields: operator-or-internal only. Two reasons:
        # (a) spend totals + heartbeat liveness are operational visibility
        #     that peer agents shouldn't have over each other, mirroring how
        #     /mesh/system/metrics gates per-agent cost behind the same
        #     operator-or-internal tier;
        # (b) /profile is in the routing hot path (peer agents call it for
        #     capability/escalation discovery), and we don't want to pay
        #     two SQL aggregates per call on that path. Skipping them when
        #     the caller isn't operator-or-internal keeps the hot path lean.
        caller_for_gate = _resolve_agent_id("", request)
        runtime_visible = (
            _caller_is_operator(caller_for_gate, request) or _is_internal_caller(request)
        )

        runtime_fields: dict[str, object] = {}
        if runtime_visible:
            last_heartbeat_at: str | None = None
            if cron_scheduler is not None:
                hb_job = cron_scheduler.find_heartbeat_job(agent_id)
                if hb_job is not None:
                    last_heartbeat_at = hb_job.last_run
            runtime_fields["last_heartbeat_at"] = last_heartbeat_at

            spend_today_usd = 0.0
            spend_month_usd = 0.0
            if cost_tracker is not None:
                try:
                    spend_today_usd = float(
                        cost_tracker.get_spend(agent=agent_id, period="today").get("total_cost", 0.0)
                    )
                    spend_month_usd = float(
                        cost_tracker.get_spend(agent=agent_id, period="month").get("total_cost", 0.0)
                    )
                except Exception:
                    logger.debug(
                        "cost_tracker.get_spend failed for %s",
                        agent_id, exc_info=True,
                    )
            runtime_fields["spend_today_usd"] = spend_today_usd
            runtime_fields["spend_month_usd"] = spend_month_usd

        # Quarantine fields (Fix 4/5 in seam follow-up). Always present so
        # ``inspect_agents`` and the dashboard can render a stable shape,
        # even when the agent isn't quarantined (operator surface clarity).
        quarantined = False
        quarantine_reason: str | None = None
        consecutive_auth_failures = 0
        if health_monitor is not None:
            h_q = health_monitor.agents.get(agent_id)
            if h_q is not None:
                quarantined = h_q.quarantined
                quarantine_reason = h_q.quarantine_reason
                consecutive_auth_failures = h_q.consecutive_auth_failures

        response = {
            "agent_id": agent_id,
            "role": role,
            "status": status,
            "last_active": last_active,
            "last_task_event_ts": last_task_event_ts,
            "heartbeat_schedule": heartbeat_schedule,
            "subscriptions": subscriptions,
            "watches": watches,
            "recent_writes": recent_writes,
            "capabilities": capabilities,
            "interface": interface,
            # Task 8 structured routing fields.
            "interface_capabilities": interface_capabilities,
            "preferred_inputs": preferred_inputs,
            "expected_outputs": expected_outputs,
            "escalation_to": escalation_to,
            "forbidden": forbidden,
            # Fix 4/5: quarantine state surface for inspect_agents/dashboard.
            "quarantined": quarantined,
            "quarantine_reason": quarantine_reason,
            "consecutive_auth_failures": consecutive_auth_failures,
        }
        response.update(runtime_fields)
        return response

    @app.post("/mesh/agents/{agent_id}/auth-failure")
    async def report_auth_failure(
        agent_id: str, body: dict, request: Request,
    ) -> dict:
        """Agent self-reports a credential failure (Fix 4 in seam follow-up).

        The mesh counts consecutive failures and quarantines the agent
        once ``HealthMonitor.AUTH_FAILURE_THRESHOLD`` is reached. Cleared
        by a successful ``edit_agent(field='model', ...)`` or by the
        auto-expiry sweeper (``OPENLEGION_QUARANTINE_AUTO_CLEAR_SECONDS``).

        Self-report only — agents cannot report failures on each other's
        behalf. Internal callers (``x-mesh-internal``) can report for any
        agent_id (used by future heartbeat probes).
        """
        if not _is_internal_caller(request):
            requesting = _resolve_agent_id(agent_id, request)
            if requesting != agent_id:
                _record_denial(
                    "scope", caller=requesting, target=agent_id,
                    gate="auth_failure:self_report_only",
                )
                raise HTTPException(
                    403, "Agents can only report failures for themselves",
                )
            # Rate-limit only the agent-self-report path. Internal callers
            # (mesh's own ``_record_auth`` recorder threading the proxy
            # boundary) are the load-bearing trigger and must never be
            # throttled — quarantine itself caps damage at threshold=3.
            await _check_rate_limit("auth_failure", agent_id)
        if health_monitor is None:
            return {"recorded": False, "reason": "no health_monitor"}
        if not isinstance(body, dict):
            raise HTTPException(400, "body must be a JSON object")
        provider = str(body.get("provider", "unknown"))[:64]
        model = str(body.get("model", ""))[:120]
        try:
            http_status = int(body.get("http_status", 0))
        except (TypeError, ValueError):
            http_status = 0
        quarantined = health_monitor.record_auth_failure(
            agent_id, provider=provider, model=model, http_status=http_status,
        )
        return {"recorded": True, "quarantined": quarantined}

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

    _RATE_LIMITS["ext_credentials"] = (3000, 60)
    _RATE_LIMITS["ext_status"] = (6000, 60)

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

        Operator-only: pre-PR this endpoint accepted any agent's bearer
        token and leaked fleet-wide health, costs, attention list, and
        per-agent budgets to every authenticated agent. The endpoint
        was always intended as part of the operator heartbeat — gate it
        accordingly.
        """
        _require_operator_or_internal(request)

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

        # -- Per-agent cost (PR-J') --
        # Mirrors the ``agent_metrics`` per-agent cost / ratio pattern but
        # rolled up across the fleet so the operator heartbeat can spot
        # outliers in a single call. Operator excluded — it's a system
        # agent whose spend is bookkeeping, not user work.
        agent_ids_user = [aid for aid in agents if aid != "operator"]
        per_agent_cost_today: dict[str, float] = {}
        per_agent_cost_ratio: dict[str, float | None] = {}
        if cost_tracker:
            for aid in agent_ids_user:
                today_a = cost_tracker.get_spend(aid, "today").get(
                    "total_cost", 0.0,
                )
                # ``yesterday`` is "since yesterday midnight" (includes
                # today). Subtract today to isolate yesterday-only spend.
                since_yest_a = cost_tracker.get_spend(aid, "yesterday").get(
                    "total_cost", 0.0,
                )
                yest_a = max(since_yest_a - today_a, 0.0)
                per_agent_cost_today[aid] = round(today_a, 4)
                # ``None`` when no yesterday baseline; otherwise
                # today/yesterday ratio rounded to 2 decimals. Returning
                # ``None`` (rather than ``0.0``) lets the heartbeat
                # playbook distinguish "agent stopped spending today"
                # (ratio == 0.0) from "no yesterday baseline" (None).
                per_agent_cost_ratio[aid] = (
                    round(today_a / yest_a, 2) if yest_a > 0 else None
                )

        # -- Per-agent task outcome / failure / stale counts (PR-J') --
        # The two count fields supersede the legacy ``failure_rate_by_agent``
        # placeholder for the operator heartbeat: counts on small fleets
        # are stable, rates on small denominators are noise. The
        # placeholder stays (empty dict) so contract consumers don't need
        # an immediate update; treat ``{}`` as "data unavailable" still.
        # PR-U: ``outcome_rejected_24h_count`` filters on
        # ``outcome_set_at`` (when the operator rated the task), not
        # ``completed_at`` (when the agent finished) — review delay no
        # longer drops lagged rejections off the heartbeat.
        outcome_rejected_24h: dict[str, int] = {}
        execution_failures_24h: dict[str, int] = {}
        stale_tasks_24h: dict[str, int] = {}
        chain_breaks_24h: dict[str, int] = {}
        if tasks_store is not None:
            try:
                _day_seconds = 24 * 60 * 60
                outcome_rejected_24h = {
                    aid: count
                    for aid, count in tasks_store.count_outcomes_since(
                        "rejected", since_seconds=_day_seconds,
                    ).items()
                    if aid != "operator"
                }
                execution_failures_24h = {
                    aid: count
                    for aid, count in tasks_store.count_failed_status_since(
                        since_seconds=_day_seconds,
                    ).items()
                    if aid != "operator"
                }
                stale_tasks_24h = {
                    aid: count
                    for aid, count in tasks_store.count_stale_since(
                        threshold_seconds=_day_seconds,
                    ).items()
                    if aid != "operator"
                }
                # Chain-break observability — surfaces ``done`` tasks
                # whose work didn't get handed off (no child row via
                # ``parent_task_id``) and that the operator hasn't
                # actioned yet via rate/rework. Closes the
                # "task_completed_without_handoff signal has no
                # consumer" gap by giving the heartbeat playbook a
                # field to drill into via its existing
                # ``get_system_status`` call.
                chain_breaks_24h = {
                    aid: count
                    for aid, count in tasks_store.chain_breaks_24h(
                        since=time.time() - _day_seconds,
                    ).items()
                    if aid != "operator"
                }
            except Exception as e:
                # Telemetry is best-effort; never block the heartbeat
                # endpoint on a tasks_v2 hiccup.
                logger.debug("system_metrics task aggregates failed: %s", e)

        # Legacy placeholder — kept on the contract so consumers
        # treating ``{}`` as "data unavailable" don't break. The two
        # count dicts above are what the heartbeat playbook keys on now.
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
        max_teams = int(os.environ.get("OPENLEGION_MAX_TEAMS", "0"))

        # Count actual teams
        from src.cli.config import _load_projects
        current_teams = len(_load_projects())

        # -- BYOK visibility (Bug 5) --
        # Operators (and the operator agent's heartbeat playbook) need
        # to know which providers actually have credentials configured
        # so they can pick a model that won't dead-on-arrival at create
        # time. Paired with the up-front validation in
        # ``create_custom_agent`` and ``_create_agent_from_template``.
        available_providers: list[str] = []
        if credential_vault is not None:
            try:
                available_providers = sorted(
                    credential_vault.get_providers_with_credentials(),
                )
            except Exception as e:
                logger.debug("system_metrics available_providers failed: %s", e)

        return {
            "total_agents": total,
            "healthy": healthy,
            "failed": failed,
            "busy": busy,
            "total_cost_today_usd": round(cost_today, 4),
            "cost_vs_yesterday_ratio": cost_ratio,
            # Per-agent cost surface (PR-J'). Empty dicts when no spend
            # has been recorded — that case is meaningfully different
            # from "data unavailable", which would require the cost
            # tracker to be ``None``. ``per_agent_cost_vs_yesterday_ratio``
            # is ``None`` when no yesterday baseline; otherwise
            # today/yesterday rounded to 2 decimals.
            "per_agent_cost_today_usd": per_agent_cost_today,
            "per_agent_cost_vs_yesterday_ratio": per_agent_cost_ratio,
            # Per-agent task outcome / failure / stale counts (PR-J').
            # Empty dicts when tasks_v2 is disabled OR no rows match.
            "outcome_rejected_24h_count": outcome_rejected_24h,
            "execution_failures_24h_count": execution_failures_24h,
            "stale_tasks_24h_count": stale_tasks_24h,
            # Chain-break observability — per-agent count of ``done``
            # tasks with no successor (no child via ``parent_task_id``)
            # and no outcome set, within the trailing 24h window.
            # Paired with the ``task_completed_without_handoff``
            # DashboardEvent emitted by ``Tasks.update_status``.
            # Observability-only, no enforcement.
            "chain_breaks_24h_count": chain_breaks_24h,
            "failure_rate_by_agent": failure_rates,
            "agents_needing_attention": agents_attention,
            "plan_limits": {
                "max_agents": max_agents,
                "current_agents": total,
                # BOTH ``max_teams`` and the legacy ``max_projects`` keys
                # are emitted with the same value for back-compat through
                # PR 3. Same for ``current_teams`` / ``current_projects``.
                "max_teams": max_teams,
                "current_teams": current_teams,
                "max_projects": max_teams,
                "current_projects": current_teams,
            },
            # Task 5: count of warn-mode "would have denied" hits since
            # process start. Operators watch this number drop toward
            # zero before flipping ``OPENLEGION_TEAM_SCOPE_MODE`` to
            # ``enforce``. The flag itself is reported alongside so the
            # operator dashboard can render the right state. BOTH keys
            # emitted for back-compat.
            "scope_warn_total": _scope_warn_count,
            "team_scope_mode": _TEAM_SCOPE_MODE,
            "project_scope_mode": _TEAM_SCOPE_MODE,
            # Phase 3 Slice 1 (PR-O'.1) telemetry: process-lifetime count
            # of cross-team blackboard accesses (caller's team set is
            # disjoint from the existing entry's writer-team set). Pure
            # observability — informs the design doc for PR-O'.2; NO
            # enforcement effect today. Counts kinds separately so the
            # design can branch on read vs write volume. BOTH the new
            # ``blackboard_cross_team_total`` and the legacy
            # ``blackboard_cross_project_total`` keys carry the same
            # counter (kept through PR 3).
            "blackboard_cross_team_total": dict(_blackboard_xteam_count),
            "blackboard_cross_project_total": dict(_blackboard_xteam_count),
            # PR-K' minimal denial observability. 24h window, auto-reset
            # at the day boundary. Operator-readable categories:
            # ``auth`` / ``scope`` / ``role`` / ``permission`` / ``rate``.
            # Categories that have not fired today are absent from the
            # dict — defaultdict semantics; readers should treat missing
            # keys as zero.
            "tool_denials_24h": dict(_denial_counter),
            # Bug 5 — BYOK provider visibility. Sorted list of provider
            # names with credentials currently configured. Empty list
            # means no LLM keys at all (deployment broken).
            "available_providers": available_providers,
        }

    @app.get("/mesh/agents/{agent_id}/metrics")
    async def agent_metrics(agent_id: str, request: Request) -> dict:
        """Per-agent pre-computed metrics for operator heartbeat.

        Returns health, cost, and queue data for a single agent.
        Read-only — no mutations.

        Operator-only (see ``system_metrics`` rationale).
        """
        _require_operator_or_internal(request)
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

    @app.get("/mesh/agents/{agent_id}/stale-tasks")
    async def agent_stale_tasks(
        agent_id: str, request: Request, threshold_hours: int = 24,
    ) -> dict:
        """List up to 5 oldest stale (non-terminal, created >threshold ago) tasks.

        Operator-only. Powers ``inspect_agents(stale_threshold_hours=N)``
        in the operator heartbeat. Returns ``{"agent_id", "threshold_hours",
        "count", "task_ids"}``. When tasks_v2 is disabled or the agent has
        no stale tasks, ``count`` is 0 and ``task_ids`` is empty.
        """
        _require_operator_or_internal(request)
        _validate_agent_id(agent_id)
        if not (1 <= threshold_hours <= 168):
            raise HTTPException(
                400, "threshold_hours must be between 1 and 168 (1 hour to 7 days)",
            )
        threshold_seconds = float(threshold_hours) * 3600.0
        try:
            ids = tasks_store.list_stale_for_assignee(
                agent_id, threshold_seconds=threshold_seconds, limit=5,
            )
        except Exception as e:
            logger.debug("stale-tasks lookup failed for %s: %s", agent_id, e)
            ids = []
        return {
            "agent_id": agent_id,
            "threshold_hours": threshold_hours,
            "count": len(ids),
            "task_ids": ids,
        }

    # === Mesh Team Proxy Endpoints ===
    # These proxy the dashboard's /api/teams/* endpoints through the
    # mesh so operator agents can manage teams using their mesh auth
    # token. The pre-rename ``/mesh/projects/*`` aliases were removed
    # in PR 3 of the project→team rename — see CLAUDE.md Review State.

    @app.get("/mesh/teams")
    async def mesh_list_teams(request: Request, include_archived: bool = False) -> dict:
        """List teams (mesh-authed proxy).

        Excludes archived teams by default — pass ``include_archived=true``
        to include them. Each row carries ``status`` so callers can render
        the archive state when ``include_archived`` is set. The ``name``
        / ``team_name`` keys both ride on each row so callers tracking
        either field keep working.
        """
        caller = _extract_verified_agent_id(request)
        if not _caller_is_operator(caller, request) and not _is_internal_caller(request):
            raise HTTPException(403, "Only the operator can manage teams")
        from src.cli.config import _load_projects
        projects = _load_projects()
        result = []
        for pname, pdata in sorted(projects.items(), key=lambda x: x[1].get("created_at") or ""):
            status = pdata.get("status", "active") or "active"
            if not include_archived and status == "archived":
                continue
            result.append({
                "name": pname,
                "team_name": pname,
                "description": pdata.get("description", ""),
                "members": pdata.get("members", []),
                "created_at": pdata.get("created_at", ""),
                "status": status,
            })
        return {"teams": result}

    @app.post("/mesh/teams")
    async def mesh_create_team(request: Request) -> dict:
        """Create a new team (mesh-authed proxy)."""
        _require_any_auth(request)
        if _resolve_agent_id("", request) != "operator":
            raise HTTPException(403, "Only the operator can manage teams")
        import os as _os

        from src.cli.config import _create_project, _load_config, _load_projects

        _max_teams_env = _os.environ.get("OPENLEGION_MAX_TEAMS")
        if _max_teams_env is not None:
            _max_teams = int(_max_teams_env)
            if _max_teams == 0:
                raise HTTPException(
                    403,
                    "Teams are not available on your plan. Upgrade for team support.",
                )
            current_count = len(_load_projects())
            if current_count >= _max_teams:
                raise HTTPException(
                    403,
                    f"Team limit reached ({_max_teams}). Upgrade your plan for more teams.",
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
        # Real-time cron lifecycle: schedule the daily work-summary
        # fire on team creation so the operator doesn't have to wait
        # for the next mesh restart for the reconcile to pick it up.
        # Best-effort — a missing cron_scheduler or a failure here
        # mustn't fail the team-create response.
        #
        # Reads the persisted metadata back (``_create_project`` wrote
        # the file with default ``settings={}``; the read is for
        # forward-compatibility with a future create endpoint that
        # accepts initial settings, and to keep behavior consistent
        # with the unarchive path which also reads metadata).
        if cron_scheduler is not None:
            try:
                persisted = _load_projects().get(name) or {}
                _custom_schedule = (
                    (persisted.get("settings") or {}).get("summary_schedule")
                )
                cron_scheduler.ensure_summary_job(
                    scope_kind="team", scope_id=name,
                    schedule=_custom_schedule,
                )
            except Exception as e:
                logger.warning(
                    "ensure_summary_job on team create %s failed: %s", name, e,
                )
        _emit_team_event(
            event_bus, "project_created", agent="operator", name=name,
            extra={"description": description, "members": list(members)},
        )
        return {"created": True, "name": name, "team_name": name, "team_id": name, "project_id": name}

    @app.post("/mesh/teams/{team_name}/members")
    async def mesh_add_team_member(team_name: str, request: Request) -> dict:
        """Add an agent to a team (mesh-authed proxy)."""
        _require_any_auth(request)
        if _resolve_agent_id("", request) != "operator":
            raise HTTPException(403, "Only the operator can manage teams")
        from src.cli.config import _add_agent_to_project
        body = await request.json()
        agent = body.get("agent", "").strip()
        if not agent:
            raise HTTPException(400, "agent is required")
        try:
            _add_agent_to_project(team_name, agent)
        except ValueError as e:
            raise HTTPException(400, str(e))
        # Update in-memory project mapping so scoping takes effect immediately
        _agent_projects[agent] = team_name
        return {"added": True, "project": team_name, "team_name": team_name, "agent": agent}

    @app.delete("/mesh/teams/{team_name}/members/{agent}")
    async def mesh_remove_team_member(team_name: str, agent: str, request: Request) -> dict:
        """Remove an agent from a team (mesh-authed proxy)."""
        _require_any_auth(request)
        if _resolve_agent_id("", request) != "operator":
            raise HTTPException(403, "Only the operator can manage teams")
        from src.cli.config import _remove_agent_from_project
        try:
            _remove_agent_from_project(team_name, agent)
        except ValueError as e:
            raise HTTPException(400, str(e))
        _agent_projects.pop(agent, None)
        return {"removed": True, "project": team_name, "team_name": team_name, "agent": agent}

    @app.delete("/mesh/teams/{team_name}")
    async def mesh_delete_team(team_name: str, request: Request) -> dict:
        """Delete a team (mesh-authed proxy)."""
        _require_any_auth(request)
        if _resolve_agent_id("", request) != "operator":
            raise HTTPException(403, "Only the operator can manage teams")
        from src.cli.config import _delete_project
        try:
            _delete_project(team_name)
        except ValueError as e:
            raise HTTPException(404, str(e))
        # Real-time cron lifecycle: drop the daily work-summary cron
        # for the deleted team. This is the mesh DIRECT-delete path
        # (distinct from the propose/confirm flow, which also cleans
        # up). Without this the team is gone but its cron keeps
        # firing daily empty-state summaries until the next mesh
        # restart catches the orphan (codex r3 P2).
        if cron_scheduler is not None:
            try:
                existing = cron_scheduler.find_summary_job("team", team_name)
                if existing is not None:
                    cron_scheduler.remove_job(existing.id)
            except Exception as e:
                logger.warning(
                    "remove summary cron on mesh team-delete %s failed: %s",
                    team_name, e,
                )
        _emit_team_event(event_bus, "project_deleted", agent="operator", name=team_name)
        return {
            "deleted": True, "name": team_name,
            "team_name": team_name, "team_id": team_name, "project_id": team_name,
        }

    @app.put("/mesh/teams/{team_name}/context")
    async def mesh_update_team_context(team_name: str, request: Request) -> dict:
        """Update a team's description/context (mesh-authed proxy)."""
        _require_any_auth(request)
        if _resolve_agent_id("", request) != "operator":
            raise HTTPException(403, "Only the operator can manage teams")
        from src.cli.config import PROJECTS_DIR, _load_projects
        body = await request.json()
        context = sanitize_for_prompt(body.get("context", "")).strip()

        projects = _load_projects()
        if team_name not in projects:
            raise HTTPException(404, f"Team '{team_name}' not found")

        # Update metadata.yaml description in place (never delete the team)
        import yaml
        meta_file = PROJECTS_DIR / team_name / "metadata.yaml"
        if meta_file.exists():
            with open(meta_file) as f:
                meta = yaml.safe_load(f) or {}
            meta["description"] = context
            with open(meta_file, "w") as f:
                yaml.dump(meta, f, default_flow_style=False, sort_keys=False)

        # Update project.md in place
        project_md = PROJECTS_DIR / team_name / "project.md"
        project_md.write_text(f"# {team_name}\n\n{context}\n")

        _emit_team_event(
            event_bus, "project_updated", agent="operator", name=team_name,
            extra={"field": "context"},
        )

        return {
            "updated": True, "project": team_name, "team": team_name,
            "team_name": team_name, "team_id": team_name, "project_id": team_name,
        }

    @app.post("/mesh/teams/{team_name}/goal")
    async def mesh_set_team_goal(team_name: str, request: Request) -> dict:
        """Set a team's north star + success criteria (mesh-authed proxy).

        Operator-only (or internal localhost callers). Validates length
        limits then persists to ``metadata.yaml`` in place. No confirmation
        gate — this is meta-config the user explicitly asked for.
        """
        _require_any_auth(request)
        caller = _extract_verified_agent_id(request)
        if not _caller_is_operator(caller, request) and not _is_internal_caller(request):
            raise HTTPException(403, "Only the operator can manage teams")
        from src.cli.config import PROJECTS_DIR, _load_projects

        body = await request.json()
        north_star_raw = body.get("north_star")
        success_criteria_raw = body.get("success_criteria")

        # Normalize and validate.
        if north_star_raw is None:
            north_star: str | None = None
        else:
            north_star = sanitize_for_prompt(str(north_star_raw)).strip()
            if len(north_star) > 2000:
                raise HTTPException(
                    400, "north_star must be 2000 characters or fewer",
                )
            if not north_star:
                north_star = None

        success_criteria: list[str] | None
        if success_criteria_raw is None:
            success_criteria = None
        else:
            if not isinstance(success_criteria_raw, list):
                raise HTTPException(
                    400, "success_criteria must be a list of strings",
                )
            if len(success_criteria_raw) > 10:
                raise HTTPException(
                    400, "success_criteria may contain at most 10 items",
                )
            cleaned: list[str] = []
            for item in success_criteria_raw:
                sc = sanitize_for_prompt(str(item)).strip()
                if not sc:
                    continue
                if len(sc) > 200:
                    raise HTTPException(
                        400,
                        "each success_criteria entry must be 200 characters or fewer",
                    )
                cleaned.append(sc)
            success_criteria = cleaned or None

        projects = _load_projects()
        if team_name not in projects:
            raise HTTPException(404, f"Team '{team_name}' not found")

        import yaml
        meta_file = PROJECTS_DIR / team_name / "metadata.yaml"
        if not meta_file.exists():
            raise HTTPException(404, f"Team '{team_name}' has no metadata file")
        with open(meta_file) as f:
            meta = yaml.safe_load(f) or {}
        meta["north_star"] = north_star
        meta["success_criteria"] = success_criteria
        with open(meta_file, "w") as f:
            yaml.dump(meta, f, default_flow_style=False, sort_keys=False)

        _emit_team_event(
            event_bus, "project_updated", agent="operator", name=team_name,
            extra={"field": "goal"},
        )

        return {
            "success": True,
            "project_name": team_name,
            "team_name": team_name,
            "team_id": team_name,
            "project_id": team_name,
            "north_star": north_star,
            "success_criteria": success_criteria,
        }

    # === Orchestration Tasks ===

    def _reap_tasks_opportunistically() -> None:
        """Cheap reap on read paths so retention drops don't need a scheduler."""
        try:
            deleted = tasks_store.reap_expired()
            if deleted:
                logger.info("orchestration: reaped %d expired tasks", deleted)
        except Exception as e:
            logger.debug("orchestration reap failed: %s", e)

    def _is_project_member(agent_id: str, project_id: str) -> bool:
        """Membership check for read scoping. Operator + mesh are global."""
        if agent_id in {"operator", "mesh"}:
            return True
        return project_id in _caller_projects(agent_id)

    # ── Back-edge events ─────────────────────────────────────────
    #
    # When a task reaches a terminal status (or a lane-timeout flips it
    # to ``failed``), the originating agent learns via a back-edge entry
    # written to ``inbox/{origin_user}/task_event/{task_id}``. Actionable
    # events also wake the originator so workflow recovery is event-
    # driven instead of heartbeat-paced. Closure-local rate-limit state
    # coalesces bursts (e.g. lane timeout + sweep retry).
    _BACK_EDGE_KIND_FOR_STATUS = {
        "done": "task_completed",
        "failed": "task_failed",
        "blocked": "task_blocked",
        "cancelled": "task_cancelled",
    }
    _BACK_EDGE_WAKE_KINDS = frozenset({"task_failed", "task_blocked"})
    _BACK_EDGE_WAKE_WINDOW_SECONDS = 60.0
    _back_edge_wake_state: dict[str, float] = {}

    def _write_task_event_back_edge(
        task_record: dict,
        *,
        event_kind: str,
        payload_extras: dict | None = None,
    ) -> None:
        """Write a back-edge event for a terminal-status transition.

        ``task_record`` is the post-transition row dict (must include
        ``id``, ``assignee``, ``title``, ``status``, and the nested
        ``origin`` dict). ``event_kind`` is the back-edge kind name
        (``task_completed`` / ``task_failed`` / ``task_blocked`` /
        ``task_cancelled``). ``payload_extras`` carries kind-specific
        fields (``blocker_note``, ``error``, ``summary``) — sentinel
        schema keys can't be shadowed.

        Best-effort: every failure path is logged and swallowed. The
        underlying status transition has already committed and the
        back-edge must never destabilize the lifecycle.

        Wake-on-event: ``task_failed`` and ``task_blocked`` also enqueue
        a followup lane message back to the originator with a 60s rate-
        limit per task. ``task_completed`` / ``task_cancelled`` do NOT
        wake (operator picks them up via heartbeat — no need to interrupt
        successful chains or explicit cancels).
        """
        try:
            origin_dict = task_record.get("origin") or {}
            origin_kind = origin_dict.get("kind") if origin_dict else None
            origin_user = origin_dict.get("user") if origin_dict else None
            assignee = task_record.get("assignee")
            task_id = task_record.get("id")

            # Eligibility — only cross-agent agent/operator handoffs.
            # Self-handoffs (sender == recipient) suppress so an
            # originating agent's check_inbox stays clean.
            if origin_kind not in {"agent", "operator"}:
                return
            if not origin_user or origin_user == assignee:
                return
            if not task_id:
                return

            payload: dict = {
                "kind": event_kind,
                "task_id": task_id,
                "recipient": assignee,
                "title": task_record.get("title"),
                "status": task_record.get("status"),
                "ts": int(time.time()),
            }
            if payload_extras:
                # Sentinel keys above take precedence — extras can't
                # shadow the canonical schema fields.
                for k, v in payload_extras.items():
                    if k not in payload:
                        payload[k] = v
            try:
                blackboard.write(
                    f"inbox/{origin_user}/task_event/{task_id}",
                    payload, written_by="mesh", ttl=604800,  # 7 days
                )
            except Exception as e:
                logger.warning(
                    "Back-edge write failed for task %s: %s", task_id, e,
                )
                return

            # Wake-on-event for actionable kinds with per-task rate limit.
            if event_kind not in _BACK_EDGE_WAKE_KINDS:
                return
            if lane_manager is None or dispatch_loop is None:
                return
            if origin_user not in router.agent_registry:
                return
            now = time.time()
            last = _back_edge_wake_state.get(task_id, 0.0)
            if now - last < _BACK_EDGE_WAKE_WINDOW_SECONDS:
                return
            _back_edge_wake_state[task_id] = now
            try:
                wake_origin = MessageOrigin(
                    kind=origin_kind,
                    channel=str(origin_dict.get("channel") or ""),
                    user=str(origin_user),
                )
                title = task_record.get("title") or "(no title)"
                wake_msg = (
                    f"Task {task_id} ({title}) reached {event_kind}. "
                    "Call check_inbox to see the event payload."
                )
                asyncio.run_coroutine_threadsafe(
                    lane_manager.enqueue(
                        origin_user, wake_msg, mode="followup",
                        origin=wake_origin, auto_notify=False,
                        task_id=task_id,
                    ),
                    dispatch_loop,
                )
            except Exception as e:
                logger.warning(
                    "Back-edge wake for %s on task %s failed: %s",
                    origin_user, task_id, e,
                )
        except Exception as e:  # belt-and-suspenders
            logger.warning(
                "Back-edge handler crashed for task %s: %s",
                task_record.get("id"), e,
            )

    # Expose so the lane watchdog (built in runtime.py before this app)
    # can fire the same back-edge path on lane-timeout failures. Wired
    # in ``runtime.py`` post-app-construction via ``set_back_edge_fn``.
    app._write_task_event_back_edge = _write_task_event_back_edge

    @app.post("/mesh/tasks")
    async def create_task(request: Request) -> dict:
        """Create a durable task record.

        Caller must be authorised to message the assignee
        (``can_message(caller, assignee)``); operator and mesh-internal
        bypass. Body: ``{assignee, title, description?, project?,
        parent_task_id?, priority?, dependencies?}``. Origin is sourced
        from the validated ``X-Origin`` header. The legacy
        ``can_route_tasks`` toggle was retired — task creation is
        structured messaging and shares the ``can_message`` trust
        boundary, which is set to ``["*"]`` by default under collab mode
        so multi-stage workflows work out of the box.
        """
        store = tasks_store
        caller = _extract_verified_agent_id(request)
        # Round-4 forensic logging (Bug 1 still reproduces post-PR#952).
        # INFO-level entry trace so the operator's repro logs reveal
        # whether the request even reached this handler (vs short-
        # circuited upstream by transport/router) and what the request
        # body looked like at the wire. Combine with ``Tasks.create``'s
        # stored-value log to bisect the silent drop in one E2E run.
        logger.info(
            "/mesh/tasks POST entry caller=%s headers=%s",
            caller,
            {
                k: v for k, v in request.headers.items()
                if k.lower() in (
                    "x-trace-id", "x-task-id", "x-origin", "x-agent-id",
                )
            },
        )
        body = await request.json()
        # ``.strip()`` on assignee defends against a trailing newline or
        # leading space sneaking through a hand-written prompt. SQLite
        # ``=`` is byte-exact so a single whitespace divergence between
        # what the LLM emitted and what the recipient compares against
        # would silently break ``list_task_inbox`` matching — the exact
        # symptom Bug 1's repros chased.
        assignee_raw = body.get("assignee", "")
        assignee = assignee_raw.strip() if isinstance(assignee_raw, str) else ""
        if assignee != assignee_raw:
            logger.warning(
                "tasks.create normalized assignee %r → %r (whitespace "
                "stripped) for caller=%s", assignee_raw, assignee, caller,
            )
        title = sanitize_for_prompt(body.get("title", "")).strip()
        description = sanitize_for_prompt(body.get("description") or "")
        project_id = body.get("project") or body.get("project_id") or None
        parent_task_id = body.get("parent_task_id") or None
        priority = int(body.get("priority", 0) or 0)
        dependencies = body.get("dependencies") or None
        artifact_refs = body.get("artifact_refs") or None
        if not title:
            raise HTTPException(400, "title is required")
        if not assignee or not _AGENT_ID_RE.match(assignee):
            raise HTTPException(400, f"Invalid assignee: {assignee!r}")
        # Task creation is structured messaging — same trust boundary as
        # ``can_message(caller, assignee)``. Using one gate (instead of
        # the legacy double-gate with the now-defunct ``can_route_tasks``
        # toggle) means every fleet template's collab-mode default
        # (``can_message=["*"]``) automatically enables worker→worker
        # handoffs out of the box. Operator + mesh-internal bypass.
        if not _caller_is_operator(caller, request) and not _is_internal_caller(request):
            if not permissions.can_message(caller, assignee):
                _record_denial(
                    "permission", caller=caller, target=assignee,
                    gate="tasks.create:can_message",
                )
                raise HTTPException(
                    403,
                    f"Agent {caller} cannot create task for {assignee!r} "
                    "(can_message not granted)",
                )
        # Cross-project scope: callers can only create tasks in projects
        # they belong to (operators / mesh are global). Standalone is
        # permitted (project_id=None).
        if project_id and not _is_project_member(caller, project_id):
            raise HTTPException(
                403,
                f"Caller {caller} is not a member of project {project_id!r}",
            )
        # Body trace — assignee/project/parent already validated above.
        # Round-4 forensic visibility: pairs with the entry log at the
        # top so a missing log line here points to validation refusal
        # while a present log here proves the request reached the
        # store.create call site.
        logger.info(
            "/mesh/tasks POST body caller=%s assignee=%s project_id=%s "
            "parent_task_id=%s title=%r",
            caller, assignee, project_id, parent_task_id,
            (title or "")[:80],
        )

        origin = _validated_origin(request, caller)
        origin_dict = origin.model_dump() if origin is not None else None

        try:
            record = store.create(
                creator=caller,
                assignee=assignee,
                title=title,
                description=description or None,
                project_id=project_id,
                parent_task_id=parent_task_id,
                priority=priority,
                dependencies=dependencies if isinstance(dependencies, list) else None,
                artifact_refs=artifact_refs if isinstance(artifact_refs, list) else None,
                origin=origin_dict,
            )
        except RuntimeError as e:
            # ``Tasks.create``'s centralised post-write verify raises
            # RuntimeError on integrity failure. Convert to a structured
            # 500 so the agent's ``hand_off`` ``create_failed`` envelope
            # surfaces the actual reason instead of FastAPI's generic
            # "Internal Server Error" placeholder.
            logger.error("tasks.create raised RuntimeError: %s", e)
            raise HTTPException(500, str(e))
        # Belt-and-suspenders verify (Bug 1 post-mortem): ``Tasks.create``
        # already asserts the row exists via its own post-write SELECT and
        # returns that fresh record. Use the returned record (no second
        # SELECT round-trip — Tasks.create has already done it) and assert
        # the canonical fields agree with what the caller requested. A
        # corrupt row with e.g. wrong-typed ``assignee`` silently breaks
        # ``list_task_inbox`` for the intended recipient (the row exists
        # but the SELECT ``WHERE assignee = ?`` doesn't match). If anything
        # differs, 500 with a structured detail so the caller's
        # ``hand_off`` ``create_failed`` envelope fires and we get
        # diagnostic evidence on the next repro.
        if record is None:  # Tasks.create's contract forbids this, but
            # keep a defensive 500 so a future regression in the store
            # surface can't return null JSON to the agent.
            raise HTTPException(
                500,
                "Task creation returned no record (store contract violation)",
            )
        mismatches: list[str] = []
        if record.get("assignee") != assignee:
            mismatches.append(
                f"assignee: stored={record.get('assignee')!r} "
                f"requested={assignee!r}"
            )
        if record.get("creator") != caller:
            mismatches.append(
                f"creator: stored={record.get('creator')!r} "
                f"requested={caller!r}"
            )
        if record.get("project_id") != project_id:
            mismatches.append(
                f"project_id: stored={record.get('project_id')!r} "
                f"requested={project_id!r}"
            )
        if record.get("parent_task_id") != parent_task_id:
            mismatches.append(
                f"parent_task_id: stored={record.get('parent_task_id')!r} "
                f"requested={parent_task_id!r}"
            )
        if record.get("status") != "pending":
            mismatches.append(
                f"status: stored={record.get('status')!r} expected='pending'"
            )
        if mismatches:
            logger.error(
                "tasks.create post-write verify: task %s mismatch — %s",
                record["id"], "; ".join(mismatches),
            )
            raise HTTPException(
                500,
                f"Task {record['id']!r} post-write verify failed: "
                + "; ".join(mismatches),
            )
        return record

    @app.get("/mesh/tasks/inbox/{assignee}")
    async def list_inbox(assignee: str, request: Request) -> dict:
        """List ``assignee``'s inbox.

        The assignee themself, the operator, and loopback-internal
        callers are permitted. Other callers receive 403.
        """
        store = tasks_store
        caller = _extract_verified_agent_id(request)
        if (
            caller != assignee
            and not _caller_is_operator(caller, request)
            and not _is_internal_caller(request)
        ):
            raise HTTPException(
                403,
                "Only the assignee, operator, or internal callers can read this inbox",
            )
        _reap_tasks_opportunistically()
        rows = store.list_inbox(assignee)
        return {"tasks": rows, "count": len(rows)}

    @app.get("/mesh/tasks/team/{team_id}")
    async def list_team_tasks(team_id: str, request: Request) -> dict:
        """List tasks scoped to a team.

        Caller must be a member of the team (or operator / internal).
        """
        store = tasks_store
        caller = _extract_verified_agent_id(request)
        if not _is_project_member(caller, team_id):
            raise HTTPException(
                403,
                f"Caller {caller} is not a member of team {team_id!r}",
            )
        _reap_tasks_opportunistically()
        rows = store.list_project(team_id)
        return {"tasks": rows, "count": len(rows)}

    @app.get("/mesh/tasks/workflow/{root_task_id}")
    async def get_workflow_snapshot(
        root_task_id: str, request: Request,
    ) -> dict:
        """Return a workflow chain snapshot rooted at ``root_task_id``.

        Walks ``parent_task_id`` descendants from the root and reports
        every stage's status + age. Operator-only by design — workflow
        orchestration awareness is operator-tier, and individual workers
        have no business inspecting a chain they don't own.

        404 when the root does not exist (lets the operator distinguish
        a typo from an empty chain).
        """
        caller = _extract_verified_agent_id(request)
        if not (
            _caller_is_operator(caller, request)
            or _is_internal_caller(request)
        ):
            raise HTTPException(
                403,
                "workflow_snapshot is operator-only",
            )
        snapshot = tasks_store.workflow_snapshot(root_task_id)
        if snapshot is None:
            raise HTTPException(
                404, f"Root task '{root_task_id}' not found",
            )
        return snapshot

    @app.get("/mesh/tasks/{task_id}")
    async def get_task(task_id: str, request: Request) -> dict:
        """Read a task by id.

        Visible to creator, assignee, project members, operator, and
        internal callers. Other callers receive 403.
        """
        store = tasks_store
        caller = _extract_verified_agent_id(request)
        record = store.get(task_id)
        if record is None:
            raise HTTPException(404, f"Task '{task_id}' not found")
        if (
            caller in (record["creator"], record["assignee"])
            or _caller_is_operator(caller, request)
            or _is_internal_caller(request)
            or (record["project_id"] and _is_project_member(caller, record["project_id"]))
        ):
            return record
        raise HTTPException(403, "Not authorized to read this task")

    @app.get("/mesh/tasks/{task_id}/events")
    async def list_task_events(task_id: str, request: Request) -> dict:
        """Audit history for a task. Same visibility rules as get_task."""
        store = tasks_store
        caller = _extract_verified_agent_id(request)
        record = store.get(task_id)
        if record is None:
            raise HTTPException(404, f"Task '{task_id}' not found")
        if not (
            caller in (record["creator"], record["assignee"])
            or _caller_is_operator(caller, request)
            or _is_internal_caller(request)
            or (record["project_id"] and _is_project_member(caller, record["project_id"]))
        ):
            raise HTTPException(403, "Not authorized to read this task")
        events = store.list_events(task_id)
        return {"events": events, "count": len(events)}

    @app.post("/mesh/tasks/{task_id}/status")
    async def update_task_status(task_id: str, request: Request) -> dict:
        """Update a task's status.

        Caller must be the assignee, the creator, or the operator/internal.
        Status transitions are validated by the storage layer; an invalid
        transition becomes HTTP 400.

        Bug 3 fix: on terminal transitions (``done`` / ``failed`` /
        ``cancelled`` / ``blocked``) where the originating ``origin_kind``
        is ``agent`` or ``operator``, write a back-edge to the
        originator's blackboard at ``inbox/{origin_user}/task_event/{id}``
        with a 7-day TTL. Humans are excluded (they get auto-notified via
        the lane worker forward path) and self-handoffs are dropped to
        keep an originator's inbox clean. Failures are logged, never
        raised — the status update itself stays authoritative.
        """
        store = tasks_store
        caller = _extract_verified_agent_id(request)
        body = await request.json()
        status = body.get("status", "")
        blocker_note = body.get("blocker_note")
        # Bug 3 (silent model rejection): when a failed transition arrives
        # with no ``blocker_note`` but carries an ``error`` string (the
        # canonical shape ``mesh_client.set_task_status`` sends for
        # auto-close paths), promote ``error`` to ``blocker_note`` so the
        # store persists the reason. Truncate to 500 chars to match the
        # column's expected size and avoid runaway bloat from huge LLM
        # tracebacks. Existing explicit ``blocker_note`` callers win.
        if status == "failed" and not blocker_note:
            _err = body.get("error")
            if isinstance(_err, str) and _err.strip():
                blocker_note = _err.strip()[:500]
        if status not in VALID_STATUSES:
            raise HTTPException(
                400,
                f"Invalid status: {status!r}. Must be one of {sorted(VALID_STATUSES)}",
            )
        record = store.get(task_id)
        if record is None:
            raise HTTPException(404, f"Task '{task_id}' not found")
        if not (
            caller in (record["creator"], record["assignee"])
            or _caller_is_operator(caller, request)
            or _is_internal_caller(request)
        ):
            raise HTTPException(
                403,
                "Only the creator, assignee, operator, or internal can update status",
            )
        try:
            updated = store.update_status(
                task_id, status, actor=caller, blocker_note=blocker_note,
            )
        except InvalidStatusTransition as e:
            raise HTTPException(400, str(e))
        except TaskNotFound:
            raise HTTPException(404, f"Task '{task_id}' not found")

        # Back-edge to originating agent on terminal transitions. The
        # helper handles eligibility, payload shaping, and the wake-on-
        # event chain for actionable kinds.
        event_kind = _BACK_EDGE_KIND_FOR_STATUS.get(status)
        if event_kind is not None:
            fresh = store.get(task_id) or record
            raw_result = body.get("result")
            result_dict = raw_result if isinstance(raw_result, dict) else {}
            payload_extras: dict = {}
            if status == "blocked":
                payload_extras["blocker_note"] = blocker_note or ""
            elif status == "failed":
                payload_extras["error"] = (
                    body.get("error") or result_dict.get("error", "") or ""
                )
            elif status == "done":
                payload_extras["summary"] = result_dict.get("summary", "")
            _write_task_event_back_edge(
                fresh, event_kind=event_kind, payload_extras=payload_extras,
            )
        return updated

    def _check_can_schedule(target_agent: str) -> tuple[bool, dict | None]:
        """Cost-aware preflight: True when the target agent has budget headroom.

        Returns ``(allowed, info_or_none)``. ``info`` carries the
        offending agent + daily/monthly used vs limit so the operator
        product tools can surface a structured error to the user.
        Operator (and the missing-tracker case) always pass — there is
        nothing to enforce.
        """
        if cost_tracker is None or target_agent == "operator":
            return True, None
        try:
            check = cost_tracker.check_budget(target_agent)
        except Exception as e:
            logger.debug("check_budget(%s) failed: %s — allowing", target_agent, e)
            return True, None
        if check.get("allowed", True):
            return True, None
        return False, {
            "agent": target_agent,
            "daily_used": check.get("daily_used"),
            "daily_limit": check.get("daily_limit"),
            "monthly_used": check.get("monthly_used"),
            "monthly_limit": check.get("monthly_limit"),
        }

    def _try_wake_agent(
        target: str, message: str, origin: "MessageOrigin | None",
    ) -> bool:
        """Best-effort lane enqueue so an operator state change is acted on now.

        Used by reroute / retry / cancel: the task store is already
        updated when this fires, so any failure here is logged but
        non-fatal — the worker will pick the change up on its next
        heartbeat. Fire-and-forget against ``dispatch_loop`` (same
        pattern as ``/mesh/wake``) so the HTTP response doesn't block
        on the agent finishing the work. ``auto_notify`` is only set
        when a real origin was provided, so completion of the woken
        work flows back to the originating human channel.
        """
        from src.shared.types import MessageOrigin

        if lane_manager is None or dispatch_loop is None:
            return False
        if not target or target not in router.agent_registry:
            return False
        had_origin = origin is not None
        eff_origin = origin if origin is not None else MessageOrigin(
            kind="agent", channel="", user="",
        )
        # Build the coroutine first so we can close it explicitly if the
        # dispatch loop is unhealthy — otherwise an orphaned coroutine
        # would emit a "coroutine was never awaited" warning.
        coro = lane_manager.enqueue(
            target, sanitize_for_prompt(message), mode="followup",
            origin=eff_origin, auto_notify=had_origin,
        )
        try:
            asyncio.run_coroutine_threadsafe(coro, dispatch_loop)
            return True
        except Exception as e:
            coro.close()
            logger.warning("Operator wake enqueue for %s failed: %s", target, e)
            return False

    @app.post("/mesh/tasks/{task_id}/reroute")
    async def reroute_task(task_id: str, request: Request) -> dict:
        """Reassign a task. Operator-only (administrative recovery action).

        Cost-aware: refuses to reroute onto an agent that is already
        over its daily or monthly budget (HTTP 400, structured error).
        """
        store = tasks_store
        caller = _extract_verified_agent_id(request)
        if not (
            _caller_is_operator(caller, request)
            or _is_internal_caller(request)
        ):
            raise HTTPException(
                403,
                "Reroute is operator-only (administrative recovery action)",
            )
        body = await request.json()
        new_assignee = body.get("new_assignee", "")
        reason = sanitize_for_prompt(body.get("reason") or "")
        if not new_assignee or not _AGENT_ID_RE.match(new_assignee):
            raise HTTPException(400, f"Invalid new_assignee: {new_assignee!r}")
        # Budget preflight — before mutating storage.
        allowed, info = _check_can_schedule(new_assignee)
        if not allowed:
            raise HTTPException(
                400,
                json.dumps({
                    "error": "over_budget",
                    "detail": (
                        f"Agent {new_assignee!r} is over budget; refusing to "
                        "reroute task to a target that cannot run."
                    ),
                    "budget": info,
                }),
            )
        try:
            updated = store.reroute(
                task_id, new_assignee, actor=caller, reason=reason,
            )
        except TaskNotFound:
            raise HTTPException(404, f"Task '{task_id}' not found")
        except InvalidStatusTransition as e:
            raise HTTPException(400, str(e))
        # Wake the new assignee so they pick the task up immediately
        # instead of waiting for their next heartbeat. State change has
        # already succeeded; wake is best-effort.
        origin = _validated_origin(request, caller)
        title = updated.get("title") or task_id
        reason_suffix = f" ({reason})" if reason else ""
        _try_wake_agent(
            new_assignee,
            f"Operator rerouted task to you: {title!r}{reason_suffix}. "
            "Call check_inbox() to pick it up.",
            origin,
        )
        return updated

    @app.post("/mesh/tasks/{task_id}/retry")
    async def retry_failed_task(task_id: str, request: Request) -> dict:
        """Retry a failed task by cloning it as a new ``pending`` task.

        Operator product tool surface: clones the existing task into a
        new id (``task_<hex>``), optionally overriding title / description /
        assignee from the request body. Original task is left in its
        terminal state (``failed``) so the audit trail is preserved.
        Cost-aware: refuses if the (possibly overridden) target assignee
        is over budget.
        """
        store = tasks_store
        caller = _extract_verified_agent_id(request)
        if not (
            _caller_is_operator(caller, request)
            or _is_internal_caller(request)
        ):
            raise HTTPException(
                403,
                "Retry is operator-only (administrative recovery action)",
            )
        original = store.get(task_id)
        if original is None:
            raise HTTPException(404, f"Task '{task_id}' not found")
        if original["status"] != "failed":
            raise HTTPException(
                400,
                f"Only failed tasks can be retried (status={original['status']!r})",
            )
        body = await request.json() if (await request.body()) else {}
        if not isinstance(body, dict):
            body = {}
        # Optional patch: title / description / assignee
        title = sanitize_for_prompt(body.get("title") or original["title"]).strip()
        description_override = body.get("description")
        if description_override is not None:
            description = sanitize_for_prompt(description_override) or None
        else:
            description = original.get("description") or None
        new_assignee = body.get("assignee") or original["assignee"]
        if not new_assignee or not _AGENT_ID_RE.match(new_assignee):
            raise HTTPException(400, f"Invalid assignee: {new_assignee!r}")
        if not title:
            raise HTTPException(400, "title is required (and cannot be blank)")
        # Budget preflight before mutating storage.
        allowed, info = _check_can_schedule(new_assignee)
        if not allowed:
            raise HTTPException(
                400,
                json.dumps({
                    "error": "over_budget",
                    "detail": (
                        f"Agent {new_assignee!r} is over budget; refusing to "
                        "retry the task onto a target that cannot run."
                    ),
                    "budget": info,
                }),
            )
        origin = _validated_origin(request, caller)
        origin_dict = origin.model_dump() if origin is not None else None
        try:
            clone = store.create(
                creator=caller,
                assignee=new_assignee,
                title=title,
                description=description,
                project_id=original.get("project_id"),
                parent_task_id=original["id"],
                priority=original.get("priority", 0) or 0,
                origin=origin_dict,
            )
        except RuntimeError as e:
            # Retry shares Tasks.create's centralised post-write verify
            # (Bug 1 R2 closed the bypass gap) — same RuntimeError-to-500
            # conversion as the /mesh/tasks POST path above.
            logger.error("tasks.retry store.create RuntimeError: %s", e)
            raise HTTPException(500, str(e))
        # Wake the (possibly new) assignee on the clone so the retry
        # starts immediately rather than waiting for a heartbeat.
        _try_wake_agent(
            new_assignee,
            f"Operator retried failed task as {clone['id']!r}: {title!r}. "
            "Call check_inbox() to pick it up.",
            origin,
        )
        return {"clone": clone, "original_id": original["id"]}

    # === Work Summaries ===
    #
    # Operator-generated team / solo-agent summaries. Replace the
    # per-task Work-tab firehose at scale: at 30+ agents the operator's
    # mental unit shifts from individual tasks to team-level health,
    # and the user rates one summary per team per period instead of
    # rating every delivery. Operator generates via the
    # ``compose_work_summary`` skill (or cron); user (operator persona
    # via dashboard) rates via ``POST /mesh/work-summaries/{id}/rating``.
    #
    # Visibility: operator + loopback-internal see all summaries.
    # Workers see only summaries for teams they belong to (mirrors
    # ``_is_project_member`` semantics). Solo summaries are scoped to
    # the agent id — only the agent itself + operator can read.
    #
    # Permission: create + rating are operator-or-internal. Read is
    # any-auth with scope filter.

    def _can_read_summary(caller: str, request: Request, row: dict) -> bool:
        if _caller_is_operator(caller, request) or _is_internal_caller(request):
            return True
        if row["scope_kind"] == "solo":
            return caller == row["scope_id"]
        if row["scope_kind"] == "team":
            return _is_project_member(caller, row["scope_id"])
        return False

    @app.post("/mesh/work-summaries")
    async def create_work_summary(data: dict, request: Request) -> dict:
        """Operator (or internal) creates a new work summary."""
        caller = _extract_verified_agent_id(request)
        if not (_caller_is_operator(caller, request) or _is_internal_caller(request)):
            _record_denial(
                "role", caller=caller,
                gate="work_summaries.create:operator_or_internal",
            )
            raise HTTPException(
                403, "Only the operator can create work summaries",
            )
        scope_kind = (data.get("scope_kind") or "").strip()
        scope_id = (data.get("scope_id") or "").strip()
        period_start = data.get("period_start")
        period_end = data.get("period_end")
        narrative_md = (data.get("narrative_md") or "").strip()
        metrics = data.get("metrics") or {}
        recommendations = data.get("recommendations") or []
        if not (
            scope_kind and scope_id and narrative_md
            and isinstance(period_start, (int, float))
            and isinstance(period_end, (int, float))
        ):
            raise HTTPException(
                400,
                "Required fields: scope_kind, scope_id, period_start "
                "(number), period_end (number), narrative_md",
            )
        if not isinstance(metrics, dict):
            raise HTTPException(400, "metrics must be a JSON object")
        if not isinstance(recommendations, list):
            raise HTTPException(400, "recommendations must be a JSON array")
        from src.host.summaries import MAX_NARRATIVE_CHARS
        if len(narrative_md) > MAX_NARRATIVE_CHARS:
            raise HTTPException(
                413,
                f"narrative_md exceeds {MAX_NARRATIVE_CHARS} chars "
                f"(got {len(narrative_md)})",
            )
        try:
            from src.host.summaries import InvalidScope
            row = summaries_store.create(
                scope_kind=scope_kind,
                scope_id=scope_id,
                period_start=float(period_start),
                period_end=float(period_end),
                narrative_md=sanitize_for_prompt(narrative_md),
                metrics=metrics,
                recommendations=[
                    sanitize_for_prompt(str(r))[:500] for r in recommendations
                ],
                generated_by=caller,
            )
        except InvalidScope as e:
            raise HTTPException(400, str(e))
        except ValueError as e:
            # UNIQUE collision or validation failure — already-exists is
            # the common case under retries / concurrent crons.
            raise HTTPException(409, str(e))
        return row

    @app.get("/mesh/work-summaries")
    async def list_work_summaries(
        request: Request,
        scope_kind: str | None = None,
        scope_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict:
        """List recent summaries. Scope-filtered for workers."""
        caller = _extract_verified_agent_id(request)
        summaries_store._safe_reap()
        try:
            from src.host.summaries import InvalidScope
            rows = summaries_store.list_recent(
                scope_kind=scope_kind, scope_id=scope_id,
                limit=limit, offset=offset,
            )
        except InvalidScope as e:
            raise HTTPException(400, str(e))
        # Apply per-row visibility filter for non-operator callers.
        visible = [r for r in rows if _can_read_summary(caller, request, r)]
        return {"summaries": visible, "count": len(visible)}

    @app.get("/mesh/work-summaries/{summary_id}")
    async def get_work_summary(summary_id: str, request: Request) -> dict:
        """Fetch a single summary by id. Scope-checked."""
        caller = _extract_verified_agent_id(request)
        row = summaries_store.get(summary_id)
        if row is None:
            raise HTTPException(404, f"Summary {summary_id!r} not found")
        if not _can_read_summary(caller, request, row):
            _record_denial(
                "scope", caller=caller, target=summary_id,
                gate="work_summaries.get:scope",
            )
            raise HTTPException(
                403, f"Caller {caller} cannot read summary {summary_id!r}",
            )
        return row

    @app.post("/mesh/work-summaries/{summary_id}/rating")
    async def rate_work_summary(
        summary_id: str, data: dict, request: Request,
    ) -> dict:
        """Operator (acting for the user) rates a summary.

        Editable for 24h after first rating, then locked. The first
        ``rated_at`` is preserved across edits so the UI shows the
        original-rating timestamp, not the latest revision.
        """
        caller = _extract_verified_agent_id(request)
        if not (_caller_is_operator(caller, request) or _is_internal_caller(request)):
            _record_denial(
                "role", caller=caller, target=summary_id,
                gate="work_summaries.rate:operator_or_internal",
            )
            raise HTTPException(
                403, "Only the operator can rate work summaries",
            )
        rating = (data.get("rating") or "").strip()
        feedback = data.get("feedback")
        if feedback is not None and not isinstance(feedback, str):
            raise HTTPException(400, "feedback must be a string or null")
        try:
            from src.host.summaries import (
                RatingLocked,
                SummaryNotFound,
            )
            return summaries_store.set_rating(
                summary_id, rating,
                feedback=sanitize_for_prompt(feedback) if feedback else None,
                actor=caller,
            )
        except SummaryNotFound:
            raise HTTPException(404, f"Summary {summary_id!r} not found")
        except RatingLocked as e:
            raise HTTPException(409, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))

    # === Operator product surface (Task 7) — read tools ===
    #
    # Read endpoints surfaced as operator skills. They aggregate over
    # the tasks store + project metadata and respect the same scoping
    # rules as the task-store endpoints (operator + internal can see all,
    # other callers can see only their own projects).

    def _summarize_tasks(rows: list[dict]) -> dict:
        """Reduce a list of task rows to status counts + recent slices.

        Buckets: ``active`` (pending/accepted/working), ``blocked``,
        ``done``, ``failed``, ``cancelled``. ``recent_done`` carries the
        last 5 completed tasks ordered newest-first.
        """
        counts = {"active": 0, "blocked": 0, "done": 0, "failed": 0, "cancelled": 0}
        blocked_rows: list[dict] = []
        done_rows: list[dict] = []
        for r in rows:
            s = r.get("status", "")
            if s in ("pending", "accepted", "working"):
                counts["active"] += 1
            elif s == "blocked":
                counts["blocked"] += 1
                blocked_rows.append(r)
            elif s == "done":
                counts["done"] += 1
                done_rows.append(r)
            elif s == "failed":
                counts["failed"] += 1
            elif s == "cancelled":
                counts["cancelled"] += 1
        done_rows.sort(key=lambda x: x.get("completed_at") or 0, reverse=True)
        return {
            "counts": counts,
            "blockers": [
                {
                    "id": r["id"], "assignee": r["assignee"],
                    "title": r["title"], "blocker_note": r.get("blocker_note"),
                }
                for r in blocked_rows[:5]
            ],
            "recent_done": [
                {
                    "id": r["id"], "assignee": r["assignee"], "title": r["title"],
                    "completed_at": r.get("completed_at"),
                }
                for r in done_rows[:5]
            ],
        }

    @app.get("/mesh/teams/{team_id}/status")
    async def team_status(team_id: str, request: Request) -> dict:
        """Per-team status counts + recent blockers/completions.

        Caller must be a team member (or operator/internal). Returns
        the same structure as ``_summarize_tasks`` plus a ``project``
        / ``team`` field carrying name and archive status.
        """
        store = tasks_store
        caller = _extract_verified_agent_id(request)
        if not _is_project_member(caller, team_id):
            raise HTTPException(
                403,
                f"Caller {caller} is not a member of team {team_id!r}",
            )
        from src.cli.config import _load_projects
        projects = _load_projects()
        if team_id not in projects:
            raise HTTPException(404, f"Team '{team_id}' not found")
        meta = projects[team_id]
        _reap_tasks_opportunistically()
        rows = store.list_project(team_id)
        result = _summarize_tasks(rows)
        meta_block = {
            "name": team_id,
            "status": meta.get("status", "active") or "active",
            "members": meta.get("members", []),
            "description": meta.get("description", ""),
        }
        result["project"] = meta_block
        result["team"] = meta_block
        return result

    @app.get("/mesh/teams/status")
    async def all_teams_status(request: Request) -> dict:
        """List status rollups for every team the caller can see.

        Operator / internal callers see all teams (including archived);
        other callers see only their own. Each row carries the same
        ``counts``/``blockers``/``recent_done`` shape as the per-team
        endpoint, plus the team name and status.
        """
        store = tasks_store
        caller = _extract_verified_agent_id(request)
        from src.cli.config import _load_projects
        projects = _load_projects()
        visible: list[str]
        if _caller_is_operator(caller, request) or _is_internal_caller(request):
            visible = list(projects.keys())
        else:
            visible = sorted(_caller_projects(caller))
        _reap_tasks_opportunistically()
        rows: list[dict] = []
        for pid in sorted(visible):
            meta = projects.get(pid, {})
            project_rows = store.list_project(pid)
            summary = _summarize_tasks(project_rows)
            meta_block = {
                "name": pid,
                "status": meta.get("status", "active") or "active",
                "members": meta.get("members", []),
                "description": meta.get("description", ""),
            }
            summary["team"] = meta_block
            rows.append(summary)
        return {"teams": rows}

    @app.get("/mesh/agents/{agent_id}/queue")
    async def agent_queue(
        agent_id: str, request: Request, limit: int = 10,
    ) -> dict:
        """Recent tasks for an agent grouped by status.

        Returns up to ``limit`` rows per status (active / blocked / done /
        failed / cancelled). Visible to the agent itself, the operator,
        loopback-internal callers, and project members of the agent's
        project.
        """
        store = tasks_store
        caller = _extract_verified_agent_id(request)
        if not (
            caller == agent_id
            or _caller_is_operator(caller, request)
            or _is_internal_caller(request)
        ):
            agent_proj = _agent_projects.get(agent_id)
            if agent_proj is None or not _is_project_member(caller, agent_proj):
                raise HTTPException(
                    403,
                    f"Caller {caller} cannot view queue for {agent_id!r}",
                )
        try:
            limit = max(1, min(int(limit), 100))
        except (TypeError, ValueError):
            limit = 10
        _reap_tasks_opportunistically()
        rows = store.list_inbox(agent_id, include_terminal=True)
        buckets: dict[str, list[dict]] = {
            "active": [], "blocked": [], "done": [], "failed": [], "cancelled": [],
        }
        # Sort newest-first so the "last N" slice is informative.
        rows.sort(key=lambda r: r.get("updated_at") or 0, reverse=True)
        for r in rows:
            s = r.get("status", "")
            if s in ("pending", "accepted", "working"):
                key = "active"
            elif s in buckets:
                key = s
            else:
                continue
            if len(buckets[key]) >= limit:
                continue
            buckets[key].append({
                "id": r["id"], "title": r["title"],
                "status": r["status"],
                "project_id": r.get("project_id"),
                "blocker_note": r.get("blocker_note"),
                "updated_at": r.get("updated_at"),
                "completed_at": r.get("completed_at"),
            })
        return {"agent_id": agent_id, "limit": limit, "queue": buckets}

    def _parse_since(since: str | None) -> float:
        """Parse a since= filter — accepts ISO timestamp, ``"24h"``/``"7d"``, or ``""``."""
        import time as _time
        if not since:
            return _time.time() - (7 * 24 * 60 * 60)
        s = since.strip().lower()
        # Duration form: ``Nh`` / ``Nd`` / ``Nm``
        if s and s[-1] in {"s", "m", "h", "d"} and s[:-1].isdigit():
            n = int(s[:-1])
            unit = s[-1]
            mult = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
            return _time.time() - (n * mult)
        # ISO timestamp
        try:
            from datetime import datetime as _dt
            dt = _dt.fromisoformat(s.replace("z", "+00:00"))
            return dt.timestamp()
        except (ValueError, TypeError):
            return _time.time() - (7 * 24 * 60 * 60)

    @app.get("/mesh/teams/{team_id}/outputs")
    async def team_outputs(
        team_id: str, request: Request, since: str = "",
    ) -> dict:
        """Completed task artifacts for a team in a time window.

        ``since`` accepts an ISO timestamp or duration string (``"24h"``,
        ``"7d"``); default is the last 7 days. Returns one entry per
        completed task with its title, assignee, completion time, and
        artifact refs.
        """
        store = tasks_store
        caller = _extract_verified_agent_id(request)
        if not _is_project_member(caller, team_id):
            raise HTTPException(
                403,
                f"Caller {caller} is not a member of team {team_id!r}",
            )
        floor = _parse_since(since)
        _reap_tasks_opportunistically()
        rows = store.list_project(team_id, statuses=["done"])
        outputs = []
        for r in rows:
            completed_at = r.get("completed_at") or 0
            if completed_at < floor:
                continue
            outputs.append({
                "id": r["id"], "title": r["title"], "assignee": r["assignee"],
                "completed_at": completed_at,
                "artifact_refs": r.get("artifact_refs", []) or [],
            })
        outputs.sort(key=lambda x: x["completed_at"], reverse=True)
        return {
            "project_id": team_id,
            "team_id": team_id,
            "since": since,
            "outputs": outputs,
        }

    @app.get("/mesh/teams/{team_id}/summary")
    async def team_summary(team_id: str, request: Request) -> dict:
        """Synthesized status text + structured fields for a team.

        Combines status counts, blocker list, recent completions, and a
        simple ``ask_for_user`` list (currently mirrors ``blocked`` —
        operators can later swap in a richer policy here without changing
        the on-the-wire shape). The narrative ``status_text`` is plain
        prose so the operator's prompt machinery doesn't have to format
        it again.
        """
        store = tasks_store
        caller = _extract_verified_agent_id(request)
        if not _is_project_member(caller, team_id):
            raise HTTPException(
                403,
                f"Caller {caller} is not a member of team {team_id!r}",
            )
        from src.cli.config import _load_projects
        projects = _load_projects()
        if team_id not in projects:
            raise HTTPException(404, f"Team '{team_id}' not found")
        meta = projects[team_id]
        _reap_tasks_opportunistically()
        rows = store.list_project(team_id)
        s = _summarize_tasks(rows)
        active = s["counts"]["active"]
        blocked = s["counts"]["blocked"]
        done = s["counts"]["done"]
        failed = s["counts"]["failed"]
        if active == 0 and blocked == 0 and done == 0 and failed == 0:
            status_text = f"No tasks recorded for {team_id!r} yet."
        else:
            parts: list[str] = []
            if active:
                parts.append(f"{active} active")
            if blocked:
                parts.append(f"{blocked} blocked")
            if done:
                parts.append(f"{done} done")
            if failed:
                parts.append(f"{failed} failed")
            status_text = ", ".join(parts) + f" in team {team_id!r}."
        meta_block = {
            "name": team_id,
            "status": meta.get("status", "active") or "active",
            "description": meta.get("description", ""),
            "members": meta.get("members", []),
        }
        return {
            "project": meta_block,
            "team": meta_block,
            "status_text": status_text,
            "counts": s["counts"],
            "top_blockers": s["blockers"],
            "recent_completions": s["recent_done"],
            "ask_for_user": s["blockers"],
        }

    # === Operator product surface (Task 7) — archive / delete ===
    #
    # Archive endpoints flip a status flag and stop scheduling. Delete
    # endpoints proxy through the existing ``PendingActions`` store
    # (Task 2d) — a propose-then-confirm flow keyed by
    # ``target_kind="project"`` / ``"agent"`` and ``action_kind="delete"``,
    # confirmed via ``/mesh/config/confirm``. Archive must precede delete;
    # the gate is enforced server-side at propose time.

    @app.post("/mesh/teams/{team_name}/archive")
    async def archive_team_endpoint(team_name: str, request: Request) -> dict:
        """Archive a team (operator-only). Idempotent."""
        caller = _extract_verified_agent_id(request)
        if not _caller_is_operator(caller, request) and not _is_internal_caller(request):
            raise HTTPException(403, "Only the operator can archive teams")
        from src.cli.config import _archive_project, _load_projects
        if team_name not in _load_projects():
            raise HTTPException(404, f"Team '{team_name}' not found")
        try:
            _archive_project(team_name)
        except ValueError as e:
            raise HTTPException(404, str(e))
        # Real-time cron lifecycle (codex r1 P2): remove the daily
        # work-summary cron so an archived team doesn't keep firing
        # empty-state summaries until the next mesh restart. Operator
        # explicitly archived — they don't want activity here.
        if cron_scheduler is not None:
            try:
                existing = cron_scheduler.find_summary_job("team", team_name)
                if existing is not None:
                    cron_scheduler.remove_job(existing.id)
            except Exception as e:
                logger.warning(
                    "remove summary cron on archive %s failed: %s",
                    team_name, e,
                )
        _emit_team_event(event_bus, "project_archived", agent="operator", name=team_name)
        return {
            "archived": True, "project": team_name, "team": team_name,
            "team_name": team_name, "team_id": team_name, "project_id": team_name,
        }

    @app.post("/mesh/teams/{team_name}/unarchive")
    async def unarchive_team_endpoint(team_name: str, request: Request) -> dict:
        """Unarchive a team (operator-only). Idempotent."""
        caller = _extract_verified_agent_id(request)
        if not _caller_is_operator(caller, request) and not _is_internal_caller(request):
            raise HTTPException(403, "Only the operator can unarchive teams")
        from src.cli.config import _load_projects, _unarchive_project
        if team_name not in _load_projects():
            raise HTTPException(404, f"Team '{team_name}' not found")
        try:
            _unarchive_project(team_name)
        except ValueError as e:
            raise HTTPException(404, str(e))
        # Re-attach the daily work-summary cron when a team is
        # unarchived. Symmetric to the archive path above. Read the
        # team's persisted ``settings.summary_schedule`` so a custom
        # cadence configured before archive is preserved on unarchive
        # — without this lookup, archive → unarchive silently reset
        # the schedule to default (codex r2 P2).
        if cron_scheduler is not None:
            try:
                team_meta = _load_projects().get(team_name) or {}
                _custom_schedule = (
                    (team_meta.get("settings") or {}).get("summary_schedule")
                )
                cron_scheduler.ensure_summary_job(
                    scope_kind="team", scope_id=team_name,
                    schedule=_custom_schedule,
                )
            except Exception as e:
                logger.warning(
                    "ensure_summary_job on unarchive %s failed: %s",
                    team_name, e,
                )
        _emit_team_event(event_bus, "project_unarchived", agent="operator", name=team_name)
        return {
            "archived": False, "project": team_name, "team": team_name,
            "team_name": team_name, "team_id": team_name, "project_id": team_name,
        }

    @app.post("/mesh/agents/{agent_id}/archive")
    async def archive_agent_endpoint(agent_id: str, request: Request) -> dict:
        """Archive an agent (operator-only).

        Stops cron / heartbeat and removes the agent from the live
        registry. Workspace + history are retained; the agent can be
        unarchived later. Container is best-effort stopped.
        """
        caller = _extract_verified_agent_id(request)
        if not _caller_is_operator(caller, request) and not _is_internal_caller(request):
            raise HTTPException(403, "Only the operator can archive agents")
        if agent_id == "operator":
            raise HTTPException(400, "The operator agent cannot be archived")
        from src.cli.config import _archive_agent, _load_config
        cfg = _load_config()
        if agent_id not in cfg.get("agents", {}):
            raise HTTPException(404, f"Agent '{agent_id}' not found")
        try:
            _archive_agent(agent_id)
        except ValueError as e:
            raise HTTPException(404, str(e))
        # Stop scheduling: drop heartbeat and any cron jobs the agent owns.
        if cron_scheduler is not None:
            try:
                cron_scheduler.remove_agent_jobs(agent_id)
            except Exception as e:
                logger.warning("archive_agent: cron cleanup for %s failed: %s", agent_id, e)
        # Best-effort container stop. Failures here don't break archive.
        if container_manager is not None:
            try:
                container_manager.stop_agent(agent_id)
            except Exception as e:
                logger.warning("archive_agent: container stop for %s failed: %s", agent_id, e)
        if event_bus is not None:
            try:
                event_bus.emit(
                    "agent_archived", agent=agent_id,
                    data={"agent_id": agent_id},
                )
            except Exception as e:
                logger.debug("agent_archived emit failed: %s", e)
        return {"archived": True, "agent_id": agent_id}

    @app.post("/mesh/agents/{agent_id}/unarchive")
    async def unarchive_agent_endpoint(agent_id: str, request: Request) -> dict:
        """Unarchive an agent (operator-only). Restart left to operator."""
        caller = _extract_verified_agent_id(request)
        if not _caller_is_operator(caller, request) and not _is_internal_caller(request):
            raise HTTPException(403, "Only the operator can unarchive agents")
        from src.cli.config import _load_config, _unarchive_agent
        cfg = _load_config()
        if agent_id not in cfg.get("agents", {}):
            raise HTTPException(404, f"Agent '{agent_id}' not found")
        try:
            _unarchive_agent(agent_id)
        except ValueError as e:
            raise HTTPException(404, str(e))
        if event_bus is not None:
            try:
                event_bus.emit(
                    "agent_unarchived", agent=agent_id,
                    data={"agent_id": agent_id},
                )
            except Exception as e:
                logger.debug("agent_unarchived emit failed: %s", e)
        return {"archived": False, "agent_id": agent_id}

    @app.post("/mesh/teams/{team_name}/propose-delete")
    async def propose_delete_team(team_name: str, request: Request) -> dict:
        """Propose deletion of an archived team. Returns nonce for human confirm.

        Pre-conditions:
          * team exists
          * team is archived (delete on a live team rejected)

        Stores a pending action with ``target_kind="project"``,
        ``action_kind="delete"``, ``origin_kind`` from the validated
        ``X-Origin``. Confirmation goes through the existing
        ``/mesh/config/confirm`` endpoint, which now dispatches on
        ``target_kind``. ``target_kind`` stays ``"project"`` because
        the pending_actions schema predates the rename — it's a
        backend value, not a domain term.
        """
        caller = _extract_verified_agent_id(request)
        if not _caller_is_operator(caller, request) and not _is_internal_caller(request):
            raise HTTPException(403, "Only the operator can delete teams")
        from src.cli.config import _load_projects, _project_status
        projects = _load_projects()
        if team_name not in projects:
            raise HTTPException(404, f"Team '{team_name}' not found")
        status = _project_status(team_name)
        if status != "archived":
            raise HTTPException(
                400,
                "Team must be archived before delete. "
                "Call /mesh/teams/{team_name}/archive first.",
            )
        origin = _validated_origin(request, caller)
        origin_kind = origin.kind if origin is not None else None
        # Cap pending rows to bound storage growth.
        pending_actions.reap_expired()
        existing = pending_actions.list_pending()
        if len(existing) >= _MAX_PENDING:
            oldest = min(existing, key=lambda r: r["expires_at"])
            with pending_actions._conn() as _conn:
                _conn.execute(
                    "DELETE FROM pending_actions WHERE nonce=?",
                    (oldest["nonce"],),
                )
        nonce = str(_uuid.uuid4())
        members = projects[team_name].get("members", []) or []
        # Short headline shown in the inline chat card (max ~80 chars
        # so it doesn't wrap awkwardly). The longer policy explanation
        # is kept in the payload for the legacy CLI surface.
        summary = (
            f"Delete team {team_name!r} and unlink {len(members)} agent(s)"
        )
        payload = {
            "name": team_name,
            "summary": summary,
            "members": members,
        }
        record = pending_actions.store(
            nonce=nonce,
            actor="operator",
            target_kind="project",
            target_id=team_name,
            action_kind="delete",
            payload=payload,
            origin_kind=origin_kind,
            ttl=_CHANGE_TTL_SECONDS,
            summary=summary,
            preview_diff=None,
        )
        return {
            "change_id": nonce,
            "summary": summary,
            "expires_at": datetime.fromtimestamp(
                record["expires_at"], tz=timezone.utc,
            ).isoformat(),
            "payload_digest": record["payload_digest"],
            "requires_confirmation": True,
        }

    @app.post("/mesh/agents/{agent_id}/propose-delete")
    async def propose_delete_agent(agent_id: str, request: Request) -> dict:
        """Propose deletion of an archived agent. Returns nonce for human confirm.

        Pre-conditions:
          * agent exists
          * agent is archived
          * agent is not ``operator``
        """
        caller = _extract_verified_agent_id(request)
        if not _caller_is_operator(caller, request) and not _is_internal_caller(request):
            raise HTTPException(403, "Only the operator can delete agents")
        if agent_id == "operator":
            raise HTTPException(400, "The operator agent cannot be deleted")
        from src.cli.config import _agent_status, _load_config
        cfg = _load_config()
        if agent_id not in cfg.get("agents", {}):
            raise HTTPException(404, f"Agent '{agent_id}' not found")
        status = _agent_status(agent_id)
        if status != "archived":
            raise HTTPException(
                400,
                "Agent must be archived before delete. "
                "Call /mesh/agents/{agent_id}/archive first.",
            )
        origin = _validated_origin(request, caller)
        origin_kind = origin.kind if origin is not None else None
        pending_actions.reap_expired()
        existing = pending_actions.list_pending()
        if len(existing) >= _MAX_PENDING:
            oldest = min(existing, key=lambda r: r["expires_at"])
            with pending_actions._conn() as _conn:
                _conn.execute(
                    "DELETE FROM pending_actions WHERE nonce=?",
                    (oldest["nonce"],),
                )
        nonce = str(_uuid.uuid4())
        summary = f"Delete agent {agent_id!r} permanently"
        payload = {
            "agent_id": agent_id,
            "summary": summary,
        }
        record = pending_actions.store(
            nonce=nonce,
            actor="operator",
            target_kind="agent",
            target_id=agent_id,
            action_kind="delete",
            payload=payload,
            origin_kind=origin_kind,
            ttl=_CHANGE_TTL_SECONDS,
            summary=summary,
            preview_diff=None,
        )
        return {
            "change_id": nonce,
            "summary": summary,
            "expires_at": datetime.fromtimestamp(
                record["expires_at"], tz=timezone.utc,
            ).isoformat(),
            "payload_digest": record["payload_digest"],
            "requires_confirmation": True,
        }

    async def _apply_pending_delete(record: dict) -> dict:
        """Apply a consumed delete pending-action.

        Dispatches on ``target_kind``. Project deletes call
        ``_delete_project``; agent deletes stop the container, remove
        the agent from config + permissions, and tear down per-agent
        runtime state via ``app.cleanup_agent``.
        """
        kind = record["target_kind"]
        target_id = record["target_id"]
        if kind == "project":
            from src.cli.config import _delete_project, _load_projects
            if target_id not in _load_projects():
                raise HTTPException(404, f"Project '{target_id}' no longer exists")
            try:
                _delete_project(target_id)
            except ValueError as e:
                raise HTTPException(404, str(e))
            # Real-time cron lifecycle (codex r1 P2): delete the
            # daily work-summary cron when the team itself is
            # deleted. Symmetric to the archive path; archive already
            # removed it but a delete-without-archive would leak the
            # cron otherwise.
            if cron_scheduler is not None:
                try:
                    existing = cron_scheduler.find_summary_job("team", target_id)
                    if existing is not None:
                        cron_scheduler.remove_job(existing.id)
                except Exception as e:
                    logger.warning(
                        "remove summary cron on delete %s failed: %s",
                        target_id, e,
                    )
            # New audit rows use ``delete_team``; historical
            # ``delete_project`` rows in archived audit data are
            # untouched and still queryable (they retain the legacy
            # action verb).
            blackboard.log_audit(
                action="delete_team", target=target_id,
                change_id=record["nonce"],
            )
            # ``deleted`` field name retained as ``"project"`` for
            # back-compat with SDK callers / dashboards that switch on
            # it; the canonical kind is also surfaced under
            # ``deleted_kind`` so consumers can opt into the new name.
            return {
                "success": True, "deleted": "project", "deleted_kind": "team",
                "name": target_id, "team_id": target_id, "project_id": target_id,
            }
        if kind == "agent":
            from src.cli.config import _load_config, _remove_agent
            if target_id not in _load_config().get("agents", {}):
                raise HTTPException(404, f"Agent '{target_id}' no longer exists")
            try:
                _remove_agent(target_id, stop_container=container_manager is not None)
            except Exception as e:
                raise HTTPException(500, f"Failed to delete agent: {e}")
            # Drop runtime state via cleanup_agent (rate buckets, vault,
            # blackboard, pubsub, lanes, cron, costs, traces, wallets).
            try:
                app.cleanup_agent(target_id)
            except Exception as e:
                logger.warning("cleanup_agent(%s) failed during delete: %s", target_id, e)
            try:
                router.unregister_agent(target_id)
            except Exception:
                pass
            if health_monitor is not None:
                try:
                    health_monitor.unregister(target_id)
                except Exception:
                    pass
            blackboard.log_audit(
                action="delete_agent", target=target_id,
                change_id=record["nonce"],
            )
            return {"success": True, "deleted": "agent", "agent_id": target_id}
        raise HTTPException(
            400, f"Unsupported pending-delete target_kind: {kind!r}",
        )

    @app.post("/mesh/tasks/{task_id}/cancel")
    async def cancel_task(task_id: str, request: Request) -> dict:
        """Cancel a task. Creator, assignee, or operator/internal."""
        store = tasks_store
        caller = _extract_verified_agent_id(request)
        record = store.get(task_id)
        if record is None:
            raise HTTPException(404, f"Task '{task_id}' not found")
        if not (
            caller in (record["creator"], record["assignee"])
            or _caller_is_operator(caller, request)
            or _is_internal_caller(request)
        ):
            raise HTTPException(
                403,
                "Only the creator, assignee, operator, or internal can cancel",
            )
        body = await request.json() if (await request.body()) else {}
        reason = sanitize_for_prompt(body.get("reason") or "") if isinstance(body, dict) else ""
        prior_assignee = record.get("assignee") or ""
        prior_status = record.get("status") or ""
        try:
            updated = store.cancel(task_id, actor=caller, reason=reason)
        except InvalidStatusTransition as e:
            raise HTTPException(400, str(e))
        except TaskNotFound:
            raise HTTPException(404, f"Task '{task_id}' not found")
        # Tell the previous assignee to drop the work — but only if
        # they were actually in a runnable state and aren't the caller
        # (no point waking yourself to tell yourself to stop). ``blocked``
        # is included on purpose: a blocked assignee is typically waiting
        # for the operator to weigh in, and cancellation IS the answer.
        # Without this, a worker keeps churning (or keeps waiting) on a
        # task the operator already took back until the next heartbeat.
        if (
            prior_assignee
            and prior_assignee != caller
            and prior_status in ("pending", "accepted", "working", "blocked")
        ):
            origin = _validated_origin(request, caller)
            title = updated.get("title") or task_id
            reason_suffix = f" ({reason})" if reason else ""
            _try_wake_agent(
                prior_assignee,
                f"Operator cancelled task {task_id!r}: {title!r}"
                f"{reason_suffix}. Stop work on it and call check_inbox() "
                "for next steps.",
                origin,
            )
        return updated

    # === Operator Config Endpoints ===

    _CONFIG_FIELD_MAP = {
        "instructions": "initial_instructions", "soul": "initial_soul",
        "heartbeat": "initial_heartbeat",
        "interface": "initial_interface",
        "model": "model", "role": "role",
        "thinking": "thinking", "budget": "budget",
    }

    @app.get("/mesh/agents/{agent_id}/config")
    async def get_agent_config(
        agent_id: str,
        request: Request,
        fields: str = "",
    ) -> dict:
        """Read agent config in canonical edit_agent shape (operator-only).

        Returns ``{agent_id, config: {...}}`` where ``config`` mirrors the
        field surface of ``edit_agent``: ``model``, ``instructions``, ``soul``,
        ``heartbeat``, ``heartbeat_schedule``, ``interface``, ``role``,
        ``permissions``, ``budget``, ``thinking``. The value sources are
        normalized — ``initial_*`` yaml internals translated, permissions
        loaded from PermissionMatrix, heartbeat_schedule pulled from the
        cron scheduler.

        Optional ``?fields=instructions,soul`` filters the response to a
        subset (case-sensitive, comma-separated). Unknown field names are
        silently dropped (the operator tool validates upfront).
        """
        _require_any_auth(request)
        caller = _resolve_agent_id("", request)
        if not (_caller_is_operator(caller, request) or _is_internal_caller(request)):
            raise HTTPException(403, "Only the operator can read agent configs")
        await _check_rate_limit("agent_profile", "operator")

        from src.cli.config import _load_config
        agent_cfg_root = _load_config()
        agents = agent_cfg_root.get("agents", {})
        if agent_id not in agents:
            raise HTTPException(404, f"Agent '{agent_id}' not found")
        raw = agents[agent_id]

        # Translate yaml internals -> canonical edit_agent field names.
        full: dict = {
            "model": raw.get("model", ""),
            "role": raw.get("role", ""),
            "thinking": raw.get("thinking", "off") or "off",
            "budget": raw.get("budget", {}),
            "instructions": raw.get("initial_instructions", ""),
            "soul": raw.get("initial_soul", ""),
            "heartbeat": raw.get("initial_heartbeat", ""),
            "interface": raw.get("initial_interface", ""),
        }

        # Permissions live in PERMISSIONS_FILE, not agents.yaml. Load via
        # PermissionMatrix and serialize the AgentPermissions Pydantic model
        # so the operator gets the same shape edit_agent accepts.
        #
        # Strip ``agent_id`` from the dump: the field exists on the Pydantic
        # model but PermissionMatrix.reload() reconstructs it via
        # ``AgentPermissions(agent_id=agent_id, **perms)`` — leaving it in
        # would cause a ``multiple values for keyword argument 'agent_id'``
        # crash on the next reload after a round-trip read→edit, poisoning
        # ``config/permissions.json``.
        if permissions is not None:
            try:
                perms = permissions.get_permissions(agent_id)
                dumped = perms.model_dump() if hasattr(perms, "model_dump") else perms.dict()
                dumped.pop("agent_id", None)
                full["permissions"] = dumped
            except Exception:
                full["permissions"] = {}
        else:
            full["permissions"] = {}

        # heartbeat_schedule lives in the cron scheduler.
        full["heartbeat_schedule"] = ""
        if cron_scheduler is not None:
            job = cron_scheduler.find_heartbeat_job(agent_id)
            if job is not None:
                full["heartbeat_schedule"] = job.schedule or ""

        # Optional ?fields= subset filter.
        if fields:
            wanted = {f.strip() for f in fields.split(",") if f.strip()}
            full = {k: v for k, v in full.items() if k in wanted}

        return {"agent_id": agent_id, "config": full}

    # Closes operator bug 6: peer artifacts are stored on each agent's
    # private /data volume, so save_artifact mirrors metadata to the
    # blackboard but never the content. The dashboard already exposes a
    # peer-read path via transport.request; these two endpoints make the
    # same affordance available to the operator agent (and only the
    # operator) so operator can review what teammates produced.
    _PEER_ARTIFACT_MAX_BYTES = 5 * 1024 * 1024  # 5 MB cap on mesh-layer response

    _ARTIFACT_NAME_PATTERN = re.compile(r"^[\w][\w.\-/ ]{0,198}[\w.]$")

    def _validate_peer_artifact_name(name: str) -> None:
        """Reject path traversal / absolute / control / metacharacter names.

        Mirrors the agent-side ``_ARTIFACT_NAME_RE`` shape but enforced
        here so the mesh never forwards a malicious name to the agent's
        artifact-resolution code. Two-stage: explicit ``..`` rejection
        before any character-class check (defence in depth against
        regex bypass via unusual unicode forms).
        """
        if not name:
            raise HTTPException(400, "Artifact name is required")
        # Stage 1: structural rejections that don't depend on the regex.
        if name.startswith("/") or name.startswith("\\"):
            raise HTTPException(400, "Absolute paths not allowed")
        if ".." in name.split("/") or ".." in name.split("\\"):
            raise HTTPException(400, "Path traversal not allowed")
        # Stage 2: character-class allowlist (word chars, dot, hyphen,
        # slash, space — same as the agent's artifact name regex).
        if not _ARTIFACT_NAME_PATTERN.match(name):
            raise HTTPException(400, f"Invalid artifact name: {name}")

    @app.get("/mesh/agents/{agent_id}/artifacts")
    async def list_peer_artifacts(agent_id: str, request: Request) -> dict:
        """List a peer agent's artifact files. Operator-or-internal only.

        Closes operator bug 6 (read side of save_artifact). Returns
        ``{agent_id, artifacts: [{name, size, modified}, ...]}``.
        """
        _require_any_auth(request)
        caller = _resolve_agent_id("", request)
        if not (_caller_is_operator(caller, request) or _is_internal_caller(request)):
            raise HTTPException(403, "Only the operator can read peer artifacts")
        if agent_id not in router.agent_registry:
            raise HTTPException(404, f"Agent '{agent_id}' not found")
        if transport is None:
            raise HTTPException(503, "Transport not available")
        result = await transport.request(agent_id, "GET", "/artifacts", timeout=10)
        if isinstance(result, dict) and "error" in result and "artifacts" not in result:
            status = result.get("status_code", 502)
            raise HTTPException(status, result["error"])
        artifacts = result.get("artifacts", []) if isinstance(result, dict) else []
        return {"agent_id": agent_id, "artifacts": artifacts}

    @app.get("/mesh/agents/{agent_id}/artifacts/{name:path}")
    async def read_peer_artifact(
        agent_id: str, name: str, request: Request,
    ) -> dict:
        """Read a single peer artifact's content. Operator-or-internal only.

        Returns ``{agent_id, name, content, size, encoding}``. Text is
        decoded as UTF-8 (with ``errors='replace'`` for malformed bytes);
        binary content falls back to base64 with ``encoding='base64'``.
        Capped at 5 MB; oversize artifacts get HTTP 413.
        """
        _require_any_auth(request)
        caller = _resolve_agent_id("", request)
        if not (_caller_is_operator(caller, request) or _is_internal_caller(request)):
            raise HTTPException(403, "Only the operator can read peer artifacts")
        _validate_peer_artifact_name(name)
        if agent_id not in router.agent_registry:
            raise HTTPException(404, f"Agent '{agent_id}' not found")
        if transport is None:
            raise HTTPException(503, "Transport not available")

        result = await transport.request(
            agent_id, "GET", f"/artifacts/{name}", timeout=30,
        )
        if isinstance(result, dict) and "error" in result and "content" not in result:
            status = result.get("status_code", 502)
            # Map the agent's 413 back as 413 (transport returns generic
            # "HTTP 413" string); 404 maps to 404 with a clean message.
            if status == 404:
                raise HTTPException(
                    404, f"Artifact '{name}' not found on agent '{agent_id}'",
                )
            if status == 413:
                raise HTTPException(
                    413,
                    f"Artifact '{name}' exceeds the per-agent size cap",
                )
            raise HTTPException(status, result["error"])

        size = int(result.get("size", 0)) if isinstance(result, dict) else 0
        if size > _PEER_ARTIFACT_MAX_BYTES:
            raise HTTPException(
                413,
                f"Artifact too large ({size} bytes, max {_PEER_ARTIFACT_MAX_BYTES})",
            )
        content = result.get("content", "") if isinstance(result, dict) else ""
        encoding = result.get("encoding", "utf-8") if isinstance(result, dict) else "utf-8"
        return {
            "agent_id": agent_id,
            "name": name,
            "content": content,
            "size": size,
            "encoding": encoding,
        }

    async def _apply_pending_change(
        change_id: str, change: dict,
        *, undoable: bool = False, is_undo: bool = False,
    ) -> dict:
        """Apply a consumed pending change — shared by soft, confirm, and undo paths.

        ``undoable`` flags the audit row as Revert-eligible. Only the
        soft-edit caller passes True (the change_id is the
        ``change_history`` undo_token there). Hard-edit confirms and the
        undo apply itself pass False — their change_id is a pending-action
        nonce or already-consumed undo token, so the dashboard's Revert
        button must not show on those rows.
        """
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
            # Apply semantics differ between forward and undo paths:
            #   * Forward apply (caller passed partial dict): MERGE the
            #     keys into the agent's existing perms so other granted
            #     bits aren't clobbered. `setdefault` covers the rare
            #     case of a never-backfilled agent.
            #   * Undo (``is_undo=True``): REPLACE the agent's perms
            #     dict with the stored old full state. A merge would
            #     leave the granted keys in place — the original edit
            #     only wrote keys that changed, so a merge of the old
            #     full dict doesn't unset what was added. REPLACE
            #     restores correctly. The dashboard already warns when
            #     undoing an older receipt would discard newer edits.
            if isinstance(new_value, dict):
                if is_undo:
                    perms.setdefault("permissions", {})[agent_id] = new_value
                else:
                    perms.setdefault("permissions", {}).setdefault(
                        agent_id, {},
                    ).update(new_value)
                _save_permissions(perms)
            # Hot-reload the in-memory permission matrix
            if permissions is not None:
                permissions.reload()
        elif field == "heartbeat_schedule":
            # Source of truth is the cron table — not agents.yaml. The
            # cron update happens in the heartbeat-schedule sync block
            # below (shared with the legacy ``heartbeat`` field path so
            # both write paths converge on the same scheduler call).
            pass
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

        # Hot-reload: push to running agent's workspace.
        #
        # Receipt-ordering note: callers (soft-edit / confirm-edit /
        # undo) emit the operator_action_receipt event AFTER this
        # function returns. Hot-reload failures here are intentionally
        # swallowed because the YAML write is the source of truth —
        # the running agent will pick up the change on its next
        # restart even if the live PUT fails. So a receipt is emitted
        # for any change whose YAML write succeeded, regardless of
        # whether the running agent's in-memory state caught up.
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
        #
        # ``hot_reload_ok`` defaults to True for fields that don't
        # hot-reload (config-write only) so the emit doesn't lie about
        # in-process state for those. For fields that DO hot-reload, it
        # tracks the agent-side ack — surfaced as ``live: bool`` on the
        # ``agent_config_updated`` event so the SPA can render a
        # "saved — restart to apply" hint when the running agent still
        # has the old config (Docker hang, agent crash, transport timeout).
        hot_reload_ok = True
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
                    hot_reload_ok = False
                    logger.warning(
                        "Hot-reload %s for '%s' returned error: %s",
                        field, agent_id, result["error"],
                    )
                elif isinstance(result, dict) and result.get("status") not in (None, "ok"):
                    # Non-"ok" status from the agent counts as a soft failure.
                    hot_reload_ok = False
                    logger.warning(
                        "Hot-reload %s for '%s' returned non-ok status: %s",
                        field, agent_id, result.get("status"),
                    )
            except Exception as e:
                hot_reload_ok = False
                logger.warning(
                    "Failed to hot-reload %s for '%s': %s",
                    field, agent_id, e,
                )

        # Heartbeat schedule sync. Two trigger paths converge here:
        #
        #   * ``field == "heartbeat"`` — legacy: when the operator edits
        #     the heartbeat-rules markdown and the new content happens to
        #     parse as a schedule, update the cron job too. Best-effort
        #     pattern match.
        #
        #   * ``field == "heartbeat_schedule"`` — PR-L': dedicated soft
        #     field for cadence only. Always retargets the cron job;
        #     value is pre-validated by the operator-tool layer.
        if cron_scheduler is not None and isinstance(new_value, str) and field in ("heartbeat", "heartbeat_schedule"):
            sched = new_value.strip()
            _is_schedule = (
                bool(re.fullmatch(r"every\s+\d+[smhd]", sched, re.IGNORECASE))
                or (len(sched.split()) == 5 and all(
                    re.match(r"^[\d,\-\*/]+$", p) for p in sched.split()
                ))
            )
            if _is_schedule:
                hb_job = cron_scheduler.find_heartbeat_job(agent_id)
                if hb_job is None and field == "heartbeat_schedule":
                    # ``heartbeat_schedule`` always implies a heartbeat
                    # job exists or should be created — agents start
                    # with one via ``ensure_heartbeat`` at registration.
                    # Auto-create here too so the soft-edit doesn't
                    # silently no-op against an agent that never
                    # registered (test fixtures, partial bring-up).
                    try:
                        hb_job = cron_scheduler.ensure_heartbeat(agent_id, schedule=sched)
                        logger.info(
                            "Created heartbeat job for '%s': %s",
                            agent_id, sched,
                        )
                    except Exception as e:
                        logger.warning(
                            "Failed to create heartbeat job for '%s': %s",
                            agent_id, e,
                        )
                        hb_job = None
                if hb_job is not None and hb_job.schedule != sched:
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
            undoable=undoable,
        )

        # Emit a "config updated" event so the dashboard's agent config
        # card flips to the new value live without a full reload. Two
        # rules guard this emit:
        #
        #   * Gated to hard fields. Soft fields are already covered by
        #     ``operator_action_receipt`` (which carries the value
        #     diff). Firing both for a soft edit is redundant noise the
        #     SPA would have to dedupe.
        #
        #   * No ``new_value`` / ``old_value`` on the wire. The SPA
        #     handler refetches the agent detail anyway (it never reads
        #     the diff out of this event), and ``permissions`` ACL
        #     diffs are structurally sensitive — keeping them out of WS
        #     frames avoids leaking on a misconfigured listener.
        #
        # ``live`` reports whether the agent-side hot-reload (model /
        # thinking) actually took. ``True`` for fields that don't
        # hot-reload. ``False`` means the YAML / config-store write
        # landed but the running container still has the old config —
        # the SPA can show "saved — restart to apply".
        if field in HARD_EDIT_FIELDS and event_bus is not None:
            try:
                event_bus.emit(
                    "agent_config_updated", agent=agent_id,
                    data={
                        "agent_id": agent_id,
                        "field": field,
                        "live": hot_reload_ok,
                    },
                )
            except Exception as e:
                logger.debug("agent_config_updated emit failed: %s", e)

        # Fix 4 (seam follow-up): a successful ``model`` change is the
        # operator's "fix the credential" signal — clear any standing
        # quarantine implicitly so the lane resumes dispatching. No
        # separate ``clear_quarantine`` operator tool needed.
        if field == "model" and health_monitor is not None:
            try:
                health_monitor.clear_quarantine(
                    agent_id, reason="model changed via edit_agent",
                )
            except Exception as e:
                logger.debug("clear_quarantine on edit failed: %s", e)

        return {"success": True, "agent_id": agent_id, "field": field}

    # === PR 1 — soft-edit / undo flow ===
    #
    # Soft fields apply directly with no propose+confirm round-trip.
    # Hard fields keep the existing propose/confirm path. The split
    # is a UX pattern — both still pass through the agent's
    # provenance gate at the operator-tool layer. Canonical
    # definitions live in :mod:`src.shared.types`; aliased here for
    # readability inside this app factory.
    _SOFT_EDIT_FIELDS = SOFT_EDIT_FIELDS
    _HARD_EDIT_FIELDS = HARD_EDIT_FIELDS

    def _humanize_field(field: str) -> str:
        """Display name for a field — used in receipt summaries."""
        return {
            "instructions": "instructions",
            "soul": "personality",
            "heartbeat": "heartbeat",
            "heartbeat_schedule": "heartbeat schedule",
            "interface": "interface contract",
            "role": "role",
            "model": "model",
            "permissions": "permissions",
            "budget": "budget",
            "thinking": "thinking config",
        }.get(field, field)

    _EDITABLE_FIELDS = _SOFT_EDIT_FIELDS | _HARD_EDIT_FIELDS

    @app.post("/mesh/agents/{agent_id}/edit-soft")
    async def edit_agent_soft(agent_id: str, request: Request) -> dict:
        """Apply an agent-config edit immediately and emit a revertible receipt.

        Path constraints:
          * Caller must be ``operator`` (or an internal caller for tests).
          * ``X-Origin`` must validate (``_validated_origin``) — the
            operator-tool layer already did its own provenance check, so
            here we just require *some* validatable origin and store the
            kind on the audit trail. All edits intentionally do NOT
            require ``kind="human"`` because the receipt+undo card is
            the safety net (the user can always revert).
          * ``field`` must be in :data:`_EDITABLE_FIELDS` (the union of
            soft and hard fields). Both classes apply immediately; the
            only difference is the undo-receipt TTL — see
            ``_ttl_for_field`` (5 min for personality/instructions/role
            cluster, 30 min for model/permissions/budget/thinking).

        Returns ``{success, undo_token, expires_at, summary, ttl_seconds}``.
        The ``operator_action_receipt`` event is emitted on the bus so the
        dashboard can render the inline receipt card immediately. For
        hard fields, ``agent_config_updated`` ALSO fires (handled by
        ``_apply_pending_change``).

        Path name retained as ``/edit-soft`` for backward compatibility
        with the dashboard SPA and any external scripts; semantically it
        is now "edit-apply".
        """
        _require_any_auth(request)
        caller = _extract_verified_agent_id(request)
        if not _caller_is_operator(caller, request) and not _is_internal_caller(request):
            raise HTTPException(403, "Only the operator can edit agent config")
        # We require *a* validatable origin so request-level audit fields
        # are populated; we do NOT require human kind (see docstring).
        origin = _validated_origin(request, caller)
        origin_kind = origin.kind if origin is not None else None

        data = await request.json()
        field = data.get("field", "")
        new_value = data.get("value")
        reason = data.get("reason", "user_asked")
        if not field:
            raise HTTPException(400, "field is required")
        if field not in _EDITABLE_FIELDS:
            raise HTTPException(
                400,
                f"Invalid field {field!r}. Editable fields: {sorted(_EDITABLE_FIELDS)}",
            )

        # Self-modification block. The operator-tool layer also blocks
        # this, but enforce server-side too in case a future caller
        # bypasses the tool.
        if agent_id == "operator":
            raise HTTPException(400, "The operator agent cannot be edited via soft-edit")

        # Validate runtime-critical values for hard fields before they
        # can be persisted. Mirrors the validation block in the now-
        # deprecated /propose endpoint.
        if field == "model":
            if not isinstance(new_value, str) or not new_value:
                raise HTTPException(400, "model must be a non-empty string")
            # Match create-agent BYOK validation (PR #901). Use the
            # live vault, not raw env, so OAuth-only providers count;
            # skip when no vault is wired (test harnesses) so existing
            # vault-less tests don't regress to 400.
            if credential_vault is not None:
                from src.shared.models import resolve_provider_for_model
                _provider = resolve_provider_for_model(new_value)
                if _provider:
                    _available = credential_vault.get_providers_with_credentials()
                    if _provider not in _available:
                        _available_list = sorted(_available) if _available else "none"
                        raise HTTPException(
                            400,
                            f"Model '{new_value}' requires '{_provider}' "
                            f"credentials, but no {_provider.upper()} key is "
                            f"configured. Available providers: "
                            f"{_available_list}. Set "
                            f"OPENLEGION_SYSTEM_{_provider.upper()}_API_KEY or "
                            f"pick a different model.",
                        )
                # Credential-kind-aware check: OAuth-only providers only
                # accept specific models. See Fix 2 in seam follow-up.
                _compatible, _reason = credential_vault.is_model_compatible(new_value)
                if not _compatible:
                    raise HTTPException(
                        400, _reason or f"Model '{new_value}' is not compatible.",
                    )
        elif field == "thinking":
            from src.agent.llm import LLMClient
            if new_value not in LLMClient.VALID_THINKING_LEVELS:
                raise HTTPException(
                    400,
                    f"thinking must be one of: {sorted(LLMClient.VALID_THINKING_LEVELS)}",
                )

        from src.cli.config import _load_config
        agent_cfg = _load_config()
        agents = agent_cfg.get("agents", {})
        if agent_id not in agents:
            raise HTTPException(404, f"Agent '{agent_id}' not found")

        # ``heartbeat_schedule`` is sourced from the live cron job, not
        # YAML — the cron table is the source of truth for an agent's
        # actual heartbeat cadence. Permissions live in permissions.json,
        # not agents.yaml. Everything else (model, budget, thinking,
        # instructions, soul, ...) reads from the agent's YAML row.
        if field == "heartbeat_schedule":
            old_value = ""
            if cron_scheduler is not None:
                hb_job = cron_scheduler.find_heartbeat_job(agent_id)
                if hb_job is not None:
                    old_value = hb_job.schedule
        elif field == "permissions":
            from src.cli.config import _load_permissions
            perms = _load_permissions()
            old_value = perms.get("permissions", {}).get(agent_id, {})
        else:
            yaml_key = _CONFIG_FIELD_MAP.get(field, field)
            old_value = agents[agent_id].get(yaml_key, "")

        # Apply directly via the same write helper used by the confirm
        # path. ``_apply_pending_change`` is async and handles audit
        # logging, hot-reload, and heartbeat schedule sync.
        # ``undoable=True`` here marks the audit row as Revert-eligible
        # — the change_id IS the undo_token for the change_history row
        # we record below, so the dashboard can wire Revert directly to
        # ``/changes/undo/{change_id}``.
        undo_token = str(_uuid.uuid4())
        await _apply_pending_change(
            undo_token,
            {
                "agent_id": agent_id,
                "field": field,
                "old_value": old_value,
                "new_value": new_value,
            },
            undoable=True,
        )

        # Detect older unconsumed receipts on the same field BEFORE we
        # record the new one. They stay revertible, but rolling them
        # back from the latest value would silently overwrite this
        # edit; the dashboard renders a "superseded" warning so the
        # operator knows. We compute the list before record() so we
        # don't have to filter ourselves out.
        prior_unconsumed = change_history.list_unconsumed_for_field(
            agent_id, field,
        )

        # Record the change for undo. Summary uses humanized field names
        # so receipt cards read naturally ("Updated writer's personality")
        # without the dashboard having to do its own field translation.
        # TTL is field-aware: hard fields (model/permissions/budget/
        # thinking) get the longer 30-min window so the user has more
        # time to catch a costly edit; soft fields keep the snappy 5 min.
        summary = f"Updated {agent_id}'s {_humanize_field(field)}"
        record = change_history.record(
            undo_token=undo_token,
            actor=caller,
            agent_id=agent_id,
            field=field,
            old_value=old_value,
            new_value=new_value,
            summary=summary,
            reason=reason,
            ttl=_ttl_for_field(field),
        )

        # Emit the receipt event. Dashboard listens for this and appends
        # the receipt card to the operator's chat so the user sees what
        # happened with [View diff] [Undo] buttons. ``supersedes_count``
        # tells the UI how many older revertible receipts on the same
        # field this edit makes stale (the older receipts can still be
        # undone — but doing so would erase this edit, so the dashboard
        # surfaces a warning).
        if event_bus is not None:
            try:
                event_bus.emit(
                    "operator_action_receipt",
                    agent="operator",
                    data={
                        "actor": caller,
                        "agent_id": agent_id,
                        "field": field,
                        "summary": summary,
                        "old_value": old_value,
                        "new_value": new_value,
                        "undo_token": undo_token,
                        "expires_at": record["expires_at"],
                        "reason": reason,
                        "origin_kind": origin_kind,
                        "supersedes_count": len(prior_unconsumed),
                    },
                )
            except Exception as e:
                logger.debug("operator_action_receipt emit failed: %s", e)

        # For each older unconsumed receipt on this field, emit a
        # ``operator_action_receipt_superseded`` event so the dashboard
        # can transition the older card into a "superseded by newer
        # edits" state. The older receipt is still revertible — the
        # event is purely a UX hint that an undo here would also
        # discard the newer edits.
        if event_bus is not None and prior_unconsumed:
            for prior in prior_unconsumed:
                try:
                    event_bus.emit(
                        "operator_action_receipt_superseded",
                        agent="operator",
                        data={
                            "undo_token": prior["undo_token"],
                            "agent_id": agent_id,
                            "field": field,
                            "superseded_by_token": undo_token,
                            "superseded_by_count": 1,
                        },
                    )
                except Exception as e:
                    logger.debug(
                        "operator_action_receipt_superseded emit failed: %s", e,
                    )

        # Fix 4 (seam follow-up): a successful ``model`` change is the
        # operator's "fix the credential" signal — clear any standing
        # quarantine implicitly so the lane resumes dispatching. No
        # separate ``clear_quarantine`` operator tool needed.
        if field == "model" and health_monitor is not None:
            try:
                health_monitor.clear_quarantine(
                    agent_id, reason="model changed via edit_agent",
                )
            except Exception as e:
                logger.debug("clear_quarantine on edit failed: %s", e)

        return {
            "success": True,
            "agent_id": agent_id,
            "field": field,
            "undo_token": undo_token,
            "expires_at": datetime.fromtimestamp(
                record["expires_at"], tz=timezone.utc,
            ).isoformat(),
            "ttl_seconds": _ttl_for_field(field),
            "field_class": "hard" if field in _HARD_EDIT_FIELDS else "soft",
            "summary": summary,
            "supersedes_count": len(prior_unconsumed),
        }

    @app.post("/mesh/operator/internet-access")
    async def operator_internet_access(request: Request) -> dict:
        """Toggle the operator's ability to use http_request / web_search.

        Body: ``{"enabled": bool}``. Operator-only or internal callers.

        Two-step apply: (1) flip ``operator.can_use_internet`` in
        permissions.json + reload the mesh-side matrix; (2) push to the
        operator container's ``/config`` endpoint so the agent loop's
        ``_runtime_disabled_tools`` is updated immediately — the next
        LLM tool surface filters ``http_request`` and ``web_search`` out
        when disabled, or restores them when re-enabled. ``hot_reload_ok``
        in the response reports whether the container-side push
        succeeded; the permissions-file write is the source of truth
        regardless (the container picks it up on next restart).

        Emits ``agent_config_updated`` (agent=``operator``,
        ``field="can_use_internet"``, ``live=<bool>``) and writes an
        ``edit_agent`` audit row tagged ``undoable=False`` (the toggle
        is its own UI affordance — flipping it back is just a re-click).
        """
        _require_any_auth(request)
        caller = _extract_verified_agent_id(request)
        if not _caller_is_operator(caller, request) and not _is_internal_caller(request):
            raise HTTPException(
                403, "Only the operator can toggle internet access",
            )

        data = await request.json()
        if "enabled" not in data:
            raise HTTPException(400, "'enabled' is required")
        enabled = data.get("enabled")
        if not isinstance(enabled, bool):
            raise HTTPException(400, "'enabled' must be a boolean")

        from src.cli.config import (
            _OPERATOR_AGENT_ID,
            _load_permissions,
            _save_permissions,
        )

        perms = _load_permissions()
        op_perms = perms.setdefault("permissions", {}).setdefault(
            _OPERATOR_AGENT_ID, {},
        )
        previous = bool(op_perms.get("can_use_internet", False))
        op_perms["can_use_internet"] = enabled
        _save_permissions(perms)
        if permissions is not None:
            permissions.reload()

        # Push to the operator's container so the runtime tool surface
        # flips immediately. ``hot_reload_ok`` defaults True when the
        # container isn't registered (e.g. mid-restart) — the durable
        # write is what matters; the container picks it up on next boot.
        hot_reload_ok = True
        if transport is not None and _OPERATOR_AGENT_ID in router.agent_registry:
            try:
                result = await transport.request(
                    _OPERATOR_AGENT_ID, "POST", "/config",
                    json={"internet_access_enabled": enabled}, timeout=10,
                )
                if isinstance(result, dict) and "error" in result:
                    hot_reload_ok = False
                    logger.warning(
                        "Operator /config push failed: %s", result["error"],
                    )
            except Exception as e:
                hot_reload_ok = False
                logger.warning(
                    "Operator /config push raised: %s", e,
                )

        blackboard.log_audit(
            action="edit_agent",
            target=_OPERATOR_AGENT_ID,
            field="can_use_internet",
            before_value=json.dumps(previous),
            after_value=json.dumps(enabled),
            undoable=False,
        )

        if event_bus is not None:
            try:
                event_bus.emit(
                    "agent_config_updated", agent=_OPERATOR_AGENT_ID,
                    data={
                        "agent_id": _OPERATOR_AGENT_ID,
                        "field": "can_use_internet",
                        "live": hot_reload_ok,
                    },
                )
            except Exception as e:
                logger.debug(
                    "agent_config_updated emit failed for internet-access: %s", e,
                )

        return {
            "success": True,
            "enabled": enabled,
            "previous": previous,
            "live": hot_reload_ok,
        }

    @app.get("/mesh/operator/internet-access")
    async def operator_internet_access_status(request: Request) -> dict:
        """Read the operator's current internet-access state.

        Returned shape: ``{"enabled": bool}``. The Operator Settings UI
        polls this to render the toggle's initial state on mount.
        """
        _require_any_auth(request)
        caller = _extract_verified_agent_id(request)
        if not _caller_is_operator(caller, request) and not _is_internal_caller(request):
            raise HTTPException(
                403, "Only the operator can read internet-access state",
            )
        from src.cli.config import _OPERATOR_AGENT_ID, _load_permissions
        perms = _load_permissions()
        op_perms = perms.get("permissions", {}).get(_OPERATOR_AGENT_ID, {})
        return {"enabled": bool(op_perms.get("can_use_internet", False))}

    @app.post("/mesh/changes/undo/{undo_token}")
    async def undo_change(undo_token: str, request: Request) -> dict:
        """Reverse a recent soft edit. 5-minute TTL.

        Looks up the token, atomically claims it (single-shot, no
        double-undo), and reapplies the OLD value via
        ``_apply_pending_change``. Caller must be the operator or an
        internal caller — same bar as the soft-edit endpoint.
        """
        _require_any_auth(request)
        caller = _extract_verified_agent_id(request)
        if not _caller_is_operator(caller, request) and not _is_internal_caller(request):
            raise HTTPException(403, "Only the operator can undo changes")

        record = change_history.consume_for_undo(undo_token)
        if record is None:
            # Distinguish "never existed" from "expired/consumed" via
            # peek so the dashboard can render the right copy. Both
            # collapse to 404 from the HTTP boundary.
            raise HTTPException(
                404, "Undo token unknown, expired, or already used",
            )

        # Reapply the old value. Note swap: new=old (the value the user
        # had before the edit), old=new (the value being reverted).
        # ``is_undo=True`` flips the permissions branch from merge to
        # replace so a partial-grant edit can be unset (merge would
        # leave the granted keys in place since the stored old_value
        # was the full pre-edit dict).
        try:
            await _apply_pending_change(
                undo_token,
                {
                    "agent_id": record["agent_id"],
                    "field": record["field"],
                    "old_value": record["new_value"],
                    "new_value": record["old_value"],
                },
                is_undo=True,
            )
        except HTTPException:
            # Agent might have been deleted between record + undo.
            # Surface a clean 4xx; the row has already been marked
            # consumed so it can't be retried.
            raise
        except Exception as e:
            raise HTTPException(500, f"Undo apply failed: {e}")

        # Emit a "receipt undone" event so the dashboard can transition
        # the original receipt card into a "Reverted" state.
        if event_bus is not None:
            try:
                event_bus.emit(
                    "operator_action_receipt_undone",
                    agent="operator",
                    data={
                        "actor": caller,
                        "agent_id": record["agent_id"],
                        "field": record["field"],
                        "summary": (
                            f"Reverted {record['agent_id']}'s "
                            f"{_humanize_field(record['field'])}"
                        ),
                        "undo_token": undo_token,
                        "restored_value": record["old_value"],
                    },
                )
            except Exception as e:
                logger.debug("operator_action_receipt_undone emit failed: %s", e)

        return {
            "success": True,
            "agent_id": record["agent_id"],
            "field": record["field"],
            "restored_value": record["old_value"],
        }

    def _confirm_origin_check(request: Request, caller: str = "") -> None:
        """Task 2d: confirm-side gate — refuse non-human origins.

        Pending operator-config edits are durable, so an agent that
        spoofed an X-Origin header (or a buggy adapter that promoted a
        non-human origin) must not be able to flip the lever. The
        propose endpoint records the proposer's origin kind on the row;
        ``pending_actions.consume(require_origin_kind="human")`` then
        rejects rows whose stored origin is not ``human``. We *also*
        re-check the *current* request's origin here so a confirm
        attempt that arrives without a human X-Origin is refused
        immediately, before we ever consume the row.
        """
        confirm_origin = _validated_origin(request, caller)
        if confirm_origin is None or confirm_origin.kind != "human":
            raise HTTPException(403, "Confirmation requires human origin")

    @app.post("/mesh/config/confirm")
    async def confirm_config_change(request: Request) -> dict:
        """Apply a destructive pending action by change_id only.

        Post PR #927 this endpoint is the consume side of the
        delete-confirmation flow only. Config edits no longer flow
        through propose+confirm — they apply immediately via
        ``/mesh/agents/{id}/edit-soft`` and emit an undo receipt
        instead. The endpoint name is retained for SDK / dashboard
        back-compat with the existing ``MeshClient.confirm_config_change``
        method, which the destructive-action review surface (delete-team,
        delete-agent) still uses.

        Accepted shape: ``target_kind in {"project", "agent"}`` +
        ``action_kind="delete"``. Any other ``action_kind`` returns
        HTTP 400 — non-delete pending rows are no longer produced.

        The endpoint inherits ``require_origin_kind="human"`` on
        ``PendingActions.consume`` and the additional confirm-side
        ``_confirm_origin_check`` so a forged X-Origin or buggy
        non-human caller can't flip the lever.
        """
        caller = _extract_verified_agent_id(request)
        if not _caller_is_operator(caller, request) and not _is_internal_caller(request):
            raise HTTPException(403, "Only the operator can confirm config changes")
        _confirm_origin_check(request, caller)
        data = await request.json()
        change_id = data.get("change_id", "")
        client_digest = data.get("payload_digest")

        # Consume + apply. ``consume`` is atomic: same nonce can only
        # be applied once. If the apply raises, the row is already
        # gone — the caller must propose a new change.
        record = pending_actions.consume(
            change_id,
            confirmer="operator",
            require_origin_kind="human",
            expected_payload_digest=client_digest,
        )
        if not record:
            raise HTTPException(400, "Pending action invalid or expired")

        # Post PR #927 this dispatcher is the delete-only path.
        # ``action_kind`` is always ``"delete"`` and ``target_kind`` is
        # always one of {"project", "agent"} — the only producers
        # (/mesh/teams/{name}/propose-delete and
        # /mesh/agents/{id}/propose-delete) hard-code those values.
        if record.get("action_kind") == "delete" and record.get("target_kind") in {
            "project", "agent",
        }:
            return await _apply_pending_delete(record)

        # Defensive: a stray non-delete row would otherwise reach the
        # retired legacy edit-apply path. Refuse it loudly so the
        # producer surfaces in logs.
        logger.warning(
            "rejecting non-delete pending row on /mesh/config/confirm: "
            "action_kind=%r target_kind=%r",
            record.get("action_kind"),
            record.get("target_kind"),
        )
        raise HTTPException(
            400,
            "Pending action is not a delete confirmation; "
            "/mesh/config/confirm no longer accepts config edits "
            "(use /mesh/agents/{id}/edit-soft instead).",
        )

    # === Task 9 — Pending action review surface ===
    #
    # The dashboard's System > Operator panel and the inline chat
    # bubble both call these endpoints. Confirm wraps the existing
    # ``/mesh/config/confirm`` path; cancel is the additive escape
    # hatch (delete-without-apply) backed by ``PendingActions.cancel``.
    # Both inherit CSRF protection (X-Requested-With) from the
    # dashboard's fetch wrapper and the `_csrf_check` middleware.

    @app.get("/mesh/pending")
    async def list_pending_actions(request: Request) -> dict:
        """List every non-expired pending action.

        Operator-or-internal only. Reaps expired rows on the read so
        the response is immediately accurate. Each row is shape-compatible
        with the dashboard's existing pending-edit display logic.
        """
        _require_operator_or_internal(request)
        pending_actions.reap_expired()
        rows = pending_actions.list_pending()
        return {"pending": rows}

    @app.post("/mesh/pending/{nonce}/confirm")
    async def pending_confirm(nonce: str, request: Request) -> dict:
        """Confirm a pending action by nonce.

        Thin wrapper over the existing ``/mesh/config/confirm`` flow:
        injects ``change_id=nonce`` into the body and dispatches to
        :func:`confirm_config_change`. Keeps the new surface stable
        without duplicating the destructive-vs-edit branch logic.
        """
        # Resolve and inject the nonce so the underlying confirm
        # handler reads it from the body the same way the legacy
        # callers do.
        body = {}
        try:
            body = await request.json()
        except Exception:
            body = {}
        body["change_id"] = nonce

        async def _receive():
            return {
                "type": "http.request",
                "body": json.dumps(body).encode("utf-8"),
                "more_body": False,
            }

        # Build a shallow copy of the request with the rewritten body.
        from starlette.requests import Request as _StarletteRequest
        forwarded = _StarletteRequest(request.scope, _receive)
        return await confirm_config_change(forwarded)

    @app.post("/mesh/pending/{nonce}/cancel")
    async def pending_cancel(nonce: str, request: Request) -> dict:
        """Cancel a pending action by nonce.

        Operator-or-internal only. Calls ``PendingActions.cancel`` which
        deletes the row + emits ``pending_action_resolved`` with
        ``status="cancelled"`` so the dashboard panel and chat bubble
        clear immediately.
        """
        caller = _extract_verified_agent_id(request)
        if not _caller_is_operator(caller, request) and not _is_internal_caller(request):
            raise HTTPException(403, "Only the operator can cancel pending actions")
        record = pending_actions.cancel(nonce, actor=caller or "operator")
        if record is None:
            raise HTTPException(404, "Pending action not found or already expired")
        return {
            "ok": True,
            "nonce": nonce,
            "target_kind": record["target_kind"],
            "target_id": record["target_id"],
            "action_kind": record["action_kind"],
        }

    @app.post("/mesh/audit/archive")
    async def audit_archive(request: Request) -> dict:
        """Bulk-archive operator audit entries older than ``before_date``.

        Operator-or-internal only. Soft-archive: rows are flipped to
        ``archived=1`` and dropped from the default audit-log view.
        Use ``GET /api/operator-audit?include_archived=true`` to see
        archived rows. Returns ``{archived_count: N, truncated: bool}``
        — ``truncated`` is ``True`` when the per-call hard cap (100k
        rows) was hit and additional matching rows remain. The caller
        identity is recorded as the actor on an audit-of-audit row so
        archive operations remain traceable even after the original
        rows are hidden.
        """
        caller = _extract_verified_agent_id(request)
        if not _caller_is_operator(caller, request) and not _is_internal_caller(request):
            raise HTTPException(403, "Only the operator can archive audit entries")
        try:
            body = await request.json()
        except Exception:
            body = {}
        before_date = (body or {}).get("before_date") if isinstance(body, dict) else None
        if not before_date or not isinstance(before_date, str):
            raise HTTPException(400, "before_date is required (ISO 8601 string)")
        # Normalise the cutoff to the SQLite ``datetime('now')`` shape
        # (``YYYY-MM-DD HH:MM:SS``) so date-string comparison against
        # the audit_log.timestamp column behaves predictably across
        # bare-date / Z-suffixed / offset-suffixed inputs.
        try:
            dt = datetime.fromisoformat(before_date.replace("Z", "+00:00"))
            if dt.tzinfo is not None:
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            normalised = dt.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            raise HTTPException(
                400,
                f"Invalid before_date: {before_date}. Expected ISO 8601"
                " (e.g., '2026-04-01' or '2026-04-01T00:00:00Z')",
            )
        # Best-effort actor: real operator/internal caller, else "operator".
        actor = caller or "operator"
        try:
            result = blackboard.archive_audit_before(normalised, actor=actor)
        except ValueError as e:
            raise HTTPException(400, str(e))
        except Exception as e:
            logger.warning("audit archive failed: %s", e)
            raise HTTPException(500, "Failed to archive audit entries")
        return {
            "ok": True,
            "archived_count": int(result.get("archived_count", 0)),
            "truncated": bool(result.get("truncated", False)),
            "before_date": before_date,
        }

    # === Browser Service Proxy ===

    import httpx as _httpx
    _browser_proxy_client = _httpx.AsyncClient(timeout=60)

    _last_browser_boot_id: str | None = None
    # Throttle for `_check_browser_boot_id_changed`. The probe was
    # previously fired on every browser command (one extra HTTP round-trip
    # per navigate / click / type), even though the browser service
    # restarts at most once per deploy. 30s is a good balance: a real
    # restart is detected within one tick, and the per-command overhead
    # collapses from "always probe" to "almost never probe."
    _last_boot_id_check_ts: float = 0.0
    _BOOT_ID_CHECK_INTERVAL_S = 30.0

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
        """Check if the browser service restarted by comparing boot_id.

        Throttled to one probe per ``_BOOT_ID_CHECK_INTERVAL_S`` seconds
        so the per-command overhead is bounded; a real restart is
        detected on the next tick after it happens. Treats the first
        successful contact (when ``_last_browser_boot_id`` is still
        ``None``) as a restart so the cold-start race — where the
        deferred startup push at ``+5s`` ran before the browser service
        was reachable — self-heals on the first browser command instead
        of leaving every agent permanently in "no proxy" mode.
        """
        nonlocal _last_browser_boot_id, _last_boot_id_check_ts
        if not container_manager:
            return False
        svc_url = getattr(container_manager, "browser_service_url", None)
        svc_token = getattr(container_manager, "browser_auth_token", "")
        if not svc_url:
            return False
        now = time.monotonic()
        if now - _last_boot_id_check_ts < _BOOT_ID_CHECK_INTERVAL_S:
            return False
        _last_boot_id_check_ts = now
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
                # First successful contact. The deferred startup push at
                # +5s may have run when the browser service wasn't ready
                # yet (silently failing per-agent); treating this as a
                # restart triggers a re-push so agents that missed the
                # initial window pick up their proxy config.
                _last_browser_boot_id = boot_id
                return True
            if boot_id != _last_browser_boot_id:
                _last_browser_boot_id = boot_id
                return True
        except Exception:
            pass
        return False

    async def _deferred_push_browser_proxies() -> None:
        """Push proxy config for all agents after a delay (non-blocking background task)."""
        nonlocal _last_browser_boot_id, _last_boot_id_check_ts
        await asyncio.sleep(5)  # Wait for agents to register
        if not container_manager:
            return
        svc_url = getattr(container_manager, "browser_service_url", None)
        if not svc_url:
            return
        agents = list(router.agent_registry.keys())
        if not agents:
            return
        # Push in parallel — sequential awaits could add seconds of
        # startup latency on a fleet with many agents. Each
        # ``_push_browser_proxy`` already swallows its own exceptions
        # (line ~2745); ``return_exceptions=True`` is belt-and-suspenders
        # so one transient failure doesn't cancel the rest.
        await asyncio.gather(
            *(_push_browser_proxy(agent_id) for agent_id in agents),
            return_exceptions=True,
        )
        # Seed ``_last_browser_boot_id`` so the first browser command's
        # restart-detection probe doesn't fire a redundant re-push for
        # all agents. If the deferred push above failed (e.g. browser
        # container slow to start), boot_id stays None and the first-
        # contact recovery path in ``_check_browser_boot_id_changed``
        # still corrects the race. We bypass the throttle here — this
        # is a one-time seed at startup, not the per-command hot path.
        try:
            svc_token = getattr(container_manager, "browser_auth_token", "")
            headers: dict = {}
            if svc_token:
                headers["Authorization"] = f"Bearer {svc_token}"
            resp = await _browser_proxy_client.get(
                f"{svc_url}/browser/status", headers=headers, timeout=5,
            )
            boot_id = resp.json().get("boot_id")
            if boot_id:
                _last_browser_boot_id = boot_id
                _last_boot_id_check_ts = time.monotonic()
        except Exception:
            # Browser container not ready yet — leave boot_id unset so
            # the first-contact branch in `_check_browser_boot_id_changed`
            # picks up the race correction.
            pass
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
            # Stamp boot_id into the per-payload event data so the dashboard
            # can detect a browser-service restart mid-session and flush its
            # local history (otherwise post-restart seq=1..N would interleave
            # with stale pre-restart entries — the dedup-by-seq path on the
            # JS side wouldn't catch it because the seqs don't collide,
            # they're just from a different counter generation). The mesh
            # poller already resets its own watermark on boot_id change
            # above; this propagates the same signal one hop further.
            event_data = (
                {**payload, "boot_id": boot_id} if boot_id else payload
            )
            event_bus.emit(event_type, agent=agent_id, data=event_data)

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
            caller_id, body.get("target_agent_id") or "", request,
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

        # Phase 6 §9.1 operator kill-switch for the network-inspection
        # surface. Mirrors the BROWSER_DOWNLOADS_DISABLED pattern so
        # operators can disable read-only request logging fleet-wide
        # without removing the action from `browser_actions` per agent.
        if action == "inspect_requests":
            from src.browser.flags import get_bool
            if get_bool(
                "BROWSER_NETWORK_INSPECT_DISABLED", False, agent_id=req_agent_id,
            ):
                raise HTTPException(403, detail={
                    "success": False,
                    "error": {
                        "code": "forbidden",
                        "message": "Network inspection disabled by operator",
                        "retry_after_ms": None,
                    },
                })

        # Check for browser service restart — re-push proxy config for ALL agents
        try:
            restarted = await _check_browser_boot_id_changed()
            if restarted:
                all_agents = list(router.agent_registry.keys())
                if all_agents:
                    logger.info(
                        "Browser service restarted, re-pushing proxy for %d agents",
                        len(all_agents),
                    )
                    # Parallel — sequential awaits added cumulative latency
                    # to the triggering agent's command on large fleets.
                    await asyncio.gather(
                        *(_push_browser_proxy(_aid) for _aid in all_agents),
                        return_exceptions=True,
                    )
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

    def _stage_paths(handle: str) -> tuple[_Path, _Path]:
        bin_path = resolve_under_root(_UPLOAD_STAGE_DIR, f"{handle}.bin")
        meta_path = resolve_under_root(_UPLOAD_STAGE_DIR, f"{handle}.json")
        if bin_path is None or meta_path is None:
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
    _stage_locks_guard: asyncio.Lock | None = None
    _stage_locks_guard_loop: asyncio.AbstractEventLoop | None = None

    def _get_stage_locks_guard() -> asyncio.Lock:
        nonlocal _stage_locks_guard, _stage_locks_guard_loop
        loop = asyncio.get_running_loop()
        if _stage_locks_guard is None or _stage_locks_guard_loop is not loop:
            _stage_locks_guard = asyncio.Lock()
            _stage_locks_guard_loop = loop
            # Per-handle locks were bound to the previous loop; using
            # them on the new loop raises ``RuntimeError: ... attached
            # to a different loop``. Clear the dict so subsequent
            # ``_get_stage_lock`` calls construct fresh locks.
            _stage_locks.clear()
        return _stage_locks_guard

    async def _get_stage_lock(handle: str) -> asyncio.Lock:
        async with _get_stage_locks_guard():
            lock = _stage_locks.get(handle)
            if lock is None:
                lock = asyncio.Lock()
                _stage_locks[handle] = lock
            return lock

    async def _release_stage_lock_if_idle(handle: str) -> None:
        async with _get_stage_locks_guard():
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
            caller_id, body.get("target_agent_id") or "", request,
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

    # Exposed for tests so they can pre-fill rate-limit buckets instead
    # of looping thousands of times against the spam-only ceilings. Not
    # used by production code paths.
    app.state.rate_ts = _rate_ts
    app.state.rate_limits = _RATE_LIMITS

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
            caller_id, body.get("target_agent_id") or "", request,
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
            await websocket.send_text(dumps_safe(evt))
        try:
            while True:
                await websocket.receive_text()  # keep-alive
        except Exception as e:
            logger.debug("WebSocket disconnected: %s", e)
        finally:
            event_bus.unsubscribe(websocket)

    # ── Per-agent VNC reverse proxy ──────────────────────────────────────
    # ``/agent-vnc/{agent_id}/{path:path}`` forwards both HTTP (noVNC
    # static client) and WebSocket (the actual /websockify VNC stream)
    # through the browser service, which looks up the agent's allocated
    # KasmVNC port via the display allocator and proxies onward.
    #
    # Distinct prefix from any other ``/vnc/`` path so KasmVNC's
    # relative asset URLs (``vendor/foo.js``, ``app/ui.js``,
    # ``core/rfb.js``) — which resolve from the iframe's document base
    # to ``/agent-vnc/{agent_id}/vendor/foo.js`` — match the same
    # route and forward correctly.

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

    async def _verify_vnc_dashboard_session(request_or_ws) -> str | None:
        """Run the same dashboard session-cookie check both VNC routes need.
        Returns an error string or None on success.
        """
        from src.dashboard.auth import verify_session_cookie
        cookies = getattr(request_or_ws, "cookies", {}) or {}
        cookie_value = cookies.get("ol_session", "")
        return verify_session_cookie(cookie_value)

    @app.get("/agent-vnc/{agent_id}/{path:path}")
    async def vnc_http_proxy_per_agent(
        agent_id: str, path: str, request: Request,
    ):
        """Per-agent VNC HTTP proxy → browser service /agent-vnc/{agent_id}/...."""
        _reject_agent_tokens(request)
        auth_error = await _verify_vnc_dashboard_session(request)
        if auth_error is not None:
            raise HTTPException(401, auth_error)
        if not _AGENT_ID_RE.fullmatch(agent_id):
            raise HTTPException(404)
        svc_url = getattr(container_manager, "browser_service_url", None)
        svc_token = getattr(container_manager, "browser_auth_token", "")
        if not svc_url:
            raise HTTPException(502, "Browser service not available")
        target = f"{svc_url}/agent-vnc/{agent_id}/{path}"
        if request.url.query:
            target += f"?{request.url.query}"
        import httpx
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    target,
                    headers={"Authorization": f"Bearer {svc_token}"},
                )
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            logger.warning(
                "Per-agent VNC HTTP proxy failed %s -> %s: %s",
                agent_id, target, exc,
            )
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

    @app.websocket("/agent-vnc/{agent_id}/{path:path}")
    async def vnc_ws_proxy_per_agent(
        websocket: WebSocket, agent_id: str, path: str,
    ):
        """Per-agent VNC WebSocket proxy → browser service /agent-vnc/{agent_id}/...."""
        auth_error = await _verify_vnc_dashboard_session(websocket)
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
                        await websocket.close(
                            code=1008, reason="Agent access denied",
                        )
                        return
        if not _AGENT_ID_RE.fullmatch(agent_id):
            await websocket.close(code=1008, reason="invalid agent_id")
            return
        svc_url = getattr(container_manager, "browser_service_url", None)
        svc_token = getattr(container_manager, "browser_auth_token", "")
        if not svc_url:
            await websocket.close(
                code=1011, reason="Browser service not available",
            )
            return

        # browser_service_url is http://...; convert to ws:// for the
        # upstream connection. Preserve any query string the iframe sent
        # — urlencode keeps it safe if any future param value contains
        # ``&`` or ``=``. (KasmVNC's /websockify is typically called
        # without query params, but encoding is one line of insurance.)
        target = svc_url.replace("http://", "ws://").replace("https://", "wss://")
        target = f"{target}/agent-vnc/{agent_id}/{path}"
        if websocket.query_params:
            from urllib.parse import urlencode
            qs = urlencode(list(websocket.query_params.multi_items()))
            if qs:
                target += f"?{qs}"

        await websocket.accept(subprotocol="binary")
        try:
            import websockets

            async with websockets.connect(
                target,
                subprotocols=["binary"],
                **_websockets_headers_kw(
                    websockets.connect,
                    {"Authorization": f"Bearer {svc_token}"},
                ),
                compression=None,
                ping_interval=None,
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
                        logger.warning(
                            "Per-agent VNC client→upstream error: %s", e,
                        )

                async def upstream_to_client():
                    try:
                        async for msg in upstream:
                            if isinstance(msg, bytes):
                                await websocket.send_bytes(msg)
                            else:
                                await websocket.send_text(msg)
                    except Exception as e:
                        logger.warning(
                            "Per-agent VNC upstream→client error: %s", e,
                        )

                async def browser_keepalive():
                    """Touch the viewed agent's browser every 30s while VNC
                    is open. Per-agent endpoint — does NOT touch other
                    agents' browsers, so opening one operator's VNC tab
                    doesn't extend everyone's idle window.
                    """
                    if not svc_url:
                        return
                    try:
                        import httpx as _httpx
                        async with _httpx.AsyncClient(timeout=5) as _client:
                            while True:
                                await asyncio.sleep(30)
                                with contextlib.suppress(Exception):
                                    await _client.post(
                                        f"{svc_url}/browser/{agent_id}/keepalive",
                                        headers={
                                            "Authorization": f"Bearer {svc_token}",
                                        },
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
            logger.warning(
                "Per-agent VNC WebSocket proxy error %s -> %s: %s",
                agent_id, target, exc,
            )
        finally:
            with contextlib.suppress(Exception):
                await websocket.close()

    return app
