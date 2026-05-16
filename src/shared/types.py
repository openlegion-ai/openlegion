"""Pydantic models for all inter-component messages, events, and state.

This is THE contract between every component in OpenLegion.
Agent containers and the host process share only these types.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


def _generate_id(prefix: str, length: int = 12) -> str:
    return f"{prefix}{uuid.uuid4().hex[:length]}"


# === Protocol Constants ===

SILENT_REPLY_TOKEN = "__SILENT__"
"""Sentinel returned by agents to suppress empty responses."""

RESERVED_AGENT_IDS = frozenset({"mesh", "operator", "canary-probe"})
"""Internal component names that must not be used as agent IDs.

``canary-probe`` is the stable agent-id used by the stealth canary
(§5.4) for its dedicated profile. Reserving it prevents a user from
creating a real agent with the same id — which would otherwise have
its profile silently stomped the next time an operator ran the canary.
"""

AGENT_ID_RE_PATTERN = r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$"
"""Canonical agent ID regex — 1-64 chars, alphanumeric start, then alphanumeric/hyphen/underscore."""

HARD_EDIT_FIELDS = frozenset({"model", "permissions", "budget", "thinking"})
"""Agent-config fields that require explicit propose+confirm (the "hard"
review path). Mirrors :data:`SOFT_EDIT_FIELDS` — the union is the full
set of editable fields surfaced through the operator-tool layer.

These are the consequential edits — model swaps, permission grants,
budget tweaks, thinking-level changes — that:

* Take the longer 30-min review TTL on the pending-action store
  (vs 5-min for soft edits) so the user has time to read the diff.
* Refuse the soft `/edit-soft` shortcut and force the operator tool to
  fall through to the propose+confirm path.

Single source of truth — imported by :mod:`src.host.server` and
:mod:`src.agent.builtins.operator_tools`.
"""

SOFT_EDIT_FIELDS = frozenset({
    "instructions", "soul", "heartbeat", "heartbeat_schedule",
    "interface", "role",
})
"""Agent-config fields that apply immediately with a 5-min Undo receipt
(the "soft" review path). See :data:`HARD_EDIT_FIELDS` for the inverse.

``heartbeat_schedule`` retargets the agent's cron job in lockstep with
the YAML write — the operator can adjust monitoring cadence without a
new tool surface (PR-L'). Validation lives in
:func:`src.agent.builtins.operator_tools._validate_heartbeat_schedule`.
"""

# === Inter-Component Messages ===


class MessageOrigin(BaseModel):
    """Typed origin for a message flowing through the mesh.

    The ``kind`` field is the authorization-relevant piece: durable
    actions gate on ``kind == "human"`` (pending-action confirm), worker
    → operator wakes are blocked when ``kind == "agent"``, and
    unverifiable channel claims are downgraded.

    ``channel`` and ``user`` are free-form; they identify which surface
    the message came from (cli, dashboard, telegram, …) and the end-user
    id when one is available.

    The model is ``frozen=True`` — origins are stamped once at the entry
    point (CLI REPL, dashboard chat, channel adapter, cron tick, …) and
    must not be mutated mid-flight.
    """

    kind: Literal["human", "operator", "agent", "system", "heartbeat", "cron"]
    channel: str = ""
    user: str = ""

    model_config = {"frozen": True}

    def to_header_value(self) -> str:
        """Serialize to a JSON ``X-Origin`` header value.

        Format keeps the existing JSON shape (``{"channel": ..., "user": ...}``)
        and adds a ``kind`` key. Old mesh nodes parsing this with the
        legacy parser whitelist only ``channel`` and ``user`` and silently
        drop ``kind`` — so the rolling-deploy invariant holds.
        """
        import json as _json
        return _json.dumps(
            {"kind": self.kind, "channel": self.channel, "user": self.user},
            separators=(",", ":"),
        )

    @classmethod
    def from_header_value(
        cls, value: str | None, *, trust_kind: bool = False,
    ) -> "MessageOrigin | None":
        """Parse an ``X-Origin`` header value back into a model.

        Returns ``None`` on missing or malformed input (never a partial
        model). Treats a missing ``kind`` segment as ``kind="agent"``
        (the least-trusted origin). Unless ``trust_kind`` is true, an
        explicit privileged kind from the header is also downgraded to
        ``kind="agent"`` so request headers cannot self-assert human,
        operator, system, heartbeat, or cron authority.

        Field length bounds match the legacy parser
        (``channel`` ≤ 32, ``user`` ≤ 128, raw blob ≤ 512).
        """
        if not value or len(value) > 512:
            return None
        import json as _json
        try:
            parsed = _json.loads(value)
        except _json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None

        ch = parsed.get("channel", "")
        us = parsed.get("user", "")
        kind = parsed.get("kind")

        # Channel/user are required for legacy compatibility unless this
        # is a typed origin without an addressable surface (cron, system,
        # heartbeat). The legacy parser rejected empty channel/user, so
        # legacy headers without ``kind`` must still meet that bar.
        if not isinstance(ch, str) or not isinstance(us, str):
            return None

        if kind is None:
            # Legacy header — least-trusted default.
            kind = "agent"
            if not ch or not us:
                # Legacy parser rejected empty channel/user for headers
                # without a ``kind`` segment. Preserve that behavior so
                # downstream auto-notify paths do not see a half-shaped
                # legacy origin.
                return None

        if not isinstance(kind, str):
            return None

        if not trust_kind:
            # Inbound request headers are not authority. Preserve addressable
            # channel/user metadata, but downgrade all caller-supplied kinds.
            kind = "agent"
            if not ch or not us:
                return None

        if len(ch) > _MAX_ORIGIN_CHANNEL_LEN or len(us) > _MAX_ORIGIN_USER_LEN:
            return None

        try:
            return cls(kind=kind, channel=ch, user=us)
        except Exception:
            # Pydantic validation error — bad ``kind`` value.
            return None


# Field-length caps used by ``MessageOrigin.from_header_value`` and
# ``src.shared.trace.parse_origin_header``. Kept here so the type module
# is self-contained.
_MAX_ORIGIN_CHANNEL_LEN = 32
_MAX_ORIGIN_USER_LEN = 128


class AgentMessage(BaseModel):
    """Every message between agents passes through the mesh in this format."""

    id: str = Field(default_factory=lambda: _generate_id("msg_"))
    from_agent: str
    to: str
    type: Literal["task_request", "task_result", "event", "query", "cancel"]
    payload: dict[str, Any]
    workflow_id: str | None = None
    reply_to: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ttl: int = 300
    priority: Literal["low", "normal", "high", "urgent"] = "normal"


class TokenBudget(BaseModel):
    """Token budget for a task. Prevents runaway API spend."""

    max_tokens: int = 500_000
    used_tokens: int = 0
    max_cost_usd: float = 5.0
    estimated_cost_usd: float = 0.0

    def can_spend(self, estimated_tokens: int) -> bool:
        return (self.used_tokens + estimated_tokens) <= self.max_tokens

    def record_usage(self, tokens: int, model: str = "") -> None:
        self.used_tokens += tokens
        from src.shared.models import estimate_cost

        self.estimated_cost_usd += estimate_cost(model, total_tokens=tokens)


class TaskAssignment(BaseModel):
    """Sent to an agent to begin work on a task."""

    task_id: str = Field(default_factory=lambda: _generate_id("task_"))
    workflow_id: str
    step_id: str
    task_type: str
    input_data: dict[str, Any]
    context: dict[str, Any] = {}
    timeout: int = 120
    max_retries: int = 0
    token_budget: TokenBudget | None = None


class TaskResult(BaseModel):
    """Returned by an agent when a task completes or fails."""

    task_id: str
    status: Literal["complete", "failed", "cancelled", "timeout", "skipped", "pending"]
    result: dict[str, Any] | None = None
    error: str | None = None
    promote_to_blackboard: dict[str, Any] = {}
    tokens_used: int = 0
    duration_ms: int = 0


class AgentStatus(BaseModel):
    """Returned by agent health check endpoint."""

    agent_id: str
    role: str
    state: Literal["idle", "working", "blocked", "failed", "starting"]
    current_task: str | None = None
    capabilities: list[str] = []
    uptime_seconds: float = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    context_tokens: int = 0
    context_max: int = 0
    context_pct: float = 0.0


# === Blackboard & Events ===


class BlackboardEntry(BaseModel):
    """A single entry in the shared blackboard."""

    key: str
    value: dict[str, Any]
    written_by: str
    workflow_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ttl: int | None = None
    version: int = 1


class MeshEvent(BaseModel):
    """Published to topics via pub/sub."""

    id: str = Field(default_factory=lambda: _generate_id("evt_"))
    topic: str
    source: str
    payload: dict[str, Any] = {}
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# === Permissions ===


class AgentPermissions(BaseModel):
    """What an agent is allowed to do on the mesh."""

    agent_id: str
    can_message: list[str] = []
    can_publish: list[str] = []
    can_subscribe: list[str] = []
    blackboard_read: list[str] = []
    blackboard_write: list[str] = []
    allowed_apis: list[str] = []
    allowed_credentials: list[str] = []
    can_use_browser: bool = False
    browser_actions: list[str] | None = None  # None = all known actions
                                               # (default-allow UX).
                                               # ["*"] = all (explicit form).
                                               # Specific list = only those
                                               # (opt-out restriction).
                                               # [] = no actions (equivalent
                                               # to can_use_browser=False).
    # ``can_use_internet`` gates external HTTPS / web-search tool calls
    # (``http_request``, ``web_search``). Workers default to ``False``
    # but their tools are also default-ungated historically, so the
    # field has no effect for them unless the agent-side filter explicitly
    # consults it. The operator gets ``True`` via
    # ``_ensure_operator_agent`` so it can reach the internet by default;
    # the Operator Settings UI surfaces a toggle that flips this field.
    can_use_internet: bool = False
    # ``can_spawn`` (Task 3 narrowed semantics): gates EPHEMERAL spawning
    # only — subagents, cron-triggered spawns, and template applies that
    # produce short-lived workers. Durable fleet operations (creating
    # named agents, managing projects, editing config, viewing fleet
    # metrics, routing tasks, requesting user credentials) now live on
    # the dedicated control-plane permissions below. Workers default to
    # ``can_spawn=False``; the operator gets it via _ensure_operator_agent.
    can_spawn: bool = False
    can_manage_cron: bool = False
    # Control-plane permissions split from ``can_spawn`` (Task 3).
    # Workers default to False; operator defaults to True. Missing fields
    # on existing agent records default to False at load time so
    # pre-existing configs don't accidentally grant durable powers.
    can_manage_fleet: bool = False         # /mesh/agents/create, register/deregister
    # ``can_manage_teams`` (new canonical name; ``can_manage_projects``
    # kept as a back-compat alias). Either flag granted upstream is
    # mirrored onto the other via :meth:`_unify_manage_teams_alias` so
    # existing yaml configs that still spell ``can_manage_projects`` keep
    # working until PR 3 retires the alias.
    can_manage_teams: bool = False         # team create/archive, membership
    can_manage_projects: bool = False      # DEPRECATED: alias for can_manage_teams
    can_edit_agent_config: bool = False    # propose/confirm edits to instructions/soul/etc.
    can_view_fleet_metrics: bool = False   # /mesh/system/metrics, /mesh/agents/{id}/metrics
    can_route_tasks: bool = False          # durable task records (Task 6)
    can_request_user_credentials: bool = False  # request_credential, request_browser_login
    can_use_wallet: bool = False
    wallet_allowed_chains: list[str] = []
    wallet_spend_limit_per_tx_usd: float = 0.0
    wallet_spend_limit_daily_usd: float = 0.0
    wallet_rate_limit_per_hour: int = 0
    wallet_allowed_contracts: list[str] = []

    @model_validator(mode="after")
    def _unify_manage_teams_alias(self) -> "AgentPermissions":
        # Either flag granted on input implies both are set downstream so
        # callers can read either field without an OR. PR 3 removes the
        # alias entirely.
        if self.can_manage_teams or self.can_manage_projects:
            object.__setattr__(self, "can_manage_teams", True)
            object.__setattr__(self, "can_manage_projects", True)
        return self


# === Agent Configuration ===


class AgentConfig(BaseModel):
    """Structured fields for an agent entry in ``config/agents.yaml``.

    Today an entry is a free-form dict (role/model/initial_instructions/…).
    Task 8 introduces five structured fields for routing — what the agent
    does, what it accepts, what it produces, where it escalates, what it
    refuses. These fields are the source of truth for the operator and
    routing layer; ``INTERFACE.md`` is preserved as a free-text companion
    for the agent's own context but is no longer parsed at routing time.

    The model is read-only metadata: the loader still persists yaml as a
    dict (for back-compat with existing tooling that diffs the file). The
    model is used by callers that want a typed view of the entry, by
    tests, and by the `/mesh/agents/{id}/profile` endpoint to validate
    the surfaced shape.

    All five new fields default cleanly so existing agents.yaml files
    without them load unchanged. ``_derive_capabilities_from_interface``
    in ``src/agent/workspace.py`` provides a one-shot back-fill from
    ``INTERFACE.md`` headings on first read; the structured field is the
    source of truth thereafter.
    """

    role: str = ""
    model: str = ""
    skills_dir: str = ""
    initial_instructions: str = ""
    initial_soul: str = ""
    initial_heartbeat: str = ""
    initial_interface: str = ""
    thinking: str = ""
    status: str = "active"
    budget: dict[str, Any] | None = None
    resources: dict[str, Any] | None = None

    # Task 8 — structured routing metadata.
    capabilities: list[str] = Field(default_factory=list)
    preferred_inputs: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)
    escalation_to: str | None = None
    forbidden: list[str] = Field(default_factory=list)

    model_config = {"extra": "allow"}


# === Teams ===


class TeamMetadata(BaseModel):
    """Team definition loaded from config/teams/<name>/metadata.yaml.

    ``status`` defaults to ``"active"``; operator product tools use
    ``"archived"`` to stop scheduling and hide the team from default
    list views without deleting its data. Archive is reversible; delete
    requires archive first plus a separate human-confirmed step.

    ``north_star`` and ``success_criteria`` capture the team's goal as
    first-class fields. Both are nullable for backwards compatibility —
    teams that predate this schema simply have ``None`` and the UI
    renders an empty-state placeholder.

    A back-compat alias :data:`ProjectMetadata` is exported below — it
    points at the same class so existing imports keep working through
    the project→team rename. PR 3 removes the alias.
    """

    name: str
    description: str = ""
    members: list[str] = []
    created_at: str | None = None
    status: str = "active"
    settings: dict[str, Any] = {}
    north_star: str | None = None
    success_criteria: list[str] | None = None


# Back-compat alias — keep until PR 3.
ProjectMetadata = TeamMetadata


# === Coordination Requests ===


class BlackboardWatchRequest(BaseModel):
    """Request to watch blackboard keys matching a glob pattern."""

    agent_id: str
    pattern: str


class BlackboardClaimRequest(BaseModel):
    """Request for atomic compare-and-swap blackboard write."""

    agent_id: str
    key: str
    value: dict[str, Any]
    expected_version: int


# === Memory (inside agent container) ===


class MemoryFact(BaseModel):
    """A single fact in agent's private memory."""

    id: str = Field(default_factory=lambda: _generate_id("fact_", 8))
    key: str
    value: str
    category: str = "general"
    source: str = "agent"
    confidence: float = 1.0
    embedding: list[float] | None = None
    access_count: int = 0
    last_accessed: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    decay_score: float = 1.0


class MemoryLog(BaseModel):
    """An entry in the agent's action log."""

    id: str = Field(default_factory=lambda: _generate_id("log_", 8))
    action: str
    input_summary: str
    output_summary: str
    task_id: str | None = None
    tokens_used: int = 0
    duration_ms: int = 0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# === API Proxy (agent -> mesh -> external service) ===


class APIProxyRequest(BaseModel):
    """Agent requests external API call through mesh."""

    service: str
    action: str
    params: dict[str, Any] = {}
    timeout: int = 30


class APIProxyResponse(BaseModel):
    """Mesh returns external API result to agent."""

    success: bool
    data: dict[str, Any] | None = None
    error: str | None = None
    status_code: int | None = None


# === LLM Response (standardized across providers) ===


class ToolCallInfo(BaseModel):
    """A tool/function call requested by the LLM."""

    name: str
    arguments: dict[str, Any]


class LLMResponse(BaseModel):
    """Standardized response from any LLM provider via mesh proxy."""

    content: str = ""
    thinking_content: str | None = None
    tool_calls: list[ToolCallInfo] | None = None
    tokens_used: int = 0
    model: str = ""
    stop_reason: str | None = None


# === Chat Mode ===


class ChatMessage(BaseModel):
    """Incoming chat message from user to agent."""

    message: str


class ChatResponse(BaseModel):
    """Agent's response to a chat message."""

    response: str
    tool_outputs: list[dict[str, Any]] = []
    tokens_used: int = 0


class BrowserCommand(BaseModel):
    """Browser command sent from agent through mesh to browser service."""

    agent_id: str
    action: str  # navigate, snapshot, click, type, evaluate, screenshot, reset, focus, status, detect_captcha
    params: dict[str, Any] = {}


class BrowserResult(BaseModel):
    """Result from browser service back to agent."""

    success: bool
    data: dict[str, Any] = {}
    error: str | None = None


class SteerMessage(BaseModel):
    """Injected into an agent's active conversation mid-execution."""

    message: str


class NotifyRequest(BaseModel):
    """Agent requests to send a notification to the user."""

    agent_id: str
    message: str


# === Dashboard Events ===


class DashboardEvent(BaseModel):
    """Real-time event broadcast to connected dashboard WebSocket clients."""

    id: str = Field(default_factory=lambda: _generate_id("evt_"))
    type: Literal[
        "agent_state",
        "message_sent",
        "message_received",
        "tool_start",
        "tool_result",
        "text_delta",
        "llm_call",
        "blackboard_write",
        "health_change",
        "notification",
        "workspace_updated",
        "heartbeat_complete",
        "cron_change",
        "chat_user_message",
        "chat_done",
        "chat_reset",
        "credit_exhausted",
        "credential_request",
        "credential_request_cancelled",
        "credential_stored",
        "browser_login_request",
        "browser_login_completed",
        "browser_login_cancelled",
        # Phase 8 §11.14 — operator CAPTCHA handoff. Without these literals
        # DashboardEvent rejects the emit and the SPA card never renders.
        "browser_captcha_help_request",
        "browser_captcha_help_completed",
        "browser_captcha_help_cancelled",
        # Phase 4 §4.6 / Phase 7 §10.1 — per-minute browser aggregates
        # forwarded from the browser service to the dashboard via
        # _poll_browser_metrics_once. Without these literals
        # DashboardEvent rejects the emit and the panel never renders.
        "browser_metrics",
        "browser_nav_probe",
        # Task 9 — Workplace tab + pending-action review surfaces.
        # ``task_*`` are emitted from ``host/orchestration.py`` (Tasks
        # store) on create / status_change / reroute / cancel.
        # ``pending_action_*`` are emitted from ``host/pending_actions.py``
        # on store / consume(success) / reap_expired so the dashboard's
        # System > Operator panel and inline chat bubbles render the
        # operator's review queue without polling.
        "task_created",
        "task_status_changed",
        # ``task_outcome`` is emitted by ``host/orchestration.py:set_outcome``
        # when an operator (or system) rates a delivered task. Without this
        # literal, ``DashboardEvent`` validation rejects the emit and the
        # event silently disappears (swallowed by ``_safe_emit``), so the
        # notifications producer below never sees deliveries.
        "task_outcome",
        "task_artifact_added",
        "pending_action_created",
        "pending_action_resolved",
        "pending_action_expired",
        # PR — close EventBus coverage gaps. Without these literals,
        # ``EventBus.emit`` raises a Pydantic ValidationError that the
        # debug-level ``except Exception`` swallows and the dashboard
        # silently misses the event. Soft-edit Undo receipts and undo
        # confirmations were emitted but never delivered until these
        # literals were added — the visible regression that prompted
        # this audit.
        "operator_action_receipt",
        "operator_action_receipt_undone",
        "operator_action_receipt_superseded",
        # Live notification bell — emitted by ``_notifications_producer``
        # immediately after each ``notification_store.add`` call so the
        # bell badge updates without waiting for the 60s poll.
        "notification_added",
        # Agent / project lifecycle — archive/unarchive emit so the SPA
        # refreshes the relevant list without a full reload.
        "agent_archived",
        "agent_unarchived",
        # Agent restart — split start/finish so the SPA can render a
        # pulsing "Restarting" indicator and clear it on completion.
        "agent_restarting",
        "agent_restarted",
        # Hard-field config apply — emitted from
        # ``_apply_pending_change`` so the agent config card flips to
        # the new value live (secret fields are redacted in the
        # payload).
        "agent_config_updated",
        # Team (formerly "project") CRUD — create / update / delete so
        # the teams list refreshes without polling. Both the legacy
        # ``project_*`` literals and the new ``team_*`` literals are kept
        # through PR 3 — emitters fire BOTH so old listeners keep
        # working.
        "project_created",
        "project_updated",
        "project_deleted",
        "project_archived",
        "project_unarchived",
        "team_created",
        "team_updated",
        "team_deleted",
        "team_archived",
        "team_unarchived",
        # Blackboard delete — mirror of ``blackboard_write`` for the
        # delete endpoint so the SPA can reflect removals live.
        "blackboard_delete",
    ]
    agent: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data: dict[str, Any] = {}
