"""Pydantic models for all inter-component messages, events, and state.

This is THE contract between every component in OpenLegion.
Agent containers and the host process share only these types.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, Literal
from urllib.parse import urlparse

from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)


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

MCP_SERVER_NAME_RE_PATTERN = r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$"
"""MCP server name regex — matches :data:`AGENT_ID_RE_PATTERN` convention.

Used as a prefix when an MCP tool conflicts with a built-in tool or
another server's tool (``mcp_<server>_<tool>`` — see
:mod:`src.agent.mcp_client`).
"""

CRED_HANDLE_RE = re.compile(r"\$CRED\{([^}]+)\}")
"""Credential handle regex — ``$CRED{name}`` references resolved by the
mesh against :class:`src.host.credentials.CredentialVault` at the point
of use.

This is the canonical location. ``src.agent.builtins.CRED_HANDLE_RE``
is a mirrored constant kept for back-compat; prefer importing from
here.
"""

HARD_EDIT_FIELDS = frozenset(
    {"model", "permissions", "budget", "thinking", "max_output_tokens",
     "max_tool_rounds", "llm_timeout_seconds"}
)
"""Agent-config fields that earn the longer 30-min Undo window (the
"hard" review path). Mirrors :data:`SOFT_EDIT_FIELDS` — the union is
the full set of editable fields surfaced through the operator-tool
layer.

These are the consequential edits — model swaps, permission grants,
budget tweaks, thinking-level changes, output-cap changes. All edits
apply immediately
via ``/edit-soft``; the only difference between hard and soft fields
is the receipt TTL, i.e. how long the user has to click Undo:

* Hard fields → 30-min Undo window so the user has time to read the
  diff before the receipt expires.
* Soft fields → 5-min Undo window (see :data:`SOFT_EDIT_FIELDS`).

There is no propose+confirm step — the legacy gated flow was retired
in PR #927. Single source of truth — imported by :mod:`src.host.server`
and :mod:`src.agent.builtins.operator_tools`.
"""

SOFT_EDIT_FIELDS = frozenset({
    "instructions", "soul", "heartbeat", "heartbeat_schedule",
    "interface", "role",
})
"""Agent-config fields that apply immediately with a 5-min Undo receipt.
See :data:`HARD_EDIT_FIELDS` for the longer 30-min Undo window.

``heartbeat_schedule`` retargets the agent's cron job in lockstep with
the YAML write — the operator can adjust monitoring cadence without a
new tool surface (PR-L'). Validation lives in
:func:`src.agent.builtins.operator_tools._validate_heartbeat_schedule`.
"""

# Versioned markers embedded in system-managed heartbeat templates
# (currently operator's). When a new sentinel is added here AND the
# template, idempotent migrators in :func:`src.cli.config._ensure_operator_agent`
# and :class:`src.agent.workspace.WorkspaceManager._ensure_scaffold`
# detect the mismatch and roll the live agents.yaml + HEARTBEAT.md
# forward. User-customised heartbeats (no sentinel) are left alone.
# Add new markers to the END of the tuple to keep older sentinels as
# evidence that a workspace was previously migrated.
HEARTBEAT_SENTINELS: tuple[str, ...] = (
    "heartbeat_v2_workflow_aware",
    "heartbeat_v3_rate_delivery",
    "heartbeat_v4_goal_seeding",
    "heartbeat_v5_fleet_health",
    "heartbeat_v6_agent_retro",
)

# Same contract for the operator's INSTRUCTIONS playbook
# (``operator_playbooks._OPERATOR_CORE``). The config-side migrator
# refreshes the agents.yaml ``initial_instructions`` payload when the
# canonical playbook gains a new sentinel (the workspace migrator then
# APPENDS the matching addendum to the live INSTRUCTIONS.md — it never
# rewrites the file, so the operator's self-evolved content survives).
# Without the config-side refresh the container keeps receiving the
# creation-time payload forever and no workspace addendum ever fires —
# discovered live on a production box whose operator predated every
# sentinel and had silently missed two playbook generations.
PLAYBOOK_SENTINELS: tuple[str, ...] = (
    "playbook_v2",
    "playbook_v3_handoff_briefs",
    "playbook_v4_watch_mode",
    "playbook_v5_verification_wake",
    "playbook_v6_chat_delivery",
)

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
    # Loop liveness signal (Bug 1) — surfaced so the mesh health monitor can
    # detect a dead inner loop with a live FastAPI thread. ``last_iteration_ts``
    # is wall-clock seconds (time.time()) stamped at the head of each
    # task/chat iteration; ``iterations_since_boot`` is monotonically
    # increasing across the agent's lifetime. Both default to safe values
    # so older serialized payloads parse cleanly.
    last_iteration_ts: float | None = None
    iterations_since_boot: int = 0


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
    # Per-agent SKILL.md pack allowlist (names). A skill is *data*, not a
    # capability — this only controls which packs the agent can DISCOVER via
    # skills_list / skill_view, to keep context lean and relevant. The agent's
    # effective set is the union of this list and the fleet-wide ``fleet_skills``
    # (a top-level key in permissions.json), resolved by
    # ``PermissionMatrix.get_effective_skills``. Empty = no skills until
    # assigned (the operator + standalone agents see the full catalog).
    allowed_skills: list[str] = []
    can_use_browser: bool = False
    browser_actions: list[str] | None = None  # None = all known actions
                                               # (default-allow UX).
                                               # ["*"] = all (explicit form).
                                               # Specific list = only those
                                               # (opt-out restriction).
                                               # [] = no actions (equivalent
                                               # to can_use_browser=False).
    # ``can_use_internet`` gates external HTTPS / web-search tool calls
    # (``http_request``, ``web_search``). Field default is ``False`` so the
    # deny-all fallback for an unknown agent stays restrictive — exactly like
    # ``can_use_browser`` above. New agents nonetheless get internet ON: every
    # create path persists ``can_use_internet: True`` via the base defaults in
    # ``cli/config._add_agent_permissions`` (mirroring how that base flips
    # ``can_use_browser`` to True). Worker internet tools are also
    # default-ungated historically, so this field mainly drives the operator's
    # agent-side tool filter + the dashboard badge. The Operator Settings UI
    # surfaces a toggle that flips this field.
    can_use_internet: bool = False
    # ``can_spawn`` (Task 3 narrowed semantics): gates EPHEMERAL fleet-spawn
    # only — the ``POST /mesh/spawn`` / ``spawn_fleet_agent`` capability that
    # creates short-lived TTL-bounded peer agents. Durable fleet operations
    # (creating named agents, managing projects, editing config, viewing fleet
    # metrics, routing tasks, requesting user credentials) live on the
    # dedicated control-plane permissions below; the in-container
    # ``spawn_subagent`` helper is ungated and separate.
    # The FIELD default is ``False`` — this is the RECURSION WALL: an ephemeral
    # ``spawn-*`` agent is never written to permissions.json, so it resolves
    # via ``PermissionMatrix.get_permissions`` ("default" record / bare
    # fallback → this default), keeping it spawn-incapable so spawn trees stay
    # bounded at one level. New NAMED agents nonetheless get ``can_spawn=True``
    # via the create-path base in ``cli/config._add_agent_permissions`` (same
    # pattern as ``can_use_browser`` / ``can_use_internet``).
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
    can_edit_agent_config: bool = False    # edit another agent's instructions/soul/etc. (apply-immediately + undo)
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


class MCPServerConfig(BaseModel):
    """Shape of a single stdio MCP (Model Context Protocol) server.

    This is the ``MCP_SERVERS`` container-env contract: the mesh
    serializes a list of these (with ``$CRED{...}`` handles resolved)
    into each agent container, where :mod:`src.agent.mcp_client`
    launches every entry as a stdio subprocess. Fleet-level definitions
    live in the connector catalog as :class:`MCPConnector` (this model
    plus assignment); there is no per-agent MCP config.

    Credential handles (``$CRED{name}``) may appear in ``env`` values
    and in ``args`` strings; the mesh resolves them against the
    credential vault at agent start. They are **not** permitted in
    ``command`` — rejected here so users get a clear validation error
    instead of a confusing "executable not found" failure later.

    ``extra="forbid"`` so typos like ``commnad`` fail loudly.
    """

    model_config = {"extra": "forbid"}

    name: str = Field(
        min_length=1, max_length=64, pattern=MCP_SERVER_NAME_RE_PATTERN,
    )
    command: str = Field(min_length=1, max_length=256)
    args: list[str] = Field(default_factory=list, max_length=32)
    env: dict[str, str] | None = None

    @field_validator("command")
    @classmethod
    def _command_no_cred_handle(cls, v: str) -> str:
        if CRED_HANDLE_RE.search(v):
            raise ValueError(
                "Credential handles ($CRED{...}) are not allowed in `command` "
                "— use `env` or `args` instead.",
            )
        return v

    @field_validator("args")
    @classmethod
    def _args_per_item_length(cls, v: list[str]) -> list[str]:
        for i, a in enumerate(v):
            if len(a) > 512:
                raise ValueError(f"args[{i}] too long (max 512 chars)")
        return v

    @field_validator("env")
    @classmethod
    def _env_shape(
        cls, v: dict[str, str] | None,
    ) -> dict[str, str] | None:
        if v is None:
            return v
        if len(v) > 32:
            raise ValueError("env may have at most 32 entries")
        for k, val in v.items():
            if not k or len(k) > 128:
                raise ValueError(
                    f"invalid env key: {k!r} (must be 1-128 chars)",
                )
            if len(val) > 4096:
                raise ValueError(
                    f"env value for {k!r} too long (max 4096 chars)",
                )
        return v


CONNECTOR_ALL_AGENTS = "*"
"""Sentinel in :attr:`MCPConnector.agents` meaning "assigned to every
agent". Matches the glob convention used by
:attr:`AgentPermissions.can_message`."""


def _connector_agents_shape(v: list[str]) -> list[str]:
    """Shared ``agents`` validation for both connector transports:
    ``'*'`` must be the sole element; ids validated and deduped with
    order preserved."""
    if CONNECTOR_ALL_AGENTS in v and len(v) > 1:
        raise ValueError(
            "agents: '*' (all agents) cannot be combined with explicit ids",
        )
    seen: set[str] = set()
    out: list[str] = []
    for a in v:
        if a != CONNECTOR_ALL_AGENTS and not re.match(AGENT_ID_RE_PATTERN, a):
            raise ValueError(f"agents: invalid agent id {a!r}")
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


class MCPConnector(MCPServerConfig):
    """A fleet-level MCP connector: a stdio server definition plus its
    agent assignment. The single source of truth for which agents run
    which MCP servers, persisted in ``config/connectors.json`` and
    managed by :class:`src.host.connectors.ConnectorStore`.

    ``agents`` is either ``["*"]`` (every agent) or an explicit list of
    agent ids. An agent-specific server is simply a connector assigned
    to one agent — there is no separate per-agent MCP config layer.
    Inherits every :class:`MCPServerConfig` validator (name pattern, no
    ``$CRED`` in ``command``, args/env caps) unchanged.

    ``transport`` discriminates the :data:`Connector` union — this is
    the stdio variant; :class:`HttpConnector` is the remote one. The
    default lets pre-union files (and hand-written records) omit the
    key.
    """

    transport: Literal["stdio"] = "stdio"
    agents: list[str] = Field(default_factory=list, max_length=128)

    @field_validator("agents")
    @classmethod
    def _agents_shape(cls, v: list[str]) -> list[str]:
        return _connector_agents_shape(v)

    def applies_to(self, agent_id: str) -> bool:
        """True when this connector is assigned to ``agent_id``."""
        return CONNECTOR_ALL_AGENTS in self.agents or agent_id in self.agents

    def server_dict(self) -> dict:
        """The ``MCP_SERVERS``-shaped dict for this connector: the
        :class:`MCPServerConfig` fields only — catalog-only fields
        stripped. ``model_dump`` with an exclude set (rather than
        re-listing the server fields) so a field added to
        ``MCPServerConfig`` can never be silently dropped from the
        container env."""
        return self.model_dump(exclude={"agents", "transport"}, exclude_none=False)


class ConnectorAuth(BaseModel):
    """Auth binding for a remote (http) connector.

    The secret itself always lives in the vault; this only names it.
    ``bearer`` → vault credential injected as ``Authorization: Bearer``
    by the mesh gateway. Vault-existence is checked at dashboard PUT
    time, but deliberately NOT per-agent ``can_access_credential`` —
    the token is mesh-held and never enters a container (plan D14).
    ``oauth`` → vault connection key, set by the Phase-3 connect flow
    (refresh-on-resolve).
    """

    model_config = {"extra": "forbid"}

    kind: Literal["none", "bearer", "oauth"] = "none"
    cred: str | None = Field(default=None, max_length=128)
    connection: str | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def _kind_fields(self) -> "ConnectorAuth":
        if self.kind == "bearer" and not self.cred:
            raise ValueError("auth.kind='bearer' requires auth.cred")
        return self


class HttpConnector(BaseModel):
    """A fleet-level remote MCP server: a streamable-HTTP endpoint plus
    its agent assignment. Calls are proxied by the mesh-side gateway;
    auth is resolved from the vault per call.

    NEVER serialized into ``MCP_SERVERS`` — ``ConnectorStore``
    ``snapshot_for_agent`` filters to stdio before building container
    env, so this record (including ``auth``) cannot enter an agent
    container. Deliberately has no ``server_dict()``: anything that
    tries to treat it as a container server fails loudly.
    """

    model_config = {"extra": "forbid"}

    transport: Literal["http"]
    name: str = Field(
        min_length=1, max_length=64, pattern=MCP_SERVER_NAME_RE_PATTERN,
    )
    url: str = Field(min_length=1, max_length=512)
    auth: ConnectorAuth = Field(default_factory=ConnectorAuth)
    agents: list[str] = Field(default_factory=list, max_length=128)

    @field_validator("url")
    @classmethod
    def _url_shape(cls, v: str) -> str:
        p = urlparse(v)
        if not p.hostname:
            raise ValueError("Connector URL must include a host")
        if p.username or p.password:
            # A token in the URL would persist plaintext on disk and
            # echo verbatim in every GET/audit row — the one surface
            # the remote design promises secrets never appear on.
            raise ValueError(
                "Credentials are not allowed in the connector URL — "
                "use the auth field (vault credential) instead.",
            )
        if p.scheme == "https":
            return v
        # Self-hosted/dev MCP on the mesh host itself is legitimate.
        # This is a config-shape check only; the gateway's resolved-IP
        # blocklist (plan D16) is the SSRF layer.
        if p.scheme == "http" and p.hostname in ("localhost", "127.0.0.1", "::1"):
            return v
        raise ValueError(
            "Connector URL must be https:// (http:// allowed for localhost only)",
        )

    @field_validator("agents")
    @classmethod
    def _agents_shape(cls, v: list[str]) -> list[str]:
        return _connector_agents_shape(v)

    def applies_to(self, agent_id: str) -> bool:
        """True when this connector is assigned to ``agent_id``."""
        return CONNECTOR_ALL_AGENTS in self.agents or agent_id in self.agents


Connector = Annotated[MCPConnector | HttpConnector, Field(discriminator="transport")]
"""The connector-catalog record union, discriminated on ``transport``.
Validate untrusted records through :data:`CONNECTOR_ADAPTER` so both
variants share one entry point (and one 400-error shape)."""


def _default_transport(v: Any) -> Any:
    """Tag-extraction shim: discriminated unions do not fall back to
    field defaults for a missing tag, but pre-union catalog files (and
    hand-written records) legitimately omit ``transport``. Inject the
    stdio default for dict input only; model instances pass through."""
    if isinstance(v, dict) and "transport" not in v:
        return {**v, "transport": "stdio"}
    return v


CONNECTOR_ADAPTER: TypeAdapter = TypeAdapter(
    Annotated[Connector, BeforeValidator(_default_transport)],
)


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
    tools_dir: str = ""
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

    # Per-agent override of the task-loop iteration cap (AgentLoop.MAX_ITERATIONS,
    # default 20). High-fan-out workers (e.g. a translator emitting one PR per
    # locale) need more headroom than the default. None = inherit the global
    # OPENLEGION_MAX_ITERATIONS / hard-coded default. The agent-side reader
    # (_clamp_env in src/agent/loop.py) clamps to 1-100 regardless of source,
    # so an absurd value here can't blow the cap.
    max_iterations: int | None = Field(default=None, ge=1, le=100)

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
    """

    name: str
    description: str = ""
    members: list[str] = []
    created_at: str | None = None
    status: str = "active"
    settings: dict[str, Any] = {}
    north_star: str | None = None
    success_criteria: list[str] | None = None


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
    # Memory v2: provenance + recency for prefer-recent retrieval.
    source_type: str = "conversation"
    date: datetime | None = None


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
    # ``error_type`` (seam follow-up Fix 3): when the upstream call raised
    # a distinguished exception (``LLMAuthError`` / ``LLMConfigError``),
    # the mesh tags the response so the agent can route to the correct
    # loop branch instead of treating it as a generic RuntimeError. Values
    # are stable string keys: ``'auth_failure'`` / ``'config_error'``.
    # The mesh ALSO records the auth failure directly via HealthMonitor —
    # ``error_type`` is for client-side bookkeeping, the quarantine path
    # does not depend on the agent self-reporting.
    error_type: str | None = None
    error_meta: dict[str, Any] | None = None


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
        # OAuth connection refresh hard-failed (e.g. ``invalid_grant`` after
        # the user revoked access at the provider). Emitted once per failure
        # episode from the mesh ``/mesh/vault/resolve`` catch site;
        # ``runtime._system_signal_producer`` reroutes it into a ``/chat/note``
        # in the operator thread so the operator knows to reconnect via the
        # Connectors page.
        "connection_refresh_failed",
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
        # ``task_completed_without_handoff`` is emitted by
        # ``host/orchestration.py:update_status`` when a task transitions
        # to ``done`` and no child task references it via ``parent_task_id``
        # — i.e. the agent finished the work without handing off to a
        # successor. Observability-only signal for the dashboard "chain
        # break" surface; NO enforcement effect.
        "task_completed_without_handoff",
        # ``task_outcome`` is emitted by ``host/orchestration.py:set_outcome``
        # when an operator (or system) rates a delivered task. Without this
        # literal, ``DashboardEvent`` validation rejects the emit and the
        # event silently disappears (swallowed by ``_safe_emit``), so
        # Work-tab consumers never see deliveries.
        "task_outcome",
        "task_artifact_added",
        # Work summaries — emitted by ``host/summaries.WorkSummariesStore``
        # on create / set_rating. Drives the Summary cards on the Work
        # tab's new default landing (PR-B). Without these literals the
        # emits get rejected by Pydantic validation and the dashboard
        # silently misses the live update.
        "work_summary_created",
        "work_summary_rated",
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
        # the teams list refreshes without polling. The 5 legacy
        # ``project_*`` literals were dropped on 2026-05-18 (see
        # refactor/drop-dead-project-literals): PR 3 of the rename had
        # already stopped emitting them, and no consumer / dispatcher /
        # historical-record code in src/ or tests/ still references them.
        "team_created",
        "team_updated",
        "team_deleted",
        "team_archived",
        "team_unarchived",
        # Blackboard delete — mirror of ``blackboard_write`` for the
        # delete endpoint so the SPA can reflect removals live.
        "blackboard_delete",
        # Lane queue depth/busy changed — emitted by ``host/lanes.LaneManager``
        # on enqueue / dequeue / quarantine / terminal completion so the SPA
        # refreshes queue badges live instead of polling ``/api/queues`` every
        # 2s. Payload: ``{"agent": <agent_id>}``.
        "queue_changed",
        # System-tab config changed — single generic event emitted by the
        # dashboard settings mutation endpoints. ``data.scope`` discriminates
        # the panel (e.g. browser_settings / channels / integrations / webhooks
        # / api_keys / network_proxy / captcha_solver / system_settings /
        # storage / uploads / skills / wallet) so the SPA re-fetches just that
        # panel without polling.
        "config_changed",
    ]
    agent: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data: dict[str, Any] = {}
