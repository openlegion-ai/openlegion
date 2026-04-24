"""Pydantic models for all inter-component messages, events, and state.

This is THE contract between every component in OpenLegion.
Agent containers and the host process share only these types.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


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

# === Inter-Component Messages ===


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
    can_spawn: bool = False
    can_manage_cron: bool = False
    can_use_wallet: bool = False
    wallet_allowed_chains: list[str] = []
    wallet_spend_limit_per_tx_usd: float = 0.0
    wallet_spend_limit_daily_usd: float = 0.0
    wallet_rate_limit_per_hour: int = 0
    wallet_allowed_contracts: list[str] = []


# === Projects ===


class ProjectMetadata(BaseModel):
    """Project definition loaded from config/projects/<name>/metadata.yaml."""

    name: str
    description: str = ""
    members: list[str] = []
    created_at: str | None = None
    settings: dict[str, Any] = {}


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
        "credential_stored",
        "browser_login_request",
        "browser_login_completed",
        "browser_login_cancelled",
    ]
    agent: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data: dict[str, Any] = {}
