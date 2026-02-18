"""Pydantic models for all inter-component messages, events, and state.

This is THE contract between every component in OpenLegion.
Agent containers and the host process share only these types.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

# === Inter-Component Messages ===


class AgentMessage(BaseModel):
    """Every message between agents passes through the mesh in this format."""

    id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:12]}")
    from_agent: str
    to: str
    type: Literal["task_request", "task_result", "event", "query", "cancel"]
    payload: dict[str, Any]
    workflow_id: Optional[str] = None
    reply_to: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    ttl: int = 300
    priority: Literal["low", "normal", "high", "urgent"] = "normal"


class TokenBudget(BaseModel):
    """Token budget for a task or workflow. Prevents runaway API spend."""

    max_tokens: int = 500_000
    used_tokens: int = 0
    max_cost_usd: float = 5.0
    estimated_cost_usd: float = 0.0

    def can_spend(self, estimated_tokens: int) -> bool:
        return (self.used_tokens + estimated_tokens) <= self.max_tokens

    def record_usage(self, tokens: int, model: str = "") -> None:
        self.used_tokens += tokens
        cost_per_1k = {
            "anthropic/claude-sonnet-4-5-20250929": 0.003,
            "anthropic/claude-haiku-4-5-20251001": 0.0008,
            "text-embedding-3-small": 0.00002,
        }
        rate = cost_per_1k.get(model, 0.003)
        self.estimated_cost_usd += (tokens / 1000) * rate


class TaskAssignment(BaseModel):
    """Sent by orchestrator to an agent to begin work."""

    task_id: str = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:12]}")
    workflow_id: str
    step_id: str
    task_type: str
    input_data: dict[str, Any]
    context: dict[str, Any] = {}
    timeout: int = 120
    max_retries: int = 0
    token_budget: Optional[TokenBudget] = None


class TaskResult(BaseModel):
    """Returned by an agent when a task completes or fails."""

    task_id: str
    status: Literal["complete", "failed", "cancelled", "timeout"]
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    promote_to_blackboard: dict[str, Any] = {}
    tokens_used: int = 0
    duration_ms: int = 0


class AgentStatus(BaseModel):
    """Returned by agent health check endpoint."""

    agent_id: str
    role: str
    state: Literal["idle", "working", "blocked", "failed", "starting"]
    current_task: Optional[str] = None
    capabilities: list[str] = []
    uptime_seconds: float = 0
    tasks_completed: int = 0
    tasks_failed: int = 0


# === Blackboard & Events ===


class BlackboardEntry(BaseModel):
    """A single entry in the shared blackboard."""

    key: str
    value: dict[str, Any]
    written_by: str
    workflow_id: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    ttl: Optional[int] = None
    version: int = 1


class MeshEvent(BaseModel):
    """Published to topics via pub/sub."""

    id: str = Field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:12]}")
    topic: str
    source: str
    payload: dict[str, Any] = {}
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# === Workflow Definitions (parsed from YAML) ===


class WorkflowStep(BaseModel):
    """A single step in a workflow."""

    id: str
    agent: Optional[str] = None
    capability: Optional[str] = None
    task_type: str
    input_from: Optional[str] = None
    depends_on: list[str] = []
    timeout: int = 120
    condition: Optional[str] = None
    on_failure: Literal["retry", "skip", "abort", "fallback"] = "abort"
    max_retries: int = 3
    fallback_agent: Optional[str] = None
    requires_approval: bool = False
    approval_timeout: int = 3600


class WorkflowDefinition(BaseModel):
    """A complete workflow parsed from YAML."""

    name: str
    trigger: str
    steps: list[WorkflowStep]
    timeout: int = 600
    on_complete: Optional[str] = None


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


# === Memory (inside agent container) ===


class MemoryFact(BaseModel):
    """A single fact in agent's private memory."""

    id: str = Field(default_factory=lambda: f"fact_{uuid.uuid4().hex[:8]}")
    key: str
    value: str
    category: str = "general"
    source: str = "agent"
    confidence: float = 1.0
    embedding: Optional[list[float]] = None
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    decay_score: float = 1.0


class MemoryLog(BaseModel):
    """An entry in the agent's action log."""

    id: str = Field(default_factory=lambda: f"log_{uuid.uuid4().hex[:8]}")
    action: str
    input_summary: str
    output_summary: str
    task_id: Optional[str] = None
    tokens_used: int = 0
    duration_ms: int = 0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


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
    data: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    status_code: Optional[int] = None


# === LLM Response (standardized across providers) ===


class ToolCallInfo(BaseModel):
    """A tool/function call requested by the LLM."""

    name: str
    arguments: dict[str, Any]


class LLMResponse(BaseModel):
    """Standardized response from any LLM provider via mesh proxy."""

    content: str = ""
    tool_calls: Optional[list[ToolCallInfo]] = None
    tokens_used: int = 0
    model: str = ""
    stop_reason: Optional[str] = None


# === Chat Mode ===


class ChatMessage(BaseModel):
    """Incoming chat message from user to agent."""

    message: str


class ChatResponse(BaseModel):
    """Agent's response to a chat message."""

    response: str
    tool_outputs: list[dict[str, Any]] = []
    tokens_used: int = 0
