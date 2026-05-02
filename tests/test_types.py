"""Unit tests for shared Pydantic types."""

from src.shared.types import (
    AgentConfig,
    AgentMessage,
    AgentStatus,
    TaskAssignment,
    TaskResult,
    TokenBudget,
)


def test_agent_message_defaults():
    msg = AgentMessage(from_agent="a1", to="a2", type="task_request", payload={"x": 1})
    assert msg.id.startswith("msg_")
    assert msg.ttl == 300
    assert msg.priority == "normal"


def test_token_budget_can_spend():
    budget = TokenBudget(max_tokens=10_000, used_tokens=0)
    assert budget.can_spend(5_000)
    assert budget.can_spend(10_000)
    assert not budget.can_spend(10_001)


def test_token_budget_record_usage():
    budget = TokenBudget(max_tokens=100_000)
    budget.record_usage(1000, "anthropic/claude-sonnet-4-5-20250929")
    assert budget.used_tokens == 1000
    assert budget.estimated_cost_usd > 0


def test_task_assignment_defaults():
    ta = TaskAssignment(workflow_id="wf_1", step_id="s1", task_type="research", input_data={"q": "test"})
    assert ta.task_id.startswith("task_")
    assert ta.timeout == 120
    assert ta.context == {}


def test_task_result_serialization():
    tr = TaskResult(task_id="t1", status="complete", result={"key": "val"}, promote_to_blackboard={"ctx/a": "b"})
    d = tr.model_dump()
    assert d["status"] == "complete"
    assert d["promote_to_blackboard"] == {"ctx/a": "b"}


def test_agent_status_fields():
    status = AgentStatus(agent_id="a1", role="research", state="idle")
    assert status.tasks_completed == 0
    assert status.capabilities == []


def test_token_budget_record_usage_uses_unified_costs():
    """WU2: record_usage delegates to estimate_cost from costs.py (18+ models)."""
    budget = TokenBudget(max_tokens=1_000_000)
    # Test a model that was NOT in the old inline dict
    budget.record_usage(1000, "openai/gpt-4o")
    assert budget.estimated_cost_usd > 0
    # Test unknown model gets a reasonable fallback
    budget2 = TokenBudget(max_tokens=1_000_000)
    budget2.record_usage(1000, "unknown/model")
    assert budget2.estimated_cost_usd > 0


# ── Task 8: AgentConfig structured routing fields ──


def test_agent_config_defaults_empty():
    """All five Task-8 routing fields default cleanly so existing
    agents.yaml entries without them load unchanged."""
    cfg = AgentConfig(role="researcher", model="openai/gpt-4o-mini")
    assert cfg.capabilities == []
    assert cfg.preferred_inputs == []
    assert cfg.expected_outputs == []
    assert cfg.escalation_to is None
    assert cfg.forbidden == []


def test_agent_config_accepts_structured_fields():
    cfg = AgentConfig(
        role="pm",
        model="openai/gpt-4o-mini",
        capabilities=["Break down specs", "Coordinate handoffs"],
        preferred_inputs=["User requests"],
        expected_outputs=["Task specs"],
        escalation_to="operator",
        forbidden=["Writing code directly"],
    )
    assert cfg.capabilities == ["Break down specs", "Coordinate handoffs"]
    assert cfg.preferred_inputs == ["User requests"]
    assert cfg.expected_outputs == ["Task specs"]
    assert cfg.escalation_to == "operator"
    assert cfg.forbidden == ["Writing code directly"]


def test_agent_config_extra_fields_allowed():
    """Extra keys (e.g., legacy ``initial_interface``) round-trip via
    ``extra='allow'`` so loading isn't a strict-validation gate."""
    cfg = AgentConfig(role="pm", model="x", initial_interface="hello")
    assert cfg.initial_interface == "hello"
    cfg2 = AgentConfig(
        role="pm",
        model="x",
        capabilities=["a"],
        legacy_field="ignored-but-kept",  # type: ignore[call-arg]
    )
    dumped = cfg2.model_dump()
    assert dumped["legacy_field"] == "ignored-but-kept"
    assert dumped["capabilities"] == ["a"]


