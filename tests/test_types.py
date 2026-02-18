"""Unit tests for shared Pydantic types."""

from src.shared.types import (
    AgentMessage,
    AgentStatus,
    TaskAssignment,
    TaskResult,
    TokenBudget,
    WorkflowDefinition,
    WorkflowStep,
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


def test_workflow_step_defaults():
    step = WorkflowStep(id="s1", task_type="research")
    assert step.on_failure == "abort"
    assert step.max_retries == 3
    assert step.depends_on == []


def test_workflow_definition_parsing():
    wf = WorkflowDefinition(
        name="test",
        trigger="webhook",
        steps=[WorkflowStep(id="s1", task_type="research")],
    )
    assert len(wf.steps) == 1
    assert wf.timeout == 600
