"""Unit tests for the orchestrator: condition parser, DAG resolution, failure modes."""

import pytest

from src.host.orchestrator import (
    Orchestrator,
    WorkflowExecution,
    _parse_literal,
    _resolve_path,
    _safe_evaluate_condition,
)
from src.shared.types import TaskResult, WorkflowDefinition, WorkflowStep

# === Safe Condition Evaluator ===


def test_condition_greater_than():
    results = {"qualify": TaskResult(task_id="t1", status="complete", result={"score": 0.8})}
    assert _safe_evaluate_condition("qualify.result.score > 0.7", results)
    assert not _safe_evaluate_condition("qualify.result.score > 0.9", results)


def test_condition_equals():
    results = {"research": TaskResult(task_id="t1", status="complete")}
    assert _safe_evaluate_condition("research.status == 'complete'", results)
    assert not _safe_evaluate_condition("research.status == 'failed'", results)


def test_condition_not_equals():
    results = {"s1": TaskResult(task_id="t1", status="failed")}
    assert _safe_evaluate_condition("s1.status != 'complete'", results)


def test_condition_less_than_or_equal():
    results = {"s1": TaskResult(task_id="t1", status="complete", result={"count": 5})}
    assert _safe_evaluate_condition("s1.result.count <= 10", results)
    assert _safe_evaluate_condition("s1.result.count <= 5", results)
    assert not _safe_evaluate_condition("s1.result.count <= 4", results)


def test_condition_empty_returns_true():
    assert _safe_evaluate_condition("", {})
    assert _safe_evaluate_condition("  ", {})
    assert _safe_evaluate_condition(None, {})


def test_condition_invalid_syntax_returns_false():
    assert not _safe_evaluate_condition("import os; os.system('rm -rf /')", {})
    assert not _safe_evaluate_condition("eval('bad')", {})
    assert not _safe_evaluate_condition("__import__('os')", {})


def test_condition_nonexistent_path_returns_false():
    results = {"s1": TaskResult(task_id="t1", status="complete")}
    assert not _safe_evaluate_condition("s1.result.nonexistent > 0", results)


# === Parse Literal ===


def test_parse_literal_string():
    assert _parse_literal("'hello'") == "hello"
    assert _parse_literal('"world"') == "world"


def test_parse_literal_number():
    assert _parse_literal("42") == 42
    assert _parse_literal("3.14") == 3.14


def test_parse_literal_boolean():
    assert _parse_literal("true") is True
    assert _parse_literal("false") is False


# === Resolve Path ===


def test_resolve_path_dict():
    data = {"a": {"b": {"c": 42}}}
    assert _resolve_path("a.b.c", data) == 42


def test_resolve_path_task_result():
    results = {"s1": TaskResult(task_id="t1", status="complete", result={"score": 0.9})}
    assert _resolve_path("s1.status", results) == "complete"
    assert _resolve_path("s1.result.score", results) == 0.9


def test_resolve_path_missing():
    assert _resolve_path("a.b.c", {"a": {}}) is None


# === WorkflowExecution ===


def test_workflow_execution_step_ready():
    wf = WorkflowDefinition(
        name="test",
        trigger="test",
        steps=[
            WorkflowStep(id="s1", task_type="t1"),
            WorkflowStep(id="s2", task_type="t2", depends_on=["s1"]),
        ],
    )
    ex = WorkflowExecution(wf, {})

    s2 = wf.steps[1]
    assert not ex.is_step_ready(s2)

    ex.step_results["s1"] = TaskResult(task_id="t1", status="complete")
    assert ex.is_step_ready(s2)


def test_workflow_execution_step_not_ready_if_dep_failed():
    wf = WorkflowDefinition(
        name="test",
        trigger="test",
        steps=[
            WorkflowStep(id="s1", task_type="t1"),
            WorkflowStep(id="s2", task_type="t2", depends_on=["s1"]),
        ],
    )
    ex = WorkflowExecution(wf, {})
    ex.step_results["s1"] = TaskResult(task_id="t1", status="failed", error="boom")
    assert not ex.is_step_ready(wf.steps[1])


def test_workflow_execution_condition_evaluation():
    wf = WorkflowDefinition(
        name="test",
        trigger="test",
        steps=[WorkflowStep(id="s1", task_type="t1", condition="s0.result.score > 0.5")],
    )
    ex = WorkflowExecution(wf, {})
    ex.step_results["s0"] = TaskResult(task_id="t0", status="complete", result={"score": 0.8})
    assert ex.evaluate_condition(wf.steps[0])


# === Orchestrator Workflow Loading ===


def test_orchestrator_loads_workflows(tmp_path):
    wf_dir = tmp_path / "workflows"
    wf_dir.mkdir()
    (wf_dir / "test.yaml").write_text(
        "name: test_wf\ntrigger: test\nsteps:\n  - id: s1\n    task_type: t1\n"
    )
    orch = Orchestrator(mesh_url="http://localhost:8420", workflows_dir=str(wf_dir))
    assert "test_wf" in orch.workflows


@pytest.mark.asyncio
async def test_orchestrator_trigger_unknown_workflow():
    orch = Orchestrator(mesh_url="http://localhost:8420", workflows_dir="/nonexistent")
    with pytest.raises(ValueError, match="Unknown workflow"):
        await orch.trigger_workflow("nonexistent", {})


def test_orchestrator_resolve_input_trigger():
    orch = Orchestrator(mesh_url="http://localhost:8420", workflows_dir="/nonexistent")
    wf = WorkflowDefinition(
        name="test",
        trigger="test",
        steps=[WorkflowStep(id="s1", task_type="t1", input_from="trigger.payload")],
    )
    ex = WorkflowExecution(wf, {"company": "Acme"})
    result = orch._resolve_input(ex, wf.steps[0])
    assert result == {"company": "Acme"}


def test_orchestrator_resolve_input_previous_step():
    orch = Orchestrator(mesh_url="http://localhost:8420", workflows_dir="/nonexistent")
    wf = WorkflowDefinition(
        name="test",
        trigger="test",
        steps=[
            WorkflowStep(id="s1", task_type="t1"),
            WorkflowStep(id="s2", task_type="t2", input_from="s1"),
        ],
    )
    ex = WorkflowExecution(wf, {})
    ex.step_results["s1"] = TaskResult(task_id="t1", status="complete", result={"data": "found"})
    result = orch._resolve_input(ex, wf.steps[1])
    assert result == {"data": "found"}


def test_orchestrator_execution_status():
    orch = Orchestrator(mesh_url="http://localhost:8420", workflows_dir="/nonexistent")
    wf = WorkflowDefinition(name="test", trigger="test", steps=[])
    ex = WorkflowExecution(wf, {})
    orch.active_executions[ex.id] = ex

    status = orch.get_execution_status(ex.id)
    assert status is not None
    assert status["status"] == "running"
    assert status["workflow"] == "test"

    assert orch.get_execution_status("nonexistent") is None


# === Push-Based Task Resolution ===


def test_resolve_task_result():
    """resolve_task_result resolves a pending future."""
    import asyncio

    orch = Orchestrator(mesh_url="http://localhost:8420", workflows_dir="/nonexistent")
    loop = asyncio.new_event_loop()

    future = loop.create_future()
    orch._pending_results["task_1"] = future

    result = TaskResult(task_id="task_1", status="complete", result={"data": "ok"})
    assert orch.resolve_task_result("task_1", result) is True
    assert future.done()
    assert future.result() is result
    assert "task_1" not in orch._pending_results

    loop.close()


def test_resolve_unknown_task_id():
    """resolve_task_result returns False for unknown task IDs."""
    orch = Orchestrator(mesh_url="http://localhost:8420", workflows_dir="/nonexistent")
    result = TaskResult(task_id="unknown", status="complete")
    assert orch.resolve_task_result("unknown", result) is False


@pytest.mark.asyncio
async def test_execute_step_uses_future():
    """_execute_step creates a future and resolves via resolve_task_result."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    orch = Orchestrator(mesh_url="http://localhost:8420", workflows_dir="/nonexistent")
    orch.container_manager = MagicMock()
    orch.container_manager.get_agent_url = MagicMock(return_value="http://localhost:8401")

    # Mock the HTTP client to simulate agent accepting the task
    mock_client = AsyncMock()
    mock_client.is_closed = False
    mock_response = MagicMock()
    mock_response.json.return_value = {"accepted": True}
    mock_client.post = AsyncMock(return_value=mock_response)
    orch._client = mock_client

    wf = WorkflowDefinition(
        name="test", trigger="test",
        steps=[WorkflowStep(id="s1", task_type="research", agent="alpha")],
    )
    execution = WorkflowExecution(wf, {"query": "test"})

    # Run _execute_step in background; it will await the future
    step_task = asyncio.create_task(orch._execute_step(execution, wf.steps[0]))

    # Give the task a moment to register the future
    await asyncio.sleep(0.05)

    # Find the pending task_id and resolve it
    assert len(orch._pending_results) == 1
    task_id = next(iter(orch._pending_results))
    push_result = TaskResult(task_id=task_id, status="complete", result={"found": True})
    orch.resolve_task_result(task_id, push_result)

    result = await step_task
    assert result.status == "complete"
    assert result.result == {"found": True}


@pytest.mark.asyncio
async def test_execute_step_timeout():
    """_execute_step times out if no result is pushed and no polling fallback."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    orch = Orchestrator(mesh_url="http://localhost:8420", workflows_dir="/nonexistent")
    orch.container_manager = MagicMock()
    orch.container_manager.get_agent_url = MagicMock(return_value="http://localhost:8401")

    # Mock client
    mock_client = AsyncMock()
    mock_client.is_closed = False
    mock_response = MagicMock()
    mock_response.json.return_value = {"accepted": True}
    mock_client.post = AsyncMock(return_value=mock_response)
    # Mock polling to also timeout (return timeout result)
    mock_status_resp = MagicMock()
    mock_status_resp.json.return_value = {"state": "working"}
    mock_client.get = AsyncMock(return_value=mock_status_resp)
    orch._client = mock_client

    wf = WorkflowDefinition(
        name="test", trigger="test",
        steps=[WorkflowStep(id="s1", task_type="research", agent="alpha", timeout=1)],
    )
    execution = WorkflowExecution(wf, {})

    result = await orch._execute_step(execution, wf.steps[0])
    assert result.status == "timeout"
