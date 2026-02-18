"""DAG executor and workflow engine.

Parses workflow YAML files and executes them as directed acyclic graphs.
Each step is a task assignment sent to an agent via HTTP.

Patterns supported:
  1. Pipeline (sequential): A -> B -> C
  2. Fan-out/fan-in (parallel): A -> [B, C, D] -> E
  3. Conditional: A -> B (if condition) else skip
  4. Failure handling: retry, skip, abort, fallback

Conditions use a SAFE regex-based parser -- NO eval().
"""

from __future__ import annotations

import asyncio
import operator
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import httpx
import yaml

from src.shared.types import TaskAssignment, TaskResult, WorkflowDefinition, WorkflowStep
from src.shared.utils import generate_id, setup_logging

if TYPE_CHECKING:
    from src.host.containers import ContainerManager
    from src.host.mesh import Blackboard, PubSub

logger = setup_logging("host.orchestrator")


# === SAFE CONDITION EVALUATOR (no eval!) ===

CONDITION_OPS = {
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}


def _safe_evaluate_condition(condition: str, step_results: dict) -> bool:
    """Safe condition evaluation. Only supports: step_id.result.key <op> value.

    Examples: "qualify.result.score > 0.7", "research.status == 'complete'"
    NO eval(), NO exec(), NO arbitrary Python.
    """
    if not condition or not condition.strip():
        return True
    pattern = r"^([\w.]+)\s*(>=|<=|!=|==|>|<)\s*(.+)$"
    match = re.match(pattern, condition.strip())
    if not match:
        logger.warning(f"Invalid condition syntax: {condition}")
        return False
    left_path, op_str, right_raw = match.groups()
    left_value = _resolve_path(left_path, step_results)
    right_value = _parse_literal(right_raw.strip())
    try:
        return CONDITION_OPS[op_str](left_value, right_value)
    except (TypeError, KeyError):
        return False


def _resolve_path(path: str, data: dict):
    """Resolve a dot-separated path against step results."""
    parts = path.split(".")
    current = data
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        elif hasattr(current, part):
            current = getattr(current, part)
        elif hasattr(current, "result") and part != "result":
            current = (current.result or {}).get(part)
        else:
            return None
    return current


def _parse_literal(raw: str):
    """Parse a string literal, boolean, or number from a condition RHS."""
    if (raw.startswith("'") and raw.endswith("'")) or (raw.startswith('"') and raw.endswith('"')):
        return raw[1:-1]
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    try:
        return float(raw) if "." in raw else int(raw)
    except ValueError:
        return raw


class WorkflowExecution:
    """Tracks state of a running workflow."""

    def __init__(self, workflow: WorkflowDefinition, trigger_payload: dict):
        self.id = generate_id("wf")
        self.workflow = workflow
        self.trigger_payload = trigger_payload
        self.step_results: dict[str, TaskResult] = {}
        self.status: str = "running"
        self.started_at: float = time.time()

    def is_step_ready(self, step: WorkflowStep) -> bool:
        """Check if all dependencies for a step are satisfied."""
        for dep_id in step.depends_on:
            if dep_id not in self.step_results:
                return False
            if self.step_results[dep_id].status != "complete":
                return False
        return True

    def evaluate_condition(self, step: WorkflowStep) -> bool:
        """Evaluate a step's condition using SAFE expression parser."""
        if not step.condition:
            return True
        return _safe_evaluate_condition(step.condition, self.step_results)


class Orchestrator:
    """Workflow engine that executes multi-step DAGs across agents."""

    def __init__(
        self,
        mesh_url: str,
        workflows_dir: str = "config/workflows",
        blackboard: Blackboard | None = None,
        pubsub: PubSub | None = None,
        container_manager: ContainerManager | None = None,
    ):
        self.mesh_url = mesh_url
        self.workflows: dict[str, WorkflowDefinition] = {}
        self.active_executions: dict[str, WorkflowExecution] = {}
        self.blackboard = blackboard
        self.pubsub = pubsub
        self.container_manager = container_manager
        self._load_workflows(workflows_dir)

    def _load_workflows(self, workflows_dir: str) -> None:
        """Load all workflow YAML files."""
        wf_path = Path(workflows_dir)
        if not wf_path.exists():
            logger.warning(f"Workflows directory not found: {workflows_dir}")
            return
        for yaml_file in wf_path.glob("*.yaml"):
            try:
                with open(yaml_file) as f:
                    data = yaml.safe_load(f)
                wf = WorkflowDefinition(**data)
                self.workflows[wf.name] = wf
                logger.info(f"Loaded workflow: {wf.name}")
            except Exception as e:
                logger.error(f"Failed to load workflow {yaml_file}: {e}")

    async def trigger_workflow(self, workflow_name: str, payload: dict) -> str:
        """Start a workflow execution. Returns workflow execution ID."""
        if workflow_name not in self.workflows:
            raise ValueError(f"Unknown workflow: {workflow_name}")

        wf = self.workflows[workflow_name]
        execution = WorkflowExecution(wf, payload)
        self.active_executions[execution.id] = execution

        task = asyncio.create_task(self._run_workflow(execution))
        execution._task = task

        return execution.id

    async def _run_workflow(self, execution: WorkflowExecution) -> None:
        """Execute a workflow DAG to completion."""
        wf = execution.workflow
        pending_steps = {step.id: step for step in wf.steps}

        try:
            while pending_steps and execution.status == "running":
                elapsed = time.time() - execution.started_at
                if elapsed > wf.timeout:
                    execution.status = "failed"
                    logger.error(f"Workflow {execution.id} timed out after {elapsed:.0f}s")
                    break

                ready = [step for step in pending_steps.values() if execution.is_step_ready(step)]

                if not ready:
                    if all(sid in execution.step_results for sid in pending_steps):
                        break
                    await asyncio.sleep(0.5)
                    continue

                tasks = []
                steps_for_tasks = []
                for step in ready:
                    if execution.evaluate_condition(step):
                        tasks.append(self._execute_step(execution, step))
                        steps_for_tasks.append(step)
                    else:
                        execution.step_results[step.id] = TaskResult(
                            task_id=f"skipped_{step.id}",
                            status="complete",
                            result={"skipped": True, "reason": "condition_not_met"},
                        )
                        del pending_steps[step.id]

                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    for step, result in zip(steps_for_tasks, results):
                        if isinstance(result, Exception):
                            execution.step_results[step.id] = TaskResult(
                                task_id=f"error_{step.id}",
                                status="failed",
                                error=str(result),
                            )
                        else:
                            execution.step_results[step.id] = result

                        if execution.step_results[step.id].status != "complete":
                            await self._handle_failure(execution, step)
                        elif self.blackboard:
                            promotions = execution.step_results[step.id].promote_to_blackboard
                            for key, value in promotions.items():
                                self.blackboard.write(
                                    key=key,
                                    value=value if isinstance(value, dict) else {"value": value},
                                    written_by=step.agent or "orchestrator",
                                    workflow_id=execution.id,
                                )

                        if step.id in pending_steps:
                            del pending_steps[step.id]

            if execution.status == "running":
                execution.status = "complete"
                if wf.on_complete and self.pubsub:
                    await self._publish_event(
                        wf.on_complete,
                        {
                            "workflow_id": execution.id,
                            "results": {sid: r.model_dump(mode="json") for sid, r in execution.step_results.items()},
                        },
                    )

        except Exception as e:
            execution.status = "failed"
            logger.error(f"Workflow {execution.id} failed: {e}", exc_info=True)

    async def _execute_step(self, execution: WorkflowExecution, step: WorkflowStep) -> TaskResult:
        """Execute a single workflow step by sending task to agent."""
        input_data = self._resolve_input(execution, step)

        context = {}
        if self.blackboard:
            entries = self.blackboard.list_by_prefix("context/")
            for entry in entries:
                context[entry.key] = entry.value

        assignment = TaskAssignment(
            workflow_id=execution.id,
            step_id=step.id,
            task_type=step.task_type,
            input_data=input_data,
            context=context,
            timeout=step.timeout,
            max_retries=step.max_retries,
        )

        agent_id = step.agent or step.capability
        agent_url = None
        if self.container_manager and agent_id:
            agent_url = self.container_manager.get_agent_url(agent_id)

        if not agent_url:
            return TaskResult(
                task_id=assignment.task_id,
                status="failed",
                error=f"No agent available: {agent_id}",
            )

        try:
            async with httpx.AsyncClient(timeout=step.timeout) as client:
                response = await client.post(
                    f"{agent_url}/task",
                    json=assignment.model_dump(mode="json"),
                )
                resp_data = response.json()

            if resp_data.get("accepted"):
                # Agent accepted -- poll for result via mesh message
                # For now, wait for the task to complete via status polling
                return await self._wait_for_task_result(agent_url, assignment, step.timeout)
            else:
                return TaskResult(
                    task_id=assignment.task_id,
                    status="failed",
                    error=resp_data.get("error", "Agent rejected task"),
                )
        except Exception as e:
            return TaskResult(
                task_id=assignment.task_id,
                status="failed",
                error=str(e),
            )

    async def _wait_for_task_result(self, agent_url: str, assignment: TaskAssignment, timeout: int) -> TaskResult:
        """Poll agent status until task completes, then fetch the real result."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    response = await client.get(f"{agent_url}/status")
                    status = response.json()
                    agent_state = status.get("state")
                    if agent_state != "working":
                        result_resp = await client.get(f"{agent_url}/result")
                        if result_resp.status_code == 200:
                            result_data = result_resp.json()
                            logger.info(f"Got result from agent: status={result_data.get('status')}")
                            return TaskResult(**result_data)
                        logger.warning(
                            f"Agent state={agent_state} but /result returned {result_resp.status_code}"
                        )
                        return TaskResult(
                            task_id=assignment.task_id,
                            status="failed",
                            error=f"Agent state={agent_state}, no result (HTTP {result_resp.status_code})",
                        )
            except Exception as e:
                logger.warning(f"Polling agent at {agent_url}: {e}")
            await asyncio.sleep(1)

        return TaskResult(
            task_id=assignment.task_id,
            status="timeout",
            error=f"Task timed out after {timeout}s",
        )

    def _resolve_input(self, execution: WorkflowExecution, step: WorkflowStep) -> dict:
        """Resolve the input data for a step."""
        if not step.input_from:
            return execution.trigger_payload

        if step.input_from == "trigger.payload":
            return execution.trigger_payload

        if step.input_from.startswith("blackboard://"):
            key = step.input_from.replace("blackboard://", "")
            for var_name, var_value in execution.trigger_payload.items():
                key = key.replace(f"{{{var_name}}}", str(var_value))
            if self.blackboard:
                entry = self.blackboard.read(key)
                return entry.value if entry else {}
            return {}

        if step.input_from in execution.step_results:
            result = execution.step_results[step.input_from]
            return result.result or {}

        return execution.trigger_payload

    async def _handle_failure(self, execution: WorkflowExecution, step: WorkflowStep) -> None:
        """Handle a failed step according to its failure policy."""
        result = execution.step_results[step.id]

        if step.on_failure == "retry" and step.max_retries > 0:
            for attempt in range(step.max_retries):
                await asyncio.sleep(2**attempt)
                retry_result = await self._execute_step(execution, step)
                if retry_result.status == "complete":
                    execution.step_results[step.id] = retry_result
                    return
            # All retries exhausted -- abort the workflow
            logger.error(f"Step {step.id} failed after {step.max_retries} retries")
            execution.status = "failed"

        elif step.on_failure == "skip":
            execution.step_results[step.id] = TaskResult(
                task_id=result.task_id,
                status="complete",
                result={"skipped": True, "reason": "step_failed"},
            )

        elif step.on_failure == "abort":
            execution.status = "failed"

        elif step.on_failure == "fallback" and step.fallback_agent:
            fallback_step = step.model_copy(update={"agent": step.fallback_agent})
            fallback_result = await self._execute_step(execution, fallback_step)
            execution.step_results[step.id] = fallback_result

    async def _publish_event(self, topic: str, payload: dict) -> None:
        """Publish an event via the mesh pub/sub."""
        if not self.pubsub:
            return
        from src.shared.types import MeshEvent

        event = MeshEvent(topic=topic, source="orchestrator", payload=payload)
        self.pubsub.publish(topic, event)

    def get_execution_status(self, execution_id: str) -> Optional[dict]:
        """Get the status of a workflow execution."""
        if execution_id not in self.active_executions:
            return None
        ex = self.active_executions[execution_id]
        return {
            "id": ex.id,
            "workflow": ex.workflow.name,
            "status": ex.status,
            "steps_completed": len(ex.step_results),
            "steps_total": len(ex.workflow.steps),
            "elapsed_seconds": time.time() - ex.started_at,
        }
