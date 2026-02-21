"""In-container subagent spawning for parallel work.

Subagents are lightweight AgentLoop instances that run in the same process.
They share the parent's LLM client and mesh client (stateless httpx) but
get their own memory store (in-memory SQLite) and workspace directory.

Limits: max 3 concurrent per parent, max depth 2 (no grandchildren),
default TTL 300s, default 10 max iterations.

NOTE: Subagents should not use browser tools concurrently with the parent
because browser state is stored in module-level globals.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING

from src.agent.skills import SkillRegistry, _skill_staging, skill
from src.shared.utils import generate_id, setup_logging

if TYPE_CHECKING:
    from src.agent.llm import LLMClient

logger = setup_logging("agent.subagent")

# Module-level state
_active_subagents: dict[str, dict[str, asyncio.Task]] = {}  # parent_id -> {subagent_id -> task}
_depth_map: dict[str, int] = {}  # agent_id -> nesting depth
_parent_llm_refs: dict[str, LLMClient] = {}  # agent_id -> LLM reference

MAX_CONCURRENT = 3
MAX_DEPTH = 2
DEFAULT_TTL = 300
DEFAULT_MAX_ITERATIONS = 10

# Skills that subagents should NOT have (they could cause recursion or contention)
_UNSAFE_SKILLS = frozenset({"create_skill", "reload_skills", "spawn_subagent"})


def register_parent_llm(agent_id: str, llm: LLMClient) -> None:
    """Register the parent's LLM client for subagent reuse. Called at agent startup."""
    _parent_llm_refs[agent_id] = llm
    _depth_map.setdefault(agent_id, 0)


def _get_parent_llm(parent_id: str) -> LLMClient | None:
    """Retrieve the LLM client for a parent agent."""
    return _parent_llm_refs.get(parent_id)


def _get_depth(agent_id: str) -> int:
    return _depth_map.get(agent_id, 0)


def _set_depth(agent_id: str, depth: int) -> None:
    _depth_map[agent_id] = depth


def _cleanup_depth(agent_id: str) -> None:
    _depth_map.pop(agent_id, None)


def _clone_skill_registry(workspace_manager=None) -> SkillRegistry:
    """Create a cloned SkillRegistry from the current staging, removing unsafe skills.

    The clone uses ``__new__`` to skip module re-execution, copies the current
    skill dict, removes unsafe skills, and disables ``reload()``.
    """
    clone = SkillRegistry.__new__(SkillRegistry)
    clone.skills_dir = ""
    clone._mcp_client = None
    # Copy current staging (which has all discovered skills)
    clone.skills = {k: v for k, v in _skill_staging.items() if k not in _UNSAFE_SKILLS}
    # Disable reload on clones
    clone.reload = lambda: len(clone.skills)  # type: ignore[assignment]
    return clone


async def _run_subagent(
    parent_id: str,
    subagent_id: str,
    task_text: str,
    role: str,
    ttl_seconds: int,
    max_iterations: int,
    mesh_client,
) -> dict:
    """Run a subagent loop and return the result."""
    from src.agent.loop import AgentLoop
    from src.agent.memory import MemoryStore
    from src.agent.workspace import WorkspaceManager

    llm = _get_parent_llm(parent_id)
    if llm is None:
        return {"error": "Parent LLM not found"}

    # Each subagent gets isolated memory and workspace
    memory = MemoryStore(db_path=":memory:")
    ws_dir = f"/data/workspace/subagents/{subagent_id}"
    workspace = WorkspaceManager(workspace_dir=ws_dir)
    skills = _clone_skill_registry()

    _set_depth(subagent_id, _get_depth(parent_id) + 1)

    loop = AgentLoop(
        agent_id=subagent_id,
        role=role,
        memory=memory,
        skills=skills,
        llm=llm,
        mesh_client=mesh_client,
        system_prompt=f"You are a subagent. Your task: {task_text}",
        workspace=workspace,
    )
    loop.MAX_ITERATIONS = max_iterations

    from src.shared.types import TaskAssignment
    assignment = TaskAssignment(
        task_id=subagent_id,
        workflow_id=f"subagent_{parent_id}",
        step_id=subagent_id,
        task_type="subagent",
        input_data={"task": task_text},
    )

    try:
        result = await asyncio.wait_for(
            loop.execute_task(assignment),
            timeout=ttl_seconds,
        )
        result_data = {
            "status": result.status,
            "result": result.result,
            "tokens_used": result.tokens_used,
            "duration_ms": result.duration_ms,
        }
    except asyncio.TimeoutError:
        result_data = {
            "status": "timeout",
            "result": f"Subagent timed out after {ttl_seconds}s",
            "tokens_used": 0,
            "iterations": 0,
        }
    except Exception as e:
        result_data = {
            "status": "error",
            "result": str(e),
            "tokens_used": 0,
            "iterations": 0,
        }
    finally:
        _cleanup_depth(subagent_id)
        memory.close()

    # Write result to blackboard
    result_key = f"subagent_results/{parent_id}/{subagent_id}"
    try:
        await mesh_client.write_blackboard(result_key, result_data)
    except Exception as e:
        logger.warning("Failed to write subagent result to blackboard: %s", e)

    return result_data


@skill(
    name="spawn_subagent",
    description=(
        "Spawn a lightweight subagent to handle a subtask in parallel. "
        "The subagent runs in the same container with its own memory and workspace. "
        "Results are written to the blackboard at subagent_results/<your_id>/<subagent_id>. "
        "Use read_shared_state to check results. Max 3 concurrent, max depth 2. "
        "NOTE: Subagents should not use browser tools (shared browser state)."
    ),
    parameters={
        "task": {
            "type": "string",
            "description": "Task description for the subagent",
        },
        "role": {
            "type": "string",
            "description": "Role for the subagent (default 'assistant')",
            "default": "assistant",
        },
        "ttl_seconds": {
            "type": "integer",
            "description": "Max time in seconds before timeout (default 300)",
            "default": 300,
        },
    },
)
async def spawn_subagent(
    task: str,
    role: str = "assistant",
    ttl_seconds: int = DEFAULT_TTL,
    *,
    mesh_client=None,
) -> dict:
    """Spawn a background subagent for parallel work."""
    if mesh_client is None:
        return {"error": "No mesh_client available"}

    # Determine parent from mesh_client
    parent_id = getattr(mesh_client, "agent_id", "unknown")

    # Depth check
    depth = _get_depth(parent_id)
    if depth >= MAX_DEPTH:
        return {"error": f"Max subagent depth ({MAX_DEPTH}) reached. Cannot spawn deeper."}

    # Concurrent check
    parent_tasks = _active_subagents.get(parent_id, {})
    active_count = sum(1 for t in parent_tasks.values() if not t.done())
    if active_count >= MAX_CONCURRENT:
        return {"error": f"Max concurrent subagents ({MAX_CONCURRENT}) reached. Wait for one to finish."}

    subagent_id = f"sub_{generate_id()}"
    result_key = f"subagent_results/{parent_id}/{subagent_id}"

    # Launch as background task
    coro = _run_subagent(
        parent_id=parent_id,
        subagent_id=subagent_id,
        task_text=task,
        role=role,
        ttl_seconds=ttl_seconds,
        max_iterations=DEFAULT_MAX_ITERATIONS,
        mesh_client=mesh_client,
    )
    async_task = asyncio.create_task(coro)

    if parent_id not in _active_subagents:
        _active_subagents[parent_id] = {}
    _active_subagents[parent_id][subagent_id] = async_task

    logger.info("Spawned subagent %s for parent %s: %s", subagent_id, parent_id, task[:80])
    return {
        "spawned": True,
        "subagent_id": subagent_id,
        "result_key": result_key,
    }


@skill(
    name="list_subagents",
    description="List active subagents spawned by this agent and their status.",
    parameters={},
)
async def list_subagents(*, mesh_client=None) -> dict:
    """List active subagents for the current agent."""
    parent_id = getattr(mesh_client, "agent_id", "unknown") if mesh_client else "unknown"
    parent_tasks = _active_subagents.get(parent_id, {})

    subagents = []
    for sid, task in parent_tasks.items():
        subagents.append({
            "subagent_id": sid,
            "done": task.done(),
            "result_key": f"subagent_results/{parent_id}/{sid}",
        })

    return {
        "parent_id": parent_id,
        "count": len(subagents),
        "active": sum(1 for s in subagents if not s["done"]),
        "subagents": subagents,
    }
