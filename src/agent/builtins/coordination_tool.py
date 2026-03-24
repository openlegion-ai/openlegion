"""Coordination tools: structured multi-agent handoffs and status.

Higher-level wrappers over the blackboard that encode the standard
coordination protocol.  Agents use these instead of raw blackboard
operations for inter-agent work handoffs.
"""

from __future__ import annotations

import json
import re
import time

from src.agent.skills import skill
from src.shared.types import AGENT_ID_RE_PATTERN
from src.shared.utils import generate_id, sanitize_for_prompt, setup_logging

logger = setup_logging("agent.builtins.coordination_tool")

# Keys follow the standard protocol sections:
#   status/{agent_id}         — current state
#   output/{agent_id}/{name}  — completed work products
#   tasks/{agent_id}/{id}     — inbox (pending work)

_STANDALONE_ERROR = (
    "Coordination tools are not available — this agent is not assigned to "
    "any project. Use memory_save/memory_search for private storage."
)

_HANDOFF_TTL = 86_400  # 24 hours — safety net for unprocessed handoffs


@skill(
    name="hand_off",
    description=(
        "Hand off work to a teammate. Creates a task in their inbox and "
        "wakes them up automatically. This is your PRIMARY coordination "
        "tool — use it whenever you've completed work that another agent "
        "should act on.\n\n"
        "If you have output data to share, pass it as 'data' (JSON string). "
        "It will be written to the blackboard and the task will include a "
        "pointer so the recipient can read it. For lightweight handoffs "
        "where the summary is enough context, omit 'data'.\n\n"
        "The target agent sees the task in their inbox (via check_inbox) "
        "with your summary and a pointer to your output."
    ),
    parameters={
        "to": {
            "type": "string",
            "description": "Agent ID to hand off to (use list_agents to discover IDs)",
        },
        "summary": {
            "type": "string",
            "description": (
                "What you did and what they should do next. Be specific — "
                "this is the main context the recipient sees."
            ),
        },
        "data": {
            "type": "string",
            "description": (
                "Optional JSON string of output data to share. Written to "
                "the blackboard so the recipient can read the full details. "
                "Omit for lightweight handoffs where the summary is enough."
            ),
            "default": "",
        },
    },
)
async def hand_off(
    to: str, summary: str, data: str = "", *, mesh_client=None,
) -> dict:
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    if mesh_client.is_standalone:
        return {"error": _STANDALONE_ERROR}

    # Validate target agent ID format (defense-in-depth if list_agents fails)
    if not re.fullmatch(AGENT_ID_RE_PATTERN, to):
        return {"error": f"Invalid agent ID: '{to}'"}

    # Validate target agent exists
    try:
        registry = await mesh_client.list_agents()
        if to not in registry:
            available = ", ".join(sorted(registry.keys()))
            return {"error": f"Agent '{to}' not found. Available: {available}"}
    except Exception as e:
        logger.debug("Fleet roster check failed, proceeding with validated ID: %s", e)

    handoff_id = generate_id("ho")
    from_agent = mesh_client.agent_id
    summary = sanitize_for_prompt(summary)
    output_key = None

    # Write output data if provided
    if data and data.strip():
        try:
            parsed_data = json.loads(data)
        except json.JSONDecodeError:
            parsed_data = {"text": data}
        output_key = f"output/{from_agent}/{handoff_id}"
        try:
            await mesh_client.write_blackboard(output_key, parsed_data, ttl=_HANDOFF_TTL)
        except Exception as e:
            return {"error": f"Failed to write output: {e}"}

    # Create task in recipient's inbox
    task_record = {
        "from": from_agent,
        "summary": summary,
        "status": "pending",
        "ts": time.time(),
    }
    if output_key:
        task_record["output_key"] = output_key

    task_key = f"tasks/{to}/{handoff_id}"
    try:
        await mesh_client.write_blackboard(task_key, task_record, ttl=_HANDOFF_TTL)
    except Exception as e:
        # Clean up orphaned output if task write fails
        if output_key:
            logger.warning("Task write failed, orphaned output at %s", output_key)
        return {"error": f"Failed to create task: {e}"}

    result = {
        "handed_off": True,
        "to": to,
        "task_key": task_key,
        "handoff_id": handoff_id,
    }
    if output_key:
        result["output_key"] = output_key
    return result


@skill(
    name="check_inbox",
    description=(
        "Check your task inbox for pending work from teammates. Returns "
        "a list of tasks with who sent them, a summary of what to do, "
        "and a pointer to their output data on the blackboard.\n\n"
        "Call this:\n"
        "- At the start of a session\n"
        "- During heartbeats\n"
        "- When you receive a coordination notification\n\n"
        "After reading a task, use read_shared_state to fetch the full "
        "output data via the output_key. When done processing, call "
        "complete_task to mark it finished so it won't appear again."
    ),
    parameters={},
)
async def check_inbox(*, mesh_client=None) -> dict:
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    if mesh_client.is_standalone:
        return {"error": _STANDALONE_ERROR}

    agent_id = mesh_client.agent_id
    prefix = f"tasks/{agent_id}/"

    try:
        entries = await mesh_client.list_blackboard(prefix)
    except Exception as e:
        return {"error": f"Failed to check inbox: {e}"}

    tasks = []
    for entry in entries:
        value = entry.get("value", {})
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                value = {"text": value}
        # Ensure value is a dict (guard against corrupted entries)
        if not isinstance(value, dict):
            value = {"text": str(value)}
        # Skip completed tasks
        if value.get("status") == "done":
            continue
        task = {
            "key": sanitize_for_prompt(entry.get("key", "")),
            "from": sanitize_for_prompt(str(value.get("from", "unknown"))),
            "summary": sanitize_for_prompt(str(value.get("summary", ""))),
            "status": value.get("status", "pending"),
        }
        output_key = value.get("output_key")
        if output_key:
            task["output_key"] = sanitize_for_prompt(str(output_key))
        ts = value.get("ts")
        if ts:
            task["ts"] = ts
        tasks.append(task)

    return {"tasks": tasks, "count": len(tasks)}


@skill(
    name="update_status",
    description=(
        "Update your status on the blackboard so teammates know what "
        "you're doing. Call this when you start work, finish work, or "
        "get blocked. Teammates can see your status to decide whether "
        "to wait or proceed."
    ),
    parameters={
        "state": {
            "type": "string",
            "enum": ["idle", "working", "blocked", "done"],
            "description": "Your current state",
        },
        "summary": {
            "type": "string",
            "description": "Brief description of current activity or blocker",
            "default": "",
        },
    },
)
async def update_status(
    state: str, summary: str = "", *, mesh_client=None,
) -> dict:
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    if mesh_client.is_standalone:
        return {"error": _STANDALONE_ERROR}

    agent_id = mesh_client.agent_id
    status_data = {
        "state": state,
        "summary": sanitize_for_prompt(summary),
        "ts": time.time(),
    }

    try:
        await mesh_client.write_blackboard(f"status/{agent_id}", status_data)
    except Exception as e:
        return {"error": f"Failed to update status: {e}"}

    return {"updated": True, "state": state}


@skill(
    name="complete_task",
    description=(
        "Mark a task from your inbox as done so it won't appear in "
        "check_inbox() again. Call this after you've finished processing "
        "a task. Pass the task key from the check_inbox() result."
    ),
    parameters={
        "task_key": {
            "type": "string",
            "description": "The task key from check_inbox() (e.g. 'tasks/you/ho_abc123')",
        },
    },
)
async def complete_task(task_key: str, *, mesh_client=None) -> dict:
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    if mesh_client.is_standalone:
        return {"error": _STANDALONE_ERROR}

    # Ownership check — only complete tasks in your own inbox
    agent_id = mesh_client.agent_id
    expected_prefix = f"tasks/{agent_id}/"
    if not task_key.startswith(expected_prefix):
        return {"error": f"Can only complete your own tasks (expected prefix: tasks/{agent_id}/)"}

    try:
        # Read task to find associated output for cleanup
        existing = await mesh_client.read_blackboard(task_key)
        if existing is None:
            return {"error": f"Task '{task_key}' not found"}

        # Delete the task entry
        await mesh_client.delete_blackboard(task_key)

        # Best-effort cleanup of associated output data
        value = existing.get("value", existing)
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                value = {}
        if isinstance(value, dict):
            output_key = value.get("output_key")
            if output_key:
                try:
                    await mesh_client.delete_blackboard(output_key)
                except Exception as exc:
                    logger.debug("Output cleanup for %s skipped: %s", output_key, exc)
    except Exception as e:
        return {"error": f"Failed to complete task: {e}"}

    return {"completed": True, "task_key": task_key}
