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
    "Not available — this agent is not assigned to a project. "
    "Use memory_save/memory_search for private storage."
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

    # Validate target agent ID format (defense-in-depth if list_agents fails)
    if not re.fullmatch(AGENT_ID_RE_PATTERN, to):
        return {"error": f"Invalid agent ID: '{to}'"}

    # Resolve target agent's project for cross-project coordination, and
    # detect whether it advertises a fleet-global scope (operator-style).
    target_project: str | None = None
    target_is_global = False
    try:
        registry = await mesh_client.list_agents()
        if to not in registry:
            available = ", ".join(sorted(registry.keys()))
            return {"error": f"Agent '{to}' not found. Available: {available}"}
        target_info = registry.get(to, {})
        if isinstance(target_info, dict):
            target_project = target_info.get("project")
            target_is_global = target_info.get("scope") == "global"
    except Exception as e:
        # Standalone senders MUST resolve the target project to write to
        # the correct namespace.  Fail closed rather than writing to the
        # global scope where the recipient will never find the task.
        if not mesh_client.project_name:
            return {"error": f"Cannot hand off: fleet roster unavailable ({e})"}
        logger.debug("Fleet roster check failed, proceeding with validated ID: %s", e)

    # Determine which project scope to use for blackboard writes.
    # Fleet-global agents (operator) need their inbox in a global namespace
    # because they have no project scope to read from. We trigger on the
    # literal reserved name AND on the registry hint, so the path stays
    # correct even when the registry roster lookup fails (no hint available)
    # and forward-compatible if other global agents are added later.
    write_project: str | None = None  # None = use sender's default scope
    write_global = False
    if to == "operator" or target_is_global:
        write_global = True
    elif target_project and target_project != mesh_client.project_name:
        write_project = target_project

    handoff_id = generate_id("ho")
    from_agent = mesh_client.agent_id
    summary = sanitize_for_prompt(summary)
    output_key = None

    # Propagate origin so the target agent's lane can auto-notify the
    # originating channel user when the handed-off task completes.
    from src.shared.trace import current_origin as _current_origin
    origin = _current_origin.get()

    # Write output data if provided
    if data and data.strip():
        try:
            parsed_data = json.loads(data)
        except json.JSONDecodeError:
            parsed_data = {"text": data}
        output_key = (
            f"global/output/{from_agent}/{handoff_id}"
            if write_global
            else f"output/{from_agent}/{handoff_id}"
        )
        try:
            await mesh_client.write_blackboard(
                output_key, parsed_data, ttl=_HANDOFF_TTL,
                project=write_project, global_scope=write_global,
            )
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
    if origin:
        if hasattr(origin, "model_dump"):
            task_record["origin"] = origin.model_dump()
        elif isinstance(origin, dict):
            task_record["origin"] = dict(origin)

    task_key = (
        f"global/tasks/{to}/{handoff_id}"
        if write_global
        else f"tasks/{to}/{handoff_id}"
    )
    try:
        await mesh_client.write_blackboard(
            task_key, task_record, ttl=_HANDOFF_TTL,
            project=write_project, global_scope=write_global,
        )
    except Exception as e:
        # Clean up orphaned output if task write fails
        if output_key:
            logger.warning("Task write failed, orphaned output at %s", output_key)
        return {"error": f"Failed to create task: {e}"}

    # Wake the target agent so it processes the task immediately
    # instead of waiting for its next heartbeat.  Pass origin so
    # the mesh lane worker can auto-notify the originating user.
    try:
        await mesh_client.wake_agent(
            to, f"New task from {from_agent}: {summary[:200]}",
            origin=origin,
        )
    except Exception as e:
        logger.debug("Wake for %s failed (task still queued): %s", to, e)

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
        "After reading a task, use read_blackboard to fetch the full "
        "output data via the output_key. When done processing, call "
        "complete_task to mark it finished so it won't appear again."
    ),
    parameters={},
)
async def check_inbox(*, mesh_client=None) -> dict:
    if mesh_client is None:
        return {"error": "No mesh_client available"}

    agent_id = mesh_client.agent_id
    is_operator = (agent_id == "operator")

    # Standalone agents normally have no inbox (blackboard is project-scoped).
    # The operator is the exception: its inbox lives in a fleet-global namespace
    # so cross-project workers can hand off back to it without knowing its scope.
    if mesh_client.is_standalone and not is_operator:
        return {"error": _STANDALONE_ERROR}

    if is_operator:
        prefix = "global/tasks/operator/"
        try:
            entries = await mesh_client.list_blackboard(prefix, global_scope=True)
        except Exception as e:
            return {"error": f"Failed to check inbox: {e}"}
    else:
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

    agent_id = mesh_client.agent_id
    is_operator = (agent_id == "operator")

    if mesh_client.is_standalone and not is_operator:
        return {"error": _STANDALONE_ERROR}

    status_data = {
        "state": state,
        "summary": sanitize_for_prompt(summary),
        "ts": time.time(),
    }

    # Operator status writes to the fleet-global status namespace so it
    # is reachable from any project scope (operator has no project of its
    # own to scope writes to).
    status_key = f"global/status/{agent_id}" if is_operator else f"status/{agent_id}"
    try:
        await mesh_client.write_blackboard(
            status_key, status_data, global_scope=is_operator,
        )
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

    agent_id = mesh_client.agent_id
    is_operator = (agent_id == "operator")

    if mesh_client.is_standalone and not is_operator:
        return {"error": _STANDALONE_ERROR}

    # Ownership check — only complete tasks in your own inbox
    allowed_prefixes = [f"tasks/{agent_id}/"]
    if is_operator:
        allowed_prefixes.append("global/tasks/operator/")
    if not any(task_key.startswith(p) for p in allowed_prefixes):
        return {"error": f"Can only complete your own tasks (expected prefix: tasks/{agent_id}/)"}

    is_global_task = task_key.startswith("global/")

    try:
        # Read task to find associated output for cleanup
        existing = await mesh_client.read_blackboard(task_key, global_scope=is_global_task)
        if existing is None:
            return {"error": f"Task '{task_key}' not found"}

        # Delete the task entry
        await mesh_client.delete_blackboard(task_key, global_scope=is_global_task)

        # Best-effort cleanup of associated output data
        value = existing.get("value", existing)
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                value = {}
        if isinstance(value, dict):
            output_key = value.get("output_key")
            if isinstance(output_key, str) and output_key:
                try:
                    handoff_id = task_key.rsplit("/", 1)[-1]
                    sender = value.get("from")
                    expected_output = None
                    if isinstance(sender, str) and re.fullmatch(AGENT_ID_RE_PATTERN, sender):
                        expected_output = (
                            f"global/output/{sender}/{handoff_id}"
                            if is_global_task
                            else f"output/{sender}/{handoff_id}"
                        )
                    if output_key == expected_output:
                        await mesh_client.delete_blackboard(
                            output_key, global_scope=is_global_task,
                        )
                    else:
                        logger.debug(
                            "Skipping unexpected output cleanup for task %s: %s",
                            task_key, output_key,
                        )
                except Exception as exc:
                    logger.debug("Output cleanup for %s skipped: %s", output_key, exc)
    except Exception as e:
        return {"error": f"Failed to complete task: {e}"}

    return {"completed": True, "task_key": task_key}
