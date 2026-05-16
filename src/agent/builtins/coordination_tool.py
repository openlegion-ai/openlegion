"""Coordination tools: structured multi-agent handoffs and status.

Higher-level wrappers over the blackboard that encode the standard
coordination protocol.  Agents use these instead of raw blackboard
operations for inter-agent work handoffs.

Task 6 introduces a durable ``tasks`` SQLite table behind the
``OPENLEGION_ORCHESTRATION_TASKS_V2`` flag. When the mesh advertises
that the flag is on (probed once via ``mesh_client.orchestration_v2_enabled``),
each tool routes through the new ``/mesh/tasks*`` endpoints. When off,
the legacy blackboard path runs unchanged.
"""

from __future__ import annotations

import json
import re
import time

from src.agent.skills import skill
from src.shared.task_titles import (
    LONG_TITLE_THRESHOLD,
    normalize_title_and_description,
)
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

# Mirrors ``host/orchestration.TERMINAL_STATUSES`` — the set of statuses
# that a task can no longer transition out of. Defined locally so the
# coordination tool stays decoupled from host internals.
_TERMINAL_STATES: frozenset[str] = frozenset({"done", "failed", "cancelled"})


@skill(
    name="hand_off",
    description=(
        "Hand off work to a teammate. Creates a task in their inbox and "
        "wakes them up automatically. This is your PRIMARY coordination "
        "tool — use it whenever you've completed work that another agent "
        "should act on.\n\n"
        "Keep the 'summary' SHORT — it becomes the task title in the "
        "recipient's inbox and on the dashboard. Aim for ≤80 characters, "
        "like a Git commit subject line. Examples: 'Draft Q3 launch brief' "
        "or 'Review pricing change for SKU-123'. If you need to send a "
        "full instruction or spec, put it in 'data' (JSON) — that's where "
        "long context belongs. A long summary will still work (the system "
        "auto-splits it into a short title + description) but a hand-"
        "written short summary reads better.\n\n"
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
                "Short title for the task — what you handed off. "
                "Aim for ≤80 characters, like a commit subject. "
                "Put long instructions in 'data', not here."
            ),
        },
        "data": {
            "type": "string",
            "description": (
                "Optional JSON string of output data or a full instruction "
                "for the recipient. Written to the blackboard so the "
                "recipient can read the full details. This is where long "
                "context belongs — keep 'summary' to a short title."
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

    # Task 6: route through the durable tasks table when the mesh
    # advertises v2. Probe is cached per process. On any error we fall
    # through to the legacy blackboard path so existing fleets keep
    # working until the flag is flipped.
    try:
        v2_enabled = await mesh_client.orchestration_v2_enabled()
    except Exception:
        v2_enabled = False
    if v2_enabled:
        return await _hand_off_v2(to, summary, data, mesh_client=mesh_client)

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
        # Persist as plain dict so the blackboard JSON write doesn't
        # choke on a Pydantic instance and downstream readers see the
        # same shape they always have.
        task_record["origin"] = origin.model_dump()

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
    # Wake failures are surfaced to the caller via ``wake_failed`` +
    # ``wake_error`` (handed_off stays True — the task IS queued in
    # SQLite, only the wake signal failed; the caller can decide
    # whether to retry the wake or wait for cron).
    wake_error: str | None = None
    try:
        await mesh_client.wake_agent(
            to, f"New task from {from_agent}: {summary[:200]}",
            origin=origin,
        )
    except Exception as e:
        wake_error = str(e)[:200]
        logger.warning("Wake for %s failed (task still queued): %s", to, e)

    result = {
        "handed_off": True,
        "to": to,
        "task_key": task_key,
        "handoff_id": handoff_id,
    }
    if output_key:
        result["output_key"] = output_key
    if wake_error is not None:
        result["wake_failed"] = True
        result["wake_error"] = wake_error
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

    # Task 6: route through the durable tasks table when v2 is on.
    try:
        v2_enabled = await mesh_client.orchestration_v2_enabled()
    except Exception:
        v2_enabled = False
    if v2_enabled:
        return await _check_inbox_v2(mesh_client=mesh_client)

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
        "to wait or proceed.\n\n"
        "When you have multiple active tasks, pass task_id explicitly "
        "to disambiguate which one this status update applies to. "
        "Otherwise the call returns ambiguous_task with the active task "
        "ids so you can pick the right one."
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
        "task_id": {
            "type": "string",
            "description": (
                "Optional task id to update a specific task. Required when "
                "you have more than one active task; otherwise omit."
            ),
            "default": "",
        },
    },
)
async def update_status(
    state: str, summary: str = "", task_id: str = "",
    *, mesh_client=None,
) -> dict:
    if mesh_client is None:
        return {"error": "No mesh_client available"}

    agent_id = mesh_client.agent_id
    is_operator = (agent_id == "operator")

    # Task 6: when v2 is on, ``update_status`` updates the assignee's
    # currently-active task. The legacy blackboard ``status/{agent_id}``
    # write still happens for back-compat surfaces that read directly
    # from the blackboard, but only when v2 is OFF — under v2 the
    # per-task status is the source of truth.
    try:
        v2_enabled = await mesh_client.orchestration_v2_enabled()
    except Exception:
        v2_enabled = False
    if v2_enabled:
        return await _update_status_v2(
            state, summary, task_id=task_id or None, mesh_client=mesh_client,
        )

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

    # Task 6: when v2 is on, ``task_key`` is the task id (the storage
    # layer's primary key). The hand_off v2 path returns this id as
    # ``handoff_id`` for back-compat with the old result shape.
    try:
        v2_enabled = await mesh_client.orchestration_v2_enabled()
    except Exception:
        v2_enabled = False
    if v2_enabled:
        return await _complete_task_v2(task_key, mesh_client=mesh_client)

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


# ── Task 6: durable orchestration tasks v2 helpers ─────────────────


def _v2_task_to_legacy_dict(task: dict) -> dict:
    """Map a v2 task record to the legacy ``check_inbox`` row shape.

    The LLM-facing tool surface stayed the same so existing prompts and
    skills don't need to change when the flag flips. ``key`` is the
    task id (also exposed as ``task_id`` for clarity), ``output_key``
    is the first artifact_ref if present.
    """
    artifact_refs = task.get("artifact_refs") or []
    out: dict = {
        "key": sanitize_for_prompt(str(task.get("id", ""))),
        "task_id": str(task.get("id", "")),
        "from": sanitize_for_prompt(str(task.get("creator", "unknown"))),
        "summary": sanitize_for_prompt(str(task.get("title", ""))),
        "status": str(task.get("status", "pending")),
    }
    if artifact_refs:
        out["output_key"] = sanitize_for_prompt(str(artifact_refs[0]))
    created_at = task.get("created_at")
    if created_at:
        out["ts"] = created_at
    return out


async def _hand_off_v2(
    to: str, summary: str, data: str, *, mesh_client,
) -> dict:
    """Task 6 hand_off path — creates a durable task row.

    Mirrors the legacy result shape (``handed_off``, ``to``,
    ``handoff_id``, ``output_key``, ``task_key``) so callers don't need
    a flag-aware result handler.
    """
    summary = sanitize_for_prompt(summary)
    # The handoff ``summary`` carries both the headline (becomes the
    # task title) and the full instruction (becomes the description).
    # Agents historically dumped multi-sentence instructions into
    # ``summary`` — that produced wall-of-text titles in the dashboard.
    # ``Tasks.create`` applies the same policy as a backstop, but we
    # mirror it here so the title we surface in result envelopes, wake
    # messages, and downstream logs is already short.
    if len(summary) > LONG_TITLE_THRESHOLD:
        title, description = normalize_title_and_description(summary, None)
        # ``description`` is the full original ``summary`` here — keep
        # it intact so the recipient still sees the complete instruction.
    else:
        title = summary
        description = summary
    artifact_ref: str | None = None

    # Validate target exists in the registry — same fail-closed behavior
    # as the legacy path.
    target_project: str | None = None
    try:
        registry = await mesh_client.list_agents()
        if to not in registry:
            available = ", ".join(sorted(registry.keys()))
            return {"error": f"Agent '{to}' not found. Available: {available}"}
        target_info = registry.get(to, {})
        if isinstance(target_info, dict):
            target_project = target_info.get("project")
    except Exception as e:
        # Standalone senders need to resolve target project to write
        # the task into the correct scope.
        if not mesh_client.project_name:
            return {"error": f"Cannot hand off: fleet roster unavailable ({e})"}
        logger.debug("Fleet roster check failed, proceeding with validated ID: %s", e)

    # Pick the project scope:
    #   - operator handoffs: project=None (fleet-global)
    #   - cross-project worker handoffs: target's project
    #   - same-project handoffs: caller's project
    if to == "operator":
        write_project = None
    elif target_project:
        write_project = target_project
    else:
        write_project = mesh_client.project_name

    if data and data.strip():
        # Optional: stash the payload under an artifact_ref the
        # recipient can fetch later. We reuse the blackboard for the
        # actual bytes — the v2 task carries only the reference. This
        # keeps the table small and matches the pre-existing
        # output/{from}/{handoff_id} convention.
        try:
            parsed_data = json.loads(data)
        except json.JSONDecodeError:
            parsed_data = {"text": data}
        # Generate a placeholder ref now — we'll write under that key.
        artifact_ref = f"output/{mesh_client.agent_id}/{generate_id('ho')}"
        try:
            await mesh_client.write_blackboard(
                artifact_ref, parsed_data, ttl=_HANDOFF_TTL,
                project=write_project,
            )
        except Exception as e:
            return {"error": f"Failed to write output: {e}"}

    # Read the origin contextvar once so both create_task and wake_agent
    # propagate the same provenance. Without origin on create_task the
    # receiving agent's lane worker has no way to auto-notify the
    # originating channel/user when the handed-off task completes —
    # wake_agent already passes it; this brings create_task into parity.
    from src.shared.trace import current_origin as _current_origin
    origin = _current_origin.get()

    try:
        record = await mesh_client.create_task(
            assignee=to,
            title=title or "(handoff)",
            description=description,
            project=write_project,
            priority=0,
            artifact_refs=[artifact_ref] if artifact_ref else None,
            origin=origin,
        )
    except Exception as e:
        return {"error": f"Failed to create task: {e}"}

    task_id = record.get("id", "")

    # Wake the target so they pick up the task immediately. The task
    # is queued in SQLite either way; wake failures are surfaced to the
    # caller via ``wake_failed`` + ``wake_error`` so they can decide
    # whether to retry the wake or wait for the recipient's next cron.
    wake_error: str | None = None
    try:
        await mesh_client.wake_agent(
            to, f"New task from {mesh_client.agent_id}: {summary[:200]}",
            origin=origin,
        )
    except Exception as e:
        wake_error = str(e)[:200]
        logger.warning("Wake for %s failed (task still queued): %s", to, e)

    result = {
        "handed_off": True,
        "to": to,
        "handoff_id": task_id,
        "task_key": task_id,
        "task_id": task_id,
    }
    if artifact_ref:
        result["output_key"] = artifact_ref
    if wake_error is not None:
        result["wake_failed"] = True
        result["wake_error"] = wake_error
    return result


async def _check_inbox_v2(*, mesh_client) -> dict:
    """Task 6 check_inbox path — reads the durable tasks table."""
    try:
        rows = await mesh_client.list_task_inbox(mesh_client.agent_id)
    except Exception as e:
        return {"error": f"Failed to check inbox: {e}"}
    tasks = [_v2_task_to_legacy_dict(r) for r in rows]
    return {"tasks": tasks, "count": len(tasks)}


async def _update_status_v2(
    state: str, summary: str, *, task_id: str | None = None, mesh_client,
) -> dict:
    """Task 6 update_status path — transitions one of the agent's tasks.

    When the agent has exactly one non-terminal task we transition it
    transparently. When the agent has 2+ non-terminal tasks and no
    ``task_id`` is supplied we return ``ambiguous_task`` rather than
    silently picking ``rows[-1]`` — the legacy "most recent wins" rule
    masked the case where an agent juggling multiple handoffs marked
    the wrong task ``done``. When ``task_id`` is supplied we route the
    transition to that exact task or return ``task_not_found``.
    """
    # Map the legacy ``state`` to the v2 status name.  Both share the
    # same vocabulary except the legacy ``idle`` which has no v2
    # equivalent — we treat it as a no-op since "idle" wasn't a per-task
    # status to begin with.
    if state == "idle":
        return {"updated": True, "state": state, "noop": "idle has no v2 mapping"}
    try:
        rows = await mesh_client.list_task_inbox(mesh_client.agent_id)
    except Exception as e:
        return {"error": f"Failed to load inbox: {e}"}

    if task_id is None:
        active = [
            r for r in rows
            if str(r.get("status", "")) not in _TERMINAL_STATES
        ]
        if len(active) > 1:
            # Augment the active list with title + state so the LLM can
            # pick a ``task_id`` directly without a follow-up
            # ``check_inbox`` call.
            active_summaries = [
                {
                    "id": r.get("id"),
                    "title": str(r.get("title", ""))[:80],
                    "state": str(r.get("status", "")),
                }
                for r in active
            ]
            return {
                "error": "ambiguous_task",
                "active": active_summaries,
                "hint": (
                    "You have multiple active tasks. Pass task_id "
                    "explicitly to update a specific task."
                ),
            }
        if active:
            target = active[0]
        elif rows:
            # No active rows but inbox has terminal entries — preserve
            # the legacy "most recent wins" no-op behavior so the call
            # still resolves to {updated: False, reason: ...} downstream
            # rather than blowing up. Picking the last terminal row is
            # deliberately benign: ``set_task_status`` will reject any
            # transition out of a terminal state, so the call surfaces
            # the failure as an error rather than silently mutating
            # something the caller didn't intend.
            return {
                "updated": False, "state": state, "reason": "no active tasks",
            }
        else:
            # Empty inbox — standalone agents and just-joined agents on a
            # fresh fleet hit this constantly. Return the legacy
            # ``{updated: False, ...}`` no-op shape so callers don't see
            # an LLM-visible error for the common "no work yet" case.
            return {
                "updated": False, "state": state, "reason": "no active tasks",
            }
    else:
        target = next(
            (r for r in rows if r.get("id") == task_id), None,
        )
        if target is None:
            return {"error": "task_not_found", "task_id": task_id}

    blocker_note = sanitize_for_prompt(summary) if state == "blocked" else None
    try:
        await mesh_client.set_task_status(
            target["id"], state, blocker_note=blocker_note,
        )
    except Exception as e:
        return {"error": f"Failed to update status: {e}"}
    return {"updated": True, "state": state, "task_id": target["id"]}


async def _complete_task_v2(task_key: str, *, mesh_client) -> dict:
    """Task 6 complete_task path — transitions a task to ``done``."""
    # ``task_key`` may be either the raw task id (preferred) or one of
    # the legacy blackboard-style keys (``tasks/x/ho_abc``). Strip the
    # legacy prefix when present so flag-flipped fleets can still pass
    # through the old key shapes for one transition cycle.
    task_id = task_key.rsplit("/", 1)[-1] if "/" in task_key else task_key
    try:
        record = await mesh_client.set_task_status(task_id, "done")
    except Exception as e:
        return {"error": f"Failed to complete task: {e}"}
    return {
        "completed": True,
        "task_key": task_key,
        "task_id": record.get("id", task_id),
    }
