"""Coordination tools: structured multi-agent handoffs and status.

Higher-level wrappers over the durable ``tasks`` SQLite table that
encode the standard coordination protocol. Agents use these instead
of raw blackboard / task-store operations for inter-agent work
handoffs. The 4 tools (``hand_off`` / ``check_inbox`` /
``update_status`` / ``complete_task``) all route through the mesh's
``/mesh/tasks*`` endpoints — the legacy blackboard-dict path was
sunset after the v2 rollout (PR #835) and removed entirely once no
production fleet was running with the kill-switch off.
"""

from __future__ import annotations

import json
import re

from src.agent.tools import tool
from src.shared.redaction import redact_text_with_urls
from src.shared.task_titles import (
    LONG_TITLE_THRESHOLD,
    normalize_title_and_description,
)
from src.shared.types import AGENT_ID_RE_PATTERN
from src.shared.utils import generate_id, sanitize_for_prompt, setup_logging

logger = setup_logging("agent.builtins.coordination_tool")

# Inbox / status / completion live in the durable tasks table. Handoff
# artifacts (the optional ``data`` payload) still go to the blackboard
# under ``output/{agent_id}/{handoff_id}`` — the task record carries
# only the artifact_ref, keeping the SQLite table small.
_HANDOFF_TTL = 86_400  # 24 hours — safety net for unprocessed handoffs

# Mirrors ``host/orchestration.TERMINAL_STATUSES`` — the set of statuses
# that a task can no longer transition out of. Defined locally so the
# coordination tool stays decoupled from host internals.
_TERMINAL_STATES: frozenset[str] = frozenset({"done", "failed", "cancelled"})

# Cap on back-edge events returned by ``check_inbox``. The operator is the
# originator of nearly every workflow, so on a busy fleet the back-edge
# event list (7-day TTL, never pruned by the call) can grow to hundreds of
# entries and flood the LLM context (~80k-170k tokens) on every heartbeat.
# Actionable kinds (task_failed / task_blocked) are NEVER dropped; the cap
# only evicts the newest-surviving informational events.
_MAX_INBOX_EVENTS = 25

# Back-edge kinds the originator must act on. Mirrors the host-side
# ``_BACK_EDGE_WAKE_KINDS`` — these survive capping in ``check_inbox``.
_ACTIONABLE_EVENT_KINDS: frozenset[str] = frozenset({"task_failed", "task_blocked"})


def _failed_transition_envelope(
    *, kind: str, detail: str, exc: Exception, extras: dict | None = None,
) -> dict:
    """Build the directive failure envelope used by terminal-transition
    coordination tools (``update_status`` / ``complete_task``).

    Mirrors the Bug G ``wake_failed`` shape and the Bug H
    ``create_failed`` / ``output_write_failed`` envelopes used by
    ``hand_off`` — handed_off-style boolean flag (``<kind>``) + an
    ``error`` field with "MUST NOT report success" + a directive
    ``recovery_hint`` that tells the LLM to surface to the operator
    rather than silently retrying.

    ``exc`` is redacted (``redact_text_with_urls``) and truncated to 200
    chars before reaching the LLM context — HTTP-level exceptions from
    the mesh client can quote URLs with ``?api_key=...`` in the query
    string, same precaution as the wake_error path.

    Codex r6: ``extras`` are merged FIRST, then sentinel fields
    (``<kind>``, ``error``, ``recovery_hint``, ``detail``) overwrite —
    so a future caller cannot accidentally shadow a sentinel by passing
    e.g. ``extras={"error": "..."}``. Reserved keys are documented in
    ``_RESERVED_ENVELOPE_KEYS`` for fast inspection.
    """
    redacted = redact_text_with_urls(str(exc))[:200]
    envelope: dict = dict(extras) if extras else {}
    envelope.update({
        kind: True,
        "error": (
            f"{kind}: {detail} ({redacted}). The transition may not "
            "have landed — verify before continuing. You MUST NOT "
            "report success."
        ),
        "recovery_hint": (
            "Surface the failure to the operator/user. DO NOT mark "
            "this work as complete in your final response. A blind "
            "retry is unsafe — the underlying failure may have left "
            "the state partially applied."
        ),
        "detail": redacted,
    })
    return envelope


# Keys reserved by ``_failed_transition_envelope``. Callers passing
# ``extras`` containing any of these will see their value overwritten
# by the helper. ``kind`` is variable, so the set lists the static
# sentinels only — the dynamic flag (e.g. ``update_status_failed``,
# ``complete_task_failed``) is also reserved but cannot be enumerated
# statically. Documented here so future maintainers don't bury a
# silent overwrite in production.
_RESERVED_ENVELOPE_KEYS: frozenset[str] = frozenset(
    {"error", "recovery_hint", "detail"},
)


@tool(
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
    else:
        title = summary
        description = summary
    artifact_ref: str | None = None

    # Validate target exists in the registry — fail closed if the roster
    # is unreachable and we can't resolve the target's project scope.
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
        # Standalone senders need to resolve target project to write
        # the task into the correct scope.
        if not mesh_client.team_name:
            return {"error": f"Cannot hand off: fleet roster unavailable ({e})"}
        logger.debug("Fleet roster check failed, proceeding with validated ID: %s", e)

    # Pick the project scope:
    #   - operator + any other fleet-global agent (``scope: "global"``
    #     on the registry): project=None. Triggering on both the literal
    #     reserved name AND the registry hint keeps the path correct
    #     when the roster lookup fails (no hint available) AND stays
    #     forward-compatible if other global agents are added later.
    #   - cross-project worker handoffs: target's project
    #   - same-project handoffs: caller's project
    if to == "operator" or target_is_global:
        write_project = None
    elif target_project:
        write_project = target_project
    else:
        write_project = mesh_client.team_name
    # Round-4 forensic trace: if the registry doesn't surface a
    # ``project`` key for the target (e.g. post-rename drift) we fall
    # back to sender's team_name — log both inputs so the operator can
    # see the resolution path on the next E2E.
    logger.info(
        "hand_off scope resolution to=%s target_project=%r "
        "target_is_global=%s sender_team=%r write_project=%r",
        to, target_project, target_is_global,
        mesh_client.team_name, write_project,
    )

    if data and data.strip():
        # Optional: stash the payload under an artifact_ref the
        # recipient can fetch later. We reuse the blackboard for the
        # actual bytes — the task carries only the reference. This
        # keeps the table small and matches the pre-existing
        # output/{from}/{handoff_id} convention.
        try:
            parsed_data = json.loads(data)
        except json.JSONDecodeError:
            parsed_data = {"text": data}
        # Operator-bound handoff reports publish their output to the
        # ``global/output/{sender}/`` namespace rather than the sender's
        # team scope. That namespace is the one the operator's read
        # carve-out (mesh_tool.read_blackboard) AND the permission model
        # already target: permissions.can_read_blackboard restricts
        # ``global/output/`` to the operator + the writing agent's own
        # prefix, so the report is readable by exactly the operator and
        # its author — nobody else. Writing it team-scoped (the prior
        # behaviour) left it unreadable by the team-less operator while
        # needlessly exposing it to the sender's whole team. ``to ==
        # "operator"`` is the robust signal: it holds even when the fleet
        # roster lookup above failed and ``target_is_global`` is unknown.
        # ``global_scope`` keeps the key un-prefixed so it lands exactly
        # at the key we record in ``artifact_refs``.
        operator_bound = to == "operator"
        if operator_bound:
            artifact_ref = (
                f"global/output/{mesh_client.agent_id}/{generate_id('ho')}"
            )
        else:
            artifact_ref = f"output/{mesh_client.agent_id}/{generate_id('ho')}"
        try:
            await mesh_client.write_blackboard(
                artifact_ref, parsed_data, ttl=_HANDOFF_TTL,
                project=write_project,
                global_scope=operator_bound,
            )
        except Exception as e:
            # Bug H: a bare ``{"error": ...}`` envelope lets agents
            # report "done" to the user while no handoff data ever
            # reached the peer. Directive shape with explicit MUST NOT
            # report success + DO NOT mark complete in the recovery hint.
            redacted = redact_text_with_urls(str(e))[:200]
            logger.warning(
                "hand_off output write for %s failed: %s", to, redacted,
            )
            return {
                "handed_off": False,
                "task_queued": False,
                "output_write_failed": True,
                "to": to,
                "write_error": redacted,
                "error": (
                    f"output_write_failed: hand_off to '{to}' did not "
                    f"complete ({redacted}). The handoff data may not "
                    "have landed and no task was created — verify "
                    "before retrying. You MUST NOT report success."
                ),
                "recovery_hint": (
                    f"Surface the failure to the operator/user — "
                    f"hand_off to peer '{to}' did not complete. DO NOT "
                    "mark this work as complete in your final "
                    "response. DO NOT retry hand_off without verifying "
                    "via check_inbox or asking the operator to inspect "
                    "the recipient's queue — a post-commit failure may "
                    "have left a partial artifact, and a blind retry "
                    "would create a duplicate."
                ),
            }

    # Read origin + current task id contextvars once so both create_task
    # and wake_agent propagate the same provenance. ``current_task_id``
    # links the new task into the parent's workflow chain so
    # ``workflow_snapshot`` can walk the descendants from the kickoff
    # root. Outside a task context (heartbeats, free chat) the var is
    # ``None`` and the new task lands as a workflow root.
    from src.shared.trace import (
        current_origin as _current_origin,
    )
    from src.shared.trace import (
        current_task_id as _current_task_id,
    )
    origin = _current_origin.get()
    parent_task_id = _current_task_id.get()

    try:
        record = await mesh_client.create_task(
            assignee=to,
            title=title or "(handoff)",
            description=description,
            project=write_project,
            priority=0,
            parent_task_id=parent_task_id,
            artifact_refs=[artifact_ref] if artifact_ref else None,
            origin=origin,
        )
    except Exception as e:
        # Bug H: operator hit a live case where seo-strategist called
        # hand_off, create_task raised, and the agent still reported
        # "Brief completed" — recipient never received the task.
        # ``redact_text_with_urls`` strips URL credentials before the
        # exception string reaches the LLM context.
        redacted = redact_text_with_urls(str(e))[:200]
        logger.warning("create_task for %s failed: %s", to, redacted)
        return {
            "handed_off": False,
            "task_queued": False,
            "create_failed": True,
            "to": to,
            "create_error": redacted,
            "error": (
                f"create_failed: hand_off to '{to}' did not complete "
                f"({redacted}). The task row may not exist — verify "
                "before retrying. You MUST NOT report success."
            ),
            "recovery_hint": (
                f"Surface the failure to the operator/user — hand_off "
                f"to peer '{to}' did not complete. DO NOT mark this "
                "work as complete in your final response. DO NOT retry "
                "hand_off without first verifying via check_inbox or "
                "asking the operator to inspect the recipient's queue "
                "— a post-commit failure may have left a partial row, "
                "and a blind retry would create a duplicate."
            ),
        }

    task_id = record.get("id", "")
    # Round-4 forensic trace: create_task succeeded with this id. Pairs
    # with the ``tasks.create stored`` server log so the chain from
    # agent → mesh → SQLite can be reconstructed from the operator's
    # log on the next E2E. Surfacing the assignee + parent_task_id +
    # creator surfaced from the server's POST handler exposes any
    # value drift between request and storage at a glance.
    logger.info(
        "hand_off create_task returned task_id=%s to=%s "
        "stored_assignee=%s stored_creator=%s stored_parent=%s "
        "stored_team_id=%s",
        task_id, to,
        record.get("assignee"), record.get("creator"),
        record.get("parent_task_id"), record.get("project_id"),
    )

    # Wake the target so they pick up the task immediately. The task
    # is queued in SQLite either way; wake failures surface via
    # ``handed_off=False`` + ``task_queued=True`` so the caller can
    # decide whether to retry the wake or wait for the recipient's
    # next cron. Forward task_id so the recipient's lane → /chat → loop
    # chain auto-closes the task on completion (Constraint #16).
    wake_error: str | None = None
    wake_status: int | None = None
    try:
        await mesh_client.wake_agent(
            to, f"New task from {mesh_client.agent_id}: {summary[:200]}",
            origin=origin,
            task_id=task_id or None,
        )
    except Exception as e:
        # Redact BEFORE truncation — HTTP-level exceptions can quote
        # the failing URL with an API key in the query string.
        wake_error = redact_text_with_urls(str(e))[:200]
        # Capture the HTTP status (httpx.HTTPStatusError carries .response)
        # so we can distinguish the BY-DESIGN worker→operator 403 from a
        # genuine infra failure (500 / network) below.
        wake_status = getattr(getattr(e, "response", None), "status_code", None)
        logger.warning("Wake for %s failed (task still queued): %s", to, e)

    result = {
        "handed_off": wake_error is None,
        "to": to,
        "handoff_id": task_id,
        "task_key": task_id,
        "task_id": task_id,
    }
    if artifact_ref:
        result["output_key"] = artifact_ref
    _operator_by_design_403 = (
        to == "operator"
        and (wake_status == 403 or "403 forbidden" in (wake_error or "").lower())
    )
    if wake_error is not None and _operator_by_design_403:
        # By design, worker agents CANNOT synchronously wake the operator —
        # the mesh /mesh/wake endpoint returns 403 for worker→operator and
        # the operator discovers queued work on its heartbeat poll. The
        # durable task row was persisted above, so a queued handoff to the
        # operator IS the intended successful outcome. Classifying the
        # expected 403 as ``wake_failed`` (the branch below) marked the
        # ORIGINATING task ``failed`` and surfaced a scary, technical
        # "403 Forbidden … host.docker.internal" note to the user. The mesh
        # permission boundary is untouched; we only stop mislabelling the
        # expected denial as a failure. A NON-403 operator wake error (500,
        # network) is a genuine infra failure and still flows to the
        # ``wake_failed`` envelope below so it stays visible.
        result["handed_off"] = True
        result["queued_for_heartbeat"] = True
    elif wake_error is not None:
        # Bug G: the durable task row sits in SQLite at status='pending'
        # for the next-heartbeat discovery path — we don't transition it
        # to failed because a transient wake error (network blip, agent
        # restarting) shouldn't kill a task whose row was persisted.
        # Recovery hint MUST NOT instruct "retry hand_off" — each call
        # creates a brand-new task row, so a retry leaves orphan
        # duplicates. Direct callers to notify operator with the task_id.
        result["task_queued"] = True
        result["wake_failed"] = True
        result["wake_error"] = wake_error
        result["error"] = (
            f"wake_failed: peer '{to}' did not wake ({wake_error}). "
            f"The task row is queued in SQLite (task_id={task_id}) but "
            "the recipient has not been notified — you MUST NOT report "
            "success. Surface this to the operator with the task_id so "
            "they can re-wake or reroute."
        )
        result["recovery_hint"] = (
            f"Notify operator with task_id={task_id} so they can re-wake "
            f"'{to}' or reroute. DO NOT retry hand_off (creates a "
            "duplicate row). DO NOT mark this work as complete in your "
            "final response."
        )
    # Round-4 forensic trace: final envelope shape the LLM sees. Pairs
    # with the ``hand_off create_task returned`` line above so a future
    # repro shows whether the LLM ignored a failure envelope OR the
    # envelope was malformed in a way that hid the failure.
    logger.info(
        "hand_off result to=%s handed_off=%s task_id=%s flags=%s",
        to, result.get("handed_off"), task_id,
        [k for k in (
            "create_failed", "wake_failed", "output_write_failed",
            "task_queued",
        ) if result.get(k)],
    )
    return result


@tool(
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
    """Read the agent's task inbox from the durable tasks table, plus
    any back-edge events from the blackboard at
    ``inbox/{agent_id}/task_event/`` so an originating agent learns when
    a handed-off task reached a terminal state. The event fetch is
    best-effort — a blackboard hiccup degrades to an empty event list
    rather than failing the whole call (Constraint #16).
    """
    if mesh_client is None:
        return {"error": "No mesh_client available"}

    try:
        rows = await mesh_client.list_task_inbox(mesh_client.agent_id)
    except Exception as e:
        return {"error": f"Failed to check inbox: {e}"}
    tasks = [_task_to_inbox_row(r) for r in rows]

    events: list[dict] = []
    try:
        prefix = f"inbox/{mesh_client.agent_id}/task_event/"
        entries = await mesh_client.list_blackboard(prefix, global_scope=True)
        for entry in entries:
            value = entry.get("value") or {}
            # Free-text fields here re-enter THIS agent's LLM context, so
            # sanitize them at the input boundary (a failed peer's note can
            # echo arbitrary/injected content). Enum/id fields are safe.
            events.append({
                "key": entry.get("key", ""),
                "kind": value.get("kind", ""),
                "task_id": value.get("task_id", ""),
                "recipient": value.get("recipient", ""),
                "title": sanitize_for_prompt(value.get("title", "")),
                "status": value.get("status", ""),
                "ts": value.get("ts"),
                "summary": sanitize_for_prompt(value.get("summary", "")),
                "error": sanitize_for_prompt(value.get("error", "")),
                "blocker_note": sanitize_for_prompt(value.get("blocker_note", "")),
            })
    except Exception as e:
        logger.warning(
            "check_inbox event fetch failed for %s: %s",
            mesh_client.agent_id, e,
        )

    # Bound the event list before returning. The operator originates almost
    # every workflow, so without a cap this list grows to hundreds of entries
    # (7-day TTL) and re-floods the LLM context on every heartbeat. Actionable
    # events (task_failed / task_blocked) must NEVER be dropped; informational
    # events (task_completed / task_cancelled) are evicted oldest-first to make
    # room. Within the returned list, actionable events come first (newest
    # actionable first) so the LLM sees what it must act on at the top.
    events_total = len(events)

    def _event_ts(ev: dict) -> float:
        ts = ev.get("ts")
        try:
            return float(ts) if ts is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    actionable = [e for e in events if e.get("kind") in _ACTIONABLE_EVENT_KINDS]
    informational = [e for e in events if e.get("kind") not in _ACTIONABLE_EVENT_KINDS]
    actionable.sort(key=_event_ts, reverse=True)
    informational.sort(key=_event_ts, reverse=True)

    # Actionable events are always retained (never dropped). Informational
    # events fill whatever slots remain under the cap, newest first.
    remaining = max(_MAX_INBOX_EVENTS - len(actionable), 0)
    capped_events = actionable + informational[:remaining]

    return {
        "tasks": tasks,
        "count": len(tasks),
        "events": capped_events,
        "event_count": len(capped_events),
        "events_total": events_total,
        "events_truncated": events_total > len(capped_events),
    }


@tool(
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
    """Transition one of the agent's tasks on the durable tasks table.

    When the agent has exactly one non-terminal task we transition it
    transparently. When the agent has 2+ non-terminal tasks and no
    ``task_id`` is supplied we return ``ambiguous_task`` rather than
    silently picking ``rows[-1]`` — the legacy "most recent wins" rule
    masked the case where an agent juggling multiple handoffs marked
    the wrong task ``done``. When ``task_id`` is supplied we route the
    transition to that exact task or return ``task_not_found``.
    """
    if mesh_client is None:
        return {"error": "No mesh_client available"}

    task_id = task_id or None

    # ``idle`` was a legacy blackboard-only state with no per-task
    # mapping — preserved as a no-op so existing prompts don't error.
    if state == "idle":
        return {"updated": True, "state": state, "noop": "idle has no task mapping"}

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
        else:
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
        # Bug H: set_task_status to a terminal state ('done' / 'failed'
        # / 'cancelled') that raises post-commit could leave the row
        # mid-transition. Directive envelope so the LLM doesn't claim
        # success on a state that didn't land.
        return _failed_transition_envelope(
            kind="update_status_failed",
            detail=(
                f"could not transition task '{target['id']}' to "
                f"state='{state}'"
            ),
            exc=e,
            extras={"state": state, "task_id": target["id"]},
        )
    return {"updated": True, "state": state, "task_id": target["id"]}


@tool(
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
    """Mark a task done on the durable tasks table.

    ``task_key`` is the task id (preferred). Legacy callers may pass
    one of the old blackboard-style keys (``tasks/x/ho_abc``); we
    strip the prefix and use the trailing segment as the id so prompts
    that still mention the legacy shape resolve cleanly.
    """
    if mesh_client is None:
        return {"error": "No mesh_client available"}

    task_id = task_key.rsplit("/", 1)[-1] if "/" in task_key else task_key
    try:
        # A handed-off report/notification task usually sits in ``pending``
        # (or ``accepted``) because its assignee never moved it through
        # ``working`` — the canonical case is the fleet operator clearing a
        # completion report another agent handed it. The task state machine
        # forbids ``pending → done`` directly (you may abandon un-started
        # work, but not claim success on it), so a naive complete_task 400s
        # and the task wedges, re-surfacing in check_inbox every cycle. Step
        # it through ``working`` first so the terminal close is valid. The
        # status probe is best-effort: if it fails we fall back to the
        # direct close (preserving the original behaviour for tasks already
        # in a completable state).
        try:
            current = await mesh_client.get_task(task_id)
        except Exception:
            current = None
        if current and current.get("status") in ("pending", "accepted"):
            await mesh_client.set_task_status(task_id, "working")
        record = await mesh_client.set_task_status(task_id, "done")
    except Exception as e:
        # Bug H: terminal transition — silent failure would leave the
        # task pending while the agent claims completion.
        return _failed_transition_envelope(
            kind="complete_task_failed",
            detail=f"could not mark task '{task_id}' as done",
            exc=e,
            extras={"task_id": task_id, "task_key": task_key},
        )
    return {
        "completed": True,
        "task_key": task_key,
        "task_id": record.get("id", task_id),
    }


# ── Helpers ─────────────────────────────────────────────────────────


def _task_to_inbox_row(task: dict) -> dict:
    """Map a durable task record to the LLM-facing inbox row shape.

    ``key`` is the task id (also exposed as ``task_id`` for clarity),
    ``output_key`` is the first artifact_ref if present.
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
