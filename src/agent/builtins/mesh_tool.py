"""Mesh interaction tools: shared state, fleet awareness, artifacts.

Framework-level skills available to every agent. Agents coordinate through
the shared blackboard (not through direct conversations with each other).
"""

from __future__ import annotations

import json
from pathlib import Path

from src.agent.skills import skill
from src.shared.utils import sanitize_for_prompt, setup_logging

logger = setup_logging("agent.builtins.mesh_tool")

_NOTIFY_COOLDOWNS: dict[str, float] = {}
_NOTIFY_COOLDOWN_SECONDS = 120  # 2 minutes between similar notifications


@skill(
    name="notify_user",
    description=(
        "Send a notification to the user across all connected channels "
        "(CLI, Telegram, Discord, Slack, etc.). This is your PRIMARY way "
        "to report back to the user when working autonomously (heartbeat, "
        "cron jobs, long-running tasks). Use it for progress updates, "
        "completed work summaries, errors needing attention, or anything "
        "the user should know about. Keep messages concise and actionable."
    ),
    parameters={
        "message": {
            "type": "string",
            "description": "The notification message to send to the user",
        },
    },
)
async def notify_user(message: str, *, mesh_client=None, workspace_manager=None) -> dict:
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    try:
        import time as _time

        # Deduplicate similar notifications within cooldown window
        notify_key = message[:80].lower().strip()
        now = _time.time()
        last_sent = _NOTIFY_COOLDOWNS.get(notify_key, 0)
        if now - last_sent < _NOTIFY_COOLDOWN_SECONDS:
            return {"sent": False, "reason": "Similar notification sent recently. Wait before sending again."}
        _NOTIFY_COOLDOWNS[notify_key] = now

        await mesh_client.notify_user(message)
        if workspace_manager:
            workspace_manager.append_chat_message("notification", message)
        return {"sent": True}
    except Exception as e:
        return {"error": f"Failed to notify user: {e}"}


@skill(
    name="list_agents",
    description=(
        "List agents in your project (or just yourself if standalone). Returns "
        "each agent's name, role, and capabilities. Use this to discover who "
        "else is working in your project. For detailed collaboration info "
        "(what they accept/produce, their status, INTERFACE.md contract), "
        "follow up with get_agent_profile(agent_id)."
    ),
    parameters={},
)
async def list_agents(*, mesh_client=None) -> dict:
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    try:
        registry = await mesh_client.list_agents()
        agents = []
        for name, info in registry.items():
            entry = {"name": name}
            if isinstance(info, dict):
                entry["role"] = info.get("role", "")
                entry["capabilities"] = info.get("capabilities", [])
            agents.append(entry)
        return {"agents": agents, "count": len(agents)}
    except Exception as e:
        return {"error": f"Failed to list agents: {e}"}


_STANDALONE_ERROR = (
    "Not available — this agent is not assigned to a project. "
    "Use memory_save/memory_search for private storage."
)


def _parse_json_value(value):
    """Parse a string as JSON, falling back to a ``{"text": ...}`` wrapper."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {"text": value}


def _sanitize_value(value):
    """Recursively sanitize all strings inside a value (dict, list, or scalar)."""
    if isinstance(value, str):
        return sanitize_for_prompt(value)
    if isinstance(value, dict):
        return {sanitize_for_prompt(k): _sanitize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    return value


@skill(
    name="read_blackboard",
    description=(
        "Read a value from the shared blackboard. Returns the full value "
        "another agent wrote, or null if the key doesn't exist. Use "
        "list_blackboard first if you don't know the exact key."
    ),
    parameters={
        "key": {
            "type": "string",
            "description": (
                "Blackboard key to read (e.g. 'status/researcher', "
                "'research/acme_findings')"
            ),
        },
    },
)
async def read_blackboard(key: str, *, mesh_client=None) -> dict:
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    if mesh_client.is_standalone:
        return {"error": _STANDALONE_ERROR}
    try:
        entry = await mesh_client.read_blackboard(key)
        if entry is None:
            return {"key": key, "exists": False, "value": None}
        value = _sanitize_value(entry.get("value", entry))
        return {"key": key, "exists": True, "value": value}
    except Exception as e:
        return {"error": f"Failed to read '{key}': {e}"}


@skill(
    name="write_blackboard",
    description=(
        "Write a value to the shared blackboard for OTHER AGENTS to read. "
        "The blackboard is persistent agent-to-agent storage — do NOT use "
        "it to report to the user (use notify_user or chat for that).\n\n"
        "For standard coordination, prefer the dedicated tools:\n"
        "- hand_off() for sending work to a teammate (writes to tasks/ and output/)\n"
        "- update_status() for broadcasting your state (writes to status/)\n"
        "- complete_task() for marking tasks done\n\n"
        "Use write_blackboard for custom data that doesn't fit the "
        "coordination protocol (e.g. research/, drafts/, analysis/).\n"
        "- artifacts/ — managed by save_artifact, don't write directly\n\n"
        "Values must be JSON-serializable."
    ),
    parameters={
        "key": {
            "type": "string",
            "description": (
                "Blackboard key (e.g. 'status/researcher', "
                "'research/acme_findings', 'tasks/pending/review')"
            ),
        },
        "value": {
            "type": "string",
            "description": "JSON string of the value to store",
        },
    },
)
async def write_blackboard(key: str, value: str, *, mesh_client=None) -> dict:
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    if mesh_client.is_standalone:
        return {"error": _STANDALONE_ERROR}
    parsed = _parse_json_value(value)
    try:
        result = await mesh_client.write_blackboard(key, parsed)
        return {"key": key, "written": True, "version": result.get("version", 1)}
    except Exception as e:
        return {"error": f"Failed to write '{key}': {e}"}


@skill(
    name="list_blackboard",
    description=(
        "Discover what's on the shared blackboard by listing entries matching a key "
        "prefix. Returns key names, authors, timestamps, and value previews — but "
        "NOT full values (use read_blackboard for that). Use prefix='' to see "
        "everything, or a domain prefix like 'research/' or 'status/' to filter."
    ),
    parameters={
        "prefix": {
            "type": "string",
            "description": (
                "Key prefix to filter by (e.g. 'status/', 'research/', "
                "'artifacts/', '' for all)"
            ),
            "default": "",
        },
    },
)
async def list_blackboard(prefix: str = "", *, mesh_client=None) -> dict:
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    if mesh_client.is_standalone:
        return {"error": _STANDALONE_ERROR}
    try:
        entries = await mesh_client.list_blackboard(prefix)
        items = []
        for entry in entries:
            raw = json.dumps(entry.get("value", {}), default=str)
            preview = sanitize_for_prompt(raw[:200] + "..." if len(raw) > 200 else raw)
            items.append({
                "key": sanitize_for_prompt(entry.get("key", "")),
                "written_by": entry.get("written_by", ""),
                "updated_at": entry.get("updated_at", ""),
                "value_preview": preview,
            })
        return {"prefix": prefix, "entries": items, "count": len(items)}
    except Exception as e:
        return {"error": f"Failed to list '{prefix}': {e}"}


@skill(
    name="publish_event",
    description=(
        "Broadcast a one-time signal to other agents via pub/sub. Subscribed "
        "agents receive it immediately as a steer message — no polling needed. "
        "Use this for ephemeral notifications (e.g. 'research_complete', "
        "'build_failed'). The event is NOT stored — if no one is subscribed, "
        "it's lost. For data that should persist, use write_blackboard instead."
    ),
    parameters={
        "topic": {
            "type": "string",
            "description": "Event topic (e.g. 'research_complete', 'deploy_ready')",
        },
        "data": {
            "type": "string",
            "description": "JSON string of event payload",
            "default": "{}",
        },
    },
)
async def publish_event(
    topic: str, data: str = "{}", *, mesh_client=None,
) -> dict:
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    if mesh_client.is_standalone:
        return {"error": _STANDALONE_ERROR}
    parsed = _parse_json_value(data)
    try:
        result = await mesh_client.publish_event(topic, parsed)
        return {"published": True, "topic": topic, **result}
    except Exception as e:
        return {"error": f"Failed to publish to '{topic}': {e}"}


@skill(
    name="subscribe_event",
    description=(
        "Subscribe to a pub/sub topic for one-time signals. Once subscribed, "
        "events published to this topic arrive as steer messages between your "
        "tool rounds — no polling needed. Use for ephemeral notifications "
        "where you need to react immediately (e.g. 'research_complete', "
        "'deploy_ready'). Events are fire-and-forget — if you subscribe "
        "after an event fires, you won't see it. For reacting to persistent "
        "data changes, use watch_blackboard instead."
    ),
    parameters={
        "topic": {
            "type": "string",
            "description": "Event topic to subscribe to (e.g. 'research_complete')",
        },
    },
)
async def subscribe_event(topic: str, *, mesh_client=None) -> dict:
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    if mesh_client.is_standalone:
        return {"error": _STANDALONE_ERROR}
    try:
        result = await mesh_client.subscribe_topic(topic)
        return {"subscribed": True, "topic": topic, **result}
    except Exception as e:
        return {"error": f"Failed to subscribe to '{topic}': {e}"}


@skill(
    name="watch_blackboard",
    description=(
        "Watch blackboard keys matching a glob pattern. When any matching "
        "key is written by another agent, you receive a notification "
        "between tool rounds — no polling needed. Use this to react to "
        "persistent data changes (e.g. watch 'sources/*' for new research "
        "briefs, 'feedback/*' for revision requests). Set up watches once "
        "during setup or your first heartbeat — they persist across sessions. "
        "For one-time event signals, use subscribe_event instead."
    ),
    parameters={
        "pattern": {
            "type": "string",
            "description": "Glob pattern for keys to watch (e.g. 'tasks/*', 'status/*', 'research/*')",
        },
    },
)
async def watch_blackboard(pattern: str, *, mesh_client=None) -> dict:
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    if mesh_client.is_standalone:
        return {"error": _STANDALONE_ERROR}
    try:
        result = await mesh_client.watch_blackboard(pattern)
        return {"watching": True, "pattern": pattern, **result}
    except Exception as e:
        return {"error": f"Failed to watch '{pattern}': {e}"}


@skill(
    name="claim_task",
    description=(
        "Atomically claim a task from the shared blackboard. Reads the current "
        "entry, then attempts a compare-and-swap write with your claim value. "
        "If another agent claimed it first, returns {claimed: false}. Use this "
        "to prevent duplicate work when multiple agents might pick up the same task."
    ),
    parameters={
        "key": {
            "type": "string",
            "description": "Blackboard key of the task to claim (e.g. 'tasks/pending/research_acme')",
        },
        "claim_value": {
            "type": "string",
            "description": "JSON string of the claim value (e.g. '{\"status\": \"claimed\", \"claimed_by\": \"me\"}')",
        },
    },
)
async def claim_task(key: str, claim_value: str, *, mesh_client=None) -> dict:
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    if mesh_client.is_standalone:
        return {"error": _STANDALONE_ERROR}
    parsed = _parse_json_value(claim_value)
    try:
        # Read current entry to get its version
        entry = await mesh_client.read_blackboard(key)
        if entry is None:
            return {"claimed": False, "reason": f"Key '{key}' does not exist"}
        current_version = entry.get("version", 1)
        result = await mesh_client.claim_blackboard(key, parsed, current_version)
        if result is None:
            return {"claimed": False, "reason": "Version conflict — another agent claimed it first"}
        return {"claimed": True, "key": key, "version": result.get("version", 0)}
    except Exception as e:
        return {"error": f"Failed to claim '{key}': {e}"}


@skill(
    name="save_artifact",
    description=(
        "Save a deliverable file (report, export, code) to your workspace "
        "artifacts. Artifacts are visible to the user and shared with teammates "
        "in multi-agent projects. Use this for finished output you want to "
        "deliver — NOT for intermediate working files (use write_file for those)."
    ),
    parameters={
        "name": {
            "type": "string",
            "description": "Artifact name (e.g. 'market_report.md', 'analysis.json')",
        },
        "content": {
            "type": "string",
            "description": "File content to save",
        },
        "description": {
            "type": "string",
            "description": "Short description of the artifact for teammates and the user",
            "default": "",
        },
    },
)
async def save_artifact(
    name: str, content: str, description: str = "",
    *, mesh_client=None, workspace_manager=None,
) -> dict:
    if workspace_manager is None:
        return {"error": "No workspace_manager available"}
    try:
        artifacts_dir = Path(workspace_manager.root) / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        filepath = (artifacts_dir / name).resolve()
        if not filepath.is_relative_to(artifacts_dir.resolve()):
            return {"error": f"Invalid artifact name: {name}"}
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content)

        # Register on blackboard (skip for standalone agents — no project blackboard)
        if mesh_client and not mesh_client.is_standalone:
            agent_id = mesh_client.agent_id
            key = f"artifacts/{agent_id}/{name}"
            meta = {
                "path": str(filepath),
                "size": len(content),
                "name": name,
            }
            if description:
                meta["description"] = description
            await mesh_client.write_blackboard(key, meta)

        return {"saved": True, "path": str(filepath), "name": name}
    except Exception as e:
        return {"error": f"Failed to save artifact '{name}': {e}"}


@skill(
    name="set_cron",
    description=(
        "Schedule a recurring job. Three modes:\n"
        "1. TOOL MODE (tool_name set) — invoke a tool directly each tick, no LLM, zero token cost. "
        "Use for deterministic actions (polling, notifications, blackboard writes).\n"
        "2. MESSAGE MODE (message set) — wake you with a message each tick for LLM processing. "
        "Use when the action needs reasoning. Costs API credits.\n"
        "3. HEARTBEAT MODE (heartbeat=true) — update your autonomous wakeup schedule. "
        "Only change if the USER explicitly asks.\n"
        "Schedules: cron ('0 9 * * 1-5') or intervals ('every 30m'). "
        "tool_name and message are mutually exclusive."
    ),
    parameters={
        "schedule": {
            "type": "string",
            "description": "Cron expression or interval (e.g. '0 9 * * 1-5', 'every 30m', 'every 5s')",
        },
        "tool_name": {
            "type": "string",
            "description": (
                "Name of a tool to invoke directly on each tick (no LLM). "
                "Mutually exclusive with message. Use for deterministic, "
                "low-cost recurring actions."
            ),
            "default": "",
        },
        "tool_params": {
            "type": "string",
            "description": (
                "JSON string of params to pass to the tool (e.g. '{\"key\": \"value\"}'). "
                "Only used when tool_name is set. Omit or pass '{}' for no params."
            ),
            "default": "{}",
        },
        "message": {
            "type": "string",
            "description": (
                "Message the mesh will send you on each trigger (LLM processes it). "
                "Mutually exclusive with tool_name. Use when the action requires "
                "reasoning or dynamic context."
            ),
            "default": "",
        },
        "heartbeat": {
            "type": "boolean",
            "description": (
                "If true, updates your autonomous heartbeat wakeup schedule. "
                "Only change if the USER explicitly asks — each heartbeat costs API credits."
            ),
            "default": False,
        },
    },
)
async def set_cron(
    schedule: str,
    tool_name: str = "",
    tool_params: str = "{}",
    message: str = "",
    heartbeat: bool = False,
    *, mesh_client=None,
) -> dict:
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    if tool_name and message:
        return {"error": "tool_name and message are mutually exclusive — use one or the other"}
    try:
        if heartbeat:
            jobs = await mesh_client.list_cron()
            existing = next((j for j in jobs if j.get("heartbeat")), None)
            if existing:
                result = await mesh_client.update_cron(existing["id"], schedule=schedule)
                return {"updated": True, "type": "heartbeat", **result}
            result = await mesh_client.create_cron(
                schedule=schedule, message=message or "heartbeat", heartbeat=True,
            )
            return {"created": True, "type": "heartbeat", **result}

        if tool_name:
            # Normalise: treat "{}" as no params (cleaner stored state)
            raw = tool_params.strip() if tool_params else ""
            params_str: str | None = None
            if raw and raw != "{}":
                try:
                    json.loads(raw)
                except json.JSONDecodeError:
                    return {"error": "tool_params must be valid JSON (e.g. '{\"key\": \"value\"}')"}
                params_str = raw
            result = await mesh_client.create_cron(
                schedule=schedule,
                tool_name=tool_name,
                tool_params=params_str,
            )
            return {"created": True, "type": "tool", **result}

        if not message:
            return {"error": "message is required when tool_name is not set (and heartbeat is false)"}
        result = await mesh_client.create_cron(schedule=schedule, message=message)
        return {"created": True, "type": "message", **result}
    except Exception as e:
        return {"error": f"Failed to create cron job: {e}"}


@skill(
    name="list_cron",
    description=(
        "List your scheduled cron jobs. Returns each job's ID, schedule, "
        "message, run count, and last run time."
    ),
    parameters={},
)
async def list_cron(*, mesh_client=None) -> dict:
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    try:
        jobs = await mesh_client.list_cron()
        return {"jobs": jobs, "count": len(jobs)}
    except Exception as e:
        return {"error": f"Failed to list cron jobs: {e}"}


@skill(
    name="remove_cron",
    description="Remove one of your scheduled cron jobs by its ID.",
    parameters={
        "job_id": {
            "type": "string",
            "description": "The cron job ID to remove",
        },
    },
)
async def remove_cron(job_id: str, *, mesh_client=None) -> dict:
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    try:
        result = await mesh_client.remove_cron(job_id)
        return {"removed": True, **result}
    except Exception as e:
        return {"error": f"Failed to remove cron job: {e}"}


@skill(
    name="spawn_fleet_agent",
    description=(
        "Spawn a new agent in its own isolated container with independent tools, "
        "memory, and environment. The agent joins the fleet as a peer — it has "
        "its own browser, filesystem, and mesh identity. Automatically cleaned "
        "up after TTL. Use this for work that needs isolation, different tools, "
        "or long-running autonomy. For quick parallel subtasks within your own "
        "container, use spawn_subagent instead."
    ),
    parameters={
        "role": {
            "type": "string",
            "description": "Role/specialty for the new agent (e.g. 'researcher', 'analyst')",
        },
        "system_prompt": {
            "type": "string",
            "description": "Instructions for the new agent",
            "default": "",
        },
        "ttl": {
            "type": "integer",
            "description": "Time-to-live in seconds (default 3600 = 1 hour)",
            "default": 3600,
        },
    },
)
async def spawn_agent(
    role: str, system_prompt: str = "", ttl: int = 3600, *, mesh_client=None,
) -> dict:
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    try:
        result = await mesh_client.spawn_agent(
            role=role, system_prompt=system_prompt, ttl=ttl,
        )
        return {"spawned": True, **result}
    except Exception as e:
        return {"error": f"Failed to spawn agent: {e}"}


@skill(
    name="read_agent_history",
    description=(
        "Read another agent's workspace daily logs to understand their recent "
        "activity — tasks worked on, tools called, and facts learned. Use this "
        "to get context before coordinating with another agent. Permission-checked: "
        "you can only read agents you're allowed to message."
    ),
    parameters={
        "agent_id": {
            "type": "string",
            "description": "ID of the agent whose history to read",
        },
    },
)
async def read_agent_history(agent_id: str, *, mesh_client=None) -> dict:
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    try:
        result = await mesh_client.get_agent_history(agent_id)
        return _sanitize_value(result)
    except Exception as e:
        return {"error": f"Failed to read history of '{agent_id}': {e}"}


@skill(
    name="get_agent_profile",
    description=(
        "Read another agent's public profile — mesh-verified metadata plus "
        "their collaboration interface. Use this to understand HOW to work "
        "with another agent: what inputs they expect, what they produce, "
        "what events they listen to, and their current status. More detailed "
        "than list_agents — call this when you need to coordinate with a "
        "specific peer. Permission-checked: you can only read profiles of "
        "agents you're allowed to message."
    ),
    parameters={
        "agent_id": {
            "type": "string",
            "description": "ID of the agent whose profile to read",
        },
    },
)
async def get_agent_profile(agent_id: str, *, mesh_client=None) -> dict:
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    try:
        result = await mesh_client.get_agent_profile(agent_id)
        return _sanitize_value(result)
    except Exception as e:
        return {"error": f"Failed to read profile of '{agent_id}': {e}"}


@skill(
    name="update_workspace",
    description=(
        "Update a workspace file to improve across sessions. These persist and "
        "shape your future behavior.\n"
        "- SOUL.md: identity, communication style, behavioral principles\n"
        "- INSTRUCTIONS.md: procedures, workflow rules, domain knowledge\n"
        "- USER.md: user preferences, corrections, project context\n"
        "- HEARTBEAT.md: autonomous wakeup rules and checks\n"
        "- INTERFACE.md: public collaboration contract for other agents\n"
        "Update when you discover something lasting, not every turn. "
        "Always read_file first — merge new knowledge, don't overwrite. "
        "When errors repeat, distill the pattern into INSTRUCTIONS.md."
    ),
    parameters={
        "filename": {
            "type": "string",
            "enum": ["SOUL.md", "INSTRUCTIONS.md", "USER.md", "HEARTBEAT.md", "INTERFACE.md"],
            "description": (
                "File to update: SOUL.md (identity/tone), INSTRUCTIONS.md "
                "(procedures/rules), USER.md (user prefs), HEARTBEAT.md "
                "(autonomous rules), or INTERFACE.md (public collaboration contract)"
            ),
        },
        "content": {
            "type": "string",
            "description": "New content for the file (replaces existing content)",
        },
    },
)
async def update_workspace(
    filename: str, content: str, *, workspace_manager=None, mesh_client=None,
) -> dict:
    if workspace_manager is None:
        return {"error": "No workspace_manager available"}
    content = sanitize_for_prompt(content)

    # Capture old content for diff summary (single read)
    old_content = workspace_manager._read_file(filename) or ""

    result = workspace_manager.update_file(filename, content)
    if result.get("error"):
        return result

    # Notify the user with a meaningful summary of what changed
    if mesh_client:
        try:
            agent_id = getattr(mesh_client, "agent_id", "unknown")
            old_stripped = old_content.strip()
            if old_stripped == content.strip():
                summary = f"[{agent_id}] Re-saved {filename} (no changes)."
            elif not old_stripped or old_stripped in (
                "# Heartbeat Rules",
                "# User Context",
                "# Identity",
                "# Instructions",
                "# Long-Term Memory",
                "# Interface",
            ) or old_stripped.startswith((
                "# Heartbeat Rules\n\nYou are woken",
                "# User Context\n\nYour user",
                "# Identity\n\nPersonality, tone",
                "# Instructions\n\nOperating procedures",
            )):
                summary = f"[{agent_id}] Initialized {filename} with custom content."
            else:
                summary = f"[{agent_id}] Updated {filename} based on what I've learned."
            await mesh_client.notify_user(summary)
        except Exception as e:
            logger.debug("Non-fatal: workspace notification failed: %s", e)
    # Warn if content exceeds bootstrap cap (file is saved in full,
    # but only the capped portion loads into the system prompt).
    from src.agent.workspace import BOOTSTRAP_CAPS
    cap = BOOTSTRAP_CAPS.get(filename)
    if cap and len(content) > cap:
        result["bootstrap_cap"] = cap
        result["content_length"] = len(content)
        result["over_cap"] = len(content) - cap
        result["warning"] = (
            f"Saved all {len(content)} chars, but only the first {cap} "
            f"will load into your active context. The last {len(content) - cap} "
            f"chars won't guide your behavior. Consider trimming."
        )
    return result
