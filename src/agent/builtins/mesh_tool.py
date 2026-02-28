"""Mesh interaction tools: shared state, fleet awareness, artifacts.

Framework-level skills available to every agent. Agents coordinate through
the shared blackboard (not through direct conversations with each other).
"""

from __future__ import annotations

import json

from src.agent.skills import skill
from src.shared.utils import setup_logging

logger = setup_logging("agent.builtins.mesh_tool")


@skill(
    name="notify_user",
    description=(
        "Send a notification to the user across all connected channels "
        "(CLI, Telegram, Discord, Slack, etc.). This is your PRIMARY way "
        "to report back to the user when working autonomously (heartbeat, "
        "cron jobs, long-running tasks). Use it for progress updates, "
        "completed work summaries, errors needing attention, or anything "
        "the user should know about. Keep messages concise and actionable. "
        "Do NOT write user-facing updates to the blackboard — the blackboard "
        "is for agent-to-agent collaboration only."
    ),
    parameters={
        "message": {
            "type": "string",
            "description": "The notification message to send to the user",
        },
    },
)
async def notify_user(message: str, *, mesh_client=None) -> dict:
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    try:
        await mesh_client.notify_user(message)
        return {"sent": True}
    except Exception as e:
        return {"error": f"Failed to notify user: {e}"}


@skill(
    name="list_agents",
    description=(
        "List agents in your project (or just yourself if standalone). Returns "
        "each agent's name and role. Use this to discover who else is working "
        "in your project and what artifacts they may have published to the blackboard."
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
    "Blackboard is not available — this agent is not assigned to any project. "
    "Use memory_save/memory_search for private storage."
)


@skill(
    name="read_shared_state",
    description=(
        "Read a value from the shared blackboard. The blackboard is for "
        "agent-to-agent collaboration — data that other agents have shared "
        "for you to act on. Keys are hierarchical: context/market_analysis, "
        "goals/researcher, signals/urgent. Returns null if the key does not exist."
    ),
    parameters={
        "key": {
            "type": "string",
            "description": "Blackboard key to read (e.g. 'goals/researcher', 'context/market')",
        },
    },
)
async def read_shared_state(key: str, *, mesh_client=None) -> dict:
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    if mesh_client.is_standalone:
        return {"error": _STANDALONE_ERROR}
    try:
        entry = await mesh_client.read_blackboard(key)
        if entry is None:
            return {"key": key, "exists": False, "value": None}
        return {"key": key, "exists": True, "value": entry.get("value", entry)}
    except Exception as e:
        return {"error": f"Failed to read '{key}': {e}"}


@skill(
    name="write_shared_state",
    description=(
        "Write a value to the shared blackboard for OTHER AGENTS to read. "
        "The blackboard is for agent-to-agent collaboration only — do NOT "
        "use it for status updates or reporting to the user (use notify_user "
        "or chat responses for that). Write here when another agent needs "
        "specific data to do their work. Use hierarchical keys: goals/ for "
        "objectives, context/ for shared knowledge, signals/ for alerts, "
        "tasks/ for work requests. Values must be JSON-serializable."
    ),
    parameters={
        "key": {
            "type": "string",
            "description": "Blackboard key (e.g. 'goals/engineer', 'context/market')",
        },
        "value": {
            "type": "string",
            "description": "JSON string of the value to store",
        },
    },
)
async def write_shared_state(key: str, value: str, *, mesh_client=None) -> dict:
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    if mesh_client.is_standalone:
        return {"error": _STANDALONE_ERROR}
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError:
        parsed = {"text": value}
    try:
        result = await mesh_client.write_blackboard(key, parsed)
        return {"key": key, "written": True, "version": result.get("version", 1)}
    except Exception as e:
        return {"error": f"Failed to write '{key}': {e}"}


@skill(
    name="list_shared_state",
    description=(
        "Discover what's on the shared blackboard by listing entries matching a key "
        "prefix. Returns key names, authors, timestamps, and value previews — but "
        "NOT full values (use read_shared_state for that). Use when you don't know "
        "the exact key: prefix='tasks/' to find tasks, prefix='' to see everything."
    ),
    parameters={
        "prefix": {
            "type": "string",
            "description": "Key prefix to filter by (e.g. 'goals/', 'context/', 'signals/')",
        },
    },
)
async def list_shared_state(prefix: str, *, mesh_client=None) -> dict:
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    if mesh_client.is_standalone:
        return {"error": _STANDALONE_ERROR}
    try:
        entries = await mesh_client.list_blackboard(prefix)
        items = []
        for entry in entries:
            items.append({
                "key": entry.get("key", ""),
                "written_by": entry.get("written_by", ""),
                "updated_at": entry.get("updated_at", ""),
                "value_preview": _preview(entry.get("value", {})),
            })
        return {"prefix": prefix, "entries": items, "count": len(items)}
    except Exception as e:
        return {"error": f"Failed to list '{prefix}': {e}"}


@skill(
    name="publish_event",
    description=(
        "Publish an event to the mesh pub/sub system. Other agents subscribed "
        "to the topic will receive the event. Use this for broadcasting updates "
        "like 'research_complete' or 'build_passed'."
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
    try:
        parsed = json.loads(data) if isinstance(data, str) else data
    except json.JSONDecodeError:
        parsed = {"text": data}
    try:
        result = await mesh_client.publish_event(topic, parsed)
        return {"published": True, "topic": topic, **result}
    except Exception as e:
        return {"error": f"Failed to publish to '{topic}': {e}"}


@skill(
    name="subscribe_event",
    description=(
        "Subscribe to a pub/sub topic at runtime. Once subscribed, events "
        "published to this topic by other agents will arrive as steer "
        "messages between your tool rounds — you don't need to poll. "
        "Use this to react to coordination signals like 'research_complete' "
        "or 'deploy_ready'."
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
    try:
        result = await mesh_client.subscribe_topic(topic)
        return {"subscribed": True, "topic": topic, **result}
    except Exception as e:
        return {"error": f"Failed to subscribe to '{topic}': {e}"}


@skill(
    name="watch_blackboard",
    description=(
        "Watch blackboard keys matching a glob pattern. When any matching "
        "key is written by another agent, you'll receive a notification "
        "between tool rounds — no polling needed. Use this to react "
        "when shared state changes (e.g. watch 'tasks/*' to know when "
        "new tasks appear)."
    ),
    parameters={
        "pattern": {
            "type": "string",
            "description": "Glob pattern for keys to watch (e.g. 'tasks/*', 'context/research_*')",
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
    try:
        parsed = json.loads(claim_value) if isinstance(claim_value, str) else claim_value
    except json.JSONDecodeError:
        parsed = {"text": claim_value}
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
        "Save a named output file to your workspace and register it on the "
        "shared blackboard so other agents can find it. Use this for deliverables "
        "like reports, code files, or data exports."
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
    },
)
async def save_artifact(
    name: str, content: str, *, mesh_client=None, workspace_manager=None,
) -> dict:
    if workspace_manager is None:
        return {"error": "No workspace_manager available"}
    try:
        from pathlib import Path
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
            await mesh_client.write_blackboard(key, {
                "path": str(filepath),
                "size": len(content),
                "name": name,
            })

        return {"saved": True, "path": str(filepath), "name": name}
    except Exception as e:
        return {"error": f"Failed to save artifact '{name}': {e}"}


@skill(
    name="set_cron",
    description=(
        "Schedule a recurring job for yourself. The mesh will send you the "
        "specified message on the given schedule. Use cron syntax "
        "(e.g. '0 9 * * 1-5' for weekdays at 9 AM) or natural intervals "
        "(e.g. 'every 30m'). Set heartbeat=true to update your autonomous "
        "wakeup schedule instead."
    ),
    parameters={
        "schedule": {
            "type": "string",
            "description": "Cron expression or interval (e.g. '*/5 * * * *', 'every 30m')",
        },
        "message": {
            "type": "string",
            "description": "Message the mesh will send you on each trigger",
            "default": "",
        },
        "heartbeat": {
            "type": "boolean",
            "description": (
                "If true, sets your autonomous heartbeat schedule (finds and "
                "updates existing heartbeat, or creates one). Only change if "
                "the USER explicitly asks — each heartbeat costs API credits."
            ),
            "default": False,
        },
    },
)
async def set_cron(
    schedule: str, message: str = "", heartbeat: bool = False,
    *, mesh_client=None,
) -> dict:
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    try:
        if heartbeat:
            # Check for existing heartbeat job and update it
            jobs = await mesh_client.list_cron()
            existing = next((j for j in jobs if j.get("heartbeat")), None)
            if existing:
                result = await mesh_client.update_cron(
                    existing["id"], schedule=schedule,
                )
                return {"updated": True, "type": "heartbeat", **result}
            # No existing heartbeat — create one
            result = await mesh_client.create_cron(
                schedule=schedule,
                message=message or "heartbeat",
                heartbeat=True,
            )
            return {"created": True, "type": "heartbeat", **result}
        # Regular cron job
        if not message:
            return {"error": "message is required for non-heartbeat cron jobs"}
        result = await mesh_client.create_cron(schedule=schedule, message=message)
        return {"created": True, **result}
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
    name="spawn_agent",
    description=(
        "Spawn a new ephemeral agent to help with a specific task. The agent "
        "runs in its own isolated container and is automatically cleaned up "
        "after the TTL expires. Use this to delegate specialized work."
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
        return result
    except Exception as e:
        return {"error": f"Failed to read history of '{agent_id}': {e}"}


@skill(
    name="update_workspace",
    description=(
        "Update one of your writable workspace files to get better over time. "
        "These files persist across sessions and shape your future behavior.\n\n"
        "- SOUL.md: your identity — communication style, tone, behavioral "
        "principles. Refine based on user feedback about how you interact.\n"
        "- INSTRUCTIONS.md: your operating manual — procedures, workflow "
        "rules, tool patterns, domain knowledge. Update when you discover "
        "better approaches or learn new domain constraints.\n"
        "- USER.md: your user's context — their preferences, communication "
        "style, project background, and important facts so you serve them "
        "better in future sessions.\n"
        "- HEARTBEAT.md: your autonomous rules — what to check and do on "
        "periodic wakeups. Drop wasteful checks, add useful ones.\n\n"
        "Update these when you discover something lasting, not every turn. "
        "Read the current content first (via read_file) to avoid losing "
        "existing information — merge new knowledge in, don't overwrite blindly. "
        "A backup is saved automatically."
    ),
    parameters={
        "filename": {
            "type": "string",
            "enum": ["SOUL.md", "INSTRUCTIONS.md", "USER.md", "HEARTBEAT.md"],
            "description": (
                "File to update: SOUL.md (identity/tone), INSTRUCTIONS.md "
                "(procedures/rules), USER.md (user prefs), or HEARTBEAT.md "
                "(autonomous rules)"
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
    from src.shared.utils import sanitize_for_prompt
    content = sanitize_for_prompt(content)

    # Capture old content for diff summary (single read)
    old_content = workspace_manager._read_file(filename) or ""

    result = workspace_manager.update_file(filename, content)
    if result.get("error"):
        return result

    # Notify the user with a meaningful summary of what changed
    if mesh_client:
        try:
            agent_id = getattr(mesh_client, "agent_id", "agent")
            if old_content.strip() == content.strip():
                summary = f"[{agent_id}] Re-saved {filename} (no changes)."
            elif not old_content.strip() or old_content.strip().startswith((
                "# Heartbeat Rules\n\nYou are woken",
                "# User Context\n\nRecord user",
                "# User Context\n\nYour user",
                "# Identity\n\nDefine personality",
                "# Identity\n\nPersonality, tone",
                "# Agent Instructions\n\nAdd operating",
                "# Instructions\n\nOperating procedures",
            )):
                summary = f"[{agent_id}] Initialized {filename} with custom content."
            else:
                summary = f"[{agent_id}] Updated {filename} based on what I've learned."
            await mesh_client.notify_user(summary)
        except Exception as e:
            logger.debug("Non-fatal: workspace notification failed: %s", e)
    return result


def _preview(value: dict, max_len: int = 200) -> str:
    text = json.dumps(value, default=str)
    return text[:max_len] + "..." if len(text) > max_len else text
