"""Mesh interaction tools: shared state, fleet awareness, artifacts.

Framework-level skills available to every agent. Agents coordinate through
the shared blackboard (not through direct conversations with each other).
"""

from __future__ import annotations

import json

from src.agent.skills import skill


@skill(
    name="list_agents",
    description=(
        "List all agents currently running in the fleet. Returns each agent's "
        "name and capabilities. Use this to discover who else is working and "
        "what artifacts they may have published to the blackboard."
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
                entry["capabilities"] = info.get("capabilities", [])
            agents.append(entry)
        return {"agents": agents, "count": len(agents)}
    except Exception as e:
        return {"error": f"Failed to list agents: {e}"}


@skill(
    name="read_shared_state",
    description=(
        "Read a value from the shared blackboard. The blackboard is the team's "
        "shared memory — any agent can write data that others can read. "
        "Keys are hierarchical: context/market_analysis, goals/researcher, signals/urgent. "
        "Returns null if the key does not exist."
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
        "Write a value to the shared blackboard for other agents to read. "
        "Use hierarchical keys: goals/ for objectives, context/ for shared "
        "knowledge, signals/ for alerts. Values must be JSON-serializable. "
        "Example: write_shared_state(key='goals/engineer', value='{\"priority\": "
        "\"fix auth bug\"}')"
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
        "List all entries on the shared blackboard matching a key prefix. "
        "Use this to browse what's available: list_shared_state(prefix='goals/') "
        "to see all goals, or list_shared_state(prefix='context/') for shared context."
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
        filepath = artifacts_dir / name
        filepath.write_text(content)

        if mesh_client:
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
        "(e.g. 'every 30m'). Returns the job ID."
    ),
    parameters={
        "schedule": {
            "type": "string",
            "description": "Cron expression or interval (e.g. '*/5 * * * *', 'every 30m')",
        },
        "message": {
            "type": "string",
            "description": "Message the mesh will send you on each trigger",
        },
    },
)
async def set_cron(schedule: str, message: str, *, mesh_client=None) -> dict:
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    try:
        result = await mesh_client.create_cron(schedule=schedule, message=message)
        return {"created": True, **result}
    except Exception as e:
        return {"error": f"Failed to create cron job: {e}"}


@skill(
    name="set_heartbeat",
    description=(
        "Enable a heartbeat for yourself. The mesh will periodically run cheap "
        "deterministic probes (disk usage, pending tasks, signals) and only wake "
        "you with an LLM call if something needs attention. Much more efficient "
        "than a regular cron for autonomous monitoring. Define your monitoring "
        "rules in HEARTBEAT.md in your workspace."
    ),
    parameters={
        "schedule": {
            "type": "string",
            "description": "How often to probe (e.g. 'every 15m', 'every 1h', '*/30 * * * *')",
        },
    },
)
async def set_heartbeat(schedule: str, *, mesh_client=None) -> dict:
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    try:
        result = await mesh_client.create_cron(
            schedule=schedule, message="heartbeat", heartbeat=True,
        )
        return {"created": True, "type": "heartbeat", **result}
    except Exception as e:
        return {"error": f"Failed to create heartbeat: {e}"}


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
        "Read another agent's conversation history (daily logs). Use this to "
        "understand what another agent has been doing, what it learned, and "
        "what context it has. Permission-checked — you can only read agents "
        "you're allowed to message."
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


def _preview(value: dict, max_len: int = 200) -> str:
    text = json.dumps(value, default=str)
    return text[:max_len] + "..." if len(text) > max_len else text
