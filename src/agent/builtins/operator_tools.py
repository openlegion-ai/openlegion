"""Operator agent tools -- propose/confirm edits, observations, agent/project management."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone


from src.agent.skills import skill
from src.shared.utils import setup_logging

logger = setup_logging("agent.builtins.operator_tools")

def _is_operator() -> bool:
    """Defence-in-depth: only the operator agent has ALLOWED_TOOLS set.

    Non-operator agents should never execute these tools even if they
    appear in the tool list via auto-discovery.  Evaluated at call time
    so env changes (and test overrides) are respected.
    """
    return os.environ.get("ALLOWED_TOOLS", "") != ""

# Permission ceiling: operator cannot grant permissions beyond these limits
_OPERATOR_PERMISSION_CEILING = {
    "can_use_browser": True,
    "can_spawn": False,       # Created agents can't spawn others
    "can_manage_cron": True,
    "can_use_wallet": False,  # Requires explicit user setup
    "blackboard_read": ["*"],
    "blackboard_write": ["tasks/*", "context/*", "status/*"],
}

_VALID_FIELDS = frozenset({
    "instructions", "soul", "model", "role", "heartbeat",
    "thinking", "budget", "permissions",
})

_OPERATOR_AGENT_ID = "operator"


@skill(
    name="propose_edit",
    description=(
        "Propose a change to an agent's configuration. Returns a preview diff "
        "and change_id for confirmation. Always show the preview to the user "
        "and wait for their approval before calling confirm_edit.\n\n"
        "Fields: instructions, soul, model, role, heartbeat, thinking, budget, permissions.\n"
        "Value format: string for text fields, object for budget/permissions.\n"
        "- budget: {\"daily_usd\": float, \"monthly_usd\": float}\n"
        "- permissions: {\"can_use_browser\": bool, ...}\n"
        "- thinking: \"off\", \"low\", \"medium\", \"high\"\n"
        "- model: e.g. \"anthropic/claude-sonnet-4-20250514\""
    ),
    parameters={
        "agent_id": {
            "type": "string",
            "description": "Target agent ID (use list_agents to find IDs)",
        },
        "field": {
            "type": "string",
            "description": "Config field to change",
            "enum": [
                "instructions", "soul", "model", "role", "heartbeat",
                "thinking", "budget", "permissions",
            ],
        },
        "value": {
            "type": ["string", "object"],
            "description": "New value for the field",
        },
    },
)
async def propose_edit(agent_id: str, field: str, value, *, mesh_client=None, **_kw) -> dict:
    """Propose a config change for an agent. Returns preview for user review."""
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if mesh_client is None:
        return {"error": "No mesh_client available"}

    # Self-modification prevention
    if agent_id.lower() == _OPERATOR_AGENT_ID:
        return {
            "error": (
                "Cannot modify the operator agent. "
                "Use the dashboard to change operator settings."
            ),
        }

    # Validate field
    if field not in _VALID_FIELDS:
        return {
            "error": (
                f"Invalid field '{field}'. "
                f"Must be one of: {sorted(_VALID_FIELDS)}"
            ),
        }

    # Permission ceiling validation
    if field == "permissions" and isinstance(value, dict):
        for key, max_val in _OPERATOR_PERMISSION_CEILING.items():
            if key not in value:
                continue
            if isinstance(max_val, bool):
                if value[key] and not max_val:
                    return {
                        "error": (
                            f"Permission ceiling exceeded: '{key}' cannot be set "
                            "to True by the operator. Use the dashboard for "
                            "advanced permissions."
                        ),
                    }
            elif isinstance(max_val, list):
                requested = set(value.get(key, []))
                allowed = set(max_val)
                if "*" not in allowed and not requested.issubset(allowed):
                    excess = requested - allowed
                    return {
                        "error": (
                            f"Permission ceiling exceeded: '{key}' patterns "
                            f"{excess} exceed allowed {allowed}. Use the "
                            "dashboard for advanced permissions."
                        ),
                    }

    # Budget validation
    if field == "budget" and isinstance(value, dict):
        daily = value.get("daily_usd", 0)
        monthly = value.get("monthly_usd", 0)
        if not isinstance(daily, (int, float)) or not (0.01 <= daily <= 1000):
            return {"error": f"daily_usd must be 0.01-1000, got {daily}"}
        if not isinstance(monthly, (int, float)) or not (0.10 <= monthly <= 30000):
            return {"error": f"monthly_usd must be 0.10-30000, got {monthly}"}

    # Thinking validation
    if field == "thinking" and value not in ("off", "low", "medium", "high"):
        return {
            "error": (
                f"thinking must be 'off', 'low', 'medium', or 'high', "
                f"got '{value}'"
            ),
        }

    try:
        result = await mesh_client.propose_config_change(agent_id, field, value)
        return result
    except Exception as e:
        return {"error": f"Failed to propose edit: {e}"}


@skill(
    name="confirm_edit",
    description=(
        "Apply a previously proposed agent config change. Only call this after "
        "the user has seen the preview and explicitly confirmed. Will fail if "
        "the user has not confirmed in the conversation."
    ),
    parameters={
        "change_id": {
            "type": "string",
            "description": "Change ID from propose_edit",
        },
    },
)
async def confirm_edit(change_id: str, *, mesh_client=None, _messages=None, **_kw) -> dict:
    """Apply a proposed config change after user confirmation."""
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if mesh_client is None:
        return {"error": "No mesh_client available"}

    # Provenance check: require user confirmation
    from src.agent.loop import _last_message_is_user_origin

    if _messages is not None and not _last_message_is_user_origin(_messages):
        return {
            "error": "provenance_check_failed",
            "detail": (
                "User confirmation required. Please show the proposed change "
                "to the user and ask them to confirm before calling confirm_edit."
            ),
        }

    try:
        result = await mesh_client.confirm_config_change(change_id)
        return result
    except Exception as e:
        error_str = str(e)
        if "404" in error_str or "not found" in error_str.lower():
            return {
                "error": "change_expired_or_lost",
                "detail": (
                    "The proposed change was not found (expired or server "
                    "restarted). Please call propose_edit again."
                ),
            }
        return {"error": f"Failed to confirm edit: {e}"}


# ── Observations ─────────────────────────────────────────────


_MAX_OBSERVATIONS_CHARS = 1500
_MAX_HISTORY_ENTRIES = 50


@skill(
    name="save_observations",
    description=(
        "Save fleet health observations from your monitoring check. "
        "Writes structured data to OBSERVATIONS.md for the Fleet Digest display."
    ),
    parameters={
        "fleet_summary": {
            "type": "string",
            "description": "One-line fleet health summary (e.g. '5/6 healthy, cost stable')",
        },
        "agents_attention": {
            "type": "array",
            "description": "Agents needing attention: [{agent_id, issue, severity}]",
            "items": {"type": "object"},
            "default": [],
        },
        "cost_trend": {
            "type": "string",
            "description": "Cost trend (e.g. 'up_40pct', 'stable', 'down_15pct')",
        },
        "notes": {
            "type": "string",
            "description": "Optional freeform notes",
            "default": "",
        },
    },
)
async def save_observations(
    fleet_summary: str,
    cost_trend: str,
    agents_attention: list | None = None,
    notes: str = "",
    *,
    workspace_manager=None,
    **_kw,
) -> dict:
    """Save fleet observations to workspace for dashboard display."""
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if workspace_manager is None:
        return {"error": "No workspace_manager available"}

    timestamp = datetime.now(timezone.utc).isoformat()

    obs = {
        "timestamp": timestamp,
        "fleet_summary": fleet_summary,
        "agents_attention": agents_attention or [],
        "cost_trend": cost_trend,
        "notes": notes,
    }

    # Build markdown with JSON block
    content = (
        f"# Fleet Observations\nUpdated: {timestamp}\n\n"
        f"```json\n{json.dumps(obs, indent=2)}\n```\n"
    )

    # Enforce char cap by truncating notes
    while len(content) > _MAX_OBSERVATIONS_CHARS and notes:
        notes = notes[:-50] + "..." if len(notes) > 50 else ""
        obs["notes"] = notes
        content = (
            f"# Fleet Observations\nUpdated: {timestamp}\n\n"
            f"```json\n{json.dumps(obs, indent=2)}\n```\n"
        )

    # Write OBSERVATIONS.md directly to workspace root (not in AGENT_WRITABLE)
    obs_path = workspace_manager.root / "OBSERVATIONS.md"
    obs_path.write_text(content)

    # Append to OBSERVATIONS_HISTORY.md (rolling window)
    history_path = workspace_manager.root / "OBSERVATIONS_HISTORY.md"
    history_content = ""
    if history_path.exists():
        try:
            history_content = history_path.read_text(errors="replace")
        except OSError:
            pass
    history_lines = [e for e in history_content.strip().split("\n---\n") if e.strip()]
    history_lines.append(json.dumps(obs))
    if len(history_lines) > _MAX_HISTORY_ENTRIES:
        history_lines = history_lines[-_MAX_HISTORY_ENTRIES:]
    history_path.write_text("\n---\n".join(history_lines) + "\n")

    return {"saved": True, "timestamp": timestamp, "chars": len(content)}


# ── Agent History ────────────────────────────────────────────


@skill(
    name="read_agent_history",
    description=(
        "Read an agent's activity history for a time period. "
        "Shows tasks completed, failures, and key events."
    ),
    parameters={
        "agent_id": {
            "type": "string",
            "description": "Agent ID to check",
        },
        "period": {
            "type": "string",
            "description": "Time period",
            "enum": ["today", "yesterday", "week"],
            "default": "today",
        },
    },
)
async def read_agent_history(
    agent_id: str,
    period: str = "today",
    *,
    mesh_client=None,
    **_kw,
) -> dict:
    """Read an agent's activity history via the mesh."""
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    try:
        response = await mesh_client._get_with_retry(
            f"{mesh_client.mesh_url}/mesh/agents/{agent_id}/history",
            params={"period": period},
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": f"Failed to read history: {e}"}


# ── Create Agent ─────────────────────────────────────────────


@skill(
    name="create_agent",
    description=(
        "Create a new custom agent with specified role, model, and instructions. "
        "Essential for Basic plan users who need a custom agent that doesn't "
        "match any template. Requires user confirmation."
    ),
    parameters={
        "name": {
            "type": "string",
            "description": "Agent ID (lowercase, alphanumeric + hyphens, 1-32 chars)",
        },
        "role": {
            "type": "string",
            "description": "Human-readable role description",
        },
        "model": {
            "type": "string",
            "description": "LLM model (optional, defaults to system default)",
            "default": "",
        },
        "instructions": {
            "type": "string",
            "description": "Initial instructions for the agent",
        },
        "soul": {
            "type": "string",
            "description": "Optional personality/identity",
            "default": "",
        },
    },
)
async def create_agent(
    name: str,
    role: str,
    instructions: str,
    model: str = "",
    soul: str = "",
    *,
    mesh_client=None,
    _messages=None,
    **_kw,
) -> dict:
    """Create a new custom agent. Provenance-gated."""
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if mesh_client is None:
        return {"error": "No mesh_client available"}

    # Provenance check
    from src.agent.loop import _last_message_is_user_origin

    if _messages is not None and not _last_message_is_user_origin(_messages):
        return {
            "error": "provenance_check_failed",
            "detail": "User confirmation required to create an agent.",
        }

    try:
        return await mesh_client.create_custom_agent(
            name, role, model, instructions, soul,
        )
    except Exception as e:
        error_str = str(e)
        if "409" in error_str:
            return {"error": f"Agent '{name}' already exists or plan limit reached."}
        return {"error": f"Failed to create agent: {e}"}


# ── Project Management ───────────────────────────────────────


@skill(
    name="list_projects",
    description="List all projects with their members and descriptions.",
    parameters={},
)
async def list_projects(*, mesh_client=None, **_kw) -> dict:
    """List all projects (read-only, no provenance gate)."""
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    try:
        return await mesh_client.list_projects()
    except Exception as e:
        return {"error": f"Failed to list projects: {e}"}


@skill(
    name="get_project",
    description="Get details for a specific project including members and description.",
    parameters={
        "project_name": {
            "type": "string",
            "description": "Project name",
        },
    },
)
async def get_project(project_name: str, *, mesh_client=None, **_kw) -> dict:
    """Get a single project's details (read-only, no provenance gate)."""
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    try:
        result = await mesh_client.list_projects()
        projects = result.get("projects", [])
        for p in projects:
            if p.get("name") == project_name:
                return p
        return {"error": f"Project '{project_name}' not found"}
    except Exception as e:
        return {"error": f"Failed to get project: {e}"}


@skill(
    name="create_project",
    description=(
        "Create a new project and optionally assign agents to it. "
        "Requires user confirmation."
    ),
    parameters={
        "name": {
            "type": "string",
            "description": "Project name",
        },
        "description": {
            "type": "string",
            "description": "Project brief / description",
        },
        "agent_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Agent IDs to assign to the project",
            "default": [],
        },
    },
)
async def create_project(
    name: str,
    description: str,
    agent_ids: list | None = None,
    *,
    mesh_client=None,
    _messages=None,
    **_kw,
) -> dict:
    """Create a new project. Provenance-gated."""
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if mesh_client is None:
        return {"error": "No mesh_client available"}

    from src.agent.loop import _last_message_is_user_origin

    if _messages is not None and not _last_message_is_user_origin(_messages):
        return {
            "error": "provenance_check_failed",
            "detail": "User confirmation required to create a project.",
        }

    try:
        return await mesh_client.create_project(
            name, description, agent_ids or [],
        )
    except Exception as e:
        return {"error": f"Failed to create project: {e}"}


@skill(
    name="add_agents_to_project",
    description="Add one or more agents to an existing project. Requires user confirmation.",
    parameters={
        "project_name": {
            "type": "string",
            "description": "Project name",
        },
        "agent_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Agent IDs to add",
        },
    },
)
async def add_agents_to_project(
    project_name: str,
    agent_ids: list,
    *,
    mesh_client=None,
    _messages=None,
    **_kw,
) -> dict:
    """Add agents to a project. Provenance-gated."""
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if mesh_client is None:
        return {"error": "No mesh_client available"}

    from src.agent.loop import _last_message_is_user_origin

    if _messages is not None and not _last_message_is_user_origin(_messages):
        return {
            "error": "provenance_check_failed",
            "detail": "User confirmation required to add agents to a project.",
        }

    results = []
    for aid in agent_ids:
        try:
            r = await mesh_client.add_agent_to_project(project_name, aid)
            results.append(r)
        except Exception as e:
            results.append({"agent": aid, "error": str(e)})
    return {"project": project_name, "results": results}


@skill(
    name="remove_agents_from_project",
    description="Remove one or more agents from a project. Requires user confirmation.",
    parameters={
        "project_name": {
            "type": "string",
            "description": "Project name",
        },
        "agent_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Agent IDs to remove",
        },
    },
)
async def remove_agents_from_project(
    project_name: str,
    agent_ids: list,
    *,
    mesh_client=None,
    _messages=None,
    **_kw,
) -> dict:
    """Remove agents from a project. Provenance-gated."""
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if mesh_client is None:
        return {"error": "No mesh_client available"}

    from src.agent.loop import _last_message_is_user_origin

    if _messages is not None and not _last_message_is_user_origin(_messages):
        return {
            "error": "provenance_check_failed",
            "detail": "User confirmation required to remove agents from a project.",
        }

    results = []
    for aid in agent_ids:
        try:
            r = await mesh_client.remove_agent_from_project(project_name, aid)
            results.append(r)
        except Exception as e:
            results.append({"agent": aid, "error": str(e)})
    return {"project": project_name, "results": results}


@skill(
    name="update_project_context",
    description=(
        "Update a project's description / context. "
        "Requires user confirmation."
    ),
    parameters={
        "project_name": {
            "type": "string",
            "description": "Project name",
        },
        "context": {
            "type": "string",
            "description": "New project description / context text",
        },
    },
)
async def update_project_context(
    project_name: str,
    context: str,
    *,
    mesh_client=None,
    _messages=None,
    **_kw,
) -> dict:
    """Update project description/context. Provenance-gated."""
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if mesh_client is None:
        return {"error": "No mesh_client available"}

    from src.agent.loop import _last_message_is_user_origin

    if _messages is not None and not _last_message_is_user_origin(_messages):
        return {
            "error": "provenance_check_failed",
            "detail": "User confirmation required to update project context.",
        }

    try:
        return await mesh_client.update_project_context(project_name, context)
    except Exception as e:
        return {"error": f"Failed to update project context: {e}"}
