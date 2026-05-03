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
    "blackboard_write": ["tasks/*", "context/*", "status/*", "output/*", "artifacts/*"],
}

_VALID_FIELDS = frozenset({
    "instructions", "soul", "model", "role", "heartbeat",
    "interface", "thinking", "budget", "permissions",
})

_OPERATOR_AGENT_ID = "operator"


@skill(
    name="propose_edit",
    description=(
        "Propose a change to an agent's config. Returns a preview diff and "
        "change_id; show the preview and get user approval before calling "
        "confirm_edit. See INSTRUCTIONS.md for field formats."
    ),
    parameters={
        "agent_id": {
            "type": "string",
            "description": "Target agent ID",
        },
        "field": {
            "type": "string",
            "description": "Config field to change",
            "enum": [
                "instructions", "soul", "model", "role", "heartbeat",
                "interface", "thinking", "budget", "permissions",
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

    if _messages is None or not _last_message_is_user_origin(_messages):
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
        # Task 2d migrated /mesh/config/confirm from 404 -> 400 with
        # "Pending action invalid or expired" for unknown/expired/
        # digest-mismatch/origin-failure rows. Match either form so
        # existing operator UX (re-propose) keeps working.
        lower = error_str.lower()
        if (
            "404" in error_str
            or "not found" in lower
            or "invalid or expired" in lower
        ):
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


# ── Create Agent ─────────────────────────────────────────────


@skill(
    name="create_agent",
    description=(
        "Create a new custom agent with role/model/instructions. "
        "Requires user confirmation."
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

    if _messages is None or not _last_message_is_user_origin(_messages):
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
    name="inspect_projects",
    description=(
        "Read project info. detail='names' lists name+description; "
        "detail='status' adds task-count rollups (requires v2). "
        "Setting project_name returns full detail for that project."
    ),
    parameters={
        "detail": {
            "type": "string",
            "description": "names | status | full",
            "enum": ["names", "status", "full"],
            "default": "names",
        },
        "project_name": {
            "type": "string",
            "description": "Optional — return full detail for this project only",
            "default": "",
        },
    },
)
async def inspect_projects(
    detail: str = "names",
    project_name: str = "",
    *,
    mesh_client=None,
    **_kw,
) -> dict:
    """Consolidated project read tool (replaces list_projects /
    get_project / list_project_status).
    """
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if mesh_client is None:
        return {"error": "No mesh_client available"}

    # Single-project lookup always returns full detail.
    if project_name:
        try:
            result = await mesh_client.list_projects()
        except Exception as e:
            return {"error": f"Failed to inspect project: {e}"}
        for p in result.get("projects", []):
            if p.get("name") == project_name:
                return p
        return {"error": f"Project '{project_name}' not found"}

    if detail == "status":
        if not _orchestration_v2_on():
            return {"error": _TASKS_V2_DISABLED}
        try:
            return await mesh_client.all_projects_status()
        except Exception as e:
            return {"error": f"Failed to read project status: {e}"}

    # detail == "names" (also the default for "full" without project_name)
    try:
        return await mesh_client.list_projects()
    except Exception as e:
        return {"error": f"Failed to list projects: {e}"}


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

    if _messages is None or not _last_message_is_user_origin(_messages):
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

    if _messages is None or not _last_message_is_user_origin(_messages):
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

    if _messages is None or not _last_message_is_user_origin(_messages):
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

    if _messages is None or not _last_message_is_user_origin(_messages):
        return {
            "error": "provenance_check_failed",
            "detail": "User confirmation required to update project context.",
        }

    try:
        return await mesh_client.update_project_context(project_name, context)
    except Exception as e:
        return {"error": f"Failed to update project context: {e}"}


@skill(
    name="set_project_goal",
    description=(
        "Set or update a project's north star (vision statement) and "
        "success criteria (measurable outcomes). The north star answers "
        "'what are we moving toward'; success criteria are the concrete "
        "checks that tell us we got there. No confirmation required — "
        "this is meta-config the user explicitly asked for.\n\n"
        "Call this proactively whenever the user describes a goal for a "
        "project so it becomes a first-class artifact visible in the "
        "workplace tab."
    ),
    parameters={
        "project_name": {
            "type": "string",
            "description": "Project to set the goal on (use list_projects to find names)",
        },
        "north_star": {
            "type": "string",
            "description": (
                "Free-text vision statement, ≤2000 characters. "
                "e.g. 'Ship a $10k MRR SaaS landing page in 2 weeks'."
            ),
        },
        "success_criteria": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Up to 10 measurable outcomes, each ≤200 characters. "
                "Optional — pass an empty list or omit to clear."
            ),
        },
    },
)
async def set_project_goal(
    project_name: str,
    north_star: str,
    success_criteria: list[str] | None = None,
    *,
    mesh_client=None,
    **_kw,
) -> dict:
    """Set the project's north_star + success_criteria. No gate."""
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if mesh_client is None:
        return {"error": "No mesh_client available"}

    if not isinstance(project_name, str) or not project_name.strip():
        return {"error": "project_name is required"}
    if not isinstance(north_star, str):
        return {"error": "north_star must be a string"}
    if len(north_star) > 2000:
        return {"error": "north_star must be 2000 characters or fewer"}

    cleaned_criteria: list[str] | None
    if success_criteria is None:
        cleaned_criteria = None
    else:
        if not isinstance(success_criteria, list):
            return {"error": "success_criteria must be a list of strings"}
        if len(success_criteria) > 10:
            return {"error": "success_criteria may contain at most 10 items"}
        cleaned_criteria = []
        for item in success_criteria:
            if not isinstance(item, str):
                return {"error": "each success_criteria entry must be a string"}
            if len(item) > 200:
                return {
                    "error": "each success_criteria entry must be 200 characters or fewer",
                }
            stripped = item.strip()
            if stripped:
                cleaned_criteria.append(stripped)
        if not cleaned_criteria:
            cleaned_criteria = None

    try:
        return await mesh_client.set_project_goal(
            project_name, north_star.strip() or None, cleaned_criteria,
        )
    except Exception as e:
        return {"error": f"Failed to set project goal: {e}"}


# ── Task 7: Operator product tools ───────────────────────────


_TASKS_V2_DISABLED = (
    "Orchestration tasks not enabled — flip OPENLEGION_ORCHESTRATION_TASKS_V2=1"
)


def _orchestration_v2_on() -> bool:
    """Read the orchestration v2 flag at call time so monkeypatch tests work.

    Default-on (rollout). Setting the env var to ``0`` disables the v2
    path; any other value is treated as on.
    """
    return os.environ.get("OPENLEGION_ORCHESTRATION_TASKS_V2", "1") != "0"


def _parse_over_budget(error: Exception) -> dict | None:
    """If a mesh HTTP error wraps an over_budget JSON payload, surface it.

    The reroute / retry endpoints encode a structured budget error in the
    400 body. ``httpx.HTTPStatusError`` stringifies as something like
    ``"Client error '400 Bad Request' for url ... \\nFor more ..."``;
    the JSON body is on ``error.response`` when the client wraps it.
    Returns the structured dict or None if the error isn't a budget one.
    """
    response = getattr(error, "response", None)
    if response is None:
        return None
    try:
        body = response.json()
    except Exception:
        return None
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, str):
        try:
            parsed = json.loads(detail)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, dict) and parsed.get("error") == "over_budget":
            return parsed
    if isinstance(detail, dict) and detail.get("error") == "over_budget":
        return detail
    return None


# ── Read tools ──────────────────────────────────────────────


@skill(
    name="list_agent_queue",
    description=(
        "Read an agent's task queue: current and recent tasks grouped by "
        "status (active / blocked / done / failed / cancelled), up to "
        "`limit` rows per bucket."
    ),
    parameters={
        "agent_id": {
            "type": "string",
            "description": "Agent ID to inspect",
        },
        "limit": {
            "type": "integer",
            "description": "Max rows per status bucket (default 10, max 100)",
            "default": 10,
        },
    },
)
async def list_agent_queue(
    agent_id: str,
    limit: int = 10,
    *,
    mesh_client=None,
    **_kw,
) -> dict:
    """Per-agent task queue grouped by status."""
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    if not _orchestration_v2_on():
        return {"error": _TASKS_V2_DISABLED}
    try:
        return await mesh_client.agent_queue(agent_id, limit=limit)
    except Exception as e:
        return {"error": f"Failed to read queue for {agent_id}: {e}"}


@skill(
    name="get_team_outputs",
    description=(
        "Completed task artifacts for a project in a time window. "
        "`since` accepts ISO timestamps or duration strings ('24h', '7d'); "
        "default is the last 7 days."
    ),
    parameters={
        "project_id": {
            "type": "string",
            "description": "Project ID",
        },
        "since": {
            "type": "string",
            "description": "ISO timestamp or duration string (e.g. '24h', '7d')",
            "default": "",
        },
    },
)
async def get_team_outputs(
    project_id: str,
    since: str = "",
    *,
    mesh_client=None,
    **_kw,
) -> dict:
    """Completed task artifacts for a project."""
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    if not _orchestration_v2_on():
        return {"error": _TASKS_V2_DISABLED}
    try:
        return await mesh_client.project_outputs(project_id, since=since)
    except Exception as e:
        return {"error": f"Failed to read outputs for {project_id}: {e}"}


@skill(
    name="summarize_project_progress",
    description=(
        "Synthesized status summary for a project: structured counts + "
        "narrative status_text + top blockers + recent completions + "
        "ask_for_user list."
    ),
    parameters={
        "project_id": {
            "type": "string",
            "description": "Project ID",
        },
    },
)
async def summarize_project_progress(
    project_id: str,
    *,
    mesh_client=None,
    **_kw,
) -> dict:
    """Synthesized progress summary for a project."""
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    if not _orchestration_v2_on():
        return {"error": _TASKS_V2_DISABLED}
    try:
        return await mesh_client.project_summary(project_id)
    except Exception as e:
        return {"error": f"Failed to summarize {project_id}: {e}"}


@skill(
    name="inspect_agents",
    description=(
        "Read agents. Without agent_id: roster summary. With agent_id: "
        "depth='profile' returns role/capabilities/INTERFACE; "
        "depth='history' adds recent activity log. depth defaults to summary."
    ),
    parameters={
        "agent_id": {
            "type": "string",
            "description": "Optional — target agent for profile/history",
            "default": "",
        },
        "depth": {
            "type": "string",
            "description": "summary | profile | history",
            "enum": ["summary", "profile", "history"],
            "default": "summary",
        },
    },
)
async def inspect_agents(
    agent_id: str = "",
    depth: str = "summary",
    *,
    mesh_client=None,
    **_kw,
) -> dict:
    """Consolidated agent read tool (replaces operator's use of list_agents
    / get_agent_profile / read_agent_history).
    """
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if mesh_client is None:
        return {"error": "No mesh_client available"}

    # No agent_id → roster (always summary regardless of depth).
    if not agent_id:
        try:
            registry = await mesh_client.list_agents()
        except Exception as e:
            return {"error": f"Failed to list agents: {e}"}
        agents = []
        for name, info in registry.items():
            entry: dict = {"name": name}
            if isinstance(info, dict):
                entry["role"] = info.get("role", "")
                entry["capabilities"] = info.get("capabilities", [])
            agents.append(entry)
        return {"agents": agents, "count": len(agents)}

    if depth == "history":
        try:
            return await mesh_client.get_agent_history(agent_id)
        except Exception as e:
            return {"error": f"Failed to read history for {agent_id}: {e}"}

    # depth == "profile" or "summary" with an agent_id → profile call
    try:
        return await mesh_client.get_agent_profile(agent_id)
    except Exception as e:
        return {"error": f"Failed to read profile for {agent_id}: {e}"}


# ── Action tools ────────────────────────────────────────────


@skill(
    name="manage_task",
    description=(
        "Cancel, reroute, or retry a task. action='cancel' stops the task; "
        "action='reroute' moves it to new_assignee (required); "
        "action='retry' clones a failed task (optionally overriding "
        "assignee/title/description via with_changes). Reroute and retry "
        "refuse if the target agent is over budget."
    ),
    parameters={
        "task_id": {
            "type": "string",
            "description": "Task ID",
        },
        "action": {
            "type": "string",
            "description": "cancel | reroute | retry",
            "enum": ["cancel", "reroute", "retry"],
        },
        "new_assignee": {
            "type": "string",
            "description": "Required for reroute; optional override for retry",
            "default": "",
        },
        "reason": {
            "type": "string",
            "description": "Optional reason recorded on the audit trail",
            "default": "",
        },
        "with_changes": {
            "type": "object",
            "description": (
                "retry only: optional patch with 'title', 'description', "
                "'assignee' overrides"
            ),
            "default": {},
        },
    },
)
async def manage_task(
    task_id: str,
    action: str,
    new_assignee: str = "",
    reason: str = "",
    with_changes: dict | None = None,
    *,
    mesh_client=None,
    **_kw,
) -> dict:
    """Consolidated task action tool (replaces cancel_task / reroute_task /
    retry_failed_task).
    """
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    if not _orchestration_v2_on():
        return {"error": _TASKS_V2_DISABLED}

    if action == "cancel":
        try:
            return await mesh_client.cancel_task(task_id, reason=reason)
        except Exception as e:
            return {"error": f"Failed to cancel task: {e}"}

    if action == "reroute":
        if not new_assignee:
            return {"error": "reroute requires new_assignee"}
        try:
            return await mesh_client.reroute_task(
                task_id, new_assignee, reason=reason,
            )
        except Exception as e:
            budget = _parse_over_budget(e)
            if budget is not None:
                return {
                    "error": "over_budget",
                    "detail": budget.get("detail") or (
                        f"Agent "
                        f"{budget.get('budget', {}).get('agent', new_assignee)!r} "
                        "is over budget."
                    ),
                    "budget": budget.get("budget"),
                }
            return {"error": f"Failed to reroute task: {e}"}

    if action == "retry":
        patch = dict(with_changes or {})
        # `new_assignee` is a convenience shortcut for retry overrides.
        if new_assignee and "assignee" not in patch:
            patch["assignee"] = new_assignee
        try:
            return await mesh_client.retry_task(
                task_id,
                title=patch.get("title"),
                description=patch.get("description"),
                assignee=patch.get("assignee"),
            )
        except Exception as e:
            budget = _parse_over_budget(e)
            if budget is not None:
                return {
                    "error": "over_budget",
                    "detail": budget.get("detail") or "Target agent is over budget.",
                    "budget": budget.get("budget"),
                }
            return {"error": f"Failed to retry task: {e}"}

    return {"error": f"Unknown action {action!r}; use cancel|reroute|retry"}


@skill(
    name="manage_project",
    description=(
        "Archive or delete a project. action='archive' is reversible and "
        "stops scheduling. action='delete' returns a confirmation nonce; "
        "the project must already be archived."
    ),
    parameters={
        "project_name": {
            "type": "string",
            "description": "Project name",
        },
        "action": {
            "type": "string",
            "description": "archive | delete",
            "enum": ["archive", "delete"],
        },
    },
)
async def manage_project(
    project_name: str,
    action: str,
    *,
    mesh_client=None,
    _messages=None,
    **_kw,
) -> dict:
    """Consolidated project lifecycle tool (archive | delete).
    Provenance-gated.
    """
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    from src.agent.loop import _last_message_is_user_origin
    if _messages is None or not _last_message_is_user_origin(_messages):
        return {
            "error": "provenance_check_failed",
            "detail": f"User confirmation required to {action} a project.",
        }

    if action == "archive":
        try:
            return await mesh_client.archive_project(project_name)
        except Exception as e:
            return {"error": f"Failed to archive project: {e}"}

    if action == "delete":
        try:
            result = await mesh_client.propose_delete_project(project_name)
            result.setdefault("requires_confirmation", True)
            return result
        except Exception as e:
            msg = str(e)
            if "must be archived" in msg.lower() or "400" in msg:
                return {
                    "error": "archive_required",
                    "detail": (
                        f"Project {project_name!r} must be archived first. "
                        "Call manage_project(action='archive') first."
                    ),
                }
            return {"error": f"Failed to propose delete: {e}"}

    return {"error": f"Unknown action {action!r}; use archive|delete"}


@skill(
    name="manage_agent",
    description=(
        "Archive or delete an agent. action='archive' is reversible and "
        "stops scheduling. action='delete' returns a confirmation nonce; "
        "the agent must already be archived."
    ),
    parameters={
        "agent_id": {
            "type": "string",
            "description": "Agent ID",
        },
        "action": {
            "type": "string",
            "description": "archive | delete",
            "enum": ["archive", "delete"],
        },
    },
)
async def manage_agent(
    agent_id: str,
    action: str,
    *,
    mesh_client=None,
    _messages=None,
    **_kw,
) -> dict:
    """Consolidated agent lifecycle tool (archive | delete).
    Provenance-gated.
    """
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    if agent_id.lower() == _OPERATOR_AGENT_ID:
        return {"error": f"Cannot {action} the operator agent."}
    from src.agent.loop import _last_message_is_user_origin
    if _messages is None or not _last_message_is_user_origin(_messages):
        return {
            "error": "provenance_check_failed",
            "detail": f"User confirmation required to {action} an agent.",
        }

    if action == "archive":
        try:
            return await mesh_client.archive_agent(agent_id)
        except Exception as e:
            return {"error": f"Failed to archive agent: {e}"}

    if action == "delete":
        try:
            result = await mesh_client.propose_delete_agent(agent_id)
            result.setdefault("requires_confirmation", True)
            return result
        except Exception as e:
            msg = str(e)
            if "must be archived" in msg.lower() or "400" in msg:
                return {
                    "error": "archive_required",
                    "detail": (
                        f"Agent {agent_id!r} must be archived first. "
                        "Call manage_agent(action='archive') first."
                    ),
                }
            return {"error": f"Failed to propose delete: {e}"}

    return {"error": f"Unknown action {action!r}; use archive|delete"}
