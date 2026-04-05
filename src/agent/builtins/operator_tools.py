"""Operator agent tools -- propose/confirm edits, observations, agent/project management."""
from __future__ import annotations

from src.agent.skills import skill
from src.shared.utils import setup_logging

logger = setup_logging("agent.builtins.operator_tools")

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
            if key in value and isinstance(max_val, bool) and value[key] and not max_val:
                return {
                    "error": (
                        f"Permission ceiling exceeded: '{key}' cannot be set "
                        "to True by the operator. Use the dashboard for "
                        "advanced permissions."
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
        # The confirm endpoint uses the agent_id from the pending change.
        # We pass a placeholder and let the server resolve from change_id.
        result = await mesh_client.confirm_config_change("_pending", change_id)
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
