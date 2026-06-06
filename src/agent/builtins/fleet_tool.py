"""Fleet template tools -- list and apply agent team templates."""

from __future__ import annotations

import os

from src.agent.tools import tool


def _is_operator() -> bool:
    """Defence-in-depth: only the operator agent has ALLOWED_TOOLS set."""
    return os.environ.get("ALLOWED_TOOLS", "") != ""


@tool(
    name="list_templates",
    description=(
        "List available fleet templates that can be used to create agent teams. "
        "Returns template names, descriptions, and agent counts."
    ),
    parameters={},
)
async def list_templates(*, mesh_client=None, **_kw) -> dict:
    """List available fleet templates."""
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    try:
        return await mesh_client.list_fleet_templates()
    except Exception as e:
        return {"error": f"Failed to list templates: {e}"}


@tool(
    name="apply_template",
    description=(
        "Create a team of agents from a fleet template. Use list_templates first "
        "to see available templates. This creates all agents defined in the template "
        "and starts them. Returns the list of created agent IDs. Pass "
        "`agent_overrides` to tune individual slots (model, instructions, soul, "
        "heartbeat, interface) at creation time -- avoids 5+ follow-up edit_agent "
        "round-trips. Note: `role` is fixed by the template and cannot be overridden. "
        "Note: agent creation is per-slot -- a mid-loop failure leaves earlier-created "
        "agents in place. Verify the returned `created` list matches your intent and "
        "inspect the on-disk fleet."
    ),
    parameters={
        "template": {
            "type": "string",
            "description": "Template name (e.g. 'sales', 'content', 'devteam')",
        },
        "model": {
            "type": "string",
            "description": "Optional model override for ALL agents. Defaults to system default.",
            "default": "",
        },
        "agent_overrides": {
            "type": "object",
            "description": (
                "Optional per-agent overrides keyed by agent name from the template. "
                "Each value is an object with any of: 'model', 'instructions' "
                "(<=12000 chars), 'soul' (<=4000 chars), 'heartbeat' (no cap), "
                "'interface' (<=4000 chars). Unknown agent names or fields are "
                "rejected with HTTP 400; oversized fields with HTTP 413. "
                "Note: `role` is intentionally NOT overrideable; templates fix the "
                "role per slot. Use a separate fleet template if you need different roles."
            ),
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "model": {"type": "string"},
                    "instructions": {"type": "string"},
                    "soul": {"type": "string"},
                    "heartbeat": {"type": "string"},
                    "interface": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
    },
)
async def apply_template(
    template: str,
    model: str = "",
    agent_overrides: dict[str, dict] | None = None,
    *,
    mesh_client=None,
    _messages=None,
    **_kw,
) -> dict:
    """Apply a fleet template to create a team of agents."""
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    # Provenance check -- require user confirmation
    from src.agent.loop import _last_message_is_user_origin

    if _messages is None or not _last_message_is_user_origin(_messages):
        return {
            "error": "provenance_check_failed",
            "detail": "User confirmation required to create agents.",
        }
    try:
        return await mesh_client.apply_fleet_template(
            template, model=model, agent_overrides=agent_overrides,
        )
    except Exception as e:
        return {"error": f"Failed to apply template '{template}': {e}"}
