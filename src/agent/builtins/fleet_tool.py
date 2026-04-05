"""Fleet template tools -- list and apply agent team templates."""

from __future__ import annotations

import os

from src.agent.skills import skill


def _is_operator() -> bool:
    """Defence-in-depth: only the operator agent has ALLOWED_TOOLS set."""
    return os.environ.get("ALLOWED_TOOLS", "") != ""


def _is_operator() -> bool:
    """Defence-in-depth: only the operator agent has ALLOWED_TOOLS set."""
    return os.environ.get("ALLOWED_TOOLS", "") != ""


@skill(
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


@skill(
    name="apply_template",
    description=(
        "Create a team of agents from a fleet template. Use list_templates first "
        "to see available templates. This creates all agents defined in the template "
        "and starts them. Returns the list of created agent IDs."
    ),
    parameters={
        "template": {
            "type": "string",
            "description": "Template name (e.g. 'sales', 'content', 'devteam')",
        },
        "model": {
            "type": "string",
            "description": "Optional model override for all agents. Defaults to system default.",
            "default": "",
        },
    },
)
async def apply_template(
    template: str, model: str = "", *, mesh_client=None, _messages=None, **_kw,
) -> dict:
    """Apply a fleet template to create a team of agents."""
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    # Provenance check -- require user confirmation
    from src.agent.loop import _last_message_is_user_origin

    if _messages is not None and not _last_message_is_user_origin(_messages):
        return {
            "error": "provenance_check_failed",
            "detail": "User confirmation required to create agents.",
        }
    try:
        return await mesh_client.apply_fleet_template(template, model=model)
    except Exception as e:
        return {"error": f"Failed to apply template '{template}': {e}"}
