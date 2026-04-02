"""Fleet template tools — list and apply agent team templates."""

from __future__ import annotations

from src.agent.skills import skill
from src.shared.utils import setup_logging

logger = setup_logging("agent.builtins.fleet_tool")


@skill(
    name="list_templates",
    description=(
        "List available fleet templates that can be used to create agent teams. "
        "Returns template names, descriptions, and agent counts. Use this to "
        "show the user what team configurations are available before applying one."
    ),
    parameters={},
)
async def list_templates(*, mesh_client=None, **_kw) -> dict:
    """List available fleet templates."""
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
        "to see available templates. This creates all agents defined in the "
        "template and starts them. Returns the list of created agent IDs."
    ),
    parameters={
        "template": {
            "type": "string",
            "description": "Template name (e.g. 'sales', 'content', 'devteam')",
        },
        "model": {
            "type": "string",
            "description": (
                "Optional model override for all agents "
                "(e.g. 'anthropic/claude-sonnet-4-20250514'). "
                "Defaults to the system's configured default model."
            ),
        },
    },
)
async def apply_template(template: str, model: str = "", *, mesh_client=None, **_kw) -> dict:
    """Apply a fleet template to create a team of agents."""
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    try:
        return await mesh_client.apply_fleet_template(template, model=model)
    except Exception as e:
        return {"error": f"Failed to apply template '{template}': {e}"}
