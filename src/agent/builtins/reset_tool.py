"""Reset request tool for agents.

Allows an agent to request a full system reset from the user.
The actual reset is performed through the dashboard after user
confirmation — the agent cannot reset the system directly.
"""

from __future__ import annotations

from src.agent.skills import skill
from src.shared.utils import setup_logging

logger = setup_logging("agent.reset_tool")


@skill(
    name="request_reset",
    description=(
        "Request a full system reset from the user. This sends a notification "
        "to the dashboard where the user can confirm or dismiss the request. "
        "A reset wipes all agent configs, projects, skills, memory, and runtime "
        "data. Use this only when the system is in a fundamentally broken state "
        "that cannot be fixed otherwise."
    ),
    parameters={
        "reason": {
            "type": "string",
            "description": "Explain why a system reset is needed",
        },
    },
)
async def request_reset(
    reason: str,
    *,
    mesh_client=None,
    **_kw,
) -> dict:
    """Request a system reset from the user via the dashboard."""
    if not mesh_client:
        return {"error": "Reset tool requires mesh connectivity"}

    if not reason or not reason.strip():
        return {"error": "reason is required — explain why a reset is needed"}

    try:
        await mesh_client.request_reset_from_user(reason=reason.strip())
    except Exception as e:
        logger.warning("Failed to send reset request: %s", e)
        return {"error": f"Failed to send reset request: {e}"}

    return {
        "requested": True,
        "message": (
            "Reset request sent to the user. They will see a confirmation "
            "prompt in the dashboard. The system will only reset if the user "
            "explicitly confirms."
        ),
    }
