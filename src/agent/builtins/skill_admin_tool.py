"""Operator-only skill-pack administration: install / remove SKILL.md packs.

Mirrors ``fleet_tool.apply_template`` — operator-gated, requires a user-origin
message before mutating, and forwards to the mesh (which re-checks
``can_manage_fleet``). Installing a Skill adds no code to the runtime and
widens no permission; it only makes a procedure available fleet-wide.
"""

from __future__ import annotations

import os

from src.agent.tools import tool


def _is_operator() -> bool:
    """Defence-in-depth: only the operator agent has ALLOWED_TOOLS set."""
    return os.environ.get("ALLOWED_TOOLS", "") != ""


@tool(
    name="install_skill",
    description=(
        "Install a SKILL.md skill pack from a public git repo so the fleet can "
        "use it. The repo must have a SKILL.md (name + description) at its root. "
        "Operator-only; requires the user to have asked for it."
    ),
    parameters={
        "repo_url": {
            "type": "string",
            "description": "Git repo URL (https:// or git@) containing SKILL.md.",
        },
        "ref": {
            "type": "string",
            "description": "Optional branch/tag to install.",
            "default": "",
        },
    },
    # Mutates the shared on-disk skills store (clone → rmtree → os.replace).
    # Must not run concurrently with another install/remove or the gather-batched
    # loop could interleave two filesystem mutations on the same store. Mirrors
    # assign_skill's opt-out.
    parallel_safe=False,
)
async def install_skill(repo_url: str, ref: str = "", *, mesh_client=None, _messages=None, **_kw) -> dict:
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    from src.agent.loop import _last_message_is_user_origin
    if _messages is None or not _last_message_is_user_origin(_messages):
        return {
            "error": "provenance_check_failed",
            "detail": "User confirmation required to install a skill.",
        }
    return await mesh_client.install_skill(repo_url, ref=ref)


@tool(
    name="remove_skill",
    description=(
        "Remove a previously installed skill pack by name. Operator-only; "
        "requires the user to have asked for it. Bundled skills cannot be removed."
    ),
    parameters={
        "name": {"type": "string", "description": "Installed skill name to remove."},
    },
    # Mutates the shared on-disk skills store — serialize like install_skill.
    parallel_safe=False,
)
async def remove_skill(name: str, *, mesh_client=None, _messages=None, **_kw) -> dict:
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    from src.agent.loop import _last_message_is_user_origin
    if _messages is None or not _last_message_is_user_origin(_messages):
        return {
            "error": "provenance_check_failed",
            "detail": "User confirmation required to remove a skill.",
        }
    return await mesh_client.remove_skill(name)
