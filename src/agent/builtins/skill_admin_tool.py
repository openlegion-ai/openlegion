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


@tool(
    name="list_skill_assignments",
    description=(
        "Show which skill packs are assigned where: the fleet-wide list (every "
        "agent sees these) and each agent's per-agent list. Operator-only. Read "
        "this before assigning so you edit the right list."
    ),
    parameters={},
)
async def list_skill_assignments(*, mesh_client=None, **_kw) -> dict:
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    return await mesh_client.get_skill_assignments()


@tool(
    name="assign_skill",
    description=(
        "Give or take away a skill pack for an agent (or the whole fleet). "
        "Skills are scoped per agent so each agent only sees the procedures it "
        "needs — assigning one makes it visible to that agent's skills_list. "
        "Set fleet=true to assign to EVERY agent; otherwise pass agent_id for a "
        "single agent. action='add' grants, action='remove' revokes. "
        "Operator-only; requires the user to have asked for it."
    ),
    parameters={
        "skill": {"type": "string", "description": "Skill pack name (see skills_list / list_skill_assignments)."},
        "agent_id": {
            "type": "string",
            "description": "Agent to assign to. Ignored when fleet=true.",
            "default": "",
        },
        "fleet": {
            "type": "boolean",
            "description": "Assign fleet-wide (every agent) instead of to one agent.",
            "default": False,
        },
        "action": {
            "type": "string",
            "description": "'add' to grant, 'remove' to revoke.",
            "default": "add",
        },
    },
)
async def assign_skill(
    skill: str,
    agent_id: str = "",
    fleet: bool = False,
    action: str = "add",
    *,
    mesh_client=None,
    _messages=None,
    **_kw,
) -> dict:
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    from src.agent.loop import _last_message_is_user_origin
    if _messages is None or not _last_message_is_user_origin(_messages):
        return {
            "error": "provenance_check_failed",
            "detail": "User confirmation required to change skill assignments.",
        }
    skill = (skill or "").strip()
    if not skill:
        return {"error": "skill is required"}
    if action not in ("add", "remove"):
        return {"error": "action must be 'add' or 'remove'"}
    agent_id = (agent_id or "").strip()
    if not fleet and not agent_id:
        return {"error": "Pass agent_id, or set fleet=true to assign to every agent."}

    # Read current assignment, mutate the relevant list, write it back. The
    # mesh endpoints set the full list, so we compute the new full list here.
    assignments = await mesh_client.get_skill_assignments()
    if fleet:
        current = list(assignments.get("fleet_skills", []))
    else:
        current = list(assignments.get("per_agent", {}).get(agent_id, []))

    names = set(current)
    if action == "add":
        names.add(skill)
    else:
        names.discard(skill)
    new_list = sorted(names)

    if fleet:
        result = await mesh_client.set_fleet_skills(new_list)
    else:
        result = await mesh_client.assign_skills(agent_id, new_list)

    # Surface the no-op-removal case: removing a skill from one agent has no
    # effect while it's still assigned fleet-wide.
    if (
        action == "remove"
        and not fleet
        and skill in set(assignments.get("fleet_skills", []))
    ):
        result["note"] = (
            f"'{skill}' is still assigned fleet-wide, so {agent_id} continues to "
            "see it. Remove it from the fleet (fleet=true) to fully revoke."
        )
    return result
