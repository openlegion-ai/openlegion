"""Progressive-disclosure tools for SKILL.md skill packs.

These two tools are READ-ONLY: they let an agent discover and read
procedural skill packs (instructions for using the tools it already has).
They add no capability and widen no permission — a Skill can only ever
orchestrate Tools the agent is already allowed to call. See
``src/agent/skills.py`` for the loader/store.
"""

from __future__ import annotations

from src.agent.skills import SkillStore, render_text
from src.agent.tools import tool


def _declared_requirements(metadata: dict) -> dict:
    """Surface config/env a skill declares so the agent knows what to set.

    Reads both the top-level agentskills.io ``required_environment_variables``
    and any vendor ``metadata.*.config`` block, without interpreting/resolving
    them (resolution + sandbox env passthrough are deferred). Returns only the
    keys that are present, so a plain skill adds nothing to the payload.
    """
    out: dict = {}
    env = metadata.get("required_environment_variables")
    if env:
        out["required_environment_variables"] = env
    vendors = metadata.get("metadata")
    if isinstance(vendors, dict):
        for vendor in vendors.values():
            if isinstance(vendor, dict) and vendor.get("config"):
                out["config"] = vendor["config"]
                break
    return out


@tool(
    name="skills_list",
    description=(
        "List available skill packs — saved step-by-step procedures for "
        "common jobs, written in plain language. Returns each skill's name "
        "and a one-line description. When a skill looks relevant to your "
        "task, call skill_view(name) to read its full instructions and "
        "follow them. Skills use only tools you already have."
    ),
    parameters={},
)
def skills_list() -> dict:
    skills = SkillStore().list()
    return {
        "skills": [{"name": s.name, "description": s.description} for s in skills],
        "count": len(skills),
        "hint": "Call skill_view(name) to read a skill's full instructions.",
    }


@tool(
    name="skill_view",
    description=(
        "Read a skill pack's full instructions. Call with just 'name' to get "
        "the procedure body; pass 'path' to read a bundled reference file the "
        "skill points you to (e.g. 'references/checklist.md'). Discover names "
        "with skills_list first."
    ),
    parameters={
        "name": {
            "type": "string",
            "description": "Skill name as returned by skills_list.",
        },
        "path": {
            "type": "string",
            "description": (
                "Optional reference file path within the skill, relative to "
                "the skill directory (e.g. 'references/checklist.md')."
            ),
            "default": "",
        },
    },
)
def skill_view(name: str, path: str = "") -> dict:
    store = SkillStore()
    skill = store.get(name)
    if skill is None:
        return {
            "error": (
                f"Skill '{name}' not found. Call skills_list to see available skills."
            ),
        }
    if path:
        content = store.read_reference(name, path)
        if content is None:
            return {"error": f"Reference '{path}' not found in skill '{name}'."}
        return {"name": name, "path": path, "content": render_text(content, skill)}
    result = {
        "name": skill.name,
        "description": skill.description,
        "version": skill.version,
        "body": render_text(skill.body, skill),
    }
    # Only present when the skill actually declares config/env requirements.
    result.update(_declared_requirements(skill.metadata))
    return result
