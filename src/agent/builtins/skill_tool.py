"""Self-authoring: agents can create their own skills at runtime.

The agent writes a Python function, which is validated (syntax-checked),
saved to the skills directory, and hot-reloaded into the registry.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from src.agent.skills import skill

_FORBIDDEN_IMPORTS = frozenset({
    "os.system", "subprocess", "shutil.rmtree", "ctypes",
    "importlib", "socket",
})
_FORBIDDEN_CALLS = frozenset({"eval", "exec", "__import__"})
_MAX_SKILL_SIZE = 10_000


def _validate_skill_code(code: str) -> str | None:
    """Validate skill code. Returns error message or None if valid."""
    if len(code) > _MAX_SKILL_SIZE:
        return f"Skill code too large ({len(code)} > {_MAX_SKILL_SIZE} chars)"
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"Syntax error: {e}"

    has_skill_decorator = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name) and dec.func.id == "skill":
                    has_skill_decorator = True
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = ""
            if isinstance(node, ast.ImportFrom) and node.module:
                module = node.module
            elif isinstance(node, ast.Import):
                module = node.names[0].name if node.names else ""
            if any(f in module for f in _FORBIDDEN_IMPORTS):
                return f"Forbidden import: {module}"
        if isinstance(node, ast.Call):
            func_name = ""
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            if func_name in _FORBIDDEN_CALLS:
                return f"Forbidden call: {func_name}()"

    if not has_skill_decorator:
        return "Code must contain at least one function with the @skill decorator"
    return None


def _sanitize_filename(name: str) -> str:
    """Convert a skill name to a safe Python filename."""
    safe = re.sub(r"[^a-z0-9_]", "_", name.lower())
    return f"custom_{safe}.py"


@skill(
    name="create_skill",
    description=(
        "Create a new tool/skill for yourself. Write Python code with the @skill "
        "decorator. The skill will be validated, saved, and immediately available. "
        "You must import 'from src.agent.skills import skill' and decorate your "
        "function with @skill(name=..., description=..., parameters=...). "
        "Example:\n"
        "  from src.agent.skills import skill\n"
        "  @skill(name='my_tool', description='Does X', parameters={'x': {'type': 'string'}})\n"
        "  def my_tool(x: str) -> dict:\n"
        "      return {'result': x.upper()}"
    ),
    parameters={
        "name": {
            "type": "string",
            "description": "Name for this skill (used as filename)",
        },
        "code": {
            "type": "string",
            "description": "Python source code with @skill decorator",
        },
    },
)
def create_skill(name: str, code: str, *, workspace_manager=None) -> dict:
    if workspace_manager is None:
        return {"error": "No workspace_manager available"}

    error = _validate_skill_code(code)
    if error:
        return {"error": f"Validation failed: {error}"}

    # Write to /data/custom_skills (writable) — /app/skills is read-only
    skills_dir = Path("/data/custom_skills")
    skills_dir.mkdir(parents=True, exist_ok=True)

    filename = _sanitize_filename(name)
    filepath = skills_dir / filename
    filepath.write_text(code)

    return {
        "created": True,
        "name": name,
        "file": str(filepath),
        "note": "Skill saved. Use reload_skills to activate it.",
    }


@skill(
    name="reload_skills",
    description=(
        "Reload all skills including any new ones you've created with create_skill. "
        "Call this after creating a new skill to make it available for use."
    ),
    parameters={},
)
def reload_skills() -> dict:
    return {
        "reload_requested": True,
        "note": "Skills will be reloaded. New tools will be available on next turn.",
    }


@skill(
    name="list_custom_skills",
    description="List all custom skills you've created in your skills directory.",
    parameters={},
)
def list_custom_skills() -> dict:
    skills_dir = Path("/data/custom_skills")
    if not skills_dir.exists():
        return {"skills": [], "count": 0}
    files = [f.name for f in skills_dir.glob("*.py") if not f.name.startswith("_")]
    return {"skills": files, "count": len(files), "directory": str(skills_dir)}
