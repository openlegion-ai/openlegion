"""Skill discovery and registry for agent tools.

Skills are plain Python functions with a @skill decorator.
Auto-discovered from a directory at startup. No plugin system.
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
from pathlib import Path
from typing import Any

from src.shared.utils import setup_logging

logger = setup_logging("agent.skills")

# Global registry populated by the @skill decorator
_skill_staging: dict[str, dict] = {}


def skill(name: str, description: str, parameters: dict):
    """Decorator to register a function as an agent skill."""

    def decorator(func):
        _skill_staging[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "function": func,
        }
        return func

    return decorator


class SkillRegistry:
    """Auto-discovers and manages agent skills from a directory.

    Loads built-in tools first, then custom skills (which can override builtins).
    Supports hot-reload when agents create new skills at runtime.
    """

    CUSTOM_SKILLS_DIR = "/data/custom_skills"

    def __init__(self, skills_dir: str):
        self.skills_dir = skills_dir
        self.skills: dict[str, dict] = {}
        self._discover_builtins()
        self._discover(skills_dir)
        self._discover(self.CUSTOM_SKILLS_DIR)
        self.skills = dict(_skill_staging)

    def _discover_builtins(self) -> None:
        """Load all tool modules from the builtins package."""
        builtins_dir = Path(__file__).parent / "builtins"
        if not builtins_dir.exists():
            return
        self._load_modules_from(builtins_dir, label="builtin")

    def _discover(self, skills_dir: str) -> None:
        """Load all .py files from skills_dir and register decorated functions."""
        skills_path = Path(skills_dir)
        if not skills_path.exists():
            logger.warning(f"Skills directory not found: {skills_dir}")
            return
        self._load_modules_from(skills_path, label="skill")

    def _load_modules_from(self, directory: Path, label: str) -> None:
        """Load all .py modules from a directory, registering decorated skills."""
        for py_file in directory.glob("**/*.py"):
            if py_file.name.startswith("_"):
                continue
            try:
                spec = importlib.util.spec_from_file_location(py_file.stem, str(py_file))
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
            except Exception as e:
                logger.warning(f"Failed to load {label} {py_file}: {e}")

    def reload(self) -> int:
        """Re-discover skills from builtins and skills_dir. Returns new skill count."""
        _skill_staging.clear()
        self._discover_builtins()
        self._discover(self.skills_dir)
        self._discover(self.CUSTOM_SKILLS_DIR)
        self.skills = dict(_skill_staging)
        logger.info(f"Reloaded {len(self.skills)} skills")
        return len(self.skills)

    async def execute(
        self,
        name: str,
        arguments: dict,
        mesh_client: Any = None,
        workspace_manager: Any = None,
        memory_store: Any = None,
    ) -> Any:
        """Execute a skill by name with given arguments."""
        if name not in self.skills:
            raise ValueError(f"Unknown skill: {name}")

        func = self.skills[name]["function"]
        call_args = dict(arguments)

        sig = inspect.signature(func)
        if "mesh_client" in sig.parameters:
            call_args["mesh_client"] = mesh_client
        if "workspace_manager" in sig.parameters:
            call_args["workspace_manager"] = workspace_manager
        if "memory_store" in sig.parameters:
            call_args["memory_store"] = memory_store

        if inspect.iscoroutinefunction(func):
            return await func(**call_args)
        return await asyncio.get_running_loop().run_in_executor(None, lambda: func(**call_args))

    def list_skills(self) -> list[str]:
        """Return list of available skill names."""
        return list(self.skills.keys())

    def get_descriptions(self) -> str:
        """Return human-readable descriptions of all skills."""
        lines = []
        for name, info in self.skills.items():
            params = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in info["parameters"].items())
            lines.append(f"- {name}({params}): {info['description']}")
        return "\n".join(lines)

    def get_tool_definitions(self) -> list[dict]:
        """Return OpenAI-compatible tool definitions for LLM function calling."""
        tools = []
        for name, info in self.skills.items():
            properties = {}
            required = []
            for param_name, param_info in info["parameters"].items():
                properties[param_name] = {
                    "type": param_info.get("type", "string"),
                    "description": param_info.get("description", ""),
                }
                if "default" not in param_info:
                    required.append(param_name)

            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": info["description"],
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            })
        return tools
