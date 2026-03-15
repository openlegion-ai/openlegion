"""Skill discovery and registry for agent tools.

Skills are plain Python functions with a @skill decorator.
Auto-discovered from a directory at startup. No plugin system.
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.shared.utils import setup_logging

if TYPE_CHECKING:
    from src.agent.mcp_client import MCPClient

logger = setup_logging("agent.skills")

# Global registry populated by the @skill decorator.
# Protected by _skill_staging_lock during reload to prevent
# corruption if hot-reload runs concurrently with module import.
_skill_staging: dict[str, dict] = {}
_skill_staging_lock = threading.Lock()


def skill(
    name: str,
    description: str,
    parameters: dict,
    parallel_safe: bool = True,
    loop_exempt: bool = False,
):
    """Decorator to register a function as an agent skill.

    *parallel_safe* — when ``True`` (default), the tool may be executed
    concurrently with other parallel-safe tools via ``asyncio.gather``.
    Set to ``False`` for tools that hold exclusive resources (e.g. browser).

    *loop_exempt* — when ``True``, the tool is exempt from warn/block
    escalation in the loop detector (but still respects the terminate
    threshold as a hard safety cap).
    """

    def decorator(func):
        sig = inspect.signature(func)
        param_names = set(sig.parameters.keys())
        _skill_staging[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "function": func,
            # Cached signature metadata — avoids inspect.signature() on every execute()
            "_sig_params": param_names,
            "_sig_has_var_keyword": any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            ),
            "_sig_is_coroutine": inspect.iscoroutinefunction(func),
            "_parallel_safe": parallel_safe,
            "_loop_exempt": loop_exempt,
        }
        return func

    return decorator


class SkillRegistry:
    """Auto-discovers and manages agent skills from a directory.

    Loads built-in tools first, then custom skills (which can override builtins).
    Supports hot-reload when agents create new skills at runtime.
    """

    CUSTOM_SKILLS_DIR = "/data/custom_skills"
    MARKETPLACE_SKILLS_DIR = "/app/marketplace_skills"

    def __init__(self, skills_dir: str, mcp_client: MCPClient | None = None):
        self.skills_dir = skills_dir
        self._mcp_client = mcp_client
        self.skills: dict[str, dict] = {}
        self._builtin_functions: frozenset = frozenset()
        # Memoization caches — cleared on reload()
        self._tool_defs_cache: dict[frozenset[str] | None, list[dict]] = {}
        self._descriptions_cache: dict[frozenset[str] | None, str] = {}
        with _skill_staging_lock:
            self._discover_builtins()
            self._builtin_functions = frozenset(
                info["function"] for info in _skill_staging.values()
                if callable(info.get("function"))
            )
            self._discover(skills_dir)
            self._discover(self.CUSTOM_SKILLS_DIR)
            self._discover(self.MARKETPLACE_SKILLS_DIR)
            self.skills = dict(_skill_staging)
        self._register_mcp_tools()

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

    def _register_mcp_tools(self) -> None:
        """Register tools from connected MCP servers."""
        if not getattr(self, "_mcp_client", None):
            return
        for tool_def in self._mcp_client.list_tools():
            name = tool_def["name"]
            if name not in self.skills:
                self.skills[name] = tool_def
            # Conflicts already handled with prefixing in MCPClient.start()

    def reload(self) -> int:
        """Re-discover skills from builtins and skills_dir. Returns new skill count."""
        with _skill_staging_lock:
            _skill_staging.clear()
            self._discover_builtins()
            self._builtin_functions = frozenset(
                info["function"] for info in _skill_staging.values()
                if callable(info.get("function"))
            )
            self._discover(self.skills_dir)
            self._discover(self.CUSTOM_SKILLS_DIR)
            self._discover(self.MARKETPLACE_SKILLS_DIR)
            self.skills = dict(_skill_staging)
        self._register_mcp_tools()
        self._tool_defs_cache.clear()
        self._descriptions_cache.clear()
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
        if getattr(self, "_mcp_client", None) and self._mcp_client.has_tool(name):
            return await self._mcp_client.call_tool(name, arguments)

        if name not in self.skills:
            raise ValueError(f"Unknown skill: {name}")

        info = self.skills[name]
        func = info["function"]
        call_args = dict(arguments)

        # Use cached signature metadata (computed at registration time via @skill)
        sig_params = info.get("_sig_params")
        if sig_params is not None:
            has_var_keyword = info["_sig_has_var_keyword"]
            is_coroutine = info["_sig_is_coroutine"]
        else:
            # Fallback for dynamically registered skills (MCP, marketplace)
            sig = inspect.signature(func)
            sig_params = set(sig.parameters.keys())
            has_var_keyword = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            )
            is_coroutine = inspect.iscoroutinefunction(func)

        # Inject framework-provided dependencies if the function accepts them.
        if "mesh_client" in sig_params:
            call_args["mesh_client"] = mesh_client
        if "workspace_manager" in sig_params:
            call_args["workspace_manager"] = workspace_manager
        if "memory_store" in sig_params:
            call_args["memory_store"] = memory_store

        # Filter out LLM-hallucinated parameters that the function doesn't
        # accept.  Without this, an LLM sending e.g. {"raw": ""} to a
        # zero-parameter tool like vault_list() causes a TypeError crash.
        # We only filter when the function does NOT accept **kwargs.
        if not has_var_keyword:
            extra = set(call_args) - sig_params
            if extra:
                logger.debug(
                    "Dropping unknown args %s for skill '%s'", extra, name,
                )
                call_args = {k: v for k, v in call_args.items() if k in sig_params}

        if is_coroutine:
            return await func(**call_args)
        return await asyncio.get_running_loop().run_in_executor(None, lambda: func(**call_args))

    def get_tool_sources(self, exclude: frozenset[str] | None = None) -> dict[str, str]:
        """Return a mapping of skill name → source tag.

        Tags: ``"builtin"`` (core platform tools), ``"mcp"`` (MCP server tools),
        ``"custom"`` (agent-created or marketplace skills).

        Uses function-object identity rather than name lookup so that a custom
        skill that overrides a builtin by the same name is correctly tagged
        ``"custom"``.
        """
        result = {}
        for name, info in self.skills.items():
            if exclude and name in exclude:
                continue
            func = info.get("function")
            if func == "mcp":
                result[name] = "mcp"
            elif func in self._builtin_functions:
                result[name] = "builtin"
            else:
                result[name] = "custom"
        return result

    def is_parallel_safe(self, name: str) -> bool:
        """Return whether a skill is safe to execute concurrently."""
        info = self.skills.get(name)
        if not info:
            return True  # unknown tools default to safe
        return info.get("_parallel_safe", True)

    def get_loop_exempt_tools(self) -> frozenset[str]:
        """Return the set of tool names marked loop_exempt."""
        return frozenset(
            name for name, info in self.skills.items()
            if info.get("_loop_exempt", False)
        )

    def list_skills(self, exclude: frozenset[str] | None = None) -> list[str]:
        """Return list of available skill names."""
        if exclude:
            return [n for n in self.skills if n not in exclude]
        return list(self.skills.keys())

    def get_descriptions(self, exclude: frozenset[str] | None = None) -> str:
        """Return human-readable descriptions of all skills (memoized)."""
        cache = getattr(self, "_descriptions_cache", None)
        if cache is None:
            self._descriptions_cache = cache = {}
        cache_key = exclude
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        lines = []
        for name, info in self.skills.items():
            if exclude and name in exclude:
                continue
            raw_params = info["parameters"]
            # MCP tools have full JSON Schema; extract from "properties"
            if info.get("function") == "mcp":
                props = raw_params.get("properties", {})
                params = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in props.items())
            else:
                params = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in raw_params.items())
            desc = " ".join(info["description"].split())
            lines.append(f"- {name}({params}): {desc}")
        result = "\n".join(lines)
        self._descriptions_cache[cache_key] = result
        return result

    def get_tool_definitions(self, exclude: frozenset[str] | None = None) -> list[dict]:
        """Return OpenAI-compatible tool definitions for LLM function calling (memoized)."""
        cache = getattr(self, "_tool_defs_cache", None)
        if cache is None:
            self._tool_defs_cache = cache = {}
        cache_key = exclude
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        tools = []
        for name, info in self.skills.items():
            if exclude and name in exclude:
                continue
            params = info["parameters"]

            # MCP tools provide a full JSON Schema (with "type": "object",
            # "properties", etc.) — use it directly.
            if info.get("function") == "mcp":
                tools.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": info["description"],
                        "parameters": params,
                    },
                })
                continue

            # Built-in skills store a flat {param_name: {type, description, ...}} dict.
            properties = {}
            required = []
            for param_name, param_info in params.items():
                prop: dict = {
                    "type": param_info.get("type", "string"),
                    "description": param_info.get("description", ""),
                }
                if "enum" in param_info:
                    prop["enum"] = param_info["enum"]
                properties[param_name] = prop
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
        self._tool_defs_cache[cache_key] = tools
        return tools
