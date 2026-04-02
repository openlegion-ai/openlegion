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
    "os", "subprocess", "shutil", "ctypes",
    "importlib", "socket", "sys", "signal",
    "multiprocessing", "threading",
    # I/O and shell access outside the sandbox
    "pathlib", "io", "tempfile", "pty", "code",
    # Introspection and deserialization
    "gc", "inspect", "pickle", "shelve",
    # Network and server
    "http", "asyncio", "resource",
    # Builtin access (prevents aliasing eval/exec/open to bypass _FORBIDDEN_CALLS)
    "builtins",
})
_FORBIDDEN_CALLS = frozenset({
    "eval", "exec", "__import__", "compile",
    "globals", "locals", "getattr", "setattr", "delattr",
    "breakpoint", "open",
    # Dynamic class/object creation
    "type", "vars", "dir", "memoryview", "super",
})
_FORBIDDEN_ATTRS = frozenset({
    "__builtins__", "__import__", "__subclasses__", "__class__",
    "__bases__", "__mro__", "__globals__", "__code__", "__reduce__",
    "__dict__",
    "builtins",
})
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
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name) and dec.func.id == "skill":
                    has_skill_decorator = True
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            elif isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            for module in modules:
                # Check each segment of the module path (e.g. "os.path" → ["os", "path"])
                parts = module.split(".")
                if any(p in _FORBIDDEN_IMPORTS for p in parts) or module in _FORBIDDEN_IMPORTS:
                    return f"Forbidden import: {module}"
        if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_ATTRS:
            return f"Forbidden attribute access: {node.attr}"
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


_SKILL_AUTHORING_GUIDE = """\
# Skill Authoring Guide

## mesh_client Methods (all async)

- `browser_command(action, params)` — send browser commands.
  Actions: navigate, snapshot, click, type, hover, scroll, screenshot,
  press_key, wait_for, reset, go_back, go_forward, switch_tab.
  Same API that browser_navigate/browser_click/etc use internally.
- `notify_user(message)` — send notification to user
- `read_blackboard(key)` / `write_blackboard(key, value)` — shared data store
- `publish_event(topic, payload)` — pub/sub events
- `vault_resolve(name)` — resolve a $CRED{name} handle to its value
- `image_generate(prompt, size, provider)` — generate an image
- `list_agents()` — list fleet agents

## Example: Browser Automation Skill

```python
from src.agent.skills import skill

@skill(
    name="post_tweet",
    description="Post a tweet",
    parameters={"text": {"type": "string", "description": "Tweet text"}},
)
async def post_tweet(text: str, *, mesh_client=None) -> dict:
    await mesh_client.browser_command(
        "navigate", {"url": "https://x.com/compose/post", "wait_until": "networkidle"},
    )
    await mesh_client.browser_command(
        "wait_for", {"selector": '[data-testid="tweetTextarea_0"]', "state": "visible"},
    )
    await mesh_client.browser_command(
        "type", {"selector": '[data-testid="tweetTextarea_0"]', "text": text},
    )
    await mesh_client.browser_command(
        "click", {"selector": '[data-testid="tweetButton"]'},
    )
    return {"posted": True}
```

## Example: Simple Skill (no dependencies)

```python
from src.agent.skills import skill

@skill(
    name="my_tool",
    description="Does X",
    parameters={"x": {"type": "string"}},
)
def my_tool(x: str) -> dict:
    return {"result": x.upper()}
```

## Important
- Do NOT import other tool modules — use injected parameters instead.
- Functions using `await` must be `async def`.
- Forbidden imports: os, subprocess, sys, socket, pathlib, etc. (sandbox enforced).
"""


def _ensure_skill_guide() -> None:
    """Lazily write the skill authoring guide to /data/ if missing.

    Uses atomic write (temp file + rename) to avoid partial-write corruption.
    """
    import os
    import tempfile

    guide_path = Path("/data/SKILL_AUTHORING.md")
    if not guide_path.exists():
        tmp = None
        try:
            fd, tmp = tempfile.mkstemp(dir="/data", suffix=".md")
            with os.fdopen(fd, "w") as f:
                f.write(_SKILL_AUTHORING_GUIDE)
            os.replace(tmp, str(guide_path))
            tmp = None  # rename succeeded, don't clean up
        except (FileNotFoundError, PermissionError, OSError):
            pass  # /data not mounted or not writable
        finally:
            if tmp:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass


@skill(
    name="create_skill",
    description=(
        "Create a new tool by writing Python code with the @skill decorator. "
        "Import 'from src.agent.skills import skill' and decorate your function "
        "with @skill(name=..., description=..., parameters=...). "
        "Declare keyword parameters for injected dependencies: mesh_client "
        "(mesh API — browser_command, notify_user, read/write_blackboard, "
        "vault_resolve, image_generate, list_agents), workspace_manager, "
        "or memory_store. Async functions must use 'async def'. "
        "Call reload_skills after creation to activate. "
        "Read /data/SKILL_AUTHORING.md for mesh_client API details and examples."
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

    _ensure_skill_guide()

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
        "Reload all skills, picking up any new ones from create_skill or "
        "marketplace installs. Your newly created skill is NOT available "
        "until you call this. Call once after create_skill — you do not need "
        "to call it before every tool use."
    ),
    parameters={},
)
def reload_skills() -> dict:
    return {
        "reload_requested": True,
        "note": "Skills will be reloaded. New tools will be available on next turn.",
    }
