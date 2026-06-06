"""Self-authoring: agents can create their own tools at runtime.

The agent writes a Python function, which is validated (syntax-checked),
saved to the tools directory, and hot-reloaded into the registry.

Security framing — the AST validation below is authoring HYGIENE, not a
sandbox or a security boundary. The ``_FORBIDDEN_IMPORTS`` /
``_FORBIDDEN_CALLS`` / ``_FORBIDDEN_ATTRS`` checks and the forgotten-``await``
detector exist to catch obvious footguns in self-authored code (e.g. an
accidental ``open()`` or a sync function that forgot to ``await`` a
mesh_client coroutine), NOT to contain a malicious agent. Agents already
have ``run_command`` — in-container code execution is part of the design —
so a determined agent never needs to smuggle anything past this validator.
The REAL boundary is the Docker container hardening (non-root UID 1000,
``cap_drop=ALL``, ``no-new-privileges``, read-only root fs, memory/CPU/PID
limits). Treat these AST checks as a lint pass that keeps honest tools
well-formed, never as a confinement mechanism.

Marketplace tools (``ToolRegistry.MARKETPLACE_TOOLS_DIR``) are loaded
WITHOUT any load-time AST validation — that path runs arbitrary module
code at import. They are trusted because the marketplace directory is
operator-populated and mounted read-only. If a remote or agent-reachable
marketplace install path is ever added, pin installs to a verified commit
SHA so the code under review is the code that runs.

See ``docs/security-remediation-review-2026-05-29.md`` (M1, H15).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from src.agent.tools import tool

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
_MAX_TOOL_SIZE = 10_000


def _validate_tool_code(code: str) -> str | None:
    """Validate tool code. Returns error message or None if valid."""
    if len(code) > _MAX_TOOL_SIZE:
        return f"Tool code too large ({len(code)} > {_MAX_TOOL_SIZE} chars)"
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"Syntax error: {e}"

    has_tool_decorator = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name) and dec.func.id == "tool":
                    has_tool_decorator = True
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

    if not has_tool_decorator:
        return "Code must contain at least one function with the @tool decorator"

    # Check: sync functions that call mesh_client methods must be async
    # (all mesh_client methods are coroutines — forgetting await silently
    # creates an unawaited coroutine that never executes).
    _ASYNC_PARAMS = {"mesh_client", "memory_store"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):  # only sync defs
            continue
        param_names = {a.arg for a in node.args.args} | {
            a.arg for a in node.args.kwonlyargs
        }
        async_params_used = param_names & _ASYNC_PARAMS
        if not async_params_used:
            continue
        # Sync function declares mesh_client/memory_store — must be async
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and isinstance(child.func.value, ast.Name)
                and child.func.value.id in async_params_used
            ):
                return (
                    f"Function '{node.name}' calls {child.func.value.id}."
                    f"{child.func.attr}() but is not async. "
                    f"Use 'async def {node.name}(...)' and "
                    f"'await {child.func.value.id}.{child.func.attr}(...)'"
                )

    return None


def _sanitize_filename(name: str) -> str:
    """Convert a tool name to a safe Python filename."""
    safe = re.sub(r"[^a-z0-9_]", "_", name.lower())
    return f"custom_{safe}.py"


_TOOL_AUTHORING_GUIDE = """\
# Tool Authoring Guide

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

## Example: Browser Automation Tool

```python
from src.agent.tools import tool

@tool(
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

## Example: Simple Tool (no dependencies)

```python
from src.agent.tools import tool

@tool(
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


def _ensure_tool_guide() -> None:
    """Lazily write the tool authoring guide to /data/ if missing.

    Uses atomic write (temp file + rename) to avoid partial-write corruption.
    """
    import os
    import tempfile

    guide_path = Path("/data/TOOL_AUTHORING.md")
    if not guide_path.exists():
        tmp = None
        try:
            fd, tmp = tempfile.mkstemp(dir="/data", suffix=".md")
            with os.fdopen(fd, "w") as f:
                f.write(_TOOL_AUTHORING_GUIDE)
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


@tool(
    name="create_tool",
    description=(
        "Create a new tool by writing Python code with the @tool decorator. "
        "Import 'from src.agent.tools import tool' and decorate your function "
        "with @tool(name=..., description=..., parameters=...). "
        "Declare keyword parameters for injected dependencies: mesh_client "
        "(mesh API — browser_command, notify_user, read/write_blackboard, "
        "vault_resolve, image_generate, list_agents), workspace_manager, "
        "or memory_store. Async functions must use 'async def'. "
        "Call reload_tools after creation to activate. "
        "Read /data/TOOL_AUTHORING.md for mesh_client API details and examples."
    ),
    parameters={
        "name": {
            "type": "string",
            "description": "Name for this tool (used as filename)",
        },
        "code": {
            "type": "string",
            "description": "Python source code with @tool decorator",
        },
    },
)
def create_tool(name: str, code: str, *, workspace_manager=None) -> dict:
    if workspace_manager is None:
        return {"error": "No workspace_manager available"}

    _ensure_tool_guide()

    error = _validate_tool_code(code)
    if error:
        return {"error": f"Validation failed: {error}"}

    # Write to /data/custom_tools (writable) — /app/tools is read-only
    tools_dir = Path("/data/custom_tools")
    tools_dir.mkdir(parents=True, exist_ok=True)

    filename = _sanitize_filename(name)
    filepath = tools_dir / filename
    filepath.write_text(code)

    return {
        "created": True,
        "name": name,
        "file": str(filepath),
        "note": "Tool saved. Use reload_tools to activate it.",
    }


@tool(
    name="reload_tools",
    description=(
        "Reload all tools, picking up any new ones from create_tool or "
        "marketplace installs. Your newly created tool is NOT available "
        "until you call this. Call once after create_tool — you do not need "
        "to call it before every tool use."
    ),
    parameters={},
)
def reload_tools() -> dict:
    return {
        "reload_requested": True,
        "note": "Tools will be reloaded. New tools will be available on next turn.",
    }
