"""File I/O tools scoped to the agent's /data volume.

All paths are resolved and validated to stay within the allowed
root directory, preventing directory traversal attacks.
"""

from __future__ import annotations

from pathlib import Path

from src.agent.skills import skill

_ALLOWED_ROOT = "/data"
_MAX_READ = 500_000

# Workspace identity files that must not be written via write_file.
# Agents should use the update_workspace tool for writable files
# (HEARTBEAT.md, USER.md) and cannot modify the rest (SOUL.md, AGENTS.md).
_PROTECTED_WORKSPACE_FILES = frozenset({
    "SOUL.md", "AGENTS.md", "HEARTBEAT.md", "USER.md", "MEMORY.md",
})


def _safe_path(path: str) -> Path:
    """Resolve a path and ensure it stays within the allowed root."""
    resolved = Path(_ALLOWED_ROOT, path).resolve()
    root = Path(_ALLOWED_ROOT).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Path escapes allowed root: {path}")
    return resolved


def _is_protected_workspace_file(resolved: Path) -> bool:
    """Check if a resolved path points to a protected workspace file."""
    workspace_root = Path(_ALLOWED_ROOT, "workspace").resolve()
    if not resolved.is_relative_to(workspace_root):
        return False
    relative = resolved.relative_to(workspace_root)
    return relative.name in _PROTECTED_WORKSPACE_FILES and len(relative.parts) == 1


@skill(
    name="read_file",
    description="Read the contents of a file. Returns the text content.",
    parameters={
        "path": {"type": "string", "description": "File path relative to /data"},
        "offset": {
            "type": "integer",
            "description": "Start reading from this line (0-based, default 0)",
            "default": 0,
        },
        "limit": {
            "type": "integer",
            "description": "Max lines to read (default all)",
            "default": 0,
        },
    },
)
def read_file(path: str, offset: int = 0, limit: int = 0) -> dict:
    """Read file contents with optional line offset/limit."""
    safe = _safe_path(path)
    if not safe.exists():
        return {"error": f"File not found: {path}"}
    if not safe.is_file():
        return {"error": f"Not a file: {path}"}

    content = safe.read_text(errors="replace")[:_MAX_READ]
    if offset or limit:
        lines = content.splitlines(keepends=True)
        end = offset + limit if limit else len(lines)
        content = "".join(lines[offset:end])

    return {"content": content, "size": safe.stat().st_size}


@skill(
    name="write_file",
    description="Write content to a file. Creates parent directories if needed.",
    parameters={
        "path": {"type": "string", "description": "File path relative to /data"},
        "content": {"type": "string", "description": "Content to write"},
        "append": {
            "type": "boolean",
            "description": "Append instead of overwrite (default false)",
            "default": False,
        },
    },
)
def write_file(path: str, content: str, append: bool = False) -> dict:
    """Write or append content to a file."""
    safe = _safe_path(path)
    if _is_protected_workspace_file(safe):
        return {
            "error": (
                f"Cannot write to workspace file '{safe.name}' directly. "
                "Use the update_workspace tool for HEARTBEAT.md and USER.md. "
                "SOUL.md and AGENTS.md are read-only (human-controlled)."
            ),
        }
    safe.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with safe.open(mode) as f:
        f.write(content)
    return {"path": str(safe), "bytes_written": len(content.encode())}


@skill(
    name="list_files",
    description="List files and directories. Supports glob patterns.",
    parameters={
        "path": {
            "type": "string",
            "description": "Directory path relative to /data (default '.')",
            "default": ".",
        },
        "pattern": {
            "type": "string",
            "description": "Glob pattern to filter (default '*')",
            "default": "*",
        },
        "recursive": {
            "type": "boolean",
            "description": "Search recursively (default false)",
            "default": False,
        },
    },
)
def list_files(path: str = ".", pattern: str = "*", recursive: bool = False) -> dict:
    """List files matching a pattern."""
    safe = _safe_path(path)
    if not safe.exists():
        return {"error": f"Directory not found: {path}"}
    if not safe.is_dir():
        return {"error": f"Not a directory: {path}"}

    glob_fn = safe.rglob if recursive else safe.glob
    entries = []
    for item in sorted(glob_fn(pattern))[:500]:
        rel = item.relative_to(Path(_ALLOWED_ROOT).resolve())
        entries.append({
            "path": str(rel),
            "type": "dir" if item.is_dir() else "file",
            "size": item.stat().st_size if item.is_file() else 0,
        })

    return {"entries": entries, "count": len(entries)}
