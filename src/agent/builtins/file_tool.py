"""File I/O tools scoped to the agent's /data volume.

All paths are resolved and validated to stay within the allowed
root directory, preventing directory traversal attacks.
"""

from __future__ import annotations

from pathlib import Path

from src.agent.skills import skill

_ALLOWED_ROOT = "/data"
_MAX_READ = 500_000


def _safe_path(path: str) -> Path:
    """Resolve a path and ensure it stays within the allowed root."""
    resolved = Path(_ALLOWED_ROOT, path).resolve()
    root = Path(_ALLOWED_ROOT).resolve()
    if not str(resolved).startswith(str(root)):
        raise ValueError(f"Path escapes allowed root: {path}")
    return resolved


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
