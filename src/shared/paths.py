"""Shared filesystem path helpers — primitives for safe path containment.

Kept separate from ``utils.py`` because path-containment is a security
primitive and benefits from being discoverable in a path-named module
rather than mixed with JSON/id/text helpers.
"""
from __future__ import annotations

from pathlib import Path


def resolve_under_root(root: Path, name: str | Path) -> Path | None:
    """Resolve ``root / name`` and return it iff it stays within ``root``.

    Both ``root`` and the final candidate are resolved (symlinks followed,
    ``..`` collapsed) so the check rejects:

    - traversal via ``..`` (e.g. ``"../etc/passwd"``)
    - **any absolute ``name``** (e.g. ``"/etc/passwd"``, or even
      ``"/x/root_resolved/file"`` that happens to land inside ``root`` —
      relative-name semantics are enforced unconditionally so callers
      never accidentally accept a fully-qualified path)
    - symlinks whose target escapes ``root``
    - suffix-collision prefixes (e.g. ``/tmp/artifacts2`` is NOT inside
      ``/tmp/artifacts`` — ``startswith`` would false-positive here)

    Returns ``None`` when ``name`` is absolute or when the resolved path
    escapes ``root``. Callers own the error response (HTTPException,
    dict-error, silent ``None``).

    Not solved by this helper:

    - TOCTOU between resolve and subsequent file ops (caller's concern)
    - Filename validation (empty string, ``"."`` etc. resolve to ``root``
      itself and pass the containment check; callers should reject those
      upstream via regex/allowlist)
    """
    if Path(name).is_absolute():
        return None
    root_resolved = root.resolve()
    target = (root_resolved / name).resolve()
    if not target.is_relative_to(root_resolved):
        return None
    return target
