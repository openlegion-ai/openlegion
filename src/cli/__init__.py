"""CLI package for OpenLegion."""

from __future__ import annotations

import sys
from pathlib import Path

# Guard: detect if the editable install points at a stale worktree.
# This happens when `pip install -e .` runs inside a git worktree —
# it silently hijacks the global `openlegion` entry point to the
# worktree's frozen source, ignoring the main checkout entirely.
_src_root = Path(__file__).resolve().parent.parent.parent
if ".claude" in _src_root.parts and "pytest" not in sys.modules:
    print(
        f"\n  ERROR: openlegion is running from a worktree: {_src_root}\n"
        f"  This means code changes on main are NOT being used.\n"
        f"  Fix: run 'pip install -e .' from your main checkout.\n",
        file=sys.stderr,
    )
    raise SystemExit(1)

from src.cli.main import cli  # noqa: F401, E402
