"""Tests for ``src.shared.paths.resolve_under_root``."""
from __future__ import annotations

import os
from pathlib import Path

from src.shared.paths import resolve_under_root


def test_valid_nested_path_returns_resolved(tmp_path: Path):
    """Valid name under root returns the resolved absolute path."""
    (tmp_path / "sub").mkdir()
    result = resolve_under_root(tmp_path, "sub/file.txt")
    assert result == (tmp_path / "sub" / "file.txt").resolve()


def test_dot_dot_traversal_returns_none(tmp_path: Path):
    """`..` traversal that escapes root returns None."""
    assert resolve_under_root(tmp_path, "../escape.txt") is None
    assert resolve_under_root(tmp_path, "sub/../../escape.txt") is None


def test_absolute_path_escapes_returns_none(tmp_path: Path):
    """Absolute path RHS overrides root via `/` operator — must be rejected."""
    assert resolve_under_root(tmp_path, "/etc/passwd") is None


def test_absolute_path_inside_root_still_rejected(tmp_path: Path):
    """Even an absolute name that *would* land inside root is rejected.

    This pins the unconditional ``is_absolute()`` guard: callers must pass
    relative names, otherwise a tightly-targeted absolute path could
    accept a fully-qualified path that happens to fall under root and
    bypass relative-name input validation upstream.
    """
    # Construct an absolute path whose resolved value IS under tmp_path.
    absolute_inside = str((tmp_path / "subdir" / "file.txt").resolve())
    assert resolve_under_root(tmp_path, absolute_inside) is None


def test_symlink_inside_root_resolves(tmp_path: Path):
    """Positive case: symlink whose target IS inside root must succeed.

    Pairs with the negative ``test_symlink_escape_returns_none`` — the
    helper must not reject all symlinks, only those whose target escapes.
    """
    target = tmp_path / "real_file.txt"
    target.write_text("hi")
    link = tmp_path / "link_to_real"
    os.symlink(target, link)
    try:
        result = resolve_under_root(tmp_path, "link_to_real")
        assert result == target.resolve()
    finally:
        link.unlink()
        target.unlink()


def test_symlink_escape_returns_none(tmp_path: Path):
    """A symlink whose target is outside root must be rejected."""
    outside = tmp_path.parent / "outside_target.txt"
    outside.write_text("secret")
    link = tmp_path / "symlink"
    os.symlink(outside, link)
    try:
        assert resolve_under_root(tmp_path, "symlink") is None
    finally:
        link.unlink()
        outside.unlink()


def test_suffix_collision_returns_none(tmp_path: Path):
    """`/a/bcd` is NOT inside `/a/b` — startswith-style checks would
    false-positive here, but is_relative_to correctly rejects."""
    sibling = tmp_path.parent / (tmp_path.name + "_sibling")
    sibling.mkdir()
    try:
        # Build a relative path that escapes root then re-enters a sibling
        # whose name shares a prefix with root.
        rel = f"../{sibling.name}/file.txt"
        assert resolve_under_root(tmp_path, rel) is None
    finally:
        sibling.rmdir()


def test_empty_name_resolves_to_root(tmp_path: Path):
    """Empty name returns root itself (caller validates filename separately)."""
    # Empty / "." both resolve to root and pass containment.
    assert resolve_under_root(tmp_path, "") == tmp_path.resolve()
    assert resolve_under_root(tmp_path, ".") == tmp_path.resolve()


def test_nonexistent_path_still_validates(tmp_path: Path):
    """Helper does not require the candidate to exist — pure path math."""
    result = resolve_under_root(tmp_path, "does_not_exist.md")
    assert result == (tmp_path / "does_not_exist.md").resolve()


def test_nonexistent_root_still_validates(tmp_path: Path):
    """`root.resolve()` does not require existence."""
    fake_root = tmp_path / "does_not_exist_root"
    result = resolve_under_root(fake_root, "file.txt")
    assert result == (fake_root / "file.txt").resolve()


def test_path_argument_accepted(tmp_path: Path):
    """Accepts Path as the name argument too, not only str."""
    result = resolve_under_root(tmp_path, Path("sub/file.txt"))
    assert result == (tmp_path / "sub" / "file.txt").resolve()
