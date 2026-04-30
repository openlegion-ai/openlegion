"""Tests for the browser cap autodetect logic in :mod:`src.browser.__main__`.

Covers:
  * autodetect math at representative box sizes (4/8/16/32 GB)
  * cgroups v2 path takes precedence over /proc/meminfo
  * graceful fallback when both probes fail
  * ``OPENLEGION_BROWSER_MAX_CONCURRENT`` env override beats autodetect
  * legacy ``MAX_BROWSERS`` env override also wins over autodetect
  * clamp to [1, 64]
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.browser.__main__ import (
    _FALLBACK_DEFAULT,
    _MAX_CAP,
    _autodetect_default_max_browsers,
    _detect_total_memory_mb,
    _max_from_memory,
    _resolve_max_browsers,
)


class TestMaxFromMemory:
    """Tests for the pure :func:`_max_from_memory` helper.

    No filesystem mocking required — these exercise the math directly.
    Reading /proc/meminfo is tested separately in :class:`TestMemoryDetection`.
    """

    @pytest.mark.parametrize("total_mb,expected", [
        (4 * 1024, 2),    # 4 GB box → 2 browsers
        (8 * 1024, 11),   # 8 GB box → 11 browsers
        (16 * 1024, 29),  # 16 GB box → 29 browsers
        (32 * 1024, 64),  # 32 GB box → clamped at 64
        (64 * 1024, 64),  # 64 GB box → still clamped
    ])
    def test_math_at_typical_sizes(self, total_mb, expected):
        assert _max_from_memory(total_mb) == expected

    def test_tiny_box_gets_minimum_one(self):
        """A 2 GB box (below the 3 GB headroom) still gets 1 browser slot.

        We never want to fail closed at zero — a single working browser
        is more useful than a clean refusal at startup.
        """
        assert _max_from_memory(2 * 1024) == 1

    def test_exactly_headroom_gets_minimum_one(self):
        """At exactly headroom, available is 0 → still get 1."""
        from src.browser.__main__ import _HEADROOM_MB
        assert _max_from_memory(_HEADROOM_MB) == 1

    def test_below_headroom_gets_minimum_one(self):
        """Below headroom (negative available) → still get 1, not crash."""
        from src.browser.__main__ import _HEADROOM_MB
        assert _max_from_memory(_HEADROOM_MB - 100) == 1

    def test_none_returns_fallback(self):
        """``None`` total (detection failure) → conservative fallback."""
        assert _max_from_memory(None) == _FALLBACK_DEFAULT

    def test_clamp_respects_max_cap_constant(self):
        """Sanity check the clamp uses _MAX_CAP, not a hardcoded number."""
        assert _max_from_memory(1024 * 1024) == _MAX_CAP

    def test_just_enough_for_two_browsers(self):
        """Boundary: exactly 2 browsers' worth of available memory → 2."""
        from src.browser.__main__ import _HEADROOM_MB, _MEM_PER_BROWSER_MB
        assert _max_from_memory(_HEADROOM_MB + 2 * _MEM_PER_BROWSER_MB) == 2


class TestMemoryDetection:
    """Tests for :func:`_detect_total_memory_mb`."""

    def test_cgroups_v2_path_wins(self, tmp_path):
        """When /sys/fs/cgroup/memory.max has a numeric value, use it."""
        cgroup = tmp_path / "memory.max"
        # 8 GB in bytes.
        cgroup.write_text(str(8 * 1024 * 1024 * 1024) + "\n")
        with patch(
            "src.browser.__main__.Path",
            side_effect=lambda p: cgroup if p == "/sys/fs/cgroup/memory.max" else Path(p),
        ):
            mb = _detect_total_memory_mb()
        assert mb == 8 * 1024  # 8 GB == 8192 MB

    def test_cgroups_v2_unbounded_falls_through(self, tmp_path):
        """``memory.max == 'max'`` means no limit; fall through to meminfo."""
        cgroup = tmp_path / "memory.max"
        cgroup.write_text("max\n")
        meminfo = tmp_path / "meminfo"
        # 4 GB in kB.
        meminfo.write_text("MemTotal:       4194304 kB\nSomeOther:    foo\n")

        def _path_factory(p):
            if p == "/sys/fs/cgroup/memory.max":
                return cgroup
            if p == "/proc/meminfo":
                return meminfo
            return Path(p)

        with patch("src.browser.__main__.Path", side_effect=_path_factory):
            mb = _detect_total_memory_mb()
        assert mb == 4 * 1024

    def test_cgroups_v2_garbage_falls_through(self, tmp_path):
        """A non-numeric, non-'max' memory.max value should fall through.

        Hostile / malformed cgroup files (e.g. truncated reads, unicode
        garbage) would otherwise raise ValueError on ``int()`` and crash
        startup if uncaught.  The handler swallows + falls through.
        """
        cgroup = tmp_path / "memory.max"
        cgroup.write_text("not a number\n")
        meminfo = tmp_path / "meminfo"
        meminfo.write_text("MemTotal:       8388608 kB\n")  # 8 GB

        def _path_factory(p):
            if p == "/sys/fs/cgroup/memory.max":
                return cgroup
            if p == "/proc/meminfo":
                return meminfo
            return Path(p)

        with patch("src.browser.__main__.Path", side_effect=_path_factory):
            mb = _detect_total_memory_mb()
        # Garbage in v2 → fell through to meminfo, which had 8 GB.
        assert mb == 8 * 1024

    def test_no_cgroups_meminfo_fallback(self, tmp_path):
        """No cgroups file → use /proc/meminfo."""
        nonexistent = tmp_path / "nope"
        meminfo = tmp_path / "meminfo"
        meminfo.write_text("MemTotal:       16777216 kB\n")  # 16 GB

        def _path_factory(p):
            if p == "/sys/fs/cgroup/memory.max":
                return nonexistent
            if p == "/proc/meminfo":
                return meminfo
            return Path(p)

        with patch("src.browser.__main__.Path", side_effect=_path_factory):
            mb = _detect_total_memory_mb()
        assert mb == 16 * 1024

    def test_meminfo_without_memtotal_returns_none(self, tmp_path):
        """A /proc/meminfo missing MemTotal returns None (caller falls
        through to fallback default)."""
        nonexistent = tmp_path / "nope"
        meminfo = tmp_path / "meminfo"
        meminfo.write_text("Buffers:    1234 kB\nCached:    5678 kB\n")

        def _path_factory(p):
            if p == "/sys/fs/cgroup/memory.max":
                return nonexistent
            if p == "/proc/meminfo":
                return meminfo
            return Path(p)

        with patch("src.browser.__main__.Path", side_effect=_path_factory):
            mb = _detect_total_memory_mb()
        assert mb is None

    def test_meminfo_garbage_memtotal_returns_none(self, tmp_path):
        """A MemTotal line with non-numeric value returns None safely."""
        nonexistent = tmp_path / "nope"
        meminfo = tmp_path / "meminfo"
        meminfo.write_text("MemTotal:       garbage kB\n")

        def _path_factory(p):
            if p == "/sys/fs/cgroup/memory.max":
                return nonexistent
            if p == "/proc/meminfo":
                return meminfo
            return Path(p)

        with patch("src.browser.__main__.Path", side_effect=_path_factory):
            mb = _detect_total_memory_mb()
        assert mb is None

    def test_returns_none_when_both_probes_fail(self, tmp_path):
        """No probe succeeds → None, callers fall through to safe default."""
        nonexistent = tmp_path / "nope"

        def _path_factory(p):
            return nonexistent

        with patch("src.browser.__main__.Path", side_effect=_path_factory):
            mb = _detect_total_memory_mb()
        assert mb is None


class TestAutodetectDefault:
    """End-to-end :func:`_autodetect_default_max_browsers` — the real
    function used by the resolver.  Math is delegated to
    :func:`_max_from_memory` (covered above); this just verifies the
    composition of probe + math."""

    def test_with_known_memory(self):
        with patch(
            "src.browser.__main__._detect_total_memory_mb",
            return_value=8 * 1024,
        ):
            assert _autodetect_default_max_browsers() == 11

    def test_falls_back_when_detection_fails(self):
        with patch(
            "src.browser.__main__._detect_total_memory_mb",
            return_value=None,
        ):
            assert _autodetect_default_max_browsers() == _FALLBACK_DEFAULT


class TestResolveMaxBrowsers:
    """Tests for the public :func:`_resolve_max_browsers` precedence chain.

    Patch :func:`_detect_total_memory_mb` (the actual probe) rather than
    the autodetect helper, because the resolver now reads memory once
    and forwards the value through :func:`_max_from_memory`.  Patching
    the probe exercises the real composition end-to-end.
    """

    def test_canonical_env_wins_over_autodetect(self, monkeypatch):
        """``OPENLEGION_BROWSER_MAX_CONCURRENT`` is the explicit-override path
        the provisioner uses; it must win over both legacy and autodetect."""
        monkeypatch.setenv("OPENLEGION_BROWSER_MAX_CONCURRENT", "20")
        monkeypatch.delenv("MAX_BROWSERS", raising=False)
        with patch(
            "src.browser.__main__._detect_total_memory_mb",
            return_value=4 * 1024,  # autodetects to 2
        ):
            assert _resolve_max_browsers() == 20

    def test_legacy_env_wins_over_autodetect(self, monkeypatch):
        """``MAX_BROWSERS`` (legacy name) still works when canonical unset."""
        monkeypatch.delenv("OPENLEGION_BROWSER_MAX_CONCURRENT", raising=False)
        monkeypatch.setenv("MAX_BROWSERS", "12")
        with patch(
            "src.browser.__main__._detect_total_memory_mb",
            return_value=4 * 1024,
        ):
            assert _resolve_max_browsers() == 12

    def test_autodetect_used_when_no_env(self, monkeypatch):
        """Self-host path: no env → autodetect is the floor.

        16 GB box autodetects to 29 — verify that's what we get when
        no override is in play.
        """
        monkeypatch.delenv("OPENLEGION_BROWSER_MAX_CONCURRENT", raising=False)
        monkeypatch.delenv("MAX_BROWSERS", raising=False)
        with patch(
            "src.browser.__main__._detect_total_memory_mb",
            return_value=16 * 1024,
        ):
            assert _resolve_max_browsers() == 29

    def test_canonical_env_overrides_legacy(self, monkeypatch):
        """When both are set, the canonical name takes precedence."""
        monkeypatch.setenv("OPENLEGION_BROWSER_MAX_CONCURRENT", "30")
        monkeypatch.setenv("MAX_BROWSERS", "5")
        with patch(
            "src.browser.__main__._detect_total_memory_mb",
            return_value=4 * 1024,
        ):
            assert _resolve_max_browsers() == 30

    def test_clamp_applies_to_env_overrides(self, monkeypatch):
        """``OPENLEGION_BROWSER_MAX_CONCURRENT=999`` clamps to 64, not crash."""
        monkeypatch.setenv("OPENLEGION_BROWSER_MAX_CONCURRENT", "999")
        with patch(
            "src.browser.__main__._detect_total_memory_mb",
            return_value=4 * 1024,
        ):
            assert _resolve_max_browsers() == _MAX_CAP

    def test_clamp_rejects_zero(self, monkeypatch):
        """``MAX=0`` would deadlock; min clamp = 1."""
        monkeypatch.setenv("OPENLEGION_BROWSER_MAX_CONCURRENT", "0")
        with patch(
            "src.browser.__main__._detect_total_memory_mb",
            return_value=8 * 1024,
        ):
            assert _resolve_max_browsers() == 1

    def test_falls_back_to_default_when_memory_undetectable(self, monkeypatch):
        """Detection failure + no env override → :data:`_FALLBACK_DEFAULT`."""
        monkeypatch.delenv("OPENLEGION_BROWSER_MAX_CONCURRENT", raising=False)
        monkeypatch.delenv("MAX_BROWSERS", raising=False)
        with patch(
            "src.browser.__main__._detect_total_memory_mb",
            return_value=None,
        ):
            assert _resolve_max_browsers() == _FALLBACK_DEFAULT
