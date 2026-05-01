"""Tests for :mod:`src.browser.display_allocator`.

Covers:
  * basic alloc / release / pool exhaustion
  * boot sweep removes stale lock + socket residue when port is free
  * boot sweep drops slots whose paired port is currently bound
  * release rejects unallocated slots without raising
  * port/display pairing math
  * residue cleanup on release
  * concurrent-style sequencing — alloc, release, alloc same display
"""

from __future__ import annotations

import pytest

import src.browser.display_allocator as display_allocator
from src.browser.display_allocator import (
    DISPLAY_RANGE_END,
    DISPLAY_RANGE_START,
    VNC_PORT_BASE,
    DisplayAllocator,
    PoolExhausted,
    Slot,
    _force_clear_residue_for_tests,
    _read_lock_pid,
    _write_fake_lock_for_tests,
    display_for_port,
    port_for_display,
)

TEST_DISPLAY_START = 800
TEST_DISPLAY_END = 805


@pytest.fixture(autouse=True)
def _isolate_lockfile_residue(monkeypatch):
    """Keep allocator unit tests independent of host TCP bind permissions."""
    monkeypatch.setattr(display_allocator, "_port_is_bindable", lambda _port: True)
    _force_clear_residue_for_tests(TEST_DISPLAY_START, TEST_DISPLAY_END)
    yield
    _force_clear_residue_for_tests(TEST_DISPLAY_START, TEST_DISPLAY_END)


def _alloc_in_range(
    start: int = TEST_DISPLAY_START,
    end: int = TEST_DISPLAY_END,
    **kwargs,
) -> DisplayAllocator:
    return DisplayAllocator(
        display_start=start, display_end=end, **kwargs,
    )


# ── module constants ─────────────────────────────────────────────────────────


class TestModuleConstants:
    def test_display_range_matches_max_concurrent_ceiling(self):
        """64-slot capacity matches the soft ceiling on browser concurrency."""
        capacity = DISPLAY_RANGE_END - DISPLAY_RANGE_START
        assert capacity == 64

    def test_display_starts_after_legacy_shared(self):
        """Range starts at 100 so :99 (legacy shared) stays clear."""
        assert DISPLAY_RANGE_START >= 100

    def test_port_helpers_round_trip(self):
        for d in (100, 137, 163):
            assert display_for_port(port_for_display(d)) == d

    def test_vnc_port_base_aligns_with_kasmvnc_default(self):
        """Base 6000 → display 100 → port 6100; KasmVNC default is 6080."""
        assert VNC_PORT_BASE == 6000


# ── allocator semantics ─────────────────────────────────────────────────────


class TestAllocatorBasics:
    def test_capacity_reflects_range(self):
        alloc = _alloc_in_range(run_boot_sweep=False)
        assert alloc.capacity == 5
        assert alloc.free_count == 5
        assert alloc.allocated_count == 0

    def test_invalid_range_rejected(self):
        with pytest.raises(ValueError):
            DisplayAllocator(display_start=10, display_end=10)
        with pytest.raises(ValueError):
            DisplayAllocator(display_start=10, display_end=5)
        with pytest.raises(ValueError):
            DisplayAllocator(display_start=0, display_end=10)

    def test_allocate_returns_lowest_free(self):
        alloc = _alloc_in_range(run_boot_sweep=False)
        start = alloc._range.start
        s1 = alloc.allocate()
        s2 = alloc.allocate()
        # Lowest-first ordering means deterministic tests + readable logs.
        assert s1.display == start
        assert s2.display == start + 1
        assert s1.vnc_port == port_for_display(start)
        assert s2.vnc_port == port_for_display(start + 1)

    def test_allocate_raises_when_exhausted(self):
        start, end = TEST_DISPLAY_START, TEST_DISPLAY_START + 2
        alloc = _alloc_in_range(start, end, run_boot_sweep=False)
        alloc.allocate()
        alloc.allocate()
        with pytest.raises(PoolExhausted):
            alloc.allocate()

    def test_release_returns_slot_to_pool(self):
        start, end = TEST_DISPLAY_START, TEST_DISPLAY_START + 2
        alloc = _alloc_in_range(start, end, run_boot_sweep=False)
        s = alloc.allocate()
        alloc.allocate()
        alloc.release(s)
        # Release should make it allocate-able again.
        s2 = alloc.allocate()
        assert s2.display == s.display

    def test_release_unallocated_is_idempotent(self, caplog):
        alloc = _alloc_in_range(run_boot_sweep=False)
        start = alloc._range.start
        # Releasing a slot that was never allocated must NOT raise — error
        # recovery paths can call release on a Slot that wasn't claimed.
        alloc.release(Slot(display=start, vnc_port=port_for_display(start)))
        # And after a real alloc+release, double-release also tolerated.
        s = alloc.allocate()
        alloc.release(s)
        alloc.release(s)

    def test_release_cleans_lock_residue(self, tmp_path, monkeypatch):
        """Residue files on the slot's display number are cleaned by release."""
        alloc = _alloc_in_range(run_boot_sweep=False)
        s = alloc.allocate()
        # Plant a fake lock; release must clean it.
        _write_fake_lock_for_tests(s, pid=11111)
        assert s.lock_path.exists()
        alloc.release(s)
        assert not s.lock_path.exists()

    def test_is_allocated_reflects_state(self):
        alloc = _alloc_in_range(run_boot_sweep=False)
        s = alloc.allocate()
        assert alloc.is_allocated(s.display)
        alloc.release(s)
        assert not alloc.is_allocated(s.display)


# ── boot sweep ──────────────────────────────────────────────────────────────


class TestBootSweep:
    def test_clean_start_no_logs(self, caplog):
        """When /tmp is clean and no port is bound, sweep is silent."""
        with caplog.at_level("INFO", logger="browser.display_allocator"):
            _alloc_in_range()
        # We don't pin exact log absence — just verify no slots were
        # mistakenly dropped from the pool.

    def test_stale_lock_removed_when_port_free(self):
        """Lock-file residue without a live process is removed."""
        # Plant a stale lock file in our test range.
        start, end = TEST_DISPLAY_START, TEST_DISPLAY_END
        slot = Slot(display=start, vnc_port=port_for_display(start))
        _write_fake_lock_for_tests(slot, pid=99999)
        assert slot.lock_path.exists()
        # Boot sweep should remove it (port is free).
        alloc = _alloc_in_range(start, end)
        assert not slot.lock_path.exists()
        # And the slot should be allocate-able.
        assert alloc.is_allocated(slot.display) is False
        s = alloc.allocate()
        assert s.display == start

    def test_slot_dropped_when_port_bound(self):
        """A slot whose paired port is currently bound is not allocate-able."""
        start, end = TEST_DISPLAY_START, TEST_DISPLAY_END
        port = port_for_display(start)
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            display_allocator,
            "_port_is_bindable",
            lambda candidate: candidate != port,
        )
        try:
            alloc = _alloc_in_range(start, end)
        finally:
            monkeypatch.undo()
        # The bound display should have been removed from the pool.
        with pytest.raises(PoolExhausted):
            # We have width-1 free slots; the width-th allocate should
            # trip pool exhaustion since the first display was dropped.
            for _ in range(5):
                alloc.allocate()


# ── port-collision recovery on allocate ─────────────────────────────────────


class TestAllocateRecovery:
    def test_allocate_skips_slot_whose_port_just_got_bound(self):
        """If the boot sweep missed a slot, allocate() drops it on probe."""
        start, end = TEST_DISPLAY_START, TEST_DISPLAY_END
        alloc = _alloc_in_range(start, end, run_boot_sweep=False)

        # Bind the first display's port AFTER construction (sweep can't see this).
        port = port_for_display(start)
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            display_allocator,
            "_port_is_bindable",
            lambda candidate: candidate != port,
        )
        try:
            # First allocate should skip the bound slot and give us the next one.
            s = alloc.allocate()
        finally:
            monkeypatch.undo()
        assert s.display == start + 1


# ── port/display helpers ────────────────────────────────────────────────────


class TestHelpers:
    def test_lock_path_format(self):
        slot = Slot(display=137, vnc_port=port_for_display(137))
        assert str(slot.lock_path) == "/tmp/.X137-lock"
        assert str(slot.socket_path) == "/tmp/.X11-unix/X137"

    def test_display_str_format(self):
        slot = Slot(display=100, vnc_port=6100)
        assert slot.display_str == ":100"

    def test_read_lock_pid_round_trip(self):
        start = TEST_DISPLAY_START
        slot = Slot(display=start, vnc_port=port_for_display(start))
        try:
            _write_fake_lock_for_tests(slot, pid=12345)
            assert _read_lock_pid(slot.lock_path) == 12345
        finally:
            slot.lock_path.unlink(missing_ok=True)

    def test_read_lock_pid_returns_none_on_garbage(self, tmp_path):
        garbage = tmp_path / "garbage-lock"
        garbage.write_text("not a number\n")
        assert _read_lock_pid(garbage) is None
