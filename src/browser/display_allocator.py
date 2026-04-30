"""Per-agent X11 display + KasmVNC port pair allocator.

PR 1 of the per-agent VNC isolation work.  The shared-display design
(one ``Xvnc :99`` for every agent — see :mod:`src.browser.__main__`'s
legacy path) leaks every agent's browser content to every VNC viewer
because KasmVNC streams the whole framebuffer.  Per-agent isolation
requires one Xvnc per agent.

This module owns the allocation of ``(display_num, vnc_port)`` pairs.
Display ``N`` pairs with port ``VNC_PORT_BASE + N`` so a single integer
identifies a slot — no separate port allocator state.

Boot sweep
----------
``/tmp/.X{N}-lock`` files from a previous crash hold their display
number against allocation.  :class:`DisplayAllocator` runs a boot phase
that walks the configured display range, decides per-slot whether the
display is genuinely in use (paired TCP port currently bound), and
removes lock-file + abstract-socket residue when the slot is free.
Without this sweep every container restart loses pool slots until
manual cleanup.

Probe-based occupancy check (rather than parsing the PID out of the
lock file and killing it) keeps us out of the kill-the-wrong-process
failure mode.  If the port is bound, something is using the slot —
leave it alone.  If the port is free, no X server can be live on the
display, so any lock-file residue is safe to remove.

Range
-----
* display ``100..163`` — 64 slots, the soft ceiling for concurrent
  browsers (matches the ``OPENLEGION_BROWSER_MAX_CONCURRENT`` clamp in
  :mod:`src.browser.__main__`).
* port ``6100..6163`` — paired 1:1.  The legacy shared display lives on
  ``:99`` / ``:6080``; starting per-agent slots at 100 lets both code
  paths coexist during the flag-gated rollout (PR 2 deletes the
  legacy path).

The allocator itself is sync — slots are claimed under :class:`asyncio.Lock`
in :class:`BrowserManager` so we don't need a second lock here.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass
from pathlib import Path

from src.shared.utils import setup_logging

logger = setup_logging("browser.display_allocator")


# Display N → port (VNC_PORT_BASE + N).  Keep base aligned with KasmVNC's
# default 6080 so per-agent ports are nearby and operators recognise the
# range — :6100 reads as "the new VNC ports" without a docs lookup.
VNC_PORT_BASE = 6000

# Inclusive on both ends; Python ranges exclude the upper bound, so
# 164 = 64 slots starting at 100.
DISPLAY_RANGE_START = 100
DISPLAY_RANGE_END = 164


def display_for_port(port: int) -> int:
    return port - VNC_PORT_BASE


def port_for_display(display: int) -> int:
    return VNC_PORT_BASE + display


@dataclass(frozen=True)
class Slot:
    """An allocated (display, port) pair.

    ``display`` is the X11 display number (e.g. ``100`` → ``DISPLAY=:100``).
    ``vnc_port`` is the KasmVNC websocket port paired with that display.
    """
    display: int
    vnc_port: int

    @property
    def display_str(self) -> str:
        return f":{self.display}"

    @property
    def lock_path(self) -> Path:
        return Path(f"/tmp/.X{self.display}-lock")

    @property
    def socket_path(self) -> Path:
        return Path(f"/tmp/.X11-unix/X{self.display}")


class DisplayAllocator:
    """Pool allocator for ``(display, vnc_port)`` slots.

    Construction runs the boot sweep so the caller doesn't have to
    remember it.  Idempotent re-construction is safe: the sweep only
    touches lock/socket residue when the paired port is genuinely free.

    Tests can override the range via the keyword args without touching
    module-level constants.
    """

    def __init__(
        self,
        *,
        display_start: int = DISPLAY_RANGE_START,
        display_end: int = DISPLAY_RANGE_END,
        run_boot_sweep: bool = True,
    ):
        if display_start <= 0 or display_end <= display_start:
            raise ValueError(
                f"Invalid display range [{display_start}, {display_end})",
            )
        self._range = range(display_start, display_end)
        # Free slots, ordered low→high so allocations are deterministic
        # (test-friendly + easier to read in logs).
        self._free: list[int] = list(self._range)
        # Currently-allocated displays.  Set semantics catch double-release
        # bugs loudly.
        self._allocated: set[int] = set()
        if run_boot_sweep:
            self._boot_sweep()

    # ── public API ────────────────────────────────────────────────────────

    def allocate(self) -> Slot:
        """Reserve the lowest-numbered free slot.

        Verifies the slot's port is currently bindable before returning;
        if the boot sweep missed something (e.g. a peer X server bound the
        port between sweep and allocate), the slot is dropped from the
        pool and the next candidate tried.

        Raises :class:`PoolExhausted` when no slot is available.
        """
        while self._free:
            display = self._free.pop(0)
            port = port_for_display(display)
            if not _port_is_bindable(port):
                # Something legitimately holds the port — drop the slot
                # permanently.  Surfacing in logs so an operator notices
                # if many slots leak this way.
                logger.warning(
                    "Skipping display :%d — port %d not bindable; "
                    "dropping slot from pool",
                    display, port,
                )
                continue
            self._allocated.add(display)
            slot = Slot(display=display, vnc_port=port)
            logger.debug(
                "Allocated display :%d (port %d); %d slots remain",
                display, port, len(self._free),
            )
            return slot
        raise PoolExhausted(
            f"No free slots in range [{self._range.start}, "
            f"{self._range.stop}); concurrent browser cap reached",
        )

    def release(self, slot: Slot) -> None:
        """Return a slot to the pool.

        Idempotent on already-released slots (logs a warning rather than
        raising — release is called from teardown paths where double-call
        is plausible during error recovery, and we'd rather leak a log
        line than fail the cleanup).

        Cleans residual lock + socket files so the next allocator on this
        slot starts from a known-clean state.  Does NOT verify the
        corresponding processes are dead — that is the caller's job
        (see :class:`BrowserManager._stop_instance` process-group teardown).
        """
        if slot.display not in self._allocated:
            logger.warning(
                "release(:%d) called but slot is not allocated; ignoring",
                slot.display,
            )
            return
        self._allocated.discard(slot.display)
        _remove_residue(slot)
        # Re-insert in sorted order so allocate() stays deterministic.
        self._free.append(slot.display)
        self._free.sort()
        logger.debug(
            "Released display :%d (port %d); %d slots free",
            slot.display, slot.vnc_port, len(self._free),
        )

    def is_allocated(self, display: int) -> bool:
        return display in self._allocated

    @property
    def free_count(self) -> int:
        return len(self._free)

    @property
    def allocated_count(self) -> int:
        return len(self._allocated)

    @property
    def capacity(self) -> int:
        return self._range.stop - self._range.start

    # ── boot sweep ────────────────────────────────────────────────────────

    def _boot_sweep(self) -> None:
        """Reclaim slots whose lock/socket residue survived a crash.

        Probe-based: for each display N in our range, if port ``VNC_PORT_BASE+N``
        is currently bindable, no live X server can hold the display, so any
        ``/tmp/.X{N}-lock`` and abstract-socket file is residue we can safely
        remove.  If the port is NOT bindable, drop the slot from the pool —
        something genuinely holds it and we shouldn't fight.

        Only logs at INFO level when residue is actually removed; quiet
        when the container starts clean.
        """
        cleaned = 0
        live = 0
        for display in list(self._range):
            port = port_for_display(display)
            if _port_is_bindable(port):
                # Slot is genuinely free; nuke any residue.
                slot = Slot(display=display, vnc_port=port)
                if _remove_residue(slot):
                    cleaned += 1
            else:
                # Port held by something — drop slot from pool.  When the
                # holder is the legacy shared Xvnc (which lives on :99 +
                # :6080, not in our range), this branch is never reached
                # for our slots.  When the holder is a same-range X server
                # left by a peer process (rare), dropping is correct.
                self._free.remove(display)
                live += 1
        if cleaned or live:
            logger.info(
                "Display-allocator boot sweep: %d residue cleaned, "
                "%d slot(s) reserved by live processes",
                cleaned, live,
            )


class PoolExhausted(RuntimeError):
    """Raised when no free display/port slot is available."""


# ── helpers ────────────────────────────────────────────────────────────────


def _port_is_bindable(port: int) -> bool:
    """Return True if ``port`` can be bound on 0.0.0.0 right now.

    Uses ``SO_REUSEADDR`` so a port in TIME_WAIT from a recently-killed
    Xvnc still reports as bindable — TIME_WAIT shouldn't reserve a slot
    in our pool.  We close immediately; the actual bind by Xvnc happens
    seconds later, so a TOCTOU race here is theoretical (same-host
    container, no peer service spawning on these ports).
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", port))
        except OSError:
            return False
        return True
    finally:
        sock.close()


def _remove_residue(slot: Slot) -> bool:
    """Remove stale ``/tmp/.X{N}-lock`` and abstract socket for ``slot``.

    Returns True if any residue was actually removed (caller logs).  Best
    effort: failures are logged but don't propagate — the next allocator
    pass will retry, and Xvnc itself will refuse to start cleanly if a
    real peer holds the resources, which is the safer failure mode.
    """
    removed = False
    for path in (slot.lock_path, slot.socket_path):
        try:
            if path.exists() or path.is_symlink():
                path.unlink()
                removed = True
        except OSError as exc:
            logger.warning(
                "Could not remove residue %s: %s", path, exc,
            )
    return removed


def is_per_agent_display_enabled(*, agent_id: str | None = None) -> bool:
    """Return True when the per-agent X stack should be used.

    Reads ``OPENLEGION_BROWSER_PER_AGENT_DISPLAY`` via the centralised
    flag layer so operators can flip it via ``config/settings.json`` or
    the agent-override path without redeploying.  Defaults to ``False``
    so PR 1 ships dark — PR 2's mesh+dashboard work flips the default.
    """
    # Local import to avoid a hard dependency on the flag module from
    # tests that exercise the allocator standalone.
    from src.browser.flags import get_bool

    return get_bool(
        "OPENLEGION_BROWSER_PER_AGENT_DISPLAY",
        _DEFAULT_PER_AGENT_DISPLAY,
        agent_id=agent_id,
    )


# Indirection so tests can patch the default without touching env state.
_DEFAULT_PER_AGENT_DISPLAY = False


# Convenience for tests / debugging — clear the env so a fresh allocator
# pass observes a pristine /tmp.  NOT used in production.
def _force_clear_residue_for_tests(display_start: int, display_end: int) -> None:
    for display in range(display_start, display_end):
        slot = Slot(display=display, vnc_port=port_for_display(display))
        _remove_residue(slot)


def _read_lock_pid(path: Path) -> int | None:
    """Read the PID from an X11 lock file.

    Format: 10-char zero-padded ASCII PID + newline.  Used only by tests
    that simulate stale lock files; production code paths don't need to
    read the PID.  Returns None on parse failure.
    """
    try:
        raw = path.read_text().strip()
        return int(raw)
    except (OSError, ValueError):
        return None


def _write_fake_lock_for_tests(slot: Slot, pid: int = 99999) -> None:
    """Test helper — write a plausible /tmp/.X{N}-lock file."""
    slot.lock_path.parent.mkdir(parents=True, exist_ok=True)
    slot.lock_path.write_text(f"{pid:>10}\n")


__all__ = [
    "DisplayAllocator",
    "PoolExhausted",
    "Slot",
    "VNC_PORT_BASE",
    "DISPLAY_RANGE_START",
    "DISPLAY_RANGE_END",
    "display_for_port",
    "port_for_display",
    "is_per_agent_display_enabled",
]


def _self_check() -> None:
    """Used by ``python -m src.browser.display_allocator`` smoke tests.

    Intentionally not behind ``if __name__ == '__main__'`` so callers can
    invoke it directly in CI without Argparse plumbing.
    """
    alloc = DisplayAllocator()
    s1 = alloc.allocate()
    s2 = alloc.allocate()
    assert s1.display != s2.display
    assert s1.vnc_port == port_for_display(s1.display)
    assert alloc.allocated_count == 2
    alloc.release(s1)
    alloc.release(s2)
    assert alloc.allocated_count == 0
    print(
        f"display_allocator OK — capacity={alloc.capacity}, "
        f"free={alloc.free_count}",
    )


if __name__ == "__main__":  # pragma: no cover
    _self_check()
