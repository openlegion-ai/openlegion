"""Browser service entry point.

Starts the FastAPI browser command server. Each agent's browser is
spawned with its own Xvnc + Openbox + unclutter stack inside
:meth:`BrowserManager._start_browser` (no global X stack at boot).
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from src.browser.server import create_browser_app
from src.browser.service import BrowserManager
from src.shared.utils import setup_logging

logger = setup_logging("browser.main")


def _assert_websockets_available() -> None:
    """Fail fast at startup if the WebSocket transport library is missing.

    uvicorn ships WebSocket support only via the ``[standard]`` extra
    (which pulls in ``websockets``); plain ``uvicorn`` rejects every
    WS upgrade with HTTP 404 at the protocol layer, BEFORE our route
    handler runs. The per-agent VNC iframe needs WS upgrades to work,
    so a missing ``websockets`` package silently breaks the dashboard.
    Catching it here turns "users see a broken VNC iframe" into
    "container exits loud at boot".

    Lives in a function (not module-level) so importing this module
    in tests / introspection scripts doesn't ``sys.exit(1)`` when the
    package happens to be absent in that environment.
    """
    try:
        import websockets  # noqa: F401
    except ImportError:  # pragma: no cover — regression guard
        logger.critical(
            "websockets package not installed. The browser service cannot "
            "serve WS upgrades for /agent-vnc/{agent_id}/{path}. Install "
            "``uvicorn[standard]`` (or add ``websockets`` explicitly) in "
            "Dockerfile.browser.",
        )
        sys.exit(1)


_API_PORT = int(os.environ.get("API_PORT", "8500"))


# Memory budget per Camoufox + per-agent X stack, in MiB.  ~400 MB is
# the high-water mark of a typical Camoufox instance under sustained
# use; ~50 MB covers the per-agent Xvnc + Openbox + unclutter.
_MEM_PER_BROWSER_MB = 450
# Reserved for OS + mesh + agent containers + Docker overhead.  The
# CLAUDE.md resource budget for the engine quotes ~2.5 GB overhead;
# add a 512 MB buffer so a fresh-install on a small box has slack
# instead of teetering on the edge of OOM.
_HEADROOM_MB = 3072
# Hard ceiling — matches the display-allocator pool size.  Going above
# this requires raising both knobs together; the comment in
# src/browser/display_allocator.py:DISPLAY_RANGE_END is the source.
_MAX_CAP = 64
# Last-resort floor when memory autodetect fails.  4 GB box × the
# budget above gives ~2 browsers; 5 was the legacy default and has
# shipped for a year so we preserve it as the conservative fallback.
_FALLBACK_DEFAULT = 5


def _detect_total_memory_mb() -> int | None:
    """Return total memory available to this process in MiB, or None.

    Probes cgroups v2 first (the modern container path), falls back to
    ``/proc/meminfo`` for bare-metal / VM, and gives up silently on
    anything else.  Catches every exception so a hostile / missing
    /proc /sys never crashes startup — the caller falls through to
    :data:`_FALLBACK_DEFAULT`.

    On cgroups v1 (the EOL Hetzner path is already on cgroups v2 since
    Ubuntu 22.04, and CLAUDE.md pins Ubuntu 24.04) we'd skip the v2
    file and read /proc/meminfo, which reports the host's memory not
    the cgroup limit.  That's still useful as a coarse upper bound;
    misconfigurations report a too-large value but the cap clamp keeps
    us safe.
    """
    try:
        v2 = Path("/sys/fs/cgroup/memory.max")
        if v2.exists():
            raw = v2.read_text().strip()
            if raw and raw != "max":
                return int(raw) // (1024 * 1024)
    except Exception:
        pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                # Format: ``MemTotal:       16389184 kB``
                return int(line.split()[1]) // 1024
    except Exception:
        pass
    return None


def _max_from_memory(total_mb: int | None) -> int:
    """Pure helper — derive a safe cap from a known memory size.

    Split out from :func:`_autodetect_default_max_browsers` so the
    resolver can share a single ``_detect_total_memory_mb`` reading
    with its log line, and so tests can exercise the math without
    mocking ``Path``.

    Math: ``(total_mem_mb - headroom_mb) // mem_per_browser_mb``,
    clamped to ``[1, _MAX_CAP]``.

    * ``None`` total → :data:`_FALLBACK_DEFAULT` (probes failed,
      typically a non-Linux dev environment)
    * available < one browser's budget → 1 (don't fail closed at 0)
    """
    if total_mb is None:
        return _FALLBACK_DEFAULT
    available = total_mb - _HEADROOM_MB
    if available < _MEM_PER_BROWSER_MB:
        # Tiny box — give one browser slot so the service can still
        # serve a single agent rather than failing closed at zero.
        return 1
    return max(1, min(_MAX_CAP, available // _MEM_PER_BROWSER_MB))


def _autodetect_default_max_browsers() -> int:
    """Estimate a safe default cap from observable memory.

    Used only when ``OPENLEGION_BROWSER_MAX_CONCURRENT`` is unset —
    i.e. self-host installs that haven't tuned anything. Hetzner-
    provisioned VPSes get the cap from the provisioner, which always
    wins over the autodetect (the env-var layer is checked first in
    :func:`_resolve_max_browsers`).

    Reference table:

    * 4 GB box   → (4096 − 3072) / 450 = 2 browsers
    * 8 GB box   → (8192 − 3072) / 450 = 11 browsers
    * 16 GB box  → (16384 − 3072) / 450 = 29 browsers
    * 32 GB box  → (32768 − 3072) / 450 = 64 (clamped)
    """
    return _max_from_memory(_detect_total_memory_mb())


def _resolve_max_browsers() -> int:
    """Return the per-service browser concurrency cap.

    Startup-only — runtime reconfig is non-trivial (would need to bound an
    asyncio.Semaphore mid-flight, drain over-budget instances, and unwind any
    in-flight ``acquire`` waiters). Operators restart the browser service to
    change this. Plan §10.2 keeps this as the explicit decision after the v3.6
    review flagged the runtime path's complexity.

    Precedence (highest → lowest):
      1. ``OPENLEGION_BROWSER_MAX_CONCURRENT`` — canonical env override,
         set by the provisioner per VPS plan and overridable by
         self-hosters. Listed in :data:`src.browser.flags.KNOWN_FLAGS`.
      2. Memory-derived autodetect (:func:`_autodetect_default_max_browsers`).
         Replaces the previous hardcoded ``5`` floor — a 4 GB laptop and
         a 32 GB VPS deserve different defaults.
    """
    from src.browser.flags import get_int

    # Read memory once and forward the result so the autodetect helper
    # and the log line stay consistent — calling _detect_total_memory_mb
    # twice was harmless but cosmetically inconsistent (a thread or
    # cgroup change between calls could disagree).
    total_mb = _detect_total_memory_mb()
    autodetected = _max_from_memory(total_mb)

    # ``min_value=1`` rejects nonsense like ``MAX=0`` (would deadlock the
    # acquire loop). ``max_value=64`` matches the display-allocator pool
    # ceiling — raising one without the other would cause allocator
    # exhaustion under per-agent display mode.
    final = get_int(
        "OPENLEGION_BROWSER_MAX_CONCURRENT",
        autodetected,
        min_value=1,
        max_value=_MAX_CAP,
    )
    # Single log line carrying both the resolved cap and the autodetect
    # baseline, so an operator can tell at a glance whether the env
    # override is doing what they expect.  ``memory_mb=None`` means the
    # autodetect probes failed (typically a non-Linux dev environment),
    # in which case ``autodetected`` is the conservative fallback.
    suffix = " (env override)" if final != autodetected else ""
    logger.info(
        "Browser cap: %d%s — autodetect=%d, memory=%s MB",
        final, suffix, autodetected, total_mb,
    )
    return final


_MAX_BROWSERS = _resolve_max_browsers()
_IDLE_TIMEOUT = int(os.environ.get("IDLE_TIMEOUT_MINUTES", "30"))


def _cleanup_orphan_downloads() -> None:
    """Blanket-delete the download staging dir on startup to clear crashed-tab orphans.

    Idempotent and non-fatal; logs but never raises.
    """
    from src.browser.flags import get_str
    dl_dir = Path(get_str("BROWSER_DOWNLOAD_DIR", "/tmp/downloads"))
    if not dl_dir.is_dir():
        return
    removed = 0
    for entry in dl_dir.iterdir():
        try:
            if entry.is_file() or entry.is_symlink():
                entry.unlink(missing_ok=True)
                removed += 1
        except OSError as e:
            logger.warning("Could not remove orphan download %s: %s", entry, e)
    if removed:
        logger.info("Cleared %d orphan download(s) from %s", removed, dl_dir)


def main() -> None:
    """Start the FastAPI browser command server.

    Per-agent mode: every agent's browser gets its own
    Xvnc + Openbox + unclutter stack inside
    :meth:`BrowserManager._start_browser`. No global X processes
    exist at boot; nothing to start or tear down here besides the
    FastAPI app and the manager's cleanup loop.
    """
    _assert_websockets_available()
    manager = BrowserManager(
        profiles_dir="/data/profiles",
        max_concurrent=_MAX_BROWSERS,
        idle_timeout_minutes=_IDLE_TIMEOUT,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _cleanup_orphan_downloads()
        # §11.14 simplified cost counter — load any prior month's state on
        # startup (skips rolled-over months). Snapshotted on shutdown below
        # so restarts mid-month don't lose accumulated spend.
        from src.browser import captcha_cost_counter as _cost_counter
        try:
            await _cost_counter.restore()
        except Exception as exc:
            logger.warning("captcha_cost_counter.restore failed: %s", exc)
        await manager.start_cleanup_loop()
        logger.info("Browser service ready (max=%d, idle_timeout=%dm)", _MAX_BROWSERS, _IDLE_TIMEOUT)
        yield
        try:
            await _cost_counter.snapshot()
        except Exception as exc:
            logger.warning("captcha_cost_counter.snapshot failed: %s", exc)
        # ``stop_all`` tears down every per-agent X stack via
        # ``_teardown_per_agent_x_stack``; no global processes to reap.
        await manager.stop_all()
        logger.info("Browser service shut down")

    app = create_browser_app(manager, lifespan=lifespan)
    uvicorn.run(app, host="0.0.0.0", port=_API_PORT)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("Browser service failed to start: %s", e)
        sys.exit(1)
