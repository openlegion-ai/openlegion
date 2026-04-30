"""Browser service entry point.

Starts KasmVNC (Xvnc + web client), Openbox WM, and the FastAPI
browser command server.

Uses Xvnc directly (not the vncserver wrapper) for full control over
flags. Xvnc serves as combined X server + VNC server + web client host.
No separate Xvfb is needed.
"""

from __future__ import annotations

import os
import secrets
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from src.browser.server import create_browser_app
from src.browser.service import BrowserManager
from src.shared.utils import setup_logging

logger = setup_logging("browser.main")

_DISPLAY = ":99"
_VNC_PORT = int(os.environ.get("VNC_PORT", "6080"))
_API_PORT = int(os.environ.get("API_PORT", "8500"))


# Memory budget per Camoufox + per-agent X stack, in MiB.  ~400 MB is
# the high-water mark of a typical Camoufox instance under sustained
# use; ~50 MB covers the per-agent Xvnc + Openbox + unclutter when
# OPENLEGION_BROWSER_PER_AGENT_DISPLAY is on.  Sized for the safer of
# the two paths so the autodetected default is correct in both modes.
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

    Used only when neither ``OPENLEGION_BROWSER_MAX_CONCURRENT`` nor
    ``MAX_BROWSERS`` is set — i.e. self-host installs that haven't
    tuned anything.  Hetzner-provisioned VPSes get the cap from the
    provisioner, which always wins over the autodetect (the env-var
    layer is checked first in :func:`_resolve_max_browsers`).

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
      1. ``OPENLEGION_BROWSER_MAX_CONCURRENT`` — canonical name, set by
         the provisioner per VPS plan and overridable by self-hosters.
         Listed in :data:`src.browser.flags.KNOWN_FLAGS`.
      2. ``MAX_BROWSERS`` — legacy name kept for back-compat with existing
         deployments / Docker compose files. Removable after one release.
      3. Memory-derived autodetect (:func:`_autodetect_default_max_browsers`).
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
    legacy_default = get_int(
        "MAX_BROWSERS", autodetected, min_value=1, max_value=_MAX_CAP,
    )
    final = get_int(
        "OPENLEGION_BROWSER_MAX_CONCURRENT",
        legacy_default,
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


def _start_kasmvnc() -> subprocess.Popen:
    """Start KasmVNC Xvnc — combined X server + VNC server + web server.

    KasmVNC 1.4.0 requires -disableBasicAuth so the web client in the
    dashboard iframe can connect without an auth prompt. The WebSocket
    endpoint is /websockify and requires an Origin header but no credentials
    when Basic Auth is disabled.

    The browser container is only reachable from the host via Docker port
    mapping, so disabling auth is safe.
    """
    # .kasmpasswd still needed for KasmVNC internals even with -disableBasicAuth
    vnc_password = secrets.token_urlsafe(16)
    kasmpasswd = os.path.expanduser("~/.kasmpasswd")
    subprocess.run(
        ["kasmvncpasswd", "-u", "browser", "-ow", kasmpasswd],
        input=f"{vnc_password}\n{vnc_password}\n",
        text=True,
        capture_output=True,
    )
    logger.debug("Created .kasmpasswd for KasmVNC")

    cmd = [
        "Xvnc", _DISPLAY,
        "-geometry", "1920x1080",
        "-depth", "24",
        "-websocketPort", str(_VNC_PORT),
        "-httpd", "/usr/share/kasmvnc/www",
        "-sslOnly", "0",
        "-SecurityTypes", "None",
        "-disableBasicAuth",
        "-AlwaysShared",
        "-interface", "0.0.0.0",
        # Allow iframe embedding from dashboard (different port = different origin)
        "-http-header", "X-Frame-Options=ALLOWALL",
        "-http-header", "Access-Control-Allow-Origin=*",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.5)
    if proc.poll() is not None:
        raise RuntimeError(
            f"Xvnc exited immediately (code {proc.returncode})"
        )
    os.environ["DISPLAY"] = _DISPLAY
    # Set X root window to a dark color matching Firefox's dark theme so the
    # toolbar area doesn't show harsh black while the browser is loading.
    try:
        subprocess.run(
            ["xsetroot", "-solid", "#1e1e2e"],
            env={**os.environ, "DISPLAY": _DISPLAY},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        logger.debug("xsetroot not found, skipping root window color")
    logger.info("KasmVNC Xvnc started on display %s, web on :%d", _DISPLAY, _VNC_PORT)
    return proc


def _start_unclutter() -> subprocess.Popen | None:
    """Hide the X11 system cursor via XFixes.

    KasmVNC streams both the X11 cursor and the browser's rendered content.
    Camoufox's ``humanize=True`` draws its own visual cursor (red dot)
    inside the browser page.  Without hiding the X11 cursor, VNC viewers
    see two cursors — the red dot that moves with automation and a stale
    system cursor that only moves on xdotool calls.  Hiding the X11
    cursor leaves only the Camoufox cursor visible.
    """
    try:
        proc = subprocess.Popen(
            ["unclutter", "--timeout", "0"],
            env={**os.environ, "DISPLAY": _DISPLAY},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info("unclutter started — X11 cursor hidden (pid=%d)", proc.pid)
        return proc
    except FileNotFoundError:
        logger.warning("unclutter not found — X11 cursor will remain visible")
        return None


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


def _start_openbox() -> subprocess.Popen:
    """Start Openbox window manager on the KasmVNC display."""
    proc = subprocess.Popen(
        ["openbox"],
        env={**os.environ, "DISPLAY": _DISPLAY},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(0.3)
    logger.info("Openbox started (pid=%d)", proc.pid)
    return proc


def main() -> None:
    """Start all services and run the FastAPI server.

    When ``OPENLEGION_BROWSER_PER_AGENT_DISPLAY`` is on, the global
    ``Xvnc :99`` / Openbox / unclutter are NOT started — each agent
    gets its own X stack inside :class:`BrowserManager._start_browser`
    instead.  The legacy path remains the default during the flag-
    gated rollout (PR 1); PR 2 flips the default and PR 3 deletes the
    legacy code path entirely.
    """
    from src.browser.display_allocator import is_per_agent_display_enabled

    per_agent_default = is_per_agent_display_enabled()
    if per_agent_default:
        logger.info(
            "Per-agent display mode active — skipping global Xvnc/Openbox/"
            "unclutter; each agent's browser will get its own X stack",
        )
        kasmvnc_proc = None
        unclutter_proc = None
        openbox_proc = None
    else:
        kasmvnc_proc = _start_kasmvnc()
        unclutter_proc = _start_unclutter()
        openbox_proc = _start_openbox()

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
        await manager.stop_all()
        # Tear down the global X stack only when it was started in
        # legacy mode.  Per-agent mode has no global processes to reap;
        # individual agents' X stacks were already torn down by
        # ``manager.stop_all()`` via ``_teardown_per_agent_x_stack``.
        procs = [p for p in (openbox_proc, kasmvnc_proc, unclutter_proc) if p]
        for proc in procs:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
                proc.wait()
        logger.info("Browser service shut down")

    app = create_browser_app(manager, lifespan=lifespan)
    uvicorn.run(app, host="0.0.0.0", port=_API_PORT)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("Browser service failed to start: %s", e)
        sys.exit(1)
