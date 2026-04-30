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


def _resolve_max_browsers() -> int:
    """Return the per-service browser concurrency cap.

    Startup-only — runtime reconfig is non-trivial (would need to bound an
    asyncio.Semaphore mid-flight, drain over-budget instances, and unwind any
    in-flight ``acquire`` waiters). Operators restart the browser service to
    change this. Plan §10.2 keeps this as the explicit decision after the v3.6
    review flagged the runtime path's complexity.

    Precedence (highest → lowest):
      1. ``OPENLEGION_BROWSER_MAX_CONCURRENT`` — canonical name, listed in
         :data:`src.browser.flags.KNOWN_FLAGS`.
      2. ``MAX_BROWSERS`` — legacy name kept for back-compat with existing
         deployments / Docker compose files. Removable after one release.
      3. Default of 5.
    """
    from src.browser.flags import get_int

    # ``min_value=1`` rejects nonsense like ``MAX=0`` (would deadlock the
    # acquire loop). ``max_value=64`` is a soft ceiling — a single VPS won't
    # have RAM for that many Camoufox instances anyway.
    legacy_default = get_int("MAX_BROWSERS", 5, min_value=1, max_value=64)
    return get_int(
        "OPENLEGION_BROWSER_MAX_CONCURRENT",
        legacy_default,
        min_value=1,
        max_value=64,
    )


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
