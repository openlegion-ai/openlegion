"""Browser service entry point.

Starts KasmVNC (Xvnc + web client), Openbox WM, and the FastAPI
browser command server.

Uses Xvnc directly (not the vncserver wrapper) for full control over
flags. Xvnc serves as combined X server + VNC server + web client host.
No separate Xvfb is needed.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from src.browser.server import create_browser_app
from src.browser.service import BrowserManager
from src.shared.utils import setup_logging

logger = setup_logging("browser.main")

_DISPLAY = ":99"
_VNC_PORT = int(os.environ.get("VNC_PORT", "6080"))
_API_PORT = int(os.environ.get("API_PORT", "8500"))
_MAX_BROWSERS = int(os.environ.get("MAX_BROWSERS", "5"))
_IDLE_TIMEOUT = int(os.environ.get("IDLE_TIMEOUT_MINUTES", "30"))


def _start_kasmvnc() -> subprocess.Popen:
    """Start KasmVNC Xvnc — combined X server + VNC server + web server.

    Uses Xvnc directly with:
    - ``-sslOnly 0``: disable TLS (access control via Docker port mapping)
    - ``-SecurityTypes None``: no VNC authentication
    - ``-disableBasicAuth``: no HTTP basic auth on the web client
    - ``-httpd``: serve the KasmVNC web client files
    - ``-AlwaysShared``: allow multiple VNC viewers
    """
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
    logger.info("KasmVNC Xvnc started on display %s, web on :%d", _DISPLAY, _VNC_PORT)
    return proc


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
    """Start all services and run the FastAPI server."""
    kasmvnc_proc = _start_kasmvnc()
    openbox_proc = _start_openbox()

    manager = BrowserManager(
        profiles_dir="/data/profiles",
        max_concurrent=_MAX_BROWSERS,
        idle_timeout_minutes=_IDLE_TIMEOUT,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await manager.start_cleanup_loop()
        logger.info("Browser service ready (max=%d, idle_timeout=%dm)", _MAX_BROWSERS, _IDLE_TIMEOUT)
        yield
        await manager.stop_all()
        for proc in (openbox_proc, kasmvnc_proc):
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
        logger.info("Browser service shut down")

    app = create_browser_app(manager, lifespan=lifespan)
    uvicorn.run(app, host="0.0.0.0", port=_API_PORT)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("Browser service failed to start: %s", e)
        sys.exit(1)
