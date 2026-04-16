"""FastAPI server for the browser service.

Exposes per-agent browser control endpoints. Auth via Bearer token
(same pattern as agent→mesh auth).
"""

from __future__ import annotations

import asyncio
import hmac
import json
import os

from fastapi import FastAPI, HTTPException, Request

from src.browser.service import BrowserManager
from src.shared.utils import setup_logging

logger = setup_logging("browser.server")


def create_browser_app(manager: BrowserManager, lifespan=None) -> FastAPI:
    """Create the browser service FastAPI application."""
    kwargs = {"title": "OpenLegion Browser Service"}
    if lifespan:
        kwargs["lifespan"] = lifespan
    app = FastAPI(**kwargs)
    auth_token = os.environ.get("BROWSER_AUTH_TOKEN", "")
    if not auth_token:
        if os.environ.get("MESH_AUTH_TOKEN"):
            raise RuntimeError(
                "BROWSER_AUTH_TOKEN not set but MESH_AUTH_TOKEN is configured. "
                "Browser auth cannot be disabled in production. "
                "Set BROWSER_AUTH_TOKEN or BROWSER_AUTH_INSECURE=1 to override."
            )
        logger.warning("BROWSER_AUTH_TOKEN not set — browser service auth is DISABLED")

    def _verify_auth(request: Request) -> None:
        if not auth_token:
            return
        header = request.headers.get("Authorization", "")
        if not header.startswith("Bearer ") or not hmac.compare_digest(header[7:], auth_token):
            raise HTTPException(401, "Unauthorized")

    async def _apply_delay():
        """Sleep for the configured inter-action delay after a browser action."""
        from src.browser.timing import inter_action_delay
        d = inter_action_delay()
        if d > 0:
            await asyncio.sleep(d)

    @app.get("/browser/status")
    async def service_status(request: Request):
        _verify_auth(request)
        return await manager.get_service_status()

    @app.post("/browser/keepalive")
    async def keepalive(request: Request):
        """Touch all running browser instances to reset their idle timers.

        Called by the VNC WebSocket proxy while a viewer is connected, so
        a browser stays alive as long as someone is watching the display.
        """
        _verify_auth(request)
        touched = await manager.touch_all()
        await manager.refocus_active()
        return {"touched": touched}

    @app.get("/browser/{agent_id}/status")
    async def agent_status(agent_id: str, request: Request):
        _verify_auth(request)
        return await manager.get_status(agent_id)

    @app.post("/browser/{agent_id}/navigate")
    async def navigate(agent_id: str, request: Request):
        _verify_auth(request)
        body = await request.json()
        url = body.get("url", "")
        if not url:
            raise HTTPException(400, "url required")
        wait_ms = body.get("wait_ms", 1000)
        wait_until = body.get("wait_until", "domcontentloaded")
        snapshot_after = body.get("snapshot_after", False)
        result = await manager.navigate(agent_id, url, wait_ms, wait_until,
                                        snapshot_after=snapshot_after)
        await _apply_delay()
        return result

    @app.post("/browser/{agent_id}/snapshot")
    async def snapshot(agent_id: str, request: Request):
        _verify_auth(request)
        return await manager.snapshot(agent_id)

    @app.post("/browser/{agent_id}/click")
    async def click(agent_id: str, request: Request):
        _verify_auth(request)
        body = await request.json()
        result = await manager.click(
            agent_id,
            ref=body.get("ref"),
            selector=body.get("selector"),
            force=body.get("force", False),
            snapshot_after=body.get("snapshot_after", False),
            timeout_ms=body.get("timeout_ms"),
        )
        await _apply_delay()
        return result

    @app.post("/browser/{agent_id}/wait_for")
    async def wait_for(agent_id: str, request: Request):
        _verify_auth(request)
        body = await request.json()
        selector = body.get("selector", "")
        if not selector:
            raise HTTPException(400, "selector required")
        return await manager.wait_for_element(
            agent_id,
            selector=selector,
            state=body.get("state", "visible"),
            timeout_ms=body.get("timeout_ms", 10000),
        )

    @app.post("/browser/{agent_id}/hover")
    async def hover(agent_id: str, request: Request):
        _verify_auth(request)
        body = await request.json()
        result = await manager.hover(
            agent_id,
            ref=body.get("ref"),
            selector=body.get("selector"),
        )
        await _apply_delay()
        return result

    @app.post("/browser/{agent_id}/type")
    async def type_text(agent_id: str, request: Request):
        _verify_auth(request)
        body = await request.json()
        result = await manager.type_text(
            agent_id,
            ref=body.get("ref"),
            selector=body.get("selector"),
            text=body.get("text", ""),
            clear=body.get("clear", True),
            fast=body.get("fast", False),
            snapshot_after=body.get("snapshot_after", False),
        )
        await _apply_delay()
        return result

    # /browser/{agent_id}/evaluate endpoint intentionally removed —
    # arbitrary JS execution is an SSRF/sandbox-escape vector.
    # The evaluate() method remains on BrowserManager for internal use only.

    @app.post("/browser/{agent_id}/screenshot")
    async def screenshot(agent_id: str, request: Request):
        _verify_auth(request)
        body = await request.json()
        return await manager.screenshot(agent_id, full_page=body.get("full_page", False))

    @app.post("/browser/{agent_id}/reset")
    async def reset(agent_id: str, request: Request):
        _verify_auth(request)
        await manager.reset(agent_id)
        return {"success": True, "data": {"message": f"Browser reset for {agent_id}"}}

    @app.put("/browser/{agent_id}/proxy")
    async def set_agent_proxy(agent_id: str, request: Request):
        _verify_auth(request)
        body_bytes = await request.body()
        body = None
        if body_bytes:
            body = json.loads(body_bytes)
        was_focused = manager._user_focused_agent == agent_id
        manager.set_proxy_config(agent_id, body)
        # If a browser is already running for this agent, restart it so the
        # new proxy config takes effect on the next get_or_start() call.
        # reset() is a no-op when no instance exists (stop() checks under lock).
        try:
            await manager.reset(agent_id)
        except Exception:
            logger.warning("Failed to reset browser for '%s' after proxy change", agent_id, exc_info=True)
        # Re-focus the agent if it was focused before the reset, so the VNC
        # viewer doesn't show a purple/blank screen.
        if was_focused:
            try:
                # Re-check: another request may have shifted focus during reset.
                # stop() clears _user_focused_agent to None, so if it's still
                # None nobody else claimed focus. If it's our agent, also safe.
                if manager._user_focused_agent is None or manager._user_focused_agent == agent_id:
                    await manager.focus(agent_id)
            except Exception:
                logger.debug("Failed to re-focus '%s' after proxy reset", agent_id)
        return {"success": True}

    @app.post("/browser/{agent_id}/focus")
    async def focus(agent_id: str, request: Request):
        _verify_auth(request)
        ok = await manager.focus(agent_id)
        return {"success": ok, "data": {"focused": ok}}

    @app.post("/browser/{agent_id}/control")
    async def browser_control(agent_id: str, request: Request):
        """Toggle user browser control — pauses agent X11 input."""
        _verify_auth(request)
        body = await request.json()
        enabled = bool(body.get("user_control", False))
        return await manager.set_user_control(agent_id, enabled)

    @app.post("/browser/{agent_id}/scroll")
    async def scroll(agent_id: str, request: Request):
        _verify_auth(request)
        body = await request.json()
        result = await manager.scroll(
            agent_id,
            direction=body.get("direction", "down"),
            amount=body.get("amount", 0),
            ref=body.get("ref"),
        )
        await _apply_delay()
        return result

    @app.post("/browser/{agent_id}/detect_captcha")
    async def detect_captcha(agent_id: str, request: Request):
        _verify_auth(request)
        return await manager.detect_captcha(agent_id)

    @app.post("/browser/{agent_id}/press_key")
    async def press_key(agent_id: str, request: Request):
        _verify_auth(request)
        body = await request.json()
        key = body.get("key", "")
        if not key:
            raise HTTPException(400, "key required")
        result = await manager.press_key(agent_id, key)
        await _apply_delay()
        return result

    @app.post("/browser/{agent_id}/go_back")
    async def go_back(agent_id: str, request: Request):
        _verify_auth(request)
        result = await manager.go_back(agent_id)
        await _apply_delay()
        return result

    @app.post("/browser/{agent_id}/go_forward")
    async def go_forward(agent_id: str, request: Request):
        _verify_auth(request)
        result = await manager.go_forward(agent_id)
        await _apply_delay()
        return result

    @app.post("/browser/{agent_id}/switch_tab")
    async def switch_tab(agent_id: str, request: Request):
        _verify_auth(request)
        body = await request.json()
        result = await manager.switch_tab(
            agent_id, tab_index=body.get("tab_index", -1),
        )
        await _apply_delay()
        return result

    # ── Settings ───────────────────────────────────────────────────────────

    @app.get("/browser/settings")
    async def get_settings(request: Request):
        """Return current browser speed and delay settings."""
        _verify_auth(request)
        from src.browser.timing import get_delay, get_speed
        return {"speed": get_speed(), "delay": get_delay()}

    @app.post("/browser/settings")
    async def update_settings(request: Request):
        """Update browser speed and delay settings at runtime."""
        _verify_auth(request)
        body = await request.json()
        speed = body.get("speed")
        if speed is not None:
            from src.browser.timing import set_speed
            set_speed(float(speed))
        delay = body.get("delay")
        if delay is not None:
            from src.browser.timing import set_delay
            set_delay(float(delay))
        from src.browser.timing import get_delay, get_speed
        return {"speed": get_speed(), "delay": get_delay()}

    # ── User uploads file serving ─────────────────────────────────────────
    # Serves files from /app/uploads (user-managed, read-only mount).
    # No auth required: this port is not internet-exposed and all content
    # was placed here by the authenticated dashboard user.
    # Agents navigate the browser to http://localhost:8500/uploads/<filename>.

    _UPLOADS_ROOT = "/app/uploads"

    @app.get("/uploads/{path:path}")
    async def serve_upload(path: str):
        """Serve a user-uploaded file so the VNC browser can navigate to it."""
        import mimetypes
        from pathlib import Path

        from fastapi.responses import Response

        root = Path(_UPLOADS_ROOT)
        if not root.is_dir():
            raise HTTPException(503, "Uploads directory not available")
        root = root.resolve()
        try:
            p = Path(path)
        except ValueError:
            raise HTTPException(400, "Invalid path")
        if p.is_absolute() or ".." in p.parts:
            raise HTTPException(400, "Invalid path")
        candidate = (root / path).resolve()
        if not candidate.is_relative_to(root):
            raise HTTPException(400, "Path traversal not allowed")
        if not candidate.exists() or not candidate.is_file():
            raise HTTPException(404, f"Upload not found: {path}")
        mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
        with candidate.open("rb") as fh:
            content = fh.read()
        return Response(content=content, media_type=mime)

    return app
