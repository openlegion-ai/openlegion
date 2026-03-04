"""FastAPI server for the browser service.

Exposes per-agent browser control endpoints. Auth via Bearer token
(same pattern as agent→mesh auth).
"""

from __future__ import annotations

import hmac
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

    def _verify_auth(request: Request) -> None:
        if not auth_token:
            return
        header = request.headers.get("Authorization", "")
        if not header.startswith("Bearer ") or not hmac.compare_digest(header[7:], auth_token):
            raise HTTPException(401, "Unauthorized")

    @app.get("/browser/status")
    async def service_status(request: Request):
        _verify_auth(request)
        return await manager.get_service_status()

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
        return await manager.navigate(agent_id, url, wait_ms)

    @app.post("/browser/{agent_id}/snapshot")
    async def snapshot(agent_id: str, request: Request):
        _verify_auth(request)
        return await manager.snapshot(agent_id)

    @app.post("/browser/{agent_id}/click")
    async def click(agent_id: str, request: Request):
        _verify_auth(request)
        body = await request.json()
        return await manager.click(
            agent_id,
            ref=body.get("ref"),
            selector=body.get("selector"),
        )

    @app.post("/browser/{agent_id}/type")
    async def type_text(agent_id: str, request: Request):
        _verify_auth(request)
        body = await request.json()
        return await manager.type_text(
            agent_id,
            ref=body.get("ref"),
            selector=body.get("selector"),
            text=body.get("text", ""),
            clear=body.get("clear", True),
            is_credential=body.get("is_credential", False),
        )

    @app.post("/browser/{agent_id}/evaluate")
    async def evaluate(agent_id: str, request: Request):
        _verify_auth(request)
        body = await request.json()
        expression = body.get("expression", "")
        if not expression:
            raise HTTPException(400, "expression required")
        return await manager.evaluate(agent_id, expression)

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

    @app.post("/browser/{agent_id}/focus")
    async def focus(agent_id: str, request: Request):
        _verify_auth(request)
        ok = await manager.focus(agent_id)
        return {"success": ok, "data": {"focused": ok}}

    @app.post("/browser/{agent_id}/solve_captcha")
    async def solve_captcha(agent_id: str, request: Request):
        _verify_auth(request)
        return await manager.solve_captcha(agent_id)

    return app
