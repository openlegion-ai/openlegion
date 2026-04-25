"""FastAPI server for the browser service.

Exposes per-agent browser control endpoints. Auth via Bearer token
(same pattern as agent→mesh auth).
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import os
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

from src.browser.service import BrowserManager
from src.shared.trace import TRACE_HEADER, current_trace_id
from src.shared.utils import setup_logging

logger = setup_logging("browser.server")


def create_browser_app(manager: BrowserManager, lifespan=None) -> FastAPI:
    """Create the browser service FastAPI application."""
    kwargs = {"title": "OpenLegion Browser Service"}
    if lifespan:
        kwargs["lifespan"] = lifespan
    app = FastAPI(**kwargs)

    # §2.5 trace propagation: read X-Trace-Id on every request and bind it
    # to the ContextVar so log records under this request carry the trace id.
    # Reset on exit so the thread-local state never leaks to a different
    # request sharing the same worker. Using FastAPI middleware (not a
    # dependency) so it also covers non-endpoint paths (errors, static).
    @app.middleware("http")
    async def _trace_id_propagation(request: Request, call_next):
        incoming = request.headers.get(TRACE_HEADER)
        token = current_trace_id.set(incoming) if incoming else None
        try:
            return await call_next(request)
        finally:
            if token is not None:
                current_trace_id.reset(token)

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

    @app.get("/browser/metrics")
    async def service_metrics(request: Request, since: int = 0):
        """Return per-agent metric payloads buffered since ``since``.

        Polled by the mesh host every ~60s to forward aggregate browser
        metrics into the dashboard EventBus (§5.1/§5.2). The mesh tracks
        the high-water ``current_seq`` across calls and only replays new
        payloads; ``boot_id`` lets it detect a service restart and reset.
        """
        _verify_auth(request)
        try:
            since_seq = max(0, int(since))
        except (TypeError, ValueError):
            since_seq = 0
        return manager.get_recent_metrics(since_seq=since_seq)

    @app.post("/browser/_canary")
    async def run_canary_endpoint(request: Request):
        """Phase 2 §5.4: run the stealth canary on demand.

        Gated by ``BROWSER_CANARY_ENABLED`` — returns 403 when the flag
        is off. Rate-limited to 1 run / ~24h across the service; passing
        ``{"force": true}`` bypasses the rate limit for operator-triggered
        debug runs.

        Output: a structured report with per-scanner status, saved
        screenshots, and best-effort numeric score when parseable.
        Never raises on individual scanner failure — the remaining
        sites still run.
        """
        _verify_auth(request)
        from src.browser.canary import (
            CanaryDisabledError,
            CanaryRateLimitedError,
            run_canary,
        )
        body: dict = {}
        with contextlib.suppress(Exception):
            body = await request.json()
        try:
            return await run_canary(manager, force=bool(body.get("force")))
        except CanaryDisabledError as e:
            raise HTTPException(403, str(e))
        except CanaryRateLimitedError as e:
            raise HTTPException(
                429,
                detail={"error": str(e), "retry_after_s": e.retry_after_s},
            )

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
        # §6.5 referer: forward the param ONLY when the caller included
        # it. ``"referer" not in body`` ⇒ leave the kwarg default (None)
        # so the service-side picker runs. Explicit empty-string ⇒
        # caller wants direct nav. Any other string ⇒ caller override.
        nav_kwargs: dict = {"snapshot_after": snapshot_after}
        if "referer" in body:
            nav_kwargs["referer"] = body["referer"]
        result = await manager.navigate(
            agent_id, url, wait_ms, wait_until, **nav_kwargs,
        )
        await _apply_delay()
        return result

    @app.post("/browser/{agent_id}/snapshot")
    async def snapshot(agent_id: str, request: Request):
        _verify_auth(request)
        try:
            body = await request.json()
        except Exception:
            body = {}
        return await manager.snapshot(
            agent_id,
            filter=(body or {}).get("filter"),
            from_ref=(body or {}).get("from_ref"),
            diff_from_last=bool((body or {}).get("diff_from_last", False)),
        )

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
        return await manager.screenshot(
            agent_id,
            full_page=body.get("full_page", False),
            format=body.get("format", "webp"),
            quality=body.get("quality", 75),
            scale=body.get("scale", 1.0),
        )

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

    # ── File-transfer endpoints (Phase 1.5) ──────────────────────────────
    #
    # These accept already-staged local paths (for uploads) or produce a
    # local path (for downloads). The mesh-side proxy orchestrates streaming
    # bytes between agent and browser containers — neither end-user agents
    # nor the browser service itself handles that transfer directly.

    @app.post("/browser/{agent_id}/upload_file")
    async def upload_file(agent_id: str, request: Request):
        """Drive a file-chooser with files already staged by the mesh.

        Body: ``{"ref": "e7", "paths": ["/tmp/upload-stage/nonce-1"]}``.
        ``paths`` must be readable inside the browser container — the mesh
        places them there via the staging volume / streaming endpoint.
        """
        _verify_auth(request)
        body = await request.json()
        ref = body.get("ref", "")
        paths = body.get("paths") or []
        if not ref:
            raise HTTPException(400, "ref required")
        if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
            raise HTTPException(400, "paths must be a list of strings")
        if not paths:
            raise HTTPException(400, "paths must not be empty")
        if len(paths) > 5:
            raise HTTPException(400, "at most 5 files per upload")
        result = await manager.upload_file(agent_id, ref, paths)
        await _apply_delay()
        return result

    @app.post("/browser/{agent_id}/_stage_upload")
    async def stage_upload(agent_id: str, request: Request):
        """Mesh-internal byte-stream ingest endpoint for §4.5 file uploads.

        Reads the request body as raw bytes, writes them under
        ``OPENLEGION_UPLOAD_RECV_DIR`` (default ``/tmp/upload-recv``) with
        a freshly-generated nonce filename, and returns the on-disk path
        for the mesh to forward into ``/browser/{agent_id}/upload_file``.

        Two gates: (1) ``X-Mesh-Internal: 1`` header AND (2) bearer
        ``Authorization`` token. Both must pass even when
        ``BROWSER_AUTH_INSECURE=1`` is set — unlike public endpoints,
        internal byte-ingest never opens its bearer gate in dev mode.
        """
        if request.headers.get("x-mesh-internal", "") != "1":
            raise HTTPException(403, "Mesh-internal endpoint")
        _verify_auth(request)
        if (
            os.environ.get("BROWSER_AUTH_INSECURE")
            and not request.headers.get("authorization")
        ):
            raise HTTPException(
                403,
                "Internal endpoint requires bearer auth even in dev mode",
            )
        from src.browser.flags import get_int as _flag_int
        from src.browser.flags import get_str as _flag_str
        max_mb = _flag_int(
            "OPENLEGION_UPLOAD_STAGE_MAX_MB", 50, min_value=1, max_value=1024,
        )
        max_bytes = max_mb * 1024 * 1024
        recv_dir = Path(
            _flag_str("OPENLEGION_UPLOAD_RECV_DIR", "/tmp/upload-recv"),
        )
        try:
            recv_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise HTTPException(500, f"Cannot create receive dir: {e}")

        nonce = uuid.uuid4().hex
        suggested = request.query_params.get("suggested_filename", "")
        suffix = ""
        if suggested:
            base = os.path.basename(suggested).strip()
            base = base.replace("\x00", "")
            base = base.replace("/", "").replace("\\", "")
            if base and not base.startswith("."):
                suffix = "-" + base[:80]
        target = recv_dir / f"{nonce}{suffix or '.bin'}"

        size = 0
        try:
            with open(target, "wb") as fh:
                async for chunk in request.stream():
                    if not chunk:
                        continue
                    size += len(chunk)
                    if size > max_bytes:
                        fh.close()
                        with contextlib.suppress(OSError):
                            target.unlink()
                        raise HTTPException(
                            413,
                            f"Upload exceeds {max_mb}MB limit",
                        )
                    fh.write(chunk)
        except HTTPException:
            raise
        except OSError as e:
            with contextlib.suppress(OSError):
                target.unlink()
            raise HTTPException(500, f"Write failed: {e}")
        return {"path": str(target), "size_bytes": size}

    @app.post("/browser/{agent_id}/download")
    async def download_trigger(agent_id: str, request: Request):
        """Click a ref that triggers a download and return a local path.

        Response ``data`` contains ``{path, size_bytes, suggested_filename,
        mime_type}``. The caller (mesh proxy) is expected to stream the
        file from ``path`` to the agent's ``/artifacts/ingest`` endpoint
        and then delete it from the browser container.
        """
        _verify_auth(request)
        body = await request.json()
        ref = body.get("ref", "")
        if not ref:
            raise HTTPException(400, "ref required")
        timeout_ms = int(body.get("timeout_ms", 30000))
        result = await manager.download(agent_id, ref, timeout_ms=timeout_ms)
        # No action delay — downloads can be long-running and the client
        # needs to act on the result immediately to free the tmp file.
        return result

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

    @app.post("/browser/{agent_id}/find_text")
    async def find_text(agent_id: str, request: Request):
        _verify_auth(request)
        body = await request.json()
        query = body.get("query", "")
        if not query:
            raise HTTPException(400, "query required")
        scroll = bool(body.get("scroll", True))
        result = await manager.find_text(agent_id, query, scroll=scroll)
        await _apply_delay()
        return result

    @app.post("/browser/{agent_id}/open_tab")
    async def open_tab(agent_id: str, request: Request):
        _verify_auth(request)
        body = await request.json()
        url = body.get("url", "")
        if not url:
            raise HTTPException(400, "url required")
        snapshot_after = bool(body.get("snapshot_after", False))
        result = await manager.open_tab(
            agent_id, url, snapshot_after=snapshot_after,
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
