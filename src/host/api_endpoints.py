"""Named API endpoints that dispatch to agents.

Each endpoint gets a unique URL. External services POST payloads to it,
and the mesh routes the content to the configured agent as a chat message.

State persisted to config/api_endpoints.json.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from src.shared.utils import generate_id, sanitize_for_prompt, setup_logging

logger = setup_logging("host.api_endpoints")


def _build_message(hook: dict, body_json: str, *, test: bool = False) -> str:
    """Build the dispatch message from hook config and payload."""
    label = f"API endpoint '{hook['name']}'" + (" (test)" if test else "") + " received:"
    instructions = hook.get("instructions", "").strip()
    suffix = instructions if instructions else "Process this webhook payload."
    message = f"{label}\n```json\n{body_json[:3000]}\n```\n{suffix}"
    return sanitize_for_prompt(message)


class ApiEndpointManager:
    """Manages named API endpoints that trigger agent actions."""

    def __init__(
        self,
        config_path: str = "config/api_endpoints.json",
        dispatch_fn: Callable | None = None,
    ):
        self.config_path = Path(config_path)
        self.hooks: dict[str, dict] = {}
        self.dispatch_fn = dispatch_fn
        self._load()

    def _load(self) -> None:
        if not self.config_path.exists():
            # Auto-migrate from old config path
            old_path = self.config_path.parent / "webhooks.json"
            if old_path.exists():
                try:
                    self.hooks = json.loads(old_path.read_text()).get("hooks", {})
                    self._save()
                    logger.info("Migrated config from %s to %s", old_path, self.config_path)
                except Exception as e:
                    logger.warning(f"Failed to migrate webhook config: {e}")
            return
        try:
            self.hooks = json.loads(self.config_path.read_text()).get("hooks", {})
        except Exception as e:
            logger.warning(f"Failed to load API endpoint config: {e}")

    def _save(self) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps({"hooks": self.hooks}, indent=2) + "\n")

    def add_endpoint(
        self,
        agent: str,
        name: str,
        *,
        require_signature: bool = False,
        instructions: str = "",
    ) -> dict:
        hook_id = generate_id("hook")[:16]
        hook: dict = {
            "id": hook_id,
            "agent": agent,
            "name": name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "call_count": 0,
        }
        if require_signature:
            hook["secret"] = secrets.token_hex(32)
        if instructions.strip():
            hook["instructions"] = instructions.strip()
        self.hooks[hook_id] = hook
        self._save()
        logger.info(f"Added API endpoint {hook_id}: agent={agent} name={name}")
        return hook

    def remove_endpoint(self, hook_id: str) -> bool:
        if hook_id not in self.hooks:
            return False
        del self.hooks[hook_id]
        self._save()
        return True

    def update_endpoint(
        self,
        hook_id: str,
        *,
        name: str | None = None,
        agent: str | None = None,
        instructions: str | None = None,
        require_signature: bool | None = None,
        regenerate_secret: bool = False,
    ) -> dict | None:
        """Update an existing endpoint's configuration.

        Only provided fields are changed. Returns updated hook dict with
        secret included only when a new secret was generated, or None if
        hook_id not found.
        """
        hook = self.hooks.get(hook_id)
        if hook is None:
            return None

        # Validate before mutating to avoid partial updates on error
        if name is not None and not name.strip():
            raise ValueError("name must not be empty")
        if agent is not None and not agent.strip():
            raise ValueError("agent must not be empty")

        # Apply mutations (safe — validation already passed)
        if name is not None:
            hook["name"] = name.strip()
        if agent is not None:
            hook["agent"] = agent.strip()
        if instructions is not None:
            if instructions.strip():
                hook["instructions"] = instructions.strip()
            else:
                hook.pop("instructions", None)

        new_secret = False
        if require_signature is True and "secret" not in hook:
            hook["secret"] = secrets.token_hex(32)
            new_secret = True
        elif require_signature is False:
            hook.pop("secret", None)

        if regenerate_secret and "secret" in hook and not new_secret:
            hook["secret"] = secrets.token_hex(32)
            new_secret = True

        self._save()
        logger.info("Updated API endpoint %s", hook_id)

        result = dict(hook)
        if not new_secret:
            result.pop("secret", None)
        return result

    def list_endpoints(self) -> list[dict]:
        return [dict(h) for h in self.hooks.values()]

    def create_router(self) -> APIRouter:
        """Create a FastAPI router for API inbound endpoints."""
        router = APIRouter()
        manager = self

        _MAX_BODY = 1_048_576  # 1 MB

        async def _handle_inbound(endpoint_id: str, request: Request) -> dict:
            hook = manager.hooks.get(endpoint_id)
            if not hook:
                raise HTTPException(status_code=404, detail="Unknown endpoint")

            content_length = request.headers.get("content-length")
            if content_length:
                try:
                    if int(content_length) > _MAX_BODY:
                        raise HTTPException(status_code=413, detail="Payload too large")
                except ValueError:
                    pass  # Malformed header; body-length check below is authoritative
            raw_body = await request.body()
            if len(raw_body) > _MAX_BODY:
                raise HTTPException(status_code=413, detail="Payload too large")

            # HMAC-SHA256 signature verification (when hook has a secret)
            hook_secret = hook.get("secret")
            if hook_secret:
                # Accept x-signature first, fall back to x-webhook-signature
                sig_header = request.headers.get("x-signature") or request.headers.get("x-webhook-signature", "")
                expected = hmac.new(
                    hook_secret.encode(), raw_body, hashlib.sha256,
                ).hexdigest()
                if not hmac.compare_digest(sig_header, expected):
                    raise HTTPException(status_code=401, detail="Invalid signature")

            try:
                body = json.loads(raw_body)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.debug("API endpoint body is not valid JSON, using raw: %s", e)
                body = {"raw": raw_body.decode(errors="replace")[:5000]}

            hook["call_count"] = hook.get("call_count", 0) + 1
            manager._save()

            message = _build_message(hook, json.dumps(body, indent=2, default=str))

            if manager.dispatch_fn:
                await manager.dispatch_fn(hook["agent"], message)

            return {"status": "processed", "hook": hook["name"]}

        @router.post("/api/inbound/{endpoint_id}")
        async def receive_inbound(endpoint_id: str, request: Request) -> dict:
            return await _handle_inbound(endpoint_id, request)

        @router.post("/webhook/hook/{endpoint_id}")
        async def receive_webhook_compat(endpoint_id: str, request: Request) -> JSONResponse:
            """Deprecated: use /api/inbound/{endpoint_id} instead."""
            result = await _handle_inbound(endpoint_id, request)
            return JSONResponse(content=result, headers={"Deprecation": "true"})

        return router

    async def test_endpoint(self, hook_id: str, payload: dict) -> dict | None:
        """Manually fire an endpoint for testing."""
        hook = self.hooks.get(hook_id)
        if not hook:
            return None

        message = _build_message(
            hook, json.dumps(payload, indent=2, default=str), test=True,
        )

        if self.dispatch_fn:
            response = await self.dispatch_fn(hook["agent"], message)
            return {"status": "processed", "response": response}
        return {"status": "no_dispatch_fn"}
