"""Named webhook endpoints that dispatch to agents.

Each webhook gets a unique URL. External services POST payloads to it,
and the mesh routes the content to the configured agent as a chat message.

State persisted to config/webhooks.json.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from fastapi import APIRouter, HTTPException, Request

from src.shared.utils import generate_id, sanitize_for_prompt, setup_logging

logger = setup_logging("host.webhooks")


def _build_message(hook: dict, body_json: str, *, test: bool = False) -> str:
    """Build the dispatch message from hook config and payload."""
    label = f"Webhook '{hook['name']}'" + (" (test)" if test else "") + " received:"
    instructions = hook.get("instructions", "").strip()
    suffix = instructions if instructions else "Process this webhook payload."
    message = f"{label}\n```json\n{body_json[:3000]}\n```\n{suffix}"
    return sanitize_for_prompt(message)


class WebhookManager:
    """Manages named webhook endpoints that trigger agent actions."""

    def __init__(
        self,
        config_path: str = "config/webhooks.json",
        dispatch_fn: Callable | None = None,
    ):
        self.config_path = Path(config_path)
        self.hooks: dict[str, dict] = {}
        self.dispatch_fn = dispatch_fn
        self._load()

    def _load(self) -> None:
        if not self.config_path.exists():
            return
        try:
            self.hooks = json.loads(self.config_path.read_text()).get("hooks", {})
        except Exception as e:
            logger.warning(f"Failed to load webhook config: {e}")

    def _save(self) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps({"hooks": self.hooks}, indent=2) + "\n")

    def add_hook(
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
        logger.info(f"Added webhook {hook_id}: agent={agent} name={name}")
        return hook

    def remove_hook(self, hook_id: str) -> bool:
        if hook_id not in self.hooks:
            return False
        del self.hooks[hook_id]
        self._save()
        return True

    def list_hooks(self) -> list[dict]:
        return [dict(h) for h in self.hooks.values()]

    def create_router(self) -> APIRouter:
        """Create a FastAPI router for webhook endpoints."""
        router = APIRouter(prefix="/webhook")
        manager = self

        _MAX_WEBHOOK_BODY = 1_048_576  # 1 MB

        @router.post("/hook/{hook_id}")
        async def receive_webhook(hook_id: str, request: Request) -> dict:
            hook = manager.hooks.get(hook_id)
            if not hook:
                raise HTTPException(status_code=404, detail="Unknown webhook")

            content_length = request.headers.get("content-length")
            if content_length:
                try:
                    if int(content_length) > _MAX_WEBHOOK_BODY:
                        raise HTTPException(status_code=413, detail="Payload too large")
                except ValueError:
                    pass  # Malformed header; body-length check below is authoritative
            raw_body = await request.body()
            if len(raw_body) > _MAX_WEBHOOK_BODY:
                raise HTTPException(status_code=413, detail="Payload too large")

            # HMAC-SHA256 signature verification (when hook has a secret)
            hook_secret = hook.get("secret")
            if hook_secret:
                sig_header = request.headers.get("x-webhook-signature", "")
                expected = hmac.new(
                    hook_secret.encode(), raw_body, hashlib.sha256,
                ).hexdigest()
                if not hmac.compare_digest(sig_header, expected):
                    raise HTTPException(status_code=401, detail="Invalid signature")

            try:
                body = json.loads(raw_body)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.debug("Webhook body is not valid JSON, using raw: %s", e)
                body = {"raw": raw_body.decode(errors="replace")[:5000]}

            hook["call_count"] = hook.get("call_count", 0) + 1
            manager._save()

            message = _build_message(hook, json.dumps(body, indent=2, default=str))

            if manager.dispatch_fn:
                await manager.dispatch_fn(hook["agent"], message)

            return {"status": "processed", "hook": hook["name"]}

        return router

    async def test_hook(self, hook_id: str, payload: dict) -> dict | None:
        """Manually fire a webhook for testing."""
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
