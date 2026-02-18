"""Named webhook endpoints that dispatch to agents.

Each webhook gets a unique URL. External services POST payloads to it,
and the mesh routes the content to the configured agent as a chat message.

State persisted to config/webhooks.json.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Optional

from fastapi import APIRouter, HTTPException, Request

from src.shared.utils import generate_id, setup_logging

logger = setup_logging("host.webhooks")


class WebhookManager:
    """Manages named webhook endpoints that trigger agent actions."""

    def __init__(
        self,
        config_path: str = "config/webhooks.json",
        dispatch_fn: Optional[Callable] = None,
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

    def add_hook(self, agent: str, name: str) -> dict:
        hook_id = generate_id("hook")[:16]
        hook = {
            "id": hook_id,
            "agent": agent,
            "name": name,
            "created_at": datetime.now(UTC).isoformat(),
            "call_count": 0,
        }
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
        return list(self.hooks.values())

    def create_router(self) -> APIRouter:
        """Create a FastAPI router for webhook endpoints."""
        router = APIRouter(prefix="/webhook")
        manager = self

        @router.post("/hook/{hook_id}")
        async def receive_webhook(hook_id: str, request: Request) -> dict:
            hook = manager.hooks.get(hook_id)
            if not hook:
                raise HTTPException(status_code=404, detail="Unknown webhook")

            try:
                body = await request.json()
            except Exception:
                body = {"raw": (await request.body()).decode(errors="replace")[:5000]}

            hook["call_count"] = hook.get("call_count", 0) + 1
            manager._save()

            message = (
                f"Webhook '{hook['name']}' received:\n"
                f"```json\n{json.dumps(body, indent=2, default=str)[:3000]}\n```\n"
                f"Process this webhook payload."
            )

            response = None
            if manager.dispatch_fn:
                response = await manager.dispatch_fn(hook["agent"], message)

            return {"status": "processed", "hook": hook["name"], "response": response}

        return router

    async def test_hook(self, hook_id: str, payload: dict) -> dict | None:
        """Manually fire a webhook for testing."""
        hook = self.hooks.get(hook_id)
        if not hook:
            return None

        message = (
            f"Webhook '{hook['name']}' (test):\n"
            f"```json\n{json.dumps(payload, indent=2, default=str)[:3000]}\n```\n"
            f"Process this webhook payload."
        )

        if self.dispatch_fn:
            response = await self.dispatch_fn(hook["agent"], message)
            return {"status": "processed", "response": response}
        return {"status": "no_dispatch_fn"}
