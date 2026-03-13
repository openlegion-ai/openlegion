"""Generic webhook channel adapter.

Converts HTTP POST requests into workflow triggers
and provides status polling endpoints.
"""

from __future__ import annotations

import hmac
import os
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request

if TYPE_CHECKING:
    from src.host.orchestrator import Orchestrator

def _verify_webhook_auth(request: Request) -> None:
    """Check Bearer token against WEBHOOK_SECRET. Fail closed if unconfigured."""
    secret = os.environ.get("WEBHOOK_SECRET", "")
    if not secret:
        raise HTTPException(status_code=503, detail="Webhook not configured")
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer ") or not hmac.compare_digest(auth[7:], secret):
        raise HTTPException(status_code=401, detail="Invalid or missing webhook secret")


def create_webhook_router(orchestrator: Orchestrator) -> APIRouter:
    """Create the webhook FastAPI router."""
    router = APIRouter(prefix="/webhook")

    @router.post("/trigger/{workflow_name}")
    async def trigger_workflow(workflow_name: str, payload: dict, request: Request) -> dict:
        """Trigger a workflow via HTTP webhook."""
        _verify_webhook_auth(request)
        try:
            execution_id = await orchestrator.trigger_workflow(workflow_name, payload)
            return {"execution_id": execution_id, "status": "started"}
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

    @router.get("/status/{execution_id}")
    async def get_status(execution_id: str, request: Request) -> dict:
        """Check workflow execution status."""
        _verify_webhook_auth(request)
        status = orchestrator.get_execution_status(execution_id)
        if not status:
            raise HTTPException(status_code=404, detail="Execution not found")
        return status

    return router
