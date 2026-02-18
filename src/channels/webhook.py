"""Generic webhook channel adapter.

Converts HTTP POST requests into workflow triggers
and provides status polling endpoints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException

if TYPE_CHECKING:
    from src.host.orchestrator import Orchestrator


def create_webhook_router(orchestrator: Orchestrator) -> APIRouter:
    """Create the webhook FastAPI router."""
    router = APIRouter(prefix="/webhook")

    @router.post("/trigger/{workflow_name}")
    async def trigger_workflow(workflow_name: str, payload: dict) -> dict:
        """Trigger a workflow via HTTP webhook."""
        try:
            execution_id = await orchestrator.trigger_workflow(workflow_name, payload)
            return {"execution_id": execution_id, "status": "started"}
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

    @router.get("/status/{execution_id}")
    async def get_status(execution_id: str) -> dict:
        """Check workflow execution status."""
        status = orchestrator.get_execution_status(execution_id)
        if not status:
            raise HTTPException(status_code=404, detail="Execution not found")
        return status

    return router
