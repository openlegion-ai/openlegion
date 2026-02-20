"""FastAPI server for agent containers.

Exposes endpoints for the mesh/orchestrator to interact with:
  POST /task     - accept a task assignment
  POST /cancel   - cancel current task
  GET  /status   - agent health check
  GET  /result   - last task result
  GET  /capabilities - list available skills
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import json as json_module

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from src.shared.types import AgentMessage, AgentStatus, ChatMessage, ChatResponse, SteerMessage, TaskAssignment, TaskResult
from src.shared.utils import setup_logging

if TYPE_CHECKING:
    from src.agent.loop import AgentLoop

logger = setup_logging("agent.server")


def create_agent_app(loop: AgentLoop) -> FastAPI:
    """Create the FastAPI application for an agent container."""
    app = FastAPI(title=f"OpenLegion Agent: {loop.agent_id}")

    @app.post("/task")
    async def receive_task(assignment: TaskAssignment) -> dict:
        """Accept a task. Returns immediately; result sent back via mesh."""
        if loop.state != "idle":
            return {"accepted": False, "status": "busy", "error": "Agent is working"}

        # Transition to "working" immediately so the orchestrator never
        # sees a stale "idle" state between accept and task start.
        loop.state = "working"
        loop.current_task = assignment.task_id

        async def run() -> None:
            result = await loop.execute_task(assignment)
            try:
                await loop.mesh_client.send_system_message(
                    to="orchestrator",
                    msg_type="task_result",
                    payload=result.model_dump(mode="json"),
                )
            except Exception as e:
                logger.error(f"Failed to send task result to orchestrator: {e}")

        task = asyncio.create_task(run())
        task.add_done_callback(_log_task_exception)
        loop._current_task_handle = task
        return {"accepted": True, "status": "executing"}

    @app.post("/cancel")
    async def cancel_task() -> dict:
        """Cancel the current task."""
        handle = loop._current_task_handle
        if handle and not handle.done():
            loop._cancel_requested = True
            try:
                await asyncio.wait_for(asyncio.shield(handle), timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                if not handle.done():
                    handle.cancel()
        return {"status": "cancelled"}

    @app.get("/status", response_model=AgentStatus)
    async def get_status() -> AgentStatus:
        """Return current agent status."""
        return loop.get_status()

    @app.get("/result", response_model=TaskResult)
    async def get_result() -> TaskResult:
        """Return the last completed task result."""
        if loop._last_result is None:
            raise HTTPException(404, "No task result available")
        return loop._last_result

    @app.get("/capabilities")
    async def get_capabilities() -> dict:
        """Return agent capabilities and tool definitions."""
        return {
            "agent_id": loop.agent_id,
            "role": loop.role,
            "skills": loop.skills.list_skills(),
            "tool_definitions": loop.skills.get_tool_definitions(),
        }

    @app.post("/chat", response_model=ChatResponse)
    async def chat(msg: ChatMessage) -> ChatResponse:
        """Interactive chat with the agent. Supports tool use."""
        result = await loop.chat(msg.message)
        return ChatResponse(**result)

    @app.post("/chat/steer")
    async def chat_steer(msg: SteerMessage) -> dict:
        """Inject a message into the active conversation. Does NOT acquire _chat_lock."""
        injected = await loop.inject_steer(msg.message)
        return {"injected": injected, "agent_state": loop.state}

    @app.post("/chat/stream")
    async def chat_stream(msg: ChatMessage) -> StreamingResponse:
        """Streaming chat. Returns SSE events for tool use and text deltas."""
        async def event_generator():
            async for event in loop.chat_stream(msg.message):
                yield f"data: {json_module.dumps(event, default=str)}\n\n"
        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @app.post("/chat/reset")
    async def chat_reset() -> dict:
        """Clear conversation history."""
        await loop.reset_chat()
        return {"status": "ok"}

    @app.get("/history")
    async def get_history(days: int = 3) -> dict:
        """Return this agent's daily logs for inter-agent context sharing."""
        if not loop.workspace:
            return {"agent_id": loop.agent_id, "logs": [], "memory": ""}
        daily = loop.workspace.load_daily_logs(days=days)
        memory = loop.workspace.load_memory()
        return {
            "agent_id": loop.agent_id,
            "role": loop.role,
            "logs": daily,
            "memory": memory[:5000] if memory else "",
        }

    @app.post("/message")
    async def receive_message(msg: AgentMessage) -> dict:
        """Receive an async message from another agent via mesh routing.

        Stores the message in the agent's memory so it has context on
        the next task, chat, or heartbeat activation.
        """
        content = (
            f"Message from {msg.from_agent} ({msg.type}): "
            f"{_summarize_payload(msg.payload)}"
        )
        loop.workspace.append_daily_log(content)
        logger.info(f"Received message from {msg.from_agent}: {msg.type}")
        return {"received": True, "from": msg.from_agent, "type": msg.type}

    return app


def _summarize_payload(payload: dict, max_len: int = 500) -> str:
    """Compact a message payload for memory storage."""
    import json
    text = json.dumps(payload, default=str)
    return text[:max_len] + "..." if len(text) > max_len else text


def _log_task_exception(task: asyncio.Task) -> None:
    """Log unhandled exceptions from fire-and-forget tasks."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc:
        logger.error(f"Background task failed: {exc}", exc_info=exc)
