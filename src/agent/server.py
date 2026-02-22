"""FastAPI server for agent containers.

Exposes endpoints for the mesh/orchestrator to interact with:
  POST /task     - accept a task assignment
  POST /cancel   - cancel current task
  GET  /status   - agent health check
  GET  /result   - last task result
  GET  /capabilities - list available skills
  GET  /workspace - list workspace files
  GET  /workspace/{filename} - read workspace file
  PUT  /workspace/{filename} - write workspace file
  PUT  /project - update fleet-wide PROJECT.md (pushed by mesh)
  GET  /workspace-logs - read daily logs (read-only)
  GET  /workspace-learnings - read errors and corrections (read-only)
  GET  /heartbeat-context - single-call heartbeat bootstrap data
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import json as json_module

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from src.shared.types import AgentMessage, AgentStatus, ChatMessage, ChatResponse, SteerMessage, TaskAssignment, TaskResult
from src.shared.utils import sanitize_for_prompt, setup_logging

if TYPE_CHECKING:
    from src.agent.loop import AgentLoop

logger = setup_logging("agent.server")


def create_agent_app(loop: AgentLoop) -> FastAPI:
    """Create the FastAPI application for an agent container."""
    app = FastAPI(title=f"OpenLegion Agent: {loop.agent_id}")

    @app.post("/task")
    async def receive_task(assignment: TaskAssignment, request: Request) -> dict:
        """Accept a task. Returns immediately; result sent back via mesh."""
        if loop.state != "idle":
            return {"accepted": False, "status": "busy", "error": "Agent is working"}

        # Transition to "working" immediately so the orchestrator never
        # sees a stale "idle" state between accept and task start.
        loop.state = "working"
        loop.current_task = assignment.task_id
        _trace_id = request.headers.get("x-trace-id")

        async def run() -> None:
            result = await loop.execute_task(assignment, trace_id=_trace_id)
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
    async def chat(msg: ChatMessage, request: Request) -> ChatResponse:
        """Interactive chat with the agent. Supports tool use."""
        result = await loop.chat(
            sanitize_for_prompt(msg.message),
            trace_id=request.headers.get("x-trace-id"),
        )
        return ChatResponse(**result)

    @app.post("/chat/steer")
    async def chat_steer(msg: SteerMessage) -> dict:
        """Inject a message into the active conversation. Does NOT acquire _chat_lock."""
        injected = await loop.inject_steer(sanitize_for_prompt(msg.message))
        return {"injected": injected, "agent_state": loop.state}

    @app.post("/chat/stream")
    async def chat_stream(msg: ChatMessage, request: Request) -> StreamingResponse:
        """Streaming chat. Returns SSE events for tool use and text deltas."""
        _trace_id = request.headers.get("x-trace-id")
        async def event_generator():
            async for event in loop.chat_stream(
                sanitize_for_prompt(msg.message), trace_id=_trace_id,
            ):
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

    # ── Workspace API ────────────────────────────────────────

    _WORKSPACE_ALLOWLIST = frozenset({"SOUL.md", "HEARTBEAT.md", "USER.md", "AGENTS.md", "MEMORY.md"})
    _DEFAULT_HEARTBEAT_PREFIX = "# Heartbeat Rules\n\nYou are woken periodically"

    _FILE_CAPS = {
        "SOUL.md": 4000,
        "AGENTS.md": 8000,
        "USER.md": 4000,
        "MEMORY.md": 16000,
        "HEARTBEAT.md": None,
    }
    _DEFAULT_PREFIXES = {
        "SOUL.md": "# Identity\n\nDefine personality",
        "AGENTS.md": "# Agent Instructions\n\nAdd operating",
        "USER.md": "# User Context\n\nRecord user",
        "MEMORY.md": "# Long-Term Memory\n\nCurated facts",
        "HEARTBEAT.md": "# Heartbeat Rules\n\nYou are woken periodically",
    }

    @app.get("/workspace")
    async def list_workspace() -> dict:
        """List editable workspace files with sizes, caps, and default status."""
        if not loop.workspace:
            return {"files": []}
        files = []
        for filename in sorted(_WORKSPACE_ALLOWLIST):
            path = loop.workspace.root / filename
            size = path.stat().st_size if path.exists() else 0
            # Detect default/empty files
            is_default = True
            if size > 0:
                content = loop.workspace._read_file(filename)
                if content:
                    prefix = _DEFAULT_PREFIXES.get(filename)
                    is_default = not content.strip() or (
                        prefix and content.strip().startswith(prefix)
                    )
            files.append({
                "name": filename,
                "size": size,
                "cap": _FILE_CAPS.get(filename),
                "is_default": is_default,
            })
        return {"files": files}

    @app.get("/workspace/{filename}")
    async def read_workspace_file(filename: str) -> dict:
        """Read a workspace file by name."""
        if filename not in _WORKSPACE_ALLOWLIST:
            raise HTTPException(400, f"File not allowed: {filename}")
        if not loop.workspace:
            raise HTTPException(503, "Workspace not available")
        content = loop.workspace._read_file(filename) or ""
        return {"filename": filename, "content": content}

    @app.put("/workspace/{filename}")
    async def write_workspace_file(filename: str, request: Request) -> dict:
        """Write content to a workspace file."""
        if filename not in _WORKSPACE_ALLOWLIST:
            raise HTTPException(400, f"File not allowed: {filename}")
        if not loop.workspace:
            raise HTTPException(503, "Workspace not available")
        body = await request.json()
        content = body.get("content", "")
        if not isinstance(content, str):
            raise HTTPException(400, "content must be a string")
        content = sanitize_for_prompt(content)
        path = loop.workspace.root / filename
        path.write_text(content)
        return {"filename": filename, "size": path.stat().st_size}

    @app.put("/project")
    async def update_project(request: Request) -> dict:
        """Accept an updated PROJECT.md from the mesh host.

        PROJECT.md is fleet-wide (not per-agent), so it's separate from
        the identity file allowlist. The mesh pushes updates here after
        the user edits it on the dashboard.
        """
        if not loop.workspace:
            raise HTTPException(503, "Workspace not available")
        body = await request.json()
        content = body.get("content", "")
        if not isinstance(content, str):
            raise HTTPException(400, "content must be a string")
        content = sanitize_for_prompt(content)
        path = loop.workspace.root / "PROJECT.md"
        path.write_text(content)
        return {"updated": True, "size": path.stat().st_size}

    @app.get("/workspace-logs")
    async def workspace_logs(days: int = 3) -> dict:
        """Return daily logs for the dashboard (read-only)."""
        if not loop.workspace:
            return {"logs": ""}
        days = max(1, min(days, 14))
        content = loop.workspace.load_daily_logs(days=days)
        if len(content) > 16000:
            content = content[:16000] + "\n\n... (truncated)"
        return {"logs": content}

    @app.get("/workspace-learnings")
    async def workspace_learnings() -> dict:
        """Return errors and corrections for the dashboard (read-only)."""
        if not loop.workspace:
            return {"errors": "", "corrections": ""}
        errors = loop.workspace._read_file("learnings/errors.md") or ""
        corrections = loop.workspace._read_file("learnings/corrections.md") or ""
        if len(errors) > 8000:
            errors = errors[-8000:]
        if len(corrections) > 8000:
            corrections = corrections[-8000:]
        return {"errors": errors, "corrections": corrections}

    @app.get("/heartbeat-context")
    async def heartbeat_context() -> dict:
        """Return everything a heartbeat needs in a single call."""
        if not loop.workspace:
            return {
                "heartbeat_rules": "",
                "daily_logs": "",
                "is_default_heartbeat": True,
                "has_recent_activity": False,
            }
        rules = loop.workspace.load_heartbeat_rules()
        daily = loop.workspace.load_daily_logs(days=2)
        is_default = (
            not rules.strip()
            or rules.strip().startswith(_DEFAULT_HEARTBEAT_PREFIX)
        )
        # Cap daily logs at 8000 chars for transport
        if len(daily) > 8000:
            daily = daily[:8000] + "\n\n... (truncated)"
        return {
            "heartbeat_rules": rules,
            "daily_logs": daily,
            "is_default_heartbeat": is_default,
            "has_recent_activity": bool(daily.strip()),
        }

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
