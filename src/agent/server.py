"""FastAPI server for agent containers.

Exposes endpoints for the mesh to interact with:
  POST /task     - accept a task assignment
  POST /cancel   - cancel current task
  GET  /status   - agent health check
  GET  /result   - last task result
  GET  /capabilities - list available skills and tool definitions
  POST /invoke   - execute a named tool directly (no LLM)
  GET  /workspace - list workspace files
  GET  /workspace/{filename} - read workspace file
  PUT  /workspace/{filename} - write workspace file
  PUT  /project - update fleet-wide PROJECT.md (pushed by mesh)
  GET  /workspace-logs - read daily logs (read-only)
  GET  /workspace-learnings - read errors and corrections (read-only)
  GET  /heartbeat-context - single-call heartbeat bootstrap data
  GET  /artifacts - list artifact files
  GET  /artifacts/{name} - read artifact content
  DELETE /artifacts/{name} - delete artifact file
  GET  /files?path=.&recursive=false - list all /data files
  GET  /files/{path} - read any /data file (text or base64)
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json as json_module
import mimetypes
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from src.shared.types import (
    AgentMessage,
    ChatMessage,
    ChatResponse,
    SteerMessage,
    TaskAssignment,
    TaskResult,
)
from src.shared.utils import sanitize_for_prompt, setup_logging

if TYPE_CHECKING:
    from src.agent.loop import AgentLoop

logger = setup_logging("agent.server")

_MAX_HISTORY_MEMORY_CHARS = 5000
_MAX_HISTORY_LOGS_CHARS = 16000
_MAX_HISTORY_LEARNINGS_CHARS = 8000


def create_agent_app(loop: AgentLoop) -> FastAPI:
    """Create the FastAPI application for an agent container."""
    app = FastAPI(title=f"OpenLegion Agent: {loop.agent_id}")
    _task_accept_lock = asyncio.Lock()

    @app.post("/task")
    async def receive_task(assignment: TaskAssignment, request: Request) -> dict:
        """Accept a task. Returns immediately; result sent back via mesh."""
        async with _task_accept_lock:
            if loop.state != "idle":
                return {"accepted": False, "status": "busy", "error": "Agent is working"}

            # Transition to "working" immediately so callers never
            # see a stale "idle" state between accept and task start.
            loop.state = "working"
            loop.current_task = assignment.task_id
        _trace_id = request.headers.get("x-trace-id")

        async def run() -> None:
            try:
                await loop.execute_task(assignment, trace_id=_trace_id)
            except asyncio.CancelledError:
                loop.state = "idle"
                loop.current_task = None

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
            # Give the loop up to 2s to notice _cancel_requested and exit cleanly.
            done, _ = await asyncio.wait({handle}, timeout=2.0)
            if not done and not handle.done():
                handle.cancel()
        return {"status": "cancelled"}

    @app.get("/status")
    async def get_status() -> dict:
        """Return current agent status, including task checkpoint info."""
        s = loop.get_status()
        s_dict = s.model_dump()
        has_checkpoint = False
        if loop.memory:
            try:
                cp = await loop.memory._run_db(loop.memory.load_task_checkpoint)
                has_checkpoint = cp is not None
            except Exception:
                pass
        s_dict["has_task_checkpoint"] = has_checkpoint
        return s_dict

    @app.get("/result", response_model=TaskResult)
    async def get_result() -> TaskResult:
        """Return the last completed task result."""
        if loop._last_result is None:
            raise HTTPException(404, "No task result available")
        return loop._last_result

    @app.get("/capabilities")
    async def get_capabilities() -> dict:
        """Return agent capabilities and tool definitions.

        Respects the same exclude filter used when calling the LLM, so the
        dashboard accurately reflects what the agent can actually use.
        """
        exc = loop._excluded_tools
        alw = loop._allowed_tools if isinstance(loop._allowed_tools, frozenset) else None
        return {
            "agent_id": loop.agent_id,
            "role": loop.role,
            "skills": loop.skills.list_skills(exclude=exc, allowed=alw),
            "tool_definitions": loop.skills.get_tool_definitions(exclude=exc, allowed=alw),
            "tool_sources": loop.skills.get_tool_sources(exclude=exc, allowed=alw),
        }

    @app.post("/invoke")
    async def invoke_tool(request: Request) -> dict:
        """Execute a named tool directly without LLM involvement.

        Called by the cron scheduler for tool-type jobs. The tool runs with
        the same dependency injection (mesh_client, workspace, memory) as it
        would during a normal agent turn.

        Body: {"tool": str, "params": {}}
        """
        body = await request.json()
        name = body.get("tool", "")
        params = body.get("params") or {}

        if not name:
            raise HTTPException(400, "tool name is required")
        if not isinstance(params, dict):
            raise HTTPException(400, "params must be a JSON object")
        allowed = loop._allowed_tools if isinstance(loop._allowed_tools, frozenset) else None
        excluded = loop._excluded_tools or frozenset()
        if allowed is not None and name not in allowed:
            raise HTTPException(403, f"Tool '{name}' is not available to this agent")
        elif allowed is None and name in excluded:
            raise HTTPException(403, f"Tool '{name}' is not available to this agent")

        try:
            result = await loop.skills.execute(
                name, params,
                mesh_client=loop.mesh_client,
                workspace_manager=loop.workspace,
                memory_store=loop.memory,
            )
            return {"result": result}
        except ValueError as e:
            raise HTTPException(404, str(e))
        except Exception as e:
            logger.warning("invoke_tool '%s' failed: %s", name, e)
            return {"error": str(e)}

    @app.post("/heartbeat")
    async def heartbeat(msg: ChatMessage) -> dict:
        """Execute an autonomous heartbeat — separate from chat."""
        result = await loop.execute_heartbeat(sanitize_for_prompt(msg.message))
        return result

    @app.get("/activity")
    async def get_activity(limit: int = 100) -> dict:
        """Return the activity log (heartbeat and autonomous work)."""
        if not loop.workspace:
            return {"activity": [], "count": 0}
        limit = max(1, min(limit, 500))
        entries = loop.workspace.load_activity(limit=limit)
        return {"activity": entries, "count": len(entries)}

    @app.post("/chat", response_model=ChatResponse)
    async def chat(msg: ChatMessage, request: Request) -> ChatResponse:
        """Interactive chat with the agent. Supports tool use."""
        from src.shared.trace import parse_origin_header
        origin = parse_origin_header(request.headers.get("x-origin"))
        result = await loop.chat(
            sanitize_for_prompt(msg.message),
            trace_id=request.headers.get("x-trace-id"),
            origin=origin,
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
        from src.shared.trace import parse_origin_header
        _trace_id = request.headers.get("x-trace-id")
        _origin = parse_origin_header(request.headers.get("x-origin"))
        async def event_generator():
            stream = loop.chat_stream(
                sanitize_for_prompt(msg.message), trace_id=_trace_id,
                origin=_origin,
            )
            stream_iter = stream.__aiter__()
            # Use asyncio.wait (not wait_for) so the pending __anext__
            # is never cancelled — cancellation would close the async
            # generator and silently drop the response.
            next_event = asyncio.ensure_future(stream_iter.__anext__())
            try:
                while True:
                    done, _ = await asyncio.wait(
                        {next_event}, timeout=15,
                    )
                    if not done:
                        yield ": keepalive\n\n"
                        continue
                    try:
                        event = next_event.result()
                    except StopAsyncIteration:
                        break
                    yield f"data: {json_module.dumps(event, default=str)}\n\n"
                    next_event = asyncio.ensure_future(stream_iter.__anext__())
            finally:
                if not next_event.done():
                    next_event.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await next_event
        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @app.post("/chat/reset")
    async def chat_reset() -> dict:
        """Clear conversation history."""
        await loop.reset_chat()
        return {"status": "ok"}

    @app.get("/chat/history")
    async def chat_history() -> dict:
        """Return the persistent chat transcript.

        Reads from the workspace transcript file so conversation history
        survives context compaction, container restarts, and is accessible
        from any device.  Falls back to in-memory messages when the
        transcript is unavailable.
        """
        messages = loop.get_chat_messages()
        return {"messages": messages, "count": len(messages)}

    @app.get("/history")
    async def get_history(days: int = 3) -> dict:
        """Return this agent's daily logs for inter-agent context sharing."""
        days = max(1, min(days, 14))
        if not loop.workspace:
            return {"agent_id": loop.agent_id, "logs": [], "memory": ""}
        daily = loop.workspace.load_daily_logs(days=days)
        memory = loop.workspace.load_memory()
        return {
            "agent_id": loop.agent_id,
            "role": loop.role,
            "logs": daily,
            "memory": memory[:_MAX_HISTORY_MEMORY_CHARS] if memory else "",
        }

    @app.post("/message")
    async def receive_message(msg: AgentMessage) -> dict:
        """Receive an async message from another agent via mesh routing.

        Stores the message in the agent's memory so it has context on
        the next task, chat, or heartbeat activation.
        """
        content = sanitize_for_prompt(
            f"Message from {msg.from_agent} ({msg.type}): "
            f"{_summarize_payload(msg.payload)}"
        )
        if loop.workspace:
            loop.workspace.append_daily_log(content)
        logger.info(f"Received message from {msg.from_agent}: {msg.type}")
        return {"received": True, "from": msg.from_agent, "type": msg.type}

    # ── Workspace API ────────────────────────────────────────

    _WORKSPACE_ALLOWLIST = frozenset({
        "SOUL.md", "HEARTBEAT.md", "USER.md", "INSTRUCTIONS.md", "AGENTS.md", "MEMORY.md",
        "INTERFACE.md",
    })
    _DEFAULT_HEARTBEAT_HEADING = "# Heartbeat Rules"

    _FILE_CAPS = {
        "SOUL.md": 4000,
        "INSTRUCTIONS.md": 12000,
        "AGENTS.md": 12000,
        "USER.md": 4000,
        "MEMORY.md": 16000,
        "HEARTBEAT.md": None,
        "INTERFACE.md": 4000,
    }
    _DEFAULT_HEADINGS = {
        "SOUL.md": "# Identity",
        "INSTRUCTIONS.md": "# Instructions",
        "AGENTS.md": "# Agent Instructions",
        "USER.md": "# User Context",
        "MEMORY.md": "# Long-Term Memory",
        "HEARTBEAT.md": "# Heartbeat Rules",
        "INTERFACE.md": "# Interface",
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
                    heading = _DEFAULT_HEADINGS.get(filename)
                    is_default = not content.strip() or (
                        heading is not None and content.strip() == heading
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
        """Write content to a workspace file.

        This endpoint is for mesh host / dashboard use only (human-initiated
        edits). Agent-side writes go through the update_workspace skill which
        enforces its own allowlist and versioning. We require X-Mesh-Internal
        header to prevent agents from calling their own endpoint via
        http_request or exec+curl.
        """
        if not request.headers.get("x-mesh-internal"):
            raise HTTPException(
                403,
                "Workspace writes via this endpoint require X-Mesh-Internal header. "
                "Agents should use the update_workspace tool instead.",
            )
        if filename not in _WORKSPACE_ALLOWLIST:
            raise HTTPException(400, f"File not allowed: {filename}")
        if not loop.workspace:
            raise HTTPException(503, "Workspace not available")
        body = await request.json()
        content = body.get("content", "")
        if not isinstance(content, str):
            raise HTTPException(400, "content must be a string")
        content = sanitize_for_prompt(content)
        cap = _FILE_CAPS.get(filename)
        if cap is not None and len(content) > cap:
            raise HTTPException(413, f"{filename} exceeds cap ({len(content)} > {cap} chars)")
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
        if not request.headers.get("x-mesh-internal"):
            raise HTTPException(
                403,
                "Project updates via this endpoint require X-Mesh-Internal header.",
            )
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

    @app.post("/config")
    async def update_runtime_config(request: Request) -> dict:
        """Hot-reload runtime config (model, thinking) without container restart.

        Mesh pushes here after the operator confirms a model/thinking edit
        so the next LLM call uses the new setting. Fields read from env vars
        at startup (LLM_MODEL, THINKING) don't get picked up by YAML edits
        on their own — this endpoint closes that gap.
        """
        if not request.headers.get("x-mesh-internal"):
            raise HTTPException(
                403,
                "Runtime config updates require X-Mesh-Internal header.",
            )
        if not loop.llm:
            raise HTTPException(503, "LLM client not available")
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(400, "Body must be a JSON object")

        # Validate all fields before applying any (atomicity).
        from src.agent.llm import LLMClient
        model = body.get("model") if "model" in body else None
        thinking = body.get("thinking") if "thinking" in body else None
        if "model" in body and (not isinstance(model, str) or not model):
            raise HTTPException(400, "model must be a non-empty string")
        if "thinking" in body and thinking not in LLMClient.VALID_THINKING_LEVELS:
            raise HTTPException(
                400,
                f"thinking must be one of: {sorted(LLMClient.VALID_THINKING_LEVELS)}",
            )

        updated: dict[str, str] = {}
        if "model" in body:
            loop.llm.default_model = model
            updated["model"] = model
        if "thinking" in body:
            loop.llm.thinking = thinking
            updated["thinking"] = thinking
        return {"updated": updated}

    @app.get("/workspace-logs")
    async def workspace_logs(days: int = 3) -> dict:
        """Return daily logs for the dashboard (read-only)."""
        if not loop.workspace:
            return {"logs": ""}
        days = max(1, min(days, 14))
        content = loop.workspace.load_daily_logs(days=days)
        if len(content) > _MAX_HISTORY_LOGS_CHARS:
            content = content[:_MAX_HISTORY_LOGS_CHARS] + "\n\n... (truncated)"
        return {"logs": content}

    @app.get("/workspace-learnings")
    async def workspace_learnings() -> dict:
        """Return errors and corrections for the dashboard (read-only)."""
        if not loop.workspace:
            return {"errors": "", "corrections": ""}
        errors = loop.workspace._read_file("learnings/errors.md") or ""
        corrections = loop.workspace._read_file("learnings/corrections.md") or ""
        if len(errors) > _MAX_HISTORY_LEARNINGS_CHARS:
            errors = errors[-_MAX_HISTORY_LEARNINGS_CHARS:]
        if len(corrections) > _MAX_HISTORY_LEARNINGS_CHARS:
            corrections = corrections[-_MAX_HISTORY_LEARNINGS_CHARS:]
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
                "is_standalone": loop.mesh_client.is_standalone,
            }
        rules = loop.workspace.load_heartbeat_rules()
        daily = loop.workspace.load_daily_logs(days=2)
        is_default = (
            not rules.strip()
            or rules.strip() == _DEFAULT_HEARTBEAT_HEADING
        )
        # Cap daily logs at _MAX_HISTORY_LEARNINGS_CHARS chars for transport
        if len(daily) > _MAX_HISTORY_LEARNINGS_CHARS:
            daily = daily[:_MAX_HISTORY_LEARNINGS_CHARS] + "\n\n... (truncated)"
        return {
            "heartbeat_rules": rules,
            "daily_logs": daily,
            "is_default_heartbeat": is_default,
            "has_recent_activity": bool(daily.strip()),
            "is_standalone": loop.mesh_client.is_standalone,
        }

    # ── Artifact API ─────────────────────────────────────────

    _MAX_ARTIFACT_BYTES = 2 * 1024 * 1024  # 2 MB cap for content transfer (read)
    # 50 MB cap for ingested artifacts (browser downloads, mesh-streamed files).
    # Larger than read cap because ingestion streams; reads are loaded in memory.
    # Overridable via env so operators with specialty workflows can tune.
    import os as _os_ingest
    _MAX_ARTIFACT_INGEST_BYTES = int(_os_ingest.environ.get(
        "OPENLEGION_ARTIFACT_INGEST_MAX_MB", "50",
    )) * 1024 * 1024
    del _os_ingest
    _ARTIFACT_NAME_RE = re.compile(r"^[\w][\w.\-/ ]{0,198}[\w.]$")

    @app.get("/artifacts")
    async def list_artifacts() -> dict:
        """List artifact files in the workspace."""
        if not loop.workspace:
            return {"artifacts": []}
        artifacts_dir = Path(loop.workspace.root) / "artifacts"
        if not artifacts_dir.is_dir():
            return {"artifacts": []}
        items = []
        for f in sorted(artifacts_dir.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(artifacts_dir)
            items.append({
                "name": str(rel),
                "size": f.stat().st_size,
                "modified": f.stat().st_mtime,
            })
        return {"artifacts": items}

    @app.get("/artifacts/{name:path}")
    async def read_artifact(name: str) -> dict:
        """Return artifact content for dashboard preview/download.

        Text files are returned as-is.  Binary files are base64-encoded.
        Capped at 2 MB to prevent transport overload.
        """
        if not loop.workspace:
            raise HTTPException(503, "Workspace not available")
        if not _ARTIFACT_NAME_RE.match(name):
            raise HTTPException(400, f"Invalid artifact name: {name}")
        artifacts_dir = Path(loop.workspace.root) / "artifacts"
        filepath = (artifacts_dir / name).resolve()
        if not filepath.is_relative_to(artifacts_dir.resolve()):
            raise HTTPException(400, "Path traversal not allowed")
        if not filepath.is_file():
            raise HTTPException(404, f"Artifact not found: {name}")
        size = filepath.stat().st_size
        if size > _MAX_ARTIFACT_BYTES:
            raise HTTPException(413, f"Artifact too large ({size} bytes, max {_MAX_ARTIFACT_BYTES})")
        # Try text first, fall back to base64 for binary
        mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
        try:
            content = filepath.read_text(encoding="utf-8")
            return {"name": name, "content": content, "size": size,
                    "mime_type": mime, "encoding": "utf-8"}
        except UnicodeDecodeError:
            raw = filepath.read_bytes()
            return {"name": name, "content": base64.b64encode(raw).decode("ascii"),
                    "size": size, "mime_type": mime, "encoding": "base64"}

    @app.get("/files")
    async def list_data_files(
        path: str = ".",
        recursive: bool = False,
        pattern: str = "*",
    ) -> dict:
        """List files under /data (the agent's full data volume)."""
        from src.agent.builtins.file_tool import list_files
        try:
            return list_files(path=path, pattern=pattern, recursive=recursive)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.get("/files/{path:path}")
    async def read_data_file(path: str) -> dict:
        """Read any file from /data. Text returned as-is; binary base64-encoded."""
        from src.agent.builtins.file_tool import _MAX_READ, _safe_path
        try:
            safe = _safe_path(path)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if not safe.exists():
            raise HTTPException(404, f"File not found: {path}")
        if not safe.is_file():
            raise HTTPException(400, f"Not a file: {path}")
        size = safe.stat().st_size
        mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
        # Read at most _MAX_READ bytes without loading the full file first.
        with safe.open("rb") as fh:
            raw = fh.read(_MAX_READ)
        try:
            content = raw.decode("utf-8")
            return {"path": path, "content": content, "size": size,
                    "mime_type": mime, "encoding": "utf-8",
                    "truncated": size > _MAX_READ}
        except UnicodeDecodeError:
            return {"path": path, "content": base64.b64encode(raw).decode("ascii"),
                    "size": size, "mime_type": mime, "encoding": "base64",
                    "truncated": size > _MAX_READ}

    @app.post("/artifacts/ingest/{name:path}")
    async def ingest_artifact(name: str, request: Request) -> dict:
        """Stream-write an artifact into the workspace (Phase 1.5).

        Intended as the landing endpoint for browser downloads: the mesh
        streams bytes from the browser service here, and they arrive in
        the agent's workspace as a regular artifact (same API surface as
        artifacts saved by the agent itself via ``save_artifact``).

        Disciplines:
        - **X-Mesh-Internal required.** Mirrors the workspace-write endpoint
          — agents shouldn't be calling their own ingest from a tool, only
          the mesh should stream into it.
        - **Streaming size cap.** Counts bytes as they arrive and aborts if
          the 50 MB default cap is exceeded. Does NOT trust Content-Length.
        - **Atomic write.** Streams into ``{name}.partial``, then
          ``os.replace`` on success. A crash mid-stream leaves an orphan
          ``.partial`` that ``delete_artifact``/operator cleanup can remove.
        - **Filename collision.** If ``{name}`` exists, the name gains a
          numeric suffix: ``foo.pdf`` → ``foo-1.pdf`` → ``foo-2.pdf`` … .
          The final name is returned so the caller can look it up.
        - **Same path-traversal guards** as existing artifact endpoints.
        """
        if not loop.workspace:
            raise HTTPException(503, "Workspace not available")
        if not request.headers.get("x-mesh-internal"):
            raise HTTPException(
                403,
                "Artifact ingest requires X-Mesh-Internal header.",
            )
        if not _ARTIFACT_NAME_RE.match(name):
            raise HTTPException(400, f"Invalid artifact name: {name}")

        artifacts_dir = Path(loop.workspace.root) / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        target = (artifacts_dir / name).resolve()
        resolved_dir = artifacts_dir.resolve()
        if not target.is_relative_to(resolved_dir):
            raise HTTPException(400, "Path traversal not allowed")
        target.parent.mkdir(parents=True, exist_ok=True)

        # Collision avoidance — never overwrite. Append numeric suffix.
        final = _disambiguate_artifact_name(target)
        partial = final.with_suffix(final.suffix + ".partial")

        max_bytes = _MAX_ARTIFACT_INGEST_BYTES
        bytes_written = 0
        try:
            with partial.open("wb") as fh:
                async for chunk in request.stream():
                    bytes_written += len(chunk)
                    if bytes_written > max_bytes:
                        raise HTTPException(
                            413,
                            f"Artifact exceeds {max_bytes} bytes",
                        )
                    fh.write(chunk)
        except HTTPException:
            # Size-cap or client disconnect — clean up partial.
            with contextlib.suppress(FileNotFoundError, OSError):
                partial.unlink()
            raise
        except Exception as e:
            with contextlib.suppress(FileNotFoundError, OSError):
                partial.unlink()
            raise HTTPException(500, f"Ingest failed: {e}") from e

        # Reject zero-byte: almost always a client error, better to 400
        # than silently create an empty artifact that agents later hit.
        if bytes_written == 0:
            with contextlib.suppress(FileNotFoundError, OSError):
                partial.unlink()
            raise HTTPException(400, "Empty request body")

        os.replace(partial, final)
        rel_name = str(final.relative_to(resolved_dir))
        mime = mimetypes.guess_type(rel_name)[0] or "application/octet-stream"
        return {
            "artifact_name": rel_name,
            "size_bytes": bytes_written,
            "mime_type": mime,
        }

    @app.delete("/artifacts/{name:path}")
    async def delete_artifact(name: str) -> dict:
        """Delete an artifact file from the workspace."""
        if not loop.workspace:
            raise HTTPException(503, "Workspace not available")
        if not _ARTIFACT_NAME_RE.match(name):
            raise HTTPException(400, f"Invalid artifact name: {name}")
        artifacts_dir = Path(loop.workspace.root) / "artifacts"
        filepath = (artifacts_dir / name).resolve()
        if not filepath.is_relative_to(artifacts_dir.resolve()):
            raise HTTPException(400, "Path traversal not allowed")
        if not filepath.is_file():
            raise HTTPException(404, f"Artifact not found: {name}")
        filepath.unlink()
        # Clean up empty parent directories up to (but not including) artifacts_dir
        resolved_artifacts = artifacts_dir.resolve()
        parent = filepath.parent
        while parent != resolved_artifacts and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent
        return {"deleted": True, "name": name}

    return app


def _summarize_payload(payload: dict, max_len: int = 500) -> str:
    """Compact a message payload for memory storage."""
    text = json_module.dumps(payload, default=str)
    return text[:max_len] + "..." if len(text) > max_len else text


def _disambiguate_artifact_name(target: Path) -> Path:
    """Return a unique file path next to ``target``.

    If ``target`` doesn't exist, returns it unchanged. If it does, appends
    ``-1``, ``-2``, … before the suffix until a free name is found. Stops
    at 999 attempts to avoid pathological loops on a corrupt dir.
    """
    if not target.exists():
        return target
    stem = target.stem
    suffix = target.suffix
    parent = target.parent
    for n in range(1, 1000):
        candidate = parent / f"{stem}-{n}{suffix}"
        if not candidate.exists():
            return candidate
    # 1000 collisions is unrecoverable; let the caller handle as a server error.
    raise RuntimeError(f"Too many collisions disambiguating {target}")


def _log_task_exception(task: asyncio.Task) -> None:
    """Log unhandled exceptions from fire-and-forget tasks."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc:
        logger.error(f"Background task failed: {exc}", exc_info=exc)
