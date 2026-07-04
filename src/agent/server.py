"""FastAPI server for agent containers.

Exposes endpoints for the mesh to interact with:
  POST /task     - accept a task assignment
  POST /cancel   - cancel current task
  GET  /status   - agent health check
  GET  /result   - last task result
  GET  /capabilities - list available tools and tool definitions
  POST /invoke   - execute a named tool directly (no LLM)
  GET  /workspace - list workspace files
  GET  /workspace/{filename} - read workspace file
  PUT  /workspace/{filename} - write workspace file
  PUT  /team    - update fleet-wide TEAM.md (pushed by mesh)
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
import mimetypes
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from src.shared.paths import resolve_under_root
from src.shared.types import (
    AgentMessage,
    ChatMessage,
    ChatResponse,
    SteerMessage,
    TaskAssignment,
    TaskResult,
)
from src.shared.utils import dumps_safe, sanitize_for_prompt, setup_logging

if TYPE_CHECKING:
    from src.agent.loop import AgentLoop

logger = setup_logging("agent.server")


def _max_body_bytes() -> int:
    """Resolve the request body-size cap (env-configurable, default 8 MiB).

    Mirrors the browser service cap. Anything larger than this is either
    operator error or a memory-DoS attempt against the JSON parser — an
    unbounded body is buffered into RAM before parsing, so a multi-GB POST
    from an authenticated agent could OOM the single agent process. Env
    override ``OPENLEGION_MAX_BODY_MB`` (default 8).
    """
    try:
        mb = float(os.environ.get("OPENLEGION_MAX_BODY_MB", "8"))
    except ValueError:
        mb = 8.0
    if mb <= 0:
        mb = 8.0
    return int(mb * 1024 * 1024)


def _install_body_size_limit(app: FastAPI) -> None:
    """Register an outer HTTP middleware that rejects oversized bodies.

    Two layers of defence:
      1. ``Content-Length`` header check — cheap, rejects honest clients early.
      2. Streaming byte counter — a client can omit Content-Length (chunked
         transfer) to bypass the header check, so we also wrap the ASGI
         receive channel and abort with HTTP 413 the moment the streamed body
         exceeds the cap, before the whole body is buffered into RAM.

    This is an outer middleware: it runs before routing and does not strip
    headers, so it does not interfere with downstream auth / CSRF
    dependencies.
    """
    from starlette.responses import JSONResponse as _StarletteJSONResponse

    @app.middleware("http")
    async def _enforce_body_size(request: Request, call_next):
        max_bytes = _max_body_bytes()
        cl = request.headers.get("content-length")
        if cl is not None:
            try:
                size = int(cl)
            except ValueError:
                return _StarletteJSONResponse(
                    {"detail": "invalid Content-Length"}, status_code=400,
                )
            if size > max_bytes:
                return _StarletteJSONResponse(
                    {"detail": "request body too large"}, status_code=413,
                )

        # Streaming guard for chunked / Content-Length-absent bodies: a
        # client can omit Content-Length to bypass the header check, so we
        # drain the ASGI receive channel here with a running byte counter and
        # bail out the moment the cap is exceeded — the endpoint is never
        # invoked and at most ``max_bytes`` ever lands in RAM. Buffered
        # messages are then replayed to the handler unchanged so routing /
        # auth / CSRF dependencies see the body exactly as sent.
        original_receive = request._receive
        received = 0
        buffered: list[dict] = []
        while True:
            message = await original_receive()
            buffered.append(message)
            if message["type"] != "http.request":
                # http.disconnect or other — stop draining, replay as-is.
                break
            received += len(message.get("body", b""))
            if received > max_bytes:
                return _StarletteJSONResponse(
                    {"detail": "request body too large"}, status_code=413,
                )
            if not message.get("more_body", False):
                break

        _replay = iter(buffered)

        async def _replay_receive():
            try:
                return next(_replay)
            except StopIteration:
                # Body fully consumed; defer to the live channel for any
                # trailing http.disconnect.
                return await original_receive()

        request._receive = _replay_receive
        return await call_next(request)


_MAX_HISTORY_MEMORY_CHARS = 5000
_MAX_HISTORY_LOGS_CHARS = 16000
_MAX_HISTORY_LEARNINGS_CHARS = 8000

# Background memory-maintenance pass cadence. The first run is delayed so boot
# and the first user turn settle; thereafter it ticks periodically. The actual
# work (consolidation + decay) is internally 6h-gated, so the short delay also
# gives a frequently-restarted agent (e.g. the operator) a maintenance attempt
# soon after each boot rather than never reaching a long interval.
_MAINTENANCE_INITIAL_DELAY_S = 180
_MAINTENANCE_TICK_S = 30 * 60

# /chat/note size bound — mirrors the mesh notify path's _NOTIFY_MAX_LEN.
# Oversized notes are truncated, not rejected: the mesh side already caps
# its payloads, so this is a backstop against transcript-rotation churn.
_NOTE_MAX_LEN = 2000


def _origin_from_mesh_request(request: Request):
    """Parse origin from a host-to-agent request.

    ``X-Origin`` is authorization-bearing only after the mesh/dashboard/CLI
    stamped it. The host transport adds ``x-mesh-internal`` on those hops, so
    preserve the typed kind only for that trusted internal path. Direct callers
    still get the strict parser, which downgrades caller-supplied kinds.
    """
    from src.shared.trace import parse_origin_header
    from src.shared.types import MessageOrigin

    raw = request.headers.get("x-origin")
    if request.headers.get("x-mesh-internal"):
        return MessageOrigin.from_header_value(raw, trust_kind=True)
    return parse_origin_header(raw)


def _system_note_from_mesh_request(request: Request) -> bool:
    """True when the mesh marked this message as SYSTEM-composed.

    ``x-system-wake`` makes the transcript record the inbound message with
    role ``system`` (a dim divider in the dashboard) instead of role
    ``user`` (the human's own bubble). Same trust rule as the origin kind:
    honoured only on the mesh-internal hop, so a direct caller can't make
    its message vanish into the de-emphasized style.
    """
    return bool(
        request.headers.get("x-system-wake")
        and request.headers.get("x-mesh-internal")
    )


def _install_protocol_version_guard(app: FastAPI) -> None:
    """Reject mesh→agent requests from a version-incompatible mesh.

    Fires only when BOTH the mesh-internal marker and a protocol-version
    header are present and the major version is incompatible — a rolling
    upgrade that left this container talking to a differently-versioned mesh.
    A request with no version header is always allowed (agent self-calls,
    reachability probes, first-party callers), so the check is non-breaking.
    ``/status`` is exempt so health/reachability polling never trips it.
    """
    from starlette.responses import JSONResponse as _StarletteJSONResponse
    from src.shared.trace import PROTOCOL_VERSION, PROTOCOL_VERSION_HEADER, protocol_compatible

    @app.middleware("http")
    async def _enforce_protocol_version(request: Request, call_next):
        if (
            request.headers.get("x-mesh-internal")
            and request.url.path != "/status"
            and not protocol_compatible(request.headers.get(PROTOCOL_VERSION_HEADER))
        ):
            return _StarletteJSONResponse(
                {
                    "detail": (
                        f"protocol version mismatch: mesh sent "
                        f"{request.headers.get(PROTOCOL_VERSION_HEADER)!r}, "
                        f"agent speaks {PROTOCOL_VERSION!r}"
                    )
                },
                status_code=426,
            )
        return await call_next(request)


def create_agent_app(loop: AgentLoop) -> FastAPI:
    """Create the FastAPI application for an agent container."""
    # M19: disable interactive API docs / OpenAPI schema by default; gate dev
    # access behind OPENLEGION_ENABLE_DOCS.
    _docs_kwargs = (
        {}
        if os.environ.get("OPENLEGION_ENABLE_DOCS", "").lower() in ("1", "true", "yes", "on")
        else {"docs_url": None, "redoc_url": None, "openapi_url": None}
    )
    app = FastAPI(title=f"OpenLegion Agent: {loop.agent_id}", **_docs_kwargs)
    _install_body_size_limit(app)
    _install_protocol_version_guard(app)
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

        Includes two MCP side-channels for the dashboard:

        * ``mcp_servers`` — per-server startup/discovery status registry
          (state, tools_count, error). Drives the per-server status dot
          and the click-to-see-error UX.
        * ``mcp_tool_to_server`` — ``{tool_name: server_name}`` mapping
          for filtering the tool list by server. Sidesteps the OpenAI
          tool definition format which has no place for per-tool source
          metadata.

        Both are omitted when no MCP client is wired to the loop.
        """
        exc = loop._excluded_tools
        alw = loop._allowed_tools if isinstance(loop._allowed_tools, frozenset) else None
        result: dict[str, Any] = {
            "agent_id": loop.agent_id,
            "role": loop.role,
            "tools": loop.tools.list_tools(exclude=exc, allowed=alw),
            "tool_definitions": loop.tools.get_tool_definitions(exclude=exc, allowed=alw),
            "tool_sources": loop.tools.get_tool_sources(exclude=exc, allowed=alw),
        }
        mcp_client = getattr(loop.tools, "_mcp_client", None)
        if mcp_client is not None:
            result["mcp_servers"] = mcp_client.list_server_statuses()
            result["mcp_tool_to_server"] = mcp_client.get_tool_to_server()
        remote_statuses = getattr(
            loop.tools, "remote_connector_statuses", None,
        )
        if remote_statuses is not None:
            statuses = remote_statuses()
            if statuses:
                result["remote_connectors"] = statuses
        return result

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
            result = await loop.tools.execute(
                name, params,
                mesh_client=loop.mesh_client,
                workspace_manager=loop.workspace,
                memory_store=loop.memory,
                agent_loop=loop,
            )
            return {"result": result}
        except ValueError as e:
            raise HTTPException(404, str(e))
        except Exception as e:
            logger.warning("invoke_tool '%s' failed: %s", name, e)
            return {"error": str(e)}

    @app.post("/heartbeat")
    async def heartbeat(msg: ChatMessage, request: Request) -> dict:
        """Execute an autonomous heartbeat — separate from chat.

        Task 2b: parse ``X-Origin`` so the heartbeat path stamps the
        contextvar with ``kind="heartbeat"`` for any tool / coordination
        call made during the heartbeat run.

        Bug 6 (codex P2 r2): read the ``x-force-llm`` header set by
        ``cli/runtime.py:heartbeat_dispatch`` when the cron job has
        ``force_llm=True``. Without this propagation the cron-side skip
        is bypassed but the loop-side skip still fires for empty
        HEARTBEAT.md, leaving pipeline-kicker agents silent.
        """
        from src.shared.trace import current_origin
        origin = _origin_from_mesh_request(request)
        force_llm = (request.headers.get("x-force-llm") or "").lower() in ("1", "true", "yes")
        token = current_origin.set(origin)
        try:
            result = await loop.execute_heartbeat(
                sanitize_for_prompt(msg.message), force_llm=force_llm,
            )
        finally:
            current_origin.reset(token)
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
        origin = _origin_from_mesh_request(request)
        task_id = request.headers.get("x-task-id") or None
        result = await loop.chat(
            sanitize_for_prompt(msg.message),
            trace_id=request.headers.get("x-trace-id"),
            origin=origin,
            task_id=task_id,
            system_note=_system_note_from_mesh_request(request),
        )
        # Round-4 forensic trace (Bug 3 still reproduces post-PR#952).
        # The operator reported turns ending with no visible chat
        # output after tool calls. INFO-level shape log makes the next
        # repro diagnosable from the agent's container log alone:
        # ``response_len=0 tool_outputs=N`` would prove the fallback
        # never fired; non-zero response_len would point to a
        # dashboard-side rendering bug instead.
        resp = result.get("response") or ""
        tool_outputs = result.get("tool_outputs") or []
        logger.info(
            "/chat EXIT task_id=%s response_len=%d tool_outputs=%d "
            "tokens_used=%s flags=%s",
            task_id, len(resp), len(tool_outputs),
            result.get("tokens_used"),
            [k for k in (
                "auth_failure", "config_error", "exception_caught",
                "tool_limit_reached", "silent_reply",
            ) if result.get(k)],
        )
        return ChatResponse(**result)

    @app.post("/chat/steer")
    async def chat_steer(msg: SteerMessage, request: Request) -> dict:
        """Inject a message into the active conversation. Does NOT acquire _chat_lock."""
        injected = await loop.inject_steer(
            sanitize_for_prompt(msg.message),
            system_note=_system_note_from_mesh_request(request),
        )
        return {"injected": injected, "agent_state": loop.state}

    @app.post("/chat/stream")
    async def chat_stream(msg: ChatMessage, request: Request) -> StreamingResponse:
        """Streaming chat. Returns SSE events for tool use and text deltas."""
        _trace_id = request.headers.get("x-trace-id")
        _origin = _origin_from_mesh_request(request)
        async def event_generator():
            # Seed the trace/origin contextvars HERE, in the pump's own
            # task context. Each ``__anext__`` below is wrapped in its own
            # task, and a task runs in a COPY of the creating context —
            # so sets made INSIDE the generator (loop.chat_stream) live
            # and die with one step's copy and are invisible to the next.
            # In production that meant every tool call in a streamed chat
            # saw ``current_origin = None``: dashboard-initiated chains
            # stored NULL origins and the entire delegate-and-subscribe
            # surface (chain completion delivery, stall nudges, failure
            # recovery wakes) silently never fired, and ``llm.py`` sent
            # no x-trace-id (no per-call traces). Seeding in THIS context
            # makes every per-step task copy inherit the values.
            from src.shared.trace import current_origin, current_trace_id
            _tid_token = current_trace_id.set(_trace_id)
            _origin_token = current_origin.set(_origin)
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
                    yield f"data: {dumps_safe(event)}\n\n"
                    next_event = asyncio.ensure_future(stream_iter.__anext__())
            finally:
                if not next_event.done():
                    next_event.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await next_event
                # Deterministically finalize the inner generator. On a
                # client disconnect it is suspended INSIDE loop.chat_stream
                # while still holding _chat_lock — without an explicit
                # aclose() the lock is only released when the GC's
                # async-gen finalizer eventually runs, and until then the
                # agent's next chat blocks on the lock. aclose() also runs
                # chat_stream's finally (session checkpoint) now, in this
                # context, instead of at an unspecified later point.
                with contextlib.suppress(Exception):
                    await stream_iter.aclose()
                # Tokens were minted in THIS context, so the resets are
                # safe for normal completion AND client-disconnect
                # cancellation (Starlette cancels the same task). The
                # suppress covers the one remaining path: a GC-driven
                # async-generator finalization can run this ``finally``
                # in a different context, where reset raises ValueError —
                # functionally irrelevant (the context dies anyway) but
                # it would surface as 'Exception ignored in <async_gen>'
                # noise during cleanup.
                with contextlib.suppress(ValueError):
                    current_origin.reset(_origin_token)
                    current_trace_id.reset(_tid_token)
        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @app.post("/chat/note")
    async def chat_note(msg: ChatMessage) -> JSONResponse:
        """Append a notification-role transcript row — no LLM turn.

        Mesh-side chain-outcome delivery target: the ChainWatcher claims
        a delivery only on this endpoint's positive ack, so the write
        path RAISES on failure (``raise_on_error=True``) — a swallowed
        disk error acked as ok would silently lose the outcome. Renders
        as the standard amber notification bubble (same row shape
        ``notify_user`` persists).
        """
        if loop.workspace is None:
            return JSONResponse(
                status_code=503,
                content={"ok": False, "error": "no workspace configured"},
            )
        text = sanitize_for_prompt(msg.message)[:_NOTE_MAX_LEN]
        if not text.strip():
            return JSONResponse(
                status_code=400,
                content={"ok": False, "error": "empty note"},
            )
        try:
            loop.workspace.append_chat_message(
                "notification", text, raise_on_error=True,
            )
        except Exception as e:
            logger.error("chat note write failed: %s", e)
            return JSONResponse(
                status_code=500,
                content={"ok": False, "error": f"transcript write failed: {e}"},
            )
        return JSONResponse(content={"ok": True})

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
        # Operator-tracked goals. GOALS.json is the structured sidecar
        # the dashboard reads; GOALS.md is the rendered markdown view.
        # Only the operator agent writes them in practice.
        "GOALS.md", "GOALS.json",
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
        # GOALS.md intentionally uncapped here — size is tool-enforced
        # (10 goals × name/note limits in manage_goals); an HTTP cap on
        # the PUT path would clash at max load with what the tool can
        # legitimately produce (~7k chars).
    }
    _DEFAULT_HEADINGS = {
        "SOUL.md": "# Identity",
        "INSTRUCTIONS.md": "# Instructions",
        "AGENTS.md": "# Agent Instructions",
        "USER.md": "# User Context",
        "MEMORY.md": "# Long-Term Memory",
        "HEARTBEAT.md": "# Heartbeat Rules",
        "INTERFACE.md": "# Interface",
        "GOALS.md": "# Goals",
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
        edits). Agent-side writes go through the update_workspace tool which
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

    @app.put("/team")
    async def update_team(request: Request) -> dict:
        """Accept an updated TEAM.md from the mesh host.

        TEAM.md is fleet-wide (not per-agent), so it's separate from the
        identity file allowlist. The mesh pushes updates here after the
        user edits it on the dashboard. PR 3 of the project→team rename
        dropped the ``PUT /project`` alias and the dual-write to
        ``PROJECT.md`` — only the canonical name is touched.
        """
        if not request.headers.get("x-mesh-internal"):
            raise HTTPException(
                403,
                "Team updates via this endpoint require X-Mesh-Internal header.",
            )
        if not loop.workspace:
            raise HTTPException(503, "Workspace not available")
        body = await request.json()
        content = body.get("content", "")
        if not isinstance(content, str):
            raise HTTPException(400, "content must be a string")
        content = sanitize_for_prompt(content)
        team_path = loop.workspace.root / "TEAM.md"
        team_path.write_text(content)
        return {"updated": True, "size": team_path.stat().st_size}

    @app.post("/learnings/feedback")
    async def record_outcome_feedback(request: Request) -> dict:
        """Accept rating feedback for one of this agent's completed tasks.

        A1 (rating → learning loop): the mesh pushes the operator's /
        user's ``rework`` / ``rejected`` feedback here; it lands in the
        corrections learnings file and rides ``get_learnings_context``
        into every future prompt. Mesh-internal only — same gate as
        ``PUT /team`` so agents can't forge corrections into their own
        (or via SSRF, a peer's) learnings.
        """
        if not request.headers.get("x-mesh-internal"):
            raise HTTPException(
                403,
                "Feedback writes via this endpoint require X-Mesh-Internal header.",
            )
        if not loop.workspace:
            raise HTTPException(503, "Workspace not available")
        body = await request.json()
        feedback = body.get("feedback", "")
        if not isinstance(feedback, str) or not feedback.strip():
            raise HTTPException(400, "feedback must be a non-empty string")
        task_id = str(body.get("task_id", ""))[:64]
        title = str(body.get("title", ""))[:200]
        outcome = str(body.get("outcome", ""))[:32] or "rework"
        original = f"Task {task_id}: {title}" if title else f"Task {task_id}"
        loop.workspace.record_correction(
            original=sanitize_for_prompt(original),
            correction=f"[{outcome}] {sanitize_for_prompt(feedback)}",
        )
        return {"recorded": True}

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
        max_tokens = body.get("max_tokens") if "max_tokens" in body else None
        max_tool_rounds = body.get("max_tool_rounds") if "max_tool_rounds" in body else None
        llm_timeout = body.get("llm_timeout_seconds") if "llm_timeout_seconds" in body else None
        internet = body.get("internet_access_enabled") if "internet_access_enabled" in body else None
        browser = body.get("browser_access_enabled") if "browser_access_enabled" in body else None
        if "model" in body and (not isinstance(model, str) or not model):
            raise HTTPException(400, "model must be a non-empty string")
        if "thinking" in body and thinking not in LLMClient.VALID_THINKING_LEVELS:
            raise HTTPException(
                400,
                f"thinking must be one of: {sorted(LLMClient.VALID_THINKING_LEVELS)}",
            )
        from src.shared.limits import MAX_OUTPUT_TOKENS_MAX, MAX_OUTPUT_TOKENS_MIN
        if "max_tokens" in body and (
            not isinstance(max_tokens, int)
            or isinstance(max_tokens, bool)
            or not (MAX_OUTPUT_TOKENS_MIN <= max_tokens <= MAX_OUTPUT_TOKENS_MAX)
        ):
            raise HTTPException(
                400,
                f"max_tokens must be an integer between "
                f"{MAX_OUTPUT_TOKENS_MIN} and {MAX_OUTPUT_TOKENS_MAX}",
            )
        if "internet_access_enabled" in body and not isinstance(internet, bool):
            raise HTTPException(
                400, "internet_access_enabled must be a boolean",
            )
        if "browser_access_enabled" in body and not isinstance(browser, bool):
            raise HTTPException(
                400, "browser_access_enabled must be a boolean",
            )
        # Per-agent operational caps — validated against the central limits
        # clamp spec (single source of truth). bool rejected (int subclass).
        from src.shared import limits as _limits
        for _k, _val in (
            ("max_tool_rounds", max_tool_rounds),
            ("llm_timeout_seconds", llm_timeout),
        ):
            if _k in body:
                _d, _lo, _hi = _limits.LIMIT_SPECS[_limits.AGENT_CONFIG_KEYS[_k]]
                if (
                    not isinstance(_val, int)
                    or isinstance(_val, bool)
                    or not (_lo <= _val <= _hi)
                ):
                    raise HTTPException(
                        400, f"{_k} must be an integer between {_lo} and {_hi}",
                    )

        updated: dict = {}
        if "model" in body:
            loop.llm.default_model = model
            updated["model"] = model
        if "thinking" in body:
            loop.llm.thinking = thinking
            updated["thinking"] = thinking
        if "max_tokens" in body:
            loop.llm.max_output_tokens = max_tokens
            updated["max_tokens"] = max_tokens
        if "max_tool_rounds" in body:
            # Honour the TASK <= CHAT invariant (mirrors AgentLoop.__init__).
            loop.TASK_MAX_TOOL_ROUNDS = min(max_tool_rounds, loop.CHAT_MAX_TOOL_ROUNDS)
            updated["max_tool_rounds"] = loop.TASK_MAX_TOOL_ROUNDS
        if "llm_timeout_seconds" in body:
            loop.llm.timeout_seconds = llm_timeout
            # Reset the pooled httpx client so the new timeout takes effect on
            # the next call (httpx fixes the timeout at client construction).
            await loop.llm.close()
            updated["llm_timeout_seconds"] = llm_timeout
        if "internet_access_enabled" in body:
            # Mesh-side push from the Operator Settings → Internet
            # access toggle. When False, hide http_request + web_search
            # from the next LLM tool surface; when True, restore them.
            loop.set_runtime_gate("internet", internet)
            updated["internet_access_enabled"] = internet
        if "browser_access_enabled" in body:
            # Mesh-side push from the Operator Settings → Browser access
            # toggle. When False, hide the browser_* tools from the next
            # LLM tool surface; when True, restore them.
            loop.set_runtime_gate("browser", browser)
            updated["browser_access_enabled"] = browser
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
        filepath = resolve_under_root(artifacts_dir, name)
        if filepath is None:
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

    # Hard ceiling on a single /files read so a caller can pull more than the
    # 500 KB default read cap (e.g. a peer-file download) without ever loading
    # an unbounded amount into memory. Larger files are paged via ``offset``.
    _MAX_FILE_READ_BYTES = 5 * 1024 * 1024  # 5 MB

    @app.get("/files/{path:path}")
    async def read_data_file(
        path: str, offset: int = 0, max_bytes: int = 0,
    ) -> dict:
        """Read any file from /data. Text returned as-is; binary base64-encoded.

        ``offset`` seeks into the file and ``max_bytes`` caps how much is
        returned in one call, so a caller can page through a file larger than
        the default 500 KB read cap. ``max_bytes`` defaults to that legacy cap
        (back-compat for existing dashboard/agent readers) and is clamped to a
        5 MB hard ceiling per request. ``next_offset`` is where the next page
        starts; ``truncated`` is True while more bytes remain.
        """
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
        offset = max(0, offset)
        limit = max_bytes if max_bytes > 0 else _MAX_READ
        limit = min(limit, _MAX_FILE_READ_BYTES)
        # Seek to offset and read at most ``limit`` bytes — never loads the
        # whole file into memory.
        with safe.open("rb") as fh:
            if offset:
                fh.seek(offset)
            raw = fh.read(limit)
        next_offset = offset + len(raw)
        truncated = next_offset < size
        try:
            content = raw.decode("utf-8")
            encoding = "utf-8"
        except UnicodeDecodeError:
            # A chunk boundary may split a multibyte char; base64 keeps the
            # page byte-exact so a paged download reassembles losslessly.
            content = base64.b64encode(raw).decode("ascii")
            encoding = "base64"
        return {"path": path, "content": content, "size": size,
                "mime_type": mime, "encoding": encoding,
                "offset": offset, "next_offset": next_offset,
                "truncated": truncated}

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
        trace_id = request.headers.get("x-trace-id", "")

        artifacts_dir = Path(loop.workspace.root) / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        target = resolve_under_root(artifacts_dir, name)
        if target is None:
            raise HTTPException(400, "Path traversal not allowed")
        target.parent.mkdir(parents=True, exist_ok=True)

        # Collision avoidance — never overwrite.  ``_open_partial_exclusive``
        # atomically reserves a ``{name}-N.partial`` slot via ``O_CREAT|O_EXCL``,
        # so two concurrent ingest requests for the same base name can never
        # claim the same suffix (the loser retries with the next number).
        final, partial, fd = _open_partial_exclusive(target)

        max_bytes = _MAX_ARTIFACT_INGEST_BYTES
        bytes_written = 0
        try:
            with os.fdopen(fd, "wb") as fh:
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
        rel_name = str(final.relative_to(artifacts_dir.resolve()))
        mime = mimetypes.guess_type(rel_name)[0] or "application/octet-stream"
        logger.info(
            "Ingested artifact %s (%d bytes)", rel_name, bytes_written,
            extra={"trace_id": trace_id},
        )
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
        filepath = resolve_under_root(artifacts_dir, name)
        if filepath is None:
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
    text = dumps_safe(payload)
    return text[:max_len] + "..." if len(text) > max_len else text


def _open_partial_exclusive(target: Path) -> tuple[Path, Path, int]:
    """Atomically reserve a ``.partial`` slot and return ``(final, partial, fd)``.

    Collision-avoidance WITHOUT a TOCTOU window: tries the base name first,
    then ``-1``, ``-2``, …, using ``os.O_CREAT | os.O_EXCL`` each time to
    guarantee only one caller can own each candidate. Two concurrent
    ingests of the same base name land on distinct final names —
    `report.pdf` and `report-1.pdf` — with no overwrite race.

    The caller receives an OS-level file descriptor open for writing; wrap
    in ``os.fdopen(fd, "wb")`` to get a file-like.  The ``partial`` path
    will be renamed to ``final`` on successful completion.
    """
    stem = target.stem
    suffix = target.suffix
    parent = target.parent
    for n in range(0, 1000):
        if n == 0:
            final = target
        else:
            final = parent / f"{stem}-{n}{suffix}"
        # Write to a .partial sidecar first; atomic replace on success.
        # Exclusive create on the .partial owns the slot — if a peer is
        # mid-stream for the same name, its .partial blocks ours and we
        # try the next suffix.
        partial = final.with_suffix(final.suffix + ".partial")
        try:
            fd = os.open(
                partial,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o644,
            )
        except FileExistsError:
            continue
        # If final already exists (committed by a peer), the .partial we
        # just created is pointless — clean it up and try the next suffix.
        if final.exists():
            os.close(fd)
            with contextlib.suppress(FileNotFoundError, OSError):
                partial.unlink()
            continue
        return final, partial, fd
    raise RuntimeError(f"Too many collisions disambiguating {target}")


def _log_task_exception(task: asyncio.Task) -> None:
    """Log unhandled exceptions from fire-and-forget tasks."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc:
        logger.error(f"Background task failed: {exc}", exc_info=exc)


async def run_maintenance_loop(loop: AgentLoop) -> None:
    """Periodic background memory-maintenance pass for an agent's lifetime.

    Launched from the agent process lifespan (``__main__``). The first run is
    delayed so boot + the first user turn settle; thereafter it ticks
    periodically. ``loop.run_maintenance`` is internally idle-guarded and
    6h-gated, so a frequent tick is cheap and never races a live turn.
    """
    delay = _MAINTENANCE_INITIAL_DELAY_S
    while True:
        await asyncio.sleep(delay)
        delay = _MAINTENANCE_TICK_S
        try:
            await loop.run_maintenance()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("maintenance pass failed: %s", e)
