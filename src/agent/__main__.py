"""Agent container entry point.

Reads configuration from environment variables, wires all components,
and starts the FastAPI server.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager, suppress
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from src.agent.builtins.http_tool import close_client
from src.agent.context import ContextManager
from src.agent.llm import LLMClient
from src.agent.loop import AgentLoop
from src.agent.memory import MemoryStore
from src.agent.mesh_client import MeshClient
from src.agent.server import create_agent_app, run_maintenance_loop
from src.agent.tools import ToolRegistry
from src.agent.workspace import WorkspaceManager
from src.shared import limits
from src.shared.trace import new_trace_id
from src.shared.utils import setup_logging

logger = setup_logging("agent.main")

_MAX_REGISTRATION_ATTEMPTS = 5
_REGISTRATION_BACKOFF_SECONDS = 1
_REGISTRATION_TIMEOUT_SECONDS = 5


def main() -> None:
    for var in ("AGENT_ID", "AGENT_ROLE", "MESH_URL"):
        if var not in os.environ:
            logger.error(f"Missing required environment variable: {var}")
            sys.exit(1)

    agent_id = os.environ["AGENT_ID"]
    role = os.environ["AGENT_ROLE"]
    mesh_url = os.environ["MESH_URL"]

    # Phase 4 log correlation: stamp this process's identity so every log line
    # in the container is attributable. The structured formatter also falls
    # back to the AGENT_ID env var for per-request contexts that never inherit
    # this set(), but setting it here covers code that reads the contextvar.
    from src.shared.trace import current_agent_id
    current_agent_id.set(agent_id)
    tools_dir = os.environ.get("TOOLS_DIR", "/app/tools")

    llm_model = os.environ.get("LLM_MODEL", "openai/gpt-4o-mini")
    embedding_model = os.environ.get("EMBEDDING_MODEL", "")
    try:
        embedding_dim = int(os.environ.get("EMBEDDING_DIM", "1536"))
    except ValueError:
        logger.warning("Invalid EMBEDDING_DIM, using default 1536")
        embedding_dim = 1536
    thinking = os.environ.get("THINKING", "off")
    try:
        max_output_tokens = int(os.environ.get("LLM_MAX_TOKENS", "16384"))
    except ValueError:
        logger.warning("Invalid LLM_MAX_TOKENS, using default 16384")
        max_output_tokens = 16384
    max_output_tokens = max(
        limits.MAX_OUTPUT_TOKENS_MIN,
        min(max_output_tokens, limits.MAX_OUTPUT_TOKENS_MAX),
    )
    llm = LLMClient(
        mesh_url=mesh_url, agent_id=agent_id,
        default_model=llm_model, embedding_model=embedding_model,
        thinking=thinking, max_output_tokens=max_output_tokens,
    )
    # Every worker is scoped: its real team, or its own agent id as a
    # private team-of-one namespace (ratified decision #5). The env is set
    # by the boot path (cli/runtime.py); the fallback here covers container
    # starts that predate it (mesh-side create/template paths) so a worker
    # can never run unscoped. Only the operator runs with team_name=None.
    team_name = os.environ.get("TEAM_NAME", "")
    if not team_name and agent_id != "operator":
        team_name = agent_id
    mesh_client = MeshClient(
        mesh_url=mesh_url, agent_id=agent_id, team_name=team_name or None,
    )
    embed_fn = llm.embed if embedding_model and embedding_model.lower() != "none" else None
    memory = MemoryStore(
        db_path=f"/data/{agent_id}.db", embed_fn=embed_fn, embedding_dim=embedding_dim,
    )

    # MCP server support — parse config from environment
    mcp_client = None
    mcp_servers_json = os.environ.get("MCP_SERVERS", "")
    if mcp_servers_json:
        from src.agent.mcp_client import MCPClient
        mcp_client = MCPClient()
        logger.info(f"MCP servers configured for agent '{agent_id}'")

    initial_instructions = os.environ.get("INITIAL_INSTRUCTIONS", "")
    initial_soul = os.environ.get("INITIAL_SOUL", "")
    initial_heartbeat = os.environ.get("INITIAL_HEARTBEAT", "")
    initial_interface = os.environ.get("INITIAL_INTERFACE", "")

    tools = ToolRegistry(tools_dir=tools_dir, mcp_client=mcp_client)

    # Write tool authoring reference guide if not present
    from src.agent.builtins.tool_authoring import _ensure_tool_guide
    _ensure_tool_guide()

    workspace = WorkspaceManager(
        workspace_dir="/data/workspace",
        initial_instructions=initial_instructions,
        initial_soul=initial_soul,
        initial_heartbeat=initial_heartbeat,
        initial_interface=initial_interface,
    )

    # PR-L' — first-message greeting seeded into the chat transcript.
    # Fires only on the very first boot of a fresh agent (gated on a
    # sentinel file that lives in the persistent /data volume so it
    # survives container restarts AND chat resets). The greeting is
    # tagged with ``_origin == "bootstrap_greeting"`` so the LLM
    # context layer can distinguish it from genuine assistant output.
    initial_greeting = os.environ.get("INITIAL_GREETING", "")
    if initial_greeting:
        try:
            workspace.seed_bootstrap_greeting(initial_greeting)
        except Exception as e:
            logger.debug("Greeting seed skipped: %s", e)

    # Copy host-mounted TEAM.md into the workspace. Mounted at /app to
    # avoid Docker creating /data/workspace as root and breaking
    # permissions. Only the canonical name is written; PR 3 of the
    # The project→team rename dropped the legacy ``PROJECT.md`` write but
    # the workspace bootstrap retains a read fallback for any
    # pre-migration files left behind.
    host_team = Path("/app/TEAM.md")
    if host_team.exists() and host_team.is_file():
        try:
            content = host_team.read_text()
            Path("/data/workspace/TEAM.md").write_text(content)
        except OSError:
            logger.debug("Could not copy TEAM.md into workspace")

    async def _notify_memory_update() -> None:
        await mesh_client.notify_user(f"[{agent_id}] MEMORY.md updated during context compaction.")

    context_mgr = ContextManager(
        llm=llm, workspace=workspace, memory=memory, model=llm_model,
        on_memory_update=_notify_memory_update,
    )

    allowed_tools_env = os.environ.get("ALLOWED_TOOLS", "")
    allowed_tools: frozenset[str] | None = None
    if allowed_tools_env:
        allowed_tools = frozenset(t.strip() for t in allowed_tools_env.split(",") if t.strip())

    loop = AgentLoop(
        agent_id=agent_id,
        role=role,
        memory=memory,
        tools=tools,
        llm=llm,
        mesh_client=mesh_client,
        workspace=workspace,
        context_manager=context_mgr,
        allowed_tools=allowed_tools,
    )

    from src.agent.builtins.subagent_tool import register_parent_llm
    register_parent_llm(agent_id, llm)

    agent_port = int(os.environ.get("AGENT_PORT", "8400"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Start MCP servers if configured
        if mcp_client and mcp_servers_json:
            try:
                server_configs = json.loads(mcp_servers_json)
                builtin_names = set(tools.list_tools())
                await mcp_client.start(server_configs, builtin_names=builtin_names)
                # Re-register MCP tools now that servers are running
                tools._register_mcp_tools()
                logger.info(f"MCP servers started for agent '{agent_id}'")
            except Exception as e:
                logger.error(f"Failed to start MCP servers: {e}")

        # Remote (http) connectors: fetch sanitized tool schemas from
        # the mesh gateway and register them AFTER stdio MCP so the
        # conflict-prefix check sees both namespaces. Calls route back
        # through the mesh per-call; no token enters this container.
        # Degrades to no remote tools (warn) — the mesh being briefly
        # unreachable must not crash-loop the agent.
        try:
            remote_connectors = await mesh_client.list_connector_tools()
            if remote_connectors:
                tools.register_remote_tools(remote_connectors)
                logger.info(
                    "Registered remote connector tools for agent '%s': %s",
                    agent_id,
                    ", ".join(sorted(remote_connectors)),
                )
        except Exception as e:
            logger.warning(
                "Remote connector discovery failed (%s); starting "
                "without remote tools — restart the agent to retry", e,
            )

        registered = False
        for attempt in range(1, _MAX_REGISTRATION_ATTEMPTS + 1):
            try:
                await mesh_client.register(
                    capabilities=tools.list_tools(),
                    port=agent_port,
                    timeout=_REGISTRATION_TIMEOUT_SECONDS,
                )
                logger.info(f"Agent '{agent_id}' registered with mesh")
                registered = True
                break
            except Exception as e:
                if attempt == _MAX_REGISTRATION_ATTEMPTS:
                    logger.error(f"Failed to register with mesh after {attempt} attempts: {e}")
                else:
                    logger.debug(f"Mesh registration attempt {attempt} failed, retrying...")
                    await asyncio.sleep(_REGISTRATION_BACKOFF_SECONDS)
        if not registered:
            logger.warning(f"Agent '{agent_id}' started without mesh registration")

        # Generate SYSTEM.md from live system state
        if registered:
            try:
                from src.agent.workspace import generate_system_md
                info = await mesh_client.introspect("all")
                system_md = generate_system_md(info, agent_id)
                Path("/data/workspace").mkdir(parents=True, exist_ok=True)
                Path("/data/workspace/SYSTEM.md").write_text(system_md)
                logger.info(f"Generated SYSTEM.md for '{agent_id}'")
            except Exception as e:
                logger.debug(f"Could not generate SYSTEM.md: {e}")

        # Auto-resume: check for task checkpoint and restart the task
        if loop.memory:
            try:
                cp = await loop.memory._run_db(loop.memory.load_task_checkpoint)
                if cp:
                    from src.shared.types import TaskAssignment
                    _resume_trace_id = new_trace_id()
                    assignment = TaskAssignment.model_validate_json(cp["assignment_json"])
                    logger.info(
                        "Auto-resuming task %s from checkpoint (iteration %d) trace=%s",
                        cp["task_id"], cp["iteration"], _resume_trace_id,
                    )

                    # Set state BEFORE yield (before server accepts requests).
                    # This prevents a race where an incoming POST /task sees
                    # state="idle" and is accepted before the auto-resume
                    # coroutine starts executing.
                    loop.state = "working"
                    loop.current_task = assignment.task_id

                    def _log_auto_resume_exception(t: asyncio.Task) -> None:
                        if not t.cancelled() and t.exception():
                            logger.error("Auto-resume task failed: %s", t.exception())

                    async def _auto_resume() -> None:
                        try:
                            await loop.execute_task(
                                assignment, trace_id=_resume_trace_id,
                            )
                        except asyncio.CancelledError:
                            loop.state = "idle"
                            loop.current_task = None
                        except Exception:
                            # If execute_task raises before its own try/except
                            # (e.g. during checkpoint restore or context build),
                            # reset state so the agent doesn't stay stuck on
                            # "working" permanently.
                            loop.state = "idle"
                            loop.current_task = None
                            raise  # re-raise so _log_auto_resume_exception logs it

                    _resume_task = asyncio.create_task(_auto_resume())
                    _resume_task.add_done_callback(_log_auto_resume_exception)
                    loop._current_task_handle = _resume_task
            except Exception as e:
                # Reset state if we set it to "working" but failed before
                # launching the resume task.
                if loop.state == "working" and loop._current_task_handle is None:
                    loop.state = "idle"
                    loop.current_task = None
                logger.warning("Auto-resume check failed: %s", e)

        # Background memory-maintenance pass (consolidation + salience decay),
        # off the live turn. Internally idle-guarded + 6h-gated.
        _maintenance_task = asyncio.create_task(run_maintenance_loop(loop))

        yield

        _maintenance_task.cancel()
        with suppress(asyncio.CancelledError):
            await _maintenance_task
        if loop.state == "working":
            loop._cancel_requested = True
            handle = loop._current_task_handle
            if handle and not handle.done():
                try:
                    await asyncio.wait_for(handle, timeout=5.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    # Force-cancel the stuck task to prevent asyncio task leak
                    if not handle.done():
                        handle.cancel()
                        with suppress(asyncio.TimeoutError, asyncio.CancelledError):
                            await asyncio.wait_for(handle, timeout=2.0)
        await close_client()
        if mcp_client:
            await mcp_client.stop()
        await mesh_client.close()
        await llm.close()
        memory.close()
        logger.info(f"Agent '{agent_id}' shut down")

    app = create_agent_app(loop)
    app.router.lifespan_context = lifespan

    uvicorn.run(app, host="0.0.0.0", port=agent_port)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Agent failed to start: {e}")
        sys.exit(1)
