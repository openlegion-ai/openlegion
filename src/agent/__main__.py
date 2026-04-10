"""Agent container entry point.

Reads configuration from environment variables, wires all components,
and starts the FastAPI server.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
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
from src.agent.server import create_agent_app
from src.agent.skills import SkillRegistry
from src.agent.workspace import WorkspaceManager
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
    skills_dir = os.environ.get("SKILLS_DIR", "/app/skills")

    llm_model = os.environ.get("LLM_MODEL", "openai/gpt-4o-mini")
    embedding_model = os.environ.get("EMBEDDING_MODEL", "")
    thinking = os.environ.get("THINKING", "off")
    llm = LLMClient(
        mesh_url=mesh_url, agent_id=agent_id,
        default_model=llm_model, embedding_model=embedding_model,
        thinking=thinking,
    )
    project_name = os.environ.get("PROJECT_NAME", "")
    mesh_client = MeshClient(
        mesh_url=mesh_url, agent_id=agent_id, project_name=project_name or None,
    )
    embed_fn = llm.embed if embedding_model and embedding_model.lower() != "none" else None
    memory = MemoryStore(db_path=f"/data/{agent_id}.db", embed_fn=embed_fn)

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

    skills = SkillRegistry(skills_dir=skills_dir, mcp_client=mcp_client)

    # Write skill authoring reference guide if not present
    from src.agent.builtins.skill_tool import _ensure_skill_guide
    _ensure_skill_guide()

    workspace = WorkspaceManager(
        workspace_dir="/data/workspace",
        initial_instructions=initial_instructions,
        initial_soul=initial_soul,
        initial_heartbeat=initial_heartbeat,
        initial_interface=initial_interface,
    )

    # Copy host-mounted PROJECT.md into workspace (mounted at /app to avoid
    # Docker creating /data/workspace as root and breaking permissions)
    host_project = Path("/app/PROJECT.md")
    ws_project = Path("/data/workspace/PROJECT.md")
    if host_project.exists() and host_project.is_file():
        try:
            ws_project.write_text(host_project.read_text())
        except OSError:
            logger.debug("Could not copy PROJECT.md into workspace")

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
        skills=skills,
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
                builtin_names = set(skills.list_skills())
                await mcp_client.start(server_configs, builtin_names=builtin_names)
                # Re-register MCP tools now that servers are running
                skills._register_mcp_tools()
                logger.info(f"MCP servers started for agent '{agent_id}'")
            except Exception as e:
                logger.error(f"Failed to start MCP servers: {e}")

        registered = False
        for attempt in range(1, _MAX_REGISTRATION_ATTEMPTS + 1):
            try:
                await mesh_client.register(
                    capabilities=skills.list_skills(),
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
                system_md = generate_system_md(
                    info, agent_id, is_standalone=mesh_client.is_standalone,
                )
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
                    _resume_trace_id = f"tr_{secrets.token_hex(6)}"
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

        yield
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
