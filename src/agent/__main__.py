"""Agent container entry point.

Reads configuration from environment variables, wires all components,
and starts the FastAPI server.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI

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
    system_prompt = os.environ.get("SYSTEM_PROMPT", "")

    llm_model = os.environ.get("LLM_MODEL", "openai/gpt-4o-mini")
    llm = LLMClient(mesh_url=mesh_url, agent_id=agent_id, default_model=llm_model)
    mesh_client = MeshClient(mesh_url=mesh_url, agent_id=agent_id)
    memory = MemoryStore(db_path=f"/data/{agent_id}.db", embed_fn=llm.embed)

    # MCP server support — parse config from environment
    mcp_client = None
    mcp_servers_json = os.environ.get("MCP_SERVERS", "")
    if mcp_servers_json:
        from src.agent.mcp_client import MCPClient
        mcp_client = MCPClient()
        logger.info(f"MCP servers configured for agent '{agent_id}'")

    skills = SkillRegistry(skills_dir=skills_dir, mcp_client=mcp_client)
    workspace = WorkspaceManager(workspace_dir="/data/workspace")

    # Copy host-mounted PROJECT.md into workspace (mounted at /app to avoid
    # Docker creating /data/workspace as root and breaking permissions)
    host_project = Path("/app/PROJECT.md")
    ws_project = Path("/data/workspace/PROJECT.md")
    if host_project.exists() and host_project.is_file():
        try:
            ws_project.write_text(host_project.read_text())
        except OSError:
            logger.debug("Could not copy PROJECT.md into workspace")

    context_mgr = ContextManager(max_tokens=128_000, llm=llm, workspace=workspace, memory=memory)

    loop = AgentLoop(
        agent_id=agent_id,
        role=role,
        memory=memory,
        skills=skills,
        llm=llm,
        mesh_client=mesh_client,
        system_prompt=system_prompt,
        workspace=workspace,
        context_manager=context_mgr,
    )

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
        yield
        if loop.state == "working":
            loop._cancel_requested = True
            handle = loop._current_task_handle
            if handle and not handle.done():
                try:
                    await asyncio.wait_for(handle, timeout=5.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
        if mcp_client:
            await mcp_client.stop()
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
