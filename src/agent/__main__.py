"""Agent container entry point.

Reads configuration from environment variables, wires all components,
and starts the FastAPI server.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager

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


def main() -> None:
    agent_id = os.environ["AGENT_ID"]
    role = os.environ["AGENT_ROLE"]
    mesh_url = os.environ["MESH_URL"]
    skills_dir = os.environ.get("SKILLS_DIR", "/app/skills")
    system_prompt = os.environ.get("SYSTEM_PROMPT", "")

    llm_model = os.environ.get("LLM_MODEL", "openai/gpt-4o-mini")
    llm = LLMClient(mesh_url=mesh_url, agent_id=agent_id, default_model=llm_model)
    mesh_client = MeshClient(mesh_url=mesh_url, agent_id=agent_id)
    memory = MemoryStore(db_path=f"/data/{agent_id}.db", embed_fn=llm.embed)
    skills = SkillRegistry(skills_dir=skills_dir)
    workspace = WorkspaceManager(workspace_dir="/data/workspace")
    context_mgr = ContextManager(max_tokens=128_000, llm=llm, workspace=workspace)

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
        await mesh_client.register(capabilities=skills.list_skills(), port=agent_port)
        logger.info(f"Agent '{agent_id}' registered with mesh")
        yield
        if loop.state == "working":
            loop._cancel_requested = True
            handle = loop._current_task_handle
            if handle and not handle.done():
                try:
                    await asyncio.wait_for(handle, timeout=5.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
        await llm.close()
        memory.close()
        logger.info(f"Agent '{agent_id}' shut down")

    app = create_agent_app(loop)
    app.router.lifespan_context = lifespan

    uvicorn.run(app, host="0.0.0.0", port=agent_port)


if __name__ == "__main__":
    main()
