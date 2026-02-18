"""End-to-end test: persistent memory across chat sessions.

Core acceptance criterion for Phase 1:
  Agent remembers a fact from session 1 when asked in session 2.

The /data/workspace/ directory persists on the Docker volume across
chat resets (which only clear in-memory message history).

Requires:
  - Docker running and accessible
  - OPENAI_API_KEY or OPENLEGION_CRED_OPENAI_API_KEY set
  - openlegion-agent:latest Docker image built

Skipped automatically if prerequisites are missing.
"""

from __future__ import annotations

import os
import threading
import time

import httpx
import pytest
import uvicorn
from dotenv import load_dotenv

load_dotenv()

MESH_PORT = 8437


def _docker_available() -> bool:
    try:
        import docker

        docker.from_env().ping()
        return True
    except Exception:
        return False


def _has_llm_key() -> bool:
    return bool(
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("OPENLEGION_CRED_OPENAI_API_KEY")
    )


skip_no_docker = pytest.mark.skipif(not _docker_available(), reason="Docker not available")
skip_no_key = pytest.mark.skipif(not _has_llm_key(), reason="OPENAI_API_KEY not set")


@pytest.fixture(scope="module")
def memory_stack(tmp_path_factory):
    """Start mesh + agent for memory E2E tests."""
    if not _docker_available() or not _has_llm_key():
        pytest.skip("Docker or API key not available")

    if not os.environ.get("OPENLEGION_CRED_OPENAI_API_KEY"):
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        if openai_key:
            os.environ["OPENLEGION_CRED_OPENAI_API_KEY"] = openai_key

    from src.host.containers import ContainerManager
    from src.host.credentials import CredentialVault
    from src.host.mesh import Blackboard, MessageRouter, PubSub
    from src.host.permissions import PermissionMatrix
    from src.host.server import create_mesh_app

    tmp_dir = tmp_path_factory.mktemp("e2e_memory")
    bb = Blackboard(db_path=str(tmp_dir / "blackboard.db"))
    pubsub = PubSub()
    perms = PermissionMatrix()
    vault = CredentialVault()
    cm = ContainerManager(mesh_host_port=MESH_PORT, use_host_network=True)
    router = MessageRouter(perms, {})

    app = create_mesh_app(bb, pubsub, router, perms, vault)

    config = uvicorn.Config(app, host="0.0.0.0", port=MESH_PORT, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    for _ in range(20):
        try:
            r = httpx.get(f"http://localhost:{MESH_PORT}/mesh/agents", timeout=2)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.5)

    url = cm.start_agent(
        agent_id="membot",
        role="assistant",
        skills_dir="",
        system_prompt=(
            "You are a helpful assistant with persistent memory. "
            "When the user tells you a fact, use the memory_save tool to remember it. "
            "When asked about something you saved, refer to your memory."
        ),
        model="openai/gpt-4o-mini",
    )
    router.register_agent("membot", url)

    agent_ready = False
    for _ in range(60):
        try:
            r = httpx.get(f"{url}/status", timeout=2)
            if r.status_code == 200:
                agent_ready = True
                break
        except Exception:
            pass
        time.sleep(1)

    if not agent_ready:
        cm.stop_all()
        server.should_exit = True
        pytest.fail("Agent container failed to become healthy")

    yield {"agent_url": url, "cm": cm}

    cm.stop_all()
    server.should_exit = True
    bb.close()
    thread.join(timeout=5)


@skip_no_docker
@skip_no_key
def test_memory_persists_across_sessions(memory_stack):
    """Session 1: tell agent a fact. Session 2: ask about it."""
    url = memory_stack["agent_url"]

    # Reset to start fresh
    httpx.post(f"{url}/chat/reset", timeout=5)

    # Session 1: tell the agent a unique fact
    r = httpx.post(
        f"{url}/chat",
        json={
            "message": (
                "Remember this important fact: my cat's name is Whiskerino. "
                "Please save this to your memory."
            ),
        },
        timeout=60,
    )
    assert r.status_code == 200
    data = r.json()
    assert data["tokens_used"] > 0

    # Reset chat = simulate new session (workspace files persist on disk)
    r = httpx.post(f"{url}/chat/reset", timeout=5)
    assert r.status_code == 200

    # Session 2: ask about the fact
    r = httpx.post(
        f"{url}/chat",
        json={"message": "What is my cat's name?"},
        timeout=60,
    )
    assert r.status_code == 200
    data = r.json()
    assert "Whiskerino" in data["response"], (
        f"Agent did not recall the fact. Response: {data['response'][:200]}"
    )


@skip_no_docker
@skip_no_key
def test_workspace_files_exist(memory_stack):
    """Verify workspace scaffold was created inside the container."""
    url = memory_stack["agent_url"]

    # Use exec tool to check workspace files
    httpx.post(f"{url}/chat/reset", timeout=5)
    r = httpx.post(
        f"{url}/chat",
        json={"message": "List the files in /data/workspace/ using the exec tool with 'ls -la /data/workspace/'"},
        timeout=60,
    )
    assert r.status_code == 200
    data = r.json()
    response_lower = data["response"].lower()
    assert "agents.md" in response_lower or "memory" in response_lower, (
        f"Workspace files not found. Response: {data['response'][:300]}"
    )


@skip_no_docker
@skip_no_key
def test_memory_search_tool_works(memory_stack):
    """Agent can use memory_search to find saved information."""
    url = memory_stack["agent_url"]

    httpx.post(f"{url}/chat/reset", timeout=5)

    # First, save something
    httpx.post(
        f"{url}/chat",
        json={"message": "Save this fact: the project deadline is March 15th 2026."},
        timeout=60,
    )

    # Then search for it
    r = httpx.post(
        f"{url}/chat",
        json={"message": "Search your memory for information about the project deadline."},
        timeout=60,
    )
    assert r.status_code == 200
    data = r.json()
    assert "March" in data["response"] or "deadline" in data["response"].lower(), (
        f"Memory search failed. Response: {data['response'][:200]}"
    )


@skip_no_docker
@skip_no_key
def test_builtin_memory_tools_available(memory_stack):
    """Agent has memory_search and memory_save in capabilities."""
    url = memory_stack["agent_url"]
    r = httpx.get(f"{url}/capabilities", timeout=5)
    assert r.status_code == 200
    caps = r.json()
    assert "memory_search" in caps["skills"]
    assert "memory_save" in caps["skills"]
