"""End-to-end test: agent chat mode with built-in tools.

Tests the chat data path:
  CLI chat -> agent container -> LLM -> exec/file tools -> response

Requires:
  - Docker running and accessible
  - OPENAI_API_KEY environment variable set (or OPENLEGION_CRED_OPENAI_API_KEY)
  - openlegion-agent:latest Docker image built

Skip conditions are applied so the unit test suite still passes without these.
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

MESH_PORT = 8436


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
def chat_stack(tmp_path_factory):
    """Start mesh + a general-purpose agent for chat testing."""
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

    tmp_dir = tmp_path_factory.mktemp("e2e_chat")
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

    # Start agent with no custom skills (builtins only)
    url = cm.start_agent(
        agent_id="chatbot",
        role="assistant",
        skills_dir="",
        system_prompt=(
            "You are a helpful assistant with access to shell commands, "
            "file operations, and HTTP tools. Use them when asked."
        ),
        model="openai/gpt-4o-mini",
    )
    router.register_agent("chatbot", url)

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
        try:
            container = cm.containers["chatbot"]["container"]
            container.reload()
            logs = container.logs(tail=50).decode()
            print(f"\n=== Agent container logs ===\n{logs}\n===")
        except Exception:
            pass
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
def test_chat_simple_response(chat_stack):
    """Agent responds to a simple question without tools."""
    url = chat_stack["agent_url"]
    r = httpx.post(f"{url}/chat", json={"message": "What is 2+2?"}, timeout=30)
    assert r.status_code == 200
    data = r.json()
    assert "4" in data["response"]
    assert data["tokens_used"] > 0


@skip_no_docker
@skip_no_key
def test_chat_exec_tool(chat_stack):
    """Agent uses exec tool to run a command when asked."""
    url = chat_stack["agent_url"]

    # Reset conversation first
    httpx.post(f"{url}/chat/reset", timeout=5)

    r = httpx.post(
        f"{url}/chat",
        json={"message": "Run the command 'echo hello_openlegion' and tell me the output."},
        timeout=60,
    )
    assert r.status_code == 200
    data = r.json()
    assert "hello_openlegion" in data["response"]
    assert data["tokens_used"] > 0


@skip_no_docker
@skip_no_key
def test_chat_file_tools(chat_stack):
    """Agent uses file tools to write and read a file."""
    url = chat_stack["agent_url"]
    httpx.post(f"{url}/chat/reset", timeout=5)

    msg = (
        "Write the text 'test content 42' to the file /data/test_output.txt, "
        "then read it back and confirm the contents."
    )
    r = httpx.post(f"{url}/chat", json={"message": msg}, timeout=60)
    assert r.status_code == 200
    data = r.json()
    assert "test content 42" in data["response"] or "42" in data["response"]


@skip_no_docker
@skip_no_key
def test_chat_has_builtin_tools(chat_stack):
    """Agent container has built-in tools discovered."""
    url = chat_stack["agent_url"]
    r = httpx.get(f"{url}/capabilities", timeout=5)
    assert r.status_code == 200
    caps = r.json()
    assert "exec" in caps["skills"]
    assert "read_file" in caps["skills"]
    assert "write_file" in caps["skills"]
    assert "http_request" in caps["skills"]
    assert "browser_navigate" in caps["skills"]


@skip_no_docker
@skip_no_key
def test_chat_reset(chat_stack):
    """Chat reset clears conversation history."""
    url = chat_stack["agent_url"]

    httpx.post(f"{url}/chat", json={"message": "Remember: the code is 'alpha'."}, timeout=30)
    r = httpx.post(f"{url}/chat/reset", timeout=5)
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
