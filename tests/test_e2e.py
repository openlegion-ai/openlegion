"""End-to-end test: mesh server + Docker agent container + real LLM.

Tests the full data path:
  webhook trigger -> orchestrator -> agent container -> LLM proxy -> skills -> result

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

MESH_PORT = 8435


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
def e2e_stack(tmp_path_factory):
    """Start the full OpenLegion stack: mesh server + research agent container."""
    if not _docker_available() or not _has_llm_key():
        pytest.skip("Docker or OPENAI_API_KEY not available")

    # Ensure the credential vault env var is set from the standard env var
    if not os.environ.get("OPENLEGION_CRED_OPENAI_API_KEY"):
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        if openai_key:
            os.environ["OPENLEGION_CRED_OPENAI_API_KEY"] = openai_key

    from src.channels.webhook import create_webhook_router
    from src.host.containers import ContainerManager
    from src.host.credentials import CredentialVault
    from src.host.mesh import Blackboard, MessageRouter, PubSub
    from src.host.orchestrator import Orchestrator
    from src.host.permissions import PermissionMatrix
    from src.host.server import create_mesh_app

    tmp_dir = tmp_path_factory.mktemp("e2e")
    bb = Blackboard(db_path=str(tmp_dir / "blackboard.db"))
    pubsub = PubSub()
    perms = PermissionMatrix()
    vault = CredentialVault()
    cm = ContainerManager(mesh_host_port=MESH_PORT, use_host_network=True)
    router = MessageRouter(perms, {})

    orch = Orchestrator(
        mesh_url=f"http://localhost:{MESH_PORT}",
        blackboard=bb,
        pubsub=pubsub,
        container_manager=cm,
    )

    # Create mesh app and start it BEFORE the agent container,
    # because the agent registers with the mesh during its startup.
    app = create_mesh_app(bb, pubsub, router, perms, vault)
    app.include_router(create_webhook_router(orch))

    config = uvicorn.Config(app, host="0.0.0.0", port=MESH_PORT, log_level="info")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for mesh to be ready
    mesh_ready = False
    for _ in range(20):
        try:
            r = httpx.get(f"http://localhost:{MESH_PORT}/mesh/agents", timeout=2)
            if r.status_code == 200:
                mesh_ready = True
                break
        except (httpx.ConnectError, httpx.TimeoutException):
            pass
        time.sleep(0.5)

    if not mesh_ready:
        server.should_exit = True
        pytest.fail("Mesh server failed to start")

    # Now start the agent container (mesh is ready to accept registration)
    skills_dir = os.path.abspath("skills/research")
    url = cm.start_agent(
        agent_id="research",
        role="research",
        skills_dir=skills_dir,
        system_prompt=(
            "You are a research specialist. Your job is to gather comprehensive "
            "information about companies using available tools. "
            "Focus on: company size, revenue, tech stack, recent news, key contacts. "
            "When done, return your findings as JSON with 'result' and 'promote' keys."
        ),
        model="openai/gpt-4o-mini",
    )
    router.register_agent("research", url)

    # Wait for agent container to become healthy
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
        # Check container logs for debugging
        try:
            container = cm.containers["research"]["container"]
            container.reload()
            logs = container.logs(tail=50).decode()
            print(f"\n=== Agent container logs ===\n{logs}\n===")
        except Exception:
            pass
        cm.stop_all()
        server.should_exit = True
        pytest.fail("Agent container failed to become healthy within 60s")

    yield {
        "mesh_url": f"http://localhost:{MESH_PORT}",
        "agent_url": url,
        "blackboard": bb,
        "container_manager": cm,
        "orchestrator": orch,
    }

    # Cleanup
    cm.stop_all()
    server.should_exit = True
    bb.close()
    thread.join(timeout=5)


@skip_no_docker
@skip_no_key
def test_agent_container_is_healthy(e2e_stack):
    """Verify the agent container is running and reporting status."""
    url = e2e_stack["agent_url"]
    r = httpx.get(f"{url}/status", timeout=5)
    assert r.status_code == 200
    status = r.json()
    assert status["agent_id"] == "research"
    assert status["role"] == "research"
    assert status["state"] == "idle"


@skip_no_docker
@skip_no_key
def test_agent_registered_with_mesh(e2e_stack):
    """Verify the agent registered itself with the mesh."""
    mesh_url = e2e_stack["mesh_url"]
    r = httpx.get(f"{mesh_url}/mesh/agents", timeout=5)
    assert r.status_code == 200
    agents = r.json()
    assert "research" in agents


@skip_no_docker
@skip_no_key
def test_agent_has_capabilities(e2e_stack):
    """Verify the agent exposes its skill definitions."""
    url = e2e_stack["agent_url"]
    r = httpx.get(f"{url}/capabilities", timeout=5)
    assert r.status_code == 200
    caps = r.json()
    assert "web_search" in caps["skills"]
    assert "company_lookup" in caps["skills"]


@skip_no_docker
@skip_no_key
def test_research_workflow_e2e(e2e_stack):
    """Full E2E: trigger research_only workflow with real LLM, verify completion."""
    mesh_url = e2e_stack["mesh_url"]

    # Trigger the workflow
    r = httpx.post(
        f"{mesh_url}/webhook/trigger/research_only",
        json={"company": "Stripe", "lead_id": "test_001"},
        timeout=10,
    )
    assert r.status_code == 200
    data = r.json()
    assert "execution_id" in data
    exec_id = data["execution_id"]

    # Poll until complete or timeout (300s -- agent with builtins does real work)
    final_status = None
    for _ in range(300):
        try:
            r = httpx.get(f"{mesh_url}/webhook/status/{exec_id}", timeout=10)
            if r.status_code == 200:
                final_status = r.json()
                if final_status.get("status") in ("complete", "failed"):
                    break
        except (httpx.TimeoutException, httpx.ConnectError):
            pass
        time.sleep(1)

    assert final_status is not None, "Never got a status response"

    # Dump agent logs if workflow didn't complete for debugging
    if final_status.get("status") != "complete":
        try:
            container = e2e_stack["container_manager"].containers["research"]["container"]
            container.reload()
            logs = container.logs(tail=80).decode()
            print(f"\n=== Agent container logs ===\n{logs}\n===")
        except Exception:
            pass

    assert final_status["status"] == "complete", (
        f"Workflow did not complete successfully: {final_status}"
    )
    assert final_status["steps_completed"] >= 1

    # Verify agent went back to idle (retry a few times, agent may still be finishing)
    agent_url = e2e_stack["agent_url"]
    status = None
    for _ in range(10):
        try:
            r = httpx.get(f"{agent_url}/status", timeout=5)
            status = r.json()
            if status["state"] == "idle":
                break
        except Exception:
            pass
        time.sleep(1)
    assert status is not None
    assert status["state"] == "idle"
    assert status["tasks_completed"] >= 1

    # Verify agent has a result
    r = httpx.get(f"{agent_url}/result", timeout=5)
    assert r.status_code == 200
    result = r.json()
    assert result["status"] == "complete"
    assert result["result"] is not None
    assert result["tokens_used"] > 0

    print("\n=== Workflow completed ===")
    print(f"  Execution ID: {exec_id}")
    print(f"  Steps completed: {final_status['steps_completed']}")
    print(f"  Tokens used: {result['tokens_used']}")
    print(f"  Result keys: {list(result['result'].keys()) if isinstance(result['result'], dict) else 'raw'}")
    print(f"  Duration: {result.get('duration_ms', 0)}ms")
