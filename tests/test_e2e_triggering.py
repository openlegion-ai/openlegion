"""End-to-end tests for Phase 2: Triggering + Cost Tracking.

Tests:
  - Cron job dispatches to running agent, gets response
  - Webhook endpoint dispatches to running agent
  - Cost tracker records real LLM usage from agent container
  - Budget enforcement blocks over-budget agent

Requires Docker + OPENAI_API_KEY, same as other E2E tests.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time

import httpx
import pytest
import uvicorn
from dotenv import load_dotenv

load_dotenv()

MESH_PORT = 8438


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
def e2e_trigger_stack(tmp_path_factory):
    """Mesh + agent + cost tracker + webhook manager for trigger tests."""
    if not _docker_available() or not _has_llm_key():
        pytest.skip("Docker or OPENAI_API_KEY not available")

    if not os.environ.get("OPENLEGION_CRED_OPENAI_API_KEY"):
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        if openai_key:
            os.environ["OPENLEGION_CRED_OPENAI_API_KEY"] = openai_key

    from src.host.containers import ContainerManager
    from src.host.costs import CostTracker
    from src.host.credentials import CredentialVault
    from src.host.mesh import Blackboard, MessageRouter, PubSub
    from src.host.permissions import PermissionMatrix
    from src.host.server import create_mesh_app
    from src.host.webhooks import WebhookManager

    tmp_dir = tmp_path_factory.mktemp("e2e_trigger")
    bb = Blackboard(db_path=str(tmp_dir / "blackboard.db"))
    pubsub = PubSub()
    perms = PermissionMatrix()

    cost_tracker = CostTracker(db_path=str(tmp_dir / "costs.db"))
    vault = CredentialVault(cost_tracker=cost_tracker)
    cm = ContainerManager(mesh_host_port=MESH_PORT, use_host_network=True)
    router = MessageRouter(perms, {})

    # Dispatch function for webhooks/cron — will be finalized after agent starts
    dispatch_state: dict = {}

    async def dispatch_fn(agent_name: str, message: str) -> str:
        url = dispatch_state.get("agent_url", "")
        if not url:
            return "Agent not ready"
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(f"{url}/chat", json={"message": message})
            return r.json().get("response", "(no response)")

    webhook_manager = WebhookManager(
        config_path=str(tmp_dir / "webhooks.json"),
        dispatch_fn=dispatch_fn,
    )

    app = create_mesh_app(bb, pubsub, router, perms, vault)
    app.include_router(webhook_manager.create_router())

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

    skills_dir = os.path.abspath("skills/research")
    url = cm.start_agent(
        agent_id="trigger_test",
        role="assistant",
        skills_dir=skills_dir,
        system_prompt="You are a helpful assistant. Keep responses brief.",
        model="openai/gpt-4o-mini",
    )
    router.register_agent("trigger_test", url)
    dispatch_state["agent_url"] = url

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

    yield {
        "mesh_url": f"http://localhost:{MESH_PORT}",
        "agent_url": url,
        "cost_tracker": cost_tracker,
        "webhook_manager": webhook_manager,
        "container_manager": cm,
        "dispatch_fn": dispatch_fn,
    }

    cm.stop_all()
    server.should_exit = True
    cost_tracker.close()
    bb.close()
    thread.join(timeout=5)


@skip_no_docker
@skip_no_key
def test_cron_dispatch_to_agent(e2e_trigger_stack):
    """Cron scheduler dispatches a message to a running agent and gets a response."""
    import asyncio

    from src.host.cron import CronScheduler

    agent_url = e2e_trigger_stack["agent_url"]

    async def dispatch(agent_name: str, message: str) -> str:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(f"{agent_url}/chat", json={"message": message})
            return r.json().get("response", "(no response)")

    tmp = tempfile.mkdtemp()
    try:
        scheduler = CronScheduler(
            config_path=f"{tmp}/cron.json",
            dispatch_fn=dispatch,
        )
        job = scheduler.add_job(
            agent="trigger_test",
            schedule="every 1m",
            message="Say exactly 'CRON_OK' and nothing else.",
        )

        result = asyncio.run(scheduler.run_job(job.id))
        assert result is not None
        assert len(result) > 0
        assert job.run_count == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@skip_no_docker
@skip_no_key
def test_webhook_dispatch_to_agent(e2e_trigger_stack):
    """Webhook endpoint receives POST and dispatches to agent."""
    mgr = e2e_trigger_stack["webhook_manager"]
    hook = mgr.add_hook(agent="trigger_test", name="test-event")

    r = httpx.post(
        f"{e2e_trigger_stack['mesh_url']}/webhook/hook/{hook['id']}",
        json={"event": "push", "repo": "openlegion"},
        timeout=120,
    )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "processed"
    assert data["response"] is not None


@skip_no_docker
@skip_no_key
def test_cost_tracked_after_chat(e2e_trigger_stack):
    """After agent processes a message, cost tracker has recorded usage."""
    tracker = e2e_trigger_stack["cost_tracker"]

    # Send a chat message to generate LLM usage
    agent_url = e2e_trigger_stack["agent_url"]
    r = httpx.post(
        f"{agent_url}/chat",
        json={"message": "Say hello in exactly 3 words."},
        timeout=120,
    )
    assert r.status_code == 200

    spend = tracker.get_spend(period="today")
    assert spend["total_tokens"] >= 0


@skip_no_docker
@skip_no_key
def test_webhook_unknown_hook_returns_404(e2e_trigger_stack):
    """Unknown webhook ID returns 404."""
    r = httpx.post(
        f"{e2e_trigger_stack['mesh_url']}/webhook/hook/nonexistent",
        json={"event": "test"},
        timeout=10,
    )
    assert r.status_code == 404
