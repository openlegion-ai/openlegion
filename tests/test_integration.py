"""Integration tests for the mesh server API.

Tests the FastAPI endpoints directly using TestClient (no Docker required).
"""


import pytest
from fastapi.testclient import TestClient

from src.host.mesh import Blackboard, MessageRouter, PubSub
from src.host.orchestrator import Orchestrator
from src.host.permissions import PermissionMatrix
from src.host.server import create_mesh_app
from src.shared.types import AgentPermissions


@pytest.fixture
def mesh_components(tmp_path):
    """Create all mesh components with test configuration."""
    bb = Blackboard(db_path=str(tmp_path / "bb.db"))

    pubsub = PubSub()

    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {
        "research": AgentPermissions(
            agent_id="research",
            can_message=["orchestrator"],
            can_publish=["research_complete"],
            can_subscribe=["new_lead"],
            blackboard_read=["context/*", "tasks/*"],
            blackboard_write=["context/research_*", "context/prospect_*"],
            allowed_apis=["anthropic", "brave_search"],
        ),
        "qualify": AgentPermissions(
            agent_id="qualify",
            can_message=["orchestrator"],
            blackboard_read=["context/*"],
            blackboard_write=["context/qualify_*"],
            allowed_apis=["anthropic"],
        ),
    }

    router = MessageRouter(permissions=perms, agent_registry={})

    app = create_mesh_app(bb, pubsub, router, perms, credential_vault=None)
    client = TestClient(app)

    return {"client": client, "blackboard": bb, "pubsub": pubsub, "router": router, "perms": perms}


def test_register_agent(mesh_components):
    client = mesh_components["client"]
    response = client.post(
        "/mesh/register", json={"agent_id": "research", "capabilities": ["web_search"], "port": 8401}
    )
    assert response.status_code == 200
    assert response.json()["registered"] is True

    agents = client.get("/mesh/agents").json()
    assert "research" in agents


def test_blackboard_write_and_read(mesh_components):
    client = mesh_components["client"]

    # Write
    response = client.put(
        "/mesh/blackboard/context/research_acme",
        params={"agent_id": "research"},
        json={"company": "Acme", "employees": 500},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["key"] == "context/research_acme"

    # Read
    response = client.get("/mesh/blackboard/context/research_acme", params={"agent_id": "research"})
    assert response.status_code == 200
    assert response.json()["value"]["company"] == "Acme"


def test_blackboard_permission_denied(mesh_components):
    client = mesh_components["client"]

    # Agent "qualify" cannot write to context/research_*
    response = client.put(
        "/mesh/blackboard/context/research_data",
        params={"agent_id": "qualify"},
        json={"data": "test"},
    )
    assert response.status_code == 403


def test_blackboard_not_found(mesh_components):
    client = mesh_components["client"]
    response = client.get("/mesh/blackboard/context/nonexistent", params={"agent_id": "research"})
    assert response.status_code == 404


def test_blackboard_list_by_prefix(mesh_components):
    client = mesh_components["client"]

    client.put("/mesh/blackboard/context/prospect_1", params={"agent_id": "research"}, json={"name": "Acme"})
    client.put("/mesh/blackboard/context/prospect_2", params={"agent_id": "research"}, json={"name": "Beta"})

    response = client.get("/mesh/blackboard/", params={"agent_id": "research", "prefix": "context/prospect_"})
    assert response.status_code == 200
    entries = response.json()
    assert len(entries) == 2


def test_subscribe(mesh_components):
    client = mesh_components["client"]
    response = client.post("/mesh/subscribe", params={"topic": "new_lead", "agent_id": "research"})
    assert response.status_code == 200
    assert response.json()["subscribed"] is True

    subs = mesh_components["pubsub"].get_subscribers("new_lead")
    assert "research" in subs


def test_subscribe_permission_denied(mesh_components):
    client = mesh_components["client"]
    response = client.post("/mesh/subscribe", params={"topic": "new_lead", "agent_id": "qualify"})
    assert response.status_code == 403


def test_api_proxy_no_vault(mesh_components):
    client = mesh_components["client"]
    response = client.post(
        "/mesh/api",
        params={"agent_id": "research"},
        json={"service": "brave_search", "action": "search", "params": {"query": "test"}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert "No credential vault" in data["error"]


def test_api_proxy_permission_denied(mesh_components):
    client = mesh_components["client"]
    response = client.post(
        "/mesh/api",
        params={"agent_id": "qualify"},
        json={"service": "brave_search", "action": "search", "params": {}},
    )
    assert response.status_code == 403


def test_list_agents(mesh_components):
    client = mesh_components["client"]
    router = mesh_components["router"]
    router.register_agent("research", "http://localhost:8401", ["web_search"])
    router.register_agent("qualify", "http://localhost:8402", ["score_lead"])

    response = client.get("/mesh/agents")
    assert response.status_code == 200
    agents = response.json()
    assert "research" in agents
    assert "qualify" in agents


def test_webhook_integration(tmp_path):
    """Test webhook endpoint triggers orchestrator (without actual agents)."""
    from src.channels.webhook import create_webhook_router

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()

    wf_dir = tmp_path / "workflows"
    wf_dir.mkdir()
    (wf_dir / "test_wf.yaml").write_text(
        "name: test_wf\ntrigger: test\nsteps:\n  - id: s1\n    task_type: t1\n    agent: test_agent\n"
    )

    orch = Orchestrator(
        mesh_url="http://localhost:8420",
        workflows_dir=str(wf_dir),
        blackboard=bb,
        pubsub=pubsub,
    )

    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(create_webhook_router(orch))
    client = TestClient(app)

    response = client.post("/webhook/trigger/test_wf", json={"company": "Acme"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "started"
    assert "execution_id" in data

    response = client.get(f"/webhook/status/{data['execution_id']}")
    assert response.status_code == 200
    status = response.json()
    assert status["workflow"] == "test_wf"


def test_webhook_unknown_workflow(tmp_path):
    """Test triggering an unknown workflow returns 404."""
    from src.channels.webhook import create_webhook_router

    orch = Orchestrator(mesh_url="http://localhost:8420", workflows_dir="/nonexistent")

    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(create_webhook_router(orch))
    client = TestClient(app)

    response = client.post("/webhook/trigger/nonexistent", json={})
    assert response.status_code == 404


def test_mesh_message_to_orchestrator(tmp_path):
    """Messages to 'orchestrator' with type 'task_result' resolve pending futures."""
    import asyncio

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {
        "research": AgentPermissions(
            agent_id="research",
            can_message=["orchestrator"],
            blackboard_read=["context/*"],
            blackboard_write=[],
            allowed_apis=[],
        ),
    }
    router = MessageRouter(permissions=perms, agent_registry={})
    orch = Orchestrator(mesh_url="http://localhost:8420", workflows_dir="/nonexistent")

    # Create a pending future
    loop = asyncio.new_event_loop()
    future = loop.create_future()
    orch._pending_results["task_123"] = future

    app = create_mesh_app(bb, pubsub, router, perms, orchestrator=orch)
    client = TestClient(app)

    response = client.post("/mesh/message", json={
        "from_agent": "research",
        "to": "orchestrator",
        "type": "task_result",
        "payload": {"task_id": "task_123", "status": "complete", "result": {"data": "ok"}},
    })
    assert response.status_code == 200
    data = response.json()
    assert data["delivered"] is True
    assert future.done()

    loop.close()
