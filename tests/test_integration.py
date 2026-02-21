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


# ── Vault endpoint tests ──────────────────────────────────────


@pytest.fixture
def vault_components(tmp_path):
    """Mesh components with vault support and can_manage_vault permission."""
    from src.host.credentials import CredentialVault

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()

    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {
        "trusted": AgentPermissions(
            agent_id="trusted",
            can_message=["orchestrator"],
            blackboard_read=["context/*"],
            blackboard_write=[],
            allowed_apis=["llm"],
            can_manage_vault=True,
        ),
        "untrusted": AgentPermissions(
            agent_id="untrusted",
            can_message=["orchestrator"],
            blackboard_read=[],
            blackboard_write=[],
            allowed_apis=[],
            can_manage_vault=False,
        ),
    }

    router = MessageRouter(permissions=perms, agent_registry={})
    vault = CredentialVault()

    # Patch _persist_to_env to avoid writing to real .env
    import src.host.credentials as cred_mod
    cred_mod._persist_to_env = lambda *a, **kw: None

    app = create_mesh_app(bb, pubsub, router, perms, credential_vault=vault)
    client = TestClient(app)

    return {"client": client, "vault": vault}


def test_vault_store_endpoint(vault_components):
    client = vault_components["client"]
    response = client.post(
        "/mesh/vault/store",
        json={"agent_id": "trusted", "name": "brave_search", "value": "sk-test-123"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["stored"] is True
    assert data["handle"] == "$CRED{brave_search}"
    # Value must NOT be in response
    assert "sk-test-123" not in str(data)


def test_vault_store_permission_denied(vault_components):
    client = vault_components["client"]
    response = client.post(
        "/mesh/vault/store",
        json={"agent_id": "untrusted", "name": "key", "value": "secret"},
    )
    assert response.status_code == 403


def test_vault_list_endpoint(vault_components):
    client = vault_components["client"]
    vault = vault_components["vault"]
    vault.credentials["test_key"] = "value"

    response = client.get("/mesh/vault/list", params={"agent_id": "trusted"})
    assert response.status_code == 200
    data = response.json()
    assert "test_key" in data["credentials"]
    assert data["count"] >= 1


def test_vault_status_endpoint(vault_components):
    client = vault_components["client"]
    vault = vault_components["vault"]
    vault.credentials["exists_key"] = "val"

    response = client.get(
        "/mesh/vault/status/exists_key", params={"agent_id": "trusted"},
    )
    assert response.status_code == 200
    assert response.json()["exists"] is True

    response = client.get(
        "/mesh/vault/status/missing_key", params={"agent_id": "trusted"},
    )
    assert response.status_code == 200
    assert response.json()["exists"] is False


def test_vault_resolve_endpoint(vault_components):
    client = vault_components["client"]
    vault = vault_components["vault"]
    vault.credentials["resolve_me"] = "secret_value"

    response = client.post(
        "/mesh/vault/resolve",
        json={"agent_id": "trusted", "name": "resolve_me"},
    )
    assert response.status_code == 200
    assert response.json()["value"] == "secret_value"


def test_vault_resolve_not_found(vault_components):
    client = vault_components["client"]
    response = client.post(
        "/mesh/vault/resolve",
        json={"agent_id": "trusted", "name": "nonexistent"},
    )
    assert response.status_code == 404


# ── Auth token tests ──────────────────────────────────────────


@pytest.fixture
def authed_components(tmp_path):
    """Mesh components with auth tokens configured."""
    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()

    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {
        "research": AgentPermissions(
            agent_id="research",
            can_message=["orchestrator"],
            can_publish=["research_complete"],
            can_subscribe=["new_lead"],
            blackboard_read=["context/*"],
            blackboard_write=["context/research_*"],
            allowed_apis=["anthropic"],
        ),
    }

    router = MessageRouter(permissions=perms, agent_registry={})

    tokens = {"research": "test-token-abc123"}
    app = create_mesh_app(bb, pubsub, router, perms, auth_tokens=tokens)
    client = TestClient(app)

    return {"client": client, "tokens": tokens}


def test_auth_valid_token_passes(authed_components):
    """Request with correct Bearer token succeeds."""
    client = authed_components["client"]
    response = client.post(
        "/mesh/register",
        json={"agent_id": "research", "capabilities": [], "port": 8401},
        headers={"Authorization": "Bearer test-token-abc123"},
    )
    assert response.status_code == 200
    assert response.json()["registered"] is True


def test_auth_missing_token_rejected(authed_components):
    """Request without auth header returns 401."""
    client = authed_components["client"]
    response = client.post(
        "/mesh/register",
        json={"agent_id": "research", "capabilities": [], "port": 8401},
    )
    assert response.status_code == 401


def test_auth_wrong_token_rejected(authed_components):
    """Request with wrong token returns 401."""
    client = authed_components["client"]
    response = client.post(
        "/mesh/register",
        json={"agent_id": "research", "capabilities": [], "port": 8401},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert response.status_code == 401


def test_auth_not_required_without_config(mesh_components):
    """Without auth_tokens configured, requests pass without token."""
    client = mesh_components["client"]
    response = client.post(
        "/mesh/register",
        json={"agent_id": "research", "capabilities": [], "port": 8401},
    )
    assert response.status_code == 200


def test_auth_mesh_agent_id_bypasses(authed_components):
    """The 'mesh' and 'orchestrator' agent IDs bypass auth."""
    client = authed_components["client"]
    response = client.post(
        "/mesh/message",
        json={
            "from_agent": "mesh",
            "to": "research",
            "type": "system",
            "payload": {"text": "hello"},
        },
    )
    # Should not get 401 — mesh identity bypasses auth
    assert response.status_code != 401


# ── Vault rate limiting tests ─────────────────────────────────


def test_vault_resolve_rate_limited(vault_components):
    """After 5 resolves, subsequent ones return 429."""
    client = vault_components["client"]
    vault = vault_components["vault"]
    vault.credentials["rate_test"] = "value"

    # First 5 should succeed
    for _ in range(5):
        resp = client.post(
            "/mesh/vault/resolve",
            json={"agent_id": "trusted", "name": "rate_test"},
        )
        assert resp.status_code == 200

    # 6th should be rate limited
    resp = client.post(
        "/mesh/vault/resolve",
        json={"agent_id": "trusted", "name": "rate_test"},
    )
    assert resp.status_code == 429


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


# === Notify Endpoint Tests ===


def test_notify_endpoint_success(tmp_path):
    """POST /mesh/notify calls notify_fn and returns sent=True."""
    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {
        "agent1": AgentPermissions(agent_id="agent1"),
    }
    router = MessageRouter(permissions=perms, agent_registry={})

    calls = []

    async def mock_notify(agent_name: str, message: str):
        calls.append((agent_name, message))

    app = create_mesh_app(bb, pubsub, router, perms, notify_fn=mock_notify)
    client = TestClient(app)

    response = client.post("/mesh/notify", json={
        "agent_id": "agent1",
        "message": "Task completed: report ready",
    })
    assert response.status_code == 200
    assert response.json() == {"sent": True}
    assert len(calls) == 1
    assert calls[0] == ("agent1", "Task completed: report ready")

    bb.close()


def test_notify_endpoint_no_notify_fn(tmp_path):
    """POST /mesh/notify returns 503 when notify_fn is not configured."""
    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {}
    router = MessageRouter(permissions=perms, agent_registry={})

    app = create_mesh_app(bb, pubsub, router, perms, notify_fn=None)
    client = TestClient(app)

    response = client.post("/mesh/notify", json={
        "agent_id": "agent1",
        "message": "hello",
    })
    assert response.status_code == 503

    bb.close()


def test_notify_endpoint_truncates_long_message(tmp_path):
    """Messages longer than _NOTIFY_MAX_LEN (2000) are truncated."""
    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {}
    router = MessageRouter(permissions=perms, agent_registry={})

    calls = []

    async def mock_notify(agent_name: str, message: str):
        calls.append((agent_name, message))

    app = create_mesh_app(bb, pubsub, router, perms, notify_fn=mock_notify)
    client = TestClient(app)

    long_message = "x" * 3000
    response = client.post("/mesh/notify", json={
        "agent_id": "agent1",
        "message": long_message,
    })
    assert response.status_code == 200
    assert len(calls) == 1
    assert len(calls[0][1]) == 2000
    assert calls[0][1] == "x" * 2000

    bb.close()
