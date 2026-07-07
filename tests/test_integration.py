"""Integration tests for the mesh server API.

Tests the FastAPI endpoints directly using TestClient (no Docker required).
"""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

from src.host.mesh import Blackboard, MessageRouter, PubSub
from src.host.permissions import PermissionMatrix
from src.host.server import create_mesh_app
from src.shared.types import AgentPermissions

# Mirror of src/host/server.py:_RATE_LIMITS["vault_resolve"]; bump together.
EXPECTED_VAULT_RESOLVE_LIMIT = 10000


def _prefill_rate_bucket(app, endpoint: str, agent_id: str, count: int) -> None:
    """Stuff ``count`` timestamps into the rate-limit bucket so the next
    request lands at ``count + 1``. Used so high-ceiling tests don't have
    to make thousands of HTTP calls."""
    import time

    bucket = app.state.rate_ts[f"{endpoint}:{agent_id}"]
    now = time.time()
    for _ in range(count):
        bucket.append(now)


@pytest.fixture
def mesh_components(tmp_path, monkeypatch):
    """Create all mesh components with test configuration.

    Pinned to ``OPENLEGION_TEAM_SCOPE_MODE=warn`` because these tests
    hit ``/mesh/agents`` with anonymous TestClient calls; the new
    ``enforce`` default would filter the response down to {operator}
    and break the contract assertions.
    """
    monkeypatch.setenv("OPENLEGION_TEAM_SCOPE_MODE", "warn")
    import src.host.server as server_module
    importlib.reload(server_module)
    # Use the freshly-reloaded module's factory so the new
    # ``_PROJECT_SCOPE_MODE`` value is honored. Other fixtures in this
    # file still use the top-level ``create_mesh_app`` import — they
    # don't drive ``/mesh/agents`` so the default is fine for them.
    fresh_create_mesh_app = server_module.create_mesh_app

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))

    pubsub = PubSub()

    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {
        "research": AgentPermissions(
            agent_id="research",
            can_message=["mesh"],
            can_publish=["research_complete"],
            # Scoped: a teamless worker is a team-of-one, so runtime
            # subscriptions carry its own teams/{id}/ prefix (ratified #5).
            can_subscribe=["teams/research/new_lead"],
            blackboard_read=["context/*", "tasks/*"],
            blackboard_write=["context/research_*", "context/prospect_*"],
            allowed_apis=["anthropic", "brave_search"],
        ),
        "qualify": AgentPermissions(
            agent_id="qualify",
            can_message=["mesh"],
            blackboard_read=["context/*"],
            blackboard_write=["context/qualify_*"],
            allowed_apis=["anthropic"],
        ),
    }

    router = MessageRouter(permissions=perms, agent_registry={})

    app = fresh_create_mesh_app(bb, pubsub, router, perms, credential_vault=None)
    client = TestClient(app)

    yield {"client": client, "blackboard": bb, "pubsub": pubsub, "router": router, "perms": perms, "app": app}

    monkeypatch.delenv("OPENLEGION_TEAM_SCOPE_MODE", raising=False)
    importlib.reload(server_module)


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


def test_blackboard_key_too_long(mesh_components):
    """Keys longer than 512 chars are rejected."""
    client = mesh_components["client"]
    long_key = "context/" + "x" * 510
    response = client.put(
        f"/mesh/blackboard/{long_key}",
        params={"agent_id": "research"},
        json={"data": "test"},
    )
    assert response.status_code == 400
    assert "Key too long" in response.json()["detail"]


def test_blackboard_value_too_large(mesh_components):
    """Values larger than 256 KB are rejected."""
    client = mesh_components["client"]
    big_value = {"data": "x" * 300_000}
    response = client.put(
        "/mesh/blackboard/context/big",
        params={"agent_id": "research"},
        json=big_value,
    )
    assert response.status_code == 413
    assert "Value too large" in response.json()["detail"]


def test_blackboard_claim_key_too_long(mesh_components):
    """CAS claim rejects keys longer than 512 chars."""
    client = mesh_components["client"]
    long_key = "context/" + "x" * 510
    response = client.post(
        "/mesh/blackboard/claim",
        json={
            "agent_id": "research",
            "key": long_key,
            "value": {"status": "claimed"},
            "expected_version": 1,
        },
    )
    assert response.status_code == 400


def test_blackboard_claim_value_too_large(mesh_components):
    """CAS claim rejects values larger than 256 KB."""
    client = mesh_components["client"]
    response = client.post(
        "/mesh/blackboard/claim",
        json={
            "agent_id": "research",
            "key": "context/big_claim",
            "value": {"data": "x" * 300_000},
            "expected_version": 1,
        },
    )
    assert response.status_code == 413


def test_blackboard_key_at_limit_succeeds(mesh_components):
    """Keys exactly at 512 chars are accepted."""
    client = mesh_components["client"]
    key = "context/research_" + "x" * (512 - len("context/research_"))
    assert len(key) == 512
    response = client.put(
        f"/mesh/blackboard/{key}",
        params={"agent_id": "research"},
        json={"data": "ok"},
    )
    assert response.status_code == 200


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
    response = client.post(
        "/mesh/subscribe",
        params={"topic": "teams/research/new_lead", "agent_id": "research"},
    )
    assert response.status_code == 200
    assert response.json()["subscribed"] is True

    subs = mesh_components["pubsub"].get_subscribers("teams/research/new_lead")
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


def test_list_agents_scoped_by_project(mesh_components, tmp_path):
    """When team param is set, only that team's members are returned."""
    client = mesh_components["client"]
    router = mesh_components["router"]
    app = mesh_components["app"]
    router.register_agent("alice", "http://localhost:8401")
    router.register_agent("bob", "http://localhost:8402")
    router.register_agent("charlie", "http://localhost:8403")

    app.teams_store.create_team("teamA")
    app.teams_store.add_member("teamA", "alice")
    app.teams_store.add_member("teamA", "bob")

    # Scoped by team
    resp = client.get("/mesh/agents", params={"team": "teamA"})
    assert resp.status_code == 200
    agents = resp.json()
    assert "alice" in agents
    assert "bob" in agents
    assert "charlie" not in agents


def test_list_agents_scoped_by_agent_id(mesh_components):
    """When agent_id param is set, only that agent is returned."""
    client = mesh_components["client"]
    router = mesh_components["router"]
    router.register_agent("solo", "http://localhost:8401")
    router.register_agent("other", "http://localhost:8402")

    resp = client.get("/mesh/agents", params={"agent_id": "solo"})
    assert resp.status_code == 200
    agents = resp.json()
    assert "solo" in agents
    assert "other" not in agents


def test_list_agents_unknown_agent_id(mesh_components):
    """Standalone agent not in registry gets empty dict."""
    client = mesh_components["client"]
    resp = client.get("/mesh/agents", params={"agent_id": "ghost"})
    assert resp.status_code == 200
    assert resp.json() == {}


# ── Vault endpoint tests ──────────────────────────────────────


@pytest.fixture
def vault_components(tmp_path):
    """Mesh components with vault support and allowed_credentials permission."""
    from src.host.credentials import CredentialVault

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()

    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {
        "trusted": AgentPermissions(
            agent_id="trusted",
            can_message=["mesh"],
            blackboard_read=["context/*"],
            blackboard_write=[],
            allowed_apis=["llm"],
            allowed_credentials=["*"],
        ),
        "untrusted": AgentPermissions(
            agent_id="untrusted",
            can_message=["mesh"],
            blackboard_read=[],
            blackboard_write=[],
            allowed_apis=[],
            allowed_credentials=[],
        ),
        "scoped": AgentPermissions(
            agent_id="scoped",
            can_message=["mesh"],
            blackboard_read=[],
            blackboard_write=[],
            allowed_apis=[],
            allowed_credentials=["brightdata_*"],
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


def test_vault_store_blocks_system_credential(vault_components):
    """Agents cannot store/overwrite system credentials like anthropic_api_key."""
    client = vault_components["client"]
    response = client.post(
        "/mesh/vault/store",
        json={"agent_id": "trusted", "name": "anthropic_api_key", "value": "malicious"},
    )
    assert response.status_code == 403
    assert "system credential" in response.json()["detail"].lower()


def test_vault_store_strips_whitespace(vault_components):
    """Credential values with leading/trailing whitespace are stripped."""
    client = vault_components["client"]
    vault = vault_components["vault"]
    response = client.post(
        "/mesh/vault/store",
        json={"agent_id": "trusted", "name": "notion", "value": "  ntn_secret_abc  "},
    )
    assert response.status_code == 200
    assert vault.credentials["notion"] == "ntn_secret_abc"


def test_vault_list_endpoint(vault_components):
    client = vault_components["client"]
    vault = vault_components["vault"]
    vault.credentials["test_key"] = "value"

    response = client.get("/mesh/vault/list", params={"agent_id": "trusted"})
    assert response.status_code == 200
    data = response.json()
    assert "test_key" in data["credentials"]
    assert data["count"] >= 1


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
            can_message=["mesh"],
            can_publish=["research_complete"],
            # Scoped: a teamless worker is a team-of-one, so runtime
            # subscriptions carry its own teams/{id}/ prefix (ratified #5).
            can_subscribe=["teams/research/new_lead"],
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
    """The 'mesh' agent ID bypasses auth."""
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


# ── Vault credential scoping tests ────────────────────────────


def test_vault_resolve_system_credential_blocked(vault_components):
    """System credentials (e.g. anthropic_api_key) are never resolvable by agents."""
    client = vault_components["client"]
    vault = vault_components["vault"]
    vault.credentials["anthropic_api_key"] = "sk-system-secret"

    response = client.post(
        "/mesh/vault/resolve",
        json={"agent_id": "trusted", "name": "anthropic_api_key"},
    )
    assert response.status_code == 403


def test_vault_resolve_scoped_agent_allowed(vault_components):
    """Agent with allowed_credentials: ['brightdata_*'] can resolve matching creds."""
    client = vault_components["client"]
    vault = vault_components["vault"]
    vault.credentials["brightdata_cdp_url"] = "wss://test"

    response = client.post(
        "/mesh/vault/resolve",
        json={"agent_id": "scoped", "name": "brightdata_cdp_url"},
    )
    assert response.status_code == 200
    assert response.json()["value"] == "wss://test"


def test_vault_resolve_scoped_agent_denied(vault_components):
    """Agent with allowed_credentials: ['brightdata_*'] cannot resolve non-matching creds."""
    client = vault_components["client"]
    vault = vault_components["vault"]
    vault.credentials["myapp_password"] = "secret"

    response = client.post(
        "/mesh/vault/resolve",
        json={"agent_id": "scoped", "name": "myapp_password"},
    )
    assert response.status_code == 403


def test_vault_list_filters_by_scoping(vault_components):
    """vault_list returns only credentials the agent can access."""
    client = vault_components["client"]
    vault = vault_components["vault"]
    vault.credentials["anthropic_api_key"] = "sk-system"
    vault.credentials["brightdata_cdp_url"] = "wss://test"
    vault.credentials["myapp_password"] = "secret"

    # Scoped agent should only see brightdata_*
    response = client.get("/mesh/vault/list", params={"agent_id": "scoped"})
    assert response.status_code == 200
    data = response.json()
    assert "brightdata_cdp_url" in data["credentials"]
    assert "anthropic_api_key" not in data["credentials"]
    assert "myapp_password" not in data["credentials"]


def test_vault_list_wildcard_excludes_system(vault_components):
    """Even with allowed_credentials: ['*'], system creds are excluded from list."""
    client = vault_components["client"]
    vault = vault_components["vault"]
    vault.credentials["anthropic_api_key"] = "sk-system"
    vault.credentials["brightdata_cdp_url"] = "wss://test"

    response = client.get("/mesh/vault/list", params={"agent_id": "trusted"})
    assert response.status_code == 200
    data = response.json()
    assert "brightdata_cdp_url" in data["credentials"]
    assert "anthropic_api_key" not in data["credentials"]


# ── Vault rate limiting tests ─────────────────────────────────


def test_vault_resolve_rate_limited(vault_components):
    """Once the per-agent vault_resolve bucket is full, the next resolve
    returns 429. Looping the full ``limit + 1`` requests would be slow
    against the spam-only ceiling, so we pre-fill the bucket instead.
    """
    client = vault_components["client"]
    vault = vault_components["vault"]
    vault.credentials["rate_test"] = "value"

    limit = EXPECTED_VAULT_RESOLVE_LIMIT
    _prefill_rate_bucket(client.app, "vault_resolve", "trusted", limit - 1)

    # One more should succeed (consumes the final token).
    resp = client.post(
        "/mesh/vault/resolve",
        json={"agent_id": "trusted", "name": "rate_test"},
    )
    assert resp.status_code == 200, resp.text

    # The next call must be 429.
    resp = client.post(
        "/mesh/vault/resolve",
        json={"agent_id": "trusted", "name": "rate_test"},
    )
    assert resp.status_code == 429, resp.text


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


# === Introspect Endpoint ===


def test_introspect_returns_permissions(tmp_path):
    """GET /mesh/introspect returns agent permissions."""
    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {
        "alice": AgentPermissions(
            agent_id="alice",
            blackboard_read=["context/*", "tasks/*"],
            blackboard_write=["context/alice_*"],
            can_message=["bob"],
            can_publish=["updates"],
            can_subscribe=["alerts"],
            allowed_apis=["anthropic"],
            allowed_credentials=["brightdata_*"],
        ),
    }
    router = MessageRouter(permissions=perms, agent_registry={})

    app = create_mesh_app(bb, pubsub, router, perms)
    client = TestClient(app)

    response = client.get(
        "/mesh/introspect",
        params={"section": "permissions"},
        headers={"X-Agent-ID": "alice"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "permissions" in data
    assert data["permissions"]["blackboard_read"] == ["context/*", "tasks/*"]
    assert data["permissions"]["can_message"] == ["bob"]
    assert data["permissions"]["allowed_apis"] == ["anthropic"]
    assert data["permissions"]["allowed_credentials"] == ["brightdata_*"]

    # Sync guard: introspect response keys must match INTROSPECT_PERM_KEYS
    from src.agent.workspace import INTROSPECT_PERM_KEYS
    assert set(data["permissions"].keys()) == set(INTROSPECT_PERM_KEYS), (
        f"Introspect permissions keys out of sync with INTROSPECT_PERM_KEYS. "
        f"Endpoint: {sorted(data['permissions'].keys())}. "
        f"Constant: {sorted(INTROSPECT_PERM_KEYS)}."
    )

    bb.close()


def test_introspect_returns_fleet_standalone(tmp_path):
    """GET /mesh/introspect section=fleet for standalone agent shows only self."""

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {
        "alice": AgentPermissions(agent_id="alice", can_message=["bob"]),
    }
    router = MessageRouter(permissions=perms, agent_registry={})
    router.register_agent("alice", "http://localhost:8401")
    router.register_agent("bob", "http://localhost:8402")

    app = create_mesh_app(bb, pubsub, router, perms)
    client = TestClient(app)

    # alice is standalone (not in any team) — sees only herself
    response = client.get(
        "/mesh/introspect",
        params={"section": "fleet"},
        headers={"X-Agent-ID": "alice"},
    )
    assert response.status_code == 200
    data = response.json()
    ids = [a["id"] for a in data["fleet"]]
    assert ids == ["alice"]

    bb.close()


def test_introspect_returns_fleet_project_scoped(tmp_path):
    """GET /mesh/introspect section=fleet for team agent shows team peers."""
    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {}
    router = MessageRouter(permissions=perms, agent_registry={})
    router.register_agent("alice", "http://localhost:8401")
    router.register_agent("bob", "http://localhost:8402")
    router.register_agent("carol", "http://localhost:8403")

    app = create_mesh_app(bb, pubsub, router, perms)
    client = TestClient(app)

    app.teams_store.create_team("teamX")
    app.teams_store.add_member("teamX", "alice")
    app.teams_store.add_member("teamX", "bob")

    response = client.get(
        "/mesh/introspect",
        params={"section": "fleet"},
        headers={"X-Agent-ID": "alice"},
    )
    assert response.status_code == 200
    data = response.json()
    ids = [a["id"] for a in data["fleet"]]
    assert "alice" in ids
    assert "bob" in ids
    assert "carol" not in ids

    bb.close()


def test_introspect_operator_fleet_includes_models(tmp_path):
    """Operator's introspect fleet entries include the current model per agent.

    This is how the operator learns about dashboard-initiated model changes
    without restarting — its SYSTEM.md refresh cycle picks up the fresh YAML.
    """
    from unittest.mock import patch

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {}
    router = MessageRouter(permissions=perms, agent_registry={})
    router.register_agent("operator", "http://localhost:8400")
    router.register_agent("sales", "http://localhost:8401")
    router.register_agent("writer", "http://localhost:8402")

    app = create_mesh_app(bb, pubsub, router, perms)
    client = TestClient(app)

    fake_cfg = {
        "llm": {"default_model": "openai/gpt-4o-mini"},
        "agents": {
            "operator": {"model": "openai/gpt-4o"},
            "sales": {"model": "anthropic/claude-sonnet-4-5"},
            # writer omits model → falls back to default_model
            "writer": {},
        },
    }
    with patch("src.cli.config._load_config", return_value=fake_cfg):
        response = client.get(
            "/mesh/introspect",
            params={"section": "fleet"},
            headers={"X-Agent-ID": "operator"},
        )
    assert response.status_code == 200
    fleet = {a["id"]: a for a in response.json()["fleet"]}
    assert fleet["operator"]["model"] == "openai/gpt-4o"
    assert fleet["sales"]["model"] == "anthropic/claude-sonnet-4-5"
    assert fleet["writer"]["model"] == "openai/gpt-4o-mini"

    bb.close()


def test_introspect_operator_fleet_model_falls_back_to_runtime_default(tmp_path):
    """When YAML has no llm.default_model, fleet entries show the agent
    runtime default (matches src/agent/__main__.py), not empty string."""
    from unittest.mock import patch

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {}
    router = MessageRouter(permissions=perms, agent_registry={})
    router.register_agent("writer", "http://localhost:8401")

    app = create_mesh_app(bb, pubsub, router, perms)
    client = TestClient(app)

    # Config has agents but no llm.default_model set
    fake_cfg = {"agents": {"writer": {}}}
    with patch("src.cli.config._load_config", return_value=fake_cfg):
        response = client.get(
            "/mesh/introspect",
            params={"section": "fleet"},
            headers={"X-Agent-ID": "operator"},
        )
    assert response.status_code == 200
    fleet = {a["id"]: a for a in response.json()["fleet"]}
    assert fleet["writer"]["model"] == "openai/gpt-4o-mini"

    bb.close()


def test_introspect_non_operator_fleet_omits_model(tmp_path):
    """Non-operator agents don't see peer models — keeps their context lean."""

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {}
    router = MessageRouter(permissions=perms, agent_registry={})
    router.register_agent("sales", "http://localhost:8401")

    app = create_mesh_app(bb, pubsub, router, perms)
    client = TestClient(app)

    response = client.get(
        "/mesh/introspect",
        params={"section": "fleet"},
        headers={"X-Agent-ID": "sales"},
    )
    assert response.status_code == 200
    for entry in response.json()["fleet"]:
        assert "model" not in entry

    bb.close()


def test_introspect_returns_budget_when_cost_tracker_present(tmp_path):
    """GET /mesh/introspect section=budget returns budget info."""
    from src.host.costs import CostTracker

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {}
    router = MessageRouter(permissions=perms, agent_registry={})
    cost_tracker = CostTracker(db_path=str(tmp_path / "costs.db"))
    cost_tracker.budgets = {"alice": {"daily_usd": 5.0, "monthly_usd": 100.0}}

    app = create_mesh_app(bb, pubsub, router, perms, cost_tracker=cost_tracker)
    client = TestClient(app)

    response = client.get(
        "/mesh/introspect",
        params={"section": "budget"},
        headers={"X-Agent-ID": "alice"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "budget" in data
    assert data["budget"]["daily_limit"] == 5.0
    assert data["budget"]["monthly_limit"] == 100.0
    assert data["budget"]["allowed"] is True

    bb.close()


def test_introspect_all_returns_multiple_sections(tmp_path):
    """GET /mesh/introspect section=all returns permissions + fleet."""
    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {
        "alice": AgentPermissions(
            agent_id="alice",
            blackboard_read=["*"],
        ),
    }
    router = MessageRouter(permissions=perms, agent_registry={})
    router.register_agent("alice", "http://localhost:8401")

    app = create_mesh_app(bb, pubsub, router, perms)
    client = TestClient(app)

    response = client.get(
        "/mesh/introspect",
        params={"section": "all"},
        headers={"X-Agent-ID": "alice"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "permissions" in data
    assert "fleet" in data
    # alice can always see herself even without can_message patterns
    assert len(data["fleet"]) == 1
    assert data["fleet"][0]["id"] == "alice"

    bb.close()


def test_introspect_unknown_agent_gets_default_perms(tmp_path):
    """Introspect for unregistered agent falls back to default permissions."""
    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {}
    router = MessageRouter(permissions=perms, agent_registry={})

    app = create_mesh_app(bb, pubsub, router, perms)
    client = TestClient(app)

    response = client.get(
        "/mesh/introspect",
        params={"section": "permissions"},
        headers={"X-Agent-ID": "unknown_agent"},
    )
    assert response.status_code == 200
    data = response.json()
    # Should still return a permissions dict (deny-all defaults)
    assert "permissions" in data
    assert data["permissions"]["blackboard_read"] == []

    bb.close()


def test_introspect_returns_cron_for_agent(tmp_path):
    """GET /mesh/introspect section=cron returns jobs for the requesting agent."""
    from src.host.cron import CronScheduler

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {}
    router = MessageRouter(permissions=perms, agent_registry={})
    cron = CronScheduler(config_path=str(tmp_path / "cron.json"))
    cron.add_job(agent="alice", schedule="*/15 * * * *", message="heartbeat", heartbeat=True)
    cron.add_job(agent="bob", schedule="0 9 * * *", message="morning")

    app = create_mesh_app(bb, pubsub, router, perms, cron_scheduler=cron)
    client = TestClient(app)

    response = client.get(
        "/mesh/introspect",
        params={"section": "cron"},
        headers={"X-Agent-ID": "alice"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "cron" in data
    # Only alice's jobs should appear
    assert len(data["cron"]) == 1
    assert data["cron"][0]["agent"] == "alice"
    assert data["cron"][0]["heartbeat"] is True

    bb.close()


def test_introspect_health_returns_none_for_unmonitored_agent(tmp_path):
    """GET /mesh/introspect section=health returns null for unknown agent."""
    from unittest.mock import MagicMock

    from src.host.health import HealthMonitor

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {}
    router = MessageRouter(permissions=perms, agent_registry={})
    health = HealthMonitor(
        runtime=MagicMock(), transport=MagicMock(), router=router,
    )

    app = create_mesh_app(bb, pubsub, router, perms, health_monitor=health)
    client = TestClient(app)

    response = client.get(
        "/mesh/introspect",
        params={"section": "health"},
        headers={"X-Agent-ID": "nonexistent"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "health" in data
    assert data["health"] is None

    bb.close()


# ── PubSub Project Isolation Tests ─────────────────────────────


def test_publish_event_project_isolation(tmp_path):
    """Project agents can only publish to topics prefixed with their project."""
    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {
        "alice": AgentPermissions(
            agent_id="alice", can_message=["*"],
            can_publish=["*"], can_subscribe=["*"],
            blackboard_read=["*"], blackboard_write=["*"],
            allowed_apis=[],
        ),
    }
    router = MessageRouter(permissions=perms, agent_registry={})
    app = create_mesh_app(bb, pubsub, router, perms)
    app.teams_store.create_team("sales")
    app.teams_store.add_member("sales", "alice")
    client = TestClient(app)

    # Publishing to project-scoped topic should succeed
    resp = client.post("/mesh/publish", json={
        "topic": "teams/sales/research_done",
        "source": "alice",
        "payload": {"msg": "done"},
    })
    assert resp.status_code == 200

    # Publishing to wrong project prefix should fail
    resp = client.post("/mesh/publish", json={
        "topic": "teams/engineering/research_done",
        "source": "alice",
        "payload": {},
    })
    assert resp.status_code == 403

    # Publishing to raw topic (no project prefix) should fail for project agents
    resp = client.post("/mesh/publish", json={
        "topic": "research_done",
        "source": "alice",
        "payload": {},
    })
    assert resp.status_code == 403

    bb.close()


def test_subscribe_project_isolation(tmp_path):
    """Project agents can only subscribe to topics prefixed with their project."""
    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {
        "bob": AgentPermissions(
            agent_id="bob", can_message=["*"],
            can_publish=["*"], can_subscribe=["*"],
            blackboard_read=["*"], blackboard_write=["*"],
            allowed_apis=[],
        ),
    }
    router = MessageRouter(permissions=perms, agent_registry={})
    app = create_mesh_app(bb, pubsub, router, perms)
    app.teams_store.create_team("engineering")
    app.teams_store.add_member("engineering", "bob")
    client = TestClient(app)

    # Subscribing to project-scoped topic should succeed
    resp = client.post("/mesh/subscribe", params={
        "topic": "teams/engineering/deploy_ready",
        "agent_id": "bob",
    })
    assert resp.status_code == 200

    # Subscribing to other project's topic should fail
    resp = client.post("/mesh/subscribe", params={
        "topic": "teams/sales/new_lead",
        "agent_id": "bob",
    })
    assert resp.status_code == 403

    bb.close()


def test_solo_agent_prefix_locked_to_own_namespace(tmp_path):
    """Solo agents = team-of-one (ratified #5): a teamless worker is
    prefix-locked to ``teams/{its-own-id}/`` instead of skipping the
    team-prefix gate — raw topics are 403, its own namespace works."""
    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {
        "solo": AgentPermissions(
            agent_id="solo", can_message=["*"],
            can_publish=["*"], can_subscribe=["*"],
            blackboard_read=["*"], blackboard_write=["*"],
            allowed_apis=[],
        ),
    }
    router = MessageRouter(permissions=perms, agent_registry={})
    # solo is not in any team — the app's empty store resolves None, so
    # the effective team falls back to the agent's own id.
    app = create_mesh_app(bb, pubsub, router, perms)
    client = TestClient(app)

    resp = client.post("/mesh/publish", json={
        "topic": "research_done",
        "source": "solo",
        "payload": {},
    })
    assert resp.status_code == 403

    resp = client.post("/mesh/publish", json={
        "topic": "teams/solo/research_done",
        "source": "solo",
        "payload": {},
    })
    assert resp.status_code == 200

    resp = client.post("/mesh/subscribe", params={
        "topic": "updates",
        "agent_id": "solo",
    })
    assert resp.status_code == 403

    resp = client.post("/mesh/subscribe", params={
        "topic": "teams/solo/updates",
        "agent_id": "solo",
    })
    assert resp.status_code == 200

    bb.close()


# ── Blackboard Claim (CAS) Tests ──────────────────────────────


def test_claim_endpoint_success(tmp_path):
    """Claim endpoint succeeds when version matches."""
    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {
        "agent1": AgentPermissions(
            agent_id="agent1", can_message=[],
            blackboard_read=["*"], blackboard_write=["*"],
            allowed_apis=[],
        ),
    }
    router = MessageRouter(permissions=perms, agent_registry={})
    app = create_mesh_app(bb, pubsub, router, perms)
    client = TestClient(app)

    # Write initial value
    client.put("/mesh/blackboard/tasks/t1", params={"agent_id": "agent1"},
               json={"status": "pending"})

    # Claim with correct version
    resp = client.post("/mesh/blackboard/claim", json={
        "agent_id": "agent1",
        "key": "tasks/t1",
        "value": {"status": "claimed"},
        "expected_version": 1,
    })
    assert resp.status_code == 200
    assert resp.json()["version"] == 2

    bb.close()


def test_claim_endpoint_conflict(tmp_path):
    """Claim endpoint returns 409 on version mismatch."""
    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {
        "agent1": AgentPermissions(
            agent_id="agent1", can_message=[],
            blackboard_read=["*"], blackboard_write=["*"],
            allowed_apis=[],
        ),
    }
    router = MessageRouter(permissions=perms, agent_registry={})
    app = create_mesh_app(bb, pubsub, router, perms)
    client = TestClient(app)

    client.put("/mesh/blackboard/tasks/t1", params={"agent_id": "agent1"},
               json={"status": "pending"})
    # Update to version 2
    client.put("/mesh/blackboard/tasks/t1", params={"agent_id": "agent1"},
               json={"status": "updated"})

    # Claim with stale version 1 should fail
    resp = client.post("/mesh/blackboard/claim", json={
        "agent_id": "agent1",
        "key": "tasks/t1",
        "value": {"status": "claimed"},
        "expected_version": 1,
    })
    assert resp.status_code == 409

    bb.close()


# ── Blackboard Watch Endpoint Tests ───────────────────────────


def test_watch_blackboard_endpoint(tmp_path):
    """Watch endpoint registers a glob pattern."""
    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {
        "watcher": AgentPermissions(
            agent_id="watcher", can_message=[],
            blackboard_read=["tasks/*"], blackboard_write=[],
            allowed_apis=[],
        ),
    }
    router = MessageRouter(permissions=perms, agent_registry={})
    app = create_mesh_app(bb, pubsub, router, perms)
    client = TestClient(app)

    resp = client.post("/mesh/blackboard/watch", json={
        "agent_id": "watcher",
        "pattern": "tasks/*",
    })
    assert resp.status_code == 200
    assert resp.json()["watching"] is True

    # Verify watcher was registered
    assert "watcher" in bb.get_watchers_for_key("tasks/foo")

    bb.close()


def test_watch_blackboard_permission_denied(tmp_path):
    """Watch endpoint rejects patterns the agent can't read."""
    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {
        "limited": AgentPermissions(
            agent_id="limited", can_message=[],
            blackboard_read=["context/*"], blackboard_write=[],
            allowed_apis=[],
        ),
    }
    router = MessageRouter(permissions=perms, agent_registry={})
    app = create_mesh_app(bb, pubsub, router, perms)
    client = TestClient(app)

    resp = client.post("/mesh/blackboard/watch", json={
        "agent_id": "limited",
        "pattern": "tasks/*",  # limited can't read tasks/*
    })
    assert resp.status_code == 403

    bb.close()


# ── Project Cost Endpoint Tests ───────────────────────────────


def test_project_cost_endpoint(tmp_path):
    """Project cost endpoint returns aggregated spend."""
    from src.host.costs import CostTracker
    from src.host.teams import TeamStore

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {}
    router = MessageRouter(permissions=perms, agent_registry={})
    tracker = CostTracker(db_path=str(tmp_path / "costs.db"))
    store = TeamStore(db_path=":memory:")
    store.create_team("teamA")
    store.add_member("teamA", "alice")
    store.add_member("teamA", "bob")
    store.set_budget("teamA", 50.0, 500.0)
    tracker.set_team_store(store)
    tracker.track("alice", "openai/gpt-4o-mini", 1000, 500)
    tracker.track("bob", "openai/gpt-4o-mini", 2000, 1000)

    app = create_mesh_app(
        bb, pubsub, router, perms, cost_tracker=tracker, teams_store=store,
    )
    client = TestClient(app)

    resp = client.get("/mesh/costs/team/teamA", params={"period": "today"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["team"] == "teamA"
    assert data["total_tokens"] == 4500
    assert data["total_cost"] > 0
    assert len(data["agents"]) == 2
    assert data["daily_limit"] == 50.0

    bb.close()
    tracker.close()


# ── Register Auto-Subscription Scoping Tests ──────────────────


def test_register_scopes_subscriptions(tmp_path):
    """register_agent auto-subscribes with project-scoped topic names."""
    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {
        "alice": AgentPermissions(
            agent_id="alice", can_message=["*"],
            can_publish=["*"],
            can_subscribe=["research_complete", "deploy_ready"],
            blackboard_read=["*"], blackboard_write=["*"],
            allowed_apis=[],
        ),
    }
    router = MessageRouter(permissions=perms, agent_registry={})
    app = create_mesh_app(bb, pubsub, router, perms)
    app.teams_store.create_team("sales")
    app.teams_store.add_member("sales", "alice")
    client = TestClient(app)

    resp = client.post("/mesh/register", json={
        "agent_id": "alice", "capabilities": [], "port": 8401,
    })
    assert resp.status_code == 200

    # Topics should be scoped
    assert "alice" in pubsub.get_subscribers("teams/sales/research_complete")
    assert "alice" in pubsub.get_subscribers("teams/sales/deploy_ready")
    # Should NOT be subscribed to raw topic
    assert "alice" not in pubsub.get_subscribers("research_complete")

    bb.close()


# ── Watcher Notification on CAS Claim Tests ───────────────────


def test_claim_endpoint_notifies_watchers(tmp_path):
    """CAS claim triggers watcher notification via lane_manager."""
    from unittest.mock import AsyncMock, MagicMock

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {
        "agent1": AgentPermissions(
            agent_id="agent1", can_message=[],
            blackboard_read=["*"], blackboard_write=["*"],
            allowed_apis=[],
        ),
    }
    router = MessageRouter(permissions=perms, agent_registry={})

    # Set up mock lane_manager and dispatch_loop
    mock_lane = MagicMock()
    mock_lane.enqueue = AsyncMock()
    mock_loop = MagicMock()

    app = create_mesh_app(
        bb, pubsub, router, perms,
        lane_manager=mock_lane, dispatch_loop=mock_loop,
    )
    client = TestClient(app)

    # Register a watcher
    bb.add_watch("watcher_agent", "tasks/*")

    # Write initial value
    client.put("/mesh/blackboard/tasks/t1", params={"agent_id": "agent1"},
               json={"status": "pending"})

    # Claim via CAS
    resp = client.post("/mesh/blackboard/claim", json={
        "agent_id": "agent1",
        "key": "tasks/t1",
        "value": {"status": "claimed"},
        "expected_version": 1,
    })
    assert resp.status_code == 200

    # The steer notification should have been scheduled via
    # asyncio.run_coroutine_threadsafe — verify the watch was registered
    # and the watcher is in the list for this key
    assert "watcher_agent" in bb.get_watchers_for_key("tasks/t1")

    bb.close()


def test_cleanup_removes_watchers(tmp_path):
    """cleanup_agent removes blackboard watchers for the agent."""
    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {}
    router = MessageRouter(permissions=perms, agent_registry={})
    app = create_mesh_app(bb, pubsub, router, perms)

    # Add a watcher
    bb.add_watch("agent1", "tasks/*")
    assert "agent1" in bb.get_watchers_for_key("tasks/foo")

    # Trigger cleanup (as health monitor/dashboard would)
    app.cleanup_agent("agent1")

    # Watcher should be gone
    assert "agent1" not in bb.get_watchers_for_key("tasks/foo")

    bb.close()


def test_cleanup_agent_full(tmp_path):
    """cleanup_agent removes watchers, pubsub subs, lanes, and cron jobs."""
    from unittest.mock import MagicMock

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {}
    router = MessageRouter(permissions=perms, agent_registry={})

    # Create mock lane_manager and cron_scheduler
    lane_manager = MagicMock()
    cron_scheduler = MagicMock()
    cron_scheduler.remove_agent_jobs.return_value = 2

    app = create_mesh_app(
        bb, pubsub, router, perms,
        lane_manager=lane_manager,
        cron_scheduler=cron_scheduler,
    )

    # Set up state to clean
    bb.add_watch("agent1", "tasks/*")
    pubsub.subscribe("topic1", "agent1")
    pubsub.subscribe("topic2", "agent1")

    assert "agent1" in bb.get_watchers_for_key("tasks/foo")
    assert "agent1" in pubsub.get_subscribers("topic1")

    # Trigger full cleanup
    app.cleanup_agent("agent1")

    # All state should be cleaned
    assert "agent1" not in bb.get_watchers_for_key("tasks/foo")
    assert "agent1" not in pubsub.get_subscribers("topic1")
    assert "agent1" not in pubsub.get_subscribers("topic2")
    lane_manager.remove_lane.assert_called_once_with("agent1")
    cron_scheduler.remove_agent_jobs.assert_called_once_with("agent1")

    bb.close()


def test_delete_blackboard_entry(tmp_path):
    """DELETE /mesh/blackboard/{key} removes the entry."""
    from starlette.testclient import TestClient

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {"agent1": AgentPermissions(
        agent_id="agent1", blackboard_read=["*"], blackboard_write=["*"],
    )}
    router = MessageRouter(permissions=perms, agent_registry={})
    app = create_mesh_app(bb, pubsub, router, perms)
    client = TestClient(app)

    # Write an entry
    client.put("/mesh/blackboard/test/key1", params={"agent_id": "agent1"}, json={"data": "hello"})
    assert bb.read("test/key1") is not None

    # Delete it
    resp = client.delete("/mesh/blackboard/test/key1", params={"agent_id": "agent1"})
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True

    # Verify it's gone
    assert bb.read("test/key1") is None

    bb.close()


def test_delete_rejects_history_namespace(tmp_path):
    """DELETE endpoint blocks deletion from history/ namespace, including project-scoped."""
    from starlette.testclient import TestClient

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {"agent1": AgentPermissions(
        agent_id="agent1", blackboard_read=["*"], blackboard_write=["*"],
    )}
    router = MessageRouter(permissions=perms, agent_registry={})
    app = create_mesh_app(bb, pubsub, router, perms)
    client = TestClient(app)

    # Direct history key
    resp = client.delete("/mesh/blackboard/history/item1", params={"agent_id": "agent1"})
    assert resp.status_code == 400
    assert "history" in resp.json()["detail"].lower()

    # Project-scoped history key
    resp = client.delete(
        "/mesh/blackboard/teams/myproject/history/item1",
        params={"agent_id": "agent1"},
    )
    assert resp.status_code == 400
    assert "history" in resp.json()["detail"].lower()

    bb.close()


def test_write_blackboard_ttl_validation(tmp_path):
    """PUT /mesh/blackboard/{key} rejects non-positive TTL."""
    from starlette.testclient import TestClient

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {"agent1": AgentPermissions(
        agent_id="agent1", blackboard_read=["*"], blackboard_write=["*"],
    )}
    router = MessageRouter(permissions=perms, agent_registry={})
    app = create_mesh_app(bb, pubsub, router, perms)
    client = TestClient(app)

    # Zero TTL should be rejected
    resp = client.put(
        "/mesh/blackboard/test/key1",
        params={"agent_id": "agent1", "ttl": "0"},
        json={"data": "hello"},
    )
    assert resp.status_code == 400

    # Negative TTL should be rejected
    resp = client.put(
        "/mesh/blackboard/test/key1",
        params={"agent_id": "agent1", "ttl": "-1"},
        json={"data": "hello"},
    )
    assert resp.status_code == 400

    # Positive TTL should work
    resp = client.put(
        "/mesh/blackboard/test/key1",
        params={"agent_id": "agent1", "ttl": "3600"},
        json={"data": "hello"},
    )
    assert resp.status_code == 200

    bb.close()


# === External API (API-key authenticated) ===


@pytest.fixture
def ext_api_components(tmp_path, monkeypatch):
    """Mesh components with ApiKeyManager for external API tests."""
    from unittest.mock import MagicMock

    from src.host.api_keys import ApiKeyManager
    from src.host.costs import CostTracker
    from src.host.credentials import CredentialVault
    from src.host.health import HealthMonitor

    monkeypatch.delenv("OPENLEGION_API_KEY", raising=False)

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()

    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {}

    router = MessageRouter(permissions=perms, agent_registry={})
    router.register_agent("card-agent", "http://localhost:8401")

    vault = CredentialVault()

    import src.host.credentials as cred_mod
    monkeypatch.setattr(cred_mod, "_persist_to_env", lambda *a, **kw: None)
    monkeypatch.setattr(cred_mod, "_remove_from_env", lambda *a, **kw: None)

    api_key_mgr = ApiKeyManager(config_path=str(tmp_path / "api_keys.json"))
    _key_id, raw_key = api_key_mgr.create_key("test-key")

    runtime_mock = MagicMock()
    transport_mock = MagicMock()
    router_mock_for_health = MagicMock()
    health_monitor = HealthMonitor(
        runtime=runtime_mock, transport=transport_mock, router=router_mock_for_health,
    )
    health_monitor.register("card-agent")
    health_monitor.agents["card-agent"].status = "healthy"

    cost_tracker = CostTracker(db_path=str(tmp_path / "costs.db"))

    lane_manager = MagicMock()
    lane_manager.get_status.return_value = {
        "card-agent": {"queued": 0, "pending": 0, "busy": False},
    }

    app = create_mesh_app(
        bb, pubsub, router, perms,
        credential_vault=vault,
        health_monitor=health_monitor,
        cost_tracker=cost_tracker,
        lane_manager=lane_manager,
        api_key_manager=api_key_mgr,
    )
    client = TestClient(app)

    yield {
        "client": client, "vault": vault, "bb": bb,
        "cost_tracker": cost_tracker, "health_monitor": health_monitor,
        "raw_key": raw_key, "api_key_manager": api_key_mgr,
    }

    cost_tracker.close()
    bb.close()


def _api_headers(key: str) -> dict:
    return {"X-API-Key": key}


def test_ext_store_credential(ext_api_components):
    """POST /mesh/credentials stores a credential and returns a handle."""
    client = ext_api_components["client"]
    h = _api_headers(ext_api_components["raw_key"])
    resp = client.post(
        "/mesh/credentials",
        json={"name": "company_sess1_ssn", "value": "123-45-6789"},
        headers=h,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["stored"] is True
    assert data["handle"] == "$CRED{company_sess1_ssn}"
    assert data["name"] == "company_sess1_ssn"
    assert "123-45-6789" not in str(data)


def test_ext_store_credential_missing_fields(ext_api_components):
    """POST /mesh/credentials with missing name/value returns 400."""
    client = ext_api_components["client"]
    h = _api_headers(ext_api_components["raw_key"])
    resp = client.post("/mesh/credentials", json={"name": "", "value": ""}, headers=h)
    assert resp.status_code == 400


def test_ext_store_credential_invalid_name(ext_api_components):
    """POST /mesh/credentials with invalid name returns 400."""
    client = ext_api_components["client"]
    h = _api_headers(ext_api_components["raw_key"])
    resp = client.post("/mesh/credentials", json={"name": "has spaces!", "value": "abc"}, headers=h)
    assert resp.status_code == 400


def test_ext_store_credential_blocks_system_creds(ext_api_components):
    """POST /mesh/credentials rejects system credential names."""
    client = ext_api_components["client"]
    h = _api_headers(ext_api_components["raw_key"])
    resp = client.post("/mesh/credentials", json={"name": "openai_api_key", "value": "sk-evil"}, headers=h)
    assert resp.status_code == 403


def test_ext_store_credential_no_api_key(ext_api_components):
    """POST /mesh/credentials without X-API-Key returns 401."""
    client = ext_api_components["client"]
    resp = client.post("/mesh/credentials", json={"name": "test", "value": "val"})
    assert resp.status_code == 401


def test_ext_store_credential_wrong_api_key(ext_api_components):
    """POST /mesh/credentials with wrong API key returns 401."""
    client = ext_api_components["client"]
    resp = client.post(
        "/mesh/credentials",
        json={"name": "test", "value": "val"},
        headers=_api_headers("wrong-key"),
    )
    assert resp.status_code == 401


def test_ext_remove_credential(ext_api_components):
    """DELETE /mesh/credentials/{name} removes a stored credential."""
    client = ext_api_components["client"]
    vault = ext_api_components["vault"]
    h = _api_headers(ext_api_components["raw_key"])
    vault.add_credential("company_sess1_ssn", "123-45-6789")
    resp = client.delete("/mesh/credentials/company_sess1_ssn", headers=h)
    assert resp.status_code == 200
    assert resp.json()["removed"] is True
    assert not vault.has_credential("company_sess1_ssn")


def test_ext_remove_credential_not_found(ext_api_components):
    """DELETE /mesh/credentials/{name} returns 404 for missing credential."""
    client = ext_api_components["client"]
    h = _api_headers(ext_api_components["raw_key"])
    resp = client.delete("/mesh/credentials/nonexistent", headers=h)
    assert resp.status_code == 404


def test_ext_remove_credential_blocks_system(ext_api_components):
    """DELETE /mesh/credentials/{name} rejects system credential names."""
    client = ext_api_components["client"]
    h = _api_headers(ext_api_components["raw_key"])
    resp = client.delete("/mesh/credentials/anthropic_api_key", headers=h)
    assert resp.status_code == 403


def test_ext_list_credentials(ext_api_components):
    """GET /mesh/credentials lists agent-tier credential names."""
    client = ext_api_components["client"]
    vault = ext_api_components["vault"]
    h = _api_headers(ext_api_components["raw_key"])
    vault.add_credential("company_sess1_ssn", "secret1")
    vault.add_credential("company_sess1_income", "secret2")
    resp = client.get("/mesh/credentials", headers=h)
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] >= 2
    assert "company_sess1_ssn" in data["credentials"]
    assert "company_sess1_income" in data["credentials"]


def test_ext_credential_exists(ext_api_components):
    """GET /mesh/credentials/{name}/exists checks credential existence."""
    client = ext_api_components["client"]
    vault = ext_api_components["vault"]
    h = _api_headers(ext_api_components["raw_key"])
    vault.add_credential("company_sess1_ssn", "secret")
    resp = client.get("/mesh/credentials/company_sess1_ssn/exists", headers=h)
    assert resp.status_code == 200
    assert resp.json()["exists"] is True
    resp = client.get("/mesh/credentials/nonexistent/exists", headers=h)
    assert resp.status_code == 200
    assert resp.json()["exists"] is False


def test_ext_agent_status(ext_api_components):
    """GET /mesh/agents/{id}/ext-status returns health, queue, and budget."""
    client = ext_api_components["client"]
    h = _api_headers(ext_api_components["raw_key"])
    resp = client.get("/mesh/agents/card-agent/ext-status", headers=h)
    assert resp.status_code == 200
    data = resp.json()
    assert data["agent_id"] == "card-agent"
    assert data["health"]["status"] == "healthy"
    assert data["queue"]["busy"] is False
    assert data["queue"]["queued"] == 0
    assert "budget" in data
    assert data["budget"]["allowed"] is True


def test_ext_agent_status_not_found(ext_api_components):
    """GET /mesh/agents/{id}/ext-status returns 404 for unknown agent."""
    client = ext_api_components["client"]
    h = _api_headers(ext_api_components["raw_key"])
    resp = client.get("/mesh/agents/nonexistent/ext-status", headers=h)
    assert resp.status_code == 404


def test_ext_api_not_configured(tmp_path, monkeypatch):
    """External API returns 503 when no API keys exist."""
    monkeypatch.delenv("OPENLEGION_API_KEY", raising=False)

    from src.host.api_keys import ApiKeyManager
    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {}
    router = MessageRouter(permissions=perms, agent_registry={})
    empty_mgr = ApiKeyManager(config_path=str(tmp_path / "empty_keys.json"))

    app = create_mesh_app(bb, pubsub, router, perms, api_key_manager=empty_mgr)
    client = TestClient(app)

    resp = client.get("/mesh/credentials", headers={"X-API-Key": "anything"})
    assert resp.status_code == 503

    bb.close()


def test_ext_store_and_remove_lifecycle(ext_api_components):
    """Full lifecycle: store, verify exists, list, remove, verify gone."""
    client = ext_api_components["client"]
    h = _api_headers(ext_api_components["raw_key"])

    resp = client.post(
        "/mesh/credentials",
        json={"name": "company_abc_ssn", "value": "999-88-7777"},
        headers=h,
    )
    assert resp.status_code == 200
    assert resp.json()["handle"] == "$CRED{company_abc_ssn}"

    resp = client.get("/mesh/credentials/company_abc_ssn/exists", headers=h)
    assert resp.json()["exists"] is True

    resp = client.get("/mesh/credentials", headers=h)
    assert "company_abc_ssn" in resp.json()["credentials"]

    resp = client.delete("/mesh/credentials/company_abc_ssn", headers=h)
    assert resp.status_code == 200

    resp = client.get("/mesh/credentials/company_abc_ssn/exists", headers=h)
    assert resp.json()["exists"] is False


# ── Agent profile endpoint ────────────────────────────────────────────


def test_agent_profile_basic(mesh_components):
    """GET /mesh/agents/{id}/profile returns metadata for a registered agent."""
    client = mesh_components["client"]
    bb = mesh_components["blackboard"]
    pubsub = mesh_components["pubsub"]

    # Register the agent
    client.post("/mesh/register", json={
        "agent_id": "research",
        "capabilities": ["web_search", "memory_save"],
        "port": 8401,
    })

    # Set up some state the profile should reflect
    pubsub.subscribe("teams/teamA/research_complete", "research")
    bb.add_watch("research", "teams/teamA/feedback/*")
    bb.write("teams/teamA/sources/topic-1", {"data": "brief"}, written_by="research")

    # No requesting_agent — falls through to _require_any_auth which is
    # a no-op when no auth_tokens are configured (test fixture).
    resp = client.get("/mesh/agents/research/profile")
    assert resp.status_code == 200
    data = resp.json()
    assert data["agent_id"] == "research"
    # Role is empty string — /mesh/register does not accept a role param
    assert data["role"] == ""
    assert data["capabilities"] == ["web_search", "memory_save"]
    # No agent_teams in fixture, so project prefix is NOT stripped
    assert "teams/teamA/research_complete" in data["subscriptions"]
    assert "teams/teamA/feedback/*" in data["watches"]
    assert "teams/teamA/sources/topic-1" in data["recent_writes"]
    assert data["interface"] is None  # No INTERFACE.md on the container


def test_agent_profile_not_found(mesh_components):
    """GET /mesh/agents/{id}/profile returns 404 for unknown agent."""
    client = mesh_components["client"]
    # No requesting_agent — bypasses can_message check (no auth_tokens in fixture).
    resp = client.get("/mesh/agents/nonexistent/profile")
    assert resp.status_code == 404


def test_agent_profile_permission_denied(tmp_path):
    """GET /mesh/agents/{id}/profile enforces can_message on the VERIFIED
    bearer identity — not the (untrusted) ``requesting_agent`` query param.

    M24 stopped trusting ``requesting_agent``; the gate now keys off the
    token-verified caller. This builds a mesh app WITH ``auth_tokens`` so a
    real verified identity exists, then asserts a caller lacking
    ``can_message`` for the target is denied 403 even with NO
    ``requesting_agent`` param, while a caller that DOES have ``can_message``
    is allowed 200.
    """
    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()

    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {
        "research": AgentPermissions(
            agent_id="research",
            can_message=["mesh"],
            allowed_apis=["anthropic"],
        ),
        # "qualify" can message "mesh" but NOT "research".
        "qualify": AgentPermissions(
            agent_id="qualify",
            can_message=["mesh"],
            allowed_apis=["anthropic"],
        ),
        # "scout" CAN message "research" — used for the positive case.
        "scout": AgentPermissions(
            agent_id="scout",
            can_message=["mesh", "research"],
            allowed_apis=["anthropic"],
        ),
    }

    router = MessageRouter(permissions=perms, agent_registry={})

    # Each agent has its own token — ``/mesh/register`` resolves the
    # registered identity from the bearer (not the supplied agent_id) when
    # auth is on, so each agent must register under its own token.
    tokens = {
        "research": "research-token-xyz",
        "qualify": "qualify-token-xyz",
        "scout": "scout-token-xyz",
    }
    app = create_mesh_app(bb, pubsub, router, perms, auth_tokens=tokens)
    client = TestClient(app)

    # Register target + callers (each authenticated as itself).
    for agent_id, token, port in (
        ("research", "research-token-xyz", 8401),
        ("qualify", "qualify-token-xyz", 8402),
        ("scout", "scout-token-xyz", 8403),
    ):
        reg = client.post(
            "/mesh/register",
            json={"agent_id": agent_id, "capabilities": [], "port": port},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert reg.status_code == 200, reg.text

    # Verified caller "qualify" lacks can_message=["research"] → 403,
    # WITHOUT any requesting_agent param (the param is no longer trusted).
    resp = client.get(
        "/mesh/agents/research/profile",
        headers={"Authorization": "Bearer qualify-token-xyz"},
    )
    assert resp.status_code == 403

    # Spoofing the param does NOT change the outcome — still keyed on the
    # verified bearer identity ("qualify"), so still denied.
    resp = client.get(
        "/mesh/agents/research/profile",
        params={"requesting_agent": "scout"},
        headers={"Authorization": "Bearer qualify-token-xyz"},
    )
    assert resp.status_code == 403

    # Verified caller "scout" HAS can_message=["research"] → allowed 200.
    resp = client.get(
        "/mesh/agents/research/profile",
        headers={"Authorization": "Bearer scout-token-xyz"},
    )
    assert resp.status_code == 200
    assert resp.json()["agent_id"] == "research"


# ── Fix 4: parse_origin_header input validation ─────────────────


class TestParseOriginHeader:
    def test_valid_header(self):
        from src.shared.trace import parse_origin_header
        from src.shared.types import MessageOrigin

        result = parse_origin_header('{"channel":"whatsapp","user":"+1234"}')
        assert isinstance(result, MessageOrigin)
        # Legacy header (no ``kind``) → least-trusted ``kind="agent"``.
        assert result.kind == "agent"
        assert result.channel == "whatsapp"
        assert result.user == "+1234"

    def test_none_returns_none(self):
        from src.shared.trace import parse_origin_header
        assert parse_origin_header(None) is None
        assert parse_origin_header("") is None

    def test_invalid_json_returns_none(self):
        from src.shared.trace import parse_origin_header
        assert parse_origin_header("not-json") is None
        assert parse_origin_header("{") is None

    def test_non_dict_returns_none(self):
        from src.shared.trace import parse_origin_header
        assert parse_origin_header('"just a string"') is None
        assert parse_origin_header("[1, 2, 3]") is None

    def test_missing_fields_returns_none(self):
        from src.shared.trace import parse_origin_header
        assert parse_origin_header('{"channel":"whatsapp"}') is None
        assert parse_origin_header('{"user":"+1"}') is None

    def test_empty_fields_returns_none(self):
        from src.shared.trace import parse_origin_header
        # Legacy header (no ``kind``) still requires non-empty channel/user.
        assert parse_origin_header('{"channel":"","user":"+1"}') is None
        assert parse_origin_header('{"channel":"whatsapp","user":""}') is None

    def test_non_string_fields_returns_none(self):
        from src.shared.trace import parse_origin_header
        assert parse_origin_header('{"channel":1,"user":"+1"}') is None
        assert parse_origin_header('{"channel":"whatsapp","user":123}') is None

    def test_extra_fields_stripped(self):
        from src.shared.trace import parse_origin_header
        from src.shared.types import MessageOrigin

        result = parse_origin_header(
            '{"channel":"whatsapp","user":"+1","extra":"dropped","nested":{}}'
        )
        assert isinstance(result, MessageOrigin)
        # Only ``kind`` / ``channel`` / ``user`` survive.
        assert result.channel == "whatsapp"
        assert result.user == "+1"
        # No way for arbitrary keys to leak onto the model.
        assert not hasattr(result, "extra")
        assert not hasattr(result, "nested")

    def test_oversized_raw_header_returns_none(self):
        from src.shared.trace import parse_origin_header
        # Raw header >512 bytes is rejected before JSON parsing
        big = '{"channel":"whatsapp","user":"' + ("x" * 600) + '"}'
        assert parse_origin_header(big) is None

    def test_oversized_user_field_returns_none(self):
        from src.shared.trace import parse_origin_header
        # Even if the raw blob fits under 512, a >128 char user is dropped
        raw = '{"channel":"whatsapp","user":"' + ("x" * 200) + '"}'
        # Raw is ~230 chars, under the 512 cap, so length-per-field check kicks in
        assert parse_origin_header(raw) is None

    def test_oversized_channel_field_returns_none(self):
        from src.shared.trace import parse_origin_header
        raw = '{"channel":"' + ("c" * 50) + '","user":"+1"}'
        assert parse_origin_header(raw) is None


# ── Fix 4: /mesh/wake origin header propagation ───────────────────


def _wake_test_app(tmp_path):
    """Build a minimal mesh app wired with a real dispatch loop + mock lane.

    Returns ``(client, captured_enqueue_kwargs)`` where captured_enqueue_kwargs
    is a list the test can inspect after POSTing to /mesh/wake.
    """
    import asyncio
    import threading

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {
        "operator": AgentPermissions(
            agent_id="operator", can_message=["*"],
            blackboard_read=["*"], blackboard_write=["*"],
            allowed_apis=[],
        ),
        "worker": AgentPermissions(
            agent_id="worker", can_message=["chef"],
            blackboard_read=[], blackboard_write=[],
            allowed_apis=[],
        ),
    }
    router = MessageRouter(permissions=perms, agent_registry={})
    router.register_agent("chef", "http://fake", role="chef")

    captured: list[dict] = []

    class _FakeLane:
        async def enqueue(self, agent, message, **kwargs):
            captured.append({"agent": agent, "message": message, **kwargs})
            return ""

    lane_manager = _FakeLane()

    # Real loop running in a daemon thread so run_coroutine_threadsafe works.
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def _run():
        asyncio.set_event_loop(loop)
        loop.call_soon(ready.set)
        loop.run_forever()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    ready.wait()

    app = create_mesh_app(
        bb, pubsub, router, perms,
        lane_manager=lane_manager,
        dispatch_loop=loop,
    )
    # Mesh endpoint resolves the caller agent from auth tokens; bypass that
    # by pinning a request-scoped agent_id via _extract_verified_agent_id's
    # fallback behaviour — no auth_tokens on the app means no verification,
    # so the endpoint falls back to "mesh" as caller. That's fine for this test.
    client = TestClient(app)
    return client, captured, loop, bb


def test_mesh_wake_propagates_origin_header(tmp_path):
    """POST /mesh/wake with X-Origin passes parsed origin + auto_notify=True to enqueue."""
    import json
    import time

    client, captured, loop, bb = _wake_test_app(tmp_path)
    try:
        resp = client.post(
            "/mesh/wake",
            params={"target": "chef", "message": "check inbox"},
            headers={
                "x-origin": json.dumps({"channel": "whatsapp", "user": "+1234"}),
                "X-Agent-ID": "operator",
            },
        )
        assert resp.status_code == 200
        # Give the dispatch loop a moment to run the enqueue coroutine
        deadline = time.monotonic() + 2.0
        while not captured and time.monotonic() < deadline:
            time.sleep(0.01)
        assert captured, "lane_manager.enqueue was never called"
        call = captured[0]
        assert call["agent"] == "chef"
        assert call["mode"] == "followup"
        # ``parse_origin_header`` returns a typed ``MessageOrigin``; legacy
        # ``X-Origin`` headers without a ``kind`` segment default to
        # ``kind="agent"`` (least-trusted).
        from src.shared.types import MessageOrigin
        assert isinstance(call["origin"], MessageOrigin)
        assert call["origin"].kind == "agent"
        assert call["origin"].channel == "whatsapp"
        assert call["origin"].user == "+1234"
        assert call["auto_notify"] is True
    finally:
        loop.call_soon_threadsafe(loop.stop)
        bb.close()


def test_mesh_wake_no_origin_header_disables_auto_notify(tmp_path):
    """POST /mesh/wake without X-Origin downgrades to kind="agent" (Task 2b).

    Auto-notify stays disabled because there's no addressable channel/user
    to notify, but the origin itself is no longer ``None`` — every lane
    payload now carries an explicit ``kind`` from this slice forward.
    """
    import time

    from src.shared.types import MessageOrigin

    client, captured, loop, bb = _wake_test_app(tmp_path)
    try:
        resp = client.post(
            "/mesh/wake",
            params={"target": "chef", "message": "check inbox"},
            headers={"X-Agent-ID": "operator"},
        )
        assert resp.status_code == 200
        deadline = time.monotonic() + 2.0
        while not captured and time.monotonic() < deadline:
            time.sleep(0.01)
        assert captured
        call = captured[0]
        assert isinstance(call["origin"], MessageOrigin)
        assert call["origin"].kind == "agent"
        assert call["origin"].channel == ""
        assert call["origin"].user == ""
        assert call["auto_notify"] is False
    finally:
        loop.call_soon_threadsafe(loop.stop)
        bb.close()


def test_mesh_wake_invalid_origin_header_ignored(tmp_path):
    """POST /mesh/wake with malformed X-Origin downgrades to kind="agent"."""
    import time

    from src.shared.types import MessageOrigin

    client, captured, loop, bb = _wake_test_app(tmp_path)
    try:
        resp = client.post(
            "/mesh/wake",
            params={"target": "chef", "message": "check inbox"},
            headers={"x-origin": "not-json", "X-Agent-ID": "operator"},
        )
        assert resp.status_code == 200
        deadline = time.monotonic() + 2.0
        while not captured and time.monotonic() < deadline:
            time.sleep(0.01)
        assert captured
        assert isinstance(captured[0]["origin"], MessageOrigin)
        assert captured[0]["origin"].kind == "agent"
        assert captured[0]["auto_notify"] is False
    finally:
        loop.call_soon_threadsafe(loop.stop)
        bb.close()


# ── Task 2e: worker → operator synchronous wake is blocked ────────


def _wake_block_app(tmp_path):
    """Build a mesh app wired with a worker (``scout``) + operator.

    Mirrors the Task 1 characterization fixture but factored out so the
    flipped block test and the four positive-path tests share setup.
    Returns ``(client, captured, loop, bb)``.
    """
    import asyncio
    import threading

    from fastapi.testclient import TestClient

    from src.host.mesh import Blackboard, MessageRouter, PubSub
    from src.host.permissions import PermissionMatrix
    from src.host.server import create_mesh_app
    from src.shared.types import AgentPermissions

    bb = Blackboard(db_path=str(tmp_path / "wake_op_bb.db"))
    pubsub = PubSub()

    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {
        "scout": AgentPermissions(
            agent_id="scout",
            can_message=["*"],
            blackboard_read=["teams/growth/*"],
            blackboard_write=["teams/growth/*"],
        ),
        "operator": AgentPermissions(
            agent_id="operator",
            can_message=["*"],
            blackboard_read=["*"],
            blackboard_write=["*"],
        ),
    }

    router = MessageRouter(permissions=perms, agent_registry={})
    router.register_agent("scout", "http://scout:8400", role="scout")
    router.register_agent("operator", "http://operator:8400", role="operator")
    router.register_agent("chef", "http://chef:8400", role="chef")

    captured: list[dict] = []

    class _FakeLane:
        async def enqueue(self, agent, message, **kwargs):
            captured.append({"agent": agent, "message": message, **kwargs})
            return ""

    lane_manager = _FakeLane()

    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def _run():
        asyncio.set_event_loop(loop)
        loop.call_soon(ready.set)
        loop.run_forever()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    ready.wait()

    app = create_mesh_app(
        bb, pubsub, router, perms,
        lane_manager=lane_manager,
        dispatch_loop=loop,
    )
    client = TestClient(app)
    return client, captured, loop, bb


def test_project_worker_cannot_wake_operator_directly(tmp_path):
    """Workers (origin kind=agent) get HTTP 403 from ``/mesh/wake?target=operator``.

    Post-Task-2e expectation: a worker agent with ``can_message=["*"]``
    can no longer synchronously steer the operator. It must hand off via
    ``global/tasks/operator/<id>`` (Task 0 hotfix path); the operator
    polls those records on heartbeat. Permissions still gate the
    request, but the wake-the-operator path is gated on top.
    """
    import time

    client, captured, loop, bb = _wake_block_app(tmp_path)
    try:
        resp = client.post(
            "/mesh/wake",
            params={"target": "operator", "message": "task done"},
            headers={"X-Agent-ID": "scout"},
        )
        assert resp.status_code == 403, resp.text
        body = resp.json()
        assert "Hand off via" in body.get("detail", ""), body

        # The lane must NOT have been driven — the block fires before
        # ``lane_manager.enqueue`` runs.
        deadline = time.monotonic() + 0.5
        while not captured and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not captured, "operator lane was driven despite 403"
    finally:
        loop.call_soon_threadsafe(loop.stop)
        bb.close()


def test_operator_self_wake_not_blocked(tmp_path):
    """The operator can still wake itself (caller == "operator")."""
    import time

    client, captured, loop, bb = _wake_block_app(tmp_path)
    try:
        resp = client.post(
            "/mesh/wake",
            params={"target": "operator", "message": "self-resume"},
            headers={"X-Agent-ID": "operator"},
        )
        assert resp.status_code == 200, resp.text
        deadline = time.monotonic() + 2.0
        while not captured and time.monotonic() < deadline:
            time.sleep(0.01)
        assert captured, "operator self-wake was not enqueued"
        assert captured[0]["agent"] == "operator"
    finally:
        loop.call_soon_threadsafe(loop.stop)
        bb.close()


def test_human_origin_wake_operator_not_blocked(tmp_path, monkeypatch):
    """A real human action (paired channel + paired user) can wake the operator.

    ``_validated_origin`` resolves ``kind="human"`` after rechecking the
    pairing record, so the Task 2e block must not fire.
    """
    import json
    import time

    from src.host.server import _invalidate_pairing_cache

    # Pairing record: telegram user "42" is the paired owner.
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "telegram_paired.json").write_text(
        json.dumps({"owner": 42, "allowed": [42]}),
    )
    _patch_pairing_project_root(tmp_path, monkeypatch)
    _invalidate_pairing_cache()

    client, captured, loop, bb = _wake_block_app(tmp_path)
    try:
        # ``scout`` is the (untrusted) caller, but the typed origin
        # carries a verified human claim (telegram/42 is paired). The
        # block must defer to the human-origin path.
        resp = client.post(
            "/mesh/wake",
            params={"target": "operator", "message": "ping"},
            headers={
                "X-Agent-ID": "scout",
                "x-origin": json.dumps(
                    {"kind": "human", "channel": "telegram", "user": "42"},
                ),
            },
        )
        assert resp.status_code == 200, resp.text
        deadline = time.monotonic() + 2.0
        while not captured and time.monotonic() < deadline:
            time.sleep(0.01)
        assert captured, "human-origin wake was not enqueued"
        assert captured[0]["agent"] == "operator"
        assert captured[0]["origin"].kind == "human"
    finally:
        loop.call_soon_threadsafe(loop.stop)
        bb.close()
        _invalidate_pairing_cache()


def test_worker_can_still_wake_non_operator_target(tmp_path):
    """Control: the Task 2e block applies only to ``target == "operator"``.

    A worker waking a peer (here ``chef``) is still allowed by the
    existing ``can_message`` gate.
    """
    import time

    client, captured, loop, bb = _wake_block_app(tmp_path)
    try:
        resp = client.post(
            "/mesh/wake",
            params={"target": "chef", "message": "review this"},
            headers={"X-Agent-ID": "scout"},
        )
        assert resp.status_code == 200, resp.text
        deadline = time.monotonic() + 2.0
        while not captured and time.monotonic() < deadline:
            time.sleep(0.01)
        assert captured, "worker → chef wake was not enqueued"
        assert captured[0]["agent"] == "chef"
    finally:
        loop.call_soon_threadsafe(loop.stop)
        bb.close()


class _LoopbackClient:
    """Stand-in for ``Request.client`` with a configurable ``host`` IP."""

    def __init__(self, host: str) -> None:
        self.host = host


class _FakeRequestWithClient:
    """``_FakeRequest`` superset that also carries a ``client`` attribute.

    ``_is_internal_caller`` reads ``request.client.host`` for the
    loopback check; the bare ``_FakeRequest`` (used by Task 2c origin
    tests) doesn't expose one.
    """

    def __init__(self, headers: dict[str, str], client_host: str | None) -> None:
        self.headers = _ReqHeaders(headers)
        self.client = _LoopbackClient(client_host) if client_host is not None else None


def test_is_internal_caller_predicate():
    """Unit-test the ``_is_internal_caller`` predicate directly.

    Both the ``x-mesh-internal`` header AND a loopback peer must be
    present. The header alone or a loopback peer alone is insufficient.
    """
    from src.host.server import _is_internal_caller

    # Both header + loopback IP → True.
    assert _is_internal_caller(
        _FakeRequestWithClient({"x-mesh-internal": "1"}, "127.0.0.1"),
    )
    assert _is_internal_caller(
        _FakeRequestWithClient({"x-mesh-internal": "1"}, "::1"),
    )
    # Header without loopback peer → False.
    assert not _is_internal_caller(
        _FakeRequestWithClient({"x-mesh-internal": "1"}, "203.0.113.7"),
    )
    # Loopback peer without header → False.
    assert not _is_internal_caller(
        _FakeRequestWithClient({}, "127.0.0.1"),
    )
    # No client at all → False (defensive: ASGI in some test setups
    # sets ``request.client`` to None).
    assert not _is_internal_caller(
        _FakeRequestWithClient({"x-mesh-internal": "1"}, None),
    )
    # Garbage host string → False (ipaddress raises, helper catches).
    assert not _is_internal_caller(
        _FakeRequestWithClient({"x-mesh-internal": "1"}, "not-an-ip"),
    )


# ── Fix 4: RuntimeContext._handle_notify_origin routing ───────────


def _stub_runtime_with_channel(channel_type: str, channel_obj):
    """Build a minimal RuntimeContext with a mocked channel_manager."""
    from unittest.mock import MagicMock

    from src.cli.runtime import RuntimeContext

    rt = RuntimeContext.__new__(RuntimeContext)
    rt.channel_manager = MagicMock()
    rt.channel_manager._channel_map = {channel_type: channel_obj}
    return rt


class _FakeChannel:
    def __init__(self, has_loop: bool = False):
        from unittest.mock import AsyncMock

        self.sent: list[tuple[str, str]] = []
        self._channel_loop = None
        if has_loop:
            import asyncio as _aio
            self._channel_loop = _aio.get_event_loop()

        async def _send(user_id: str, text: str) -> None:
            self.sent.append((user_id, text))

        self.send_to_user = AsyncMock(side_effect=_send)


class TestHandleNotifyOrigin:
    @pytest.mark.asyncio
    async def test_routes_to_channel_with_agent_label(self):
        from src.shared.types import MessageOrigin

        ch = _FakeChannel()
        rt = _stub_runtime_with_channel("whatsapp", ch)
        await rt._handle_notify_origin(
            MessageOrigin(kind="human", channel="whatsapp", user="+1234"),
            "dinner is ready",
            "chef",
        )
        assert ch.sent == [("+1234", "[chef] dinner is ready")]

    @pytest.mark.asyncio
    async def test_no_agent_label_when_agent_name_empty(self):
        from src.shared.types import MessageOrigin

        ch = _FakeChannel()
        rt = _stub_runtime_with_channel("whatsapp", ch)
        await rt._handle_notify_origin(
            MessageOrigin(kind="human", channel="whatsapp", user="+1234"),
            "raw message",
            "",
        )
        assert ch.sent == [("+1234", "raw message")]

    @pytest.mark.asyncio
    async def test_drops_when_channel_not_connected(self):
        from unittest.mock import MagicMock

        from src.cli.runtime import RuntimeContext
        from src.shared.types import MessageOrigin
        rt = RuntimeContext.__new__(RuntimeContext)
        rt.channel_manager = MagicMock()
        rt.channel_manager._channel_map = {}  # empty
        # No exception raised, no call made
        await rt._handle_notify_origin(
            MessageOrigin(kind="human", channel="discord", user="99"),
            "hi",
            "chef",
        )

    @pytest.mark.asyncio
    async def test_drops_when_channel_manager_missing(self):
        from src.cli.runtime import RuntimeContext
        from src.shared.types import MessageOrigin
        rt = RuntimeContext.__new__(RuntimeContext)
        rt.channel_manager = None
        # Must be a no-op, not raise
        await rt._handle_notify_origin(
            MessageOrigin(kind="human", channel="whatsapp", user="+1"),
            "hi",
            "chef",
        )

    @pytest.mark.asyncio
    async def test_drops_on_invalid_origin(self):
        from src.shared.types import MessageOrigin

        ch = _FakeChannel()
        rt = _stub_runtime_with_channel("whatsapp", ch)
        # Missing user (empty string user)
        await rt._handle_notify_origin(
            MessageOrigin(kind="human", channel="whatsapp", user=""),
            "hi", "chef",
        )
        # Missing channel
        await rt._handle_notify_origin(
            MessageOrigin(kind="human", channel="", user="+1"),
            "hi", "chef",
        )
        # No origin at all
        await rt._handle_notify_origin(None, "hi", "chef")
        assert ch.sent == []

    @pytest.mark.asyncio
    async def test_send_failure_is_caught_not_raised(self):
        from unittest.mock import AsyncMock

        from src.shared.types import MessageOrigin

        ch = _FakeChannel()
        ch.send_to_user = AsyncMock(side_effect=RuntimeError("boom"))
        rt = _stub_runtime_with_channel("whatsapp", ch)
        # Must not raise — the warning is logged and swallowed
        await rt._handle_notify_origin(
            MessageOrigin(kind="human", channel="whatsapp", user="+1"),
            "hi",
            "chef",
        )


# ── Task 2c: server-side channel origin pairing recheck ──────────


class _ReqHeaders:
    """Minimal stand-in for ``Request.headers`` (case-insensitive get)."""

    def __init__(self, mapping: dict[str, str]) -> None:
        self._lower = {k.lower(): v for k, v in mapping.items()}

    def get(self, key: str, default: str | None = None) -> str | None:
        return self._lower.get(key.lower(), default)


class _FakeRequest:
    def __init__(self, headers: dict[str, str], client_host: str | None = None) -> None:
        self.headers = _ReqHeaders(headers)
        self.client = None
        if client_host is not None:
            self.client = type("_Client", (), {"host": client_host})()


def _write_pairing(tmp_path, channel: str, payload: dict | str) -> None:
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    path = cfg_dir / f"{channel}_paired.json"
    if isinstance(payload, str):
        path.write_text(payload)
    else:
        import json as _json
        path.write_text(_json.dumps(payload))


def _patch_pairing_project_root(tmp_path, monkeypatch) -> None:
    from src.cli import config as cli_config
    monkeypatch.setattr(cli_config, "PROJECT_ROOT", tmp_path)


def _origin_header(kind: str, channel: str, user: str) -> str:
    import json as _json
    return _json.dumps({"kind": kind, "channel": channel, "user": user})


class TestValidatedOrigin:
    """Server-side recheck of inbound ``X-Origin`` channel claims (Task 2c)."""

    def setup_method(self):
        from src.host.server import _invalidate_pairing_cache
        _invalidate_pairing_cache()

    def teardown_method(self):
        from src.host.server import _invalidate_pairing_cache
        _invalidate_pairing_cache()

    def test_paired_user_keeps_kind_human(self, tmp_path, monkeypatch):
        from src.host.server import _validated_origin
        from src.shared.types import MessageOrigin

        _patch_pairing_project_root(tmp_path, monkeypatch)
        _write_pairing(tmp_path, "telegram", {"owner": 42, "allowed": [42]})
        req = _FakeRequest({"x-origin": _origin_header("human", "telegram", "42")})
        result = _validated_origin(req)
        assert isinstance(result, MessageOrigin)
        assert result.kind == "human"
        assert result.channel == "telegram"
        assert result.user == "42"

    def test_paired_user_in_allowed_list_keeps_human(self, tmp_path, monkeypatch):
        from src.host.server import _validated_origin

        _patch_pairing_project_root(tmp_path, monkeypatch)
        _write_pairing(tmp_path, "discord", {"owner": "owner-id", "allowed": ["alice", "bob"]})
        req = _FakeRequest({"x-origin": _origin_header("human", "discord", "alice")})
        result = _validated_origin(req)
        assert result.kind == "human"
        assert result.user == "alice"

    def test_unpaired_user_downgraded_to_agent(self, tmp_path, monkeypatch):
        from src.host.server import _validated_origin

        _patch_pairing_project_root(tmp_path, monkeypatch)
        _write_pairing(tmp_path, "telegram", {"owner": 42, "allowed": [42]})
        req = _FakeRequest({"x-origin": _origin_header("human", "telegram", "9999")})
        result = _validated_origin(req)
        assert result.kind == "agent"
        # Channel/user metadata preserved for downstream addressing.
        assert result.channel == "telegram"
        assert result.user == "9999"

    def test_empty_user_downgraded(self, tmp_path, monkeypatch):
        """A typed ``kind="human"`` header with empty user passes the
        parser (``trust_kind=True`` skips the empty-user reject), so the
        validator's own empty-user branch must catch it and downgrade
        — an empty user id cannot be verified against any pairing list.
        """
        from src.host.server import _validated_origin

        _patch_pairing_project_root(tmp_path, monkeypatch)
        _write_pairing(tmp_path, "telegram", {"owner": 42, "allowed": [42]})
        req = _FakeRequest(
            {"x-origin": _origin_header("human", "telegram", "")},
        )
        result = _validated_origin(req)
        assert result is not None
        assert result.kind == "agent"
        assert result.channel == "telegram"
        assert result.user == ""

    def test_missing_pairing_file_downgrades(self, tmp_path, monkeypatch):
        from src.host.server import _validated_origin

        _patch_pairing_project_root(tmp_path, monkeypatch)
        # No config/telegram_paired.json at all.
        req = _FakeRequest({"x-origin": _origin_header("human", "telegram", "42")})
        result = _validated_origin(req)
        assert result.kind == "agent"

    def test_malformed_pairing_file_downgrades(self, tmp_path, monkeypatch):
        from src.host.server import _validated_origin

        _patch_pairing_project_root(tmp_path, monkeypatch)
        _write_pairing(tmp_path, "telegram", "not-json-at-all{")
        req = _FakeRequest({"x-origin": _origin_header("human", "telegram", "42")})
        result = _validated_origin(req)
        assert result.kind == "agent"

    def test_pairing_file_non_dict_downgrades(self, tmp_path, monkeypatch):
        from src.host.server import _validated_origin

        _patch_pairing_project_root(tmp_path, monkeypatch)
        _write_pairing(tmp_path, "telegram", "[1, 2, 3]")
        req = _FakeRequest({"x-origin": _origin_header("human", "telegram", "42")})
        result = _validated_origin(req)
        assert result.kind == "agent"

    def test_non_paired_channel_downgrades_for_untrusted_caller(self, tmp_path, monkeypatch):
        """Untrusted agent callers cannot self-assert CLI/dashboard human origin."""
        from src.host.server import _validated_origin

        _patch_pairing_project_root(tmp_path, monkeypatch)
        req = _FakeRequest({"x-origin": _origin_header("human", "cli", "jeff")})
        result = _validated_origin(req)
        assert result.kind == "agent"
        assert result.channel == "cli"

        req2 = _FakeRequest({"x-origin": _origin_header("human", "dashboard", "op-1")})
        result2 = _validated_origin(req2)
        assert result2.kind == "agent"
        assert result2.channel == "dashboard"

    def test_trusted_caller_keeps_non_paired_human_origin(self, tmp_path, monkeypatch):
        """Operator/mesh callers already passed their own authentication layer."""
        from src.host.server import _validated_origin

        _patch_pairing_project_root(tmp_path, monkeypatch)
        req = _FakeRequest({"x-origin": _origin_header("human", "dashboard", "op-1")})
        result = _validated_origin(req, caller="operator")
        assert result.kind == "human"
        assert result.channel == "dashboard"

    def test_unknown_human_channel_downgrades_for_untrusted_caller(self, tmp_path, monkeypatch):
        from src.host.server import _validated_origin

        _patch_pairing_project_root(tmp_path, monkeypatch)
        req = _FakeRequest({"x-origin": _origin_header("human", "rss", "feed-1")})
        result = _validated_origin(req)
        assert result.kind == "agent"
        assert result.channel == "rss"

    def test_non_human_kinds_downgrade_for_untrusted_caller(self, tmp_path, monkeypatch):
        """Untrusted callers cannot self-assert operator/system/cron/heartbeat."""
        from src.host.server import _validated_origin

        _patch_pairing_project_root(tmp_path, monkeypatch)
        req = _FakeRequest({"x-origin": _origin_header("agent", "telegram", "42")})
        result = _validated_origin(req)
        assert result is not None
        assert result.kind == "agent"

        for kind in ("system", "heartbeat", "cron", "operator"):
            req = _FakeRequest({"x-origin": _origin_header(kind, "telegram", "42")})
            result = _validated_origin(req)
            assert result is not None, f"kind={kind} should parse"
            assert result.kind == "agent", f"kind={kind} should downgrade"

    def test_trusted_loopback_internal_keeps_non_human_kind(self, tmp_path, monkeypatch):
        from src.host.server import _validated_origin

        _patch_pairing_project_root(tmp_path, monkeypatch)
        req = _FakeRequest(
            {
                "x-origin": _origin_header("heartbeat", "heartbeat", ""),
                "x-mesh-internal": "1",
            },
            client_host="127.0.0.1",
        )
        result = _validated_origin(req)
        assert result is not None
        assert result.kind == "heartbeat"

    def test_no_origin_header_returns_none(self, tmp_path, monkeypatch):
        from src.host.server import _validated_origin

        _patch_pairing_project_root(tmp_path, monkeypatch)
        req = _FakeRequest({})
        assert _validated_origin(req) is None

    def test_malformed_origin_header_returns_none(self, tmp_path, monkeypatch):
        from src.host.server import _validated_origin

        _patch_pairing_project_root(tmp_path, monkeypatch)
        req = _FakeRequest({"x-origin": "not-json"})
        assert _validated_origin(req) is None

    def test_cache_ttl_serves_stale_within_window(self, tmp_path, monkeypatch):
        """Reads inside the TTL window hit the cache. After TTL expires
        the helper re-reads the file."""
        from src.host import server as _server

        _patch_pairing_project_root(tmp_path, monkeypatch)
        _write_pairing(tmp_path, "telegram", {"owner": 42, "allowed": [42]})
        req_paired = _FakeRequest(
            {"x-origin": _origin_header("human", "telegram", "42")},
        )
        # Prime cache (records the current monotonic timestamp).
        assert _server._validated_origin(req_paired).kind == "human"

        # Revoke owner on disk → ``42`` is no longer paired.
        _write_pairing(tmp_path, "telegram", {"owner": 99, "allowed": []})
        # Within TTL → cached record still says owner=42 → still kind=human.
        assert _server._validated_origin(req_paired).kind == "human"

        # Force expiry by rewriting the cache entry with a stale timestamp
        # (cleaner than monkeypatching the global ``time`` module).
        cached = _server._pairing_cache.get("telegram")
        assert cached is not None
        _server._pairing_cache["telegram"] = (
            cached[0] - _server._PAIRING_CACHE_TTL - 1.0,
            cached[1],
        )
        # Next call re-reads → owner is now 99 → user 42 unpaired → downgrade.
        assert _server._validated_origin(req_paired).kind == "agent"

    def test_invalidate_cache_forces_reread(self, tmp_path, monkeypatch):
        from src.host import server as _server

        _patch_pairing_project_root(tmp_path, monkeypatch)
        _write_pairing(tmp_path, "telegram", {"owner": 42, "allowed": [42]})
        req = _FakeRequest({"x-origin": _origin_header("human", "telegram", "42")})
        assert _server._validated_origin(req).kind == "human"

        # Revoke on disk and invalidate.
        _write_pairing(tmp_path, "telegram", {"owner": 42, "allowed": []})
        _server._invalidate_pairing_cache("telegram")
        # User 42 still owner → still paired.
        assert _server._validated_origin(req).kind == "human"

        # Now flip the owner.
        _write_pairing(tmp_path, "telegram", {"owner": 99, "allowed": []})
        _server._invalidate_pairing_cache("telegram")
        assert _server._validated_origin(req).kind == "agent"

    def test_invalidate_cache_global(self, tmp_path, monkeypatch):
        """Invalidate with no argument clears all entries."""
        from src.host import server as _server

        _patch_pairing_project_root(tmp_path, monkeypatch)
        _write_pairing(tmp_path, "telegram", {"owner": 42, "allowed": []})
        _write_pairing(tmp_path, "discord", {"owner": "alice", "allowed": []})

        req_t = _FakeRequest({"x-origin": _origin_header("human", "telegram", "42")})
        req_d = _FakeRequest({"x-origin": _origin_header("human", "discord", "alice")})
        assert _server._validated_origin(req_t).kind == "human"
        assert _server._validated_origin(req_d).kind == "human"

        _server._invalidate_pairing_cache()
        assert _server._pairing_cache == {}

    def test_pairing_record_with_non_list_allowed(self, tmp_path, monkeypatch):
        """A malformed pairing record with non-list ``allowed`` falls back
        gracefully — the helper must not raise on bad shapes."""
        from src.host.server import _validated_origin

        _patch_pairing_project_root(tmp_path, monkeypatch)
        _write_pairing(tmp_path, "telegram", {"owner": 42, "allowed": "not-a-list"})
        # User 42 is the owner → still paired (owner check doesn't touch allowed).
        req_owner = _FakeRequest({"x-origin": _origin_header("human", "telegram", "42")})
        assert _validated_origin(req_owner).kind == "human"
        # User 9999 is neither owner nor in (treated-as-empty) allowed → downgrade.
        req_other = _FakeRequest({"x-origin": _origin_header("human", "telegram", "9999")})
        assert _validated_origin(req_other).kind == "agent"


def test_mesh_wake_downgrades_unpaired_human_origin(tmp_path, monkeypatch):
    """``/mesh/wake`` with ``kind="human"`` from an unpaired channel user
    enqueues a lane payload with origin downgraded to ``kind="agent"``.

    Wake itself still succeeds (current capture behaviour — Task 2e
    will tighten worker→operator wakes). The lane payload's origin is
    the authoritative trust signal that downstream gates will read.
    """
    import time

    from src.host import server as _server
    from src.shared.types import MessageOrigin

    _patch_pairing_project_root(tmp_path, monkeypatch)
    _server._invalidate_pairing_cache()
    # Pair owner=42; the wake claim uses user=not-paired so the recheck
    # downgrades.
    _write_pairing(tmp_path, "telegram", {"owner": 42, "allowed": [42]})

    client, captured, loop, bb = _wake_test_app(tmp_path)
    try:
        resp = client.post(
            "/mesh/wake",
            params={"target": "chef", "message": "check inbox"},
            headers={
                "x-origin": _origin_header("human", "telegram", "not-paired"),
                "X-Agent-ID": "worker",
            },
        )
        assert resp.status_code == 200
        deadline = time.monotonic() + 2.0
        while not captured and time.monotonic() < deadline:
            time.sleep(0.01)
        assert captured, "lane_manager.enqueue was never called"
        call = captured[0]
        assert isinstance(call["origin"], MessageOrigin)
        assert call["origin"].kind == "agent"
        # Channel/user metadata is preserved so the auto-notify path can
        # still address the originating surface (Task 2b semantics).
        assert call["origin"].channel == "telegram"
        assert call["origin"].user == "not-paired"
        assert call["auto_notify"] is True
    finally:
        loop.call_soon_threadsafe(loop.stop)
        bb.close()
        _server._invalidate_pairing_cache()


def test_mesh_wake_paired_human_origin_keeps_kind(tmp_path, monkeypatch):
    """``/mesh/wake`` with a verified channel claim keeps ``kind="human"``
    on the lane payload — downstream gates can trust it."""
    import time

    from src.host import server as _server
    from src.shared.types import MessageOrigin

    _patch_pairing_project_root(tmp_path, monkeypatch)
    _server._invalidate_pairing_cache()
    _write_pairing(tmp_path, "telegram", {"owner": 42, "allowed": [42]})

    client, captured, loop, bb = _wake_test_app(tmp_path)
    try:
        resp = client.post(
            "/mesh/wake",
            params={"target": "chef", "message": "check inbox"},
            headers={
                "x-origin": _origin_header("human", "telegram", "42"),
                "X-Agent-ID": "worker",
            },
        )
        assert resp.status_code == 200
        deadline = time.monotonic() + 2.0
        while not captured and time.monotonic() < deadline:
            time.sleep(0.01)
        assert captured
        call = captured[0]
        assert isinstance(call["origin"], MessageOrigin)
        assert call["origin"].kind == "human"
        assert call["origin"].channel == "telegram"
        assert call["origin"].user == "42"
        assert call["auto_notify"] is True
    finally:
        loop.call_soon_threadsafe(loop.stop)
        bb.close()
        _server._invalidate_pairing_cache()


# === M8: LLM-proxy input size cap ===

def test_proxy_input_oversize_rejected(vault_components, monkeypatch):
    """M8: a serialized proxy input above the cap is rejected (413) before
    dispatch — the vault handler is never reached."""
    import src.host.server as server_module

    # Shrink the cap so the test payload is cheap to build.
    monkeypatch.setattr(server_module, "_PROXY_INPUT_MAX_BYTES", 1024)
    # Pin the agent to the model it requests so the request clears the
    # model-pin gate and reaches the input-cap gate under test.
    monkeypatch.setattr(
        "src.cli.config._load_config",
        lambda *a, **k: {
            "agents": {"trusted": {"model": "anthropic/claude-sonnet-4-6"}},
        },
    )
    # The pin also runs an is_model_compatible() check — give the fixture
    # vault an Anthropic key so the requested model clears it and the
    # request reaches the input-cap gate under test.
    vault_components["vault"].system_credentials["anthropic_api_key"] = "sk-ant-test"
    client = vault_components["client"]
    big = "x" * 4096
    resp = client.post(
        "/mesh/api",
        params={"agent_id": "trusted"},
        json={
            "service": "llm", "action": "chat",
            "params": {
                "model": "anthropic/claude-sonnet-4-6",
                "messages": [{"role": "user", "content": big}],
            },
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is False
    assert data["status_code"] == 413
    assert "too large" in data["error"].lower()


def test_proxy_input_under_cap_not_rejected_for_size(vault_components, monkeypatch):
    """M8: a normal-sized prompt is NOT rejected by the size gate (it may
    still fail downstream for other reasons, but not with a 413)."""
    import src.host.server as server_module

    monkeypatch.setattr(server_module, "_PROXY_INPUT_MAX_BYTES", 4 * 1024 * 1024)
    # Pin the agent to the model it requests so the request clears the
    # model-pin gate and reaches the input-cap gate under test.
    monkeypatch.setattr(
        "src.cli.config._load_config",
        lambda *a, **k: {
            "agents": {"trusted": {"model": "anthropic/claude-sonnet-4-6"}},
        },
    )
    # The pin also runs an is_model_compatible() check — give the fixture
    # vault an Anthropic key so the requested model clears it and the
    # request reaches the input-cap gate under test.
    vault_components["vault"].system_credentials["anthropic_api_key"] = "sk-ant-test"
    client = vault_components["client"]
    resp = client.post(
        "/mesh/api",
        params={"agent_id": "trusted"},
        json={
            "service": "llm", "action": "chat",
            "params": {
                "model": "anthropic/claude-sonnet-4-6",
                "messages": [{"role": "user", "content": "hi"}],
            },
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    # Whatever happens downstream, it is NOT a size rejection.
    assert data.get("status_code") != 413


def test_proxy_stream_input_oversize_rejected(vault_components, monkeypatch):
    """M8: the streaming proxy rejects oversized input with HTTP 413."""
    import src.host.server as server_module

    monkeypatch.setattr(server_module, "_PROXY_INPUT_MAX_BYTES", 1024)
    # Pin the agent to the model it requests so the request clears the
    # model-pin gate and reaches the input-cap gate under test.
    monkeypatch.setattr(
        "src.cli.config._load_config",
        lambda *a, **k: {
            "agents": {"trusted": {"model": "anthropic/claude-sonnet-4-6"}},
        },
    )
    # The pin also runs an is_model_compatible() check — give the fixture
    # vault an Anthropic key so the requested model clears it and the
    # request reaches the input-cap gate under test.
    vault_components["vault"].system_credentials["anthropic_api_key"] = "sk-ant-test"
    client = vault_components["client"]
    big = "y" * 4096
    resp = client.post(
        "/mesh/api/stream",
        params={"agent_id": "trusted"},
        json={
            "service": "llm", "action": "chat",
            "params": {
                "model": "anthropic/claude-sonnet-4-6",
                "messages": [{"role": "user", "content": big}],
            },
        },
    )
    assert resp.status_code == 413
