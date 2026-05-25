"""Tests for BYOK-aware model-provider validation at agent creation.

Bug 5 — ``create_custom_agent`` and ``_create_agent_from_template`` used
to pick a hardcoded default model (``openai/gpt-4o-mini``) without
checking which providers actually had credentials. On a BYOK deployment
with only ``OPENLEGION_SYSTEM_ANTHROPIC_API_KEY`` set, agents were
created successfully but died silently on first LLM call.

These tests pin the up-front validation: HTTP 400 (mesh) /
``ValueError`` (CLI template path) when the resolved model's provider
has no key in env, success when it does, and a clear "no providers
configured" message when env has nothing at all.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.host.credentials import CredentialVault
from src.host.mesh import Blackboard, MessageRouter, PubSub
from src.host.permissions import PermissionMatrix
from src.host.server import create_mesh_app
from src.shared.models import get_available_providers, resolve_provider_for_model
from src.shared.types import AgentPermissions

# ── resolve_provider_for_model edge cases ──────────────────


def test_resolve_provider_empty_string():
    assert resolve_provider_for_model("") is None


def test_resolve_provider_none_typed():
    # Non-string inputs (defensive) — should return None, not raise.
    assert resolve_provider_for_model(None) is None  # type: ignore[arg-type]


def test_resolve_provider_no_slash_unknown():
    # Bare name without slash and not in the override table → None.
    assert resolve_provider_for_model("some-random-model") is None


def test_resolve_provider_standard_slash_form():
    assert resolve_provider_for_model("openai/gpt-4o-mini") == "openai"
    assert resolve_provider_for_model("anthropic/claude-sonnet-4-5-20250929") == "anthropic"
    assert resolve_provider_for_model("gemini/gemini-2.5-pro") == "gemini"


def test_resolve_provider_openrouter_namespaced():
    # ``openrouter/anthropic/claude-...`` — outer provider wins because
    # that's what determines the API key / routing path.
    assert (
        resolve_provider_for_model("openrouter/anthropic/claude-3.5-sonnet")
        == "openrouter"
    )


def test_resolve_provider_bare_openai_prefixes():
    assert resolve_provider_for_model("gpt-4o-mini") == "openai"
    assert resolve_provider_for_model("o1-preview") == "openai"
    assert resolve_provider_for_model("o3-mini") == "openai"
    assert resolve_provider_for_model("o4-mini") == "openai"


def test_resolve_provider_ollama_alt_prefix():
    assert resolve_provider_for_model("ollama_chat/llama3") == "ollama"


# ── get_available_providers env inspection ──────────────────


def _clear_system_env(monkeypatch):
    """Strip every ``OPENLEGION_SYSTEM_*`` env var for a clean slate."""
    import os
    for key in list(os.environ):
        if key.startswith("OPENLEGION_SYSTEM_"):
            monkeypatch.delenv(key, raising=False)


def test_get_available_providers_empty(monkeypatch):
    _clear_system_env(monkeypatch)
    assert get_available_providers() == set()


def test_get_available_providers_picks_up_api_key(monkeypatch):
    _clear_system_env(monkeypatch)
    monkeypatch.setenv("OPENLEGION_SYSTEM_ANTHROPIC_API_KEY", "sk-ant")
    assert get_available_providers() == {"anthropic"}


def test_get_available_providers_multiple(monkeypatch):
    _clear_system_env(monkeypatch)
    monkeypatch.setenv("OPENLEGION_SYSTEM_ANTHROPIC_API_KEY", "sk-ant")
    monkeypatch.setenv("OPENLEGION_SYSTEM_OPENAI_API_KEY", "sk-oai")
    assert get_available_providers() == {"anthropic", "openai"}


def test_get_available_providers_oauth_counts_as_openai(monkeypatch):
    _clear_system_env(monkeypatch)
    monkeypatch.setenv("OPENLEGION_SYSTEM_OPENAI_OAUTH", '{"token": "..."}')
    assert "openai" in get_available_providers()


def test_get_available_providers_ollama_api_base(monkeypatch):
    _clear_system_env(monkeypatch)
    monkeypatch.setenv("OPENLEGION_SYSTEM_OLLAMA_API_BASE", "http://localhost:11434")
    assert "ollama" in get_available_providers()


def test_get_available_providers_ignores_empty_values(monkeypatch):
    _clear_system_env(monkeypatch)
    monkeypatch.setenv("OPENLEGION_SYSTEM_ANTHROPIC_API_KEY", "")
    assert get_available_providers() == set()


# ── mesh create_custom_agent validation ────────────────────


@pytest.fixture()
def _mesh_env(tmp_path, monkeypatch):
    """Patch config module paths so agent creation writes to tmp_path."""
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()

    monkeypatch.setattr("src.cli.config.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("src.cli.config.AGENTS_FILE", cfg_dir / "agents.yaml")
    monkeypatch.setattr("src.cli.config.PERMISSIONS_FILE", cfg_dir / "permissions.json")
    monkeypatch.setattr("src.cli.config.CONFIG_FILE", cfg_dir / "mesh.yaml")


@pytest.fixture()
def container_mgr():
    mgr = MagicMock()
    mgr.start_agent.return_value = "http://localhost:9001"
    mgr.wait_for_agent = AsyncMock(return_value=True)
    mgr.agents = {}
    return mgr


def _build_mesh_app(tmp_path, container_mgr, vault: CredentialVault | None):
    """Build a test mesh with operator perms and an optional vault."""
    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    import json as _json
    perms_file = tmp_path / "config" / "permissions.json"
    perms_file.parent.mkdir(parents=True, exist_ok=True)
    perms_file.write_text(_json.dumps({
        "permissions": {
            "operator": {
                "can_message": ["*"],
                "can_spawn": True,
                "can_manage_fleet": True,
                "blackboard_read": ["*"],
                "blackboard_write": ["*"],
            },
        },
    }))
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {
        "operator": AgentPermissions(
            agent_id="operator",
            can_message=["*"],
            can_spawn=True,
            can_manage_fleet=True,
            blackboard_read=["*"],
            blackboard_write=["*"],
        ),
    }
    perms._config_path = str(perms_file)
    router = MessageRouter(permissions=perms, agent_registry={})
    app = create_mesh_app(
        bb, pubsub, router, perms,
        container_manager=container_mgr,
        credential_vault=vault,
    )
    return TestClient(app), bb


def test_create_agent_rejects_model_without_credentials(
    tmp_path, _mesh_env, container_mgr, monkeypatch,
):
    """Anthropic-only deployment, ask for openai/gpt-4o-mini → 400."""
    _clear_system_env(monkeypatch)
    monkeypatch.setenv("OPENLEGION_SYSTEM_ANTHROPIC_API_KEY", "sk-ant-only")
    vault = CredentialVault()

    client, bb = _build_mesh_app(tmp_path, container_mgr, vault)
    try:
        resp = client.post(
            "/mesh/agents/create",
            json={
                "agent_id": "operator",
                "name": "byok_blind",
                "role": "research",
                "model": "openai/gpt-4o-mini",
            },
        )
        assert resp.status_code == 400, resp.text
        detail = resp.json()["detail"]
        # Message must name the missing provider AND list what's available.
        assert "openai" in detail.lower()
        assert "anthropic" in detail.lower()
        assert "OPENLEGION_SYSTEM_OPENAI_API_KEY" in detail
        # Container should NOT have been started.
        container_mgr.start_agent.assert_not_called()
    finally:
        bb.close()


def test_create_agent_accepts_model_with_credentials(
    tmp_path, _mesh_env, container_mgr, monkeypatch,
):
    """Anthropic key present, ask for anthropic/... → success."""
    _clear_system_env(monkeypatch)
    monkeypatch.setenv("OPENLEGION_SYSTEM_ANTHROPIC_API_KEY", "sk-ant-only")
    vault = CredentialVault()

    client, bb = _build_mesh_app(tmp_path, container_mgr, vault)
    try:
        resp = client.post(
            "/mesh/agents/create",
            json={
                "agent_id": "operator",
                "name": "byok_happy",
                "role": "research",
                "model": "anthropic/claude-sonnet-4-5-20250929",
            },
        )
        assert resp.status_code == 200, resp.text
        container_mgr.start_agent.assert_called_once()
    finally:
        bb.close()


def test_create_agent_rejects_when_no_provider_keys_present(
    tmp_path, _mesh_env, container_mgr, monkeypatch,
):
    """Zero provider keys → rejection message mentions 'none' available."""
    _clear_system_env(monkeypatch)
    vault = CredentialVault()

    client, bb = _build_mesh_app(tmp_path, container_mgr, vault)
    try:
        resp = client.post(
            "/mesh/agents/create",
            json={
                "agent_id": "operator",
                "name": "no_keys_at_all",
                "role": "test",
                "model": "openai/gpt-4o-mini",
            },
        )
        # Rejection is graceful — proper 400, clear copy.
        assert resp.status_code == 400, resp.text
        detail = resp.json()["detail"]
        assert "none" in detail.lower()
        container_mgr.start_agent.assert_not_called()
    finally:
        bb.close()


def test_create_agent_skips_validation_when_no_vault(
    tmp_path, _mesh_env, container_mgr, monkeypatch,
):
    """No CredentialVault wired (test harness / sandbox) → skip the
    validation rather than block. The vault is the only thing that can
    enumerate OAuth state, so without it we'd be guessing.
    """
    _clear_system_env(monkeypatch)
    client, bb = _build_mesh_app(tmp_path, container_mgr, vault=None)
    try:
        resp = client.post(
            "/mesh/agents/create",
            json={
                "agent_id": "operator",
                "name": "no_vault",
                "role": "test",
                "model": "openai/gpt-4o-mini",
            },
        )
        assert resp.status_code == 200, resp.text
    finally:
        bb.close()


# ── CLI template-path validation ───────────────────────────


def test_apply_template_rejects_model_without_credentials(tmp_path, monkeypatch):
    """``_create_agent_from_template`` must reject up front when the
    resolved model's provider has no key in env. Mirrors the mesh-side
    check so ``apply_template`` doesn't silently mint dead agents.
    """
    _clear_system_env(monkeypatch)
    monkeypatch.setenv("OPENLEGION_SYSTEM_ANTHROPIC_API_KEY", "sk-ant-only")

    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    monkeypatch.setattr("src.cli.config.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("src.cli.config.AGENTS_FILE", cfg_dir / "agents.yaml")
    monkeypatch.setattr("src.cli.config.PERMISSIONS_FILE", cfg_dir / "permissions.json")
    monkeypatch.setattr("src.cli.config.CONFIG_FILE", cfg_dir / "mesh.yaml")

    # Stub the template loader so we don't depend on the on-disk YAMLs.
    fake_template = {
        "starter": {
            "agents": {
                "worker": {
                    "role": "worker",
                    "model": "openai/gpt-4o-mini",  # forces the openai provider
                    "instructions": "do work",
                },
            },
        },
    }
    monkeypatch.setattr("src.cli.config._load_templates", lambda: fake_template)

    from src.cli.config import _create_agent_from_template
    with pytest.raises(ValueError) as exc:
        _create_agent_from_template(name="worker", template_id="starter/worker", model="")
    msg = str(exc.value).lower()
    assert "openai" in msg
    assert "anthropic" in msg


def test_apply_template_accepts_model_with_credentials(tmp_path, monkeypatch):
    """Inverse of the above — anthropic key in env, anthropic model → OK."""
    _clear_system_env(monkeypatch)
    monkeypatch.setenv("OPENLEGION_SYSTEM_ANTHROPIC_API_KEY", "sk-ant-only")

    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    monkeypatch.setattr("src.cli.config.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("src.cli.config.AGENTS_FILE", cfg_dir / "agents.yaml")
    monkeypatch.setattr("src.cli.config.PERMISSIONS_FILE", cfg_dir / "permissions.json")
    monkeypatch.setattr("src.cli.config.CONFIG_FILE", cfg_dir / "mesh.yaml")

    fake_template = {
        "starter": {
            "agents": {
                "worker": {
                    "role": "worker",
                    "model": "anthropic/claude-sonnet-4-5-20250929",
                    "instructions": "do work",
                },
            },
        },
    }
    monkeypatch.setattr("src.cli.config._load_templates", lambda: fake_template)

    from src.cli.config import _create_agent_from_template
    # Should not raise.
    _create_agent_from_template(name="worker", template_id="starter/worker", model="")
    # And the agent should be present in agents.yaml.
    import yaml
    data = yaml.safe_load((cfg_dir / "agents.yaml").read_text())
    assert "worker" in data.get("agents", {})


# ── Seam follow-up Fix 2: OAuth-only model compat at write paths ──


def test_create_agent_rejects_non_oauth_allowed_model_under_oauth_only(
    tmp_path, _mesh_env, container_mgr, monkeypatch,
):
    """OAuth-only OpenAI creds + non-OAuth-allowed model → 400.

    Naming the allowed alternatives in the error message is the operator
    UX contract — they should never have to guess which models work.
    """
    _clear_system_env(monkeypatch)
    monkeypatch.setenv(
        "OPENLEGION_SYSTEM_OPENAI_OAUTH",
        '{"access_token":"tok","refresh_token":"ref"}',
    )
    vault = CredentialVault()

    client, bb = _build_mesh_app(tmp_path, container_mgr, vault)
    try:
        # gpt-4.1-mini is NOT in OAUTH_ALLOWED_MODELS_OPENAI
        resp = client.post(
            "/mesh/agents/create",
            json={
                "agent_id": "operator",
                "name": "oauth_blocker",
                "role": "research",
                "model": "openai/gpt-4.1-mini",
            },
        )
        assert resp.status_code == 400, resp.text
        detail = resp.json()["detail"]
        assert "OAuth-allowed models" in detail
        # Suggested replacement model is named.
        assert "openai/gpt-5.3-codex" in detail
        container_mgr.start_agent.assert_not_called()
    finally:
        bb.close()


def test_apply_template_rejects_non_oauth_allowed_model_under_oauth_only(
    tmp_path, monkeypatch,
):
    """Same gate fires on the CLI template path."""
    _clear_system_env(monkeypatch)
    monkeypatch.setenv(
        "OPENLEGION_SYSTEM_OPENAI_OAUTH",
        '{"access_token":"tok","refresh_token":"ref"}',
    )

    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    monkeypatch.setattr("src.cli.config.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("src.cli.config.AGENTS_FILE", cfg_dir / "agents.yaml")
    monkeypatch.setattr("src.cli.config.PERMISSIONS_FILE", cfg_dir / "permissions.json")
    monkeypatch.setattr("src.cli.config.CONFIG_FILE", cfg_dir / "mesh.yaml")

    fake_template = {
        "starter": {
            "agents": {
                "worker": {
                    "role": "worker",
                    "model": "openai/gpt-4.1-mini",  # NOT in OAuth allowlist
                    "instructions": "do work",
                },
            },
        },
    }
    monkeypatch.setattr("src.cli.config._load_templates", lambda: fake_template)

    from src.cli.config import _create_agent_from_template
    with pytest.raises(ValueError) as exc:
        _create_agent_from_template(
            name="worker", template_id="starter/worker", model="",
        )
    assert "OAuth-allowed models" in str(exc.value)


def test_introspect_llm_surface_includes_allowed_models_and_kinds(
    tmp_path, _mesh_env, monkeypatch,
):
    """``GET /mesh/introspect?section=llm`` must surface the new fields
    so list_available_models has a stable contract."""
    _clear_system_env(monkeypatch)
    monkeypatch.setenv("OPENLEGION_SYSTEM_OPENAI_API_KEY", "sk-test")
    vault = CredentialVault()

    container_mgr = MagicMock()
    container_mgr.agents = {}
    client, bb = _build_mesh_app(tmp_path, container_mgr, vault)
    try:
        resp = client.get(
            "/mesh/introspect",
            params={"section": "llm", "agent_id": "operator"},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert "llm" in data
        llm = data["llm"]
        assert "available_providers" in llm
        assert "allowed_models" in llm
        assert "credential_kinds" in llm
        # OpenAI API key is configured — should appear with kind=api_key.
        assert "openai" in llm["available_providers"]
        assert llm["credential_kinds"].get("openai") == "api_key"
        assert "openai" in llm["allowed_models"]
        assert len(llm["allowed_models"]["openai"]) > 0
    finally:
        bb.close()


# ── Seam follow-up Fix 4 / 5: auth-failure endpoint + profile quarantine ──


def test_report_auth_failure_endpoint_records_and_quarantines(
    tmp_path, _mesh_env, monkeypatch,
):
    """POST /mesh/agents/{id}/auth-failure increments counter and
    quarantines at the threshold."""
    from src.host.health import HealthMonitor

    _clear_system_env(monkeypatch)
    vault = CredentialVault()

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    import json as _json
    perms_file = tmp_path / "config" / "permissions.json"
    perms_file.parent.mkdir(parents=True, exist_ok=True)
    perms_file.write_text(_json.dumps({
        "permissions": {
            "agent-a": {
                "can_message": ["*"],
                "blackboard_read": ["*"],
                "blackboard_write": ["*"],
            },
        },
    }))
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {
        "agent-a": AgentPermissions(
            agent_id="agent-a", can_message=["*"],
            blackboard_read=["*"], blackboard_write=["*"],
        ),
    }
    perms._config_path = str(perms_file)
    router = MessageRouter(permissions=perms, agent_registry={})

    health_monitor = HealthMonitor(
        runtime=MagicMock(), transport=MagicMock(), router=router,
    )
    health_monitor.register("agent-a")

    container_mgr = MagicMock()
    container_mgr.agents = {}
    app = create_mesh_app(
        bb, pubsub, router, perms,
        container_manager=container_mgr,
        credential_vault=vault,
        health_monitor=health_monitor,
    )
    client = TestClient(app)
    try:
        # First two reports stay below threshold.
        for _ in range(2):
            resp = client.post(
                "/mesh/agents/agent-a/auth-failure",
                json={"provider": "openai", "model": "openai/gpt-5", "http_status": 401},
            )
            assert resp.status_code == 200
            assert resp.json() == {"recorded": True, "quarantined": False}
        # Third trips the quarantine.
        resp = client.post(
            "/mesh/agents/agent-a/auth-failure",
            json={"provider": "openai", "model": "openai/gpt-5", "http_status": 401},
        )
        assert resp.status_code == 200
        assert resp.json() == {"recorded": True, "quarantined": True}
        assert health_monitor.is_quarantined("agent-a") is True
    finally:
        bb.close()


def test_report_auth_failure_endpoint_rejects_cross_agent_report(
    tmp_path, _mesh_env, monkeypatch,
):
    """Agents must only be able to self-report.

    With auth tokens unset (dev mode), ``_resolve_agent_id`` trusts the
    caller. To assert the cross-agent rejection path we wire auth_tokens
    so the verified agent ID diverges from the URL agent_id.
    """
    from src.host.health import HealthMonitor

    _clear_system_env(monkeypatch)
    vault = CredentialVault()

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    import json as _json
    perms_file = tmp_path / "config" / "permissions.json"
    perms_file.parent.mkdir(parents=True, exist_ok=True)
    perms_file.write_text(_json.dumps({"permissions": {}}))
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {}
    perms._config_path = str(perms_file)
    router = MessageRouter(permissions=perms, agent_registry={})

    health_monitor = HealthMonitor(
        runtime=MagicMock(), transport=MagicMock(), router=router,
    )
    health_monitor.register("alice")
    health_monitor.register("bob")

    container_mgr = MagicMock()
    container_mgr.agents = {}
    app = create_mesh_app(
        bb, pubsub, router, perms,
        container_manager=container_mgr,
        credential_vault=vault,
        health_monitor=health_monitor,
        auth_tokens={"alice": "tok-alice", "bob": "tok-bob"},
    )
    client = TestClient(app)
    try:
        # Bob's token, but URL targets alice → 403.
        resp = client.post(
            "/mesh/agents/alice/auth-failure",
            json={"provider": "openai", "model": "x", "http_status": 401},
            headers={"Authorization": "Bearer tok-bob"},
        )
        assert resp.status_code == 403
        assert health_monitor.agents["alice"].consecutive_auth_failures == 0
        # Self-report (alice with alice's token) works → 200.
        resp = client.post(
            "/mesh/agents/alice/auth-failure",
            json={"provider": "openai", "model": "x", "http_status": 401},
            headers={"Authorization": "Bearer tok-alice"},
        )
        assert resp.status_code == 200
        assert health_monitor.agents["alice"].consecutive_auth_failures == 1
    finally:
        bb.close()


def test_auth_failure_endpoint_is_rate_limited(
    tmp_path, _mesh_env, monkeypatch,
):
    """Principal-eng follow-up: /auth-failure must enforce the
    ``auth_failure`` rate-limit bucket on agent self-reports.

    Quarantine threshold is 3 so legitimate traffic never approaches
    the limit (default 60/min). This bucket exists to cap notification-store
    writes and event-bus emits when a runaway agent retries on a broken
    credential before its lane gate latches. Internal callers
    (``x-mesh-internal``) intentionally bypass — they are the
    load-bearing trigger fired from inside the proxy boundary.
    """
    from src.host import server as host_server
    from src.host.health import HealthMonitor

    _clear_system_env(monkeypatch)
    vault = CredentialVault()

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    import json as _json
    perms_file = tmp_path / "config" / "permissions.json"
    perms_file.parent.mkdir(parents=True, exist_ok=True)
    perms_file.write_text(_json.dumps({"permissions": {}}))
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {}
    perms._config_path = str(perms_file)
    router = MessageRouter(
        permissions=perms,
        agent_registry={"alice": "http://localhost:9999"},
    )

    health_monitor = HealthMonitor(
        runtime=MagicMock(), transport=MagicMock(), router=router,
    )
    health_monitor.register("alice")

    container_mgr = MagicMock()
    container_mgr.agents = {}
    app = create_mesh_app(
        bb, pubsub, router, perms,
        container_manager=container_mgr,
        credential_vault=vault,
        health_monitor=health_monitor,
        auth_tokens={"alice": "tok-alice"},
    )
    client = TestClient(app)
    # Drop the rate counter to a clean slate so prior tests don't
    # pollute this test's window.
    host_server._denial_counter["rate"] = 0

    try:
        # Burn through the bucket. Limit is 60/min — fire 60 OK then
        # the 61st must 429. Body content is irrelevant past threshold;
        # the limiter raises before record_auth_failure executes.
        for i in range(60):
            resp = client.post(
                "/mesh/agents/alice/auth-failure",
                json={"provider": "openai", "model": "x", "http_status": 401},
                headers={"Authorization": "Bearer tok-alice"},
            )
            assert resp.status_code == 200, (
                f"request {i} unexpectedly failed: {resp.status_code} {resp.text}"
            )
        resp = client.post(
            "/mesh/agents/alice/auth-failure",
            json={"provider": "openai", "model": "x", "http_status": 401},
            headers={"Authorization": "Bearer tok-alice"},
        )
        assert resp.status_code == 429, resp.text
        # ``rate`` denial counter bumped.
        assert host_server._denial_counter["rate"] >= 1
    finally:
        bb.close()


@pytest.mark.asyncio
async def test_auth_failure_endpoint_internal_caller_bypasses_rate_limit(
    tmp_path, _mesh_env, monkeypatch,
):
    """Internal (loopback + ``x-mesh-internal``) callers must not be
    rate-limited — they are the mesh's own ``_record_auth`` recorder
    threading the credential-vault proxy boundary, which is the
    load-bearing quarantine trigger. Throttling that path would mute
    quarantine signals for legitimate operator credentials going bad.

    ``_is_internal_caller`` requires loopback peer + the header; the
    Starlette ``TestClient`` reports peer ``"testclient"`` which fails
    the loopback parse, so this test uses ``AsyncClient`` over
    ``ASGITransport`` which presents ``127.0.0.1``.
    """
    from httpx import ASGITransport, AsyncClient

    from src.host.health import HealthMonitor

    _clear_system_env(monkeypatch)
    vault = CredentialVault()

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    import json as _json
    perms_file = tmp_path / "config" / "permissions.json"
    perms_file.parent.mkdir(parents=True, exist_ok=True)
    perms_file.write_text(_json.dumps({"permissions": {}}))
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {}
    perms._config_path = str(perms_file)
    router = MessageRouter(
        permissions=perms,
        agent_registry={"alice": "http://localhost:9999"},
    )

    health_monitor = HealthMonitor(
        runtime=MagicMock(), transport=MagicMock(), router=router,
    )
    health_monitor.register("alice")

    container_mgr = MagicMock()
    container_mgr.agents = {}
    app = create_mesh_app(
        bb, pubsub, router, perms,
        container_manager=container_mgr,
        credential_vault=vault,
        health_monitor=health_monitor,
    )
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            # 70 internal calls — bypasses the rate limit because the
            # internal-caller branch returns before _check_rate_limit.
            for i in range(70):
                resp = await client.post(
                    "/mesh/agents/alice/auth-failure",
                    json={"provider": "openai", "model": "x", "http_status": 401},
                    headers={"x-mesh-internal": "1"},
                )
                assert resp.status_code == 200, (
                    f"internal request {i} hit rate limit: {resp.text}"
                )
    finally:
        bb.close()


def test_apply_template_endpoint_rejects_override_outside_oauth_allowlist(
    tmp_path, _mesh_env, container_mgr, monkeypatch,
):
    """Bug 3 follow-up: POST /mesh/templates/apply must validate every
    effective model with ``is_model_compatible`` UPFRONT — before any
    agent is created. An OAuth-only deployment + a non-OAuth-allowed
    override should reject with 400 and zero containers started."""
    _clear_system_env(monkeypatch)
    monkeypatch.setenv(
        "OPENLEGION_SYSTEM_OPENAI_OAUTH",
        '{"access_token":"tok","refresh_token":"ref"}',
    )
    vault = CredentialVault()

    # Stub the template loader so we don't depend on the on-disk YAMLs.
    fake_template = {
        "minteam": {
            "description": "stub",
            "agents": {
                "worker": {
                    "role": "worker",
                    "model": "openai/gpt-5",  # in OAuth allowlist
                    "instructions": "do work",
                },
            },
        },
    }
    monkeypatch.setattr("src.cli.config._load_templates", lambda: fake_template)

    client, bb = _build_mesh_app(tmp_path, container_mgr, vault)
    try:
        resp = client.post(
            "/mesh/fleet/apply",
            json={
                "spawned_by": "operator",
                "template": "minteam",
                "agent_overrides": {
                    "worker": {"model": "openai/gpt-4.1-mini"},  # NOT in OAuth allowlist
                },
            },
        )
        assert resp.status_code == 400, resp.text
        detail = resp.json()["detail"]
        assert "OAuth-allowed models" in detail
        # Slot name is named for actionable error.
        assert "worker" in detail
        # No agent was created.
        container_mgr.start_agent.assert_not_called()
    finally:
        bb.close()


def test_apply_template_endpoint_rejects_template_default_when_incompatible(
    tmp_path, _mesh_env, container_mgr, monkeypatch,
):
    """Even WITHOUT overrides, the template-default model is validated
    upfront. OAuth-only deployment + template defaulting to
    openai/gpt-4o-mini (not in OAuth allowlist) → 400."""
    _clear_system_env(monkeypatch)
    monkeypatch.setenv(
        "OPENLEGION_SYSTEM_OPENAI_OAUTH",
        '{"access_token":"tok","refresh_token":"ref"}',
    )
    vault = CredentialVault()

    fake_template = {
        "badtmpl": {
            "description": "stub",
            "agents": {
                "worker": {
                    "role": "worker",
                    "model": "openai/gpt-4o-mini",  # NOT in OAuth allowlist
                    "instructions": "do work",
                },
            },
        },
    }
    monkeypatch.setattr("src.cli.config._load_templates", lambda: fake_template)

    client, bb = _build_mesh_app(tmp_path, container_mgr, vault)
    try:
        resp = client.post(
            "/mesh/fleet/apply",
            json={"spawned_by": "operator", "template": "badtmpl"},
        )
        assert resp.status_code == 400, resp.text
        detail = resp.json()["detail"]
        assert "OAuth-allowed models" in detail
        container_mgr.start_agent.assert_not_called()
    finally:
        bb.close()


def test_apply_template_endpoint_top_level_model_override_validated(
    tmp_path, _mesh_env, container_mgr, monkeypatch,
):
    """Top-level ``model`` (the legacy override that applies to every slot)
    is validated upfront too. OAuth-only deployment + top-level
    incompatible model → 400."""
    _clear_system_env(monkeypatch)
    monkeypatch.setenv(
        "OPENLEGION_SYSTEM_OPENAI_OAUTH",
        '{"access_token":"tok","refresh_token":"ref"}',
    )
    vault = CredentialVault()

    fake_template = {
        "okt": {
            "description": "stub",
            "agents": {
                "worker": {
                    "role": "worker",
                    "model": "openai/gpt-5",  # template default IS OK
                    "instructions": "do work",
                },
            },
        },
    }
    monkeypatch.setattr("src.cli.config._load_templates", lambda: fake_template)

    client, bb = _build_mesh_app(tmp_path, container_mgr, vault)
    try:
        resp = client.post(
            "/mesh/fleet/apply",
            json={
                "spawned_by": "operator",
                "template": "okt",
                "model": "openai/gpt-4o-mini",  # forces all slots → BAD
            },
        )
        assert resp.status_code == 400, resp.text
        detail = resp.json()["detail"]
        assert "OAuth-allowed models" in detail
        container_mgr.start_agent.assert_not_called()
    finally:
        bb.close()


def test_profile_endpoint_surfaces_quarantine_fields(
    tmp_path, _mesh_env, monkeypatch,
):
    """GET /mesh/agents/{id}/profile must include quarantine state."""
    from src.host.health import HealthMonitor

    _clear_system_env(monkeypatch)
    vault = CredentialVault()

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    import json as _json
    perms_file = tmp_path / "config" / "permissions.json"
    perms_file.parent.mkdir(parents=True, exist_ok=True)
    perms_file.write_text(_json.dumps({"permissions": {}}))
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {}
    perms._config_path = str(perms_file)
    router = MessageRouter(
        permissions=perms,
        agent_registry={"agent-a": "http://localhost:9999"},
    )

    health_monitor = HealthMonitor(
        runtime=MagicMock(), transport=MagicMock(), router=router,
    )
    health_monitor.register("agent-a")
    # Force quarantine.
    for _ in range(3):
        health_monitor.record_auth_failure(
            "agent-a", provider="openai", model="x", http_status=401,
        )

    container_mgr = MagicMock()
    container_mgr.agents = {}
    app = create_mesh_app(
        bb, pubsub, router, perms,
        container_manager=container_mgr,
        credential_vault=vault,
        health_monitor=health_monitor,
    )
    client = TestClient(app)
    try:
        resp = client.get("/mesh/agents/agent-a/profile")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["quarantined"] is True
        assert body["quarantine_reason"] is not None
        assert body["consecutive_auth_failures"] == 3
    finally:
        bb.close()


# ── Codex P1.1 / P1.2: slot model resolution helper ────────


def test_resolve_slot_model_slot_override_wins_over_top_level():
    """Slot override beats top-level model_override (precedence rule 1)."""
    from src.shared.models import resolve_slot_model
    slot_def = {"model": "openai/gpt-4o-mini"}
    out = resolve_slot_model(
        slot_name="worker",
        slot_def=slot_def,
        agent_overrides={"worker": {"model": "anthropic/claude-sonnet-4-5-20250929"}},
        model_override="openai/gpt-5",  # would otherwise win
        default_model="openai/gpt-4o-mini",
    )
    assert out == "anthropic/claude-sonnet-4-5-20250929"


def test_resolve_slot_model_top_level_beats_template_default():
    """Top-level model_override beats template default (precedence rule 2)."""
    from src.shared.models import resolve_slot_model
    slot_def = {"model": "openai/gpt-4o-mini"}
    out = resolve_slot_model(
        slot_name="worker",
        slot_def=slot_def,
        agent_overrides=None,
        model_override="openai/gpt-5",
        default_model="openai/gpt-4o-mini",
    )
    assert out == "openai/gpt-5"


def test_resolve_slot_model_template_default_used_when_no_override():
    """Template default applies when neither override is set."""
    from src.shared.models import resolve_slot_model
    slot_def = {"model": "openai/gpt-4o-mini"}
    out = resolve_slot_model(
        slot_name="worker",
        slot_def=slot_def,
        agent_overrides=None,
        model_override="",
        default_model="anthropic/claude-sonnet-4-5-20250929",
    )
    assert out == "openai/gpt-4o-mini"


def test_resolve_slot_model_config_default_when_template_absent():
    """Falls through to config default when slot_def has no model key."""
    from src.shared.models import resolve_slot_model
    out = resolve_slot_model(
        slot_name="worker",
        slot_def={},
        agent_overrides=None,
        model_override="",
        default_model="anthropic/claude-sonnet-4-5-20250929",
    )
    assert out == "anthropic/claude-sonnet-4-5-20250929"


def test_resolve_slot_model_handles_null_model_in_slot_def():
    """P1.2 — a template slot with ``model: null`` (Python None) must
    coerce to the config default. Without this, ``dict.get('model',
    default)`` returns None (because the KEY is present), and the
    downstream ``.replace()`` raises ``AttributeError``."""
    from src.shared.models import resolve_slot_model
    out = resolve_slot_model(
        slot_name="worker",
        slot_def={"model": None},  # explicit Python None
        agent_overrides=None,
        model_override="",
        default_model="openai/gpt-4o-mini",
    )
    assert out == "openai/gpt-4o-mini"


def test_resolve_slot_model_substitutes_default_model_sentinel():
    """A template string ``"{default_model}"`` is replaced with the
    config default after precedence resolution."""
    from src.shared.models import resolve_slot_model
    out = resolve_slot_model(
        slot_name="worker",
        slot_def={"model": "{default_model}"},
        agent_overrides=None,
        model_override="",
        default_model="anthropic/claude-sonnet-4-5-20250929",
    )
    assert out == "anthropic/claude-sonnet-4-5-20250929"


def test_resolve_slot_model_empty_slot_override_falls_through():
    """A slot override with empty/whitespace ``model`` falls through to
    the next precedence layer rather than emitting an empty string.
    Prevents accidental ``{"worker": {"model": ""}}`` from breaking
    container start."""
    from src.shared.models import resolve_slot_model
    out = resolve_slot_model(
        slot_name="worker",
        slot_def={"model": "openai/gpt-5"},
        agent_overrides={"worker": {"model": ""}},
        model_override="",
        default_model="openai/gpt-4o-mini",
    )
    assert out == "openai/gpt-5"


def test_apply_template_validation_matches_creation_resolution(
    tmp_path, _mesh_env, container_mgr, monkeypatch,
):
    """P1.1 — validation and agent creation MUST resolve the per-slot
    model the same way. Otherwise an operator could set a slot override
    (validated) while a top-level model_override (used at creation)
    produces a different result, and an incompatible model bypasses
    the up-front credential check.

    Setup: OAuth-only OpenAI deployment. Template slot defaults to
    ``openai/gpt-5`` (in OAuth allowlist). Operator supplies a slot
    override of ``openai/gpt-5-mini`` (also allowed) AND a top-level
    model_override of ``openai/gpt-4.1-mini`` (NOT in OAuth allowlist).

    The canonical precedence is slot override > top-level. If both
    sites obey it, validation passes (slot model is allowed), creation
    starts the container with the slot model (allowed), end-to-end ok.

    If validation and creation drift (e.g. validation honors slot
    override but creation falls back to top-level), creation would
    silently hand the container an incompatible model and the slot
    would die on first LLM call with no surfaced reason.
    """
    _clear_system_env(monkeypatch)
    monkeypatch.setenv(
        "OPENLEGION_SYSTEM_OPENAI_OAUTH",
        '{"access_token":"tok","refresh_token":"ref"}',
    )
    vault = CredentialVault()

    fake_template = {
        "tmpl": {
            "description": "stub",
            "agents": {
                "worker": {
                    "role": "worker",
                    "model": "openai/gpt-5",  # template default — allowed
                    "instructions": "do work",
                },
            },
        },
    }
    monkeypatch.setattr("src.cli.config._load_templates", lambda: fake_template)

    client, bb = _build_mesh_app(tmp_path, container_mgr, vault)
    try:
        resp = client.post(
            "/mesh/fleet/apply",
            json={
                "spawned_by": "operator",
                "template": "tmpl",
                # Slot override is OAuth-allowed — should win.
                "agent_overrides": {
                    "worker": {"model": "openai/gpt-5-mini"},
                },
                # Top-level is NOT OAuth-allowed. If creation lets
                # this win over the slot override, the slot starts
                # dead. If precedence holds, the slot override wins
                # at BOTH validation and creation and we're fine.
                "model": "openai/gpt-4.1-mini",
            },
        )
        assert resp.status_code == 200, resp.text
        # The container manager call records the model that was
        # actually used at creation — assert it matches what
        # validation accepted (the slot override).
        container_mgr.start_agent.assert_called_once()
        call_kwargs = container_mgr.start_agent.call_args.kwargs
        assert call_kwargs.get("model") == "openai/gpt-5-mini", (
            "Validation accepted the slot override but creation used a "
            "different model — validation/creation precedence drifted. "
            f"start_agent called with model={call_kwargs.get('model')!r}"
        )
    finally:
        bb.close()


def test_apply_template_slot_with_null_model_uses_default(
    tmp_path, _mesh_env, container_mgr, monkeypatch,
):
    """P1.2 — a template slot with ``model: null`` (explicit Python
    None) must coerce to the config default. Without this,
    ``slot_def.get('model', default)`` returns None and ``.replace()``
    crashes with ``AttributeError``.

    Use the OAuth-only happy path so credential validation accepts the
    default ``openai/gpt-5`` and the create flow can run.
    """
    _clear_system_env(monkeypatch)
    monkeypatch.setenv(
        "OPENLEGION_SYSTEM_OPENAI_OAUTH",
        '{"access_token":"tok","refresh_token":"ref"}',
    )
    vault = CredentialVault()

    fake_template = {
        "tmpl_null": {
            "description": "stub",
            "agents": {
                "worker": {
                    "role": "worker",
                    "model": None,  # the bug-trigger
                    "instructions": "do work",
                },
            },
        },
    }
    monkeypatch.setattr("src.cli.config._load_templates", lambda: fake_template)

    # Force the config default to a model the OAuth-only deployment
    # accepts, so once None coerces correctly, validation passes.
    import src.cli.config as cli_config_mod
    monkeypatch.setattr(
        cli_config_mod,
        "_load_config",
        lambda: {"llm": {"default_model": "openai/gpt-5"}},
    )
    # The mesh route also calls ``_load_config`` directly — patch the
    # host server's import the same way.
    import src.host.server as host_server_mod  # noqa: F401
    monkeypatch.setattr(
        "src.cli.config._load_config",
        lambda: {"llm": {"default_model": "openai/gpt-5"}},
    )

    client, bb = _build_mesh_app(tmp_path, container_mgr, vault)
    try:
        resp = client.post(
            "/mesh/fleet/apply",
            json={
                "spawned_by": "operator",
                "template": "tmpl_null",
            },
        )
        # Pre-fix, validation site would crash with AttributeError on
        # the .replace() call before any HTTP 4xx/5xx envelope. With
        # the helper in place, the None coerces to the config default
        # and the request succeeds.
        assert resp.status_code == 200, resp.text
        container_mgr.start_agent.assert_called_once()
        call_kwargs = container_mgr.start_agent.call_args.kwargs
        # Creation must end up with the config default, not None.
        assert call_kwargs.get("model") == "openai/gpt-5"
    finally:
        bb.close()
