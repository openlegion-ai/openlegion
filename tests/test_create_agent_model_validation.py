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
