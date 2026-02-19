"""Unit tests for credential vault."""

import pytest

from src.host.credentials import CredentialVault
from src.shared.types import APIProxyRequest


@pytest.fixture
def vault():
    v = CredentialVault()
    return v


def test_load_credentials_from_env(monkeypatch):
    monkeypatch.setenv("OPENLEGION_CRED_ANTHROPIC_API_KEY", "sk-test-123")
    monkeypatch.setenv("OPENLEGION_CRED_BRAVE_SEARCH_API_KEY", "bsk-456")
    v = CredentialVault()
    assert v.credentials.get("anthropic_api_key") == "sk-test-123"
    assert v.credentials.get("brave_search_api_key") == "bsk-456"


async def test_unknown_service_returns_error(vault):
    req = APIProxyRequest(service="nonexistent", action="do_thing")
    result = await vault.execute_api_call(req)
    assert not result.success
    assert "Unknown service" in result.error


async def test_missing_api_key_returns_error(monkeypatch):
    monkeypatch.delenv("OPENLEGION_CRED_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    v = CredentialVault()
    req = APIProxyRequest(service="anthropic", action="chat")
    result = await v.execute_api_call(req)
    assert not result.success
    assert "not configured" in result.error


def test_handler_dispatch():
    v = CredentialVault()
    assert "llm" in v.service_handlers
    assert "anthropic" in v.service_handlers
    assert "apollo" in v.service_handlers
    assert "hunter" in v.service_handlers
    assert "brave_search" in v.service_handlers
    assert "openai" in v.service_handlers


async def test_llm_missing_key_returns_error(monkeypatch):
    monkeypatch.delenv("OPENLEGION_CRED_OPENAI_API_KEY", raising=False)
    v = CredentialVault()
    req = APIProxyRequest(
        service="llm",
        action="chat",
        params={"model": "openai/gpt-4o-mini", "messages": []},
    )
    result = await v.execute_api_call(req)
    assert not result.success
    assert "No API key" in result.error


def test_get_api_key_for_model(monkeypatch):
    monkeypatch.setenv("OPENLEGION_CRED_OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("OPENLEGION_CRED_ANTHROPIC_API_KEY", "sk-anthropic")
    v = CredentialVault()
    assert v._get_api_key_for_model("openai/gpt-4o-mini") == "sk-openai"
    assert v._get_api_key_for_model("gpt-4o") == "sk-openai"
    assert v._get_api_key_for_model("anthropic/claude-sonnet-4-5-20250929") == "sk-anthropic"
    assert v._get_api_key_for_model("unknown/model") is None


async def test_unknown_action_returns_error(monkeypatch):
    monkeypatch.setenv("OPENLEGION_CRED_ANTHROPIC_API_KEY", "sk-test")
    v = CredentialVault()
    req = APIProxyRequest(service="anthropic", action="nonexistent_action")
    result = await v.execute_api_call(req)
    assert not result.success
    assert "Unknown action" in result.error
