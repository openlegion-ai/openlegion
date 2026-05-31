"""Tests for the operator ``list_available_models`` tool (Fix 2 in seam follow-up).

Operator-only tool that queries the mesh ``/mesh/introspect?section=llm``
payload and returns a stable shape (per-provider allowed models + credential
kinds) so the operator never has to memorize which models work with OAuth-only
credentials.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture(autouse=True)
def _set_operator_env(monkeypatch):
    """list_available_models requires ALLOWED_TOOLS to be set (defence-in-depth)."""
    monkeypatch.setenv("ALLOWED_TOOLS", "list_available_models")


def _mesh_with_llm_payload(payload: dict) -> MagicMock:
    mc = MagicMock()
    mc.introspect = AsyncMock(return_value={"llm": payload})
    return mc


@pytest.mark.asyncio
async def test_list_available_models_all_providers():
    """No provider arg → returns per-provider dict with kind + models."""
    from src.agent.builtins.operator_tools import list_available_models

    mc = _mesh_with_llm_payload({
        "available_providers": ["anthropic", "openai"],
        "allowed_models": {
            "openai": ["openai/gpt-5", "openai/gpt-5-mini"],
            "anthropic": ["anthropic/claude-sonnet-4-6"],
        },
        "credential_kinds": {"openai": "api_key", "anthropic": "oauth"},
    })

    result = await list_available_models(mesh_client=mc)
    assert "providers" in result
    assert set(result["providers"].keys()) == {"openai", "anthropic"}
    assert result["providers"]["openai"]["credential_kind"] == "api_key"
    assert result["providers"]["openai"]["allowed_models"] == [
        "openai/gpt-5", "openai/gpt-5-mini",
    ]
    assert result["providers"]["anthropic"]["credential_kind"] == "oauth"


@pytest.mark.asyncio
async def test_list_available_models_provider_filter():
    """Provider arg → narrowed payload with kind + models."""
    from src.agent.builtins.operator_tools import list_available_models

    mc = _mesh_with_llm_payload({
        "available_providers": ["anthropic", "openai"],
        "allowed_models": {
            "openai": ["openai/gpt-5"],
            "anthropic": ["anthropic/claude-sonnet-4-6"],
        },
        "credential_kinds": {"openai": "oauth", "anthropic": "api_key"},
    })

    result = await list_available_models(provider="openai", mesh_client=mc)
    assert result == {
        "provider": "openai",
        "credential_kind": "oauth",
        "allowed_models": ["openai/gpt-5"],
    }


@pytest.mark.asyncio
async def test_list_available_models_unconfigured_provider_filter():
    """Asking for a provider with no creds returns kind='none', empty list."""
    from src.agent.builtins.operator_tools import list_available_models

    mc = _mesh_with_llm_payload({
        "available_providers": ["openai"],
        "allowed_models": {"openai": ["openai/gpt-5"]},
        "credential_kinds": {"openai": "api_key"},
    })

    result = await list_available_models(provider="gemini", mesh_client=mc)
    assert result["provider"] == "gemini"
    assert result["credential_kind"] == "none"
    assert result["allowed_models"] == []


@pytest.mark.asyncio
async def test_list_available_models_rejects_non_operator(monkeypatch):
    """ALLOWED_TOOLS empty → non-operator → error."""
    from src.agent.builtins.operator_tools import list_available_models

    monkeypatch.setenv("ALLOWED_TOOLS", "")  # Non-operator
    mc = _mesh_with_llm_payload({})
    result = await list_available_models(mesh_client=mc)
    assert "error" in result
    assert "operator" in result["error"]


@pytest.mark.asyncio
async def test_list_available_models_handles_mesh_failure():
    """A mesh error returns a friendly error dict, not a raise."""
    from src.agent.builtins.operator_tools import list_available_models

    mc = MagicMock()
    mc.introspect = AsyncMock(side_effect=RuntimeError("mesh unreachable"))
    result = await list_available_models(mesh_client=mc)
    assert "error" in result
    assert "mesh" in result["error"].lower()


@pytest.mark.asyncio
async def test_list_available_models_handles_missing_llm_key():
    """If the mesh response doesn't wrap under 'llm', tool reads top-level."""
    from src.agent.builtins.operator_tools import list_available_models

    # Some mesh paths return the section content unwrapped.
    mc = MagicMock()
    mc.introspect = AsyncMock(return_value={
        "allowed_models": {"openai": ["openai/gpt-5"]},
        "credential_kinds": {"openai": "api_key"},
    })
    result = await list_available_models(mesh_client=mc)
    assert "providers" in result
    assert "openai" in result["providers"]
