"""Tests for vault tools (credential-blind agent tools)."""

from unittest.mock import AsyncMock

import pytest


class TestVaultGenerateSecret:
    @pytest.mark.asyncio
    async def test_no_value_in_return(self):
        """The actual secret value must NEVER appear in the return dict."""
        from src.agent.builtins.vault_tool import vault_generate_secret

        mock_client = AsyncMock()
        mock_client.vault_store.return_value = {"stored": True, "handle": "$CRED{test_key}"}

        result = await vault_generate_secret(
            name="test_key", length=32, charset="urlsafe", mesh_client=mock_client,
        )
        assert result["stored"] is True
        assert result["handle"] == "$CRED{test_key}"
        assert "value" not in result

        # Verify vault_store was called with a non-empty value
        call_args = mock_client.vault_store.call_args
        assert len(call_args[0][1]) > 0  # value is non-empty

    @pytest.mark.asyncio
    async def test_charsets_hex(self):
        from src.agent.builtins.vault_tool import vault_generate_secret

        mock_client = AsyncMock()
        mock_client.vault_store.return_value = {"stored": True, "handle": "$CRED{hex_key}"}

        result = await vault_generate_secret(
            name="hex_key", length=16, charset="hex", mesh_client=mock_client,
        )
        assert result["stored"] is True

        # Verify hex charset used
        stored_value = mock_client.vault_store.call_args[0][1]
        assert all(c in "0123456789abcdef" for c in stored_value)
        assert len(stored_value) == 16

    @pytest.mark.asyncio
    async def test_charsets_alphanumeric(self):
        from src.agent.builtins.vault_tool import vault_generate_secret

        mock_client = AsyncMock()
        mock_client.vault_store.return_value = {"stored": True, "handle": "$CRED{alnum_key}"}

        result = await vault_generate_secret(
            name="alnum_key", length=24, charset="alphanumeric", mesh_client=mock_client,
        )
        assert result["stored"] is True

        stored_value = mock_client.vault_store.call_args[0][1]
        assert stored_value.isalnum()
        assert len(stored_value) == 24

    @pytest.mark.asyncio
    async def test_charsets_urlsafe(self):
        from src.agent.builtins.vault_tool import vault_generate_secret

        mock_client = AsyncMock()
        mock_client.vault_store.return_value = {"stored": True, "handle": "$CRED{url_key}"}

        result = await vault_generate_secret(
            name="url_key", length=32, charset="urlsafe", mesh_client=mock_client,
        )
        assert result["stored"] is True
        assert "value" not in result

    @pytest.mark.asyncio
    async def test_unknown_charset_returns_error(self):
        from src.agent.builtins.vault_tool import vault_generate_secret

        mock_client = AsyncMock()
        result = await vault_generate_secret(
            name="key", charset="bogus", mesh_client=mock_client,
        )
        assert "error" in result
        assert "Unknown charset" in result["error"]

    @pytest.mark.asyncio
    async def test_no_mesh_client_returns_error(self):
        from src.agent.builtins.vault_tool import vault_generate_secret

        result = await vault_generate_secret(name="key", mesh_client=None)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_empty_name_returns_error(self):
        from src.agent.builtins.vault_tool import vault_generate_secret

        mock_client = AsyncMock()
        result = await vault_generate_secret(name="", mesh_client=mock_client)
        assert "error" in result


class TestVaultListTool:
    @pytest.mark.asyncio
    async def test_vault_list_returns_names(self):
        from src.agent.builtins.vault_tool import vault_list

        mock_client = AsyncMock()
        mock_client.vault_list.return_value = ["brave_search", "anthropic_api_key"]

        result = await vault_list(mesh_client=mock_client)
        assert result["credentials"] == ["brave_search", "anthropic_api_key"]
        assert result["count"] == 2

    @pytest.mark.asyncio
    async def test_vault_list_no_mesh_client(self):
        from src.agent.builtins.vault_tool import vault_list

        result = await vault_list(mesh_client=None)
        assert "error" in result


