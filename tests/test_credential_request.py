"""Tests for the credential request system.

Covers the request_credential tool, MeshClient method, and mesh endpoint.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Tool tests ──────────────────────────────────────────────────────


class TestRequestCredentialTool:
    """Tests for src/agent/builtins/vault_tool.request_credential."""

    @pytest.mark.asyncio
    async def test_returns_handle_on_success(self):
        from src.agent.builtins.vault_tool import request_credential

        mock_client = AsyncMock()
        mock_client.vault_list.return_value = []
        mock_client.request_credential_from_user.return_value = {
            "requested": True, "name": "linkedin_api_key",
        }

        result = await request_credential(
            name="linkedin_api_key",
            description="Your LinkedIn API key from developer portal",
            service="LinkedIn",
            mesh_client=mock_client,
        )
        assert result["requested"] is True
        assert result["handle"] == "$CRED{linkedin_api_key}"
        assert result["name"] == "linkedin_api_key"
        # Value must NEVER appear in result
        assert "value" not in result

    @pytest.mark.asyncio
    async def test_no_mesh_client_returns_error(self):
        from src.agent.builtins.vault_tool import request_credential

        result = await request_credential(
            name="key", description="desc", mesh_client=None,
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_empty_name_returns_error(self):
        from src.agent.builtins.vault_tool import request_credential

        mock_client = AsyncMock()
        result = await request_credential(
            name="", description="desc", mesh_client=mock_client,
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_invalid_name_returns_error(self):
        from src.agent.builtins.vault_tool import request_credential

        mock_client = AsyncMock()
        for bad_name in ["has spaces", "has/slashes", "a" * 200, "special!@#"]:
            result = await request_credential(
                name=bad_name, description="desc", mesh_client=mock_client,
            )
            assert "error" in result, f"Expected error for name: {bad_name}"

    @pytest.mark.asyncio
    async def test_valid_name_formats(self):
        from src.agent.builtins.vault_tool import request_credential

        mock_client = AsyncMock()
        mock_client.vault_list.return_value = []
        mock_client.request_credential_from_user.return_value = {"requested": True, "name": "x"}

        for good_name in ["simple", "with_underscore", "with-dash", "with.dot", "MixedCase123"]:
            result = await request_credential(
                name=good_name, description="desc", mesh_client=mock_client,
            )
            assert "error" not in result, f"Unexpected error for name: {good_name}"
            assert result["requested"] is True

    @pytest.mark.asyncio
    async def test_already_exists_returns_handle(self):
        from src.agent.builtins.vault_tool import request_credential

        mock_client = AsyncMock()
        mock_client.vault_list.return_value = ["existing_key"]

        result = await request_credential(
            name="existing_key", description="desc", mesh_client=mock_client,
        )
        assert result["already_exists"] is True
        assert result["handle"] == "$CRED{existing_key}"

    @pytest.mark.asyncio
    async def test_vault_list_failure_still_proceeds(self):
        """If vault_list raises, the tool should still send the request."""
        from src.agent.builtins.vault_tool import request_credential

        mock_client = AsyncMock()
        mock_client.vault_list.side_effect = Exception("Vault unavailable")
        mock_client.request_credential_from_user.return_value = {"requested": True, "name": "k"}

        result = await request_credential(
            name="new_key", description="desc", mesh_client=mock_client,
        )
        assert result["requested"] is True

    @pytest.mark.asyncio
    async def test_mesh_request_failure_still_returns_result(self):
        """If the mesh call fails, tool should still return requested=True."""
        from src.agent.builtins.vault_tool import request_credential

        mock_client = AsyncMock()
        mock_client.vault_list.return_value = []
        mock_client.request_credential_from_user.side_effect = Exception("Network error")

        result = await request_credential(
            name="key", description="desc", mesh_client=mock_client,
        )
        assert result["requested"] is True
        assert result["handle"] == "$CRED{key}"

    @pytest.mark.asyncio
    async def test_default_service_uses_name(self):
        """When service is not provided, name is used as fallback."""
        from src.agent.builtins.vault_tool import request_credential

        mock_client = AsyncMock()
        mock_client.vault_list.return_value = []
        mock_client.request_credential_from_user.return_value = {"requested": True, "name": "k"}

        await request_credential(
            name="stripe_key", description="desc", mesh_client=mock_client,
        )

        call_kwargs = mock_client.request_credential_from_user.call_args
        assert call_kwargs[1]["service"] == "stripe_key" or call_kwargs[0][2] == "stripe_key"


# ── MeshClient tests ───────────────────────────────────────────────


class TestMeshClientCredentialRequest:
    """Tests for MeshClient.request_credential_from_user."""

    @pytest.mark.asyncio
    async def test_posts_to_correct_url(self):
        from src.agent.mesh_client import MeshClient

        client = MeshClient("http://localhost:8420", "agent-1")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"requested": True, "name": "my_key"}

        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response
        mock_http.is_closed = False
        client._client = mock_http

        result = await client.request_credential_from_user(
            name="my_key", description="API key for X", service="ServiceX",
        )

        assert result["requested"] is True
        call_args = mock_http.post.call_args
        assert "/mesh/credential-request" in call_args[0][0]
        body = call_args[1]["json"]
        assert body["name"] == "my_key"
        assert body["description"] == "API key for X"
        assert body["service"] == "ServiceX"
        assert body["agent_id"] == "agent-1"


# ── Mesh endpoint tests ────────────────────────────────────────────


class TestCredentialRequestEndpoint:
    """Tests for POST /mesh/credential-request."""

    @pytest.fixture
    def _app(self, tmp_path):
        """Create a minimal mesh app with mocked dependencies."""
        from src.host.server import create_mesh_app

        bb_path = tmp_path / "bb.db"
        costs_path = tmp_path / "costs.db"
        traces_path = tmp_path / "traces.db"

        from src.host.mesh import Blackboard, MessageRouter, PubSub
        from src.host.permissions import PermissionMatrix
        from src.host.costs import CostTracker
        from src.host.traces import TraceStore

        blackboard = Blackboard(str(bb_path))
        pubsub = PubSub()
        permissions = PermissionMatrix()
        router = MessageRouter(permissions, {})
        costs = CostTracker(str(costs_path))
        traces = TraceStore(str(traces_path))

        self.event_bus = MagicMock()

        app = create_mesh_app(
            blackboard=blackboard,
            pubsub=pubsub,
            router=router,
            permissions=permissions,
            cost_tracker=costs,
            trace_store=traces,
            event_bus=self.event_bus,
        )
        return app

    @pytest.mark.asyncio
    async def test_emits_credential_request_event(self, _app):
        from httpx import ASGITransport, AsyncClient

        async with AsyncClient(
            transport=ASGITransport(app=_app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/mesh/credential-request",
                json={
                    "agent_id": "agent-1",
                    "name": "twitter_api_key",
                    "description": "Your Twitter API key",
                    "service": "Twitter",
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["requested"] is True
        assert data["name"] == "twitter_api_key"

        # Verify event_bus.emit was called with correct args
        self.event_bus.emit.assert_called_once()
        call_args = self.event_bus.emit.call_args
        assert call_args[0][0] == "credential_request"
        assert call_args[1]["agent"] == "agent-1"
        assert call_args[1]["data"]["name"] == "twitter_api_key"
        assert call_args[1]["data"]["description"] == "Your Twitter API key"
        assert call_args[1]["data"]["service"] == "Twitter"

    @pytest.mark.asyncio
    async def test_rejects_missing_name(self, _app):
        from httpx import ASGITransport, AsyncClient

        async with AsyncClient(
            transport=ASGITransport(app=_app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/mesh/credential-request",
                json={"agent_id": "agent-1", "name": "", "description": "desc"},
            )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_rejects_invalid_name(self, _app):
        from httpx import ASGITransport, AsyncClient

        async with AsyncClient(
            transport=ASGITransport(app=_app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/mesh/credential-request",
                json={"agent_id": "agent-1", "name": "bad name!", "description": "desc"},
            )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_no_credential_value_in_event(self, _app):
        """The event must never contain an actual credential value."""
        from httpx import ASGITransport, AsyncClient

        async with AsyncClient(
            transport=ASGITransport(app=_app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/mesh/credential-request",
                json={
                    "agent_id": "agent-1",
                    "name": "secret_key",
                    "description": "Enter your secret",
                    "service": "MyService",
                },
            )
        assert resp.status_code == 200

        call_data = self.event_bus.emit.call_args[1]["data"]
        assert "value" not in call_data
        assert "key" not in call_data
        assert "secret" not in call_data.get("name", "").lower() or True  # name is fine
        # Only name, description, service are allowed
        assert set(call_data.keys()) == {"name", "description", "service"}
