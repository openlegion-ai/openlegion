"""Tests for dashboard workspace proxy endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from src.dashboard.server import create_dashboard_router


def _make_dashboard_app(transport=None, agent_registry=None):
    """Create a FastAPI app with dashboard router for testing."""
    from fastapi import FastAPI

    if agent_registry is None:
        agent_registry = {"test_agent": "http://localhost:8401"}

    blackboard = MagicMock()
    blackboard.list_by_prefix = MagicMock(return_value=[])
    health_monitor = MagicMock()
    health_monitor.get_status = MagicMock(return_value=[])
    cost_tracker = MagicMock()
    cost_tracker.get_all_agents_spend = MagicMock(return_value=[])
    cost_tracker.get_spend = MagicMock(return_value={})
    cost_tracker.check_budget = MagicMock(return_value={})

    router = create_dashboard_router(
        blackboard=blackboard,
        health_monitor=health_monitor,
        cost_tracker=cost_tracker,
        trace_store=None,
        event_bus=None,
        agent_registry=agent_registry,
        transport=transport,
    )
    app = FastAPI()
    app.include_router(router)
    return app


class TestWorkspaceProxy:
    @pytest.mark.asyncio
    async def test_list_workspace_proxies(self):
        """GET /workspace proxies to agent transport."""
        transport = AsyncMock()
        transport.request = AsyncMock(return_value={
            "files": [{"name": "SOUL.md", "size": 100}],
        })
        app = _make_dashboard_app(transport=transport)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.get("/dashboard/api/agents/test_agent/workspace")
            assert resp.status_code == 200
            assert resp.json()["files"][0]["name"] == "SOUL.md"
            transport.request.assert_called_once_with(
                "test_agent", "GET", "/workspace", timeout=10,
            )

    @pytest.mark.asyncio
    async def test_read_workspace_proxies(self):
        """GET /workspace/{filename} proxies to agent."""
        transport = AsyncMock()
        transport.request = AsyncMock(return_value={
            "filename": "SOUL.md", "content": "# Test Soul",
        })
        app = _make_dashboard_app(transport=transport)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.get(
                "/dashboard/api/agents/test_agent/workspace/SOUL.md",
            )
            assert resp.status_code == 200
            assert "Test Soul" in resp.json()["content"]

    @pytest.mark.asyncio
    async def test_write_workspace_proxies(self):
        """PUT /workspace/{filename} proxies to agent with sanitized content."""
        transport = AsyncMock()
        transport.request = AsyncMock(return_value={
            "filename": "SOUL.md", "size": 42,
        })
        app = _make_dashboard_app(transport=transport)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.put(
                "/dashboard/api/agents/test_agent/workspace/SOUL.md",
                json={"content": "# Updated Soul"},
            )
            assert resp.status_code == 200
            # Verify transport was called with correct args
            call_args = transport.request.call_args
            assert call_args[0][0] == "test_agent"
            assert call_args[0][1] == "PUT"
            assert "/workspace/SOUL.md" in call_args[0][2]

    @pytest.mark.asyncio
    async def test_disallowed_filename_returns_400(self):
        """Disallowed filenames are rejected at the dashboard level."""
        transport = AsyncMock()
        app = _make_dashboard_app(transport=transport)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.get(
                "/dashboard/api/agents/test_agent/workspace/SECRET.md",
            )
            assert resp.status_code == 400

            resp = await client.put(
                "/dashboard/api/agents/test_agent/workspace/PROJECT.md",
                json={"content": "hacked"},
            )
            assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_agent_not_found_returns_404(self):
        """Non-existent agent returns 404."""
        transport = AsyncMock()
        app = _make_dashboard_app(transport=transport)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.get(
                "/dashboard/api/agents/nonexistent/workspace",
            )
            assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_transport_failure_returns_502(self):
        """Transport errors return 502."""
        transport = AsyncMock()
        transport.request = AsyncMock(side_effect=RuntimeError("connection refused"))
        app = _make_dashboard_app(transport=transport)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.get(
                "/dashboard/api/agents/test_agent/workspace",
            )
            assert resp.status_code == 502

    @pytest.mark.asyncio
    async def test_write_sanitizes_content(self):
        """Content with invisible chars is sanitized before forwarding."""
        transport = AsyncMock()
        transport.request = AsyncMock(return_value={"filename": "USER.md", "size": 10})
        app = _make_dashboard_app(transport=transport)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.put(
                "/dashboard/api/agents/test_agent/workspace/USER.md",
                json={"content": "clean\u200Bvalue\u202Ehere"},
            )
            assert resp.status_code == 200
            # Check sanitized content was forwarded
            call_kwargs = transport.request.call_args[1]
            forwarded_content = call_kwargs["json"]["content"]
            assert "\u200B" not in forwarded_content
            assert "\u202E" not in forwarded_content
            assert "cleanvaluehere" in forwarded_content
