"""Tests for agent workspace and heartbeat-context endpoints."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from src.agent.server import create_agent_app


def _make_app(workspace_dir: str | None = None) -> tuple:
    """Create agent app with mock loop and optional real workspace."""
    loop = MagicMock()
    loop.agent_id = "test_agent"
    loop.role = "researcher"
    loop.state = "idle"
    loop.skills = MagicMock()
    loop.skills.list_skills = MagicMock(return_value=[])
    loop.skills.get_tool_definitions = MagicMock(return_value=[])

    if workspace_dir:
        from src.agent.workspace import WorkspaceManager
        loop.workspace = WorkspaceManager(workspace_dir=workspace_dir)
    else:
        loop.workspace = None

    app = create_agent_app(loop)
    return app, loop


@pytest.fixture
def tmp_workspace():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


class TestWorkspaceList:
    @pytest.mark.asyncio
    async def test_list_with_workspace(self, tmp_workspace):
        app, _ = _make_app(tmp_workspace)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/workspace")
            assert resp.status_code == 200
            data = resp.json()
            names = [f["name"] for f in data["files"]]
            assert "SOUL.md" in names
            assert "HEARTBEAT.md" in names
            assert "MEMORY.md" in names

    @pytest.mark.asyncio
    async def test_list_without_workspace(self):
        app, _ = _make_app(None)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/workspace")
            assert resp.status_code == 200
            assert resp.json()["files"] == []


class TestWorkspaceReadWrite:
    @pytest.mark.asyncio
    async def test_round_trip(self, tmp_workspace):
        app, _ = _make_app(tmp_workspace)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            # Write
            resp = await client.put(
                "/workspace/SOUL.md",
                json={"content": "# My Soul\n\nI am a test agent."},
            )
            assert resp.status_code == 200
            assert resp.json()["size"] > 0

            # Read back
            resp = await client.get("/workspace/SOUL.md")
            assert resp.status_code == 200
            assert "My Soul" in resp.json()["content"]

    @pytest.mark.asyncio
    async def test_disallowed_filename_rejected(self, tmp_workspace):
        app, _ = _make_app(tmp_workspace)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/workspace/SECRET.md")
            assert resp.status_code == 400

            resp = await client.put(
                "/workspace/PROJECT.md",
                json={"content": "hacked"},
            )
            assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_content_sanitized(self, tmp_workspace):
        app, _ = _make_app(tmp_workspace)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            # Write with invisible chars
            resp = await client.put(
                "/workspace/USER.md",
                json={"content": "clean\u200Bvalue\u202Ehere"},
            )
            assert resp.status_code == 200

            resp = await client.get("/workspace/USER.md")
            content = resp.json()["content"]
            assert "\u200B" not in content
            assert "\u202E" not in content
            assert "cleanvaluehere" in content


class TestHeartbeatContext:
    @pytest.mark.asyncio
    async def test_fields_present(self, tmp_workspace):
        app, _ = _make_app(tmp_workspace)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/heartbeat-context")
            assert resp.status_code == 200
            data = resp.json()
            assert "heartbeat_rules" in data
            assert "daily_logs" in data
            assert "is_default_heartbeat" in data
            assert "has_recent_activity" in data

    @pytest.mark.asyncio
    async def test_default_detection(self, tmp_workspace):
        app, _ = _make_app(tmp_workspace)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            # Default scaffold HEARTBEAT.md should be detected
            resp = await client.get("/heartbeat-context")
            assert resp.json()["is_default_heartbeat"] is True

            # Custom HEARTBEAT.md should not be detected as default
            await client.put(
                "/workspace/HEARTBEAT.md",
                json={"content": "# My Custom Rules\n\nCheck email every hour."},
            )
            resp = await client.get("/heartbeat-context")
            assert resp.json()["is_default_heartbeat"] is False

    @pytest.mark.asyncio
    async def test_no_workspace(self):
        app, _ = _make_app(None)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/heartbeat-context")
            assert resp.status_code == 200
            data = resp.json()
            assert data["is_default_heartbeat"] is True
            assert data["has_recent_activity"] is False

    @pytest.mark.asyncio
    async def test_daily_logs_reflected(self, tmp_workspace):
        app, loop = _make_app(tmp_workspace)
        # Write a daily log entry
        loop.workspace.append_daily_log("Did some work")

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/heartbeat-context")
            data = resp.json()
            assert data["has_recent_activity"] is True
            assert "Did some work" in data["daily_logs"]
