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

    @pytest.mark.asyncio
    async def test_list_includes_cap_and_is_default(self, tmp_workspace):
        """Workspace list includes cap and is_default fields per file."""
        app, _ = _make_app(tmp_workspace)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/workspace")
            assert resp.status_code == 200
            files = resp.json()["files"]
            soul = next(f for f in files if f["name"] == "SOUL.md")
            assert soul["cap"] == 4000
            assert soul["is_default"] is True
            memory = next(f for f in files if f["name"] == "MEMORY.md")
            assert memory["cap"] == 16000
            heartbeat = next(f for f in files if f["name"] == "HEARTBEAT.md")
            assert heartbeat["cap"] is None

    @pytest.mark.asyncio
    async def test_list_detects_customized_file(self, tmp_workspace):
        """Custom content marks is_default as false."""
        app, _ = _make_app(tmp_workspace)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            # Write custom content
            await client.put(
                "/workspace/SOUL.md",
                json={"content": "# My Custom Soul\n\nI am unique."},
            )
            resp = await client.get("/workspace")
            files = resp.json()["files"]
            soul = next(f for f in files if f["name"] == "SOUL.md")
            assert soul["is_default"] is False


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


class TestProjectEndpoint:
    @pytest.mark.asyncio
    async def test_update_project(self, tmp_workspace):
        """PUT /project writes PROJECT.md to workspace."""
        app, loop = _make_app(tmp_workspace)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.put(
                "/project",
                json={"content": "# My Project\n\nBuild a web app."},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["updated"] is True
            assert data["size"] > 0

            # Verify file was written
            project_path = Path(tmp_workspace) / "PROJECT.md"
            assert project_path.exists()
            assert "My Project" in project_path.read_text()

    @pytest.mark.asyncio
    async def test_update_project_sanitizes_content(self, tmp_workspace):
        """PUT /project sanitizes invisible characters."""
        app, _ = _make_app(tmp_workspace)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.put(
                "/project",
                json={"content": "clean\u200Btext\u202Ehere"},
            )
            assert resp.status_code == 200
            content = (Path(tmp_workspace) / "PROJECT.md").read_text()
            assert "\u200B" not in content
            assert "\u202E" not in content
            assert "cleantexthere" in content

    @pytest.mark.asyncio
    async def test_update_project_no_workspace(self):
        """PUT /project returns 503 without workspace."""
        app, _ = _make_app(None)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.put(
                "/project",
                json={"content": "anything"},
            )
            assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_update_project_rejects_non_string(self, tmp_workspace):
        """PUT /project rejects non-string content."""
        app, _ = _make_app(tmp_workspace)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.put(
                "/project",
                json={"content": 12345},
            )
            assert resp.status_code == 400


class TestWorkspaceLogs:
    @pytest.mark.asyncio
    async def test_workspace_logs_returns_content(self, tmp_workspace):
        """Logs endpoint returns daily log entries."""
        app, loop = _make_app(tmp_workspace)
        loop.workspace.append_daily_log("Built the feature")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/workspace-logs")
            assert resp.status_code == 200
            assert "Built the feature" in resp.json()["logs"]

    @pytest.mark.asyncio
    async def test_workspace_logs_empty_without_workspace(self):
        """No workspace returns empty string."""
        app, _ = _make_app(None)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/workspace-logs")
            assert resp.status_code == 200
            assert resp.json()["logs"] == ""

    @pytest.mark.asyncio
    async def test_workspace_logs_days_clamped(self, tmp_workspace):
        """Extreme days value is clamped without error."""
        app, _ = _make_app(tmp_workspace)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/workspace-logs?days=100")
            assert resp.status_code == 200
            assert "logs" in resp.json()


class TestWorkspaceLearnings:
    @pytest.mark.asyncio
    async def test_workspace_learnings_returns_content(self, tmp_workspace):
        """Learnings endpoint returns errors and corrections."""
        app, loop = _make_app(tmp_workspace)
        loop.workspace.record_error("web_search", "timeout after 30s")
        loop.workspace.record_correction("use google", "use duckduckgo instead")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/workspace-learnings")
            assert resp.status_code == 200
            data = resp.json()
            assert "timeout" in data["errors"]
            assert "duckduckgo" in data["corrections"]

    @pytest.mark.asyncio
    async def test_workspace_learnings_empty_without_workspace(self):
        """No workspace returns empty strings."""
        app, _ = _make_app(None)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/workspace-learnings")
            assert resp.status_code == 200
            data = resp.json()
            assert data["errors"] == ""
            assert data["corrections"] == ""
