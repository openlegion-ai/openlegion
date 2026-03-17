"""Tests for dashboard workspace proxy endpoints."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from src.dashboard.server import create_dashboard_router

_CSRF_HEADERS = {"X-Requested-With": "XMLHttpRequest"}


def _make_dashboard_app(transport=None, agent_registry=None, runtime=None):
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
        runtime=runtime,
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
                headers=_CSRF_HEADERS,
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
                headers=_CSRF_HEADERS,
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
                headers=_CSRF_HEADERS,
            )
            assert resp.status_code == 200
            # Check sanitized content was forwarded
            call_kwargs = transport.request.call_args[1]
            forwarded_content = call_kwargs["json"]["content"]
            assert "\u200B" not in forwarded_content
            assert "\u202E" not in forwarded_content
            assert "cleanvaluehere" in forwarded_content


class TestLogsProxy:
    @pytest.mark.asyncio
    async def test_logs_proxy(self):
        """GET /workspace-logs proxies to agent transport."""
        transport = AsyncMock()
        transport.request = AsyncMock(return_value={"logs": "- [10:00] Did work"})
        app = _make_dashboard_app(transport=transport)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.get(
                "/dashboard/api/agents/test_agent/workspace-logs?days=5",
            )
            assert resp.status_code == 200
            assert "Did work" in resp.json()["logs"]
            transport.request.assert_called_once_with(
                "test_agent", "GET", "/workspace-logs?days=5", timeout=10,
            )

    @pytest.mark.asyncio
    async def test_logs_agent_not_found(self):
        """Non-existent agent returns 404."""
        transport = AsyncMock()
        app = _make_dashboard_app(transport=transport)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.get(
                "/dashboard/api/agents/nonexistent/workspace-logs",
            )
            assert resp.status_code == 404


class TestLearningsProxy:
    @pytest.mark.asyncio
    async def test_learnings_proxy(self):
        """GET /workspace-learnings proxies to agent transport."""
        transport = AsyncMock()
        transport.request = AsyncMock(return_value={
            "errors": "- timeout error",
            "corrections": "- use duckduckgo",
        })
        app = _make_dashboard_app(transport=transport)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.get(
                "/dashboard/api/agents/test_agent/workspace-learnings",
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "timeout" in data["errors"]
            assert "duckduckgo" in data["corrections"]
            transport.request.assert_called_once_with(
                "test_agent", "GET", "/workspace-learnings", timeout=10,
            )

    @pytest.mark.asyncio
    async def test_learnings_agent_not_found(self):
        """Non-existent agent returns 404."""
        transport = AsyncMock()
        app = _make_dashboard_app(transport=transport)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.get(
                "/dashboard/api/agents/nonexistent/workspace-learnings",
            )
            assert resp.status_code == 404


class TestProjectProxy:
    @pytest.fixture
    def tmp_project_dir(self):
        d = tempfile.mkdtemp()
        yield Path(d)
        shutil.rmtree(d, ignore_errors=True)

    def _make_runtime(self, project_root: Path):
        runtime = MagicMock()
        runtime.project_root = project_root
        return runtime

    def _setup_project(self, projects_dir: Path, name: str = "testproj", *, with_md: bool = False):
        """Create a project directory with metadata for testing."""
        import yaml
        proj_dir = projects_dir / name
        proj_dir.mkdir(parents=True, exist_ok=True)
        (proj_dir / "metadata.yaml").write_text(
            yaml.dump({"name": name, "members": ["test_agent"]})
        )
        if with_md:
            (proj_dir / "project.md").write_text("# My Project")
        return proj_dir

    @pytest.mark.asyncio
    async def test_read_project_requires_param(self):
        """GET /api/project without project param returns 400."""
        runtime = MagicMock()
        app = _make_dashboard_app(runtime=runtime)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.get("/dashboard/api/project")
            assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_read_project_empty(self, tmp_project_dir):
        """GET /api/project?project=X returns empty when project.md doesn't exist."""
        from unittest.mock import patch
        projects_dir = tmp_project_dir / "projects"
        self._setup_project(projects_dir)
        runtime = self._make_runtime(tmp_project_dir)
        app = _make_dashboard_app(runtime=runtime)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            with patch("src.cli.config.PROJECTS_DIR", projects_dir):
                resp = await client.get("/dashboard/api/project", params={"project": "testproj"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["content"] == ""
            assert data["exists"] is False

    @pytest.mark.asyncio
    async def test_read_project_with_content(self, tmp_project_dir):
        """GET /api/project?project=X returns content from project.md."""
        from unittest.mock import patch
        projects_dir = tmp_project_dir / "projects"
        self._setup_project(projects_dir, with_md=True)
        runtime = self._make_runtime(tmp_project_dir)
        app = _make_dashboard_app(runtime=runtime)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            with patch("src.cli.config.PROJECTS_DIR", projects_dir):
                resp = await client.get("/dashboard/api/project", params={"project": "testproj"})
            assert resp.status_code == 200
            data = resp.json()
            assert "My Project" in data["content"]
            assert data["exists"] is True

    @pytest.mark.asyncio
    async def test_read_project_no_runtime(self):
        """GET /api/project?project=X returns 503 without runtime."""
        app = _make_dashboard_app(runtime=None)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.get("/dashboard/api/project", params={"project": "testproj"})
            assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_write_project_requires_param(self):
        """PUT /api/project without project param returns 400."""
        runtime = MagicMock()
        app = _make_dashboard_app(runtime=runtime)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.put(
                "/dashboard/api/project",
                json={"content": "anything"},
                headers=_CSRF_HEADERS,
            )
            assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_write_project_saves_and_pushes(self, tmp_project_dir):
        """PUT /api/project?project=X saves to project dir and pushes to members."""
        from unittest.mock import patch
        projects_dir = tmp_project_dir / "projects"
        proj_dir = self._setup_project(projects_dir)
        runtime = self._make_runtime(tmp_project_dir)
        transport_mock = AsyncMock()
        transport_mock.request = AsyncMock(return_value={"updated": True, "size": 20})
        app = _make_dashboard_app(
            transport=transport_mock, runtime=runtime,
        )
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            with patch("src.cli.config.PROJECTS_DIR", projects_dir):
                resp = await client.put(
                    "/dashboard/api/project",
                    params={"project": "testproj"},
                    json={"content": "# Updated Project"},
                    headers=_CSRF_HEADERS,
                )
            assert resp.status_code == 200
            data = resp.json()
            assert data["saved"] is True
            assert data["size"] > 0
            assert data["pushed"]["test_agent"] is True

            # Verify file was written to project dir
            content = (proj_dir / "project.md").read_text()
            assert "Updated Project" in content

            # Verify transport was called to push to member agent
            transport_mock.request.assert_called_once()
            call_args = transport_mock.request.call_args
            assert call_args[0][0] == "test_agent"
            assert call_args[0][1] == "PUT"
            assert call_args[0][2] == "/project"

    @pytest.mark.asyncio
    async def test_write_project_no_runtime(self):
        """PUT /api/project?project=X returns 503 without runtime."""
        app = _make_dashboard_app(runtime=None)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.put(
                "/dashboard/api/project",
                params={"project": "testproj"},
                json={"content": "anything"},
                headers=_CSRF_HEADERS,
            )
            assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_write_project_sanitizes_content(self, tmp_project_dir):
        """PUT /api/project?project=X sanitizes invisible characters."""
        from unittest.mock import patch
        projects_dir = tmp_project_dir / "projects"
        self._setup_project(projects_dir)
        runtime = self._make_runtime(tmp_project_dir)
        app = _make_dashboard_app(runtime=runtime)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            with patch("src.cli.config.PROJECTS_DIR", projects_dir):
                resp = await client.put(
                    "/dashboard/api/project",
                    params={"project": "testproj"},
                    json={"content": "clean\u200Btext\u202Ehere"},
                    headers=_CSRF_HEADERS,
                )
            assert resp.status_code == 200
            proj_dir = projects_dir / "testproj"
            content = (proj_dir / "project.md").read_text()
            assert "\u200B" not in content
            assert "\u202E" not in content
            assert "cleantexthere" in content
