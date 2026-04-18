"""Tests for agent workspace and heartbeat-context endpoints."""

from __future__ import annotations

import asyncio
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
    loop._excluded_tools = frozenset()
    loop.memory = None
    loop.mesh_client = MagicMock()
    loop.skills = MagicMock()
    loop.skills.list_skills = MagicMock(return_value=[])
    loop.skills.get_tool_definitions = MagicMock(return_value=[])
    loop.skills.get_tool_sources = MagicMock(return_value={})
    loop.skills.execute = AsyncMock(return_value={"ok": True})

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
                headers={"x-mesh-internal": "1"},
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
                headers={"x-mesh-internal": "1"},
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
                headers={"x-mesh-internal": "1"},
            )
            assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_write_requires_mesh_internal_header(self, tmp_workspace):
        """PUT /workspace rejects requests without X-Mesh-Internal header."""
        app, _ = _make_app(tmp_workspace)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.put(
                "/workspace/SOUL.md",
                json={"content": "# Hacked"},
            )
            assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_content_sanitized(self, tmp_workspace):
        app, _ = _make_app(tmp_workspace)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            # Write with invisible chars
            resp = await client.put(
                "/workspace/USER.md",
                json={"content": "clean\u200Bvalue\u202Ehere"},
                headers={"x-mesh-internal": "1"},
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
                headers={"x-mesh-internal": "1"},
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
                headers={"x-mesh-internal": "1"},
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
                headers={"x-mesh-internal": "1"},
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
                headers={"x-mesh-internal": "1"},
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
                headers={"x-mesh-internal": "1"},
            )
            assert resp.status_code == 400


class TestRuntimeConfig:
    @pytest.mark.asyncio
    async def test_update_model_hot_reloads_llm(self, tmp_workspace):
        """POST /config updates llm.default_model immediately."""
        app, loop = _make_app(tmp_workspace)
        loop.llm = MagicMock()
        loop.llm.default_model = "openai/gpt-4o-mini"
        loop.llm.thinking = "off"
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/config",
                json={"model": "openai/gpt-4o"},
                headers={"x-mesh-internal": "1"},
            )
            assert resp.status_code == 200
            assert resp.json() == {"updated": {"model": "openai/gpt-4o"}}
            assert loop.llm.default_model == "openai/gpt-4o"

    @pytest.mark.asyncio
    async def test_update_thinking_hot_reloads_llm(self, tmp_workspace):
        """POST /config updates llm.thinking immediately."""
        app, loop = _make_app(tmp_workspace)
        loop.llm = MagicMock()
        loop.llm.default_model = "openai/gpt-4o-mini"
        loop.llm.thinking = "off"
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/config",
                json={"thinking": "high"},
                headers={"x-mesh-internal": "1"},
            )
            assert resp.status_code == 200
            assert loop.llm.thinking == "high"

    @pytest.mark.asyncio
    async def test_update_config_requires_mesh_header(self, tmp_workspace):
        """POST /config rejects requests without X-Mesh-Internal header."""
        app, _ = _make_app(tmp_workspace)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/config", json={"model": "openai/gpt-4o"})
            assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_update_config_rejects_invalid_thinking(self, tmp_workspace):
        """POST /config rejects invalid thinking values."""
        app, loop = _make_app(tmp_workspace)
        loop.llm = MagicMock()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/config",
                json={"thinking": "extreme"},
                headers={"x-mesh-internal": "1"},
            )
            assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_update_config_rejects_empty_model(self, tmp_workspace):
        """POST /config rejects empty model string."""
        app, loop = _make_app(tmp_workspace)
        loop.llm = MagicMock()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/config",
                json={"model": ""},
                headers={"x-mesh-internal": "1"},
            )
            assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_update_model_and_thinking_together(self, tmp_workspace):
        """POST /config updates both fields in a single request."""
        app, loop = _make_app(tmp_workspace)
        loop.llm = MagicMock()
        loop.llm.default_model = "openai/gpt-4o-mini"
        loop.llm.thinking = "off"
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/config",
                json={"model": "openai/gpt-4o", "thinking": "medium"},
                headers={"x-mesh-internal": "1"},
            )
            assert resp.status_code == 200
            assert loop.llm.default_model == "openai/gpt-4o"
            assert loop.llm.thinking == "medium"
            assert resp.json()["updated"] == {
                "model": "openai/gpt-4o", "thinking": "medium",
            }

    @pytest.mark.asyncio
    async def test_update_config_atomic_on_invalid_thinking(self, tmp_workspace):
        """Invalid thinking rejects the whole request; model not partially applied."""
        app, loop = _make_app(tmp_workspace)
        loop.llm = MagicMock()
        loop.llm.default_model = "openai/gpt-4o-mini"
        loop.llm.thinking = "off"
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/config",
                json={"model": "openai/gpt-4o", "thinking": "bogus"},
                headers={"x-mesh-internal": "1"},
            )
            assert resp.status_code == 400
            # Neither field was applied
            assert loop.llm.default_model == "openai/gpt-4o-mini"
            assert loop.llm.thinking == "off"

    @pytest.mark.asyncio
    async def test_update_config_empty_body_is_noop(self, tmp_workspace):
        """POST /config with empty body returns empty updated dict, doesn't error."""
        app, loop = _make_app(tmp_workspace)
        loop.llm = MagicMock()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/config", json={}, headers={"x-mesh-internal": "1"},
            )
            assert resp.status_code == 200
            assert resp.json() == {"updated": {}}

    @pytest.mark.asyncio
    async def test_update_config_ignores_unknown_fields(self, tmp_workspace):
        """Unknown fields in body are silently ignored (not 400)."""
        app, loop = _make_app(tmp_workspace)
        loop.llm = MagicMock()
        loop.llm.default_model = "openai/gpt-4o-mini"
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/config",
                json={"foo": "bar", "model": "openai/gpt-4o"},
                headers={"x-mesh-internal": "1"},
            )
            assert resp.status_code == 200
            assert loop.llm.default_model == "openai/gpt-4o"
            assert "foo" not in resp.json()["updated"]

    @pytest.mark.asyncio
    async def test_update_config_rejects_non_string_model(self, tmp_workspace):
        """POST /config rejects non-string model value."""
        app, loop = _make_app(tmp_workspace)
        loop.llm = MagicMock()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/config",
                json={"model": 123},
                headers={"x-mesh-internal": "1"},
            )
            assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_update_config_returns_503_without_llm(self, tmp_workspace):
        """POST /config returns 503 if loop.llm is missing."""
        app, loop = _make_app(tmp_workspace)
        loop.llm = None
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/config",
                json={"model": "openai/gpt-4o"},
                headers={"x-mesh-internal": "1"},
            )
            assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_update_config_rejects_non_object_body(self, tmp_workspace):
        """POST /config rejects a JSON array/string body."""
        app, loop = _make_app(tmp_workspace)
        loop.llm = MagicMock()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/config",
                json=["model", "openai/gpt-4o"],
                headers={"x-mesh-internal": "1"},
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


class TestTaskCancellation:
    @pytest.mark.asyncio
    async def test_cancelled_task_resets_state_to_idle(self):
        """If a task is cancelled before execute_task completes, state resets to idle."""
        loop = MagicMock()
        loop.agent_id = "test_agent"
        loop.role = "researcher"
        loop.state = "idle"
        loop.current_task = None
        loop._current_task_handle = None
        loop._cancel_requested = False
        loop.skills = MagicMock()
        loop.skills.list_skills = MagicMock(return_value=[])
        loop.skills.get_tool_definitions = MagicMock(return_value=[])
        loop.workspace = None
        loop.mesh_client = AsyncMock()

        # Make execute_task block forever then raise CancelledError when cancelled
        cancel_event = asyncio.Event()

        async def slow_execute(*args, **kwargs):
            cancel_event.set()
            await asyncio.sleep(999)

        loop.execute_task = AsyncMock(side_effect=slow_execute)

        app = create_agent_app(loop)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            # Submit a task
            from src.shared.types import TaskAssignment
            assignment = TaskAssignment(
                workflow_id="wf1", step_id="s1",
                task_type="research", input_data={"q": "test"},
            )
            resp = await client.post("/task", json=assignment.model_dump(mode="json"))
            assert resp.json()["accepted"] is True
            assert loop.state == "working"

            # Wait for execute_task to start
            await cancel_event.wait()

            # Cancel the task
            resp = await client.post("/cancel")
            assert resp.json()["status"] == "cancelled"

            # Give the cancellation a moment to propagate
            await asyncio.sleep(0.1)

            # State should be reset to idle
            assert loop.state == "idle"
            assert loop.current_task is None


class TestArtifactDelete:
    @pytest.mark.asyncio
    async def test_delete_artifact(self, tmp_workspace):
        app, _ = _make_app(tmp_workspace)
        artifacts_dir = Path(tmp_workspace) / "artifacts"
        artifacts_dir.mkdir()
        (artifacts_dir / "report.md").write_text("# Report")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.delete("/artifacts/report.md")
            assert resp.status_code == 200
            assert resp.json()["deleted"] is True
            assert not (artifacts_dir / "report.md").exists()

    @pytest.mark.asyncio
    async def test_delete_artifact_not_found(self, tmp_workspace):
        app, _ = _make_app(tmp_workspace)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.delete("/artifacts/missing.txt")
            assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_artifact_invalid_name(self, tmp_workspace):
        app, _ = _make_app(tmp_workspace)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.delete("/artifacts/!invalid")
            assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_delete_artifact_cleans_empty_dirs(self, tmp_workspace):
        app, _ = _make_app(tmp_workspace)
        artifacts_dir = Path(tmp_workspace) / "artifacts"
        subdir = artifacts_dir / "sub" / "deep"
        subdir.mkdir(parents=True)
        (subdir / "file.txt").write_text("data")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.delete("/artifacts/sub/deep/file.txt")
            assert resp.status_code == 200
            # Empty parent dirs should be cleaned up
            assert not subdir.exists()
            assert not (artifacts_dir / "sub").exists()
            # artifacts_dir itself should remain
            assert artifacts_dir.exists()

    @pytest.mark.asyncio
    async def test_delete_artifact_preserves_nonempty_parent(self, tmp_workspace):
        app, _ = _make_app(tmp_workspace)
        artifacts_dir = Path(tmp_workspace) / "artifacts"
        subdir = artifacts_dir / "reports"
        subdir.mkdir(parents=True)
        (subdir / "a.txt").write_text("keep")
        (subdir / "b.txt").write_text("delete")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.delete("/artifacts/reports/b.txt")
            assert resp.status_code == 200
            # Sibling file keeps the directory alive
            assert subdir.exists()
            assert (subdir / "a.txt").exists()

    @pytest.mark.asyncio
    async def test_delete_artifact_no_workspace(self):
        app, _ = _make_app(workspace_dir=None)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.delete("/artifacts/file.txt")
            assert resp.status_code == 503


class TestInvokeTool:
    """Tests for POST /invoke — direct tool execution without LLM."""

    @pytest.mark.asyncio
    async def test_invoke_success(self):
        app, loop = _make_app()
        loop.skills.execute = AsyncMock(return_value={"number": 42})

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/invoke", json={"tool": "random_number", "params": {}})
        assert resp.status_code == 200
        assert resp.json() == {"result": {"number": 42}}
        loop.skills.execute.assert_called_once_with(
            "random_number", {},
            mesh_client=loop.mesh_client,
            workspace_manager=loop.workspace,
            memory_store=loop.memory,
        )

    @pytest.mark.asyncio
    async def test_invoke_with_params(self):
        app, loop = _make_app()
        loop.skills.execute = AsyncMock(return_value={"sent": True})

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/invoke", json={"tool": "notify_user", "params": {"message": "hello"}}
            )
        assert resp.status_code == 200
        loop.skills.execute.assert_called_once_with(
            "notify_user", {"message": "hello"},
            mesh_client=loop.mesh_client,
            workspace_manager=loop.workspace,
            memory_store=loop.memory,
        )

    @pytest.mark.asyncio
    async def test_invoke_missing_name_returns_400(self):
        app, _ = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/invoke", json={"params": {}})
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_invoke_params_not_dict_returns_400(self):
        app, _ = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/invoke", json={"tool": "notify_user", "params": "not-a-dict"})
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_invoke_unknown_tool_returns_404(self):
        app, loop = _make_app()
        loop.skills.execute = AsyncMock(side_effect=ValueError("Unknown skill: missing_tool"))

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/invoke", json={"tool": "missing_tool", "params": {}})
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_invoke_excluded_tool_returns_403(self):
        app, loop = _make_app()
        loop._excluded_tools = frozenset({"blackboard_read"})

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/invoke", json={"tool": "blackboard_read", "params": {}})
        assert resp.status_code == 403
        loop.skills.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_invoke_tool_runtime_error_returns_error_dict(self):
        app, loop = _make_app()
        loop.skills.execute = AsyncMock(side_effect=RuntimeError("tool exploded"))

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/invoke", json={"tool": "broken_tool", "params": {}})
        assert resp.status_code == 200
        assert "error" in resp.json()


# ── Fix 4f: X-Origin header propagation ────────────────────────

class TestChatOriginHeader:
    @pytest.mark.asyncio
    async def test_chat_passes_origin_to_loop(self):
        """POST /chat with X-Origin header calls loop.chat with parsed origin."""
        import json

        app, mock_loop = _make_app()
        mock_loop.chat = AsyncMock(return_value={
            "response": "hi", "tool_outputs": [], "tokens_used": 0,
        })
        mock_loop.current_task = None

        origin = {"channel": "whatsapp", "user": "+1234"}
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/chat",
                json={"message": "hello"},
                headers={"x-origin": json.dumps(origin)},
            )
        assert resp.status_code == 200
        mock_loop.chat.assert_awaited_once()
        call_kwargs = mock_loop.chat.call_args.kwargs
        assert call_kwargs.get("origin") == origin

    @pytest.mark.asyncio
    async def test_chat_no_origin_header_passes_none(self):
        """POST /chat without X-Origin passes origin=None to loop.chat."""
        app, mock_loop = _make_app()
        mock_loop.chat = AsyncMock(return_value={
            "response": "hi", "tool_outputs": [], "tokens_used": 0,
        })
        mock_loop.current_task = None

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/chat", json={"message": "hello"})
        assert resp.status_code == 200
        call_kwargs = mock_loop.chat.call_args.kwargs
        assert call_kwargs.get("origin") is None

    @pytest.mark.asyncio
    async def test_chat_invalid_origin_header_passes_none(self):
        """Malformed X-Origin silently becomes origin=None."""
        app, mock_loop = _make_app()
        mock_loop.chat = AsyncMock(return_value={
            "response": "hi", "tool_outputs": [], "tokens_used": 0,
        })
        mock_loop.current_task = None

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/chat",
                json={"message": "hello"},
                headers={"x-origin": "not-valid-json"},
            )
        assert resp.status_code == 200
        call_kwargs = mock_loop.chat.call_args.kwargs
        assert call_kwargs.get("origin") is None
