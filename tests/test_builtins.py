"""Tests for built-in agent tools: exec, file, http, browser."""

import math
import os
import shutil
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest


def _make_embedding(seed: float = 0.1) -> list[float]:
    """Create a deterministic 1536-dim embedding for testing."""
    return [math.sin(seed * (i + 1)) for i in range(1536)]


async def _mock_embed(text: str) -> list[float]:
    """Mock embed function that returns deterministic embeddings based on text hash."""
    seed = sum(ord(c) for c in text) / 100.0
    return _make_embedding(seed)


async def _mock_categorize(key: str, value: str) -> str:
    """Mock categorize function that returns a category based on key prefix."""
    if "user" in key.lower():
        return "preferences"
    if "project" in key.lower():
        return "project_info"
    return "general"

# ── exec_tool ────────────────────────────────────────────────


class TestExecTool:
    @pytest.mark.asyncio
    async def test_exec_simple_command(self):
        from src.agent.builtins.exec_tool import exec_command

        result = await exec_command(command="echo hello", workdir="/tmp")
        assert result["exit_code"] == 0
        assert "hello" in result["stdout"]

    @pytest.mark.asyncio
    async def test_exec_returns_stderr(self):
        from src.agent.builtins.exec_tool import exec_command

        result = await exec_command(command="echo err >&2", workdir="/tmp")
        assert result["exit_code"] == 0
        assert "err" in result["stderr"]

    @pytest.mark.asyncio
    async def test_exec_nonzero_exit(self):
        from src.agent.builtins.exec_tool import exec_command

        result = await exec_command(command="false", workdir="/tmp")
        assert result["exit_code"] != 0

    @pytest.mark.asyncio
    async def test_exec_timeout(self):
        from src.agent.builtins.exec_tool import exec_command

        result = await exec_command(command="sleep 10", timeout=1, workdir="/tmp")
        assert result["exit_code"] == -1
        assert "timed out" in result["stderr"]


# ── file_tool ────────────────────────────────────────────────


class TestFileTool:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        import src.agent.builtins.file_tool as ft

        self._ft = ft
        self._original_root = ft._ALLOWED_ROOT
        ft._ALLOWED_ROOT = self._tmpdir

    def teardown_method(self):
        self._ft._ALLOWED_ROOT = self._original_root
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_write_and_read_file(self):
        result = self._ft.write_file(path="test.txt", content="hello world")
        assert result["bytes_written"] == 11
        result = self._ft.read_file(path="test.txt")
        assert result["content"] == "hello world"

    def test_read_nonexistent_file(self):
        result = self._ft.read_file(path="nope.txt")
        assert "error" in result

    def test_write_creates_dirs(self):
        result = self._ft.write_file(path="sub/dir/file.txt", content="nested")
        assert result["bytes_written"] > 0

    def test_list_files(self):
        self._ft.write_file(path="a.txt", content="a")
        self._ft.write_file(path="b.txt", content="b")
        result = self._ft.list_files(path=".")
        names = [e["path"] for e in result["entries"]]
        assert "a.txt" in names
        assert "b.txt" in names

    def test_path_traversal_blocked(self):
        with pytest.raises(ValueError, match="(escapes|traversal)"):
            self._ft._safe_path("../../etc/passwd")

    def test_absolute_path_blocked(self):
        with pytest.raises(ValueError, match="Absolute"):
            self._ft._safe_path("/etc/passwd")

    def test_read_with_offset_limit(self):
        self._ft.write_file(path="lines.txt", content="line0\nline1\nline2\nline3\n")
        result = self._ft.read_file(path="lines.txt", offset=1, limit=2)
        assert "line1" in result["content"]
        assert "line2" in result["content"]
        assert "line0" not in result["content"]


class TestFileToolWorkspaceGuard:
    """write_file must block writes to workspace identity files."""

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        import src.agent.builtins.file_tool as ft

        self._ft = ft
        self._original_root = ft._ALLOWED_ROOT
        ft._ALLOWED_ROOT = self._tmpdir
        # Create workspace directory structure
        workspace = os.path.join(self._tmpdir, "workspace")
        os.makedirs(workspace, exist_ok=True)

    def teardown_method(self):
        self._ft._ALLOWED_ROOT = self._original_root
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_write_soul_md_blocked(self):
        result = self._ft.write_file(path="workspace/SOUL.md", content="hacked")
        assert "error" in result
        assert "update_workspace" in result["error"]

    def test_write_agents_md_blocked(self):
        result = self._ft.write_file(path="workspace/AGENTS.md", content="hacked")
        assert "error" in result

    def test_write_instructions_md_blocked(self):
        result = self._ft.write_file(path="workspace/INSTRUCTIONS.md", content="hacked")
        assert "error" in result

    def test_write_heartbeat_md_blocked(self):
        result = self._ft.write_file(path="workspace/HEARTBEAT.md", content="hacked")
        assert "error" in result

    def test_write_user_md_blocked(self):
        result = self._ft.write_file(path="workspace/USER.md", content="hacked")
        assert "error" in result

    def test_write_memory_md_blocked(self):
        result = self._ft.write_file(path="workspace/MEMORY.md", content="hacked")
        assert "error" in result

    def test_write_other_workspace_files_allowed(self):
        result = self._ft.write_file(path="workspace/notes.md", content="ok")
        assert "error" not in result
        assert result["bytes_written"] == 2

    def test_write_non_workspace_files_allowed(self):
        result = self._ft.write_file(path="other/data.txt", content="fine")
        assert "error" not in result

    def test_write_nested_workspace_not_blocked(self):
        """Files inside workspace subdirs with same name are NOT blocked."""
        result = self._ft.write_file(path="workspace/subdir/SOUL.md", content="ok")
        assert "error" not in result


# ── update_workspace (mesh_tool) ────────────────────────────


class TestUpdateWorkspaceTool:
    """Skill-level tests for the update_workspace tool in mesh_tool.py."""

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _make_ws(self):
        from src.agent.workspace import WorkspaceManager
        return WorkspaceManager(workspace_dir=self._tmpdir)

    @pytest.mark.asyncio
    async def test_returns_error_when_no_workspace_manager(self):
        from src.agent.builtins.mesh_tool import update_workspace
        result = await update_workspace(
            filename="HEARTBEAT.md", content="rules",
            workspace_manager=None, mesh_client=None,
        )
        assert "error" in result
        assert "workspace_manager" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_blocks_memory_md(self):
        """MEMORY.md is system-managed — cannot be written via update_workspace."""
        from src.agent.builtins.mesh_tool import update_workspace
        ws = self._make_ws()
        result = await update_workspace(
            filename="MEMORY.md", content="hacked",
            workspace_manager=ws, mesh_client=None,
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_writes_soul_md(self):
        from src.agent.builtins.mesh_tool import update_workspace
        ws = self._make_ws()
        result = await update_workspace(
            filename="SOUL.md", content="# Identity\nI am a pirate.",
            workspace_manager=ws, mesh_client=None,
        )
        assert result.get("updated") is True
        from pathlib import Path
        content = (Path(self._tmpdir) / "SOUL.md").read_text()
        assert "pirate" in content

    @pytest.mark.asyncio
    async def test_writes_instructions_md(self):
        from src.agent.builtins.mesh_tool import update_workspace
        ws = self._make_ws()
        result = await update_workspace(
            filename="INSTRUCTIONS.md", content="# Instructions\nAlways use JSON.",
            workspace_manager=ws, mesh_client=None,
        )
        assert result.get("updated") is True
        from pathlib import Path
        content = (Path(self._tmpdir) / "INSTRUCTIONS.md").read_text()
        assert "Always use JSON" in content

    @pytest.mark.asyncio
    async def test_writes_heartbeat_md(self):
        from src.agent.builtins.mesh_tool import update_workspace
        ws = self._make_ws()
        result = await update_workspace(
            filename="HEARTBEAT.md", content="# My Rules\nCheck inbox",
            workspace_manager=ws, mesh_client=None,
        )
        assert result.get("updated") is True
        from pathlib import Path
        content = (Path(self._tmpdir) / "HEARTBEAT.md").read_text()
        assert "Check inbox" in content

    @pytest.mark.asyncio
    async def test_writes_user_md(self):
        from src.agent.builtins.mesh_tool import update_workspace
        ws = self._make_ws()
        result = await update_workspace(
            filename="USER.md", content="# User\nPrefers short answers",
            workspace_manager=ws, mesh_client=None,
        )
        assert result.get("updated") is True

    @pytest.mark.asyncio
    async def test_sanitizes_content(self):
        from src.agent.builtins.mesh_tool import update_workspace
        ws = self._make_ws()
        result = await update_workspace(
            filename="HEARTBEAT.md", content="clean\u200Bvalue\u202Ehere",
            workspace_manager=ws, mesh_client=None,
        )
        assert result.get("updated") is True
        from pathlib import Path
        content = (Path(self._tmpdir) / "HEARTBEAT.md").read_text()
        assert "\u200B" not in content
        assert "\u202E" not in content

    @pytest.mark.asyncio
    async def test_notify_no_changes(self):
        from src.agent.builtins.mesh_tool import update_workspace
        ws = self._make_ws()
        from pathlib import Path
        (Path(self._tmpdir) / "HEARTBEAT.md").write_text("same content")
        mc = AsyncMock()
        mc.agent_id = "test_agent"
        await update_workspace(
            filename="HEARTBEAT.md", content="same content",
            workspace_manager=ws, mesh_client=mc,
        )
        mc.notify_user.assert_called_once()
        msg = mc.notify_user.call_args[0][0]
        assert "no changes" in msg.lower()
        assert "test_agent" in msg

    @pytest.mark.asyncio
    async def test_notify_initialized(self):
        from src.agent.builtins.mesh_tool import update_workspace
        ws = self._make_ws()
        # Default scaffold content should trigger "initialized"
        mc = AsyncMock()
        mc.agent_id = "test_agent"
        await update_workspace(
            filename="HEARTBEAT.md", content="# Custom Rules\nDo things",
            workspace_manager=ws, mesh_client=mc,
        )
        mc.notify_user.assert_called_once()
        msg = mc.notify_user.call_args[0][0]
        assert "initialized" in msg.lower()

    @pytest.mark.asyncio
    async def test_notify_updated(self):
        from src.agent.builtins.mesh_tool import update_workspace
        ws = self._make_ws()
        from pathlib import Path
        # Write non-default content first
        (Path(self._tmpdir) / "USER.md").write_text("# User\nLikes Python")
        mc = AsyncMock()
        mc.agent_id = "bot"
        await update_workspace(
            filename="USER.md", content="# User\nLikes Python and Rust",
            workspace_manager=ws, mesh_client=mc,
        )
        mc.notify_user.assert_called_once()
        msg = mc.notify_user.call_args[0][0]
        assert "updated" in msg.lower()
        assert "bot" in msg

    @pytest.mark.asyncio
    async def test_write_succeeds_if_notify_fails(self):
        from src.agent.builtins.mesh_tool import update_workspace
        ws = self._make_ws()
        mc = AsyncMock()
        mc.agent_id = "test"
        mc.notify_user = AsyncMock(side_effect=RuntimeError("notify down"))
        result = await update_workspace(
            filename="HEARTBEAT.md", content="# Rules",
            workspace_manager=ws, mesh_client=mc,
        )
        # Write should still succeed even though notification failed
        assert result.get("updated") is True


# ── http_tool ────────────────────────────────────────────────


class TestHttpTool:
    @pytest.mark.asyncio
    async def test_http_get(self):
        from src.agent.builtins.http_tool import http_request

        result = await http_request(url="https://httpbin.org/get", timeout=10)
        assert result["status_code"] == 200
        assert "body" in result

    @pytest.mark.asyncio
    async def test_http_bad_url(self):
        from src.agent.builtins.http_tool import http_request

        result = await http_request(url="http://192.0.2.1:1", timeout=2)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_cred_handle_in_headers(self):
        """$CRED{name} handles in headers are resolved via vault."""
        from unittest.mock import AsyncMock, patch

        from src.agent.builtins.http_tool import http_request

        mock_mesh = AsyncMock()
        mock_mesh.vault_resolve = AsyncMock(return_value="secret-token-123")

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.text = '{"ok": true}'
        mock_response.headers = {"content-type": "application/json"}
        mock_response.is_closed = False

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)

        with patch("src.agent.builtins.http_tool._get_client", return_value=mock_client), \
             patch("src.agent.builtins.http_tool._is_private_url", return_value=False):
            result = await http_request(
                url="https://api.github.com/user",
                headers={"Authorization": "Bearer $CRED{github_token}"},
                mesh_client=mock_mesh,
            )

        mock_mesh.vault_resolve.assert_called_once_with("github_token")
        # Verify the resolved value was sent, not the handle
        call_kwargs = mock_client.request.call_args
        assert call_kwargs.kwargs["headers"]["Authorization"] == "Bearer secret-token-123"
        assert result["status_code"] == 200

    @pytest.mark.asyncio
    async def test_cred_handle_missing_credential(self):
        """Missing credentials return an error, not a crash."""
        from unittest.mock import AsyncMock

        from src.agent.builtins.http_tool import http_request

        mock_mesh = AsyncMock()
        mock_mesh.vault_resolve = AsyncMock(return_value=None)

        result = await http_request(
            url="https://8.8.8.8/api",
            headers={"Authorization": "Bearer $CRED{nonexistent}"},
            mesh_client=mock_mesh,
        )
        assert "error" in result
        assert "Credential not found" in result["error"]

    @pytest.mark.asyncio
    async def test_cred_handle_no_mesh_client(self):
        """$CRED{} handles without mesh connectivity return an error."""
        from src.agent.builtins.http_tool import http_request

        result = await http_request(
            url="https://8.8.8.8/api",
            headers={"Authorization": "Bearer $CRED{token}"},
        )
        assert "error" in result
        assert "mesh connectivity" in result["error"]

    @pytest.mark.asyncio
    async def test_no_cred_handles_skips_resolution(self):
        """Requests without $CRED{} handles work without mesh_client."""
        from unittest.mock import AsyncMock, patch

        from src.agent.builtins.http_tool import http_request

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.text = "ok"
        mock_response.headers = {}

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)

        with patch("src.agent.builtins.http_tool._get_client", return_value=mock_client), \
             patch("src.agent.builtins.http_tool._is_private_url", return_value=False):
            result = await http_request(
                url="https://example.com",
                headers={"Accept": "application/json"},
            )
        assert result["status_code"] == 200

    @pytest.mark.asyncio
    async def test_cred_handle_in_url(self):
        """$CRED{name} handles in URL are resolved."""
        from unittest.mock import AsyncMock, patch

        from src.agent.builtins.http_tool import http_request

        mock_mesh = AsyncMock()
        mock_mesh.vault_resolve = AsyncMock(return_value="my-api-key")

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.text = "ok"
        mock_response.headers = {}

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)

        with patch("src.agent.builtins.http_tool._get_client", return_value=mock_client), \
             patch("src.agent.builtins.http_tool._is_private_url", return_value=False):
            await http_request(
                url="https://api.example.com/data?key=$CRED{api_key}",
                mesh_client=mock_mesh,
            )

        mock_mesh.vault_resolve.assert_called_once_with("api_key")
        call_kwargs = mock_client.request.call_args
        assert "my-api-key" in call_kwargs.kwargs["url"]
        assert "$CRED" not in call_kwargs.kwargs["url"]

    @pytest.mark.asyncio
    async def test_cred_handle_in_body(self):
        """$CRED{name} handles in body are resolved."""
        from unittest.mock import AsyncMock, patch

        from src.agent.builtins.http_tool import http_request

        mock_mesh = AsyncMock()
        mock_mesh.vault_resolve = AsyncMock(return_value="secret-value")

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.text = "ok"
        mock_response.headers = {}

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)

        with patch("src.agent.builtins.http_tool._get_client", return_value=mock_client):
            await http_request(
                url="https://8.8.8.8/api",
                method="POST",
                body='{"token": "$CRED{my_token}"}',
                mesh_client=mock_mesh,
            )

        mock_mesh.vault_resolve.assert_called_once_with("my_token")
        call_kwargs = mock_client.request.call_args
        assert "secret-value" in call_kwargs.kwargs["content"]
        assert "$CRED" not in call_kwargs.kwargs["content"]

    @pytest.mark.asyncio
    async def test_response_body_redacted(self):
        """Secret values echoed in response body are redacted."""
        from unittest.mock import AsyncMock, patch

        from src.agent.builtins.http_tool import http_request

        mock_mesh = AsyncMock()
        mock_mesh.vault_resolve = AsyncMock(return_value="secret-token-123")

        mock_response = AsyncMock()
        mock_response.status_code = 200
        # API echoes back the token in its response
        mock_response.text = '{"token": "secret-token-123", "user": "alice"}'
        mock_response.headers = {"content-type": "application/json"}

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)

        with patch("src.agent.builtins.http_tool._get_client", return_value=mock_client):
            result = await http_request(
                url="https://8.8.8.8/api",
                headers={"Authorization": "Bearer $CRED{token}"},
                mesh_client=mock_mesh,
            )

        assert result["status_code"] == 200
        assert "secret-token-123" not in result["body"]
        assert "[REDACTED]" in result["body"]
        assert "alice" in result["body"]  # non-secret data preserved

    @pytest.mark.asyncio
    async def test_response_headers_redacted(self):
        """Secret values in response headers are redacted."""
        from unittest.mock import AsyncMock, patch

        from src.agent.builtins.http_tool import http_request

        mock_mesh = AsyncMock()
        mock_mesh.vault_resolve = AsyncMock(return_value="secret-key-abc")

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.text = "ok"
        # Server echoes the credential in a response header
        mock_response.headers = {
            "content-type": "application/json",
            "x-api-key": "secret-key-abc",
        }

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)

        with patch("src.agent.builtins.http_tool._get_client", return_value=mock_client), \
             patch("src.agent.builtins.http_tool._is_private_url", return_value=False):
            result = await http_request(
                url="https://api.example.com?key=$CRED{api_key}",
                mesh_client=mock_mesh,
            )

        assert "secret-key-abc" not in result["headers"]["x-api-key"]
        assert "[REDACTED]" in result["headers"]["x-api-key"]

    @pytest.mark.asyncio
    async def test_error_message_redacted(self):
        """Secret values in error messages are redacted."""
        from unittest.mock import AsyncMock, patch

        from src.agent.builtins.http_tool import http_request

        mock_mesh = AsyncMock()
        mock_mesh.vault_resolve = AsyncMock(return_value="super-secret-key")

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(
            side_effect=Exception("Connection failed to https://api.example.com?key=super-secret-key")
        )

        with patch("src.agent.builtins.http_tool._get_client", return_value=mock_client), \
             patch("src.agent.builtins.http_tool._is_private_url", return_value=False):
            result = await http_request(
                url="https://api.example.com?key=$CRED{key}",
                mesh_client=mock_mesh,
            )

        assert "error" in result
        assert "super-secret-key" not in result["error"]
        assert "[REDACTED]" in result["error"]

    @pytest.mark.asyncio
    async def test_partial_resolution_no_leak(self):
        """When one credential resolves but a later one fails, the first secret must not leak."""
        from unittest.mock import AsyncMock

        from src.agent.builtins.http_tool import http_request

        mock_mesh = AsyncMock()
        # First credential resolves, second returns None
        mock_mesh.vault_resolve = AsyncMock(
            side_effect=["resolved-secret-value", None]
        )

        result = await http_request(
            url="https://8.8.8.8/api",
            headers={
                "X-First": "$CRED{good_token}",
                "X-Second": "$CRED{bad_token}",
            },
            mesh_client=mock_mesh,
        )

        assert "error" in result
        assert "Credential not found" in result["error"]
        # The resolved secret from the first credential must not appear anywhere
        assert "resolved-secret-value" not in str(result)

    @pytest.mark.asyncio
    async def test_no_redaction_without_credentials(self):
        """Responses without credential resolution are returned as-is."""
        from unittest.mock import AsyncMock, patch

        from src.agent.builtins.http_tool import http_request

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.text = '{"data": "normal response"}'
        mock_response.headers = {"content-type": "application/json"}

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)

        with patch("src.agent.builtins.http_tool._get_client", return_value=mock_client), \
             patch("src.agent.builtins.http_tool._is_private_url", return_value=False):
            result = await http_request(url="https://example.com")

        assert result["body"] == '{"data": "normal response"}'
        assert result["headers"] == {"content-type": "application/json"}


# ── SkillRegistry discovers builtins ─────────────────────────


# ── browser_tool ─────────────────────────────────────────────


class TestFlattenTree:
    def test_flatten_tree_basic(self):
        from src.agent.builtins.browser_tool import _flatten_tree

        tree = {
            "role": "WebArea",
            "name": "Test Page",
            "children": [
                {"role": "heading", "name": "Welcome", "level": 1},
                {
                    "role": "generic",
                    "name": "",
                    "children": [
                        {"role": "button", "name": "Submit"},
                        {"role": "textbox", "name": "Email"},
                        {"role": "link", "name": "Sign up"},
                    ],
                },
            ],
        }
        result = _flatten_tree(tree)
        assert len(result) == 4
        roles = [el["role"] for el in result]
        assert "heading" in roles
        assert "button" in roles
        assert "textbox" in roles
        assert "link" in roles

    def test_flatten_tree_excludes_unnamed(self):
        from src.agent.builtins.browser_tool import _flatten_tree

        tree = {
            "role": "WebArea",
            "name": "Page",
            "children": [
                {"role": "button", "name": ""},
                {"role": "button", "name": "OK"},
            ],
        }
        result = _flatten_tree(tree)
        assert len(result) == 1
        assert result[0]["name"] == "OK"

    def test_flatten_tree_includes_value(self):
        from src.agent.builtins.browser_tool import _flatten_tree

        tree = {
            "role": "WebArea",
            "name": "Page",
            "children": [
                {"role": "textbox", "name": "Search", "value": "hello"},
            ],
        }
        result = _flatten_tree(tree)
        assert len(result) == 1
        assert result[0]["value"] == "hello"

    def test_flatten_tree_checkbox_state(self):
        from src.agent.builtins.browser_tool import _flatten_tree

        tree = {
            "role": "WebArea",
            "name": "Page",
            "children": [
                {"role": "checkbox", "name": "Agree", "checked": True},
            ],
        }
        result = _flatten_tree(tree)
        assert len(result) == 1
        assert result[0]["checked"] is True


class TestBrowserSnapshot:
    @pytest.mark.asyncio
    async def test_snapshot_stores_refs(self):
        import src.agent.builtins.browser_tool as bt

        mock_page = AsyncMock()
        mock_page.url = "https://example.com"
        mock_page.title = AsyncMock(return_value="Example")
        mock_page.accessibility = MagicMock()
        mock_page.accessibility.snapshot = AsyncMock(return_value={
            "role": "WebArea",
            "name": "Example",
            "children": [
                {"role": "button", "name": "Login"},
                {"role": "textbox", "name": "Username"},
            ],
        })
        mock_locator = MagicMock()
        mock_page.get_by_role = MagicMock(return_value=mock_locator)

        with patch.object(bt, "_get_page", return_value=mock_page):
            bt._page_refs.clear()
            result = await bt.browser_snapshot()

        assert result["element_count"] == 2
        assert result["elements"][0]["ref"] == "e1"
        assert result["elements"][1]["ref"] == "e2"
        assert "e1" in bt._page_refs
        assert "e2" in bt._page_refs

    @pytest.mark.asyncio
    async def test_snapshot_duplicate_names(self):
        import src.agent.builtins.browser_tool as bt

        mock_page = AsyncMock()
        mock_page.url = "https://example.com"
        mock_page.title = AsyncMock(return_value="Example")
        mock_page.accessibility = MagicMock()
        mock_page.accessibility.snapshot = AsyncMock(return_value={
            "role": "WebArea",
            "name": "Example",
            "children": [
                {"role": "button", "name": "Delete"},
                {"role": "button", "name": "Delete"},
            ],
        })
        mock_locator = MagicMock()
        mock_nth_locator = MagicMock()
        mock_locator.nth = MagicMock(return_value=mock_nth_locator)
        mock_page.get_by_role = MagicMock(return_value=mock_locator)

        with patch.object(bt, "_get_page", return_value=mock_page):
            bt._page_refs.clear()
            result = await bt.browser_snapshot()

        assert result["element_count"] == 2
        # First "Delete" gets base locator, second gets .nth(1)
        assert bt._page_refs["e1"] is mock_locator
        assert bt._page_refs["e2"] is mock_nth_locator
        mock_locator.nth.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_snapshot_caps_at_200(self):
        import src.agent.builtins.browser_tool as bt

        children = [{"role": "button", "name": f"Btn{i}"} for i in range(250)]
        mock_page = AsyncMock()
        mock_page.url = "https://example.com"
        mock_page.title = AsyncMock(return_value="Example")
        mock_page.accessibility = MagicMock()
        mock_page.accessibility.snapshot = AsyncMock(return_value={
            "role": "WebArea",
            "name": "Example",
            "children": children,
        })
        mock_page.get_by_role = MagicMock(return_value=MagicMock())

        with patch.object(bt, "_get_page", return_value=mock_page):
            bt._page_refs.clear()
            result = await bt.browser_snapshot()

        assert result["element_count"] == 200
        assert result["truncated"] is True
        assert len(bt._page_refs) == 200


class TestParseAriaSnapshot:
    def test_parse_basic_elements(self):
        from src.agent.builtins.browser_tool import _parse_aria_snapshot

        yaml_text = '''- heading "Welcome to Example" [level=1]
- navigation "Main":
  - list:
    - listitem:
      - link "Home"
    - listitem:
      - link "About"
- main:
  - textbox "Search"
  - button "Submit"
  - checkbox "Remember me"'''

        elements = _parse_aria_snapshot(yaml_text)
        roles = [e["role"] for e in elements]
        assert "heading" in roles
        assert "link" in roles
        assert "textbox" in roles
        assert "button" in roles
        assert "checkbox" in roles

    def test_parse_extracts_names(self):
        from src.agent.builtins.browser_tool import _parse_aria_snapshot

        yaml_text = '- button "Click Me"\n- link "Go Home"'
        elements = _parse_aria_snapshot(yaml_text)
        assert elements[0]["name"] == "Click Me"
        assert elements[1]["name"] == "Go Home"

    def test_parse_extracts_attributes(self):
        from src.agent.builtins.browser_tool import _parse_aria_snapshot

        yaml_text = '- heading "Title" [level=2]'
        elements = _parse_aria_snapshot(yaml_text)
        assert elements[0]["level"] == "2"

    def test_parse_skips_non_actionable_roles(self):
        from src.agent.builtins.browser_tool import _parse_aria_snapshot

        yaml_text = '- paragraph "Some text"\n- button "OK"'
        elements = _parse_aria_snapshot(yaml_text)
        assert len(elements) == 1
        assert elements[0]["role"] == "button"

    def test_parse_skips_unnamed_elements(self):
        from src.agent.builtins.browser_tool import _parse_aria_snapshot

        yaml_text = '- button\n- button "Named"'
        elements = _parse_aria_snapshot(yaml_text)
        assert len(elements) == 1
        assert elements[0]["name"] == "Named"

    def test_parse_empty_input(self):
        from src.agent.builtins.browser_tool import _parse_aria_snapshot

        assert _parse_aria_snapshot("") == []
        assert _parse_aria_snapshot("  \n  ") == []


class TestBrowserClickRef:
    @pytest.mark.asyncio
    async def test_click_with_ref(self):
        import src.agent.builtins.browser_tool as bt

        mock_locator = AsyncMock()
        mock_page = AsyncMock()
        mock_page.url = "https://example.com"
        bt._page_refs["e1"] = mock_locator

        with patch.object(bt, "_get_page", return_value=mock_page):
            result = await bt.browser_click(ref="e1")

        assert result["clicked"] == "e1"
        mock_locator.click.assert_awaited_once_with(timeout=10000)

    @pytest.mark.asyncio
    async def test_click_unknown_ref(self):
        import src.agent.builtins.browser_tool as bt

        mock_page = AsyncMock()
        bt._page_refs.clear()

        with patch.object(bt, "_get_page", return_value=mock_page):
            result = await bt.browser_click(ref="e99")

        assert "error" in result
        assert "Unknown ref" in result["error"]

    @pytest.mark.asyncio
    async def test_click_no_params(self):
        import src.agent.builtins.browser_tool as bt

        result = await bt.browser_click()
        assert "error" in result
        assert "Provide either" in result["error"]

    @pytest.mark.asyncio
    async def test_click_selector_backward_compat(self):
        import src.agent.builtins.browser_tool as bt

        mock_page = AsyncMock()
        mock_page.url = "https://example.com"

        with patch.object(bt, "_get_page", return_value=mock_page):
            result = await bt.browser_click(selector="#submit-btn")

        assert result["clicked"] == "#submit-btn"
        mock_page.click.assert_awaited_once_with("#submit-btn", timeout=10000)


class TestBrowserTypeRef:
    @pytest.mark.asyncio
    async def test_type_with_ref(self):
        import src.agent.builtins.browser_tool as bt

        mock_locator = AsyncMock()
        mock_page = AsyncMock()
        bt._page_refs["e5"] = mock_locator

        with patch.object(bt, "_get_page", return_value=mock_page):
            result = await bt.browser_type(ref="e5", text="hello@test.com")

        assert result["typed"] == "hello@test.com"
        assert result["ref"] == "e5"
        mock_locator.fill.assert_awaited_once_with("hello@test.com", timeout=10000)

    @pytest.mark.asyncio
    async def test_type_no_text(self):
        import src.agent.builtins.browser_tool as bt

        result = await bt.browser_type(text="", ref="e1")
        assert "error" in result
        assert "'text'" in result["error"]


class TestBrowserTypeCredentialHandles:
    @pytest.mark.asyncio
    async def test_cred_handle_resolved(self):
        """$CRED{name} is resolved and return shows [credential] not the value."""
        import src.agent.builtins.browser_tool as bt

        mock_locator = AsyncMock()
        mock_page = AsyncMock()
        bt._page_refs["e5"] = mock_locator

        mock_client = AsyncMock()
        mock_client.vault_resolve.return_value = "actual-secret-value"

        with patch.object(bt, "_get_page", return_value=mock_page):
            result = await bt.browser_type(
                text="$CRED{my_api_key}", ref="e5", mesh_client=mock_client,
            )

        assert result["typed"] == "[credential]"
        assert "actual-secret-value" not in str(result)
        # Verify fill() was called with the actual value
        mock_locator.fill.assert_awaited_once_with("actual-secret-value", timeout=10000)

    @pytest.mark.asyncio
    async def test_cred_handle_not_found(self):
        """$CRED{nonexistent} returns error."""
        import src.agent.builtins.browser_tool as bt

        mock_client = AsyncMock()
        mock_client.vault_resolve.return_value = None

        result = await bt.browser_type(
            text="$CRED{nonexistent}", ref="e1", mesh_client=mock_client,
        )
        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_cred_handle_no_mesh_client(self):
        """$CRED{} without mesh_client returns error."""
        import src.agent.builtins.browser_tool as bt

        result = await bt.browser_type(
            text="$CRED{some_key}", ref="e1", mesh_client=None,
        )
        assert "error" in result
        assert "mesh connectivity" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_mixed_text_and_cred(self):
        """Text with $CRED{} embedded resolves and returns [credential]."""
        import src.agent.builtins.browser_tool as bt

        mock_locator = AsyncMock()
        mock_page = AsyncMock()
        bt._page_refs["e1"] = mock_locator

        mock_client = AsyncMock()
        mock_client.vault_resolve.return_value = "secret123"

        with patch.object(bt, "_get_page", return_value=mock_page):
            result = await bt.browser_type(
                text="Bearer $CRED{token}", ref="e1", mesh_client=mock_client,
            )

        assert result["typed"] == "[credential]"
        mock_locator.fill.assert_awaited_once_with("Bearer secret123", timeout=10000)

    @pytest.mark.asyncio
    async def test_cred_handle_tracks_resolved_value(self):
        """$CRED{} resolution adds value to _resolved_credential_values."""
        import src.agent.builtins.browser_tool as bt

        bt._resolved_credential_values.clear()
        mock_locator = AsyncMock()
        mock_page = AsyncMock()
        bt._page_refs["e1"] = mock_locator

        mock_client = AsyncMock()
        mock_client.vault_resolve.return_value = "MySecretP@ss123"

        with patch.object(bt, "_get_page", return_value=mock_page):
            await bt.browser_type(
                text="$CRED{password}", ref="e1", mesh_client=mock_client,
            )

        assert "MySecretP@ss123" in bt._resolved_credential_values
        bt._resolved_credential_values.clear()

    @pytest.mark.asyncio
    async def test_cred_handle_skips_short_values(self):
        """Resolved values shorter than 4 chars are not tracked (false-positive risk)."""
        import src.agent.builtins.browser_tool as bt

        bt._resolved_credential_values.clear()
        mock_locator = AsyncMock()
        mock_page = AsyncMock()
        bt._page_refs["e1"] = mock_locator

        mock_client = AsyncMock()
        mock_client.vault_resolve.return_value = "abc"

        with patch.object(bt, "_get_page", return_value=mock_page):
            await bt.browser_type(
                text="$CRED{pin}", ref="e1", mesh_client=mock_client,
            )

        assert "abc" not in bt._resolved_credential_values
        bt._resolved_credential_values.clear()


class TestResolvedCredentialRedaction:
    """Tests for the readback-attack mitigation.

    After browser_type resolves a $CRED{} handle, the actual value must be
    redacted from all subsequent browser output (evaluate, snapshot, navigate).
    """

    @pytest.mark.asyncio
    async def test_evaluate_redacts_resolved_credential(self):
        """browser_evaluate redacts a resolved credential value read back from DOM."""
        import src.agent.builtins.browser_tool as bt

        bt._resolved_credential_values = {"MySecretP@ss123"}
        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value="MySecretP@ss123")

        with patch.object(bt, "_get_page", return_value=mock_page):
            result = await bt.browser_evaluate(script="document.querySelector('input').value")

        assert "MySecretP@ss123" not in str(result)
        assert result["result"] == "[REDACTED]"
        bt._resolved_credential_values.clear()

    @pytest.mark.asyncio
    async def test_evaluate_redacts_credential_in_dict(self):
        """browser_evaluate redacts resolved credentials in dict values."""
        import src.agent.builtins.browser_tool as bt

        bt._resolved_credential_values = {"hunter2secret"}
        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock(
            return_value={"password": "hunter2secret", "username": "admin"},
        )

        with patch.object(bt, "_get_page", return_value=mock_page):
            result = await bt.browser_evaluate(script="getFormData()")

        assert "hunter2secret" not in str(result)
        assert result["result"]["password"] == "[REDACTED]"
        assert result["result"]["username"] == "admin"
        bt._resolved_credential_values.clear()

    @pytest.mark.asyncio
    async def test_evaluate_redacts_credential_in_list(self):
        """browser_evaluate redacts resolved credentials in list items."""
        import src.agent.builtins.browser_tool as bt

        bt._resolved_credential_values = {"secret-token-xyz"}
        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock(
            return_value=["safe text", "secret-token-xyz"],
        )

        with patch.object(bt, "_get_page", return_value=mock_page):
            result = await bt.browser_evaluate(script="getValues()")

        assert "secret-token-xyz" not in str(result)
        assert result["result"][0] == "safe text"
        assert result["result"][1] == "[REDACTED]"
        bt._resolved_credential_values.clear()

    @pytest.mark.asyncio
    async def test_navigate_redacts_resolved_credential_in_content(self):
        """browser_navigate redacts resolved credentials from page content."""
        import src.agent.builtins.browser_tool as bt

        bt._resolved_credential_values = {"MySecretP@ss123"}
        mock_page = AsyncMock()
        mock_page.url = "https://example.com"
        mock_page.title = AsyncMock(return_value="Example")
        mock_page.wait_for_timeout = AsyncMock()
        mock_page.inner_text = AsyncMock(
            return_value="Your API key is MySecretP@ss123. Keep it safe.",
        )
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_page.goto = AsyncMock(return_value=mock_response)

        with patch.object(bt, "_get_page", return_value=mock_page):
            result = await bt.browser_navigate(url="https://example.com")

        assert "MySecretP@ss123" not in result["content"]
        assert "[REDACTED]" in result["content"]
        bt._resolved_credential_values.clear()

    @pytest.mark.asyncio
    async def test_evaluate_redacts_nested_dict(self):
        """browser_evaluate redacts credentials in nested structures."""
        import src.agent.builtins.browser_tool as bt

        bt._resolved_credential_values = {"deep-secret-value"}
        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock(
            return_value={"data": {"nested": {"token": "deep-secret-value"}}},
        )

        with patch.object(bt, "_get_page", return_value=mock_page):
            result = await bt.browser_evaluate(script="getNestedData()")

        assert "deep-secret-value" not in str(result)
        assert result["result"]["data"]["nested"]["token"] == "[REDACTED]"
        bt._resolved_credential_values.clear()

    @pytest.mark.asyncio
    async def test_evaluate_error_redacted(self):
        """browser_evaluate redacts credentials from error messages."""
        import src.agent.builtins.browser_tool as bt

        bt._resolved_credential_values = {"MySecretP@ss123"}
        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock(
            side_effect=Exception("Failed with value MySecretP@ss123 in context"),
        )

        with patch.object(bt, "_get_page", return_value=mock_page):
            result = await bt.browser_evaluate(script="badScript()")

        assert "MySecretP@ss123" not in str(result)
        assert "[REDACTED]" in result["error"]
        bt._resolved_credential_values.clear()

    @pytest.mark.asyncio
    async def test_type_error_redacted(self):
        """browser_type redacts credentials from fill() error messages."""
        import src.agent.builtins.browser_tool as bt

        bt._resolved_credential_values = {"actual-secret"}
        mock_locator = AsyncMock()
        mock_locator.fill = AsyncMock(
            side_effect=Exception("fill failed: actual-secret not accepted"),
        )
        mock_page = AsyncMock()
        bt._page_refs["e1"] = mock_locator

        mock_client = AsyncMock()
        mock_client.vault_resolve.return_value = "actual-secret"

        with patch.object(bt, "_get_page", return_value=mock_page):
            result = await bt.browser_type(
                text="$CRED{password}", ref="e1", mesh_client=mock_client,
            )

        assert "actual-secret" not in str(result)
        assert "[REDACTED]" in result["error"]
        bt._resolved_credential_values.clear()

    @pytest.mark.asyncio
    async def test_auto_recovery_preserves_tracked_credentials(self):
        """Dead CDP auto-recovery does NOT clear _resolved_credential_values."""
        import src.agent.builtins.browser_tool as bt

        bt._resolved_credential_values = {"must-survive-recovery"}

        mock_page = AsyncMock()
        mock_page.url = "https://example.com"
        mock_page.title = AsyncMock(return_value="Example")
        mock_page.wait_for_timeout = AsyncMock()
        mock_page.inner_text = AsyncMock(return_value="safe content")
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_page.goto = AsyncMock(
            side_effect=[
                Exception("Page.navigate limit reached"),  # triggers auto-recovery
                mock_response,  # retry succeeds
            ],
        )

        with (
            patch.object(bt, "_get_page", return_value=mock_page),
            patch.object(bt, "_browser_cleanup_soft", new_callable=AsyncMock),
        ):
            await bt.browser_navigate(url="https://example.com")

        # Tracked credentials must survive the auto-recovery
        assert "must-survive-recovery" in bt._resolved_credential_values
        bt._resolved_credential_values.clear()

    def test_deep_redact_nested_structures(self):
        """_deep_redact handles arbitrarily nested dicts and lists."""
        import src.agent.builtins.browser_tool as bt
        from src.agent.builtins.browser_tool import _deep_redact

        bt._resolved_credential_values = {"secret-val"}

        # Nested dict
        assert _deep_redact({"a": {"b": "secret-val"}}) == {"a": {"b": "[REDACTED]"}}
        # List of dicts
        assert _deep_redact([{"k": "secret-val"}]) == [{"k": "[REDACTED]"}]
        # Mixed nesting
        assert _deep_redact({"items": ["ok", "secret-val"]}) == {"items": ["ok", "[REDACTED]"]}
        # Non-string leaves pass through
        assert _deep_redact({"count": 42, "flag": True}) == {"count": 42, "flag": True}
        # None and empty
        assert _deep_redact(None) is None
        assert _deep_redact("") == ""
        bt._resolved_credential_values.clear()

    def test_redact_resolved_credentials_basic(self):
        """_redact_resolved_credentials replaces tracked values."""
        import src.agent.builtins.browser_tool as bt
        from src.agent.builtins.browser_tool import _redact_resolved_credentials

        bt._resolved_credential_values = {"secret123", "p@ssw0rd"}

        assert _redact_resolved_credentials("the key is secret123") == "the key is [REDACTED]"
        assert _redact_resolved_credentials("pw: p@ssw0rd") == "pw: [REDACTED]"
        assert _redact_resolved_credentials("safe text here") == "safe text here"
        assert _redact_resolved_credentials("") == ""
        bt._resolved_credential_values.clear()

    def test_redact_resolved_credentials_noop_when_empty(self):
        """_redact_resolved_credentials is a no-op when no values are tracked."""
        import src.agent.builtins.browser_tool as bt
        from src.agent.builtins.browser_tool import _redact_resolved_credentials

        bt._resolved_credential_values = set()

        text = "sk-abcdefghijklmnopqrstuvwxyz"
        assert _redact_resolved_credentials(text) == text


class TestBrowserSnapshotRedaction:
    def test_redacts_api_key_patterns(self):
        from src.agent.builtins.browser_tool import _redact_credentials

        assert _redact_credentials("sk-abcdefghijklmnopqrstuvwxyz") == "[REDACTED]"
        assert _redact_credentials("ghp_abcdefghijklmnopqrstuvwxyz0123456789") == "[REDACTED]"
        assert _redact_credentials("xoxb-123-456-abcdefghijklmnop") == "[REDACTED]"
        assert _redact_credentials("AKIAIOSFODNN7EXAMPLE") == "[REDACTED]"

    def test_preserves_normal_text(self):
        from src.agent.builtins.browser_tool import _redact_credentials

        assert _redact_credentials("Submit") == "Submit"
        assert _redact_credentials("Enter your email") == "Enter your email"
        assert _redact_credentials("Price: $42.00") == "Price: $42.00"
        assert _redact_credentials("") == ""

    @pytest.mark.asyncio
    async def test_snapshot_redacts_element_values(self):
        """Snapshot redacts secret patterns in element name/value fields."""
        import src.agent.builtins.browser_tool as bt

        mock_page = AsyncMock()
        mock_page.url = "https://dashboard.example.com"
        mock_page.title = AsyncMock(return_value="Dashboard")
        mock_page.accessibility = MagicMock()
        mock_page.accessibility.snapshot = AsyncMock(return_value={
            "role": "WebArea",
            "name": "Dashboard",
            "children": [
                {"role": "textbox", "name": "API Key", "value": "sk-abcdefghijklmnopqrstuvwxyz"},
                {"role": "button", "name": "Copy"},
            ],
        })
        mock_page.get_by_role = MagicMock(return_value=MagicMock())

        with patch.object(bt, "_get_page", return_value=mock_page):
            bt._page_refs.clear()
            result = await bt.browser_snapshot()

        # The API key value should be redacted
        api_key_el = next(e for e in result["elements"] if e["role"] == "textbox")
        assert "[REDACTED]" in api_key_el["value"]
        assert "sk-" not in api_key_el["value"]

        # Normal button name should be preserved
        button_el = next(e for e in result["elements"] if e["role"] == "button")
        assert button_el["name"] == "Copy"


class TestBrowserNavigateRedaction:
    @pytest.mark.asyncio
    async def test_navigate_redacts_api_keys_in_content(self):
        """browser_navigate redacts secret patterns in page content."""
        import src.agent.builtins.browser_tool as bt

        mock_page = AsyncMock()
        mock_page.url = "https://example.com/settings"
        mock_page.title = AsyncMock(return_value="Settings")
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_page.goto = AsyncMock(return_value=mock_response)
        # Page content contains an API key
        mock_page.inner_text = AsyncMock(
            return_value="Your API key: sk-abcdefghijklmnopqrstuvwxyz"
        )

        with patch.object(bt, "_get_page", return_value=mock_page):
            result = await bt.browser_navigate(url="https://example.com/settings")

        assert "sk-" not in result["content"]
        assert "[REDACTED]" in result["content"]

    @pytest.mark.asyncio
    async def test_navigate_preserves_normal_content(self):
        """Normal page content passes through unredacted."""
        import src.agent.builtins.browser_tool as bt

        mock_page = AsyncMock()
        mock_page.url = "https://example.com"
        mock_page.title = AsyncMock(return_value="Example")
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_page.goto = AsyncMock(return_value=mock_response)
        mock_page.inner_text = AsyncMock(return_value="Welcome to Example.com")

        with patch.object(bt, "_get_page", return_value=mock_page):
            result = await bt.browser_navigate(url="https://example.com")

        assert result["content"] == "Welcome to Example.com"


class TestBrowserEvaluateRedaction:
    @pytest.mark.asyncio
    async def test_evaluate_redacts_string_result(self):
        """browser_evaluate redacts secret patterns in string results."""
        import src.agent.builtins.browser_tool as bt

        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock(
            return_value="sk-abcdefghijklmnopqrstuvwxyz"
        )

        with patch.object(bt, "_get_page", return_value=mock_page):
            result = await bt.browser_evaluate(script="document.querySelector('.key').textContent")

        assert "sk-" not in str(result)
        assert result["result"] == "[REDACTED]"

    @pytest.mark.asyncio
    async def test_evaluate_redacts_dict_result(self):
        """browser_evaluate redacts secret patterns in dict values."""
        import src.agent.builtins.browser_tool as bt

        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock(
            return_value={"key": "ghp_abcdefghijklmnopqrstuvwxyz0123456789", "label": "API Key"}
        )

        with patch.object(bt, "_get_page", return_value=mock_page):
            result = await bt.browser_evaluate(script="getConfig()")

        assert "ghp_" not in str(result)
        assert result["result"]["key"] == "[REDACTED]"
        assert result["result"]["label"] == "API Key"

    @pytest.mark.asyncio
    async def test_evaluate_redacts_list_result(self):
        """browser_evaluate redacts secret patterns in list items."""
        import src.agent.builtins.browser_tool as bt

        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock(
            return_value=["sk-abcdefghijklmnopqrstuvwxyz", "normal text", 42]
        )

        with patch.object(bt, "_get_page", return_value=mock_page):
            result = await bt.browser_evaluate(script="getKeys()")

        assert "sk-" not in str(result)
        assert result["result"][0] == "[REDACTED]"
        assert result["result"][1] == "normal text"
        assert result["result"][2] == 42

    @pytest.mark.asyncio
    async def test_evaluate_preserves_normal_result(self):
        """Normal evaluate results pass through unchanged."""
        import src.agent.builtins.browser_tool as bt

        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value="Hello World")

        with patch.object(bt, "_get_page", return_value=mock_page):
            result = await bt.browser_evaluate(script="document.title")

        assert result["result"] == "Hello World"


class TestBrowserNavigateClearsRefs:
    @pytest.mark.asyncio
    async def test_navigate_clears_refs(self):
        import src.agent.builtins.browser_tool as bt

        bt._page_refs["e1"] = MagicMock()
        bt._page_refs["e2"] = MagicMock()

        mock_page = AsyncMock()
        mock_page.url = "https://new.example.com"
        mock_page.title = AsyncMock(return_value="New Page")
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_page.goto = AsyncMock(return_value=mock_response)
        mock_page.inner_text = AsyncMock(return_value="page content")

        with patch.object(bt, "_get_page", return_value=mock_page):
            result = await bt.browser_navigate(url="https://new.example.com")

        assert len(bt._page_refs) == 0
        assert result["url"] == "https://new.example.com"


# ── browser_reset & auto-recovery ─────────────────────────────


class TestBrowserReset:
    @pytest.mark.asyncio
    async def test_browser_reset_clears_state(self):
        """browser_reset tears down session and clears refs."""
        import src.agent.builtins.browser_tool as bt

        bt._page_refs["e1"] = MagicMock()
        bt._credential_filled_refs.add("e1")
        bt._resolved_credential_values.add("some-secret")
        bt._page = MagicMock()
        bt._page.is_closed = MagicMock(return_value=False)
        bt._page.close = AsyncMock()
        bt._context = MagicMock()
        bt._context.close = AsyncMock()
        bt._browser = MagicMock()
        bt._browser.close = AsyncMock()

        result = await bt.browser_reset()

        assert result["status"] == "reset"
        assert bt._page is None
        assert bt._browser is None
        assert bt._context is None
        assert len(bt._page_refs) == 0
        assert len(bt._credential_filled_refs) == 0
        assert len(bt._resolved_credential_values) == 0

    @pytest.mark.asyncio
    async def test_browser_reset_safe_when_no_session(self):
        """browser_reset works even with no active session."""
        import src.agent.builtins.browser_tool as bt

        bt._page = None
        bt._browser = None
        bt._context = None
        bt._pw = None

        result = await bt.browser_reset()

        assert result["status"] == "reset"


class TestBrowserNavigateAutoRecovery:
    @pytest.mark.asyncio
    async def test_navigate_auto_recovers_from_navigation_limit(self):
        """browser_navigate resets and retries on 'Page.navigate limit reached'."""
        import src.agent.builtins.browser_tool as bt

        dead_page = AsyncMock()
        dead_page.goto = AsyncMock(
            side_effect=Exception("Protocol error (Page.navigate): Page.navigate limit reached")
        )

        fresh_page = AsyncMock()
        fresh_page.url = "https://example.com"
        fresh_page.title = AsyncMock(return_value="Example")
        fresh_response = AsyncMock()
        fresh_response.status = 200
        fresh_page.goto = AsyncMock(return_value=fresh_response)
        fresh_page.inner_text = AsyncMock(return_value="Hello")

        call_count = 0

        async def mock_get_page(*, mesh_client=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return dead_page
            return fresh_page

        with patch.object(bt, "_get_page", side_effect=mock_get_page), \
             patch.object(bt, "_browser_cleanup_soft", new_callable=AsyncMock) as mock_cleanup:
            result = await bt.browser_navigate(url="https://example.com")

        mock_cleanup.assert_awaited_once()
        assert result["status"] == 200
        assert result["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_navigate_auto_recovers_from_tunnel_failure(self):
        """browser_navigate resets and retries on ERR_TUNNEL_CONNECTION_FAILED."""
        import src.agent.builtins.browser_tool as bt

        dead_page = AsyncMock()
        dead_page.goto = AsyncMock(
            side_effect=Exception("net::ERR_TUNNEL_CONNECTION_FAILED")
        )

        fresh_page = AsyncMock()
        fresh_page.url = "https://example.com"
        fresh_page.title = AsyncMock(return_value="Example")
        fresh_response = AsyncMock()
        fresh_response.status = 200
        fresh_page.goto = AsyncMock(return_value=fresh_response)
        fresh_page.inner_text = AsyncMock(return_value="Works now")

        call_count = 0

        async def mock_get_page(*, mesh_client=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return dead_page
            return fresh_page

        with patch.object(bt, "_get_page", side_effect=mock_get_page), \
             patch.object(bt, "_browser_cleanup_soft", new_callable=AsyncMock):
            result = await bt.browser_navigate(url="https://example.com")

        assert result["content"] == "Works now"

    @pytest.mark.asyncio
    async def test_navigate_returns_error_for_non_cdp_failures(self):
        """Normal errors (e.g. timeout) are returned without retry."""
        import src.agent.builtins.browser_tool as bt

        mock_page = AsyncMock()
        mock_page.goto = AsyncMock(
            side_effect=Exception("Timeout 30000ms exceeded")
        )

        with patch.object(bt, "_get_page", return_value=mock_page), \
             patch.object(bt, "_browser_cleanup_soft", new_callable=AsyncMock) as mock_cleanup:
            result = await bt.browser_navigate(url="https://slow-site.com")

        mock_cleanup.assert_not_awaited()
        assert "error" in result
        assert "Timeout" in result["error"]

    @pytest.mark.asyncio
    async def test_navigate_returns_error_if_retry_also_fails(self):
        """If the fresh session also fails, return the error."""
        import src.agent.builtins.browser_tool as bt

        mock_page = AsyncMock()
        mock_page.goto = AsyncMock(
            side_effect=Exception("Protocol error (Page.navigate): Page.navigate limit reached")
        )

        with patch.object(bt, "_get_page", return_value=mock_page), \
             patch.object(bt, "_browser_cleanup_soft", new_callable=AsyncMock):
            result = await bt.browser_navigate(url="https://example.com")

        assert "error" in result


class TestIsDeadSessionError:
    def test_detects_navigation_limit(self):
        from src.agent.builtins.browser_tool import _is_dead_session_error

        assert _is_dead_session_error("Protocol error (Page.navigate): Page.navigate limit reached")

    def test_detects_tunnel_failure(self):
        from src.agent.builtins.browser_tool import _is_dead_session_error

        assert _is_dead_session_error("net::ERR_TUNNEL_CONNECTION_FAILED")

    def test_detects_target_closed(self):
        from src.agent.builtins.browser_tool import _is_dead_session_error

        assert _is_dead_session_error("Target closed")

    def test_detects_connection_refused(self):
        from src.agent.builtins.browser_tool import _is_dead_session_error

        assert _is_dead_session_error("Connection refused")

    def test_detects_connection_closed(self):
        from src.agent.builtins.browser_tool import _is_dead_session_error

        assert _is_dead_session_error("Connection closed while reading from the driver")

    def test_ignores_timeout(self):
        from src.agent.builtins.browser_tool import _is_dead_session_error

        assert not _is_dead_session_error("Timeout 30000ms exceeded")

    def test_ignores_normal_errors(self):
        from src.agent.builtins.browser_tool import _is_dead_session_error

        assert not _is_dead_session_error("Element not found")

    def test_ignores_generic_protocol_error(self):
        from src.agent.builtins.browser_tool import _is_dead_session_error

        assert not _is_dead_session_error("Protocol error (Page.navigate): Cannot navigate to invalid URL")

class TestCleanupStaleProfile:
    def test_removes_singleton_lock_files(self, tmp_path):
        """_cleanup_stale_profile removes SingletonLock/Socket/Cookie."""
        import src.agent.builtins.browser_tool as bt

        profile_dir = tmp_path / "browser_profile"
        profile_dir.mkdir()
        (profile_dir / "SingletonLock").touch()
        (profile_dir / "SingletonSocket").touch()
        (profile_dir / "SingletonCookie").touch()
        (profile_dir / "Cookies").touch()  # should NOT be removed

        with patch.object(bt, "Path", return_value=profile_dir):
            # We need to patch Path("/data/browser_profile") to return our tmp dir
            pass

        # Directly call with patched profile path
        original_path = bt.Path
        with patch("src.agent.builtins.browser_tool.Path") as mock_path:
            mock_path.return_value = profile_dir
            mock_path.__truediv__ = original_path.__truediv__
            bt._cleanup_stale_profile()

        assert not (profile_dir / "SingletonLock").exists()
        assert not (profile_dir / "SingletonSocket").exists()
        assert not (profile_dir / "SingletonCookie").exists()
        assert (profile_dir / "Cookies").exists()  # untouched

    def test_noop_when_profile_dir_missing(self):
        """_cleanup_stale_profile does nothing when /data/browser_profile doesn't exist."""
        import src.agent.builtins.browser_tool as bt

        # Default /data/browser_profile won't exist in test env — should not raise
        bt._cleanup_stale_profile()

    def test_handles_pkill_failure(self, tmp_path):
        """_cleanup_stale_profile handles pkill not found gracefully."""
        import src.agent.builtins.browser_tool as bt

        profile_dir = tmp_path / "browser_profile"
        profile_dir.mkdir()
        (profile_dir / "SingletonLock").touch()

        with patch("subprocess.run", side_effect=FileNotFoundError("pkill not found")), \
             patch("src.agent.builtins.browser_tool.Path") as mock_path:
            mock_path.return_value = profile_dir
            bt._cleanup_stale_profile()

        # Lock file should still be removed even if pkill fails
        assert not (profile_dir / "SingletonLock").exists()


class TestBrowserCleanupSoftCallsStaleCleanup:
    @pytest.mark.asyncio
    async def test_soft_cleanup_calls_stale_profile_cleanup(self):
        """_browser_cleanup_soft calls _cleanup_stale_profile after closing."""
        import src.agent.builtins.browser_tool as bt

        bt._page = MagicMock()
        bt._page.is_closed = MagicMock(return_value=False)
        bt._page.close = AsyncMock()
        bt._context = MagicMock()
        bt._context.close = AsyncMock()
        bt._browser = None
        bt._pw = MagicMock()
        bt._pw.stop = AsyncMock()

        with patch.object(bt, "_cleanup_stale_profile") as mock_cleanup:
            await bt._browser_cleanup_soft()

        mock_cleanup.assert_called_once()
        assert bt._page is None
        assert bt._context is None


# ── SkillRegistry discovers builtins ─────────────────────────


class TestBuiltinDiscovery:
    def test_builtins_auto_discovered(self):
        from src.agent.skills import SkillRegistry

        registry = SkillRegistry(skills_dir="/nonexistent/path")
        assert "exec" in registry.skills
        assert "read_file" in registry.skills
        assert "write_file" in registry.skills
        assert "list_files" in registry.skills
        assert "http_request" in registry.skills
        assert "browser_navigate" in registry.skills
        assert "browser_snapshot" in registry.skills
        assert "browser_reset" in registry.skills
        assert "memory_search" in registry.skills
        assert "category" in registry.skills["memory_search"]["parameters"]


# ── memory_tool ────────────────────────────────────────────────


class TestMemorySearchWithCategory:
    @pytest.mark.asyncio
    async def test_memory_search_with_category_basic(self, tmp_path):
        """memory_search with category searches only the structured fact DB."""
        from src.agent.builtins.memory_tool import memory_search
        from src.agent.memory import MemoryStore

        store = MemoryStore(db_path=str(tmp_path / "recall.db"))
        await store.store_fact("user_lang", "Python")
        await store.store_fact("user_editor", "Vim")

        result = await memory_search("user preferences", max_results=5, memory_store=store)
        assert result["count"] >= 1
        assert len(result["results"]) >= 1
        store.close()

    @pytest.mark.asyncio
    async def test_memory_search_with_category_filter(self, tmp_path):
        """category param filters results (without embeddings, uses text field)."""
        from src.agent.builtins.memory_tool import memory_search
        from src.agent.memory import MemoryStore

        store = MemoryStore(db_path=str(tmp_path / "recall_cat.db"))
        await store.store_fact("user_lang", "Python", category="preference")
        await store.store_fact("project_name", "OpenLegion", category="fact")

        result = await memory_search("Python", category="preference", max_results=5, memory_store=store)
        for r in result["results"]:
            assert r["category"] == "preference"
        store.close()

    @pytest.mark.asyncio
    async def test_memory_search_category_filter_with_auto_categorize(self, tmp_path):
        """category filter works with auto-categorized facts (embed_fn + categorize_fn)."""
        from src.agent.builtins.memory_tool import memory_search
        from src.agent.memory import MemoryStore

        store = MemoryStore(
            db_path=str(tmp_path / "recall_autocat.db"),
            embed_fn=_mock_embed,
            categorize_fn=_mock_categorize,
        )
        await store.store_fact("user_lang", "Python is my preferred language")
        await store.store_fact("user_editor", "Vim is my editor of choice")
        await store.store_fact("project_name", "OpenLegion is the project")

        # Filter by auto-assigned category
        result = await memory_search("Python", category="preferences", max_results=5, memory_store=store)
        assert result["count"] >= 1, "Should find preference-categorized facts"
        for r in result["results"]:
            assert r["category"] == "preferences"
        store.close()

    @pytest.mark.asyncio
    async def test_memory_search_category_no_store(self):
        """memory_search with category but no memory_store returns error."""
        from src.agent.builtins.memory_tool import memory_search

        result = await memory_search("anything", category="test", memory_store=None)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_memory_search_category_no_store_with_workspace(self):
        """memory_search with category + workspace but no memory_store returns error, not workspace results."""
        from src.agent.builtins.memory_tool import memory_search

        mock_ws = MagicMock()
        mock_ws.search.return_value = [{"file": "MEMORY.md", "snippet": "something"}]

        result = await memory_search("anything", category="test", workspace_manager=mock_ws, memory_store=None)
        assert "error" in result
        # Should NOT fall through to workspace search
        mock_ws.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_memory_search_category_no_matches(self, tmp_path):
        """memory_search with category returns empty results when no facts match, not an error."""
        from src.agent.builtins.memory_tool import memory_search
        from src.agent.memory import MemoryStore

        store = MemoryStore(db_path=str(tmp_path / "no_match.db"))
        await store.store_fact("user_lang", "Python", category="preference")

        result = await memory_search("Python", category="nonexistent_category", max_results=5, memory_store=store)
        assert "error" not in result
        assert result["count"] == 0
        store.close()

    @pytest.mark.asyncio
    async def test_memory_search_combined_sources(self, tmp_path):
        """memory_search returns both workspace and DB results."""
        from src.agent.builtins.memory_tool import memory_search
        from src.agent.memory import MemoryStore

        store = MemoryStore(db_path=str(tmp_path / "search_combo.db"))
        await store.store_fact("api_key_type", "OAuth2 tokens")

        mock_ws = MagicMock()
        mock_ws.search.return_value = [{"file": "MEMORY.md", "snippet": "User prefers dark mode"}]

        result = await memory_search("user preferences", max_results=5, workspace_manager=mock_ws, memory_store=store)
        sources = {r.get("source") for r in result["results"]}
        assert "workspace" in sources
        # DB results may or may not match depending on keyword search
        assert result["count"] >= 1
        store.close()


class TestParseFact:
    def test_colon_separator(self):
        from src.agent.builtins.memory_tool import _parse_fact

        key, value = _parse_fact("favorite color: blue")
        assert key == "favorite color"
        assert value == "blue"

    def test_dash_separator(self):
        from src.agent.builtins.memory_tool import _parse_fact

        key, value = _parse_fact("User preference - dark mode enabled")
        assert key == "User preference"
        assert value == "dark mode enabled"

    def test_no_separator_short(self):
        from src.agent.builtins.memory_tool import _parse_fact

        key, value = _parse_fact("likes pizza")
        assert key == "likes pizza"
        assert value == "likes pizza"

    def test_no_separator_long(self):
        from src.agent.builtins.memory_tool import _parse_fact

        long_text = (
            "The user mentioned they prefer to work late at night"
            " and want all notifications disabled after 10pm"
        )
        key, value = _parse_fact(long_text)
        assert len(key) <= 60
        assert value == long_text

    def test_strips_whitespace(self):
        from src.agent.builtins.memory_tool import _parse_fact

        key, value = _parse_fact("  name : Alice  ")
        assert key == "name"
        assert value == "Alice"


class TestMemorySave:
    @pytest.mark.asyncio
    async def test_save_workspace_only(self):
        """memory_save with only workspace_manager stores to daily log."""
        from src.agent.builtins.memory_tool import memory_save

        mock_ws = MagicMock()
        result = await memory_save("user likes Python", workspace_manager=mock_ws)
        assert result["saved"] is True
        assert result["saved_workspace"] is True
        assert result["saved_db"] is False
        mock_ws.append_daily_log.assert_called_once_with("user likes Python")

    @pytest.mark.asyncio
    async def test_save_db_only(self, tmp_path):
        """memory_save with only memory_store stores to structured DB."""
        from src.agent.builtins.memory_tool import memory_save
        from src.agent.memory import MemoryStore

        store = MemoryStore(db_path=str(tmp_path / "save_db.db"))
        result = await memory_save("favorite color: blue", memory_store=store)
        assert result["saved"] is True
        assert result["saved_workspace"] is False
        assert result["saved_db"] is True

        # Verify fact is in DB
        facts = await store.search("favorite color", top_k=5)
        assert len(facts) >= 1
        assert any("blue" in f.value for f in facts)
        store.close()

    @pytest.mark.asyncio
    async def test_save_both_backends(self, tmp_path):
        """memory_save stores to both workspace and DB when both available."""
        from src.agent.builtins.memory_tool import memory_save
        from src.agent.memory import MemoryStore

        mock_ws = MagicMock()
        store = MemoryStore(db_path=str(tmp_path / "save_both.db"))

        result = await memory_save("project name: OpenLegion", workspace_manager=mock_ws, memory_store=store)
        assert result["saved"] is True
        assert result["saved_workspace"] is True
        assert result["saved_db"] is True
        mock_ws.append_daily_log.assert_called_once()
        store.close()

    @pytest.mark.asyncio
    async def test_save_no_backends(self):
        """memory_save with no backends returns error."""
        from src.agent.builtins.memory_tool import memory_save

        result = await memory_save("anything")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_save_then_search(self, tmp_path):
        """Facts saved via memory_save are findable via memory_search."""
        from src.agent.builtins.memory_tool import memory_save, memory_search
        from src.agent.memory import MemoryStore

        store = MemoryStore(db_path=str(tmp_path / "save_recall.db"))

        await memory_save("user timezone: America/New_York", memory_store=store)
        result = await memory_search("timezone", max_results=5, memory_store=store)
        assert result["count"] >= 1
        assert any("New_York" in r["value"] for r in result["results"])
        store.close()


class TestSetCronHeartbeat:
    @pytest.mark.asyncio
    async def test_set_cron_heartbeat_creates_new(self):
        """set_cron(heartbeat=True) creates a heartbeat job when none exists."""
        from src.agent.builtins.mesh_tool import set_cron

        mock_client = AsyncMock()
        mock_client.list_cron.return_value = []
        mock_client.create_cron.return_value = {"id": "hb-1"}

        result = await set_cron(schedule="every 30m", heartbeat=True, mesh_client=mock_client)
        assert result["created"] is True
        assert result["type"] == "heartbeat"
        mock_client.create_cron.assert_called_once_with(
            schedule="every 30m", message="heartbeat", heartbeat=True,
        )

    @pytest.mark.asyncio
    async def test_set_cron_heartbeat_updates_existing(self):
        """set_cron(heartbeat=True) updates existing heartbeat job."""
        from src.agent.builtins.mesh_tool import set_cron

        mock_client = AsyncMock()
        mock_client.list_cron.return_value = [{"id": "hb-1", "heartbeat": True}]
        mock_client.update_cron.return_value = {"id": "hb-1"}

        result = await set_cron(schedule="every 1h", heartbeat=True, mesh_client=mock_client)
        assert result["updated"] is True
        assert result["type"] == "heartbeat"
        mock_client.update_cron.assert_called_once_with("hb-1", schedule="every 1h")

    @pytest.mark.asyncio
    async def test_set_cron_heartbeat_custom_message(self):
        """set_cron(heartbeat=True, message='check') uses provided message."""
        from src.agent.builtins.mesh_tool import set_cron

        mock_client = AsyncMock()
        mock_client.list_cron.return_value = []
        mock_client.create_cron.return_value = {"id": "hb-2"}

        await set_cron(schedule="every 5m", message="check", heartbeat=True, mesh_client=mock_client)
        mock_client.create_cron.assert_called_once_with(
            schedule="every 5m", message="check", heartbeat=True,
        )

    @pytest.mark.asyncio
    async def test_set_cron_regular_requires_message(self):
        """set_cron without heartbeat requires a non-empty message."""
        from src.agent.builtins.mesh_tool import set_cron

        mock_client = AsyncMock()
        result = await set_cron(schedule="every 1h", message="", mesh_client=mock_client)
        assert "error" in result
        assert "message" in result["error"].lower()


class TestSearchWithFallback:
    """Tests for the _search_with_fallback helper in memory_tool."""

    @pytest.mark.asyncio
    async def test_hierarchical_success(self):
        """Returns hierarchical results when it succeeds."""
        from src.agent.builtins.memory_tool import _search_with_fallback

        store = AsyncMock()
        store.search_hierarchical.return_value = [MagicMock(key="k", value="v")]
        result = await _search_with_fallback(store, "query", 5)
        assert len(result) == 1
        store.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_fallback_to_flat(self):
        """Falls back to flat search when hierarchical raises."""
        from src.agent.builtins.memory_tool import _search_with_fallback

        store = AsyncMock()
        store.search_hierarchical.side_effect = RuntimeError("no embeddings")
        store.search.return_value = [MagicMock(key="k", value="v")]
        result = await _search_with_fallback(store, "query", 5)
        assert len(result) == 1
        store.search.assert_called_once_with("query", top_k=5)

    @pytest.mark.asyncio
    async def test_both_fail_returns_none(self):
        """Returns None when both searches fail."""
        from src.agent.builtins.memory_tool import _search_with_fallback

        store = AsyncMock()
        store.search_hierarchical.side_effect = RuntimeError("fail")
        store.search.side_effect = RuntimeError("fail")
        result = await _search_with_fallback(store, "query", 5)
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_list_returned_not_none(self):
        """Empty list from hierarchical is returned as-is, not as None."""
        from src.agent.builtins.memory_tool import _search_with_fallback

        store = AsyncMock()
        store.search_hierarchical.return_value = []
        result = await _search_with_fallback(store, "query", 5)
        assert result == []
        assert result is not None
        store.search.assert_not_called()


# ── labeled screenshots ──────────────────────────────────────

_has_pillow = True
try:
    import PIL  # noqa: F401
except ImportError:
    _has_pillow = False


@pytest.mark.skipif(not _has_pillow, reason="Pillow not installed")
class TestLabeledScreenshot:
    @pytest.mark.asyncio
    async def test_labeled_false_unchanged(self, tmp_path):
        """No labels key when labeled=False."""
        import src.agent.builtins.browser_tool as bt

        mock_page = AsyncMock()
        mock_page.is_closed.return_value = False
        img_path = tmp_path / "shot.png"

        # Create a minimal valid PNG
        from PIL import Image
        Image.new("RGB", (100, 100), "white").save(str(img_path))

        async def mock_screenshot(path, full_page):
            import shutil
            shutil.copy2(str(img_path), path)

        mock_page.screenshot = mock_screenshot

        with patch.object(bt, "_get_page", return_value=mock_page):
            with patch("src.agent.builtins.browser_tool.Path") as MockPath:
                mock_save_path = MagicMock()
                mock_save_path.parent.mkdir = MagicMock()
                mock_save_path.__str__ = lambda self: str(img_path)
                mock_save_path.stat.return_value.st_size = 100
                MockPath.__truediv__ = lambda self, other: mock_save_path
                MockPath.return_value.__truediv__ = lambda self, other: mock_save_path
                # Simpler approach: just mock Path("/data") / filename
                result = await bt.browser_screenshot(filename="shot.png", labeled=False)

        assert "labels" not in result

    @pytest.mark.asyncio
    async def test_labeled_auto_snapshots(self):
        """Auto-calls snapshot when _page_refs empty and labeled=True."""
        import src.agent.builtins.browser_tool as bt

        mock_page = AsyncMock()
        mock_page.is_closed.return_value = False
        mock_page.url = "https://example.com"
        mock_page.title = AsyncMock(return_value="Example")

        bt._page_refs.clear()
        snapshot_called = []

        original_snapshot = bt._browser_snapshot_inner

        async def mock_snapshot(*, mesh_client=None):
            snapshot_called.append(True)
            bt._page_refs["e1"] = AsyncMock()
            bt._page_refs["e1"].bounding_box = AsyncMock(return_value=None)
            return {"element_count": 1}

        bt._browser_snapshot_inner = mock_snapshot

        try:
            with patch.object(bt, "_get_page", return_value=mock_page):
                # Mock the file operations
                with patch("src.agent.builtins.browser_tool.Path") as MockPath:
                    mock_save = MagicMock()
                    mock_save.parent.mkdir = MagicMock()
                    mock_save.__str__ = MagicMock(return_value="/data/shot.png")
                    mock_save.stat.return_value.st_size = 500
                    MockPath.__truediv__ = lambda self, other: mock_save
                    MockPath.return_value.__truediv__ = lambda self, other: mock_save

                    with patch.object(bt, "_draw_labels", return_value={}):
                        result = await bt.browser_screenshot(
                            filename="shot.png", labeled=True
                        )

            assert len(snapshot_called) == 1
            assert result.get("labeled") is True
        finally:
            bt._browser_snapshot_inner = original_snapshot
            bt._page_refs.clear()

    @pytest.mark.asyncio
    async def test_labeled_draws_labels(self, tmp_path):
        """Mock locators with bounding boxes, verify label_count > 0."""
        import src.agent.builtins.browser_tool as bt

        mock_loc1 = AsyncMock()
        mock_loc1.bounding_box = AsyncMock(return_value={"x": 10, "y": 20, "width": 80, "height": 30})
        mock_loc2 = AsyncMock()
        mock_loc2.bounding_box = AsyncMock(return_value={"x": 50, "y": 100, "width": 120, "height": 40})

        bt._page_refs.clear()
        bt._page_refs["e1"] = mock_loc1
        bt._page_refs["e2"] = mock_loc2

        # Create a test image
        from PIL import Image
        img_path = str(tmp_path / "test_labels.png")
        Image.new("RGB", (300, 200), "white").save(img_path)

        labels = await bt._draw_labels(img_path)

        assert len(labels) == 2
        assert "1" in labels
        assert "2" in labels
        assert "80x30" in labels["1"]

        bt._page_refs.clear()

    @pytest.mark.asyncio
    async def test_labeled_skips_offscreen(self, tmp_path):
        """bbox=None elements excluded from label map."""
        import src.agent.builtins.browser_tool as bt

        mock_visible = AsyncMock()
        mock_visible.bounding_box = AsyncMock(return_value={"x": 10, "y": 20, "width": 80, "height": 30})
        mock_offscreen = AsyncMock()
        mock_offscreen.bounding_box = AsyncMock(return_value=None)

        bt._page_refs.clear()
        bt._page_refs["e1"] = mock_visible
        bt._page_refs["e2"] = mock_offscreen

        from PIL import Image
        img_path = str(tmp_path / "test_offscreen.png")
        Image.new("RGB", (300, 200), "white").save(img_path)

        labels = await bt._draw_labels(img_path)

        assert len(labels) == 1
        assert "1" in labels
        assert "2" not in labels

        bt._page_refs.clear()

    @pytest.mark.asyncio
    async def test_labeled_pillow_import_error(self, tmp_path):
        """Graceful fallback returns empty labels when Pillow not available."""
        import src.agent.builtins.browser_tool as bt

        bt._page_refs["e1"] = AsyncMock()

        with patch.dict("sys.modules", {"PIL": None, "PIL.Image": None, "PIL.ImageDraw": None, "PIL.ImageFont": None}):
            # Need to re-import to trigger the ImportError
            labels = await bt._draw_labels(str(tmp_path / "fake.png"))

        assert labels == {}
        bt._page_refs.clear()


# ── notify_user ──────────────────────────────────────────────


class TestNotifyUser:
    @pytest.mark.asyncio
    async def test_notify_user_success(self):
        from src.agent.builtins.mesh_tool import notify_user

        mock_mesh = AsyncMock()
        mock_mesh.notify_user = AsyncMock()

        result = await notify_user(message="Task done", mesh_client=mock_mesh)
        assert result == {"sent": True}
        mock_mesh.notify_user.assert_awaited_once_with("Task done")

    @pytest.mark.asyncio
    async def test_notify_user_no_mesh_client(self):
        from src.agent.builtins.mesh_tool import notify_user

        result = await notify_user(message="hello", mesh_client=None)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_notify_user_failure(self):
        from src.agent.builtins.mesh_tool import notify_user

        mock_mesh = AsyncMock()
        mock_mesh.notify_user = AsyncMock(side_effect=RuntimeError("connection refused"))

        result = await notify_user(message="hello", mesh_client=mock_mesh)
        assert "error" in result
        assert "connection refused" in result["error"]


# ── LLMClient embedding model ───────────────────────────────


class TestLLMClientEmbeddingModel:
    @pytest.mark.asyncio
    async def test_embed_uses_configured_model(self):
        from src.agent.llm import LLMClient

        llm = LLMClient(
            mesh_url="http://localhost:8420",
            agent_id="test",
            embedding_model="custom/embed-v2",
        )

        # Mock the HTTP client to capture the request
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "data": {"embedding": [0.1, 0.2, 0.3]},
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        llm._client = mock_client

        result = await llm.embed("hello world")
        assert result == [0.1, 0.2, 0.3]

        # Verify the request used the configured model
        call_kwargs = mock_client.post.call_args
        request_body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert request_body["params"]["model"] == "custom/embed-v2"

    @pytest.mark.asyncio
    async def test_embed_default_model(self):
        """Default embedding_model is empty (runtime.py sets it via env var)."""
        from src.agent.llm import LLMClient

        llm = LLMClient(mesh_url="http://localhost:8420", agent_id="test")
        assert llm.embedding_model == ""


# ── artifact path traversal ──────────────────────────────────


class TestArtifactPathTraversal:
    @pytest.mark.asyncio
    async def test_path_traversal_blocked(self):
        """save_artifact rejects names that escape the artifacts dir."""
        from src.agent.builtins.mesh_tool import save_artifact

        ws = MagicMock()
        ws.root = tempfile.mkdtemp()
        try:
            result = await save_artifact(
                name="../../escape.txt", content="pwned",
                workspace_manager=ws, mesh_client=None,
            )
            assert "error" in result
            assert "Invalid artifact name" in result["error"]
        finally:
            shutil.rmtree(ws.root, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_normal_artifact_allowed(self):
        """save_artifact accepts valid names."""
        from src.agent.builtins.mesh_tool import save_artifact

        ws = MagicMock()
        ws.root = tempfile.mkdtemp()
        try:
            result = await save_artifact(
                name="report.txt", content="hello",
                workspace_manager=ws, mesh_client=None,
            )
            assert "error" not in result
            assert result["saved"] is True
            assert result["name"] == "report.txt"
        finally:
            shutil.rmtree(ws.root, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_absolute_path_blocked(self):
        """save_artifact rejects absolute paths."""
        from src.agent.builtins.mesh_tool import save_artifact

        ws = MagicMock()
        ws.root = tempfile.mkdtemp()
        try:
            result = await save_artifact(
                name="/etc/passwd", content="pwned",
                workspace_manager=ws, mesh_client=None,
            )
            assert "error" in result
        finally:
            shutil.rmtree(ws.root, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_sibling_directory_blocked(self):
        """save_artifact rejects names that resolve to a sibling of artifacts dir."""
        from src.agent.builtins.mesh_tool import save_artifact

        ws = MagicMock()
        ws.root = tempfile.mkdtemp()
        try:
            # Create a sibling directory that starts with "artifacts"
            sibling = os.path.join(ws.root, "artifacts_evil")
            os.makedirs(sibling)
            result = await save_artifact(
                name="../artifacts_evil/steal.txt", content="pwned",
                workspace_manager=ws, mesh_client=None,
            )
            assert "error" in result
            assert "Invalid artifact name" in result["error"]
        finally:
            shutil.rmtree(ws.root, ignore_errors=True)


# ── Standalone agent blackboard guards ───────────────────────────


class TestStandaloneBlackboardGuards:
    """Standalone agents (no project) get clear errors from blackboard tools."""

    def _standalone_client(self):
        from src.agent.mesh_client import MeshClient
        mc = MagicMock(spec=MeshClient)
        mc.is_standalone = True
        return mc

    def _project_client(self):
        from src.agent.mesh_client import MeshClient
        mc = MagicMock(spec=MeshClient)
        mc.is_standalone = False
        return mc

    @pytest.mark.asyncio
    async def test_read_shared_state_blocked_for_standalone(self):
        from src.agent.builtins.mesh_tool import read_shared_state
        result = await read_shared_state(key="foo", mesh_client=self._standalone_client())
        assert "error" in result
        assert "not assigned to any project" in result["error"]

    @pytest.mark.asyncio
    async def test_write_shared_state_blocked_for_standalone(self):
        from src.agent.builtins.mesh_tool import write_shared_state
        result = await write_shared_state(
            key="foo", value="bar", mesh_client=self._standalone_client(),
        )
        assert "error" in result
        assert "not assigned to any project" in result["error"]

    @pytest.mark.asyncio
    async def test_list_shared_state_blocked_for_standalone(self):
        from src.agent.builtins.mesh_tool import list_shared_state
        result = await list_shared_state(prefix="", mesh_client=self._standalone_client())
        assert "error" in result
        assert "not assigned to any project" in result["error"]

    @pytest.mark.asyncio
    async def test_save_artifact_skips_blackboard_for_standalone(self, tmp_path):
        """Standalone agents can save artifacts locally but skip blackboard."""
        from src.agent.builtins.mesh_tool import save_artifact
        ws = MagicMock()
        ws.root = str(tmp_path)
        mc = self._standalone_client()
        mc.write_blackboard = AsyncMock()
        result = await save_artifact(
            name="report.txt", content="hello",
            workspace_manager=ws, mesh_client=mc,
        )
        assert result.get("saved") is True
        assert (tmp_path / "artifacts" / "report.txt").read_text() == "hello"
        mc.write_blackboard.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_read_shared_state_allowed_for_project_agent(self):
        """Project agents are NOT blocked by the standalone guard."""
        from src.agent.builtins.mesh_tool import read_shared_state
        mc = self._project_client()
        mc.read_blackboard = AsyncMock(return_value={"key": "foo", "value": "bar"})
        result = await read_shared_state(key="foo", mesh_client=mc)
        assert "not assigned" not in result.get("error", "")
        mc.read_blackboard.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_write_shared_state_allowed_for_project_agent(self):
        """Project agents can write to blackboard."""
        from src.agent.builtins.mesh_tool import write_shared_state
        mc = self._project_client()
        mc.write_blackboard = AsyncMock(return_value=True)
        result = await write_shared_state(key="k", value="v", mesh_client=mc)
        assert "not assigned" not in result.get("error", "")
        mc.write_blackboard.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_list_shared_state_allowed_for_project_agent(self):
        """Project agents can list blackboard entries."""
        from src.agent.builtins.mesh_tool import list_shared_state
        mc = self._project_client()
        mc.list_blackboard = AsyncMock(return_value=[])
        result = await list_shared_state(prefix="", mesh_client=mc)
        assert "not assigned" not in result.get("error", "")
        mc.list_blackboard.assert_awaited_once()


# ── Introspect Tool ─────────────────────────────────────────────


class TestIntrospectTool:
    @pytest.mark.asyncio
    async def test_introspect_success(self):
        from src.agent.builtins.introspect_tool import introspect_tool

        mock_data = {
            "permissions": {"blackboard_read": ["context/*"]},
            "budget": {"daily_used": 1.5, "daily_limit": 10.0},
            "fleet": [{"id": "alice", "role": "researcher"}],
        }
        mock_mesh = AsyncMock()
        mock_mesh.introspect = AsyncMock(return_value=mock_data)

        result = await introspect_tool(section="all", mesh_client=mock_mesh)
        assert result == mock_data
        mock_mesh.introspect.assert_awaited_once_with("all")

    @pytest.mark.asyncio
    async def test_introspect_specific_section(self):
        from src.agent.builtins.introspect_tool import introspect_tool

        mock_mesh = AsyncMock()
        mock_mesh.introspect = AsyncMock(return_value={"budget": {"daily_used": 0.5}})

        result = await introspect_tool(section="budget", mesh_client=mock_mesh)
        assert "budget" in result
        mock_mesh.introspect.assert_awaited_once_with("budget")

    @pytest.mark.asyncio
    async def test_introspect_no_mesh_client(self):
        from src.agent.builtins.introspect_tool import introspect_tool

        result = await introspect_tool(section="all", mesh_client=None)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_introspect_mesh_failure(self):
        from src.agent.builtins.introspect_tool import introspect_tool

        mock_mesh = AsyncMock()
        mock_mesh.introspect = AsyncMock(side_effect=RuntimeError("connection refused"))

        result = await introspect_tool(section="all", mesh_client=mock_mesh)
        assert "error" in result
        assert "connection refused" in result["error"]


# ── browser lifecycle ─────────────────────────────────────────


class TestBrowserLifecycle:
    @pytest.mark.asyncio
    async def test_get_page_calls_launch_persistent(self):
        """_get_page() connects to Chrome via _launch_persistent()."""
        import src.agent.builtins.browser_tool as bt
        bt._browser = bt._context = bt._page = None
        with patch.object(bt, "_launch_persistent", new_callable=AsyncMock) as mock_persistent:
            mock_page = AsyncMock()
            mock_page.is_closed.return_value = False
            mock_persistent.return_value = (None, MagicMock(), mock_page)
            await bt._get_page()
            mock_persistent.assert_called_once()

    @pytest.mark.asyncio
    async def test_browser_reset_uses_soft_cleanup(self):
        """browser_reset uses _browser_cleanup_soft (preserves VNC + profile)."""
        import src.agent.builtins.browser_tool as bt

        bt._page_refs["e1"] = MagicMock()
        bt._page = MagicMock()
        bt._page.is_closed = MagicMock(return_value=False)
        bt._page.close = AsyncMock()
        bt._context = MagicMock()
        bt._context.close = AsyncMock()
        bt._browser = None

        with patch.object(bt, "_browser_cleanup_soft", new_callable=AsyncMock) as mock_soft:
            result = await bt.browser_reset()

        mock_soft.assert_awaited_once()
        assert result["status"] == "reset"
        assert "preserved" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_start_browser_launches_stack(self):
        """start_browser starts KasmVNC Xvnc and browser."""
        import src.agent.builtins.browser_tool as bt

        bt._page = None
        bt._context = None
        bt._browser = None
        bt._pw = None
        bt._vnc_proc = None

        popen_calls = []

        def mock_popen(cmd, **kwargs):
            popen_calls.append(cmd)
            m = MagicMock()
            m.poll.return_value = None  # process still running
            return m

        with patch.object(bt, "_find_chromium_binary", return_value="/usr/bin/chromium"), \
             patch.object(bt, "_inject_vnc_input_fix", new_callable=AsyncMock), \
             patch("subprocess.Popen", side_effect=mock_popen), \
             patch("asyncio.sleep", new_callable=AsyncMock), \
             patch("pathlib.Path.mkdir"):
            await bt.start_browser()

        # Three Popen calls: KasmVNC Xvnc + openbox + Chrome subprocess
        assert len(popen_calls) == 3
        xvnc_cmd = popen_calls[0]
        assert "Xvnc" in xvnc_cmd[0]
        assert popen_calls[1] == ["openbox"]
        chrome_cmd = popen_calls[2]
        assert "/usr/bin/chromium" in chrome_cmd[0]
        assert "--remote-debugging-port=9222" in chrome_cmd
        # Default web port 6080 passed via -websocketPort
        assert "-websocketPort" in xvnc_cmd
        assert "6080" in xvnc_cmd
        # Auth must be fully disabled (both VNC and BasicAuth layers)
        assert "-SecurityTypes" in xvnc_cmd
        assert "-disableBasicAuth" in xvnc_cmd

    @pytest.mark.asyncio
    async def test_start_browser_uses_vnc_port_env(self):
        """VNC_PORT env var overrides the default KasmVNC web port."""
        import src.agent.builtins.browser_tool as bt

        bt._page = None
        bt._context = None
        bt._browser = None
        bt._pw = None
        bt._vnc_proc = None

        popen_calls = []

        def mock_popen(cmd, **kwargs):
            popen_calls.append(cmd)
            m = MagicMock()
            m.poll.return_value = None
            return m

        with patch.dict(os.environ, {"VNC_PORT": "9999"}):
            with patch.object(bt, "_find_chromium_binary", return_value="/usr/bin/chromium"), \
                 patch.object(bt, "_inject_vnc_input_fix", new_callable=AsyncMock), \
                 patch("subprocess.Popen", side_effect=mock_popen), \
                 patch("asyncio.sleep", new_callable=AsyncMock), \
                 patch("pathlib.Path.mkdir"):
                await bt.start_browser()

        # KasmVNC should use port 9999
        assert "9999" in popen_calls[0]

    @pytest.mark.asyncio
    async def test_start_browser_raises_on_vnc_crash(self):
        """start_browser raises if KasmVNC Xvnc exits immediately."""
        import src.agent.builtins.browser_tool as bt

        bt._page = None
        bt._context = None
        bt._browser = None
        bt._pw = None
        bt._vnc_proc = None

        def mock_popen(cmd, **kwargs):
            m = MagicMock()
            m.poll.return_value = 1  # exited immediately
            m.returncode = 1
            return m

        with patch.object(bt, "_get_page", new_callable=AsyncMock), \
             patch("subprocess.Popen", side_effect=mock_popen), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(RuntimeError, match="KasmVNC"):
                await bt.start_browser()


class TestListAgentsProjectScope:
    """Tests that list_agents skill passes project scope via MeshClient."""

    @pytest.mark.asyncio
    async def test_list_agents_project_scope(self):
        """MeshClient.list_agents passes project param when in a project."""
        from src.agent.mesh_client import MeshClient
        client = MeshClient("http://mesh:8420", "bot1", project_name="teamA")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"bot1": {"url": "...", "role": "dev"}}
        mock_response.raise_for_status = MagicMock()
        http_client = AsyncMock()
        http_client.get = AsyncMock(return_value=mock_response)
        http_client.is_closed = False
        client._client = http_client

        await client.list_agents()

        http_client.get.assert_called_once()
        call_kwargs = http_client.get.call_args
        assert call_kwargs.kwargs.get("params", {}).get("project") == "teamA"
        assert "agent_id" not in call_kwargs.kwargs.get("params", {})

    @pytest.mark.asyncio
    async def test_list_agents_standalone_scope(self):
        """MeshClient.list_agents passes agent_id param when standalone."""
        from src.agent.mesh_client import MeshClient
        client = MeshClient("http://mesh:8420", "solo", project_name=None)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"solo": {"url": "...", "role": ""}}
        mock_response.raise_for_status = MagicMock()
        http_client = AsyncMock()
        http_client.get = AsyncMock(return_value=mock_response)
        http_client.is_closed = False
        client._client = http_client

        await client.list_agents()

        http_client.get.assert_called_once()
        call_kwargs = http_client.get.call_args
        assert call_kwargs.kwargs.get("params", {}).get("agent_id") == "solo"
        assert "project" not in call_kwargs.kwargs.get("params", {})


# ── CAPTCHA detection / solving / injection ──────────────────


class TestCaptchaDetection:
    @pytest.mark.asyncio
    async def test_detect_recaptcha_v2(self):
        from src.agent.builtins.captcha import detect_captcha

        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value={
            "type": "recaptcha_v2",
            "sitekey": "6LeIxAcTAAAAAJcZVRqyHh71UMIEGNQ_MXjiZKhI",
        })

        result = await detect_captcha(mock_page)
        assert result is not None
        assert result["type"] == "recaptcha_v2"
        assert result["sitekey"] == "6LeIxAcTAAAAAJcZVRqyHh71UMIEGNQ_MXjiZKhI"

    @pytest.mark.asyncio
    async def test_detect_hcaptcha(self):
        from src.agent.builtins.captcha import detect_captcha

        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value={
            "type": "hcaptcha",
            "sitekey": "10000000-ffff-ffff-ffff-000000000001",
        })

        result = await detect_captcha(mock_page)
        assert result is not None
        assert result["type"] == "hcaptcha"

    @pytest.mark.asyncio
    async def test_detect_turnstile(self):
        from src.agent.builtins.captcha import detect_captcha

        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value={
            "type": "turnstile",
            "sitekey": "0x4AAAAAAAC",
        })

        result = await detect_captcha(mock_page)
        assert result is not None
        assert result["type"] == "turnstile"

    @pytest.mark.asyncio
    async def test_detect_none(self):
        from src.agent.builtins.captcha import detect_captcha

        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value=None)

        result = await detect_captcha(mock_page)
        assert result is None

    @pytest.mark.asyncio
    async def test_detect_handles_js_error(self):
        from src.agent.builtins.captcha import detect_captcha

        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock(side_effect=Exception("Page crashed"))

        result = await detect_captcha(mock_page)
        assert result is None


class TestCaptchaSolving:
    @pytest.mark.asyncio
    async def test_solve_with_capsolver(self):
        from src.agent.builtins.captcha import solve_captcha

        mock_mesh = AsyncMock()
        mock_mesh.vault_resolve = AsyncMock(side_effect=lambda name: (
            "test-capsolver-key" if name == "capsolver_key" else None
        ))

        create_resp = MagicMock()
        create_resp.status_code = 200
        create_resp.raise_for_status = MagicMock()
        create_resp.json.return_value = {"taskId": "task-123"}

        result_resp = MagicMock()
        result_resp.status_code = 200
        result_resp.raise_for_status = MagicMock()
        result_resp.json.return_value = {
            "status": "ready",
            "solution": {"gRecaptchaResponse": "solved-token-abc"},
        }

        with patch("src.agent.builtins.captcha.asyncio.sleep", new_callable=AsyncMock):
            with patch("src.agent.builtins.captcha.httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(side_effect=[create_resp, result_resp])
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client_cls.return_value = mock_client

                token = await solve_captcha(
                    {"type": "recaptcha_v2", "sitekey": "test-key"},
                    "https://example.com",
                    mock_mesh,
                )

        assert token == "solved-token-abc"

    @pytest.mark.asyncio
    async def test_solve_with_2captcha_fallback(self):
        from src.agent.builtins.captcha import solve_captcha

        mock_mesh = AsyncMock()
        # capsolver_key returns None, 2captcha_key returns a key
        mock_mesh.vault_resolve = AsyncMock(side_effect=lambda name: (
            "test-2captcha-key" if name == "2captcha_key" else None
        ))

        create_resp = MagicMock()
        create_resp.status_code = 200
        create_resp.raise_for_status = MagicMock()
        create_resp.json.return_value = {"taskId": "task-456"}

        result_resp = MagicMock()
        result_resp.status_code = 200
        result_resp.raise_for_status = MagicMock()
        result_resp.json.return_value = {
            "status": "ready",
            "solution": {"token": "2cap-token-xyz"},
        }

        with patch("src.agent.builtins.captcha.asyncio.sleep", new_callable=AsyncMock):
            with patch("src.agent.builtins.captcha.httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(side_effect=[create_resp, result_resp])
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client_cls.return_value = mock_client

                token = await solve_captcha(
                    {"type": "hcaptcha", "sitekey": "test-key"},
                    "https://example.com",
                    mock_mesh,
                )

        assert token == "2cap-token-xyz"
        # Verify it called 2captcha API URL
        call_args = mock_client.post.call_args_list[0]
        assert "2captcha.com" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_solve_no_api_key(self):
        from src.agent.builtins.captcha import solve_captcha

        mock_mesh = AsyncMock()
        mock_mesh.vault_resolve = AsyncMock(return_value=None)

        token = await solve_captcha(
            {"type": "recaptcha_v2", "sitekey": "test-key"},
            "https://example.com",
            mock_mesh,
        )
        assert token is None

    @pytest.mark.asyncio
    async def test_solve_polls_until_ready(self):
        from src.agent.builtins.captcha import solve_captcha

        mock_mesh = AsyncMock()
        mock_mesh.vault_resolve = AsyncMock(side_effect=lambda name: (
            "key" if name == "capsolver_key" else None
        ))

        create_resp = MagicMock()
        create_resp.status_code = 200
        create_resp.raise_for_status = MagicMock()
        create_resp.json.return_value = {"taskId": "t1"}

        processing_resp = MagicMock()
        processing_resp.status_code = 200
        processing_resp.raise_for_status = MagicMock()
        processing_resp.json.return_value = {"status": "processing"}

        ready_resp = MagicMock()
        ready_resp.status_code = 200
        ready_resp.raise_for_status = MagicMock()
        ready_resp.json.return_value = {
            "status": "ready",
            "solution": {"gRecaptchaResponse": "tok"},
        }

        with patch("src.agent.builtins.captcha.asyncio.sleep", new_callable=AsyncMock):
            with patch("src.agent.builtins.captcha.httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(
                    side_effect=[create_resp, processing_resp, processing_resp, ready_resp],
                )
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client_cls.return_value = mock_client

                token = await solve_captcha(
                    {"type": "recaptcha_v2", "sitekey": "k"},
                    "https://example.com",
                    mock_mesh,
                )

        assert token == "tok"
        # 1 createTask + 3 getTaskResult polls
        assert mock_client.post.call_count == 4


class TestCaptchaInjection:
    @pytest.mark.asyncio
    async def test_inject_recaptcha_token(self):
        from src.agent.builtins.captcha import inject_captcha_token

        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value=True)

        result = await inject_captcha_token(
            mock_page, {"type": "recaptcha_v2"}, "test-token",
        )
        assert result is True
        mock_page.evaluate.assert_called_once()
        # Token should be passed as argument
        call_args = mock_page.evaluate.call_args
        assert call_args[0][1] == "test-token"

    @pytest.mark.asyncio
    async def test_inject_hcaptcha_token(self):
        from src.agent.builtins.captcha import inject_captcha_token

        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value=True)

        result = await inject_captcha_token(
            mock_page, {"type": "hcaptcha"}, "hcap-token",
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_inject_turnstile_token(self):
        from src.agent.builtins.captcha import inject_captcha_token

        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value=True)

        result = await inject_captcha_token(
            mock_page, {"type": "turnstile"}, "cf-token",
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_inject_unknown_type_returns_false(self):
        from src.agent.builtins.captcha import inject_captcha_token

        mock_page = AsyncMock()

        result = await inject_captcha_token(
            mock_page, {"type": "unknown_captcha"}, "token",
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_inject_handles_js_error(self):
        from src.agent.builtins.captcha import inject_captcha_token

        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock(side_effect=Exception("JS error"))

        result = await inject_captcha_token(
            mock_page, {"type": "recaptcha_v2"}, "token",
        )
        assert result is False


class TestBrowserNavigateCaptcha:
    @pytest.mark.asyncio
    async def test_navigate_auto_solves_captcha(self):
        """browser_navigate detects and solves CAPTCHAs automatically."""
        import src.agent.builtins.browser_tool as bt

        mock_page = AsyncMock()
        mock_page.url = "https://protected.example.com"
        mock_page.title = AsyncMock(return_value="Protected")
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_page.goto = AsyncMock(return_value=mock_response)
        # First inner_text: page with captcha, second: after solve
        mock_page.inner_text = AsyncMock(
            side_effect=["Please verify you are human", "Welcome! Content unlocked."],
        )

        mock_mesh = AsyncMock()

        with patch.object(bt, "_get_page", return_value=mock_page), \
             patch(
                 "src.agent.builtins.captcha.detect_captcha",
                 new_callable=AsyncMock,
                 return_value={"type": "recaptcha_v2", "sitekey": "test"},
             ), \
             patch(
                 "src.agent.builtins.captcha.solve_captcha",
                 new_callable=AsyncMock,
                 return_value="solved-token",
             ), \
             patch(
                 "src.agent.builtins.captcha.inject_captcha_token",
                 new_callable=AsyncMock,
                 return_value=True,
             ):
            result = await bt.browser_navigate(
                url="https://protected.example.com",
                mesh_client=mock_mesh,
            )

        assert result["captcha_solved"] == "recaptcha_v2"
        assert "Welcome" in result["content"]

    @pytest.mark.asyncio
    async def test_navigate_reports_captcha_solve_failed(self):
        """browser_navigate reports captcha_detected when solving fails."""
        import src.agent.builtins.browser_tool as bt

        mock_page = AsyncMock()
        mock_page.url = "https://protected.example.com"
        mock_page.title = AsyncMock(return_value="Protected")
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_page.goto = AsyncMock(return_value=mock_response)
        mock_page.inner_text = AsyncMock(return_value="Captcha page")

        mock_mesh = AsyncMock()

        with patch.object(bt, "_get_page", return_value=mock_page), \
             patch(
                 "src.agent.builtins.captcha.detect_captcha",
                 new_callable=AsyncMock,
                 return_value={"type": "hcaptcha", "sitekey": "test"},
             ), \
             patch(
                 "src.agent.builtins.captcha.solve_captcha",
                 new_callable=AsyncMock,
                 return_value=None,
             ):
            result = await bt.browser_navigate(
                url="https://protected.example.com",
                mesh_client=mock_mesh,
            )

        assert result["captcha_detected"] == "hcaptcha"
        assert "vault" in result["captcha_note"].lower()
        assert "could not be solved" in result["captcha_note"].lower()

    @pytest.mark.asyncio
    async def test_navigate_no_captcha_unchanged(self):
        """browser_navigate works normally when no CAPTCHA detected."""
        import src.agent.builtins.browser_tool as bt

        mock_page = AsyncMock()
        mock_page.url = "https://example.com"
        mock_page.title = AsyncMock(return_value="Example")
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_page.goto = AsyncMock(return_value=mock_response)
        mock_page.inner_text = AsyncMock(return_value="Normal page")

        with patch.object(bt, "_get_page", return_value=mock_page), \
             patch(
                 "src.agent.builtins.captcha.detect_captcha",
                 new_callable=AsyncMock,
                 return_value=None,
             ):
            result = await bt.browser_navigate(url="https://example.com")

        assert "captcha_solved" not in result
        assert "captcha_detected" not in result
        assert result["content"] == "Normal page"


class TestBrowserSolveCaptchaSkill:
    @pytest.mark.asyncio
    async def test_solve_captcha_skill_success(self):
        """browser_solve_captcha skill detects and solves a CAPTCHA."""
        import src.agent.builtins.browser_tool as bt

        mock_page = AsyncMock()
        mock_page.url = "https://protected.example.com"
        mock_page.is_closed.return_value = False

        mock_mesh = AsyncMock()

        with patch.object(bt, "_get_page", return_value=mock_page), \
             patch(
                 "src.agent.builtins.captcha.detect_captcha",
                 new_callable=AsyncMock,
                 return_value={"type": "turnstile", "sitekey": "key"},
             ), \
             patch(
                 "src.agent.builtins.captcha.solve_captcha",
                 new_callable=AsyncMock,
                 return_value="cf-token-123",
             ), \
             patch(
                 "src.agent.builtins.captcha.inject_captcha_token",
                 new_callable=AsyncMock,
                 return_value=True,
             ):
            result = await bt.browser_solve_captcha(mesh_client=mock_mesh)

        assert result["status"] == "solved"
        assert result["captcha_type"] == "turnstile"
        assert result["injected"] is True

    @pytest.mark.asyncio
    async def test_solve_captcha_skill_no_captcha(self):
        """browser_solve_captcha returns no_captcha when page is clean."""
        import src.agent.builtins.browser_tool as bt

        mock_page = AsyncMock()
        mock_page.is_closed.return_value = False

        with patch.object(bt, "_get_page", return_value=mock_page), \
             patch(
                 "src.agent.builtins.captcha.detect_captcha",
                 new_callable=AsyncMock,
                 return_value=None,
             ):
            result = await bt.browser_solve_captcha(mesh_client=AsyncMock())

        assert result["status"] == "no_captcha"

    @pytest.mark.asyncio
    async def test_solve_captcha_skill_solve_failed(self):
        """browser_solve_captcha reports solve_failed when solving fails."""
        import src.agent.builtins.browser_tool as bt

        mock_page = AsyncMock()
        mock_page.url = "https://example.com"
        mock_page.is_closed.return_value = False

        mock_mesh = AsyncMock()

        with patch.object(bt, "_get_page", return_value=mock_page), \
             patch(
                 "src.agent.builtins.captcha.detect_captcha",
                 new_callable=AsyncMock,
                 return_value={"type": "recaptcha_v2", "sitekey": "k"},
             ), \
             patch(
                 "src.agent.builtins.captcha.solve_captcha",
                 new_callable=AsyncMock,
                 return_value=None,
             ):
            result = await bt.browser_solve_captcha(mesh_client=mock_mesh)

        assert result["status"] == "solve_failed"
        assert result["captcha_type"] == "recaptcha_v2"
        assert "vault" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_solve_captcha_skill_no_mesh_client(self):
        """browser_solve_captcha returns no_client when mesh_client is None."""
        import src.agent.builtins.browser_tool as bt

        mock_page = AsyncMock()
        mock_page.is_closed.return_value = False

        with patch.object(bt, "_get_page", return_value=mock_page), \
             patch(
                 "src.agent.builtins.captcha.detect_captcha",
                 new_callable=AsyncMock,
                 return_value={"type": "turnstile", "sitekey": "k"},
             ):
            result = await bt.browser_solve_captcha(mesh_client=None)

        assert result["status"] == "no_client"
        assert result["captcha_type"] == "turnstile"

    @pytest.mark.asyncio
    async def test_solve_captcha_skill_injection_fails(self):
        """browser_solve_captcha reports injected=False when injection fails."""
        import src.agent.builtins.browser_tool as bt

        mock_page = AsyncMock()
        mock_page.url = "https://example.com"
        mock_page.is_closed.return_value = False

        mock_mesh = AsyncMock()

        with patch.object(bt, "_get_page", return_value=mock_page), \
             patch(
                 "src.agent.builtins.captcha.detect_captcha",
                 new_callable=AsyncMock,
                 return_value={"type": "hcaptcha", "sitekey": "k"},
             ), \
             patch(
                 "src.agent.builtins.captcha.solve_captcha",
                 new_callable=AsyncMock,
                 return_value="token-xyz",
             ), \
             patch(
                 "src.agent.builtins.captcha.inject_captcha_token",
                 new_callable=AsyncMock,
                 return_value=False,
             ):
            result = await bt.browser_solve_captcha(mesh_client=mock_mesh)

        assert result["status"] == "solved"
        assert result["injected"] is False
        # Should NOT wait_for_timeout when injection fails
        mock_page.wait_for_timeout.assert_not_called()


class TestCaptchaSolvingEdgeCases:
    @pytest.mark.asyncio
    async def test_solve_create_task_api_error(self):
        """solve_captcha returns None when createTask API call fails."""
        from src.agent.builtins.captcha import solve_captcha

        mock_mesh = AsyncMock()
        mock_mesh.vault_resolve = AsyncMock(side_effect=lambda name: (
            "key" if name == "capsolver_key" else None
        ))

        with patch("src.agent.builtins.captcha.asyncio.sleep", new_callable=AsyncMock):
            with patch("src.agent.builtins.captcha.httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(
                    side_effect=httpx.HTTPStatusError(
                        "Internal Server Error",
                        request=MagicMock(),
                        response=MagicMock(status_code=500),
                    ),
                )
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client_cls.return_value = mock_client

                token = await solve_captcha(
                    {"type": "recaptcha_v2", "sitekey": "k"},
                    "https://example.com",
                    mock_mesh,
                )

        assert token is None

    @pytest.mark.asyncio
    async def test_solve_create_task_returns_error(self):
        """solve_captcha returns None when createTask returns an error response."""
        from src.agent.builtins.captcha import solve_captcha

        mock_mesh = AsyncMock()
        mock_mesh.vault_resolve = AsyncMock(side_effect=lambda name: (
            "key" if name == "capsolver_key" else None
        ))

        error_resp = MagicMock()
        error_resp.status_code = 200
        error_resp.raise_for_status = MagicMock()
        error_resp.json.return_value = {
            "errorId": 1,
            "errorCode": "ERROR_KEY_DOES_NOT_EXIST",
            "errorDescription": "Account not found",
        }

        with patch("src.agent.builtins.captcha.asyncio.sleep", new_callable=AsyncMock):
            with patch("src.agent.builtins.captcha.httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(return_value=error_resp)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client_cls.return_value = mock_client

                token = await solve_captcha(
                    {"type": "recaptcha_v2", "sitekey": "k"},
                    "https://example.com",
                    mock_mesh,
                )

        assert token is None

    @pytest.mark.asyncio
    async def test_solve_task_failed_status(self):
        """solve_captcha returns None when task status is 'failed'."""
        from src.agent.builtins.captcha import solve_captcha

        mock_mesh = AsyncMock()
        mock_mesh.vault_resolve = AsyncMock(side_effect=lambda name: (
            "key" if name == "capsolver_key" else None
        ))

        create_resp = MagicMock()
        create_resp.status_code = 200
        create_resp.raise_for_status = MagicMock()
        create_resp.json.return_value = {"taskId": "t1"}

        failed_resp = MagicMock()
        failed_resp.status_code = 200
        failed_resp.raise_for_status = MagicMock()
        failed_resp.json.return_value = {
            "status": "failed",
            "errorDescription": "CAPTCHA_UNSOLVABLE",
        }

        with patch("src.agent.builtins.captcha.asyncio.sleep", new_callable=AsyncMock):
            with patch("src.agent.builtins.captcha.httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(side_effect=[create_resp, failed_resp])
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client_cls.return_value = mock_client

                token = await solve_captcha(
                    {"type": "hcaptcha", "sitekey": "k"},
                    "https://example.com",
                    mock_mesh,
                )

        assert token is None

    @pytest.mark.asyncio
    async def test_solve_timeout(self):
        """solve_captcha returns None after polling timeout."""
        from src.agent.builtins.captcha import solve_captcha

        mock_mesh = AsyncMock()
        mock_mesh.vault_resolve = AsyncMock(side_effect=lambda name: (
            "key" if name == "capsolver_key" else None
        ))

        create_resp = MagicMock()
        create_resp.status_code = 200
        create_resp.raise_for_status = MagicMock()
        create_resp.json.return_value = {"taskId": "t1"}

        processing_resp = MagicMock()
        processing_resp.status_code = 200
        processing_resp.raise_for_status = MagicMock()
        processing_resp.json.return_value = {"status": "processing"}

        with patch("src.agent.builtins.captcha.asyncio.sleep", new_callable=AsyncMock), \
             patch("src.agent.builtins.captcha._POLL_TIMEOUT", 6), \
             patch("src.agent.builtins.captcha._POLL_INTERVAL", 3):
            with patch("src.agent.builtins.captcha.httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                # createTask + 2 polls (at 3s and 6s) then timeout
                mock_client.post = AsyncMock(
                    side_effect=[create_resp, processing_resp, processing_resp],
                )
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client_cls.return_value = mock_client

                token = await solve_captcha(
                    {"type": "turnstile", "sitekey": "k"},
                    "https://example.com",
                    mock_mesh,
                )

        assert token is None

    @pytest.mark.asyncio
    async def test_solve_unsupported_captcha_type(self):
        """solve_captcha returns None for an unrecognized CAPTCHA type."""
        from src.agent.builtins.captcha import solve_captcha

        mock_mesh = AsyncMock()
        mock_mesh.vault_resolve = AsyncMock(side_effect=lambda name: (
            "key" if name == "capsolver_key" else None
        ))

        token = await solve_captcha(
            {"type": "unknown_captcha_v99", "sitekey": "k"},
            "https://example.com",
            mock_mesh,
        )

        assert token is None


class TestCaptchaDetectionEdgeCases:
    @pytest.mark.asyncio
    async def test_detect_empty_dict(self):
        """detect_captcha returns None for empty dict result."""
        from src.agent.builtins.captcha import detect_captcha

        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value={})

        result = await detect_captcha(mock_page)
        assert result is None

    @pytest.mark.asyncio
    async def test_detect_dict_without_type(self):
        """detect_captcha returns None for dict missing 'type' key."""
        from src.agent.builtins.captcha import detect_captcha

        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value={"sitekey": "abc123"})

        result = await detect_captcha(mock_page)
        assert result is None

    @pytest.mark.asyncio
    async def test_detect_non_dict_result(self):
        """detect_captcha returns None for non-dict truthy results."""
        from src.agent.builtins.captcha import detect_captcha

        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value="unexpected string")

        result = await detect_captcha(mock_page)
        assert result is None


class TestBrowserNavigateCaptchaEdgeCases:
    @pytest.mark.asyncio
    async def test_navigate_captcha_detected_no_mesh_client(self):
        """browser_navigate reports captcha when no mesh_client available."""
        import src.agent.builtins.browser_tool as bt

        mock_page = AsyncMock()
        mock_page.url = "https://protected.example.com"
        mock_page.title = AsyncMock(return_value="Protected")
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_page.goto = AsyncMock(return_value=mock_response)
        mock_page.inner_text = AsyncMock(return_value="Captcha page")

        with patch.object(bt, "_get_page", return_value=mock_page), \
             patch(
                 "src.agent.builtins.captcha.detect_captcha",
                 new_callable=AsyncMock,
                 return_value={"type": "turnstile", "sitekey": "test"},
             ):
            # No mesh_client passed
            result = await bt.browser_navigate(
                url="https://protected.example.com",
            )

        assert result["captcha_detected"] == "turnstile"
        assert "captcha_note" in result

    @pytest.mark.asyncio
    async def test_navigate_captcha_exception_is_swallowed(self):
        """browser_navigate swallows CAPTCHA errors and returns normal content."""
        import src.agent.builtins.browser_tool as bt

        mock_page = AsyncMock()
        mock_page.url = "https://example.com"
        mock_page.title = AsyncMock(return_value="Example")
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_page.goto = AsyncMock(return_value=mock_response)
        mock_page.inner_text = AsyncMock(return_value="Normal content")

        with patch.object(bt, "_get_page", return_value=mock_page), \
             patch(
                 "src.agent.builtins.captcha.detect_captcha",
                 new_callable=AsyncMock,
                 side_effect=RuntimeError("Something broke in captcha module"),
             ):
            result = await bt.browser_navigate(url="https://example.com")

        # Should return content normally despite captcha error
        assert result["content"] == "Normal content"
        assert "error" not in result
        assert "captcha_solved" not in result
        assert "captcha_detected" not in result


# ── Coordination Tools Tests (subscribe_event, watch_blackboard, claim_task) ──


class TestSubscribeEventTool:
    @pytest.mark.asyncio
    async def test_subscribe_event_success(self):
        from src.agent.builtins.mesh_tool import subscribe_event

        mock_client = AsyncMock()
        mock_client.subscribe_topic = AsyncMock(return_value={"subscribed": True})
        result = await subscribe_event(topic="research_complete", mesh_client=mock_client)
        assert result["subscribed"] is True
        assert result["topic"] == "research_complete"
        mock_client.subscribe_topic.assert_awaited_once_with("research_complete")

    @pytest.mark.asyncio
    async def test_subscribe_event_no_mesh_client(self):
        from src.agent.builtins.mesh_tool import subscribe_event

        result = await subscribe_event(topic="test", mesh_client=None)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_subscribe_event_error(self):
        from src.agent.builtins.mesh_tool import subscribe_event

        mock_client = AsyncMock()
        mock_client.subscribe_topic = AsyncMock(side_effect=RuntimeError("fail"))
        result = await subscribe_event(topic="test", mesh_client=mock_client)
        assert "error" in result


class TestWatchBlackboardTool:
    @pytest.mark.asyncio
    async def test_watch_blackboard_success(self):
        from src.agent.builtins.mesh_tool import watch_blackboard

        mock_client = AsyncMock()
        mock_client.is_standalone = False
        mock_client.watch_blackboard = AsyncMock(return_value={"watching": True})
        result = await watch_blackboard(pattern="tasks/*", mesh_client=mock_client)
        assert result["watching"] is True
        assert result["pattern"] == "tasks/*"
        mock_client.watch_blackboard.assert_awaited_once_with("tasks/*")

    @pytest.mark.asyncio
    async def test_watch_blackboard_standalone(self):
        from src.agent.builtins.mesh_tool import watch_blackboard

        mock_client = AsyncMock()
        mock_client.is_standalone = True
        result = await watch_blackboard(pattern="tasks/*", mesh_client=mock_client)
        assert "error" in result
        assert "not assigned" in result["error"]

    @pytest.mark.asyncio
    async def test_watch_blackboard_no_mesh_client(self):
        from src.agent.builtins.mesh_tool import watch_blackboard

        result = await watch_blackboard(pattern="tasks/*", mesh_client=None)
        assert "error" in result


class TestClaimTaskTool:
    @pytest.mark.asyncio
    async def test_claim_task_success(self):
        from src.agent.builtins.mesh_tool import claim_task

        mock_client = AsyncMock()
        mock_client.is_standalone = False
        mock_client.read_blackboard = AsyncMock(return_value={
            "key": "tasks/t1", "value": {"status": "pending"}, "version": 1,
        })
        mock_client.claim_blackboard = AsyncMock(return_value={
            "key": "tasks/t1", "version": 2,
        })
        result = await claim_task(
            key="tasks/t1",
            claim_value='{"status": "claimed"}',
            mesh_client=mock_client,
        )
        assert result["claimed"] is True
        assert result["version"] == 2

    @pytest.mark.asyncio
    async def test_claim_task_conflict(self):
        from src.agent.builtins.mesh_tool import claim_task

        mock_client = AsyncMock()
        mock_client.is_standalone = False
        mock_client.read_blackboard = AsyncMock(return_value={
            "key": "tasks/t1", "value": {"status": "pending"}, "version": 1,
        })
        mock_client.claim_blackboard = AsyncMock(return_value=None)
        result = await claim_task(
            key="tasks/t1",
            claim_value='{"status": "claimed"}',
            mesh_client=mock_client,
        )
        assert result["claimed"] is False
        assert "conflict" in result["reason"].lower()

    @pytest.mark.asyncio
    async def test_claim_task_key_not_found(self):
        from src.agent.builtins.mesh_tool import claim_task

        mock_client = AsyncMock()
        mock_client.is_standalone = False
        mock_client.read_blackboard = AsyncMock(return_value=None)
        result = await claim_task(
            key="tasks/nope",
            claim_value='{"status": "claimed"}',
            mesh_client=mock_client,
        )
        assert result["claimed"] is False
        assert "does not exist" in result["reason"]

    @pytest.mark.asyncio
    async def test_claim_task_standalone(self):
        from src.agent.builtins.mesh_tool import claim_task

        mock_client = AsyncMock()
        mock_client.is_standalone = True
        result = await claim_task(
            key="tasks/t1",
            claim_value='{"status": "claimed"}',
            mesh_client=mock_client,
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_claim_task_no_mesh_client(self):
        from src.agent.builtins.mesh_tool import claim_task

        result = await claim_task(
            key="tasks/t1",
            claim_value='{"status": "claimed"}',
            mesh_client=None,
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_claim_task_invalid_json_fallback(self):
        """claim_task with non-JSON string falls back to {text: value}."""
        from src.agent.builtins.mesh_tool import claim_task

        mock_client = AsyncMock()
        mock_client.is_standalone = False
        mock_client.read_blackboard = AsyncMock(return_value={
            "key": "tasks/t1", "value": {"status": "pending"}, "version": 1,
        })
        mock_client.claim_blackboard = AsyncMock(return_value={
            "key": "tasks/t1", "version": 2,
        })
        result = await claim_task(
            key="tasks/t1",
            claim_value="not valid json",
            mesh_client=mock_client,
        )
        assert result["claimed"] is True
        # Verify the parsed value was {"text": "not valid json"}
        call_args = mock_client.claim_blackboard.call_args
        assert call_args[0][1] == {"text": "not valid json"}
