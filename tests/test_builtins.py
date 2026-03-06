"""Tests for built-in agent tools: exec, file, http, browser."""

import math
import os
import shutil
import tempfile
from unittest.mock import AsyncMock, MagicMock

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

        captured = {}
        async def _capture_request(client, method, url, headers, content, timeout):
            captured.update(method=method, url=url, headers=headers, content=content)
            return mock_response

        with patch("src.agent.builtins.http_tool._request_with_pinned_dns", side_effect=_capture_request):
            result = await http_request(
                url="https://api.github.com/user",
                headers={"Authorization": "Bearer $CRED{github_token}"},
                mesh_client=mock_mesh,
            )

        mock_mesh.vault_resolve.assert_called_once_with("github_token")
        assert captured["headers"]["Authorization"] == "Bearer secret-token-123"
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

        async def _mock_pinned(client, method, url, headers, content, timeout):
            return mock_response

        with patch("src.agent.builtins.http_tool._request_with_pinned_dns", side_effect=_mock_pinned):
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

        captured = {}
        async def _capture_request(client, method, url, headers, content, timeout):
            captured.update(url=url)
            return mock_response

        with patch("src.agent.builtins.http_tool._request_with_pinned_dns", side_effect=_capture_request):
            await http_request(
                url="https://api.example.com/data?key=$CRED{api_key}",
                mesh_client=mock_mesh,
            )

        mock_mesh.vault_resolve.assert_called_once_with("api_key")
        assert "my-api-key" in captured["url"]
        assert "$CRED" not in captured["url"]

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

        captured = {}
        async def _capture_request(client, method, url, headers, content, timeout):
            captured.update(content=content)
            return mock_response

        with patch("src.agent.builtins.http_tool._request_with_pinned_dns", side_effect=_capture_request):
            await http_request(
                url="https://8.8.8.8/api",
                method="POST",
                body='{"token": "$CRED{my_token}"}',
                mesh_client=mock_mesh,
            )

        mock_mesh.vault_resolve.assert_called_once_with("my_token")
        assert "secret-value" in captured["content"]
        assert "$CRED" not in captured["content"]

    @pytest.mark.asyncio
    async def test_response_body_redacted(self):
        """Secret values echoed in response body are redacted."""
        from unittest.mock import AsyncMock, patch

        from src.agent.builtins.http_tool import http_request

        mock_mesh = AsyncMock()
        mock_mesh.vault_resolve = AsyncMock(return_value="secret-token-123")

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.text = '{"token": "secret-token-123", "user": "alice"}'
        mock_response.headers = {"content-type": "application/json"}

        async def _mock_pinned(client, method, url, headers, content, timeout):
            return mock_response

        with patch("src.agent.builtins.http_tool._request_with_pinned_dns", side_effect=_mock_pinned):
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
        mock_response.headers = {
            "content-type": "application/json",
            "x-api-key": "secret-key-abc",
        }

        async def _mock_pinned(client, method, url, headers, content, timeout):
            return mock_response

        with patch("src.agent.builtins.http_tool._request_with_pinned_dns", side_effect=_mock_pinned):
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

        async def _fail_pinned(client, method, url, headers, content, timeout):
            raise Exception("Connection failed to https://api.example.com?key=super-secret-key")

        with patch("src.agent.builtins.http_tool._request_with_pinned_dns", side_effect=_fail_pinned):
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

        async def _mock_pinned(client, method, url, headers, content, timeout):
            return mock_response

        with patch("src.agent.builtins.http_tool._request_with_pinned_dns", side_effect=_mock_pinned):
            result = await http_request(url="https://example.com")

        assert result["body"] == '{"data": "normal response"}'
        assert result["headers"] == {"content-type": "application/json"}

    @pytest.mark.asyncio
    async def test_dns_pinning_blocks_private_ip(self):
        """_resolve_and_pin blocks requests to private IPs after DNS resolution."""
        from src.agent.builtins.http_tool import _resolve_and_pin

        # Direct private IP should be blocked
        with pytest.raises(ValueError, match="SSRF"):
            _resolve_and_pin("http://127.0.0.1/secret")

        with pytest.raises(ValueError, match="SSRF"):
            _resolve_and_pin("http://10.0.0.1/internal")

    @pytest.mark.asyncio
    async def test_dns_pinning_blocks_rebinding(self):
        """DNS resolution to a private IP is blocked even when hostname looks public."""
        from unittest.mock import patch

        from src.agent.builtins.http_tool import _resolve_and_pin

        # Mock DNS to resolve to a private IP (simulating DNS rebinding)
        fake_dns = [(2, 1, 6, '', ('192.168.1.1', 80))]
        with patch("src.agent.builtins.http_tool.socket.getaddrinfo", return_value=fake_dns):
            with pytest.raises(ValueError, match="SSRF"):
                _resolve_and_pin("http://evil.example.com/steal")

    @pytest.mark.asyncio
    async def test_redirect_revalidates_dns(self):
        """Redirects re-resolve DNS and block if target is private."""
        from unittest.mock import AsyncMock, patch

        from src.agent.builtins.http_tool import _request_with_pinned_dns

        # First request returns a redirect to a private IP
        redirect_response = AsyncMock()
        redirect_response.status_code = 302
        redirect_response.headers = {"location": "http://169.254.169.254/metadata"}
        redirect_response.request = AsyncMock()

        mock_client = AsyncMock()

        # Mock _send_pinned_request: first call succeeds, redirect target blocked
        call_count = 0
        async def _mock_send(client, method, url, headers, content, timeout):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return redirect_response
            raise ValueError("SSRF protection: requests to private/internal addresses are blocked")

        with patch("src.agent.builtins.http_tool._send_pinned_request", side_effect=_mock_send):
            with pytest.raises(ValueError, match="SSRF"):
                await _request_with_pinned_dns(
                    mock_client, "GET", "https://evil.example.com",
                    {}, None, 30,
                )

    @pytest.mark.asyncio
    async def test_max_redirects_enforced(self):
        """Exceeding max redirects raises an error."""
        from unittest.mock import AsyncMock, patch

        from src.agent.builtins.http_tool import _request_with_pinned_dns

        redirect_response = AsyncMock()
        redirect_response.status_code = 302
        redirect_response.headers = {"location": "https://example.com/loop"}
        redirect_response.request = AsyncMock()

        async def _always_redirect(client, method, url, headers, content, timeout):
            return redirect_response

        with patch("src.agent.builtins.http_tool._send_pinned_request", side_effect=_always_redirect):
            with pytest.raises(Exception, match="redirect"):
                await _request_with_pinned_dns(
                    AsyncMock(), "GET", "https://example.com/start",
                    {}, None, 30,
                )

    @pytest.mark.asyncio
    async def test_https_sni_set_correctly(self):
        """HTTPS requests set SNI extension to original hostname."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from src.agent.builtins.http_tool import _send_pinned_request

        mock_request = MagicMock()
        mock_request.extensions = {}

        mock_response = AsyncMock()
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.build_request = MagicMock(return_value=mock_request)
        mock_client.send = AsyncMock(return_value=mock_response)

        # Resolve to a public IP
        fake_dns = [(2, 1, 6, '', ('93.184.216.34', 443))]
        with patch("src.agent.builtins.http_tool.socket.getaddrinfo", return_value=fake_dns):
            await _send_pinned_request(
                mock_client, "GET", "https://example.com/path",
                {}, None, 30,
            )

        # Verify SNI was set to the original hostname
        assert mock_request.extensions["sni_hostname"] == b"example.com"
        # Verify Host header was set
        build_kwargs = mock_client.build_request.call_args
        assert build_kwargs.kwargs["headers"]["Host"] == "example.com"

    @pytest.mark.asyncio
    async def test_dns_pinning_blocks_ipv4_mapped_ipv6(self):
        """IPv4-mapped IPv6 addresses like ::ffff:127.0.0.1 are blocked."""
        from src.agent.builtins.http_tool import _resolve_and_pin

        with pytest.raises(ValueError, match="SSRF"):
            _resolve_and_pin("http://[::ffff:127.0.0.1]/secret")

    @pytest.mark.asyncio
    async def test_is_blocked_ip_link_local(self):
        """Link-local addresses (169.254.x.x) are blocked."""
        import ipaddress

        from src.agent.builtins.http_tool import _is_blocked_ip

        assert _is_blocked_ip(ipaddress.ip_address("169.254.1.1")) is True

    @pytest.mark.asyncio
    async def test_is_blocked_ip_ipv6_loopback(self):
        """IPv6 loopback (::1) is blocked."""
        import ipaddress

        from src.agent.builtins.http_tool import _is_blocked_ip

        assert _is_blocked_ip(ipaddress.ip_address("::1")) is True

    @pytest.mark.asyncio
    async def test_is_blocked_ip_public_is_allowed(self):
        """Public IPs are not blocked."""
        import ipaddress

        from src.agent.builtins.http_tool import _is_blocked_ip

        assert _is_blocked_ip(ipaddress.ip_address("8.8.8.8")) is False
        assert _is_blocked_ip(ipaddress.ip_address("93.184.216.34")) is False

    @pytest.mark.asyncio
    async def test_resolve_and_pin_rejects_non_http_scheme(self):
        """Non-http/https schemes (file://, gopher://) are rejected."""
        from src.agent.builtins.http_tool import _resolve_and_pin

        with pytest.raises(ValueError, match="only http and https"):
            _resolve_and_pin("file:///etc/passwd")

        with pytest.raises(ValueError, match="only http and https"):
            _resolve_and_pin("gopher://evil.com/steal")

    @pytest.mark.asyncio
    async def test_dns_failure_fails_closed(self):
        """DNS resolution failure blocks the request (fail-closed)."""
        import socket
        from unittest.mock import patch

        from src.agent.builtins.http_tool import _resolve_and_pin

        with patch("src.agent.builtins.http_tool.socket.getaddrinfo", side_effect=socket.gaierror("NXDOMAIN")):
            with pytest.raises(ValueError, match="DNS resolution failed"):
                _resolve_and_pin("http://nonexistent.invalid/path")


# ── browser_tool (thin HTTP client via mesh) ─────────────────


class TestBrowserNavigateHttpClient:
    """browser_navigate sends navigate command through mesh_client."""

    @pytest.mark.asyncio
    async def test_navigate_calls_browser_command(self):
        from src.agent.builtins.browser_tool import browser_navigate

        mc = AsyncMock()
        mc.browser_command = AsyncMock(return_value={
            "url": "https://example.com",
            "title": "Example",
            "content": "Hello world",
        })

        result = await browser_navigate(url="https://example.com", mesh_client=mc)

        mc.browser_command.assert_awaited_once_with("navigate", {"url": "https://example.com", "wait_ms": 1000})
        assert result["url"] == "https://example.com"
        assert result["content"] == "Hello world"

    @pytest.mark.asyncio
    async def test_navigate_passes_wait_ms(self):
        from src.agent.builtins.browser_tool import browser_navigate

        mc = AsyncMock()
        mc.browser_command = AsyncMock(return_value={"url": "https://example.com"})

        await browser_navigate(url="https://example.com", wait_ms=5000, mesh_client=mc)

        mc.browser_command.assert_awaited_once_with("navigate", {"url": "https://example.com", "wait_ms": 5000})

    @pytest.mark.asyncio
    async def test_navigate_no_mesh_client(self):
        from src.agent.builtins.browser_tool import browser_navigate

        result = await browser_navigate(url="https://example.com", mesh_client=None)
        assert "error" in result
        assert "mesh" in result["error"].lower()


class TestBrowserSnapshotHttpClient:
    """browser_snapshot sends snapshot command through mesh_client."""

    @pytest.mark.asyncio
    async def test_snapshot_calls_browser_command(self):
        from src.agent.builtins.browser_tool import browser_snapshot

        mc = AsyncMock()
        mc.browser_command = AsyncMock(return_value={
            "element_count": 3,
            "elements": [
                {"ref": "e1", "role": "button", "name": "Submit"},
                {"ref": "e2", "role": "textbox", "name": "Email"},
                {"ref": "e3", "role": "link", "name": "Sign up"},
            ],
        })

        result = await browser_snapshot(mesh_client=mc)

        mc.browser_command.assert_awaited_once_with("snapshot", {})
        assert result["element_count"] == 3


class TestBrowserClickHttpClient:
    """browser_click sends click command through mesh_client."""

    @pytest.mark.asyncio
    async def test_click_with_ref(self):
        from src.agent.builtins.browser_tool import browser_click

        mc = AsyncMock()
        mc.browser_command = AsyncMock(return_value={"clicked": "e1"})

        result = await browser_click(ref="e1", mesh_client=mc)

        mc.browser_command.assert_awaited_once_with("click", {"ref": "e1", "selector": ""})
        assert result["clicked"] == "e1"

    @pytest.mark.asyncio
    async def test_click_with_selector(self):
        from src.agent.builtins.browser_tool import browser_click

        mc = AsyncMock()
        mc.browser_command = AsyncMock(return_value={"clicked": "#btn"})

        result = await browser_click(selector="#btn", mesh_client=mc)

        mc.browser_command.assert_awaited_once_with("click", {"ref": "", "selector": "#btn"})
        assert result["clicked"] == "#btn"

    @pytest.mark.asyncio
    async def test_click_no_params_returns_error(self):
        from src.agent.builtins.browser_tool import browser_click

        result = await browser_click(mesh_client=AsyncMock())
        assert "error" in result
        assert "Provide either" in result["error"]


class TestBrowserTypeHttpClient:
    """browser_type sends type command through mesh_client (plain text)."""

    @pytest.mark.asyncio
    async def test_type_plain_text(self):
        from src.agent.builtins.browser_tool import browser_type

        mc = AsyncMock()
        mc.browser_command = AsyncMock(return_value={"success": True, "data": {"typed": "hello", "ref": "e5"}})

        result = await browser_type(text="hello", ref="e5", mesh_client=mc)

        mc.browser_command.assert_awaited_once_with(
            "type", {"ref": "e5", "selector": "", "text": "hello", "clear": True,
                     "is_credential": False},
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_type_empty_text_returns_error(self):
        from src.agent.builtins.browser_tool import browser_type

        result = await browser_type(text="", ref="e1", mesh_client=AsyncMock())
        assert "error" in result
        assert "'text'" in result["error"]

    @pytest.mark.asyncio
    async def test_type_no_ref_or_selector_returns_error(self):
        from src.agent.builtins.browser_tool import browser_type

        result = await browser_type(text="hello", mesh_client=AsyncMock())
        assert "error" in result
        assert "Provide either" in result["error"]


class TestBrowserTypeCredentialHandles:
    """$CRED{} handles are resolved agent-side before sending to browser service."""

    @pytest.mark.asyncio
    async def test_cred_handle_resolved_and_redacted(self):
        """$CRED{name} is resolved via vault and return shows [credential]."""
        from src.agent.builtins.browser_tool import browser_type

        mc = AsyncMock()
        mc.vault_resolve = AsyncMock(return_value="actual-secret-value")
        mc.browser_command = AsyncMock(return_value={"success": True, "data": {"typed": "x"}})

        result = await browser_type(text="$CRED{my_api_key}", ref="e5", mesh_client=mc)

        # Verify actual secret was sent to browser_command with is_credential flag
        mc.browser_command.assert_awaited_once()
        call_params = mc.browser_command.call_args[0][1]
        assert call_params["text"] == "actual-secret-value"
        assert call_params["is_credential"] is True
        # Return shows [credential] instead of actual value
        assert result["data"]["typed"] == "[credential]"
        assert "actual-secret-value" not in str(result)

    @pytest.mark.asyncio
    async def test_cred_handle_not_found(self):
        """$CRED{nonexistent} returns error."""
        from src.agent.builtins.browser_tool import browser_type

        mc = AsyncMock()
        mc.vault_resolve = AsyncMock(return_value=None)

        result = await browser_type(text="$CRED{nonexistent}", ref="e1", mesh_client=mc)
        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_cred_handle_no_mesh_client(self):
        """$CRED{} without mesh_client returns error."""
        from src.agent.builtins.browser_tool import browser_type

        result = await browser_type(text="$CRED{some_key}", ref="e1", mesh_client=None)
        assert "error" in result
        assert "mesh connectivity" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_mixed_text_and_cred(self):
        """Text with $CRED{} embedded resolves and returns [credential]."""
        from src.agent.builtins.browser_tool import browser_type

        mc = AsyncMock()
        mc.vault_resolve = AsyncMock(return_value="secret123")
        mc.browser_command = AsyncMock(return_value={"success": True, "data": {"typed": "x"}})

        result = await browser_type(text="Bearer $CRED{token}", ref="e1", mesh_client=mc)

        call_params = mc.browser_command.call_args[0][1]
        assert call_params["text"] == "Bearer secret123"
        assert result["data"]["typed"] == "[credential]"

    @pytest.mark.asyncio
    async def test_cred_handle_tracks_resolved_value(self):
        """$CRED{} resolution adds value to _resolved_credential_values."""
        import src.agent.builtins.browser_tool as bt
        from src.agent.builtins.browser_tool import browser_type

        bt._resolved_credential_values.clear()

        mc = AsyncMock()
        mc.vault_resolve = AsyncMock(return_value="MySecretP@ss123")
        mc.browser_command = AsyncMock(return_value={"success": True, "data": {}})

        await browser_type(text="$CRED{password}", ref="e1", mesh_client=mc)

        assert "MySecretP@ss123" in bt._resolved_credential_values
        bt._resolved_credential_values.clear()

    @pytest.mark.asyncio
    async def test_cred_handle_skips_short_values(self):
        """Resolved values shorter than 4 chars are not tracked."""
        import src.agent.builtins.browser_tool as bt
        from src.agent.builtins.browser_tool import browser_type

        bt._resolved_credential_values.clear()

        mc = AsyncMock()
        mc.vault_resolve = AsyncMock(return_value="abc")
        mc.browser_command = AsyncMock(return_value={"success": True, "data": {}})

        await browser_type(text="$CRED{pin}", ref="e1", mesh_client=mc)

        assert "abc" not in bt._resolved_credential_values
        bt._resolved_credential_values.clear()


class TestCredentialRedaction:
    """Tests for _deep_redact, _redact_credentials, _redact_resolved_credentials."""

    def test_deep_redact_nested_structures(self):
        import src.agent.builtins.browser_tool as bt
        from src.agent.builtins.browser_tool import _deep_redact

        bt._resolved_credential_values = {"secret-val"}

        assert _deep_redact({"a": {"b": "secret-val"}}) == {"a": {"b": "[REDACTED]"}}
        assert _deep_redact([{"k": "secret-val"}]) == [{"k": "[REDACTED]"}]
        assert _deep_redact({"items": ["ok", "secret-val"]}) == {"items": ["ok", "[REDACTED]"]}
        assert _deep_redact({"count": 42, "flag": True}) == {"count": 42, "flag": True}
        assert _deep_redact(None) is None
        assert _deep_redact("") == ""
        bt._resolved_credential_values.clear()

    def test_redact_credentials_api_key_patterns(self):
        from src.agent.builtins.browser_tool import _redact_credentials

        assert _redact_credentials("sk-abcdefghijklmnopqrstuvwxyz") == "[REDACTED]"
        assert _redact_credentials("xoxb-123-456-abcdefghijklmnop") == "[REDACTED]"
        assert _redact_credentials("AKIAIOSFODNN7EXAMPLE") == "[REDACTED]"

    def test_redact_credentials_preserves_normal_text(self):
        from src.agent.builtins.browser_tool import _redact_credentials

        assert _redact_credentials("Submit") == "Submit"
        assert _redact_credentials("Price: $42.00") == "Price: $42.00"
        assert _redact_credentials("") == ""

    def test_redact_resolved_credentials_basic(self):
        import src.agent.builtins.browser_tool as bt
        from src.agent.builtins.browser_tool import _redact_resolved_credentials

        bt._resolved_credential_values = {"secret123", "p@ssw0rd"}
        assert _redact_resolved_credentials("the key is secret123") == "the key is [REDACTED]"
        assert _redact_resolved_credentials("safe text here") == "safe text here"
        bt._resolved_credential_values.clear()

    def test_redact_resolved_credentials_noop_when_empty(self):
        import src.agent.builtins.browser_tool as bt
        from src.agent.builtins.browser_tool import _redact_resolved_credentials

        bt._resolved_credential_values = set()
        text = "sk-abcdefghijklmnopqrstuvwxyz"
        assert _redact_resolved_credentials(text) == text

    def test_browser_command_redacts_response(self):
        """_browser_command applies _deep_redact to the response."""
        import asyncio

        import src.agent.builtins.browser_tool as bt
        from src.agent.builtins.browser_tool import _browser_command

        bt._resolved_credential_values = {"leaked-secret"}

        mc = AsyncMock()
        mc.browser_command = AsyncMock(return_value={"content": "found leaked-secret here"})

        result = asyncio.get_event_loop().run_until_complete(
            _browser_command(mc, "navigate", {"url": "https://x.com"})
        )

        assert "leaked-secret" not in str(result)
        assert "[REDACTED]" in result["content"]
        bt._resolved_credential_values.clear()

    def test_browser_command_redacts_error(self):
        """_browser_command redacts errors too."""
        import asyncio

        import src.agent.builtins.browser_tool as bt
        from src.agent.builtins.browser_tool import _browser_command

        bt._resolved_credential_values = {"my-secret"}

        mc = AsyncMock()
        mc.browser_command = AsyncMock(side_effect=Exception("fail: my-secret exposed"))

        result = asyncio.get_event_loop().run_until_complete(
            _browser_command(mc, "navigate", {})
        )

        assert "my-secret" not in str(result)
        assert "[REDACTED]" in result["error"]
        bt._resolved_credential_values.clear()


class TestBrowserEvaluateHttpClient:
    """browser_evaluate sends evaluate command through mesh_client."""

    @pytest.mark.asyncio
    async def test_evaluate_calls_browser_command(self):
        from src.agent.builtins.browser_tool import browser_evaluate

        mc = AsyncMock()
        mc.browser_command = AsyncMock(return_value={"result": "Hello World"})

        result = await browser_evaluate(script="document.title", mesh_client=mc)

        mc.browser_command.assert_awaited_once_with("evaluate", {"expression": "document.title"})
        assert result["result"] == "Hello World"


class TestBrowserResetHttpClient:
    """browser_reset sends reset command and clears credential tracking."""

    @pytest.mark.asyncio
    async def test_reset_clears_credential_values(self):
        import src.agent.builtins.browser_tool as bt
        from src.agent.builtins.browser_tool import browser_reset

        bt._resolved_credential_values.add("some-secret")

        mc = AsyncMock()
        mc.browser_command = AsyncMock(return_value={"status": "reset"})

        result = await browser_reset(mesh_client=mc)

        mc.browser_command.assert_awaited_once_with("reset", {})
        assert result["status"] == "reset"
        assert len(bt._resolved_credential_values) == 0

    @pytest.mark.asyncio
    async def test_reset_no_mesh_client(self):
        import src.agent.builtins.browser_tool as bt
        from src.agent.builtins.browser_tool import browser_reset

        bt._resolved_credential_values.add("leftover")

        result = await browser_reset(mesh_client=None)

        assert "error" in result
        # Credentials should still be cleared even if mesh call fails
        assert len(bt._resolved_credential_values) == 0


class TestBrowserNoMeshClient:
    """All browser commands return errors when mesh_client is None."""

    @pytest.mark.asyncio
    async def test_navigate_no_mesh(self):
        from src.agent.builtins.browser_tool import browser_navigate
        result = await browser_navigate(url="https://x.com", mesh_client=None)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_snapshot_no_mesh(self):
        from src.agent.builtins.browser_tool import browser_snapshot
        result = await browser_snapshot(mesh_client=None)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_click_no_mesh(self):
        from src.agent.builtins.browser_tool import browser_click
        result = await browser_click(ref="e1", mesh_client=None)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_evaluate_no_mesh(self):
        from src.agent.builtins.browser_tool import browser_evaluate
        result = await browser_evaluate(script="1+1", mesh_client=None)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_screenshot_no_mesh(self):
        from src.agent.builtins.browser_tool import browser_screenshot
        result = await browser_screenshot(mesh_client=None)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_solve_captcha_no_mesh(self):
        from src.agent.builtins.browser_tool import browser_solve_captcha
        result = await browser_solve_captcha(mesh_client=None)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_scroll_no_mesh(self):
        from src.agent.builtins.browser_tool import browser_scroll
        result = await browser_scroll(mesh_client=None)
        assert "error" in result


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
        assert "browser_scroll" in registry.skills
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


# ── browser_screenshot ────────────────────────────────────────


class TestBrowserScreenshotHttpClient:
    """browser_screenshot sends screenshot command through mesh_client."""

    @pytest.mark.asyncio
    async def test_screenshot_default_params(self):
        from src.agent.builtins.browser_tool import browser_screenshot

        mc = AsyncMock()
        mc.browser_command = AsyncMock(return_value={
            "image_base64": "iVBORw0KGgo=",
            "width": 1280,
            "height": 720,
        })

        result = await browser_screenshot(mesh_client=mc)

        mc.browser_command.assert_awaited_once_with("screenshot", {"full_page": False})
        assert result["image_base64"] == "iVBORw0KGgo="

    @pytest.mark.asyncio
    async def test_screenshot_full_page(self):
        from src.agent.builtins.browser_tool import browser_screenshot

        mc = AsyncMock()
        mc.browser_command = AsyncMock(return_value={"image_base64": "data"})

        await browser_screenshot(full_page=True, mesh_client=mc)

        mc.browser_command.assert_awaited_once_with("screenshot", {"full_page": True})


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


# ── Blackboard Sanitization Tests ─────────────────────────────────


class TestBlackboardSanitization:
    """Verify that blackboard reads deep-sanitize values."""

    def _project_client(self):
        mc = AsyncMock()
        mc.is_standalone = False
        return mc

    @pytest.mark.asyncio
    async def test_read_sanitizes_nested_dict(self):
        """Strings inside nested dicts are sanitized."""
        from src.agent.builtins.mesh_tool import read_shared_state
        mc = self._project_client()
        # \u200b is a zero-width space — should be stripped by sanitize_for_prompt
        mc.read_blackboard = AsyncMock(return_value={
            "key": "k",
            "value": {"nested": {"deep": "hello\u200bworld"}},
        })
        result = await read_shared_state(key="k", mesh_client=mc)
        assert result["value"]["nested"]["deep"] == "helloworld"

    @pytest.mark.asyncio
    async def test_read_sanitizes_list_values(self):
        """Strings inside lists are sanitized."""
        from src.agent.builtins.mesh_tool import read_shared_state
        mc = self._project_client()
        mc.read_blackboard = AsyncMock(return_value={
            "key": "k",
            "value": ["clean", "has\u200binvisible"],
        })
        result = await read_shared_state(key="k", mesh_client=mc)
        assert result["value"] == ["clean", "hasinvisible"]

    @pytest.mark.asyncio
    async def test_read_sanitizes_plain_string(self):
        """Plain string values are sanitized (existing behavior)."""
        from src.agent.builtins.mesh_tool import read_shared_state
        mc = self._project_client()
        mc.read_blackboard = AsyncMock(return_value={
            "key": "k",
            "value": "text\u200bhere",
        })
        result = await read_shared_state(key="k", mesh_client=mc)
        assert result["value"] == "texthere"

    @pytest.mark.asyncio
    async def test_list_preview_sanitized(self):
        """List previews are sanitized."""
        from src.agent.builtins.mesh_tool import list_shared_state
        mc = self._project_client()
        mc.list_blackboard = AsyncMock(return_value=[{
            "key": "k",
            "written_by": "agent1",
            "updated_at": "2026-01-01",
            "value": {"data": "has\u200binvisible"},
        }])
        result = await list_shared_state(prefix="", mesh_client=mc)
        preview = result["entries"][0]["value_preview"]
        assert "\u200b" not in preview

    @pytest.mark.asyncio
    async def test_read_sanitizes_dict_keys(self):
        """Dict keys with invisible chars are sanitized."""
        from src.agent.builtins.mesh_tool import read_shared_state
        mc = self._project_client()
        mc.read_blackboard = AsyncMock(return_value={
            "key": "k",
            "value": {"inject\u200bed": "data"},
        })
        result = await read_shared_state(key="k", mesh_client=mc)
        keys = list(result["value"].keys())
        assert keys == ["injected"]

    def test_sanitize_value_preserves_non_strings(self):
        """_sanitize_value passes through numbers, bools, None."""
        from src.agent.builtins.mesh_tool import _sanitize_value
        assert _sanitize_value(42) == 42
        assert _sanitize_value(True) is True
        assert _sanitize_value(None) is None
        assert _sanitize_value(3.14) == 3.14


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


# ── CAPTCHA (browser_tool solve_captcha via mesh) ──────────────────


class TestBrowserSolveCaptchaHttpClient:
    """browser_solve_captcha sends solve_captcha command through mesh_client."""

    @pytest.mark.asyncio
    async def test_solve_captcha_calls_browser_command(self):
        from src.agent.builtins.browser_tool import browser_solve_captcha

        mc = AsyncMock()
        mc.browser_command = AsyncMock(return_value={
            "status": "solved",
            "captcha_type": "turnstile",
            "injected": True,
        })

        result = await browser_solve_captcha(mesh_client=mc)

        mc.browser_command.assert_awaited_once_with("solve_captcha", {})
        assert result["status"] == "solved"
        assert result["captcha_type"] == "turnstile"

    @pytest.mark.asyncio
    async def test_solve_captcha_no_mesh_client(self):
        from src.agent.builtins.browser_tool import browser_solve_captcha

        result = await browser_solve_captcha(mesh_client=None)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_solve_captcha_service_error(self):
        from src.agent.builtins.browser_tool import browser_solve_captcha

        mc = AsyncMock()
        mc.browser_command = AsyncMock(side_effect=Exception("service unavailable"))

        result = await browser_solve_captcha(mesh_client=mc)
        assert "error" in result


class TestBrowserScrollHttpClient:
    """browser_scroll sends scroll command through mesh_client."""

    @pytest.mark.asyncio
    async def test_scroll_calls_browser_command(self):
        from src.agent.builtins.browser_tool import browser_scroll

        mc = AsyncMock()
        mc.browser_command = AsyncMock(return_value={
            "success": True,
            "data": {"direction": "down", "pixels": 720},
        })

        result = await browser_scroll(direction="down", amount=720, mesh_client=mc)

        mc.browser_command.assert_awaited_once_with(
            "scroll", {"direction": "down", "amount": 720}
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_scroll_with_ref(self):
        from src.agent.builtins.browser_tool import browser_scroll

        mc = AsyncMock()
        mc.browser_command = AsyncMock(return_value={
            "success": True,
            "data": {"scrolled_to_ref": "e5"},
        })

        result = await browser_scroll(ref="e5", mesh_client=mc)

        mc.browser_command.assert_awaited_once_with(
            "scroll", {"direction": "down", "amount": 0, "ref": "e5"}
        )
        assert result["data"]["scrolled_to_ref"] == "e5"

    @pytest.mark.asyncio
    async def test_scroll_no_mesh_client(self):
        from src.agent.builtins.browser_tool import browser_scroll

        result = await browser_scroll(mesh_client=None)
        assert "error" in result


# ── Coordination Tools Tests (publish_event, subscribe_event, watch_blackboard, claim_task) ──


class TestPublishEventTool:
    @pytest.mark.asyncio
    async def test_publish_event_success(self):
        from src.agent.builtins.mesh_tool import publish_event

        mock_client = AsyncMock()
        mock_client.is_standalone = False
        mock_client.publish_event = AsyncMock(return_value={"published": True})
        result = await publish_event(topic="research_complete", data='{"done": true}', mesh_client=mock_client)
        assert result["published"] is True
        assert result["topic"] == "research_complete"

    @pytest.mark.asyncio
    async def test_publish_event_standalone(self):
        from src.agent.builtins.mesh_tool import publish_event

        mock_client = AsyncMock()
        mock_client.is_standalone = True
        result = await publish_event(topic="test", mesh_client=mock_client)
        assert "error" in result
        assert "not assigned" in result["error"]

    @pytest.mark.asyncio
    async def test_publish_event_no_mesh_client(self):
        from src.agent.builtins.mesh_tool import publish_event

        result = await publish_event(topic="test", mesh_client=None)
        assert "error" in result


class TestSubscribeEventTool:
    @pytest.mark.asyncio
    async def test_subscribe_event_success(self):
        from src.agent.builtins.mesh_tool import subscribe_event

        mock_client = AsyncMock()
        mock_client.is_standalone = False
        mock_client.subscribe_topic = AsyncMock(return_value={"subscribed": True})
        result = await subscribe_event(topic="research_complete", mesh_client=mock_client)
        assert result["subscribed"] is True
        assert result["topic"] == "research_complete"
        mock_client.subscribe_topic.assert_awaited_once_with("research_complete")

    @pytest.mark.asyncio
    async def test_subscribe_event_standalone(self):
        from src.agent.builtins.mesh_tool import subscribe_event

        mock_client = AsyncMock()
        mock_client.is_standalone = True
        result = await subscribe_event(topic="test", mesh_client=mock_client)
        assert "error" in result
        assert "not assigned" in result["error"]

    @pytest.mark.asyncio
    async def test_subscribe_event_no_mesh_client(self):
        from src.agent.builtins.mesh_tool import subscribe_event

        result = await subscribe_event(topic="test", mesh_client=None)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_subscribe_event_error(self):
        from src.agent.builtins.mesh_tool import subscribe_event

        mock_client = AsyncMock()
        mock_client.is_standalone = False
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
