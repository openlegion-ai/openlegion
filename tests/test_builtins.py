"""Tests for built-in agent tools: exec, file, http."""

import shutil
import tempfile

import pytest

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
        with pytest.raises(ValueError, match="escapes"):
            self._ft._safe_path("../../etc/passwd")

    def test_read_with_offset_limit(self):
        self._ft.write_file(path="lines.txt", content="line0\nline1\nline2\nline3\n")
        result = self._ft.read_file(path="lines.txt", offset=1, limit=2)
        assert "line1" in result["content"]
        assert "line2" in result["content"]
        assert "line0" not in result["content"]


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
