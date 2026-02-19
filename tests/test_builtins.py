"""Tests for built-in agent tools: exec, file, http, browser."""

import shutil
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

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
