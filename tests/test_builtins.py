"""Tests for built-in agent tools: exec, file, http, browser."""

import math
import os
import shutil
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

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
        assert "memory_recall" in registry.skills


# ── memory_tool ────────────────────────────────────────────────


class TestMemoryRecall:
    @pytest.mark.asyncio
    async def test_memory_recall_basic(self, tmp_path):
        """memory_recall calls search_hierarchical and returns results."""
        from src.agent.builtins.memory_tool import memory_recall
        from src.agent.memory import MemoryStore

        store = MemoryStore(db_path=str(tmp_path / "recall.db"))
        await store.store_fact("user_lang", "Python")
        await store.store_fact("user_editor", "Vim")

        result = await memory_recall("user preferences", max_results=5, memory_store=store)
        assert result["count"] >= 1
        assert len(result["results"]) >= 1
        store.close()

    @pytest.mark.asyncio
    async def test_memory_recall_with_category_filter(self, tmp_path):
        """category param filters results (without embeddings, uses text field)."""
        from src.agent.builtins.memory_tool import memory_recall
        from src.agent.memory import MemoryStore

        store = MemoryStore(db_path=str(tmp_path / "recall_cat.db"))
        await store.store_fact("user_lang", "Python", category="preference")
        await store.store_fact("project_name", "OpenLegion", category="fact")

        result = await memory_recall("Python", category="preference", max_results=5, memory_store=store)
        for r in result["results"]:
            assert r["category"] == "preference"
        store.close()

    @pytest.mark.asyncio
    async def test_memory_recall_category_filter_with_auto_categorize(self, tmp_path):
        """category filter works with auto-categorized facts (embed_fn + categorize_fn)."""
        from src.agent.builtins.memory_tool import memory_recall
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
        result = await memory_recall("Python", category="preferences", max_results=5, memory_store=store)
        assert result["count"] >= 1, "Should find preference-categorized facts"
        for r in result["results"]:
            assert r["category"] == "preferences"
        store.close()

    @pytest.mark.asyncio
    async def test_memory_recall_no_store(self):
        """memory_recall without memory_store returns error."""
        from src.agent.builtins.memory_tool import memory_recall

        result = await memory_recall("anything", memory_store=None)
        assert "error" in result

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

        long_text = "The user mentioned they prefer to work late at night and want all notifications disabled after 10pm"
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
    async def test_save_then_recall(self, tmp_path):
        """Facts saved via memory_save are findable via memory_recall."""
        from src.agent.builtins.memory_tool import memory_recall, memory_save
        from src.agent.memory import MemoryStore

        store = MemoryStore(db_path=str(tmp_path / "save_recall.db"))

        await memory_save("user timezone: America/New_York", memory_store=store)
        result = await memory_recall("timezone", max_results=5, memory_store=store)
        assert result["count"] >= 1
        assert any("New_York" in r["value"] for r in result["results"])
        store.close()


# ── browser backend selection ─────────────────────────────────


class TestBrowserBackendSelection:
    @pytest.mark.asyncio
    async def test_default_is_basic(self):
        """Without BROWSER_BACKEND, _get_page() launches basic Chromium."""
        import src.agent.builtins.browser_tool as bt
        bt._browser = bt._context = bt._page = None
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BROWSER_BACKEND", None)
            with patch.object(bt, "_launch_basic", new_callable=AsyncMock) as mock_basic:
                mock_page = AsyncMock()
                mock_page.is_closed.return_value = False
                mock_basic.return_value = (MagicMock(), MagicMock(), mock_page)
                await bt._get_page()
                mock_basic.assert_called_once()

    @pytest.mark.asyncio
    async def test_stealth_dispatches_to_camoufox(self):
        """BROWSER_BACKEND=stealth calls _launch_stealth()."""
        import src.agent.builtins.browser_tool as bt
        bt._browser = bt._context = bt._page = None
        with patch.dict(os.environ, {"BROWSER_BACKEND": "stealth"}):
            with patch.object(bt, "_launch_stealth", new_callable=AsyncMock) as mock_stealth:
                mock_page = AsyncMock()
                mock_page.is_closed.return_value = False
                mock_stealth.return_value = (MagicMock(), MagicMock(), mock_page)
                await bt._get_page()
                mock_stealth.assert_called_once()

    @pytest.mark.asyncio
    async def test_advanced_dispatches_to_brightdata(self):
        """BROWSER_BACKEND=advanced calls _launch_advanced(mesh_client)."""
        import src.agent.builtins.browser_tool as bt
        bt._browser = bt._context = bt._page = None
        mock_mesh = MagicMock()
        with patch.dict(os.environ, {"BROWSER_BACKEND": "advanced"}):
            with patch.object(bt, "_launch_advanced", new_callable=AsyncMock) as mock_adv:
                mock_page = AsyncMock()
                mock_page.is_closed.return_value = False
                mock_adv.return_value = (MagicMock(), MagicMock(), mock_page)
                await bt._get_page(mesh_client=mock_mesh)
                mock_adv.assert_called_once_with(mock_mesh)

    @pytest.mark.asyncio
    async def test_advanced_requires_mesh_client(self):
        """_launch_advanced() raises without mesh_client."""
        import src.agent.builtins.browser_tool as bt
        with pytest.raises(RuntimeError, match="mesh connectivity"):
            await bt._launch_advanced(None)

    @pytest.mark.asyncio
    async def test_advanced_requires_vault_credential(self):
        """_launch_advanced() raises when credential not found."""
        import src.agent.builtins.browser_tool as bt
        mock_mesh = AsyncMock()
        mock_mesh.vault_resolve = AsyncMock(return_value=None)
        with pytest.raises(RuntimeError, match="brightdata_cdp_url"):
            await bt._launch_advanced(mock_mesh)

    @pytest.mark.asyncio
    async def test_stealth_import_error(self):
        """_launch_stealth() gives helpful error when camoufox not installed."""
        import src.agent.builtins.browser_tool as bt
        with patch.dict("sys.modules", {"camoufox": None, "camoufox.async_api": None}):
            with pytest.raises(RuntimeError, match="camoufox is not installed"):
                await bt._launch_stealth()


# ── labeled screenshots ──────────────────────────────────────


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

        original_snapshot = bt.browser_snapshot_func

        async def mock_snapshot(*, mesh_client=None):
            snapshot_called.append(True)
            bt._page_refs["e1"] = AsyncMock()
            bt._page_refs["e1"].bounding_box = AsyncMock(return_value=None)
            return {"element_count": 1}

        bt.browser_snapshot_func = mock_snapshot

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
            bt.browser_snapshot_func = original_snapshot
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
