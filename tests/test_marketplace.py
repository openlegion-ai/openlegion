"""Tests for tool marketplace: install, list, remove."""

import json
from pathlib import Path
from unittest.mock import patch

from src.marketplace import (
    _parse_tool_manifest,
    _validate_all_tools,
    install_tool,
    list_tools,
    remove_tool,
)


def _create_valid_tool_dir(path: Path, name: str = "test-tool", version: str = "1.0.0") -> None:
    """Create a minimal valid marketplace tool directory."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "TOOL.md").write_text(
        f"---\nname: {name}\nversion: {version}\ndescription: A test tool\n---\n# {name}\n"
    )
    (path / "my_tool.py").write_text(
        'from src.agent.tools import tool\n\n'
        '@tool(name="my_tool", description="Test", parameters={"x": {"type": "string"}})\n'
        'def my_tool(x: str) -> dict:\n'
        '    return {"result": x}\n'
    )


# ── manifest parsing ─────────────────────────────────────────


class TestParseManifest:
    def test_parse_manifest_valid(self, tmp_path):
        _create_valid_tool_dir(tmp_path)
        meta = _parse_tool_manifest(tmp_path)
        assert meta is not None
        assert meta["name"] == "test-tool"
        assert meta["version"] == "1.0.0"
        assert meta["description"] == "A test tool"

    def test_parse_manifest_missing_fields(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        (tmp_path / "TOOL.md").write_text("---\nname: foo\n---\nBody\n")
        meta = _parse_tool_manifest(tmp_path)
        assert meta is None  # Missing version and description

    def test_parse_manifest_no_file(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        meta = _parse_tool_manifest(tmp_path)
        assert meta is None

    def test_parse_manifest_no_front_matter(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        (tmp_path / "TOOL.md").write_text("# Just a readme\nNo front matter.\n")
        meta = _parse_tool_manifest(tmp_path)
        assert meta is None

    def test_parse_manifest_path_traversal_name(self, tmp_path):
        """Names with path traversal characters are rejected."""
        tmp_path.mkdir(exist_ok=True)
        (tmp_path / "TOOL.md").write_text(
            "---\nname: ../../etc/evil\nversion: 1.0\ndescription: Bad\n---\n"
        )
        meta = _parse_tool_manifest(tmp_path)
        assert meta is None

    def test_parse_manifest_slash_in_name(self, tmp_path):
        """Names with forward slashes are rejected."""
        tmp_path.mkdir(exist_ok=True)
        (tmp_path / "TOOL.md").write_text(
            "---\nname: foo/bar\nversion: 1.0\ndescription: Bad\n---\n"
        )
        meta = _parse_tool_manifest(tmp_path)
        assert meta is None

    def test_parse_manifest_yaml_missing(self, tmp_path):
        """Helpful error when PyYAML is not installed."""
        tmp_path.mkdir(exist_ok=True)
        (tmp_path / "TOOL.md").write_text(
            "---\nname: test\nversion: 1.0\ndescription: Test\n---\n"
        )
        with patch.dict("sys.modules", {"yaml": None}):
            meta = _parse_tool_manifest(tmp_path)
        assert meta is None


# ── install ───────────────────────────────────────────────────


class TestInstallTool:
    def test_install_tool_success(self, tmp_path):
        """Mock git clone, valid tool + manifest."""
        marketplace_dir = tmp_path / "marketplace"

        def mock_clone(cmd, **kwargs):
            # Simulate git clone by creating the directory with valid content
            clone_target = cmd[-1]
            _create_valid_tool_dir(Path(clone_target))
            return type("Result", (), {"returncode": 0, "stderr": ""})()

        with patch("src.marketplace.subprocess.run", side_effect=mock_clone):
            result = install_tool("https://github.com/user/test-tool.git", marketplace_dir)

        assert result.get("installed") is True
        assert result["name"] == "test-tool"
        assert (marketplace_dir / "test-tool" / ".installed.json").exists()

    def test_install_tool_leaves_no_staging_dir(self, tmp_path):
        """Staging is a unique mkdtemp dir (no shared _tmp_install to race) and
        is gone after a successful install."""
        marketplace_dir = tmp_path / "marketplace"

        def mock_clone(cmd, **kwargs):
            _create_valid_tool_dir(Path(cmd[-1]))
            return type("Result", (), {"returncode": 0, "stderr": ""})()

        with patch("src.marketplace.subprocess.run", side_effect=mock_clone):
            result = install_tool("https://github.com/user/test-tool.git", marketplace_dir)
        assert result.get("installed") is True
        assert list(marketplace_dir.glob("_tmp_install*")) == []

    def test_install_tool_validation_failure(self, tmp_path):
        """Tool with eval() is rejected."""
        marketplace_dir = tmp_path / "marketplace"

        def mock_clone(cmd, **kwargs):
            clone_target = cmd[-1]
            target = Path(clone_target)
            target.mkdir(parents=True, exist_ok=True)
            (target / "TOOL.md").write_text(
                "---\nname: evil\nversion: 1.0\ndescription: Bad tool\n---\n"
            )
            (target / "evil.py").write_text(
                'from src.agent.tools import tool\n\n'
                '@tool(name="evil", description="Bad", parameters={})\n'
                'def evil() -> dict:\n'
                '    return eval("1+1")\n'
            )
            return type("Result", (), {"returncode": 0, "stderr": ""})()

        with patch("src.marketplace.subprocess.run", side_effect=mock_clone):
            result = install_tool("https://github.com/user/evil.git", marketplace_dir)

        assert "error" in result
        assert "validation" in result["error"].lower()

    def test_install_tool_no_manifest(self, tmp_path):
        """Error when no TOOL.md."""
        marketplace_dir = tmp_path / "marketplace"

        def mock_clone(cmd, **kwargs):
            clone_target = cmd[-1]
            target = Path(clone_target)
            target.mkdir(parents=True, exist_ok=True)
            (target / "tool.py").write_text("print('hello')\n")
            return type("Result", (), {"returncode": 0, "stderr": ""})()

        with patch("src.marketplace.subprocess.run", side_effect=mock_clone):
            result = install_tool("https://github.com/user/no-manifest.git", marketplace_dir)

        assert "error" in result
        assert "TOOL.md" in result["error"]

    def test_install_tool_git_failure(self, tmp_path):
        """Error when git clone fails."""
        marketplace_dir = tmp_path / "marketplace"

        def mock_clone(cmd, **kwargs):
            return type("Result", (), {"returncode": 128, "stderr": "repo not found"})()

        with patch("src.marketplace.subprocess.run", side_effect=mock_clone):
            result = install_tool("https://github.com/user/nonexistent.git", marketplace_dir)

        assert "error" in result
        assert "clone failed" in result["error"].lower()

    def test_install_tool_with_ref(self, tmp_path):
        """--ref flag is passed to git clone."""
        marketplace_dir = tmp_path / "marketplace"
        captured_cmds: list = []

        def mock_clone(cmd, **kwargs):
            captured_cmds.append(cmd)
            clone_target = cmd[-1]
            _create_valid_tool_dir(Path(clone_target))
            return type("Result", (), {"returncode": 0, "stderr": ""})()

        with patch("src.marketplace.subprocess.run", side_effect=mock_clone):
            install_tool("https://github.com/user/test-tool.git", marketplace_dir, ref="v2.0")

        assert "--branch" in captured_cmds[0]
        assert "v2.0" in captured_cmds[0]


# ── list / remove ─────────────────────────────────────────────


class TestListTools:
    def test_list_tools(self, tmp_path):
        marketplace_dir = tmp_path / "marketplace"
        tool_dir = marketplace_dir / "my-tool"
        _create_valid_tool_dir(tool_dir)
        (tool_dir / ".installed.json").write_text(
            json.dumps({"name": "my-tool", "version": "1.0.0", "description": "Test"})
        )

        tools = list_tools(marketplace_dir)
        assert len(tools) == 1
        assert tools[0]["name"] == "my-tool"

    def test_list_tools_empty(self, tmp_path):
        tools = list_tools(tmp_path / "nonexistent")
        assert tools == []

    def test_list_tools_missing_metadata(self, tmp_path):
        """Tool dir without .installed.json still shows up."""
        marketplace_dir = tmp_path / "marketplace"
        (marketplace_dir / "orphan-tool").mkdir(parents=True)
        tools = list_tools(marketplace_dir)
        assert len(tools) == 1
        assert tools[0]["name"] == "orphan-tool"


class TestRemoveTool:
    def test_remove_tool(self, tmp_path):
        marketplace_dir = tmp_path / "marketplace"
        tool_dir = marketplace_dir / "my-tool"
        _create_valid_tool_dir(tool_dir)

        result = remove_tool("my-tool", marketplace_dir)
        assert result["removed"] is True
        assert not tool_dir.exists()

    def test_remove_nonexistent(self, tmp_path):
        marketplace_dir = tmp_path / "marketplace"
        marketplace_dir.mkdir(parents=True)

        result = remove_tool("ghost", marketplace_dir)
        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_remove_path_traversal(self, tmp_path):
        """Path traversal in remove name is rejected."""
        marketplace_dir = tmp_path / "marketplace"
        marketplace_dir.mkdir(parents=True)

        result = remove_tool("../../etc/passwd", marketplace_dir)
        assert "error" in result
        assert "invalid" in result["error"].lower()

    def test_remove_tool_rejects_catalog_wipe(self, tmp_path):
        """'.' / '' must not resolve to marketplace_dir itself and rmtree the
        whole catalog (the weak ..-only check let '.' through)."""
        marketplace_dir = tmp_path / "marketplace"
        _create_valid_tool_dir(marketplace_dir / "keep")

        for bad in (".", "", "   "):
            result = remove_tool(bad, marketplace_dir)
            assert "error" in result
        # Catalog (and the marketplace dir) survived.
        assert (marketplace_dir / "keep").exists()


# ── code validation ───────────────────────────────────────────


class TestValidateAllTools:
    def test_validates_clean_code(self, tmp_path):
        _create_valid_tool_dir(tmp_path)
        errors = _validate_all_tools(tmp_path)
        assert errors == []

    def test_catches_forbidden_call(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        (tmp_path / "bad.py").write_text(
            'from src.agent.tools import tool\n\n'
            '@tool(name="bad", description="Bad", parameters={})\n'
            'def bad() -> dict:\n'
            '    return eval("1+1")\n'
        )
        errors = _validate_all_tools(tmp_path)
        assert len(errors) == 1
        assert "eval" in errors[0]
