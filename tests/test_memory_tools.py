"""Tests for memory_search, memory_save, and memory_recall built-in skills."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from src.agent.builtins.memory_tool import memory_recall, memory_save, memory_search
from src.agent.workspace import WorkspaceManager


class TestMemorySearch:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.ws = WorkspaceManager(workspace_dir=self._tmpdir)

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        (Path(self._tmpdir) / "MEMORY.md").write_text(
            "User is a Python developer who works on ML projects.\n"
        )
        result = await memory_search("Python developer", workspace_manager=self.ws)
        assert result["count"] > 0
        assert result["results"][0]["file"] == "MEMORY.md"

    @pytest.mark.asyncio
    async def test_search_without_workspace_returns_error(self):
        result = await memory_search("anything")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_search_respects_max_results(self):
        for i in range(10):
            (Path(self._tmpdir) / f"note_{i}.md").write_text(f"Python topic {i}")
        result = await memory_search("Python", max_results=2, workspace_manager=self.ws)
        assert result["count"] <= 2


class TestMemorySave:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.ws = WorkspaceManager(workspace_dir=self._tmpdir)

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_save_writes_to_daily_log(self):
        result = await memory_save("User prefers dark mode", workspace_manager=self.ws)
        assert result["saved"] is True
        assert result["saved_workspace"] is True
        log_files = list((Path(self._tmpdir) / "memory").glob("*.md"))
        assert len(log_files) == 1
        content = log_files[0].read_text()
        assert "dark mode" in content

    @pytest.mark.asyncio
    async def test_save_without_workspace_returns_error(self):
        result = await memory_save("anything")
        assert "error" in result


class TestMemoryToolDiscovery:
    def test_memory_tools_auto_discovered(self):
        from src.agent.skills import SkillRegistry

        registry = SkillRegistry(skills_dir="/nonexistent/path")
        assert "memory_search" in registry.skills
        assert "memory_save" in registry.skills
        assert "memory_recall" in registry.skills
