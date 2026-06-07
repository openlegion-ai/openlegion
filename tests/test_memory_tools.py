"""Tests for memory_search, memory_save, and memory_recall built-in tools."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.agent.builtins.memory_tool import memory_save, memory_search, memory_think
from src.agent.workspace import WorkspaceManager
from src.shared.types import LLMResponse, MemoryFact


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
        from src.agent.tools import ToolRegistry

        registry = ToolRegistry(tools_dir="/nonexistent/path")
        assert "memory_search" in registry.tools
        assert "memory_save" in registry.tools
        assert "memory_think" in registry.tools
        assert "category" in registry.tools["memory_search"]["parameters"]


def _make_facts(*pairs) -> list[MemoryFact]:
    return [
        MemoryFact(key=k, value=v, category="general", confidence=0.9)
        for k, v in pairs
    ]


def _fake_memory_store(facts: list[MemoryFact]) -> SimpleNamespace:
    """A memory_store stub whose search returns the given facts."""
    store = SimpleNamespace()
    store.search_hierarchical = AsyncMock(return_value=facts)
    store.search = AsyncMock(return_value=facts)
    return store


def _fake_mesh_client(agent_id: str = "agent-think") -> SimpleNamespace:
    return SimpleNamespace(agent_id=agent_id)


class TestMemoryThink:
    def setup_method(self):
        # Isolate the parent-LLM registry per test.
        from src.agent.builtins import subagent_tool

        self._subagent_tool = subagent_tool
        self._saved_refs = dict(subagent_tool._parent_llm_refs)
        subagent_tool._parent_llm_refs.clear()

    def teardown_method(self):
        self._subagent_tool._parent_llm_refs.clear()
        self._subagent_tool._parent_llm_refs.update(self._saved_refs)

    @pytest.mark.asyncio
    async def test_synthesis_path(self):
        facts = _make_facts(
            ("favourite colour", "blue"),
            ("home city", "Lisbon"),
            ("role", "ML engineer"),
        )
        store = _fake_memory_store(facts)
        mesh = _fake_mesh_client("agent-think")

        llm = SimpleNamespace()
        llm.chat = AsyncMock(
            return_value=LLMResponse(
                content="The user likes blue [1] and lives in Lisbon [2].\n"
                "Unknown / not in memory: their age.",
            )
        )
        self._subagent_tool.register_parent_llm("agent-think", llm)

        result = await memory_think(
            "Tell me about the user", memory_store=store, mesh_client=mesh,
        )

        assert result["answer"]
        assert "blue" in result["answer"]
        assert result["evidence_count"] > 0
        assert len(result["citations"]) == result["evidence_count"]
        # Exercised the LLM exactly once with our tuned params.
        llm.chat.assert_awaited_once()
        kwargs = llm.chat.await_args.kwargs
        assert kwargs["max_tokens"] == 512
        assert kwargs["temperature"] == 0.2

    @pytest.mark.asyncio
    async def test_degraded_path_no_llm_registered(self):
        facts = _make_facts(("favourite colour", "blue"))
        store = _fake_memory_store(facts)
        mesh = _fake_mesh_client("unregistered-agent")

        result = await memory_think(
            "what colour", memory_store=store, mesh_client=mesh,
        )

        assert result["answer"] is None
        assert result["results"]
        assert result["evidence_count"] == 1
        assert "Synthesis unavailable" in result["note"]

    @pytest.mark.asyncio
    async def test_empty_retrieval(self):
        store = _fake_memory_store([])
        mesh = _fake_mesh_client("agent-think")

        result = await memory_think(
            "anything", memory_store=store, mesh_client=mesh,
        )

        assert result["evidence_count"] == 0
        assert result["answer"] == ""
        assert "No relevant memory" in result["note"]

    @pytest.mark.asyncio
    async def test_no_backends_returns_error(self):
        result = await memory_think("anything")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_synthesis_failure_falls_back_to_raw(self):
        facts = _make_facts(("favourite colour", "blue"))
        store = _fake_memory_store(facts)
        mesh = _fake_mesh_client("agent-think")

        llm = SimpleNamespace()
        llm.chat = AsyncMock(side_effect=RuntimeError("boom"))
        self._subagent_tool.register_parent_llm("agent-think", llm)

        result = await memory_think(
            "what colour", memory_store=store, mesh_client=mesh,
        )

        assert result["answer"] is None
        assert result["results"]
        assert result["evidence_count"] == 1
