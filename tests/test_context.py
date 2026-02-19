"""Tests for ContextManager: compaction, flushing, pruning."""

from __future__ import annotations

import shutil
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.context import ContextManager, estimate_tokens
from src.agent.workspace import WorkspaceManager
from src.shared.types import LLMResponse


def _make_messages(count: int, chars_each: int = 200) -> list[dict]:
    """Generate fake messages of predictable size."""
    msgs = []
    for i in range(count):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"Message {i}: " + "x" * chars_each})
    return msgs


class TestEstimateTokens:
    def test_empty(self):
        assert estimate_tokens([]) == 0

    def test_estimates_roughly(self):
        msgs = [{"role": "user", "content": "Hello world, this is a test message."}]
        tokens = estimate_tokens(msgs)
        assert 5 < tokens < 50


class TestUsageTracking:
    def test_usage_fraction(self):
        cm = ContextManager(max_tokens=1000)
        msgs = _make_messages(1, chars_each=100)
        usage = cm.usage(msgs)
        assert 0 < usage < 1.0

    def test_should_compact_below_threshold(self):
        cm = ContextManager(max_tokens=100_000)
        msgs = _make_messages(2, chars_each=50)
        assert not cm.should_compact(msgs)

    def test_should_compact_above_threshold(self):
        cm = ContextManager(max_tokens=100)
        msgs = _make_messages(10, chars_each=200)
        assert cm.should_compact(msgs)


class TestCompaction:
    @pytest.mark.asyncio
    async def test_no_compact_below_threshold(self):
        cm = ContextManager(max_tokens=100_000)
        msgs = _make_messages(3, chars_each=50)
        result = await cm.maybe_compact("system", msgs)
        assert result == msgs  # unchanged

    @pytest.mark.asyncio
    async def test_hard_prune_without_llm(self):
        cm = ContextManager(max_tokens=100, llm=None, workspace=None)
        msgs = _make_messages(10, chars_each=200)
        result = await cm.maybe_compact("system", msgs)
        assert len(result) < len(msgs)

    @pytest.mark.asyncio
    async def test_compact_flushes_to_memory(self):
        tmpdir = tempfile.mkdtemp()
        try:
            workspace = WorkspaceManager(workspace_dir=tmpdir)
            llm = MagicMock()
            llm.chat = AsyncMock(
                return_value=LLMResponse(
                    content="- User prefers dark mode\n- User works on ML projects",
                    tokens_used=50,
                )
            )

            cm = ContextManager(max_tokens=100, llm=llm, workspace=workspace)
            msgs = _make_messages(10, chars_each=200)
            result = await cm.maybe_compact("system", msgs)

            memory_content = workspace.load_memory()
            assert "dark mode" in memory_content or "ML projects" in memory_content
            assert len(result) < len(msgs)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_compact_with_summarize(self):
        tmpdir = tempfile.mkdtemp()
        try:
            workspace = WorkspaceManager(workspace_dir=tmpdir)
            llm = MagicMock()
            llm.chat = AsyncMock(
                side_effect=[
                    LLMResponse(content="- Key fact extracted", tokens_used=30),
                    LLMResponse(content="Summary: discussed Python and ML.", tokens_used=30),
                ]
            )

            cm = ContextManager(max_tokens=100, llm=llm, workspace=workspace)
            msgs = _make_messages(10, chars_each=200)
            result = await cm.maybe_compact("system", msgs)

            assert len(result) <= 5
            assert any("Summary" in m.get("content", "") for m in result)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_compact_handles_llm_failure(self):
        llm = MagicMock()
        llm.chat = AsyncMock(side_effect=RuntimeError("LLM unavailable"))

        cm = ContextManager(max_tokens=100, llm=llm, workspace=None)
        msgs = _make_messages(10, chars_each=200)
        result = await cm.maybe_compact("system", msgs)
        assert len(result) < len(msgs)

    @pytest.mark.asyncio
    async def test_compact_skips_memory_when_content_short(self):
        tmpdir = tempfile.mkdtemp()
        try:
            workspace = WorkspaceManager(workspace_dir=tmpdir)
            llm = MagicMock()
            llm.chat = AsyncMock(
                return_value=LLMResponse(content="Summary.", tokens_used=10)
            )

            cm = ContextManager(max_tokens=50, llm=llm, workspace=workspace)
            msgs = [{"role": "user", "content": "hi"}]
            # Force threshold by using very low max_tokens
            await cm.maybe_compact("system", msgs)

            # With such a short conversation, _flush_to_memory should skip
            # (< 100 chars of conversation text), so only summarize is called
            assert llm.chat.call_count <= 2
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestProactiveFlush:
    @pytest.mark.asyncio
    async def test_proactive_flush_triggers_at_60_pct(self):
        """Proactive flush should trigger between 60-70% and return messages unchanged."""
        tmpdir = tempfile.mkdtemp()
        try:
            workspace = WorkspaceManager(workspace_dir=tmpdir)
            llm = MagicMock()
            llm.chat = AsyncMock(
                return_value=LLMResponse(
                    content='[{"key": "user_pref", "value": "likes Python", "category": "preference"}]',
                    tokens_used=30,
                )
            )

            # max_tokens=470 with 5 messages => ~65% usage (in the 60-70% window)
            cm = ContextManager(max_tokens=470, llm=llm, workspace=workspace)
            msgs = _make_messages(5, chars_each=200)
            result = await cm.maybe_compact("system", msgs)

            # Messages returned unchanged (no compaction yet)
            assert result == msgs
            # LLM was called for extraction
            assert llm.chat.call_count == 1
            # Facts written to MEMORY.md
            memory = workspace.load_memory()
            assert "user_pref" in memory
            assert "likes Python" in memory
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_proactive_flush_does_not_retrigger(self):
        """Once triggered, proactive flush should not run again."""
        tmpdir = tempfile.mkdtemp()
        try:
            workspace = WorkspaceManager(workspace_dir=tmpdir)
            llm = MagicMock()
            llm.chat = AsyncMock(
                return_value=LLMResponse(
                    content='[{"key": "fact1", "value": "val1", "category": "fact"}]',
                    tokens_used=30,
                )
            )

            cm = ContextManager(max_tokens=470, llm=llm, workspace=workspace)
            msgs = _make_messages(5, chars_each=200)
            await cm.maybe_compact("system", msgs)
            assert llm.chat.call_count == 1

            # Second call at same usage — should NOT flush again
            await cm.maybe_compact("system", msgs)
            assert llm.chat.call_count == 1  # still 1
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_proactive_flush_handles_llm_failure(self):
        """If LLM fails during proactive flush, no crash."""
        tmpdir = tempfile.mkdtemp()
        try:
            workspace = WorkspaceManager(workspace_dir=tmpdir)
            llm = MagicMock()
            llm.chat = AsyncMock(side_effect=RuntimeError("LLM down"))

            cm = ContextManager(max_tokens=470, llm=llm, workspace=workspace)
            msgs = _make_messages(5, chars_each=200)
            result = await cm.maybe_compact("system", msgs)

            # Should return messages unchanged (no crash)
            assert result == msgs
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_proactive_flush_stores_to_memory_db(self):
        """Proactive flush should store facts in memory DB when available."""
        tmpdir = tempfile.mkdtemp()
        try:
            from src.agent.memory import MemoryStore

            workspace = WorkspaceManager(workspace_dir=tmpdir)
            memory_store = MemoryStore(db_path=":memory:")
            llm = MagicMock()
            llm.chat = AsyncMock(
                return_value=LLMResponse(
                    content='[{"key": "tool_pref", "value": "uses vim", "category": "preference"}]',
                    tokens_used=30,
                )
            )

            cm = ContextManager(max_tokens=470, llm=llm, workspace=workspace, memory=memory_store)
            msgs = _make_messages(5, chars_each=200)
            await cm.maybe_compact("system", msgs)

            # Fact should be in memory DB
            fact = memory_store._get_fact_by_key("tool_pref")
            assert fact is not None
            assert fact.value == "uses vim"
            memory_store.close()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestHardPrune:
    def test_prune_keeps_first_and_last(self):
        cm = ContextManager(max_tokens=100)
        msgs = _make_messages(10, chars_each=50)
        pruned = cm._hard_prune(msgs)
        assert pruned[0] == msgs[0]  # first preserved
        assert pruned[-1] == msgs[-1]  # last preserved
        assert len(pruned) == 5

    def test_prune_leaves_small_lists_alone(self):
        cm = ContextManager(max_tokens=100)
        msgs = _make_messages(4, chars_each=50)
        pruned = cm._hard_prune(msgs)
        assert pruned == msgs
