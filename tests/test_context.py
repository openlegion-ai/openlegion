"""Tests for ContextManager: compaction, flushing, pruning."""

from __future__ import annotations

import json
import shutil
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.context import (
    MODEL_CONTEXT_WINDOWS,
    ContextManager,
    _DEFAULT_CONTEXT_WINDOW,
    estimate_tokens,
)
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
                side_effect=[
                    # flush extraction
                    LLMResponse(
                        content='[{"key": "user_theme", "value": "prefers dark mode", "category": "preference"}]',
                        tokens_used=50,
                    ),
                    # summarization
                    LLMResponse(content="Summary of the conversation.", tokens_used=30),
                ]
            )

            cm = ContextManager(max_tokens=100, llm=llm, workspace=workspace)
            msgs = _make_messages(10, chars_each=200)
            result = await cm.maybe_compact("system", msgs)

            memory_content = workspace.load_memory()
            assert "dark mode" in memory_content
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
                    LLMResponse(content='[{"key": "topic", "value": "discussed Python and ML", "category": "fact"}]', tokens_used=30),
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


    @pytest.mark.asyncio
    async def test_flush_triggered_resets_after_compaction(self):
        """After compaction, proactive flush can fire again on next growth."""
        tmpdir = tempfile.mkdtemp()
        try:
            workspace = WorkspaceManager(workspace_dir=tmpdir)
            flush_json = '[{"key": "f1", "value": "v1", "category": "fact"}]'
            summary_text = "Summary of conversation."
            llm = MagicMock()
            llm.chat = AsyncMock(
                side_effect=[
                    # 1st: proactive flush at 60%
                    LLMResponse(content=flush_json, tokens_used=30),
                    # 2nd: compaction flush at 70%
                    LLMResponse(content=flush_json, tokens_used=30),
                    # 3rd: summarization
                    LLMResponse(content=summary_text, tokens_used=30),
                    # 4th: second proactive flush (after reset)
                    LLMResponse(content=flush_json, tokens_used=30),
                ]
            )

            # Phase 1: trigger proactive flush at ~63%
            cm = ContextManager(max_tokens=470, llm=llm, workspace=workspace)
            msgs_60 = _make_messages(5, chars_each=200)
            await cm.maybe_compact("system", msgs_60)
            assert cm._flush_triggered is True
            assert llm.chat.call_count == 1

            # Phase 2: force compaction at 70%+
            msgs_70 = _make_messages(10, chars_each=200)
            result = await cm.maybe_compact("system", msgs_70)
            assert cm._flush_triggered is False  # reset after compaction
            assert len(result) < len(msgs_70)

            # Phase 3: proactive flush fires again
            msgs_60b = _make_messages(5, chars_each=200)
            await cm.maybe_compact("system", msgs_60b)
            assert cm._flush_triggered is True
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_compaction_flush_stores_to_memory_db(self):
        """70% compaction flush stores structured facts in memory DB."""
        tmpdir = tempfile.mkdtemp()
        try:
            from src.agent.memory import MemoryStore

            workspace = WorkspaceManager(workspace_dir=tmpdir)
            memory_store = MemoryStore(db_path=":memory:")
            llm = MagicMock()
            llm.chat = AsyncMock(
                side_effect=[
                    # flush
                    LLMResponse(
                        content='[{"key": "compact_fact", "value": "from compaction", "category": "fact"}]',
                        tokens_used=30,
                    ),
                    # summarize
                    LLMResponse(content="Conversation summary.", tokens_used=30),
                ]
            )

            cm = ContextManager(max_tokens=100, llm=llm, workspace=workspace, memory=memory_store)
            msgs = _make_messages(10, chars_each=200)
            await cm.maybe_compact("system", msgs)

            fact = memory_store._get_fact_by_key("compact_fact")
            assert fact is not None
            assert fact.value == "from compaction"
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


class TestEstimateTokensAccuracy:
    def test_openai_tiktoken_returns_positive(self):
        """tiktoken for OpenAI models should return int > 0."""
        msgs = [{"role": "user", "content": "Hello, world!"}]
        tokens = estimate_tokens(msgs, model="openai/gpt-4o")
        assert isinstance(tokens, int)
        assert tokens > 0

    def test_anthropic_uses_3_5_ratio(self):
        """Anthropic models should use ~3.5 chars per token."""
        msgs = [{"role": "user", "content": "x" * 350}]
        tokens = estimate_tokens(msgs, model="anthropic/claude-sonnet-4-5-20250929")
        # 350 content chars + JSON overhead, divided by 3.5
        chars = sum(len(json.dumps(m)) for m in msgs)
        assert tokens == int(chars / 3.5)

    def test_unknown_model_uses_4_ratio(self):
        """Unknown models should fall back to 4 chars/token."""
        msgs = [{"role": "user", "content": "x" * 400}]
        tokens = estimate_tokens(msgs, model="custom/my-model")
        chars = sum(len(json.dumps(m)) for m in msgs)
        assert tokens == chars // 4

    def test_empty_model_uses_fallback(self):
        """Empty model string should use 4 chars/token fallback."""
        msgs = [{"role": "user", "content": "Hello"}]
        tokens_default = estimate_tokens(msgs, model="")
        tokens_no_model = estimate_tokens(msgs)
        assert tokens_default == tokens_no_model

    def test_unknown_openai_model_falls_back(self):
        """An unrecognized OpenAI model should fall back gracefully."""
        msgs = [{"role": "user", "content": "Hello, world!"}]
        tokens = estimate_tokens(msgs, model="openai/gpt-99-nonexistent")
        assert isinstance(tokens, int)
        assert tokens > 0


class TestModelContextWindows:
    def test_auto_detect_gpt4o(self):
        cm = ContextManager(model="openai/gpt-4o")
        assert cm.max_tokens == 128_000

    def test_auto_detect_claude(self):
        cm = ContextManager(model="anthropic/claude-sonnet-4-5-20250929")
        assert cm.max_tokens == 200_000

    def test_unknown_model_defaults_128k(self):
        cm = ContextManager(model="custom/unknown-model")
        assert cm.max_tokens == _DEFAULT_CONTEXT_WINDOW

    def test_explicit_max_tokens_overrides(self):
        cm = ContextManager(max_tokens=50_000, model="openai/gpt-4o")
        assert cm.max_tokens == 50_000


class TestContextWarning:
    def test_no_warning_below_80(self):
        cm = ContextManager(max_tokens=10_000)
        msgs = _make_messages(2, chars_each=50)  # very small
        assert cm.context_warning(msgs) is None

    def test_warning_at_80_pct(self):
        cm = ContextManager(max_tokens=100)
        msgs = _make_messages(5, chars_each=200)  # well over 80%
        warning = cm.context_warning(msgs)
        assert warning is not None
        assert "CONTEXT WARNING" in warning

    def test_warning_contains_token_counts(self):
        cm = ContextManager(max_tokens=100)
        msgs = _make_messages(5, chars_each=200)
        warning = cm.context_warning(msgs)
        assert warning is not None
        assert "100" in warning  # max_tokens
        assert "tokens" in warning


class TestFlushResetOnChatReset:
    @pytest.mark.asyncio
    async def test_reset_clears_flush_flag(self):
        """After reset(), proactive flush fires again at 60%+ usage."""
        tmpdir = tempfile.mkdtemp()
        try:
            workspace = WorkspaceManager(workspace_dir=tmpdir)
            llm = MagicMock()
            llm.chat = AsyncMock(
                return_value=LLMResponse(
                    content='[{"key": "pref", "value": "dark mode", "category": "preference"}]',
                    tokens_used=30,
                )
            )

            cm = ContextManager(max_tokens=470, llm=llm, workspace=workspace)
            msgs = _make_messages(5, chars_each=200)

            # First pass: proactive flush triggers
            await cm.maybe_compact("system", msgs)
            assert cm._flush_triggered is True
            assert llm.chat.call_count == 1

            # Second pass: no re-trigger (flag is set)
            await cm.maybe_compact("system", msgs)
            assert llm.chat.call_count == 1

            # Reset (simulates chat reset)
            cm.reset()
            assert cm._flush_triggered is False

            # Third pass: proactive flush fires AGAIN
            await cm.maybe_compact("system", msgs)
            assert llm.chat.call_count == 2
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestTokenCount:
    def test_returns_positive_int(self):
        cm = ContextManager(max_tokens=10_000)
        msgs = _make_messages(3, chars_each=100)
        count = cm.token_count(msgs)
        assert isinstance(count, int)
        assert count > 0

    def test_with_model(self):
        cm = ContextManager(max_tokens=10_000, model="anthropic/claude-sonnet-4-5-20250929")
        msgs = _make_messages(3, chars_each=100)
        count = cm.token_count(msgs)
        assert isinstance(count, int)
        assert count > 0
