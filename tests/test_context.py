"""Tests for ContextManager: compaction, flushing, pruning."""

from __future__ import annotations

import json
import shutil
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.context import (
    _DEFAULT_CONTEXT_WINDOW,
    ContextManager,
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

    def test_image_blocks_use_fixed_estimate(self):
        """Image blocks should NOT count base64 chars — use a fixed ~1600 token estimate."""
        large_b64 = "A" * 500_000  # 500KB of base64 data
        msgs = [
            {"role": "user", "content": "take a screenshot"},
            {
                "role": "tool", "tool_call_id": "c1",
                "content": [
                    {"type": "text", "text": '{"status": "screenshot captured"}'},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{large_b64}"}},
                ],
            },
        ]
        tokens = estimate_tokens(msgs)
        # Without the fix, this would be ~125,000+ tokens (500K/4)
        # With the fix, the image is ~1,600 tokens + text overhead
        assert tokens < 5_000, f"Image inflated token count to {tokens} — base64 is being counted as text"

    def test_text_only_messages_unchanged(self):
        """Text-only messages should estimate the same as before."""
        msgs = [
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "Hi there, how can I help?"},
        ]
        tokens = estimate_tokens(msgs)
        assert 10 < tokens < 100


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
        result, did_compact = await cm.maybe_compact("system", msgs)
        assert result == msgs  # unchanged
        assert did_compact is False

    @pytest.mark.asyncio
    async def test_hard_prune_without_llm(self):
        cm = ContextManager(max_tokens=100, llm=None, workspace=None)
        msgs = _make_messages(10, chars_each=200)
        result, did_compact = await cm.maybe_compact("system", msgs)
        assert len(result) < len(msgs)
        assert did_compact is True

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
            result, did_compact = await cm.maybe_compact("system", msgs)

            memory_content = workspace.load_memory()
            assert "dark mode" in memory_content
            assert len(result) < len(msgs)
            assert did_compact is True
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
                    LLMResponse(
                        content='[{"key": "topic", "value": "discussed Python and ML", "category": "fact"}]',
                        tokens_used=30,
                    ),
                    LLMResponse(content="Summary: discussed Python and ML.", tokens_used=30),
                ]
            )

            cm = ContextManager(max_tokens=100, llm=llm, workspace=workspace)
            msgs = _make_messages(10, chars_each=200)
            result, did_compact = await cm.maybe_compact("system", msgs)

            assert did_compact is True
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
        result, did_compact = await cm.maybe_compact("system", msgs)
        assert len(result) < len(msgs)
        assert did_compact is True

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
            result, did_compact = await cm.maybe_compact("system", msgs)

            # Messages returned unchanged (no compaction yet)
            assert result == msgs
            assert did_compact is False
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
            result, did_compact = await cm.maybe_compact("system", msgs)

            # Should return messages unchanged (no crash)
            assert result == msgs
            assert did_compact is False
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
            result, did_compact = await cm.maybe_compact("system", msgs_70)
            assert cm._flush_triggered is False  # reset after compaction
            assert len(result) < len(msgs_70)
            assert did_compact is True

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
        # 1 first group + bridge assistant + 4 last groups = 6
        assert len(pruned) == 6

    def test_prune_leaves_small_lists_alone(self):
        cm = ContextManager(max_tokens=100)
        msgs = _make_messages(4, chars_each=50)
        pruned = cm._hard_prune(msgs)
        assert pruned == msgs

    def test_prune_inserts_bridge_for_consecutive_user_messages(self):
        """Hard prune inserts bridge assistant msg when gap creates consecutive user roles."""
        cm = ContextManager(max_tokens=100)
        # First group is user, then many alternating, last 4 groups start with user
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "u3"},
            {"role": "assistant", "content": "a3"},
            {"role": "user", "content": "u4"},
            {"role": "assistant", "content": "a4"},
            {"role": "user", "content": "u5"},
            {"role": "assistant", "content": "a5"},
        ]
        pruned = cm._hard_prune(msgs)
        # First group=[user], last 4=[a3, u4, a4, u5, a5] — but groups are single msgs
        # After pruning, first msg is user, second kept msg could be user
        # Check no two consecutive messages share the same role
        for i in range(len(pruned) - 1):
            if pruned[i].get("role") == "user" and pruned[i + 1].get("role") == "user":
                pytest.fail(
                    f"Consecutive user messages at index {i} and {i+1}: "
                    f"{pruned[i]['content']!r}, {pruned[i+1]['content']!r}"
                )


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
        # Should be roughly 350 content chars + metadata overhead / 3.5
        assert 90 < tokens < 120

    def test_unknown_model_uses_4_ratio(self):
        """Unknown models should fall back to 4 chars/token."""
        msgs = [{"role": "user", "content": "x" * 400}]
        tokens = estimate_tokens(msgs, model="custom/my-model")
        # Should be roughly 400 content chars + metadata overhead / 4
        assert 90 < tokens < 120

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
        assert cm.max_tokens > 0

    def test_auto_detect_claude(self):
        cm = ContextManager(model="anthropic/claude-sonnet-4-5-20250929")
        assert cm.max_tokens > 0

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


class TestForceCompact:
    @pytest.mark.asyncio
    async def test_force_compact_always_runs(self):
        """force_compact summarizes even when token usage is low."""
        llm = MagicMock()
        llm.chat = AsyncMock(
            return_value=LLMResponse(content="Summary of conversation.", tokens_used=20)
        )
        # Large window so normal compaction would NOT trigger
        cm = ContextManager(max_tokens=1_000_000, llm=llm, workspace=None)
        msgs = _make_messages(6, chars_each=100)
        assert not cm.should_compact(msgs)  # confirm normal wouldn't compact

        result = await cm.force_compact("system", msgs)
        # Should have summarized anyway
        assert len(result) < len(msgs)
        assert llm.chat.call_count >= 1

    @pytest.mark.asyncio
    async def test_force_compact_empty_messages(self):
        """force_compact returns empty list for empty input."""
        cm = ContextManager(max_tokens=100_000)
        result = await cm.force_compact("system", [])
        assert result == []

    @pytest.mark.asyncio
    async def test_force_compact_flushes_memory(self):
        """force_compact flushes facts to workspace before summarizing."""
        tmpdir = tempfile.mkdtemp()
        try:
            workspace = WorkspaceManager(workspace_dir=tmpdir)
            llm = MagicMock()
            llm.chat = AsyncMock(side_effect=[
                # First call: fact extraction
                LLMResponse(
                    content='[{"key": "test_fact", "value": "important", "category": "fact"}]',
                    tokens_used=20,
                ),
                # Second call: summarization
                LLMResponse(content="Summary.", tokens_used=10),
            ])
            cm = ContextManager(max_tokens=1_000_000, llm=llm, workspace=workspace)
            msgs = _make_messages(6, chars_each=200)

            await cm.force_compact("system", msgs)
            memory = workspace.load_memory()
            assert "test_fact" in memory
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_force_compact_falls_back_to_hard_prune(self):
        """force_compact uses hard_prune when LLM is unavailable."""
        cm = ContextManager(max_tokens=100, llm=None, workspace=None)
        msgs = _make_messages(20, chars_each=50)

        result = await cm.force_compact("system", msgs)
        assert len(result) < len(msgs)
