"""Tests for ContextManager: compaction, flushing, pruning."""

from __future__ import annotations

import shutil
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.context import (
    ContextManager,
    estimate_tokens,
    group_messages_by_tool_call,
)
from src.agent.workspace import WorkspaceManager
from src.shared.models import _DEFAULT_CONTEXT_WINDOW
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
    async def test_compact_summary_is_sanitized_before_reinjection(self):
        """M2: the auto-compaction summary is passed through
        sanitize_for_prompt before being re-injected as a user message —
        invisible/control chars (e.g. zero-width space, RTL override) are
        stripped, matching the memory/bootstrap entry paths. Lossless for
        normal text, so summary quality is preserved."""
        tmpdir = tempfile.mkdtemp()
        try:
            workspace = WorkspaceManager(workspace_dir=tmpdir)
            # Summary laced with a zero-width space (U+200B) and a
            # right-to-left override (U+202E). Both are Unicode Cf chars
            # that sanitize_for_prompt strips.
            dirty = "Summary: discussed​Python‮and ML."
            llm = MagicMock()
            llm.chat = AsyncMock(
                side_effect=[
                    LLMResponse(
                        content='[{"key": "topic", "value": "ok", "category": "fact"}]',
                        tokens_used=30,
                    ),
                    LLMResponse(content=dirty, tokens_used=30),
                ]
            )

            cm = ContextManager(max_tokens=100, llm=llm, workspace=workspace)
            msgs = _make_messages(10, chars_each=200)
            result, did_compact = await cm.maybe_compact("system", msgs)

            assert did_compact is True
            summary_msg = next(
                m for m in result
                if "Conversation Summary" in m.get("content", "")
            )
            content = summary_msg["content"]
            # Invisible chars stripped...
            assert "​" not in content
            assert "‮" not in content
            # ...but the visible text survives (sanitize is lossless).
            assert "discussedPython" in content
            assert "and ML." in content
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


# ── Bug 8 fix: _summarize_compact must preserve tool-call group atomicity ──
#
# Before the fix, _summarize_compact sliced ``messages[-4:]`` as the recent
# tail. When an assistant turn made 4+ tool calls, that slice landed on
# ``[tool_a, tool_b, tool_c, tool_d]`` and orphaned the parent
# ``assistant(tool_calls)`` at position -5. The next LLM call would reject
# the messages (tool messages without preceding assistant_with_tool_calls),
# the exception fired in the chat loop, and the post-tool continuation
# silently failed. _hard_prune was already group-aware; _summarize_compact
# was the odd one out.
#
# These tests pin: (1) the shared group_messages_by_tool_call helper,
# (2) _summarize_compact's group-aware tail slicing, and (3) the bridge
# logic that no longer produces consecutive assistant messages.


def _make_multimsg(*roles_and_content: tuple[str, str]) -> list[dict]:
    """Compact helper to build heterogeneous message lists for the
    group-aware compaction tests."""
    return [{"role": r, "content": c} for r, c in roles_and_content]


class TestGroupMessagesByToolCall:
    """Pin the shared grouping helper. Three call sites use it
    (_summarize_compact, _hard_prune, AgentLoop._trim_context) — drift
    here would re-introduce the orphan-tool bug at any of them."""

    def test_empty_messages_returns_no_groups(self):
        assert group_messages_by_tool_call([]) == []

    def test_standalone_messages_each_become_own_group(self):
        msgs = _make_multimsg(
            ("user", "hi"), ("assistant", "hello"), ("user", "bye"),
        )
        groups = group_messages_by_tool_call(msgs)
        assert len(groups) == 3
        assert all(len(g) == 1 for g in groups)

    def test_assistant_with_tools_groups_with_following_tools(self):
        msgs = [
            {"role": "user", "content": "do stuff"},
            {"role": "assistant", "tool_calls": [
                {"id": "t1", "function": {"name": "x", "arguments": "{}"}},
                {"id": "t2", "function": {"name": "y", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "t1", "content": "r1"},
            {"role": "tool", "tool_call_id": "t2", "content": "r2"},
        ]
        groups = group_messages_by_tool_call(msgs)
        # [user], [asst+tools, tool_t1, tool_t2]
        assert len(groups) == 2
        assert len(groups[0]) == 1
        assert len(groups[1]) == 3
        assert groups[1][0]["role"] == "assistant"
        assert groups[1][1]["role"] == "tool"
        assert groups[1][2]["role"] == "tool"

    def test_assistant_text_only_does_not_eat_following_tools(self):
        """An assistant message WITHOUT tool_calls must not absorb
        subsequent tool messages — those tools belong to a different
        (preceding) assistant or are themselves orphans we don't
        rewrite."""
        msgs = [
            {"role": "assistant", "content": "ack"},
            # Pathological orphan tool — grouping must not eat it.
            {"role": "tool", "tool_call_id": "orphan", "content": "r"},
        ]
        groups = group_messages_by_tool_call(msgs)
        assert len(groups) == 2
        assert groups[0][0]["role"] == "assistant"
        assert groups[1][0]["role"] == "tool"

    def test_multi_tool_groups_chained_in_one_turn(self):
        """Multiple assistant-with-tools turns in sequence each form
        their own group, with tool results properly attached to their
        parent assistant."""
        msgs = [
            {"role": "assistant", "tool_calls": [{"id": "a1", "function": {"name": "x", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "a1", "content": "ra"},
            {"role": "assistant", "tool_calls": [{"id": "b1", "function": {"name": "y", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "b1", "content": "rb"},
        ]
        groups = group_messages_by_tool_call(msgs)
        assert len(groups) == 2
        assert all(g[0]["role"] == "assistant" and g[0].get("tool_calls") for g in groups)
        assert all(g[1]["role"] == "tool" for g in groups)


class TestSummarizeCompactGroupAware:
    """End-to-end pin: _summarize_compact preserves tool-call atomicity
    so the next LLM call after compaction never sees orphan tool messages."""

    @pytest.mark.asyncio
    async def test_four_tool_turn_at_tail_not_orphaned(self):
        """A turn with 4 tool calls right before compaction must NOT
        be split — pre-fix, ``messages[-4:]`` landed on
        ``[tool_a, tool_b, tool_c, tool_d]`` and orphaned the parent
        ``assistant(tool_calls)`` at position -5."""
        llm = MagicMock()
        llm.chat = AsyncMock(return_value=LLMResponse(
            content="Conversation summary.", tokens_used=10,
        ))
        cm = ContextManager(max_tokens=100, llm=llm, workspace=None)

        msgs = [
            {"role": "user", "content": "warm-up"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "do 4 things in parallel"},
            {"role": "assistant", "tool_calls": [
                {"id": "t1", "function": {"name": "a", "arguments": "{}"}},
                {"id": "t2", "function": {"name": "b", "arguments": "{}"}},
                {"id": "t3", "function": {"name": "c", "arguments": "{}"}},
                {"id": "t4", "function": {"name": "d", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "t1", "content": "r1"},
            {"role": "tool", "tool_call_id": "t2", "content": "r2"},
            {"role": "tool", "tool_call_id": "t3", "content": "r3"},
            {"role": "tool", "tool_call_id": "t4", "content": "r4"},
        ]

        result = await cm._summarize_compact("system", msgs)

        # Invariant: every tool message must be immediately preceded by
        # an assistant message that carries the matching ``tool_call_id``
        # (or by another tool message from the same assistant group).
        tool_call_ids: set[str] = set()
        for i, m in enumerate(result):
            if m.get("role") == "tool":
                assert i > 0, "tool message at index 0 has no parent assistant"
                assert m["tool_call_id"] in tool_call_ids, (
                    f"tool message at index {i} (id={m['tool_call_id']}) "
                    f"has no preceding assistant with matching tool_call — "
                    f"orphan after compaction. result: {result}"
                )
            elif m.get("role") == "assistant" and m.get("tool_calls"):
                tool_call_ids = {tc["id"] for tc in m["tool_calls"]}
            else:
                tool_call_ids = set()

    @pytest.mark.asyncio
    async def test_legacy_consecutive_assistants_bug_fixed(self):
        """Six-message alternating conversation triggered the pre-fix
        bridge logic to produce two consecutive assistants:
        ``[summary(user), bridge(asst), asst_2(asst), user_3, asst_3]``.
        With group-aware drop-leading-user-group, the bridge is no
        longer needed because dropping atomically lands on an assistant
        (alternation: user(summary) → assistant → ...)."""
        llm = MagicMock()
        llm.chat = AsyncMock(return_value=LLMResponse(
            content="Summary.", tokens_used=10,
        ))
        cm = ContextManager(max_tokens=100, llm=llm, workspace=None)

        msgs = _make_multimsg(
            ("user", "u1"), ("assistant", "a1"),
            ("user", "u2"), ("assistant", "a2"),
            ("user", "u3"), ("assistant", "a3"),
        )

        result = await cm._summarize_compact("system", msgs)

        # Walk result, assert no consecutive same-role between
        # neighbours of role in {user, assistant} (tool messages are
        # allowed adjacent to anything by API contract).
        for i in range(len(result) - 1):
            r_a = result[i].get("role")
            r_b = result[i + 1].get("role")
            if r_a in ("user", "assistant") and r_b in ("user", "assistant"):
                assert r_a != r_b, (
                    f"consecutive {r_a} messages at index {i},{i+1} — "
                    f"breaks LLM API alternation. result: {result}"
                )

    @pytest.mark.asyncio
    async def test_recent_starting_with_user_drops_leading_group_no_bridge(self):
        """When the leading recent group is a standalone user message,
        the fix drops it (atomic, since the group has one message) and
        does NOT add a bridge — the next group starts with assistant
        so summary(user) → assistant alternation is already valid."""
        llm = MagicMock()
        llm.chat = AsyncMock(return_value=LLMResponse(
            content="Summary.", tokens_used=10,
        ))
        cm = ContextManager(max_tokens=100, llm=llm, workspace=None)

        msgs = _make_multimsg(
            ("user", "u0"), ("assistant", "a0"),
            ("user", "u1"), ("assistant", "a1"),
            ("user", "u2"), ("assistant", "a2"),
        )
        result = await cm._summarize_compact("system", msgs)

        # First message is the summary (user role)
        assert result[0].get("role") == "user"
        assert "Summary" in result[0]["content"]
        # Second must be assistant — alternation is valid without bridge
        assert result[1].get("role") == "assistant"
        # The legacy bridge text must NOT appear (it would, pre-fix,
        # produce a consecutive-assistant bug)
        assert not any(
            m.get("role") == "assistant"
            and m.get("content") == "Understood, continuing from the summary above."
            for m in result
        )

    @pytest.mark.asyncio
    async def test_single_recent_group_starting_with_user_keeps_bridge(self):
        """Edge case: only 2 groups total — older has 1 group,
        recent has 1 group that starts with user. We can't drop the
        only recent group (would lose the kept turn entirely). Insert
        the bridge assistant between summary and the user message."""
        llm = MagicMock()
        llm.chat = AsyncMock(return_value=LLMResponse(
            content="Summary.", tokens_used=10,
        ))
        cm = ContextManager(max_tokens=100, llm=llm, workspace=None)

        # 2 groups total: [asst], [user]. Recent (keep_n=1) = [[user]].
        msgs = _make_multimsg(("assistant", "a0"), ("user", "u0"))
        result = await cm._summarize_compact("system", msgs)

        # summary(user) → bridge(asst) → u0(user) is correct alternation
        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "Understood, continuing from the summary above."
        assert result[2]["role"] == "user"
        assert result[2]["content"] == "u0"

    @pytest.mark.asyncio
    async def test_single_group_returns_as_is(self):
        """Cannot meaningfully compact with only one group — there's
        nothing older to summarize without losing the only turn.
        Return messages unchanged."""
        cm = ContextManager(max_tokens=100, llm=MagicMock(), workspace=None)
        msgs = [{"role": "user", "content": "single message"}]
        result = await cm._summarize_compact("system", msgs)
        assert result == msgs

    @pytest.mark.asyncio
    async def test_first_input_is_orphan_tool_grouped_as_standalone(self):
        """Codex P1 follow-up: ``group_messages_by_tool_call`` treats a
        bare leading ``tool`` message as its own standalone group (no
        assistant to absorb into). Pins that the helper does not
        consume orphans into a non-existent assistant group."""
        msgs = [
            {"role": "tool", "tool_call_id": "orphan_1", "content": "r"},
            {"role": "user", "content": "hi"},
        ]
        groups = group_messages_by_tool_call(msgs)
        assert len(groups) == 2
        assert groups[0][0]["role"] == "tool"
        assert groups[1][0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_summarize_empty_messages_no_compact(self):
        """Codex P1 follow-up: zero-message input returns as-is without
        invoking the LLM. The threshold gate above this layer should
        prevent the call, but defense-in-depth."""
        cm = ContextManager(max_tokens=100, llm=MagicMock(), workspace=None)
        result = await cm._summarize_compact("system", [])
        assert result == []

    @pytest.mark.asyncio
    async def test_summarize_first_retained_role_never_tool(self):
        """Codex P1.1 follow-up: pathological input where the recent
        tail leads with an orphan ``tool`` group must NOT produce
        ``[summary(user), tool, ...]``. Either drop the orphans
        atomically (when 2+ groups available) OR fall back to
        ``_hard_prune`` (the orphan-leads-alone case)."""
        llm = MagicMock()
        llm.chat = AsyncMock(return_value=LLMResponse(
            content="Summary.", tokens_used=10,
        ))
        cm = ContextManager(max_tokens=100, llm=llm, workspace=None)

        # 4 standalone-tool groups + 1 assistant group at the tail.
        # group_messages_by_tool_call yields 5 groups; with keep_n=4
        # the recent tail leads with the second orphan-tool group.
        # The fix drops those leading tools and lands on the assistant.
        msgs = [
            {"role": "user", "content": "kickoff"},
            {"role": "tool", "tool_call_id": "x1", "content": "orph1"},
            {"role": "tool", "tool_call_id": "x2", "content": "orph2"},
            {"role": "tool", "tool_call_id": "x3", "content": "orph3"},
            {"role": "tool", "tool_call_id": "x4", "content": "orph4"},
            {"role": "assistant", "content": "final"},
        ]
        result = await cm._summarize_compact("system", msgs)
        # First retained message must not be a tool — would orphan in
        # the API. The summary is at index 0; index 1 is the first
        # retained from recent.
        if len(result) > 1:
            assert result[1].get("role") != "tool", (
                f"orphan tool leaked into recent tail: {result}"
            )

    @pytest.mark.asyncio
    async def test_summarize_two_leading_user_groups_no_consecutive_users(self):
        """Codex P1.2 follow-up: when recent has multiple consecutive
        user groups (pathological: queued chat inputs / reconnect race
        producing two user messages back-to-back), dropping just the
        first would leave a second user group at the head — still
        consecutive with summary(user). The fix loops the drop so all
        leading user groups are stripped."""
        llm = MagicMock()
        llm.chat = AsyncMock(return_value=LLMResponse(
            content="Summary.", tokens_used=10,
        ))
        cm = ContextManager(max_tokens=100, llm=llm, workspace=None)

        # 5 groups: 1 older + 4 recent. Recent leads with 2 user
        # groups, then assistant turns. Need keep_n=4 to fire the
        # multi-drop path.
        msgs = _make_multimsg(
            ("assistant", "a0"),  # older
            ("user", "u1"),       # recent[0]
            ("user", "u2"),       # recent[1] — also user! consecutive
            ("assistant", "a1"),  # recent[2]
            ("user", "u3"),       # recent[3]
        )
        # group_messages_by_tool_call: 5 standalone groups.
        groups = group_messages_by_tool_call(msgs)
        assert len(groups) == 5, f"setup error: got {len(groups)} groups"

        result = await cm._summarize_compact("system", msgs)

        # No two consecutive user/assistant pairs in the output.
        for i in range(len(result) - 1):
            r_a = result[i].get("role")
            r_b = result[i + 1].get("role")
            if r_a in ("user", "assistant") and r_b in ("user", "assistant"):
                assert r_a != r_b, (
                    f"consecutive {r_a} messages at index {i},{i+1} — "
                    f"breaks LLM API alternation. result: {result}"
                )

    @pytest.mark.asyncio
    async def test_summarize_all_orphan_tools_returns_summary_only(self):
        """Codex r2 P1: when every retained group is an orphan ``tool``
        (pathological input — every message is a stray tool with no
        parent assistant), the dedup loop strips all but the last
        group. That last group still leads with ``tool``, which can't
        be bridged (a tool needs a matching parent assistant, not just
        any assistant). Drop the tail entirely; the summary already
        captured everything older. Returning ``[summary]`` is strictly
        better than ``[summary, tool, ...]`` (API-invalid)."""
        llm = MagicMock()
        llm.chat = AsyncMock(return_value=LLMResponse(
            content="Summary.", tokens_used=10,
        ))
        cm = ContextManager(max_tokens=100, llm=llm, workspace=None)

        # 5 standalone orphan-tool groups + nothing else.
        msgs = [
            {"role": "tool", "tool_call_id": f"orph_{i}", "content": f"r{i}"}
            for i in range(5)
        ]
        result = await cm._summarize_compact("system", msgs)
        # Result is just the summary; no orphan tools leaked out.
        assert len(result) == 1
        assert result[0].get("role") == "user"
        assert "Summary" in result[0]["content"]

    @pytest.mark.asyncio
    async def test_summarize_unknown_role_drops_tail(self):
        """Codex r2 P2: unknown roles (``developer``, future provider-
        native roles, malformed payloads) must NOT pass through to
        the LLM API. The original code's ``else`` branch accepted
        them implicitly; the explicit allowlist (``assistant`` /
        ``system``) drops them and returns summary-only."""
        llm = MagicMock()
        llm.chat = AsyncMock(return_value=LLMResponse(
            content="Summary.", tokens_used=10,
        ))
        cm = ContextManager(max_tokens=100, llm=llm, workspace=None)

        # 2 groups: [user_only], [developer_message]. After dedup,
        # recent[0].role is "developer" — drops to summary-only.
        msgs = [
            {"role": "user", "content": "kickoff"},
            {"role": "developer", "content": "weird leaked role"},
        ]
        result = await cm._summarize_compact("system", msgs)
        # Tail dropped, only summary returned.
        assert len(result) == 1
        assert result[0].get("role") == "user"
        # The unknown-role message must NOT have leaked into the result.
        assert not any(
            m.get("role") == "developer" for m in result
        )

    @pytest.mark.asyncio
    async def test_large_multi_tool_group_stays_atomic(self):
        """A single assistant turn with 10 tool calls forms one
        12-message group (1 + 1 + 10). The group must NOT be split by
        a 4-message tail boundary. With only 2 groups total, the
        recent kept group is the whole tool turn — atomicity wins
        over shrinkage."""
        llm = MagicMock()
        llm.chat = AsyncMock(return_value=LLMResponse(
            content="Summary.", tokens_used=10,
        ))
        cm = ContextManager(max_tokens=100, llm=llm, workspace=None)

        tool_calls = [
            {"id": f"t{i}", "function": {"name": "x", "arguments": "{}"}}
            for i in range(10)
        ]
        msgs = [
            {"role": "user", "content": "kick off 10 parallel things"},
            {"role": "assistant", "tool_calls": tool_calls},
        ] + [
            {"role": "tool", "tool_call_id": f"t{i}", "content": f"r{i}"}
            for i in range(10)
        ]

        result = await cm._summarize_compact("system", msgs)
        # The 10-tool group must remain intact in the result. The
        # summary lands at index 0; the assistant_with_10_tools is
        # immediately followed by all 10 tool results.
        asst_idx = next(
            i for i, m in enumerate(result)
            if m.get("role") == "assistant" and m.get("tool_calls")
        )
        tool_ids_after = [
            m["tool_call_id"]
            for m in result[asst_idx + 1:]
            if m.get("role") == "tool"
        ]
        expected_ids = {f"t{i}" for i in range(10)}
        assert set(tool_ids_after) == expected_ids, (
            f"10-tool group split by compaction — missing tool ids: "
            f"{expected_ids - set(tool_ids_after)}"
        )


def _tool_group_messages(num_groups: int, chars_each: int = 4000) -> list[dict]:
    """Build a message list with ``num_groups`` tool-call groups.

    Shape: an initial user message, then repeating
    ``assistant(tool_calls) -> tool(result) -> assistant(text) -> user`` so the
    grouping helper produces clearly-bounded groups that can be pruned at
    boundaries without orphaning a tool result.
    """
    msgs: list[dict] = [{"role": "user", "content": "INITIAL " + "i" * chars_each}]
    for g in range(num_groups):
        msgs.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": f"c{g}", "function": {"name": "do", "arguments": "{}"}}],
        })
        msgs.append({"role": "tool", "tool_call_id": f"c{g}", "content": "R" + "r" * chars_each})
        msgs.append({"role": "assistant", "content": "A" + "a" * chars_each})
        msgs.append({"role": "user", "content": "U" + "u" * chars_each})
    return msgs


class TestEstimateRequestTokens:
    def test_includes_system_and_tools(self):
        cm = ContextManager(max_tokens=100_000, model="anthropic/claude")
        msgs = _make_messages(4, chars_each=400)
        msgs_only = cm.token_count(msgs)
        with_sys = cm.estimate_request_tokens(msgs, system_prompt="S" * 4000)
        tools = [{"type": "function", "function": {"name": "x", "description": "d" * 4000}}]
        with_all = cm.estimate_request_tokens(msgs, system_prompt="S" * 4000, tools=tools)
        assert with_sys > msgs_only, "system prompt must add to the estimate"
        assert with_all > with_sys, "tool schema must add to the estimate"

    def test_no_system_no_tools_equals_messages(self):
        cm = ContextManager(max_tokens=100_000)
        msgs = _make_messages(4, chars_each=400)
        assert cm.estimate_request_tokens(msgs) == cm.token_count(msgs)


class TestPruneToFit:
    def test_reduces_over_limit_list_below_ceiling(self):
        # Window small enough that several groups blow past 0.9 * max_tokens.
        cm = ContextManager(max_tokens=8_000, model="anthropic/claude")
        msgs = _tool_group_messages(8, chars_each=4000)
        assert cm.estimate_request_tokens(msgs) > cm.max_tokens * 0.90
        pruned = cm.prune_to_fit(msgs)
        assert cm.estimate_request_tokens(pruned) <= cm.max_tokens * 0.90

    def test_keeps_first_message(self):
        cm = ContextManager(max_tokens=8_000, model="anthropic/claude")
        msgs = _tool_group_messages(8, chars_each=4000)
        pruned = cm.prune_to_fit(msgs)
        assert pruned[0].get("content", "").startswith("INITIAL"), \
            "first (initial-context) message must always be kept"

    def test_preserves_role_alternation(self):
        cm = ContextManager(max_tokens=8_000, model="anthropic/claude")
        msgs = _tool_group_messages(8, chars_each=4000)
        pruned = cm.prune_to_fit(msgs)
        for a, b in zip(pruned, pruned[1:]):
            ra, rb = a.get("role"), b.get("role")
            assert not (ra == rb == "user"), "consecutive user messages"
            assert not (ra == rb == "assistant"), "consecutive assistant messages"

    def test_never_orphans_a_tool_group(self):
        cm = ContextManager(max_tokens=8_000, model="anthropic/claude")
        msgs = _tool_group_messages(8, chars_each=4000)
        pruned = cm.prune_to_fit(msgs)
        # Every tool message must be immediately preceded (within its group) by
        # an assistant carrying tool_calls — i.e. no orphaned tool results.
        groups = group_messages_by_tool_call(pruned)
        for g in groups:
            if any(m.get("role") == "tool" for m in g):
                assert g[0].get("role") == "assistant" and g[0].get("tool_calls"), \
                    "tool result orphaned from its parent assistant"

    def test_noop_when_under_ceiling(self):
        cm = ContextManager(max_tokens=1_000_000, model="anthropic/claude")
        msgs = _make_messages(4, chars_each=400)
        assert cm.prune_to_fit(msgs) is msgs

    def test_aggressive_ceiling_prunes_more(self):
        cm = ContextManager(max_tokens=8_000, model="anthropic/claude")
        msgs = _tool_group_messages(8, chars_each=4000)
        lenient = cm.prune_to_fit(msgs, ceiling_frac=0.90)
        aggressive = cm.prune_to_fit(msgs, ceiling_frac=0.50)
        assert len(aggressive) <= len(lenient)
        assert cm.estimate_request_tokens(aggressive) <= cm.max_tokens * 0.50

    def test_target_tokens_overrides_frac_ceiling(self):
        # An absolute target forces pruning below it even though the default
        # frac-ceiling would no-op. This is the forced-progress lever the
        # self-heal uses when the estimate (uncalibrated) thinks we fit.
        cm = ContextManager(max_tokens=1_000_000, model="anthropic/claude")
        msgs = _tool_group_messages(8, chars_each=4000)
        # Well under 0.90 * 1M, so the default ceiling is a no-op...
        assert cm.prune_to_fit(msgs) is msgs
        before = cm.estimate_request_tokens(msgs)
        # ...but an absolute target below `before` must shed real content.
        pruned = cm.prune_to_fit(msgs, target_tokens=int(before * 0.5))
        assert len(pruned) < len(msgs)
        assert cm.estimate_request_tokens(pruned) <= before * 0.5


class TestEstimateCalibration:
    """Regression for the map-agent wedge (2026-06-24): the chars/token estimate
    read ~2x under the real tokenizer on dense CSV/JSON content (504,884 est vs
    1,001,150 actual), so the 0.75 emergency ceiling on a 1M window saw 504K,
    pruned NOTHING, and every retry 400'd identically. Calibrating from the
    provider-reported count fixes both the no-op prune and future turns."""

    def test_correction_defaults_to_identity(self):
        cm = ContextManager(max_tokens=1_000_000, model="anthropic/claude")
        assert cm._estimate_correction == 1.0
        msgs = _make_messages(4, chars_each=400)
        assert cm.estimate_request_tokens(msgs) == cm._raw_request_tokens(msgs)

    def test_calibrate_ratchets_correction_and_scales_estimate(self):
        cm = ContextManager(max_tokens=1_000_000, model="anthropic/claude")
        msgs = _make_messages(6, chars_each=4000)
        raw = cm._raw_request_tokens(msgs)
        # Provider says the request was actually ~2x our raw estimate.
        cm.calibrate_from_overflow(actual_tokens=raw * 2, messages=msgs)
        assert cm._estimate_correction == pytest.approx(2.0, abs=0.05)
        assert cm.estimate_request_tokens(msgs) == pytest.approx(raw * 2, rel=0.05)

    def test_correction_only_ratchets_up(self):
        cm = ContextManager(max_tokens=1_000_000, model="anthropic/claude")
        msgs = _make_messages(6, chars_each=4000)
        raw = cm._raw_request_tokens(msgs)
        cm.calibrate_from_overflow(actual_tokens=raw * 2, messages=msgs)
        # A later, smaller ratio must NOT lower the learned correction.
        cm.calibrate_from_overflow(actual_tokens=int(raw * 1.2), messages=msgs)
        assert cm._estimate_correction == pytest.approx(2.0, abs=0.05)

    def test_correction_is_clamped(self):
        from src.agent.context import _MAX_ESTIMATE_CORRECTION
        cm = ContextManager(max_tokens=1_000_000, model="anthropic/claude")
        msgs = _make_messages(6, chars_each=4000)
        raw = cm._raw_request_tokens(msgs)
        cm.calibrate_from_overflow(actual_tokens=raw * 100, messages=msgs)
        assert cm._estimate_correction == _MAX_ESTIMATE_CORRECTION

    def test_calibration_turns_a_noop_prune_into_a_real_one(self):
        # The exact cake shape: raw estimate sits UNDER the 0.75 ceiling so the
        # emergency prune would no-op — until calibration scales it over.
        cm = ContextManager(max_tokens=100_000, model="anthropic/claude")
        # ~50K raw est: comfortably under the 0.75 ceiling (so the default prune
        # no-ops) yet over 0.375*max, so doubling it via calibration clears the
        # ceiling and forces a real prune.
        msgs = _tool_group_messages(14, chars_each=4000)
        raw = cm._raw_request_tokens(msgs)
        assert raw <= cm.max_tokens * 0.75, "precondition: default prune would no-op"
        assert cm.prune_to_fit(msgs, ceiling_frac=0.75) is msgs
        # API reports the request was actually ~2x — calibrate, then re-prune.
        cm.calibrate_from_overflow(actual_tokens=raw * 2, messages=msgs)
        assert cm.estimate_request_tokens(msgs) > cm.max_tokens * 0.75
        pruned = cm.prune_to_fit(msgs, ceiling_frac=0.75)
        assert len(pruned) < len(msgs), "after calibration the prune must drop groups"
        assert cm.estimate_request_tokens(pruned) <= cm.max_tokens * 0.75


class TestSummarizeTailCap:
    @pytest.mark.asyncio
    async def test_tail_cap_drops_oversized_kept_groups(self):
        # The retained recent tail (default keep_n up to 4 groups) is capped at
        # ~0.5 * max_tokens. With huge groups, the tail must shed older kept
        # groups so it stays under that cap.
        llm = MagicMock()
        llm.chat = AsyncMock(return_value=LLMResponse(content="SUMMARY", tokens_used=10))
        cm = ContextManager(max_tokens=10_000, llm=llm, model="anthropic/claude")
        # 6 standalone assistant/user groups, each ~big.
        msgs = _tool_group_messages(6, chars_each=4000)
        result = await cm._summarize_compact("system", msgs)
        # Result = summary + retained tail; tail (excluding the summary msg)
        # must be under 0.5 * max_tokens.
        tail_tokens = estimate_tokens(result[1:], cm.model)
        assert tail_tokens <= cm.max_tokens * 0.5 + 50, \
            f"retained tail {tail_tokens} exceeds 0.5*max_tokens cap"


class TestChunkedSummarization:
    """B1: compaction summarizes ALL older history in chunks instead of
    truncating to the first 20k chars (and 200 chars per tool result)."""

    @staticmethod
    def _recording_llm(fold_content: str = "FOLDED", part_content: str = "part summary"):
        llm = MagicMock()
        prompts: list[str] = []

        async def chat(system, messages, max_tokens=None, temperature=None, **kw):
            prompt = messages[0]["content"]
            prompts.append(prompt)
            if prompt.startswith("Merge these sequential summaries"):
                return LLMResponse(content=fold_content, tokens_used=10)
            return LLMResponse(content=part_content, tokens_used=10)

        llm.chat = AsyncMock(side_effect=chat)
        return llm, prompts

    @pytest.mark.asyncio
    async def test_content_beyond_20k_reaches_summarizer(self):
        llm, prompts = self._recording_llm()
        cm = ContextManager(max_tokens=100, llm=llm, workspace=None)
        msgs = []
        for i in range(30):
            role = "user" if i % 2 == 0 else "assistant"
            msgs.append({"role": role, "content": f"MARKER_{i} " + "x" * 2000})

        result = await cm._summarize_compact("system", msgs)

        joined = "\n".join(prompts)
        # Multiple chunk calls were made...
        assert len(prompts) >= 2
        # ...and content past the legacy 20k truncation point survived.
        assert "MARKER_20" in joined
        assert "MARKER_25" in joined
        # The stitched summary carries part headers.
        summary_msg = result[0]["content"]
        assert "Part 1 of" in summary_msg

    @pytest.mark.asyncio
    async def test_tool_results_not_over_truncated(self):
        llm, prompts = self._recording_llm()
        cm = ContextManager(max_tokens=100, llm=llm, workspace=None)
        msgs = [
            {"role": "user", "content": "do deep research"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "t1", "function": {"name": "web_search", "arguments": "{}"}}],
            },
            # Marker sits past the legacy 200-char tool-result cut.
            {"role": "tool", "tool_call_id": "t1", "content": "x" * 1000 + " TOOLMARKER " + "y" * 200},
        ]
        # Trailing small groups so the tool group lands in "older".
        for i in range(8):
            role = "user" if i % 2 == 0 else "assistant"
            msgs.append({"role": role, "content": f"tail {i}"})

        await cm._summarize_compact("system", msgs)

        assert "TOOLMARKER" in "\n".join(prompts)

    @pytest.mark.asyncio
    async def test_long_partials_get_folded_once(self):
        llm, prompts = self._recording_llm(part_content="p" * 5000)
        cm = ContextManager(max_tokens=100, llm=llm, workspace=None)
        msgs = []
        for i in range(30):
            role = "user" if i % 2 == 0 else "assistant"
            msgs.append({"role": role, "content": f"m{i} " + "x" * 2000})

        result = await cm._summarize_compact("system", msgs)

        fold_calls = [p for p in prompts if p.startswith("Merge these sequential summaries")]
        assert len(fold_calls) == 1
        assert "FOLDED" in result[0]["content"]

    @pytest.mark.asyncio
    async def test_single_chunk_keeps_single_call(self):
        llm, prompts = self._recording_llm()
        cm = ContextManager(max_tokens=100, llm=llm, workspace=None)
        msgs = _make_messages(10, chars_each=200)

        result = await cm._summarize_compact("system", msgs)

        assert len(prompts) == 1
        assert "part summary" in result[0]["content"]

    def test_chunk_message_texts_boundaries(self):
        msgs = [{"role": "user", "content": f"msg{i} " + "z" * 40} for i in range(10)]
        chunks = ContextManager._chunk_message_texts(msgs, chunk_size=120, tool_chars=200)
        assert len(chunks) > 1
        assert all(len(c) <= 120 for c in chunks)
        # Every message's marker survives across the chunk set.
        joined = "".join(chunks)
        for i in range(10):
            assert f"msg{i}" in joined

    def test_chunk_message_texts_truncates_oversized_message(self):
        msgs = [{"role": "user", "content": "B" * 500}]
        chunks = ContextManager._chunk_message_texts(msgs, chunk_size=100, tool_chars=200)
        assert len(chunks) == 1
        assert len(chunks[0]) <= 100
