"""Tests for agent chat mode and chat endpoints."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.agent.loop import AgentLoop
from src.agent.server import create_agent_app
from src.shared.types import LLMResponse, ToolCallInfo


def _make_loop(llm_responses: list[LLMResponse] | None = None) -> AgentLoop:
    """Create an AgentLoop with mock dependencies."""
    memory = MagicMock()
    memory.get_high_salience_facts = MagicMock(return_value=[])
    memory.search = AsyncMock(return_value=[])
    memory.log_action = AsyncMock()
    memory._run_db = AsyncMock(return_value=None)

    skills = MagicMock()
    skills.get_tool_definitions = MagicMock(return_value=[])
    skills.get_descriptions = MagicMock(return_value="- no tools")
    skills.list_skills = MagicMock(return_value=[])
    skills.is_parallel_safe = MagicMock(return_value=True)
    skills.get_loop_exempt_tools = MagicMock(return_value=frozenset())

    llm = MagicMock()
    if llm_responses:
        llm.chat = AsyncMock(side_effect=llm_responses)
    else:
        llm.chat = AsyncMock(return_value=LLMResponse(content="Hello!", tokens_used=50))
    llm.default_model = "test-model"

    mesh_client = MagicMock()
    mesh_client.send_system_message = AsyncMock(return_value={})

    return AgentLoop(
        agent_id="test_agent",
        role="assistant",
        memory=memory,
        skills=skills,
        llm=llm,
        mesh_client=mesh_client,
    )


# ── AgentLoop.chat() tests ───────────────────────────────────


class TestChatMode:
    @pytest.mark.asyncio
    async def test_simple_chat(self):
        loop = _make_loop()
        result = await loop.chat("Hi there")
        assert result["response"] == "Hello!"
        assert result["tokens_used"] == 50
        assert loop.state == "idle"

    @pytest.mark.asyncio
    async def test_chat_preserves_history(self):
        loop = _make_loop([
            LLMResponse(content="First reply", tokens_used=30),
            LLMResponse(content="Second reply", tokens_used=40),
        ])
        await loop.chat("Hello")
        await loop.chat("Again")

        # History: user, assistant, user, assistant = 4 messages
        assert len(loop._chat_messages) == 4
        assert loop._chat_messages[0]["role"] == "user"
        assert loop._chat_messages[1]["role"] == "assistant"
        assert loop._chat_messages[2]["role"] == "user"
        assert loop._chat_messages[3]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_chat_with_tool_calls(self):
        tool_response = LLMResponse(
            content="",
            tool_calls=[ToolCallInfo(name="exec", arguments={"command": "ls"})],
            tokens_used=30,
        )
        final_response = LLMResponse(content="Here are your files", tokens_used=20)

        loop = _make_loop([tool_response, final_response])
        loop.skills.execute = AsyncMock(return_value={"exit_code": 0, "stdout": "file.txt"})
        loop.skills.get_tool_definitions = MagicMock(
            return_value=[{"type": "function", "function": {"name": "exec"}}]
        )

        result = await loop.chat("List files")
        assert result["response"] == "Here are your files"
        assert len(result["tool_outputs"]) == 1
        assert result["tool_outputs"][0]["tool"] == "exec"
        assert result["tokens_used"] == 50

    @pytest.mark.asyncio
    async def test_chat_reset(self):
        loop = _make_loop()
        await loop.chat("Hello")
        assert len(loop._chat_messages) > 0
        await loop.reset_chat()
        assert len(loop._chat_messages) == 0

    @pytest.mark.asyncio
    async def test_chat_reset_flushes_memory(self):
        """reset_chat flushes to memory before clearing when context_manager exists."""
        import shutil
        import tempfile

        from src.agent.context import ContextManager
        from src.agent.workspace import WorkspaceManager

        tmpdir = tempfile.mkdtemp()
        try:
            workspace = WorkspaceManager(workspace_dir=tmpdir)
            flush_llm = MagicMock()
            flush_llm.chat = AsyncMock(
                return_value=LLMResponse(
                    content='[{"key": "chat_fact", "value": "from chat", "category": "fact"}]',
                    tokens_used=30,
                )
            )
            cm = ContextManager(max_tokens=200_000, llm=flush_llm, workspace=workspace)

            loop = _make_loop()
            loop.context_manager = cm
            # Build enough conversation for flush to consider it worth extracting
            long_msg = "Please help me set up my ML pipeline. " * 10
            await loop.chat(long_msg)
            assert len(loop._chat_messages) > 0

            await loop.reset_chat()
            assert len(loop._chat_messages) == 0
            # Flush should have been called
            assert flush_llm.chat.call_count == 1
            memory_content = workspace.load_memory()
            assert "chat_fact" in memory_content
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_chat_queues_while_locked(self):
        """Concurrent chat calls queue via lock instead of being rejected."""
        import asyncio

        loop = _make_loop()
        results = []

        async def delayed_chat(msg: str) -> dict:
            r = await loop.chat(msg)
            results.append(r)
            return r

        r1, r2 = await asyncio.gather(
            delayed_chat("First"),
            delayed_chat("Second"),
        )
        assert r1["response"] == "Hello!"
        assert r2["response"] == "Hello!"
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_chat_error_recovery(self):
        loop = _make_loop()
        loop.llm.chat = AsyncMock(side_effect=RuntimeError("LLM down"))
        result = await loop.chat("Hello")
        assert "Error" in result["response"]
        assert loop.state == "idle"


# ── Server chat endpoints ────────────────────────────────────


class TestChatEndpoints:
    def test_post_chat(self):
        loop = _make_loop()
        app = create_agent_app(loop)
        client = TestClient(app)

        resp = client.post("/chat", json={"message": "Hello"})
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        assert data["response"] == "Hello!"

    def test_post_chat_reset(self):
        loop = _make_loop()
        app = create_agent_app(loop)
        client = TestClient(app)

        client.post("/chat", json={"message": "Hello"})
        resp = client.post("/chat/reset")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert len(loop._chat_messages) == 0

    def test_post_chat_steer_returns_200(self):
        """POST /chat/steer returns 200 with injected key."""
        loop = _make_loop()
        app = create_agent_app(loop)
        client = TestClient(app)

        resp = client.post("/chat/steer", json={"message": "redirect"})
        assert resp.status_code == 200
        data = resp.json()
        assert "injected" in data
        assert "agent_state" in data

    def test_chat_steer_does_not_block(self):
        """Steer endpoint returns immediately without acquiring _chat_lock."""
        import asyncio

        loop = _make_loop()
        app = create_agent_app(loop)
        client = TestClient(app)

        # Manually acquire the chat lock to simulate a busy agent
        acquired = asyncio.get_event_loop().run_until_complete(loop._chat_lock.acquire())
        assert acquired

        try:
            # Steer should still work since it bypasses _chat_lock
            resp = client.post("/chat/steer", json={"message": "urgent"})
            assert resp.status_code == 200
            assert resp.json()["injected"] is False  # idle state
        finally:
            loop._chat_lock.release()

    def test_status_returns_context_fields(self):
        """GET /status should include context_tokens, context_max, context_pct."""
        from src.agent.context import ContextManager

        loop = _make_loop()
        loop.context_manager = ContextManager(max_tokens=50_000)
        app = create_agent_app(loop)
        client = TestClient(app)

        # Send a chat message to populate _chat_messages
        client.post("/chat", json={"message": "Hello"})

        resp = client.get("/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "context_tokens" in data
        assert "context_max" in data
        assert "context_pct" in data
        assert data["context_max"] == 50_000
        assert data["context_tokens"] >= 0


# ── Auto-continue session tests ──────────────────────────────


class TestAutoContinueSession:
    @pytest.mark.asyncio
    async def test_auto_continue_resets_round_counter(self):
        """Hitting CHAT_MAX_TOTAL_ROUNDS triggers auto-continue instead of error."""
        loop = _make_loop()
        loop._chat_total_rounds = loop.CHAT_MAX_TOTAL_ROUNDS
        loop._chat_messages = [
            {"role": "user", "content": "msg"},
            {"role": "assistant", "content": "reply"},
        ]

        result = await loop.chat("Continue working")
        # Should get a normal response, not the old error message
        assert "absolute limit" not in result["response"]
        assert result["response"] == "Hello!"
        # Round counter should have been reset
        assert loop._chat_total_rounds == 0
        assert loop._chat_auto_continues == 1

    @pytest.mark.asyncio
    async def test_auto_continue_with_context_manager(self):
        """Auto-continue calls force_compact on the context manager."""
        from src.agent.context import ContextManager

        loop = _make_loop()
        cm = MagicMock(spec=ContextManager)
        cm.force_compact = AsyncMock(return_value=[
            {"role": "user", "content": "## Conversation Summary\n\nSummary here"},
        ])
        cm.maybe_compact = AsyncMock(side_effect=lambda s, m: (m, False))
        cm.context_warning = MagicMock(return_value=None)
        loop.context_manager = cm

        loop._chat_total_rounds = loop.CHAT_MAX_TOTAL_ROUNDS
        loop._chat_messages = [
            {"role": "user", "content": "msg"},
            {"role": "assistant", "content": "reply"},
        ]

        result = await loop.chat("Continue")
        assert result["response"] == "Hello!"
        cm.force_compact.assert_called_once()
        assert loop._chat_auto_continues == 1
        assert loop._chat_total_rounds == 0

    @pytest.mark.asyncio
    async def test_absolute_limit_after_max_continues(self):
        """After _MAX_SESSION_CONTINUES, session returns an error."""
        loop = _make_loop()
        loop._chat_total_rounds = loop.CHAT_MAX_TOTAL_ROUNDS
        loop._chat_auto_continues = loop._MAX_SESSION_CONTINUES

        result = await loop.chat("Keep going")
        assert "absolute limit" in result["response"]
        assert loop.state == "idle"

    @pytest.mark.asyncio
    async def test_reset_clears_auto_continues(self):
        """reset_chat resets the auto-continue counter."""
        loop = _make_loop()
        loop._chat_auto_continues = 3
        loop._chat_total_rounds = 150
        await loop.reset_chat()
        assert loop._chat_auto_continues == 0
        assert loop._chat_total_rounds == 0

    @pytest.mark.asyncio
    async def test_round_warning_in_system_prompt(self):
        """System prompt includes session note at 80% of round limit."""
        loop = _make_loop()
        loop._chat_total_rounds = loop._CHAT_ROUND_WARNING

        prompt = loop._build_chat_system_prompt()
        assert "Session Note" in prompt
        assert "auto-refreshed" in prompt

    @pytest.mark.asyncio
    async def test_no_warning_below_threshold(self):
        """No session note below the warning threshold."""
        loop = _make_loop()
        loop._chat_total_rounds = loop._CHAT_ROUND_WARNING - 1

        prompt = loop._build_chat_system_prompt()
        assert "Session Note" not in prompt

    @pytest.mark.asyncio
    async def test_auto_continue_in_tool_loop(self):
        """Auto-continue triggers mid-tool-loop without breaking the session."""
        tool_response = LLMResponse(
            content="",
            tool_calls=[ToolCallInfo(name="exec", arguments={"command": "ls"})],
            tokens_used=10,
        )
        final_response = LLMResponse(content="Done", tokens_used=10)

        loop = _make_loop([tool_response, final_response])
        loop.skills.execute = AsyncMock(return_value={"result": "ok"})
        loop.skills.get_tool_definitions = MagicMock(
            return_value=[{"type": "function", "function": {"name": "exec"}}]
        )
        # Set rounds to just below the limit so the first tool round triggers it
        loop._chat_total_rounds = loop.CHAT_MAX_TOTAL_ROUNDS - 1

        result = await loop.chat("Do something")
        assert result["response"] == "Done"
        # Should have auto-continued: counter reset after hitting limit
        assert loop._chat_auto_continues == 1
        assert loop._chat_total_rounds == 0

    @pytest.mark.asyncio
    async def test_auto_continue_survives_compaction_failure(self):
        """Round counter resets even when force_compact throws."""
        from src.agent.context import ContextManager

        loop = _make_loop()
        cm = MagicMock(spec=ContextManager)
        cm.force_compact = AsyncMock(side_effect=RuntimeError("LLM down"))
        cm.maybe_compact = AsyncMock(side_effect=lambda s, m: (m, False))
        cm.context_warning = MagicMock(return_value=None)
        loop.context_manager = cm

        loop._chat_total_rounds = loop.CHAT_MAX_TOTAL_ROUNDS
        loop._chat_messages = [
            {"role": "user", "content": "msg"},
            {"role": "assistant", "content": "reply"},
        ]

        result = await loop.chat("Continue despite failure")
        # Should still work — fell back to trim
        assert result["response"] == "Hello!"
        assert loop._chat_auto_continues == 1
        # Counter must reset even though compaction failed
        assert loop._chat_total_rounds == 0

    @pytest.mark.asyncio
    async def test_auto_continue_streaming_entry(self):
        """Streaming chat auto-continues at the round limit."""
        loop = _make_loop()

        async def _failing_stream(**kwargs):
            raise RuntimeError("no stream")
            yield  # noqa: unreachable — makes this an async generator

        loop.llm.chat_stream = _failing_stream
        loop._chat_total_rounds = loop.CHAT_MAX_TOTAL_ROUNDS
        loop._chat_messages = [
            {"role": "user", "content": "msg"},
            {"role": "assistant", "content": "reply"},
        ]

        events = []
        async for event in loop.chat_stream("Continue streaming"):
            events.append(event)

        done_events = [e for e in events if e.get("type") == "done"]
        assert len(done_events) == 1
        # Should get a normal response, not the absolute limit error
        assert "absolute limit" not in done_events[0]["response"]
        assert loop._chat_auto_continues == 1
        assert loop._chat_total_rounds == 0

    @pytest.mark.asyncio
    async def test_streaming_absolute_limit(self):
        """Streaming chat returns error at absolute limit."""
        loop = _make_loop()
        loop._chat_total_rounds = loop.CHAT_MAX_TOTAL_ROUNDS
        loop._chat_auto_continues = loop._MAX_SESSION_CONTINUES

        events = []
        async for event in loop.chat_stream("Keep going"):
            events.append(event)

        text_events = [e for e in events if e.get("type") == "text_delta"]
        assert any("absolute limit" in e["content"] for e in text_events)

    @pytest.mark.asyncio
    async def test_auto_continue_writes_system_separator(self):
        """_auto_continue_session writes a system message to the transcript."""
        import shutil
        import tempfile

        from src.agent.workspace import WorkspaceManager

        tmpdir = tempfile.mkdtemp()
        try:
            loop = _make_loop()
            loop.workspace = WorkspaceManager(workspace_dir=tmpdir)
            loop._chat_total_rounds = loop.CHAT_MAX_TOTAL_ROUNDS
            loop._chat_messages = [
                {"role": "user", "content": "msg"},
                {"role": "assistant", "content": "reply"},
            ]

            await loop.chat("Continue")

            transcript = loop.workspace.load_chat_transcript()
            system_msgs = [m for m in transcript if m.get("role") == "system"]
            assert len(system_msgs) == 1
            assert "Session continued" in system_msgs[0]["content"]
            assert str(loop.CHAT_MAX_TOTAL_ROUNDS) in system_msgs[0]["content"]
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestToolLimitReached:
    @pytest.mark.asyncio
    async def test_tool_limit_sets_flag_in_response(self):
        """When CHAT_MAX_TOOL_ROUNDS is exhausted, response has tool_limit_reached=True."""
        # Each call uses a unique arg so the loop detector doesn't fire first.
        responses = [
            LLMResponse(
                content="",
                tool_calls=[ToolCallInfo(name="exec", arguments={"command": f"step_{i}"})],
                tokens_used=10,
            )
            for i in range(AgentLoop.CHAT_MAX_TOOL_ROUNDS)
        ] + [LLMResponse(content="I've done what I can.", tokens_used=10)]

        loop = _make_loop(responses)
        loop.skills.execute = AsyncMock(return_value={"result": "ok"})
        loop.skills.get_tool_definitions = MagicMock(
            return_value=[{"type": "function", "function": {"name": "exec"}}]
        )

        result = await loop.chat("Do a long task")
        assert result.get("tool_limit_reached") is True
        assert result["response"] == "I've done what I can."

    @pytest.mark.asyncio
    async def test_normal_response_has_no_tool_limit_flag(self):
        """Normal responses do not set tool_limit_reached."""
        loop = _make_loop()
        result = await loop.chat("Hello")
        assert result.get("tool_limit_reached") is not True

    @pytest.mark.asyncio
    async def test_tool_limit_streaming_sets_flag(self):
        """Streaming path also emits tool_limit_reached=True on the done event."""
        responses = [
            LLMResponse(
                content="",
                tool_calls=[ToolCallInfo(name="exec", arguments={"command": f"step_{i}"})],
                tokens_used=10,
            )
            for i in range(AgentLoop.CHAT_MAX_TOOL_ROUNDS)
        ] + [LLMResponse(content="Done.", tokens_used=10)]

        loop = _make_loop(responses)

        async def _no_stream(**kwargs):
            raise RuntimeError("no streaming")
            yield  # noqa: makes it an async generator

        loop.llm.chat_stream = _no_stream
        loop.skills.execute = AsyncMock(return_value={"result": "ok"})
        loop.skills.get_tool_definitions = MagicMock(
            return_value=[{"type": "function", "function": {"name": "exec"}}]
        )

        events = []
        async for event in loop.chat_stream("Do a long task"):
            events.append(event)

        done_events = [e for e in events if e.get("type") == "done"]
        assert len(done_events) == 1
        assert done_events[0].get("tool_limit_reached") is True


class TestStreamingThinkingContent:
    """Verify chat_stream handles thinking-only LLM responses."""

    @pytest.mark.asyncio
    async def test_thinking_only_response_via_streaming(self):
        """When LLM returns only thinking_content, it becomes the response."""
        thinking = "<think>Let me reason about this</think>\nThe answer is 42."
        llm_resp = LLMResponse(
            content="", thinking_content=thinking, tokens_used=50,
        )

        loop = _make_loop()

        async def _stream_with_thinking(**kwargs):
            yield {"type": "done", "response": llm_resp}

        loop.llm.chat_stream = _stream_with_thinking

        events = []
        async for event in loop.chat_stream("What is the answer?"):
            events.append(event)

        done_events = [e for e in events if e.get("type") == "done"]
        assert len(done_events) == 1
        # <think> tags should be stripped; answer extracted
        assert done_events[0]["response"] == "The answer is 42."
        # Conversation history should also have clean content
        assistant_msgs = [
            m for m in loop._chat_messages if m.get("role") == "assistant"
        ]
        assert assistant_msgs[-1]["content"] == "The answer is 42."

    @pytest.mark.asyncio
    async def test_thinking_only_nonstreaming_fallback(self):
        """Non-streaming fallback also handles thinking-only responses."""
        thinking = "<think>reasoning</think>\nClean answer."
        loop = _make_loop([
            LLMResponse(content="", thinking_content=thinking, tokens_used=50),
        ])

        result = await loop.chat("Question?")
        assert result["response"] == "Clean answer."


class TestCompactionSystemMessage:
    @pytest.mark.asyncio
    async def test_compaction_writes_system_message(self):
        """_compact_chat_context writes a system message when compaction fires."""
        import shutil
        import tempfile

        from src.agent.context import ContextManager
        from src.agent.workspace import WorkspaceManager

        tmpdir = tempfile.mkdtemp()
        try:
            workspace = WorkspaceManager(workspace_dir=tmpdir)
            tool_response = LLMResponse(
                content="",
                tool_calls=[ToolCallInfo(name="exec", arguments={"command": "ls"})],
                tokens_used=10,
            )
            final_response = LLMResponse(content="Done.", tokens_used=10)

            loop = _make_loop([tool_response, final_response])
            loop.workspace = workspace
            loop.skills.execute = AsyncMock(return_value={"result": "ok"})
            loop.skills.get_tool_definitions = MagicMock(
                return_value=[{"type": "function", "function": {"name": "exec"}}]
            )

            # Patch maybe_compact to signal compaction fired
            cm = MagicMock(spec=ContextManager)
            compacted_msgs = [{"role": "user", "content": "## Conversation Summary\n\nSummary"}]
            cm.maybe_compact = AsyncMock(return_value=(compacted_msgs, True))
            cm.context_warning = MagicMock(return_value=None)
            loop.context_manager = cm

            await loop.chat("Trigger compaction")

            transcript = workspace.load_chat_transcript()
            system_msgs = [m for m in transcript if m.get("role") == "system"]
            assert len(system_msgs) == 1
            assert "Context compacted" in system_msgs[0]["content"]
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_no_system_message_when_no_compaction(self):
        """No system message is written when compaction does not fire."""
        import shutil
        import tempfile

        from src.agent.context import ContextManager
        from src.agent.workspace import WorkspaceManager

        tmpdir = tempfile.mkdtemp()
        try:
            workspace = WorkspaceManager(workspace_dir=tmpdir)
            loop = _make_loop()
            loop.workspace = workspace

            cm = MagicMock(spec=ContextManager)
            cm.maybe_compact = AsyncMock(
                side_effect=lambda s, m: (m, False)
            )
            cm.context_warning = MagicMock(return_value=None)
            loop.context_manager = cm

            await loop.chat("Short message, no compaction")

            transcript = workspace.load_chat_transcript()
            system_msgs = [m for m in transcript if m.get("role") == "system"]
            assert len(system_msgs) == 0
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ── _strip_think_tags, _extract_json_response & _resolve_content tests ──


class TestExtractJsonResponse:
    """Verify JSON chain-of-thought extraction."""

    def test_extracts_response_field(self):
        import json

        from src.agent.loop import _extract_json_response
        text = json.dumps({"thought": {"x": 1}, "response": "The answer"})
        assert _extract_json_response(text) == "The answer"

    def test_plain_text_unchanged(self):
        from src.agent.loop import _extract_json_response
        assert _extract_json_response("Hello world") == "Hello world"

    def test_json_without_response_key(self):
        import json

        from src.agent.loop import _extract_json_response
        text = json.dumps({"data": 123})
        assert _extract_json_response(text) == text

    def test_invalid_json_unchanged(self):
        from src.agent.loop import _extract_json_response
        text = '{"broken json'
        assert _extract_json_response(text) == text

    def test_json_array_unchanged(self):
        from src.agent.loop import _extract_json_response
        text = '[1, 2, 3]'
        assert _extract_json_response(text) == text

    def test_empty_string(self):
        from src.agent.loop import _extract_json_response
        assert _extract_json_response("") == ""

    def test_response_with_whitespace(self):
        import json

        from src.agent.loop import _extract_json_response
        text = "  " + json.dumps({"response": "padded"}) + "  "
        assert _extract_json_response(text) == "padded"

    def test_numeric_response(self):
        import json

        from src.agent.loop import _extract_json_response
        text = json.dumps({"response": 42})
        assert _extract_json_response(text) == "42"


class TestStripThinkTags:
    """Verify <think> tag stripping for reasoning models."""

    def test_no_tags(self):
        from src.agent.loop import _strip_think_tags
        assert _strip_think_tags("Hello world") == "Hello world"

    def test_strips_leading_think_block(self):
        from src.agent.loop import _strip_think_tags
        text = "<think>internal reasoning</think>\n\nThe answer is 42."
        assert _strip_think_tags(text) == "The answer is 42."

    def test_preserves_content_when_no_answer_after_tag(self):
        from src.agent.loop import _strip_think_tags
        text = "<think>only thinking, no answer</think>"
        # No content after tag — return original to avoid empty response
        assert _strip_think_tags(text) == text

    def test_unclosed_tag_returns_original(self):
        from src.agent.loop import _strip_think_tags
        text = "<think>still thinking, model interrupted..."
        assert _strip_think_tags(text) == text

    def test_multiple_think_blocks(self):
        from src.agent.loop import _strip_think_tags
        text = "<think>first</think>\n<think>second</think>\nFinal answer."
        assert _strip_think_tags(text) == "Final answer."

    def test_think_in_middle_preserved(self):
        from src.agent.loop import _strip_think_tags
        text = "Prefix text <think>should stay</think> suffix"
        assert _strip_think_tags(text) == text

    def test_empty_string(self):
        from src.agent.loop import _strip_think_tags
        assert _strip_think_tags("") == ""


class TestResolveContentThinkingFallback:
    """Verify _resolve_content falls back to thinking_content and strips tags."""

    def test_content_only(self):
        resp = LLMResponse(content="Normal answer", tokens_used=10)
        assert AgentLoop._resolve_content(resp) == "Normal answer"

    def test_thinking_only_with_tags(self):
        resp = LLMResponse(
            content="",
            thinking_content="<think>reasoning</think>\nThe answer.",
            tokens_used=10,
        )
        assert AgentLoop._resolve_content(resp) == "The answer."

    def test_thinking_only_without_tags(self):
        resp = LLMResponse(
            content="", thinking_content="Plain thinking text", tokens_used=10,
        )
        assert AgentLoop._resolve_content(resp) == "Plain thinking text"

    def test_content_with_think_tags_stripped(self):
        resp = LLMResponse(
            content="<think>reasoning</think>\nClean answer", tokens_used=10,
        )
        assert AgentLoop._resolve_content(resp) == "Clean answer"

    def test_both_content_and_thinking_uses_content(self):
        resp = LLMResponse(
            content="Real answer",
            thinking_content="<think>internal</think>",
            tokens_used=10,
        )
        assert AgentLoop._resolve_content(resp) == "Real answer"

    def test_silent_token_falls_back_to_thinking(self):
        from src.shared.types import SILENT_REPLY_TOKEN
        resp = LLMResponse(
            content=SILENT_REPLY_TOKEN,
            thinking_content="<think>r</think>\nFallback answer",
            tokens_used=10,
        )
        assert AgentLoop._resolve_content(resp) == "Fallback answer"

    def test_json_cot_response_extracted(self):
        """JSON chain-of-thought wrapper is unwrapped to just the response."""
        import json
        cot = json.dumps({
            "thought": {"intent": "greeting"},
            "response": "Hi there! How can I help?",
        })
        resp = LLMResponse(content=cot, tokens_used=10)
        assert AgentLoop._resolve_content(resp) == "Hi there! How can I help?"

    def test_json_without_response_key_unchanged(self):
        """JSON that doesn't have a 'response' key is left as-is."""
        import json
        data = json.dumps({"result": "some data", "status": "ok"})
        resp = LLMResponse(content=data, tokens_used=10)
        assert AgentLoop._resolve_content(resp) == data

    def test_plain_text_not_parsed_as_json(self):
        resp = LLMResponse(content="Just a normal answer", tokens_used=10)
        assert AgentLoop._resolve_content(resp) == "Just a normal answer"

    def test_json_response_null_unchanged(self):
        """JSON with response: null should not return 'None'."""
        import json
        data = json.dumps({"thought": "hmm", "response": None})
        resp = LLMResponse(content=data, tokens_used=10)
        # null response — return original since we'd lose content
        assert AgentLoop._resolve_content(resp) == data

    def test_think_tags_wrapping_json_cot(self):
        """Think tags are stripped before JSON extraction (ordering test)."""
        import json
        cot = json.dumps({"thought": "x", "response": "The answer"})
        text = f"<think>reasoning here</think>\n{cot}"
        resp = LLMResponse(content=text, tokens_used=10)
        assert AgentLoop._resolve_content(resp) == "The answer"

    def test_thinking_fallback_with_think_tags_and_json(self):
        """Full pipeline: thinking fallback → strip tags → extract JSON."""
        import json
        cot = json.dumps({"thought": "x", "response": "extracted"})
        thinking = f"<think>reasoning</think>\n{cot}"
        resp = LLMResponse(
            content="", thinking_content=thinking, tokens_used=10,
        )
        assert AgentLoop._resolve_content(resp) == "extracted"
