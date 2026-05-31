"""Tests for chat mode with workspace integration.

Verifies:
- Workspace files are loaded into system prompt
- Memory search runs on first message
- Daily log is written during conversation
- Context manager is used instead of _trim_context
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.context import ContextManager
from src.agent.loop import AgentLoop
from src.agent.workspace import WorkspaceManager
from src.shared.types import SILENT_REPLY_TOKEN, LLMResponse, ToolCallInfo


def _make_loop_with_workspace(
    tmpdir: str,
    llm_responses: list[LLMResponse] | None = None,
    context_max_tokens: int = 100_000,
) -> AgentLoop:
    """Create an AgentLoop with real workspace and mock LLM."""
    memory = MagicMock()
    memory.get_high_salience_facts = MagicMock(return_value=[])
    memory.search = AsyncMock(return_value=[])
    memory.log_action = AsyncMock()
    memory._run_db = AsyncMock(return_value=None)

    tools = MagicMock()
    tools.get_tool_definitions = MagicMock(return_value=[])
    tools.get_descriptions = MagicMock(return_value="- memory_search\n- memory_save")
    tools.list_tools = MagicMock(return_value=["memory_search", "memory_save"])
    tools.is_parallel_safe = MagicMock(return_value=True)
    tools.get_loop_exempt_tools = MagicMock(return_value=frozenset())

    llm = MagicMock()
    if llm_responses:
        llm.chat = AsyncMock(side_effect=llm_responses)
    else:
        llm.chat = AsyncMock(return_value=LLMResponse(content="Hello!", tokens_used=50))
    llm.default_model = "test-model"

    mesh_client = MagicMock()
    mesh_client.send_system_message = AsyncMock(return_value={})

    workspace = WorkspaceManager(workspace_dir=tmpdir)
    context_mgr = ContextManager(max_tokens=context_max_tokens, llm=llm, workspace=workspace)

    return AgentLoop(
        agent_id="test_agent",
        role="assistant",
        memory=memory,
        tools=tools,
        llm=llm,
        mesh_client=mesh_client,
        workspace=workspace,
        context_manager=context_mgr,
    )


class TestChatWithWorkspace:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_system_prompt_includes_workspace_files(self):
        (Path(self._tmpdir) / "INSTRUCTIONS.md").write_text("Always respond in haiku.")
        (Path(self._tmpdir) / "SOUL.md").write_text("You are a poet.")
        loop = _make_loop_with_workspace(self._tmpdir)

        await loop.chat("Hello")

        # Verify LLM was called with system prompt containing workspace content
        call_args = loop.llm.chat.call_args
        system_prompt = call_args.kwargs.get("system") or call_args[1].get("system", "")
        assert "haiku" in system_prompt
        assert "poet" in system_prompt

    @pytest.mark.asyncio
    async def test_system_prompt_includes_memory(self):
        (Path(self._tmpdir) / "MEMORY.md").write_text("User prefers Python.")
        loop = _make_loop_with_workspace(self._tmpdir)

        await loop.chat("Hello")

        call_args = loop.llm.chat.call_args
        system_prompt = call_args.kwargs.get("system") or call_args[1].get("system", "")
        assert "Python" in system_prompt

    @pytest.mark.asyncio
    async def test_daily_logs_not_in_system_prompt(self):
        """Daily logs are accessed via memory_search, not injected into prompt."""
        ws = WorkspaceManager(workspace_dir=self._tmpdir)
        ws.append_daily_log("Discussed API design")
        loop = _make_loop_with_workspace(self._tmpdir)

        await loop.chat("Hello")

        call_args = loop.llm.chat.call_args
        system_prompt = call_args.kwargs.get("system") or call_args[1].get("system", "")
        assert "API design" not in system_prompt

    @pytest.mark.asyncio
    async def test_first_message_preloads_relevant_memory(self):
        # Write to a daily log file (not a bootstrap file like MEMORY.md)
        # because bootstrap files are excluded from auto-search to avoid
        # duplicate content — they're already in the system prompt.
        daily_dir = Path(self._tmpdir) / "memory"
        daily_dir.mkdir(exist_ok=True)
        (daily_dir / "2026-03-01.md").write_text(
            "User is building a machine learning pipeline.\n"
        )
        loop = _make_loop_with_workspace(self._tmpdir)

        await loop.chat("Tell me about machine learning")

        # The user message sent to LLM should contain auto-loaded memory
        call_args = loop.llm.chat.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages", [])
        user_msg = messages[0]["content"]
        assert "machine learning" in user_msg.lower()
        assert "auto-loaded" in user_msg.lower()

    @pytest.mark.asyncio
    async def test_chat_writes_daily_log(self):
        loop = _make_loop_with_workspace(self._tmpdir)

        await loop.chat("What is Python?")

        log_files = list((Path(self._tmpdir) / "memory").glob("*.md"))
        assert len(log_files) == 1
        content = log_files[0].read_text()
        assert "Python" in content

    @pytest.mark.asyncio
    async def test_chat_with_tool_passes_workspace_manager(self):
        tool_response = LLMResponse(
            content="",
            tool_calls=[ToolCallInfo(name="memory_search", arguments={"query": "test"})],
            tokens_used=30,
        )
        final_response = LLMResponse(content="Found it!", tokens_used=20)

        loop = _make_loop_with_workspace(
            self._tmpdir, llm_responses=[tool_response, final_response]
        )
        loop.tools.execute = AsyncMock(return_value={"results": [], "count": 0})
        loop.tools.get_tool_definitions = MagicMock(
            return_value=[{"type": "function", "function": {"name": "memory_search"}}]
        )

        await loop.chat("Search for test")

        # Verify workspace_manager was passed to tools.execute
        call_args = loop.tools.execute.call_args
        assert call_args.kwargs.get("workspace_manager") is not None

    @pytest.mark.asyncio
    async def test_reset_chat_clears_but_workspace_persists(self):
        loop = _make_loop_with_workspace(self._tmpdir)
        loop.workspace.append_daily_log("Important fact from session 1")

        await loop.chat("Hello")
        await loop.reset_chat()

        # Workspace files persist after reset
        assert loop.workspace.load_daily_logs(days=1) != ""
        assert "Important fact from session 1" in loop.workspace.load_daily_logs()

        # But chat messages are cleared
        assert len(loop._chat_messages) == 0


class TestCrossSessionMemory:
    """Verify the core acceptance criterion: agent remembers across sessions."""

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_fact_persists_across_sessions(self):
        """Session 1 saves a fact. Session 2 sees it in system prompt."""
        # Session 1: save a fact
        loop1 = _make_loop_with_workspace(self._tmpdir)
        loop1.workspace.append_daily_log("User's dog is named Biscuit")
        loop1.workspace.append_memory("- User has a dog named Biscuit")
        await loop1.chat("My dog's name is Biscuit")

        # Session 2: new AgentLoop, same workspace dir
        loop2 = _make_loop_with_workspace(self._tmpdir)
        await loop2.chat("What is my dog's name?")

        # Verify system prompt in session 2 contains the fact from session 1
        call_args = loop2.llm.chat.call_args
        system_prompt = call_args.kwargs.get("system") or call_args[1].get("system", "")
        assert "Biscuit" in system_prompt

    @pytest.mark.asyncio
    async def test_daily_log_accessible_via_search(self):
        """Daily logs are searchable via workspace search, not in system prompt."""
        ws = WorkspaceManager(workspace_dir=self._tmpdir)
        ws.append_daily_log("Deployed v2.3 to production")

        # Daily log should be findable via search
        results = ws.search("v2.3 production")
        assert len(results) > 0

        # But NOT in system prompt
        loop = _make_loop_with_workspace(self._tmpdir)
        await loop.chat("What did we do today?")

        call_args = loop.llm.chat.call_args
        system_prompt = call_args.kwargs.get("system") or call_args[1].get("system", "")
        assert "v2.3" not in system_prompt

    @pytest.mark.asyncio
    async def test_memory_search_finds_old_session_data(self):
        """BM25 search finds data from a previous session's daily log."""
        ws = WorkspaceManager(workspace_dir=self._tmpdir)
        memory_dir = Path(self._tmpdir) / "memory"
        (memory_dir / "2026-02-16.md").write_text(
            "- [09:00] Discussed Kubernetes deployment strategy\n"
            "- [10:00] Decided to use Helm charts\n"
        )

        results = ws.search("Kubernetes deployment")
        assert len(results) > 0
        assert any("Kubernetes" in r["snippet"] for r in results)


class TestChatTranscriptIntegration:
    """Verify chat messages are persisted to the workspace transcript."""

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_chat_persists_user_and_assistant(self):
        loop = _make_loop_with_workspace(self._tmpdir)
        await loop.chat("Hello agent")

        transcript = loop.workspace.load_chat_transcript()
        assert len(transcript) == 2
        assert transcript[0]["role"] == "user"
        assert transcript[0]["content"] == "Hello agent"
        assert transcript[1]["role"] == "assistant"
        assert transcript[1]["content"] == "Hello!"

    @pytest.mark.asyncio
    async def test_chat_persists_tool_names(self):
        tool_response = LLMResponse(
            content="",
            tool_calls=[ToolCallInfo(name="memory_search", arguments={"query": "test"})],
            tokens_used=30,
        )
        final_response = LLMResponse(content="Found it!", tokens_used=20)

        loop = _make_loop_with_workspace(
            self._tmpdir, llm_responses=[tool_response, final_response],
        )
        loop.tools.execute = AsyncMock(return_value={"results": []})
        loop.tools.get_tool_definitions = MagicMock(
            return_value=[{"type": "function", "function": {"name": "memory_search"}}],
        )

        await loop.chat("Search for test")

        transcript = loop.workspace.load_chat_transcript()
        assistant_msg = [m for m in transcript if m["role"] == "assistant"][0]
        tools = assistant_msg.get("tools", [])
        tool_names = [t["name"] if isinstance(t, dict) else t for t in tools]
        assert "memory_search" in tool_names

    @pytest.mark.asyncio
    async def test_reset_archives_transcript(self):
        loop = _make_loop_with_workspace(self._tmpdir)
        await loop.chat("Hello")
        await loop.reset_chat()

        # Transcript should be archived, not present
        assert loop.workspace.load_chat_transcript() == []

        # Archive file should exist
        archive_dir = Path(self._tmpdir) / "chat_archive"
        assert archive_dir.exists()
        assert len(list(archive_dir.glob("*.jsonl"))) == 1

    @pytest.mark.asyncio
    async def test_get_chat_messages_reads_transcript(self):
        loop = _make_loop_with_workspace(self._tmpdir)
        await loop.chat("Hello")

        # get_chat_messages should return transcript data
        messages = loop.get_chat_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert "ts" in messages[0]  # transcript entries have timestamps

    @pytest.mark.asyncio
    async def test_transcript_survives_container_restart(self):
        """Simulate container restart: new AgentLoop, same workspace dir."""
        loop1 = _make_loop_with_workspace(self._tmpdir)
        await loop1.chat("Hello from session 1")

        # New loop = simulated container restart
        loop2 = _make_loop_with_workspace(self._tmpdir)
        assert len(loop2._chat_messages) == 0  # in-memory is empty

        # But transcript is available
        messages = loop2.get_chat_messages()
        assert len(messages) == 2
        assert messages[0]["content"] == "Hello from session 1"

    @pytest.mark.asyncio
    async def test_multiple_turns_accumulate(self):
        responses = [
            LLMResponse(content="First reply", tokens_used=50),
            LLMResponse(content="Second reply", tokens_used=50),
        ]
        loop = _make_loop_with_workspace(self._tmpdir, llm_responses=responses)

        await loop.chat("First message")
        await loop.chat("Second message")

        transcript = loop.workspace.load_chat_transcript()
        assert len(transcript) == 4
        assert transcript[0]["content"] == "First message"
        assert transcript[1]["content"] == "First reply"
        assert transcript[2]["content"] == "Second message"
        assert transcript[3]["content"] == "Second reply"

    @pytest.mark.asyncio
    async def test_fallback_when_no_workspace(self):
        """Without workspace, get_chat_messages falls back to in-memory."""
        memory = MagicMock()
        memory.get_high_salience_facts = MagicMock(return_value=[])
        memory.search = AsyncMock(return_value=[])
        memory.log_action = AsyncMock()
        memory.get_tool_history = MagicMock(return_value=[])
        memory._run_db = AsyncMock(return_value=None)

        tools = MagicMock()
        tools.get_tool_definitions = MagicMock(return_value=[])
        tools.get_descriptions = MagicMock(return_value="none")
        tools.list_tools = MagicMock(return_value=[])

        llm = MagicMock()
        llm.chat = AsyncMock(return_value=LLMResponse(content="Reply", tokens_used=10))
        llm.default_model = "test"

        mesh = MagicMock()
        mesh.send_system_message = AsyncMock(return_value={})

        loop = AgentLoop(
            agent_id="no_ws", role="test",
            memory=memory, tools=tools, llm=llm, mesh_client=mesh,
            workspace=None,
        )
        await loop.chat("Hello")
        messages = loop.get_chat_messages()
        # Falls back to in-memory filtering
        assert len(messages) >= 1
        assert any(m["role"] == "user" for m in messages)

    @pytest.mark.asyncio
    async def test_silent_response_persists_user_only(self):
        """SILENT_REPLY_TOKEN responses persist user message but not empty assistant."""
        loop = _make_loop_with_workspace(
            self._tmpdir,
            llm_responses=[LLMResponse(content=SILENT_REPLY_TOKEN, tokens_used=10)],
        )
        await loop.chat("Background ack")

        transcript = loop.workspace.load_chat_transcript()
        assert len(transcript) == 1
        assert transcript[0]["role"] == "user"
        assert transcript[0]["content"] == "Background ack"

    @pytest.mark.asyncio
    async def test_error_path_persists_to_transcript(self):
        """When chat raises an exception, the error response is persisted."""
        loop = _make_loop_with_workspace(self._tmpdir)
        loop.llm.chat = AsyncMock(side_effect=ValueError("LLM exploded"))

        result = await loop.chat("Trigger error")

        assert "Error:" in result["response"]
        transcript = loop.workspace.load_chat_transcript()
        assert len(transcript) == 2
        assert transcript[0]["role"] == "user"
        assert transcript[1]["role"] == "assistant"
        assert "LLM exploded" in transcript[1]["content"]


# ── Fix B — Partial chat persistence for in-flight tool calls ────────


class TestChatPartialPersistence:
    """Fix B — assistant turns that fire long-running tool calls must
    persist a ``partial`` transcript entry BEFORE tool dispatch so a
    dashboard refresh mid-flight doesn't lose the assistant bubble.
    The final entry (written at turn close) supersedes the partial via
    ``turn_id`` dedup in ``load_chat_transcript``.
    """

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_legacy_transcript_without_turn_id_unchanged(self):
        """Pre-Fix-B transcripts have no ``turn_id`` on any entry. Loading
        such a transcript must pass every entry through unchanged — no
        dedup, no reordering."""
        ws = WorkspaceManager(workspace_dir=self._tmpdir)
        ws.append_chat_message("user", "first")
        ws.append_chat_message("assistant", "first reply")
        ws.append_chat_message("user", "second")

        transcript = ws.load_chat_transcript()
        assert len(transcript) == 3
        assert [t["role"] for t in transcript] == ["user", "assistant", "user"]
        assert [t["content"] for t in transcript] == [
            "first", "first reply", "second",
        ]
        # No entry has turn_id or partial set.
        assert all("turn_id" not in t for t in transcript)
        assert all("partial" not in t for t in transcript)

    def test_load_transcript_dedupes_two_partials_keeps_latest(self):
        """Two entries with the same ``turn_id``: ``load_chat_transcript``
        keeps only the LATER one. Validates the core dedup contract."""
        ws = WorkspaceManager(workspace_dir=self._tmpdir)
        ws.append_chat_message(
            "assistant", "in flight", turn_id="t1", partial=True,
        )
        ws.append_chat_message(
            "assistant", "final answer", turn_id="t1", partial=False,
        )

        transcript = ws.load_chat_transcript()
        assert len(transcript) == 1
        assert transcript[0]["content"] == "final answer"
        assert transcript[0]["turn_id"] == "t1"
        # ``partial`` flag was dropped on the second write (omitted).
        assert "partial" not in transcript[0]

    def test_dedup_preserves_chronological_order_of_first_occurrence(self):
        """When a partial gets superseded by a final, the entry must stay
        in its ORIGINAL position in the transcript — not jump to the
        end. Otherwise the chat panel would visually reshuffle on every
        turn close."""
        ws = WorkspaceManager(workspace_dir=self._tmpdir)
        ws.append_chat_message("user", "u1")
        ws.append_chat_message(
            "assistant", "partial reply", turn_id="t1", partial=True,
        )
        ws.append_chat_message("user", "u2")
        # Final lands after a subsequent user message — it must replace
        # the partial in position 1, not move to position 3.
        ws.append_chat_message(
            "assistant", "final reply", turn_id="t1", partial=False,
        )

        transcript = ws.load_chat_transcript()
        assert [t["content"] for t in transcript] == [
            "u1", "final reply", "u2",
        ]

    def test_dedup_applies_limit_after(self):
        """``limit`` trims AFTER dedup so a final entry that supersedes
        an earlier partial doesn't get dropped just because the partial
        consumed a slot."""
        ws = WorkspaceManager(workspace_dir=self._tmpdir)
        ws.append_chat_message(
            "assistant", "partial", turn_id="t1", partial=True,
        )
        ws.append_chat_message("user", "u2")
        ws.append_chat_message("user", "u3")
        ws.append_chat_message(
            "assistant", "final", turn_id="t1", partial=False,
        )

        # Pre-dedup, the file has 4 lines. After dedup it has 3 entries.
        # limit=3 must return all 3 — no further trim.
        transcript = ws.load_chat_transcript(limit=3)
        assert len(transcript) == 3
        assert transcript[0]["content"] == "final"

    @pytest.mark.asyncio
    async def test_partial_chat_entry_persisted_on_tool_dispatch(self):
        """During a tool-calling round, the loop writes a ``partial=True``
        assistant entry BEFORE the tool fires. Simulate by raising mid-
        tool — the partial must remain in the transcript even though
        the turn never completed."""
        tool_response = LLMResponse(
            content="thinking…",
            tool_calls=[
                ToolCallInfo(name="slow_tool", arguments={"k": "v"}),
            ],
            tokens_used=10,
        )

        loop = _make_loop_with_workspace(
            self._tmpdir, llm_responses=[tool_response],
        )
        # Tool execution raises a base exception so the partial we wrote
        # before dispatch is the LAST persistent transcript event for
        # this turn — the simulated mid-tool crash.
        loop.tools.execute = AsyncMock(
            side_effect=RuntimeError("tool blew up"),
        )
        loop.tools.get_tool_definitions = MagicMock(
            return_value=[{"type": "function", "function": {"name": "slow_tool"}}],
        )

        await loop.chat("trigger slow tool")

        # Read raw file lines (no dedup) — we want to verify the partial
        # was actually written to disk before the tool dispatched.
        from pathlib import Path as _P
        path = _P(self._tmpdir) / WorkspaceManager.CHAT_TRANSCRIPT
        raw_lines = path.read_text().strip().splitlines()
        import json as _json
        parsed = [_json.loads(line) for line in raw_lines if line.strip()]
        partial_entries = [
            e for e in parsed
            if e.get("role") == "assistant" and e.get("partial") is True
        ]
        assert len(partial_entries) >= 1, (
            "Expected a partial assistant entry written before tool dispatch"
        )
        assert partial_entries[0].get("turn_id"), (
            "Partial entry must carry a turn_id so the final can supersede it"
        )

    @pytest.mark.asyncio
    async def test_partial_superseded_by_final_with_same_turn_id(self):
        """A complete turn produces exactly ONE assistant entry visible
        via ``load_chat_transcript`` — the partial-pre-dispatch entry
        is hidden behind the final."""
        tool_response = LLMResponse(
            content="",
            tool_calls=[
                ToolCallInfo(name="memory_search", arguments={"query": "x"}),
            ],
            tokens_used=10,
        )
        final_response = LLMResponse(content="Final answer", tokens_used=20)

        loop = _make_loop_with_workspace(
            self._tmpdir, llm_responses=[tool_response, final_response],
        )
        loop.tools.execute = AsyncMock(return_value={"results": []})
        loop.tools.get_tool_definitions = MagicMock(
            return_value=[{"type": "function", "function": {"name": "memory_search"}}],
        )

        await loop.chat("ask question")

        transcript = loop.workspace.load_chat_transcript()
        assistant_entries = [
            e for e in transcript if e.get("role") == "assistant"
        ]
        assert len(assistant_entries) == 1, (
            f"Expected exactly one visible assistant entry after dedup, "
            f"got {len(assistant_entries)}: {assistant_entries}"
        )
        assert assistant_entries[0]["content"] == "Final answer"
        # Final entry has no partial flag set.
        assert assistant_entries[0].get("partial") is not True

    @pytest.mark.asyncio
    async def test_streaming_chat_persists_partial_before_tool_dispatch(self):
        """Streaming path also writes a partial=True assistant entry before
        tool dispatch — mirrors the _chat_inner behavior so a dashboard
        refresh during a streamed long tool call doesn't lose the bubble."""
        tool_response = LLMResponse(
            content="streaming thinking…",
            tool_calls=[
                ToolCallInfo(name="slow_tool", arguments={"k": "v"}),
            ],
            tokens_used=10,
        )
        final_response = LLMResponse(content="Final.", tokens_used=20)

        loop = _make_loop_with_workspace(
            self._tmpdir, llm_responses=[tool_response, final_response],
        )

        # Force the streaming path to fall back to non-streaming for the
        # LLM call (the partial-write happens AFTER the LLM call regardless
        # of whether it streamed or not).
        async def _no_stream(**kwargs):
            raise RuntimeError("no streaming")
            yield  # makes it an async generator

        loop.llm.chat_stream = _no_stream
        loop.tools.execute = AsyncMock(return_value={"results": []})
        loop.tools.get_tool_definitions = MagicMock(
            return_value=[
                {"type": "function", "function": {"name": "slow_tool"}},
            ],
        )

        # Drive the async generator to completion.
        events = []
        async for event in loop.chat_stream("trigger slow tool"):
            events.append(event)

        # Read raw transcript lines (no dedup) — verify a partial=True
        # assistant entry was written to disk before the final.
        from pathlib import Path as _P
        path = _P(self._tmpdir) / WorkspaceManager.CHAT_TRANSCRIPT
        raw_lines = path.read_text().strip().splitlines()
        import json as _json
        parsed = [_json.loads(line) for line in raw_lines if line.strip()]
        partial_entries = [
            e for e in parsed
            if e.get("role") == "assistant" and e.get("partial") is True
        ]
        assert len(partial_entries) >= 1, (
            "Streaming path must persist a partial assistant entry "
            "before tool dispatch — same contract as _chat_inner."
        )
        assert partial_entries[0].get("turn_id"), (
            "Partial entry must carry a turn_id for dedup against the final"
        )
        # Fix 2 (codex P1.2) — when no text_delta events streamed
        # (streaming raised in this test, so accumulated_text is empty),
        # the partial must fall back to ``llm_response.content`` instead
        # of writing an empty bubble.
        assert partial_entries[0].get("content") == "streaming thinking…", (
            "Partial must fall back to llm_response.content when "
            f"accumulated_text is empty, got {partial_entries[0].get('content')!r}"
        )
        # After dedup, exactly one assistant entry visible (the final).
        transcript = loop.workspace.load_chat_transcript()
        assistant_entries = [
            e for e in transcript if e.get("role") == "assistant"
        ]
        assert len(assistant_entries) == 1
        assert assistant_entries[0]["content"] == "Final."

    @pytest.mark.asyncio
    async def test_streaming_partial_falls_back_to_llm_response_content_when_no_deltas(self):
        """Some LLM providers return content as a single block instead of
        streamed text_delta events. The streaming partial-write must then
        fall back to llm_response.content so the bubble isn't empty during
        a mid-flight refresh.

        Scenario: ``chat_stream`` emits a ``done`` event carrying the
        full LLMResponse (no text_delta events along the way).
        ``accumulated_text`` is empty but ``llm_response.content`` has
        the prose. The partial-write must use the content.
        """
        tool_response = LLMResponse(
            content="single-block prose body",
            tool_calls=[
                ToolCallInfo(name="slow_tool", arguments={"k": "v"}),
            ],
            tokens_used=10,
        )
        final_response = LLMResponse(content="Final.", tokens_used=20)

        loop = _make_loop_with_workspace(
            self._tmpdir, llm_responses=[tool_response, final_response],
        )

        # Streaming path emits only ``done`` (no text_delta events) —
        # the LLM returned content as one block. ``accumulated_text``
        # stays empty.
        async def _done_only_stream(**kwargs):
            yield {"type": "done", "response": tool_response}

        async def _done_only_stream_final(**kwargs):
            yield {"type": "done", "response": final_response}

        call_count = {"n": 0}

        async def _dispatch(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                async for evt in _done_only_stream(**kwargs):
                    yield evt
            else:
                async for evt in _done_only_stream_final(**kwargs):
                    yield evt

        loop.llm.chat_stream = _dispatch
        loop.tools.execute = AsyncMock(return_value={"results": []})
        loop.tools.get_tool_definitions = MagicMock(
            return_value=[
                {"type": "function", "function": {"name": "slow_tool"}},
            ],
        )

        events = []
        async for event in loop.chat_stream("trigger slow tool"):
            events.append(event)

        # Read raw transcript — verify the partial entry contains the
        # fallback content (not empty).
        from pathlib import Path as _P
        path = _P(self._tmpdir) / WorkspaceManager.CHAT_TRANSCRIPT
        raw_lines = path.read_text().strip().splitlines()
        import json as _json
        parsed = [_json.loads(line) for line in raw_lines if line.strip()]
        partial_entries = [
            e for e in parsed
            if e.get("role") == "assistant" and e.get("partial") is True
        ]
        assert len(partial_entries) >= 1, (
            "Expected a partial assistant entry written before tool dispatch"
        )
        # The partial must carry the LLM-block content as a fallback.
        assert partial_entries[0].get("content") == "single-block prose body", (
            "Partial must fall back to llm_response.content when no "
            f"text_delta events streamed, got {partial_entries[0].get('content')!r}"
        )

    @pytest.mark.asyncio
    async def test_multi_round_tool_dispatch_dedupes_to_single_assistant_entry(self):
        """A turn with 3 rounds of tool calls + final text should produce
        multiple partial writes (one per round) but ONLY ONE assistant
        entry visible after load_chat_transcript dedup. The visible final
        carries every tool name used across the turn (cumulative)."""
        round1 = LLMResponse(
            content="round 1 prose",
            tool_calls=[ToolCallInfo(name="tool_a", arguments={"x": 1})],
            tokens_used=10,
        )
        round2 = LLMResponse(
            content="round 2 prose",
            tool_calls=[ToolCallInfo(name="tool_b", arguments={"x": 2})],
            tokens_used=10,
        )
        round3 = LLMResponse(
            content="round 3 prose",
            tool_calls=[ToolCallInfo(name="tool_c", arguments={"x": 3})],
            tokens_used=10,
        )
        final_response = LLMResponse(
            content="Final synthesized answer", tokens_used=20,
        )

        loop = _make_loop_with_workspace(
            self._tmpdir,
            llm_responses=[round1, round2, round3, final_response],
        )
        loop.tools.execute = AsyncMock(return_value={"ok": True})
        loop.tools.get_tool_definitions = MagicMock(
            return_value=[
                {"type": "function", "function": {"name": "tool_a"}},
                {"type": "function", "function": {"name": "tool_b"}},
                {"type": "function", "function": {"name": "tool_c"}},
            ],
        )

        await loop.chat("multi-round work")

        # Raw file: 3 partial assistant entries (one per tool round) +
        # final assistant entry = at least 3 partials present.
        from pathlib import Path as _P
        path = _P(self._tmpdir) / WorkspaceManager.CHAT_TRANSCRIPT
        raw_lines = path.read_text().strip().splitlines()
        import json as _json
        parsed = [_json.loads(line) for line in raw_lines if line.strip()]
        partial_entries = [
            e for e in parsed
            if e.get("role") == "assistant" and e.get("partial") is True
        ]
        assert len(partial_entries) >= 3, (
            f"Expected at least 3 partial entries (one per tool round), "
            f"got {len(partial_entries)}"
        )

        # After dedup, exactly ONE visible assistant entry.
        transcript = loop.workspace.load_chat_transcript()
        assistant_entries = [
            e for e in transcript if e.get("role") == "assistant"
        ]
        assert len(assistant_entries) == 1, (
            f"Expected 1 visible assistant entry, got {len(assistant_entries)}"
        )
        # The visible entry shows the FINAL content.
        assert assistant_entries[0]["content"] == "Final synthesized answer"
        # The visible entry's tools list (from _log_chat_turn, sourced
        # from tool_outputs) includes all 3 tool names. ``_log_chat_turn``
        # stores tools as a list of dicts with a ``name`` field.
        visible_tools = assistant_entries[0].get("tools") or []
        visible_tool_names = {
            (t.get("name") if isinstance(t, dict) else t)
            for t in visible_tools
        }
        assert visible_tool_names == {"tool_a", "tool_b", "tool_c"}, (
            f"Final entry tools should include all 3 names, "
            f"got {visible_tool_names}"
        )

        # PE review follow-up — the LAST partial (just before the final
        # supersedes it) should carry the cumulative tool list across
        # all 3 rounds + cumulative content joined with newlines.
        # ``append_chat_message`` stores ``tool_names`` (list[str]) under
        # the ``tools`` key when no full ``tools`` dict list is passed.
        last_partial = partial_entries[-1]
        last_partial_tools = last_partial.get("tools") or []
        last_partial_tool_names = {
            (t.get("name") if isinstance(t, dict) else t)
            for t in last_partial_tools
        }
        assert last_partial_tool_names == {"tool_a", "tool_b", "tool_c"}, (
            f"Last partial entry must carry cumulative tool names, "
            f"got {last_partial_tool_names}"
        )
        # Cumulative content: all 3 rounds' prose joined by newlines.
        assert "round 1 prose" in last_partial.get("content", "")
        assert "round 2 prose" in last_partial.get("content", "")
        assert "round 3 prose" in last_partial.get("content", "")

    @pytest.mark.asyncio
    async def test_partial_cumulative_content_then_final_keeps_final(self):
        """During mid-flight, the partial entry carries CUMULATIVE content
        across rounds (newline-joined). After turn close, the final entry
        via _log_chat_turn carries only the final round's content. dedup
        keeps the final — the cumulative partial commentary is intentional
        mid-flight UX only, not preserved in the post-completion history.
        """
        tool_resp_1 = LLMResponse(
            content="Step 1 commentary.",
            tool_calls=[ToolCallInfo(name="tool_a", arguments={})],
            tokens_used=10,
        )
        tool_resp_2 = LLMResponse(
            content="Step 2 commentary.",
            tool_calls=[ToolCallInfo(name="tool_b", arguments={})],
            tokens_used=10,
        )
        final_resp = LLMResponse(content="Done.", tokens_used=10)
        loop = _make_loop_with_workspace(
            self._tmpdir,
            llm_responses=[tool_resp_1, tool_resp_2, final_resp],
        )
        loop.tools.execute = AsyncMock(return_value={"ok": True})
        loop.tools.get_tool_definitions = MagicMock(
            return_value=[
                {"type": "function", "function": {"name": "tool_a"}},
                {"type": "function", "function": {"name": "tool_b"}},
            ],
        )
        await loop.chat("ask")
        transcript = loop.workspace.load_chat_transcript()
        assistants = [e for e in transcript if e.get("role") == "assistant"]
        assert len(assistants) == 1
        # The final's content is just the final round's text — NOT the
        # cumulative "Step 1 commentary.\nStep 2 commentary.\nDone."
        # This pins the pre-existing _log_chat_turn behavior.
        assert assistants[0]["content"] == "Done."

    def test_partial_survives_when_rotation_drops_first_half(self, monkeypatch):
        """When append_chat_message rotates the transcript (first half
        dropped) BETWEEN the partial-write and the final-write, the
        final-write must still cleanly land and load_chat_transcript
        must return the final (not a stale partial)."""
        ws = WorkspaceManager(workspace_dir=self._tmpdir)
        # Lower the rotation threshold so we trip it easily.
        monkeypatch.setattr(
            WorkspaceManager, "_MAX_TRANSCRIPT_SIZE", 256, raising=False,
        )
        # Write a partial that will be in the FIRST half (gets dropped).
        ws.append_chat_message(
            "assistant", "early partial",
            turn_id="t1", partial=True,
        )
        # Pad the file with enough small user messages to exceed the cap.
        for i in range(20):
            ws.append_chat_message("user", f"padding {i}" * 10)
        # Write the final — file may rotate during this append.
        ws.append_chat_message(
            "assistant", "final answer",
            turn_id="t1", partial=False,
        )
        transcript = ws.load_chat_transcript()
        # Find the assistant entry — there should be exactly one (or zero
        # if rotation dropped both, but that would be a separate bug).
        assistants = [e for e in transcript if e.get("role") == "assistant"]
        if assistants:
            # If anything survives, it must be the final — the partial
            # being kept solo would mean the final couldn't supersede,
            # which is a bug.
            assert assistants[-1]["content"] == "final answer", (
                f"Expected final to win after rotation, got: {assistants}"
            )
            assert assistants[-1].get("partial") is not True
