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

    skills = MagicMock()
    skills.get_tool_definitions = MagicMock(return_value=[])
    skills.get_descriptions = MagicMock(return_value="- memory_search\n- memory_save")
    skills.list_skills = MagicMock(return_value=["memory_search", "memory_save"])
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

    workspace = WorkspaceManager(workspace_dir=tmpdir)
    context_mgr = ContextManager(max_tokens=context_max_tokens, llm=llm, workspace=workspace)

    return AgentLoop(
        agent_id="test_agent",
        role="assistant",
        memory=memory,
        skills=skills,
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
        loop.skills.execute = AsyncMock(return_value={"results": [], "count": 0})
        loop.skills.get_tool_definitions = MagicMock(
            return_value=[{"type": "function", "function": {"name": "memory_search"}}]
        )

        await loop.chat("Search for test")

        # Verify workspace_manager was passed to skills.execute
        call_args = loop.skills.execute.call_args
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
        loop.skills.execute = AsyncMock(return_value={"results": []})
        loop.skills.get_tool_definitions = MagicMock(
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

        skills = MagicMock()
        skills.get_tool_definitions = MagicMock(return_value=[])
        skills.get_descriptions = MagicMock(return_value="none")
        skills.list_skills = MagicMock(return_value=[])

        llm = MagicMock()
        llm.chat = AsyncMock(return_value=LLMResponse(content="Reply", tokens_used=10))
        llm.default_model = "test"

        mesh = MagicMock()
        mesh.send_system_message = AsyncMock(return_value={})

        loop = AgentLoop(
            agent_id="no_ws", role="test",
            memory=memory, skills=skills, llm=llm, mesh_client=mesh,
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
        loop.skills.execute = AsyncMock(
            side_effect=RuntimeError("tool blew up"),
        )
        loop.skills.get_tool_definitions = MagicMock(
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
        loop.skills.execute = AsyncMock(return_value={"results": []})
        loop.skills.get_tool_definitions = MagicMock(
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
