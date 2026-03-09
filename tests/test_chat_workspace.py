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
    memory.store_fact = AsyncMock(return_value="fact_123")
    memory.log_action = AsyncMock()

    skills = MagicMock()
    skills.get_tool_definitions = MagicMock(return_value=[])
    skills.get_descriptions = MagicMock(return_value="- memory_search\n- memory_save")
    skills.list_skills = MagicMock(return_value=["memory_search", "memory_save"])

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
        (Path(self._tmpdir) / "MEMORY.md").write_text(
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
        assert "memory_search" in assistant_msg.get("tools", [])

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
