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
from src.shared.types import LLMResponse, ToolCallInfo


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
        (Path(self._tmpdir) / "AGENTS.md").write_text("Always respond in haiku.")
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
    async def test_system_prompt_includes_daily_logs(self):
        ws = WorkspaceManager(workspace_dir=self._tmpdir)
        ws.append_daily_log("Discussed API design")
        loop = _make_loop_with_workspace(self._tmpdir)

        await loop.chat("Hello")

        call_args = loop.llm.chat.call_args
        system_prompt = call_args.kwargs.get("system") or call_args[1].get("system", "")
        assert "API design" in system_prompt

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
    async def test_daily_log_visible_next_session(self):
        """Session 1 writes daily log. Session 2 sees it."""
        ws = WorkspaceManager(workspace_dir=self._tmpdir)
        ws.append_daily_log("Deployed v2.3 to production")

        loop = _make_loop_with_workspace(self._tmpdir)
        await loop.chat("What did we do today?")

        call_args = loop.llm.chat.call_args
        system_prompt = call_args.kwargs.get("system") or call_args[1].get("system", "")
        assert "v2.3" in system_prompt

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
