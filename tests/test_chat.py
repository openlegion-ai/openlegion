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
    memory.store_fact = AsyncMock(return_value="fact_123")
    memory.log_action = AsyncMock()

    skills = MagicMock()
    skills.get_tool_definitions = MagicMock(return_value=[])
    skills.get_descriptions = MagicMock(return_value="- no tools")
    skills.list_skills = MagicMock(return_value=[])

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
        loop.reset_chat()
        assert len(loop._chat_messages) == 0

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
