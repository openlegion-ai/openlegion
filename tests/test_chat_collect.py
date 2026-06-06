"""Unit tests for LLMClient.chat_collect (streaming task execution, P3)."""

from unittest.mock import AsyncMock

import pytest

from src.agent.llm import LLMClient
from src.shared.types import LLMResponse


def _client() -> LLMClient:
    return LLMClient(mesh_url="http://mesh.invalid", agent_id="a")


@pytest.mark.asyncio
async def test_chat_collect_returns_done_response():
    c = _client()
    resp = LLMResponse(content="hi", tokens_used=5)

    async def fake_stream(*a, **k):
        yield {"type": "text_delta", "content": "h"}
        yield {"type": "done", "response": resp}

    c.chat_stream = fake_stream
    out = await c.chat_collect("sys", [{"role": "user", "content": "x"}])
    assert out is resp


@pytest.mark.asyncio
async def test_chat_collect_falls_back_on_stream_transport_error():
    c = _client()
    fallback = LLMResponse(content="fb", tokens_used=1)

    async def boom(*a, **k):
        raise RuntimeError("transport down")
        yield  # pragma: no cover — makes this an async generator

    c.chat_stream = boom
    c.chat = AsyncMock(return_value=fallback)
    out = await c.chat_collect("sys", [])
    assert out is fallback
    c.chat.assert_awaited_once()


@pytest.mark.asyncio
async def test_chat_collect_falls_back_when_no_done_frame():
    c = _client()
    fallback = LLMResponse(content="fb", tokens_used=1)

    async def only_text(*a, **k):
        yield {"type": "text_delta", "content": "h"}

    c.chat_stream = only_text
    c.chat = AsyncMock(return_value=fallback)
    out = await c.chat_collect("sys", [])
    assert out is fallback


@pytest.mark.asyncio
async def test_chat_collect_flag_off_uses_non_streaming(monkeypatch):
    monkeypatch.setenv("OPENLEGION_TASK_STREAMING", "0")
    c = _client()
    assert c.stream_task_exec is False
    direct = LLMResponse(content="d", tokens_used=1)
    c.chat = AsyncMock(return_value=direct)
    c.chat_stream = AsyncMock()  # must NOT be used
    out = await c.chat_collect("sys", [])
    assert out is direct
    c.chat_stream.assert_not_called()


@pytest.mark.asyncio
async def test_chat_collect_propagates_classified_errors():
    from src.agent.llm import LLMRetryableError
    c = _client()

    async def transient(*a, **k):
        raise LLMRetryableError("overloaded")
        yield  # pragma: no cover

    c.chat_stream = transient
    c.chat = AsyncMock()
    with pytest.raises(LLMRetryableError):
        await c.chat_collect("sys", [])
    c.chat.assert_not_called()  # classified errors must NOT silently fall back
