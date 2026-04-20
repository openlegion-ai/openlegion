"""LiteLLM provider — wraps litellm.acompletion for all non-Anthropic models.

This is the fallback provider: it handles every model string that the
AnthropicProvider does not claim (i.e. everything except anthropic/* and
bare claude-* models).
"""

from __future__ import annotations

from typing import AsyncIterator, Any

from src.shared.utils import setup_logging
from .base import LLMProvider, LLMResponse, StreamChunk
from .anthropic import AnthropicProvider

logger = setup_logging("host.providers.litellm")

# Singleton for model prefix checks (avoids instantiating AnthropicProvider repeatedly)
_anthropic_provider = AnthropicProvider()


def _extract_content(raw_content) -> tuple[str, str | None]:
    """Extract text and thinking from LLM response content.

    LiteLLM returns content as a list of blocks when extended thinking is
    enabled: [{"type": "thinking", "thinking": "..."}, {"type": "text", ...}]
    """
    if isinstance(raw_content, str):
        return raw_content, None
    if not isinstance(raw_content, list):
        return str(raw_content) if raw_content else "", None

    text_parts, thinking_parts = [], []
    for block in raw_content:
        if not isinstance(block, dict):
            text_parts.append(str(block))
            continue
        btype = block.get("type", "")
        if btype == "thinking":
            thinking_parts.append(block.get("thinking", ""))
        elif btype == "text":
            text_parts.append(block.get("text", ""))
        else:
            text_parts.append(block.get("text", str(block)))

    return "".join(text_parts), "".join(thinking_parts) if thinking_parts else None


class LiteLLMProvider(LLMProvider):
    """LLM provider backed by litellm — handles all non-Anthropic models."""

    def __init__(self, api_key: str = "", api_base: str = "", **_kwargs) -> None:
        self._api_key = api_key
        self._api_base = api_base

    @property
    def name(self) -> str:
        return "litellm"

    def supports_model(self, model: str) -> bool:
        """Return True for everything AnthropicProvider does not handle."""
        return not _anthropic_provider.supports_model(model)

    async def complete(self, params: dict[str, Any]) -> LLMResponse:
        """Non-streaming completion via litellm.acompletion."""
        import litellm
        litellm.drop_params = True

        kwargs: dict = {k: v for k, v in params.items()}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._api_base:
            kwargs["api_base"] = self._api_base

        result = await litellm.acompletion(**kwargs)
        if not getattr(result, "choices", None):
            raise RuntimeError(
                f"LiteLLM returned empty response (no choices) for model {params.get('model')}"
            )

        msg = result.choices[0].message
        usage = result.usage
        content, thinking = _extract_content(msg.content)
        if thinking is None:
            thinking = getattr(msg, "reasoning_content", None) or None
        if not content and thinking:
            content = thinking

        return LLMResponse(
            content=content,
            tool_calls=[
                {"name": tc.function.name, "arguments": tc.function.arguments}
                for tc in (msg.tool_calls or [])
            ] or None,
            thinking=thinking,
            model=params.get("model", ""),
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            finish_reason=result.choices[0].finish_reason or "stop",
        )

    async def stream(self, params: dict[str, Any]) -> AsyncIterator[StreamChunk]:
        """Streaming completion via litellm.acompletion(stream=True)."""
        import asyncio
        import contextlib
        import litellm
        litellm.drop_params = True

        kwargs: dict = {**params, "stream": True}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._api_base:
            kwargs["api_base"] = self._api_base

        response = await litellm.acompletion(**kwargs)

        collected_tool_calls: list[dict] = []
        _KEEPALIVE_INTERVAL = 15
        chunk_iter = response.__aiter__()
        next_chunk = asyncio.ensure_future(chunk_iter.__anext__())
        try:
            while True:
                done, _ = await asyncio.wait({next_chunk}, timeout=_KEEPALIVE_INTERVAL)
                if not done:
                    # keepalive — no chunk to yield at this layer
                    continue
                try:
                    chunk = next_chunk.result()
                except StopAsyncIteration:
                    break

                delta = chunk.choices[0].delta if chunk.choices else None
                if delta is not None:
                    if delta.content:
                        yield StreamChunk(type="text", content=delta.content)

                    reasoning = getattr(delta, "reasoning_content", None)
                    if reasoning and isinstance(reasoning, str):
                        yield StreamChunk(type="thinking", content=reasoning)

                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            idx = tc.index if hasattr(tc, "index") else 0
                            while len(collected_tool_calls) <= idx:
                                collected_tool_calls.append({"name": "", "arguments": ""})
                            if tc.function and tc.function.name:
                                collected_tool_calls[idx]["name"] = tc.function.name
                            if tc.function and tc.function.arguments:
                                collected_tool_calls[idx]["arguments"] += tc.function.arguments

                next_chunk = asyncio.ensure_future(chunk_iter.__anext__())
        finally:
            if not next_chunk.done():
                next_chunk.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await next_chunk

        # Emit completed tool calls
        for tc in collected_tool_calls:
            yield StreamChunk(type="tool_use", tool_call=tc)

        yield StreamChunk(type="done", finish_reason="stop")
