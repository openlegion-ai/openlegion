"""LLM client for agents. All calls route through the mesh API proxy.

Agents have NO external network access and hold NO API keys.
The mesh tracks token usage and enforces budgets.
"""

from __future__ import annotations

import json
import os
from typing import AsyncIterator

import httpx

from src.shared.types import APIProxyRequest, LLMResponse, ToolCallInfo
from src.shared.utils import setup_logging

logger = setup_logging("agent.llm")


class LLMClient:
    """LLM interface that routes all calls through the mesh API proxy."""

    _THINKING_BUDGETS = {"low": 5_000, "medium": 10_000, "high": 25_000}
    _VALID_THINKING_LEVELS = {"off", "low", "medium", "high"}

    def __init__(
        self,
        mesh_url: str,
        agent_id: str = "agent",
        default_model: str = "openai/gpt-4o-mini",
        embedding_model: str = "",
        thinking: str = "off",
    ):
        if thinking and thinking not in self._VALID_THINKING_LEVELS:
            logger.warning(
                "Invalid thinking level '%s', falling back to 'off'. "
                "Valid: %s", thinking, ", ".join(sorted(self._VALID_THINKING_LEVELS)),
            )
            thinking = "off"
        self.mesh_url = mesh_url
        self.agent_id = agent_id
        self.default_model = default_model
        self.embedding_model = embedding_model
        self.thinking = thinking
        self._client: httpx.AsyncClient | None = None
        self._auth_token: str = os.environ.get("MESH_AUTH_TOKEN", "")

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers: dict[str, str] = {}
            if self._auth_token:
                headers["Authorization"] = f"Bearer {self._auth_token}"
            self._client = httpx.AsyncClient(timeout=120, headers=headers)
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def _get_thinking_params(self, model: str | None = None) -> dict:
        """Build provider-specific thinking/reasoning parameters."""
        if self.thinking == "off":
            return {}
        m = model or self.default_model
        if m.startswith("anthropic/"):
            budget = self._THINKING_BUDGETS.get(self.thinking, 10_000)
            # Anthropic requires max_tokens > budget_tokens; ensure enough room
            # for both thinking and output text.
            return {
                "thinking": {"type": "enabled", "budget_tokens": budget},
                "temperature": 1.0,
                "max_tokens": budget + 4096,
            }
        if m.startswith("openai/o") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4"):
            return {"reasoning_effort": self.thinking}
        return {}

    async def chat(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs,
    ) -> LLMResponse:
        """Send a chat completion request through the mesh proxy."""
        params: dict = {
            "model": model or self.default_model,
            "messages": [{"role": "system", "content": system}] + messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            params["tools"] = tools
        params.update(kwargs)
        thinking_params = self._get_thinking_params(model)
        for k, v in thinking_params.items():
            if k not in kwargs:
                params[k] = v

        request = APIProxyRequest(service="llm", action="chat", params=params, timeout=120)

        from src.shared.trace import trace_headers

        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/api",
            json=request.model_dump(mode="json"),
            params={"agent_id": self.agent_id},
            headers=trace_headers(),
        )
        response.raise_for_status()
        data = response.json()

        if not data.get("success"):
            raise RuntimeError(f"LLM call failed: {data.get('error')}")

        result = data["data"]
        tool_calls = []
        for tc in result.get("tool_calls", []):
            args = tc["arguments"]
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    logger.warning(f"Malformed tool arguments for {tc['name']}, using raw string")
                    args = {"raw": args}
            tool_calls.append(ToolCallInfo(name=tc["name"], arguments=args))

        return LLMResponse(
            content=result.get("content", ""),
            thinking_content=result.get("thinking_content"),
            tool_calls=tool_calls if tool_calls else None,
            tokens_used=result.get("tokens_used", 0),
            model=result.get("model", ""),
        )

    async def chat_stream(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs,
    ) -> AsyncIterator[dict]:
        """Stream a chat completion via the mesh proxy SSE endpoint.

        Yields dicts:
          {"type": "text_delta", "content": str}   — incremental token
          {"type": "done", "response": LLMResponse} — final assembled response
        On error, falls back by raising so caller can retry non-streaming.
        """
        params: dict = {
            "model": model or self.default_model,
            "messages": [{"role": "system", "content": system}] + messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            params["tools"] = tools
        params.update(kwargs)
        thinking_params = self._get_thinking_params(model)
        for k, v in thinking_params.items():
            if k not in kwargs:
                params[k] = v

        request = APIProxyRequest(service="llm", action="chat", params=params, timeout=120)

        from src.shared.trace import trace_headers

        client = await self._get_client()
        async with client.stream(
            "POST",
            f"{self.mesh_url}/mesh/api/stream",
            json=request.model_dump(mode="json"),
            params={"agent_id": self.agent_id},
            headers=trace_headers(),
            timeout=120,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                if "error" in data:
                    raise RuntimeError(f"LLM stream error: {data['error']}")

                if data.get("type") == "text_delta":
                    yield {"type": "text_delta", "content": data.get("content", "")}

                elif data.get("type") == "done":
                    # Assemble final LLMResponse from the done event
                    tool_calls = []
                    for tc in data.get("tool_calls", []):
                        args = tc.get("arguments", "")
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                args = {"raw": args}
                        tool_calls.append(ToolCallInfo(name=tc.get("name", ""), arguments=args))

                    llm_resp = LLMResponse(
                        content=data.get("content", ""),
                        thinking_content=data.get("thinking_content"),
                        tool_calls=tool_calls if tool_calls else None,
                        tokens_used=data.get("tokens_used", 0),
                        model=data.get("model", ""),
                    )
                    yield {"type": "done", "response": llm_resp}

    async def embed(self, text: str, model: str | None = None) -> list[float]:
        """Generate an embedding vector through the mesh proxy."""
        request = APIProxyRequest(
            service="llm",
            action="embed",
            params={"model": model or self.embedding_model, "text": text},
            timeout=30,
        )
        from src.shared.trace import trace_headers

        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/api",
            json=request.model_dump(mode="json"),
            params={"agent_id": self.agent_id},
            headers=trace_headers(),
        )
        response.raise_for_status()
        data = response.json()
        if not data.get("success"):
            raise RuntimeError(f"Embedding call failed: {data.get('error')}")
        return data["data"]["embedding"]
