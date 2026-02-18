"""LLM client for agents. All calls route through the mesh API proxy.

Agents have NO external network access and hold NO API keys.
The mesh tracks token usage and enforces budgets.
"""

from __future__ import annotations

import json

import httpx

from src.shared.types import APIProxyRequest, LLMResponse, ToolCallInfo
from src.shared.utils import setup_logging

logger = setup_logging("agent.llm")


class LLMClient:
    """LLM interface that routes all calls through the mesh API proxy."""

    def __init__(self, mesh_url: str, agent_id: str = "agent", default_model: str = "openai/gpt-4o-mini"):
        self.mesh_url = mesh_url
        self.agent_id = agent_id
        self.default_model = default_model
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=120)
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def chat(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
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

        request = APIProxyRequest(service="llm", action="chat", params=params, timeout=120)

        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/api",
            json=request.model_dump(mode="json"),
            params={"agent_id": self.agent_id},
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
            tool_calls=tool_calls if tool_calls else None,
            tokens_used=result.get("tokens_used", 0),
            model=result.get("model", ""),
        )

    async def embed(self, text: str, model: str | None = None) -> list[float]:
        """Generate an embedding vector through the mesh proxy."""
        request = APIProxyRequest(
            service="llm",
            action="embed",
            params={"model": model or "text-embedding-3-small", "text": text},
            timeout=30,
        )
        client = await self._get_client()
        response = await client.post(
            f"{self.mesh_url}/mesh/api",
            json=request.model_dump(mode="json"),
            params={"agent_id": self.agent_id},
        )
        response.raise_for_status()
        data = response.json()
        if not data.get("success"):
            raise RuntimeError(f"Embedding call failed: {data.get('error')}")
        return data["data"]["embedding"]
