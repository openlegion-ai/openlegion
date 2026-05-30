"""LLM client for agents. All calls route through the mesh API proxy.

Agents have NO external network access and hold NO API keys.
The mesh tracks token usage and enforces budgets.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator

import httpx

from src.shared.types import APIProxyRequest, LLMResponse, ToolCallInfo
from src.shared.utils import setup_logging

logger = setup_logging("agent.llm")


class LLMRetryableError(RuntimeError):
    """LLM error that should be retried (rate limits, transient failures)."""

    pass


class LLMClient:
    """LLM interface that routes all calls through the mesh API proxy."""

    _THINKING_BUDGETS = {"low": 5_000, "medium": 10_000, "high": 25_000}
    VALID_THINKING_LEVELS = {"off", "low", "medium", "high"}

    def __init__(
        self,
        mesh_url: str,
        agent_id: str = "agent",
        default_model: str = "openai/gpt-4o-mini",
        embedding_model: str = "",
        thinking: str = "off",
        max_output_tokens: int = 8192,
    ):
        if thinking and thinking not in self.VALID_THINKING_LEVELS:
            logger.warning(
                "Invalid thinking level '%s', falling back to 'off'. "
                "Valid: %s", thinking, ", ".join(sorted(self.VALID_THINKING_LEVELS)),
            )
            thinking = "off"
        self.mesh_url = mesh_url
        self.agent_id = agent_id
        self.default_model = default_model
        self.embedding_model = embedding_model
        self.thinking = thinking
        # Default output-token cap for chat()/chat_stream() when the caller
        # doesn't pass max_tokens explicitly. Configurable per-agent via the
        # LLM_MAX_TOKENS env var (see __main__.py) and hot-reloadable via the
        # agent /config endpoint. The legacy hardcoded 4096 was too small for
        # tool calls with large argument payloads (e.g. write_file of a
        # 15-20KB file): the model hit the cap mid-tool-call, the JSON never
        # closed, and the call failed permanently with "Truncated tool-call
        # arguments". 8192 is the safe floor across common modern models
        # (gpt-4o family = 16384, Claude 3.5 Sonnet = 8192, Claude 4.x = 64K+).
        self.max_output_tokens = max_output_tokens
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()
        self._auth_token: str = os.environ.get("MESH_AUTH_TOKEN", "")

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is not None and not self._client.is_closed:
            return self._client
        async with self._client_lock:
            # Double-check after acquiring lock
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
        if m.startswith("openrouter/"):
            m = m[len("openrouter/"):]
        if m.startswith("openlegion/"):
            m = m[len("openlegion/"):]
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

    def _apply_thinking_params(self, params: dict, model: str | None, kwargs: dict) -> None:
        """Merge thinking/reasoning params into *params*, respecting caller overrides."""
        for k, v in self._get_thinking_params(model).items():
            if k not in kwargs:
                params[k] = v

    @staticmethod
    def _raise_classified_error(
        error_msg: str,
        error_type: str | None,
        error_meta: dict,
        *,
        prefix: str,
    ) -> None:
        """Raise the typed exception for a mesh error payload. Always raises.

        Fix 3 (seam follow-up): the mesh proxy tags distinguished exceptions on
        APIProxyResponse.error_type so we can re-raise the same type the agent
        loop expects to catch. Without this, every credential failure looks
        like a generic RuntimeError to the loop and the quarantine path doesn't
        trigger via the agent's report channel (mesh also records directly, but
        the agent-side branches still need the type).

        Shared by ``chat`` (``prefix="LLM call failed"``) and ``chat_stream``
        (``prefix="LLM stream error"``) so the streaming path gets the same
        ``transient`` + substring-backstop retry handling the non-streaming path
        has always had.
        """
        if error_type == "auth_failure":
            from src.shared.errors import LLMAuthError
            raise LLMAuthError(
                error_msg,
                provider=error_meta.get("provider", "unknown"),
                model=error_meta.get("model"),
                http_status=error_meta.get("http_status"),
            )
        if error_type == "config_error":
            from src.shared.errors import LLMConfigError
            raise LLMConfigError(
                error_msg,
                provider=error_meta.get("provider", "unknown"),
                model=error_meta.get("model", ""),
                allowed_models=set(error_meta.get("allowed_models", [])),
                http_status=error_meta.get("http_status"),
            )
        if error_type == "transient":
            # Mesh-tagged transient (Claude subscription throttle, stream
            # interruption, empty-choices response). The mesh already
            # classified it at the source — surface as retryable so
            # ``_llm_call_with_retry`` backs off.
            raise LLMRetryableError(f"{prefix}: {error_msg}")
        # Backstop for un-tagged transient signals — third-party SDKs, future
        # wrapper sites not yet routed through the typed channel, or paths the
        # mesh outer handler didn't catch. Each substring corresponds to a known
        # transient pattern; ``retrying may help`` is the deliberate suffix
        # appended by ``friendly_streaming_error`` (src/shared/utils.py:78), so
        # matching on it recognizes the helper's contract rather than
        # whack-a-mole'ing message text.
        _retryable = (
            "rate limit", "ratelimit", "429", "too many requests",
            "overloaded", "503", "empty response", "retrying may help",
        )
        if any(kw in error_msg.lower() for kw in _retryable):
            raise LLMRetryableError(f"{prefix}: {error_msg}")
        raise RuntimeError(f"{prefix}: {error_msg}")

    @staticmethod
    def _parse_tool_calls(raw_calls: list[dict]) -> list[ToolCallInfo] | None:
        """Parse raw tool-call dicts into ToolCallInfo objects."""
        tool_calls: list[ToolCallInfo] = []
        for tc in raw_calls:
            args = tc.get("arguments", "")
            if args is None:
                args = {}
            elif isinstance(args, str):
                if args.strip() == "":
                    # Legitimate no-arg tool call (e.g. check_inbox()).
                    # Empty / whitespace-only arguments are valid and MUST
                    # keep working — do not raise.
                    args = {}
                else:
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError as e:
                        # Non-empty arguments that won't parse means the tool
                        # call was truncated mid-stream (provider hit the output
                        # token cap mid-tool-call, or the stream dropped before
                        # the JSON closed). Treat as retryable so
                        # ``_llm_call_with_retry``'s backoff re-issues the call
                        # (and the streaming caller falls back to non-streaming
                        # where a complete response can assemble) rather than
                        # dispatching a ``{"raw": <truncated>}`` stub that fails
                        # downstream with a confusing missing-required-param
                        # error. (Incident: ``edit_agent`` received a payload
                        # whose JSON cut off mid-string, e.g.
                        # ``{"agent_id": "page-validator", "field": ...``.)
                        raise LLMRetryableError(
                            f"Truncated tool-call arguments for "
                            f"{tc.get('name')!r}: {args[:200]}"
                        ) from e
            # json.loads can return non-dict types (null→None, "42"→int,
            # "[1,2]"→list).  Normalise to dict so ToolCallInfo validation
            # and downstream execute() don't crash.
            if not isinstance(args, dict):
                args = {}
            tool_calls.append(ToolCallInfo(name=tc.get("name", ""), arguments=args))
        return tool_calls if tool_calls else None

    async def chat(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> LLMResponse:
        """Send a chat completion request through the mesh proxy."""
        if max_tokens is None:
            max_tokens = self.max_output_tokens
        params: dict = {
            "model": model or self.default_model,
            "messages": [{"role": "system", "content": system}] + messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            params["tools"] = tools
        params.update(kwargs)
        self._apply_thinking_params(params, model, kwargs)

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
            self._raise_classified_error(
                data.get("error", ""),
                data.get("error_type"),
                data.get("error_meta") or {},
                prefix="LLM call failed",
            )

        result = data["data"]
        return LLMResponse(
            content=result.get("content", ""),
            thinking_content=result.get("thinking_content"),
            tool_calls=self._parse_tool_calls(result.get("tool_calls", [])),
            tokens_used=result.get("tokens_used", 0),
            model=result.get("model", ""),
        )

    async def chat_stream(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> AsyncIterator[dict]:
        """Stream a chat completion via the mesh proxy SSE endpoint.

        Yields dicts:
          {"type": "text_delta", "content": str}   — incremental token
          {"type": "done", "response": LLMResponse} — final assembled response
        On error, falls back by raising so caller can retry non-streaming.
        """
        if max_tokens is None:
            max_tokens = self.max_output_tokens
        params: dict = {
            "model": model or self.default_model,
            "messages": [{"role": "system", "content": system}] + messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            params["tools"] = tools
        params.update(kwargs)
        self._apply_thinking_params(params, model, kwargs)

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
                    # Distinguished error types ride the SSE payload so the
                    # agent loop can route to the auth-failure / config-error
                    # branches. Shared helper also adds the ``transient`` +
                    # substring-backstop retry handling that ``chat`` has —
                    # closing the streaming-path asymmetry (a mid-stream
                    # transient is now retried rather than raised as a bare
                    # RuntimeError).
                    self._raise_classified_error(
                        data["error"],
                        data.get("error_type"),
                        data.get("error_meta") or {},
                        prefix="LLM stream error",
                    )

                if data.get("type") == "text_delta":
                    yield {"type": "text_delta", "content": data.get("content", "")}

                elif data.get("type") == "done":
                    llm_resp = LLMResponse(
                        content=data.get("content", ""),
                        thinking_content=data.get("thinking_content"),
                        tool_calls=self._parse_tool_calls(data.get("tool_calls", [])),
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
