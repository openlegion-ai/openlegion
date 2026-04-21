"""Anthropic SDK provider — Claude Code subscription OAuth tokens only.

Standard Anthropic API key calls (sk-ant-...) are handled by LiteLLMProvider,
which already supports them natively. This provider only activates when a
Claude Code subscription OAuth token (sk-ant-oat01-...) is available, because
those tokens require special headers that LiteLLM does not send:
  - anthropic-beta: oauth-2025-04-20
  - x-app: cli
  - anthropic-dangerous-direct-browser-access: true

Token resolution order:
  1. ``auth_token`` kwarg
  2. ``CLAUDE_CODE_OAUTH_TOKEN`` env var
  3. ``OPENLEGION_SYSTEM_ANTHROPIC_OAUTH`` env var (JSON with access_token field)

When none of these yield an OAuth token, ``supports_model()`` returns False and
the factory falls through to LiteLLMProvider.
"""

from __future__ import annotations

import json
import os
from typing import AsyncIterator, Any

from src.shared.utils import setup_logging
from .base import LLMProvider, LLMResponse, StreamChunk

logger = setup_logging("host.providers.anthropic")

# OAuth token prefix — tokens from `claude setup-token`
_OAUTH_TOKEN_PREFIX = "sk-ant-oat01-"

# Headers required for Claude Code OAuth bearer auth
_CLAUDE_CLI_VERSION = "2.1.84"

# Required Claude Code identity system block for OAuth requests
_CLAUDE_CODE_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."

# thinking budget_tokens map for named levels
_THINKING_BUDGET: dict[str, int] = {
    "off": 0,
    "low": 1024,
    "medium": 8000,
    "high": 32000,
}


def _is_oauth_token(token: str) -> bool:
    """Return True if token is a Claude Code subscription OAuth token."""
    return token.startswith(_OAUTH_TOKEN_PREFIX)


def _normalize_tool_schema_for_anthropic(schema: object) -> object:
    """Normalize tool JSON Schema fragments for Anthropic requests.

    Anthropic's tool ``input_schema`` validator rejects array-valued ``type``
    fields (e.g. ``"type": ["string", "object"]``) even though they are valid
    JSON Schema. Convert them to the equivalent ``anyOf`` form, which Anthropic
    does accept. Preserves ``null`` member types — Anthropic supports
    nullability via ``anyOf`` with a ``null`` branch.
    """
    if isinstance(schema, list):
        return [_normalize_tool_schema_for_anthropic(item) for item in schema]
    if not isinstance(schema, dict):
        return schema

    normalized = {
        key: _normalize_tool_schema_for_anthropic(value)
        for key, value in schema.items()
    }

    schema_type = normalized.get("type")
    if isinstance(schema_type, list):
        types = [t for t in schema_type if isinstance(t, str)]
        if not types:
            normalized.pop("type", None)
        elif len(types) == 1:
            normalized["type"] = types[0]
        elif any(k in normalized for k in ("anyOf", "oneOf", "allOf")):
            normalized.pop("type", None)
        else:
            normalized.pop("type", None)
            normalized["anyOf"] = [{"type": t} for t in types]

    return normalized


def _convert_openai_image_blocks(content: list) -> list:
    """Convert OpenAI image_url content blocks to Anthropic image format.

    Imports lazily to avoid circular imports — this mirrors the logic in
    src/agent/attachments.py:convert_openai_image_blocks.
    """
    try:
        from src.agent.attachments import convert_openai_image_blocks
        return convert_openai_image_blocks(content)
    except ImportError:
        # Fallback: pass through unchanged
        return content


def _build_anthropic_body(params: dict) -> dict:
    """Convert LiteLLM/OpenAI-style params to Anthropic Messages API format.

    Extracted from CredentialVault._build_anthropic_body in credentials.py.
    """
    messages = params.get("messages", [])
    model = params.get("model", "").removeprefix("anthropic/")

    system_parts: list[str] = []
    non_system: list[dict] = []
    for m in messages:
        if m.get("role") == "system":
            c = m.get("content", "")
            if isinstance(c, str):
                system_parts.append(c)
            elif isinstance(c, list):
                system_parts.append(" ".join(
                    b.get("text", "") for b in c if isinstance(b, dict)
                ))
        else:
            non_system.append(m)

    # Convert OpenAI-format tool messages to Anthropic Messages API format.
    converted: list[dict] = []
    for m in non_system:
        role = m.get("role", "")

        if role == "assistant" and m.get("tool_calls"):
            # Assistant + tool_calls → tool_use content blocks
            content_blocks: list[dict] = []
            text = m.get("content", "")
            if isinstance(text, str) and text:
                content_blocks.append({"type": "text", "text": text})
            elif isinstance(text, list):
                content_blocks.extend(text)
            for tc in m["tool_calls"]:
                func = tc.get("function", tc)
                args = func.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, ValueError):
                        args = {"raw": args}
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id", ""),
                    "name": func.get("name", ""),
                    "input": args,
                })
            converted.append({"role": "assistant", "content": content_blocks})

        elif role == "tool":
            # Tool result → user message with tool_result block
            raw_content = m.get("content", "")
            if isinstance(raw_content, list):
                raw_content = _convert_openai_image_blocks(raw_content)
            tool_result = {
                "type": "tool_result",
                "tool_use_id": m.get("tool_call_id", ""),
                "content": raw_content,
            }
            # Merge consecutive tool results into one user message
            if (converted
                    and converted[-1].get("role") == "user"
                    and isinstance(converted[-1].get("content"), list)):
                converted[-1]["content"].append(tool_result)
            else:
                converted.append({"role": "user", "content": [tool_result]})

        else:
            # Convert OpenAI image_url blocks to Anthropic image format
            if isinstance(m.get("content"), list):
                m = {**m, "content": _convert_openai_image_blocks(m["content"])}
            converted.append(m)

    body: dict = {
        "model": model,
        "messages": converted,
        "max_tokens": params.get("max_tokens", 4096),
    }
    if system_parts:
        body["system"] = "\n\n".join(system_parts)

    temp = params.get("temperature")
    if temp is not None:
        body["temperature"] = temp
    top_p = params.get("top_p")
    if top_p is not None:
        body["top_p"] = top_p

    # Convert OpenAI function-calling tools to Anthropic format
    tools = params.get("tools")
    if tools:
        anthropic_tools = []
        for t in tools:
            if not isinstance(t, dict):
                anthropic_tools.append(t)
                continue
            func = t.get("function")
            if isinstance(func, dict) and "parameters" in func:
                anthropic_tools.append({
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "input_schema": _normalize_tool_schema_for_anthropic(
                        func["parameters"]
                    ),
                })
            elif "input_schema" in t:
                out = {k: v for k, v in t.items() if k not in ("function", "type")}
                out["input_schema"] = _normalize_tool_schema_for_anthropic(t["input_schema"])
                anthropic_tools.append(out)
            else:
                anthropic_tools.append(t)
        body["tools"] = anthropic_tools

    # Tool choice — convert OpenAI format to Anthropic format
    tool_choice = params.get("tool_choice")
    if tool_choice is not None and tools:
        if tool_choice == "auto":
            body["tool_choice"] = {"type": "auto"}
        elif tool_choice == "required":
            body["tool_choice"] = {"type": "any"}
        elif tool_choice == "none":
            body.pop("tools", None)
        elif isinstance(tool_choice, dict) and "function" in tool_choice:
            body["tool_choice"] = {
                "type": "tool",
                "name": tool_choice["function"]["name"],
            }

    # Extended thinking — map named levels or pass through dicts directly
    thinking = params.get("thinking")
    if thinking:
        if isinstance(thinking, str):
            budget = _THINKING_BUDGET.get(thinking, 0)
            if budget > 0:
                body["thinking"] = {"type": "enabled", "budget_tokens": budget}
            # "off" → omit thinking entirely
        elif isinstance(thinking, dict):
            body["thinking"] = thinking

    return body


def _patch_oauth_body(body: dict) -> None:
    """Patch an Anthropic body for OAuth bearer auth.

    1. Prepends the Claude Code identity as a mandatory first system block.
    2. Drops temperature/top_p if not exactly 1.0 (OAuth policy restriction).
    """
    identity = _CLAUDE_CODE_IDENTITY
    system_blocks: list[dict] = [{"type": "text", "text": identity}]
    existing = body.get("system", "")
    if existing:
        if isinstance(existing, str):
            system_blocks.append({"type": "text", "text": existing})
        elif isinstance(existing, list):
            system_blocks.extend(existing)
    body["system"] = system_blocks

    temp = body.get("temperature")
    if temp is not None and temp != 1.0:
        logger.debug("Anthropic OAuth: dropping temperature=%s (only 1.0 permitted)", temp)
        body.pop("temperature", None)
    top_p = body.get("top_p")
    if top_p is not None and top_p != 1.0:
        logger.debug("Anthropic OAuth: dropping top_p=%s (only 1.0 permitted)", top_p)
        body.pop("top_p", None)


def _make_sdk_kwargs(body: dict) -> dict:
    """Extract keyword arguments for the Anthropic SDK from a converted body."""
    sdk_kwargs: dict = {
        "model": body["model"],
        "messages": body["messages"],
        "max_tokens": body.get("max_tokens", 4096),
    }
    if "system" in body:
        sdk_kwargs["system"] = body["system"]
    if "temperature" in body:
        sdk_kwargs["temperature"] = body["temperature"]
    if "top_p" in body:
        sdk_kwargs["top_p"] = body["top_p"]
    if body.get("tools"):
        sdk_kwargs["tools"] = body["tools"]
    if "tool_choice" in body:
        sdk_kwargs["tool_choice"] = body["tool_choice"]
    if "thinking" in body:
        sdk_kwargs["thinking"] = body["thinking"]
    return sdk_kwargs


class AnthropicProvider(LLMProvider):
    """LLM provider for Claude Code subscription OAuth tokens.

    Only activates when an OAuth token (sk-ant-oat01-...) is present.
    Standard API key calls fall through to LiteLLMProvider.
    """

    def __init__(self, auth_token: str = "", **_kwargs) -> None:
        self._auth_token = auth_token or self._resolve_oauth_token()

    @staticmethod
    def _resolve_oauth_token() -> str:
        """Return the first available Claude Code OAuth token, or empty string."""
        if token := os.getenv("CLAUDE_CODE_OAUTH_TOKEN", ""):
            if _is_oauth_token(token):
                return token
        if oauth_json := os.getenv("OPENLEGION_SYSTEM_ANTHROPIC_OAUTH", ""):
            try:
                oauth = json.loads(oauth_json)
                if isinstance(oauth, dict):
                    token = oauth.get("access_token", "")
                    if _is_oauth_token(token):
                        return token
            except (json.JSONDecodeError, KeyError):
                pass
        return ""

    @property
    def name(self) -> str:
        return "anthropic-oauth"

    def supports_model(self, model: str) -> bool:
        """Only claim anthropic/* and claude-* models when an OAuth token is available."""
        if not self._auth_token:
            return False
        return model.startswith("anthropic/") or model.startswith("claude-")

    def _make_client(self):
        """Create an AsyncAnthropic client configured for OAuth bearer auth."""
        import anthropic
        return anthropic.AsyncAnthropic(
            api_key=None,
            auth_token=self._auth_token,
            default_headers={
                "accept": "application/json",
                "anthropic-dangerous-direct-browser-access": "true",
                "anthropic-beta": (
                    "claude-code-20250219,"
                    "oauth-2025-04-20,"
                    "fine-grained-tool-streaming-2025-05-14"
                ),
                "User-Agent": f"claude-cli/{_CLAUDE_CLI_VERSION}",
                "x-app": "cli",
            },
            max_retries=0,
            timeout=120.0,
        )

    async def complete(self, params: dict[str, Any]) -> LLMResponse:
        """Non-streaming completion via Anthropic SDK."""
        body = _build_anthropic_body(params)
        _patch_oauth_body(body)
        sdk_kwargs = _make_sdk_kwargs(body)
        model_str = f"anthropic/{body['model']}"
        client = self._make_client()
        try:
            response = await client.messages.create(**sdk_kwargs)
            content = ""
            thinking = ""
            tool_calls: list[dict] = []
            for block in response.content:
                btype = getattr(block, "type", "")
                if btype == "text":
                    content += getattr(block, "text", "")
                elif btype == "thinking":
                    thinking += getattr(block, "thinking", "")
                elif btype == "tool_use":
                    tool_calls.append({
                        "name": block.name,
                        "arguments": json.dumps(block.input),
                    })
            usage = response.usage
            return LLMResponse(
                content=content,
                tool_calls=tool_calls or None,
                thinking=thinking or None,
                model=model_str,
                input_tokens=usage.input_tokens if usage else 0,
                output_tokens=usage.output_tokens if usage else 0,
                finish_reason=response.stop_reason or "stop",
            )
        finally:
            await client.close()

    async def stream(self, params: dict[str, Any]) -> AsyncIterator[StreamChunk]:
        """Streaming completion via Anthropic SDK. Yields StreamChunk objects."""
        body = _build_anthropic_body(params)
        _patch_oauth_body(body)
        sdk_kwargs = _make_sdk_kwargs(body)
        model_str = f"anthropic/{body['model']}"

        collected_tool_calls: list[dict] = []
        current_tool_idx = -1
        input_tokens = 0
        output_tokens = 0

        client = self._make_client()
        try:
            stream = await client.messages.create(**sdk_kwargs, stream=True)

            async for event in stream:
                etype = event.type

                if etype == "message_start":
                    if hasattr(event, "message") and hasattr(event.message, "usage"):
                        input_tokens = event.message.usage.input_tokens

                elif etype == "content_block_start":
                    cb = event.content_block
                    if cb.type == "tool_use":
                        current_tool_idx += 1
                        collected_tool_calls.append({
                            "id": cb.id,
                            "name": cb.name,
                            "arguments": "",
                        })

                elif etype == "content_block_delta":
                    delta = event.delta
                    dtype = delta.type
                    if dtype == "text_delta":
                        yield StreamChunk(type="text", content=delta.text)
                    elif dtype == "thinking_delta":
                        yield StreamChunk(type="thinking", content=delta.thinking)
                    elif dtype == "input_json_delta":
                        if current_tool_idx >= 0:
                            collected_tool_calls[current_tool_idx]["arguments"] += delta.partial_json

                elif etype == "message_delta":
                    if hasattr(event, "usage"):
                        output_tokens = event.usage.output_tokens

            # Emit tool_use chunks for completed tool calls
            for tc in collected_tool_calls:
                yield StreamChunk(
                    type="tool_use",
                    tool_call={
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    },
                )

            yield StreamChunk(
                type="done",
                finish_reason="end_turn",
                tool_call=None,
            )

        except Exception as e:
            import anthropic
            if isinstance(e, anthropic.AuthenticationError):
                msg = "Auth failed"
                if hasattr(e, "body") and isinstance(e.body, dict):
                    msg = e.body.get("error", {}).get("message", msg)
                yield StreamChunk(
                    type="error",
                    content=f"OAuth auth failed (token may have expired): {msg}",
                )
            elif isinstance(e, anthropic.APIStatusError):
                detail = ""
                if hasattr(e, "body") and isinstance(e.body, dict):
                    detail = e.body.get("error", {}).get("message", "")
                msg = f"Anthropic API error (HTTP {e.status_code})"
                if detail:
                    msg += f": {detail}"
                yield StreamChunk(type="error", content=msg)
            else:
                yield StreamChunk(type="error", content=str(e))
        finally:
            await client.close()
