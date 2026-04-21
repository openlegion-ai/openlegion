"""Provider factory — selects the right provider for a given model string.

Controlled by the ``OPENLEGION_SYSTEM_LLMPROVIDER`` environment variable:

  LITELLM (default)    — LiteLLMProvider handles all models. AnthropicProvider
                         is still tried first for Claude Code OAuth tokens
                         (sk-ant-oat01-...) because LiteLLM cannot send the
                         required OAuth headers.
  ANTHROPIC_SDK        — AnthropicProvider is used for all anthropic/* and
                         claude-* models. Falls through to LiteLLM for other
                         providers.
  <anything else>      — raises ValueError at startup.
"""
import os

from .anthropic import AnthropicProvider
from .litellm import LiteLLMProvider
from .base import LLMProvider

_VALID_PROVIDERS = ("LITELLM", "ANTHROPIC_SDK")


def _get_provider_setting() -> str:
    val = os.getenv("OPENLEGION_SYSTEM_LLMPROVIDER", "LITELLM").upper()
    if val not in _VALID_PROVIDERS:
        raise ValueError(
            f"Unknown OPENLEGION_SYSTEM_LLMPROVIDER={val!r}. "
            f"Valid values: {', '.join(_VALID_PROVIDERS)}"
        )
    return val


def get_provider(model: str, **credentials) -> LLMProvider:
    """Return the appropriate provider instance for the given model string."""
    setting = _get_provider_setting()

    if setting == "ANTHROPIC_SDK":
        # Explicit Anthropic SDK — use AnthropicProvider for Anthropic models,
        # fall through to LiteLLM for everything else (e.g. openai/*, gemini/*).
        # supports_model() check is skipped here — the operator explicitly asked
        # for the SDK, so we honour that even without an OAuth token (a standard
        # API key via ANTHROPIC_API_KEY will be picked up by the client).
        is_anthropic = model.startswith("anthropic/") or model.startswith("claude-")
        if is_anthropic:
            return AnthropicProvider(**credentials)
        return LiteLLMProvider(**credentials)

    # LITELLM (default) — still try AnthropicProvider first so that Claude Code
    # OAuth tokens (sk-ant-oat01-...) get the correct OAuth headers. If no OAuth
    # token is present, supports_model() returns False and we fall through.
    candidate = AnthropicProvider(**credentials)
    if candidate.supports_model(model):
        return candidate
    return LiteLLMProvider(**credentials)
