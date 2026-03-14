"""Dynamic model registry backed by LiteLLM.

Provides model costs, context windows, and provider model lists from
litellm.model_cost (~2,600 models).  All litellm imports are lazy so this
module works in agent containers where litellm isn't installed.

Public API:
    get_model_cost(model)        -> (input_per_1k, output_per_1k)
    get_context_window(model)    -> int
    get_provider_models(provider) -> list[str]
    get_all_model_costs()        -> dict[str, tuple[float, float]]
"""

from __future__ import annotations

from functools import lru_cache

# ── Fallback data (used when litellm is unavailable) ─────────

_FALLBACK_COSTS: dict[str, tuple[float, float]] = {
    "openai/gpt-4o": (0.0025, 0.01),
    "openai/gpt-4o-mini": (0.00015, 0.0006),
    "openai/gpt-4.1": (0.002, 0.008),
    "openai/gpt-4.1-mini": (0.0004, 0.0016),
    "openai/gpt-4.1-nano": (0.0001, 0.0004),
    "openai/o3": (0.002, 0.008),
    "openai/o3-mini": (0.0011, 0.0044),
    "openai/o4-mini": (0.0011, 0.0044),
    "anthropic/claude-opus-4-6": (0.005, 0.025),
    "anthropic/claude-sonnet-4-6": (0.003, 0.015),
    "anthropic/claude-sonnet-4-5-20250929": (0.003, 0.015),
    "anthropic/claude-haiku-4-5-20251001": (0.0008, 0.004),
    "minimax/MiniMax-M2.5": (0.0003, 0.0012),
    "minimax/MiniMax-M2.1": (0.001, 0.005),
    "minimax/MiniMax-M2.1-lightning": (0.0005, 0.002),
    "minimax/MiniMax-M2": (0.001, 0.005),
    "minimax/MiniMax-M2.5-Lightning": (0.0003, 0.0024),
    "zai/glm-5": (0.001, 0.0032),
}

_DEFAULT_COST = (0.003, 0.015)

_FALLBACK_CONTEXT: dict[str, int] = {
    "openai/gpt-4o": 128_000,
    "openai/gpt-4o-mini": 128_000,
    "openai/gpt-4.1": 1_047_576,
    "openai/gpt-4.1-mini": 1_047_576,
    "openai/gpt-4.1-nano": 1_047_576,
    "openai/o3": 200_000,
    "openai/o3-mini": 200_000,
    "openai/o4-mini": 200_000,
    "anthropic/claude-opus-4-6": 200_000,
    "anthropic/claude-sonnet-4-6": 200_000,
    "anthropic/claude-sonnet-4-5-20250929": 200_000,
    "anthropic/claude-haiku-4-5-20251001": 200_000,
}

_DEFAULT_CONTEXT_WINDOW = 128_000

# Featured models: curated ordering for UI dropdowns.
# These always appear first for each provider.
_FEATURED_MODELS: dict[str, list[str]] = {
    "openai": [
        "openai/gpt-5.2",
        "openai/gpt-5.2-pro",
        "openai/gpt-5.1-codex",
        "openai/gpt-5",
        "openai/gpt-5-mini",
        "openai/o3",
        "openai/o4-mini",
        "openai/gpt-4.1",
        "openai/gpt-4.1-mini",
    ],
    "anthropic": [
        "anthropic/claude-opus-4-6",
        "anthropic/claude-sonnet-4-6",
        "anthropic/claude-sonnet-4-5-20250929",
        "anthropic/claude-haiku-4-5-20251001",
    ],
    "gemini": [
        "gemini/gemini-3-pro-preview",
        "gemini/gemini-3-flash-preview",
        "gemini/gemini-2.5-pro",
        "gemini/gemini-2.5-flash",
    ],
    "xai": [
        "xai/grok-4-1-fast-reasoning",
        "xai/grok-4",
        "xai/grok-3",
        "xai/grok-3-mini",
    ],
    "deepseek": [
        "deepseek/deepseek-chat",
        "deepseek/deepseek-reasoner",
    ],
    "moonshot": [
        "moonshot/kimi-k2.5",
        "moonshot/kimi-k2",
        "moonshot/moonshot-v1-128k",
    ],
    "groq": [
        "groq/llama-3.3-70b-versatile",
        "groq/llama-3.1-8b-instant",
        "groq/llama-3-groq-70b-tool-use",
    ],
    "minimax": [
        "minimax/MiniMax-M2.5",
        "minimax/MiniMax-M2.5-Lightning",
        "minimax/MiniMax-M2.1",
        "minimax/MiniMax-M2.1-lightning",
        "minimax/MiniMax-M2",
    ],
    "zai": [
        "zai/glm-5",
    ],
    "ollama": [
        "ollama/llama3.3",
        "ollama/qwen2.5",
        "ollama/deepseek-r1",
        "ollama/phi4",
        "ollama/mistral",
        "ollama/gemma2",
        "ollama/command-r",
        "ollama/codellama",
    ],
}

# Providers that use bare model names in litellm (no prefix).
# For these, "openai/gpt-4.1" resolves to litellm key "gpt-4.1".
_BARE_NAME_PROVIDERS = {"openai", "anthropic"}

# Exclude these patterns from auto-discovered models
_EXCLUDE_PATTERNS = (
    "embed", "tts", "whisper", "dall-e", "image", "audio",
    "moderation", "ft:", "fine_tune", "-instruct-preview",
)


def _resolve_litellm_key(model: str) -> str | None:
    """Resolve our ``provider/model`` format to a litellm model_cost key.

    LiteLLM uses bare names for OpenAI/Anthropic (``gpt-4.1``,
    ``claude-sonnet-4-6``) but prefixed names for others
    (``xai/grok-3``, ``gemini/gemini-2.5-pro``).

    Returns the matching key, or None if not found.
    """
    try:
        from litellm import model_cost
    except ImportError:
        return None

    # Exact match first (works for prefixed providers)
    if model in model_cost:
        return model

    # Strip provider prefix for bare-name providers only
    if "/" in model:
        provider = model.split("/", 1)[0]
        if provider in _BARE_NAME_PROVIDERS:
            bare = model.split("/", 1)[1]
            if bare in model_cost:
                return bare

    return None


def get_model_cost(model: str) -> tuple[float, float]:
    """Return ``(input_per_1k, output_per_1k)`` USD cost for a model.

    Converts litellm's per-token costs to our per-1K-token format.
    Falls back to hardcoded data, then a conservative default.
    Local providers (Ollama) are always free.
    """
    if model.startswith("ollama/") or model.startswith("ollama_chat/"):
        return (0.0, 0.0)

    key = _resolve_litellm_key(model)
    if key is not None:
        try:
            from litellm import model_cost
            info = model_cost[key]
            input_per_token = info.get("input_cost_per_token", 0) or 0
            output_per_token = info.get("output_cost_per_token", 0) or 0
            return (input_per_token * 1000, output_per_token * 1000)
        except (ImportError, KeyError):
            pass

    return _FALLBACK_COSTS.get(model, _DEFAULT_COST)


_OLLAMA_DEFAULT_CONTEXT = 4096


def get_context_window(model: str) -> int:
    """Return the max input tokens for a model.

    Uses litellm's ``max_input_tokens``, falling back to hardcoded data.
    Ollama models default to 4096 (Ollama's default ``num_ctx``).
    """
    if model.startswith("ollama/") or model.startswith("ollama_chat/"):
        return _OLLAMA_DEFAULT_CONTEXT

    key = _resolve_litellm_key(model)
    if key is not None:
        try:
            from litellm import model_cost
            info = model_cost[key]
            max_input = info.get("max_input_tokens")
            if max_input and max_input > 0:
                return max_input
        except (ImportError, KeyError):
            pass

    return _FALLBACK_CONTEXT.get(model, _DEFAULT_CONTEXT_WINDOW)


@lru_cache(maxsize=32)
def get_provider_models(provider: str) -> list[str]:
    """Return chat models for a provider in ``provider/name`` format.

    Featured models (curated) appear first, then additional chat models
    from litellm sorted alphabetically.  Non-chat models are excluded.
    """
    featured = list(_FEATURED_MODELS.get(provider, []))
    featured_set = set(featured)

    try:
        from litellm import model_cost
    except ImportError:
        return featured

    extra: list[str] = []
    is_bare = provider in _BARE_NAME_PROVIDERS
    prefix = f"{provider}/"

    for litellm_key, info in model_cost.items():
        if info.get("mode") != "chat":
            continue

        # Determine if this model belongs to our provider
        if is_bare:
            # For openai/anthropic: look for bare names matching the provider's naming
            if "/" in litellm_key:
                continue  # Skip prefixed entries for bare-name providers
            # Heuristic: openai models start with gpt/o/chatgpt, anthropic with claude
            if provider == "openai" and not (
                litellm_key.startswith("gpt-")
                or litellm_key.startswith("chatgpt-")
                or (litellm_key[:1] == "o" and len(litellm_key) > 1 and litellm_key[1:2].isdigit())
            ):
                continue
            if provider == "anthropic" and not litellm_key.startswith("claude"):
                continue
            canonical = f"{provider}/{litellm_key}"
        else:
            if not litellm_key.startswith(prefix):
                continue
            canonical = litellm_key  # Already in provider/name format

        # Skip excluded patterns
        lower = canonical.lower()
        if any(pat in lower for pat in _EXCLUDE_PATTERNS):
            continue

        if canonical not in featured_set:
            extra.append(canonical)

    extra.sort()
    return featured + extra


def get_all_model_costs() -> dict[str, tuple[float, float]]:
    """Return costs for all models across our provider list.

    Used by the dashboard settings endpoint.
    """
    providers = list(_FEATURED_MODELS.keys())
    result: dict[str, tuple[float, float]] = {}
    for provider in providers:
        for model in get_provider_models(provider):
            result[model] = get_model_cost(model)
    return result
