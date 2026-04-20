"""Provider factory — selects the right provider for a given model string."""
from .anthropic import AnthropicProvider
from .litellm import LiteLLMProvider
from .base import LLMProvider


_PROVIDERS: list[type[LLMProvider]] = [
    AnthropicProvider,
    LiteLLMProvider,  # fallback — handles everything via LiteLLM
]


def get_provider(model: str, **credentials) -> LLMProvider:
    """Return the appropriate provider instance for the given model string.

    Credentials are passed as kwargs:
      - ``api_key``: standard API key (sk-ant-... or other provider key)
      - ``auth_token``: OAuth token (sk-ant-oat01-...)

    If no credentials are passed, the provider resolves them from environment
    variables (e.g. CLAUDE_CODE_OAUTH_TOKEN, ANTHROPIC_API_KEY).
    """
    for provider_cls in _PROVIDERS:
        instance = provider_cls(**credentials)
        if instance.supports_model(model):
            return instance
    raise ValueError(f"No provider found for model: {model}")
