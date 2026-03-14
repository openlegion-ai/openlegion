"""Tests for the dynamic model registry (src/shared/models.py)."""

from __future__ import annotations

from unittest.mock import patch

from src.shared.models import (
    _DEFAULT_CONTEXT_WINDOW,
    _DEFAULT_COST,
    _FALLBACK_CONTEXT,
    _FALLBACK_COSTS,
    _FEATURED_MODELS,
    _resolve_litellm_key,
    get_all_model_costs,
    get_context_window,
    get_model_cost,
    get_provider_models,
)


class TestResolveLitellmKey:
    def test_exact_match_prefixed(self):
        """Prefixed keys like xai/grok-3 match directly."""
        key = _resolve_litellm_key("xai/grok-3")
        assert key == "xai/grok-3"

    def test_bare_name_fallback_openai(self):
        """OpenAI models resolve via bare name (gpt-4.1)."""
        key = _resolve_litellm_key("openai/gpt-4.1")
        assert key == "gpt-4.1"

    def test_bare_name_fallback_anthropic(self):
        """Anthropic models resolve via bare name (claude-sonnet-4-6)."""
        key = _resolve_litellm_key("anthropic/claude-sonnet-4-6")
        assert key == "claude-sonnet-4-6"

    def test_bare_name_only_for_openai_anthropic(self):
        """Non-bare-name providers (e.g. xai) don't try bare-name fallback."""
        # xai/grok-3 should resolve via exact match, not bare-name fallback
        key = _resolve_litellm_key("xai/grok-3")
        # If litellm has it as "xai/grok-3", exact match works.
        # But crucially, we should NOT strip prefix for non-bare-name providers.
        # Verify by testing a fake provider with a name that matches a bare key.
        from unittest.mock import patch as _patch
        fake_cost = {"gpt-4o": {"input_cost_per_token": 0.0025e-3}}
        with _patch("litellm.model_cost", fake_cost):
            # "fake/gpt-4o" should NOT resolve to bare "gpt-4o" since
            # "fake" is not in _BARE_NAME_PROVIDERS
            key = _resolve_litellm_key("fake/gpt-4o")
            assert key is None

    def test_unknown_model_returns_none(self):
        """Unknown model returns None."""
        key = _resolve_litellm_key("fake/nonexistent-model-xyz")
        assert key is None

    def test_no_litellm_returns_none(self):
        """Without litellm, returns None."""
        with patch.dict("sys.modules", {"litellm": None}):
            key = _resolve_litellm_key("openai/gpt-4o")
            assert key is None


class TestGetModelCost:
    def test_known_model_from_litellm(self):
        """Known model returns litellm costs converted to per-1K."""
        cost = get_model_cost("openai/gpt-4o")
        assert isinstance(cost, tuple)
        assert len(cost) == 2
        assert cost[0] > 0
        assert cost[1] > 0

    def test_per_token_to_per_1k_conversion(self):
        """Verify litellm per-token costs are multiplied by 1000."""
        import litellm
        info = litellm.model_cost.get("gpt-4o", {})
        expected_input = (info.get("input_cost_per_token", 0) or 0) * 1000
        expected_output = (info.get("output_cost_per_token", 0) or 0) * 1000
        cost = get_model_cost("openai/gpt-4o")
        assert abs(cost[0] - expected_input) < 1e-10
        assert abs(cost[1] - expected_output) < 1e-10

    def test_unknown_model_returns_default(self):
        """Unknown model returns the conservative default."""
        cost = get_model_cost("fake/nonexistent-model-xyz")
        assert cost == _DEFAULT_COST

    def test_fallback_without_litellm(self):
        """Without litellm, falls back to hardcoded costs."""
        with patch("src.shared.models._resolve_litellm_key", return_value=None):
            cost = get_model_cost("openai/gpt-4o")
            assert cost == _FALLBACK_COSTS["openai/gpt-4o"]

    def test_opus_cost_corrected(self):
        """Claude Opus 4.6 cost should reflect litellm's correct pricing."""
        cost = get_model_cost("anthropic/claude-opus-4-6")
        # litellm has $5/$25 per 1M = 0.005/0.025 per 1K
        assert cost[0] < 0.01  # Not the old wrong $15/1M = 0.015/1K


class TestGetContextWindow:
    def test_known_model(self):
        """Known model returns a positive context window."""
        window = get_context_window("openai/gpt-4o")
        assert window > 0

    def test_gpt41_large_context(self):
        """GPT-4.1 has ~1M context window."""
        window = get_context_window("openai/gpt-4.1")
        assert window >= 1_000_000

    def test_unknown_model_returns_default(self):
        """Unknown model returns the default 128K."""
        window = get_context_window("fake/nonexistent-model-xyz")
        assert window == _DEFAULT_CONTEXT_WINDOW

    def test_fallback_without_litellm(self):
        """Without litellm, falls back to hardcoded context windows."""
        with patch("src.shared.models._resolve_litellm_key", return_value=None):
            window = get_context_window("openai/gpt-4o")
            assert window == _FALLBACK_CONTEXT["openai/gpt-4o"]


class TestGetProviderModels:
    def test_featured_models_first(self):
        """Featured models appear first in the returned list."""
        models = get_provider_models.__wrapped__("anthropic")
        featured = _FEATURED_MODELS["anthropic"]
        for i, expected in enumerate(featured):
            assert models[i] == expected

    def test_extra_chat_models_included(self):
        """Additional chat models from litellm are appended."""
        models = get_provider_models.__wrapped__("anthropic")
        # litellm has many more claude models than our featured list
        assert len(models) > len(_FEATURED_MODELS["anthropic"])

    def test_non_chat_excluded(self):
        """Non-chat models (embeddings, audio) are not included."""
        models = get_provider_models.__wrapped__("openai")
        for m in models:
            lower = m.lower()
            assert "embed" not in lower
            assert "tts" not in lower
            assert "whisper" not in lower
            assert "dall-e" not in lower

    def test_fallback_returns_featured_only(self):
        """Without litellm, returns only featured models."""
        with patch.dict("sys.modules", {"litellm": None}):
            models = get_provider_models.__wrapped__("anthropic")
            assert models == _FEATURED_MODELS["anthropic"]

    def test_unknown_provider_returns_empty(self):
        """Unknown provider returns empty list."""
        models = get_provider_models.__wrapped__("nonexistent_provider")
        assert models == []

    def test_all_providers_return_models(self):
        """All featured providers return at least their featured models."""
        for provider in _FEATURED_MODELS:
            models = get_provider_models.__wrapped__(provider)
            assert len(models) >= len(_FEATURED_MODELS[provider])

    def test_models_have_provider_prefix(self):
        """All returned models have provider/ prefix."""
        models = get_provider_models.__wrapped__("openai")
        for m in models:
            assert m.startswith("openai/")


class TestOllamaModels:
    """Tests for Ollama (local) model support."""

    def test_ollama_cost_is_free(self):
        """Ollama models always return zero cost."""
        assert get_model_cost("ollama/llama3") == (0.0, 0.0)
        assert get_model_cost("ollama/some-custom-model:7b") == (0.0, 0.0)

    def test_ollama_chat_cost_is_free(self):
        """ollama_chat/ prefix also returns zero cost."""
        assert get_model_cost("ollama_chat/llama3") == (0.0, 0.0)

    def test_ollama_context_window(self):
        """Ollama models return the default Ollama context window."""
        from src.shared.models import _OLLAMA_DEFAULT_CONTEXT
        assert get_context_window("ollama/llama3") == _OLLAMA_DEFAULT_CONTEXT
        assert get_context_window("ollama_chat/llama3") == _OLLAMA_DEFAULT_CONTEXT

    def test_ollama_featured_models_exist(self):
        """Ollama has featured models defined."""
        assert "ollama" in _FEATURED_MODELS
        assert len(_FEATURED_MODELS["ollama"]) > 0
        for m in _FEATURED_MODELS["ollama"]:
            assert m.startswith("ollama/")

    def test_ollama_provider_models_start_with_featured(self):
        """get_provider_models returns featured models first for ollama."""
        models = get_provider_models.__wrapped__("ollama")
        featured = _FEATURED_MODELS["ollama"]
        assert models[: len(featured)] == featured


class TestGetAllModelCosts:
    def test_returns_dict_with_costs(self):
        """Returns a dict mapping model names to cost tuples."""
        costs = get_all_model_costs()
        assert isinstance(costs, dict)
        assert len(costs) > 0
        for model, cost in costs.items():
            assert isinstance(model, str)
            assert isinstance(cost, tuple)
            assert len(cost) == 2

    def test_covers_featured_models(self):
        """All featured models have cost entries."""
        costs = get_all_model_costs()
        for provider, models in _FEATURED_MODELS.items():
            for model in models:
                assert model in costs, f"Missing cost for {model}"

    def test_ollama_costs_are_zero(self):
        """All Ollama models in cost table have zero cost."""
        costs = get_all_model_costs()
        for model, cost in costs.items():
            if model.startswith("ollama/"):
                assert cost == (0.0, 0.0), f"{model} should be free"
