"""Credential vault and API proxy handlers.

Stores and manages API credentials loaded from environment variables.
Agents NEVER see credentials -- they request API calls through the mesh,
and the vault authenticates on their behalf.

Env var format: OPENLEGION_CRED_<SERVICE>_<KEY_NAME>
  e.g. OPENLEGION_CRED_ANTHROPIC_API_KEY
"""

from __future__ import annotations

import os

import httpx

from src.shared.types import APIProxyRequest, APIProxyResponse
from src.shared.utils import setup_logging

logger = setup_logging("host.credentials")


class CredentialVault:
    """Stores API credentials and executes proxied API calls."""

    def __init__(self, cost_tracker: object | None = None) -> None:
        self.credentials: dict[str, str] = {}
        self.service_handlers: dict[str, callable] = {}
        self.cost_tracker = cost_tracker
        self._load_credentials()
        self._register_handlers()

    def _load_credentials(self) -> None:
        """Load credentials from environment variables."""
        prefix = "OPENLEGION_CRED_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                cred_name = key[len(prefix) :].lower()
                self.credentials[cred_name] = value
        loaded = list(self.credentials.keys())
        if loaded:
            logger.info(f"Loaded credentials: {', '.join(loaded)}")

    def _register_handlers(self) -> None:
        """Register API call handlers for each supported service."""
        self.service_handlers = {
            "llm": self._handle_llm,
            "anthropic": self._handle_anthropic,
            "openai": self._handle_openai,
            "apollo": self._handle_apollo,
            "hunter": self._handle_hunter,
            "brave_search": self._handle_brave_search,
        }

    async def execute_api_call(
        self, request: APIProxyRequest, agent_id: str = "",
    ) -> APIProxyResponse:
        """Execute an API call on behalf of an agent."""
        if self.cost_tracker and agent_id and request.service in ("llm", "anthropic", "openai"):
            budget = self.cost_tracker.check_budget(agent_id)
            if not budget["allowed"]:
                return APIProxyResponse(
                    success=False,
                    error=(
                        f"Budget exceeded: ${budget['daily_used']:.2f}/${budget['daily_limit']:.2f} daily, "
                        f"${budget['monthly_used']:.2f}/${budget['monthly_limit']:.2f} monthly"
                    ),
                )

        handler = self.service_handlers.get(request.service)
        if not handler:
            return APIProxyResponse(success=False, error=f"Unknown service: {request.service}")
        try:
            response = await handler(request)

            if self.cost_tracker and agent_id and response.success and response.data:
                tokens_used = response.data.get("tokens_used", 0)
                if tokens_used:
                    model = request.params.get("model", "unknown")
                    prompt_tokens = int(tokens_used * 0.7)
                    completion_tokens = tokens_used - prompt_tokens
                    self.cost_tracker.track(agent_id, model, prompt_tokens, completion_tokens)

            return response
        except Exception as e:
            logger.error(f"API call failed for {request.service}/{request.action}: {e}")
            return APIProxyResponse(success=False, error=str(e))

    def _get_api_key_for_model(self, model: str) -> str | None:
        """Resolve the API key for a model based on its provider prefix."""
        provider_key_map = {
            "anthropic/": "anthropic_api_key",
            "openai/": "openai_api_key",
            "gpt-": "openai_api_key",
            "o1": "openai_api_key",
            "groq/": "groq_api_key",
            "gemini/": "gemini_api_key",
            "text-embedding-": "openai_api_key",
        }
        for prefix, key_name in provider_key_map.items():
            if model.startswith(prefix):
                return self.credentials.get(key_name)
        return None

    async def _handle_llm(self, request: APIProxyRequest) -> APIProxyResponse:
        """Unified LLM handler. Auto-detects provider from model prefix via LiteLLM."""
        import litellm

        model = request.params.get("model", "")
        api_key = self._get_api_key_for_model(model)
        if not api_key:
            return APIProxyResponse(
                success=False,
                error=f"No API key configured for model: {model}",
            )

        if request.action == "chat":
            response = await litellm.acompletion(
                model=model,
                messages=request.params.get("messages", []),
                api_key=api_key,
                **{k: v for k, v in request.params.items() if k not in ("model", "messages")},
            )
            msg = response.choices[0].message
            return APIProxyResponse(
                success=True,
                data={
                    "content": msg.content or "",
                    "tokens_used": response.usage.total_tokens if response.usage else 0,
                    "model": model,
                    "tool_calls": [
                        {"name": tc.function.name, "arguments": tc.function.arguments}
                        for tc in (msg.tool_calls or [])
                    ],
                },
            )

        elif request.action == "embed":
            response = await litellm.aembedding(
                model=request.params.get("model", "text-embedding-3-small"),
                input=request.params.get("text", ""),
                api_key=api_key,
            )
            item = response.data[0]
            embedding = item["embedding"] if isinstance(item, dict) else item.embedding
            return APIProxyResponse(success=True, data={"embedding": embedding})

        return APIProxyResponse(success=False, error=f"Unknown action: {request.action}")

    async def _handle_anthropic(self, request: APIProxyRequest) -> APIProxyResponse:
        """Handle Anthropic API calls (LLM completions, embeddings) via LiteLLM."""
        api_key = self.credentials.get("anthropic_api_key")
        if not api_key:
            return APIProxyResponse(success=False, error="Anthropic API key not configured")

        import litellm

        if request.action == "chat":
            response = await litellm.acompletion(
                model=request.params.get("model", "anthropic/claude-sonnet-4-5-20250929"),
                messages=request.params.get("messages", []),
                api_key=api_key,
                **{k: v for k, v in request.params.items() if k not in ("model", "messages")},
            )
            msg = response.choices[0].message
            return APIProxyResponse(
                success=True,
                data={
                    "content": msg.content or "",
                    "tokens_used": response.usage.total_tokens if response.usage else 0,
                    "tool_calls": [
                        {"name": tc.function.name, "arguments": tc.function.arguments}
                        for tc in (msg.tool_calls or [])
                    ],
                },
            )

        elif request.action == "embed":
            response = await litellm.aembedding(
                model=request.params.get("model", "text-embedding-3-small"),
                input=request.params.get("text", ""),
                api_key=api_key,
            )
            item = response.data[0]
            embedding = item["embedding"] if isinstance(item, dict) else item.embedding
            return APIProxyResponse(success=True, data={"embedding": embedding})

        return APIProxyResponse(success=False, error=f"Unknown action: {request.action}")

    async def _handle_openai(self, request: APIProxyRequest) -> APIProxyResponse:
        """Handle OpenAI API calls (embeddings)."""
        api_key = self.credentials.get("openai_api_key")
        if not api_key:
            return APIProxyResponse(success=False, error="OpenAI API key not configured")

        import litellm

        if request.action == "embed":
            response = await litellm.aembedding(
                model=request.params.get("model", "text-embedding-3-small"),
                input=request.params.get("text", ""),
                api_key=api_key,
            )
            item = response.data[0]
            embedding = item["embedding"] if isinstance(item, dict) else item.embedding
            return APIProxyResponse(success=True, data={"embedding": embedding})

        return APIProxyResponse(success=False, error=f"Unknown action: {request.action}")

    async def _handle_apollo(self, request: APIProxyRequest) -> APIProxyResponse:
        """Handle Apollo.io API calls."""
        api_key = self.credentials.get("apollo_api_key")
        if not api_key:
            return APIProxyResponse(success=False, error="Apollo API key not configured")

        async with httpx.AsyncClient() as client:
            if request.action == "search_people":
                response = await client.post(
                    "https://api.apollo.io/api/v1/mixed_people/search",
                    headers={"X-Api-Key": api_key},
                    json=request.params,
                    timeout=request.timeout,
                )
                return APIProxyResponse(
                    success=response.is_success,
                    data=response.json() if response.is_success else None,
                    error=response.text if not response.is_success else None,
                    status_code=response.status_code,
                )

        return APIProxyResponse(success=False, error=f"Unknown action: {request.action}")

    async def _handle_hunter(self, request: APIProxyRequest) -> APIProxyResponse:
        """Handle Hunter.io API calls."""
        api_key = self.credentials.get("hunter_api_key")
        if not api_key:
            return APIProxyResponse(success=False, error="Hunter API key not configured")

        async with httpx.AsyncClient() as client:
            if request.action == "domain_search":
                response = await client.get(
                    "https://api.hunter.io/v2/domain-search",
                    params={"domain": request.params.get("domain"), "api_key": api_key},
                    timeout=request.timeout,
                )
                return APIProxyResponse(
                    success=response.is_success,
                    data=response.json() if response.is_success else None,
                    error=response.text if not response.is_success else None,
                    status_code=response.status_code,
                )

        return APIProxyResponse(success=False, error=f"Unknown action: {request.action}")

    async def _handle_brave_search(self, request: APIProxyRequest) -> APIProxyResponse:
        """Handle Brave Search API calls."""
        api_key = self.credentials.get("brave_search_api_key")
        if not api_key:
            return APIProxyResponse(success=False, error="Brave Search API key not configured")

        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers={"X-Subscription-Token": api_key},
                params=request.params,
                timeout=request.timeout,
            )
            return APIProxyResponse(
                success=response.is_success,
                data=response.json() if response.is_success else None,
                error=response.text if not response.is_success else None,
                status_code=response.status_code,
            )
