"""Credential vault and API proxy handlers.

Stores and manages API credentials loaded from environment variables.
Agents NEVER see credentials -- they request API calls through the mesh,
and the vault authenticates on their behalf.

Env var format: OPENLEGION_CRED_<SERVICE>_<KEY_NAME>
  e.g. OPENLEGION_CRED_ANTHROPIC_API_KEY
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable

import httpx

from src.host.transcript import sanitize_for_provider
from src.shared.types import APIProxyRequest, APIProxyResponse
from src.shared.utils import setup_logging

logger = setup_logging("host.credentials")


def _persist_to_env(env_key: str, value: str, env_file: str = "") -> None:
    """Persist an environment variable to .env and os.environ.

    If *env_file* is empty, defaults to ``PROJECT_ROOT / ".env"``.
    """
    from pathlib import Path

    if not env_file:
        env_file = str(Path(__file__).resolve().parent.parent.parent / ".env")

    env_path = Path(env_file)
    lines: list[str] = []
    found = False

    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith(f"{env_key}=") or line.startswith(f"# {env_key}="):
                lines.append(f"{env_key}={value}")
                found = True
            else:
                lines.append(line)

    if not found:
        lines.append(f"{env_key}={value}")

    env_path.write_text("\n".join(lines) + "\n")
    os.environ[env_key] = value


class CredentialVault:
    """Stores API credentials and executes proxied API calls."""

    def __init__(
        self,
        cost_tracker: object | None = None,
        failover_config: dict[str, list[str]] | None = None,
    ) -> None:
        self.credentials: dict[str, str] = {}
        self.service_handlers: dict[str, Callable] = {}
        self.cost_tracker = cost_tracker
        self._http_client: httpx.AsyncClient | None = None
        self._load_credentials()
        self._register_handlers()

        from src.host.failover import FailoverChain, ModelHealthTracker
        self._health_tracker = ModelHealthTracker()
        self._failover_chain = FailoverChain(
            chains=failover_config or {}, health=self._health_tracker,
        )

    async def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=30)
        return self._http_client

    async def close(self) -> None:
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

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

    def add_credential(self, name: str, value: str) -> str:
        """Store a credential in memory and persist to .env. Returns a $CRED{name} handle."""
        cred_key = name.lower()
        self.credentials[cred_key] = value
        env_key = f"OPENLEGION_CRED_{name.upper()}"
        _persist_to_env(env_key, value)
        logger.info(f"Credential stored: {cred_key}")
        return f"$CRED{{{name}}}"

    def resolve_credential(self, name: str) -> str | None:
        """Resolve a credential name to its value. Returns None if not found."""
        return self.credentials.get(name.lower())

    def list_credential_names(self) -> list[str]:
        """Return a list of credential names (never values)."""
        return list(self.credentials.keys())

    def has_credential(self, name: str) -> bool:
        """Check if a credential exists by name."""
        return name.lower() in self.credentials

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
                    model = response.data.get(
                        "model", request.params.get("model", "unknown"),
                    )
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
            "o3": "openai_api_key",
            "o4": "openai_api_key",
            "xai/": "xai_api_key",
            "groq/": "groq_api_key",
            "gemini/": "gemini_api_key",
            "moonshot/": "moonshot_api_key",
            "deepseek/": "deepseek_api_key",
            "text-embedding-": "openai_api_key",
        }
        for prefix, key_name in provider_key_map.items():
            if model.startswith(prefix):
                return self.credentials.get(key_name)
        return None

    @staticmethod
    def _is_permanent_error(error: Exception) -> bool:
        """Return True if the error should NOT cascade to fallback models.

        BadRequestError covers its subclasses: ContentPolicyViolationError,
        ContextWindowExceededError, UnsupportedParamsError, etc.
        NotFoundError means the model name itself is invalid — cascading
        would silently mask bad config.
        """
        import litellm
        if isinstance(error, (litellm.BadRequestError, litellm.NotFoundError)):
            return True
        return False

    @staticmethod
    def _get_status_code(error: Exception) -> int:
        """Extract HTTP status code from a litellm exception."""
        return getattr(error, "status_code", 0)

    async def _call_llm_with_failover(
        self, requested_model: str, call_fn,
    ) -> tuple:
        """Try *call_fn(model, api_key)* across the failover chain.

        Returns ``(result, used_model)`` on success.
        Raises the last exception if all models are exhausted.
        """
        models = self._failover_chain.get_models_to_try(requested_model)
        last_error: Exception | None = None

        for model in models:
            api_key = self._get_api_key_for_model(model)
            if not api_key:
                logger.debug(f"No API key for failover candidate '{model}', skipping")
                continue
            try:
                result = await call_fn(model, api_key)
                self._health_tracker.record_success(model)
                if model != requested_model:
                    logger.info(
                        f"Failover: '{requested_model}' → '{model}' succeeded",
                    )
                return result, model
            except Exception as e:
                status_code = self._get_status_code(e)
                self._health_tracker.record_failure(
                    model, type(e).__name__, status_code,
                )
                if self._is_permanent_error(e):
                    raise
                last_error = e

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"No API key configured for model: {requested_model}")

    def get_model_health(self) -> list[dict]:
        """Return diagnostic model-health data."""
        return self._health_tracker.get_status()

    async def _handle_llm(self, request: APIProxyRequest) -> APIProxyResponse:
        """Unified LLM handler. Auto-detects provider from model prefix via LiteLLM."""
        import litellm

        requested_model = request.params.get("model", "")

        if request.action == "chat":
            async def _chat(model: str, api_key: str):
                sanitized = sanitize_for_provider(request.params.get("messages", []), model)
                return await litellm.acompletion(
                    model=model,
                    messages=sanitized,
                    api_key=api_key,
                    **{k: v for k, v in request.params.items() if k not in ("model", "messages")},
                )

            response, used_model = await self._call_llm_with_failover(
                requested_model, _chat,
            )
            msg = response.choices[0].message
            return APIProxyResponse(
                success=True,
                data={
                    "content": msg.content or "",
                    "tokens_used": response.usage.total_tokens if response.usage else 0,
                    "model": used_model,
                    "tool_calls": [
                        {"name": tc.function.name, "arguments": tc.function.arguments}
                        for tc in (msg.tool_calls or [])
                    ],
                },
            )

        elif request.action == "embed":
            # Embedding models produce incompatible vector spaces — no failover
            api_key = self._get_api_key_for_model(requested_model)
            if not api_key:
                return APIProxyResponse(
                    success=False,
                    error=f"No API key configured for model: {requested_model}",
                )
            response = await litellm.aembedding(
                model=request.params.get("model", "text-embedding-3-small"),
                input=request.params.get("text", ""),
                api_key=api_key,
            )
            item = response.data[0]
            embedding = item["embedding"] if isinstance(item, dict) else item.embedding
            return APIProxyResponse(success=True, data={"embedding": embedding})

        return APIProxyResponse(success=False, error=f"Unknown action: {request.action}")

    async def stream_llm(self, request: APIProxyRequest, agent_id: str = ""):
        """Streaming LLM handler. Yields SSE-formatted chunks.

        Each yielded string is a complete SSE line: ``data: <json>\\n\\n``.
        The final chunk has ``"done": true``.
        Supports failover: if connection setup fails for one model, the next
        model in the chain is tried. Once streaming starts, we stay on that model.
        """
        import litellm

        if self.cost_tracker and agent_id and request.service in ("llm", "anthropic", "openai"):
            budget = self.cost_tracker.check_budget(agent_id)
            if not budget["allowed"]:
                yield f"data: {json.dumps({'error': 'Budget exceeded'})}\n\n"
                return

        requested_model = request.params.get("model", "")
        models_to_try = self._failover_chain.get_models_to_try(requested_model)

        response = None
        used_model = requested_model
        last_error: Exception | None = None

        for model in models_to_try:
            api_key = self._get_api_key_for_model(model)
            if not api_key:
                continue
            try:
                sanitized = sanitize_for_provider(request.params.get("messages", []), model)
                response = await litellm.acompletion(
                    model=model,
                    messages=sanitized,
                    api_key=api_key,
                    stream=True,
                    **{k: v for k, v in request.params.items() if k not in ("model", "messages")},
                )
                used_model = model
                if model != requested_model:
                    logger.info(f"Stream failover: '{requested_model}' → '{model}'")
                break
            except Exception as e:
                status_code = self._get_status_code(e)
                self._health_tracker.record_failure(model, type(e).__name__, status_code)
                if self._is_permanent_error(e):
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
                    return
                last_error = e

        if response is None:
            error_msg = str(last_error) if last_error else f"No API key for model: {requested_model}"
            yield f"data: {json.dumps({'error': error_msg})}\n\n"
            return

        try:
            collected_content = ""
            collected_tool_calls: list[dict] = []

            async for chunk in response:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta is None:
                    continue

                if delta.content:
                    collected_content += delta.content
                    yield f"data: {json.dumps({'type': 'text_delta', 'content': delta.content})}\n\n"

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index if hasattr(tc, 'index') else 0
                        while len(collected_tool_calls) <= idx:
                            collected_tool_calls.append({"name": "", "arguments": ""})
                        if tc.function and tc.function.name:
                            collected_tool_calls[idx]["name"] = tc.function.name
                        if tc.function and tc.function.arguments:
                            collected_tool_calls[idx]["arguments"] += tc.function.arguments

            # Emit final summary
            tokens_used = 0
            if hasattr(response, 'usage') and response.usage:
                tokens_used = response.usage.total_tokens

            self._health_tracker.record_success(used_model)
            yield f"data: {json.dumps({'type': 'done', 'content': collected_content, 'tool_calls': collected_tool_calls, 'tokens_used': tokens_used})}\n\n"

            if self.cost_tracker and agent_id and tokens_used:
                prompt_tokens = int(tokens_used * 0.7)
                completion_tokens = tokens_used - prompt_tokens
                self.cost_tracker.track(agent_id, used_model, prompt_tokens, completion_tokens)

        except Exception as e:
            logger.error(f"Streaming LLM call failed: {e}")
            self._health_tracker.record_failure(
                used_model, type(e).__name__, self._get_status_code(e),
            )
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    async def _handle_anthropic(self, request: APIProxyRequest) -> APIProxyResponse:
        """Handle Anthropic API calls (LLM completions, embeddings) via LiteLLM."""
        api_key = self.credentials.get("anthropic_api_key")
        if not api_key:
            return APIProxyResponse(success=False, error="Anthropic API key not configured")

        import litellm

        if request.action == "chat":
            model = request.params.get("model", "anthropic/claude-sonnet-4-5-20250929")
            sanitized = sanitize_for_provider(request.params.get("messages", []), model)
            response = await litellm.acompletion(
                model=model,
                messages=sanitized,
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

        client = await self._get_http_client()
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

        client = await self._get_http_client()
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

        client = await self._get_http_client()
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
