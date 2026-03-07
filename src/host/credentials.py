"""Credential vault and API proxy handlers.

Stores and manages API credentials loaded from environment variables.
Agents NEVER see credentials -- they request API calls through the mesh,
and the vault authenticates on their behalf.

Two-tier credential system:
  OPENLEGION_SYSTEM_<NAME>  — System-tier. Used by mesh proxy internally.
                              Never resolvable by agents.
  OPENLEGION_CRED_<NAME>    — Agent-tier. Accessible based on
                              ``allowed_credentials`` permission patterns.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable

import httpx

from src.host.transcript import sanitize_for_provider
from src.shared.types import APIProxyRequest, APIProxyResponse
from src.shared.utils import friendly_streaming_error, setup_logging

logger = setup_logging("host.credentials")

# ── OAuth setup-token constants ───────────────────────────
# Anthropic's unofficial OAuth path for Claude Pro/Max subscriptions.
# Tokens from `claude setup-token` use Bearer auth instead of x-api-key.
_OAUTH_TOKEN_PREFIX = "sk-ant-oat01-"
_CLAUDE_CLI_VERSION = "2.1.62"
_ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"


def _extract_content(raw_content) -> tuple[str, str | None]:
    """Extract text and thinking from LLM response content.

    LiteLLM returns content as a list of blocks when extended thinking is enabled:
      [{"type": "thinking", "thinking": "..."}, {"type": "text", "text": "..."}]
    """
    if isinstance(raw_content, str):
        return raw_content, None
    if not isinstance(raw_content, list):
        return str(raw_content) if raw_content else "", None

    text_parts, thinking_parts = [], []
    for block in raw_content:
        if not isinstance(block, dict):
            text_parts.append(str(block))
            continue
        btype = block.get("type", "")
        if btype == "thinking":
            thinking_parts.append(block.get("thinking", ""))
        elif btype == "text":
            text_parts.append(block.get("text", ""))
        else:
            text_parts.append(block.get("text", str(block)))

    return "".join(text_parts), "".join(thinking_parts) if thinking_parts else None


def _persist_to_env(env_key: str, value: str, env_file: str = "") -> None:
    """Persist an environment variable to .env and os.environ.

    If *env_file* is empty, defaults to ``PROJECT_ROOT / ".env"``.
    Values are single-quoted to prevent python-dotenv from mangling
    special characters (``$``, ``#``).  Production loads with
    ``interpolate=False`` as a second layer of defense.
    """
    import re
    from pathlib import Path

    # Reject newlines/carriage returns to prevent env injection
    if re.search(r"[\r\n]", env_key) or re.search(r"[\r\n]", value):
        raise ValueError("env key and value must not contain newline characters")
    # Validate env key format
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", env_key):
        raise ValueError(f"Invalid env key name: {env_key}")

    if not env_file:
        env_file = str(Path(__file__).resolve().parent.parent.parent / ".env")

    # Quote to prevent python-dotenv from mangling values.
    # Single quotes: no interpolation, no escape processing — safest.
    # Double quotes (for values with '): still safe because we load with
    # interpolate=False; only need to escape \ and " which dotenv always
    # processes inside double quotes.
    if "'" not in value:
        formatted = f"{env_key}='{value}'"
    else:
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        formatted = f'{env_key}="{escaped}"'

    env_path = Path(env_file)
    lines: list[str] = []
    found = False

    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith(f"{env_key}=") or line.startswith(f"# {env_key}="):
                lines.append(formatted)
                found = True
            else:
                lines.append(line)

    if not found:
        lines.append(formatted)

    env_path.write_text("\n".join(lines) + "\n")
    os.chmod(env_file, 0o600)
    os.environ[env_key] = value


def _remove_from_env(env_key: str, env_file: str = "") -> None:
    """Remove an environment variable from .env and os.environ."""
    from pathlib import Path

    if not env_file:
        env_file = str(Path(__file__).resolve().parent.parent.parent / ".env")

    env_path = Path(env_file)
    if env_path.exists():
        lines = [
            line for line in env_path.read_text().splitlines()
            if not line.startswith(f"{env_key}=")
        ]
        env_path.write_text("\n".join(lines) + "\n")
        os.chmod(env_file, 0o600)

    os.environ.pop(env_key, None)


# ── Prefix constants ───────────────────────────────────────
SYSTEM_PREFIX = "OPENLEGION_SYSTEM_"
AGENT_PREFIX = "OPENLEGION_CRED_"

# System credential patterns — used by is_system_credential() for
# defense-in-depth permission checks and by CLI/dashboard for
# auto-detecting LLM provider keys.  Derived from _PROVIDER_KEY_MAP.
SYSTEM_CREDENTIAL_PROVIDERS = frozenset({
    "anthropic", "openai", "gemini", "deepseek", "moonshot",
    "minimax", "xai", "groq", "zai",
})
SYSTEM_CREDENTIAL_SUFFIXES = ("_api_key", "_api_base")


def is_oauth_token(token: str) -> bool:
    """Check if a token is an Anthropic OAuth setup-token."""
    return token.startswith(_OAUTH_TOKEN_PREFIX)


def is_system_credential(name: str) -> bool:
    """Check if a credential name is a system-level credential.

    System credentials are provider API keys and base URLs that should
    only be used internally by the mesh proxy — never resolvable by agents.
    """
    lower = name.lower()
    for suffix in SYSTEM_CREDENTIAL_SUFFIXES:
        if lower.endswith(suffix):
            prefix = lower[: -len(suffix)]
            if prefix in SYSTEM_CREDENTIAL_PROVIDERS:
                return True
    return False


class CredentialVault:
    """Stores API credentials and executes proxied API calls."""

    def __init__(
        self,
        cost_tracker: object | None = None,
        failover_config: dict[str, list[str]] | None = None,
    ) -> None:
        self.system_credentials: dict[str, str] = {}
        self.credentials: dict[str, str] = {}
        self.api_bases: dict[str, str] = {}
        self.service_handlers: dict[str, Callable] = {}
        self.cost_tracker = cost_tracker
        self._http_client: httpx.AsyncClient | None = None
        self._http_client_lock = asyncio.Lock()
        self._budget_locks: dict[str, asyncio.Lock] = {}
        self._load_credentials()
        self._register_handlers()

        from src.host.failover import FailoverChain, ModelHealthTracker
        self._health_tracker = ModelHealthTracker()
        self._failover_chain = FailoverChain(
            chains=failover_config or {}, health=self._health_tracker,
        )

    async def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is not None and not self._http_client.is_closed:
            return self._http_client
        async with self._http_client_lock:
            if self._http_client is None or self._http_client.is_closed:
                self._http_client = httpx.AsyncClient(timeout=30)
            return self._http_client

    async def close(self) -> None:
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
        self._http_client = None

    def cleanup_agent(self, agent_id: str) -> None:
        """Remove per-agent state (budget locks) for a deregistered agent."""
        self._budget_locks.pop(agent_id, None)

    def _load_credentials(self) -> None:
        """Load credentials from environment variables (two-phase).

        Phase 1: Scan ``OPENLEGION_SYSTEM_*`` → ``system_credentials`` / ``api_bases``.
        Phase 2: Scan ``OPENLEGION_CRED_*`` → ``credentials`` (agent tier) /
                 ``api_bases`` (if not already set by system prefix).

        No auto-promotion: ``OPENLEGION_CRED_`` provider keys are treated as
        agent-tier.  Use ``OPENLEGION_SYSTEM_`` for LLM provider keys.
        """
        # Phase 1: explicit system-tier credentials
        for key, value in os.environ.items():
            if key.startswith(SYSTEM_PREFIX):
                cred_name = key[len(SYSTEM_PREFIX):].lower()
                if cred_name.endswith("_api_base"):
                    self.api_bases[cred_name] = value
                else:
                    self.system_credentials[cred_name] = value

        # Phase 2: agent-tier credentials
        for key, value in os.environ.items():
            if key.startswith(AGENT_PREFIX):
                cred_name = key[len(AGENT_PREFIX):].lower()
                if cred_name.endswith("_api_base"):
                    # Only store if not already set by SYSTEM_ prefix
                    if cred_name not in self.api_bases:
                        self.api_bases[cred_name] = value
                else:
                    self.credentials[cred_name] = value

        loaded_system = list(self.system_credentials.keys())
        loaded_agent = list(self.credentials.keys())
        if loaded_system:
            logger.info(f"Loaded system credentials: {', '.join(loaded_system)}")
        if loaded_agent:
            logger.info(f"Loaded agent credentials: {', '.join(loaded_agent)}")
        if self.api_bases:
            logger.info(f"Loaded custom API bases: {', '.join(self.api_bases.keys())}")

    def add_credential(self, name: str, value: str, *, system: bool = False) -> str:
        """Store a credential in memory and persist to .env.

        Args:
            system: If True, stores in ``system_credentials`` with the
                    ``OPENLEGION_SYSTEM_`` prefix. Otherwise stores in
                    ``credentials`` (agent tier) with ``OPENLEGION_CRED_``.

        Returns a ``$CRED{name}`` handle.
        """
        cred_key = name.lower()
        prefix = SYSTEM_PREFIX if system else AGENT_PREFIX
        if cred_key.endswith("_api_base"):
            self.api_bases[cred_key] = value
        elif system:
            self.system_credentials[cred_key] = value
        else:
            self.credentials[cred_key] = value
        env_key = f"{prefix}{name.upper()}"
        _persist_to_env(env_key, value)
        tier = "system" if system else "agent"
        logger.info(f"Credential stored ({tier}): {cred_key}")
        return f"$CRED{{{name}}}"

    def resolve_credential(self, name: str) -> str | None:
        """Resolve a credential name to its value (agent-tier only).

        System-tier credentials are never returned here — they are only
        accessible internally by the mesh proxy handlers.
        """
        return self.credentials.get(name.lower())

    def list_credential_names(self) -> list[str]:
        """Return all credential names across both tiers (never values)."""
        combined = set(self.system_credentials.keys()) | set(self.credentials.keys())
        return sorted(combined)

    def list_agent_credential_names(self) -> list[str]:
        """Return agent-tier credential names only.

        Since loading already sorts credentials into the correct tier,
        this simply returns ``credentials`` keys — no filtering needed.
        """
        return list(self.credentials.keys())

    def list_system_credential_names(self) -> list[str]:
        """Return system-tier credential names only."""
        return list(self.system_credentials.keys())

    def remove_credential(self, name: str) -> bool:
        """Remove a credential from memory, .env, and os.environ.

        Searches both tiers and removes from both ``OPENLEGION_SYSTEM_``
        and ``OPENLEGION_CRED_`` env prefixes.  Returns True if it existed.
        """
        cred_key = name.lower()
        existed = False
        if cred_key.endswith("_api_base"):
            existed = cred_key in self.api_bases
            self.api_bases.pop(cred_key, None)
        else:
            if cred_key in self.system_credentials:
                existed = True
                self.system_credentials.pop(cred_key, None)
            if cred_key in self.credentials:
                existed = True
                self.credentials.pop(cred_key, None)
        # Remove from both possible env prefixes
        _remove_from_env(f"{SYSTEM_PREFIX}{name.upper()}")
        _remove_from_env(f"{AGENT_PREFIX}{name.upper()}")
        if existed:
            logger.info(f"Credential removed: {cred_key}")
        return existed

    def has_credential(self, name: str) -> bool:
        """Check if a credential exists by name (either tier)."""
        lower = name.lower()
        return lower in self.credentials or lower in self.system_credentials

    def _register_handlers(self) -> None:
        """Register API call handlers for each supported service."""
        self.service_handlers = {
            "llm": self._handle_llm,
            "apollo": self._handle_apollo,
            "hunter": self._handle_hunter,
            "brave_search": self._handle_brave_search,
        }

    async def execute_api_call(
        self, request: APIProxyRequest, agent_id: str = "",
    ) -> APIProxyResponse:
        """Execute an API call on behalf of an agent.

        Budget check + execute + cost track are serialized per-agent to
        prevent two concurrent calls from both passing the budget check
        before either is charged.
        """
        is_llm = request.service == "llm"
        use_budget_lock = bool(self.cost_tracker and agent_id and is_llm)

        if use_budget_lock:
            if agent_id not in self._budget_locks:
                self._budget_locks[agent_id] = asyncio.Lock()
            lock = self._budget_locks[agent_id]
        else:
            lock = None

        async def _execute() -> APIProxyResponse:
            if self.cost_tracker and agent_id and is_llm:
                model = request.params.get("model", "unknown")
                preflight = self.cost_tracker.preflight_check(agent_id, model)
                if not preflight["allowed"]:
                    return APIProxyResponse(
                        success=False,
                        error=(
                            f"Budget exceeded: ${preflight['daily_used']:.2f}/${preflight['daily_limit']:.2f} daily, "
                            f"${preflight['monthly_used']:.2f}/${preflight['monthly_limit']:.2f} monthly "
                            f"(estimated next call: ${preflight['estimated_cost']:.4f})"
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
                        raw_pt = response.data.get("input_tokens")
                        prompt_tokens = raw_pt if raw_pt else int(tokens_used * 0.7)
                        raw_ct = response.data.get("output_tokens")
                        completion_tokens = raw_ct if raw_ct else (tokens_used - prompt_tokens)
                        self.cost_tracker.track(agent_id, model, prompt_tokens, completion_tokens)

                return response
            except Exception as e:
                logger.error(f"API call failed for {request.service}/{request.action}: {e}")
                return APIProxyResponse(success=False, error=str(e))

        if lock is not None:
            try:
                await asyncio.wait_for(lock.acquire(), timeout=120)
            except (TimeoutError, asyncio.TimeoutError):
                logger.warning("Budget lock timed out for agent '%s'", agent_id)
                return APIProxyResponse(
                    success=False,
                    error="Budget lock contention — too many concurrent LLM calls. Retry shortly.",
                )
            try:
                return await _execute()
            finally:
                lock.release()
        return await _execute()

    # Provider prefix → credential key mapping (shared by key + base lookups)
    _PROVIDER_KEY_MAP = {
        "anthropic/": "anthropic",
        "openai/": "openai",
        "gpt-": "openai",
        "o1": "openai",
        "o3": "openai",
        "o4": "openai",
        "minimax/": "minimax",
        "zai/": "zai",
        "xai/": "xai",
        "groq/": "groq",
        "gemini/": "gemini",
        "moonshot/": "moonshot",
        "deepseek/": "deepseek",
        "text-embedding-": "openai",
    }

    def _get_api_key_for_model(self, model: str) -> str | None:
        """Resolve the API key for a model based on its provider prefix.

        Only checks system_credentials — LLM provider keys must use the
        ``OPENLEGION_SYSTEM_`` prefix.
        """
        for prefix, provider in self._PROVIDER_KEY_MAP.items():
            if model.startswith(prefix):
                key_name = f"{provider}_api_key"
                return self.system_credentials.get(key_name)
        return None

    def _get_auth_for_model(self, model: str) -> tuple[str | None, dict[str, str]]:
        """Resolve API key and any extra auth headers for a model.

        Returns ``(api_key, extra_headers)``.  OAuth tokens are handled
        separately via ``_oauth_chat`` / ``_oauth_chat_stream`` — this
        method always returns empty extra_headers for them since they
        bypass LiteLLM entirely.
        """
        api_key = self._get_api_key_for_model(model)
        if api_key is None:
            return None, {}
        return api_key, {}

    def get_providers_with_credentials(self) -> set[str]:
        """Return the set of provider names that have credentials configured."""
        providers: set[str] = set()
        for provider in SYSTEM_CREDENTIAL_PROVIDERS:
            key_name = f"{provider}_api_key"
            if key_name in self.system_credentials:
                providers.add(provider)
        return providers

    def _get_api_base_for_model(self, model: str) -> str | None:
        """Resolve a custom API base URL for a model's provider.

        Checks ``OPENLEGION_SYSTEM_<PROVIDER>_API_BASE`` first, then
        falls back to ``OPENLEGION_CRED_<PROVIDER>_API_BASE``.
        Returns *None* when no custom base is configured — LiteLLM
        uses its own defaults.
        """
        for prefix, provider in self._PROVIDER_KEY_MAP.items():
            if model.startswith(prefix):
                return self.api_bases.get(f"{provider}_api_base")
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
        """Try *call_fn(model, api_key, api_base, auth_headers)* across the failover chain.

        Returns ``(result, used_model)`` on success.
        Raises the last exception if all models are exhausted.
        """
        models = self._failover_chain.get_models_to_try(requested_model)
        last_error: Exception | None = None

        for model in models:
            api_key, auth_headers = self._get_auth_for_model(model)
            if not api_key:
                logger.debug(f"No API key for failover candidate '{model}', skipping")
                continue
            api_base = self._get_api_base_for_model(model)
            try:
                result = await call_fn(model, api_key, api_base, auth_headers)
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

    def _prepare_llm_params(
        self, request: APIProxyRequest, model: str,
        api_base: str | None = None,
        auth_headers: dict[str, str] | None = None,
    ) -> tuple[list[dict], dict]:
        """Build sanitized messages and extra kwargs for an LLM call.

        Returns ``(sanitized_messages, extra_kwargs)``.
        """
        sanitized = sanitize_for_provider(request.params.get("messages", []), model)
        extra = {k: v for k, v in request.params.items() if k not in ("model", "messages")}
        if api_base:
            extra["api_base"] = api_base
        if auth_headers:
            extra["extra_headers"] = {
                **extra.get("extra_headers", {}),
                **auth_headers,
            }
        return sanitized, extra

    # ── OAuth direct-call helpers ────────────────────────────

    @staticmethod
    def _oauth_headers(token: str) -> dict[str, str]:
        """Build Anthropic API headers for OAuth bearer auth."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "claude-code-20250219,oauth-2025-04-20",
            "user-agent": f"claude-cli/{_CLAUDE_CLI_VERSION}",
        }

    @staticmethod
    def _build_anthropic_body(params: dict) -> dict:
        """Convert LiteLLM-style params to Anthropic Messages API format."""
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
        # OpenAI uses role:"tool" for results and tool_calls on assistant msgs.
        # Anthropic uses role:"user" with tool_result blocks and role:"assistant"
        # with tool_use content blocks.
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
                tool_result = {
                    "type": "tool_result",
                    "tool_use_id": m.get("tool_call_id", ""),
                    "content": m.get("content", ""),
                }
                # Merge consecutive tool results into one user message
                if (converted
                        and converted[-1].get("role") == "user"
                        and isinstance(converted[-1].get("content"), list)):
                    converted[-1]["content"].append(tool_result)
                else:
                    converted.append({"role": "user", "content": [tool_result]})

            else:
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
                if "function" in t:
                    func = t["function"]
                    anthropic_tools.append({
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "input_schema": func.get("parameters", {"type": "object"}),
                    })
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
                body.pop("tools", None)  # Anthropic has no "none" — omit tools entirely
            elif isinstance(tool_choice, dict) and "function" in tool_choice:
                body["tool_choice"] = {
                    "type": "tool",
                    "name": tool_choice["function"]["name"],
                }

        # Extended thinking
        thinking = params.get("thinking")
        if thinking:
            body["thinking"] = thinking

        return body

    @staticmethod
    def _parse_anthropic_response(data: dict, model_prefix: str) -> dict:
        """Convert Anthropic API response to our standard result dict."""
        content = ""
        thinking_content = ""
        tool_calls: list[dict] = []

        for block in data.get("content", []):
            btype = block.get("type", "")
            if btype == "text":
                content += block.get("text", "")
            elif btype == "thinking":
                thinking_content += block.get("thinking", "")
            elif btype == "tool_use":
                tool_calls.append({
                    "name": block["name"],
                    "arguments": json.dumps(block.get("input", {})),
                })

        usage = data.get("usage", {})
        input_t = usage.get("input_tokens", 0)
        output_t = usage.get("output_tokens", 0)

        result: dict = {
            "content": content,
            "tokens_used": input_t + output_t,
            "input_tokens": input_t,
            "output_tokens": output_t,
            "model": model_prefix,
            "tool_calls": tool_calls,
        }
        if thinking_content:
            result["thinking_content"] = thinking_content
        return result

    async def _oauth_chat(
        self, request: APIProxyRequest, api_key: str, model: str,
    ) -> APIProxyResponse:
        """Direct Anthropic API call using OAuth bearer auth (non-streaming)."""
        sanitized = sanitize_for_provider(
            request.params.get("messages", []), model,
        )
        params = {**request.params, "messages": sanitized}
        body = self._build_anthropic_body(params)
        headers = self._oauth_headers(api_key)

        client = await self._get_http_client()
        try:
            resp = await client.post(
                _ANTHROPIC_API_URL, headers=headers, json=body, timeout=120,
            )
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            self._health_tracker.record_failure(model, type(e).__name__, 0)
            raise RuntimeError(f"Anthropic API connection error: {e}") from e

        if resp.status_code == 401:
            self._health_tracker.record_failure(model, "AuthError", 401)
            try:
                error_data = resp.json()
                msg = error_data.get("error", {}).get("message", "Authentication failed")
            except (json.JSONDecodeError, ValueError):
                msg = resp.text[:200] or "Authentication failed"
            if "OAuth" in msg:
                raise RuntimeError(
                    "Anthropic has disabled OAuth for third-party apps. "
                    "Use a standard API key from console.anthropic.com instead."
                )
            raise RuntimeError(f"OAuth authentication failed: {msg}")

        if not resp.is_success:
            error_text = resp.text[:500]
            self._health_tracker.record_failure(model, "HTTPError", resp.status_code)
            raise RuntimeError(
                f"Anthropic API error (HTTP {resp.status_code}): {error_text}"
            )

        data = resp.json()
        result = self._parse_anthropic_response(data, model)
        self._health_tracker.record_success(model)
        return APIProxyResponse(success=True, data=result)

    async def _oauth_chat_stream(
        self, request: APIProxyRequest, api_key: str, model: str,
        agent_id: str = "",
    ):
        """Streaming Anthropic API call using OAuth bearer auth.

        Yields SSE-formatted strings matching the ``stream_llm`` protocol.
        """
        sanitized = sanitize_for_provider(
            request.params.get("messages", []), model,
        )
        params = {**request.params, "messages": sanitized}
        body = self._build_anthropic_body(params)
        body["stream"] = True
        headers = self._oauth_headers(api_key)

        client = await self._get_http_client()
        collected_content = ""
        collected_thinking = ""
        collected_tool_calls: list[dict] = []
        input_tokens = 0
        output_tokens = 0
        current_tool_idx = -1

        try:
            async with client.stream(
                "POST", _ANTHROPIC_API_URL, headers=headers,
                json=body, timeout=120,
            ) as resp:
                if resp.status_code == 401:
                    self._health_tracker.record_failure(model, "AuthError", 401)
                    await resp.aread()
                    try:
                        error_data = resp.json()
                        msg = error_data.get("error", {}).get("message", "Auth failed")
                    except (json.JSONDecodeError, ValueError):
                        msg = resp.text[:200] or "Auth failed"
                    yield f"data: {json.dumps({'error': f'OAuth auth failed: {msg}'})}\n\n"
                    return
                if not resp.is_success:
                    self._health_tracker.record_failure(
                        model, "HTTPError", resp.status_code,
                    )
                    await resp.aread()
                    yield f"data: {json.dumps({'error': f'Anthropic API error (HTTP {resp.status_code})'})}\n\n"
                    return

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError:
                        continue

                    etype = event.get("type", "")

                    if etype == "content_block_start":
                        block = event.get("content_block", {})
                        if block.get("type") == "tool_use":
                            current_tool_idx += 1
                            collected_tool_calls.append({
                                "name": block.get("name", ""),
                                "arguments": "",
                            })

                    elif etype == "content_block_delta":
                        delta = event.get("delta", {})
                        dtype = delta.get("type", "")
                        if dtype == "text_delta":
                            text = delta.get("text", "")
                            collected_content += text
                            yield f"data: {json.dumps({'type': 'text_delta', 'content': text})}\n\n"
                        elif dtype == "thinking_delta":
                            collected_thinking += delta.get("thinking", "")
                        elif dtype == "input_json_delta":
                            if current_tool_idx >= 0:
                                collected_tool_calls[current_tool_idx]["arguments"] += delta.get("partial_json", "")

                    elif etype == "message_delta":
                        usage = event.get("usage", {})
                        output_tokens = usage.get("output_tokens", output_tokens)

                    elif etype == "message_start":
                        usage = event.get("message", {}).get("usage", {})
                        input_tokens = usage.get("input_tokens", input_tokens)

            tokens_used = input_tokens + output_tokens
            self._health_tracker.record_success(model)

            done_data: dict = {
                "type": "done", "content": collected_content,
                "tool_calls": collected_tool_calls,
                "tokens_used": tokens_used, "model": model,
            }
            if collected_thinking:
                done_data["thinking_content"] = collected_thinking
            yield f"data: {json.dumps(done_data)}\n\n"

            if self.cost_tracker and agent_id and tokens_used:
                pt = input_tokens or int(tokens_used * 0.7)
                ct = output_tokens or (tokens_used - pt)
                self.cost_tracker.track(agent_id, model, pt, ct)

        except Exception as e:
            logger.error(f"OAuth streaming call failed: {e}")
            self._health_tracker.record_failure(model, type(e).__name__, 0)
            yield f"data: {json.dumps({'error': friendly_streaming_error(e)})}\n\n"

    async def _handle_llm(self, request: APIProxyRequest) -> APIProxyResponse:
        """Unified LLM handler. Auto-detects provider from model prefix via LiteLLM.

        OAuth setup-tokens bypass LiteLLM entirely — direct httpx calls to
        Anthropic's Messages API with Bearer auth.
        """
        import litellm

        requested_model = request.params.get("model", "")

        if request.action == "chat":
            # OAuth fast-path: bypass LiteLLM for Anthropic OAuth tokens
            api_key = self._get_api_key_for_model(requested_model)
            if api_key and is_oauth_token(api_key):
                return await self._oauth_chat(request, api_key, requested_model)

            async def _chat(
                model: str, api_key: str,
                api_base: str | None = None,
                auth_headers: dict[str, str] | None = None,
            ):
                sanitized, extra = self._prepare_llm_params(
                    request, model, api_base, auth_headers,
                )
                return await litellm.acompletion(
                    model=model,
                    messages=sanitized,
                    api_key=api_key,
                    **extra,
                )

            response, used_model = await self._call_llm_with_failover(
                requested_model, _chat,
            )
            msg = response.choices[0].message
            usage = response.usage
            content, thinking_content = _extract_content(msg.content)
            # Fallback: some litellm versions put thinking in a separate attribute
            if thinking_content is None:
                thinking_content = getattr(msg, "reasoning_content", None) or None
            data = {
                "content": content,
                "tokens_used": usage.total_tokens if usage else 0,
                "input_tokens": usage.prompt_tokens if usage else 0,
                "output_tokens": usage.completion_tokens if usage else 0,
                "model": used_model,
                "tool_calls": [
                    {"name": tc.function.name, "arguments": tc.function.arguments}
                    for tc in (msg.tool_calls or [])
                ],
            }
            if thinking_content is not None:
                data["thinking_content"] = thinking_content
            return APIProxyResponse(success=True, data=data)

        elif request.action == "embed":
            # Embedding models produce incompatible vector spaces — no failover
            api_key, auth_headers = self._get_auth_for_model(requested_model)
            if not api_key:
                return APIProxyResponse(
                    success=False,
                    error=f"No API key configured for model: {requested_model}",
                )
            api_base = self._get_api_base_for_model(requested_model)
            embed_kwargs: dict = {}
            if api_base:
                embed_kwargs["api_base"] = api_base
            if auth_headers:
                embed_kwargs["extra_headers"] = auth_headers
            response = await litellm.aembedding(
                model=request.params["model"],
                input=request.params.get("text", ""),
                api_key=api_key,
                **embed_kwargs,
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

        if self.cost_tracker and agent_id and request.service == "llm":
            model = request.params.get("model", "unknown")
            preflight = self.cost_tracker.preflight_check(agent_id, model)
            if not preflight["allowed"]:
                yield f"data: {json.dumps({'error': 'Budget exceeded'})}\n\n"
                return

        requested_model = request.params.get("model", "")

        # OAuth fast-path: bypass LiteLLM for Anthropic OAuth tokens
        api_key = self._get_api_key_for_model(requested_model)
        if api_key and is_oauth_token(api_key):
            async for chunk in self._oauth_chat_stream(
                request, api_key, requested_model, agent_id,
            ):
                yield chunk
            return
        models_to_try = self._failover_chain.get_models_to_try(requested_model)

        response = None
        used_model = requested_model
        last_error: Exception | None = None

        for model in models_to_try:
            api_key, auth_headers = self._get_auth_for_model(model)
            if not api_key:
                continue
            api_base = self._get_api_base_for_model(model)
            try:
                sanitized, extra = self._prepare_llm_params(
                    request, model, api_base, auth_headers,
                )
                response = await litellm.acompletion(
                    model=model,
                    messages=sanitized,
                    api_key=api_key,
                    stream=True,
                    **extra,
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
            collected_thinking = ""
            collected_tool_calls: list[dict] = []

            async for chunk in response:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta is None:
                    continue

                if delta.content:
                    collected_content += delta.content
                    yield f"data: {json.dumps({'type': 'text_delta', 'content': delta.content})}\n\n"

                # Collect thinking/reasoning tokens but don't stream them to client
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning and isinstance(reasoning, str):
                    collected_thinking += reasoning

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
            prompt_tokens = 0
            completion_tokens = 0
            if hasattr(response, 'usage') and response.usage:
                tokens_used = response.usage.total_tokens
                prompt_tokens = getattr(response.usage, 'prompt_tokens', 0) or 0
                completion_tokens = getattr(response.usage, 'completion_tokens', 0) or 0

            self._health_tracker.record_success(used_model)
            done_data: dict = {
                'type': 'done', 'content': collected_content,
                'tool_calls': collected_tool_calls,
                'tokens_used': tokens_used, 'model': used_model,
            }
            if collected_thinking:
                done_data['thinking_content'] = collected_thinking
            yield f"data: {json.dumps(done_data)}\n\n"

            if self.cost_tracker and agent_id and tokens_used:
                pt = prompt_tokens or int(tokens_used * 0.7)
                ct = completion_tokens or (tokens_used - pt)
                self.cost_tracker.track(agent_id, used_model, pt, ct)

        except Exception as e:
            logger.error(f"Streaming LLM call failed: {e}")
            self._health_tracker.record_failure(
                used_model, type(e).__name__, self._get_status_code(e),
            )
            yield f"data: {json.dumps({'error': friendly_streaming_error(e)})}\n\n"

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
