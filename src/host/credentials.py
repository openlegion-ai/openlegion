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
import base64
import contextlib
import json
import os
import re
import time
from collections.abc import Callable
from pathlib import Path

import httpx

from src.agent.attachments import convert_openai_image_blocks
from src.host.transcript import sanitize_for_provider
from src.shared.models import KEYLESS_PROVIDERS, get_known_provider_names
from src.shared.types import APIProxyRequest, APIProxyResponse
from src.shared.utils import friendly_streaming_error, setup_logging

logger = setup_logging("host.credentials")

# ── OAuth setup-token constants ───────────────────────────
# Anthropic's unofficial OAuth path for Claude Pro/Max subscriptions.
# Tokens from `claude setup-token` use Bearer auth instead of x-api-key.
_OAUTH_TOKEN_PREFIX = "sk-ant-oat01-"
_CLAUDE_CLI_VERSION = "2.1.84"
_ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

# OpenAI Codex Responses API — uses ChatGPT subscription via OAuth.
_OPENAI_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
_OPENAI_TOKEN_URL = "https://auth.openai.com/oauth/token"
_OPENAI_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
_OPENAI_OAUTH_REDIRECT_URI = "http://localhost:1455/auth/callback"


def _model_supports_vision(model: str) -> bool:
    """Check if a model supports image content blocks.

    Uses litellm's model registry.  Returns True when unsure (safe default
    for custom/unknown models — the API will reject if wrong, and the
    error is clear and recoverable).
    """
    try:
        from litellm import supports_vision
        return supports_vision(model=model)
    except Exception:
        return True


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

    content = "\n".join(lines) + "\n"
    # Atomic write: temp file + fsync + rename ensures data survives
    # immediate process restarts (e.g., from update-all).
    # Resolve symlinks so the rename targets the real file's directory.
    real_path = env_path.resolve()
    tmp_path = real_path.parent / f"{real_path.name}.tmp"
    try:
        tmp_path.write_text(content)
        os.chmod(str(tmp_path), 0o600)
        # fsync to ensure data hits disk before rename
        fd = os.open(str(tmp_path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        tmp_path.replace(real_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    os.environ[env_key] = value


def _remove_from_env(env_key: str, env_file: str = "") -> None:
    """Remove an environment variable from .env and os.environ."""
    if not env_file:
        env_file = str(Path(__file__).resolve().parent.parent.parent / ".env")

    env_path = Path(env_file)
    if env_path.exists():
        lines = [
            line for line in env_path.read_text().splitlines()
            if not line.startswith(f"{env_key}=")
        ]
        content = "\n".join(lines) + "\n"
        # Atomic write: temp file + fsync + rename ensures data survives
        # immediate process restarts (e.g., from update-all).
        real_path = env_path.resolve()
        tmp_path = real_path.parent / f"{real_path.name}.tmp"
        try:
            tmp_path.write_text(content)
            os.chmod(str(tmp_path), 0o600)
            fd = os.open(str(tmp_path), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            tmp_path.replace(real_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    os.environ.pop(env_key, None)


# ── Prefix constants ───────────────────────────────────────
SYSTEM_PREFIX = "OPENLEGION_SYSTEM_"
AGENT_PREFIX = "OPENLEGION_CRED_"

# System credential patterns — used by is_system_credential() for
# defense-in-depth permission checks and by CLI/dashboard for
# auto-detecting LLM provider keys.  Derived dynamically from the
# model registry (src/shared/models.py) so adding a provider in one
# place propagates everywhere.
SYSTEM_CREDENTIAL_PROVIDERS = get_known_provider_names()
# Providers where LiteLLM handles routing natively (built-in support).
# Custom providers like 'openlegion' are NOT in this set — they use
# api_base with OpenAI-compatible rewrite in _rewrite_model_for_litellm().
_LITELLM_NATIVE_PROVIDERS = frozenset({
    "anthropic", "openai", "openrouter", "gemini", "mistral",
    "deepseek", "groq", "together_ai", "fireworks_ai", "perplexity",
    "minimax", "moonshot", "xai", "zai", "ollama",
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


def _load_oauth_config(env_var: str, label: str) -> dict | None:
    """Load and parse an OAuth JSON config from an environment variable.

    Returns the parsed dict if valid (must contain ``access_token``),
    or ``None`` on missing/invalid data.
    """
    raw = os.environ.get(env_var, "")
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "access_token" in parsed:
            logger.info("Loaded %s OAuth credentials", label)
            return parsed
        logger.warning(
            "Failed to parse %s OAuth config from %s: "
            "not a valid credentials object (missing access_token)",
            label, env_var,
        )
    except (json.JSONDecodeError, ValueError):
        logger.warning(
            "Failed to parse %s OAuth config from %s: not valid JSON",
            label, env_var,
        )
    return None


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
        self._openai_oauth: dict | None = None
        self._openai_oauth_lock = asyncio.Lock()
        self._anthropic_oauth: dict | None = None
        self._anthropic_oauth_lock = asyncio.Lock()
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
                if cred_name in ("openai_oauth", "anthropic_oauth"):
                    continue  # Handled separately below as structured OAuth
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

        # Load OAuth credentials (JSON blobs from environment)
        self._openai_oauth = _load_oauth_config(
            "OPENLEGION_SYSTEM_OPENAI_OAUTH", "OpenAI",
        )
        self._anthropic_oauth = _load_oauth_config(
            "OPENLEGION_SYSTEM_ANTHROPIC_OAUTH", "Anthropic",
        )

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
        # Strip ALL whitespace — terminal paste can include line breaks which
        # browsers convert to spaces, silently corrupting tokens and API keys.
        value = "".join(value.split())
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
        if self._openai_oauth is not None:
            combined.add("openai_oauth")
        if self._anthropic_oauth is not None:
            combined.add("anthropic_oauth")
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
        # Handle OAuth credential removal
        if cred_key == "openai_oauth":
            existed = self._openai_oauth is not None
            self._openai_oauth = None
            _remove_from_env("OPENLEGION_SYSTEM_OPENAI_OAUTH")
            if existed:
                logger.info("Credential removed: openai_oauth")
            return existed
        if cred_key == "anthropic_oauth":
            existed = self._anthropic_oauth is not None
            self._anthropic_oauth = None
            _remove_from_env("OPENLEGION_SYSTEM_ANTHROPIC_OAUTH")
            if existed:
                logger.info("Credential removed: anthropic_oauth")
            return existed
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
        if lower == "openai_oauth" and self._openai_oauth is not None:
            return True
        if lower == "anthropic_oauth" and self._anthropic_oauth is not None:
            return True
        return lower in self.credentials or lower in self.system_credentials

    def _register_handlers(self) -> None:
        """Register API call handlers for each supported service."""
        self.service_handlers = {
            "llm": self._handle_llm,
            "apollo": self._handle_apollo,
            "hunter": self._handle_hunter,
            "brave_search": self._handle_brave_search,
            "image_gen": self._handle_image_gen,
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

        # OAuth tokens (Anthropic/OpenAI subscription) have no per-call cost —
        # skip budget enforcement and cost tracking for them.
        _is_oauth = False
        if is_llm:
            _provider = self._resolve_provider(request.params.get("model", ""))
            # Structured OAuth takes priority (matches routing in _handle_llm)
            if _provider == "anthropic" and self._has_anthropic_oauth():
                _is_oauth = True
            elif _provider == "openai" and self._has_openai_oauth():
                _is_oauth = True
            else:
                _oauth_key = self._get_api_key_for_model(request.params.get("model", ""))
                _is_oauth = bool(_oauth_key and is_oauth_token(_oauth_key))

        _needs_budget = (is_llm and not _is_oauth) or request.service == "image_gen"
        use_budget_lock = bool(self.cost_tracker and agent_id and _needs_budget)

        if use_budget_lock:
            if agent_id not in self._budget_locks:
                self._budget_locks[agent_id] = asyncio.Lock()
            lock = self._budget_locks[agent_id]
        else:
            lock = None

        async def _execute() -> APIProxyResponse:
            if self.cost_tracker and agent_id and _needs_budget:
                if request.service == "image_gen":
                    budget_check = self.cost_tracker.check_budget(agent_id)
                    if not budget_check["allowed"]:
                        return APIProxyResponse(
                            success=False,
                            error=(
                                "Budget exceeded: "
                                f"${budget_check['daily_used']:.2f}"
                                f"/${budget_check['daily_limit']:.2f} daily, "
                                f"${budget_check['monthly_used']:.2f}"
                                f"/${budget_check['monthly_limit']:.2f} monthly"
                            ),
                        )
                else:
                    model = request.params.get("model", "unknown")
                    preflight = self.cost_tracker.preflight_check(agent_id, model)
                    if not preflight["allowed"]:
                        return APIProxyResponse(
                            success=False,
                            error=(
                                "Budget exceeded: "
                                f"${preflight['daily_used']:.2f}"
                                f"/${preflight['daily_limit']:.2f} daily, "
                                f"${preflight['monthly_used']:.2f}"
                                f"/${preflight['monthly_limit']:.2f} monthly "
                                f"(estimated next call: "
                                f"${preflight['estimated_cost']:.4f})"
                            ),
                        )

            handler = self.service_handlers.get(request.service)
            if not handler:
                return APIProxyResponse(success=False, error=f"Unknown service: {request.service}")
            try:
                response = await handler(request)

                if self.cost_tracker and agent_id and response.success and response.data:
                    if not response.data.get("oauth"):
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

                    fixed_cost = response.data.get("fixed_cost_usd")
                    if fixed_cost and fixed_cost > 0:
                        fc_model = response.data.get("model", request.service)
                        self.cost_tracker.track_fixed_cost(
                            agent_id, fc_model, fixed_cost,
                        )

                return response
            except Exception as e:
                logger.error(f"API call failed for {request.service}/{request.action}: {e}")
                return APIProxyResponse(
                    success=False, error=str(e),
                    status_code=getattr(e, "status_code", None),
                )

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

    # Overrides for non-standard model prefixes.  Standard providers
    # use ``provider/model`` format and are resolved automatically by
    # ``_resolve_provider()``.  Only bare-name prefixes (gpt-, o1, …)
    # and alternate prefixes (ollama_chat/) need explicit entries.
    _PROVIDER_PREFIX_OVERRIDES = {
        "gpt-": "openai",
        "o1": "openai",
        "o3": "openai",
        "o4": "openai",
        "ollama_chat/": "ollama",
        "text-embedding-": "openai",
    }

    def _resolve_provider(self, model: str) -> str | None:
        """Resolve a model name to its provider.

        Checks explicit overrides first (bare prefixes like ``gpt-``,
        ``o1``), then falls back to extracting the provider from the
        standard ``provider/model`` format.  This means any LiteLLM-
        supported provider works without code changes.
        """
        for prefix, provider in self._PROVIDER_PREFIX_OVERRIDES.items():
            if model.startswith(prefix):
                return provider
        if "/" in model:
            return model.split("/", 1)[0]
        return None

    def _get_api_key_for_model(self, model: str) -> str | None:
        """Resolve the API key for a model based on its provider prefix.

        Only checks system_credentials — LLM provider keys must use the
        ``OPENLEGION_SYSTEM_`` prefix.
        """
        provider = self._resolve_provider(model)
        if provider:
            return self.system_credentials.get(f"{provider}_api_key")
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

    def _is_keyless_provider(self, model: str) -> bool:
        """Check if a model belongs to a provider that doesn't need API keys.

        Uses ``_resolve_provider()`` so that prefixes like
        ``ollama_chat/`` correctly map to the ``ollama`` provider.
        """
        provider = self._resolve_provider(model)
        return provider in KEYLESS_PROVIDERS if provider else False

    def get_providers_with_credentials(self) -> set[str]:
        """Return the set of provider names that have credentials configured.

        Checks known providers from the model registry, plus any provider
        whose ``{name}_api_key`` is in system_credentials (so dynamically
        configured providers are detected automatically).

        For keyless providers (e.g. Ollama), checks if an API base is
        configured instead of an API key.  Use ``discover_ollama_models``
        for runtime availability.
        """
        providers: set[str] = set()
        for provider in SYSTEM_CREDENTIAL_PROVIDERS:
            if provider in KEYLESS_PROVIDERS:
                if f"{provider}_api_base" in self.api_bases:
                    providers.add(provider)
            else:
                if f"{provider}_api_key" in self.system_credentials:
                    providers.add(provider)

        # Also detect any provider keys not in the curated set
        for cred_name in self.system_credentials:
            if cred_name.endswith("_api_key"):
                provider = cred_name[: -len("_api_key")]
                if provider and provider not in providers:
                    providers.add(provider)

        # OpenAI Codex OAuth counts as having OpenAI credentials
        if self._has_openai_oauth():
            providers.add("openai")
        # Anthropic structured OAuth counts as having Anthropic credentials
        if self._has_anthropic_oauth():
            providers.add("anthropic")
        return providers

    _OLLAMA_DEFAULT_BASE = "http://localhost:11434"

    async def discover_ollama_models(self) -> list[str]:
        """Query the local Ollama instance for installed models.

        Returns model names in ``ollama/<name>`` format.
        Returns an empty list if Ollama is unreachable.
        """
        base = self.api_bases.get(
            "ollama_api_base", self._OLLAMA_DEFAULT_BASE,
        )
        try:
            client = await self._get_http_client()
            resp = await client.get(f"{base}/api/tags", timeout=2.0)
            if resp.status_code == 200:
                data = resp.json()
                models: list[str] = []
                for m in data.get("models", []):
                    name = m.get("name", "")
                    if not name:
                        continue
                    # Strip `:latest` suffix for cleaner display
                    if name.endswith(":latest"):
                        name = name[: -len(":latest")]
                    models.append(f"ollama/{name}")
                return sorted(set(models))
        except (httpx.HTTPError, OSError, KeyError, ValueError):
            pass
        return []

    # Exclude patterns for non-chat models from gateway discovery
    _GATEWAY_EXCLUDE_PATTERNS = ("embed", "tts", "whisper", "dall-e", "image", "audio", "moderation")

    async def discover_openlegion_models(
        self,
    ) -> tuple[list[str], dict[str, tuple[float, float]]]:
        """Query the openlegion credit proxy gateway for available models.

        Returns ``(model_ids, pricing)`` where model_ids are in
        ``openlegion/{creator}/{model}`` format and pricing maps
        ``creator/model`` → ``(input_per_1k, output_per_1k)`` USD.

        Returns ``([], {})`` if the gateway is not configured or unreachable.
        """
        api_base = self.api_bases.get("openlegion_api_base")
        api_key = self.system_credentials.get("openlegion_api_key")
        if not api_base:
            return [], {}

        try:
            client = await self._get_http_client()
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            resp = await client.get(f"{api_base}/models", headers=headers, timeout=10.0)
            if resp.status_code != 200:
                logger.warning("OpenLegion gateway /models returned %d", resp.status_code)
                return [], {}

            body = resp.json()
            models: list[str] = []
            pricing: dict[str, tuple[float, float]] = {}

            for entry in body.get("data", []):
                model_id = entry.get("id", "")
                if not model_id or not isinstance(model_id, str):
                    continue
                # Skip non-chat models
                lower = model_id.lower()
                if any(pat in lower for pat in self._GATEWAY_EXCLUDE_PATTERNS):
                    continue

                models.append(f"openlegion/{model_id}")

                # Extract pricing (per-token → per-1K-token)
                p = entry.get("pricing")
                if p and p.get("input") is not None and p.get("output") is not None:
                    try:
                        inp = float(p["input"]) * 1000
                        out = float(p["output"]) * 1000
                        pricing[model_id] = (inp, out)
                    except (ValueError, TypeError):
                        pass

            models.sort()
            return models, pricing
        except Exception:
            logger.debug("Failed to discover openlegion models", exc_info=True)
            return [], {}

    def _get_api_base_for_model(self, model: str) -> str | None:
        """Resolve a custom API base URL for a model's provider.

        Checks ``OPENLEGION_SYSTEM_<PROVIDER>_API_BASE`` first, then
        falls back to ``OPENLEGION_CRED_<PROVIDER>_API_BASE``.
        Returns *None* when no custom base is configured — LiteLLM
        uses its own defaults.
        """
        provider = self._resolve_provider(model)
        if provider:
            return self.api_bases.get(f"{provider}_api_base")
        return None

    def _rewrite_model_for_litellm(self, model: str, api_base: str | None) -> str:
        """Rewrite model strings for litellm compatibility.

        LiteLLM requires a recognized provider prefix to route API calls.

        For ``openlegion/`` models (credit proxy): the gateway expects the
        Vercel AI Gateway model ID (e.g., ``openai/gpt-5.4``) in the
        request body.  We prepend ``openai/`` so litellm treats it as an
        OpenAI-compatible endpoint — litellm strips that prefix and sends
        the remainder as the model name in the HTTP body.

        For other custom providers with ``api_base``: same ``openai/``
        prefix trick so litellm uses its OpenAI-compatible code path.
        """
        if not api_base:
            return model

        # openlegion credit proxy: strip "openlegion/" and prepend "openai/"
        # so litellm routes to the custom api_base.  litellm strips the
        # "openai/" prefix and sends the inner model ID in the request body
        # (e.g., "openai/gpt-5.4" or "anthropic/claude-sonnet-4-6").
        if model.startswith("openlegion/"):
            inner = model[len("openlegion/"):]
            return f"openai/{inner}"

        provider = self._resolve_provider(model)
        if provider and provider in _LITELLM_NATIVE_PROVIDERS:
            return model

        # Unknown provider with custom api_base → OpenAI-compatible
        if "/" in model:
            rewritten = f"openai/{model.split('/', 1)[1]}"
        else:
            rewritten = f"openai/{model}"
        logger.debug("Rewrote custom model '%s' → '%s' for litellm", model, rewritten)
        return rewritten

    @staticmethod
    def _is_permanent_error(error: Exception) -> bool:
        """Return True if the error should NOT cascade to fallback models.

        BadRequestError covers its subclasses: ContentPolicyViolationError,
        ContextWindowExceededError, UnsupportedParamsError, etc.
        NotFoundError means the model name itself is invalid — cascading
        would silently mask bad config.
        402 Payment Required means the credit proxy rejected the call for
        insufficient credits — all models route through the same proxy,
        so failover to another model is pointless.
        """
        import litellm
        if isinstance(error, (litellm.BadRequestError, litellm.NotFoundError)):
            return True
        if getattr(error, "status_code", 0) == 402:
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
            if not api_key and not self._is_keyless_provider(model):
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

    # Allowlist of LLM parameters that agents may pass through to the
    # provider.  Everything else is silently dropped so that an untrusted
    # agent cannot inject ``api_key``, ``api_base``, ``custom_llm_provider``,
    # or other security-sensitive litellm kwargs.
    _ALLOWED_LLM_PARAMS: frozenset[str] = frozenset({
        # Standard OpenAI-compatible params
        "max_tokens", "temperature", "top_p", "stop", "stream",
        "tools", "tool_choice", "response_format", "seed",
        "presence_penalty", "frequency_penalty", "logit_bias", "n",
        "logprobs", "top_logprobs", "user",
        # Anthropic thinking / extended-thinking
        "thinking", "max_completion_tokens",
        # OpenAI reasoning models
        "reasoning_effort",
    })

    def _prepare_llm_params(
        self, request: APIProxyRequest, model: str,
        api_base: str | None = None,
        auth_headers: dict[str, str] | None = None,
    ) -> tuple[list[dict], dict]:
        """Build sanitized messages and extra kwargs for an LLM call.

        Only parameters in ``_ALLOWED_LLM_PARAMS`` are forwarded from the
        agent request.  This prevents untrusted agents from injecting
        ``api_key``, ``api_base``, or other credential-bearing kwargs.

        Returns ``(sanitized_messages, extra_kwargs)``.
        """
        sanitized = sanitize_for_provider(request.params.get("messages", []), model)

        # Strip image blocks from tool messages for non-vision models
        # (e.g., Groq/Llama, DeepSeek) so they don't cause API errors.
        if not _model_supports_vision(model):
            for msg in sanitized:
                if msg.get("role") == "tool" and isinstance(msg.get("content"), list):
                    msg["content"] = " ".join(
                        b.get("text", "") for b in msg["content"]
                        if isinstance(b, dict) and b.get("type") == "text"
                    )

        extra: dict = {}
        for k, v in request.params.items():
            if k in ("model", "messages"):
                continue
            if k in self._ALLOWED_LLM_PARAMS:
                extra[k] = v
            else:
                logger.debug("Dropped non-allowlisted LLM param: %s", k)
        if api_base:
            extra["api_base"] = api_base
        if auth_headers:
            extra["extra_headers"] = {
                **extra.get("extra_headers", {}),
                **auth_headers,
            }

        # Ollama thinking mode + tool calling is broken (empty/malformed
        # output).  Disable thinking for Ollama when tools are present.
        # Uses reasoning_effort (OpenAI-standard param that LiteLLM maps
        # to Ollama's think=false) rather than the provider-specific
        # think param which LiteLLM silently drops.
        # See: https://github.com/ollama/ollama/issues/10976
        if (
            self._is_keyless_provider(model)
            and request.params.get("tools")
            and "reasoning_effort" not in extra
        ):
            extra["reasoning_effort"] = "none"

        return sanitized, extra

    # ── OAuth direct-call helpers ────────────────────────────

    # _oauth_headers has been removed — the Anthropic SDK handles auth
    # headers natively. For non-SDK callers (e.g. setup_wizard validation),
    # construct headers inline.

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
                raw_content = m.get("content", "")
                # Convert OpenAI image_url blocks to Anthropic format
                # (tool results can contain multimodal content from screenshots)
                if isinstance(raw_content, list):
                    raw_content = convert_openai_image_blocks(raw_content)
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
                # Convert OpenAI image_url blocks to Anthropic image format.
                # LiteLLM does this automatically on its code path; the OAuth
                # fast-path bypasses LiteLLM so we must convert manually.
                if isinstance(m.get("content"), list):
                    m = {**m, "content": convert_openai_image_blocks(m["content"])}
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

    # ── Anthropic OAuth helpers ───────────────────────────────

    # Required by Anthropic's OAuth backend — must be the first system block.
    _CLAUDE_CODE_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."

    @staticmethod
    def _patch_anthropic_oauth_body(body: dict) -> None:
        """Patch an Anthropic body for OAuth — required by pi-ai / openclaw.

        1. Convert ``system`` from a plain string to an array of content
           blocks with the Claude Code identity as the mandatory first block.
        2. Add an override so the model follows the agent's actual instructions
           instead of defaulting to Claude Code behavior.
        """
        identity = CredentialVault._CLAUDE_CODE_IDENTITY
        system_blocks: list[dict] = [{"type": "text", "text": identity}]
        existing = body.get("system", "")
        if existing:
            # The agent's real instructions come next — the model should
            # follow THESE, not the Claude Code identity above.
            if isinstance(existing, str):
                system_blocks.append({"type": "text", "text": existing})
            elif isinstance(existing, list):
                system_blocks.extend(existing)
        body["system"] = system_blocks

    async def _oauth_chat(
        self, request: APIProxyRequest, api_key: str, model: str,
    ) -> APIProxyResponse:
        """Anthropic OAuth call (non-streaming wrapper).

        pi-ai always streams — there is no non-streaming OAuth path.
        This method streams internally and collects the final result.
        """
        result: dict = {}
        last_error: str | None = None

        async for chunk in self._oauth_chat_stream(request, api_key, model):
            if not chunk.startswith("data: "):
                continue
            try:
                data = json.loads(chunk[6:].strip())
            except (json.JSONDecodeError, ValueError):
                continue
            if data.get("error"):
                last_error = data["error"]
            elif data.get("type") == "done":
                result = data

        if last_error:
            raise RuntimeError(f"Anthropic OAuth error: {last_error}")

        if not result:
            raise RuntimeError("Anthropic OAuth returned no response")

        result["oauth"] = True
        return APIProxyResponse(success=True, data=result)

    async def _oauth_chat_stream(
        self, request: APIProxyRequest, api_key: str, model: str,
    ):
        """Streaming Anthropic API call using OAuth bearer auth via SDK.

        Uses ``anthropic.AsyncAnthropic`` with ``auth_token`` for Bearer
        auth instead of raw httpx.  Yields SSE-formatted strings matching
        the ``stream_llm`` protocol.
        """
        import anthropic

        sanitized = sanitize_for_provider(
            request.params.get("messages", []), model,
        )
        params = {**request.params, "messages": sanitized}
        body = self._build_anthropic_body(params)
        self._patch_anthropic_oauth_body(body)

        # Extract SDK parameters from the converted body
        sdk_kwargs: dict = {
            "model": body["model"],
            "messages": body["messages"],
            "max_tokens": body.get("max_tokens", 4096),
            "stream": True,
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
        # Thinking/extended thinking support
        if "thinking" in body:
            sdk_kwargs["thinking"] = body["thinking"]

        # Debug: log token info to diagnose deployed instance issues
        token_preview = f"{api_key[:15]}...{api_key[-4:]}" if len(api_key) > 20 else "***"
        logger.info(
            "OAuth stream: model=%s, token=%s, token_len=%d",
            body["model"], token_preview, len(api_key),
        )

        client = anthropic.AsyncAnthropic(
            api_key=None,
            auth_token=api_key,
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

        collected_content = ""
        collected_thinking = ""
        collected_tool_calls: list[dict] = []
        input_tokens = 0
        output_tokens = 0
        current_tool_idx = -1

        try:
            stream = await client.messages.create(**sdk_kwargs)

            async for event in stream:
                if event.type == "message_start":
                    if hasattr(event, "message") and hasattr(event.message, "usage"):
                        input_tokens = event.message.usage.input_tokens
                elif event.type == "content_block_start":
                    cb = event.content_block
                    if cb.type == "tool_use":
                        current_tool_idx += 1
                        collected_tool_calls.append({
                            "id": cb.id,
                            "name": cb.name,
                            "arguments": "",
                        })
                    elif cb.type == "thinking":
                        pass  # thinking block started
                elif event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        collected_content += delta.text
                        yield f"data: {json.dumps({'type': 'text_delta', 'content': delta.text})}\n\n"
                    elif delta.type == "thinking_delta":
                        collected_thinking += delta.thinking
                    elif delta.type == "input_json_delta":
                        if current_tool_idx >= 0:
                            collected_tool_calls[current_tool_idx]["arguments"] += delta.partial_json
                elif event.type == "message_delta":
                    if hasattr(event, "usage"):
                        output_tokens = event.usage.output_tokens

            # Emit final done event
            tokens_used = input_tokens + output_tokens
            self._health_tracker.record_success(model)
            done_data: dict = {
                "type": "done",
                "content": collected_content,
                "tokens_used": tokens_used,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "model": f"anthropic/{body['model']}",
                "oauth": True,
                "tool_calls": [
                    {
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    }
                    for tc in collected_tool_calls
                ],
            }
            if collected_thinking:
                done_data["thinking_content"] = collected_thinking
            yield f"data: {json.dumps(done_data)}\n\n"

        except anthropic.AuthenticationError as e:
            self._health_tracker.record_failure(model, "AuthError", 401)
            msg = "Auth failed"
            if hasattr(e, "body") and isinstance(e.body, dict):
                msg = e.body.get("error", {}).get("message", msg)
            logger.error("OAuth AuthenticationError: token=%s, msg=%s", token_preview, msg)
            yield f"data: {json.dumps({'error': f'OAuth auth failed (token may have expired): {msg}'})}\n\n"

        except anthropic.APIStatusError as e:
            self._health_tracker.record_failure(model, "HTTPError", e.status_code)
            detail = ""
            if hasattr(e, "body") and isinstance(e.body, dict):
                detail = e.body.get("error", {}).get("message", "")
            msg = f"Anthropic API error (HTTP {e.status_code})"
            if detail:
                msg += f": {detail}"
            logger.error("OAuth APIStatusError (%d): token=%s, detail=%s", e.status_code, token_preview, detail)
            yield f"data: {json.dumps({'error': msg})}\n\n"

        except Exception as e:
            logger.error(f"OAuth streaming call failed: {e}")
            self._health_tracker.record_failure(model, type(e).__name__, 0)
            yield f"data: {json.dumps({'error': friendly_streaming_error(e)})}\n\n"

        finally:
            await client.close()

    # ── OpenAI Codex Responses API helpers ─────────────────

    def _has_openai_oauth(self) -> bool:
        """Return True if OpenAI OAuth credentials are available."""
        return self._openai_oauth is not None

    # ── Anthropic structured OAuth helpers ──────────────────

    def _has_anthropic_oauth(self) -> bool:
        """Return True if structured Anthropic OAuth credentials are available."""
        return self._anthropic_oauth is not None

    def store_anthropic_oauth(self, creds: dict) -> None:
        """Store Anthropic OAuth credentials in memory and .env.

        Accepts a dict with at least ``access_token``.  Optionally includes
        ``refresh_token`` and ``expires_at``.
        """
        self._anthropic_oauth = creds
        _persist_to_env("OPENLEGION_SYSTEM_ANTHROPIC_OAUTH", json.dumps(creds))
        logger.info("Anthropic OAuth credentials stored")

    @staticmethod
    def load_claude_cli_auth() -> dict | None:
        """Load OAuth credentials from ``~/.claude/.credentials.json``.

        Reads the ``claudeAiOauth`` field and normalizes camelCase keys
        to snake_case (``accessToken`` -> ``access_token``, etc.).

        Returns a normalized dict or None if unavailable.
        """
        creds_path = Path.home() / ".claude" / ".credentials.json"
        if not creds_path.exists():
            return None
        try:
            data = json.loads(creds_path.read_text())
            if not isinstance(data, dict):
                return None
            oauth_data = data.get("claudeAiOauth")
            if not isinstance(oauth_data, dict):
                return None
            if not oauth_data.get("accessToken"):
                return None
            # Normalize camelCase to snake_case
            normalized: dict = {
                "access_token": oauth_data["accessToken"],
            }
            if oauth_data.get("refreshToken"):
                normalized["refresh_token"] = oauth_data["refreshToken"]
            if oauth_data.get("expiresAt"):
                normalized["expires_at"] = oauth_data["expiresAt"]
            return normalized
        except (json.JSONDecodeError, OSError):
            return None

    async def _ensure_anthropic_oauth_token(self) -> str:
        """Return the access_token, checking expiry with a 5-minute buffer.

        Since the Anthropic OAuth refresh endpoint is not publicly documented,
        this raises ``RuntimeError`` when the token is expired rather than
        attempting a refresh.  Users should regenerate with ``claude setup-token``
        or re-import from Claude CLI.
        """
        if self._anthropic_oauth is None:
            raise RuntimeError("No Anthropic OAuth credentials configured")

        token = self._anthropic_oauth["access_token"]
        token_preview = f"{token[:15]}...{token[-4:]}" if len(token) > 20 else "***"
        logger.info("OAuth token resolved: %s (len=%d)", token_preview, len(token))

        now = int(time.time())
        expires_at = self._anthropic_oauth.get("expires_at", 0)

        # If no expiry set or still valid, return the token
        if expires_at == 0 or expires_at > now + 300:
            return self._anthropic_oauth["access_token"]

        async with self._anthropic_oauth_lock:
            # Double-check after lock
            expires_at = self._anthropic_oauth.get("expires_at", 0)
            if expires_at == 0 or expires_at > now + 300:
                return self._anthropic_oauth["access_token"]

            raise RuntimeError(
                "Anthropic OAuth token expired — regenerate with "
                "`claude setup-token` or re-import from Claude CLI"
            )

    @staticmethod
    def normalize_openai_oauth(data: dict) -> dict | None:
        """Normalize OpenAI OAuth credentials from Codex CLI nested or flat format.

        Codex CLI stores ``{tokens: {access_token, refresh_token, ...}, last_refresh}``.
        We need a flat dict with ``access_token`` at top level.

        Returns a flat dict or None if *data* lacks an ``access_token``.
        """
        # Nested: {tokens: {access_token, ...}}
        tokens = data.get("tokens")
        if isinstance(tokens, dict) and tokens.get("access_token"):
            result = {
                "access_token": tokens["access_token"],
                "refresh_token": tokens.get("refresh_token", ""),
            }
            if tokens.get("account_id"):
                result["account_id"] = tokens["account_id"]
            return result
        # Flat: {access_token, ...}
        if data.get("access_token"):
            return data
        return None

    def store_openai_oauth(self, creds: dict) -> None:
        """Store OpenAI OAuth credentials in memory and .env.

        Accepts a dict with at least ``access_token`` and ``refresh_token``.
        Automatically extracts ``account_id`` and ``expires_at`` from the JWT
        if not already present.
        """
        token = creds.get("access_token", "")
        if token and not creds.get("account_id"):
            creds["account_id"] = self._extract_account_id_from_jwt(token)
        if token and not creds.get("expires_at"):
            exp = self._extract_jwt_expiry(token)
            if exp:
                creds["expires_at"] = exp
        self._openai_oauth = creds
        _persist_to_env("OPENLEGION_SYSTEM_OPENAI_OAUTH", json.dumps(creds))
        logger.info("OpenAI OAuth credentials stored")

    @staticmethod
    def load_codex_auth() -> dict | None:
        """Load credentials from ``~/.codex/auth.json`` if present.

        Handles both flat ``{access_token, refresh_token}`` and nested
        ``{tokens: {access_token, refresh_token}}`` formats.
        """
        auth_path = Path.home() / ".codex" / "auth.json"
        if not auth_path.exists():
            return None
        try:
            data = json.loads(auth_path.read_text())
            if isinstance(data, dict):
                # Nested format: {tokens: {...}}
                if "tokens" in data and isinstance(data["tokens"], dict):
                    return data["tokens"]
                # Flat format
                if "access_token" in data:
                    return data
        except (json.JSONDecodeError, OSError):
            pass
        return None

    @staticmethod
    def _decode_jwt_payload(token: str) -> dict:
        """Decode the payload of a JWT without verification."""
        parts = token.split(".")
        if len(parts) < 2:
            return {}
        payload_b64 = parts[1]
        # Add padding
        payload_b64 += "=" * (-len(payload_b64) % 4)
        try:
            return json.loads(base64.urlsafe_b64decode(payload_b64))
        except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
            return {}

    @classmethod
    def _extract_account_id_from_jwt(cls, token: str) -> str:
        """Extract the ChatGPT account ID from a JWT claim."""
        payload = cls._decode_jwt_payload(token)
        return payload.get("https://api.openai.com/auth", {}).get("chatgpt_account_id", "")

    @classmethod
    def _extract_jwt_expiry(cls, token: str) -> int:
        """Extract the ``exp`` claim from a JWT."""
        payload = cls._decode_jwt_payload(token)
        return payload.get("exp", 0)

    async def _ensure_openai_oauth_token(self) -> tuple[str, str]:
        """Return ``(access_token, account_id)``, refreshing if needed.

        Uses a 5-minute buffer before expiry.  Double-checks after acquiring
        the lock to avoid redundant refreshes.
        """
        if self._openai_oauth is None:
            raise RuntimeError("No OpenAI OAuth credentials configured")

        now = int(time.time())
        expires_at = self._openai_oauth.get("expires_at", 0)
        if expires_at > now + 300:
            return (
                self._openai_oauth["access_token"],
                self._openai_oauth.get("account_id", ""),
            )

        async with self._openai_oauth_lock:
            # Double-check after lock
            expires_at = self._openai_oauth.get("expires_at", 0)
            if expires_at > now + 300:
                return (
                    self._openai_oauth["access_token"],
                    self._openai_oauth.get("account_id", ""),
                )

            refresh_token = self._openai_oauth.get("refresh_token", "")
            if not refresh_token:
                raise RuntimeError("No refresh_token in OpenAI OAuth credentials")

            client = await self._get_http_client()
            try:
                resp = await client.post(
                    _OPENAI_TOKEN_URL,
                    data={
                        "grant_type": "refresh_token",
                        "client_id": _OPENAI_OAUTH_CLIENT_ID,
                        "redirect_uri": _OPENAI_OAUTH_REDIRECT_URI,
                        "refresh_token": refresh_token,
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=30,
                )
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                raise RuntimeError(
                    f"OpenAI token refresh request failed: {exc}"
                ) from exc
            if not resp.is_success:
                raise RuntimeError(
                    f"OpenAI token refresh failed (HTTP {resp.status_code}): "
                    f"{resp.text[:200]}"
                )
            data = resp.json()
            new_access = data.get("access_token", "")
            new_refresh = data.get("refresh_token", refresh_token)
            account_id = self._extract_account_id_from_jwt(new_access)
            exp = self._extract_jwt_expiry(new_access)

            self._openai_oauth = {
                "access_token": new_access,
                "refresh_token": new_refresh,
                "account_id": account_id,
                "expires_at": exp,
            }
            _persist_to_env(
                "OPENLEGION_SYSTEM_OPENAI_OAUTH",
                json.dumps(self._openai_oauth),
            )
            logger.info("OpenAI OAuth token refreshed")
            return new_access, account_id

    @staticmethod
    def _openai_oauth_headers(
        access_token: str, account_id: str,
    ) -> dict[str, str]:
        """Build headers for the OpenAI Codex Responses API.

        Headers match what pi-ai sends: ``OpenAI-Beta`` is required,
        ``chatgpt-account-id`` uses lowercase (matching pi-ai's format).
        """
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "accept": "text/event-stream",
            "Authorization": f"Bearer {access_token}",
            "OpenAI-Beta": "responses=experimental",
        }
        if account_id:
            headers["chatgpt-account-id"] = account_id
        return headers

    @staticmethod
    def _build_openai_responses_body(params: dict) -> dict:
        """Convert LiteLLM-style params to Responses API format.

        The Responses API uses a different structure from Chat Completions:
        - ``instructions`` instead of system messages
        - ``input`` list with typed items instead of ``messages``
        - ``max_output_tokens`` instead of ``max_tokens``
        - Tools are unwrapped (no ``type: function`` wrapper)
        """
        model = params.get("model", "")
        if model.startswith("openai/"):
            model = model[len("openai/"):]

        # Message conversion matching pi-ai's convertResponsesMessages exactly.
        # System → instructions (NOT in input). User → {role, content} (no type).
        # Assistant → {type: "message", ...} with id. Tool calls → {type: "function_call"}.
        messages = params.get("messages", [])
        instructions_parts: list[str] = []
        input_items: list[dict] = []
        msg_idx = 0

        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")

            if role == "system":
                if isinstance(content, str):
                    instructions_parts.append(content)
                elif isinstance(content, list):
                    instructions_parts.append(" ".join(
                        b.get("text", "") for b in content if isinstance(b, dict)
                    ))

            elif role == "user":
                # pi-ai: {role: "user", content: [...]} — NO type: "message"
                if isinstance(content, str):
                    input_items.append({
                        "role": "user",
                        "content": [{"type": "input_text", "text": content}],
                    })
                elif isinstance(content, list):
                    blocks: list[dict] = []
                    for b in content:
                        if not isinstance(b, dict):
                            continue
                        btype = b.get("type", "")
                        if btype == "text":
                            blocks.append({"type": "input_text", "text": b.get("text", "")})
                        elif btype == "image_url":
                            url = b.get("image_url", {}).get("url", "")
                            blocks.append({
                                "type": "input_image",
                                "detail": "auto",
                                "image_url": url,
                            })
                    if blocks:
                        input_items.append({"role": "user", "content": blocks})
                msg_idx += 1

            elif role == "assistant":
                # pi-ai flattens assistant content into separate items
                if content:
                    text_val = content if isinstance(content, str) else str(content)
                    if text_val:
                        input_items.append({
                            "type": "message",
                            "role": "assistant",
                            "content": [{
                                "type": "output_text",
                                "text": text_val,
                                "annotations": [],
                            }],
                            "status": "completed",
                            "id": f"msg_{msg_idx}",
                        })
                for tc in (m.get("tool_calls") or []):
                    func = tc.get("function", tc)
                    input_items.append({
                        "type": "function_call",
                        "call_id": tc.get("id", ""),
                        "name": func.get("name", ""),
                        "arguments": func.get("arguments", "{}"),
                    })
                msg_idx += 1

            elif role == "tool":
                input_items.append({
                    "type": "function_call_output",
                    "call_id": m.get("tool_call_id", ""),
                    "output": content if isinstance(content, str) else json.dumps(content),
                })
                msg_idx += 1

        # Build body matching EXACTLY what pi-ai sends to the Codex backend.
        body: dict = {
            "model": model,
            "input": input_items,
            "store": False,
            "stream": True,
            "text": {"verbosity": "medium"},
            "include": ["reasoning.encrypted_content"],
        }
        if instructions_parts:
            body["instructions"] = "\n\n".join(instructions_parts)

        # Tools — only set tool_choice/parallel_tool_calls when tools exist
        tools = params.get("tools")
        if tools:
            unwrapped: list[dict] = []
            for t in tools:
                if "function" in t:
                    func = t["function"]
                    unwrapped.append({
                        "type": "function",
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "parameters": func.get("parameters", {"type": "object"}),
                        "strict": None,
                    })
                else:
                    unwrapped.append(t)
            body["tools"] = unwrapped
            body["tool_choice"] = "auto"
            body["parallel_tool_calls"] = True

        return body

    @staticmethod
    def _parse_openai_responses_response(data: dict, model_prefix: str) -> dict:
        """Parse a Responses API response into our standard result dict."""
        content = ""
        thinking_content = ""
        tool_calls: list[dict] = []

        for item in data.get("output", []):
            item_type = item.get("type", "")
            if item_type == "message":
                for block in item.get("content", []):
                    btype = block.get("type", "")
                    if btype == "output_text":
                        content += block.get("text", "")
            elif item_type == "function_call":
                tool_calls.append({
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments", "{}"),
                })
            elif item_type == "reasoning":
                for block in item.get("summary", []):
                    if isinstance(block, dict) and block.get("text"):
                        thinking_content += block["text"]

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

    async def _openai_oauth_chat(
        self, request: APIProxyRequest, model: str,
    ) -> APIProxyResponse:
        """Codex Responses API call (non-streaming wrapper).

        The Codex endpoint at ``chatgpt.com`` requires ``stream: true`` —
        there is no non-streaming mode.  This method streams internally
        and collects the final result from SSE events.
        """
        result: dict = {}
        last_error: str | None = None

        async for chunk in self._openai_oauth_chat_stream(request, model):
            if not chunk.startswith("data: "):
                continue
            try:
                data = json.loads(chunk[6:].strip())
            except (json.JSONDecodeError, ValueError):
                continue
            if data.get("error"):
                last_error = data["error"]
            elif data.get("type") == "done":
                result = data

        if last_error:
            raise RuntimeError(f"OpenAI Codex API error: {last_error}")

        if not result:
            raise RuntimeError("OpenAI Codex returned no response")

        result["oauth"] = True
        return APIProxyResponse(success=True, data=result)

    async def _openai_oauth_chat_stream(
        self, request: APIProxyRequest, model: str,
    ):
        """Streaming Codex Responses API call.

        Yields SSE-formatted strings matching the ``stream_llm`` protocol.
        SSE events: ``response.output_text.delta``,
        ``response.reasoning_summary_text.delta``,
        ``response.output_item.added``,
        ``response.function_call_arguments.delta``,
        ``response.function_call_arguments.done``,
        ``response.completed``, ``response.failed``.
        """
        sanitized = sanitize_for_provider(
            request.params.get("messages", []), model,
        )
        params = {**request.params, "messages": sanitized}
        body = self._build_openai_responses_body(params)

        collected_content = ""
        collected_thinking = ""
        collected_tool_calls: list[dict] = []
        input_tokens = 0
        output_tokens = 0
        # Map call_id → index in collected_tool_calls
        call_id_to_idx: dict[str, int] = {}

        try:
            access_token, account_id = await self._ensure_openai_oauth_token()
            headers = self._openai_oauth_headers(access_token, account_id)

            client = await self._get_http_client()
            async with client.stream(
                "POST", _OPENAI_RESPONSES_URL, headers=headers,
                json=body, timeout=120,
            ) as resp:
                if resp.status_code == 401:
                    self._health_tracker.record_failure(model, "AuthError", 401)
                    await resp.aread()
                    detail = resp.text[:500] if resp.text else ""
                    msg = f"OpenAI Codex auth failed (token may have expired): {detail}"
                    yield f"data: {json.dumps({'error': msg})}\n\n"
                    return
                if not resp.is_success:
                    self._health_tracker.record_failure(
                        model, "HTTPError", resp.status_code,
                    )
                    await resp.aread()
                    detail = resp.text[:500] if resp.text else ""
                    msg = f"OpenAI Codex API error (HTTP {resp.status_code}): {detail}"
                    yield f"data: {json.dumps({'error': msg})}\n\n"
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

                    if etype == "response.output_text.delta":
                        text = event.get("delta", "")
                        if text:
                            collected_content += text
                            yield f"data: {json.dumps({'type': 'text_delta', 'content': text})}\n\n"

                    elif etype == "response.reasoning_summary_text.delta":
                        text = event.get("delta", "")
                        if text:
                            collected_thinking += text

                    elif etype == "response.output_item.added":
                        item = event.get("item", {})
                        if item.get("type") == "function_call":
                            idx = len(collected_tool_calls)
                            collected_tool_calls.append({
                                "name": item.get("name", ""),
                                "arguments": "",
                            })
                            # Map both item.id and item.call_id so delta
                            # events match regardless of which key they
                            # reference (varies by API version/model).
                            for id_key in ("call_id", "id"):
                                id_val = item.get(id_key, "")
                                if id_val:
                                    call_id_to_idx[id_val] = idx

                    elif etype == "response.function_call_arguments.delta":
                        delta = event.get("delta", "")
                        lookup = event.get("call_id") or event.get("item_id", "")
                        idx = call_id_to_idx.get(lookup)
                        if idx is not None and delta:
                            collected_tool_calls[idx]["arguments"] += delta
                        elif delta:
                            logger.debug(
                                "Codex streaming: unmatched argument delta "
                                "(keys in event: %s, known IDs: %s)",
                                sorted(event.keys()),
                                sorted(call_id_to_idx.keys()),
                            )

                    elif etype == "response.function_call_arguments.done":
                        lookup = event.get("call_id") or event.get("item_id", "")
                        arguments = event.get("arguments", "")
                        idx = call_id_to_idx.get(lookup)
                        if idx is not None and arguments:
                            collected_tool_calls[idx]["arguments"] = arguments

                    elif etype == "response.completed":
                        response_data = event.get("response", {})
                        usage = response_data.get("usage", {})
                        input_tokens = usage.get("input_tokens", input_tokens)
                        output_tokens = usage.get("output_tokens", output_tokens)

                        # Backfill tool call arguments from the completed
                        # response.  Streaming deltas can be lost if the
                        # API changes the ID field used to correlate them;
                        # the completed response always has the final state.
                        fc_idx = 0
                        for out_item in response_data.get("output", []):
                            if out_item.get("type") != "function_call":
                                continue
                            if fc_idx < len(collected_tool_calls):
                                tc = collected_tool_calls[fc_idx]
                                final_args = out_item.get("arguments", "")
                                if not tc["arguments"] and final_args:
                                    logger.info(
                                        "Backfilling empty arguments for "
                                        "'%s' from completed response",
                                        tc["name"],
                                    )
                                    tc["arguments"] = final_args
                            fc_idx += 1

                    elif etype == "response.failed":
                        error_info = event.get("response", {}).get("error", {})
                        err_msg = error_info.get("message", "Codex request failed")
                        yield f"data: {json.dumps({'error': err_msg})}\n\n"
                        return

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

        except Exception as e:
            logger.error(f"OpenAI Codex streaming call failed: {e}")
            self._health_tracker.record_failure(model, type(e).__name__, 0)
            yield f"data: {json.dumps({'error': friendly_streaming_error(e)})}\n\n"

    async def _handle_llm(self, request: APIProxyRequest) -> APIProxyResponse:
        """Unified LLM handler. Auto-detects provider from model prefix via LiteLLM.

        OAuth setup-tokens bypass LiteLLM entirely — direct httpx calls to
        the provider's API with Bearer auth.
        """
        import litellm

        # Silently drop params unsupported by the target model (e.g. gpt-5
        # doesn't accept temperature; o-series models drop top_p etc.).
        # This is model-aware: litellm only drops params the specific model
        # doesn't support, not params it does.
        litellm.drop_params = True

        requested_model = request.params.get("model", "")

        if request.action == "chat":
            provider = self._resolve_provider(requested_model)

            # OAuth takes priority over API keys for chat — subscription
            # auth is preferred when available (no per-call cost).
            if self._has_anthropic_oauth() and provider == "anthropic":
                access_token = await self._ensure_anthropic_oauth_token()
                return await self._oauth_chat(request, access_token, requested_model)

            api_key = self._get_api_key_for_model(requested_model)
            if api_key and is_oauth_token(api_key):
                return await self._oauth_chat(request, api_key, requested_model)

            if self._has_openai_oauth() and provider == "openai":
                return await self._openai_oauth_chat(request, requested_model)

            async def _chat(
                model: str, api_key: str | None,
                api_base: str | None = None,
                auth_headers: dict[str, str] | None = None,
            ):
                sanitized, extra = self._prepare_llm_params(
                    request, model, api_base, auth_headers,
                )
                llm_kwargs: dict = {
                    "model": self._rewrite_model_for_litellm(model, api_base),
                    "messages": sanitized,
                    **extra,
                }
                if api_key:
                    llm_kwargs["api_key"] = api_key
                return await litellm.acompletion(**llm_kwargs)

            response, used_model = await self._call_llm_with_failover(
                requested_model, _chat,
            )
            msg = response.choices[0].message
            usage = response.usage
            content, thinking_content = _extract_content(msg.content)
            # Fallback: some litellm versions put thinking in a separate attribute
            if thinking_content is None:
                thinking_content = getattr(msg, "reasoning_content", None) or None
            # Use thinking as content when model produced only reasoning
            if not content and thinking_content:
                content = thinking_content
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
            if not api_key and not self._is_keyless_provider(requested_model):
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
            if api_key:
                embed_kwargs["api_key"] = api_key
            embed_model = self._rewrite_model_for_litellm(
                request.params["model"], api_base,
            )
            response = await litellm.aembedding(
                model=embed_model,
                input=request.params.get("text", ""),
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

        litellm.drop_params = True

        requested_model = request.params.get("model", "")

        provider = self._resolve_provider(requested_model)

        # OAuth takes priority over API keys — no cost tracking needed.
        if self._has_anthropic_oauth() and provider == "anthropic":
            access_token = await self._ensure_anthropic_oauth_token()
            async for chunk in self._oauth_chat_stream(request, access_token, requested_model):
                yield chunk
            return

        api_key = self._get_api_key_for_model(requested_model)
        if api_key and is_oauth_token(api_key):
            async for chunk in self._oauth_chat_stream(request, api_key, requested_model):
                yield chunk
            return

        if self._has_openai_oauth() and provider == "openai":
            async for chunk in self._openai_oauth_chat_stream(request, requested_model):
                yield chunk
            return

        if self.cost_tracker and agent_id and request.service == "llm":
            preflight = self.cost_tracker.preflight_check(agent_id, requested_model)
            if not preflight["allowed"]:
                yield f"data: {json.dumps({'error': 'Budget exceeded'})}\n\n"
                return

        models_to_try = self._failover_chain.get_models_to_try(requested_model)

        response = None
        used_model = requested_model
        last_error: Exception | None = None

        for model in models_to_try:
            api_key, auth_headers = self._get_auth_for_model(model)
            if not api_key and not self._is_keyless_provider(model):
                continue
            api_base = self._get_api_base_for_model(model)
            try:
                sanitized, extra = self._prepare_llm_params(
                    request, model, api_base, auth_headers,
                )
                llm_kwargs: dict = {
                    "model": self._rewrite_model_for_litellm(model, api_base),
                    "messages": sanitized,
                    "stream": True,
                    **extra,
                }
                if api_key:
                    llm_kwargs["api_key"] = api_key
                response = await litellm.acompletion(**llm_kwargs)
                used_model = model
                if model != requested_model:
                    logger.info(f"Stream failover: '{requested_model}' → '{model}'")
                break
            except Exception as e:
                status_code = self._get_status_code(e)
                self._health_tracker.record_failure(model, type(e).__name__, status_code)
                if self._is_permanent_error(e):
                    error_data: dict = {'error': str(e)}
                    if getattr(e, 'status_code', 0) == 402:
                        error_data['credit_exhausted'] = True
                    yield f"data: {json.dumps(error_data)}\n\n"
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
            chunk_count = 0
            # Cache once — avoids repeated string-prefix checks per chunk.
            is_local = self._is_keyless_provider(used_model)

            # Iterate with keepalive: send SSE comments every 15s so
            # downstream read timeouts (agent → mesh, dashboard → agent)
            # don't fire while waiting for slow providers like Ollama.
            # Uses asyncio.wait (not wait_for) to avoid cancelling the
            # pending __anext__ — cancellation would corrupt the iterator.
            _KEEPALIVE_INTERVAL = 15
            chunk_iter = response.__aiter__()
            next_chunk = asyncio.ensure_future(chunk_iter.__anext__())
            try:
                while True:
                    done, _ = await asyncio.wait(
                        {next_chunk}, timeout=_KEEPALIVE_INTERVAL,
                    )
                    if not done:
                        yield ": keepalive\n\n"
                        continue
                    try:
                        chunk = next_chunk.result()
                    except StopAsyncIteration:
                        break

                    chunk_count += 1
                    delta = chunk.choices[0].delta if chunk.choices else None
                    # Log the first chunk from local models to aid debugging
                    # empty-response issues.
                    if chunk_count == 1 and is_local:
                        try:
                            delta_dict = (
                                delta.model_dump()
                                if delta and hasattr(delta, "model_dump")
                                else repr(delta)
                            )
                        except Exception:
                            delta_dict = repr(delta)
                        logger.debug(
                            "First streaming chunk for %s: choices=%d delta=%s",
                            used_model,
                            len(chunk.choices) if chunk.choices else 0,
                            delta_dict,
                        )
                    if delta is not None:
                        if delta.content:
                            collected_content += delta.content
                            yield f"data: {json.dumps({'type': 'text_delta', 'content': delta.content})}\n\n"

                        # Collect thinking/reasoning tokens.  For local
                        # providers (Ollama) we also stream them so the
                        # user sees progress — they are often the only
                        # output for reasoning models (qwen3, deepseek-r1).
                        # Only stream reasoning when no real content has
                        # arrived yet; once content flows the reasoning is
                        # internal and the done event will carry the answer.
                        reasoning = getattr(delta, "reasoning_content", None)
                        if reasoning and isinstance(reasoning, str):
                            collected_thinking += reasoning
                            if is_local and not collected_content:
                                yield f"data: {json.dumps({'type': 'text_delta', 'content': reasoning})}\n\n"

                        if delta.tool_calls:
                            for tc in delta.tool_calls:
                                idx = tc.index if hasattr(tc, 'index') else 0
                                while len(collected_tool_calls) <= idx:
                                    collected_tool_calls.append({"name": "", "arguments": ""})
                                if tc.function and tc.function.name:
                                    collected_tool_calls[idx]["name"] = tc.function.name
                                if tc.function and tc.function.arguments:
                                    collected_tool_calls[idx]["arguments"] += tc.function.arguments

                    next_chunk = asyncio.ensure_future(chunk_iter.__anext__())
            finally:
                if not next_chunk.done():
                    next_chunk.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await next_chunk

            # Emit final summary
            tokens_used = 0
            prompt_tokens = 0
            completion_tokens = 0
            if hasattr(response, 'usage') and response.usage:
                tokens_used = response.usage.total_tokens
                prompt_tokens = getattr(response.usage, 'prompt_tokens', 0) or 0
                completion_tokens = getattr(response.usage, 'completion_tokens', 0) or 0

            # If the model produced only reasoning tokens (common with
            # Ollama thinking models like deepseek-r1, qwen3), use that
            # as content so the response is not empty.
            if not collected_content and collected_thinking:
                collected_content = collected_thinking
                logger.info(
                    "Model %s returned only reasoning tokens — using as content",
                    used_model,
                )
            elif not collected_content and not collected_thinking:
                logger.warning(
                    "Model %s produced no content and no reasoning tokens "
                    "(tokens_used=%d)",
                    used_model, tokens_used,
                )

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

    # ── Image generation ──────────────────────────────────────

    _GEMINI_IMAGE_MODELS = [
        "gemini-2.5-flash-image",
        "gemini-2.0-flash-image-generation",
    ]
    _GEMINI_IMAGE_BASE = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
    )

    _IMAGE_GEN_COSTS = {
        "gemini": 0.04,     # Gemini image gen per image
        "openai": 0.04,     # DALL-E 3 standard 1024x1024
    }

    _OPENAI_SIZE_MAP = {
        "square": "1024x1024",
        "landscape": "1792x1024",
        "portrait": "1024x1792",
    }

    @staticmethod
    def _default_image_provider() -> str:
        """Read the configured default image gen provider from settings."""
        try:
            p = Path("config/settings.json")
            if p.exists():
                data = json.loads(p.read_text())
                prov = data.get("image_gen_provider", "gemini")
                if prov in ("gemini", "openai"):
                    return prov
        except (json.JSONDecodeError, OSError):
            pass
        return "gemini"

    async def _handle_image_gen(self, request: APIProxyRequest) -> APIProxyResponse:
        """Dispatch image generation to the requested provider."""
        prompt = request.params.get("prompt", "").strip()
        if not prompt:
            return APIProxyResponse(success=False, error="prompt is required")

        provider = request.params.get("provider") or self._default_image_provider()
        if provider == "gemini":
            return await self._image_gen_gemini(request)
        if provider == "openai":
            return await self._image_gen_openai(request)
        return APIProxyResponse(
            success=False,
            error=f"Unknown image_gen provider: {provider}. Use 'gemini' or 'openai'.",
        )

    async def _image_gen_gemini(self, request: APIProxyRequest) -> APIProxyResponse:
        """Generate an image via Gemini's generateContent API.

        Tries models in _GEMINI_IMAGE_MODELS order, falling back on
        403/404 (model unavailable) so we survive model deprecation.
        """
        api_key = self.system_credentials.get("gemini_api_key")
        if not api_key:
            api_key = self.system_credentials.get("google_api_key")
        if not api_key:
            return APIProxyResponse(
                success=False,
                error="Gemini API key not configured (set OPENLEGION_SYSTEM_GEMINI_API_KEY)",
            )

        prompt = request.params.get("prompt", "")
        client = await self._get_http_client()
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseModalities": ["IMAGE", "TEXT"],
            },
        }

        last_error = ""
        for model in self._GEMINI_IMAGE_MODELS:
            url = f"{self._GEMINI_IMAGE_BASE}{model}:generateContent?key={api_key}"
            try:
                resp = await client.post(url, json=body, timeout=60)
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                return APIProxyResponse(
                    success=False, error=f"Gemini request failed: {e}",
                )

            # Model unavailable — try next in chain
            if resp.status_code in (403, 404):
                last_error = (
                    f"{model}: {resp.status_code} {resp.text[:200]}"
                )
                logger.warning(
                    "Gemini image model %s unavailable (%s), trying next",
                    model, resp.status_code,
                )
                continue

            if not resp.is_success:
                return APIProxyResponse(
                    success=False,
                    error=f"Gemini API error {resp.status_code}: "
                    f"{resp.text[:500]}",
                    status_code=resp.status_code,
                )

            data = resp.json()
            for candidate in data.get("candidates", []):
                for part in candidate.get("content", {}).get("parts", []):
                    inline = part.get("inlineData")
                    if inline and inline.get("mimeType", "").startswith(
                        "image/"
                    ):
                        cost = self._IMAGE_GEN_COSTS["gemini"]
                        return APIProxyResponse(
                            success=True,
                            data={
                                "image_base64": inline["data"],
                                "mime_type": inline["mimeType"],
                                "model": model,
                                "fixed_cost_usd": cost,
                            },
                        )

            return APIProxyResponse(
                success=False,
                error="Gemini returned no image data",
            )

        return APIProxyResponse(
            success=False,
            error=f"All Gemini image models unavailable: {last_error}",
        )

    async def _image_gen_openai(self, request: APIProxyRequest) -> APIProxyResponse:
        """Generate an image via OpenAI DALL-E 3 API."""
        api_key = self.system_credentials.get("openai_api_key")
        if not api_key:
            return APIProxyResponse(
                success=False,
                error="OpenAI API key not configured (set OPENLEGION_SYSTEM_OPENAI_API_KEY)",
            )

        prompt = request.params.get("prompt", "")
        size_name = request.params.get("size", "square")
        size = self._OPENAI_SIZE_MAP.get(size_name, "1024x1024")

        client = await self._get_http_client()
        body = {
            "model": "dall-e-3",
            "prompt": prompt,
            "n": 1,
            "size": size,
            "response_format": "b64_json",
        }

        try:
            resp = await client.post(
                "https://api.openai.com/v1/images/generations",
                headers={"Authorization": f"Bearer {api_key}"},
                json=body,
                timeout=60,
            )
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            return APIProxyResponse(success=False, error=f"OpenAI request failed: {e}")

        if not resp.is_success:
            return APIProxyResponse(
                success=False,
                error=f"OpenAI API error {resp.status_code}: {resp.text[:500]}",
                status_code=resp.status_code,
            )

        data = resp.json()
        images = data.get("data", [])
        if not images:
            return APIProxyResponse(success=False, error="OpenAI returned no image data")

        image_b64 = images[0].get("b64_json", "")
        cost = self._IMAGE_GEN_COSTS["openai"]
        return APIProxyResponse(
            success=True,
            data={
                "image_base64": image_b64,
                "mime_type": "image/png",
                "model": "dall-e-3",
                "fixed_cost_usd": cost,
            },
        )
