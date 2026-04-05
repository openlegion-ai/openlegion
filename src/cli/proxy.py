"""Per-agent proxy resolution and URL utilities.

Resolves proxy configuration for each agent using the resolution chain:
  1. Agent's agents.yaml proxy config (custom/direct/inherit)
  2. System proxy: BROWSER_PROXY_* env vars (managed) or OPENLEGION_SYSTEM_PROXY (self-hosted)
  3. No proxy
"""

from __future__ import annotations

import logging
import os
from urllib.parse import quote, unquote, urlparse, urlunparse

logger = logging.getLogger("proxy")

_VALID_SCHEMES = {"http", "https", "socks5"}

_MANDATORY_NO_PROXY = "host.docker.internal,127.0.0.1,localhost"


def sanitize_agent_id_for_env(agent_id: str) -> str:
    """Sanitize an agent ID for use in env var names (replace non-alphanumeric with _)."""
    import re
    return re.sub(r"[^A-Za-z0-9_]", "_", agent_id)


def validate_proxy_url(url: str) -> bool:
    """Return True if url is a valid proxy URL with supported scheme and port."""
    if not url:
        return False
    try:
        parsed = urlparse(url)
        if parsed.scheme not in _VALID_SCHEMES:
            return False
        if not parsed.hostname:
            return False
        if not parsed.port:
            return False
        return True
    except Exception:
        return False


def parse_proxy_url(url: str) -> dict | None:
    """Parse a proxy URL into components.

    Returns dict with keys: url (without auth), username, password, full_url.
    Returns None if the URL is invalid.
    """
    if not validate_proxy_url(url):
        return None
    parsed = urlparse(url)
    no_auth = urlunparse((
        parsed.scheme,
        f"{parsed.hostname}:{parsed.port}",
        parsed.path, parsed.params, parsed.query, parsed.fragment,
    ))
    return {
        "url": no_auth,
        "username": unquote(parsed.username or ""),
        "password": unquote(parsed.password or ""),
        "full_url": url,
    }


def _assemble_proxy_url(base_url: str, username: str = "", password: str = "") -> str:
    """Assemble a full proxy URL from components, URL-encoding credentials."""
    if not username:
        return base_url
    parsed = urlparse(base_url)
    user_part = quote(username, safe="")
    if password:
        user_part += ":" + quote(password, safe="")
    netloc = f"{user_part}@{parsed.hostname}:{parsed.port}"
    return urlunparse((
        parsed.scheme, netloc,
        parsed.path, parsed.params, parsed.query, parsed.fragment,
    ))


def _resolve_system_proxy() -> str | None:
    """Resolve system proxy from env vars. BROWSER_PROXY_* takes precedence."""
    browser_url = os.environ.get("BROWSER_PROXY_URL", "")
    if browser_url:
        user = os.environ.get("BROWSER_PROXY_USER", "")
        pwd = os.environ.get("BROWSER_PROXY_PASS", "")
        full = _assemble_proxy_url(browser_url, user, pwd)
        if validate_proxy_url(full):
            return full
        logger.warning("BROWSER_PROXY_URL is set but invalid: %s", browser_url)

    system_proxy = os.environ.get("OPENLEGION_SYSTEM_PROXY", "")
    if system_proxy:
        if validate_proxy_url(system_proxy):
            return system_proxy
        logger.warning("OPENLEGION_SYSTEM_PROXY is set but invalid")

    return None


def resolve_agent_proxy(
    agent_id: str,
    agents_cfg: dict,
    network_cfg: dict,
) -> str | None:
    """Resolve proxy URL for an agent using the resolution chain.

    Args:
        agent_id: The agent identifier.
        agents_cfg: Dict of agent configs from agents.yaml (keyed by agent_id).
        network_cfg: Dict from network.yaml (used for no_proxy, not proxy URL).

    Returns:
        Full proxy URL string, or None if no proxy should be used.
    """
    agent = agents_cfg.get(agent_id, {})
    proxy_cfg = agent.get("proxy", {})
    mode = proxy_cfg.get("mode", "inherit")

    if mode == "direct":
        return None

    if mode == "custom":
        cred_name = proxy_cfg.get("credential", "")
        if cred_name:
            env_key = f"OPENLEGION_CRED_{cred_name}"
            proxy_url = os.environ.get(env_key, "")
            if proxy_url and validate_proxy_url(proxy_url):
                return proxy_url
            if proxy_url:
                logger.warning(
                    "[%s] Proxy credential '%s' has invalid URL, falling through to system proxy",
                    agent_id, cred_name,
                )
            else:
                logger.warning(
                    "[%s] Proxy credential '%s' not found (env var %s), falling through to system proxy",
                    agent_id, cred_name, env_key,
                )
        else:
            logger.warning("[%s] mode=custom but no credential specified, falling through", agent_id)

    return _resolve_system_proxy()


def build_proxy_env_vars(proxy_url: str | None, no_proxy_user: str = "") -> dict[str, str]:
    """Build the env var dict to inject into an agent container.

    Returns empty dict if proxy_url is None (direct mode).
    """
    if not proxy_url:
        return {}
    no_proxy_parts = [_MANDATORY_NO_PROXY]
    if no_proxy_user:
        no_proxy_parts.append(no_proxy_user)
    return {
        "HTTP_PROXY": proxy_url,
        "HTTPS_PROXY": proxy_url,
        "NO_PROXY": ",".join(no_proxy_parts),
    }
