"""HTTP request tool for direct internet access from agent containers.

Supports ``$CRED{name}`` handles in URLs, headers, and body — resolved
via the mesh vault at execution time so the agent never sees raw secrets.
"""

from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
from urllib.parse import urlparse

import httpx

from src.agent.skills import skill

_MAX_BODY = 50_000

_CRED_HANDLE_RE = re.compile(r"\$CRED\{([^}]+)\}")

# Shared client for connection pooling across tool invocations.
# Avoids repeated TCP handshakes and TLS negotiation.
_client: httpx.AsyncClient | None = None
_client_lock = asyncio.Lock()


async def _check_redirect_ssrf(request: httpx.Request) -> None:
    """Event hook: validate every redirect target against SSRF rules.

    httpx AsyncClient calls this *before* following each redirect, so an
    attacker cannot bounce through a public URL into a private network.
    Runs DNS check in executor to avoid blocking the event loop.
    """
    loop = asyncio.get_running_loop()
    if await loop.run_in_executor(None, _is_private_url, str(request.url)):
        raise httpx.TooManyRedirects(
            "SSRF protection: redirect to private/internal address blocked",
            request=request,
        )


async def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is not None and not _client.is_closed:
        return _client
    async with _client_lock:
        if _client is None or _client.is_closed:
            _client = httpx.AsyncClient(
                follow_redirects=True,
                max_redirects=5,
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
                event_hooks={"request": [_check_redirect_ssrf]},
            )
        return _client


async def close_client() -> None:
    """Close the shared HTTP client. Called during agent shutdown."""
    global _client
    async with _client_lock:
        if _client and not _client.is_closed:
            await _client.aclose()
        _client = None


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True if the IP is in a range that should be blocked for SSRF."""
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
        return True
    # IPv4-mapped IPv6 (e.g. ::ffff:127.0.0.1) — check the mapped v4 address too
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        mapped = ip.ipv4_mapped
        if mapped.is_private or mapped.is_loopback or mapped.is_link_local or mapped.is_reserved:
            return True
    return False


def _is_private_url(url: str) -> bool:
    """Reject requests to private IP ranges, loopback, link-local, and reserved addresses."""
    parsed = urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        return True  # Reject malformed URLs
    try:
        ip = ipaddress.ip_address(hostname)
        return _is_blocked_ip(ip)
    except ValueError:
        # It's a hostname — resolve to check the actual IP
        try:
            resolved = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
            for family, _, _, _, sockaddr in resolved:
                ip = ipaddress.ip_address(sockaddr[0])
                if _is_blocked_ip(ip):
                    return True
        except (socket.gaierror, OSError):
            return True  # DNS resolution failed — fail closed (block)
        return False


async def _resolve_creds(text: str, mesh_client) -> tuple[str, list[str]]:
    """Replace all $CRED{name} handles in *text* with resolved values.

    Returns ``(resolved_text, list_of_resolved_secret_values)`` so callers
    can redact those values from output.
    """
    matches = _CRED_HANDLE_RE.findall(text)
    if not matches:
        return text, []
    if not mesh_client:
        raise ValueError("$CRED{} handles require mesh connectivity")
    secrets: list[str] = []
    for cred_name in set(matches):  # dedupe to avoid redundant vault calls
        value = await mesh_client.vault_resolve(cred_name)
        if value is None:
            raise ValueError(f"Credential not found: {cred_name}")
        text = text.replace(f"$CRED{{{cred_name}}}", value)
        secrets.append(value)
    return text, secrets


def _redact(text: str, secrets: list[str]) -> str:
    """Replace any occurrence of secret values with [REDACTED]."""
    for s in secrets:
        if s and s in text:
            text = text.replace(s, "[REDACTED]")
    return text


@skill(
    name="http_request",
    description=(
        "Make an HTTP request. Use this to call APIs, download web pages, "
        "or interact with any HTTP service. "
        "Supports $CRED{name} handles in url, headers, and body — "
        "they are resolved from the vault automatically."
    ),
    parameters={
        "url": {"type": "string", "description": "Full URL to request"},
        "method": {
            "type": "string",
            "description": "HTTP method (GET, POST, PUT, DELETE, PATCH)",
            "default": "GET",
        },
        "headers": {
            "type": "object",
            "description": (
                "HTTP headers as key-value pairs. "
                "Use $CRED{name} for secrets, e.g. "
                "{\"Authorization\": \"Bearer $CRED{github_token}\"}"
            ),
            "default": {},
        },
        "body": {
            "type": "string",
            "description": "Request body (for POST/PUT/PATCH). Supports $CRED{name} handles.",
            "default": "",
        },
        "timeout": {
            "type": "integer",
            "description": "Timeout in seconds (default 30)",
            "default": 30,
        },
    },
)
async def http_request(
    url: str,
    method: str = "GET",
    headers: dict | None = None,
    body: str = "",
    timeout: int = 30,
    *,
    mesh_client=None,
) -> dict:
    """Make an HTTP request and return status, headers, and body."""
    # Collect all resolved secret values for redaction
    all_secrets: list[str] = []

    try:
        # Resolve $CRED{name} handles in url, headers, and body
        resolved_url, url_secrets = await _resolve_creds(url, mesh_client)
        all_secrets.extend(url_secrets)

        # SSRF protection: block requests to private/internal networks.
        # Run in executor because socket.getaddrinfo() blocks the event loop.
        loop = asyncio.get_running_loop()
        if await loop.run_in_executor(None, _is_private_url, resolved_url):
            return {"error": "SSRF protection: requests to private/internal addresses are blocked", "status_code": 0}

        resolved_headers = {}
        for k, v in (headers or {}).items():
            resolved_v, hdr_secrets = await _resolve_creds(str(v), mesh_client)
            resolved_headers[k] = resolved_v
            all_secrets.extend(hdr_secrets)

        resolved_body = ""
        if body:
            resolved_body, body_secrets = await _resolve_creds(body, mesh_client)
            all_secrets.extend(body_secrets)

        client = await _get_client()
        response = await client.request(
            method=method.upper(),
            url=resolved_url,
            headers=resolved_headers,
            content=resolved_body if resolved_body else None,
            timeout=timeout,
        )
        resp_body = response.text[:_MAX_BODY]
        truncated = len(response.text) > _MAX_BODY

        # Redact any credential values that appear in the response
        if all_secrets:
            resp_body = _redact(resp_body, all_secrets)
            resp_headers = {
                k: _redact(v, all_secrets) for k, v in response.headers.items()
            }
        else:
            resp_headers = dict(response.headers)

        return {
            "status_code": response.status_code,
            "headers": resp_headers,
            "body": resp_body,
            "truncated": truncated,
        }
    except ValueError as e:
        # Credential resolution errors — redact in case earlier creds already resolved
        error_msg = _redact(str(e), all_secrets) if all_secrets else str(e)
        return {"error": error_msg, "status_code": 0}
    except httpx.TimeoutException:
        return {"error": f"Request timed out after {timeout}s", "status_code": 0}
    except Exception as e:
        error_msg = _redact(str(e), all_secrets) if all_secrets else str(e)
        return {"error": error_msg, "status_code": 0}
