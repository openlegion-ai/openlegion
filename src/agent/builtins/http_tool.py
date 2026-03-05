"""HTTP request tool for direct internet access from agent containers.

Supports ``$CRED{name}`` handles in URLs, headers, and body — resolved
via the mesh vault at execution time so the agent never sees raw secrets.

DNS rebinding protection: DNS is resolved once and the resolved IP is
pinned for the actual connection.  Redirects are followed manually with
DNS re-validation at each hop.
"""

from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
from urllib.parse import urlparse, urlunparse

import httpx

from src.agent.skills import skill

_MAX_BODY = 50_000
_MAX_REDIRECTS = 5

_CRED_HANDLE_RE = re.compile(r"\$CRED\{([^}]+)\}")

# Shared client for connection pooling across tool invocations.
# Redirects are followed manually (follow_redirects=False) so we can
# re-validate DNS at each hop to prevent DNS rebinding attacks.
_client: httpx.AsyncClient | None = None
_client_lock = asyncio.Lock()


async def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is not None and not _client.is_closed:
        return _client
    async with _client_lock:
        if _client is None or _client.is_closed:
            _client = httpx.AsyncClient(
                follow_redirects=False,
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
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
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_unspecified:
        return True
    # IPv4-mapped IPv6 (e.g. ::ffff:127.0.0.1) — check the mapped v4 address too
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        mapped = ip.ipv4_mapped
        if (mapped.is_private or mapped.is_loopback or mapped.is_link_local
                or mapped.is_reserved or mapped.is_unspecified):
            return True
    return False


def _resolve_and_pin(url: str) -> tuple[str, str, str]:
    """Resolve DNS for *url*, validate all IPs, return pinned connection info.

    Returns ``(pinned_url, original_hostname, resolved_ip)`` where
    *pinned_url* has the hostname replaced with the resolved IP and
    *original_hostname* is preserved for the Host header / SNI.

    Raises ``ValueError`` if any resolved IP is blocked or DNS fails.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("SSRF protection: only http and https schemes are allowed")
    hostname = parsed.hostname
    port = parsed.port
    if not hostname:
        raise ValueError("SSRF protection: malformed URL")

    # If hostname is already an IP literal, validate directly
    try:
        ip = ipaddress.ip_address(hostname)
        if _is_blocked_ip(ip):
            raise ValueError("SSRF protection: requests to private/internal addresses are blocked")
        return url, hostname, str(ip)
    except ValueError as e:
        if "SSRF" in str(e):
            raise
        # Not an IP literal — resolve hostname

    try:
        results = socket.getaddrinfo(hostname, port or 443, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except (socket.gaierror, OSError):
        raise ValueError("SSRF protection: DNS resolution failed (fail-closed)")

    if not results:
        raise ValueError("SSRF protection: DNS resolution returned no results")

    # Validate ALL resolved IPs — block if any is private
    resolved_ips = []
    for family, _, _, _, sockaddr in results:
        ip = ipaddress.ip_address(sockaddr[0])
        if _is_blocked_ip(ip):
            raise ValueError("SSRF protection: requests to private/internal addresses are blocked")
        resolved_ips.append(str(ip))

    # Use the first resolved IP for pinning
    pinned_ip = resolved_ips[0]

    # Build pinned URL: replace hostname with resolved IP
    # For IPv6, wrap in brackets
    if ":" in pinned_ip:
        netloc_ip = f"[{pinned_ip}]"
    else:
        netloc_ip = pinned_ip

    # Reconstruct netloc with port if present
    if port:
        pinned_netloc = f"{netloc_ip}:{port}"
    else:
        pinned_netloc = netloc_ip

    pinned_url = urlunparse((
        parsed.scheme, pinned_netloc, parsed.path,
        parsed.params, parsed.query, parsed.fragment,
    ))

    return pinned_url, hostname, pinned_ip


def _build_pinned_headers(
    headers: dict[str, str],
    original_hostname: str,
    parsed_url,
) -> dict[str, str]:
    """Add Host header and return updated headers for a pinned request."""
    out = dict(headers)
    # Set Host header to original hostname (required when URL has IP)
    port = parsed_url.port
    if port and port not in (80, 443):
        out["Host"] = f"{original_hostname}:{port}"
    else:
        out["Host"] = original_hostname
    return out


async def _send_pinned_request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    headers: dict[str, str],
    content: str | None,
    timeout: int,
) -> httpx.Response:
    """Resolve DNS, pin IP, and send a single request (no redirect following).

    For HTTPS, sets SNI to the original hostname so TLS validation works
    even though the URL contains the resolved IP.
    """
    loop = asyncio.get_running_loop()
    pinned_url, original_hostname, resolved_ip = await loop.run_in_executor(
        None, _resolve_and_pin, url,
    )

    parsed = urlparse(url)
    pinned_headers = _build_pinned_headers(headers, original_hostname, parsed)

    # Build request manually to set extensions for SNI and timeout
    request = client.build_request(
        method=method,
        url=pinned_url,
        headers=pinned_headers,
        content=content,
        timeout=timeout,
    )

    # Set SNI hostname for HTTPS so TLS validates against the original
    # hostname, not the IP address we pinned to.
    if parsed.scheme == "https":
        request.extensions["sni_hostname"] = original_hostname.encode("ascii")

    return await client.send(request, follow_redirects=False, stream=False)


async def _request_with_pinned_dns(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    headers: dict[str, str],
    content: str | None,
    timeout: int,
) -> httpx.Response:
    """Send request with DNS pinning and safe manual redirect following.

    Each redirect hop re-resolves DNS and re-validates the target IP,
    preventing DNS rebinding attacks where an attacker's DNS returns a
    public IP for the initial check but a private IP for the connection.
    """
    original_parsed = urlparse(url)
    original_origin = (original_parsed.scheme, original_parsed.hostname, original_parsed.port)

    response = await _send_pinned_request(client, method, url, headers, content, timeout)

    for _ in range(_MAX_REDIRECTS):
        if response.status_code not in (301, 302, 303, 307, 308):
            return response

        location = response.headers.get("location")
        if not location:
            return response

        # Resolve relative redirects against the current URL
        redirect_url = str(httpx.URL(url).join(location))

        # For 303, always GET; for 301/302, GET (per browser behavior);
        # for 307/308, preserve method
        if response.status_code in (301, 302, 303):
            method = "GET"
            content = None

        # Strip Authorization header on cross-origin redirects
        redirect_parsed = urlparse(redirect_url)
        redirect_origin = (redirect_parsed.scheme, redirect_parsed.hostname, redirect_parsed.port)
        if redirect_origin != original_origin:
            headers = {k: v for k, v in headers.items() if k.lower() != "authorization"}

        url = redirect_url
        # Re-resolve and re-validate DNS for the redirect target
        response = await _send_pinned_request(client, method, url, headers, content, timeout)

    raise httpx.TooManyRedirects(
        f"SSRF protection: exceeded {_MAX_REDIRECTS} redirects",
        request=response.request,
    )


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
    resolved: dict[str, str] = {}
    for cred_name in set(matches):  # dedupe to avoid redundant vault calls
        value = await mesh_client.vault_resolve(cred_name)
        if value is None:
            raise ValueError(f"Credential not found: {cred_name}")
        resolved[cred_name] = value
        secrets.append(value)
    # Single-pass replacement prevents double-resolution attacks where a
    # resolved value itself contains $CRED{...} patterns.
    text = _CRED_HANDLE_RE.sub(
        lambda m: resolved.get(m.group(1), m.group(0)), text,
    )
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
        response = await _request_with_pinned_dns(
            client,
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
        # Credential resolution errors + SSRF blocks
        error_msg = _redact(str(e), all_secrets) if all_secrets else str(e)
        return {"error": error_msg, "status_code": 0}
    except httpx.TooManyRedirects as e:
        error_msg = str(e)
        return {"error": error_msg, "status_code": 0}
    except httpx.TimeoutException:
        return {"error": f"Request timed out after {timeout}s", "status_code": 0}
    except Exception as e:
        error_msg = _redact(str(e), all_secrets) if all_secrets else str(e)
        return {"error": error_msg, "status_code": 0}
