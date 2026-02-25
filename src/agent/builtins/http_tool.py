"""HTTP request tool for direct internet access from agent containers.

Supports ``$CRED{name}`` handles in URLs, headers, and body — resolved
via the mesh vault at execution time so the agent never sees raw secrets.
"""

from __future__ import annotations

import re

import httpx

from src.agent.skills import skill

_MAX_BODY = 50_000

_CRED_HANDLE_RE = re.compile(r"\$CRED\{([^}]+)\}")

# Shared client for connection pooling across tool invocations.
# Avoids repeated TCP handshakes and TLS negotiation.
_client: httpx.AsyncClient | None = None


async def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(follow_redirects=True, max_redirects=5)
    return _client


async def _resolve_creds(text: str, mesh_client) -> str:
    """Replace all $CRED{name} handles in *text* with resolved values."""
    matches = _CRED_HANDLE_RE.findall(text)
    if not matches:
        return text
    if not mesh_client:
        raise ValueError("$CRED{} handles require mesh connectivity")
    for cred_name in matches:
        value = await mesh_client.vault_resolve(cred_name)
        if value is None:
            raise ValueError(f"Credential not found: {cred_name}")
        text = text.replace(f"$CRED{{{cred_name}}}", value)
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
                "Use $CRED{name} for secrets, e.g. {\"Authorization\": \"Bearer $CRED{github_token}\"}"
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
    try:
        # Resolve $CRED{name} handles in url, headers, and body
        resolved_url = await _resolve_creds(url, mesh_client)
        resolved_headers = {}
        for k, v in (headers or {}).items():
            resolved_headers[k] = await _resolve_creds(str(v), mesh_client)
        resolved_body = await _resolve_creds(body, mesh_client) if body else ""

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
        return {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "body": resp_body,
            "truncated": truncated,
        }
    except ValueError as e:
        # Credential resolution errors
        return {"error": str(e), "status_code": 0}
    except httpx.TimeoutException:
        return {"error": f"Request timed out after {timeout}s", "status_code": 0}
    except Exception as e:
        return {"error": str(e), "status_code": 0}
