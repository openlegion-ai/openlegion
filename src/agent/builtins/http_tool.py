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
        _client = httpx.AsyncClient(
            follow_redirects=True,
            max_redirects=5,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
    return _client


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
