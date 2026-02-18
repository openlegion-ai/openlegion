"""HTTP request tool for direct internet access from agent containers.

No mesh proxy needed -- the container has direct network access.
"""

from __future__ import annotations

import httpx

from src.agent.skills import skill

_MAX_BODY = 50_000


@skill(
    name="http_request",
    description=(
        "Make an HTTP request. Use this to call APIs, download web pages, "
        "or interact with any HTTP service."
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
            "description": "HTTP headers as key-value pairs",
            "default": {},
        },
        "body": {
            "type": "string",
            "description": "Request body (for POST/PUT/PATCH)",
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
) -> dict:
    """Make an HTTP request and return status, headers, and body."""
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.request(
                method=method.upper(),
                url=url,
                headers=headers or {},
                content=body if body else None,
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
    except httpx.TimeoutException:
        return {"error": f"Request timed out after {timeout}s", "status_code": 0}
    except Exception as e:
        return {"error": str(e), "status_code": 0}
