"""Credential-blind vault tools for agents.

Agents can generate, capture, list, and check credentials without ever
seeing the actual secret values. Values are stored in the mesh vault and
referenced via opaque ``$CRED{name}`` handles.
"""

from __future__ import annotations

import re
import secrets
import string

from src.agent.skills import skill
from src.shared.utils import setup_logging

logger = setup_logging("agent.vault")

_CHARSETS = {
    "urlsafe": None,  # uses secrets.token_urlsafe
    "hex": string.hexdigits[:16],
    "alphanumeric": string.ascii_letters + string.digits,
}


@skill(
    name="vault_generate_secret",
    description=(
        "Generate a cryptographically random secret and store it in the vault. "
        "Returns an opaque $CRED{name} handle — the actual value is NEVER returned. "
        "Use this to create API keys, passwords, or tokens for service accounts."
    ),
    parameters={
        "name": {"type": "string", "description": "Credential name (e.g. 'myservice_api_key')"},
        "length": {
            "type": "integer",
            "description": "Length of the secret (default 32)",
            "default": 32,
        },
        "charset": {
            "type": "string",
            "description": "Character set: 'urlsafe' (default), 'hex', or 'alphanumeric'",
            "default": "urlsafe",
        },
    },
)
async def vault_generate_secret(
    name: str,
    length: int = 32,
    charset: str = "urlsafe",
    *,
    mesh_client=None,
) -> dict:
    """Generate a random secret, store it, return only the handle."""
    if not mesh_client:
        return {"error": "Vault tools require mesh connectivity"}
    if not name:
        return {"error": "name is required"}

    if charset == "urlsafe":
        value = secrets.token_urlsafe(length)[:length]
    elif charset in _CHARSETS and _CHARSETS[charset]:
        value = "".join(secrets.choice(_CHARSETS[charset]) for _ in range(length))
    else:
        return {"error": f"Unknown charset: {charset}. Use: urlsafe, hex, alphanumeric"}

    try:
        result = await mesh_client.vault_store(name, value)
        return {"stored": True, "handle": result.get("handle", f"$CRED{{{name}}}")}
    except Exception as e:
        return {"error": f"Failed to store credential: {e}"}


@skill(
    name="vault_capture_from_page",
    description=(
        "Read text from a browser element and store it as a credential in the vault. "
        "Returns an opaque $CRED{name} handle — the captured value is NEVER returned. "
        "Use after signing up for a service to capture displayed API keys."
    ),
    parameters={
        "name": {"type": "string", "description": "Credential name to store under"},
        "selector": {
            "type": "string",
            "description": "CSS selector of the element containing the secret (optional if ref given)",
            "default": "",
        },
        "ref": {
            "type": "string",
            "description": "Element ref from browser_snapshot (e.g. 'e3')",
            "default": "",
        },
    },
)
async def vault_capture_from_page(
    name: str,
    selector: str = "",
    ref: str = "",
    *,
    mesh_client=None,
) -> dict:
    """Capture text from a browser element and store as credential."""
    if not mesh_client:
        return {"error": "Vault tools require mesh connectivity"}
    if not name:
        return {"error": "name is required"}
    if not selector and not ref:
        return {"error": "Provide either 'ref' (from browser_snapshot) or 'selector' (CSS)"}

    try:
        from src.agent.builtins.browser_tool import _get_page, _page_refs

        page = await _get_page()
        if ref:
            locator = _page_refs.get(ref)
            if not locator:
                return {"error": f"Unknown ref '{ref}'. Call browser_snapshot first."}
            value = await locator.inner_text(timeout=10000)
        else:
            value = await page.inner_text(selector, timeout=10000)

        value = value.strip()
        if not value:
            return {"error": "Element is empty — no text to capture"}

        result = await mesh_client.vault_store(name, value)
        return {"captured": True, "handle": result.get("handle", f"$CRED{{{name}}}")}
    except Exception as e:
        return {"error": f"Failed to capture credential: {e}"}


@skill(
    name="vault_list",
    description="List all credential names stored in the vault (names only, never values).",
    parameters={},
)
async def vault_list(*, mesh_client=None) -> dict:
    """List credential names from the vault."""
    if not mesh_client:
        return {"error": "Vault tools require mesh connectivity"}
    try:
        names = await mesh_client.vault_list()
        return {"credentials": names, "count": len(names)}
    except Exception as e:
        return {"error": f"Failed to list credentials: {e}"}


@skill(
    name="vault_status",
    description="Check if a credential exists in the vault by name.",
    parameters={
        "name": {"type": "string", "description": "Credential name to check"},
    },
)
async def vault_status(name: str, *, mesh_client=None) -> dict:
    """Check if a credential exists."""
    if not mesh_client:
        return {"error": "Vault tools require mesh connectivity"}
    if not name:
        return {"error": "name is required"}
    try:
        result = await mesh_client.vault_status(name)
        return {"name": name, "exists": result.get("exists", False)}
    except Exception as e:
        return {"error": f"Failed to check credential: {e}"}
