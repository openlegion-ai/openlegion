"""Credential-blind vault tools for agents.

Agents can generate, capture, and list credentials without ever seeing
the actual secret values. Values are stored in the mesh vault and
referenced via opaque ``$CRED{name}`` handles.
"""

from __future__ import annotations

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
