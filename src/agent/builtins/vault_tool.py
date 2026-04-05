"""Credential-blind vault tools for agents.

Agents can generate, capture, and list credentials without ever seeing
the actual secret values. Values are stored in the mesh vault and
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


_CRED_NAME_RE = re.compile(r'^[a-zA-Z0-9_.\-]{1,128}$')


@skill(
    name="request_credential",
    description=(
        "Ask the user to provide a credential (API key, password, token) "
        "through a secure input in their chat. The credential is stored "
        "directly in the vault — you never see the actual value. After "
        "the user saves it, use the handle $CRED{name} in HTTP requests "
        "or browser logins."
    ),
    parameters={
        "name": {
            "type": "string",
            "description": "Credential name (e.g. 'linkedin_api_key', 'twitter_password')",
        },
        "description": {
            "type": "string",
            "description": "Tell the user what this credential is for and where to find it",
        },
        "service": {
            "type": "string",
            "description": "Service name (e.g. 'LinkedIn', 'Twitter', 'Stripe')",
            "default": "",
        },
    },
)
async def request_credential(
    name: str, description: str, service: str = "",
    *, mesh_client=None, **_kw,
) -> dict:
    """Request a credential from the user via secure chat input."""
    if not mesh_client:
        return {"error": "Vault tools require mesh connectivity"}

    if not name:
        return {"error": "name is required"}

    if not _CRED_NAME_RE.match(name):
        return {
            "error": (
                f"Invalid credential name: {name}. "
                "Use alphanumeric, underscore, dot, or hyphen (1-128 chars)."
            ),
        }

    # Check if credential already exists
    try:
        existing = await mesh_client.vault_list()
        if name in existing:
            return {
                "already_exists": True,
                "handle": f"$CRED{{{name}}}",
                "message": f"Credential '{name}' already exists in the vault.",
            }
    except Exception:
        pass  # Vault may not be available yet; proceed with the request

    # Emit credential request event to the dashboard via the mesh
    try:
        await mesh_client.request_credential_from_user(
            name=name, description=description, service=service or name,
        )
    except Exception:
        pass  # Best effort — the tool result itself is the primary mechanism

    return {
        "requested": True,
        "name": name,
        "handle": f"$CRED{{{name}}}",
        "message": (
            f"Credential request sent to user. "
            f"Once they save it, use $CRED{{{name}}} in your requests."
        ),
    }
