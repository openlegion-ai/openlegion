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
        "or browser logins. For services needing multiple credentials "
        "(e.g. username + password), use the 'fields' parameter to request "
        "them all in a single card."
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
        "fields": {
            "type": "array",
            "description": (
                "Request multiple credentials in one card. Each field has "
                "'name' and 'description'. Example: "
                "[{\"name\": \"linkedin_username\", \"description\": \"Your email\"}, "
                "{\"name\": \"linkedin_password\", \"description\": \"Your password\"}]. "
                "When provided, 'name' and 'description' are ignored."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["name", "description"],
            },
            "default": [],
        },
    },
)
async def request_credential(
    name: str = "", description: str = "", service: str = "",
    fields: list | None = None,
    *, mesh_client=None, **_kw,
) -> dict:
    """Request a credential from the user via secure chat input."""
    if not mesh_client:
        return {"error": "Vault tools require mesh connectivity"}

    # Normalize: if fields provided, use those; otherwise wrap name/description
    if fields:
        resolved_fields = []
        for f in fields:
            fn = f.get("name", "")
            fd = f.get("description", "")
            if not fn:
                return {"error": "Each field must have a 'name'"}
            if not _CRED_NAME_RE.match(fn):
                return {
                    "error": (
                        f"Invalid credential name: {fn}. "
                        "Use alphanumeric, underscore, dot, or hyphen (1-128 chars)."
                    ),
                }
            resolved_fields.append({"name": fn, "description": fd})
    else:
        if not name:
            return {"error": "name is required (or provide 'fields')"}
        if not _CRED_NAME_RE.match(name):
            return {
                "error": (
                    f"Invalid credential name: {name}. "
                    "Use alphanumeric, underscore, dot, or hyphen (1-128 chars)."
                ),
            }
        resolved_fields = [{"name": name, "description": description}]

    # Check if credentials already exist
    handles = {}
    already_exist = []
    try:
        existing = await mesh_client.vault_list()
        for f in resolved_fields:
            if f["name"] in existing:
                already_exist.append(f["name"])
            handles[f["name"]] = f"$CRED{{{f['name']}}}"
    except Exception as exc:
        logger.warning("vault_list check failed, proceeding with request: %s", exc)
        for f in resolved_fields:
            handles[f["name"]] = f"$CRED{{{f['name']}}}"

    if len(already_exist) == len(resolved_fields):
        return {
            "already_exists": True,
            "handles": handles,
            "message": "All requested credentials already exist in the vault.",
        }

    # Filter to only request credentials that don't already exist
    fields_to_request = [f for f in resolved_fields if f["name"] not in already_exist]

    # Emit credential request event to the dashboard via the mesh
    service_name = service or name or (fields_to_request[0]["name"] if fields_to_request else "")
    try:
        await mesh_client.request_credential_from_user(
            name=service_name,
            description=description,
            service=service_name,
            fields=fields_to_request,
        )
    except Exception:
        pass  # Best effort — the tool result itself is the primary mechanism

    return {
        "requested": True,
        "handles": handles,
        "message": (
            f"Credential request sent to user. "
            f"You will be notified automatically when the user saves the credentials. "
            f"Do NOT poll vault_list — just continue with other work or wait. "
            f"Once notified, use these handles: "
            + ", ".join(f"$CRED{{{f['name']}}}" for f in resolved_fields)
        ),
    }
