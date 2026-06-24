"""Operator permission ceiling — single source of truth, shared across zones.

Lives in ``src/shared`` (shipped to BOTH the mesh host and the agent container)
because the operator tool in ``src/agent/builtins/operator_tools.py`` needs it
client-side, and the agent container ships only ``src/agent`` + ``src/shared``
(not ``src/host``). ``src/host/permissions`` re-exports these symbols so existing
host-side imports keep resolving.
"""

from __future__ import annotations

# Permission ceiling for operator-initiated agent edits. The operator
# (an LLM-driven agent) is NOT allowed to grant the one remaining escalation
# to the agents it manages — ``can_use_wallet`` requires explicit human setup
# (spending money) — and blackboard read/write patterns are bounded.
# ``can_spawn`` is a normal default-on capability now (ephemeral fleet-spawn,
# bounded one level deep — spawned agents can't re-spawn), so the operator may
# manage it like ``can_use_browser``.
#
# This is the SINGLE SOURCE OF TRUTH. It is enforced in two places:
#   1. Client-side in the operator tool (`operator_tools._validate_edit`)
#      for a fast, descriptive error to the operator LLM.
#   2. Server-side on the mesh ``/edit-soft`` endpoint, so a fooled or
#      injected operator LLM cannot route a raw permissions edit around
#      its own client-side guard (finding H1, May 2026 remediation — now
#      narrowed to the wallet, the only remaining operator-ungrantable flag).
#
# DELIBERATELY NOT enforced on the dashboard ``PUT /api/agents/{id}/
# permissions`` endpoint — that is the human operator's "advanced
# permissions" escalation path, and the ceiling is intentionally human-
# overridable there.
_OPERATOR_PERMISSION_CEILING = {
    "can_use_browser": True,
    "can_spawn": True,        # Ephemeral fleet-spawn is a default capability
    "can_manage_cron": True,
    "can_use_wallet": False,  # Requires explicit user setup (spends money)
    "blackboard_read": ["*"],
    "blackboard_write": ["tasks/*", "context/*", "status/*", "output/*", "artifacts/*"],
}


def clamp_to_operator_ceiling(field: str, new_value) -> str | None:
    """Return an error string if a permissions edit exceeds the operator ceiling.

    Single source of truth for the operator permission ceiling, shared by
    the operator tool's client-side ``_validate_edit`` and the mesh
    ``/edit-soft`` endpoint's server-side re-check (finding H1).

    Returns ``None`` when the edit is within the ceiling (or is not a
    permissions edit / not a dict — those are handled by other validators).
    """
    if field != "permissions" or not isinstance(new_value, dict):
        return None
    for key, max_val in _OPERATOR_PERMISSION_CEILING.items():
        if key not in new_value:
            continue
        if isinstance(max_val, bool):
            if new_value[key] and not max_val:
                return (
                    f"Permission ceiling exceeded: '{key}' cannot be set "
                    "to True by the operator. Use the dashboard for "
                    "advanced permissions."
                )
        elif isinstance(max_val, list):
            requested = set(new_value.get(key, []))
            allowed = set(max_val)
            if "*" not in allowed and not requested.issubset(allowed):
                excess = requested - allowed
                return (
                    f"Permission ceiling exceeded: '{key}' patterns "
                    f"{excess} exceed allowed {allowed}. Use the "
                    "dashboard for advanced permissions."
                )
    return None
