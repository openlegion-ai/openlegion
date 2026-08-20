"""Operator permission ceiling — single source of truth, shared across zones.

Lives in ``src/shared`` (shipped to BOTH the mesh host and the agent container)
because the operator tool in ``src/agent/builtins/operator_tools.py`` needs it
client-side, and the agent container ships only ``src/agent`` + ``src/shared``
(not ``src/host``). ``src/host/permissions`` re-exports these symbols so existing
host-side imports keep resolving.

Enforced in two places:
  1. Client-side in the operator tool (``operator_tools._validate_edit``) for a
     fast, descriptive error to the operator LLM.
  2. Server-side on the mesh ``/edit-soft`` endpoint, so a fooled or injected
     operator LLM cannot route a raw permissions edit around its own
     client-side guard (finding H1, May 2026 remediation).

DELIBERATELY NOT enforced on the dashboard ``PUT /api/agents/{id}/permissions``
endpoint — that is the HUMAN operator's "advanced permissions" escalation path,
and the ceiling is intentionally human-overridable there.

Deny by default
---------------
This table used to be an ALLOW-list that was iterated instead of the submitted
payload::

    for key, max_val in _OPERATOR_PERMISSION_CEILING.items():
        if key not in new_value:
            continue

Any field absent from the table was therefore never examined at all. It listed
6 of ``AgentPermissions``' 26 fields, so the operator LLM could set the other 19
to anything — including ``allowed_credentials``, the control-plane flags
(``can_manage_fleet`` / ``can_manage_teams`` / ``can_edit_agent_config``), and
every wallet spending limit. The ``can_use_wallet`` flag was blocked while
``wallet_spend_limit_daily_usd`` was not, so on an agent a human had already
enabled the wallet for, the operator could raise the daily cap without limit —
defeating the one protection the table named as its reason for existing.

The submitted payload is now what gets iterated, and every key must be
classified. Unknown keys are refused, so a field added to ``AgentPermissions``
without a decision here fails CLOSED rather than silently becoming grantable.
``tests/test_operator_ceiling.py`` pins that: the two tables below must together
cover ``AgentPermissions.model_fields`` exactly.

Not applied to undo
-------------------
The undo path calls ``_apply_pending_change`` directly rather than going
through ``/mesh/agents/{agent_id}/edit-soft``, so it never reaches this
function — deliberately. Undo REPLACES the agent's permissions with the full
pre-edit dict, which would otherwise trip the deny-by-default check on any
forbidden field the agent legitimately had. That is safe because the stored
``old_value`` is read from the server's own state at edit time, never supplied
by the caller: undo can only restore prior state, never escalate past it.

Revoke, never grant
-------------------
Forbidden fields still accept a DE-ESCALATING value — ``False`` for a flag, an
empty list, ``0`` for a numeric cap. Taking a dangerous power away is always
safe and is a genuinely useful thing for the operator to be able to do; only
handing one out requires a human. Numeric caps admit only ``0`` because this
function sees the proposed value and not the current one, so it cannot tell a
raise from a lower.
"""

from __future__ import annotations

from typing import Any

# Sentinel: the operator may set any value for this field. Used for fields
# where the SHAPE is the only constraint and the mesh validates it elsewhere.
UNRESTRICTED = object()

# Fields the operator may manage, with the ceiling on each.
#   * bool         → the maximum grantable value (True = may enable)
#   * list         → the allowed pattern set ("*" = any)
#   * UNRESTRICTED → any value
#
# Kept under the historical name because ``src/host/permissions`` and
# ``src/agent/builtins/operator_tools`` re-export it for back-compat.
_OPERATOR_PERMISSION_CEILING: dict[str, Any] = {
    # Capabilities that are normal, default-on agent management. Each is
    # already surfaced as a toggle in the operator/dashboard UI.
    "can_use_browser": True,
    "can_use_internet": True,
    # Ephemeral fleet-spawn, bounded one level deep (a spawned agent cannot
    # re-spawn), so this is not an escalation ladder.
    "can_spawn": True,
    "can_manage_cron": True,
    # Read-only view of fleet metrics.
    "can_view_fleet_metrics": True,
    # Blackboard is signals-only; ``output/*`` + ``artifacts/*`` moved to the
    # Team Drive in Phase-2 unit 4.
    "blackboard_read": ["*"],
    "blackboard_write": ["tasks/*", "context/*", "status/*"],
    # Subordinate to ``can_use_browser``, which is itself grantable above —
    # narrowing or widening the action list within a capability the operator
    # can already grant outright is not a further escalation.
    "browser_actions": UNRESTRICTED,
    # A skill is DATA, not a capability: this only controls which packs the
    # agent can discover, to keep context lean.
    "allowed_skills": UNRESTRICTED,
    # Mesh topology — who an agent may talk to and what it may publish or
    # subscribe to. This is the operator's core coordination remit, and these
    # were unrestricted before this table became deny-by-default.
    "can_message": UNRESTRICTED,
    "can_publish": UNRESTRICTED,
    "can_subscribe": UNRESTRICTED,
}

# Fields the operator may never GRANT, each with the reason surfaced to the
# LLM. Values are the human-facing explanation.
_OPERATOR_FORBIDDEN: dict[str, str] = {
    # Identity, not a permission.
    "agent_id": "an agent's identity cannot be reassigned by the operator",
    # Money.
    "can_use_wallet": "spending money requires explicit human setup",
    "wallet_spend_limit_per_tx_usd": "wallet spending limits are a human decision",
    "wallet_spend_limit_daily_usd": "wallet spending limits are a human decision",
    "wallet_rate_limit_per_hour": "wallet rate limits are a human decision",
    "wallet_allowed_chains": "wallet chain allowlists are a human decision",
    "wallet_allowed_contracts": "wallet contract allowlists are a human decision",
    # Secrets and external reach.
    "allowed_credentials": "credential access requires explicit human setup",
    "allowed_apis": "external API allowlists require explicit human setup",
    "can_request_user_credentials": (
        "asking the human for credentials requires explicit human setup"
    ),
    # Durable control-plane powers. ``can_edit_agent_config`` is the power the
    # operator is exercising right now, so granting it would let the operator
    # propagate its own privilege to an agent it manages.
    "can_manage_fleet": "durable fleet management requires explicit human setup",
    "can_manage_teams": "team management requires explicit human setup",
    "can_edit_agent_config": (
        "config-edit rights would propagate the operator's own privilege"
    ),
    "can_route_tasks": "task-routing rights require explicit human setup",
}


def _is_de_escalation(value: Any) -> bool:
    """True when ``value`` can only ever REDUCE a permission.

    ``False`` revokes a flag, an empty collection revokes an allowlist, and
    ``0`` zeroes a numeric cap. Any other value could be a grant, and this
    function cannot compare against the current value to tell.
    """
    if isinstance(value, bool):
        return value is False
    if isinstance(value, (int, float)):
        return value == 0
    if isinstance(value, (list, tuple, set, dict, str)):
        return len(value) == 0
    return value is None


def clamp_to_operator_ceiling(field: str, new_value) -> str | None:
    """Return an error string if a permissions edit exceeds the operator ceiling.

    Returns ``None`` when the edit is within the ceiling (or is not a
    permissions edit / not a dict — those are handled by other validators).

    Iterates the SUBMITTED payload, not the ceiling table, so an unclassified
    key is refused rather than ignored.
    """
    if field != "permissions" or not isinstance(new_value, dict):
        return None

    for key, value in new_value.items():
        if key in _OPERATOR_FORBIDDEN:
            # Revoking a dangerous power is always allowed.
            if _is_de_escalation(value):
                continue
            return (
                f"Permission ceiling exceeded: '{key}' cannot be granted by "
                f"the operator — {_OPERATOR_FORBIDDEN[key]}. Use the "
                "dashboard for advanced permissions."
            )

        if key not in _OPERATOR_PERMISSION_CEILING:
            # Fail CLOSED: an unknown key is either a typo or a field nobody
            # has classified yet. Either way the operator does not get it.
            return (
                f"Permission ceiling exceeded: '{key}' is not an "
                "operator-manageable permission. Use the dashboard for "
                "advanced permissions."
            )

        max_val = _OPERATOR_PERMISSION_CEILING[key]
        if max_val is UNRESTRICTED:
            continue
        if isinstance(max_val, bool):
            if value and not max_val:
                return (
                    f"Permission ceiling exceeded: '{key}' cannot be set "
                    "to True by the operator. Use the dashboard for "
                    "advanced permissions."
                )
        elif isinstance(max_val, list):
            requested = set(value or [])
            allowed = set(max_val)
            if "*" not in allowed and not requested.issubset(allowed):
                excess = requested - allowed
                return (
                    f"Permission ceiling exceeded: '{key}' patterns "
                    f"{excess} exceed allowed {allowed}. Use the "
                    "dashboard for advanced permissions."
                )
    return None
