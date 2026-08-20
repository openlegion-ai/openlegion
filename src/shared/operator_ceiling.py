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

Not applied to undo — and that is a KNOWN GAP, not a proof of safety
-------------------------------------------------------------------
The undo path calls ``_apply_pending_change`` directly rather than going
through ``/mesh/agents/{agent_id}/edit-soft``, so it never reaches this
function. Undo REPLACES the agent's permissions with the full pre-edit dict,
which would otherwise trip the deny-by-default check on any forbidden field the
agent legitimately had.

An earlier version of this note claimed that was safe because ``old_value`` is
server-derived. That argument is WRONG, and the hole it misses is worth
spelling out so nobody re-derives it. Server-derived proves the snapshot is
authentic, not that restoring it is monotonic:

  1. an agent holds human-granted wallet or credential authority;
  2. the operator makes an unrelated, permitted edit — the receipt captures the
     agent's ENTIRE permissions record, including that authority;
  3. a human revokes the authority via the dashboard;
  4. the operator redeems the still-live undo receipt;
  5. the full historical record is restored, re-granting what the human took
     away.

Closing it means making undo diff-aware (restore only the keys the original
edit touched) or ceiling-checking the delta, and the snapshot is also taken
outside the write lock, so it races other permission writers. Both belong with
the undo machinery in ``src/host/server.py``, not here. Tracked as a follow-up;
this module deliberately does not pretend to cover it.

Revoke, never grant
-------------------
Most forbidden fields still accept a DE-ESCALATING value — ``False`` for a
flag, an empty allowlist. Taking a dangerous power away is always safe and is a
genuinely useful thing for the operator to be able to do; only handing one out
requires a human.

That is decided PER FIELD, not by the shape of the value, because "empty" and
"zero" do not mean "less" everywhere in this codebase:

* ``wallet_allowed_contracts`` — ``PermissionMatrix.can_access_wallet_contract``
  reads ``if not contracts: return True  # Empty = allow all``. An empty list
  REMOVES the restriction.
* ``wallet_spend_limit_*`` / ``wallet_rate_limit_per_hour`` —
  ``get_wallet_limits`` documents ``0 = use global default``. Zeroing an agent
  whose per-agent cap is TIGHTER than the global default RAISES it.

Those fields therefore accept no value at all from the operator. The ones that
do accept a revoke were each checked against their enforcement site:
``can_use_api`` is ``service in perms.allowed_apis``, ``can_manage_vault`` is
``bool(perms.allowed_credentials)``, and ``can_use_wallet_chain`` requires an
explicit membership — for all three, empty genuinely means "nothing".
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

# Fields the operator may never GRANT, mapping to
# ``(reason, revoke_allowed)``. ``revoke_allowed`` is False where NO value is
# unambiguously de-escalating — see the module docstring for the two wallet
# cases where "empty" and "zero" mean MORE permission, not less.
_OPERATOR_FORBIDDEN: dict[str, tuple[str, bool]] = {
    # Identity, not a permission — no value is meaningful.
    "agent_id": ("an agent's identity cannot be reassigned by the operator", False),
    # Money.
    "can_use_wallet": ("spending money requires explicit human setup", True),
    # ``0`` means "use the global default" (get_wallet_limits), so zeroing a
    # tighter-than-global per-agent cap RAISES it. No safe revoke value.
    "wallet_spend_limit_per_tx_usd": (
        "wallet spending limits are a human decision", False,
    ),
    "wallet_spend_limit_daily_usd": (
        "wallet spending limits are a human decision", False,
    ),
    "wallet_rate_limit_per_hour": (
        "wallet rate limits are a human decision", False,
    ),
    # Membership-tested (can_use_wallet_chain), so empty really is "none".
    "wallet_allowed_chains": ("wallet chain allowlists are a human decision", True),
    # ``if not contracts: return True  # Empty = allow all`` — an empty list
    # REMOVES the restriction, so there is no safe revoke value.
    "wallet_allowed_contracts": (
        "wallet contract allowlists are a human decision", False,
    ),
    # Secrets and external reach. Both are membership-tested, so empty = none.
    "allowed_credentials": ("credential access requires explicit human setup", True),
    "allowed_apis": ("external API allowlists require explicit human setup", True),
    "can_request_user_credentials": (
        "asking the human for credentials requires explicit human setup", True,
    ),
    # Durable control-plane powers. ``can_edit_agent_config`` is the power the
    # operator is exercising right now, so granting it would let the operator
    # propagate its own privilege to an agent it manages.
    "can_manage_fleet": ("durable fleet management requires explicit human setup", True),
    "can_manage_teams": ("team management requires explicit human setup", True),
    "can_edit_agent_config": (
        "config-edit rights would propagate the operator's own privilege", True,
    ),
    "can_route_tasks": ("task-routing rights require explicit human setup", True),
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


def _type_error(key: str, value: Any) -> str | None:
    """Reject a value that ``AgentPermissions`` could not hold.

    The ``/edit-soft`` endpoint writes the submitted permissions JSON and only
    then calls ``PermissionMatrix.reload()``, which catches JSON and I/O errors
    but not model-construction errors. So a well-formed-JSON-but-wrong-TYPE
    value (``{"can_message": null}``) passed the ceiling, was persisted, and
    then blew up on reload — leaving ``permissions.json`` poisoned and live
    permission state empty or half-rebuilt, with the request returning 500.

    Probing the real model keeps this derived from ``AgentPermissions`` rather
    than a second hand-written type table.
    """
    from src.shared.types import AgentPermissions

    try:
        # strict: a permissions edit must not rely on coercion. Lax mode turns
        # "yes" and 1 into True, so a string could grant a boolean capability.
        AgentPermissions.model_validate(
            {"agent_id": "_ceiling_type_probe", key: value}, strict=True,
        )
    except Exception:
        return (
            f"Invalid value for permission '{key}': {value!r} is not a valid "
            f"{key} value. The edit was rejected before it could be written."
        )
    return None


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
            reason, revoke_allowed = _OPERATOR_FORBIDDEN[key]
            # Revoking a dangerous power is allowed — but only where some
            # value unambiguously means LESS. See the module docstring.
            if revoke_allowed and _is_de_escalation(value):
                continue
            return (
                f"Permission ceiling exceeded: '{key}' cannot be granted by "
                f"the operator — {reason}. Use the "
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

        # Type-check BEFORE the bound check: an unwritable value must never
        # reach the file, and the bound checks below assume the right shape.
        type_err = _type_error(key, value)
        if type_err:
            return type_err

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
