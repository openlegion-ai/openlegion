"""The operator permission ceiling must deny by default.

The ceiling gates what the OPERATOR — an LLM-driven agent — may grant to the
agents it manages. It is enforced client-side in ``operator_tools._validate_edit``
and re-checked server-side on the mesh ``/edit-soft`` endpoint, so a fooled or
prompt-injected operator cannot route around its own guard.

It used to iterate the ceiling TABLE rather than the submitted payload, with a
``if key not in new_value: continue``. A field absent from the table was
therefore never examined. The table named 6 of ``AgentPermissions``' 26 fields,
so the other 19 were grantable without limit — including ``allowed_credentials``,
the durable control-plane flags, and every wallet spending limit. The
``can_use_wallet`` FLAG was blocked while ``wallet_spend_limit_daily_usd`` was
not, which defeated the protection the table cited as its own reason to exist:
on an agent a human had already enabled the wallet for, the operator could raise
the daily cap without bound.

``TestEveryPermissionFieldIsClassified`` is the load-bearing test here. It
derives its expectation from ``AgentPermissions`` itself, so adding a field
without deciding whether the operator may grant it fails CLOSED.
"""

from __future__ import annotations

import pytest

from src.shared.operator_ceiling import (
    _OPERATOR_FORBIDDEN,
    _OPERATOR_PERMISSION_CEILING,
    clamp_to_operator_ceiling,
)
from src.shared.types import AgentPermissions


class TestEveryPermissionFieldIsClassified:
    """Every AgentPermissions field must be manageable OR forbidden.

    Derived from the model rather than restated, so a new field cannot be
    added without a deliberate decision about operator access.
    """

    def test_tables_cover_the_model_exactly(self):
        model = set(AgentPermissions.model_fields)
        classified = set(_OPERATOR_PERMISSION_CEILING) | set(_OPERATOR_FORBIDDEN)
        unclassified = model - classified
        assert not unclassified, (
            "These AgentPermissions fields are in neither the manageable nor "
            "the forbidden table, so the operator's access to them was never "
            f"decided: {sorted(unclassified)}. Add each to "
            "_OPERATOR_PERMISSION_CEILING (with its ceiling) or to "
            "_OPERATOR_FORBIDDEN (with the reason a human must do it)."
        )
        stale = classified - model
        assert not stale, (
            "These fields are classified but no longer exist on "
            f"AgentPermissions: {sorted(stale)}"
        )

    def test_the_two_tables_are_disjoint(self):
        both = set(_OPERATOR_PERMISSION_CEILING) & set(_OPERATOR_FORBIDDEN)
        assert not both, f"Fields both manageable and forbidden: {sorted(both)}"

    def test_an_unclassified_key_is_refused(self):
        """Fail closed on anything the tables don't name — typo or new field."""
        err = clamp_to_operator_ceiling("permissions", {"can_do_anything": True})
        assert err is not None and "not an operator-manageable permission" in err


class TestEscalationIsBlocked:
    """Each of these was ALLOWED before the table became deny-by-default."""

    @pytest.mark.parametrize(
        ("label", "payload"),
        [
            ("all credentials", {"allowed_credentials": ["*"]}),
            ("all external APIs", {"allowed_apis": ["*"]}),
            ("fleet management", {"can_manage_fleet": True}),
            ("team management", {"can_manage_teams": True}),
            ("peer config edit", {"can_edit_agent_config": True}),
            ("task routing", {"can_route_tasks": True}),
            ("ask human for creds", {"can_request_user_credentials": True}),
            ("daily wallet cap", {"wallet_spend_limit_daily_usd": 1_000_000.0}),
            ("per-tx wallet cap", {"wallet_spend_limit_per_tx_usd": 1_000_000.0}),
            ("wallet rate limit", {"wallet_rate_limit_per_hour": 10_000}),
            ("wallet chains", {"wallet_allowed_chains": ["*"]}),
            ("wallet contracts", {"wallet_allowed_contracts": ["*"]}),
            ("agent identity", {"agent_id": "someone-else"}),
        ],
    )
    def test_grant_is_refused(self, label: str, payload: dict):
        err = clamp_to_operator_ceiling("permissions", payload)
        assert err is not None, f"operator can still grant {label}: {payload}"
        assert "dashboard" in err

    def test_wallet_caps_are_blocked_not_just_the_flag(self):
        """The specific hole: the flag was gated, the limits were not.

        On an agent whose wallet a human already enabled, raising the cap was
        a pure-profit escalation that never touched the gated flag.
        """
        assert clamp_to_operator_ceiling(
            "permissions", {"can_use_wallet": True},
        ) is not None
        assert clamp_to_operator_ceiling(
            "permissions", {"wallet_spend_limit_daily_usd": 1_000_000.0},
        ) is not None

    def test_a_mixed_payload_is_refused_on_its_worst_key(self):
        """A legal key alongside an illegal one must not launder it."""
        err = clamp_to_operator_ceiling(
            "permissions",
            {"can_use_browser": True, "allowed_credentials": ["*"]},
        )
        assert err is not None and "allowed_credentials" in err


class TestRevokeIsAllowedWhereItIsUnambiguous:
    """Taking a dangerous power AWAY needs no human — where "less" is real.

    Each field below was checked against its ENFORCEMENT site, not its name:
    ``can_use_api`` is ``service in perms.allowed_apis``, ``can_manage_vault``
    is ``bool(perms.allowed_credentials)``, and ``can_use_wallet_chain``
    requires explicit membership. For all of those, empty means "nothing".
    """

    @pytest.mark.parametrize(
        "payload",
        [
            {"can_use_wallet": False},
            {"can_manage_fleet": False},
            {"can_manage_teams": False},
            {"can_edit_agent_config": False},
            {"can_route_tasks": False},
            {"can_request_user_credentials": False},
            {"allowed_credentials": []},
            {"allowed_apis": []},
            {"wallet_allowed_chains": []},
        ],
    )
    def test_de_escalation_passes(self, payload: dict):
        assert clamp_to_operator_ceiling("permissions", payload) is None, (
            f"revoking should be allowed: {payload}"
        )


class TestEmptyAndZeroAreNotAlwaysDeEscalation:
    """Two fields inverted the "empty/zero means less" assumption.

    Shape-based de-escalation is wrong here, and getting it wrong would have
    reopened the exact hole this module exists to close — through the
    de-escalation path rather than the grant path.
    """

    def test_empty_contract_allowlist_is_a_widening(self):
        """``can_access_wallet_contract``:
        ``if not contracts: return True  # Empty = allow all``.

        An agent pinned to two contract addresses would be freed to call ANY
        contract by an edit that looks like a revoke.
        """
        assert clamp_to_operator_ceiling(
            "permissions", {"wallet_allowed_contracts": []},
        ) is not None

    @pytest.mark.parametrize(
        "field",
        [
            "wallet_spend_limit_per_tx_usd",
            "wallet_spend_limit_daily_usd",
            "wallet_rate_limit_per_hour",
        ],
    )
    def test_zeroing_a_wallet_cap_is_a_widening(self, field: str):
        """``get_wallet_limits``: "0 = use global default".

        Zeroing an agent whose per-agent cap is TIGHTER than the global
        default raises it to the global default.
        """
        assert clamp_to_operator_ceiling("permissions", {field: 0}) is not None
        assert clamp_to_operator_ceiling("permissions", {field: 0.0}) is not None

    def test_blanking_agent_id_is_refused(self):
        assert clamp_to_operator_ceiling("permissions", {"agent_id": ""}) is not None

    def test_fields_without_a_safe_revoke_are_marked_as_such(self):
        """Pin the flags themselves, so the rationale can't be lost in a
        future edit that only reads the value-shape helper."""
        no_revoke = {f for f, (_r, ok) in _OPERATOR_FORBIDDEN.items() if not ok}
        assert no_revoke == {
            "agent_id",
            "wallet_allowed_contracts",
            "wallet_spend_limit_per_tx_usd",
            "wallet_spend_limit_daily_usd",
            "wallet_rate_limit_per_hour",
        }


class TestRoutineManagementStillWorks:
    """The tightening must not cost the operator its actual job."""

    @pytest.mark.parametrize(
        "payload",
        [
            {"can_use_browser": True},
            {"can_use_internet": True},
            {"can_spawn": True},
            {"can_manage_cron": True},
            {"can_view_fleet_metrics": True},
            {"blackboard_read": ["*"]},
            {"blackboard_write": ["tasks/*", "context/*", "status/*"]},
            {"browser_actions": ["click", "type"]},
            {"allowed_skills": ["research"]},
            {"can_message": ["*"]},
            {"can_publish": ["events/*"]},
            {"can_subscribe": ["events/*"]},
        ],
    )
    def test_allowed(self, payload: dict):
        assert clamp_to_operator_ceiling("permissions", payload) is None

    def test_blackboard_write_outside_the_allowlist_still_refused(self):
        """Unchanged behavior — kept so the rewrite can't silently widen it."""
        assert clamp_to_operator_ceiling(
            "permissions", {"blackboard_write": ["secrets/*"]},
        ) is not None

    def test_non_permission_edits_are_not_the_ceilings_concern(self):
        assert clamp_to_operator_ceiling("model", "claude-opus-5") is None
        assert clamp_to_operator_ceiling("permissions", "not-a-dict") is None


class TestMalformedValuesCannotBeWritten:
    """A wrong-TYPE value must be refused before it reaches permissions.json.

    ``/edit-soft`` writes the submitted permissions JSON and only THEN calls
    ``PermissionMatrix.reload()``, which catches JSON and I/O errors but not
    model-construction errors. So ``{"can_message": null}`` passed the ceiling,
    was persisted, and blew up on reload — leaving the file poisoned and live
    permission state empty or half-rebuilt, with the request returning 500.

    The check probes the real ``AgentPermissions`` model, so it stays derived
    from the model rather than becoming a second hand-written type table.
    """

    @pytest.mark.parametrize(
        "payload",
        [
            {"can_message": None},
            {"can_message": 12345},
            {"can_message": {"a": 1}},
            {"allowed_skills": 5},
            {"browser_actions": 7},
            {"blackboard_read": "not-a-list"},
        ],
    )
    def test_wrong_type_is_refused(self, payload: dict):
        err = clamp_to_operator_ceiling("permissions", payload)
        assert err is not None, f"unwritable value accepted: {payload}"
        assert "Invalid value" in err

    def test_validation_is_strict_so_coercion_cannot_grant(self):
        """Lax pydantic turns "yes" and 1 into True.

        A permissions edit must not depend on coercion — a string should never
        become a boolean capability grant.
        """
        assert clamp_to_operator_ceiling(
            "permissions", {"can_use_browser": "yes"},
        ) is not None
        assert clamp_to_operator_ceiling(
            "permissions", {"can_use_browser": 1},
        ) is not None

    @pytest.mark.parametrize(
        "payload",
        [
            {"can_use_browser": True},
            {"can_use_browser": False},
            {"can_message": ["*"]},
            {"browser_actions": None},
            {"browser_actions": ["click"]},
            {"allowed_skills": []},
        ],
    )
    def test_well_formed_values_still_pass(self, payload: dict):
        assert clamp_to_operator_ceiling("permissions", payload) is None
