"""Tests for ``ActionPolicyEngine`` (plan §8 #17).

Covers:

* Static tier registry completeness — every Phase-5 classified action
  has a tier, every gated tier has a compiled-in default.
* Default decisions per tier with no ``config/policy.yaml`` present.
* yaml override load + mtime-based reload (external hand-edit).
* Malformed yaml (bad version, non-mapping, unknown keys) falls back to
  the compiled-in defaults rather than crashing or fail-opening/closing
  beyond them.
* The irreversible-tier floor: a yaml (or per-agent override) trying to
  set it below ``hold`` is clamped back to ``hold``.
* Per-agent overrides beat the tier-level default.
* Audit rows are written for ``hold``/``deny``/``allow_audit`` but NOT
  for plain ``allow`` (the financial-tier default — see the module
  docstring's deviation note).
* No agent-reachable policy-write surface exists anywhere.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from src.host.policy import (
    ACTION_TIERS,
    DEFAULT_TIER_DECISIONS,
    TIERS,
    VALID_DECISIONS,
    ActionPolicyEngine,
)


def _engine(tmp_path, **kwargs) -> ActionPolicyEngine:
    bb = kwargs.pop("blackboard", None) or MagicMock()
    path = kwargs.pop("config_path", None) or str(tmp_path / "policy.yaml")
    return ActionPolicyEngine(bb, config_path=path, **kwargs)


def _write_yaml(path, text: str) -> None:
    path.write_text(text)


# ── Tier registry completeness ────────────────────────────────────


class TestTierRegistry:
    def test_every_phase5_action_is_classified(self):
        expected = {
            "agent_delete": "irreversible",
            "team_delete": "irreversible",
            "wallet_transfer": "financial",
            "wallet_execute": "financial",
            "notify_user": "external_visible",
            "connector_call": "external_visible",
        }
        assert ACTION_TIERS == expected

    def test_every_gated_tier_has_a_default(self):
        for action_kind, tier in ACTION_TIERS.items():
            assert tier in DEFAULT_TIER_DECISIONS, f"{action_kind} tier {tier} has no default"

    def test_reversible_internal_documented_but_not_gated(self):
        """Config edits' tier exists for documentation (module docstring's
        C.3-c section) but is deliberately absent from the gated-default
        map and from the action registry — nothing calls ``evaluate()``
        with it in this unit."""
        assert "reversible_internal" in TIERS
        assert "reversible_internal" not in DEFAULT_TIER_DECISIONS
        assert "reversible_internal" not in ACTION_TIERS.values()

    def test_valid_decisions_are_the_four_named(self):
        assert VALID_DECISIONS == {"allow", "allow_audit", "hold", "deny"}


# ── Default decisions (no yaml) ───────────────────────────────────


class TestDefaults:
    def test_irreversible_defaults_to_hold(self, tmp_path):
        engine = _engine(tmp_path)
        for kind in ("agent_delete", "team_delete"):
            d = engine.evaluate("scout", kind, summary="x")
            assert d.decision == "hold"
            assert d.tier == "irreversible"

    def test_financial_defaults_to_allow(self, tmp_path):
        engine = _engine(tmp_path)
        for kind in ("wallet_transfer", "wallet_execute"):
            d = engine.evaluate("scout", kind, summary="x")
            assert d.decision == "allow"
            assert d.tier == "financial"

    def test_external_visible_defaults_to_allow_audit(self, tmp_path):
        engine = _engine(tmp_path)
        for kind in ("notify_user", "connector_call"):
            d = engine.evaluate("scout", kind, summary="x")
            assert d.decision == "allow_audit"
            assert d.tier == "external_visible"

    def test_unknown_action_kind_raises(self, tmp_path):
        engine = _engine(tmp_path)
        with pytest.raises(ValueError):
            engine.evaluate("scout", "not_a_real_action", summary="x")

    def test_missing_yaml_file_behaves_like_no_yaml(self, tmp_path):
        """Constructing against a path that doesn't exist must not raise
        and must reproduce the compiled-in defaults exactly."""
        engine = _engine(tmp_path, config_path=str(tmp_path / "does-not-exist.yaml"))
        assert engine.evaluate("scout", "agent_delete", summary="x").decision == "hold"
        assert engine.evaluate("scout", "wallet_transfer", summary="x").decision == "allow"


# ── yaml override + mtime reload ──────────────────────────────────


class TestYamlOverrideAndReload:
    def test_tier_override_takes_effect(self, tmp_path):
        cfg = tmp_path / "policy.yaml"
        _write_yaml(cfg, "version: 1\ntiers:\n  external_visible: hold\n")
        engine = _engine(tmp_path, config_path=str(cfg))
        d = engine.evaluate("scout", "notify_user", summary="x")
        assert d.decision == "hold"

    def test_external_edit_is_picked_up_via_mtime(self, tmp_path):
        cfg = tmp_path / "policy.yaml"
        _write_yaml(cfg, "version: 1\ntiers:\n  external_visible: allow_audit\n")
        engine = _engine(tmp_path, config_path=str(cfg))
        assert engine.evaluate("scout", "notify_user", summary="x").decision == "allow_audit"

        # Hand-edit the file to hold, forcing a visible stat change
        # (mirrors ConnectorStore's own reload test).
        _write_yaml(cfg, "version: 1\ntiers:\n  external_visible: hold\n")
        os.utime(cfg, ns=(1_000_000_000, 1_000_000_000))
        assert engine.evaluate("scout", "notify_user", summary="x").decision == "hold"

    def test_deny_override_for_financial_tier(self, tmp_path):
        cfg = tmp_path / "policy.yaml"
        _write_yaml(cfg, "version: 1\ntiers:\n  financial: deny\n")
        engine = _engine(tmp_path, config_path=str(cfg))
        d = engine.evaluate("scout", "wallet_transfer", summary="x")
        assert d.decision == "deny"


# ── Malformed yaml fallback ────────────────────────────────────────


class TestMalformedYamlFallback:
    def test_unparseable_yaml_falls_back_to_defaults(self, tmp_path):
        cfg = tmp_path / "policy.yaml"
        _write_yaml(cfg, "not: [valid, yaml: structure")
        engine = _engine(tmp_path, config_path=str(cfg))
        assert engine.evaluate("scout", "agent_delete", summary="x").decision == "hold"
        assert engine.evaluate("scout", "wallet_transfer", summary="x").decision == "allow"
        assert engine.evaluate("scout", "notify_user", summary="x").decision == "allow_audit"

    def test_wrong_version_falls_back_to_defaults(self, tmp_path):
        cfg = tmp_path / "policy.yaml"
        _write_yaml(cfg, "version: 2\ntiers:\n  external_visible: deny\n")
        engine = _engine(tmp_path, config_path=str(cfg))
        assert engine.evaluate("scout", "notify_user", summary="x").decision == "allow_audit"

    def test_non_mapping_yaml_falls_back_to_defaults(self, tmp_path):
        cfg = tmp_path / "policy.yaml"
        _write_yaml(cfg, "- just\n- a\n- list\n")
        engine = _engine(tmp_path, config_path=str(cfg))
        assert engine.evaluate("scout", "agent_delete", summary="x").decision == "hold"

    def test_unknown_tier_key_ignored(self, tmp_path):
        cfg = tmp_path / "policy.yaml"
        _write_yaml(cfg, "version: 1\ntiers:\n  not_a_real_tier: deny\n")
        engine = _engine(tmp_path, config_path=str(cfg))
        # Untouched tiers keep their compiled-in defaults.
        assert engine.evaluate("scout", "notify_user", summary="x").decision == "allow_audit"

    def test_unknown_decision_value_ignored(self, tmp_path):
        cfg = tmp_path / "policy.yaml"
        _write_yaml(cfg, "version: 1\ntiers:\n  financial: sorta_allow\n")
        engine = _engine(tmp_path, config_path=str(cfg))
        assert engine.evaluate("scout", "wallet_transfer", summary="x").decision == "allow"

    def test_reversible_internal_in_yaml_is_rejected(self, tmp_path):
        """The tier this engine never gates can't be configured through
        the tiers block either -- it isn't in the gated-tier allowlist."""
        cfg = tmp_path / "policy.yaml"
        _write_yaml(cfg, "version: 1\ntiers:\n  reversible_internal: hold\n")
        engine = _engine(tmp_path, config_path=str(cfg))
        # No gated tier was touched -- everything still at compiled-in defaults.
        assert engine.evaluate("scout", "notify_user", summary="x").decision == "allow_audit"
        assert engine.evaluate("scout", "wallet_transfer", summary="x").decision == "allow"


# ── Irreversible clamp ─────────────────────────────────────────────


class TestIrreversibleClamp:
    def test_tier_default_allow_is_clamped_to_hold(self, tmp_path):
        cfg = tmp_path / "policy.yaml"
        _write_yaml(cfg, "version: 1\ntiers:\n  irreversible: allow\n")
        engine = _engine(tmp_path, config_path=str(cfg))
        d = engine.evaluate("scout", "agent_delete", summary="x")
        assert d.decision == "hold"

    def test_tier_default_allow_audit_is_clamped_to_hold(self, tmp_path):
        cfg = tmp_path / "policy.yaml"
        _write_yaml(cfg, "version: 1\ntiers:\n  irreversible: allow_audit\n")
        engine = _engine(tmp_path, config_path=str(cfg))
        assert engine.evaluate("scout", "team_delete", summary="x").decision == "hold"

    def test_deny_is_not_clamped_deny_is_stricter_than_hold(self, tmp_path):
        cfg = tmp_path / "policy.yaml"
        _write_yaml(cfg, "version: 1\ntiers:\n  irreversible: deny\n")
        engine = _engine(tmp_path, config_path=str(cfg))
        assert engine.evaluate("scout", "agent_delete", summary="x").decision == "deny"

    def test_per_agent_override_cannot_bypass_the_clamp_either(self, tmp_path):
        cfg = tmp_path / "policy.yaml"
        _write_yaml(
            cfg,
            "version: 1\nagents:\n  scout:\n    irreversible: allow\n",
        )
        engine = _engine(tmp_path, config_path=str(cfg))
        assert engine.evaluate("scout", "agent_delete", summary="x").decision == "hold"


# ── Per-agent override precedence ──────────────────────────────────


class TestPerAgentOverride:
    def test_agent_override_beats_tier_default(self, tmp_path):
        cfg = tmp_path / "policy.yaml"
        _write_yaml(
            cfg,
            "version: 1\n"
            "tiers:\n  external_visible: allow_audit\n"
            "agents:\n  scout:\n    external_visible: hold\n",
        )
        engine = _engine(tmp_path, config_path=str(cfg))
        assert engine.evaluate("scout", "notify_user", summary="x").decision == "hold"
        # A different agent still gets the tier default.
        assert engine.evaluate("analyst", "notify_user", summary="x").decision == "allow_audit"

    def test_agent_override_beats_compiled_in_default_with_no_tier_block(self, tmp_path):
        cfg = tmp_path / "policy.yaml"
        _write_yaml(cfg, "version: 1\nagents:\n  scout:\n    financial: deny\n")
        engine = _engine(tmp_path, config_path=str(cfg))
        assert engine.evaluate("scout", "wallet_transfer", summary="x").decision == "deny"
        assert engine.evaluate("analyst", "wallet_transfer", summary="x").decision == "allow"

    def test_malformed_agent_override_shape_ignored(self, tmp_path):
        cfg = tmp_path / "policy.yaml"
        _write_yaml(cfg, "version: 1\nagents:\n  scout: not_a_mapping\n")
        engine = _engine(tmp_path, config_path=str(cfg))
        assert engine.evaluate("scout", "notify_user", summary="x").decision == "allow_audit"


# ── Audit rows ──────────────────────────────────────────────────────


class TestAudit:
    def test_hold_writes_one_audit_row(self, tmp_path):
        bb = MagicMock()
        engine = _engine(tmp_path, blackboard=bb)
        cfg = tmp_path / "policy.yaml"
        _write_yaml(cfg, "version: 1\ntiers:\n  external_visible: hold\n")
        os.utime(cfg, ns=(1_000_000_000, 1_000_000_000))
        engine.evaluate("scout", "notify_user", summary="hello world")
        assert bb.log_audit.call_count == 1
        kwargs = bb.log_audit.call_args.kwargs
        assert kwargs["action"] == "policy_decision"
        assert kwargs["target"] == "scout"
        assert kwargs["field"] == "notify_user"
        assert kwargs["actor"] == "scout"
        assert "hold" in kwargs["after_value"]
        assert "hello world" in kwargs["after_value"]

    def test_deny_writes_one_audit_row(self, tmp_path):
        bb = MagicMock()
        cfg = tmp_path / "policy.yaml"
        _write_yaml(cfg, "version: 1\ntiers:\n  financial: deny\n")
        engine = _engine(tmp_path, blackboard=bb, config_path=str(cfg))
        engine.evaluate("scout", "wallet_transfer", summary="x")
        assert bb.log_audit.call_count == 1

    def test_allow_audit_writes_one_audit_row(self, tmp_path):
        bb = MagicMock()
        engine = _engine(tmp_path, blackboard=bb)
        engine.evaluate("scout", "connector_call", summary="x")
        assert bb.log_audit.call_count == 1

    def test_plain_allow_writes_no_audit_row(self, tmp_path):
        """Deviation from 'every decision is audited': the financial
        tier's compiled-in default is plain allow, and wallet.db already
        journals every attempt -- a policy-level row here would be a
        pure duplicate. See the module docstring."""
        bb = MagicMock()
        engine = _engine(tmp_path, blackboard=bb)
        engine.evaluate("scout", "wallet_transfer", summary="x")
        assert bb.log_audit.call_count == 0

    def test_summary_truncated_to_200_chars(self, tmp_path):
        bb = MagicMock()
        cfg = tmp_path / "policy.yaml"
        _write_yaml(cfg, "version: 1\ntiers:\n  external_visible: hold\n")
        engine = _engine(tmp_path, blackboard=bb, config_path=str(cfg))
        long_summary = "x" * 500
        engine.evaluate("scout", "notify_user", summary=long_summary)
        kwargs = bb.log_audit.call_args.kwargs
        # 200 x's plus the surrounding JSON -- the *summary* substring
        # itself is capped, not the whole after_value blob.
        assert "x" * 201 not in kwargs["after_value"]
        assert "x" * 200 in kwargs["after_value"]

    def test_no_blackboard_never_raises(self, tmp_path):
        cfg = tmp_path / "policy.yaml"
        _write_yaml(cfg, "version: 1\ntiers:\n  external_visible: hold\n")
        engine = ActionPolicyEngine(None, config_path=str(cfg))
        d = engine.evaluate("scout", "notify_user", summary="x")
        assert d.decision == "hold"

    def test_audit_log_failure_never_raises(self, tmp_path):
        bb = MagicMock()
        bb.log_audit.side_effect = RuntimeError("db is gone")
        cfg = tmp_path / "policy.yaml"
        _write_yaml(cfg, "version: 1\ntiers:\n  external_visible: hold\n")
        engine = _engine(tmp_path, blackboard=bb, config_path=str(cfg))
        d = engine.evaluate("scout", "notify_user", summary="x")
        assert d.decision == "hold"


# ── No agent-reachable write surface ────────────────────────────────


class TestNoWriteSurface:
    def test_engine_exposes_no_write_method(self, tmp_path):
        """Grep-style pin: the engine object itself must never grow a
        save/write/set_* method an agent-facing tool could call. Human/
        dashboard edits happen by hand-editing config/policy.yaml on
        disk -- there is no in-process write API to expose."""
        engine = _engine(tmp_path)
        forbidden = (
            "save", "write", "set_tier", "set_agent_override",
            "update", "persist", "set_decision",
        )
        for name in forbidden:
            assert not hasattr(engine, name), f"ActionPolicyEngine grew a write method: {name}"

    def test_no_mesh_route_writes_policy_yaml(self):
        """No POST/PUT/PATCH/DELETE route in the mesh app touches the
        policy surface -- mirrors the grep pin style used for other
        operator-gate invariants in this suite."""
        from src.host.mesh import Blackboard, MessageRouter, PubSub
        from src.host.permissions import PermissionMatrix
        from src.host.server import create_mesh_app

        blackboard = Blackboard(":memory:")
        app = create_mesh_app(
            blackboard=blackboard,
            pubsub=PubSub(),
            router=MessageRouter(PermissionMatrix(), {}),
            permissions=PermissionMatrix(),
        )
        write_methods = {"POST", "PUT", "PATCH", "DELETE"}
        offending = []
        for route in app.routes:
            path = getattr(route, "path", "") or ""
            methods = getattr(route, "methods", None) or set()
            if "policy" in path.lower() and methods & write_methods:
                offending.append((path, methods))
        assert offending == [], f"agent-reachable policy write route(s) found: {offending}"
