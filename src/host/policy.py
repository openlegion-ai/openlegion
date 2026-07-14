"""ActionPolicyEngine — the single mesh-side gate for consequential agent actions.

Plan §8 #17 (docs/plans/2026-07-04-agent-employee-platform-architecture.md):
one shared pre-check, called at each mesh-terminating action endpoint AFTER
the existing permission check + rate limit (the ``_check_rate_limit`` /
``_record_denial`` placement already used throughout ``src/host/server.py``),
that classifies the action into a STATIC tier and returns one of four
decisions: ``allow | allow_audit | hold | deny``.

Static tier registry (module-level, never varies at runtime):
    irreversible         agent_delete, team_delete
    financial            wallet_transfer, wallet_execute
    external_visible     notify_user, connector_call
    reversible_internal  config edits -- documented here, NOT gated
                          through this engine (see the C.3-c note below);
                          no action_kind is registered for it because
                          nothing calls :meth:`ActionPolicyEngine.evaluate`
                          with it in this unit.

U3 default policy (no ``config/policy.yaml`` present) preserves EVERY
currently-working flow exactly as today -- no new denials by default:
    irreversible      -> hold         (delete already worked this way)
    financial         -> allow        (wallet caps in
                                        ``WalletService._check_policy``
                                        remain the real governor)
    external_visible  -> allow_audit  (notify/connector calls execute,
                                        plus one audit row)

C.3-c boundary (Appendix C.3-c): approval and undo are different axes
under one policy surface, and this module is the place that says so.
  * Approval (THIS module + ``src/host/pending_actions.py``) gates BEFORE
    execution -- a held action is released only by a verified-human
    confirm (``_confirm_origin_check`` in ``server.py``).
  * Undo (``src/host/change_history.py``) reverts an APPLIED change AFTER
    the fact -- soft-edit receipts with a 5/30-minute window.
Undo is never grafted onto an irreversible or financial action: once a
delete executes or a wallet transfer broadcasts, there is nothing to
"revert" that isn't itself a brand-new consequential action needing its
own approval. Config edits are reversible-internal and live entirely on
the undo axis -- they never call :meth:`ActionPolicyEngine.evaluate`.

``config/policy.yaml`` is human/dashboard-write-only -- mirrors
``src/host/connectors.py``'s ``ConnectorStore`` posture: mtime-based
reload of a hand-edited file, no agent-facing write tool, no mesh write
endpoint. Schema v1::

    version: 1
    tiers:
      external_visible: allow_audit | hold | deny
      financial: allow | hold | deny
      irreversible: hold             # clamped -- see below
    agents:
      <agent_id>:
        <tier>: <decision>

Per-agent overrides win over the tier default. ``irreversible`` can never
be configured below ``hold`` -- a yaml author writing
``irreversible: allow`` (tier default OR a per-agent override) gets
clamped back to ``hold`` with a logged warning: deletes must never
auto-execute. Malformed yaml (bad/missing version, non-mapping shape,
unknown tier/decision keys, unreadable/corrupt file) logs an error and
falls back to the compiled-in defaults above -- never fails open into
allow-everything, never fails closed into deny-everything beyond those
defaults.

U5 addition -- probation preset (plan §8 #19), schema v1 extended with an
optional top-level ``probation:`` block::

    probation:
      enabled: true            # default false
      min_accepted: 5          # default 5, clamped 1-10000
      tiers: [external_visible, financial]   # optional narrowing, default both

Precedence (load-bearing, documented here because it is easy to get
backwards): **per-agent override > probation > tier override > compiled
default**. Probation only ever fires on a decision that was resolved from
the tier override or the compiled-in tier default -- an explicit
per-agent row for that tier wins outright and probation never runs for
that call. When probation is enabled, the action's tier is one of its
configured ``tiers`` (default both ``external_visible``/``financial`` --
``irreversible`` can never be named here; see below), and the resolved
decision is weaker than ``hold``, the agent's ACCEPTED count (see below)
is compared against ``min_accepted``; below it, the decision is escalated
to ``hold``. Probation only ESCALATES -- it can never resolve a decision
to something LESS strict than what the tier/agent config already
produced, and it can never push a decision past ``hold`` (a yaml ``deny``
stays ``deny``). ``irreversible`` is untouched either way: it is clamped
to ``hold`` unconditionally by :func:`_clamp_for_tier` before probation
ever runs, and probation's tier allowlist can't name it in the first
place.

Accepted count = the plan §6 "after N accepted deliverables" count:
:meth:`TrackRecordStore.counts_for_agent` for the agent, restricted to
``rater_kinds=AUTONOMY_RATER_KINDS`` (the rating-trust rule -- an
operator-agent's own rating never counts), summed over ``outcome ==
"accepted"`` across the ``task_outcome`` and ``summary_rating`` sources.
No ``track_record_store`` wired, or a read failure, is treated as zero
accepted outcomes -- probation fails TOWARD safety (a hold), never toward
a guessed allow, and the failure is logged.

Constraint #12 (absolute, no exceptions): thresholds set here are
OPERATOR policy. There is no lead or agent carve-out anywhere in this
module -- the ``agents:`` block is a set of per-agent-ID rows a human
hand-edits (or a future dashboard write surface edits on their behalf),
never a privilege an agent or lead can grant itself. No agent-facing tool
and no mesh endpoint write this file in this unit.

Audit: ``hold`` / ``deny`` / ``allow_audit`` decisions each write one
``blackboard.log_audit`` row (``action="policy_decision"``). Plain
``allow`` (the financial-tier default) does NOT write a row --
``wallet.db`` already journals every attempt via
``WalletService._audit``, so a policy-level row would be a pure
duplicate for the one tier whose compiled-in default is "allow". This is
a deliberate deviation from "every decision is audited" -- see the unit's
build report for the reasoning. A probation-escalated hold's audit row
additionally carries ``"probation": true`` in its payload -- present ONLY
when probation is what caused the escalation, so every pre-U5 audit
payload (and every non-probation decision today) is byte-identical to
before.

``track_record_store`` (the U3 constructor seam, wired in
``create_mesh_app``) is now READ by :meth:`evaluate` for the probation
preset above. ``cost_tracker`` remains accepted-and-unused -- no part of
plan §8 #19's f(tier, track_record, budget) built in this unit reads
budget; that stays a future seam.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.host.track_record import AUTONOMY_RATER_KINDS
from src.shared.utils import dumps_safe, setup_logging

logger = setup_logging("host.policy")

# Every tier this module knows about, including the documented-but-never-
# gated ``reversible_internal`` (config edits stay on ChangeHistory undo
# receipts -- see the module docstring's C.3-c section).
TIERS: frozenset[str] = frozenset(
    {"reversible_internal", "external_visible", "financial", "irreversible"},
)

# Tiers this engine actually resolves decisions for. ``reversible_internal``
# is excluded on purpose: nothing calls ``evaluate()`` with an action_kind
# mapped to it in this unit, so a yaml author configuring it would be
# configuring dead policy -- ``_load`` rejects it with a logged error.
_GATED_TIERS: frozenset[str] = frozenset({"external_visible", "financial", "irreversible"})

VALID_DECISIONS: frozenset[str] = frozenset({"allow", "allow_audit", "hold", "deny"})

# Severity ordering used only by the irreversible-tier clamp below.
# Higher = stricter. Not exposed -- it exists purely to compare two
# decision strings, not to rank arbitrary external input.
_SEVERITY: dict[str, int] = {"allow": 0, "allow_audit": 1, "hold": 2, "deny": 3}

# Static action-tier registry (plan §8 #17). ``action_kind`` -> tier.
# Every mesh-terminating consequential action this unit gates is listed
# here; nothing outside this module decides an action's tier.
ACTION_TIERS: dict[str, str] = {
    "agent_delete": "irreversible",
    "team_delete": "irreversible",
    "wallet_transfer": "financial",
    "wallet_execute": "financial",
    "notify_user": "external_visible",
    "connector_call": "external_visible",
}

# U3 default policy -- what every tier resolves to with no
# ``config/policy.yaml`` present (or with no override for that tier).
# Preserves every currently-working flow exactly as today; see the
# module docstring.
DEFAULT_TIER_DECISIONS: dict[str, str] = {
    "irreversible": "hold",
    "financial": "allow",
    "external_visible": "allow_audit",
}

# Tiers the probation preset (plan §8 #19) is allowed to name. Deliberately
# excludes "irreversible" -- it is already clamped to "hold" unconditionally
# (see ``_clamp_for_tier``), so naming it in ``probation.tiers`` would
# configure dead policy.
_PROBATION_TIERS: frozenset[str] = frozenset({"external_visible", "financial"})

_PROBATION_MIN_ACCEPTED_DEFAULT = 5
_PROBATION_MIN_ACCEPTED_RANGE = (1, 10000)

# Compiled-in probation defaults -- byte-identical to "no probation block
# at all" (disabled, both tiers, min_accepted=5). Sharing one dict
# instance per resolved config is safe: every value is immutable
# (bool/int/frozenset).
_DEFAULT_PROBATION: dict[str, Any] = {
    "enabled": False,
    "min_accepted": _PROBATION_MIN_ACCEPTED_DEFAULT,
    "tiers": _PROBATION_TIERS,
}


def _clamp_for_tier(tier: str, decision: str) -> str:
    """Enforce the one hard floor: ``irreversible`` can never resolve
    below ``hold`` -- a delete must never auto-execute, regardless of
    what a tier default or a per-agent override says."""
    if tier == "irreversible" and _SEVERITY[decision] < _SEVERITY["hold"]:
        logger.warning(
            "policy config attempted to set the irreversible tier's decision "
            "to %r; clamped to 'hold' -- deletes must never auto-execute",
            decision,
        )
        return "hold"
    return decision


@dataclass(frozen=True)
class PolicyDecision:
    """Result of :meth:`ActionPolicyEngine.evaluate`."""

    decision: str  # allow | allow_audit | hold | deny
    tier: str
    action_kind: str


class ActionPolicyEngine:
    """The static action-tier policy gate (plan §8 #17).

    One instance is constructed per mesh process (see
    ``create_mesh_app``) and shared by every gated endpoint. Read-mostly:
    a hand-edited ``config/policy.yaml`` is picked up via mtime-based
    reload (the same pattern as ``src.host.connectors.ConnectorStore``) --
    there is no agent-facing write path, matching the human/dashboard-
    write-only posture plan §8 #17 requires.
    """

    def __init__(
        self,
        blackboard: Any | None,
        *,
        config_path: str = "config/policy.yaml",
        track_record_store: Any | None = None,
        cost_tracker: Any | None = None,
    ) -> None:
        self._blackboard = blackboard
        self._path = Path(config_path)
        self._lock = threading.RLock()
        # (st_mtime_ns, st_size) of the file as last loaded; None = no file.
        self._loaded_stat: tuple[int, int] | None = None
        self._tier_overrides: dict[str, str] = {}
        self._agent_overrides: dict[str, dict[str, str]] = {}
        self._probation: dict[str, Any] = dict(_DEFAULT_PROBATION)
        # U5 seam (plan §8 #19's f(tier, track_record, budget)) -- accepted
        # and stored, unused, so the earned-autonomy unit can read from
        # these without another constructor-signature change.
        self.track_record_store = track_record_store
        self.cost_tracker = cost_tracker
        self._load()

    # ── config load / reload ─────────────────────────────────────────

    def _stat(self) -> tuple[int, int] | None:
        try:
            st = self._path.stat()
            return (st.st_mtime_ns, st.st_size)
        except OSError:
            return None

    def _load(self) -> None:
        """(Re)load ``config/policy.yaml``. Caller holds the lock (or is
        ``__init__``).

        Fail-closed to the compiled-in defaults on ANY problem -- missing
        file, corrupt/non-mapping yaml, unsupported version, unknown
        tier/decision keys. Never crashes boot; never silently produces a
        policy stricter or looser than the documented defaults.
        """
        self._loaded_stat = self._stat()
        self._tier_overrides = {}
        self._agent_overrides = {}
        self._probation = dict(_DEFAULT_PROBATION)
        if not self._path.exists():
            return
        try:
            raw = yaml.safe_load(self._path.read_text())
        except (yaml.YAMLError, OSError) as e:
            logger.error(
                "Policy config %s is corrupt or unreadable (%s); falling back to defaults",
                self._path, e,
            )
            return
        if raw is None:
            return
        if not isinstance(raw, dict):
            logger.error(
                "Policy config %s is not a mapping; falling back to defaults", self._path,
            )
            return
        version = raw.get("version")
        if version != 1:
            logger.error(
                "Policy config %s has unsupported version %r (expected 1); "
                "falling back to defaults",
                self._path, version,
            )
            return
        self._tier_overrides = self._parse_tier_overrides(raw.get("tiers"))
        self._agent_overrides = self._parse_agent_overrides(raw.get("agents"))
        self._probation = self._parse_probation(raw.get("probation"))

    def _parse_tier_overrides(self, tiers: Any) -> dict[str, str]:
        if tiers is None:
            return {}
        if not isinstance(tiers, dict):
            logger.error(
                "Policy config %s: 'tiers' must be a mapping; ignored", self._path,
            )
            return {}
        result: dict[str, str] = {}
        for tier, decision in tiers.items():
            if tier not in _GATED_TIERS:
                logger.error(
                    "Policy config %s: unknown or ungated tier %r ignored",
                    self._path, tier,
                )
                continue
            if decision not in VALID_DECISIONS:
                logger.error(
                    "Policy config %s: unknown decision %r for tier %r ignored",
                    self._path, decision, tier,
                )
                continue
            result[tier] = decision
        return result

    def _parse_agent_overrides(self, agents: Any) -> dict[str, dict[str, str]]:
        if agents is None:
            return {}
        if not isinstance(agents, dict):
            logger.error(
                "Policy config %s: 'agents' must be a mapping; ignored", self._path,
            )
            return {}
        result: dict[str, dict[str, str]] = {}
        for agent_id, overrides in agents.items():
            if not isinstance(overrides, dict):
                logger.error(
                    "Policy config %s: agent override for %r must be a mapping; ignored",
                    self._path, agent_id,
                )
                continue
            clean: dict[str, str] = {}
            for tier, decision in overrides.items():
                if tier not in _GATED_TIERS:
                    logger.error(
                        "Policy config %s: unknown or ungated tier %r for agent %r ignored",
                        self._path, tier, agent_id,
                    )
                    continue
                if decision not in VALID_DECISIONS:
                    logger.error(
                        "Policy config %s: unknown decision %r for agent %r tier %r ignored",
                        self._path, decision, agent_id, tier,
                    )
                    continue
                clean[tier] = decision
            if clean:
                result[str(agent_id)] = clean
        return result

    def _parse_probation(self, raw: Any) -> dict[str, Any]:
        """Parse the optional ``probation:`` block (plan §8 #19).

        Each field is validated independently and falls back to ITS OWN
        compiled-in default on a malformed value -- mirroring the
        per-field-tolerant style of :meth:`_parse_tier_overrides` /
        :meth:`_parse_agent_overrides` above, rather than discarding the
        whole block for one bad field. No ``probation:`` key at all (or
        an empty mapping) resolves to all-defaults -- disabled, both
        tiers, ``min_accepted=5`` -- which is exactly "no yaml at all"
        (the U5 regression requirement: absent/disabled must be
        byte-identical to U3-landed main).
        """
        result: dict[str, Any] = dict(_DEFAULT_PROBATION)
        if raw is None:
            return result
        if not isinstance(raw, dict):
            logger.error(
                "Policy config %s: 'probation' must be a mapping; ignored (disabled)",
                self._path,
            )
            return result
        if "enabled" in raw:
            enabled = raw["enabled"]
            if isinstance(enabled, bool):
                result["enabled"] = enabled
            else:
                logger.error(
                    "Policy config %s: probation.enabled must be a boolean; "
                    "ignored (default false)",
                    self._path,
                )
        if "min_accepted" in raw:
            min_accepted = raw["min_accepted"]
            if isinstance(min_accepted, bool) or not isinstance(min_accepted, int):
                logger.error(
                    "Policy config %s: probation.min_accepted must be an integer; "
                    "ignored (default %d)",
                    self._path, _PROBATION_MIN_ACCEPTED_DEFAULT,
                )
            else:
                lo, hi = _PROBATION_MIN_ACCEPTED_RANGE
                clamped = max(lo, min(min_accepted, hi))
                if clamped != min_accepted:
                    logger.warning(
                        "Policy config %s: probation.min_accepted %r clamped to %d",
                        self._path, min_accepted, clamped,
                    )
                result["min_accepted"] = clamped
        if "tiers" in raw:
            tiers_raw = raw["tiers"]
            if not isinstance(tiers_raw, list):
                logger.error(
                    "Policy config %s: probation.tiers must be a list; "
                    "ignored (default: both tiers)",
                    self._path,
                )
            else:
                clean_tiers: set[str] = set()
                for tier in tiers_raw:
                    if tier in _PROBATION_TIERS:
                        clean_tiers.add(tier)
                    else:
                        logger.error(
                            "Policy config %s: probation.tiers entry %r ignored "
                            "(must be 'external_visible' or 'financial')",
                            self._path, tier,
                        )
                if clean_tiers:
                    result["tiers"] = frozenset(clean_tiers)
                else:
                    logger.error(
                        "Policy config %s: probation.tiers had no valid entries; "
                        "falling back to both tiers",
                        self._path,
                    )
        return result

    def _maybe_reload(self) -> None:
        """Pick up an external edit to the file (hand-edited policy on a
        headless deploy). Caller holds the lock."""
        current = self._stat()
        if current == self._loaded_stat:
            return
        self._load()
        logger.info("Policy config %s changed on disk; reloaded", self._path)

    # ── decision ──────────────────────────────────────────────────────

    def evaluate(self, agent_id: str, action_kind: str, *, summary: str) -> PolicyDecision:
        """Classify ``action_kind`` and return the decision for ``agent_id``.

        Resolution order: per-agent override for the action's tier, then
        the tier-level override, then the compiled-in default -- with the
        irreversible-tier floor (:func:`_clamp_for_tier`) applied next no
        matter which source produced the decision, and the probation
        preset (plan §8 #19) applied last. Full precedence: **per-agent
        override > probation > tier override > compiled default**
        (irreversible's clamp sits outside this ladder entirely -- it
        always wins regardless of source).

        Raises ``ValueError`` if ``action_kind`` isn't in the static
        :data:`ACTION_TIERS` registry -- every call site in this unit
        passes a literal from that registry, so this only fires on a
        programming error, never on live traffic.

        Writes one audit row for every decision except plain ``allow``
        (see the module docstring's audit-deviation note); a probation-
        escalated hold's row additionally carries ``"probation": true``.
        """
        tier = ACTION_TIERS.get(action_kind)
        if tier is None:
            raise ValueError(f"unknown action_kind for policy evaluation: {action_kind!r}")
        with self._lock:
            self._maybe_reload()
            agent_overrides = self._agent_overrides.get(agent_id, {})
            if tier in agent_overrides:
                decision = agent_overrides[tier]
                from_per_agent = True
            elif tier in self._tier_overrides:
                decision = self._tier_overrides[tier]
                from_per_agent = False
            else:
                decision = DEFAULT_TIER_DECISIONS[tier]
                from_per_agent = False
            probation = self._probation
        decision = _clamp_for_tier(tier, decision)
        probation_applied = False
        if (
            not from_per_agent
            and probation["enabled"]
            and tier in probation["tiers"]
            and _SEVERITY[decision] < _SEVERITY["hold"]
            and self._accepted_count(agent_id) < probation["min_accepted"]
        ):
            decision = "hold"
            probation_applied = True
        if decision != "allow":
            self._write_audit(
                agent_id, action_kind, tier, decision, summary, probation=probation_applied,
            )
        return PolicyDecision(decision=decision, tier=tier, action_kind=action_kind)

    def _accepted_count(self, agent_id: str) -> int:
        """The probation preset's "after N accepted DELIVERABLES" count
        (plan §6/§8 #19): the number of DISTINCT ``(source, ref_id)``
        deliverables whose LATEST outcome is ``accepted``, restricted to
        :data:`AUTONOMY_RATER_KINDS` (the rating-trust rule -- an
        operator-agent's own rating never counts toward probation
        release, same as every other autonomy score).

        Reads :meth:`TrackRecordStore.distinct_accepted_count`, NOT the
        raw append-only ``counts_for_agent`` ledger: probation must count
        deliverables, not rating rows, so re-rating one task N times can't
        release probation (M5) and a retracted acceptance (accept-then-
        reject on the same ref) stops counting (latest event wins). The
        raw ledger stays the display/learning surface.

        No store wired, or a read failure, is treated as zero accepted
        outcomes -- fail TOWARD safety (keeps the action on hold) rather
        than guessing an agent has already earned release. A failure is
        logged so a broken ledger doesn't silently hold everyone forever
        without a trace.
        """
        store = self.track_record_store
        if store is None:
            return 0
        try:
            return store.distinct_accepted_count(agent_id, rater_kinds=AUTONOMY_RATER_KINDS)
        except Exception as e:
            logger.warning(
                "probation: track record read failed for agent %r; treating "
                "accepted count as 0 (fail toward hold): %s",
                agent_id, e,
            )
            return 0

    def _write_audit(
        self,
        agent_id: str,
        action_kind: str,
        tier: str,
        decision: str,
        summary: str,
        *,
        probation: bool = False,
    ) -> None:
        if self._blackboard is None:
            return
        payload = {
            "tier": tier,
            "decision": decision,
            "summary": (summary or "")[:200],
        }
        if probation:
            payload["probation"] = True
        try:
            self._blackboard.log_audit(
                action="policy_decision",
                target=agent_id,
                field=action_kind,
                after_value=dumps_safe(payload),
                actor=agent_id,
                provenance="agent",
            )
        except Exception as e:
            logger.warning("policy_decision audit log failed: %s", e)
