"""Permission matrix enforcement for mesh operations.

Loaded from config/permissions.json.
Uses glob patterns for blackboard path matching.
Default policy: deny everything not explicitly allowed.
"""

from __future__ import annotations

import fnmatch
import json
from pathlib import Path

from src.host.credentials import is_system_credential

# Operator permission ceiling — moved to ``src/shared/operator_ceiling`` so the
# agent container (which ships ``src/agent`` + ``src/shared`` but NOT ``src/host``)
# can import it from the operator tool. Re-exported here so existing host-side
# imports (e.g. ``from src.host.permissions import clamp_to_operator_ceiling`` in
# ``server.py``) keep resolving. Single source of truth lives in the shared module.
from src.shared.operator_ceiling import (  # noqa: F401
    _OPERATOR_PERMISSION_CEILING,
    clamp_to_operator_ceiling,
)
from src.shared.types import AgentPermissions
from src.shared.utils import setup_logging

logger = setup_logging("host.permissions")


# Known browser action names. Used ONLY for mesh-side input validation —
# catches typo'd action names with a clean 400 before reaching the browser
# service. Does NOT gate permissions (every known action is granted to any
# agent with `can_use_browser=true` unless an explicit `browser_actions`
# list narrows the set; see `can_browser_action`).
#
# As new actions are added in later phases they SHOULD be added here so the
# mesh recognizes them. Agents gain access to them automatically; operators
# who want to restrict specific actions do so per-template via
# `AgentPermissions.browser_actions`.
KNOWN_BROWSER_ACTIONS: frozenset[str] = frozenset({
    # Legacy 16 — present pre-Phase 1 refactor.
    "navigate", "snapshot", "click", "type", "hover",
    "screenshot", "reset", "focus", "status", "detect_captcha", "scroll",
    "wait_for", "press_key", "go_back", "go_forward", "switch_tab",
    # Phase 1.5 file-transfer actions. Reserved here so the mesh input
    # validation accepts the action names even before Phase 1.5's
    # endpoints land in `src/browser/server.py`. If a caller invokes
    # these before 1.5 is deployed, the browser service returns a clean
    # 404 on the unknown URL instead of the mesh rejecting the action
    # name as unknown — avoids a cross-PR merge-order dependency.
    "upload_file", "download",
    # Phase 5 §8.5 / §8.6 default-allow read-only / nav-equivalent actions.
    "find_text", "open_tab",
    # Phase 6 §9.4 compound find-text+type with CAPTCHA-mid-flow partial success
    "fill_form",
    # Phase 6 §9.3 coordinate-based click with overlay pre-check.
    "click_xy",
    # Phase 6 §9.1 read-only network inspection.
    "inspect_requests",
    # Phase 8 §11.14 explicit-trigger captcha-handling tools. Default-allow
    # alongside the other browser actions; operators who want to forbid
    # solver spend per-template can still add ``solve_captcha`` to a
    # narrowed ``browser_actions`` denylist (or set
    # ``CAPTCHA_DISABLED=true`` fleet-wide via flags.py).
    "solve_captcha", "request_captcha_help",
    # Phase 8 §11.14 + browser-login handoff.  Both endpoints emit a
    # dashboard handoff card — ``request_browser_login`` for VNC-driven
    # interactive login, ``request_captcha_help`` for human captcha
    # assistance — and both are now permission-gated at the dedicated
    # mesh endpoints (``/mesh/browser-login-request``,
    # ``/mesh/browser-captcha-help-request``).  Without these names in
    # the validator set, an operator who narrows ``browser_actions`` to
    # exclude the handoff would still see the second-call-to-dedicated-
    # endpoint succeed, defeating permission narrowing.
    "request_browser_login",
})

# Back-compat alias — retained so `host/server.py` and test fixtures that
# imported `LEGACY_BROWSER_ACTIONS` keep working. Prefer `KNOWN_BROWSER_ACTIONS`
# in new code.
LEGACY_BROWSER_ACTIONS = KNOWN_BROWSER_ACTIONS


class PermissionMatrix:
    """Enforces agent-level permissions for mesh operations."""

    def __init__(self, config_path: str = "config/permissions.json"):
        self.permissions: dict[str, AgentPermissions] = {}
        self._config_path = config_path
        # L10: names of credentials that loaded under the SYSTEM_* tier.
        # Injected post-construction by the runtime via
        # ``set_system_credential_names``. Blocks agent resolution of a
        # system secret even when its name doesn't match the provider-key
        # shape heuristic (``is_system_credential``).
        self._system_credential_names: frozenset[str] = frozenset()
        self._load(config_path)

    def set_system_credential_names(self, names) -> None:
        """Register the names of credentials loaded under the system tier.

        Lowercased + frozen. Called by the runtime once the credential
        vault is built so ``can_access_credential`` can deny by LOADED
        TIER, not just by name shape (L10).
        """
        self._system_credential_names = frozenset(
            n.lower() for n in (names or [])
        )

    def reload(self) -> None:
        """Reload permissions from disk (e.g. after adding an agent at runtime)."""
        self._load(self._config_path)

    def _load(self, config_path: str) -> None:
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Permissions file not found: {config_path}, using deny-all defaults")
            # Clear so stale permissions don't persist (fail closed)
            self.permissions.clear()
            return
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            # Corrupt or unreadable ACL file: fail CLOSED. A parse error must
            # not crash boot and must NOT leave stale grants in place — clear
            # to deny-all, exactly like the missing-file branch above.
            logger.error(
                "Permissions file %s is corrupt or unreadable (%s); "
                "falling back to deny-all defaults",
                config_path, e,
            )
            self.permissions.clear()
            return
        # Clear before repopulating so that removed entries don't persist
        self.permissions.clear()
        for agent_id, perms in data.get("permissions", {}).items():
            self.permissions[agent_id] = AgentPermissions(agent_id=agent_id, **perms)

    def get_permissions(self, agent_id: str) -> AgentPermissions:
        """Get permissions for an agent. Falls back to 'default' template, then deny-all."""
        if agent_id in self.permissions:
            return self.permissions[agent_id]
        default = self.permissions.get("default")
        if default:
            return AgentPermissions(
                agent_id=agent_id,
                can_message=default.can_message,
                can_publish=default.can_publish,
                can_subscribe=default.can_subscribe,
                blackboard_read=default.blackboard_read,
                blackboard_write=default.blackboard_write,
                allowed_apis=default.allowed_apis,
                allowed_credentials=default.allowed_credentials,
                can_use_browser=default.can_use_browser,
                browser_actions=default.browser_actions,
                can_use_internet=default.can_use_internet,
                can_spawn=default.can_spawn,
                can_manage_cron=default.can_manage_cron,
                can_manage_fleet=default.can_manage_fleet,
                can_manage_teams=default.can_manage_teams,
                can_manage_projects=default.can_manage_projects,
                can_edit_agent_config=default.can_edit_agent_config,
                can_view_fleet_metrics=default.can_view_fleet_metrics,
                can_route_tasks=default.can_route_tasks,
                can_request_user_credentials=default.can_request_user_credentials,
                can_use_wallet=default.can_use_wallet,
                wallet_allowed_chains=default.wallet_allowed_chains,
                wallet_spend_limit_per_tx_usd=default.wallet_spend_limit_per_tx_usd,
                wallet_spend_limit_daily_usd=default.wallet_spend_limit_daily_usd,
                wallet_rate_limit_per_hour=default.wallet_rate_limit_per_hour,
                wallet_allowed_contracts=default.wallet_allowed_contracts,
            )
        return AgentPermissions(agent_id=agent_id)

    @staticmethod
    def _is_trusted(agent_id: str) -> bool:
        """Check if an agent ID is a trusted internal component."""
        return agent_id == "mesh"

    def can_message(self, from_agent: str, to_agent: str) -> bool:
        if self._is_trusted(from_agent):
            return True
        perms = self.get_permissions(from_agent)
        return "*" in perms.can_message or to_agent in perms.can_message

    def can_publish(self, agent_id: str, topic: str) -> bool:
        if self._is_trusted(agent_id):
            return True
        perms = self.get_permissions(agent_id)
        return "*" in perms.can_publish or topic in perms.can_publish

    def can_subscribe(self, agent_id: str, topic: str) -> bool:
        if self._is_trusted(agent_id):
            return True
        perms = self.get_permissions(agent_id)
        return "*" in perms.can_subscribe or topic in perms.can_subscribe

    def can_read_blackboard(self, agent_id: str, key: str) -> bool:
        if self._is_trusted(agent_id):
            return True
        # Fleet-global operator handoff data is not covered by normal
        # wildcard reads.  Workers can write into the operator mailbox, but
        # only the operator may inspect queued tasks and submitted outputs.
        if key.startswith("global/tasks/operator/"):
            return agent_id == "operator"
        if key.startswith("global/output/"):
            return (
                agent_id == "operator"
                or key.startswith(f"global/output/{agent_id}/")
            )
        perms = self.get_permissions(agent_id)
        return any(fnmatch.fnmatch(key, pattern) for pattern in perms.blackboard_read)

    def can_write_blackboard(self, agent_id: str, key: str) -> bool:
        if self._is_trusted(agent_id):
            return True
        # Operator inbox is a fleet-global mailbox: any registered agent can
        # hand off a task to the operator. The operator is the only reader
        # (its blackboard_read=["*"] already covers the read side).
        if key.startswith("global/tasks/operator/"):
            return True
        # Output namespace for operator handoffs is per-sender — only the
        # writing agent can place output under their own prefix.
        if key.startswith(f"global/output/{agent_id}/"):
            return True
        perms = self.get_permissions(agent_id)
        return any(fnmatch.fnmatch(key, pattern) for pattern in perms.blackboard_write)

    def can_use_browser(self, agent_id: str) -> bool:
        """Check if agent has browser access."""
        if self._is_trusted(agent_id):
            return True
        perms = self.get_permissions(agent_id)
        return perms.can_use_browser

    def can_use_internet(self, agent_id: str) -> bool:
        """Check if agent has external-internet access (HTTPS / web search).

        Returns True for trusted mesh-internal callers. Otherwise reads
        ``AgentPermissions.can_use_internet``. The operator gets True by
        default via ``_ensure_operator_agent``; worker agents default to
        False — their http_request / web_search tools are historically
        ungated, so this only affects agents whose runtime explicitly
        consults the flag (today: operator's ``_allowed_tools`` filter).
        """
        if self._is_trusted(agent_id):
            return True
        perms = self.get_permissions(agent_id)
        return perms.can_use_internet

    def can_browser_action(self, agent_id: str, action: str) -> bool:
        """Check if agent has permission for a specific browser action.

        Default-allow UX: any agent with ``can_use_browser=true`` gets ALL
        known browser actions unless their template explicitly narrows the
        set via ``browser_actions``. Operators who want restriction use
        an explicit list.

        ``browser_actions`` semantics:
           - ``None`` (default) → **all** current and future actions.
             Equivalent to ``["*"]``; kept as the default because "turn
             browser on, agent can browse" is the common expectation.
           - ``["*"]`` → all actions (explicit form).
           - Specific list → only those actions (opt-out restriction).
           - ``[]`` → no actions (equivalent to ``can_use_browser=False``).

        ``action`` must be a known action string. Input validation against
        :data:`KNOWN_BROWSER_ACTIONS` (catching typos) happens at the mesh
        gate; this method assumes the caller has validated and does not
        re-check, allowing it to also pass for future actions not yet in
        ``KNOWN_BROWSER_ACTIONS`` at the moment a specific grant is evaluated.
        """
        if self._is_trusted(agent_id):
            return True
        perms = self.get_permissions(agent_id)
        if not perms.can_use_browser:
            return False
        allowed = perms.browser_actions
        if allowed is None:
            return True  # default-allow: all known actions
        if "*" in allowed:
            return True
        return action in allowed

    def can_spawn(self, agent_id: str) -> bool:
        """Check if agent is allowed to spawn EPHEMERAL workers.

        Task 3 narrowed the semantics: this gates short-lived spawns
        (subagent / cron-triggered / template apply for transient
        helpers) only. Durable fleet operations (creating named agents,
        managing projects, editing config, viewing fleet metrics,
        routing tasks, requesting user credentials) live on dedicated
        control-plane checks below.
        """
        if self._is_trusted(agent_id):
            return True
        perms = self.get_permissions(agent_id)
        return perms.can_spawn

    def can_manage_cron(self, agent_id: str) -> bool:
        """Check if agent is allowed to create/manage cron jobs."""
        if self._is_trusted(agent_id):
            return True
        perms = self.get_permissions(agent_id)
        return perms.can_manage_cron

    # === Control-plane permissions (Task 3) ===

    def can_manage_fleet(self, agent_id: str) -> bool:
        """Check if agent is allowed to create/register durable named agents."""
        if self._is_trusted(agent_id):
            return True
        return self.get_permissions(agent_id).can_manage_fleet

    def can_manage_teams(self, agent_id: str) -> bool:
        """Check if agent is allowed to create/archive teams and manage membership."""
        if self._is_trusted(agent_id):
            return True
        return self.get_permissions(agent_id).can_manage_teams

    def can_edit_agent_config(self, agent_id: str) -> bool:
        """Check if agent is allowed to propose/confirm edits to other agents' config."""
        if self._is_trusted(agent_id):
            return True
        return self.get_permissions(agent_id).can_edit_agent_config

    def can_view_fleet_metrics(self, agent_id: str) -> bool:
        """Check if agent is allowed to read fleet-wide metrics endpoints."""
        if self._is_trusted(agent_id):
            return True
        return self.get_permissions(agent_id).can_view_fleet_metrics

    def can_route_tasks(self, agent_id: str) -> bool:
        """Check if agent is allowed to create durable task records (Task 6)."""
        if self._is_trusted(agent_id):
            return True
        return self.get_permissions(agent_id).can_route_tasks

    def can_request_user_credentials(self, agent_id: str) -> bool:
        """Check if agent is allowed to request credentials/login from the user."""
        if self._is_trusted(agent_id):
            return True
        return self.get_permissions(agent_id).can_request_user_credentials

    def can_use_api(self, agent_id: str, service: str) -> bool:
        if self._is_trusted(agent_id):
            return True
        perms = self.get_permissions(agent_id)
        return service in perms.allowed_apis

    def can_manage_vault(self, agent_id: str) -> bool:
        """Check if agent has any vault access (has allowed_credentials patterns)."""
        if self._is_trusted(agent_id):
            return True
        perms = self.get_permissions(agent_id)
        return bool(perms.allowed_credentials)

    def can_access_credential(self, agent_id: str, credential_name: str) -> bool:
        """Check if an agent can access a specific credential.

        Returns False for system credentials (always).
        Uses fnmatch against allowed_credentials patterns.

        Defense-in-depth: TWO independent system-tier checks block agent
        access regardless of the matched ``allowed_credentials`` patterns.

        1. L10 — LOADED TIER: any name that loaded under ``OPENLEGION_SYSTEM_``
           (registered via ``set_system_credential_names``) is denied even
           if its name doesn't match the provider-key shape. Closes the gap
           where a system secret with a non-conforming name
           (e.g. ``my_internal_token``) would otherwise be agent-resolvable.
        2. NAME SHAPE — ``is_system_credential()`` blocks provider-key-shaped
           names (``<provider>_api_key`` / ``_api_base``) regardless of tier,
           catching e.g. an ``add_credential()`` without ``system=True``.

        The primary separation is still at loading time, and
        ``resolve_credential()`` only returns agent-tier values.
        """
        if self._is_trusted(agent_id):
            return True
        if credential_name.lower() in self._system_credential_names:
            return False
        if is_system_credential(credential_name):
            return False
        perms = self.get_permissions(agent_id)
        patterns = perms.allowed_credentials
        if not patterns:
            return False
        lower_name = credential_name.lower()
        return any(fnmatch.fnmatch(lower_name, p.lower()) for p in patterns)

    def get_allowed_credentials(self, agent_id: str) -> list[str]:
        """Return the allowed_credentials patterns for an agent."""
        perms = self.get_permissions(agent_id)
        return perms.allowed_credentials

    # === Wallet permissions ===

    def can_use_wallet(self, agent_id: str) -> bool:
        """Check if agent has wallet access."""
        if self._is_trusted(agent_id):
            return True
        return self.get_permissions(agent_id).can_use_wallet

    def can_use_wallet_chain(self, agent_id: str, chain: str) -> bool:
        """Check if agent can transact on a specific chain."""
        if self._is_trusted(agent_id):
            return True
        perms = self.get_permissions(agent_id)
        if not perms.can_use_wallet:
            return False
        return "*" in perms.wallet_allowed_chains or chain in perms.wallet_allowed_chains

    def get_wallet_limits(self, agent_id: str) -> tuple[float, float, int]:
        """Return (per_tx_usd, daily_usd, rate_per_hour). 0 = use global default."""
        perms = self.get_permissions(agent_id)
        return (
            perms.wallet_spend_limit_per_tx_usd,
            perms.wallet_spend_limit_daily_usd,
            perms.wallet_rate_limit_per_hour,
        )

    def can_access_wallet_contract(self, agent_id: str, contract: str) -> bool:
        """Check if agent can interact with a specific contract address."""
        if self._is_trusted(agent_id):
            return True
        contracts = self.get_permissions(agent_id).wallet_allowed_contracts
        if not contracts:
            return True  # Empty = allow all
        return "*" in contracts or contract.lower() in [c.lower() for c in contracts]
