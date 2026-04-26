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
    # Phase 6 §9.1 read-only network inspection.
    "inspect_requests",
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
        self._load(config_path)

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
        with open(path) as f:
            data = json.load(f)
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
                can_spawn=default.can_spawn,
                can_manage_cron=default.can_manage_cron,
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
        perms = self.get_permissions(agent_id)
        return any(fnmatch.fnmatch(key, pattern) for pattern in perms.blackboard_read)

    def can_write_blackboard(self, agent_id: str, key: str) -> bool:
        if self._is_trusted(agent_id):
            return True
        perms = self.get_permissions(agent_id)
        return any(fnmatch.fnmatch(key, pattern) for pattern in perms.blackboard_write)

    def can_use_browser(self, agent_id: str) -> bool:
        """Check if agent has browser access."""
        if self._is_trusted(agent_id):
            return True
        perms = self.get_permissions(agent_id)
        return perms.can_use_browser

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
        """Check if agent is allowed to spawn ephemeral agents."""
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

        Defense-in-depth: The ``is_system_credential()`` check here blocks
        agent access to provider-key-shaped names regardless of which tier
        they landed in.  The primary separation is at loading time
        (``OPENLEGION_SYSTEM_`` → ``system_credentials``, ``OPENLEGION_CRED_``
        → ``credentials``), and ``resolve_credential()`` only returns
        agent-tier values.  This check catches edge cases where a
        provider-key name might appear in the agent-tier dict (e.g. via
        ``add_credential()`` without ``system=True``).
        """
        if self._is_trusted(agent_id):
            return True
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
