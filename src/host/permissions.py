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
            return
        with open(path) as f:
            data = json.load(f)
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
            )
        return AgentPermissions(agent_id=agent_id)

    @staticmethod
    def _is_trusted(agent_id: str) -> bool:
        """Check if an agent ID is a trusted internal component."""
        return agent_id in ("mesh", "orchestrator")

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
