"""Permission matrix enforcement for mesh operations.

Loaded from config/permissions.json.
Uses glob patterns for blackboard path matching.
Default policy: deny everything not explicitly allowed.
"""

from __future__ import annotations

import fnmatch
import json
from pathlib import Path

from src.shared.types import AgentPermissions
from src.shared.utils import setup_logging

logger = setup_logging("host.permissions")


class PermissionMatrix:
    """Enforces agent-level permissions for mesh operations."""

    def __init__(self, config_path: str = "config/permissions.json"):
        self.permissions: dict[str, AgentPermissions] = {}
        self._load(config_path)

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
            )
        return AgentPermissions(agent_id=agent_id)

    def can_message(self, from_agent: str, to_agent: str) -> bool:
        if from_agent in ("mesh", "orchestrator"):
            return True
        perms = self.get_permissions(from_agent)
        return "*" in perms.can_message or to_agent in perms.can_message

    def can_publish(self, agent_id: str, topic: str) -> bool:
        if agent_id in ("mesh", "orchestrator"):
            return True
        perms = self.get_permissions(agent_id)
        return "*" in perms.can_publish or topic in perms.can_publish

    def can_subscribe(self, agent_id: str, topic: str) -> bool:
        if agent_id in ("mesh", "orchestrator"):
            return True
        perms = self.get_permissions(agent_id)
        return "*" in perms.can_subscribe or topic in perms.can_subscribe

    def can_read_blackboard(self, agent_id: str, key: str) -> bool:
        if agent_id in ("mesh", "orchestrator"):
            return True
        perms = self.get_permissions(agent_id)
        return any(fnmatch.fnmatch(key, pattern) for pattern in perms.blackboard_read)

    def can_write_blackboard(self, agent_id: str, key: str) -> bool:
        if agent_id in ("mesh", "orchestrator"):
            return True
        perms = self.get_permissions(agent_id)
        return any(fnmatch.fnmatch(key, pattern) for pattern in perms.blackboard_write)

    def can_use_api(self, agent_id: str, service: str) -> bool:
        if agent_id in ("mesh", "orchestrator"):
            return True
        perms = self.get_permissions(agent_id)
        return service in perms.allowed_apis
