"""Fleet-level MCP connector catalog.

The single source of truth for which MCP servers exist and which agents
they are assigned to, persisted in ``config/connectors.json``::

    {
      "connectors": [
        {
          "name": "sqlite",
          "command": "mcp-server-sqlite",
          "args": ["--db", "/data/analytics.db"],
          "env": {"DB_KEY": "$CRED{analytics_db_key}"},
          "agents": ["researcher", "analyst"]
        },
        {
          "name": "fetch",
          "command": "mcp-server-fetch",
          "args": [],
          "env": null,
          "agents": ["*"]
        }
      ]
    }

Catalog order is operator-meaningful: it feeds the first-server-wins
tool-name conflict policy in :class:`src.agent.mcp_client.MCPClient`.

Concurrency: one reentrant lock held across every whole
load→mutate→save inside the mutators, so this store does not carry the
lost-update gap documented on ``permissions.json``
(``src/cli/config.py``). Saves are atomic (tempfile + ``os.replace``).

Failure policy: a missing or corrupt file loads as an EMPTY catalog
with an error log. Silently-absent fleet connectors are strictly safer
than blocking agent start.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path

from src.shared.types import CONNECTOR_ALL_AGENTS, MCPConnector
from src.shared.utils import setup_logging

logger = setup_logging("host.connectors")


class ConnectorStore:
    """Thread-safe catalog of :class:`MCPConnector` records."""

    def __init__(self, config_path: str = "config/connectors.json") -> None:
        self._path = Path(config_path)
        self._lock = threading.RLock()
        self._connectors: list[MCPConnector] = []
        # Agents whose running container predates the latest catalog
        # change that affects them. In-memory by design: a full mesh
        # reboot restarts every container, which makes the catalog
        # current again.
        self._dirty_agents: set[str] = set()
        self._load()

    # ── persistence ──────────────────────────────────────────────

    def _load(self) -> None:
        """(Re)load from disk. Fail-closed to an empty catalog on a
        missing or corrupt file — mirrors ``PermissionMatrix._load``."""
        with self._lock:
            self._connectors = []
            if not self._path.exists():
                return
            try:
                data = json.loads(self._path.read_text())
            except (json.JSONDecodeError, OSError) as e:
                logger.error(
                    "Connector catalog %s is corrupt or unreadable (%s); "
                    "loading an empty catalog", self._path, e,
                )
                return
            seen: set[str] = set()
            for raw in data.get("connectors", []) or []:
                try:
                    c = MCPConnector.model_validate(raw)
                except Exception as e:
                    name = raw.get("name") if isinstance(raw, dict) else "<?>"
                    logger.error(
                        "Dropping malformed connector %r from %s: %s",
                        name, self._path, e,
                    )
                    continue
                if c.name.lower() in seen:
                    logger.error(
                        "Dropping duplicate connector name %r from %s",
                        c.name, self._path,
                    )
                    continue
                seen.add(c.name.lower())
                self._connectors.append(c)

    def _save(self) -> None:
        """Atomic write (tempfile + ``os.replace``) — a concurrent
        reader sees the old or the new complete file, never a torn one.
        Caller holds the lock."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "connectors": [
                c.model_dump(exclude_none=False) for c in self._connectors
            ],
        }
        content = json.dumps(payload, indent=2) + "\n"
        fd, tmp_path = tempfile.mkstemp(dir=str(self._path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(content)
        except BaseException:
            try:
                os.close(fd)
            except OSError:
                pass  # already closed by fdopen
            Path(tmp_path).unlink(missing_ok=True)
            raise
        try:
            os.replace(tmp_path, self._path)
        except BaseException:
            Path(tmp_path).unlink(missing_ok=True)
            raise

    def reload(self) -> None:
        self._load()

    # ── catalog access ───────────────────────────────────────────

    def list(self) -> list[MCPConnector]:
        with self._lock:
            return list(self._connectors)

    def get(self, name: str) -> MCPConnector | None:
        lower = name.lower()
        with self._lock:
            for c in self._connectors:
                if c.name.lower() == lower:
                    return c
        return None

    def upsert(self, connector: MCPConnector) -> MCPConnector | None:
        """Insert or replace by case-insensitive name (insertion order
        preserved on replace). Returns the previous record, if any."""
        lower = connector.name.lower()
        with self._lock:
            previous: MCPConnector | None = None
            for i, c in enumerate(self._connectors):
                if c.name.lower() == lower:
                    previous = c
                    self._connectors[i] = connector
                    break
            else:
                self._connectors.append(connector)
            self._save()
            return previous

    def remove(self, name: str) -> MCPConnector | None:
        """Remove by case-insensitive name. Returns the removed record,
        or ``None`` if absent."""
        lower = name.lower()
        with self._lock:
            for i, c in enumerate(self._connectors):
                if c.name.lower() == lower:
                    removed = self._connectors.pop(i)
                    self._save()
                    return removed
        return None

    # ── assignment resolution ────────────────────────────────────

    def stdio_for_agent(self, agent_id: str) -> list[dict]:
        """``MCP_SERVERS``-shaped dicts for every connector assigned to
        this agent, in catalog order. This is the runtime input."""
        with self._lock:
            return [
                c.server_dict() for c in self._connectors
                if c.applies_to(agent_id)
            ]

    def assigned_agents(self, name: str, known_agents: list[str]) -> list[str]:
        """Concrete RUNNING agent ids a connector applies to: ``'*'``
        expanded against the live registry, explicit ids intersected
        with it (an id not currently registered has no container to
        restart). Display surfaces use the raw ``agents`` field."""
        c = self.get(name)
        if c is None:
            return []
        if CONNECTOR_ALL_AGENTS in c.agents:
            return sorted(known_agents)
        known = set(known_agents)
        return sorted(a for a in c.agents if a in known)

    # ── pending-restart tracking ─────────────────────────────────

    def mark_dirty(self, agents: list[str]) -> None:
        with self._lock:
            self._dirty_agents.update(agents)

    def mark_clean(self, agent_id: str) -> None:
        with self._lock:
            self._dirty_agents.discard(agent_id)

    def pending_restart(self) -> list[str]:
        with self._lock:
            return sorted(self._dirty_agents)
