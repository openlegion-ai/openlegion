"""Fleet-level MCP connector catalog.

The single source of truth for which MCP servers exist and which agents
they are assigned to, persisted in ``config/connectors.json``::

    {
      "connectors": [
        {
          "name": "sqlite",
          "transport": "stdio",
          "command": "mcp-server-sqlite",
          "args": ["--db", "/data/analytics.db"],
          "env": {"DB_KEY": "$CRED{analytics_db_key}"},
          "agents": ["researcher", "analyst"]
        },
        {
          "name": "fetch",
          "transport": "stdio",
          "command": "mcp-server-fetch",
          "args": [],
          "env": null,
          "agents": ["*"]
        },
        {
          "name": "linear",
          "transport": "http",
          "url": "https://mcp.linear.app/mcp",
          "auth": {"kind": "bearer", "cred": "linear_token"},
          "agents": ["*"]
        }
      ]
    }

Two transports: ``stdio`` records become in-container subprocesses via
``MCP_SERVERS`` (``snapshot_for_agent`` — stdio ONLY, by design);
``http`` records are mesh-gateway-proxied (``http_for_agent``) and
never enter a container.

Catalog order is operator-meaningful: it feeds the first-server-wins
tool-name conflict policy in :class:`src.agent.mcp_client.MCPClient`.

Concurrency: one reentrant lock held across every whole
load→mutate→save inside the mutators, so this store does not carry the
lost-update gap documented on ``permissions.json``
(``src/cli/config.py``). Saves are atomic via
:func:`src.shared.utils.atomic_write_text`.

Pending-restart is a pure derivation, not mutable UI state: every
catalog change bumps a monotonic generation and stamps the affected
agents (``_touch_gen``); the runtime snapshots the catalog generation
together with the server list at container-env build time and records
it after a successful start (``_start_gen``). An agent is pending
restart iff it was touched after the generation its running container
was built from — which makes the tracking immune to the
edit-lands-during-container-build race (the late edit's touch
generation is greater than the snapshot generation, so the agent stays
correctly dirty).

External edits: read paths re-load the file when its mtime/size
changes, so a hand-edited ``connectors.json`` (headless/SSH operators)
takes effect at the next agent start without a mesh reboot; the reload
touches every agent assigned before or after so pending-restart
surfaces the change.

Failure policy: a missing or corrupt file loads as an EMPTY catalog
with an error log. Silently-absent fleet connectors are strictly safer
than blocking agent start.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from src.shared.types import CONNECTOR_ADAPTER, HttpConnector, MCPConnector
from src.shared.utils import atomic_write_text, setup_logging

logger = setup_logging("host.connectors")


class ConnectorStore:
    """Thread-safe catalog of connector records (stdio
    :class:`MCPConnector` and remote :class:`HttpConnector`)."""

    def __init__(self, config_path: str = "config/connectors.json") -> None:
        self._path = Path(config_path)
        self._lock = threading.RLock()
        self._connectors: list[MCPConnector | HttpConnector] = []
        # (st_mtime_ns, st_size) of the file as last loaded; None = no file.
        self._loaded_stat: tuple[int, int] | None = None
        # Monotonic catalog generation. Bumped on every mutation
        # (dashboard write or detected external edit).
        self._generation = 0
        # Per-agent generation stamps. In-memory by design: a full mesh
        # reboot restarts every container, making the catalog current.
        self._touch_gen: dict[str, int] = {}
        self._start_gen: dict[str, int] = {}
        self._load()

    # ── persistence ──────────────────────────────────────────────

    def _stat(self) -> tuple[int, int] | None:
        try:
            st = self._path.stat()
            return (st.st_mtime_ns, st.st_size)
        except OSError:
            return None

    def _load(self) -> None:
        """(Re)load from disk. Fail-closed to an empty catalog on a
        missing or corrupt file — mirrors ``PermissionMatrix._load``.
        Caller holds the lock (or is ``__init__``)."""
        self._loaded_stat = self._stat()
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
                c = CONNECTOR_ADAPTER.validate_python(raw)
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

    def _maybe_reload(self) -> None:
        """Pick up an external edit to the file (hand-edited catalog on
        a headless deploy). Touches every agent assigned before or
        after the reload so pending-restart reflects the change.
        Caller holds the lock."""
        current = self._stat()
        if current == self._loaded_stat:
            return
        before = {a for c in self._connectors for a in c.agents}
        self._load()
        after = {a for c in self._connectors for a in c.agents}
        touched = before | after
        if touched:
            self._touch(touched)
        logger.info(
            "Connector catalog %s changed on disk; reloaded (%d connectors)",
            self._path, len(self._connectors),
        )

    def _save(self) -> None:
        """Atomic write. Caller holds the lock."""
        payload = {
            "connectors": [
                c.model_dump(exclude_none=False) for c in self._connectors
            ],
        }
        atomic_write_text(self._path, json.dumps(payload, indent=2) + "\n")
        self._loaded_stat = self._stat()

    # ── catalog access ───────────────────────────────────────────

    def list(self) -> list[MCPConnector | HttpConnector]:
        with self._lock:
            self._maybe_reload()
            return list(self._connectors)

    def get(self, name: str) -> MCPConnector | HttpConnector | None:
        lower = name.lower()
        with self._lock:
            self._maybe_reload()
            for c in self._connectors:
                if c.name.lower() == lower:
                    return c
        return None

    def upsert(self, connector: MCPConnector | HttpConnector) -> bool:
        """Insert or replace by case-insensitive name (insertion order
        preserved on replace).

        Returns whether the edit is **restart-relevant** (the dirty
        matrix, plan D12): False only when an existing http connector
        changed nothing but ``auth`` — remote auth is resolved per call
        on the mesh, so the caller must skip ``mark_dirty`` / the
        restart prompt and instead invalidate the gateway's discovery
        cache. Every other mutation (any stdio change, http
        URL/assignment change, insert) returns True.
        """
        lower = connector.name.lower()
        with self._lock:
            self._maybe_reload()
            previous: MCPConnector | HttpConnector | None = None
            for i, c in enumerate(self._connectors):
                if c.name.lower() == lower:
                    previous = c
                    self._connectors[i] = connector
                    break
            else:
                self._connectors.append(connector)
            self._save()
            return not self._auth_only_change(previous, connector)

    @staticmethod
    def _auth_only_change(
        old: MCPConnector | HttpConnector | None,
        new: MCPConnector | HttpConnector,
    ) -> bool:
        """True when ``old`` → ``new`` is an http-connector edit that
        touches nothing outside ``auth`` (a no-op also qualifies — not
        touching generations for it is equally correct)."""
        if not isinstance(old, HttpConnector) or not isinstance(new, HttpConnector):
            return False
        # agents compared as a SET: assignment is set-semantic (only
        # connector order is meaningful, not agent-id order), and the
        # dashboard sorts the list it sends — an order-only diff against
        # a hand-edited file must not turn an auth rotation into a
        # spurious restart prompt.
        return (
            old.model_dump(exclude={"auth", "agents"})
            == new.model_dump(exclude={"auth", "agents"})
            and set(old.agents) == set(new.agents)
        )

    def remove(self, name: str) -> bool:
        """Remove by case-insensitive name. Returns True if present."""
        lower = name.lower()
        with self._lock:
            self._maybe_reload()
            for i, c in enumerate(self._connectors):
                if c.name.lower() == lower:
                    del self._connectors[i]
                    self._save()
                    return True
        return False

    def remove_agent(self, agent_id: str) -> None:
        """Agent-deletion lifecycle hook: strip ``agent_id`` from every
        explicit assignment and drop its generation stamps. Without
        this, a future agent recreated under the same name would
        silently inherit the deleted agent's connectors (and their
        ``$CRED``-bearing env). ``"*"`` assignments are untouched —
        applying to whatever agents exist is their meaning.
        """
        with self._lock:
            self._maybe_reload()
            changed = False
            for i, c in enumerate(self._connectors):
                if agent_id in c.agents:
                    self._connectors[i] = c.model_copy(
                        update={"agents": [a for a in c.agents if a != agent_id]},
                    )
                    changed = True
            if changed:
                self._save()
            self._touch_gen.pop(agent_id, None)
            self._start_gen.pop(agent_id, None)

    # ── assignment resolution ────────────────────────────────────

    def snapshot_for_agent(self, agent_id: str) -> tuple[list[dict], int]:
        """``MCP_SERVERS``-shaped dicts for every **stdio** connector
        assigned to this agent (catalog order) plus the catalog
        generation they were read at — taken under one lock so the pair
        is consistent. The runtime passes the generation back via
        :meth:`record_agent_start` after a successful start.

        The stdio filter is a security boundary, not a convenience:
        an :class:`HttpConnector` here would serialize its ``auth``
        into the container env — the exact exposure the mesh gateway
        exists to prevent. The generation still covers the WHOLE
        catalog, so pending-restart works for http-only edits too.
        """
        with self._lock:
            self._maybe_reload()
            servers = [
                c.server_dict() for c in self._connectors
                if isinstance(c, MCPConnector) and c.applies_to(agent_id)
            ]
            return servers, self._generation

    def stdio_for_agent(self, agent_id: str) -> list[dict]:
        """Server dicts only — convenience for display surfaces that
        don't participate in pending-restart tracking."""
        return self.snapshot_for_agent(agent_id)[0]

    def http_for_agent(self, agent_id: str) -> list[HttpConnector]:
        """Remote connectors assigned to this agent, catalog order.
        The mesh gateway's input; never serialized into container
        env."""
        with self._lock:
            self._maybe_reload()
            return [
                c for c in self._connectors
                if isinstance(c, HttpConnector) and c.applies_to(agent_id)
            ]

    def generation(self) -> int:
        """Current catalog generation — the gateway's tools-cache key.
        Auth-only edits deliberately don't bump it (the dashboard calls
        ``MCPGateway.invalidate`` for those instead)."""
        with self._lock:
            self._maybe_reload()
            return self._generation

    # ── pending-restart derivation ───────────────────────────────

    def _touch(self, agents: set[str]) -> None:
        """Caller holds the lock."""
        self._generation += 1
        for a in agents:
            self._touch_gen[a] = self._generation

    def mark_dirty(self, agents: list[str]) -> None:
        """Record that a catalog change affects ``agents`` (called by
        the dashboard after a write)."""
        if not agents:
            return
        with self._lock:
            self._touch(set(agents))

    def record_agent_start(self, agent_id: str, generation: int) -> None:
        """Record the catalog generation a successfully started
        container was built from (called by the runtime). Monotonic:
        never lowers an existing stamp."""
        with self._lock:
            if generation > self._start_gen.get(agent_id, -1):
                self._start_gen[agent_id] = generation

    def pending_restart(self) -> list[str]:
        """Agents whose last catalog touch postdates the generation
        their running container was built from. Pure derivation — an
        edit landing mid-container-build keeps the agent correctly
        dirty because its touch generation exceeds the build snapshot.
        May include agents that no longer exist; surfaces filter
        against the live registry."""
        with self._lock:
            return sorted(
                a for a, g in self._touch_gen.items()
                if g > self._start_gen.get(a, 0)
            )
