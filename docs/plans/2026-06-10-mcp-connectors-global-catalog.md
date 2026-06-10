# MCP Connectors — Global Catalog, Fleet Assignment, Remote Transport & OAuth

**Date:** 2026-06-10
**Status:** Proposed
**Scope:** Promote MCP from a per-agent config field to a fleet-level **Connectors** catalog
(connect once, enable for all agents or specific agents — the Skills-page interaction model),
housed on the existing Integrations settings page (relabelled "Connectors"). Phase 2 adds a
**remote (HTTP) transport with a mesh-side gateway** so connector credentials stay in the vault
and are proxied — never entering the agent container. Phase 3 adds the Claude-style
**"paste URL → Connect → OAuth redirect"** flow on top of the already-built Option-B OAuth
machinery.
**Related:**
- `docs/mcp.md` (current per-agent stdio MCP design)
- `docs/plans/2026-06-04-oauth-integrations-connect-flow.md` (Option-B OAuth: state store,
  callback, `store_connection`, refresh-on-resolve — Phase 3 reuses all of it)
- `docs/plans/2026-06-04-integrations-oauth-strategy.md` (§4 already names MCP connectors as
  the long-tail integration strategy; this plan is that section made concrete)
- `docs/plans/2026-05-31-tools-skills-rename-and-skill-packs.md` (the fleet/per-agent
  assignment pattern this plan mirrors)

---

## 1. Problem

Three gaps, in increasing order of architectural weight:

1. **No global enablement.** `mcp_servers` lives on each agent's entry in `config/agents.yaml`
   (`AgentConfig.mcp_servers`, `src/shared/types.py:526`). Connecting the same Linear MCP to five
   agents means entering the same config five times, keeping five copies in sync, and rotating a
   credential in five places. Skills already solved this shape (`fleet_skills` ∪ per-agent
   `allowed_skills`, `src/host/permissions.py:186-196`); MCP has no equivalent.

2. **stdio-only.** `MCPServerConfig` (`src/shared/types.py:414-450`) is `command`/`args`/`env` —
   a subprocess inside the agent container. There is no way to point at a remote MCP server URL
   (`https://mcp.linear.app/mcp`), which is where the ecosystem has standardized. Worse, the
   default agent image is Python-only, so the large npm-based server catalog doesn't run at all
   (`docs/mcp.md` "Node.js MCP servers").

3. **Credentials are vault-*stored* but not vault-*proxied*.** Config holds `$CRED{name}`
   handles (good: never plaintext on disk, masked on GET, redacted in audit), but
   `RuntimeBackend._build_mcp_servers_env` (`src/host/runtime.py:102-168`) resolves them to
   plaintext at container start and ships them in the `MCP_SERVERS` env var. This is *inherent*
   to stdio (the subprocess needs the secret) and is a documented asymmetry vs `http_tool`
   (`docs/mcp.md` §security) — but it means every MCP credential is exposed to the agent
   process. Only a remote transport with a mesh-side session can close this.

## 2. Decisions

| # | Decision | Rationale |
|---|---|---|
| **D1** | Relabel the Integrations tab **"Connectors"**; keep the tab **ID `integrations`** | Labels are free, IDs are frozen — the exact pattern already used for `fleet`→"Teams" (Known Constraint #5). The label changes in one place (`app.js:508`); deep-links and persisted prefs keep working. "Connectors" is the industry term (Claude's naming) and honestly covers what the page already holds: OAuth connections, channels, API keys. |
| **D2** | New file **`config/connectors.json`** + new `ConnectorStore`, not more keys in `permissions.json` | Connectors are *definitions + assignment* (config), not ACLs. `permissions.json` already carries a documented lost-update race (`src/cli/config.py:269-277`); a new store holds its lock across the whole load→mutate→save from day one instead of inheriting the gap. Fail-closed load mirrors `PermissionMatrix._load`. |
| **D3** | Assignment lives **on the connector**: `agents: ["*"]` or `agents: ["researcher", "writer"]` — one surface, one source of truth | The user's mental model is "connect a service, choose who gets it" (the Claude connectors model). Skills needed the two-surface union (fleet list + per-agent list edited on the agent) because skills are ambient discoverability; a connector is a deliberate grant. One control on one card is simpler to build, audit, and explain. `"*"` matches the existing glob convention in `AgentPermissions.can_message`. |
| **D4** | Merge fleet connectors into the agent's server list at **one choke point in `RuntimeBackend`** | Six call sites pass `mcp_servers` into `start_agent` (`cli/repl.py:478,1429`, `cli/runtime.py:627,659`, `dashboard/server.py:3162,6631`, plus `host/health.py:603` via the registry). Merging inside the backend means every start/restart path — CLI boot, dashboard restart, health-watchdog restart — inherits the behavior with zero call-site edits and zero chance of a forgotten path. |
| **D5** | Name-collision policy: **agent-local wins** over a fleet connector with the same (case-insensitive) name; log a warning | More-specific beats more-general; deterministic; lets an operator override one agent's variant of a fleet connector without touching the catalog. |
| **D6** | stdio connectors keep today's resolve-into-container-env semantics; **remote connectors are the proxied path**, and the UI labels the difference honestly | We do not pretend stdio can be proxied — it can't (the subprocess runs in the container). The Add-Connector UI says: *Remote — credentials stay in the vault; calls proxied by the mesh* vs *Local — runs inside the agent container; credentials are exposed to that agent*. Security-relevant difference surfaced at decision time, not buried in docs. |
| **D7** | Restarts are **explicit and confirmed, never automatic mass-restarts** | `MCP_SERVERS` is startup-only. A catalog edit returns the affected agent list; the UI offers "Restart now / Later" with pending-restart badges until bounced. A fleet-wide toggle silently bouncing 20 agents mid-task is not acceptable. |
| **D8** | Connector management stays **human-only** (dashboard); no operator-agent tool to add/remove connectors | Consistent with the decision already recorded in `docs/mcp.md` ("operator-requested MCP setup" deferred): chat-driven installation of arbitrary tool servers is a prompt-injection-shaped hole. The request-card pattern (agent requests, human approves) remains the tracked follow-up. |

## 3. Phasing

| Phase | Ships | Depends on |
|---|---|---|
| **1** | Global catalog + assignment + Connectors UI + restart orchestration (stdio transport, all existing plumbing) | nothing |
| **2** | `http` transport + mesh-side `MCPGateway` — vault-held, proxied credentials | Phase 1 |
| **3** | OAuth 2.1 connect flow (discovery + DCR + PKCE) for remote connectors | Phase 2 + existing Option-B OAuth machinery |

Each phase is independently shippable and independently valuable. Phase 1 is pure recombination
of proven pieces; the architectural risk is concentrated in Phase 2 and isolated to one new
module.

---

## 4. Phase 1 — Global catalog & fleet assignment

### 4.1 Data model — `src/shared/types.py`

A discriminated union keeps stdio validation byte-identical to today (`StdioConnector` inherits
every `MCPServerConfig` validator) and gives Phase 2 a clean slot:

```python
# Sentinel for "assigned to every agent" in Connector.agents. Matches the
# glob convention used by AgentPermissions.can_message.
CONNECTOR_ALL_AGENTS = "*"


def _validate_connector_agents(cls, v: list[str]) -> list[str]:
    """Shared `agents` validator: '*' must be the sole element; ids deduped,
    order preserved. Defined before the models so both can bind it via
    ``field_validator("agents")(classmethod(_validate_connector_agents))``.
    """
    if CONNECTOR_ALL_AGENTS in v and len(v) > 1:
        raise ValueError("agents: '*' (all agents) cannot be combined with explicit ids")
    seen: set[str] = set()
    out: list[str] = []
    for a in v:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


class ConnectorAuth(BaseModel):
    """Auth binding for a remote (http) connector — Phase 2/3.

    The secret itself always lives in the vault; this only names it.
    ``bearer`` → vault credential (``$CRED``-tier name) injected as
    ``Authorization: Bearer <value>`` by the mesh gateway.
    ``oauth`` → vault connection key (refresh-on-resolve, same machinery
    as the Google Option-B flow).
    """
    model_config = {"extra": "forbid"}

    kind: Literal["none", "bearer", "oauth"] = "none"
    cred: str | None = None        # required iff kind == "bearer"
    connection: str | None = None  # set by the connect flow iff kind == "oauth"

    @model_validator(mode="after")
    def _kind_fields(self) -> "ConnectorAuth":
        if self.kind == "bearer" and not self.cred:
            raise ValueError("auth.kind='bearer' requires auth.cred")
        return self


class StdioConnector(MCPServerConfig):
    """Fleet-level stdio MCP server: an MCPServerConfig plus assignment.

    Inherits every MCPServerConfig validator (name pattern, no $CRED in
    command, args/env caps) unchanged.
    """
    model_config = {"extra": "forbid"}

    transport: Literal["stdio"] = "stdio"
    agents: list[str] = Field(default_factory=list, max_length=128)

    _validate_agents = field_validator("agents")(
        classmethod(_validate_connector_agents),
    )


class HttpConnector(BaseModel):
    """Fleet-level remote MCP server — Phase 2. Mesh-gateway only; never
    serialized into MCP_SERVERS, so its credentials never enter a container."""
    model_config = {"extra": "forbid"}

    transport: Literal["http"]
    name: str = Field(min_length=1, max_length=64, pattern=MCP_SERVER_NAME_RE_PATTERN)
    url: str = Field(min_length=1, max_length=512)
    auth: ConnectorAuth = Field(default_factory=ConnectorAuth)
    agents: list[str] = Field(default_factory=list, max_length=128)

    _validate_agents = field_validator("agents")(
        classmethod(_validate_connector_agents),
    )

    @field_validator("url")
    @classmethod
    def _https_only(cls, v: str) -> str:
        from urllib.parse import urlparse
        p = urlparse(v)
        if p.scheme == "https":
            return v
        # Self-hosted/dev MCP on the mesh host itself is legitimate.
        if p.scheme == "http" and p.hostname in ("localhost", "127.0.0.1"):
            return v
        raise ValueError("Connector URL must be https:// (http:// allowed for localhost only)")


Connector = Annotated[StdioConnector | HttpConnector, Field(discriminator="transport")]
```

Notes:
- `transport: Literal["stdio"] = "stdio"` lets pre-Phase-2 files (and hand-written ones) omit
  the key; Pydantic v2 supports defaulted discriminators.
- `MCPServerConfig` and `AgentConfig.mcp_servers` are **untouched**. No migration, no shim.

### 4.2 Storage — `config/connectors.json`

```json
{
  "connectors": [
    {
      "name": "linear",
      "transport": "http",
      "url": "https://mcp.linear.app/mcp",
      "auth": { "kind": "oauth", "connection": "mcp_linear" },
      "agents": ["*"]
    },
    {
      "name": "sqlite",
      "transport": "stdio",
      "command": "mcp-server-sqlite",
      "args": ["--db", "/data/analytics.db"],
      "env": { "DB_KEY": "$CRED{analytics_db_key}" },
      "agents": ["researcher", "analyst"]
    }
  ]
}
```

A **list**, not a name-keyed dict: order is operator-meaningful because it feeds the existing
first-server-wins tool-name conflict policy in `MCPClient.start()`. Name uniqueness
(case-insensitive) is enforced at the store layer, mirroring `AgentConfig._no_duplicate_mcp_names`.

### 4.3 `ConnectorStore` — `src/host/connectors.py` (new, ~150 lines)

```python
class ConnectorStore:
    """Fleet-level MCP connector catalog backed by config/connectors.json.

    Concurrency: one reentrant lock held across the whole load→mutate→save
    inside every mutator — this store does NOT inherit the lost-update gap
    documented on permissions.json (src/cli/config.py:269-277). Saves are
    atomic (tempfile + os.replace), identical to _save_permissions.

    Failure policy: missing or corrupt file loads as an EMPTY catalog with
    an error log. Fleet connectors silently absent is strictly safer than
    blocking agent start; agent-local mcp_servers are unaffected either way.
    """

    def __init__(self, config_path: str = "config/connectors.json") -> None: ...

    def list(self) -> list[StdioConnector | HttpConnector]: ...
    def get(self, name: str) -> StdioConnector | HttpConnector | None: ...

    def upsert(self, connector: StdioConnector | HttpConnector) -> None:
        """Insert or replace by case-insensitive name. Lock held throughout."""

    def remove(self, name: str) -> bool: ...

    def assigned_agents(self, name: str, known_agents: list[str]) -> list[str]:
        """Concrete agent ids a connector applies to ('*' expanded against
        the live registry). Used to compute restart-affected sets."""

    def stdio_for_agent(self, agent_id: str) -> list[dict]:
        """MCP_SERVERS-shaped dicts (name/command/args/env only — transport
        and agents stripped) for every stdio connector assigned to this
        agent, in catalog order. This is the runtime merge input."""

    def http_for_agent(self, agent_id: str) -> list[HttpConnector]:
        """Phase 2: remote connectors assigned to this agent."""

    def reload(self) -> None: ...
```

Assignment check is trivial and inlined: `"*" in c.agents or agent_id in c.agents`.

### 4.4 Runtime merge — `src/host/runtime.py`

Mirror the existing `set_credential_resolver` wiring pattern exactly (class-level `None` default
so `__new__`-style tests stay safe, `src/host/runtime.py:63-71`):

```python
class RuntimeBackend(abc.ABC):
    _vault: CredentialVault | None = None
    _permissions: PermissionMatrix | None = None
    _connectors: "ConnectorStore | None" = None          # NEW

    def set_connector_store(self, store: "ConnectorStore | None") -> None:
        """Wire the fleet connector catalog. Until called, behavior is
        byte-identical to today (agent-local mcp_servers only)."""
        self._connectors = store

    def _effective_mcp_servers(
        self, agent_id: str, mcp_servers: list[dict] | None,
    ) -> list[dict] | None:
        """Agent-local servers ∪ assigned fleet stdio connectors.

        Agent-local wins on case-insensitive name collision (D5). A store
        read error degrades to local-only — a corrupt catalog must never
        block an agent from starting.
        """
        local = list(mcp_servers or [])
        if self._connectors is None:
            return local or None
        try:
            fleet = self._connectors.stdio_for_agent(agent_id)
        except Exception:
            logger.exception(
                "Connector catalog unreadable; starting %r with agent-local "
                "MCP servers only", agent_id,
            )
            fleet = []
        local_names = {str(s.get("name", "")).lower() for s in local}
        for s in fleet:
            if s["name"].lower() in local_names:
                logger.warning(
                    "Agent-local MCP server %r overrides fleet connector of "
                    "the same name for agent %r", s["name"], agent_id,
                )
                continue
            local.append(s)
        return local or None
```

Then a two-line change at each of the two (and only two) `MCP_SERVERS` construction sites —
`DockerBackend._start_agent_container` (`runtime.py:423-426`) and the `SandboxBackend`
equivalent (`runtime.py:1106-1109`). **Bind the merged list to a new name** — do NOT reassign
the `mcp_servers` parameter, because the same variable is stored into the runtime registry a few
lines down (`agent_info["mcp_servers"]`, `runtime.py:548`) and must stay agent-local:

```python
effective_mcp = self._effective_mcp_servers(agent_id, mcp_servers)
if effective_mcp:
    environment["MCP_SERVERS"] = self._build_mcp_servers_env(
        effective_mcp, agent_id=agent_id,
    )
```

Why this is correct everywhere with no other edits:
- The registry entry `agent_info["mcp_servers"]` (`runtime.py:548`) keeps storing the
  **agent-local** list (this is why the merged list gets its own variable above — reassigning
  the parameter would leak merged fleet entries into the registry, and a later health-watchdog
  restart would then treat them as agent-local: stale against catalog edits, spurious
  collision warnings, and they'd shadow the updated catalog under the D5 local-wins rule).
  Health-watchdog restarts (`host/health.py:603`) pass the agent-local list back through
  `start_agent`, where the merge re-applies **fresh** — so a catalog edit made between crash and
  auto-restart is picked up, and fleet servers are never double-merged. Pin this with a test:
  registry contents equal the pre-merge input.
- `$CRED{...}` resolution is downstream of the merge, so fleet-connector credentials flow
  through the existing `resolve_cred_handles` path with the existing per-agent
  `allowed_credentials` permission gate and the existing loud-failure semantics
  (`runtime.py:131-139`).

**Wiring:** instantiate `ConnectorStore` once in the runtime bootstrap and call
`set_connector_store` at both existing `set_credential_resolver` call sites
(`src/cli/runtime.py:381` and `:648`); pass the same instance into the dashboard router factory
alongside `permissions` (same dependency-injection style — no new module-level globals,
Known Constraint #8).

### 4.5 Dashboard API — `src/dashboard/server.py`

Three endpoints plus one batch-restart helper, in the existing closure style. All state-changing
routes are covered by the existing `X-Requested-With` CSRF middleware; all writes are
audit-logged with env values redacted via the existing `_redact_mcp_env_for_audit`
(`dashboard/server.py:235-259`).

```
GET    /api/connectors
       → {"connectors": [{...connector fields, env masked to env_keys,
                          assigned_agents: [...expanded...],
                          pending_restart: [...agent ids not yet bounced...]}]}

PUT    /api/connectors/{name}
       Body: full connector record (upsert; name in path is authoritative).
       Validation, in order:
         1. Pydantic Connector union → structured 400 with per-field errors
            (same shape as the existing mcp_servers PUT, server.py:2929-2945,
            so the UI reuses its inline-error rendering).
         2. stdio env preserve-or-replace: omitted `env` key preserves the
            persisted env (matched by name) — the exact GET→edit→PUT contract
            the agent-config editor already implements (server.py:2890-2922).
         3. $CRED{...} handles: for each handle, (a) vault has the credential,
            (b) every CURRENTLY-assigned agent passes
            permissions.can_access_credential. Failures → 400 listing the
            blocked agents and the fix ("grant analytics_db_key to
            'researcher' or narrow the assignment"). Agents created later
            under a '*' assignment fail at restart through the existing loud
            path (red status dot + captured stderr) — documented, not silent.
       → {"connector": {...}, "affected_agents": [...], "restart_required": true}

DELETE /api/connectors/{name}
       → {"removed": true, "affected_agents": [...]}

POST   /api/agents/restart-batch
       Body: {"agents": ["a", "b", ...]}  (cap: 32 per call)
       Sequentially runs the existing single-agent restart logic
       (api_restart_agent, server.py:3076+) so every agent gets the same
       event choreography (agent_restarting → agent_restarted /
       restart_failed) the SPA already renders.
       → {"results": {"a": "ok", "b": "failed: <reason>"}}
```

`affected_agents` = union of the assignment **before and after** the edit (an agent removed from
a connector needs a bounce to *lose* the tools, too). Each successful write emits
`_emit_config_changed("connectors")`.

Pending-restart tracking: the store keeps an in-memory `dirty_agents: set[str]` updated on every
write and cleared per-agent on successful restart (hook the existing restart success path). On
mesh reboot it resets to empty — correct, because a full reboot restarts every container anyway.

### 4.6 UI — Connectors page & per-agent view

**Tab rename** — `src/dashboard/static/js/app.js:508`:

```js
{ id: 'integrations', label: 'Connectors' },
```

ID untouched (D1). Update the page's `<h2>`/intro copy in `index.html`; existing sections
(Connected Services, Channels, Developer & API) become subsections of the relabelled page.

**New first section: "MCP Connectors"** (above Connected Services, since it's the page's new
headline capability). Follows the established card-list grammar of the Skills catalog
(`index.html:5302-5353`):

- **Card per connector:** name · transport badge (`Local` slate / `Remote` indigo) · assignment
  summary ("All agents" / "3 agents") · aggregated status dots reusing the existing per-server
  capabilities surface (green running / red failed / amber *pending restart*) · **Edit** /
  **Remove** on hover (the inline-webhooks hover pattern already used by the per-agent MCP
  editor).
- **Empty state:** "No connectors yet. Connect an MCP server once and enable it for your whole
  fleet." + **Add connector**.
- **Add/Edit modal**, two-step:
  1. *Type*: two radio cards —
     **Remote server (recommended)** "Credentials stay in the vault; calls are proxied by the
     mesh." *(disabled with a "Phase 2" tooltip until the gateway ships)* /
     **Local command** "Runs inside each agent's container; credentials are exposed to that
     agent."
  2. *Details*: Local reuses the existing agent-config MCP form components verbatim — name,
     command, args rows, env rows with the **Credential | Plain text** toggle, credential
     dropdown, and the looks-like-a-secret inline warning (`index.html:3402-3450`). Plus the new
     **Assignment** control: radio **All agents** / **Specific agents** with a checkbox list of
     the fleet. Remote (Phase 2): URL + auth picker (None / API key → credential dropdown /
     OAuth → Connect button, Phase 3) + the same Assignment control.
- **Save flow (D7):** on a response with `affected_agents`, show a confirm dialog —
  "This change affects **N agents** and takes effect after restart." **[Restart now]**
  (calls `restart-batch`, per-agent progress via the existing restart events)
  **[Later]** (amber *pending restart* chip on the connector card and on each affected agent's
  row until bounced).

**Per-agent view** (agent identity → existing MCP Servers section, `index.html:3104-3126`):
inherited fleet connectors render as **locked rows** with an indigo `fleet` badge — exactly the
Skills tab's fleet-badge convention (`index.html:3874-3924`) — linking to the Connectors page to
edit. Agent-local servers stay editable in place, unchanged. The read path is
`GET /api/connectors` filtered client-side by the selected agent; no new endpoint.

**CLI parity:** `openlegion status`-style listing (`cli/repl.py:1195-1205`) appends
`(+N fleet connectors)` when the store is non-empty. Read-only; CLI editing of the catalog is
out of scope.

### 4.7 Phase 1 explicitly does NOT

- Touch `MCPServerConfig`, `AgentConfig`, agent-container code, or `MCP_SERVERS` semantics.
- Migrate any existing per-agent config (none needed — both layers coexist by design).
- Change credential exposure for stdio (that's Phase 2's job, via a different transport).
- Give the operator agent any connector-management tool (D8).

---

## 5. Phase 2 — Remote transport & mesh gateway (the proxied path)

### 5.1 Architecture

```
Agent container                      Mesh host                        Remote MCP server
┌──────────────────┐   tool call    ┌──────────────────────┐  HTTPS  ┌─────────────────┐
│ ToolRegistry     │ ─────────────▶ │ POST /mesh/connectors │ ──────▶ │ streamable-http │
│  "mcp_remote" ───┼── mesh_client  │   /call               │  +Bearer│  MCP endpoint   │
│  entries         │ ◀───────────── │ MCPGateway            │ ◀────── │                 │
└──────────────────┘    result      │  vault.resolve (token │         └─────────────────┘
                                    │  never leaves mesh)   │
                                    └──────────────────────┘
```

The mesh owns the HTTP session and the `Authorization` header. The agent sees tool schemas and
results — never the token. This is the same trust shape as `execute_api_call` / `_handle_llm`
(`src/host/credentials.py:1043-1272`), applied to MCP.

### 5.2 `MCPGateway` — `src/host/mcp_gateway.py` (new, ~250 lines)

```python
class MCPGateway:
    """Mesh-side sessions to remote (http) MCP connectors.

    One lazily-opened session per connector, via the mcp SDK's
    streamable-HTTP client (same package the agent already uses for stdio).
    Auth header resolved through the vault at session open and re-resolved
    once on a 401 mid-session — refresh-on-resolve means OAuth connections
    (Phase 3) renew with no restart and no agent involvement.
    """

    INIT_TIMEOUT = 30   # parity with MCPClient.start (mcp_client.py:148-151)
    CALL_TIMEOUT = 60   # parity with MCPClient.call_tool (mcp_client.py:270-272)

    def __init__(self, store: ConnectorStore, vault: CredentialVault) -> None: ...

    async def list_tools(self, name: str) -> list[dict]:
        """Sanitized tool schemas for one connector. Discovery result cached
        until the connector record changes (catalog write bumps a per-name
        generation counter) — remote servers are not re-walked per agent boot."""

    async def call_tool(
        self, name: str, tool: str, arguments: dict, *, agent_id: str,
    ) -> dict:
        """Execute one tool call. Raises PermissionError unless the
        connector is assigned to agent_id — assignment IS the authz gate."""

    async def probe(self, name: str) -> dict:
        """Dashboard 'Test connection': initialize + list_tools, returning
        {ok, tools_count} or {ok: False, error, needs_auth: bool}.
        needs_auth=True on 401 → the UI surfaces the Connect button (Phase 3)."""

    async def aclose(self) -> None: ...
```

Implementation notes:
- Transport: `mcp.client.streamable_http.streamablehttp_client` — already a dependency (the
  agent image ships the `mcp` SDK; the host shares `pyproject.toml`).
- **Sanitization at the trust boundary:** tool names/descriptions/schemas from a remote server
  are untrusted text that will reach LLM prompts. The gateway applies the same metadata
  sanitization `MCPClient.start()` applies for stdio (`mcp_client.py:164-179`) before caching.
  Results flow back as ordinary tool output (identical posture to stdio results).
- Auth resolution: `await vault.resolve_credential_async(auth.cred)` for `bearer`;
  connection-key resolution (existing refresh-on-resolve) for `oauth`. One retry with a fresh
  resolve on 401, then fail the call with a masked error (full text logged mesh-side, same
  policy as the LLM proxy's masked upstream errors).
- Session failure → next call reopens. No background reconnect loops (KISS; first failing call
  pays the reconnect).

### 5.3 Mesh endpoints — `src/host/server.py`

Both agent-authenticated (standard `X-Agent-ID` + token), both rate-limited under a new
`_RATE_LIMITS` category `"connectors"` (same budget as the API-proxy category):

```
GET  /mesh/connectors/tools
     → {"connectors": {"linear": {"tools": [...sanitized schemas...]}}}
     Scope: only connectors assigned to the CALLER (store.http_for_agent).

POST /mesh/connectors/call
     Body: {"connector": "linear", "tool": "create_issue", "arguments": {...}}
     Authz: assignment check inside gateway.call_tool (deny → 403).
     Audit: audit_log(kind="connector_call", agent, connector, tool) —
     arguments truncated to 500 chars, never logged raw.
     → {"result": ...} | {"error": "..."}  (masked upstream errors)
```

### 5.4 Agent side — registration & routing

Minimal, additive, no loop changes (Known Constraint #2):

- `src/agent/mesh_client.py`: `list_connector_tools()` and
  `call_connector_tool(connector, tool, arguments)` — thin wrappers over the two endpoints.
- `src/agent/__main__.py` (startup, after stdio MCP registration): fetch remote tools, register
  each into `ToolRegistry` with `"function": "mcp_remote"` and the originating connector name.
  Name conflicts reuse the existing policy verbatim: builtin collision or duplicate →
  `mcp_{connector}_{tool}` (`mcp_client.py:154-162`). Mesh unreachable → warn and skip; the
  agent boots without remote tools rather than crash-looping.
- `src/agent/tools.py` `ToolRegistry.execute()`: one new dispatch arm next to the existing
  `"mcp"` arm (`tools.py:259-265`) routing `"mcp_remote"` → `mesh_client.call_connector_tool`.

Tool *lists* are fetched at agent startup, so editing a remote connector's tools still follows
the uniform restart-to-apply rule (D7) — predictable and identical across transports. What does
**not** need a restart: token rotation, OAuth refresh, or re-connecting auth — all resolved
per-call on the mesh. That asymmetry is the payoff of the gateway and gets a line in the UI
("Auth changes apply immediately; server changes apply on restart").

### 5.5 Security posture (Phase 2)

| Property | Mechanism |
|---|---|
| Token never enters container | Resolved mesh-side in `MCPGateway`; agents receive schemas/results only |
| Authorization | Assignment check on every `/mesh/connectors/call` (deny-all default: unassigned = 403) |
| Prompt injection via tool metadata | Gateway sanitizes schemas at discovery (stdio-parity) |
| Abuse / runaway loops | `_RATE_LIMITS["connectors"]` per-agent; 60 s call timeout |
| Audit | `connector_call` audit rows (agent, connector, tool; args truncated) |
| Egress | Mesh-host outbound to an **operator-supplied** URL — operator is full-trust, but `HttpConnector._https_only` still enforces TLS (loopback exempt for self-hosted dev) |
| Secret leakage in errors | Upstream error bodies masked for the agent, full text logged mesh-side (LLM-proxy policy) |

---

## 6. Phase 3 — "Paste URL → Connect" OAuth flow

Everything hard here is already built and battle-tested in the Option-B Google flow
(`2026-06-04-oauth-integrations-connect-flow.md`): CSRF state store (single-use, TTL,
session-bound), callback handling, `store_connection`, vault refresh-on-resolve, redaction.
Phase 3 generalizes the *front half* (where the authorize/token endpoints come from) for MCP's
OAuth 2.1 profile.

### 6.1 Flow

```
1. Operator adds a Remote connector. probe() returns 401 + needs_auth
   → card shows [Connect].
2. GET /integrations/mcp/{name}/connect
   a. Discovery: GET {mcp_origin}/.well-known/oauth-protected-resource
      (RFC 9728) → authorization_servers[0]
   b. GET AS metadata (RFC 8414) → authorization/token/registration endpoints
   c. Client identity: if registration_endpoint present → Dynamic Client
      Registration (RFC 7591), client_id (+secret) persisted on the connector
      record / vault. Else → operator-supplied client_id/secret fields in the
      modal (the BYO fallback, exactly Option B's posture).
   d. Build authorize URL: PKCE S256 (always), resource={mcp_url} (RFC 8707),
      state from the EXISTING state store.
   e. 302 → provider consent screen.
3. GET /integrations/mcp/{name}/callback?code&state
   → validate state → token exchange with code_verifier
   → store_connection(f"mcp_{name}", access, refresh, expiry, scopes)
   → connector.auth = {kind: "oauth", connection: f"mcp_{name}"} persisted
   → 302 → /#system/integrations?connected={name}
4. Gateway resolves the connection per call; refresh-on-resolve keeps it
   fresh forever. No restart needed (5.4).
```

### 6.2 Phase-3 specifics to get right

- **Deployment:** engines sit behind the Caddy `forward_auth` SSO gate. The callback arrives on
  an already-authenticated operator session, but **verify in staging that the gate passes
  `?code&state` query params through untouched** — same verification the Google callback
  needed; if it passed there, this path is identical.
- **Discovery fetches are mesh-side requests to an operator-supplied origin.** Operator is
  full-trust (it's their server), but: https enforced by the model validator, responses size-capped
  (64 KB) and content-type-checked, and discovered endpoint URLs must themselves be https.
- **DCR responses** can include a `client_secret` → straight to the vault, never the connector
  record on disk; `deep_redact` on all flow logging (existing redaction module).
- **Servers without protected-resource metadata** (non-compliant): fall back to the BYO
  client-id fields rather than failing the whole add flow. The error message says exactly which
  discovery step failed.

---

## 7. Non-regression contract

Run-through before each phase merges:

1. **No `connectors.json` ⇒ bit-identical behavior.** `set_connector_store(None)` (or absent
   file ⇒ empty catalog) makes `_effective_mcp_servers` return the local list unchanged; both
   `MCP_SERVERS` sites then behave exactly as today. Pin with a test asserting env equality.
2. **Per-agent `mcp_servers` editor untouched.** GET masking (`_mask_mcp_servers_for_get`),
   PUT preserve-or-replace, `_canonicalize_mcp_servers` diffing, restart-on-touch: zero changes.
3. **Fail-closed catalog, fail-open agent start.** Corrupt `connectors.json` ⇒ empty catalog +
   error log (matches `PermissionMatrix._load` policy); a store *exception* during start ⇒
   local-only merge, never a blocked agent.
4. **Credential gates unchanged.** Fleet stdio connectors flow through the same
   `resolve_cred_handles` + `can_access_credential` path; an unauthorized `$CRED{}` still fails
   the start loudly (`runtime.py:131-139`), now *also* pre-flighted at PUT time per assigned
   agent (4.5).
5. **No mass-restart surprises.** No code path restarts an agent without an explicit operator
   action (`restart-batch` is only ever called from the confirm dialog).
6. **Frozen surfaces respected.** Tab ID `integrations` unchanged (Constraint #5); no new
   module-level globals (Constraint #8); no agent-loop changes (Constraint #2); operator cannot
   manage connectors (D8, `docs/mcp.md` decision).
7. **Token-hygiene parity.** Remote-connector secrets must never appear in: connector GET
   responses (auth shows `kind` + names only), audit rows, event payloads, agent-visible errors,
   or `MCP_SERVERS` (http connectors are excluded from serialization by construction —
   `stdio_for_agent` only).

## 8. Testing plan

**Phase 1**
- `tests/test_connectors_store.py` — CRUD; atomic save (inject crash between tempfile and
  replace, assert old file intact); fail-closed corrupt load; `"*"` vs explicit assignment;
  case-insensitive name uniqueness; `stdio_for_agent` strips catalog-only fields.
- `tests/test_runtime.py` (extend) — merge precedence (local wins, warning logged); fleet-only
  agent gets `MCP_SERVERS`; no-store ⇒ env byte-identical to today (regression pin #1); store
  exception ⇒ local-only; merged `$CRED{}` resolution still permission-gated; registry
  `agent_info["mcp_servers"]` stays agent-local after a merged start (§4.4 pin).
- `tests/test_dashboard_connectors.py` — PUT validation error shape matches the existing
  mcp_servers 400 contract; env preserve-or-replace round-trip; per-agent cred pre-flight
  (blocked agent named in detail); affected-agents = before ∪ after; DELETE; restart-batch
  sequencing + per-agent results; CSRF header required.

**Phase 2**
- `tests/test_mcp_gateway.py` — mocked streamable-http session: discovery caching +
  invalidation on catalog write; metadata sanitization applied; 401 → single re-resolve retry;
  call timeout; assignment denial raises.
- Mesh endpoint tests — unassigned agent gets 403; rate-limit category enforced; audit row
  shape; masked upstream errors.
- Agent-side — `mcp_remote` registration with conflict prefixing; `ToolRegistry.execute`
  routing; mesh-unreachable-at-boot degrades gracefully.

**Phase 3**
- httpx `MockTransport` end-to-end: RFC 9728 → RFC 8414 → DCR → PKCE exchange; state replay
  rejected; missing protected-resource metadata falls back to BYO fields; size-capped discovery
  responses; secrets absent from logs (assert via `deep_redact` spy).

**E2E** (existing skip-without-Docker pattern): one fleet stdio connector assigned `["*"]`,
two agents, assert both report it in `/capabilities` and the tool round-trips on each.

## 9. File-by-file change summary

| File | Phase | Change |
|---|---|---|
| `src/shared/types.py` | 1 (+2) | `ConnectorAuth`, `StdioConnector`, `HttpConnector`, `Connector` union, `CONNECTOR_ALL_AGENTS` |
| `src/host/connectors.py` | 1 | **New** — `ConnectorStore` |
| `src/host/runtime.py` | 1 | `set_connector_store`, `_effective_mcp_servers`; 2-line edits at `:423` and `:1106` |
| `src/cli/runtime.py` | 1 | Instantiate store; wire at `:381` and `:648`; pass to dashboard factory |
| `src/dashboard/server.py` | 1 | `GET/PUT/DELETE /api/connectors*`, `POST /api/agents/restart-batch`; audit + events |
| `src/dashboard/static/js/app.js` | 1 | Label `'Connectors'` at `:508`; connectors state + methods (mirroring the skills-state block at `:518-529`) |
| `src/dashboard/templates/index.html` | 1 | MCP Connectors section on the Integrations page; fleet-badge rows in the per-agent MCP section |
| `src/cli/repl.py` | 1 | Status line `(+N fleet connectors)` |
| `src/host/mcp_gateway.py` | 2 | **New** — `MCPGateway` |
| `src/host/server.py` | 2 | `/mesh/connectors/tools`, `/mesh/connectors/call`; `_RATE_LIMITS["connectors"]` |
| `src/agent/mesh_client.py` | 2 | `list_connector_tools`, `call_connector_tool` |
| `src/agent/__main__.py`, `src/agent/tools.py` | 2 | Remote-tool registration; `mcp_remote` dispatch arm |
| `src/host/oauth_providers.py` + dashboard routes | 3 | MCP discovery/DCR/PKCE connect + callback (reusing Option-B state store + `store_connection`) |
| `docs/mcp.md`, `docs/dashboard.md`, `CLAUDE.md` | each | Document the layer shipped in that phase |

## 10. Out of scope / tracked follow-ups

- **Operator-agent connector requests** (request-card pattern) — deferred per `docs/mcp.md`; D8.
- **Per-tool enable/disable within a connector** — YAGNI until a real connector ships too many
  tools; the schema leaves room (`tools_allow: list[str]` on the record) without committing now.
- **Hot tool-list refresh without restart** — uniform restart rule (D7) until demand exists.
- **Curated connector directory** ("one-click Linear/GitHub/Slack") — a static JSON of presets
  that pre-fills the Add modal; pure UI sugar once Phase 3 lands. Pin installs to verified
  endpoints when this becomes agent-reachable in any form (security review M1/H15 posture).
- **`If-Match` concurrency on connector PUT** — last-write-wins, consistent with every other
  dashboard config edit under the single-operator model.
