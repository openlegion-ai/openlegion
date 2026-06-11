# MCP Connectors — Global Catalog, Fleet Assignment, Remote Transport & OAuth

**Date:** 2026-06-10 · **Re-baselined:** 2026-06-11
**Status:** ALL PHASES IMPLEMENTED. Phase 1 2026-06-10 (stricter no-backcompat
variant — see §4); Phases 2a/2b/3 2026-06-11 as the stacked PRs #1125 → #1130 → #1131,
each with its own review pass. §11's Q1–Q6 proposed defaults were applied and are flagged
in the respective PR bodies for sign-off; the Caddy `forward_auth` `?code&state` staging
check (§7.3) remains a DEPLOY gate for #1131. File:line references below are pinned to
`7cf92c6b` (pre-implementation) — read them as the seams the work was written against.
**Scope:** Phase 1 promoted MCP from per-agent config to a fleet-level **Connectors** catalog
(connect once, assign to all agents or specific agents). Phase 2a adds the remote-connector
**data model and dashboard surface**; Phase 2b adds the **mesh-side `MCPGateway`** so remote
(HTTP) MCP servers work end-to-end with vault-held, proxied credentials — never entering the
agent container. Phase 3 adds the Claude-style **"paste URL → Connect → OAuth redirect"** flow
on top of the existing Option-B OAuth machinery.
**Related:**
- `docs/mcp.md` (stdio MCP design; D8's human-only decision is recorded there at `:141`)
- `docs/plans/2026-06-04-oauth-integrations-connect-flow.md` (Option-B OAuth: state store,
  callback, `store_connection`, refresh-on-resolve — Phase 3 reuses the back half; §7.1 lists
  what the front half must generalize first)
- `docs/plans/2026-06-04-integrations-oauth-strategy.md` (§4 names MCP connectors as the
  long-tail integration strategy; this plan is that section made concrete)

---

## 1. Problem

Three gaps, in increasing order of architectural weight. **Gap 1 is closed** (Phase 1 shipped);
gaps 2–3 are what Phases 2–3 exist for:

1. ~~**No global enablement.**~~ Closed by Phase 1: `config/connectors.json` +
   `ConnectorStore` (`src/host/connectors.py`) is the single source of MCP servers; an
   agent-specific server is a connector assigned to one agent. The per-agent `mcp_servers`
   config layer was **removed**, not wrapped.

2. **stdio-only.** `MCPConnector` (`src/shared/types.py:504`) is `command`/`args`/`env` — a
   subprocess inside the agent container. There is no way to point at a remote MCP server URL
   (`https://mcp.linear.app/mcp`), which is where the ecosystem has standardized. Worse, the
   default agent image is Python-only (`Dockerfile.agent` — no Node, no `npx`/`uvx`), so the
   large npm-based server catalog doesn't run at all. The transport users actually want —
   paste a URL — doesn't exist.

3. **Credentials are vault-*stored* but not vault-*proxied*.** Config holds `$CRED{name}`
   handles, but `RuntimeBackend._build_mcp_servers_env` (`src/host/runtime.py:137`) resolves
   them to plaintext at container start and ships them in the `MCP_SERVERS` env var. This is
   *inherent* to stdio (the subprocess needs the secret) — but it means every MCP credential
   is exposed to the agent process. Only a remote transport with a mesh-side session closes
   this.

## 2. Decisions

D1–D8 are the original decisions; annotations record how Phase 1 actually shipped.
D9–D17 were added at the 2026-06-11 re-baseline and are **binding on Phases 2a/2b/3**.

| # | Decision | Rationale |
|---|---|---|
| **D1** | Relabel the Integrations tab **"Connectors"**; keep the tab **ID `integrations`** | Shipped. Labels are free, IDs are frozen (Known Constraint #5). |
| **D2** | New file **`config/connectors.json`** + `ConnectorStore`, not more keys in `permissions.json` | Shipped. The store holds its lock across whole load→mutate→save; atomic writes; fail-closed load. |
| **D3** | Assignment lives **on the connector**: `agents: ["*"]` or explicit ids | Shipped — *stricter than planned*: the per-agent `mcp_servers` layer was removed entirely rather than coexisting. `agents: []` = unassigned. One surface, one source of truth. |
| **D4** | ~~Merge fleet connectors into the agent's list at one choke point~~ | **Moot.** There is no agent-local list to merge. The runtime reads the catalog directly via `_mcp_snapshot_for` (`runtime.py:112`). |
| **D5** | ~~Agent-local wins on name collision~~ | **Moot** with D4. Catalog-internal name uniqueness (case-insensitive) is enforced at the store. |
| **D6** | stdio keeps resolve-into-container-env semantics; **remote connectors are the proxied path**; the UI labels the difference honestly | Unchanged. *Remote — credentials stay in the vault; calls proxied by the mesh* vs *Local — runs inside the agent container; credentials are exposed to that agent*. |
| **D7** | Restarts are **explicit and confirmed, never automatic mass-restarts** | Shipped: catalog edits return `affected_agents`; `/api/agents/restart-batch` runs in the **background** (not sequentially as originally sketched) with per-agent restart events; pending-restart badges until bounced. |
| **D8** | Connector management stays **human-only** (dashboard); no operator-agent write tool | Shipped; recorded in `docs/mcp.md:141`. The request-card pattern remains the tracked follow-up. |
| **D9** | **Per-call gateway sessions.** The gateway opens `streamablehttp_client` + `ClientSession` per tool call — open → initialize → call → close, inside the single request task | The SDK's clients are anyio task-group context managers: entered in request task A and exited in task B (a later call, a 401 reopen, shutdown `aclose`) they raise `RuntimeError: attempted to exit cancel scope in a different task`. The agent's `MCPClient` survives long-lived sessions only because one owner task holds the `AsyncExitStack`; a gateway serving concurrent requests from N agents cannot. Per-call costs ~1 extra round-trip (noise against LLM-paced tool calls) and deletes the 401-retry state machine, the shared-session concurrency question, and `tools/list_changed` staleness. Long-lived sessions are a later optimization behind a dedicated owner-task design, if latency ever demands it. |
| **D10** | **`mcp>=1.9` becomes a core host dependency** (lazy import + loud degrade) | The host does NOT have the SDK today: `mcp` is an optional extra (`pyproject.toml:46`), `install.sh:194` installs `.[dev,channels,wallet]` without it, and `README.md:911` documents that. Only agent containers install it (`Dockerfile.agent`). The `>=1.0` floor predates `streamablehttp_client` (~1.8). Missing SDK at runtime → gateway endpoints return 503 with an actionable reason, mirroring the agent-side degrade (`src/agent/mcp_client.py:104-119`); never a silent no-op. |
| **D11** | **`/mesh/connectors/*` stays OUT of the operator bypass set.** Assignment is the authz gate for every agent, operator included | Connectors front third-party credentials — they belong with the still-gated family (`can_use_wallet*`, `can_access_credential`; Known Constraint #12), not the coordination bypass. Because assignment isn't a `permissions.can_*` gate, the grep trip-wire (`tests/test_operator_trust_tier.py:490`) covers it in neither direction — an explicit HTTP-level test pins it: operator with no assignment → 403. |
| **D12** | **Dirty matrix:** stdio any-change → pending-restart; http URL/assignment change → pending-restart; **http auth-only change → NOT pending-restart** (+ explicit gateway cache invalidation) | Remote auth is resolved per call on the mesh (D9), so connecting/rotating it needs no container bounce — marking dirty would manufacture false "restart to apply" nags and contradict the UI promise. URL/assignment changes DO need a bounce (tool schemas register at agent boot). UI copy "auth changes apply immediately" is **scoped to remote connectors** — for stdio, rotating a vault credential does *not* reach running containers (plaintext baked into env at start; pre-existing gap, see §10). |
| **D13** | **Byte-cap remote tool results at the gateway**: 256 KiB default + `truncated` flag (the `http_tool` convention) | The agent loop serializes dict tool results uncapped; a remote server returning a base64 blob otherwise lands whole in the agent's context. The gateway is the natural choke point. Cap value is §11-Q6. |
| **D14** | **No per-agent `can_access_credential` check on `auth.cred`** — vault-existence check only at PUT time | The bearer token is mesh-held and injected by the gateway; agents never see it. Requiring per-agent grants would force granting agents the secret precisely to avoid giving it to them. (stdio `$CRED` pre-flight keeps its per-agent check — those secrets DO enter the container.) |
| **D15** | **Server-initiated `sampling/createMessage` and elicitation are rejected**, pinned by test | A mesh-side `ClientSession` is a juicier target than an in-container one — a malicious server would be asking the credential holder to run LLM calls. Pass no callbacks (the SDK default rejects); the pin test ensures an SDK upgrade or a helpful refactor can't flip it. |
| **D16** | **SSRF posture on every mesh-originated MCP fetch**: https-only (loopback http exempt), private-range IP blocklist on resolved hosts (RFC1918, loopback, link-local, CGNAT — the `http_tool` ranges), `follow_redirects=False`, and in Phase 3 the same checks applied to **discovered** URLs (AS metadata, token, registration endpoints) plus 64 KB caps and content-type checks on discovery responses | `oauth_providers.py:5-6` pins today's invariant ("Provider endpoints are fixed here — never user-supplied — so the connect/callback flow adds no SSRF surface"); Phase 3 deletes it: the *remote server* — not the trusted operator — controls everything discovery returns. `https://169.254.169.254/` must not pass. Operator trust covers the pasted URL, not what its owner serves back. |
| **D17** | **Remote is the default-selected type in the Add-connector modal** once Phase 2b ships; Local (stdio) is the secondary option | Confirmed by user 2026-06-11. Matches the ecosystem default (Claude/ChatGPT connectors are URL-first) and our reality: the Python-only agent image can't run most published stdio servers anyway. |

## 3. Phasing

| Phase | Ships | Depends on |
|---|---|---|
| **1** | Global catalog + assignment + Connectors UI + restart orchestration (stdio) | **DONE** |
| **2a** | Remote data model (`HttpConnector` union) + transport-aware dashboard PUT/GET + UI transport chooser. Shippable dark: http connectors storable, inert until 2b | Phase 1 |
| **2b** | `MCPGateway` (per-call sessions) + `/mesh/connectors/*` + agent-side registration/dispatch. Delivers **bearer/no-auth remote connectors end-to-end** — standalone value before any OAuth | 2a |
| **3** | OAuth 2.1 connect flow (discovery + DCR + PKCE) | 2b + refresh-machinery extension (§7.1) + discovery SSRF posture (D16) |

Each phase is independently shippable and testable. The architectural risk is concentrated in
2b's gateway and 3's refresh extension; both have their load-bearing constraints resolved by
decision (D9, §7.1) before code.

---

## 4. Phase 1 — as shipped (baseline for everything below)

This section describes **reality**, not proposal. Phases 2a/2b are written against these APIs.
(The original §4 proposal — two-layer merge, `_effective_mcp_servers`, dirty-set tracking — is
in git history; none of it shipped in that form.)

### 4.1 Types — `src/shared/types.py`

- `MCPServerConfig` (`types.py:431`) — name/command/args/env, `extra="forbid"`, `$CRED`
  rejected in `command`, args/env caps. This is the `MCP_SERVERS` container-env contract.
- `MCPConnector(MCPServerConfig)` (`types.py:504`) — adds `transport: Literal["stdio"] =
  "stdio"` and `agents: list[str]` (`["*"]` sentinel `CONNECTOR_ALL_AGENTS`, `applies_to()`,
  dedup + id validation). `server_dict()` strips `{agents, transport}`. The `transport` field
  exists precisely so this plan's union lands without breaking persisted records.
- There is **no per-agent MCP config**: `AgentConfig` has no `mcp_servers`; `start_agent`
  takes no such param.

### 4.2 Store — `src/host/connectors.py`

Real API (the original plan's `assigned_agents()`, `http_for_agent()`, `reload()`, and
`dirty_agents` set were never built):

| Method | Behavior |
|---|---|
| `list()` / `get(name)` | mtime-based auto-reload (`_maybe_reload`) for hand-edited files |
| `upsert(connector)` | case-insensitive replace-by-name; touches old∪new agents |
| `remove(name)` / `remove_agent(agent_id)` | lifecycle hooks; `remove_agent` drops explicit assignments + generation stamps |
| `snapshot_for_agent(agent_id)` (`connectors.py:228`) | `(server_dicts, generation)` under one lock — **the runtime merge input** |
| `stdio_for_agent(agent_id)` | display convenience = `snapshot_for_agent()[0]` |
| `mark_dirty(agents)` / `record_agent_start(agent_id, gen)` / `pending_restart()` | pending-restart is a **generation derivation**: every mutation bumps `_generation` and stamps touched agents; the runtime snapshots the generation at container-env build and records it post-start; dirty ⇔ touch-gen > start-gen. Immune to the edit-during-container-build race. In-memory by design (a full mesh reboot restarts every container). |

Failure policy: missing/corrupt file → empty catalog + error log; malformed records dropped
per-record with error logs.

### 4.3 Runtime — `src/host/runtime.py`

`set_connector_store` (`:106`) → `_mcp_snapshot_for` (`:112`) → `_build_mcp_servers_env`
(`:137`, resolves `$CRED` with per-connector degradation — a missing/denied cred drops THAT
connector, never blocks boot) → env injection at both backends (`:464` Docker, `:1156`
Sandbox) → `record_agent_start` post-start (`:595`, `:1246`).

### 4.4 Dashboard — `src/dashboard/server.py`

- `_expand_assignment` (`:2294`), `_connector_to_api` (`:2301` — masks env to `env_keys` via
  the stdio masking helper).
- `GET /api/connectors` (`:2310`), `PUT /api/connectors/{name}` (`:2334` — validates via
  `MCPConnector.model_validate` at `:2365`, stdio env preserve-or-replace, `$CRED` pre-flight
  per assigned agent, no-op detection, affected = before∪after at `:2439`), `DELETE`
  (`:2473`), `POST /api/agents/restart-batch` (`:2536` — background task + in-flight guard +
  restart events).
- UI: Connectors sub-tab (`index.html:5150+`) with a **single stdio form** — there is no
  transport chooser; 2a adds it from scratch. Per-agent view shows assigned connectors
  read-only (`index.html:3109-3116`).

---

## 5. Phase 2a — remote data model & dashboard surface

Everything here is shippable dark: http connectors become storable and visible but inert
(no gateway yet). Independently testable at the store/endpoint layer.

### 5.1 Types union — `src/shared/types.py`

Keep the shipped `MCPConnector` name for the stdio variant (it is referenced across store,
runtime, dashboard, and tests; renaming to `StdioConnector` is churn with no behavior). Add:

```python
class ConnectorAuth(BaseModel):
    """Auth binding for a remote (http) connector. The secret always lives
    in the vault; this only names it.
    ``bearer`` → vault credential injected as ``Authorization: Bearer`` by
    the mesh gateway (D14: vault-existence checked at PUT; NO per-agent
    can_access_credential — agents never see this value).
    ``oauth`` → vault connection key (Phase 3 sets it; refresh-on-resolve)."""
    model_config = {"extra": "forbid"}

    kind: Literal["none", "bearer", "oauth"] = "none"
    cred: str | None = None        # required iff kind == "bearer"
    connection: str | None = None  # set by the connect flow iff kind == "oauth"

    @model_validator(mode="after")
    def _kind_fields(self) -> "ConnectorAuth":
        if self.kind == "bearer" and not self.cred:
            raise ValueError("auth.kind='bearer' requires auth.cred")
        return self


class HttpConnector(BaseModel):
    """Fleet-level remote MCP server. Mesh-gateway only; NEVER serialized
    into MCP_SERVERS (pinned by test — §8-1), so its auth never enters a
    container."""
    model_config = {"extra": "forbid"}

    transport: Literal["http"]
    name: str = Field(min_length=1, max_length=64, pattern=MCP_SERVER_NAME_RE_PATTERN)
    url: str = Field(min_length=1, max_length=512)
    auth: ConnectorAuth = Field(default_factory=ConnectorAuth)
    agents: list[str] = Field(default_factory=list, max_length=128)
    # url validator: https:// required; http:// allowed for localhost/127.0.0.1
    # only (self-hosted dev). agents validator shared with MCPConnector.


Connector = Annotated[MCPConnector | HttpConnector, Field(discriminator="transport")]
```

`MCPConnector.transport: Literal["stdio"] = "stdio"` already defaults, so pre-2a files and
hand-written records omit the key and still validate (Pydantic v2 defaulted discriminators).

### 5.2 Store changes — `src/host/connectors.py`

1. **`_load` validates against the `Connector` union** (`TypeAdapter`), keeping the
   per-record drop-with-error-log policy.
2. **`snapshot_for_agent` filters `transport == "stdio"`** before `server_dict()`. As shipped
   it serializes *every* connector passing `applies_to()` (`connectors.py:228-240`) — without
   this filter an http record, **including `auth`, would enter `MCP_SERVERS` and the
   container**, which is the exact exposure the gateway exists to prevent. Highest-severity
   item in the whole plan; pinned by test (§8-1). The generation half of the tuple still
   covers the *whole* catalog, so pending-restart keeps working for http edits on agents with
   no stdio connectors.
3. **`http_for_agent(agent_id) -> list[HttpConnector]`** — new, for the gateway and the mesh
   tools endpoint.
4. **Dirty matrix in `upsert` (D12):** when old and new are both http and differ only in
   `auth`, save WITHOUT `_touch` (no generation bump, no pending-restart). All other
   mutations touch old∪new as today. Return whether the edit was auth-only so the dashboard
   can (a) skip the restart prompt and (b) call the gateway's cache invalidation hook (§6.3 —
   the generation can't serve as the tools-cache key for auth edits precisely because auth
   edits don't bump it).

### 5.3 Dashboard — transport-aware in one deliberate step

All four stdio-blind sites in `src/dashboard/server.py` change together, not incrementally:

1. **Validation** (`:2365`): `MCPConnector.model_validate` → `TypeAdapter(Connector)`.
   Per-field 400 shape unchanged (the UI's inline-error rendering keys off it).
2. **No-op detection** (`:2425-2432`): per-transport — stdio compares `command/args/env`;
   http compares `url/auth/agents`, and an auth-only diff routes to the no-restart path (D12).
3. **GET masking** (`_connector_to_api`, `:2301`): http connectors emit `url`, `auth.kind`,
   and the cred/connection *names* only — never token values. stdio masking unchanged.
4. **Pre-flight** (`:2391+`): stdio keeps the per-agent `can_access_credential` check
   (secrets enter the container). http: `auth.cred` vault-**existence** check only (D14).

`affected_agents` stays before∪after; auth-only http edits return `restart_required: false`.

### 5.4 UI — transport chooser (net-new; nothing to "enable")

Add-connector modal becomes two-step:

1. *Type*: two radio cards — **Remote server** ("Credentials stay in the vault; calls are
   proxied by the mesh") **default-selected (D17)**, and **Local command** ("Runs inside each
   agent's container; credentials are exposed to that agent"). Until 2b ships, Remote saves
   but its card shows an *inactive — gateway not yet deployed* state; do not block storage
   (dark shipping is what makes 2a/2b independently mergeable).
2. *Details*: Local = the existing stdio form unchanged. Remote = URL + auth picker
   (None / API key → credential dropdown / OAuth → Connect button, Phase 3) + the existing
   Assignment control.

Connector cards: transport badge (`Local` slate / `Remote` indigo). Save flow keeps D7's
confirm-restart dialog, skipped entirely for auth-only edits. Remote auth section copy:
"Auth changes apply immediately — no restart" (scoped to remote per D12).

---

## 6. Phase 2b — mesh gateway & wire-up

### 6.1 Architecture

```
Agent container                      Mesh host                          Remote MCP server
┌──────────────────┐   tool call    ┌────────────────────────┐  HTTPS  ┌─────────────────┐
│ ToolRegistry     │ ─────────────▶ │ POST /mesh/connectors/ │ ──────▶ │ streamable-http │
│  "mcp_remote"  ──┼── mesh_client  │   call                 │ +Bearer │  MCP endpoint   │
│  entries         │ ◀───────────── │ MCPGateway             │ ◀────── │                 │
└──────────────────┘    result      │  per-call session;     │         └─────────────────┘
                                    │  vault token never     │
                                    │  leaves the mesh       │
                                    └────────────────────────┘
```

Same trust shape as `execute_api_call`/`_handle_llm` (`src/host/credentials.py`): the mesh
owns the HTTP session and the `Authorization` header; the agent sees schemas and results.

### 6.2 Dependency (D10)

`pyproject.toml`: add `mcp>=1.9` to core `dependencies`; the `[mcp]` extra remains as an
alias for one release (install docs reference it). Update `README.md:911`'s "NOT installed
by `./install.sh`" note. `Dockerfile.agent` unchanged (already installs it explicitly).
Gateway imports lazily; SDK missing → `/mesh/connectors/*` return 503 with "mcp SDK not
installed on the mesh host — re-run ./install.sh", mirroring `mcp_client.py:104-119`.

### 6.3 `MCPGateway` — `src/host/mcp_gateway.py` (new)

```python
class MCPGateway:
    """Mesh-side access to remote (http) MCP connectors.

    PER-CALL sessions (D9): every operation opens streamablehttp_client +
    ClientSession, initializes, executes, and closes — all inside the one
    request task. No session state survives a request; there is nothing to
    reopen on 401 (the next call re-resolves auth from the vault) and no
    cross-task cancel-scope hazard.

    SSRF posture (D16): before opening, the URL host is resolved and every
    address checked against the private-range blocklist (the http_tool
    ranges); the httpx client is built with follow_redirects=False via
    httpx_client_factory. https enforced by the model validator (loopback
    http exempt — and loopback exempt from the blocklist for that case).

    No client callbacks are passed (D15): server-initiated sampling /
    elicitation is rejected by the SDK default; pinned by test.
    """

    INIT_TIMEOUT = 30   # parity with MCPClient startup (mcp_client.py:148-151)
    CALL_TIMEOUT = 60   # parity with MCPClient.call_tool (mcp_client.py:270-273)
    RESULT_MAX_BYTES = 262_144  # D13; truncated results carry {"truncated": true}

    def __init__(self, store: ConnectorStore, vault: CredentialVault) -> None: ...

    async def list_tools(self, name: str) -> list[dict]:
        """Sanitized tool schemas for one connector. Cached keyed on
        (name, catalog generation); auth-only edits don't bump the
        generation (D12), so the dashboard invalidates explicitly via
        invalidate(name) after auth changes — a connector that 401'd
        before Connect has no tools cached and must re-discover after."""

    def invalidate(self, name: str) -> None: ...

    async def call_tool(self, name, tool, arguments, *, agent_id) -> dict:
        """PermissionError unless the connector is assigned to agent_id —
        assignment IS the authz gate (operator included, D11). Result
        byte-capped (D13). Upstream error bodies masked for the agent,
        full text logged mesh-side (LLM-proxy policy)."""

    async def probe(self, name: str) -> dict:
        """Dashboard 'Test connection': initialize + list_tools →
        {ok, tools_count} | {ok: False, error, needs_auth: bool}.
        needs_auth=True on 401 → UI surfaces Connect (Phase 3).
        Distinguishes DNS/timeout (server unreachable) from 4xx
        (misconfigured) in the error string."""
```

- Auth resolution per call: `bearer` → `vault.resolve_credential_async(auth.cred)`
  (`credentials.py:791`); `oauth` → `ensure_connection_token` (`credentials.py:1042`,
  refresh-on-resolve; rotated refresh tokens already persisted at `:1142`).
- Tool metadata sanitization mirrors the agent's stdio path (`mcp_client.py:154-188`):
  `sanitize_for_prompt` on names/descriptions/schema strings, description caps, before
  caching. Results flow back as ordinary tool output (identical posture to stdio results).
- The private-range check duplicates `http_tool`'s ranges initially; extracting a shared
  helper into `src/shared/` is welcome if it falls out naturally, not a prerequisite.

### 6.4 Mesh endpoints — `src/host/server.py`

Agent-authenticated, new `_RATE_LIMITS["connectors"]` category (`server.py:1028`; same
6000/60 budget as `api_proxy`):

```
GET  /mesh/connectors/tools     → caller-scoped: store.http_for_agent(caller) only
POST /mesh/connectors/call      → authz inside gateway.call_tool (unassigned → 403,
                                  operator included — D11, pinned by an HTTP-level test)
                                  audit_log(kind="connector_call", agent, connector, tool;
                                  arguments truncated to 500 chars, never raw)
```

Dashboard additions: `POST /api/connectors/{name}/probe` (CSRF-covered like every
state-adjacent route) and the auth-edit → `gateway.invalidate(name)` hook.

### 6.5 Agent side — registration & dispatch

No loop changes (Known Constraint #2). Note the original plan misdescribed dispatch: there is
no `"mcp"` string arm — stdio dispatch is the `self._mcp_client.has_tool(name)` short-circuit
at `src/agent/tools.py:252`, which runs FIRST. Remote registration must account for that:

- `src/agent/mesh_client.py`: `list_connector_tools()` / `call_connector_tool(...)` — thin
  wrappers over the two mesh endpoints.
- `src/agent/__main__.py` (after stdio MCP startup): fetch remote tools, register each into
  `ToolRegistry` with `"function": "mcp_remote"` + the originating connector name.
  **Conflict policy:** a remote tool whose name collides with a builtin OR an
  stdio-registered tool registers as `mcp_{connector}_{tool}` — checked against the union of
  both namespaces, because the `has_tool` short-circuit would otherwise silently shadow it.
  Mesh unreachable at boot → warn, register nothing, boot anyway; the agent's
  `/capabilities` payload includes remote connector status (name, tool count, last error)
  for stdio-parity status dots on the Connectors page.
- `ToolRegistry.execute()`: route registered `"mcp_remote"` entries →
  `mesh_client.call_connector_tool`.

Tool lists are fetched at agent boot, so URL/assignment edits follow the uniform
restart-to-apply rule (D7/D12). Auth changes apply per-call with no restart — that asymmetry
is the payoff of the gateway and is stated in the UI (§5.4).

---

## 7. Phase 3 — "paste URL → Connect" OAuth flow

The back half is shipped and verified: `OAuthStateStore` (`src/host/oauth_state.py:37` —
single-use, TTL, session-bound), `generate_pkce` (`oauth_providers.py:173`),
`GET /integrations/{provider}/connect` (`dashboard/server.py:4742`) + `/callback` (`:4782`),
`store_connection` (`credentials.py:990`, called at `dashboard/server.py:4820`),
refresh-on-resolve with rotated-refresh-token persistence (`credentials.py:1142`).

The front half does **not** generalize as the original plan claimed — two prerequisite
changes land before any flow code:

### 7.1 Prerequisite: refresh machinery extension — `src/host/credentials.py`

As shipped, `ensure_connection_token` resolves the provider from the **static registry** and
**silently returns the stale access token** when `get_provider()` misses
(`credentials.py:1077`); client id/secret come from system-env keys; and
`exchange_oauth_code` hard-requires a `client_secret` (`credentials.py:933`) — OAuth 2.1
public clients (DCR + PKCE, no secret) cannot exchange at all. "Refresh-on-resolve keeps it
fresh forever" is false for MCP connections until this lands:

- At callback time, persist **inside the connection blob**: `token_endpoint`, `client_id`,
  optional `client_secret`, `uses_pkce`, `resource`. (The blob already holds refresh tokens —
  equivalent custody; no parallel registry of dynamic providers to garbage-collect.)
- `ensure_connection_token`: prefer blob-embedded endpoint/credentials over the registry;
  unknown-provider-with-no-blob-endpoint keeps the legacy stale-return path so existing
  Google connections are untouched.
- `exchange_oauth_code`: support public clients (no secret; PKCE verifier).

### 7.2 Flow

```
1. Operator adds a Remote connector; probe() → 401 + needs_auth → card shows [Connect].
2. GET /integrations/mcp/{name}/connect
   a. Discovery: {mcp_origin}/.well-known/oauth-protected-resource (RFC 9728)
      → authorization_servers[0]
   b. AS metadata (RFC 8414) → authorization/token/registration endpoints
   c. Client identity: registration_endpoint present → DCR (RFC 7591);
      client_secret (if any) → connection blob via the vault, never the
      connector record on disk; deep_redact on all flow logging.
      No DCR → §11-Q4 (BYO fallback: proposed CUT from v1).
   d. Authorize URL: PKCE S256 always, resource={mcp_url} (RFC 8707),
      state from the EXISTING state store.
   e. 302 → provider consent.
3. GET /integrations/mcp/{name}/callback?code&state
   → validate state → exchange (public-client capable, §7.1)
   → store_connection(f"mcp_{name}", …including blob-embedded endpoints…)
   → connector.auth = {kind: "oauth", connection: f"mcp_{name}"} persisted
     (auth-only edit → no pending-restart, D12; gateway.invalidate(name))
   → 302 → /#system/integrations?connected={name}
4. Gateway resolves per call; refresh-on-resolve keeps it fresh. No restart.
```

### 7.3 Phase-3 specifics to get right

- **Discovery SSRF (D16) is a merge gate, not a footnote.** Every discovered URL — AS
  metadata, authorize, token, registration — gets https + public-IP-blocklist validation;
  discovery fetches run with `follow_redirects=False`, a 64 KB cap, and content-type checks.
  The validator on the *pasted* URL covers nothing the server serves back.
- **Caddy `forward_auth` staging check stays a gate:** verify `?code&state` pass through
  untouched on an engine subdomain before merge (same verification the Google callback
  needed).
- **Servers issuing no refresh token:** the Google callback path rejects connections lacking
  one when `refresh_required`. Many MCP AS implementations issue short-lived tokens with no
  refresh. Policy is §11-Q3 (proposed: accept; expiry surfaces as `needs_auth` on probe →
  reconnect card). Don't inherit the Google default silently.

---

## 8. Non-regression contract

Run-through before each phase merges (rewritten for shipped reality):

1. **http connectors never reach a container.** An `HttpConnector` assigned `["*"]` appears
   in NEITHER backend's `MCP_SERVERS` env (pin on both `runtime.py:464` and `:1156` paths) —
   §5.2's `snapshot_for_agent` filter, the highest-severity pin in this plan.
2. **No `connectors.json` ⇒ behavior unchanged**; corrupt file ⇒ empty catalog + error log;
   malformed records dropped per-record. (Existing Phase-1 pins stay green.)
3. **stdio semantics byte-identical**: env masking, `$CRED` per-agent pre-flight,
   per-connector degradation at start (`runtime.py:137+`), generation-based pending-restart.
4. **Dirty matrix (D12)**: http auth-only edit → no generation bump, no restart prompt,
   gateway cache invalidated; URL/assignment edit → pending-restart as today.
5. **No mass-restart surprises** (D7): restart-batch only from the confirm dialog.
6. **Operator gating (D11)**: unassigned operator → 403 on `/mesh/connectors/call`
   (HTTP-level test; the `can_*` grep trip-wire at `tests/test_operator_trust_tier.py:490`
   does not cover assignment gates).
7. **Sampling/elicitation rejected (D15)** — pinned.
8. **Token hygiene**: remote secrets never in connector GET responses (kind + names only),
   audit rows, event payloads, agent-visible errors, or `MCP_SERVERS`.
9. **Frozen surfaces**: tab ID `integrations`; no new module-level globals (Constraint #8);
   no agent-loop changes (Constraint #2); connector writes human-only (D8).

## 9. Testing plan

**Phase 2a**
- types: union round-trip; defaulted discriminator accepts legacy stdio records (no
  `transport` key); http url validator (https required, localhost http allowed —
  `https://169.254.169.254` is ACCEPTED at the model layer by design; the gateway is the
  SSRF layer); `ConnectorAuth` kind/field coupling.
- store: `snapshot_for_agent` excludes http (regression pin §8-1); `http_for_agent`;
  auth-only upsert skips `_touch` (generation unchanged, `pending_restart()` empty) while
  URL/assignment edits bump; per-record drop of malformed/unknown-transport records.
- dashboard: `TypeAdapter` 400 shape parity; per-transport no-op detection; http GET masking
  (no token values); `auth.cred` existence-only pre-flight (D14 — explicitly assert NO
  per-agent permission check); auth-only PUT returns `restart_required: false`.

**Phase 2b**
- gateway (mocked streamable-http): per-call open/close (no state across calls — assert via
  open/close counts); concurrent calls from multiple agents don't interfere; discovery cache
  keyed on generation + explicit `invalidate`; metadata sanitization applied; 401 →
  `needs_auth` on probe; result > cap → truncated + flag; **no sampling/elicitation
  callbacks registered** (D15 pin); redirects not followed; private-IP host rejected before
  connect (D16); SDK-missing → 503 with reason (D10).
- mesh endpoints: unassigned agent 403; **operator unassigned 403** (D11 pin); rate-limit
  category; audit row shape (args truncated); masked upstream errors.
- agent side: `mcp_remote` registration; conflict prefix against builtin AND stdio names
  (the `tools.py:252` short-circuit shadow case); `execute()` routing; mesh-unreachable boot
  degrade; `/capabilities` includes remote connector status.

**Phase 3**
- httpx `MockTransport` end-to-end: RFC 9728 → RFC 8414 → DCR → PKCE exchange (public
  client, no secret); state replay rejected; discovery responses size-capped +
  content-type-checked; **discovered private-IP/non-https endpoints rejected** (D16);
  blob-embedded refresh: token endpoint taken from blob (not registry), rotated refresh
  token persisted, legacy Google connections unaffected; no-refresh-token policy per §11-Q3;
  secrets absent from logs (`deep_redact` spy).

**E2E** (existing skip-without-Docker pattern): one stdio + one bearer http connector
assigned `["*"]`, two agents; both report both in `/capabilities`; the http tool round-trips
through the gateway; the stdio connector's container env never contains http auth.

## 10. Out of scope / tracked follow-ups

- **Vault credential rotation does not pending-restart stdio-assigned agents** — plaintext
  baked into container env at start; rotating `$CRED{x}` leaves running containers on the
  stale secret with no badge. Pre-existing gap, surfaced by D12's copy-scoping. Cheap fix
  when picked up: vault store/update → `mark_dirty` agents assigned any stdio connector
  referencing the handle.
- **Operator-agent connector requests** (request-card pattern) — deferred per `docs/mcp.md`;
  D8.
- **Per-tool enable/disable within a connector** — YAGNI; schema leaves room
  (`tools_allow: list[str]`) without committing.
- **Hot tool-list refresh without restart** — uniform restart rule until demand exists.
- **Long-lived gateway sessions** — only behind a dedicated owner-task design, only if
  per-call latency ever matters (D9).
- **Curated connector directory** ("one-click Linear/GitHub/Slack" presets) — UI sugar once
  Phase 3 lands; pin to verified endpoints if it ever becomes agent-reachable (M1/H15
  posture).
- **`If-Match` concurrency on connector PUT** — last-write-wins, consistent with every other
  dashboard config edit under the single-operator model.

## 11. Open product decisions — confirm before the named phase merges

| # | Question | Proposed default | Gates |
|---|---|---|---|
| **Q1** | Does `"*"` assignment include the **operator agent**? It does today — every fleet-wide connector's tool schemas land in the operator's context, feeding the known operator token-bloat problem. | Keep `"*"` = truly all (matches shipped semantics + the `can_message` glob convention); revisit if operator context telemetry shows connector schemas as a material contributor. | 2b |
| **Q2** | **Remote edits still bounce containers** (tools register at agent boot): adding/changing a URL restarts N agents even though nothing in the container changed but schemas. Acceptable story? | Yes for v1 — uniform with D7; hot refresh is a tracked follow-up. | 2b |
| **Q3** | **MCP servers issuing no refresh token**: reject at connect (the Google path's `refresh_required` default) or accept short-lived? | Accept; expiry surfaces as `needs_auth` on probe → reconnect card. | 3 |
| **Q4** | **BYO client-id fallback** for non-DCR servers: extra form fields + a second secret-entry path. Launch targets (Linear, GitHub, …) support DCR. | Cut from v1; DCR-only. The add-flow error names the missing discovery step so the gap is diagnosable. | 3 |
| **Q5** | **Downgrade data loss**: an older engine drops http records on load (by design, per-record) but its next *save* permanently deletes them from the file. | Accept + document in `docs/mcp.md` (single-operator, short rollback windows); revisit a `version` key only if a rollback actually bites. | 2a |
| **Q6** | **Result cap value** (D13) — caps change observable agent behavior. | 256 KiB + `truncated` flag. | 2b |

Resolved: **Remote as the default-selected Add-connector type** — confirmed by user
2026-06-11 (D17).

## 12. File-by-file change summary

| File | Phase | Change |
|---|---|---|
| `src/shared/types.py` | 2a | `ConnectorAuth`, `HttpConnector`, `Connector` union (`MCPConnector` keeps its name as the stdio variant) |
| `src/host/connectors.py` | 2a | Union-aware `_load`; `snapshot_for_agent` stdio filter (§5.2 — leak pin); `http_for_agent`; auth-only dirty matrix in `upsert` |
| `src/dashboard/server.py` | 2a | Transport-aware PUT/GET/no-op/pre-flight (§5.3) |
| `src/dashboard/templates/index.html`, `static/js/app.js` | 2a | Transport chooser (net-new), Remote default-selected (D17), remote card states |
| `pyproject.toml`, `README.md` | 2b | `mcp>=1.9` to core deps; update the install note (D10) |
| `src/host/mcp_gateway.py` | 2b | **New** — per-call-session `MCPGateway` (D9, D13, D15, D16) |
| `src/host/server.py` | 2b | `/mesh/connectors/tools`, `/mesh/connectors/call`; `_RATE_LIMITS["connectors"]` |
| `src/dashboard/server.py` | 2b | `POST /api/connectors/{name}/probe`; auth-edit → `gateway.invalidate` |
| `src/agent/mesh_client.py` | 2b | `list_connector_tools`, `call_connector_tool` |
| `src/agent/__main__.py`, `src/agent/tools.py` | 2b | Remote-tool registration (conflict prefix vs builtin∪stdio), `mcp_remote` dispatch, capabilities status |
| `src/host/credentials.py` | 3 | Blob-embedded token endpoints; public-client `exchange_oauth_code` (§7.1) |
| dashboard routes | 3 | `/integrations/mcp/{name}/connect` + `/callback` (discovery, DCR, PKCE; D16 posture) |
| `docs/mcp.md`, `docs/dashboard.md`, `CLAUDE.md` | each | Document the layer shipped in that phase |
