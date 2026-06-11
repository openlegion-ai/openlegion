# MCP Integration Guide

OpenLegion supports the **Model Context Protocol (MCP)** -- the emerging standard for LLM tool interoperability. Any MCP-compatible tool server can be plugged into an agent, with tools automatically discovered and exposed to the LLM alongside built-in tools.

## Overview

MCP servers are external processes that expose tools via a standardized protocol. OpenLegion launches them as subprocesses inside agent containers using **stdio transport only** (the HTTP/SSE transports are not wired up — `MCPClient` calls `stdio_client(params)` directly), discovers their tools, and routes LLM tool calls through the MCP protocol.

```
LLM -> tool_call("read_file", {path: "/data/report.csv"})
  -> ToolRegistry.execute()
    -> MCPClient.call_tool()
      -> stdio -> MCP Server subprocess
        -> result
```

## Configuration — the connector catalog

MCP servers are **connectors**: fleet-level records in `config/connectors.json`, each pairing a server definition with an **agent assignment**. The catalog (managed by `src/host/connectors.py:ConnectorStore`) is the single source of truth — there is no per-agent MCP config. Connect a server once and enable it for the whole fleet (`agents: ["*"]`) or for specific agents; an agent-specific server is simply a connector assigned to one agent.

```json
{
  "connectors": [
    {
      "name": "fetch",
      "command": "mcp-server-fetch",
      "args": [],
      "env": null,
      "agents": ["*"]
    },
    {
      "name": "sqlite",
      "command": "mcp-server-sqlite",
      "args": ["--db", "/data/analytics.db"],
      "env": { "DB_KEY": "$CRED{analytics_db_key}" },
      "agents": ["researcher", "analyst"]
    },
    {
      "name": "linear",
      "transport": "http",
      "url": "https://mcp.linear.app/mcp",
      "auth": { "kind": "bearer", "cred": "linear_token" },
      "agents": ["*"]
    }
  ]
}
```

### Two transports

A connector is either **local** (`transport: "stdio"`, the default when the
key is omitted) or **remote** (`transport: "http"`):

- **Local** — `command`/`args`/`env`: a subprocess inside each assigned
  agent's container. `$CRED{...}` handles resolve to plaintext in the
  container env at start (inherent to stdio: the subprocess needs the
  secret).
- **Remote** — `url` + `auth`: a streamable-HTTP MCP server reached
  through the **mesh-side gateway**. `auth.kind` is `none`, `bearer`
  (`auth.cred` names a vault credential injected as
  `Authorization: Bearer` per call), or `oauth` (`auth.connection` names
  a vault connection, set by the Connect flow). The secret never enters
  a container and never appears in any API response — the dashboard
  shows credential *names* only. Auth changes apply on the gateway's
  next call with **no agent restart**; URL and assignment changes apply
  on restart like everything else.

Remote records validate via the `Connector` discriminated union
(`src.shared.types.CONNECTOR_ADAPTER`) and are **excluded from
`MCP_SERVERS` by construction** (`snapshot_for_agent` serializes stdio
records only — pinned by test).

**How remote calls flow:** at agent start, the agent fetches sanitized
tool schemas from `GET /mesh/connectors/tools` (caller-scoped) and
registers them as `mcp_remote` tools; each invocation routes through
`POST /mesh/connectors/call` to the mesh-side `MCPGateway`
(`src/host/mcp_gateway.py`), which opens a **per-call** streamable-HTTP
session (no long-lived session state), resolves auth from the vault,
applies the resolved-IP SSRF blocklist with redirects disabled, rejects
server-initiated sampling/elicitation, byte-caps results (256 KiB +
`truncated` flag), and masks upstream error bodies. Assignment is the
authorization gate — deny-all default, **operator included** (the
trust-tier carve-out does not extend here; pinned by an HTTP-level
test). The dashboard's **Test** button calls
`POST /api/connectors/{name}/probe` for a fresh initialize + discovery;
`needs_auth: true` renders the **Connect** affordance.

**OAuth (paste URL → Connect):** for `auth.kind: "oauth"`, the Connect
button starts `GET /dashboard/integrations/mcp/{name}/connect` —
endpoint discovery (RFC 9728 protected-resource metadata →
`authorization_servers[0]` → RFC 8414 AS metadata), Dynamic Client
Registration (RFC 7591; falls back from `client_secret_post` to a
public client; **no bring-your-own-client fallback** — a server
without DCR gets an error naming the failed step), then PKCE S256 +
`resource` (RFC 8707) through the existing single-use, session-bound
state store. The callback exchanges the code (public-client capable),
stores the connection with the **discovered token endpoint + client
identity embedded in the blob** — `ensure_connection_token` prefers
blob fields over the static provider registry, so refresh works with
no registry entry — and binds `auth.connection` to the connector as an
auth-only edit (no restart; gateway cache invalidated). Connections
without a refresh token are accepted: expiry surfaces as `needs_auth`
on probe → reconnect. Everything discovery returns is server-controlled
input and passes the same SSRF posture as the gateway
(`src/host/mcp_oauth.py`: https-only, resolved-IP blocklist, no
redirects, 64 KB caps).

**Downgrade note:** an older engine version loads a catalog containing
`http` records by dropping them per-record with an error log (stdio
connectors are unaffected). If that older version then *saves* the
catalog (any connector edit), the dropped `http` records are permanently
removed from the file — re-add them after rolling forward.

At every agent (re)start, the runtime asks the catalog for the agent's assigned set (`RuntimeBackend._mcp_snapshot_for` in `src/host/runtime.py`) and serializes it as JSON into the `MCP_SERVERS` container environment variable. `MCP_SERVERS` is startup-only by design — **catalog edits apply on the next restart of the affected agents** (the dashboard prompts for it; nothing restarts automatically). Catalog order is meaningful: it feeds the first-server-wins tool-name conflict policy.

A missing or corrupt `connectors.json` fails **closed to an empty catalog** (error logged, agents start with no MCP servers); a store read error at start degrades the same way rather than blocking the agent.

### Connector Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Unique identifier (1-64 chars, `^[a-zA-Z0-9][a-zA-Z0-9_-]*$`, case-insensitive uniqueness). Used as a prefix when a tool name collides with a built-in or another server's tool (`mcp_<server>_<tool>`). |
| `command` | string | Yes | Command to launch the server (max 256 chars). **Cannot contain `$CRED{name}` handles** — use `env` or `args` if a credential needs to reach the subprocess. |
| `args` | list[string] | No | Command-line arguments (defaults to `[]`, max 32 entries × 512 chars). May contain `$CRED{name}` handles — the mesh resolves them at agent start. |
| `env` | dict[string, string] | No | Environment variables for the server process (defaults to `None`, max 32 entries × 4096-char values). Values may contain `$CRED{name}` handles. |
| `agents` | list[string] | No | Assignment: `["*"]` = every agent (exclusive — cannot be combined with explicit ids), or explicit agent ids. Defaults to `[]` (staged, reaches nobody). |

The table above covers local (stdio) connectors; remote ones replace `command`/`args`/`env` with `url` (https required; plain http allowed for localhost only) and `auth`. All records are validated through the `Connector` discriminated union (`src.shared.types.CONNECTOR_ADAPTER` — stdio variant `MCPConnector`, remote variant `HttpConnector`) at store load time and at dashboard PUT time; records with an unknown `transport` are dropped per-record with an error log.

### Credential handles in `env` and `args`

`$CRED{name}` references the agent-tier credential vault. The mesh resolves them just before serializing the `MCP_SERVERS` env var for the agent container — the resolved plaintext goes into the subprocess environment, never to disk and never through the API surface.

```json
{
  "name": "linear",
  "command": "mcp-server-linear",
  "args": ["--workspace", "ol"],
  "env": { "LINEAR_API_KEY": "$CRED{linear_token}" },
  "agents": ["*"]
}
```

**Permission gate.** Each handle is checked against the assigned LIVE agents' `allowed_credentials` glob lists (via `permissions.can_access_credential`) at PUT time — the dashboard rejects an unsavable connector synchronously, naming the blocked agents. At agent start, enforcement is **per-connector degradation**: a connector whose credential is missing or denied for that agent is dropped from its `MCP_SERVERS` (error logged with the connector, agent, and reason) while the rest ship — a fleet connector must never brick an agent that didn't ask for it (e.g. an agent created after a `"*"` connector was saved, under the default deny-all credential ACL).

**What handles protect (and what they don't):**

| Surface | Protected? |
|---|---|
| `config/connectors.json` on disk | ✅ stored as `$CRED{name}` reference |
| `audit_log` table | ✅ env values redacted; only keys preserved |
| `GET /api/connectors` | ✅ `env` field omitted; `env_keys` returned instead |
| Container `MCP_SERVERS` env var | ❌ plaintext after mesh resolves |
| MCP subprocess `env` | ❌ plaintext (the subprocess needs the value to authenticate — this is a protocol fact, not a routing problem) |
| Container process memory | ❌ plaintext |

The runtime exposure is bounded by the existing container hardening (UID 1000, `cap_drop=ALL`, `read_only` filesystem, 384 MB memory cap). Persistent storage and observability surfaces are the threat-model windows that credential handles close.

> **Asymmetry vs. `http_tool`'s `$CRED{}` handles — be explicit.** When an agent uses `$CRED{name}` through `http_tool`, the handle is resolved **server-side in the mesh** and the secret never enters the agent process; responses are redacted on the way back. MCP is different: the resolved plaintext is placed into the agent container's `MCP_SERVERS` environment variable and forwarded into the MCP subprocess's `env`, so an MCP-using agent's **own process can read those secrets from its environment**. This is by design — stdio MCP needs the secret in-container to authenticate to the upstream service — but it means an MCP credential is exposed to the agent in a way an `http_tool` `$CRED{}` handle is not. Grant MCP-referenced credentials with that in mind. See `docs/security.md` and `docs/security-remediation-review-2026-05-29.md` (L14).

**Failure modes:**

- **Vault not wired.** `RuntimeBackend` raises a clear startup error if the config contains `$CRED{...}` handles but the mesh credential vault was not plumbed in via `set_credential_resolver`. Silent literal-passthrough was rejected as a footgun: a misconfigured deploy would otherwise ship literal `$CRED{...}` strings to subprocesses.
- **Credential missing or permission denied.** That connector is DROPPED for that agent at start (error logged naming the connector, agent, and credential); the agent boots with its remaining connectors. The agent's Config-tab panel shows the dropped server's status dot as gray (it never reached the container). Fix by storing the credential / extending the agent's `allowed_credentials` glob, then restarting the agent.
- **Corrupt `connectors.json`.** Loads as an empty catalog (error logged); agents start with no MCP servers rather than failing to boot.

### `GET /api/connectors` env masking

`env` values are never returned by the connectors API. Each record in the response omits the `env` key entirely and returns `env_keys: ["KEY1", "KEY2"]` instead — the dashboard renders the list of env-var names without ever holding the values.

The `env` field is **omitted**, not returned as `null`. This is deliberate: a naive `GET → edit → PUT` round-trip would otherwise lose env because the PUT handler treats "`env` present in body" as "replace wholesale." Omission lets the PUT handler tell "client preserved env" from "client wants to clear env":

- **Field absent from request body** → preserve the persisted env for this connector.
- **Field present (as `{}` or `{K: v}`)** → replace wholesale.

### No-op saves don't prompt restarts

Saving a connector that doesn't actually change the persisted record (same canonical server fields, same assignment set) returns `restart_required: false` with an empty `affected_agents` list and marks nothing dirty — a `GET → unchanged PUT` round-trip never nags the operator to restart.

### Concurrent edits

Last-write-wins per connector record. The dashboard does not use an `If-Match` / etag concurrency token — consistent with every other dashboard config edit under the single-operator model. (Unlike the old per-agent field, the store itself holds its lock across the whole load→mutate→save, so two writers can't lose each other's *unrelated* connectors.)

## Managing connectors from the dashboard

Settings → **Connectors** (sub-tab ID `integrations` — only the label changed) opens with the **MCP connectors** panel. Each catalog record renders as a row with the assignment badge (**All agents** in indigo, or the explicit agent count), the command preview, and an env-var-count chip; hover a row for **Edit** and **Remove**, or click **+ Connect MCP server**.

The agent's own Config tab keeps a **read-only** MCP Connectors panel showing the connectors assigned to that agent with live status dots (green = running, red = failed at last startup with the captured error inline, gray = pending restart) and a "Manage in Connectors →" link. Assignment is edited only on the Connectors page.

### The connector form

- **Name** and **Command** as plain inputs (validated against the same `MCPConnector` model the backend uses — same regex, same length caps, same `$CRED{...}` rejection in `command`). Name is immutable once created.
- **Args** as a list-of-rows (one input per arg + remove + `+ Add arg`). No JSON syntax to learn.
- **Env** as a list-of-rows where each row has a key field, a **Credential | Plain text** type toggle, and the value field:
  - **Credential** shows a dropdown of agent-tier vault credential names — the saved value becomes a `$CRED{name}` handle resolved by the mesh at agent start.
  - **Plain text** is a regular input. If you paste something that looks like an API key (e.g. starts with `sk-`, `ghp_`, `pat-`, `Bearer ...`, or is high-entropy), an inline warning nudges you toward Credential mode.
- A **Replace env vars** toggle (edit mode only) that defaults OFF — existing env preserves on the wire. When ON, the editor starts empty and the save replaces env wholesale; the UI explains that values are not retrievable from the masked GET (you'd need to re-supply all of them).
- **Enabled for** — the assignment control: **All agents** or **Specific agents** with a checkbox per fleet member.
- An inline Node-runtime warning chip below the command input when it starts with `npx`/`bunx`/`pnpm dlx`/`yarn dlx`/`node`/`npm`/`pnpm`/`yarn`/`bun` (the default agent image is Python-only).

### Save + restart flow (explicit, never automatic)

A successful save returns the **affected agents** — the union of the assignment before and after the edit (an agent *removed* from a connector needs a bounce to lose its tools, too). The UI shows an amber prompt: *"X affects N agents — the change applies after restart"* with **Restart now** (calls `POST /api/agents/restart-batch`, which sequentially reuses the single-agent restart flow and its event choreography) and **Later**. Until the affected agents restart, a standing notice lists them as *running an older connector setup*; the pending state clears automatically when the runtime successfully starts each agent against the current catalog.

### Failure visibility

A red status dot on the agent's panel expands to show the captured stderr from `MCPClient.start()` (truncated to 500 chars). The common ones — `command not found`, missing or denied `$CRED{...}` reference, permission error — surface there without needing to tail container logs.

Validation errors from the backend (regex failures, oversize fields, `$CRED` in `command`, a bad `agents` list, etc.) come back as a structured 400 (`{detail: {field: "connector", errors: [...]}}`) and render as **inline red text** next to the offending field; credential pre-flight failures name the blocked agents and the fix.

### Operator-tool connector management

**Not supported, by design.** The operator agent can READ which connectors an agent runs (through the existing capabilities surface) but cannot ADD or REMOVE them via chat — chat-driven installation of arbitrary tool servers is a prompt-injection-shaped hole. A follow-up will introduce an operator-requested-setup flow modeled on the existing credential-request and browser-login patterns: the operator surfaces a request card in the dashboard, the human reviews and saves, the affected agents pick up the connector on restart.

## How It Works

### Startup Sequence

1. The runtime layer (`DockerBackend` / `SandboxBackend` in `src/host/runtime.py`) asks the connector catalog for the agent's assigned set (`self._mcp_snapshot_for(agent_id)`) and serializes it as JSON into the `MCP_SERVERS` environment variable passed to the agent container (`environment["MCP_SERVERS"] = self._build_mcp_servers_env(...)`). Any `$CRED{name}` handles in `env` values or `args` strings are resolved here against the mesh credential vault — the agent container receives plaintext values; the persisted catalog retains the handle.
2. Agent container starts; `src/agent/__main__.py` reads `MCP_SERVERS`
3. `MCPClient` is created and passed to `ToolRegistry`
4. During lifespan startup, `MCPClient.start()` launches each server:
   - Creates `StdioServerParameters` from config; any `env` dict in the server config is forwarded to the subprocess environment
   - Opens stdio transport via `AsyncExitStack`
   - Establishes `ClientSession` and calls `initialize()`
   - Calls `list_tools()` to discover available tools
5. Each tool is recorded in the `MCPClient` schema table with `"function": "mcp"` and an `_mcp_original_name` field that preserves the original tool name (used at call time — see Tool Call Routing). `ToolRegistry._register_mcp_tools()` then inserts those entries alongside built-in tools.
6. Agent registers with mesh, reporting MCP tools in its capabilities

If a server fails to start at step 4, its tools are not registered but the agent continues normally with built-in tools and any successfully started MCP servers.

### Tool Call Routing

MCP tools are registered into `ToolRegistry` at startup alongside built-in tools. Name conflicts are resolved **at registration time** (not at call time) — if an MCP tool has the same name as a built-in or as another MCP tool, it is renamed to `mcp_{server_name}_{tool_name}` before insertion. This means by the time `execute()` runs, every tool has a unique name and there is no runtime priority check.

When the LLM calls a tool:

1. `ToolRegistry.execute()` looks up the name in the unified tool dict
2. If the entry has `"function": "mcp"`, it routes to `MCPClient.call_tool(name, arguments)`
3. `MCPClient` looks up which server provides the tool via its internal `_tool_to_server` map
4. The renamed-on-conflict name is mapped back to `_mcp_original_name` before being sent over the wire, so the MCP server receives the tool name it actually exposes (useful when grepping server-side logs)
5. Sends the call via the MCP session to the correct subprocess
6. Converts the MCP `CallToolResult` to a dict: text content blocks are concatenated under a `"result"` key; image and binary content blocks are silently dropped. If `result.isError` is true, the text is returned under an `"error"` key instead.

Each `call_tool()` call has a **60-second timeout** (`asyncio.wait_for(..., timeout=60)`). If the server does not respond in time, the call returns `{"error": "MCP tool '...' timed out after 60s"}`.

If an agent has `ALLOWED_TOOLS` configured (operator mode — `loop.py:277-287` sets `_is_operator = (allowed_tools is not None)`), MCP tool names must appear in that allowlist to be accessible — the restriction applies equally to built-ins and MCP tools.

### Name Conflict Resolution

Conflicts are resolved at registration time, before execution:

- If an MCP tool has the same name as a built-in tool, the MCP tool is renamed to `mcp_{server_name}_{tool_name}` and a warning is logged. The built-in always keeps its original name.
- If two MCP servers provide tools with the same name, conflicts are resolved in **catalog / registration order**: the first assigned connector keeps the unprefixed name, and any subsequent server providing the same tool name gets prefixed with `mcp_{server_name}_{tool_name}`.

After registration every tool has a unique name, so there is no runtime priority resolution.

### Graceful Failure

- If one MCP server fails to start, others still work
- Built-in tools are always available regardless of MCP server status
- Failed servers are logged but don't crash the agent
- If a server crashes mid-session, tool calls return error dicts

### Shutdown

On agent shutdown, `MCPClient.stop()` closes the `AsyncExitStack`, which terminates all server subprocesses and cleans up resources.

## Architecture

### Source Files

| File | Role |
|------|------|
| `src/agent/mcp_client.py` | `MCPClient` class -- server lifecycle, tool discovery, call routing |
| `src/agent/tools.py` | `ToolRegistry` -- MCP tool registration and execution routing |
| `src/agent/__main__.py` | Reads `MCP_SERVERS` env var, creates MCPClient, manages lifespan |
| `src/host/connectors.py` | `ConnectorStore` — the fleet connector catalog (`config/connectors.json`) |
| `src/host/runtime.py` | Resolves each agent's assigned connectors into the `MCP_SERVERS` container env var at start |
| `src/dashboard/server.py` | `GET/PUT/DELETE /api/connectors`, `POST /api/agents/restart-batch` |

### Key Classes

**`MCPClient`** (`src/agent/mcp_client.py`):
- `start(servers, builtin_names)` -- Launch servers, discover tools
- `stop()` -- Shut down all servers
- `list_tools()` -- Return discovered tools as schema dicts
- `call_tool(name, arguments)` -- Route call to correct server
- `has_tool(name)` -- Check if a tool exists

**`ToolRegistry`** (`src/agent/tools.py`):
- Accepts optional `mcp_client` parameter
- `_register_mcp_tools()` -- Register MCP tools in the tool dict
- `execute()` -- Dispatches tool calls by name from the unified tool dict; MCP tools are identified by the `"function": "mcp"` marker set at registration time
- `get_tool_definitions()` -- MCP tools included with full JSON Schema

## Popular MCP Servers

Some well-known MCP servers that work with OpenLegion:

| Server | Package | Tools |
|--------|---------|-------|
| Filesystem | `@modelcontextprotocol/server-filesystem` | Read/write/search files |
| SQLite | `@modelcontextprotocol/server-sqlite` | Query SQLite databases |
| PostgreSQL | `@modelcontextprotocol/server-postgres` | Query PostgreSQL databases |
| Brave Search | `@modelcontextprotocol/server-brave-search` | Web search via Brave API |
| GitHub | `@modelcontextprotocol/server-github` | GitHub API operations |
| Playwright | `@playwright/mcp` | Browser automation (70+ tools) |

**Caveat:** the default `Dockerfile.agent` image is Python-only — Node.js is **not** installed. npm-based MCP servers (most of the entries above, including `@modelcontextprotocol/server-filesystem`, `@modelcontextprotocol/server-github`, and `@playwright/mcp`) will fail to launch out of the box; you must build a custom agent image with Node.js installed and the server package available on `PATH`. Python-based MCP servers work with the default image — the `mcp` Python SDK is pre-installed in `Dockerfile.agent`.

If you use a custom agent image and the `mcp` package is missing, the `from mcp import ...` block at the top of `src/agent/mcp_client.py` swallows the `ImportError` silently with no log line — the only visible signal is that `MCPClient.start()` later logs `"MCP SDK not installed — cannot start MCP servers"` and returns early, but only if the agent actually has assigned connectors.

## Writing a Custom MCP Server

A minimal Python MCP server using the `mcp` SDK (requires `mcp >= 1.0` for the `FastMCP` high-level API):

```python
from mcp.server import FastMCP

server = FastMCP("my-tools")

@server.tool()
def analyze(data: str, format: str = "json") -> str:
    """Analyze data in the specified format."""
    # Your logic here
    return f"Analysis of {len(data)} bytes in {format} format"

if __name__ == "__main__":
    server.run(transport="stdio")
```

Connect it on Settings → Connectors (or write the catalog record directly):

```json
{
  "name": "my-tools",
  "command": "python",
  "args": ["/app/tools/my_server.py"],
  "agents": ["*"]
}
```

## Troubleshooting

### Server fails to start

Check agent container logs:
```bash
openlegion status  # or docker logs openlegion_<agent_id>
```

Common causes:
- Command not found in container (missing dependency)
- Permission error (file not executable)
- Missing environment variable the server requires (add it via the `env` config field)

### Tools not discovered

- Verify the server runs correctly standalone: `python my_server.py` should start and accept stdin
- Check that `list_tools()` returns tools (use the MCP inspector)
- Look for "MCP server started" log messages

### Name conflicts

If your MCP tool has the same name as a built-in (e.g., `exec_command`), it will be prefixed. Check logs for "conflicts with existing tool, renamed to" warnings.

### Dashboard shows a red status dot

The agent's `/capabilities` endpoint exposes a per-server startup registry — each entry has `{state, tools_count, error}`. Failed startups capture the exception message (truncated to 500 chars); clicking the row in the dashboard surfaces the error verbatim, so you don't need to tail container logs for the common cases (`command not found`, missing env var, permission denied on a credential handle).

Note: the registry reflects the **last startup attempt** only — it is not a live health probe. A server that started successfully but crashed mid-session is not marked failed until the next agent restart re-runs discovery.

## Future: operator-requested setup

A planned follow-up will let the operator agent request MCP setup through the existing credential-request / browser-login pattern: the operator surfaces a dashboard card describing the tool it needs, the human reviews + saves, the agent picks up the new server on restart. The operator never edits MCP config directly — this preserves the operator-first positioning while sidestepping the prompt-injection risk of giving operator chat the power to install arbitrary subprocesses.
