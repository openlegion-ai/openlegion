# MCP Integration Guide

OpenLegion supports the **Model Context Protocol (MCP)** -- the emerging standard for LLM tool interoperability. Any MCP-compatible tool server can be plugged into an agent, with tools automatically discovered and exposed to the LLM alongside built-in skills.

## Overview

MCP servers are external processes that expose tools via a standardized protocol. OpenLegion launches them as subprocesses inside agent containers using **stdio transport only** (the HTTP/SSE transports are not wired up — `MCPClient` calls `stdio_client(params)` directly), discovers their tools, and routes LLM tool calls through the MCP protocol.

```
LLM -> tool_call("read_file", {path: "/data/report.csv"})
  -> SkillRegistry.execute()
    -> MCPClient.call_tool()
      -> stdio -> MCP Server subprocess
        -> result
```

## Configuration

`mcp_servers` is an agent-level field on a fleet template (`src/templates/*.yaml`) or on a runtime agent record. The host serializes the list as JSON into the `MCP_SERVERS` environment variable when launching the agent container (`DockerBackend` in `src/host/runtime.py`).

```yaml
# src/templates/your_template.yaml
name: analyst
agents:
  analyst:
    role: "Data analyst"
    model: "openai/gpt-4.1-mini"
    mcp_servers:
      - name: filesystem
        command: mcp-server-filesystem
        args: ["/data"]
      - name: sqlite
        command: mcp-server-sqlite
        args: ["--db", "/data/analytics.db"]
```

There is no top-level `config/agents.yaml` file shipped with the repo — agents are defined either via the fleet template system (`src/cli/config.py:_load_templates`) or created at runtime through the dashboard / operator tools.

### Server Config Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Unique identifier (1-64 chars, `^[a-zA-Z0-9][a-zA-Z0-9_-]*$`). Used as a prefix when a tool name collides with a built-in or another server's tool (`mcp_<server>_<tool>`). Case-insensitive duplicate names are rejected. |
| `command` | string | Yes | Command to launch the server (max 256 chars). **Cannot contain `$CRED{name}` handles** — use `env` or `args` if a credential needs to reach the subprocess. |
| `args` | list[string] | No | Command-line arguments (defaults to `[]`, max 32 entries × 512 chars). May contain `$CRED{name}` handles — the mesh resolves them at agent start. |
| `env` | dict[string, string] | No | Environment variables for the server process (defaults to `None`, max 32 entries × 4096-char values). Values may contain `$CRED{name}` handles. |

Entries are validated by `src.shared.types.MCPServerConfig` at load time (`src/cli/config.py:_load_config`), at PUT time (`/api/agents/{id}/config`), and persisted as plain dicts so existing tooling that diffs `config/agents.yaml` keeps working.

### Credential handles in `env` and `args`

`$CRED{name}` references the agent-tier credential vault. The mesh resolves them just before serializing the `MCP_SERVERS` env var for the agent container — the resolved plaintext goes into the subprocess environment, never to disk and never through the API surface.

```yaml
mcp_servers:
  - name: linear
    command: mcp-server-linear
    args: ["--workspace", "ol"]
    env:
      LINEAR_API_KEY: "$CRED{linear_token}"
```

**Permission gate.** Each handle is checked against the agent's `allowed_credentials` glob list (via `permissions.can_access_credential`) at **both** PUT time (so the dashboard rejects an unsavable config synchronously) and runtime (so a permission revoked after save is enforced on next restart).

**What handles protect (and what they don't):**

| Surface | Protected? |
|---|---|
| `config/agents.yaml` on disk | ✅ stored as `$CRED{name}` reference |
| `audit_log` table | ✅ env values redacted; only keys preserved |
| `GET /api/agents/{id}/config` | ✅ `env` field omitted; `env_keys` returned instead |
| Container `MCP_SERVERS` env var | ❌ plaintext after mesh resolves |
| MCP subprocess `env` | ❌ plaintext (the subprocess needs the value to authenticate — this is a protocol fact, not a routing problem) |
| Container process memory | ❌ plaintext |

The runtime exposure is bounded by the existing container hardening (UID 1000, `cap_drop=ALL`, `read_only` filesystem, 384 MB memory cap). Persistent storage and observability surfaces are the threat-model windows that credential handles close.

> **Asymmetry vs. `http_tool`'s `$CRED{}` handles — be explicit.** When an agent uses `$CRED{name}` through `http_tool`, the handle is resolved **server-side in the mesh** and the secret never enters the agent process; responses are redacted on the way back. MCP is different: the resolved plaintext is placed into the agent container's `MCP_SERVERS` environment variable and forwarded into the MCP subprocess's `env`, so an MCP-using agent's **own process can read those secrets from its environment**. This is by design — stdio MCP needs the secret in-container to authenticate to the upstream service — but it means an MCP credential is exposed to the agent in a way an `http_tool` `$CRED{}` handle is not. Grant MCP-referenced credentials with that in mind. See `docs/security.md` and `docs/security-remediation-review-2026-05-29.md` (L14).

**Failure modes:**

- **Vault not wired.** `RuntimeBackend` raises a clear startup error if the config contains `$CRED{...}` handles but the mesh credential vault was not plumbed in via `set_credential_resolver`. Silent literal-passthrough was rejected as a footgun: a misconfigured deploy would otherwise ship literal `$CRED{...}` strings to subprocesses.
- **Credential missing.** Agent start raises `ValueError`; the dashboard surfaces this through the existing restart-failed UX with the credential name in the error message. Fix by storing the credential and retrying restart.
- **Permission denied.** Same as missing — agent start raises with the credential name. Fix by extending the agent's `allowed_credentials` glob.

### Dashboard `GET /config` env masking

`env` values are never returned by the dashboard config API. Each MCP server entry in the response omits the `env` key entirely and returns `env_keys: ["KEY1", "KEY2"]` instead — the dashboard renders the list of env-var names without ever holding the values.

The `env` field is **omitted**, not returned as `null`. This is deliberate: a naive `GET → edit → PUT` round-trip would otherwise lose env because the PUT handler treats "`env` present in body" as "replace wholesale." Omission lets the PUT handler tell "client preserved env" from "client wants to clear env":

- **Field absent from request body** → preserve persisted env for this server (matched by name).
- **Field present (as `{}` or `{K: v}`)** → replace wholesale.

### `mcp_touched` only fires on real diffs

Saving a config that doesn't actually change the persisted `mcp_servers` does NOT restart the agent. The PUT handler canonicalizes both sides (Pydantic model load + `model_dump(exclude_none=False)`, sorted by `name.lower()`, with `env={}` normalized to `env=None`) and only triggers the restart if the diff is non-empty.

### Concurrent edits

Last-write-wins semantics for the whole `mcp_servers` field. The dashboard does not currently use an `If-Match` / etag concurrency token, so two operators editing the same agent's MCP config simultaneously will have one overwrite the other. This is consistent with how every other dashboard config edit behaves today.

## Managing MCP servers from the dashboard

The agent settings → Config tab has an **MCP Servers** section that wraps the contract above. The visual UI is the primary path for users — fleet templates and the REST API still work but are no longer the only way in.

In display mode each configured server renders as a row with a status dot (green = running, red = failed at last startup, gray = pending restart), the command preview, and a tool-count badge. Click an agent's settings → Edit to manage.

### Adding / editing / removing servers

The edit-mode list reuses the inline Webhooks pattern: hover a row for **Edit** and **Remove** buttons, or click **+ Add MCP server** to open a fresh form. Each form has:

- **Name** and **Command** as plain inputs (validated against the same `MCPServerConfig` model the backend uses — same regex, same length caps, same `$CRED{...}` rejection in `command`).
- **Args** as a list-of-pairs (one input per arg + remove + `+ Add arg`). No JSON syntax to learn.
- **Env** as a list-of-rows where each row has a key field, a **Credential | Plain text** type toggle, and the value field:
  - **Credential** shows a dropdown filtered to the credentials the agent's `allowed_credentials` policy actually permits — the saved value becomes a `$CRED{name}` handle resolved by the mesh at agent start.
  - **Plain text** is a regular input. If you paste something that looks like an API key (e.g. starts with `sk-`, `ghp_`, `pat-`, `Bearer ...`, or is high-entropy), an inline warning nudges you toward Credential mode.
- A **Replace env vars** toggle (edit mode only) that defaults OFF — existing env preserves on the wire. When ON, the editor starts empty and the save replaces env wholesale; the UI explains that values are not retrievable from the masked GET (you'd need to re-supply all of them).
- An inline Node-runtime warning chip below the command input when it starts with `npx`/`bunx`/`pnpm dlx`/`yarn dlx`/`node`/`npm`/`pnpm`/`yarn`/`bun` (the default agent image is Python-only).

### Save + restart flow

Clicking the outer **Save** in the Config tab will auto-commit any open MCP draft (or block with a toast if the draft is incomplete) so typed entries don't get silently dropped. If `mcp_servers` actually changed (the dashboard does the same canonical diff as the backend's `mcp_touched` check), the agent is restarted automatically. The toast reports success/failure; the per-server status dot reflects the new state once the post-restart capabilities fetch lands.

If the restart itself fails (the config persisted but the container didn't come back up against it), the edit form **stays open** with a persistent red banner: *"Config saved, but agent restart failed: &lt;reason&gt;"* + a **Retry restart** button. Retry calls `POST /restart` only — the config is already on disk, no re-PUT.

### Failure visibility

A red status dot expands to show the captured stderr from `MCPClient.start()` (truncated to 500 chars). The common ones — `command not found`, missing or denied `$CRED{...}` reference, permission error — surface here without needing to tail container logs.

Validation errors from the backend (regex failures, oversize fields, `$CRED` in `command`, etc.) come back as a structured 400 and render as **inline red text** next to the offending field. Errors that don't map to a single row (e.g. case-insensitive duplicate names rejected by the `AgentConfig` field validator) render as a banner above the list.

### Operator-tool MCP management

**Not yet supported.** The operator agent can READ which MCP servers are configured on each agent (through the existing capabilities surface) but cannot ADD or REMOVE them via chat. A follow-up will introduce an operator-requested-setup flow modeled on the existing credential-request and browser-login patterns: the operator surfaces a request card in the dashboard, the human reviews and saves, the agent picks up the new server on restart. Until then, MCP writes are dashboard-only.

## How It Works

### Startup Sequence

1. The runtime layer (`DockerBackend` / `SandboxBackend` in `src/host/runtime.py`) reads `mcp_servers` from the agent's record and serializes it as JSON into the `MCP_SERVERS` environment variable passed to the agent container (`environment["MCP_SERVERS"] = self._build_mcp_servers_env(...)`). Any `$CRED{name}` handles in `env` values or `args` strings are resolved here against the mesh credential vault — the agent container receives plaintext values; the persisted config retains the handle.
2. Agent container starts; `src/agent/__main__.py` reads `MCP_SERVERS`
3. `MCPClient` is created and passed to `SkillRegistry`
4. During lifespan startup, `MCPClient.start()` launches each server:
   - Creates `StdioServerParameters` from config; any `env` dict in the server config is forwarded to the subprocess environment
   - Opens stdio transport via `AsyncExitStack`
   - Establishes `ClientSession` and calls `initialize()`
   - Calls `list_tools()` to discover available tools
5. Each tool is recorded in the `MCPClient` schema table with `"function": "mcp"` and an `_mcp_original_name` field that preserves the original tool name (used at call time — see Tool Call Routing). `SkillRegistry._register_mcp_tools()` then inserts those entries alongside built-in skills.
6. Agent registers with mesh, reporting MCP tools in its capabilities

If a server fails to start at step 4, its tools are not registered but the agent continues normally with built-in skills and any successfully started MCP servers.

### Tool Call Routing

MCP tools are registered into `SkillRegistry` at startup alongside built-in skills. Name conflicts are resolved **at registration time** (not at call time) — if an MCP tool has the same name as a built-in or as another MCP tool, it is renamed to `mcp_{server_name}_{tool_name}` before insertion. This means by the time `execute()` runs, every tool has a unique name and there is no runtime priority check.

When the LLM calls a tool:

1. `SkillRegistry.execute()` looks up the name in the unified skill dict
2. If the entry has `"function": "mcp"`, it routes to `MCPClient.call_tool(name, arguments)`
3. `MCPClient` looks up which server provides the tool via its internal `_tool_to_server` map
4. The renamed-on-conflict name is mapped back to `_mcp_original_name` before being sent over the wire, so the MCP server receives the tool name it actually exposes (useful when grepping server-side logs)
5. Sends the call via the MCP session to the correct subprocess
6. Converts the MCP `CallToolResult` to a dict: text content blocks are concatenated under a `"result"` key; image and binary content blocks are silently dropped. If `result.isError` is true, the text is returned under an `"error"` key instead.

Each `call_tool()` call has a **60-second timeout** (`asyncio.wait_for(..., timeout=60)`). If the server does not respond in time, the call returns `{"error": "MCP tool '...' timed out after 60s"}`.

If an agent has `ALLOWED_TOOLS` configured (operator mode — `loop.py:277-287` sets `_is_operator = (allowed_tools is not None)`), MCP tool names must appear in that allowlist to be accessible — the restriction applies equally to built-ins and MCP tools.

### Name Conflict Resolution

Conflicts are resolved at registration time, before execution:

- If an MCP tool has the same name as a built-in skill, the MCP tool is renamed to `mcp_{server_name}_{tool_name}` and a warning is logged. The built-in always keeps its original name.
- If two MCP servers provide tools with the same name, conflicts are resolved in **config / registration order**: the first server listed in `mcp_servers` keeps the unprefixed name, and any subsequent server providing the same tool name gets prefixed with `mcp_{server_name}_{tool_name}`.

After registration every tool has a unique name, so there is no runtime priority resolution.

### Graceful Failure

- If one MCP server fails to start, others still work
- Built-in skills are always available regardless of MCP server status
- Failed servers are logged but don't crash the agent
- If a server crashes mid-session, tool calls return error dicts

### Shutdown

On agent shutdown, `MCPClient.stop()` closes the `AsyncExitStack`, which terminates all server subprocesses and cleans up resources.

## Architecture

### Source Files

| File | Role |
|------|------|
| `src/agent/mcp_client.py` | `MCPClient` class -- server lifecycle, tool discovery, call routing |
| `src/agent/skills.py` | `SkillRegistry` -- MCP tool registration and execution routing |
| `src/agent/__main__.py` | Reads `MCP_SERVERS` env var, creates MCPClient, manages lifespan |
| `src/host/runtime.py` | Passes `mcp_servers` config as container environment variable |
| `src/host/health.py` | Preserves `mcp_servers` across agent restarts |

### Key Classes

**`MCPClient`** (`src/agent/mcp_client.py`):
- `start(servers, builtin_names)` -- Launch servers, discover tools
- `stop()` -- Shut down all servers
- `list_tools()` -- Return discovered tools as schema dicts
- `call_tool(name, arguments)` -- Route call to correct server
- `has_tool(name)` -- Check if a tool exists

**`SkillRegistry`** (`src/agent/skills.py`):
- Accepts optional `mcp_client` parameter
- `_register_mcp_tools()` -- Register MCP tools in the skill dict
- `execute()` -- Dispatches tool calls by name from the unified skill dict; MCP tools are identified by the `"function": "mcp"` marker set at registration time
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

If you use a custom agent image and the `mcp` package is missing, the `from mcp import ...` block at the top of `src/agent/mcp_client.py` swallows the `ImportError` silently with no log line — the only visible signal is that `MCPClient.start()` later logs `"MCP SDK not installed — cannot start MCP servers"` and returns early, but only if `mcp_servers` is actually configured for that agent.

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

Configure it on an agent — either in a fleet template under `src/templates/*.yaml` or by setting `mcp_servers` directly on the agent record (the host will serialize it into `MCP_SERVERS` for you):

```yaml
mcp_servers:
  - name: my-tools
    command: python
    args: ["/app/tools/my_server.py"]
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
