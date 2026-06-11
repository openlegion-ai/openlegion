"""Mesh-side gateway for remote (http) MCP connectors.

The trust shape mirrors the LLM proxy (``src/host/credentials.py``):
the mesh owns the HTTP session and the ``Authorization`` header; agents
see sanitized tool schemas and capped results — never the token.

PER-CALL sessions (plan D9): every operation opens the streamable-HTTP
client + ``ClientSession``, initializes, executes, and closes — all
inside the single request task. The SDK's clients are anyio task-group
context managers; entered in one task and exited in another they raise
``RuntimeError: attempted to exit cancel scope in a different task``,
so a lazily-opened long-lived session per connector is a footgun, not
an optimization. Per-call costs ~1 extra round-trip — noise against
LLM-paced tool calls — and deletes the 401-retry state machine, the
shared-session concurrency question, and ``tools/list_changed``
staleness. Auth is re-resolved from the vault on every call, which is
what makes token rotation and OAuth refresh restart-free.

No client callbacks are passed to ``ClientSession`` (plan D15):
server-initiated ``sampling/createMessage`` / elicitation requests are
rejected by the SDK default. A mesh-side session is a juicier target
than an in-container one — a malicious server would be asking the
credential holder to run LLM calls. Pinned by test.

SSRF posture (plan D16): the URL host is resolved and every address
checked against the private-range blocklist below before a session
opens, and the httpx client is built with ``follow_redirects=False``.
Explicit loopback hosts (``localhost``/``127.0.0.1``/``::1``) are
allowed — self-hosted MCP on the mesh host is legitimate — but a
public DNS name *resolving* to loopback or a private range is rejected
(rebinding-shaped). Best-effort TOCTOU caveat: the SDK re-resolves to
connect; redirects are disabled so the checked origin is the only one
contacted (same posture as the browser-path M20 note in CLAUDE.md).
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
from typing import TYPE_CHECKING, Any, Callable
from urllib.parse import urlparse

from src.shared.types import HttpConnector
from src.shared.utils import sanitize_for_prompt, setup_logging

if TYPE_CHECKING:
    from src.host.connectors import ConnectorStore
    from src.host.credentials import CredentialVault

logger = setup_logging("host.mcp_gateway")

INIT_TIMEOUT = 30   # parity with MCPClient startup (agent/mcp_client.py)
CALL_TIMEOUT = 60   # parity with MCPClient.call_tool
RESULT_MAX_BYTES = 262_144  # plan D13 (§11-Q6): cap + truncated flag

# Tool descriptions byte-capped exactly like the agent-side stdio path
# (mcp_client._MCP_DESCRIPTION_MAX_BYTES) so one connector can't flood
# every assigned agent's context.
_DESCRIPTION_MAX_BYTES = 8 * 1024

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}

# Mirrors the http_tool SSRF blocklist ranges (src/agent/builtins/
# http_tool.py). Duplicated rather than imported: host code doesn't
# import agent builtins, and the list is small + stable. Extract to
# src/shared/ if a third copy ever appears.
_BLOCKED_NETS = tuple(ipaddress.ip_network(n) for n in (
    "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16",   # RFC1918
    "100.64.0.0/10",                                   # CGNAT
    "169.254.0.0/16",                                  # link-local + cloud metadata
    "192.0.0.0/24", "198.18.0.0/15",                   # IETF protocol / benchmarking
    "224.0.0.0/4", "240.0.0.0/4",                      # multicast / reserved
    "fe80::/10", "fc00::/7",                           # v6 link-local / ULA
    "64:ff9b::/96", "2002::/16", "2001::/32",          # NAT64 / 6to4 / Teredo
))


class GatewayUnavailable(RuntimeError):
    """The mcp SDK is not importable on the mesh host → endpoints 503."""


class ConnectorAuthError(RuntimeError):
    """Auth could not be resolved or was rejected upstream → needs_auth."""


class ConnectorSSRFError(RuntimeError):
    """URL host resolves into a blocked range."""


class UnknownConnectorError(KeyError):
    """No http connector with that name in the catalog."""


async def _assert_public_host(url: str) -> None:
    """Resolve the URL host; reject private/reserved destinations.

    Explicit loopback hostnames pass (self-hosted dev); everything else
    must resolve to public addresses only — one private A/AAAA record
    among many fails the whole check (the SDK may pick any of them).
    """
    host = urlparse(url).hostname or ""
    if host.lower() in _LOOPBACK_HOSTS:
        return
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, None)
    except OSError as e:
        raise ConnectorAuthError(f"cannot resolve host {host!r}: {e}") from e
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        bad = (
            ip.is_loopback or ip.is_link_local or ip.is_multicast
            or ip.is_reserved or ip.is_unspecified or ip.is_private
            or any(ip in net for net in _BLOCKED_NETS)
        )
        if bad:
            raise ConnectorSSRFError(
                f"connector host {host!r} resolves to blocked address {ip}",
            )


def _default_client_factory() -> tuple[Callable, type]:
    """Lazy SDK import seam. Returns ``(open_client, ClientSession)``.

    ``open_client(url, headers)`` must return the streamable-HTTP
    async context manager yielding ``(read, write, ...)``. Tests pass a
    fake via ``MCPGateway(client_factory=...)`` so the suite runs with
    or without the SDK installed.
    """
    try:
        import httpx
        from mcp.client.session import ClientSession
        from mcp.client.streamable_http import streamablehttp_client
    except ImportError as e:
        raise GatewayUnavailable(
            "mcp SDK not installed on the mesh host — re-run ./install.sh "
            "(mcp>=1.9 is a core dependency since the remote-connector "
            "gateway)",
        ) from e

    def _httpx_factory(headers=None, timeout=None, auth=None, **_kw):
        # follow_redirects=False is load-bearing (plan D16): the SDK's
        # default client follows redirects, which would let a remote
        # server bounce the mesh to an unchecked origin.
        return httpx.AsyncClient(
            headers=headers, timeout=timeout, auth=auth,
            follow_redirects=False,
        )

    def _open_client(url: str, headers: dict[str, str] | None):
        return streamablehttp_client(
            url, headers=headers, httpx_client_factory=_httpx_factory,
        )

    return _open_client, ClientSession


def _truncate_result(shaped: dict) -> dict:
    """Byte-cap a shaped tool result (plan D13). Oversize text is cut
    and flagged with ``truncated: True`` (the http_tool convention) so
    the LLM knows it is looking at a prefix. Applies to whichever key
    the result carries (``result`` or ``error``)."""
    raw = json.dumps(shaped, ensure_ascii=False, default=str)
    if len(raw.encode("utf-8", errors="ignore")) <= RESULT_MAX_BYTES:
        return shaped
    key = "result" if "result" in shaped else "error"
    text = str(shaped.get(key, ""))
    keep = max(0, RESULT_MAX_BYTES - 1024)
    cut = text.encode("utf-8", errors="ignore")[:keep].decode(
        "utf-8", errors="ignore",
    )
    return {**shaped, key: cut, "truncated": True}


class MCPGateway:
    """Mesh-side access to remote (http) MCP connectors."""

    def __init__(
        self,
        store: "ConnectorStore",
        vault: "CredentialVault | None",
        *,
        client_factory: Callable[[], tuple[Callable, type]] | None = None,
    ) -> None:
        self._store = store
        self._vault = vault
        self._client_factory = client_factory or _default_client_factory
        # name(lower) → (catalog generation, sanitized tool schemas).
        # Generation-keyed: URL/assignment edits bump the generation and
        # naturally invalidate; auth-only edits do NOT bump it (plan
        # D12), so the dashboard calls invalidate() explicitly after
        # auth changes — a connector that 401'd before Connect has no
        # tools cached and must re-discover.
        self._tools_cache: dict[str, tuple[int, list[dict]]] = {}

    # ── catalog helpers ──────────────────────────────────────────

    def _get_http(self, name: str) -> HttpConnector:
        c = self._store.get(name)
        if not isinstance(c, HttpConnector):
            raise UnknownConnectorError(name)
        return c

    def invalidate(self, name: str) -> None:
        """Drop the cached discovery for one connector (auth edits)."""
        self._tools_cache.pop(name.lower(), None)

    # ── session plumbing ─────────────────────────────────────────

    async def _headers(self, c: HttpConnector) -> dict[str, str]:
        """Resolve auth from the vault — per call, so rotation and
        OAuth refresh-on-resolve apply with no restart."""
        # ANY failure to produce the header is ConnectorAuthError — the
        # vault RAISES on missing/dead connections (RuntimeError("No
        # connection: …"), hard 4xx grant deaths as
        # ConnectionRefreshError) rather than returning a falsy token,
        # and classifying those as generic upstream errors would hide
        # exactly the state the probe's needs_auth → Connect/reconnect
        # affordance exists for.
        if c.auth.kind == "bearer":
            token = None
            if self._vault is not None:
                try:
                    token = await self._vault.resolve_credential_async(
                        c.auth.cred,
                    )
                except Exception as e:
                    raise ConnectorAuthError(
                        f"credential {c.auth.cred!r}: {e}",
                    ) from e
            if not token:
                raise ConnectorAuthError(
                    f"credential {c.auth.cred!r} could not be resolved "
                    "from the vault",
                )
            return {"Authorization": f"Bearer {token}"}
        if c.auth.kind == "oauth":
            if not c.auth.connection or self._vault is None:
                raise ConnectorAuthError(
                    "no OAuth connection bound — use Connect on the "
                    "Connectors page",
                )
            try:
                token = await self._vault.ensure_connection_token(
                    c.auth.connection,
                )
            except Exception as e:
                raise ConnectorAuthError(
                    f"OAuth connection {c.auth.connection!r}: {e}",
                ) from e
            if not token:
                raise ConnectorAuthError(
                    f"OAuth connection {c.auth.connection!r} did not "
                    "yield a token — reconnect",
                )
            return {"Authorization": f"Bearer {token}"}
        return {}

    async def _with_session(self, c: HttpConnector, fn: Callable) -> Any:
        """Open → initialize → fn(session) → close, in this task."""
        open_client, client_session_cls = self._client_factory()
        await _assert_public_host(c.url)
        headers = await self._headers(c)
        async with open_client(c.url, headers) as streams:
            read, write = streams[0], streams[1]
            # No sampling/elicitation callbacks — SDK default REJECTS
            # server-initiated requests (plan D15; pinned by test).
            async with client_session_cls(read, write) as session:
                await asyncio.wait_for(session.initialize(), INIT_TIMEOUT)
                return await fn(session)

    # ── sanitization (stdio parity) ──────────────────────────────

    @staticmethod
    def _sanitize_tool(tool: Any) -> dict:
        """Schema → safe dict, shaped exactly like the agent-side stdio
        registration (``parameters`` key holding the full JSON Schema)
        so the agent registers both transports through one code path.
        Remote metadata is untrusted text that reaches LLM prompts —
        same sanitization posture as ``MCPClient.start()``."""
        name = sanitize_for_prompt(str(getattr(tool, "name", "")))
        desc = sanitize_for_prompt(str(getattr(tool, "description", None) or ""))
        if len(desc.encode("utf-8", errors="ignore")) > _DESCRIPTION_MAX_BYTES:
            desc = (
                desc.encode("utf-8", errors="ignore")[:_DESCRIPTION_MAX_BYTES]
                .decode("utf-8", errors="ignore")
                + "… [truncated]"
            )
        schema = getattr(tool, "inputSchema", None) or {
            "type": "object", "properties": {},
        }

        def _clean(v: Any) -> Any:
            if isinstance(v, str):
                return sanitize_for_prompt(v)
            if isinstance(v, dict):
                return {k: _clean(x) for k, x in v.items()}
            if isinstance(v, list):
                return [_clean(x) for x in v]
            return v

        return {
            "name": name,
            "description": desc,
            "parameters": _clean(schema) if isinstance(schema, dict) else {
                "type": "object", "properties": {},
            },
        }

    @staticmethod
    def _shape_result(result: Any) -> dict:
        """MCP CallToolResult → the agent-side stdio result contract:
        ``{"result": text}`` on success, ``{"error": text}`` when the
        server flags isError — one shape end-to-end, so the agent's
        registry and loop treat both transports identically."""
        parts: list[str] = []
        for block in getattr(result, "content", None) or []:
            text = getattr(block, "text", None)
            if text is not None:
                parts.append(str(text))
            else:
                btype = getattr(block, "type", "unknown")
                parts.append(f"[non-text content: {btype}]")
        text = sanitize_for_prompt("\n".join(parts))
        if bool(getattr(result, "isError", False)):
            return {"error": text or "MCP tool returned an error"}
        return {"result": text}

    # ── public surface ───────────────────────────────────────────

    async def list_tools(self, name: str) -> list[dict]:
        """Sanitized tool schemas for one connector, cached on the
        catalog generation."""
        c = self._get_http(name)
        gen = self._store.generation()
        key = name.lower()
        cached = self._tools_cache.get(key)
        if cached is not None and cached[0] == gen:
            return cached[1]
        listed = await self._with_session(
            c,
            lambda s: asyncio.wait_for(s.list_tools(), CALL_TIMEOUT),
        )
        tools = [self._sanitize_tool(t) for t in getattr(listed, "tools", [])]
        self._tools_cache[key] = (gen, tools)
        return tools

    async def tools_for_agent(self, agent_id: str) -> dict[str, dict]:
        """Discovery for /mesh/connectors/tools — caller-scoped. A
        connector that fails discovery degrades to an error entry
        rather than failing the whole response: one broken remote must
        not strip every other connector's tools from an agent."""
        out: dict[str, dict] = {}
        for c in self._store.http_for_agent(agent_id):
            try:
                out[c.name] = {"tools": await self.list_tools(c.name)}
            except GatewayUnavailable:
                raise
            except Exception as e:
                logger.warning(
                    "Connector %r discovery failed for agent %r: %s",
                    c.name, agent_id, e,
                )
                out[c.name] = {"tools": [], "error": self._mask(e)}
        return out

    async def call_tool(
        self, name: str, tool: str, arguments: dict, *, agent_id: str,
    ) -> dict:
        """Execute one tool call. Assignment IS the authz gate — the
        operator participates like any agent (plan D11)."""
        c = self._get_http(name)
        if not c.applies_to(agent_id):
            raise PermissionError(
                f"connector {name!r} is not assigned to agent {agent_id!r}",
            )
        try:
            result = await self._with_session(
                c,
                lambda s: asyncio.wait_for(
                    s.call_tool(tool, arguments), CALL_TIMEOUT,
                ),
            )
        except (GatewayUnavailable, ConnectorAuthError, ConnectorSSRFError):
            raise
        except Exception as e:
            # Full text mesh-side, masked for the agent (LLM-proxy
            # policy — upstream error bodies can leak server internals
            # or echoed headers).
            logger.error("Connector %r call %r failed: %s", name, tool, e)
            raise RuntimeError(self._mask(e)) from None
        return _truncate_result(self._shape_result(result))

    async def probe(self, name: str) -> dict:
        """Dashboard 'Test connection': fresh initialize + discovery.
        Also repopulates the tools cache on success."""
        try:
            self.invalidate(name)
            tools = await self.list_tools(name)
            return {"ok": True, "tools_count": len(tools)}
        except UnknownConnectorError:
            return {"ok": False, "error": "unknown connector", "needs_auth": False}
        except GatewayUnavailable as e:
            return {"ok": False, "error": str(e), "needs_auth": False}
        except ConnectorAuthError as e:
            return {"ok": False, "error": str(e), "needs_auth": True}
        except ConnectorSSRFError as e:
            return {"ok": False, "error": str(e), "needs_auth": False}
        except asyncio.TimeoutError:
            return {
                "ok": False,
                "error": "server did not respond (timeout) — is the URL "
                         "reachable from the mesh host?",
                "needs_auth": False,
            }
        except Exception as e:
            needs_auth = self._looks_like_auth_failure(e)
            return {
                "ok": False,
                "error": self._mask(e),
                "needs_auth": needs_auth,
            }

    # ── error shaping ────────────────────────────────────────────

    @staticmethod
    def _looks_like_auth_failure(e: Exception) -> bool:
        text = str(e)
        return "401" in text or "Unauthorized" in text or "unauthorized" in text

    @staticmethod
    def _mask(e: Exception) -> str:
        """Agent/dashboard-safe error string: exception type + a short
        classification, never the upstream body."""
        if MCPGateway._looks_like_auth_failure(e):
            return "upstream rejected authorization (401)"
        name = type(e).__name__
        text = str(e)
        if "timed out" in text.lower() or name in ("TimeoutError", "ConnectTimeout"):
            return "upstream timed out"
        if "connect" in text.lower() or "resolve" in text.lower():
            return f"upstream unreachable ({name})"
        return f"upstream error ({name})"
