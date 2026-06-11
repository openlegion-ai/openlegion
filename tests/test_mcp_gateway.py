"""Tests for the mesh-side MCP gateway (Phase 2b).

Everything runs through the ``client_factory`` seam with fakes — no
``mcp`` SDK required (the suite must pass on hosts with and without it
installed). The pins here are the plan's load-bearing decisions:
per-call sessions (D9), the operator-included assignment gate (D11),
generation-keyed discovery cache + explicit invalidate (D12), result
byte-cap (D13), no server-initiated callbacks (D15), and the
resolved-IP SSRF blocklist (D16).
"""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.host.connectors import ConnectorStore
from src.host.mcp_gateway import (
    RESULT_MAX_BYTES,
    ConnectorAuthError,
    ConnectorSSRFError,
    GatewayUnavailable,
    MCPGateway,
    UnknownConnectorError,
    _assert_public_host,
    _default_client_factory,
)
from src.shared.types import ConnectorAuth, HttpConnector, MCPConnector


def _tool(name="create_issue", desc="Create an issue", schema=None):
    return SimpleNamespace(
        name=name, description=desc,
        inputSchema=schema or {"type": "object", "properties": {}},
    )


def _text_result(text="done", is_error=False):
    return SimpleNamespace(
        content=[SimpleNamespace(text=text)], isError=is_error,
    )


class FakeSession:
    """Stands in for mcp ClientSession. Records construction args so
    the no-callbacks pin (D15) can assert on them."""

    instances: list["FakeSession"] = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.entered = False
        self.exited = False
        self.behavior = FakeSession.current_behavior
        FakeSession.instances.append(self)

    current_behavior: dict = {}

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, *exc):
        self.exited = True
        return False

    async def initialize(self):
        b = self.behavior.get("initialize")
        if isinstance(b, Exception):
            raise b
        if b == "hang":
            await asyncio.sleep(3600)

    async def list_tools(self):
        b = self.behavior.get("list_tools")
        if isinstance(b, Exception):
            raise b
        return SimpleNamespace(tools=b if b is not None else [_tool()])

    async def call_tool(self, tool, arguments):
        b = self.behavior.get("call_tool")
        if isinstance(b, Exception):
            raise b
        return b if b is not None else _text_result()


class FakeOpenClient:
    """Stands in for streamablehttp_client(...) — records opens/closes
    and the headers passed (so auth-injection is assertable)."""

    opened: list[dict] = []
    closed: int = 0

    def __init__(self, url, headers):
        self.url = url
        self.headers = headers

    async def __aenter__(self):
        FakeOpenClient.opened.append(
            {"url": self.url, "headers": dict(self.headers or {})},
        )
        return (MagicMock(name="read"), MagicMock(name="write"), None)

    async def __aexit__(self, *exc):
        FakeOpenClient.closed += 1
        return False


@pytest.fixture(autouse=True)
def _reset_fakes():
    FakeSession.instances = []
    FakeSession.current_behavior = {}
    FakeOpenClient.opened = []
    FakeOpenClient.closed = 0
    yield


def _factory():
    return (lambda url, headers: FakeOpenClient(url, headers)), FakeSession


def _gateway(tmp_path, *, vault=None, agents=("*",), auth=None):
    store = ConnectorStore(str(tmp_path / "connectors.json"))
    store.upsert(HttpConnector(
        transport="http", name="linear",
        url="https://93.184.216.34/mcp",
        auth=auth or ConnectorAuth(),
        agents=list(agents),
    ))
    return MCPGateway(store, vault, client_factory=_factory), store


# ── per-call sessions (D9) ───────────────────────────────────


class TestPerCallSessions:
    @pytest.mark.asyncio
    async def test_each_call_opens_and_closes_one_session(self, tmp_path):
        gw, _ = _gateway(tmp_path)
        await gw.call_tool("linear", "t", {}, agent_id="a")
        await gw.call_tool("linear", "t", {}, agent_id="a")
        assert len(FakeOpenClient.opened) == 2
        assert FakeOpenClient.closed == 2
        assert all(s.entered and s.exited for s in FakeSession.instances)

    @pytest.mark.asyncio
    async def test_concurrent_calls_use_independent_sessions(self, tmp_path):
        gw, _ = _gateway(tmp_path)
        results = await asyncio.gather(*(
            gw.call_tool("linear", "t", {}, agent_id="a") for _ in range(5)
        ))
        assert len(results) == 5
        assert len(FakeOpenClient.opened) == 5
        assert FakeOpenClient.closed == 5

    @pytest.mark.asyncio
    async def test_no_server_initiated_callbacks_registered(self, tmp_path):
        # D15 pin: ClientSession constructed with the two streams ONLY
        # — no sampling/elicitation callbacks. The SDK default then
        # REJECTS server-initiated requests; passing any callback here
        # must fail this test loudly.
        gw, _ = _gateway(tmp_path)
        await gw.call_tool("linear", "t", {}, agent_id="a")
        (session,) = FakeSession.instances
        assert len(session.args) == 2
        assert session.kwargs == {}


# ── authz (D11) ──────────────────────────────────────────────


class TestAssignmentGate:
    @pytest.mark.asyncio
    async def test_unassigned_agent_denied(self, tmp_path):
        gw, _ = _gateway(tmp_path, agents=("alpha",))
        with pytest.raises(PermissionError):
            await gw.call_tool("linear", "t", {}, agent_id="operator")
        assert FakeOpenClient.opened == []  # denied before any I/O

    @pytest.mark.asyncio
    async def test_unknown_or_stdio_connector_rejected(self, tmp_path):
        gw, store = _gateway(tmp_path)
        store.upsert(MCPConnector(name="fs", command="c", agents=["*"]))
        with pytest.raises(UnknownConnectorError):
            await gw.call_tool("ghost", "t", {}, agent_id="a")
        with pytest.raises(UnknownConnectorError):
            await gw.call_tool("fs", "t", {}, agent_id="a")


# ── auth resolution ──────────────────────────────────────────


class TestAuthResolution:
    def _vault(self, token="tok-123"):
        vault = MagicMock()
        vault.resolve_credential_async = AsyncMock(return_value=token)
        vault.ensure_connection_token = AsyncMock(return_value=token)
        return vault

    @pytest.mark.asyncio
    async def test_bearer_header_injected_per_call(self, tmp_path):
        vault = self._vault()
        gw, _ = _gateway(
            tmp_path, vault=vault,
            auth=ConnectorAuth(kind="bearer", cred="linear_token"),
        )
        await gw.call_tool("linear", "t", {}, agent_id="a")
        vault.resolve_credential_async.assert_awaited_once_with("linear_token")
        assert FakeOpenClient.opened[0]["headers"] == {
            "Authorization": "Bearer tok-123",
        }

    @pytest.mark.asyncio
    async def test_missing_bearer_cred_is_auth_error(self, tmp_path):
        gw, _ = _gateway(
            tmp_path, vault=self._vault(token=None),
            auth=ConnectorAuth(kind="bearer", cred="linear_token"),
        )
        with pytest.raises(ConnectorAuthError):
            await gw.call_tool("linear", "t", {}, agent_id="a")
        probe = await gw.probe("linear")
        assert probe == {
            "ok": False,
            "error": probe["error"],
            "needs_auth": True,
        }

    @pytest.mark.asyncio
    async def test_vault_raises_classify_as_auth_not_generic(self, tmp_path):
        # The vault RAISES on missing/dead connections (RuntimeError
        # "No connection", ConnectionRefreshError on revoked grants) —
        # those must surface as needs_auth on probe, not as a masked
        # generic upstream error, or the reconnect affordance never
        # renders exactly when it's needed.
        vault = MagicMock()
        vault.ensure_connection_token = AsyncMock(
            side_effect=RuntimeError("No connection: mcp_linear"),
        )
        gw, store = _gateway(tmp_path, vault=vault)
        bound = store.get("linear").model_copy(update={
            "auth": ConnectorAuth(kind="oauth", connection="mcp_linear"),
        })
        store.upsert(bound)
        with pytest.raises(ConnectorAuthError):
            await gw.call_tool("linear", "t", {}, agent_id="a")
        probe = await gw.probe("linear")
        assert probe["ok"] is False and probe["needs_auth"] is True

    @pytest.mark.asyncio
    async def test_oauth_unbound_connection_is_auth_error(self, tmp_path):
        gw, store = _gateway(tmp_path, vault=self._vault())
        # Hand-built record with kind=oauth but no connection yet.
        c = store.get("linear").model_copy(update={
            "auth": ConnectorAuth(kind="oauth"),
        })
        store.upsert(c)
        with pytest.raises(ConnectorAuthError):
            await gw.call_tool("linear", "t", {}, agent_id="a")

    @pytest.mark.asyncio
    async def test_none_auth_sends_no_header(self, tmp_path):
        gw, _ = _gateway(tmp_path)
        await gw.call_tool("linear", "t", {}, agent_id="a")
        assert FakeOpenClient.opened[0]["headers"] == {}


# ── discovery cache (D12) ────────────────────────────────────


class TestDiscoveryCache:
    @pytest.mark.asyncio
    async def test_cached_until_generation_bumps(self, tmp_path):
        gw, store = _gateway(tmp_path)
        await gw.list_tools("linear")
        await gw.list_tools("linear")
        assert len(FakeOpenClient.opened) == 1  # second hit cached
        # URL edit → restart-relevant → generation bump via mark_dirty
        moved = store.get("linear").model_copy(
            update={"url": "https://93.184.216.34:8443/mcp"},
        )
        assert store.upsert(moved) is True
        store.mark_dirty(["a"])
        await gw.list_tools("linear")
        assert len(FakeOpenClient.opened) == 2  # re-discovered

    @pytest.mark.asyncio
    async def test_auth_only_edit_needs_explicit_invalidate(self, tmp_path):
        gw, store = _gateway(tmp_path)
        await gw.list_tools("linear")
        rotated = store.get("linear").model_copy(update={
            "auth": ConnectorAuth(kind="bearer", cred="tok"),
        })
        assert store.upsert(rotated) is False  # no generation bump
        # Cache survives the auth edit (generation unchanged) — that is
        # why the dashboard MUST call invalidate() on auth edits.
        vault = MagicMock()
        vault.resolve_credential_async = AsyncMock(return_value="t")
        gw._vault = vault
        await gw.list_tools("linear")
        assert len(FakeOpenClient.opened) == 1
        gw.invalidate("linear")
        await gw.list_tools("linear")
        assert len(FakeOpenClient.opened) == 2

    @pytest.mark.asyncio
    async def test_tools_for_agent_degrades_per_connector(self, tmp_path):
        gw, store = _gateway(tmp_path)
        store.upsert(HttpConnector(
            transport="http", name="broken",
            url="https://93.184.216.99/mcp", agents=["*"],
        ))
        calls = {"n": 0}
        real_list = gw.list_tools

        async def flaky(name):
            if name == "broken":
                raise RuntimeError("boom with secret body")
            return await real_list(name)

        gw.list_tools = flaky
        out = await gw.tools_for_agent("a")
        assert out["linear"]["tools"]
        assert out["broken"]["tools"] == []
        assert "secret body" not in out["broken"]["error"]


# ── result shaping & cap (D13) ───────────────────────────────


class TestResults:
    @pytest.mark.asyncio
    async def test_text_result_matches_stdio_contract(self, tmp_path):
        # Same {"result"}/{"error"} shape as MCPClient.call_tool so the
        # agent registry treats both transports identically.
        FakeSession.current_behavior = {"call_tool": _text_result("hello")}
        gw, _ = _gateway(tmp_path)
        out = await gw.call_tool("linear", "t", {}, agent_id="a")
        assert out == {"result": "hello"}
        FakeSession.current_behavior = {
            "call_tool": _text_result("bad input", is_error=True),
        }
        out = await gw.call_tool("linear", "t", {}, agent_id="a")
        assert out == {"error": "bad input"}

    @pytest.mark.asyncio
    async def test_oversize_result_truncated_with_flag(self, tmp_path):
        FakeSession.current_behavior = {
            "call_tool": _text_result("x" * (RESULT_MAX_BYTES * 2)),
        }
        gw, _ = _gateway(tmp_path)
        out = await gw.call_tool("linear", "t", {}, agent_id="a")
        assert out["truncated"] is True
        assert len(out["result"].encode()) <= RESULT_MAX_BYTES

    @pytest.mark.asyncio
    async def test_upstream_error_masked_logged_full(self, tmp_path):
        FakeSession.current_behavior = {
            "call_tool": RuntimeError("500 body: internal stack trace XYZ"),
        }
        gw, _ = _gateway(tmp_path)
        with pytest.raises(RuntimeError) as ei:
            await gw.call_tool("linear", "t", {}, agent_id="a")
        assert "XYZ" not in str(ei.value)

    @pytest.mark.asyncio
    async def test_tool_metadata_sanitized_and_capped(self, tmp_path):
        # Byte cap + suffix mirror the agent-side stdio path
        # (mcp_client._cap_description), and the schema is emitted
        # under "parameters" — the registry's stdio key.
        FakeSession.current_behavior = {"list_tools": [
            _tool(desc="d" * 20000),
        ]}
        gw, _ = _gateway(tmp_path)
        (tool,) = await gw.list_tools("linear")
        assert tool["description"].endswith("… [truncated]")
        assert len(tool["description"].encode()) <= 8 * 1024 + 32
        assert tool["parameters"] == {"type": "object", "properties": {}}


# ── probe ────────────────────────────────────────────────────


class TestProbe:
    @pytest.mark.asyncio
    async def test_ok_with_tool_count(self, tmp_path):
        gw, _ = _gateway(tmp_path)
        assert await gw.probe("linear") == {"ok": True, "tools_count": 1}

    @pytest.mark.asyncio
    async def test_unauthorized_maps_to_needs_auth(self, tmp_path):
        FakeSession.current_behavior = {
            "initialize": RuntimeError("HTTP 401 Unauthorized"),
        }
        gw, _ = _gateway(tmp_path)
        probe = await gw.probe("linear")
        assert probe["ok"] is False and probe["needs_auth"] is True

    @pytest.mark.asyncio
    async def test_unknown_connector(self, tmp_path):
        gw, _ = _gateway(tmp_path)
        probe = await gw.probe("ghost")
        assert probe["ok"] is False and probe["needs_auth"] is False


# ── SSRF (D16) ───────────────────────────────────────────────


class TestSSRF:
    @pytest.mark.asyncio
    async def test_private_literal_rejected_before_io(self, tmp_path):
        gw, store = _gateway(tmp_path)
        store.upsert(HttpConnector(
            transport="http", name="internal",
            url="https://10.0.0.5/mcp", agents=["*"],
        ))
        with pytest.raises(ConnectorSSRFError):
            await gw.call_tool("internal", "t", {}, agent_id="a")
        assert FakeOpenClient.opened == []

    @pytest.mark.asyncio
    async def test_metadata_and_cgnat_and_v6_rejected(self, tmp_path):
        for bad in ("169.254.169.254", "100.64.0.7", "[fd00::1]"):
            with pytest.raises(ConnectorSSRFError):
                await _assert_public_host(f"https://{bad}/mcp")

    @pytest.mark.asyncio
    async def test_explicit_loopback_allowed(self, tmp_path):
        await _assert_public_host("http://localhost:9000/mcp")
        await _assert_public_host("http://127.0.0.1:9000/mcp")

    @pytest.mark.asyncio
    async def test_v4_mapped_v6_unwrapped(self, tmp_path):
        with pytest.raises(ConnectorSSRFError):
            await _assert_public_host("https://[::ffff:192.168.1.1]/mcp")


# ── SDK-missing degrade (D10) ────────────────────────────────


class TestSdkMissing:
    def test_default_factory_raises_actionable(self, monkeypatch):
        # Force the import to fail regardless of whether the SDK is
        # installed on this host (None in sys.modules → ImportError).
        monkeypatch.setitem(sys.modules, "mcp", None)
        monkeypatch.setitem(sys.modules, "mcp.client.session", None)
        monkeypatch.setitem(sys.modules, "mcp.client.streamable_http", None)
        with pytest.raises(GatewayUnavailable) as ei:
            _default_client_factory()
        assert "install" in str(ei.value)
