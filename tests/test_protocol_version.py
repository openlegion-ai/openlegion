"""Wire-protocol version handshake on the mesh↔agent contract.

The mesh advertises ``X-Protocol-Version`` on every mesh→agent hop; the agent
server rejects a request whose major version is incompatible (HTTP 426) so a
rolling upgrade that leaves a stale container talking to a new mesh fails loudly
instead of silently mis-decoding JSON. The check is non-breaking: a request
with no version header is always accepted, and ``/status`` is exempt so
reachability polling never trips it.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from src.agent.server import create_agent_app
from src.shared.trace import (
    PROTOCOL_VERSION,
    PROTOCOL_VERSION_HEADER,
    protocol_compatible,
)

# ── pure compatibility function ────────────────────────────────────────────

def test_protocol_compatible_same_major():
    assert protocol_compatible(PROTOCOL_VERSION) is True


def test_protocol_compatible_missing_is_fail_open():
    # Missing/empty/whitespace-only → compatible (fail-open), so the header is
    # non-breaking for unversioned callers. (A present garbage value is NOT
    # fail-open — see test_protocol_incompatible_non_numeric.)
    assert protocol_compatible(None) is True
    assert protocol_compatible("") is True
    assert protocol_compatible("   ") is True


def test_protocol_incompatible_different_major():
    other_major = str(int(PROTOCOL_VERSION.split(".")[0]) + 1)
    assert protocol_compatible(other_major) is False


def test_protocol_compatible_ignores_minor():
    # Same major, differing minor → still compatible (additive changes).
    assert protocol_compatible(f"{PROTOCOL_VERSION.split('.')[0]}.7") is True


def test_protocol_incompatible_non_numeric():
    # A present, non-empty, unparseable value is NOT a version we speak → reject.
    # (Distinct from missing/whitespace, which is fail-open.)
    assert protocol_compatible("abc") is False


# ── transport emits the header (mesh→agent) ────────────────────────────────

def test_transport_resolve_headers_adds_version():
    from src.host.transport import HttpTransport

    t = HttpTransport()
    headers = t._resolve_headers(None)
    assert headers[PROTOCOL_VERSION_HEADER] == PROTOCOL_VERSION
    assert headers["x-mesh-internal"] == "1"


# ── agent server enforcement ───────────────────────────────────────────────

def _make_agent_app():
    loop = MagicMock()
    loop.agent_id = "test_agent"
    loop.role = "researcher"
    loop.state = "idle"
    loop._excluded_tools = frozenset()
    loop.memory = None
    loop.mesh_client = MagicMock()
    loop.tools = MagicMock()
    loop.tools.list_tools = MagicMock(return_value=[])
    loop.tools.get_tool_definitions = MagicMock(return_value=[])
    loop.tools.get_tool_sources = MagicMock(return_value={})
    loop.tools.execute = AsyncMock(return_value={"ok": True})
    loop.workspace = None
    # /status serializes loop.get_status().model_dump(); give it a real dict so
    # the endpoint responds cleanly (needed by the exemption test).
    loop.get_status.return_value.model_dump.return_value = {"state": "idle"}
    return create_agent_app(loop)


@pytest.mark.asyncio
async def test_agent_rejects_incompatible_mesh_version():
    """x-mesh-internal + incompatible major version → 426."""
    app = _make_agent_app()
    bad = str(int(PROTOCOL_VERSION.split(".")[0]) + 1)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.get(
            "/capabilities",
            headers={"x-mesh-internal": "1", PROTOCOL_VERSION_HEADER: bad},
        )
    assert resp.status_code == 426, resp.text
    assert "protocol version mismatch" in resp.text.lower()


@pytest.mark.asyncio
async def test_agent_accepts_matching_version():
    app = _make_agent_app()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.get(
            "/capabilities",
            headers={"x-mesh-internal": "1", PROTOCOL_VERSION_HEADER: PROTOCOL_VERSION},
        )
    assert resp.status_code == 200, resp.text


@pytest.mark.asyncio
async def test_agent_accepts_missing_version_header():
    """No version header → allowed (non-breaking for unversioned callers)."""
    app = _make_agent_app()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.get("/capabilities", headers={"x-mesh-internal": "1"})
    assert resp.status_code == 200, resp.text


@pytest.mark.asyncio
async def test_status_exempt_from_version_check():
    """Reachability probe (/status) is never rejected, even on mismatch."""
    app = _make_agent_app()
    bad = str(int(PROTOCOL_VERSION.split(".")[0]) + 1)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.get(
            "/status",
            headers={"x-mesh-internal": "1", PROTOCOL_VERSION_HEADER: bad},
        )
    # The guard must not reject /status even on a version mismatch; the request
    # reaches routing (a non-426 status proves the middleware let it through —
    # the endpoint's own response shape is irrelevant to the exemption).
    assert resp.status_code != 426, resp.text


@pytest.mark.asyncio
async def test_agent_selfcall_without_internal_marker_not_checked():
    """No x-mesh-internal (agent self-call) → version check skipped entirely."""
    app = _make_agent_app()
    bad = str(int(PROTOCOL_VERSION.split(".")[0]) + 1)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.get(
            "/capabilities",
            headers={PROTOCOL_VERSION_HEADER: bad},
        )
    assert resp.status_code == 200, resp.text
