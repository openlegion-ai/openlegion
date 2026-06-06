"""Tests for credential-request cancellation (PR 3).

Covers:
  * ``POST /mesh/credential-request/{id}/cancel`` happy path.
  * Unknown / already-resolved request_id → 404.
  * Forbidden caller (non-operator, non-internal) → 403.
  * Cross-emit: ``credential_request_cancelled`` event fires.
  * Steer message enqueued to the awaiting agent (via lane_manager).
  * Agent-side ``request_credential`` returns the request_id so the
    dashboard's Cancel button can address it.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


def _build_mesh(tmp_path, *, lane_manager=None):
    from src.host.costs import CostTracker
    from src.host.mesh import Blackboard, MessageRouter, PubSub
    from src.host.permissions import PermissionMatrix
    from src.host.server import create_mesh_app
    from src.host.traces import TraceStore

    bb = Blackboard(str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix()
    # L2 gate: /mesh/credential-request enforces
    # ``can_request_user_credentials``. Grant it to ``agent-1`` so these
    # cancellation tests can reach the request endpoint.
    from src.shared.types import AgentPermissions
    for _aid in ("agent-1", "a"):
        perms.permissions[_aid] = AgentPermissions(
            agent_id=_aid, can_request_user_credentials=True,
        )
    router = MessageRouter(perms, {})
    costs = CostTracker(str(tmp_path / "costs.db"))
    traces = TraceStore(str(tmp_path / "traces.db"))
    event_bus = MagicMock()

    app = create_mesh_app(
        blackboard=bb,
        pubsub=pubsub,
        router=router,
        permissions=perms,
        cost_tracker=costs,
        trace_store=traces,
        event_bus=event_bus,
        lane_manager=lane_manager,
    )
    return app, event_bus


@pytest.mark.asyncio
async def test_request_returns_request_id(tmp_path):
    from httpx import ASGITransport, AsyncClient

    app, event_bus = _build_mesh(tmp_path)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post(
            "/mesh/credential-request",
            json={
                "agent_id": "agent-1",
                "name": "twitter_api_key",
                "description": "Your Twitter API key",
                "service": "Twitter",
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["request_id"]
    # Recorded in the in-memory registry exposed for tests
    assert data["request_id"] in app.help_requests
    rec = app.help_requests[data["request_id"]]
    assert rec["kind"] == "credential_request"
    assert rec["agent_id"] == "agent-1"
    assert rec["status"] == "open"


@pytest.mark.asyncio
async def test_cancel_happy_path_emits_and_steers(tmp_path):
    from httpx import ASGITransport, AsyncClient

    lane_manager = MagicMock()
    lane_manager.enqueue = AsyncMock()

    app, event_bus = _build_mesh(tmp_path, lane_manager=lane_manager)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post(
            "/mesh/credential-request",
            json={
                "agent_id": "agent-1",
                "name": "stripe_key",
                "description": "Your Stripe key",
            },
        )
        request_id = resp.json()["request_id"]
        event_bus.emit.reset_mock()  # ignore the request emit

        # Loopback + operator header → internal caller path
        cancel = await client.post(
            f"/mesh/credential-request/{request_id}/cancel",
            json={"reason": "user_cancelled"},
            headers={"x-mesh-internal": "1", "X-Agent-ID": "operator"},
        )
    assert cancel.status_code == 200, cancel.text
    body = cancel.json()
    assert body["ok"] is True
    assert body["status"] == "cancelled"
    assert body["request_id"] == request_id

    # The cancellation event fired with the right shape
    event_bus.emit.assert_called_once()
    call = event_bus.emit.call_args
    assert call[0][0] == "credential_request_cancelled"
    assert call[1]["agent"] == "agent-1"
    assert call[1]["data"]["request_id"] == request_id
    assert call[1]["data"]["name"] == "stripe_key"
    assert call[1]["data"]["reason"] == "user_cancelled"

    # And a steer message reached the awaiting agent so it can react
    lane_manager.enqueue.assert_awaited_once()
    enqueue_call = lane_manager.enqueue.await_args
    assert enqueue_call[0][0] == "agent-1"
    assert enqueue_call[1]["mode"] == "steer"
    assert "cancelled" in enqueue_call[0][1].lower()
    assert "stripe_key" in enqueue_call[0][1]

    # Record popped from the registry — second cancel must 404
    assert request_id not in app.help_requests


@pytest.mark.asyncio
async def test_cancel_unknown_request_id_returns_404(tmp_path):
    from httpx import ASGITransport, AsyncClient

    app, _ = _build_mesh(tmp_path)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post(
            "/mesh/credential-request/00000000-0000-0000-0000-000000000000/cancel",
            json={"reason": "user_cancelled"},
            headers={"x-mesh-internal": "1", "X-Agent-ID": "operator"},
        )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_cancel_already_cancelled_returns_404(tmp_path):
    from httpx import ASGITransport, AsyncClient

    lane_manager = MagicMock()
    lane_manager.enqueue = AsyncMock()
    app, _ = _build_mesh(tmp_path, lane_manager=lane_manager)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post(
            "/mesh/credential-request",
            json={"agent_id": "a", "name": "k", "description": "d"},
        )
        rid = resp.json()["request_id"]
        first = await client.post(
            f"/mesh/credential-request/{rid}/cancel",
            json={},
            headers={"x-mesh-internal": "1", "X-Agent-ID": "operator"},
        )
        assert first.status_code == 200
        second = await client.post(
            f"/mesh/credential-request/{rid}/cancel",
            json={},
            headers={"x-mesh-internal": "1", "X-Agent-ID": "operator"},
        )
    assert second.status_code == 404


@pytest.mark.asyncio
async def test_cancel_blocked_for_non_operator(tmp_path):
    """A worker-style caller (not operator, not loopback-internal) must
    not be able to cancel another agent's pending credential request.
    """
    from httpx import ASGITransport, AsyncClient

    app, _ = _build_mesh(tmp_path)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post(
            "/mesh/credential-request",
            json={"agent_id": "agent-1", "name": "k", "description": "d"},
        )
        rid = resp.json()["request_id"]
        # No x-mesh-internal header → not internal. X-Agent-ID is a
        # non-operator worker.
        forbid = await client.post(
            f"/mesh/credential-request/{rid}/cancel",
            json={},
            headers={"X-Agent-ID": "agent-1"},
        )
    assert forbid.status_code == 403
    # The record is still open (not popped on auth failure)
    assert rid in app.help_requests


@pytest.mark.asyncio
async def test_agent_side_request_credential_surfaces_request_id():
    """The ``request_credential`` tool must surface the server-generated
    request_id back into the tool result so downstream cancel paths can
    address it (and so the agent itself can correlate steers).
    """
    from src.agent.builtins.vault_tool import request_credential

    mc = AsyncMock()
    mc.vault_list.return_value = []
    mc.request_credential_from_user.return_value = {
        "requested": True, "name": "k", "request_id": "abc-123",
    }

    result = await request_credential(
        name="k", description="d", mesh_client=mc,
    )
    assert result["requested"] is True
    assert result["request_id"] == "abc-123"


@pytest.mark.asyncio
async def test_agent_side_request_credential_omits_request_id_when_missing():
    """Defensive: when the mesh response lacks a request_id (legacy),
    the tool result does not include the key — better to omit than
    return a confusing empty string.
    """
    from src.agent.builtins.vault_tool import request_credential

    mc = AsyncMock()
    mc.vault_list.return_value = []
    mc.request_credential_from_user.return_value = {
        "requested": True, "name": "k",
    }

    result = await request_credential(
        name="k", description="d", mesh_client=mc,
    )
    assert "request_id" not in result
