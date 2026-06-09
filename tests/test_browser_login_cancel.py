"""Tests for browser-login & captcha-help cancellation (PR 3).

Covers both ``POST /mesh/browser-login-request/{id}/cancel`` and
``POST /mesh/browser-captcha-help-request/{id}/cancel``:

  * Happy path emits ``*_cancelled`` event AND enqueues a steer to
    the awaiting agent so it can react.
  * Unknown id → 404. Already-cancelled → 404.
  * Forbidden caller → 403.
  * Agent-side ``request_browser_login`` / ``request_captcha_help``
    surface the server-generated request_id.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


def _build_mesh(tmp_path, *, lane_manager=None, perms_map=None):
    from src.host.costs import CostTracker
    from src.host.mesh import Blackboard, MessageRouter, PubSub
    from src.host.permissions import PermissionMatrix
    from src.host.server import create_mesh_app
    from src.host.traces import TraceStore
    from src.shared.types import AgentPermissions

    bb = Blackboard(str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix()
    for aid, p in (perms_map or {}).items():
        perms.permissions[aid] = AgentPermissions(agent_id=aid, **p)
    router = MessageRouter(perms, {})
    costs = CostTracker(str(tmp_path / "costs.db"))
    traces = TraceStore(str(tmp_path / "traces.db"))
    event_bus = MagicMock()

    container_manager = MagicMock()
    container_manager.browser_service_url = "http://browser-svc:8500"
    container_manager.browser_auth_token = ""

    app = create_mesh_app(
        blackboard=bb,
        pubsub=pubsub,
        router=router,
        permissions=perms,
        cost_tracker=costs,
        trace_store=traces,
        event_bus=event_bus,
        container_manager=container_manager,
        lane_manager=lane_manager,
        help_requests_db=str(tmp_path / "help_requests.db"),
    )
    return app, event_bus


# ── Browser login ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_browser_login_request_returns_request_id(tmp_path):
    from httpx import ASGITransport, AsyncClient

    app, _ = _build_mesh(
        tmp_path, perms_map={"worker": {"can_use_browser": True}},
    )
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post(
            "/mesh/browser-login-request",
            json={
                "agent_id": "worker",
                "url": "https://x.com/login",
                "service": "X",
                "description": "Log in",
            },
            headers={"X-Agent-ID": "worker"},
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["request_id"]
    assert app.help_requests_store.get(body["request_id"]) is not None


@pytest.mark.asyncio
async def test_browser_login_cancel_emits_and_steers(tmp_path):
    from httpx import ASGITransport, AsyncClient

    lane_manager = MagicMock()
    lane_manager.enqueue = AsyncMock()
    app, event_bus = _build_mesh(
        tmp_path,
        perms_map={"worker": {"can_use_browser": True}},
        lane_manager=lane_manager,
    )
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post(
            "/mesh/browser-login-request",
            json={
                "agent_id": "worker",
                "url": "https://x.com/login",
                "service": "X",
                "description": "Log in",
            },
            headers={"X-Agent-ID": "worker"},
        )
        rid = resp.json()["request_id"]
        event_bus.emit.reset_mock()

        cancel = await client.post(
            f"/mesh/browser-login-request/{rid}/cancel",
            json={"reason": "user_cancelled"},
            headers={"x-mesh-internal": "1", "X-Agent-ID": "operator"},
        )
    assert cancel.status_code == 200, cancel.text
    body = cancel.json()
    assert body["status"] == "cancelled"
    assert body["request_id"] == rid

    event_bus.emit.assert_called_once()
    call = event_bus.emit.call_args
    assert call[0][0] == "browser_login_cancelled"
    assert call[1]["agent"] == "worker"
    assert call[1]["data"]["request_id"] == rid
    assert call[1]["data"]["service"] == "X"

    lane_manager.enqueue.assert_awaited_once()
    enqueue = lane_manager.enqueue.await_args
    assert enqueue[0][0] == "worker"
    assert "X" in enqueue[0][1]
    assert enqueue[1]["mode"] == "steer"


@pytest.mark.asyncio
async def test_browser_login_cancel_unknown_id_returns_404(tmp_path):
    from httpx import ASGITransport, AsyncClient

    app, _ = _build_mesh(tmp_path)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post(
            "/mesh/browser-login-request/nope/cancel",
            json={},
            headers={"x-mesh-internal": "1", "X-Agent-ID": "operator"},
        )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_browser_login_cancel_already_cancelled_returns_404(tmp_path):
    from httpx import ASGITransport, AsyncClient

    lane_manager = MagicMock()
    lane_manager.enqueue = AsyncMock()
    app, _ = _build_mesh(
        tmp_path,
        perms_map={"worker": {"can_use_browser": True}},
        lane_manager=lane_manager,
    )
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post(
            "/mesh/browser-login-request",
            json={
                "agent_id": "worker",
                "url": "https://x.com/login",
                "service": "X",
                "description": "Log in",
            },
            headers={"X-Agent-ID": "worker"},
        )
        rid = resp.json()["request_id"]
        first = await client.post(
            f"/mesh/browser-login-request/{rid}/cancel",
            json={},
            headers={"x-mesh-internal": "1", "X-Agent-ID": "operator"},
        )
        assert first.status_code == 200
        second = await client.post(
            f"/mesh/browser-login-request/{rid}/cancel",
            json={},
            headers={"x-mesh-internal": "1", "X-Agent-ID": "operator"},
        )
    assert second.status_code == 404


@pytest.mark.asyncio
async def test_browser_login_cancel_blocked_for_non_operator(tmp_path):
    from httpx import ASGITransport, AsyncClient

    app, _ = _build_mesh(
        tmp_path, perms_map={"worker": {"can_use_browser": True}},
    )
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post(
            "/mesh/browser-login-request",
            json={
                "agent_id": "worker",
                "url": "https://x.com/login",
                "service": "X",
                "description": "Log in",
            },
            headers={"X-Agent-ID": "worker"},
        )
        rid = resp.json()["request_id"]
        forbid = await client.post(
            f"/mesh/browser-login-request/{rid}/cancel",
            json={},
            headers={"X-Agent-ID": "worker"},  # not operator, not internal
        )
    assert forbid.status_code == 403


# ── Browser CAPTCHA help ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_captcha_help_request_returns_request_id(tmp_path):
    from httpx import ASGITransport, AsyncClient

    app, _ = _build_mesh(
        tmp_path, perms_map={"worker": {"can_use_browser": True}},
    )
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post(
            "/mesh/browser-captcha-help-request",
            json={
                "agent_id": "worker",
                "service": "Cloudflare",
                "description": "Click the checkbox",
            },
            headers={"X-Agent-ID": "worker"},
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["request_id"]
    assert app.help_requests_store.get(body["request_id"]) is not None


@pytest.mark.asyncio
async def test_captcha_help_cancel_emits_and_steers(tmp_path):
    from httpx import ASGITransport, AsyncClient

    lane_manager = MagicMock()
    lane_manager.enqueue = AsyncMock()
    app, event_bus = _build_mesh(
        tmp_path,
        perms_map={"worker": {"can_use_browser": True}},
        lane_manager=lane_manager,
    )
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post(
            "/mesh/browser-captcha-help-request",
            json={
                "agent_id": "worker",
                "service": "Cloudflare",
                "description": "Click the checkbox",
            },
            headers={"X-Agent-ID": "worker"},
        )
        rid = resp.json()["request_id"]
        event_bus.emit.reset_mock()

        cancel = await client.post(
            f"/mesh/browser-captcha-help-request/{rid}/cancel",
            json={"reason": "user_cancelled"},
            headers={"x-mesh-internal": "1", "X-Agent-ID": "operator"},
        )
    assert cancel.status_code == 200, cancel.text
    event_bus.emit.assert_called_once()
    call = event_bus.emit.call_args
    assert call[0][0] == "browser_captcha_help_cancelled"
    assert call[1]["agent"] == "worker"
    assert call[1]["data"]["request_id"] == rid
    assert call[1]["data"]["service"] == "Cloudflare"

    lane_manager.enqueue.assert_awaited_once()


# ── Agent-side tools surface request_id ───────────────────────────


@pytest.mark.asyncio
async def test_request_browser_login_tool_surfaces_request_id():
    from src.agent.builtins.browser_tool import request_browser_login

    mc = AsyncMock()
    mc.browser_command = AsyncMock(return_value={"url": "https://x/login"})
    mc.request_browser_login = AsyncMock(return_value={
        "requested": True, "service": "X", "request_id": "abc",
    })
    result = await request_browser_login(
        url="https://x/login", service="X", description="d",
        mesh_client=mc,
    )
    assert result["request_id"] == "abc"


@pytest.mark.asyncio
async def test_request_browser_login_tool_omits_request_id_when_missing():
    from src.agent.builtins.browser_tool import request_browser_login

    mc = AsyncMock()
    mc.browser_command = AsyncMock(return_value={"url": "https://x/login"})
    mc.request_browser_login = AsyncMock(return_value={
        "requested": True, "service": "X",
    })
    result = await request_browser_login(
        url="https://x/login", service="X", description="d",
        mesh_client=mc,
    )
    assert "request_id" not in result


@pytest.mark.asyncio
async def test_request_captcha_help_tool_surfaces_request_id():
    from src.agent.builtins.browser_tool import request_captcha_help

    mc = AsyncMock()
    mc.browser_command = AsyncMock(return_value={})
    mc.request_captcha_help = AsyncMock(return_value={
        "requested": True, "service": "CF", "request_id": "rid-1",
    })
    result = await request_captcha_help(
        service="CF", description="d", mesh_client=mc,
    )
    assert result["request_id"] == "rid-1"
