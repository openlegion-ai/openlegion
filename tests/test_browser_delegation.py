"""Tests for browser login delegation.

Covers:
- ``request_browser_login`` skill passes ``agent_id`` through as
  ``target_agent_id`` to the mesh client.
- ``MeshClient.browser_command`` / ``request_browser_login`` include
  ``target_agent_id`` in the request body when set.
- ``POST /mesh/browser/command`` delegation: self path, delegation path,
  blocked by ``can_message``, blocked by missing target browser.
- ``POST /mesh/browser-login-request`` delegation: self path, delegation
  path (event emitted under target's identity), blocked paths.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

# ── Skill tests ─────────────────────────────────────────────────────


class TestRequestBrowserLoginSkill:
    """src/agent/builtins/browser_tool.request_browser_login."""

    @pytest.mark.asyncio
    async def test_no_agent_id_uses_self_browser(self):
        from src.agent.builtins.browser_tool import request_browser_login

        mc = AsyncMock()
        mc.browser_command = AsyncMock(return_value={"url": "https://x.com/login"})
        mc.request_browser_login = AsyncMock(return_value={"requested": True})

        result = await request_browser_login(
            url="https://x.com/login",
            service="X",
            description="Log in to X",
            mesh_client=mc,
        )

        # browser_command called with target_agent_id=None
        mc.browser_command.assert_awaited_once_with(
            "navigate", {"url": "https://x.com/login"}, target_agent_id=None,
        )
        # request_browser_login called with target_agent_id=None
        mc.request_browser_login.assert_awaited_once_with(
            url="https://x.com/login",
            service="X",
            description="Log in to X",
            target_agent_id=None,
        )
        assert result["requested"] is True
        assert result["target_agent"] is None

    @pytest.mark.asyncio
    async def test_agent_id_delegates_to_target(self):
        from src.agent.builtins.browser_tool import request_browser_login

        mc = AsyncMock()
        mc.browser_command = AsyncMock(return_value={"url": "https://x.com/login"})
        mc.request_browser_login = AsyncMock(return_value={"requested": True})

        result = await request_browser_login(
            url="https://x.com/login",
            service="X",
            description="Log in to X for social-manager",
            agent_id="social-manager",
            mesh_client=mc,
        )

        mc.browser_command.assert_awaited_once_with(
            "navigate",
            {"url": "https://x.com/login"},
            target_agent_id="social-manager",
        )
        mc.request_browser_login.assert_awaited_once_with(
            url="https://x.com/login",
            service="X",
            description="Log in to X for social-manager",
            target_agent_id="social-manager",
        )
        assert result["requested"] is True
        assert result["target_agent"] == "social-manager"

    @pytest.mark.asyncio
    async def test_whitespace_agent_id_treated_as_self(self):
        from src.agent.builtins.browser_tool import request_browser_login

        mc = AsyncMock()
        mc.browser_command = AsyncMock(return_value={})
        mc.request_browser_login = AsyncMock(return_value={"requested": True})

        await request_browser_login(
            url="https://x.com/login",
            service="X",
            description="Log in",
            agent_id="   ",
            mesh_client=mc,
        )

        mc.browser_command.assert_awaited_once_with(
            "navigate", {"url": "https://x.com/login"}, target_agent_id=None,
        )

    @pytest.mark.asyncio
    async def test_navigate_failure_returns_error(self):
        from src.agent.builtins.browser_tool import request_browser_login

        mc = AsyncMock()
        mc.browser_command = AsyncMock(side_effect=Exception("denied"))

        result = await request_browser_login(
            url="https://x.com/login",
            service="X",
            description="Log in",
            agent_id="social-manager",
            mesh_client=mc,
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_none_agent_id_treated_as_self(self):
        """LLMs sometimes pass null instead of omitting — must not crash."""
        from src.agent.builtins.browser_tool import request_browser_login

        mc = AsyncMock()
        mc.browser_command = AsyncMock(return_value={})
        mc.request_browser_login = AsyncMock(return_value={"requested": True})

        # Simulate LLM passing agent_id=None via the **_kw kwargs path
        result = await request_browser_login(
            url="https://x.com/login",
            service="X",
            description="Log in",
            agent_id=None,  # type: ignore[arg-type]
            mesh_client=mc,
        )
        mc.browser_command.assert_awaited_once_with(
            "navigate", {"url": "https://x.com/login"}, target_agent_id=None,
        )
        assert result["requested"] is True
        assert result["target_agent"] is None

    @pytest.mark.asyncio
    async def test_empty_string_agent_id_treated_as_self(self):
        """Explicit empty string should behave identically to omission."""
        from src.agent.builtins.browser_tool import request_browser_login

        mc = AsyncMock()
        mc.browser_command = AsyncMock(return_value={})
        mc.request_browser_login = AsyncMock(return_value={"requested": True})

        await request_browser_login(
            url="https://x.com/login",
            service="X",
            description="Log in",
            agent_id="",
            mesh_client=mc,
        )
        mc.browser_command.assert_awaited_once_with(
            "navigate", {"url": "https://x.com/login"}, target_agent_id=None,
        )


# ── MeshClient tests ────────────────────────────────────────────────


class TestMeshClientBrowserDelegation:
    """MeshClient.browser_command / request_browser_login delegation params."""

    @pytest.mark.asyncio
    async def test_browser_command_no_target(self):
        from src.agent.mesh_client import MeshClient

        mc = MeshClient("http://localhost:8420", "operator")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"ok": True}

        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response
        mock_http.is_closed = False
        mc._client = mock_http

        await mc.browser_command("navigate", {"url": "https://x.com"})

        body = mock_http.post.call_args[1]["json"]
        assert body["agent_id"] == "operator"
        assert body["action"] == "navigate"
        assert "target_agent_id" not in body

    @pytest.mark.asyncio
    async def test_browser_command_with_target(self):
        from src.agent.mesh_client import MeshClient

        mc = MeshClient("http://localhost:8420", "operator")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"ok": True}

        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response
        mock_http.is_closed = False
        mc._client = mock_http

        await mc.browser_command(
            "navigate", {"url": "https://x.com"}, target_agent_id="social-manager",
        )

        body = mock_http.post.call_args[1]["json"]
        assert body["agent_id"] == "operator"
        assert body["target_agent_id"] == "social-manager"

    @pytest.mark.asyncio
    async def test_request_browser_login_no_target(self):
        from src.agent.mesh_client import MeshClient

        mc = MeshClient("http://localhost:8420", "worker")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"requested": True}

        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response
        mock_http.is_closed = False
        mc._client = mock_http

        await mc.request_browser_login(
            url="https://x.com/login", service="X", description="Log in",
        )

        body = mock_http.post.call_args[1]["json"]
        assert body["agent_id"] == "worker"
        assert "target_agent_id" not in body

    @pytest.mark.asyncio
    async def test_request_browser_login_with_target(self):
        from src.agent.mesh_client import MeshClient

        mc = MeshClient("http://localhost:8420", "operator")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"requested": True}

        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response
        mock_http.is_closed = False
        mc._client = mock_http

        await mc.request_browser_login(
            url="https://x.com/login",
            service="X",
            description="Log in",
            target_agent_id="social-manager",
        )

        body = mock_http.post.call_args[1]["json"]
        assert body["agent_id"] == "operator"
        assert body["target_agent_id"] == "social-manager"


# ── Mesh endpoint tests ─────────────────────────────────────────────


def _build_app(tmp_path, *, perms_map):
    """Build a mesh app with a permissions matrix from a dict and a fake
    container_manager whose browser service URL is reachable via mock.
    """
    from src.host.costs import CostTracker
    from src.host.mesh import Blackboard, MessageRouter, PubSub
    from src.host.permissions import PermissionMatrix
    from src.host.server import create_mesh_app
    from src.host.traces import TraceStore
    from src.shared.types import AgentPermissions

    bb_path = tmp_path / "bb.db"
    costs_path = tmp_path / "costs.db"
    traces_path = tmp_path / "traces.db"

    blackboard = Blackboard(str(bb_path))
    pubsub = PubSub()
    permissions = PermissionMatrix()
    # Seed permissions directly
    for aid, perms in perms_map.items():
        permissions.permissions[aid] = AgentPermissions(agent_id=aid, **perms)

    router = MessageRouter(permissions, {})
    costs = CostTracker(str(costs_path))
    traces = TraceStore(str(traces_path))

    # Fake container_manager with a browser_service_url — we'll mock the
    # HTTP proxy client inside create_mesh_app via monkeypatching the
    # module-level _browser_proxy_client on the app's closure.
    container_manager = MagicMock()
    container_manager.browser_service_url = "http://browser-svc:8500"
    container_manager.browser_auth_token = ""

    event_bus = MagicMock()

    app = create_mesh_app(
        blackboard=blackboard,
        pubsub=pubsub,
        router=router,
        permissions=permissions,
        cost_tracker=costs,
        trace_store=traces,
        event_bus=event_bus,
        container_manager=container_manager,
    )
    return app, event_bus, container_manager


class TestBrowserCommandEndpoint:
    """POST /mesh/browser/command delegation matrix."""

    @pytest.mark.asyncio
    async def test_self_path_success(self, tmp_path, monkeypatch):
        """Caller has browser, no target_agent_id → self path success."""
        from httpx import ASGITransport, AsyncClient, Response

        app, _event_bus, _cm = _build_app(
            tmp_path,
            perms_map={
                "worker": {"can_use_browser": True},
            },
        )

        # Only intercept calls to the fake browser service URL; let the
        # test's AsyncClient (talking to the ASGI app) use the real .post.
        import httpx
        proxy_url_seen: dict = {}
        real_post = httpx.AsyncClient.post

        async def fake_post(self, url, *args, **kwargs):
            if "browser-svc" in str(url):
                proxy_url_seen["url"] = str(url)
                req = httpx.Request("POST", str(url))
                return Response(200, json={"navigated": True}, request=req)
            return await real_post(self, url, *args, **kwargs)

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/mesh/browser/command",
                json={"action": "snapshot", "params": {}},
                headers={"X-Agent-ID": "worker"},
            )
        assert resp.status_code == 200, resp.text
        # Browser service URL routes to worker's own profile
        assert "/browser/worker/snapshot" in proxy_url_seen["url"]

    @pytest.mark.asyncio
    async def test_self_path_denied_without_browser(self, tmp_path):
        """Caller has no browser, no target → 403."""
        from httpx import ASGITransport, AsyncClient

        app, _event_bus, _cm = _build_app(
            tmp_path,
            perms_map={
                "operator": {"can_use_browser": False, "can_message": ["*"]},
            },
        )

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/mesh/browser/command",
                json={"action": "snapshot", "params": {}},
                headers={"X-Agent-ID": "operator"},
            )
        assert resp.status_code == 403
        # Per Phase 1.1 (per-action permission contract), denial message
        # names the specific action rather than the generic "browser access"
        # phrase — agents and operators get actionable feedback.
        assert "denied" in resp.text.lower()
        assert "snapshot" in resp.text

    @pytest.mark.asyncio
    async def test_delegation_happy_path(self, tmp_path, monkeypatch):
        """Operator (no browser, can_message=[*]) delegates to worker (has browser)."""
        from httpx import ASGITransport, AsyncClient, Response

        app, _event_bus, _cm = _build_app(
            tmp_path,
            perms_map={
                "operator": {"can_use_browser": False, "can_message": ["*"]},
                "social-manager": {"can_use_browser": True},
            },
        )

        import httpx
        proxy_url_seen: dict = {}
        real_post = httpx.AsyncClient.post

        async def fake_post(self, url, *args, **kwargs):
            if "browser-svc" in str(url):
                proxy_url_seen["url"] = str(url)
                req = httpx.Request("POST", str(url))
                return Response(200, json={"navigated": True}, request=req)
            return await real_post(self, url, *args, **kwargs)

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/mesh/browser/command",
                json={
                    "action": "navigate",
                    "params": {"url": "https://x.com/login"},
                    "target_agent_id": "social-manager",
                },
                headers={"X-Agent-ID": "operator"},
            )
        assert resp.status_code == 200, resp.text
        # Routed to the worker's browser profile, not operator
        assert "/browser/social-manager/navigate" in proxy_url_seen["url"]
        assert "operator" not in proxy_url_seen["url"].split("/browser/", 1)[1]

    @pytest.mark.asyncio
    async def test_delegation_blocked_by_can_message(self, tmp_path):
        """Caller can't message target → 403 (not a delegation allowed)."""
        from httpx import ASGITransport, AsyncClient

        app, _event_bus, _cm = _build_app(
            tmp_path,
            perms_map={
                "worker1": {
                    "can_use_browser": False,
                    "can_message": ["worker1"],  # self only
                },
                "worker2": {"can_use_browser": True},
            },
        )

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/mesh/browser/command",
                json={
                    "action": "snapshot",
                    "params": {},
                    "target_agent_id": "worker2",
                },
                headers={"X-Agent-ID": "worker1"},
            )
        assert resp.status_code == 403
        assert "allowlist" in resp.text.lower()

    @pytest.mark.asyncio
    async def test_delegation_blocked_when_target_has_no_browser(self, tmp_path):
        """Target has no browser → 403."""
        from httpx import ASGITransport, AsyncClient

        app, _event_bus, _cm = _build_app(
            tmp_path,
            perms_map={
                "operator": {"can_use_browser": False, "can_message": ["*"]},
                "researcher": {"can_use_browser": False},
            },
        )

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/mesh/browser/command",
                json={
                    "action": "snapshot",
                    "params": {},
                    "target_agent_id": "researcher",
                },
                headers={"X-Agent-ID": "operator"},
            )
        assert resp.status_code == 403
        assert "no browser access" in resp.text.lower()

    @pytest.mark.asyncio
    async def test_target_equals_caller_uses_self_path(self, tmp_path, monkeypatch):
        """target_agent_id == caller_id should be treated as the self path,
        not a delegation (no can_message check needed against self)."""
        from httpx import ASGITransport, AsyncClient, Response

        app, _event_bus, _cm = _build_app(
            tmp_path,
            perms_map={
                # Worker can_message intentionally does NOT include itself —
                # the self path must not require can_message against self.
                "worker": {"can_use_browser": True, "can_message": []},
            },
        )

        import httpx
        proxy_url_seen: dict = {}
        real_post = httpx.AsyncClient.post

        async def fake_post(self, url, *args, **kwargs):
            if "browser-svc" in str(url):
                proxy_url_seen["url"] = str(url)
                req = httpx.Request("POST", str(url))
                return Response(200, json={"navigated": True}, request=req)
            return await real_post(self, url, *args, **kwargs)

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/mesh/browser/command",
                json={
                    "action": "snapshot",
                    "params": {},
                    "target_agent_id": "worker",
                },
                headers={"X-Agent-ID": "worker"},
            )
        assert resp.status_code == 200, resp.text
        assert "/browser/worker/snapshot" in proxy_url_seen["url"]

    @pytest.mark.asyncio
    async def test_delegation_respects_target_per_action_policy(self, tmp_path):
        """Per-action gate applies to the EFFECTIVE TARGET, not the caller.

        Regression guard: an operator with ``browser_actions=['*']`` could
        otherwise delegate any action to a worker whose own policy restricts
        what actions are allowed against its profile. The gate must enforce
        the target's policy before ever proxying to the browser service.

        We deliberately do NOT monkey-patch httpx here: the 403 must fire
        inside the mesh before any proxy attempt. (Monkey-patching
        httpx.AsyncClient.send would also short-circuit the TEST client's
        ASGITransport path, hiding the real behavior.)
        """
        from httpx import ASGITransport, AsyncClient

        app, _event_bus, _cm = _build_app(
            tmp_path,
            perms_map={
                "operator": {
                    "can_use_browser": True,
                    "browser_actions": ["*"],
                    "can_message": ["*"],
                },
                "worker": {
                    "can_use_browser": True,
                    # Worker's profile allows navigation only — no snapshot.
                    "browser_actions": ["navigate"],
                },
            },
        )

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            # Operator delegating snapshot to worker — worker's policy denies it.
            resp = await client.post(
                "/mesh/browser/command",
                json={
                    "action": "snapshot",
                    "params": {},
                    "target_agent_id": "worker",
                },
                headers={"X-Agent-ID": "operator"},
            )

        assert resp.status_code == 403
        assert "snapshot" in resp.text

    @pytest.mark.asyncio
    async def test_target_equals_caller_self_denied_without_browser(self, tmp_path):
        """target_agent_id == caller_id with caller lacking browser → 403."""
        from httpx import ASGITransport, AsyncClient

        app, _event_bus, _cm = _build_app(
            tmp_path,
            perms_map={
                "operator": {"can_use_browser": False, "can_message": ["*"]},
            },
        )

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/mesh/browser/command",
                json={
                    "action": "snapshot",
                    "params": {},
                    "target_agent_id": "operator",
                },
                headers={"X-Agent-ID": "operator"},
            )
        assert resp.status_code == 403
        # Per-action denial message (Phase 1.1 contract).
        assert "denied" in resp.text.lower()
        assert "snapshot" in resp.text

    @pytest.mark.asyncio
    async def test_unknown_target_blocked(self, tmp_path):
        """Delegation to an agent ID with no permissions row → 403.

        Defends against the case where a permissive ``default`` template
        could otherwise let an attacker hand back any string.
        """
        from httpx import ASGITransport, AsyncClient

        app, _event_bus, _cm = _build_app(
            tmp_path,
            perms_map={
                "operator": {"can_use_browser": False, "can_message": ["*"]},
            },
        )

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/mesh/browser/command",
                json={
                    "action": "snapshot",
                    "params": {},
                    "target_agent_id": "ghost-agent",
                },
                headers={"X-Agent-ID": "operator"},
            )
        # The unknown agent has no permissions row, so can_use_browser
        # falls back to AgentPermissions default (False).
        assert resp.status_code == 403
        assert "no browser access" in resp.text.lower()

    @pytest.mark.asyncio
    async def test_browser_command_rejects_non_string_target(self, tmp_path):
        """A non-string target_agent_id must produce 400, not 500.

        Defends ``_resolve_browser_target`` against arbitrary JSON shapes
        coming through ``data.get("target_agent_id")``.
        """
        from httpx import ASGITransport, AsyncClient

        app, _event_bus, _cm = _build_app(
            tmp_path,
            perms_map={
                "operator": {"can_use_browser": False, "can_message": ["*"]},
                "social-manager": {"can_use_browser": True},
            },
        )

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            for bad in ([1], {"x": 1}, 42):
                resp = await client.post(
                    "/mesh/browser/command",
                    json={
                        "action": "snapshot",
                        "params": {},
                        "target_agent_id": bad,
                    },
                    headers={"X-Agent-ID": "operator"},
                )
                assert resp.status_code == 400, (bad, resp.text)
                assert "must be a string" in resp.text


class TestBrowserLoginRequestEndpoint:
    """POST /mesh/browser-login-request delegation matrix."""

    @pytest.mark.asyncio
    async def test_self_path_emits_under_caller(self, tmp_path):
        from httpx import ASGITransport, AsyncClient

        app, event_bus, _cm = _build_app(
            tmp_path,
            perms_map={
                "worker": {"can_use_browser": True},
            },
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
        assert body["requested"] is True
        assert body["target_agent"] == "worker"

        # Event emitted under worker's identity
        event_bus.emit.assert_called_once()
        call_args = event_bus.emit.call_args
        assert call_args[0][0] == "browser_login_request"
        assert call_args[1]["agent"] == "worker"

    @pytest.mark.asyncio
    async def test_delegation_emits_under_target(self, tmp_path):
        """Operator delegates → event emitted with agent=target so the dashboard's
        cross-surfacing logic routes the login card into operator's chat."""
        from httpx import ASGITransport, AsyncClient

        app, event_bus, _cm = _build_app(
            tmp_path,
            perms_map={
                "operator": {"can_use_browser": False, "can_message": ["*"]},
                "social-manager": {"can_use_browser": True},
            },
        )

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/mesh/browser-login-request",
                json={
                    "agent_id": "operator",
                    "target_agent_id": "social-manager",
                    "url": "https://x.com/login",
                    "service": "X",
                    "description": "Log in to X for social-manager",
                },
                headers={"X-Agent-ID": "operator"},
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["requested"] is True
        assert body["target_agent"] == "social-manager"

        # Event emitted with agent=target, not operator
        event_bus.emit.assert_called_once()
        call_args = event_bus.emit.call_args
        assert call_args[0][0] == "browser_login_request"
        assert call_args[1]["agent"] == "social-manager"
        assert call_args[1]["data"]["service"] == "X"

    @pytest.mark.asyncio
    async def test_delegation_blocked_by_can_message(self, tmp_path):
        from httpx import ASGITransport, AsyncClient

        app, event_bus, _cm = _build_app(
            tmp_path,
            perms_map={
                "worker1": {
                    "can_use_browser": False,
                    "can_message": ["worker1"],
                },
                "worker2": {"can_use_browser": True},
            },
        )

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/mesh/browser-login-request",
                json={
                    "agent_id": "worker1",
                    "target_agent_id": "worker2",
                    "url": "https://x.com/login",
                    "service": "X",
                    "description": "Log in",
                },
                headers={"X-Agent-ID": "worker1"},
            )
        assert resp.status_code == 403
        assert "allowlist" in resp.text.lower()
        event_bus.emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_delegation_blocked_when_target_has_no_browser(self, tmp_path):
        from httpx import ASGITransport, AsyncClient

        app, event_bus, _cm = _build_app(
            tmp_path,
            perms_map={
                "operator": {"can_use_browser": False, "can_message": ["*"]},
                "researcher": {"can_use_browser": False},
            },
        )

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/mesh/browser-login-request",
                json={
                    "agent_id": "operator",
                    "target_agent_id": "researcher",
                    "url": "https://x.com/login",
                    "service": "X",
                    "description": "Log in",
                },
                headers={"X-Agent-ID": "operator"},
            )
        assert resp.status_code == 403
        assert "no browser access" in resp.text.lower()
        event_bus.emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_target_equals_caller_emits_under_caller(self, tmp_path):
        """target_agent_id == caller_id is treated as self path, not delegation."""
        from httpx import ASGITransport, AsyncClient

        app, event_bus, _cm = _build_app(
            tmp_path,
            perms_map={
                # Worker has can_use_browser but no can_message at all —
                # the self path must not consult can_message.
                "worker": {"can_use_browser": True, "can_message": []},
            },
        )

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/mesh/browser-login-request",
                json={
                    "agent_id": "worker",
                    "target_agent_id": "worker",
                    "url": "https://x.com/login",
                    "service": "X",
                    "description": "Log in",
                },
                headers={"X-Agent-ID": "worker"},
            )
        assert resp.status_code == 200, resp.text
        assert resp.json()["target_agent"] == "worker"
        event_bus.emit.assert_called_once()
        assert event_bus.emit.call_args[1]["agent"] == "worker"

    @pytest.mark.asyncio
    async def test_unknown_target_blocked(self, tmp_path):
        """Delegation to an agent ID with no permissions row → 403."""
        from httpx import ASGITransport, AsyncClient

        app, event_bus, _cm = _build_app(
            tmp_path,
            perms_map={
                "operator": {"can_use_browser": False, "can_message": ["*"]},
            },
        )

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/mesh/browser-login-request",
                json={
                    "agent_id": "operator",
                    "target_agent_id": "ghost-agent",
                    "url": "https://x.com/login",
                    "service": "X",
                    "description": "Log in",
                },
                headers={"X-Agent-ID": "operator"},
            )
        assert resp.status_code == 403
        assert "no browser access" in resp.text.lower()
        event_bus.emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_emitted_url_is_redacted(self, tmp_path):
        """A9: ``request_browser_login`` MUST redact the URL before
        emitting to the event bus. OAuth callback URLs (``?code=...&
        state=...``) and other query-string secrets must NOT leak to
        the dashboard event history. ``redact_url`` strips query
        params with sensitive names while preserving scheme/host/path.
        """
        from httpx import ASGITransport, AsyncClient

        app, event_bus, _cm = _build_app(
            tmp_path,
            perms_map={"worker": {"can_use_browser": True}},
        )

        sensitive_url = (
            "https://example.com/cb"
            "?code=secret_xyz_must_not_leak"
            "&state=abc"
            "&access_token=tk_live_must_not_leak"
        )

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/mesh/browser-login-request",
                json={
                    "agent_id": "worker",
                    "url": sensitive_url,
                    "service": "X",
                    "description": "Log in",
                },
                headers={"X-Agent-ID": "worker"},
            )
        assert resp.status_code == 200, resp.text

        event_bus.emit.assert_called_once()
        emitted_data = event_bus.emit.call_args[1]["data"]
        emitted_url = emitted_data["url"]
        # The sensitive query values MUST NOT appear in the emitted
        # event payload — ``redact_url`` strips them before emit.
        assert "secret_xyz_must_not_leak" not in emitted_url
        assert "tk_live_must_not_leak" not in emitted_url
        # Scheme + host + path are preserved so the operator still
        # sees what target the agent meant to log into.
        assert emitted_url.startswith("https://example.com/cb")

    @pytest.mark.asyncio
    async def test_rate_limit_charged_to_caller_not_target(self, tmp_path):
        """The notify rate limit MUST be charged to the caller, never the
        target. Defense-in-depth: the original PR attributed the limit to
        the target, which would let one noisy caller exhaust an unrelated
        peer's notify quota via repeated delegation.

        Strategy: ``notify`` is 10/min. Spam 11 delegated requests from
        operator → social-manager. If the bucket is keyed by caller
        (operator), the 11th request returns 429. If it were keyed by
        target, all 11 succeed (because operator's bucket is empty). We
        also verify that after operator is exhausted, social-manager can
        still issue its own request (target's bucket untouched).
        """
        from httpx import ASGITransport, AsyncClient

        app, _event_bus, _cm = _build_app(
            tmp_path,
            perms_map={
                "operator": {"can_use_browser": False, "can_message": ["*"]},
                "social-manager": {"can_use_browser": True},
            },
        )

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            statuses: list[int] = []
            for _ in range(11):
                resp = await client.post(
                    "/mesh/browser-login-request",
                    json={
                        "agent_id": "operator",
                        "target_agent_id": "social-manager",
                        "url": "https://x.com/login",
                        "service": "X",
                        "description": "Log in",
                    },
                    headers={"X-Agent-ID": "operator"},
                )
                statuses.append(resp.status_code)

            # First 10 succeed; 11th hits the caller's notify limit (10/min).
            assert statuses[:10] == [200] * 10, statuses
            assert statuses[10] == 429, statuses
            assert "Rate limit exceeded" in (
                resp.json().get("detail") or ""
            ), resp.text

            # social-manager's own bucket must be untouched: it can still
            # issue a self-path request even though operator just spammed
            # 10 delegated requests targeting it.
            resp = await client.post(
                "/mesh/browser-login-request",
                json={
                    "agent_id": "social-manager",
                    "url": "https://x.com/login",
                    "service": "X",
                    "description": "Log in",
                },
                headers={"X-Agent-ID": "social-manager"},
            )
            assert resp.status_code == 200, resp.text

    @pytest.mark.asyncio
    async def test_browser_login_request_rejects_non_string_target(self, tmp_path):
        """A list/dict/int target_agent_id must produce 400, not 500.

        Defends ``_resolve_browser_target`` against arbitrary JSON shapes
        coming through ``data.get("target_agent_id")``.
        """
        from httpx import ASGITransport, AsyncClient

        app, _event_bus, _cm = _build_app(
            tmp_path,
            perms_map={
                "operator": {"can_use_browser": False, "can_message": ["*"]},
                "social-manager": {"can_use_browser": True},
            },
        )

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            for bad in ([1], {"x": 1}, 42):
                resp = await client.post(
                    "/mesh/browser-login-request",
                    json={
                        "agent_id": "operator",
                        "target_agent_id": bad,
                        "url": "https://x.com/login",
                        "service": "X",
                        "description": "Log in",
                    },
                    headers={"X-Agent-ID": "operator"},
                )
                assert resp.status_code == 400, (bad, resp.text)
                assert "must be a string" in resp.text


class TestBrowserCommandSSRF:
    """Mesh-side early-reject for navigate / open_tab pointing at private IPs."""

    @pytest.mark.asyncio
    async def test_navigate_private_ip_rejected(self, tmp_path):
        from httpx import ASGITransport, AsyncClient

        app, _event_bus, _cm = _build_app(
            tmp_path,
            perms_map={"worker": {"can_use_browser": True}},
        )

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/mesh/browser/command",
                json={"action": "navigate", "params": {"url": "http://127.0.0.1/"}},
                headers={"X-Agent-ID": "worker"},
            )
        assert resp.status_code == 400, resp.text

    @pytest.mark.asyncio
    async def test_open_tab_private_ip_rejected(self, tmp_path):
        from httpx import ASGITransport, AsyncClient

        app, _event_bus, _cm = _build_app(
            tmp_path,
            perms_map={"worker": {"can_use_browser": True}},
        )

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/mesh/browser/command",
                json={"action": "open_tab", "params": {"url": "http://127.0.0.1/"}},
                headers={"X-Agent-ID": "worker"},
            )
        assert resp.status_code == 400, resp.text
