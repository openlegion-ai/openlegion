"""Trace-id propagation end-to-end (§2.5 / §4.6 wiring).

Two integration surfaces:

1. The browser-service middleware binds an incoming ``X-Trace-Id`` header
   to the :data:`current_trace_id` ContextVar for the duration of the
   request, and resets on exit.
2. The mesh's ``/mesh/browser/command`` proxy forwards ``X-Trace-Id`` to
   the browser service (verified by source inspection — the unit-test
   environment doesn't have a live browser service).
"""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient


def _mk_app_without_auth(monkeypatch):
    """Build the browser service FastAPI app with auth disabled.

    Auth is unrelated to trace propagation and adds mock overhead; we
    disable via the documented ``BROWSER_AUTH_TOKEN`` unset + no
    MESH_AUTH_TOKEN path.
    """
    monkeypatch.delenv("BROWSER_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("MESH_AUTH_TOKEN", raising=False)
    from src.browser.server import create_browser_app

    manager = MagicMock()
    manager.get_service_status = AsyncMock(return_value={"status": "ok"})
    app = create_browser_app(manager)
    return app, manager


class TestBrowserServiceTraceMiddleware:
    def test_incoming_x_trace_id_bound_during_request(self, monkeypatch):
        """A GET with X-Trace-Id header must expose the value via the
        contextvar inside endpoint handlers."""
        from src.shared.trace import current_trace_id

        observed: list[str | None] = []

        app, manager = _mk_app_without_auth(monkeypatch)

        async def _observe():
            observed.append(current_trace_id.get())
            return {"ok": True}

        app.add_api_route("/_probe", _observe, methods=["GET"])

        with TestClient(app) as client:
            client.get("/_probe", headers={"X-Trace-Id": "tr_abc123456"})

        assert observed == ["tr_abc123456"]

    def test_missing_trace_header_leaves_contextvar_none(self, monkeypatch):
        from src.shared.trace import current_trace_id

        observed: list[str | None] = []

        app, manager = _mk_app_without_auth(monkeypatch)

        async def _observe():
            observed.append(current_trace_id.get())
            return {"ok": True}

        app.add_api_route("/_probe", _observe, methods=["GET"])

        with TestClient(app) as client:
            client.get("/_probe")

        assert observed == [None]

    def test_contextvar_reset_after_request(self, monkeypatch):
        """Setting bled across requests would be a cross-request leak —
        the middleware must ``reset()`` the token on exit."""
        from src.shared.trace import current_trace_id

        observed: list[str | None] = []

        app, manager = _mk_app_without_auth(monkeypatch)

        async def _observe():
            observed.append(current_trace_id.get())
            return {"ok": True}

        app.add_api_route("/_probe", _observe, methods=["GET"])

        with TestClient(app) as client:
            client.get("/_probe", headers={"X-Trace-Id": "tr_first"})
            client.get("/_probe")  # no header on second

        assert observed == ["tr_first", None]


class TestMeshProxyForwardsTraceHeader:
    """Source-inspection guard: the mesh's browser proxy must forward
    ``X-Trace-Id`` to the browser service so downstream logs correlate."""

    def test_browser_command_forwards_trace_header(self):
        import src.host.server as host_server

        source = inspect.getsource(host_server)
        # Find the proxy call block and verify the header is added.
        proxy_call_pos = source.find('f"{browser_service_url}/browser/{req_agent_id}/{action}"')
        assert proxy_call_pos != -1, "browser proxy call site moved — update test"

        # Look backwards from proxy call for the header setup.
        before_proxy = source[:proxy_call_pos]
        # The forwarding should live in the ~200 lines immediately before
        # the proxy call (same endpoint body).
        region = before_proxy[-4000:]
        assert 'headers["X-Trace-Id"]' in region or "X-Trace-Id" in region, (
            "X-Trace-Id forwarding was removed from the browser proxy — §2.5 "
            "trace propagation is required across all hops."
        )
        assert 'request.headers.get("x-trace-id")' in region, (
            "X-Trace-Id extraction site moved — update test."
        )
