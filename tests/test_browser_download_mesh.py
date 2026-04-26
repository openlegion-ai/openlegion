"""Mesh-side tests for ``POST /mesh/browser/download`` (Phase 5 §8.2).

Covers the orchestration:
  mesh → browser (trigger)
  mesh ← browser (stream bytes by nonce)
  mesh → agent (POST /artifacts/ingest streamed)
  mesh → browser (cleanup)

Plus the operator kill switch (``BROWSER_DOWNLOADS_DISABLED``), permission
denial, browser-down + ingest-fails error paths, and the cleanup-on-error
guarantee.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest


def _build_app(tmp_path, *, perms_map, agent_urls):
    """Build a mesh app with seeded permissions and agent registry."""
    from src.host.costs import CostTracker
    from src.host.mesh import Blackboard, MessageRouter, PubSub
    from src.host.permissions import PermissionMatrix
    from src.host.server import create_mesh_app
    from src.host.traces import TraceStore
    from src.shared.types import AgentPermissions

    blackboard = Blackboard(str(tmp_path / "bb.db"))
    pubsub = PubSub()
    permissions = PermissionMatrix()
    for aid, perms in perms_map.items():
        permissions.permissions[aid] = AgentPermissions(agent_id=aid, **perms)

    router = MessageRouter(permissions, dict(agent_urls))
    costs = CostTracker(str(tmp_path / "costs.db"))
    traces = TraceStore(str(tmp_path / "traces.db"))

    container_manager = MagicMock()
    container_manager.browser_service_url = "http://browser-svc:8500"
    container_manager.browser_auth_token = ""

    app = create_mesh_app(
        blackboard=blackboard,
        pubsub=pubsub,
        router=router,
        permissions=permissions,
        cost_tracker=costs,
        trace_store=traces,
        event_bus=MagicMock(),
        container_manager=container_manager,
    )
    return app


class _FakeStream:
    """Minimal async ctx-manager mimicking httpx.AsyncClient.stream(...)."""

    def __init__(self, status_code: int, body: bytes):
        self.status_code = status_code
        self._body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def aiter_bytes(self):
        yield self._body


class TestMeshDownload:
    @pytest.mark.asyncio
    async def test_full_flow_streams_to_agent_and_cleans_up(self, tmp_path, monkeypatch):
        """Happy path: trigger → stream → ingest → cleanup."""
        from httpx import ASGITransport, AsyncClient, Response

        app = _build_app(
            tmp_path,
            perms_map={"worker": {"can_use_browser": True}},
            agent_urls={"worker": "http://worker-agent:8400"},
        )

        calls: list[str] = []
        ingest_seen: dict = {}

        real_post = httpx.AsyncClient.post
        real_stream = httpx.AsyncClient.stream

        async def fake_post(self, url, *args, **kwargs):
            url_s = str(url)
            calls.append(("POST", url_s))
            if url_s.endswith("/browser/worker/download"):
                req = httpx.Request("POST", url_s)
                return Response(200, json={
                    "success": True,
                    "data": {
                        "path": "/tmp/downloads/abcdef012345-report.pdf",
                        "nonce": "abcdef012345",
                        "size_bytes": 12,
                        "suggested_filename": "report.pdf",
                        "mime_type": "application/pdf",
                    },
                }, request=req)
            if url_s.endswith("/_download_cleanup"):
                req = httpx.Request("POST", url_s)
                return Response(200, json={"deleted": 1}, request=req)
            if url_s.endswith("/artifacts/ingest/report.pdf"):
                # Drain the streamed content so the test verifies it flows.
                content = kwargs.get("content")
                if content is not None and hasattr(content, "__aiter__"):
                    chunks = []
                    async for chunk in content:
                        chunks.append(chunk)
                    ingest_seen["body"] = b"".join(chunks)
                ingest_seen["headers"] = dict(kwargs.get("headers") or {})
                req = httpx.Request("POST", url_s)
                return Response(200, json={
                    "artifact_name": "report.pdf",
                    "size_bytes": 12,
                    "mime_type": "application/pdf",
                }, request=req)
            return await real_post(self, url, *args, **kwargs)

        def fake_stream(self, method, url, *args, **kwargs):
            url_s = str(url)
            calls.append((method, url_s))
            if "_download_stream" in url_s:
                return _FakeStream(200, b"hello-bytes!")
            return real_stream(self, method, url, *args, **kwargs)

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
        monkeypatch.setattr(httpx.AsyncClient, "stream", fake_stream)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/mesh/browser/download",
                json={"ref": "e1", "timeout_ms": 5000},
                headers={"X-Agent-ID": "worker"},
            )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["artifact_name"] == "report.pdf"
        assert body["data"]["size_bytes"] == 12
        assert body["data"]["mime_type"] == "application/pdf"
        # Bytes streamed end-to-end.
        assert ingest_seen["body"] == b"hello-bytes!"
        # X-Mesh-Internal forwarded so the agent's ingest endpoint accepts it.
        assert ingest_seen["headers"].get("X-Mesh-Internal") == "1"
        # Cleanup ran exactly once after ingest.
        cleanup_calls = [u for _m, u in calls if "_download_cleanup" in u]
        assert len(cleanup_calls) == 1

    @pytest.mark.asyncio
    async def test_disabled_flag_returns_forbidden(self, tmp_path, monkeypatch):
        from httpx import ASGITransport, AsyncClient

        monkeypatch.setenv("BROWSER_DOWNLOADS_DISABLED", "1")
        # Reset cached operator settings so the env override wins.
        from src.browser import flags as bflags
        bflags.reload_operator_settings()

        app = _build_app(
            tmp_path,
            perms_map={"worker": {"can_use_browser": True}},
            agent_urls={"worker": "http://worker-agent:8400"},
        )

        invocations: list[str] = []
        real_post = httpx.AsyncClient.post

        async def fake_post(self, url, *args, **kwargs):
            invocations.append(str(url))
            if "browser-svc" in str(url) or "/artifacts/ingest/" in str(url):
                return httpx.Response(
                    500, request=httpx.Request("POST", str(url)),
                )
            return await real_post(self, url, *args, **kwargs)

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/mesh/browser/download",
                json={"ref": "e1"},
                headers={"X-Agent-ID": "worker"},
            )

        assert resp.status_code == 403, (resp.text, resp.content)
        body = resp.json()
        # FastAPI wraps `detail=` into the response body's `detail` key.
        detail = body.get("detail", body)
        assert detail.get("success") is False
        assert detail["error"]["code"] == "forbidden"
        # Browser was never invoked.
        assert not any("browser-svc" in u for u in invocations)

    @pytest.mark.asyncio
    async def test_browser_service_unavailable_503(self, tmp_path):
        from httpx import ASGITransport, AsyncClient

        # Build an app whose container_manager has no browser_service_url.
        from src.host.costs import CostTracker
        from src.host.mesh import Blackboard, MessageRouter, PubSub
        from src.host.permissions import PermissionMatrix
        from src.host.server import create_mesh_app
        from src.host.traces import TraceStore
        from src.shared.types import AgentPermissions

        permissions = PermissionMatrix()
        permissions.permissions["worker"] = AgentPermissions(
            agent_id="worker", can_use_browser=True,
        )
        router = MessageRouter(permissions, {"worker": "http://worker:8400"})
        cm = MagicMock()
        cm.browser_service_url = ""
        cm.browser_auth_token = ""
        app = create_mesh_app(
            blackboard=Blackboard(str(tmp_path / "bb.db")),
            pubsub=PubSub(),
            router=router,
            permissions=permissions,
            cost_tracker=CostTracker(str(tmp_path / "c.db")),
            trace_store=TraceStore(str(tmp_path / "t.db")),
            event_bus=MagicMock(),
            container_manager=cm,
        )

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/mesh/browser/download",
                json={"ref": "e1"},
                headers={"X-Agent-ID": "worker"},
            )
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_permission_denied_returns_403(self, tmp_path):
        from httpx import ASGITransport, AsyncClient

        app = _build_app(
            tmp_path,
            perms_map={"worker": {
                "can_use_browser": True,
                # Empty whitelist with `*` denied — note that
                # KNOWN_BROWSER_ACTIONS gates default-allow when the
                # `browser_actions` field is unset; setting it to a
                # subset that excludes 'download' produces the denial.
                "browser_actions": ["snapshot"],
            }},
            agent_urls={"worker": "http://worker:8400"},
        )

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/mesh/browser/download",
                json={"ref": "e1"},
                headers={"X-Agent-ID": "worker"},
            )
        assert resp.status_code == 403
        assert "download" in resp.text.lower()

    @pytest.mark.asyncio
    async def test_agent_ingest_failure_runs_cleanup(self, tmp_path, monkeypatch):
        """If the agent rejects the ingest mid-stream, cleanup MUST still run."""
        from httpx import ASGITransport, AsyncClient, Response

        app = _build_app(
            tmp_path,
            perms_map={"worker": {"can_use_browser": True}},
            agent_urls={"worker": "http://worker:8400"},
        )

        cleanup_called: list[str] = []
        real_post = httpx.AsyncClient.post

        async def fake_post(self, url, *args, **kwargs):
            url_s = str(url)
            if url_s.endswith("/browser/worker/download"):
                return Response(200, json={
                    "success": True,
                    "data": {
                        "path": "/tmp/downloads/abcdef012345-x",
                        "nonce": "abcdef012345",
                        "size_bytes": 5,
                        "suggested_filename": "x.bin",
                        "mime_type": "application/octet-stream",
                    },
                }, request=httpx.Request("POST", url_s))
            if "_download_cleanup" in url_s:
                cleanup_called.append(url_s)
                return Response(200, json={"deleted": 1},
                                request=httpx.Request("POST", url_s))
            if "/artifacts/ingest/" in url_s:
                content = kwargs.get("content")
                if content is not None and hasattr(content, "__aiter__"):
                    async for _ in content:
                        pass
                return Response(507, text="Insufficient Storage",
                                request=httpx.Request("POST", url_s))
            return await real_post(self, url, *args, **kwargs)

        def fake_stream(self, method, url, *args, **kwargs):
            return _FakeStream(200, b"abcde")

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
        monkeypatch.setattr(httpx.AsyncClient, "stream", fake_stream)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/mesh/browser/download",
                json={"ref": "e1"},
                headers={"X-Agent-ID": "worker"},
            )
        assert resp.status_code == 507, resp.text
        assert len(cleanup_called) == 1, "cleanup must run even on ingest failure"

    @pytest.mark.asyncio
    async def test_browser_stream_failure_runs_cleanup(self, tmp_path, monkeypatch):
        """Stream interruption mid-transfer → cleanup still runs."""
        from httpx import ASGITransport, AsyncClient, Response

        app = _build_app(
            tmp_path,
            perms_map={"worker": {"can_use_browser": True}},
            agent_urls={"worker": "http://worker:8400"},
        )

        cleanup_called: list[str] = []
        real_post = httpx.AsyncClient.post

        async def fake_post(self, url, *args, **kwargs):
            url_s = str(url)
            if url_s.endswith("/browser/worker/download"):
                return Response(200, json={
                    "success": True,
                    "data": {
                        "path": "/tmp/downloads/abcdef012345-y",
                        "nonce": "abcdef012345",
                        "size_bytes": 1,
                        "suggested_filename": "y.bin",
                        "mime_type": "application/octet-stream",
                    },
                }, request=httpx.Request("POST", url_s))
            if "_download_cleanup" in url_s:
                cleanup_called.append(url_s)
                return Response(200, json={"deleted": 1},
                                request=httpx.Request("POST", url_s))
            return await real_post(self, url, *args, **kwargs)

        def fake_stream(self, method, url, *args, **kwargs):
            # Browser side returns an error mid-fetch.
            return _FakeStream(500, b"")

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
        monkeypatch.setattr(httpx.AsyncClient, "stream", fake_stream)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/mesh/browser/download",
                json={"ref": "e1"},
                headers={"X-Agent-ID": "worker"},
            )
        assert resp.status_code == 502, resp.text
        assert len(cleanup_called) == 1

    @pytest.mark.asyncio
    async def test_trigger_failure_passes_through(self, tmp_path, monkeypatch):
        """When the browser trigger returns an error envelope, the mesh
        forwards it without attempting stream/ingest/cleanup."""
        from httpx import ASGITransport, AsyncClient, Response

        app = _build_app(
            tmp_path,
            perms_map={"worker": {"can_use_browser": True}},
            agent_urls={"worker": "http://worker:8400"},
        )

        observed: list[str] = []
        real_post = httpx.AsyncClient.post

        async def fake_post(self, url, *args, **kwargs):
            url_s = str(url)
            observed.append(url_s)
            if url_s.endswith("/browser/worker/download"):
                return Response(200, json={
                    "success": False,
                    "error": "Ref 'e1' not found",
                }, request=httpx.Request("POST", url_s))
            return await real_post(self, url, *args, **kwargs)

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/mesh/browser/download",
                json={"ref": "e1"},
                headers={"X-Agent-ID": "worker"},
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["success"] is False
        # No attempt at stream / ingest / cleanup since trigger said "no file".
        assert not any("_download_stream" in u for u in observed)
        assert not any("_download_cleanup" in u for u in observed)
        assert not any("/artifacts/ingest" in u for u in observed)

    @pytest.mark.asyncio
    async def test_trigger_timeout_overrides_default(self, tmp_path, monkeypatch):
        """The trigger POST overrides the proxy client's default 60s timeout
        with a longer 180s value, since the browser-side trigger blocks until
        the streaming write to disk finishes for large downloads."""
        from httpx import ASGITransport, AsyncClient, Response

        app = _build_app(
            tmp_path,
            perms_map={"worker": {"can_use_browser": True}},
            agent_urls={"worker": "http://worker:8400"},
        )

        seen_timeouts: dict = {}
        real_post = httpx.AsyncClient.post

        async def fake_post(self, url, *args, **kwargs):
            url_s = str(url)
            if url_s.endswith("/browser/worker/download"):
                seen_timeouts["trigger"] = kwargs.get("timeout")
                # Return success=False so we bail out before stream/ingest.
                return Response(200, json={
                    "success": False,
                    "error": "no-op for timeout test",
                }, request=httpx.Request("POST", url_s))
            return await real_post(self, url, *args, **kwargs)

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/mesh/browser/download",
                json={"ref": "e1"},
                headers={"X-Agent-ID": "worker"},
            )

        assert resp.status_code == 200, resp.text
        # The trigger POST must override the client default with timeout=180.
        assert seen_timeouts.get("trigger") == 180, seen_timeouts
