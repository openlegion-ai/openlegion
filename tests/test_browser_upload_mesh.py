"""Tests for §4.5 / §8.1 mesh-mediated file-upload endpoints.

Covers:
- ``POST /mesh/browser/upload-stage`` — Phase A: bytes ingest, idempotency,
  size cap, permission check, rate limit.
- ``POST /mesh/browser/upload_file`` — Phase B: handle resolution,
  cross-agent abuse rejection, browser proxy orchestration.
- Stage GC loop reaps orphan files past TTL.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _build_app(tmp_path, *, perms_map, monkeypatch=None, ttl_s=60, max_mb=50):
    from src.host.costs import CostTracker
    from src.host.mesh import Blackboard, MessageRouter, PubSub
    from src.host.permissions import PermissionMatrix
    from src.host.server import create_mesh_app
    from src.host.traces import TraceStore
    from src.shared.types import AgentPermissions

    if monkeypatch is not None:
        monkeypatch.setenv(
            "OPENLEGION_UPLOAD_STAGE_DIR", str(tmp_path / "stage"),
        )
        monkeypatch.setenv(
            "OPENLEGION_UPLOAD_STAGE_TTL_S", str(ttl_s),
        )
        monkeypatch.setenv(
            "OPENLEGION_UPLOAD_STAGE_MAX_MB", str(max_mb),
        )

    blackboard = Blackboard(str(tmp_path / "bb.db"))
    pubsub = PubSub()
    permissions = PermissionMatrix()
    for aid, perms in perms_map.items():
        permissions.permissions[aid] = AgentPermissions(agent_id=aid, **perms)

    router = MessageRouter(permissions, {})
    costs = CostTracker(str(tmp_path / "costs.db"))
    traces = TraceStore(str(tmp_path / "traces.db"))

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
    return app, container_manager


class TestMeshUploadStage:
    @pytest.mark.asyncio
    async def test_stage_returns_handle_and_metadata(self, tmp_path, monkeypatch):
        from httpx import ASGITransport, AsyncClient

        app, _cm = _build_app(
            tmp_path,
            perms_map={"worker": {"can_use_browser": True}},
            monkeypatch=monkeypatch,
        )

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/mesh/browser/upload-stage",
                content=b"hello world",
                headers={"X-Agent-ID": "worker"},
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["staged_handle"].startswith("worker-")
        assert body["size_bytes"] == 11
        assert body["expires_at"]

    @pytest.mark.asyncio
    async def test_stage_413_when_oversize(self, tmp_path, monkeypatch):
        from httpx import ASGITransport, AsyncClient

        app, _cm = _build_app(
            tmp_path,
            perms_map={"worker": {"can_use_browser": True}},
            monkeypatch=monkeypatch,
            max_mb=1,
        )
        too_big = b"X" * (2 * 1024 * 1024)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/mesh/browser/upload-stage",
                content=too_big,
                headers={"X-Agent-ID": "worker"},
            )
        assert resp.status_code == 413, resp.text
        stage_dir = tmp_path / "stage"
        assert not any(p.suffix == ".bin" for p in stage_dir.iterdir())

    @pytest.mark.asyncio
    async def test_stage_idempotency_returns_same_handle(self, tmp_path, monkeypatch):
        from httpx import ASGITransport, AsyncClient

        app, _cm = _build_app(
            tmp_path,
            perms_map={"worker": {"can_use_browser": True}},
            monkeypatch=monkeypatch,
        )
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            r1 = await client.post(
                "/mesh/browser/upload-stage",
                content=b"resume bytes",
                headers={
                    "X-Agent-ID": "worker",
                    "Idempotency-Key": "abc-1",
                },
            )
            r2 = await client.post(
                "/mesh/browser/upload-stage",
                content=b"resume bytes",
                headers={
                    "X-Agent-ID": "worker",
                    "Idempotency-Key": "abc-1",
                },
            )
        assert r1.status_code == 200 and r2.status_code == 200
        assert r1.json()["staged_handle"] == r2.json()["staged_handle"]

    @pytest.mark.asyncio
    async def test_stage_idempotency_different_bytes_returns_new_handle(
        self, tmp_path, monkeypatch,
    ):
        from httpx import ASGITransport, AsyncClient

        app, _cm = _build_app(
            tmp_path,
            perms_map={"worker": {"can_use_browser": True}},
            monkeypatch=monkeypatch,
        )
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            r1 = await client.post(
                "/mesh/browser/upload-stage",
                content=b"version-A",
                headers={
                    "X-Agent-ID": "worker",
                    "Idempotency-Key": "abc-1",
                },
            )
            r2 = await client.post(
                "/mesh/browser/upload-stage",
                content=b"version-B",
                headers={
                    "X-Agent-ID": "worker",
                    "Idempotency-Key": "abc-1",
                },
            )
        assert r1.status_code == 200 and r2.status_code == 200
        assert r1.json()["staged_handle"] != r2.json()["staged_handle"]

    @pytest.mark.asyncio
    async def test_stage_denied_without_browser_permission(
        self, tmp_path, monkeypatch,
    ):
        from httpx import ASGITransport, AsyncClient

        app, _cm = _build_app(
            tmp_path,
            perms_map={
                "denied": {
                    "can_use_browser": True,
                    "browser_actions": ["click"],
                },
            },
            monkeypatch=monkeypatch,
        )
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/mesh/browser/upload-stage",
                content=b"x",
                headers={"X-Agent-ID": "denied"},
            )
        assert resp.status_code == 403, resp.text

    @pytest.mark.asyncio
    async def test_stage_gc_reaps_old_files(self, tmp_path, monkeypatch):
        import os
        import time

        from httpx import ASGITransport, AsyncClient

        app, _cm = _build_app(
            tmp_path,
            perms_map={"worker": {"can_use_browser": True}},
            monkeypatch=monkeypatch,
            ttl_s=5,
        )
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/mesh/browser/upload-stage",
                content=b"data",
                headers={"X-Agent-ID": "worker"},
            )
        handle = resp.json()["staged_handle"]
        bin_path = tmp_path / "stage" / f"{handle}.bin"
        meta_path = tmp_path / "stage" / f"{handle}.json"
        assert bin_path.is_file() and meta_path.is_file()

        old = time.time() - 10
        os.utime(bin_path, (old, old))
        os.utime(meta_path, (old, old))

        gc = app.state.upload_stage_gc_once
        reaped = await gc()
        assert reaped >= 2
        assert not bin_path.exists()
        assert not meta_path.exists()


class TestMeshUploadApply:
    @pytest.mark.asyncio
    async def test_apply_orchestrates_ingest_and_upload(self, tmp_path, monkeypatch):
        import httpx
        from httpx import ASGITransport, AsyncClient, Response

        app, _cm = _build_app(
            tmp_path,
            perms_map={"worker": {"can_use_browser": True}},
            monkeypatch=monkeypatch,
        )

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            stage = await client.post(
                "/mesh/browser/upload-stage",
                content=b"resume.pdf bytes",
                headers={"X-Agent-ID": "worker"},
            )
            handle = stage.json()["staged_handle"]

            calls: list[str] = []
            real_post = httpx.AsyncClient.post

            async def fake_post(self, url, *args, **kwargs):
                if "browser-svc" in str(url):
                    if "/_stage_upload" in str(url):
                        calls.append("ingest")
                        body = b""
                        content = kwargs.get("content")
                        if content is not None:
                            async for chunk in content:
                                body += chunk
                        return Response(
                            200,
                            json={
                                "path": f"/tmp/upload-recv/{handle}.bin",
                                "size_bytes": len(body),
                            },
                            request=httpx.Request("POST", str(url)),
                        )
                    if str(url).endswith("/upload_file"):
                        calls.append("apply")
                        sent = kwargs.get("json") or {}
                        return Response(
                            200,
                            json={
                                "success": True,
                                "data": {"uploaded": sent.get("paths", [])},
                            },
                            request=httpx.Request("POST", str(url)),
                        )
                return await real_post(self, url, *args, **kwargs)

            monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

            resp = await client.post(
                "/mesh/browser/upload_file",
                json={"ref": "e7", "staged_handles": [handle]},
                headers={"X-Agent-ID": "worker"},
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["uploaded"] == [f"/tmp/upload-recv/{handle}.bin"]
        assert calls == ["ingest", "apply"]

        bin_path = tmp_path / "stage" / f"{handle}.bin"
        meta_path = tmp_path / "stage" / f"{handle}.json"
        assert not bin_path.exists()
        assert not meta_path.exists()

    @pytest.mark.asyncio
    async def test_apply_rejects_handle_from_other_caller(self, tmp_path, monkeypatch):
        from httpx import ASGITransport, AsyncClient

        app, _cm = _build_app(
            tmp_path,
            perms_map={
                "alice": {"can_use_browser": True},
                "bob": {"can_use_browser": True},
            },
            monkeypatch=monkeypatch,
        )
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            stage = await client.post(
                "/mesh/browser/upload-stage",
                content=b"alice bytes",
                headers={"X-Agent-ID": "alice"},
            )
            handle = stage.json()["staged_handle"]

            resp = await client.post(
                "/mesh/browser/upload_file",
                json={"ref": "e1", "staged_handles": [handle]},
                headers={"X-Agent-ID": "bob"},
            )
        assert resp.status_code == 403, resp.text

    @pytest.mark.asyncio
    async def test_apply_404_when_handle_unknown(self, tmp_path, monkeypatch):
        from httpx import ASGITransport, AsyncClient

        app, _cm = _build_app(
            tmp_path,
            perms_map={"worker": {"can_use_browser": True}},
            monkeypatch=monkeypatch,
        )
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/mesh/browser/upload_file",
                json={"ref": "e1", "staged_handles": ["worker-nope"]},
                headers={"X-Agent-ID": "worker"},
            )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_apply_503_when_browser_unavailable(self, tmp_path, monkeypatch):
        from httpx import ASGITransport, AsyncClient

        app, _cm = _build_app(
            tmp_path,
            perms_map={"worker": {"can_use_browser": True}},
            monkeypatch=monkeypatch,
        )
        _cm.browser_service_url = None

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            stage = await client.post(
                "/mesh/browser/upload-stage",
                content=b"x",
                headers={"X-Agent-ID": "worker"},
            )
            handle = stage.json()["staged_handle"]
            resp = await client.post(
                "/mesh/browser/upload_file",
                json={"ref": "e1", "staged_handles": [handle]},
                headers={"X-Agent-ID": "worker"},
            )
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_apply_denied_when_target_lacks_permission(
        self, tmp_path, monkeypatch,
    ):
        from httpx import ASGITransport, AsyncClient

        app, _cm = _build_app(
            tmp_path,
            perms_map={
                "operator": {
                    "can_use_browser": False,
                    "can_message": ["*"],
                },
                "worker": {
                    "can_use_browser": True,
                    "browser_actions": ["click"],
                },
            },
            monkeypatch=monkeypatch,
        )
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            stage = await client.post(
                "/mesh/browser/upload-stage",
                content=b"x",
                headers={"X-Agent-ID": "operator"},
            )
            assert stage.status_code == 403

    @pytest.mark.asyncio
    async def test_apply_validates_ref_and_handles(self, tmp_path, monkeypatch):
        from httpx import ASGITransport, AsyncClient

        app, _cm = _build_app(
            tmp_path,
            perms_map={"worker": {"can_use_browser": True}},
            monkeypatch=monkeypatch,
        )
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            r1 = await client.post(
                "/mesh/browser/upload_file",
                json={"ref": "", "staged_handles": ["a"]},
                headers={"X-Agent-ID": "worker"},
            )
            r2 = await client.post(
                "/mesh/browser/upload_file",
                json={"ref": "e1", "staged_handles": []},
                headers={"X-Agent-ID": "worker"},
            )
            r3 = await client.post(
                "/mesh/browser/upload_file",
                json={
                    "ref": "e1",
                    "staged_handles": ["a", "b", "c", "d", "e", "f"],
                },
                headers={"X-Agent-ID": "worker"},
            )
        assert r1.status_code == 400
        assert r2.status_code == 400
        assert r3.status_code == 400
