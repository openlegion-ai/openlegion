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
    async def test_stage_idempotency_different_bytes_overwrites(
        self, tmp_path, monkeypatch,
    ):
        """Same (caller, idem_key) with different bytes overwrites the stage
        file — sha256 mismatch fails the dedupe gate, so the upload is fresh.
        The handle string is deterministic in (caller, key)."""
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
        handle = r2.json()["staged_handle"]
        bin_path = tmp_path / "stage" / f"{handle}.bin"
        assert bin_path.read_bytes() == b"version-B"

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

        import json as _json
        bin_path = tmp_path / "stage" / f"{handle}.bin"
        meta_path = tmp_path / "stage" / f"{handle}.json"
        assert not bin_path.exists()
        assert meta_path.exists()
        meta = _json.loads(meta_path.read_text())
        assert meta["status"] == "consumed"
        assert meta["consumed_at"]

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


class TestApplyRateLimit:
    @pytest.mark.asyncio
    async def test_apply_rate_limit_enforced(self, tmp_path, monkeypatch):
        """upload_apply has its own bucket; once exhausted, returns 429."""
        from httpx import ASGITransport, AsyncClient

        app, _cm = _build_app(
            tmp_path,
            perms_map={"worker": {"can_use_browser": True}},
            monkeypatch=monkeypatch,
        )

        # Stage one handle so we have something to apply.
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            stage = await client.post(
                "/mesh/browser/upload-stage",
                content=b"x",
                headers={"X-Agent-ID": "worker"},
            )
            handle = stage.json()["staged_handle"]

            # Patch the rate-limit table to be tight (2 requests/min) for
            # upload_apply via the running app's stored state.
            # We achieve this by issuing many calls; the default is 30/60s
            # per the impl. To keep the test fast we hit the cap.
            statuses: list[int] = []
            for _ in range(35):
                resp = await client.post(
                    "/mesh/browser/upload_file",
                    json={"ref": "e1", "staged_handles": [handle]},
                    headers={"X-Agent-ID": "worker"},
                )
                statuses.append(resp.status_code)
                if resp.status_code == 429:
                    break
        assert 429 in statuses, statuses


class TestIdempotencyTtl:
    @pytest.mark.asyncio
    async def test_idem_match_returns_same_handle_within_ttl(
        self, tmp_path, monkeypatch,
    ):
        from httpx import ASGITransport, AsyncClient

        app, _cm = _build_app(
            tmp_path,
            perms_map={"worker": {"can_use_browser": True}},
            monkeypatch=monkeypatch,
            ttl_s=60,
        )
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            r1 = await client.post(
                "/mesh/browser/upload-stage",
                content=b"same bytes",
                headers={
                    "X-Agent-ID": "worker",
                    "Idempotency-Key": "k1",
                },
            )
            r2 = await client.post(
                "/mesh/browser/upload-stage",
                content=b"same bytes",
                headers={
                    "X-Agent-ID": "worker",
                    "Idempotency-Key": "k1",
                },
            )
        assert r1.json()["staged_handle"] == r2.json()["staged_handle"]

    @pytest.mark.asyncio
    async def test_idem_handle_is_deterministic_in_caller_and_key(
        self, tmp_path, monkeypatch,
    ):
        """Smoke check that the handle is derived from (caller, key) without
        listing the stage directory — the same key always maps to the same
        handle string regardless of how many other files exist alongside."""
        from httpx import ASGITransport, AsyncClient

        app, _cm = _build_app(
            tmp_path,
            perms_map={"worker": {"can_use_browser": True}},
            monkeypatch=monkeypatch,
        )
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            # Populate dir with a handful of unrelated files first so the
            # dedupe path has noise to sift through if it were O(N).
            for i in range(10):
                await client.post(
                    "/mesh/browser/upload-stage",
                    content=f"f{i}".encode(),
                    headers={
                        "X-Agent-ID": "worker",
                        "Idempotency-Key": f"u{i}",
                    },
                )
            r1 = await client.post(
                "/mesh/browser/upload-stage",
                content=b"target",
                headers={
                    "X-Agent-ID": "worker",
                    "Idempotency-Key": "target-key",
                },
            )
            r2 = await client.post(
                "/mesh/browser/upload-stage",
                content=b"target",
                headers={
                    "X-Agent-ID": "worker",
                    "Idempotency-Key": "target-key",
                },
            )
        assert r1.json()["staged_handle"] == r2.json()["staged_handle"]
        # Handle is the deterministic SHA-prefix form, not a random hex.
        assert r1.json()["staged_handle"].startswith("worker-idem")

    @pytest.mark.asyncio
    async def test_idem_match_then_fresh_after_expiry(
        self, tmp_path, monkeypatch,
    ):
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
            r1 = await client.post(
                "/mesh/browser/upload-stage",
                content=b"payload",
                headers={
                    "X-Agent-ID": "worker",
                    "Idempotency-Key": "ttl-key",
                },
            )
            handle = r1.json()["staged_handle"]
            meta_path = tmp_path / "stage" / f"{handle}.json"
            bin_path = tmp_path / "stage" / f"{handle}.bin"
            old = time.time() - 10
            os.utime(meta_path, (old, old))
            os.utime(bin_path, (old, old))

            r2 = await client.post(
                "/mesh/browser/upload-stage",
                content=b"payload",
                headers={
                    "X-Agent-ID": "worker",
                    "Idempotency-Key": "ttl-key",
                },
            )
        # Same deterministic handle string but the file is rewritten with
        # current mtime — the dedupe gate fails on age.
        assert r1.json()["staged_handle"] == r2.json()["staged_handle"]
        assert (tmp_path / "stage" / f"{handle}.bin").stat().st_mtime > old + 1


class TestApplyTimeoutTaxonomy:
    @pytest.mark.asyncio
    async def test_apply_timeout_returns_503_not_502(self, tmp_path, monkeypatch):
        import httpx
        from httpx import ASGITransport, AsyncClient

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
                content=b"x",
                headers={"X-Agent-ID": "worker"},
            )
            handle = stage.json()["staged_handle"]

            real_post = httpx.AsyncClient.post

            async def fake_post(self, url, *args, **kwargs):
                if "browser-svc" in str(url):
                    raise httpx.ReadTimeout("simulated")
                return await real_post(self, url, *args, **kwargs)

            monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
            resp = await client.post(
                "/mesh/browser/upload_file",
                json={"ref": "e1", "staged_handles": [handle]},
                headers={"X-Agent-ID": "worker"},
            )
        assert resp.status_code == 503, resp.text


class TestSuggestedFilenamesPropagate:
    @pytest.mark.asyncio
    async def test_apply_forwards_suggested_filename_to_browser_ingest(
        self, tmp_path, monkeypatch,
    ):
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
                content=b"resume bytes",
                headers={"X-Agent-ID": "worker"},
            )
            handle = stage.json()["staged_handle"]

            ingest_filenames: list[str] = []
            real_post = httpx.AsyncClient.post

            async def fake_post(self, url, *args, **kwargs):
                url_str = str(url)
                if "browser-svc" in url_str and "/_stage_upload" in url_str:
                    params = kwargs.get("params") or {}
                    ingest_filenames.append(params.get("suggested_filename", ""))
                    body = b""
                    content = kwargs.get("content")
                    if content is not None:
                        async for chunk in content:
                            body += chunk
                    return Response(
                        200,
                        json={"path": "/tmp/upload-recv/x-resume.pdf", "size_bytes": len(body)},
                        request=httpx.Request("POST", url_str),
                    )
                if "browser-svc" in url_str and url_str.endswith("/upload_file"):
                    return Response(
                        200,
                        json={"success": True, "data": {"uploaded": []}},
                        request=httpx.Request("POST", url_str),
                    )
                return await real_post(self, url, *args, **kwargs)

            monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
            resp = await client.post(
                "/mesh/browser/upload_file",
                json={
                    "ref": "e7",
                    "staged_handles": [handle],
                    "suggested_filenames": ["resume.pdf"],
                },
                headers={"X-Agent-ID": "worker"},
            )
        assert resp.status_code == 200, resp.text
        assert ingest_filenames == ["resume.pdf"]


class TestAtomicWrite:
    @pytest.mark.asyncio
    async def test_no_partial_files_remain_after_successful_stage(
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
            await client.post(
                "/mesh/browser/upload-stage",
                content=b"data",
                headers={"X-Agent-ID": "worker"},
            )
        stage_dir = tmp_path / "stage"
        partials = list(stage_dir.glob("*.partial"))
        assert partials == []

    @pytest.mark.asyncio
    async def test_partial_files_cleaned_on_413(
        self, tmp_path, monkeypatch,
    ):
        from httpx import ASGITransport, AsyncClient

        app, _cm = _build_app(
            tmp_path,
            perms_map={"worker": {"can_use_browser": True}},
            monkeypatch=monkeypatch,
            max_mb=1,
        )
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/mesh/browser/upload-stage",
                content=b"X" * (2 * 1024 * 1024),
                headers={"X-Agent-ID": "worker"},
            )
        assert resp.status_code == 413
        stage_dir = tmp_path / "stage"
        assert list(stage_dir.glob("*.partial")) == []
        assert list(stage_dir.glob("*.bin")) == []


class TestApplyReplay:
    """P0.1 — apply must be idempotent on retry-after-success."""

    @pytest.mark.asyncio
    async def test_apply_retry_after_success_returns_cached_result(
        self, tmp_path, monkeypatch,
    ):
        """Same caller + same idempotency_key + same handles within TTL —
        the second call returns the cached envelope and DOES NOT drive
        the browser a second time."""
        import httpx
        from httpx import ASGITransport, AsyncClient, Response

        app, _cm = _build_app(
            tmp_path,
            perms_map={"worker": {"can_use_browser": True}},
            monkeypatch=monkeypatch,
        )

        ingest_calls: list[str] = []
        upload_calls: list[str] = []
        real_post = httpx.AsyncClient.post

        async def fake_post(self, url, *args, **kwargs):
            url_str = str(url)
            if "browser-svc" in url_str and "/_stage_upload" in url_str:
                ingest_calls.append(url_str)
                content = kwargs.get("content")
                if content is not None:
                    async for _ in content:
                        pass
                return Response(
                    200,
                    json={"path": "/tmp/upload-recv/x.bin", "size_bytes": 4},
                    request=httpx.Request("POST", url_str),
                )
            if "browser-svc" in url_str and url_str.endswith("/upload_file"):
                upload_calls.append(url_str)
                return Response(
                    200,
                    json={"success": True, "data": {"uploaded": ["/tmp/upload-recv/x.bin"]}},
                    request=httpx.Request("POST", url_str),
                )
            return await real_post(self, url, *args, **kwargs)

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            stage = await client.post(
                "/mesh/browser/upload-stage",
                content=b"data",
                headers={"X-Agent-ID": "worker"},
            )
            handle = stage.json()["staged_handle"]

            r1 = await client.post(
                "/mesh/browser/upload_file",
                json={
                    "ref": "e1",
                    "staged_handles": [handle],
                    "idempotency_key": "apply-key-1",
                },
                headers={"X-Agent-ID": "worker"},
            )
            r2 = await client.post(
                "/mesh/browser/upload_file",
                json={
                    "ref": "e1",
                    "staged_handles": [handle],
                    "idempotency_key": "apply-key-1",
                },
                headers={"X-Agent-ID": "worker"},
            )

        assert r1.status_code == 200, r1.text
        assert r2.status_code == 200, r2.text
        assert r1.json() == r2.json()
        assert len(ingest_calls) == 1, "browser ingest must run only once"
        assert len(upload_calls) == 1, "browser upload_file must run only once"

    @pytest.mark.asyncio
    async def test_apply_retry_with_different_key_returns_404(
        self, tmp_path, monkeypatch,
    ):
        """Same handle, different key — handles are consumed and the
        replay gate fails on key mismatch → 404 (handle unknown)."""
        import httpx
        from httpx import ASGITransport, AsyncClient, Response

        app, _cm = _build_app(
            tmp_path,
            perms_map={"worker": {"can_use_browser": True}},
            monkeypatch=monkeypatch,
        )

        real_post = httpx.AsyncClient.post

        async def fake_post(self, url, *args, **kwargs):
            url_str = str(url)
            if "browser-svc" in url_str and "/_stage_upload" in url_str:
                content = kwargs.get("content")
                if content is not None:
                    async for _ in content:
                        pass
                return Response(
                    200,
                    json={"path": "/tmp/upload-recv/x.bin", "size_bytes": 4},
                    request=httpx.Request("POST", url_str),
                )
            if "browser-svc" in url_str and url_str.endswith("/upload_file"):
                return Response(
                    200,
                    json={"success": True, "data": {"uploaded": []}},
                    request=httpx.Request("POST", url_str),
                )
            return await real_post(self, url, *args, **kwargs)

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            stage = await client.post(
                "/mesh/browser/upload-stage",
                content=b"data",
                headers={"X-Agent-ID": "worker"},
            )
            handle = stage.json()["staged_handle"]

            r1 = await client.post(
                "/mesh/browser/upload_file",
                json={
                    "ref": "e1",
                    "staged_handles": [handle],
                    "idempotency_key": "key-A",
                },
                headers={"X-Agent-ID": "worker"},
            )
            assert r1.status_code == 200

            r2 = await client.post(
                "/mesh/browser/upload_file",
                json={
                    "ref": "e1",
                    "staged_handles": [handle],
                    "idempotency_key": "key-B",
                },
                headers={"X-Agent-ID": "worker"},
            )
        assert r2.status_code == 404, r2.text


class TestStageConcurrency:
    """P0.2 — concurrent stages with the same idempotency_key serialize."""

    @pytest.mark.asyncio
    async def test_concurrent_stage_same_key_serializes(
        self, tmp_path, monkeypatch,
    ):
        import asyncio as _asyncio

        from httpx import ASGITransport, AsyncClient

        app, _cm = _build_app(
            tmp_path,
            perms_map={"worker": {"can_use_browser": True}},
            monkeypatch=monkeypatch,
        )

        payload = b"X" * (256 * 1024)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            results = await _asyncio.gather(
                client.post(
                    "/mesh/browser/upload-stage",
                    content=payload,
                    headers={
                        "X-Agent-ID": "worker",
                        "Idempotency-Key": "race-key",
                    },
                ),
                client.post(
                    "/mesh/browser/upload-stage",
                    content=payload,
                    headers={
                        "X-Agent-ID": "worker",
                        "Idempotency-Key": "race-key",
                    },
                ),
            )

        assert all(r.status_code == 200 for r in results), [r.text for r in results]
        bodies = [r.json() for r in results]
        # Same handle (deterministic in caller+key).
        assert bodies[0]["staged_handle"] == bodies[1]["staged_handle"]
        # Bytes on disk match the payload exactly — proof the two writers
        # did not interleave.
        handle = bodies[0]["staged_handle"]
        bin_path = tmp_path / "stage" / f"{handle}.bin"
        assert bin_path.read_bytes() == payload


class TestGcSkipsActivePartial:
    """P0.3 — GC must not unlink partials for in-flight uploads."""

    @pytest.mark.asyncio
    async def test_gc_skips_active_partial(self, tmp_path, monkeypatch):
        import os
        import time

        app, _cm = _build_app(
            tmp_path,
            perms_map={"worker": {"can_use_browser": True}},
            monkeypatch=monkeypatch,
            ttl_s=5,
        )

        # Simulate an in-flight upload by registering the handle in the
        # active set and dropping a stale ``.partial`` on disk.
        active = app.state.upload_stage_active_handles
        partial_ttl = app.state.upload_stage_partial_ttl_s
        stage_dir = tmp_path / "stage"
        stage_dir.mkdir(parents=True, exist_ok=True)
        handle = "worker-busy"
        partial = stage_dir / f"{handle}.bin.partial"
        partial.write_bytes(b"in-flight")
        old = time.time() - (partial_ttl + 60)
        os.utime(partial, (old, old))

        active.add(handle)
        try:
            gc = app.state.upload_stage_gc_once
            await gc()
            # Active partial — preserved despite stale mtime.
            assert partial.exists()
        finally:
            active.discard(handle)

        # Once cleared from the active set, the GC reaps it (mtime is
        # past _UPLOAD_STAGE_PARTIAL_TTL_S).
        await app.state.upload_stage_gc_once()
        assert not partial.exists()

    @pytest.mark.asyncio
    async def test_gc_uses_longer_ttl_for_partials(self, tmp_path, monkeypatch):
        """Partials younger than the longer partial-TTL are kept even if
        older than the regular sidecar TTL — slow uploads can complete."""
        import os
        import time

        app, _cm = _build_app(
            tmp_path,
            perms_map={"worker": {"can_use_browser": True}},
            monkeypatch=monkeypatch,
            ttl_s=5,
        )
        stage_dir = tmp_path / "stage"
        stage_dir.mkdir(parents=True, exist_ok=True)
        partial = stage_dir / "worker-slow.bin.partial"
        partial.write_bytes(b"halfway")
        # Past sidecar TTL (5s) but inside partial TTL (5 × 5 = 25s).
        recent = time.time() - 10
        os.utime(partial, (recent, recent))

        await app.state.upload_stage_gc_once()
        assert partial.exists()


class TestStageHandleValidation:
    """P1.6 — apply must reject handles that escape the stage dir."""

    @pytest.mark.asyncio
    async def test_apply_rejects_traversal_handle(self, tmp_path, monkeypatch):
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
                json={
                    "ref": "e1",
                    "staged_handles": ["../../etc/passwd"],
                },
                headers={"X-Agent-ID": "worker"},
            )
        assert resp.status_code == 400, resp.text


class TestApplyParseOrder:
    """P1.7 — rate limit fires before json parsing."""

    @pytest.mark.asyncio
    async def test_apply_rate_limit_fires_before_parse(
        self, tmp_path, monkeypatch,
    ):
        """Burn the apply bucket, then send a malformed body. We expect
        429 (rate-limit) — NOT 4xx-from-parse — because the rate-limit
        check now runs first."""
        from httpx import ASGITransport, AsyncClient

        app, _cm = _build_app(
            tmp_path,
            perms_map={"worker": {"can_use_browser": True}},
            monkeypatch=monkeypatch,
        )

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            statuses: list[int] = []
            for _ in range(40):
                # Send a body that would 400 if parsed (no ref, no
                # handles). We expect 400 until the rate-limit kicks in,
                # then 429 — the moment we see 429 we stop.
                resp = await client.post(
                    "/mesh/browser/upload_file",
                    json={},
                    headers={"X-Agent-ID": "worker"},
                )
                statuses.append(resp.status_code)
                if resp.status_code == 429:
                    break
            assert 429 in statuses, statuses

            # Confirm a malformed body (which would normally raise on
            # json parse) is now stopped at the rate-limit gate.
            resp = await client.post(
                "/mesh/browser/upload_file",
                content=b"not-json",
                headers={
                    "X-Agent-ID": "worker",
                    "content-type": "application/json",
                },
            )
            assert resp.status_code == 429, resp.text
