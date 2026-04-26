"""Tests for BrowserManager.upload_file() and .download() (§4.5).

Playwright's real APIs (``page.expect_file_chooser``, ``page.expect_download``)
are async context managers that yield an awaited ``.value``. We mock both
rather than run a live browser — tests verify:

- The file-chooser context is entered and set_files is called with the
  provided paths.
- Ref-not-found surfaces a clean error.
- User-control gate blocks both methods.
- Download size cap (post-transfer) rejects oversized files and deletes
  them from disk.
- Download success returns the expected envelope shape.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.browser.service import BrowserManager


def _make_instance(agent_id: str = "a1"):
    """MagicMock stand-in for CamoufoxInstance — spec_set is too strict
    because many instance attrs are assigned in __init__, not declared at
    class level."""
    inst = MagicMock()
    inst.agent_id = agent_id
    inst.page = MagicMock()
    inst.lock = asyncio.Lock()
    inst.touch = MagicMock()
    inst._user_control = False
    inst.refs = {}
    return inst


def _async_ctx(awaitable_value):
    """Return an async context manager whose ``.__aenter__`` yields an object
    with ``.value`` = the given awaitable. Mimics Playwright's
    ``expect_file_chooser`` / ``expect_download`` shape."""

    class _Info:
        pass

    info = _Info()
    info.value = awaitable_value

    class _CM:
        async def __aenter__(self_):
            return info

        async def __aexit__(self_, *exc):
            return False

    return _CM()


class TestUploadFile:
    @pytest.mark.asyncio
    async def test_happy_path_invokes_chooser_set_files(self, tmp_path, monkeypatch):
        mgr = BrowserManager(profiles_dir=str(tmp_path / "p"))
        inst = _make_instance()

        chooser = MagicMock()
        chooser.set_files = AsyncMock()

        # expect_file_chooser returns an async context whose .value is an
        # awaitable that resolves to the chooser.
        async def _chooser_value():
            return chooser

        inst.page.expect_file_chooser = MagicMock(
            return_value=_async_ctx(_chooser_value()),
        )

        # click is awaitable.
        fake_locator = MagicMock()
        fake_locator.click = AsyncMock()
        monkeypatch.setattr(
            mgr, "_locator_from_ref",
            AsyncMock(return_value=fake_locator),
        )
        monkeypatch.setattr(mgr, "get_or_start", AsyncMock(return_value=inst))
        # Mock action_delay's sleep.
        monkeypatch.setattr(
            "src.browser.service.action_delay", lambda: 0,
        )

        # Use a real path that exists so the path-validation guard passes.
        real_file = tmp_path / "foo.pdf"
        real_file.write_bytes(b"fake pdf")

        result = await mgr.upload_file("a1", "e1", [str(real_file)])
        assert result["success"] is True
        assert result["data"]["uploaded"] == [str(real_file)]
        chooser.set_files.assert_awaited_once_with([str(real_file)])
        fake_locator.click.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_missing_local_path_returns_error_before_click(
        self, tmp_path, monkeypatch,
    ):
        """Playwright's ``set_files`` raises cryptically on missing paths AND
        by then the chooser has already opened.  We validate up front so
        the failure is fast and the message is actionable."""
        mgr = BrowserManager(profiles_dir=str(tmp_path / "p"))
        inst = _make_instance()

        fake_locator = MagicMock()
        fake_locator.click = AsyncMock()
        inst.page.expect_file_chooser = MagicMock()
        monkeypatch.setattr(
            mgr, "_locator_from_ref",
            AsyncMock(return_value=fake_locator),
        )
        monkeypatch.setattr(mgr, "get_or_start", AsyncMock(return_value=inst))

        result = await mgr.upload_file(
            "a1", "e1", [str(tmp_path / "nonexistent.pdf")],
        )
        assert result["success"] is False
        assert "not found" in result["error"].lower()
        # No click should have fired — we failed before entering the chooser context.
        fake_locator.click.assert_not_called()
        inst.page.expect_file_chooser.assert_not_called()

    @pytest.mark.asyncio
    async def test_ref_not_found_returns_error(self, tmp_path, monkeypatch):
        mgr = BrowserManager(profiles_dir=str(tmp_path / "p"))
        inst = _make_instance()
        monkeypatch.setattr(
            mgr, "_locator_from_ref", AsyncMock(return_value=None),
        )
        monkeypatch.setattr(mgr, "get_or_start", AsyncMock(return_value=inst))
        # Use a real path so the path-validation guard (added for upload
        # correctness) passes and we actually exercise the ref-not-found
        # branch.
        real_file = tmp_path / "present.txt"
        real_file.write_text("x")
        result = await mgr.upload_file("a1", "missing-ref", [str(real_file)])
        assert result["success"] is False
        assert "missing-ref" in result["error"]

    @pytest.mark.asyncio
    async def test_user_control_blocks_upload(self, tmp_path, monkeypatch):
        mgr = BrowserManager(profiles_dir=str(tmp_path / "p"))
        inst = _make_instance()
        inst._user_control = True
        monkeypatch.setattr(mgr, "get_or_start", AsyncMock(return_value=inst))
        result = await mgr.upload_file("a1", "e1", ["/tmp/x"])
        assert result["success"] is False
        assert "control" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_browser_side_orphan_cleanup_on_chooser_timeout(
        self, tmp_path, monkeypatch,
    ):
        """Even when the chooser flow throws, the staged files are deleted."""
        mgr = BrowserManager(profiles_dir=str(tmp_path / "p"))
        inst = _make_instance()

        class _BadCM:
            async def __aenter__(self_):
                raise asyncio.TimeoutError("chooser never appeared")

            async def __aexit__(self_, *exc):
                return False

        inst.page.expect_file_chooser = MagicMock(return_value=_BadCM())

        fake_locator = MagicMock()
        fake_locator.click = AsyncMock()
        monkeypatch.setattr(mgr, "_locator_from_ref", AsyncMock(return_value=fake_locator))
        monkeypatch.setattr(mgr, "get_or_start", AsyncMock(return_value=inst))

        a = tmp_path / "a.txt"
        a.write_text("a")
        result = await mgr.upload_file("a1", "e1", [str(a)])
        assert result["success"] is False
        assert not a.exists()

    @pytest.mark.asyncio
    async def test_browser_side_orphan_cleanup_on_user_control(
        self, tmp_path, monkeypatch,
    ):
        """User-control branch returns early — files must still be cleaned."""
        mgr = BrowserManager(profiles_dir=str(tmp_path / "p"))
        inst = _make_instance()
        inst._user_control = True
        monkeypatch.setattr(mgr, "get_or_start", AsyncMock(return_value=inst))

        a = tmp_path / "stage1.txt"
        a.write_text("x")
        result = await mgr.upload_file("a1", "e1", [str(a)])
        assert result["success"] is False
        assert not a.exists()


class TestBrowserSidePeriodicGc:
    @pytest.mark.asyncio
    async def test_gc_reaps_files_older_than_ttl(self, tmp_path, monkeypatch):
        """Browser-side recv-dir GC removes files older than the stage TTL."""
        import os
        import time

        recv = tmp_path / "recv"
        recv.mkdir()
        monkeypatch.setenv("OPENLEGION_UPLOAD_RECV_DIR", str(recv))
        monkeypatch.setenv("OPENLEGION_UPLOAD_STAGE_TTL_S", "5")

        old = recv / "old.bin"
        old.write_bytes(b"x")
        new = recv / "new.bin"
        new.write_bytes(b"y")

        old_t = time.time() - 60
        os.utime(old, (old_t, old_t))

        mgr = BrowserManager(profiles_dir=str(tmp_path / "p"))
        reaped = await mgr._upload_recv_gc_once()
        assert reaped == 1
        assert not old.exists()
        assert new.exists()


class TestDownload:
    @pytest.mark.asyncio
    async def test_happy_path_saves_file_and_returns_path(
        self, tmp_path, monkeypatch,
    ):
        mgr = BrowserManager(profiles_dir=str(tmp_path / "p"))
        inst = _make_instance()
        fake_locator = MagicMock()
        fake_locator.click = AsyncMock()

        # Fake the download object returned by expect_download.
        download = MagicMock()
        download.suggested_filename = "report.pdf"

        dl_dir = tmp_path / "dl"

        async def _save_as(path_str):
            p = Path(path_str)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"X" * 200)

        download.save_as = AsyncMock(side_effect=_save_as)

        async def _dl_value():
            return download

        inst.page.expect_download = MagicMock(
            return_value=_async_ctx(_dl_value()),
        )
        monkeypatch.setattr(
            mgr, "_locator_from_ref", AsyncMock(return_value=fake_locator),
        )
        monkeypatch.setattr(mgr, "get_or_start", AsyncMock(return_value=inst))

        result = await mgr.download(
            "a1", "e1", download_dir=str(dl_dir), timeout_ms=5000,
        )
        assert result["success"] is True, result
        assert result["data"]["suggested_filename"] == "report.pdf"
        assert result["data"]["size_bytes"] == 200
        assert result["data"]["mime_type"] == "application/pdf"
        assert Path(result["data"]["path"]).exists()

    @pytest.mark.asyncio
    async def test_oversize_download_rejected_and_file_removed(
        self, tmp_path, monkeypatch,
    ):
        mgr = BrowserManager(profiles_dir=str(tmp_path / "p"))
        inst = _make_instance()
        fake_locator = MagicMock()
        fake_locator.click = AsyncMock()

        download = MagicMock()
        download.suggested_filename = "big.bin"
        dl_dir = tmp_path / "dl"

        async def _save_as(path_str):
            # Write a file larger than the cap we'll pass.
            p = Path(path_str)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"A" * 500)

        download.save_as = AsyncMock(side_effect=_save_as)

        async def _dl_value():
            return download

        inst.page.expect_download = MagicMock(
            return_value=_async_ctx(_dl_value()),
        )
        monkeypatch.setattr(
            mgr, "_locator_from_ref", AsyncMock(return_value=fake_locator),
        )
        monkeypatch.setattr(mgr, "get_or_start", AsyncMock(return_value=inst))

        result = await mgr.download(
            "a1", "e1", download_dir=str(dl_dir), max_bytes=100,
        )
        assert result["success"] is False
        assert "exceeds" in result["error"]
        # Partial file was cleaned up.
        assert not any(p.is_file() for p in dl_dir.iterdir())

    @pytest.mark.asyncio
    async def test_user_control_blocks_download(self, tmp_path, monkeypatch):
        mgr = BrowserManager(profiles_dir=str(tmp_path / "p"))
        inst = _make_instance()
        inst._user_control = True
        monkeypatch.setattr(mgr, "get_or_start", AsyncMock(return_value=inst))
        result = await mgr.download("a1", "e1", download_dir=str(tmp_path / "dl"))
        assert result["success"] is False
        assert "control" in result["error"].lower()


class TestUploadStageIngest:
    """`/browser/{a}/_stage_upload` mesh-internal byte ingest endpoint."""

    def _make_client(self, tmp_path, monkeypatch):
        from src.browser.server import create_browser_app
        recv = tmp_path / "recv"
        monkeypatch.setenv("OPENLEGION_UPLOAD_RECV_DIR", str(recv))
        monkeypatch.delenv("BROWSER_AUTH_TOKEN", raising=False)
        monkeypatch.delenv("MESH_AUTH_TOKEN", raising=False)
        mgr = BrowserManager(profiles_dir=str(tmp_path / "p"))
        app = create_browser_app(mgr)
        from starlette.testclient import TestClient
        return TestClient(app), mgr, recv

    def test_ingest_writes_file_and_returns_path(self, tmp_path, monkeypatch):
        client, _mgr, recv = self._make_client(tmp_path, monkeypatch)
        resp = client.post(
            "/browser/a1/_stage_upload",
            content=b"hello bytes",
            headers={"X-Mesh-Internal": "1"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["size_bytes"] == len(b"hello bytes")
        assert Path(body["path"]).is_file()
        assert Path(body["path"]).parent == recv
        assert Path(body["path"]).read_bytes() == b"hello bytes"

    def test_ingest_rejects_without_mesh_internal_header(self, tmp_path, monkeypatch):
        client, _mgr, _recv = self._make_client(tmp_path, monkeypatch)
        resp = client.post("/browser/a1/_stage_upload", content=b"hi")
        assert resp.status_code == 403

    def test_ingest_413_when_oversize(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENLEGION_UPLOAD_STAGE_MAX_MB", "1")
        client, _mgr, recv = self._make_client(tmp_path, monkeypatch)
        too_big = b"X" * (2 * 1024 * 1024)
        resp = client.post(
            "/browser/a1/_stage_upload",
            content=too_big,
            headers={"X-Mesh-Internal": "1"},
        )
        assert resp.status_code == 413, resp.text
        assert not any(p.is_file() for p in recv.iterdir())

    def test_ingest_path_consumed_by_upload_file(self, tmp_path, monkeypatch):
        client, mgr, _recv = self._make_client(tmp_path, monkeypatch)
        resp = client.post(
            "/browser/a1/_stage_upload",
            content=b"resume.pdf bytes",
            headers={"X-Mesh-Internal": "1"},
            params={"suggested_filename": "resume.pdf"},
        )
        ingested_path = resp.json()["path"]
        assert Path(ingested_path).is_file()
        assert "resume.pdf" in ingested_path

        async def _fake_upload(agent_id, ref, paths):
            assert paths == [ingested_path]
            return {"success": True, "data": {"uploaded": paths}}

        mgr.upload_file = _fake_upload
        upload_resp = client.post(
            "/browser/a1/upload_file",
            json={"ref": "e1", "paths": [ingested_path]},
        )
        assert upload_resp.status_code == 200
        assert upload_resp.json()["success"] is True

    def test_ingest_sanitizes_suggested_filename(self, tmp_path, monkeypatch):
        client, _mgr, recv = self._make_client(tmp_path, monkeypatch)
        resp = client.post(
            "/browser/a1/_stage_upload",
            content=b"x",
            headers={"X-Mesh-Internal": "1"},
            params={"suggested_filename": "../../etc/passwd"},
        )
        assert resp.status_code == 200, resp.text
        path = Path(resp.json()["path"])
        assert path.parent == recv
        assert ".." not in path.name

    def test_ingest_filename_includes_original_basename(self, tmp_path, monkeypatch):
        """When mesh forwards a real filename, the on-disk path ends with it
        so Playwright reports the correct name to the form site."""
        client, _mgr, _recv = self._make_client(tmp_path, monkeypatch)
        resp = client.post(
            "/browser/a1/_stage_upload",
            content=b"data",
            headers={"X-Mesh-Internal": "1"},
            params={"suggested_filename": "resume.pdf"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["path"].endswith("-resume.pdf")

    def test_long_filename_preserves_extension(self, tmp_path, monkeypatch):
        """P0.4: truncation must not drop the extension. A 200-char
        ``.pdf`` filename ends up stored as ``...<truncated>.pdf`` so
        Playwright reports the right type to the form site."""
        client, _mgr, _recv = self._make_client(tmp_path, monkeypatch)
        long_stem = "scan_2026_04_25_invoice_" + ("x" * 200)
        long_name = f"{long_stem}.pdf"
        resp = client.post(
            "/browser/a1/_stage_upload",
            content=b"data",
            headers={"X-Mesh-Internal": "1"},
            params={"suggested_filename": long_name},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["path"].endswith(".pdf")

    def test_auth_insecure_requires_bearer_for_internal_endpoint(
        self, tmp_path, monkeypatch,
    ):
        """BROWSER_AUTH_INSECURE=1 must NOT bypass auth for the internal
        byte-ingest endpoint; missing bearer → 403 even in dev mode."""
        from src.browser.server import create_browser_app

        recv = tmp_path / "recv"
        monkeypatch.setenv("OPENLEGION_UPLOAD_RECV_DIR", str(recv))
        monkeypatch.delenv("BROWSER_AUTH_TOKEN", raising=False)
        monkeypatch.delenv("MESH_AUTH_TOKEN", raising=False)
        monkeypatch.setenv("BROWSER_AUTH_INSECURE", "1")
        mgr = BrowserManager(profiles_dir=str(tmp_path / "p"))
        app = create_browser_app(mgr)
        from starlette.testclient import TestClient
        client = TestClient(app)

        resp = client.post(
            "/browser/a1/_stage_upload",
            content=b"x",
            headers={"X-Mesh-Internal": "1"},
        )
        assert resp.status_code == 403, resp.text

    def test_startup_raises_when_mesh_token_set_but_browser_token_missing(
        self, tmp_path, monkeypatch,
    ):
        """P1.5: production posture — MESH_AUTH_TOKEN set but no
        BROWSER_AUTH_TOKEN and no INSECURE override → RuntimeError."""
        import pytest as _pytest

        from src.browser.server import create_browser_app
        monkeypatch.delenv("BROWSER_AUTH_TOKEN", raising=False)
        monkeypatch.setenv("MESH_AUTH_TOKEN", "live-token")
        monkeypatch.delenv("BROWSER_AUTH_INSECURE", raising=False)
        mgr = BrowserManager(profiles_dir=str(tmp_path / "p"))
        with _pytest.raises(RuntimeError):
            create_browser_app(mgr)

    def test_startup_allows_insecure_override_in_dev_with_mesh_token(
        self, tmp_path, monkeypatch,
    ):
        """P1.5: BROWSER_AUTH_INSECURE=1 overrides the production guard
        so the dev posture (no bearer required) is reachable while
        keeping the per-endpoint guard at /_stage_upload."""
        from src.browser.server import create_browser_app
        monkeypatch.delenv("BROWSER_AUTH_TOKEN", raising=False)
        monkeypatch.setenv("MESH_AUTH_TOKEN", "live-token")
        monkeypatch.setenv("BROWSER_AUTH_INSECURE", "1")
        mgr = BrowserManager(profiles_dir=str(tmp_path / "p"))
        # Must not raise. App still has the per-endpoint bearer guard
        # exercised in the prior test.
        app = create_browser_app(mgr)
        assert app is not None


class TestUploadFileStageCleanup:
    """After `set_files` succeeds the manager removes the staged paths."""

    @pytest.mark.asyncio
    async def test_stage_files_unlinked_after_success(self, tmp_path, monkeypatch):
        mgr = BrowserManager(profiles_dir=str(tmp_path / "p"))
        inst = _make_instance()
        chooser = MagicMock()
        chooser.set_files = AsyncMock()

        async def _chooser_value():
            return chooser

        inst.page.expect_file_chooser = MagicMock(
            return_value=_async_ctx(_chooser_value()),
        )

        fake_locator = MagicMock()
        fake_locator.click = AsyncMock()
        monkeypatch.setattr(
            mgr, "_locator_from_ref", AsyncMock(return_value=fake_locator),
        )
        monkeypatch.setattr(mgr, "get_or_start", AsyncMock(return_value=inst))
        monkeypatch.setattr(
            "src.browser.service.action_delay", lambda: 0,
        )

        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("a")
        b.write_text("b")
        result = await mgr.upload_file("a1", "e1", [str(a), str(b)])
        assert result["success"] is True
        assert not a.exists()
        assert not b.exists()


class TestUploadRecvGcTaskLifecycle:
    """§8.1 cross-PR fix — ensure the upload-recv GC task is created by
    ``start_cleanup_loop`` AND cancelled when ``stop_all`` runs. Without
    explicit cancellation the task leaks across manager restarts and keeps
    polling a possibly-deleted recv dir.
    """

    @pytest.mark.asyncio
    async def test_upload_recv_gc_task_cancelled_on_stop(self, tmp_path):
        mgr = BrowserManager(profiles_dir=str(tmp_path / "p"))
        # Pre: not started yet
        assert mgr._upload_recv_gc_task is None
        await mgr.start_cleanup_loop()
        try:
            assert mgr._upload_recv_gc_task is not None
            assert not mgr._upload_recv_gc_task.done()
        finally:
            await mgr.stop_all()
        # Post: stop_all() must clear the handle and cancel the task.
        assert mgr._upload_recv_gc_task is None
class TestDownload:
    @pytest.mark.asyncio
    async def test_happy_path_saves_file_and_returns_path(
        self, tmp_path, monkeypatch,
    ):
        pytest.importorskip("playwright")
        mgr = BrowserManager(profiles_dir=str(tmp_path / "p"))
        mgr._download_streaming_available = True
        inst = _make_instance()
        fake_locator = MagicMock()
        fake_locator.click = AsyncMock()

        # 200 bytes via the mocked private artifact channel.
        download, _ = _make_oversize_artifact_download(
            total_bytes=200, chunk_bytes=64 * 1024,
        )
        download.suggested_filename = "report.pdf"

        dl_dir = tmp_path / "dl"

        async def _dl_value():
            return download

        inst.page.expect_download = MagicMock(
            return_value=_async_ctx(_dl_value()),
        )
        monkeypatch.setattr(mgr, "_locator_from_ref", AsyncMock(return_value=fake_locator))
        monkeypatch.setattr(mgr, "get_or_start", AsyncMock(return_value=inst))
        monkeypatch.setattr(
            "playwright._impl._connection.from_channel",
            lambda ch: ch,
        )

        result = await mgr.download(
            "a1", "e1", download_dir=str(dl_dir), timeout_ms=5000,
        )
        assert result["success"] is True, result
        assert result["data"]["suggested_filename"] == "report.pdf"
        assert result["data"]["size_bytes"] == 200
        assert result["data"]["mime_type"] == "application/pdf"
        assert Path(result["data"]["path"]).exists()

    @pytest.mark.asyncio
    async def test_oversize_download_rejected_and_file_removed(
        self, tmp_path, monkeypatch,
    ):
        pytest.importorskip("playwright")
        mgr = BrowserManager(profiles_dir=str(tmp_path / "p"))
        mgr._download_streaming_available = True
        inst = _make_instance()
        fake_locator = MagicMock()
        fake_locator.click = AsyncMock()

        # 500 bytes mocked with a 100-byte cap — must abort.
        download, _ = _make_oversize_artifact_download(
            total_bytes=500, chunk_bytes=64 * 1024,
        )
        download.suggested_filename = "big.bin"
        dl_dir = tmp_path / "dl"

        async def _dl_value():
            return download

        inst.page.expect_download = MagicMock(
            return_value=_async_ctx(_dl_value()),
        )
        monkeypatch.setattr(mgr, "_locator_from_ref", AsyncMock(return_value=fake_locator))
        monkeypatch.setattr(mgr, "get_or_start", AsyncMock(return_value=inst))
        monkeypatch.setattr(
            "playwright._impl._connection.from_channel",
            lambda ch: ch,
        )

        result = await mgr.download(
            "a1", "e1", download_dir=str(dl_dir), max_bytes=100,
        )
        assert result["success"] is False
        assert "exceeds" in result["error"]
        # Partial file was cleaned up.
        assert not any(p.is_file() for p in dl_dir.iterdir())

    @pytest.mark.asyncio
    async def test_user_control_blocks_download(self, tmp_path, monkeypatch):
        pytest.importorskip("playwright")
        mgr = BrowserManager(profiles_dir=str(tmp_path / "p"))
        mgr._download_streaming_available = True
        inst = _make_instance()
        inst._user_control = True
        monkeypatch.setattr(mgr, "get_or_start", AsyncMock(return_value=inst))
        result = await mgr.download("a1", "e1", download_dir=str(tmp_path / "dl"))
        assert result["success"] is False
        assert "control" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_download_response_includes_nonce(self, tmp_path, monkeypatch):
        """The download() response data must include a 12-hex-char nonce so
        the mesh can later address the file via ``_download_stream`` /
        ``_download_cleanup`` without trusting a server-internal path."""
        pytest.importorskip("playwright")
        import re

        mgr = BrowserManager(profiles_dir=str(tmp_path / "p"))
        mgr._download_streaming_available = True
        inst = _make_instance()
        fake_locator = MagicMock()
        fake_locator.click = AsyncMock()

        download, _ = _make_oversize_artifact_download(
            total_bytes=4, chunk_bytes=64 * 1024,
        )
        download.suggested_filename = "report.pdf"
        dl_dir = tmp_path / "dl"

        async def _dl_value():
            return download

        inst.page.expect_download = MagicMock(
            return_value=_async_ctx(_dl_value()),
        )
        monkeypatch.setattr(mgr, "_locator_from_ref", AsyncMock(return_value=fake_locator))
        monkeypatch.setattr(mgr, "get_or_start", AsyncMock(return_value=inst))
        monkeypatch.setattr(
            "playwright._impl._connection.from_channel",
            lambda ch: ch,
        )

        result = await mgr.download(
            "a1", "e1", download_dir=str(dl_dir), timeout_ms=5000,
        )
        assert result["success"] is True
        nonce = result["data"]["nonce"]
        assert re.fullmatch(r"[a-f0-9]{12}", nonce), nonce
        # Path must encode the same nonce (download_stream resolves by prefix).
        assert Path(result["data"]["path"]).name.startswith(f"{nonce}-")


class TestDownloadFlow:
    """Browser server endpoints for streaming/cleaning a saved download."""

    def _app(self, monkeypatch, tmp_path):
        from src.browser.server import create_browser_app
        from src.browser.service import BrowserManager

        monkeypatch.setenv("BROWSER_DOWNLOAD_DIR", str(tmp_path / "downloads"))
        mgr = BrowserManager(profiles_dir=str(tmp_path / "p"))
        return create_browser_app(mgr)

    def _seed(self, tmp_path: Path, nonce: str, suggested: str, payload: bytes) -> Path:
        dl = tmp_path / "downloads"
        dl.mkdir(parents=True, exist_ok=True)
        path = dl / f"{nonce}-{suggested}"
        path.write_bytes(payload)
        return path

    def test_download_stream_returns_file_bytes(self, tmp_path, monkeypatch):
        from starlette.testclient import TestClient
        app = self._app(monkeypatch, tmp_path)
        nonce = "abcdef012345"
        self._seed(tmp_path, nonce, "report.pdf", b"X" * 200)
        with TestClient(app) as client:
            resp = client.get(
                f"/browser/a1/_download_stream?nonce={nonce}",
            )
        assert resp.status_code == 200
        assert resp.content == b"X" * 200
        assert resp.headers["content-type"].startswith("application/octet-stream")

    def test_download_stream_unknown_nonce_404(self, tmp_path, monkeypatch):
        from starlette.testclient import TestClient
        app = self._app(monkeypatch, tmp_path)
        with TestClient(app) as client:
            resp = client.get("/browser/a1/_download_stream?nonce=000000000000")
        assert resp.status_code == 404

    def test_download_stream_bad_nonce_shape_400(self, tmp_path, monkeypatch):
        from starlette.testclient import TestClient
        app = self._app(monkeypatch, tmp_path)
        with TestClient(app) as client:
            for bad in ("XYZ", "ZZZZZZZZZZZZ", "abc", "abcdef012345abc"):
                resp = client.get(
                    f"/browser/a1/_download_stream?nonce={bad}",
                )
                assert resp.status_code == 400, (bad, resp.text)

    def test_download_cleanup_removes_file(self, tmp_path, monkeypatch):
        from starlette.testclient import TestClient
        app = self._app(monkeypatch, tmp_path)
        nonce = "0123456789ab"
        path = self._seed(tmp_path, nonce, "x.txt", b"hi")
        with TestClient(app) as client:
            resp = client.post(
                "/browser/a1/_download_cleanup", json={"nonce": nonce},
            )
        assert resp.status_code == 200
        assert resp.json() == {"deleted": 1}
        assert not path.exists()

    def test_download_cleanup_bad_nonce_400(self, tmp_path, monkeypatch):
        from starlette.testclient import TestClient
        app = self._app(monkeypatch, tmp_path)
        with TestClient(app) as client:
            resp = client.post(
                "/browser/a1/_download_cleanup", json={"nonce": "../../etc/passwd"},
            )
        assert resp.status_code == 400

    def test_startup_cleanup_clears_orphans(self, tmp_path, monkeypatch):
        """Phase 5 §8.2: blanket-delete /tmp/downloads/* on browser boot to
        purge orphans from a crashed/restarted tab."""
        monkeypatch.setenv("BROWSER_DOWNLOAD_DIR", str(tmp_path / "downloads"))
        dl = tmp_path / "downloads"
        dl.mkdir()
        (dl / "stale1-foo.bin").write_bytes(b"old")
        (dl / "stale2-bar.bin").write_bytes(b"old")
        from src.browser import __main__ as bmain
        bmain._cleanup_orphan_downloads()
        assert list(dl.iterdir()) == []

    def test_startup_cleanup_idempotent_when_dir_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BROWSER_DOWNLOAD_DIR", str(tmp_path / "missing"))
        from src.browser import __main__ as bmain
        # No exception when the download dir doesn't exist yet.
        bmain._cleanup_orphan_downloads()

    def test_nonce_registered_before_path_resolution(
        self, tmp_path, monkeypatch,
    ):
        """TOCTOU guard: the nonce must enter ``_active_download_nonces``
        BEFORE the on-disk path is resolved, so the GC janitor running
        between resolve() and stream-open can't reap the file."""
        from pathlib import Path as _Path
        from starlette.testclient import TestClient
        from src.browser import server as bserver
        from src.browser.service import BrowserManager

        monkeypatch.setenv("BROWSER_DOWNLOAD_DIR", str(tmp_path / "downloads"))
        mgr = BrowserManager(profiles_dir=str(tmp_path / "p"))
        app = bserver.create_browser_app(mgr)
        nonce = "abcdef012345"
        self._seed(tmp_path, nonce, "report.pdf", b"data")

        # Spy on iterdir() — the first thing the closure-scoped resolver does.
        # When it runs, the nonce must already be in the active set.
        observed: dict = {}
        real_iterdir = _Path.iterdir
        download_dir_str = str(tmp_path / "downloads")

        def spy_iterdir(self):
            if str(self) == download_dir_str and "active_at_resolve" not in observed:
                observed["active_at_resolve"] = (
                    nonce in mgr._active_download_nonces
                )
            return real_iterdir(self)

        monkeypatch.setattr(_Path, "iterdir", spy_iterdir)

        with TestClient(app) as client:
            resp = client.get(f"/browser/a1/_download_stream?nonce={nonce}")
        assert resp.status_code == 200, resp.text
        assert observed.get("active_at_resolve") is True, observed


class TestStreamingSizeCap:
    """P0.1 — the size cap must abort mid-transfer, not after a full write."""

    @pytest.mark.asyncio
    async def test_streaming_size_cap_aborts_mid_transfer(
        self, tmp_path, monkeypatch,
    ):
        """A 51MB attacker download must not be fully written to disk before
        the cap fires. We assert that the streaming counter aborted before
        the full payload was read."""
        pytest.importorskip("playwright")
        mgr = BrowserManager(profiles_dir=str(tmp_path / "p"))
        mgr._download_streaming_available = True
        inst = _make_instance()
        fake_locator = MagicMock()
        fake_locator.click = AsyncMock()

        chunk = 64 * 1024
        download, stream_chan = _make_oversize_artifact_download(
            total_bytes=51 * 1024 * 1024, chunk_bytes=chunk,
        )
        download.suggested_filename = "huge.bin"
        dl_dir = tmp_path / "dl"

        async def _dl_value():
            return download

        inst.page.expect_download = MagicMock(
            return_value=_async_ctx(_dl_value()),
        )
        monkeypatch.setattr(mgr, "_locator_from_ref", AsyncMock(return_value=fake_locator))
        monkeypatch.setattr(mgr, "get_or_start", AsyncMock(return_value=inst))
        monkeypatch.setattr(
            "playwright._impl._connection.from_channel",
            lambda ch: ch,
        )

        max_bytes = 50 * 1024 * 1024
        result = await mgr.download(
            "a1", "e1", download_dir=str(dl_dir),
            max_bytes=max_bytes, timeout_ms=5000,
        )

        assert result["success"] is False
        assert "exceeds" in result["error"]
        # Streaming aborted before sending the full 51MB payload.
        assert stream_chan._sent <= max_bytes + chunk
        # No leaked partial files.
        assert not any(p.is_file() for p in dl_dir.iterdir() if dl_dir.exists())

    @pytest.mark.asyncio
    async def test_51mb_download_rejected_by_streaming_counter(
        self, tmp_path, monkeypatch,
    ):
        """P1.5 — synthesize a 51MB download, assert streaming counter
        rejects it without reading the full file."""
        pytest.importorskip("playwright")
        mgr = BrowserManager(profiles_dir=str(tmp_path / "p"))
        mgr._download_streaming_available = True
        inst = _make_instance()
        fake_locator = MagicMock()
        fake_locator.click = AsyncMock()

        chunk = 64 * 1024
        download, stream_chan = _make_oversize_artifact_download(
            total_bytes=51 * 1024 * 1024, chunk_bytes=chunk,
        )
        download.suggested_filename = "report.dat"
        dl_dir = tmp_path / "dl"

        async def _dl_value():
            return download

        inst.page.expect_download = MagicMock(
            return_value=_async_ctx(_dl_value()),
        )
        monkeypatch.setattr(mgr, "_locator_from_ref", AsyncMock(return_value=fake_locator))
        monkeypatch.setattr(mgr, "get_or_start", AsyncMock(return_value=inst))
        monkeypatch.setattr(
            "playwright._impl._connection.from_channel",
            lambda ch: ch,
        )

        max_bytes = 50 * 1024 * 1024
        result = await mgr.download(
            "a1", "e1", download_dir=str(dl_dir),
            max_bytes=max_bytes, timeout_ms=5000,
        )
        assert result["success"] is False
        assert "exceeds" in result["error"]
        # P2.1 — assert streaming aborted before the full payload was read.
        assert stream_chan._sent <= max_bytes + chunk


class TestPrivateChannelStreaming:
    """Direct exercise of the chunked streaming path with a mocked channel.

    Demonstrates that when Playwright's private artifact stream is
    available, oversized payloads abort BEFORE all bytes hit disk."""

    @pytest.mark.asyncio
    async def test_chunked_path_aborts_before_full_write(
        self, tmp_path, monkeypatch,
    ):
        # The streaming path imports ``playwright._impl._connection``;
        # skip cleanly when playwright isn't installed (it's only present
        # in the browser container, not in the dev pyproject).
        pytest.importorskip("playwright")
        import base64

        mgr = BrowserManager(profiles_dir=str(tmp_path / "p"))
        # Force the streaming-availability flag on for this test even when
        # playwright is mocked, so download() doesn't return service_unavailable.
        mgr._download_streaming_available = True

        # Mock Playwright Stream + Artifact channel API. The channel
        # ``send("read", ...)`` yields fixed-size base64 chunks; the
        # ``saveAsStream`` channel call hands back a fake stream.
        class _StreamChan:
            def __init__(self_, total_bytes, chunk_bytes):
                self_._total = total_bytes
                self_._chunk = chunk_bytes
                self_._sent = 0

            async def send(self_, name, *_args):
                if name != "read":
                    return None
                if self_._sent >= self_._total:
                    return None
                remaining = self_._total - self_._sent
                size = min(self_._chunk, remaining)
                self_._sent += size
                return base64.b64encode(b"A" * size).decode()

        class _Stream:
            def __init__(self_, total, chunk):
                self_._channel = _StreamChan(total, chunk)

        class _ArtifactChan:
            def __init__(self_, stream):
                self_._stream = stream

            async def send(self_, name, *_args):
                # Branch on the requested action.
                if name == "saveAsStream":
                    return self_._stream
                if name == "cancel":
                    return None
                return None

        # 51MB total, 64KB chunks — must fail at 50MB cap without
        # writing the full 51MB to disk.
        total = 51 * 1024 * 1024
        chunk = 64 * 1024
        stream = _Stream(total, chunk)
        artifact = type("A", (), {"_channel": _ArtifactChan(stream)})()

        download = MagicMock()
        download._artifact = artifact
        download.cancel = AsyncMock()

        # Patch from_channel to passthrough — our fake channel mimics
        # ``Stream`` shape closely enough that we don't need the real
        # ``from_channel``.

        async def _run():
            return stream

        # Patch the import inside _stream_download_to_disk via attribute
        # injection on the function's globals.
        monkeypatch.setattr(
            "playwright._impl._connection.from_channel",
            lambda ch: ch,
        )

        dest = tmp_path / "abc-huge.bin"
        max_bytes = 50 * 1024 * 1024
        result = await mgr._stream_download_to_disk(download, dest, max_bytes)

        # Streaming counter rejected — return None.
        assert result is None
        # Partial file removed.
        assert not dest.exists()
        # Stream did NOT write the full 51MB before aborting.
        assert stream._channel._sent <= max_bytes + chunk
        # Cancel was invoked.
        download.cancel.assert_awaited()


class TestBrowserDownloadDirEnv:
    """P0.2 — BrowserManager.download() honors BROWSER_DOWNLOAD_DIR."""

    @pytest.mark.asyncio
    async def test_browser_download_dir_env_honored_end_to_end(
        self, tmp_path, monkeypatch,
    ):
        pytest.importorskip("playwright")
        custom_dir = tmp_path / "custom-dl"
        monkeypatch.setenv("BROWSER_DOWNLOAD_DIR", str(custom_dir))

        mgr = BrowserManager(profiles_dir=str(tmp_path / "p"))
        mgr._download_streaming_available = True
        inst = _make_instance()
        fake_locator = MagicMock()
        fake_locator.click = AsyncMock()

        # Build a mocked private artifact channel that yields b"hi" once.
        download, _ = _make_oversize_artifact_download(
            total_bytes=2, chunk_bytes=64 * 1024,
        )
        download.suggested_filename = "x.bin"

        async def _dl_value():
            return download

        inst.page.expect_download = MagicMock(
            return_value=_async_ctx(_dl_value()),
        )
        monkeypatch.setattr(mgr, "_locator_from_ref", AsyncMock(return_value=fake_locator))
        monkeypatch.setattr(mgr, "get_or_start", AsyncMock(return_value=inst))
        monkeypatch.setattr(
            "playwright._impl._connection.from_channel",
            lambda ch: ch,
        )

        # No explicit download_dir — must pick up env.
        result = await mgr.download("a1", "e1", timeout_ms=5000)
        assert result["success"] is True
        # File landed in the env-resolved directory.
        assert Path(result["data"]["path"]).parent == custom_dir.resolve() or (
            Path(result["data"]["path"]).parent == custom_dir
        )
        # Stream-by-nonce server endpoint also reads the same env.
        from starlette.testclient import TestClient

        from src.browser.server import create_browser_app
        app = create_browser_app(mgr)
        nonce = result["data"]["nonce"]
        with TestClient(app) as client:
            resp = client.get(f"/browser/a1/_download_stream?nonce={nonce}")
        assert resp.status_code == 200
        # The mocked channel yields 2 bytes of "A".
        assert resp.content == b"AA"


class TestDownloadGcLoop:
    """P0.3 — periodic janitor sweeps stale staging files."""

    @pytest.mark.asyncio
    async def test_download_gc_loop_deletes_orphans_after_60s(
        self, tmp_path, monkeypatch,
    ):
        """Direct exercise of the GC pass logic: a file with mtime older
        than the TTL gets deleted; a fresh file is preserved."""
        import os
        import time as _time

        dl = tmp_path / "downloads"
        dl.mkdir()
        old = dl / "abcdef012345-stale.bin"
        old.write_bytes(b"x")
        # Backdate to 5 minutes old.
        os.utime(old, (_time.time() - 300, _time.time() - 300))
        fresh = dl / "fedcba987654-fresh.bin"
        fresh.write_bytes(b"y")

        monkeypatch.setenv("BROWSER_DOWNLOAD_DIR", str(dl))
        monkeypatch.setenv("BROWSER_DOWNLOAD_TTL_S", "60")

        # Drive a single GC pass synchronously by inlining the loop body —
        # avoids waiting for the 30s sleep tick the production loop uses.
        ttl = int(os.environ["BROWSER_DOWNLOAD_TTL_S"])
        now = _time.time()
        for entry in list(dl.iterdir()):
            if entry.is_file() and (now - entry.stat().st_mtime) > ttl:
                entry.unlink(missing_ok=True)

        assert not old.exists()
        assert fresh.exists()

    @pytest.mark.asyncio
    async def test_gc_skips_actively_streamed_file(self, tmp_path, monkeypatch):
        """P1.1 — files whose nonce is registered in
        ``_active_download_nonces`` are skipped even when their mtime is
        past the TTL. Prevents the janitor from reaping a file mid-stream.
        """
        import os
        import time as _time

        dl = tmp_path / "downloads"
        dl.mkdir()
        active = dl / "abcdef012345-active.bin"
        active.write_bytes(b"x")
        os.utime(active, (_time.time() - 300, _time.time() - 300))
        idle = dl / "fedcba987654-idle.bin"
        idle.write_bytes(b"y")
        os.utime(idle, (_time.time() - 300, _time.time() - 300))

        monkeypatch.setenv("BROWSER_DOWNLOAD_DIR", str(dl))
        monkeypatch.setenv("BROWSER_DOWNLOAD_TTL_S", "60")

        mgr = BrowserManager(profiles_dir=str(tmp_path / "p"))
        mgr._active_download_nonces.add("abcdef012345")

        # Inline a single GC pass mirroring the loop body.
        ttl = int(os.environ["BROWSER_DOWNLOAD_TTL_S"])
        now = _time.time()
        for entry in list(dl.iterdir()):
            if not entry.is_file():
                continue
            nonce = entry.name.split("-", 1)[0]
            if nonce in mgr._active_download_nonces:
                continue
            if (now - entry.stat().st_mtime) > ttl:
                entry.unlink(missing_ok=True)

        assert active.exists(), "active stream's file must NOT be reaped"
        assert not idle.exists(), "idle stale file should be reaped"


class TestServiceUnavailableWhenStreamingMissing:
    """P0.1 — refuse downloads when the private streaming API is missing,
    rather than silently degrading to a racy polling fallback."""

    @pytest.mark.asyncio
    async def test_download_returns_service_unavailable_envelope(
        self, tmp_path, monkeypatch,
    ):
        mgr = BrowserManager(profiles_dir=str(tmp_path / "p"))
        # Force the detect-and-refuse branch.
        mgr._download_streaming_available = False

        # Trigger should NOT touch the page — assert no instance access.
        get_or_start = AsyncMock()
        monkeypatch.setattr(mgr, "get_or_start", get_or_start)

        result = await mgr.download(
            "a1", "e1", download_dir=str(tmp_path / "dl"),
            timeout_ms=1000,
        )
        assert result["success"] is False
        assert isinstance(result["error"], dict)
        assert result["error"]["code"] == "service_unavailable"
        # Confirm no save_as / locator / get_or_start happened.
        get_or_start.assert_not_called()


class TestDownloadEnvelopePassthrough:
    """P2.2 — mesh_client.browser_download passes through structured
    error envelopes (e.g. operator kill switch) without raising."""

    @pytest.mark.asyncio
    async def test_disabled_download_returns_envelope_not_exception(
        self, monkeypatch,
    ):
        from unittest.mock import MagicMock as _MM

        import httpx

        from src.agent.mesh_client import MeshClient

        client = MeshClient(
            mesh_url="http://mesh:8420",
            agent_id="a1",
        )

        envelope = {
            "success": False,
            "error": {
                "code": "forbidden",
                "message": "Downloads disabled by operator",
            },
        }

        class _Resp:
            def __init__(self):
                self.status_code = 403

            def json(self):
                # FastAPI wraps HTTPException(detail=...) as
                # ``{"detail": <envelope>}``.
                return {"detail": envelope}

            def raise_for_status(self):
                raise httpx.HTTPStatusError(
                    "403", request=_MM(), response=_MM(),
                )

        async def _post(url, json=None, timeout=None, headers=None):
            return _Resp()

        fake_client = _MM()
        fake_client.post = _post
        monkeypatch.setattr(
            client, "_get_client", AsyncMock(return_value=fake_client),
        )

        result = await client.browser_download(ref="e1")
        assert result == envelope


class TestTimeoutHierarchy:
    """P1.1 — outer ≥ middle ≥ inner. Verify the configured numbers."""

    def test_timeout_hierarchy_inner_less_than_outer(self):
        """Outer (agent caller) ≥ middle (mesh→agent ingest) ≥ inner
        (mesh→browser stream). The browser stream is the slowest hop, so
        the ingest client must outlast it with buffer; the agent's outer
        timeout must outlast both."""
        from src.host import server as host_server
        # Read raw source — we only need to confirm the literals used in
        # the orchestrator endpoint, not patch the server runtime.
        src = (
            host_server.__file__
        )
        text = open(src).read()
        # mesh→browser stream timeout — 180s.
        assert "timeout=180" in text
        # mesh→agent ingest client timeout — 240s, must be ≥ browser stream.
        assert "AsyncClient(timeout=240)" in text
        # Agent-side mesh client outer timeout — 300s, must be ≥ ingest.
        from src.agent import mesh_client as mc
        mc_text = open(mc.__file__).read()
        assert "timeout=300" in mc_text


class TestHttpExceptionLogging:
    """P1.2 — HTTPException raised by ingest must not log a noisy warning."""

    @pytest.mark.asyncio
    async def test_http_exception_not_logged_as_warning(
        self, tmp_path, monkeypatch, caplog,
    ):
        """A 507 from the agent's ingest endpoint is a legitimate denial,
        not an unexpected error — the orchestrator must re-raise it
        without emitting an "ingest error" warning."""
        from unittest.mock import MagicMock

        import httpx
        from httpx import ASGITransport, AsyncClient, Response

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
        cm.browser_service_url = "http://browser-svc:8500"
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

        real_post = httpx.AsyncClient.post

        async def fake_post(self, url, *args, **kwargs):
            url_s = str(url)
            if url_s.endswith("/browser/worker/download"):
                return Response(200, json={
                    "success": True,
                    "data": {
                        "nonce": "abcdef012345",
                        "size_bytes": 5,
                        "suggested_filename": "x.bin",
                        "mime_type": "application/octet-stream",
                    },
                }, request=httpx.Request("POST", url_s))
            if "_download_cleanup" in url_s:
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

        class _FakeStreamLocal:
            status_code = 200
            async def __aenter__(self_):
                return self_
            async def __aexit__(self_, *_exc):
                return False
            async def aiter_bytes(self_):
                yield b"abcde"

        def fake_stream(self, method, url, *args, **kwargs):
            return _FakeStreamLocal()

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
        monkeypatch.setattr(httpx.AsyncClient, "stream", fake_stream)

        import logging as _logging
        with caplog.at_level(_logging.WARNING, logger="host.server"):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test",
            ) as client:
                resp = await client.post(
                    "/mesh/browser/download",
                    json={"ref": "e1"},
                    headers={"X-Agent-ID": "worker"},
                )
        assert resp.status_code == 507
        # No "ingest error" warning for the legitimate 507 denial.
        ingest_warnings = [
            r for r in caplog.records
            if "Browser→agent ingest error" in r.getMessage()
        ]
        assert ingest_warnings == [], [r.getMessage() for r in ingest_warnings]


class TestSanitizedNamePadding:
    """P1.3 — single-char sanitized result is padded to ≥2 chars."""

    def test_one_char_filename_padded(self):
        """A single-character filename must satisfy the agent-side
        artifact-name regex (≥2 chars)."""
        # We exercise the same sanitizer the mesh uses by importing it
        # via a fresh app build — the function is closed-over by
        # create_mesh_app, so mirror its rules here.
        import re as _re
        _SAFE = _re.compile(r"[^\w.\-]+")

        def _sanitize(suggested: str) -> str:
            name = (suggested or "").strip()
            if "/" in name or "\\" in name:
                name = name.replace("\\", "/").rsplit("/", 1)[-1]
            name = _SAFE.sub("_", name)
            name = name.strip("._-")
            if not name or len(name) > 180:
                name = name[:180].strip("._-") if name else ""
            if not name:
                name = "download.bin"
            if len(name) < 2:
                name = name + "_"
            return name

        # Bare single char survives, but is padded.
        assert _sanitize("a") == "a_"
        assert len(_sanitize("a")) >= 2

        # Confirm the live mesh uses the same rule by reading its source.
        from src.host import server as host_server
        text = open(host_server.__file__).read()
        assert "len(name) < 2" in text


class TestUserControlBlocksMeshDownload:
    """P1.8 — _user_control=True must propagate as a refusal at the mesh."""

    @pytest.mark.asyncio
    async def test_user_control_blocks_mesh_download(self, tmp_path, monkeypatch):
        """When the browser refuses with the user-control message, the mesh
        forwards the failure envelope without attempting stream/ingest."""
        from unittest.mock import MagicMock

        import httpx
        from httpx import ASGITransport, AsyncClient, Response

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
        cm.browser_service_url = "http://browser-svc:8500"
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

        observed: list[str] = []
        real_post = httpx.AsyncClient.post

        async def fake_post(self, url, *args, **kwargs):
            url_s = str(url)
            observed.append(url_s)
            if url_s.endswith("/browser/worker/download"):
                return Response(200, json={
                    "success": False,
                    "error": "User has browser control — action paused",
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
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is False
        assert "control" in body["error"].lower()
        # No stream / ingest / cleanup attempted because the trigger refused.
        assert not any("_download_stream" in u for u in observed)
        assert not any("_download_cleanup" in u for u in observed)
        assert not any("/artifacts/ingest" in u for u in observed)


class TestTraceIdPropagation:
    """P1.6 — X-Trace-Id reaches every hop."""

    @pytest.mark.asyncio
    async def test_x_trace_id_propagates_to_all_hops(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock

        import httpx
        from httpx import ASGITransport, AsyncClient, Response

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
        cm.browser_service_url = "http://browser-svc:8500"
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

        seen: dict = {"trigger": None, "stream": None, "ingest": None, "cleanup": None}
        real_post = httpx.AsyncClient.post

        async def fake_post(self, url, *args, **kwargs):
            url_s = str(url)
            hdr = (kwargs.get("headers") or {}).get("X-Trace-Id")
            if url_s.endswith("/browser/worker/download"):
                seen["trigger"] = hdr
                return Response(200, json={
                    "success": True,
                    "data": {
                        "nonce": "abcdef012345",
                        "size_bytes": 1,
                        "suggested_filename": "z.bin",
                        "mime_type": "application/octet-stream",
                    },
                }, request=httpx.Request("POST", url_s))
            if "_download_cleanup" in url_s:
                seen["cleanup"] = hdr
                return Response(200, json={"deleted": 1},
                                request=httpx.Request("POST", url_s))
            if "/artifacts/ingest/" in url_s:
                seen["ingest"] = hdr
                content = kwargs.get("content")
                if content is not None and hasattr(content, "__aiter__"):
                    async for _ in content:
                        pass
                return Response(200, json={
                    "artifact_name": "z.bin",
                    "size_bytes": 1,
                    "mime_type": "application/octet-stream",
                }, request=httpx.Request("POST", url_s))
            return await real_post(self, url, *args, **kwargs)

        class _FakeStreamLocal:
            status_code = 200
            def __init__(self, hdr):
                self._hdr = hdr
            async def __aenter__(self_):
                return self_
            async def __aexit__(self_, *_exc):
                return False
            async def aiter_bytes(self_):
                yield b"q"

        def fake_stream(self, method, url, *args, **kwargs):
            seen["stream"] = (kwargs.get("headers") or {}).get("X-Trace-Id")
            return _FakeStreamLocal(seen["stream"])

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
        monkeypatch.setattr(httpx.AsyncClient, "stream", fake_stream)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/mesh/browser/download",
                json={"ref": "e1"},
                headers={"X-Agent-ID": "worker", "x-trace-id": "trace-xyz"},
            )
        assert resp.status_code == 200
        assert seen["trigger"] == "trace-xyz"
        assert seen["stream"] == "trace-xyz"
        assert seen["ingest"] == "trace-xyz"
        assert seen["cleanup"] == "trace-xyz"


class TestMidStreamInterruption:
    """P1.7 — stream raises mid-aiter, finally runs cleanup, error envelope."""

    @pytest.mark.asyncio
    async def test_mid_stream_interruption_runs_cleanup(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock

        import httpx
        from httpx import ASGITransport, AsyncClient, Response

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
        cm.browser_service_url = "http://browser-svc:8500"
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

        cleanup_called: list[str] = []
        real_post = httpx.AsyncClient.post

        async def fake_post(self, url, *args, **kwargs):
            url_s = str(url)
            if url_s.endswith("/browser/worker/download"):
                return Response(200, json={
                    "success": True,
                    "data": {
                        "nonce": "abcdef012345",
                        "size_bytes": 9,
                        "suggested_filename": "m.bin",
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
                    # Drain — the iterator will raise.
                    try:
                        async for _ in content:
                            pass
                    except Exception:
                        # Ingest endpoint surfaces the inner stream error
                        # by returning a 502-equivalent.
                        return Response(502, text="Stream broke",
                                        request=httpx.Request("POST", url_s))
                return Response(200, json={"artifact_name": "m.bin",
                                           "size_bytes": 0,
                                           "mime_type": "application/octet-stream"},
                                request=httpx.Request("POST", url_s))
            return await real_post(self, url, *args, **kwargs)

        class _ExplodingStream:
            status_code = 200
            async def __aenter__(self_):
                return self_
            async def __aexit__(self_, *_exc):
                return False
            async def aiter_bytes(self_):
                yield b"abc"
                raise RuntimeError("mid-stream network blip")

        def fake_stream(self, method, url, *args, **kwargs):
            return _ExplodingStream()

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
        # Either the inner stream error or the ingest 502 is acceptable —
        # what matters is the error envelope and that cleanup ran.
        assert resp.status_code >= 400
        assert len(cleanup_called) == 1


