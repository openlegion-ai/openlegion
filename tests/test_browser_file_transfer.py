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
        monkeypatch.setattr(mgr, "_locator_from_ref", lambda _i, _r: fake_locator)
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
            mgr, "_locator_from_ref", lambda _i, _r: fake_locator,
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
