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
            mgr, "_locator_from_ref", lambda _inst, _ref: fake_locator,
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
            mgr, "_locator_from_ref", lambda _i, _r: fake_locator,
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
        monkeypatch.setattr(mgr, "_locator_from_ref", lambda _i, _r: None)
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
        monkeypatch.setattr(mgr, "_locator_from_ref", lambda _i, _r: fake_locator)
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
        monkeypatch.setattr(mgr, "_locator_from_ref", lambda _i, _r: fake_locator)
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
