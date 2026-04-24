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

        result = await mgr.upload_file("a1", "e1", ["/tmp/foo.pdf"])
        assert result["success"] is True
        assert result["data"]["uploaded"] == ["/tmp/foo.pdf"]
        chooser.set_files.assert_awaited_once_with(["/tmp/foo.pdf"])
        fake_locator.click.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_ref_not_found_returns_error(self, tmp_path, monkeypatch):
        mgr = BrowserManager(profiles_dir=str(tmp_path / "p"))
        inst = _make_instance()
        monkeypatch.setattr(mgr, "_locator_from_ref", lambda _i, _r: None)
        monkeypatch.setattr(mgr, "get_or_start", AsyncMock(return_value=inst))
        result = await mgr.upload_file("a1", "missing", ["/tmp/x"])
        assert result["success"] is False
        assert "missing" in result["error"]

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
