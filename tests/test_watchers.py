"""Tests for FileWatcher: polling, dispatch, first-scan suppression."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.host.watchers import FileWatcher


class TestFileWatcher:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.watch_dir = Path(self._tmpdir) / "inbox"
        self.watch_dir.mkdir()

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_initial_scan_does_not_dispatch(self):
        (self.watch_dir / "existing.csv").write_text("a,b,c")
        dispatch = AsyncMock()
        watcher = FileWatcher(dispatch_fn=dispatch)
        watcher.watch(str(self.watch_dir), "*.csv", agent="test")

        await watcher._scan(dispatch=False)
        dispatch.assert_not_called()
        assert str(self.watch_dir / "existing.csv") in watcher._seen

    @pytest.mark.asyncio
    async def test_new_file_dispatches(self):
        dispatch = AsyncMock()
        watcher = FileWatcher(dispatch_fn=dispatch)
        watcher.watch(str(self.watch_dir), "*.csv", agent="researcher")

        # First scan: learn existing
        await watcher._scan(dispatch=False)

        # Add new file
        (self.watch_dir / "new.csv").write_text("x,y,z")
        await watcher._scan(dispatch=True)

        dispatch.assert_called_once()
        call_args = dispatch.call_args
        assert call_args[0][0] == "researcher"
        assert "new.csv" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_unchanged_file_no_redispatch(self):
        (self.watch_dir / "data.csv").write_text("a,b,c")
        dispatch = AsyncMock()
        watcher = FileWatcher(dispatch_fn=dispatch)
        watcher.watch(str(self.watch_dir), "*.csv", agent="test")

        await watcher._scan(dispatch=False)
        await watcher._scan(dispatch=True)
        dispatch.assert_not_called()

    @pytest.mark.asyncio
    async def test_ignores_non_matching_pattern(self):
        (self.watch_dir / "file.txt").write_text("hello")
        dispatch = AsyncMock()
        watcher = FileWatcher(dispatch_fn=dispatch)
        watcher.watch(str(self.watch_dir), "*.csv", agent="test")

        await watcher._scan(dispatch=False)
        (self.watch_dir / "file2.txt").write_text("world")
        await watcher._scan(dispatch=True)
        dispatch.assert_not_called()

    @pytest.mark.asyncio
    async def test_custom_message_template(self):
        dispatch = AsyncMock()
        watcher = FileWatcher(dispatch_fn=dispatch)
        watcher.watch(
            str(self.watch_dir), "*.csv", agent="test",
            message_template="Process {filename} at {filepath}",
        )

        await watcher._scan(dispatch=False)
        (self.watch_dir / "input.csv").write_text("data")
        await watcher._scan(dispatch=True)

        message = dispatch.call_args[0][1]
        assert "input.csv" in message

    @pytest.mark.asyncio
    async def test_nonexistent_watch_dir(self):
        dispatch = AsyncMock()
        watcher = FileWatcher(dispatch_fn=dispatch)
        watcher.watch("/nonexistent/path", "*.csv", agent="test")
        await watcher._scan(dispatch=True)
        dispatch.assert_not_called()

    def test_stop(self):
        watcher = FileWatcher()
        watcher._running = True
        watcher.stop()
        assert watcher._running is False
