"""File watcher — triggers agent actions when files appear or change.

Uses polling (not inotify) for Docker volume compatibility.
Checks every POLL_INTERVAL seconds for new/modified files matching
configured glob patterns.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable, Optional

from src.shared.utils import setup_logging

logger = setup_logging("host.watchers")


class FileWatcher:
    """Watches directories for new/modified files, dispatches to agents."""

    POLL_INTERVAL = 5

    def __init__(self, dispatch_fn: Optional[Callable] = None):
        self.watches: list[dict] = []
        self.dispatch_fn = dispatch_fn
        self._seen: dict[str, float] = {}
        self._running = False
        self._first_scan_done = False

    def watch(
        self,
        path: str,
        pattern: str,
        agent: str,
        message_template: str | None = None,
    ) -> None:
        self.watches.append({
            "path": path,
            "pattern": pattern,
            "agent": agent,
            "template": message_template or "New file detected: {filepath}\nFilename: {filename}\nProcess this file.",
        })

    async def start(self) -> None:
        self._running = True
        logger.info(f"File watcher started with {len(self.watches)} watches")
        # Initial scan: record existing files without dispatching
        await self._scan(dispatch=False)
        self._first_scan_done = True
        while self._running:
            await self._scan(dispatch=True)
            await asyncio.sleep(self.POLL_INTERVAL)

    def stop(self) -> None:
        self._running = False

    async def _scan(self, dispatch: bool = True) -> None:
        for watch in self.watches:
            watch_path = Path(watch["path"])
            if not watch_path.exists():
                continue
            for filepath in watch_path.glob(watch["pattern"]):
                if not filepath.is_file():
                    continue
                try:
                    mtime = filepath.stat().st_mtime
                except OSError:
                    continue
                key = str(filepath)
                if key in self._seen and self._seen[key] >= mtime:
                    continue
                self._seen[key] = mtime
                if dispatch and self.dispatch_fn:
                    message = watch["template"].format(
                        filepath=str(filepath),
                        filename=filepath.name,
                    )
                    await self._dispatch_safe(watch["agent"], message, filepath.name)

    async def _dispatch_safe(self, agent: str, message: str, filename: str) -> None:
        try:
            await self.dispatch_fn(agent, message)
            logger.info(f"File trigger: {filename} -> agent '{agent}'")
        except Exception as e:
            logger.error(f"File watcher dispatch failed for {filename}: {e}")
