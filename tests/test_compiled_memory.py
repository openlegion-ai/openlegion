"""Tests for MEMORY.md 'compiled truth + timeline' structure + consolidation.

Covers the compiled-head / append-only-log split in WorkspaceManager and the
context-manager consolidation step folded into compaction.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.context import ContextManager
from src.agent.workspace import (
    _MEMORY_FILE_MAX,
    MEMORY_COMPILED_BEGIN,
    MEMORY_COMPILED_END,
    WorkspaceManager,
)
from src.shared.types import LLMResponse


class TestCompiledMemorySplit:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.ws = WorkspaceManager(workspace_dir=self._tmpdir)

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_append_migrates_legacy_body_into_compiled_head(self):
        """A legacy MEMORY.md body migrates into the compiled head on first
        structured append; the appended content lands in the log."""
        (Path(self._tmpdir) / "MEMORY.md").write_text(
            "# Long-Term Memory\n\nLEGACY_BODY_FACT\n"
        )
        ws = WorkspaceManager(workspace_dir=self._tmpdir)
        ws.append_memory("## Extracted\n\nLOG_ONLY_ENTRY")

        head = ws.load_compiled_memory()
        log = ws.load_memory_log()
        assert "LEGACY_BODY_FACT" in head
        assert "LOG_ONLY_ENTRY" not in head
        assert "LOG_ONLY_ENTRY" in log
        # File is now structured.
        raw = (Path(self._tmpdir) / "MEMORY.md").read_text()
        assert MEMORY_COMPILED_BEGIN in raw
        assert MEMORY_COMPILED_END in raw

    def test_marker_strings_in_content_do_not_corrupt_split(self):
        """Literal COMPILED marker strings inside appended content are
        neutralized so they can't corrupt the head/log split or cause silent
        data loss on the next compiled rewrite."""
        self.ws.write_compiled_memory("DURABLE_HEAD_FACT")
        self.ws.append_memory(
            f"discussion of {MEMORY_COMPILED_BEGIN} and {MEMORY_COMPILED_END} markers"
        )
        # Exactly one structural marker pair survives in the file.
        raw = (Path(self._tmpdir) / "MEMORY.md").read_text()
        assert raw.count(MEMORY_COMPILED_BEGIN) == 1
        assert raw.count(MEMORY_COMPILED_END) == 1
        # Head intact; a subsequent rewrite preserves the (sanitized) log.
        assert self.ws.load_compiled_memory() == "DURABLE_HEAD_FACT"
        assert "markers" in self.ws.load_memory_log()
        self.ws.write_compiled_memory("NEW_HEAD")
        assert self.ws.load_compiled_memory() == "NEW_HEAD"
        assert "markers" in self.ws.load_memory_log()

    def test_hand_edited_stray_markers_preserve_content(self):
        """A file with marker text NOT anchored at the top (e.g. a user hand-
        edit via the workspace editor pasting marker-looking content) is treated
        as a legacy whole-head, so no content is dropped on the next append."""
        path = Path(self._tmpdir) / "MEMORY.md"
        path.write_text(
            "# Long-Term Memory\n\nIMPORTANT_USER_FACT\n\n"
            f"a note mentioning {MEMORY_COMPILED_BEGIN} and {MEMORY_COMPILED_END}\n"
        )
        ws = WorkspaceManager(workspace_dir=self._tmpdir)
        head, log = ws._split_memory(path.read_text())
        assert "IMPORTANT_USER_FACT" in head  # not split on the stray markers
        assert log == ""
        # The next append migrates that whole body into the compiled head.
        ws.append_memory("## Extracted\n\nNEW_LOG_ENTRY")
        assert "IMPORTANT_USER_FACT" in ws.load_compiled_memory()
        assert "NEW_LOG_ENTRY" in ws.load_memory_log()

    def test_bootstrap_injects_head_and_recent_log(self):
        """get_bootstrap_content injects the compiled head AND a recent log
        entry, so recently-learned facts auto-surface in later sessions."""
        self.ws.write_compiled_memory("COMPILED_HEAD_MARKER")
        self.ws.append_memory("RECENT_LOG_FACT")
        content = self.ws.get_bootstrap_content()
        assert "COMPILED_HEAD_MARKER" in content
        assert "RECENT_LOG_FACT" in content

    def test_bootstrap_excludes_old_log_beyond_recent_window(self):
        """Log entries pushed beyond the recent window are search-only (not
        injected), but stay in the searchable log."""
        from src.agent.workspace import _MEMORY_RECENT_LOG_CHARS
        self.ws.write_compiled_memory("HEAD")
        self.ws.append_memory("## OLD\n\nOLD_LOG_FACT_MARKER")
        filler = "## F\n\n" + ("x" * 1000)
        for _ in range((_MEMORY_RECENT_LOG_CHARS // 1000) + 2):
            self.ws.append_memory(filler)
        content = self.ws.get_bootstrap_content()
        assert "OLD_LOG_FACT_MARKER" not in content                 # not injected
        assert "OLD_LOG_FACT_MARKER" in self.ws.load_memory_log()   # still searchable

    def test_write_compiled_memory_replaces_head_preserves_log(self):
        self.ws.append_memory("PERSISTENT_LOG_ENTRY")
        self.ws.write_compiled_memory("NEW HEAD")

        assert self.ws.load_compiled_memory() == "NEW HEAD"
        assert "PERSISTENT_LOG_ENTRY" in self.ws.load_memory_log()

    def test_trim_memory_log_keeps_file_bounded_and_head(self):
        """Appending past _MEMORY_FILE_MAX trims the oldest log entries but
        the compiled head survives and the file stays bounded."""
        self.ws.write_compiled_memory("DURABLE_HEAD")
        # Each append is a distinct entry; push well past the cap.
        chunk = "x" * 4_000
        for i in range(40):
            self.ws.append_memory(f"## Entry {i}\n\n{chunk}")

        raw = (Path(self._tmpdir) / "MEMORY.md").read_text()
        # Head + markers + bounded log + trim notice + small overhead.
        assert len(raw) <= _MEMORY_FILE_MAX + 2_000
        assert "DURABLE_HEAD" in self.ws.load_compiled_memory()
        # The most-recent entry must survive; the oldest must be trimmed.
        assert "Entry 39" in self.ws.load_memory_log()
        assert "Entry 0\n" not in self.ws.load_memory_log()

    def test_log_entry_remains_bm25_searchable(self):
        """Regression guard: log-only content stays searchable via BM25 even
        though it is never injected into the prompt."""
        self.ws.write_compiled_memory("Head with unrelated words")
        self.ws.append_memory(
            "## Extracted\n\n- **kubernetes_namespace**: prod-payments-cluster"
        )
        results = self.ws.search("kubernetes namespace payments")
        files = {r["file"] for r in results}
        assert "MEMORY.md" in files

    def test_legacy_file_without_markers_still_injects_whole_body(self):
        """Back-compat: a MEMORY.md with no compiled markers injects its
        entire body exactly as before (load_compiled_memory == whole body)."""
        body = "# Long-Term Memory\n\nLEGACY_INJECTED_VERBATIM\n"
        (Path(self._tmpdir) / "MEMORY.md").write_text(body)
        ws = WorkspaceManager(workspace_dir=self._tmpdir)

        assert "LEGACY_INJECTED_VERBATIM" in ws.load_compiled_memory()
        assert "LEGACY_INJECTED_VERBATIM" in ws.get_bootstrap_content()
        assert ws.load_memory_log() == ""

    def test_consolidation_stamp_due_then_not_due(self):
        # Never consolidated → due.
        assert self.ws.consolidation_due(10_000) is True
        self.ws.mark_consolidated()
        # Just stamped → not due within a large window.
        assert self.ws.consolidation_due(10_000) is False
        # Zero interval → always due.
        assert self.ws.consolidation_due(0) is True


def _fake_fact(key: str, value: str):
    f = MagicMock()
    f.key = key
    f.value = value
    return f


class TestConsolidateMemory:
    def _make_workspace(self, *, due: bool, log: str, head: str = "OLD HEAD"):
        ws = MagicMock()
        ws.consolidation_due.return_value = due
        ws.load_memory_log.return_value = log
        ws.load_compiled_memory.return_value = head
        return ws

    @pytest.mark.asyncio
    async def test_consolidates_when_due_and_log_long_enough(self):
        ws = self._make_workspace(due=True, log="L" * 2_000)
        llm = MagicMock()
        llm.chat = AsyncMock(
            return_value=LLMResponse(content="NEW COMPILED HEAD", tokens_used=20)
        )
        memory = MagicMock()
        memory.get_high_salience_facts = AsyncMock(
            return_value=[_fake_fact("pref", "dark mode")]
        )

        cm = ContextManager(
            max_tokens=1000, llm=llm, workspace=ws, memory=memory,
        )
        await cm._maybe_consolidate_memory()

        llm.chat.assert_awaited_once()
        ws.write_compiled_memory.assert_called_once()
        assert "NEW COMPILED HEAD" in ws.write_compiled_memory.call_args[0][0]
        ws.mark_consolidated.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_when_not_due(self):
        ws = self._make_workspace(due=False, log="L" * 5_000)
        llm = MagicMock()
        llm.chat = AsyncMock()

        cm = ContextManager(max_tokens=1000, llm=llm, workspace=ws, memory=None)
        await cm._maybe_consolidate_memory()

        llm.chat.assert_not_called()
        ws.write_compiled_memory.assert_not_called()
        ws.mark_consolidated.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_log_too_short(self):
        ws = self._make_workspace(due=True, log="tiny")
        llm = MagicMock()
        llm.chat = AsyncMock()

        cm = ContextManager(max_tokens=1000, llm=llm, workspace=ws, memory=None)
        await cm._maybe_consolidate_memory()

        llm.chat.assert_not_called()
        ws.write_compiled_memory.assert_not_called()
