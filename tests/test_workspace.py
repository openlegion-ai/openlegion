"""Tests for WorkspaceManager: file scaffold, loading, search, daily logs."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from src.agent.workspace import WorkspaceManager, _bm25_score, _tokenize


class TestWorkspaceScaffold:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.ws = WorkspaceManager(workspace_dir=self._tmpdir)

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_scaffold_creates_default_files(self):
        root = Path(self._tmpdir)
        assert (root / "AGENTS.md").exists()
        assert (root / "SOUL.md").exists()
        assert (root / "USER.md").exists()
        assert (root / "MEMORY.md").exists()
        assert (root / "memory").is_dir()

    def test_scaffold_preserves_existing_files(self):
        root = Path(self._tmpdir)
        (root / "AGENTS.md").write_text("custom instructions")
        WorkspaceManager(workspace_dir=self._tmpdir)
        assert (root / "AGENTS.md").read_text() == "custom instructions"

    def test_load_prompt_context_returns_file_contents(self):
        root = Path(self._tmpdir)
        (root / "AGENTS.md").write_text("# My Agent\nDo X.")
        (root / "SOUL.md").write_text("# Identity\nFriendly.")
        (root / "USER.md").write_text("# User\nPrefers Python.")
        ws = WorkspaceManager(workspace_dir=self._tmpdir)
        context = ws.load_prompt_context()
        assert "Do X." in context
        assert "Friendly." in context
        assert "Prefers Python." in context

    def test_load_prompt_context_skips_empty(self):
        root = Path(self._tmpdir)
        (root / "AGENTS.md").write_text("")
        (root / "SOUL.md").write_text("Identity")
        (root / "USER.md").write_text("")
        ws = WorkspaceManager(workspace_dir=self._tmpdir)
        context = ws.load_prompt_context()
        assert "Identity" in context
        assert "---" not in context  # no separator for single file


class TestDailyLogs:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.ws = WorkspaceManager(workspace_dir=self._tmpdir)

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_append_daily_log_creates_file(self):
        self.ws.append_daily_log("User asked about Python")
        log_files = list((Path(self._tmpdir) / "memory").glob("*.md"))
        assert len(log_files) == 1
        content = log_files[0].read_text()
        assert "User asked about Python" in content

    def test_append_daily_log_has_timestamp(self):
        self.ws.append_daily_log("fact one")
        log_files = list((Path(self._tmpdir) / "memory").glob("*.md"))
        content = log_files[0].read_text()
        assert content.startswith("- [")
        assert "] fact one" in content

    def test_multiple_appends_same_file(self):
        self.ws.append_daily_log("fact one")
        self.ws.append_daily_log("fact two")
        log_files = list((Path(self._tmpdir) / "memory").glob("*.md"))
        assert len(log_files) == 1
        content = log_files[0].read_text()
        assert "fact one" in content
        assert "fact two" in content

    def test_load_daily_logs(self):
        self.ws.append_daily_log("today's fact")
        logs = self.ws.load_daily_logs(days=1)
        assert "today's fact" in logs

    def test_load_daily_logs_empty(self):
        logs = self.ws.load_daily_logs(days=2)
        assert logs == ""


class TestMemoryFile:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.ws = WorkspaceManager(workspace_dir=self._tmpdir)

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_load_memory_default(self):
        content = self.ws.load_memory()
        assert "Long-Term Memory" in content

    def test_append_memory(self):
        self.ws.append_memory("User prefers dark mode")
        content = self.ws.load_memory()
        assert "dark mode" in content

    def test_append_memory_preserves_existing(self):
        (Path(self._tmpdir) / "MEMORY.md").write_text("Existing fact\n")
        ws = WorkspaceManager(workspace_dir=self._tmpdir)
        ws.append_memory("New fact")
        content = ws.load_memory()
        assert "Existing fact" in content
        assert "New fact" in content


class TestBM25Search:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.ws = WorkspaceManager(workspace_dir=self._tmpdir)

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_search_finds_relevant_file(self):
        (Path(self._tmpdir) / "MEMORY.md").write_text(
            "# Memory\n\nUser loves Python and machine learning.\n"
        )
        (Path(self._tmpdir) / "USER.md").write_text(
            "# User\n\nPrefers TypeScript and React.\n"
        )
        results = self.ws.search("Python machine learning")
        assert len(results) > 0
        assert results[0]["file"] == "MEMORY.md"
        assert results[0]["score"] > 0

    def test_search_returns_empty_for_no_match(self):
        results = self.ws.search("xyznonexistent12345")
        assert results == []

    def test_search_respects_max_results(self):
        for i in range(10):
            (Path(self._tmpdir) / f"note_{i}.md").write_text(f"Python note {i}")
        results = self.ws.search("Python", max_results=3)
        assert len(results) <= 3

    def test_search_daily_logs(self):
        memory_dir = Path(self._tmpdir) / "memory"
        memory_dir.mkdir(exist_ok=True)
        (memory_dir / "2026-02-17.md").write_text(
            "- [10:00] Discussed API design with user\n"
            "- [11:00] User prefers RESTful APIs\n"
        )
        results = self.ws.search("API design")
        assert len(results) > 0
        assert "memory/2026-02-17.md" in results[0]["file"]

    def test_search_snippet_contains_relevant_text(self):
        (Path(self._tmpdir) / "MEMORY.md").write_text(
            "# Memory\n\nThe user's favorite color is blue.\n"
            "They also like green but not red.\n"
        )
        results = self.ws.search("favorite color")
        assert len(results) > 0
        assert "color" in results[0]["snippet"].lower() or "blue" in results[0]["snippet"].lower()


class TestTokenize:
    def test_basic_tokenize(self):
        tokens = _tokenize("Hello World!")
        assert "hello" in tokens
        assert "world" in tokens

    def test_stop_words_removed(self):
        tokens = _tokenize("the cat is on the mat")
        assert "the" not in tokens
        assert "is" not in tokens
        assert "cat" in tokens
        assert "mat" in tokens

    def test_short_words_removed(self):
        tokens = _tokenize("I a am do it go")
        assert "i" not in tokens
        assert "a" not in tokens


class TestBM25Core:
    def test_bm25_ranks_relevant_higher(self):
        docs = [
            ("a.md", _tokenize("python machine learning data science"), ""),
            ("b.md", _tokenize("javascript react frontend web"), ""),
            ("c.md", _tokenize("python deep learning neural network"), ""),
        ]
        query = _tokenize("python machine learning")
        scores = _bm25_score(query, docs)
        assert scores[0] > scores[1]  # a.md more relevant than b.md

    def test_bm25_all_zero_for_no_match(self):
        docs = [("a.md", _tokenize("completely unrelated content"), "")]
        query = _tokenize("xyznonexistent")
        scores = _bm25_score(query, docs)
        assert scores[0] == 0.0

    def test_list_memory_files(self):
        tmpdir = tempfile.mkdtemp()
        try:
            ws = WorkspaceManager(workspace_dir=tmpdir)
            ws.append_daily_log("fact")
            files = ws.list_memory_files()
            assert len(files) == 1
            assert files[0].startswith("memory/")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
