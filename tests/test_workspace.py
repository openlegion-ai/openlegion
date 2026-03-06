"""Tests for WorkspaceManager: file scaffold, loading, search, daily logs, learnings."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from src.agent.workspace import (
    _MAX_SYSTEM,
    WorkspaceManager,
    _bm25_score,
    _maybe_add_header,
    _tokenize,
    generate_system_md,
)


class TestWorkspaceScaffold:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.ws = WorkspaceManager(workspace_dir=self._tmpdir)

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_scaffold_creates_default_files(self):
        root = Path(self._tmpdir)
        assert (root / "INSTRUCTIONS.md").exists()
        assert (root / "SOUL.md").exists()
        assert (root / "USER.md").exists()
        assert (root / "MEMORY.md").exists()
        assert (root / "HEARTBEAT.md").exists()
        assert (root / "memory").is_dir()
        assert (root / "learnings").is_dir()

    def test_scaffold_preserves_existing_files(self):
        root = Path(self._tmpdir)
        (root / "INSTRUCTIONS.md").write_text("custom instructions")
        WorkspaceManager(workspace_dir=self._tmpdir)
        assert (root / "INSTRUCTIONS.md").read_text() == "custom instructions"

    def test_initial_instructions_seeds_instructions_md(self):
        """When initial_instructions is provided, INSTRUCTIONS.md is seeded with that content."""
        tmpdir = tempfile.mkdtemp()
        try:
            WorkspaceManager(workspace_dir=tmpdir, initial_instructions="You are a sales agent.\nBe concise.")
            root = Path(tmpdir)
            content = (root / "INSTRUCTIONS.md").read_text()
            assert content.startswith("# Instructions")
            assert "You are a sales agent." in content
            assert "Be concise." in content
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_initial_instructions_does_not_overwrite_existing(self):
        """If INSTRUCTIONS.md already exists, initial_instructions is ignored."""
        tmpdir = tempfile.mkdtemp()
        try:
            root = Path(tmpdir)
            root.mkdir(exist_ok=True)
            (root / "INSTRUCTIONS.md").write_text("# Existing instructions\nDo not change.")
            WorkspaceManager(workspace_dir=tmpdir, initial_instructions="New instructions")
            assert (root / "INSTRUCTIONS.md").read_text() == "# Existing instructions\nDo not change."
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_initial_instructions_empty_uses_default(self):
        """When initial_instructions is empty, default scaffold content is used."""
        tmpdir = tempfile.mkdtemp()
        try:
            WorkspaceManager(workspace_dir=tmpdir, initial_instructions="")
            root = Path(tmpdir)
            content = (root / "INSTRUCTIONS.md").read_text()
            # Default scaffold content, not custom instructions
            assert "# Instructions" in content
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_initial_soul_seeds_soul_md(self):
        """When initial_soul is provided, SOUL.md is seeded with that content."""
        tmpdir = tempfile.mkdtemp()
        try:
            WorkspaceManager(workspace_dir=tmpdir, initial_soul="You are a pirate captain.")
            root = Path(tmpdir)
            content = (root / "SOUL.md").read_text()
            assert content.startswith("# Identity")
            assert "You are a pirate captain." in content
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_initial_soul_does_not_overwrite_existing(self):
        """If SOUL.md already exists, initial_soul is ignored."""
        tmpdir = tempfile.mkdtemp()
        try:
            root = Path(tmpdir)
            root.mkdir(exist_ok=True)
            (root / "SOUL.md").write_text("# Existing soul\nDo not change.")
            WorkspaceManager(workspace_dir=tmpdir, initial_soul="New soul")
            assert (root / "SOUL.md").read_text() == "# Existing soul\nDo not change."
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_initial_soul_empty_uses_default(self):
        """When initial_soul is empty, default scaffold content is used."""
        tmpdir = tempfile.mkdtemp()
        try:
            WorkspaceManager(workspace_dir=tmpdir, initial_soul="")
            root = Path(tmpdir)
            content = (root / "SOUL.md").read_text()
            assert content.strip() == "# Identity"  # minimal stub
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_initial_heartbeat_seeds_heartbeat_md(self):
        """When initial_heartbeat is provided, HEARTBEAT.md is seeded with that content."""
        tmpdir = tempfile.mkdtemp()
        try:
            WorkspaceManager(workspace_dir=tmpdir, initial_heartbeat="Check email every hour.")
            root = Path(tmpdir)
            content = (root / "HEARTBEAT.md").read_text()
            assert content.startswith("# Heartbeat Rules")
            assert "Check email every hour." in content
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_initial_heartbeat_does_not_overwrite_existing(self):
        """If HEARTBEAT.md already exists, initial_heartbeat is ignored."""
        tmpdir = tempfile.mkdtemp()
        try:
            root = Path(tmpdir)
            root.mkdir(exist_ok=True)
            (root / "HEARTBEAT.md").write_text("# Custom rules\nDo not change.")
            WorkspaceManager(workspace_dir=tmpdir, initial_heartbeat="New rules")
            assert (root / "HEARTBEAT.md").read_text() == "# Custom rules\nDo not change."
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_all_three_seeded_together(self):
        """All three initial_* params work simultaneously."""
        tmpdir = tempfile.mkdtemp()
        try:
            WorkspaceManager(
                workspace_dir=tmpdir,
                initial_instructions="Do research.",
                initial_soul="You are curious.",
                initial_heartbeat="Monitor news.",
            )
            root = Path(tmpdir)
            assert "Do research." in (root / "INSTRUCTIONS.md").read_text()
            assert "You are curious." in (root / "SOUL.md").read_text()
            assert "Monitor news." in (root / "HEARTBEAT.md").read_text()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_migration_agents_md_to_instructions_md(self):
        """Existing AGENTS.md is renamed to INSTRUCTIONS.md on scaffold."""
        tmpdir = tempfile.mkdtemp()
        try:
            root = Path(tmpdir)
            root.mkdir(exist_ok=True)
            (root / "AGENTS.md").write_text("# My Old Instructions\nDo the thing.")
            WorkspaceManager(workspace_dir=tmpdir)
            assert not (root / "AGENTS.md").exists()
            assert (root / "INSTRUCTIONS.md").exists()
            assert "Do the thing." in (root / "INSTRUCTIONS.md").read_text()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_migration_does_not_overwrite_instructions_md(self):
        """If both AGENTS.md and INSTRUCTIONS.md exist, migration is skipped."""
        tmpdir = tempfile.mkdtemp()
        try:
            root = Path(tmpdir)
            root.mkdir(exist_ok=True)
            (root / "AGENTS.md").write_text("old content")
            (root / "INSTRUCTIONS.md").write_text("new content")
            WorkspaceManager(workspace_dir=tmpdir)
            # Both files remain; INSTRUCTIONS.md is not overwritten
            assert (root / "AGENTS.md").exists()
            assert (root / "INSTRUCTIONS.md").read_text() == "new content"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_initial_instructions_seeds_after_migration(self):
        """If AGENTS.md exists, migration renames it so initial_instructions is ignored."""
        tmpdir = tempfile.mkdtemp()
        try:
            root = Path(tmpdir)
            root.mkdir(exist_ok=True)
            (root / "AGENTS.md").write_text("# Legacy instructions")
            WorkspaceManager(workspace_dir=tmpdir, initial_instructions="New instructions")
            # Migration renamed AGENTS.md → INSTRUCTIONS.md, so initial_instructions is ignored
            assert "Legacy instructions" in (root / "INSTRUCTIONS.md").read_text()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_load_prompt_context_returns_file_contents(self):
        root = Path(self._tmpdir)
        (root / "INSTRUCTIONS.md").write_text("# My Agent\nDo X.")
        (root / "SOUL.md").write_text("# Identity\nFriendly.")
        (root / "USER.md").write_text("# User\nPrefers Python.")
        ws = WorkspaceManager(workspace_dir=self._tmpdir)
        context = ws.load_prompt_context()
        assert "Do X." in context
        assert "Friendly." in context
        assert "Prefers Python." in context

    def test_load_prompt_context_skips_empty(self):
        root = Path(self._tmpdir)
        (root / "INSTRUCTIONS.md").write_text("")
        (root / "SOUL.md").write_text("Identity")
        (root / "USER.md").write_text("")
        ws = WorkspaceManager(workspace_dir=self._tmpdir)
        context = ws.load_prompt_context()
        assert "Identity" in context
        assert "---" not in context  # no separator for single file

    def test_load_prompt_context_includes_project_md(self):
        root = Path(self._tmpdir)
        (root / "PROJECT.md").write_text("## What We're Building\nA lead gen tool.")
        ws = WorkspaceManager(workspace_dir=self._tmpdir)
        context = ws.load_prompt_context()
        assert "lead gen tool" in context

    def test_load_prompt_context_without_project_md(self):
        """PROJECT.md is optional -- no error if missing."""
        root = Path(self._tmpdir)
        (root / "PROJECT.md").unlink(missing_ok=True)
        ws = WorkspaceManager(workspace_dir=self._tmpdir)
        context = ws.load_prompt_context()
        assert "Instructions" in context


class TestBootstrapContent:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.ws = WorkspaceManager(workspace_dir=self._tmpdir)

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_bootstrap_includes_all_workspace_files(self):
        root = Path(self._tmpdir)
        (root / "INSTRUCTIONS.md").write_text("Agent instructions here")
        (root / "SOUL.md").write_text("Soul content")
        (root / "USER.md").write_text("User prefs")
        (root / "MEMORY.md").write_text("Important memory")
        ws = WorkspaceManager(workspace_dir=self._tmpdir)
        content = ws.get_bootstrap_content()
        assert "Agent instructions" in content
        assert "Soul content" in content
        assert "User prefs" in content
        assert "Important memory" in content

    def test_bootstrap_truncates_large_files(self):
        root = Path(self._tmpdir)
        # Write a MEMORY.md larger than _MAX_MEMORY (16000 chars)
        (root / "MEMORY.md").write_text("x" * 20_000)
        ws = WorkspaceManager(workspace_dir=self._tmpdir)
        content = ws.get_bootstrap_content()
        assert "truncated" in content
        assert "memory_search" in content

    def test_bootstrap_enforces_total_cap(self):
        root = Path(self._tmpdir)
        # Fill every file close to its per-file cap to exceed total
        (root / "INSTRUCTIONS.md").write_text("A" * 8_000)
        (root / "SOUL.md").write_text("S" * 4_000)
        (root / "USER.md").write_text("U" * 4_000)
        (root / "MEMORY.md").write_text("M" * 16_000)
        (root / "PROJECT.md").write_text("P" * 15_000)
        ws = WorkspaceManager(workspace_dir=self._tmpdir)
        content = ws.get_bootstrap_content()
        assert len(content) <= 40_000 + 100  # small margin for truncation text

    def test_bootstrap_skips_empty_files(self):
        root = Path(self._tmpdir)
        (root / "INSTRUCTIONS.md").write_text("")
        (root / "SOUL.md").write_text("")
        (root / "USER.md").write_text("User info")
        (root / "MEMORY.md").write_text("")
        ws = WorkspaceManager(workspace_dir=self._tmpdir)
        content = ws.get_bootstrap_content()
        assert "User info" in content
        assert content.count("---") == 0  # only one file, no separators

    def test_bootstrap_does_not_include_daily_logs(self):
        root = Path(self._tmpdir)
        memory_dir = root / "memory"
        memory_dir.mkdir(exist_ok=True)
        (memory_dir / "2026-02-19.md").write_text("Daily log entry here")
        ws = WorkspaceManager(workspace_dir=self._tmpdir)
        content = ws.get_bootstrap_content()
        assert "Daily log entry here" not in content

    def test_bootstrap_includes_project_md(self):
        root = Path(self._tmpdir)
        (root / "PROJECT.md").write_text("## Project\nWe are building X.")
        ws = WorkspaceManager(workspace_dir=self._tmpdir)
        content = ws.get_bootstrap_content()
        assert "We are building X" in content

    def test_bootstrap_adds_header_when_no_heading(self):
        """Files without a markdown heading get a descriptive header prepended."""
        root = Path(self._tmpdir)
        (root / "USER.md").write_text("Prefers concise answers.")
        ws = WorkspaceManager(workspace_dir=self._tmpdir)
        content = ws.get_bootstrap_content()
        assert "## USER.md — User Preferences & Corrections" in content
        assert "Prefers concise answers" in content

    def test_bootstrap_skips_header_when_heading_exists(self):
        """Files that already start with a markdown heading are not double-labeled."""
        root = Path(self._tmpdir)
        (root / "SOUL.md").write_text("# My Identity\n\nI am a poet.")
        ws = WorkspaceManager(workspace_dir=self._tmpdir)
        content = ws.get_bootstrap_content()
        # Should NOT have the auto-header since file starts with #
        assert "## SOUL.md — Your Identity" not in content
        assert "# My Identity" in content
        assert "I am a poet" in content

    def test_bootstrap_adds_header_to_project_md(self):
        """PROJECT.md without heading gets its header."""
        root = Path(self._tmpdir)
        (root / "PROJECT.md").write_text("We are building a fleet.")
        ws = WorkspaceManager(workspace_dir=self._tmpdir)
        content = ws.get_bootstrap_content()
        assert "## PROJECT.md — Fleet-Wide Context" in content


class TestMaybeAddHeader:
    def test_adds_header_for_known_file(self):
        result = _maybe_add_header("USER.md", "Prefers concise answers.")
        assert result.startswith("## USER.md — User Preferences & Corrections")
        assert "Prefers concise answers." in result

    def test_skips_header_when_content_starts_with_heading(self):
        content = "# My Identity\n\nI am a poet."
        result = _maybe_add_header("SOUL.md", content)
        assert result == content

    def test_skips_header_when_content_starts_with_heading_and_whitespace(self):
        content = "\n  # My Identity\n\nI am a poet."
        result = _maybe_add_header("SOUL.md", content)
        assert result == content

    def test_skips_header_for_unknown_file(self):
        result = _maybe_add_header("CUSTOM.md", "custom content")
        assert result == "custom content"

    def test_empty_content(self):
        result = _maybe_add_header("USER.md", "")
        assert "## USER.md" in result


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


class TestLearnings:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.ws = WorkspaceManager(workspace_dir=self._tmpdir)

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_record_error_creates_file(self):
        self.ws.record_error("exec", "Command not found: foo", "Tried to run foo")
        path = Path(self._tmpdir) / "learnings" / "errors.md"
        assert path.exists()
        content = path.read_text()
        assert "exec" in content
        assert "Command not found" in content
        assert "Context: Tried to run foo" in content

    def test_record_correction_creates_file(self):
        self.ws.record_correction("I used JSON format", "No, please use YAML instead")
        path = Path(self._tmpdir) / "learnings" / "corrections.md"
        assert path.exists()
        content = path.read_text()
        assert "JSON format" in content
        assert "YAML instead" in content

    def test_get_learnings_context_empty_when_no_learnings(self):
        context = self.ws.get_learnings_context()
        assert context == ""

    def test_get_learnings_context_includes_errors(self):
        self.ws.record_error("http_request", "Timeout", "url=https://example.com")
        context = self.ws.get_learnings_context()
        assert "Timeout" in context
        assert "Recent Errors" in context

    def test_get_learnings_context_includes_corrections(self):
        self.ws.record_correction("Said hello", "Actually say 'hi'")
        context = self.ws.get_learnings_context()
        assert "Corrections" in context

    def test_looks_like_correction(self):
        assert self.ws.looks_like_correction("No, I meant something else")
        assert self.ws.looks_like_correction("Wrong, that's not right")
        assert self.ws.looks_like_correction("Actually, use YAML")
        assert not self.ws.looks_like_correction("What's the weather?")
        assert not self.ws.looks_like_correction("Tell me about Python")

    def test_rotate_large_file(self):
        path = Path(self._tmpdir) / "learnings" / "errors.md"
        # Write a large file
        with path.open("w") as f:
            for i in range(5000):
                f.write(f"- Error {i}: something went wrong\n")
        original_lines = len(path.read_text().splitlines())
        self.ws._rotate_if_large(path)
        new_lines = len(path.read_text().splitlines())
        assert new_lines < original_lines

    def test_load_heartbeat_rules(self):
        rules = self.ws.load_heartbeat_rules()
        assert "Heartbeat Rules" in rules


class TestPerAgentSoul:
    """SOUL.md is per-agent — each agent has its own identity file."""

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.ws = WorkspaceManager(workspace_dir=self._tmpdir)

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_custom_soul_in_bootstrap(self):
        """Custom SOUL.md content appears in bootstrap."""
        root = Path(self._tmpdir)
        (root / "SOUL.md").write_text("# My Custom Persona\nI am a pirate.")
        ws = WorkspaceManager(workspace_dir=self._tmpdir)
        content = ws.get_bootstrap_content()
        assert "pirate" in content

    def test_default_soul_scaffold_exists(self):
        """Default SOUL.md scaffold is created on workspace init."""
        root = Path(self._tmpdir)
        assert (root / "SOUL.md").exists()
        content = (root / "SOUL.md").read_text()
        assert "# Identity" in content

    def test_all_identity_files_scaffolded(self):
        """All 5 identity files are created with default content."""
        root = Path(self._tmpdir)
        for filename in ("SOUL.md", "INSTRUCTIONS.md", "USER.md", "MEMORY.md", "HEARTBEAT.md"):
            assert (root / filename).exists(), f"{filename} not scaffolded"
            content = (root / filename).read_text()
            assert content.strip(), f"{filename} has empty scaffold"


class TestHeartbeatFile:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.ws = WorkspaceManager(workspace_dir=self._tmpdir)

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_heartbeat_scaffold(self):
        root = Path(self._tmpdir)
        assert (root / "HEARTBEAT.md").exists()
        content = (root / "HEARTBEAT.md").read_text()
        assert "Heartbeat Rules" in content

    def test_custom_heartbeat_rules(self):
        root = Path(self._tmpdir)
        (root / "HEARTBEAT.md").write_text("# Custom\n- Check email\n- Monitor API")
        ws = WorkspaceManager(workspace_dir=self._tmpdir)
        rules = ws.load_heartbeat_rules()
        assert "Check email" in rules


class TestUpdateFile:
    """Tests for agent-writable workspace file updates."""

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.ws = WorkspaceManager(workspace_dir=self._tmpdir)

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_update_heartbeat_md(self):
        result = self.ws.update_file("HEARTBEAT.md", "# Custom Rules\n- Check inbox")
        assert result["updated"] is True
        assert result["filename"] == "HEARTBEAT.md"
        content = (Path(self._tmpdir) / "HEARTBEAT.md").read_text()
        assert "Check inbox" in content

    def test_update_user_md(self):
        result = self.ws.update_file("USER.md", "# User\nPrefers short answers")
        assert result["updated"] is True
        content = (Path(self._tmpdir) / "USER.md").read_text()
        assert "Prefers short answers" in content

    def test_update_soul_md(self):
        result = self.ws.update_file("SOUL.md", "# Identity\nI am a pirate.")
        assert result["updated"] is True
        content = (Path(self._tmpdir) / "SOUL.md").read_text()
        assert "pirate" in content

    def test_update_instructions_md(self):
        result = self.ws.update_file("INSTRUCTIONS.md", "# Instructions\nAlways use JSON.")
        assert result["updated"] is True
        content = (Path(self._tmpdir) / "INSTRUCTIONS.md").read_text()
        assert "Always use JSON" in content

    def test_update_memory_md_blocked(self):
        result = self.ws.update_file("MEMORY.md", "overwrite memory")
        assert "error" in result

    def test_backup_created(self):
        # Write initial content
        (Path(self._tmpdir) / "HEARTBEAT.md").write_text("original content")
        # Update
        self.ws.update_file("HEARTBEAT.md", "new content")
        # Check backup exists
        backup_dir = Path(self._tmpdir) / "backups"
        assert backup_dir.exists()
        backups = list(backup_dir.glob("HEARTBEAT.md.*.bak"))
        assert len(backups) == 1
        assert backups[0].read_text() == "original content"

    def test_multiple_backups(self):
        """Each update creates a separate backup."""
        (Path(self._tmpdir) / "USER.md").write_text("v1")
        self.ws.update_file("USER.md", "v2")
        self.ws.update_file("USER.md", "v3")
        backups = list((Path(self._tmpdir) / "backups").glob("USER.md.*.bak"))
        assert len(backups) == 2

    def test_daily_log_entry(self):
        self.ws.update_file("HEARTBEAT.md", "new rules")
        today_log = list((Path(self._tmpdir) / "memory").glob("*.md"))
        assert len(today_log) >= 1
        content = today_log[0].read_text()
        assert "Updated workspace file: HEARTBEAT.md" in content

    def test_size_limit_enforced(self):
        """Content exceeding _MAX_WRITABLE_SIZE is rejected."""
        huge = "x" * (WorkspaceManager._MAX_WRITABLE_SIZE + 1)
        result = self.ws.update_file("HEARTBEAT.md", huge)
        assert "error" in result
        assert "too large" in result["error"].lower()

    def test_size_at_limit_accepted(self):
        """Content exactly at _MAX_WRITABLE_SIZE is accepted."""
        exact = "x" * WorkspaceManager._MAX_WRITABLE_SIZE
        result = self.ws.update_file("HEARTBEAT.md", exact)
        assert result["updated"] is True
        assert result["size"] == WorkspaceManager._MAX_WRITABLE_SIZE

    def test_backup_rotation(self):
        """Old backups are pruned when exceeding _MAX_BACKUPS_PER_FILE."""
        path = Path(self._tmpdir) / "HEARTBEAT.md"
        backup_dir = Path(self._tmpdir) / "backups"
        backup_dir.mkdir(exist_ok=True)
        # Create more backups than the limit
        for i in range(WorkspaceManager._MAX_BACKUPS_PER_FILE + 5):
            path.write_text(f"v{i}")
            bak = backup_dir / f"HEARTBEAT.md.fake_{i:04d}.bak"
            bak.write_text(f"v{i}")
        # Trigger rotation by doing a real update
        path.write_text("current")
        self.ws.update_file("HEARTBEAT.md", "new")
        backups = list(backup_dir.glob("HEARTBEAT.md.*.bak"))
        assert len(backups) <= WorkspaceManager._MAX_BACKUPS_PER_FILE


# ── generate_system_md ──────────────────────────────────────────


class TestGenerateSystemMd:
    def test_includes_preamble(self):
        """Generated SYSTEM.md contains the static architecture overview."""
        result = generate_system_md({}, "alice")
        assert "# System Architecture" in result
        assert "Credential vault" in result

    def test_ignores_introspect_data(self):
        """Permissions and fleet are no longer baked into SYSTEM.md (provided live by runtime context)."""
        data = {
            "permissions": {
                "blackboard_read": ["context/*", "tasks/*"],
            },
            "fleet": [
                {"id": "alice", "role": "researcher"},
                {"id": "bob", "role": "engineer"},
            ],
        }
        result = generate_system_md(data, "alice")
        assert "# System Architecture" in result
        assert "## Your Permissions" not in result
        assert "## Fleet" not in result

    def test_output_capped_at_max_system(self):
        """Output is capped at _MAX_SYSTEM chars."""
        result = generate_system_md({}, "agent_0")
        assert len(result) <= _MAX_SYSTEM + 20

    def test_standalone_omits_blackboard(self):
        """Standalone SYSTEM.md has no blackboard, pub/sub, or coordination sections."""
        result = generate_system_md({}, "alice", is_standalone=True)
        assert "# System Architecture" in result
        assert "Credential vault" in result
        assert "blackboard" not in result.lower()
        assert "pub/sub" not in result.lower()
        assert "Coordination Philosophy" not in result
        assert "403 Forbidden" not in result

    def test_project_includes_blackboard(self):
        """Project SYSTEM.md includes blackboard, pub/sub, and coordination."""
        result = generate_system_md({}, "alice", is_standalone=False)
        assert "Blackboard" in result
        assert "Pub/Sub" in result
        assert "Coordination Philosophy" in result


class TestBootstrapIncludesSystemMd:
    def test_system_md_loaded_in_bootstrap(self):
        """SYSTEM.md content appears in get_bootstrap_content()."""
        tmpdir = tempfile.mkdtemp()
        try:
            ws = WorkspaceManager(workspace_dir=tmpdir)
            system_content = "# System Architecture\n\nTest content for SYSTEM.md"
            (Path(tmpdir) / "SYSTEM.md").write_text(system_content)
            bootstrap = ws.get_bootstrap_content()
            assert "Test content for SYSTEM.md" in bootstrap
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_missing_system_md_is_fine(self):
        """Bootstrap works without SYSTEM.md (graceful degradation)."""
        tmpdir = tempfile.mkdtemp()
        try:
            ws = WorkspaceManager(workspace_dir=tmpdir)
            # Don't create SYSTEM.md
            bootstrap = ws.get_bootstrap_content()
            assert "System Architecture" not in bootstrap
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
