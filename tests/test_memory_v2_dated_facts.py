"""Tests for memory v2: dated + sourced facts and inject-head-only option.

Covers:
  - new (source_type, date) columns present + populated on store_fact
  - idempotent migration adds the columns to a pre-existing DB created
    WITHOUT them (an old on-disk schema) — reopen must not crash
  - prefer-recent: date breaks decay_score ties in high-salience selection
  - inject-head-only flag drops the ## Recent slice but keeps the head, and
    the log stays searchable; flag-off → injection unchanged
"""

from __future__ import annotations

import shutil
import sqlite3
import tempfile

import pytest

from src.agent.memory import MemoryStore
from src.agent.workspace import WorkspaceManager


@pytest.fixture
def memory(tmp_path):
    store = MemoryStore(db_path=str(tmp_path / "test.db"))
    yield store
    store.close()


# --- Schema + populate -------------------------------------------------------

@pytest.mark.asyncio
async def test_new_columns_present_and_populated_on_store(memory):
    fact_id = await memory.store_fact("user_lang", "english", source_type="conversation")

    cols = {r[1] for r in memory.db.execute("PRAGMA table_info(facts)").fetchall()}
    assert "source_type" in cols
    assert "date" in cols

    row = memory.db.execute(
        "SELECT source_type, date FROM facts WHERE id = ?", (fact_id,)
    ).fetchone()
    assert row[0] == "conversation"
    assert row[1]  # date stamped, non-empty

    fact = memory._get_fact(fact_id)
    assert fact.source_type == "conversation"
    assert fact.date is not None


@pytest.mark.asyncio
async def test_source_type_default_and_override(memory):
    default_id = await memory.store_fact("k1", "v1")
    custom_id = await memory.store_fact("k2", "v2", source_type="consolidation")

    assert memory._get_fact(default_id).source_type == "conversation"
    assert memory._get_fact(custom_id).source_type == "consolidation"


@pytest.mark.asyncio
async def test_batch_store_tags_context_flush_source_type(memory):
    await memory.store_facts_batch([{"key": "bk", "value": "bv"}])
    fact = memory._get_fact_by_key("bk")
    assert fact is not None
    assert fact.source_type == "context_flush"


# --- Migration on a pre-existing OLD-schema DB -------------------------------

def _create_legacy_facts_db(path: str) -> None:
    """Create a facts table WITHOUT the v2 columns, mimicking an old on-disk DB
    that predates the source_type/date migration."""
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE facts (
            id TEXT PRIMARY KEY,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            category TEXT DEFAULT 'general',
            source TEXT DEFAULT 'agent',
            confidence REAL DEFAULT 1.0,
            access_count INTEGER DEFAULT 0,
            last_accessed TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            decay_score REAL DEFAULT 1.0
        );
        INSERT INTO facts (id, key, value) VALUES ('fact_old', 'legacy_key', 'legacy_value');
        """
    )
    conn.commit()
    conn.close()


@pytest.mark.asyncio
async def test_migration_adds_columns_to_preexisting_db(tmp_path):
    db_path = str(tmp_path / "legacy.db")
    _create_legacy_facts_db(db_path)

    # Sanity: old schema lacks the v2 columns.
    conn = sqlite3.connect(db_path)
    cols_before = {r[1] for r in conn.execute("PRAGMA table_info(facts)").fetchall()}
    conn.close()
    assert "source_type" not in cols_before
    assert "date" not in cols_before

    # Reopening through MemoryStore must NOT crash and must add the columns.
    store = MemoryStore(db_path=db_path)
    try:
        cols_after = {r[1] for r in store.db.execute("PRAGMA table_info(facts)").fetchall()}
        assert "source_type" in cols_after
        assert "date" in cols_after

        # The pre-existing row survives and is readable through the mapper.
        legacy = store._get_fact_by_key("legacy_key")
        assert legacy is not None
        assert legacy.value == "legacy_value"

        # New writes against the migrated DB populate the v2 columns.
        new_id = await store.store_fact("fresh", "value", source_type="conversation")
        assert store._get_fact(new_id).source_type == "conversation"
    finally:
        store.close()


@pytest.mark.asyncio
async def test_migration_is_idempotent_on_reopen(tmp_path):
    db_path = str(tmp_path / "reopen.db")
    s1 = MemoryStore(db_path=db_path)
    await s1.store_fact("k", "v")
    s1.close()
    # Reopening an already-migrated DB must not raise on the duplicate ALTER.
    s2 = MemoryStore(db_path=db_path)
    try:
        cols = {r[1] for r in s2.db.execute("PRAGMA table_info(facts)").fetchall()}
        assert "source_type" in cols and "date" in cols
    finally:
        s2.close()


# --- Prefer-recent tie-break -------------------------------------------------

@pytest.mark.asyncio
async def test_high_salience_prefers_recent_on_tie(memory):
    # Two facts with identical decay_score; the newer date should rank first.
    old_id = await memory.store_fact("old", "old_val")
    new_id = await memory.store_fact("new", "new_val")
    # Force identical decay scores and a clearly older date on `old`.
    memory.db.execute("UPDATE facts SET decay_score = 1.0")
    memory.db.execute("UPDATE facts SET date = '2000-01-01 00:00:00' WHERE id = ?", (old_id,))
    memory.db.execute("UPDATE facts SET date = '2099-01-01 00:00:00' WHERE id = ?", (new_id,))
    memory.db.commit()

    facts = await memory.get_high_salience_facts(top_k=2)
    assert facts[0].id == new_id


# --- inject-head-only flag ---------------------------------------------------

class TestInjectHeadOnly:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.ws = WorkspaceManager(workspace_dir=self._tmpdir)
        self.ws.write_compiled_memory("COMPILED_HEAD_MARKER")
        self.ws.append_memory("## Recent entry\n\nRECENT_LOG_MARKER")

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_flag_off_injects_head_and_recent(self, monkeypatch):
        monkeypatch.delenv("OPENLEGION_INJECT_HEAD_ONLY", raising=False)
        injected = self.ws.get_memory_injection()
        assert "COMPILED_HEAD_MARKER" in injected
        assert "RECENT_LOG_MARKER" in injected
        assert "## Recent" in injected

    def test_flag_on_drops_recent_keeps_head(self, monkeypatch):
        monkeypatch.setenv("OPENLEGION_INJECT_HEAD_ONLY", "1")
        injected = self.ws.get_memory_injection()
        assert "COMPILED_HEAD_MARKER" in injected
        assert "RECENT_LOG_MARKER" not in injected
        assert "## Recent" not in injected

    def test_flag_on_log_stays_searchable(self, monkeypatch):
        monkeypatch.setenv("OPENLEGION_INJECT_HEAD_ONLY", "1")
        # Dropped from injection, but still on disk + searchable.
        assert "RECENT_LOG_MARKER" not in self.ws.get_memory_injection()
        assert "RECENT_LOG_MARKER" in self.ws.load_memory_log()
        files = {r["file"] for r in self.ws.search("RECENT_LOG_MARKER")}
        assert "MEMORY.md" in files
