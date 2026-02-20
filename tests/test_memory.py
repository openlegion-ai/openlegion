"""Unit tests for agent memory store."""

import math

import pytest

from src.agent.memory import MemoryStore


@pytest.fixture
def memory(tmp_path):
    store = MemoryStore(db_path=str(tmp_path / "test.db"))
    yield store
    store.close()


def _make_embedding(seed: float = 0.1) -> list[float]:
    """Create a deterministic 1536-dim embedding for testing."""
    return [math.sin(seed * (i + 1)) for i in range(1536)]


async def _mock_embed(text: str) -> list[float]:
    """Mock embed function that returns deterministic embeddings based on text hash."""
    seed = sum(ord(c) for c in text) / 100.0
    return _make_embedding(seed)


async def _mock_categorize(key: str, value: str) -> str:
    """Mock categorize function that returns a category based on key prefix."""
    if "user" in key.lower():
        return "preferences"
    if "project" in key.lower():
        return "project_info"
    return "general"


@pytest.fixture
def memory_with_embeddings(tmp_path):
    store = MemoryStore(db_path=str(tmp_path / "test_embed.db"), embed_fn=_mock_embed)
    yield store
    store.close()


@pytest.fixture
def memory_with_categories(tmp_path):
    store = MemoryStore(
        db_path=str(tmp_path / "test_cat.db"),
        embed_fn=_mock_embed,
        categorize_fn=_mock_categorize,
    )
    yield store
    store.close()


@pytest.mark.asyncio
async def test_store_and_retrieve_fact(memory):
    fact_id = await memory.store_fact("test_key", "test_value")
    assert fact_id.startswith("fact_")
    facts = await memory.search("test")
    assert len(facts) > 0
    assert facts[0].key == "test_key"


@pytest.mark.asyncio
async def test_update_existing_fact(memory):
    await memory.store_fact("key1", "old_value")
    await memory.store_fact("key1", "new_value")
    fact = memory._get_fact_by_key("key1")
    assert fact is not None
    assert fact.value == "new_value"
    assert fact.access_count >= 1


@pytest.mark.asyncio
async def test_keyword_search_returns_results(memory):
    await memory.store_fact("company_name", "Acme Corporation")
    await memory.store_fact("company_size", "500 employees")
    results = await memory.search("Acme")
    assert len(results) >= 1
    assert any(f.key == "company_name" for f in results)


@pytest.mark.asyncio
async def test_salience_boost_on_access(memory):
    await memory.store_fact("key1", "value1")
    initial = memory._get_fact_by_key("key1")
    assert initial is not None
    initial_score = initial.decay_score

    for _ in range(3):
        await memory.search("key1")

    boosted = memory._get_fact_by_key("key1")
    assert boosted is not None
    assert boosted.decay_score > initial_score


@pytest.mark.asyncio
async def test_decay_reduces_scores(memory):
    await memory.store_fact("key1", "value1")
    initial = memory._get_fact_by_key("key1")
    assert initial is not None
    initial_score = initial.decay_score

    memory.decay_all()

    decayed = memory._get_fact_by_key("key1")
    assert decayed is not None
    assert decayed.decay_score < initial_score


@pytest.mark.asyncio
async def test_high_salience_facts(memory):
    for i in range(5):
        await memory.store_fact(f"key_{i}", f"value_{i}")

    high = memory.get_high_salience_facts(top_k=3)
    assert len(high) == 3


@pytest.mark.asyncio
async def test_log_action(memory):
    await memory.log_action(action="test_action", input_summary="input", output_summary="output")
    logs = memory.get_recent_logs(limit=10)
    assert len(logs) == 1
    assert logs[0].action == "test_action"


@pytest.mark.asyncio
async def test_empty_search_returns_empty(memory):
    results = await memory.search("nonexistent")
    assert results == []


@pytest.mark.asyncio
async def test_store_facts_batch(memory):
    facts = [
        {"key": "user_lang", "value": "Python", "category": "preference"},
        {"key": "user_editor", "value": "Vim", "category": "preference"},
        {"key": "project_name", "value": "OpenLegion", "category": "fact"},
    ]
    stored = await memory.store_facts_batch(facts)
    assert stored == 3
    assert memory._get_fact_by_key("user_lang").value == "Python"
    assert memory._get_fact_by_key("user_editor").value == "Vim"
    assert memory._get_fact_by_key("project_name").value == "OpenLegion"


@pytest.mark.asyncio
async def test_store_facts_batch_skips_invalid(memory):
    facts = [
        {"key": "valid_key", "value": "valid_value"},
        {"key": "", "value": "no key"},  # empty key
        {"value": "missing key"},  # no key at all
        {"key": "no_value"},  # no value
    ]
    stored = await memory.store_facts_batch(facts)
    assert stored == 1
    assert memory._get_fact_by_key("valid_key") is not None


@pytest.mark.asyncio
async def test_store_facts_batch_empty(memory):
    stored = await memory.store_facts_batch([])
    assert stored == 0


# ── Categories table tests ────────────────────────────────────


def test_categories_table_created(tmp_path):
    """Schema includes categories and categories_vec tables."""
    store = MemoryStore(db_path=str(tmp_path / "schema.db"))
    tables = [
        r[0] for r in store.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    ]
    assert "categories" in tables
    store.close()


def test_category_id_column_exists(tmp_path):
    """facts table has category_id column."""
    store = MemoryStore(db_path=str(tmp_path / "schema.db"))
    cols = [r[1] for r in store.db.execute("PRAGMA table_info(facts)").fetchall()]
    assert "category_id" in cols
    store.close()


@pytest.mark.asyncio
async def test_auto_categorize_creates_category(memory_with_categories):
    """New category is created when categorize_fn returns a name that doesn't exist."""
    mem = memory_with_categories
    await mem.store_fact("user_language", "Python")

    cats = mem.get_categories()
    names = [c["name"] for c in cats]
    assert "preferences" in names


@pytest.mark.asyncio
async def test_auto_categorize_reuses_existing(memory_with_categories):
    """Multiple facts with same category reuse existing category."""
    mem = memory_with_categories
    await mem.store_fact("user_language", "Python")
    await mem.store_fact("user_editor", "Vim")

    cats = mem.get_categories()
    pref_cats = [c for c in cats if c["name"] == "preferences"]
    assert len(pref_cats) == 1
    assert pref_cats[0]["item_count"] >= 2


@pytest.mark.asyncio
async def test_category_id_assigned_on_store(memory_with_categories):
    """store_fact sets category_id on the fact."""
    mem = memory_with_categories
    fact_id = await mem.store_fact("user_theme", "dark mode")

    row = mem.db.execute("SELECT category_id FROM facts WHERE id = ?", (fact_id,)).fetchone()
    assert row is not None
    assert row[0] is not None  # category_id was set


@pytest.mark.asyncio
async def test_no_categorize_fn_graceful(memory_with_embeddings):
    """Without categorize_fn, facts are stored normally with 'general' category."""
    mem = memory_with_embeddings
    fact_id = await mem.store_fact("some_key", "some_value")
    assert fact_id.startswith("fact_")

    # Should have a general category created
    row = mem.db.execute("SELECT category_id FROM facts WHERE id = ?", (fact_id,)).fetchone()
    # category_id may or may not be set (general), but no crash
    assert row is not None


@pytest.mark.asyncio
async def test_no_embed_fn_no_categorization(memory):
    """Without embed_fn, no categorization happens (graceful degradation)."""
    fact_id = await memory.store_fact("key1", "value1")
    row = memory.db.execute("SELECT category_id FROM facts WHERE id = ?", (fact_id,)).fetchone()
    assert row is not None
    assert row[0] is None  # No embedding → no categorization


# ── Hierarchical search tests ────────────────────────────────


@pytest.mark.asyncio
async def test_search_hierarchical_finds_by_category(memory_with_categories):
    """Facts found via category-tier search."""
    mem = memory_with_categories
    await mem.store_fact("user_language", "Python is my preferred language")
    await mem.store_fact("user_editor", "Vim is my preferred editor")
    await mem.store_fact("project_name", "OpenLegion is the project name")

    results = await mem.search_hierarchical("user preferences", top_k=5)
    assert len(results) > 0


@pytest.mark.asyncio
async def test_search_hierarchical_fallback_to_flat(memory):
    """If no categories exist, falls back to flat keyword search."""
    await memory.store_fact("company", "Acme Corp")
    results = await memory.search_hierarchical("Acme", top_k=5)
    assert len(results) >= 1
    assert any(f.key == "company" for f in results)


@pytest.mark.asyncio
async def test_search_hierarchical_deduplicates(memory_with_categories):
    """No duplicate facts in merged results."""
    mem = memory_with_categories
    await mem.store_fact("user_name", "Alice")
    await mem.store_fact("user_email", "alice@example.com")

    results = await mem.search_hierarchical("user Alice", top_k=10)
    ids = [f.id for f in results]
    assert len(ids) == len(set(ids))  # No duplicates


@pytest.mark.asyncio
async def test_search_within_categories_scoped(memory_with_categories):
    """_search_within_categories only returns facts from matched categories."""
    mem = memory_with_categories
    await mem.store_fact("user_lang", "Python")
    await mem.store_fact("project_status", "Active development")

    # Get the preferences category ID
    cats = mem.get_categories()
    pref_cat = next((c for c in cats if c["name"] == "preferences"), None)
    assert pref_cat is not None, "preferences category should have been created"
    embedding = _make_embedding(0.5)
    scoped = mem._search_within_categories("language", embedding, [pref_cat["id"]], top_k=5)
    for fact in scoped:
        row = mem.db.execute(
            "SELECT category_id FROM facts WHERE id = ?", (fact.id,),
        ).fetchone()
        assert row[0] == pref_cat["id"]


@pytest.mark.asyncio
async def test_existing_facts_without_category_id(memory_with_embeddings):
    """Facts with category_id=NULL are still searchable via flat fallback."""
    mem = memory_with_embeddings
    # Store facts (they'll get category_id but let's force NULL)
    fact_id = await mem.store_fact("uncategorized_fact", "this has no category")
    mem.db.execute("UPDATE facts SET category_id = NULL WHERE id = ?", (fact_id,))
    mem.db.commit()

    # Hierarchical search should still find it via flat fallback
    results = await mem.search_hierarchical("uncategorized", top_k=5)
    assert any(f.id == fact_id for f in results)


# ── Reinforcement scoring tests ────────────────────────────


def test_continuous_boost_increases_with_access():
    """More accesses → higher boost."""
    boost_1 = MemoryStore._compute_boost(1)
    boost_5 = MemoryStore._compute_boost(5)
    boost_20 = MemoryStore._compute_boost(20)

    assert boost_1 < boost_5 < boost_20
    # All should be >= 1.0
    assert boost_1 >= 1.0


def test_recency_factor_decays():
    """Old facts get lower boost than recent ones."""
    boost_fresh = MemoryStore._compute_boost(5, days_since_last_access=0.0)
    boost_week = MemoryStore._compute_boost(5, days_since_last_access=7.0)
    boost_month = MemoryStore._compute_boost(5, days_since_last_access=20.0)

    assert boost_fresh > boost_week > boost_month
    # At 20 days, recency_factor = max(0.1, 1.0 - 20*0.05) = 0.1
    assert boost_month >= 1.0  # Still positive


def test_boost_formula_correctness():
    """Verify the boost formula: 1 + log(1 + access_count) * recency_factor."""
    access = 10
    days = 5.0
    recency = max(0.1, 1.0 - days * 0.05)
    expected = 1.0 + math.log(1 + access) * recency
    actual = MemoryStore._compute_boost(access, days)
    assert abs(actual - expected) < 1e-10


# ── Tool outcome tests ────────────────────────────────────


class TestToolOutcomes:
    def test_store_and_retrieve(self, memory):
        memory.store_tool_outcome("exec", {"command": "ls"}, "file.txt", success=True)
        history = memory.get_tool_history("exec")
        assert len(history) == 1
        assert history[0]["tool_name"] == "exec"
        assert history[0]["success"] is True

    def test_failure_recorded(self, memory):
        memory.store_tool_outcome("exec", {"command": "bad"}, "error: not found", success=False)
        history = memory.get_tool_history("exec")
        assert len(history) == 1
        assert history[0]["success"] is False

    def test_params_hash_deterministic(self, memory):
        args = {"command": "ls", "cwd": "/tmp"}
        h1 = MemoryStore._compute_params_hash(args)
        h2 = MemoryStore._compute_params_hash({"cwd": "/tmp", "command": "ls"})
        assert h1 == h2  # order-independent

    def test_params_hash_none(self, memory):
        h = MemoryStore._compute_params_hash(None)
        assert isinstance(h, str)
        assert len(h) == 16

    def test_prune_keeps_50(self, memory):
        for i in range(60):
            memory.store_tool_outcome("exec", {"i": i}, f"output_{i}")
        history = memory.get_tool_history("exec", limit=100)
        assert len(history) == 50

    def test_filter_by_params_hash(self, memory):
        memory.store_tool_outcome("exec", {"command": "ls"}, "files", success=True)
        memory.store_tool_outcome("exec", {"command": "pwd"}, "/home", success=True)
        h = MemoryStore._compute_params_hash({"command": "ls"})
        history = memory.get_tool_history("exec", params_hash=h)
        assert len(history) == 1
        assert "files" in history[0]["outcome"]

    def test_get_all_tools(self, memory):
        memory.store_tool_outcome("exec", {}, "ok")
        memory.store_tool_outcome("web_search", {}, "results")
        history = memory.get_tool_history(limit=10)
        assert len(history) == 2

    def test_tool_outcomes_table_created(self, tmp_path):
        store = MemoryStore(db_path=str(tmp_path / "schema.db"))
        tables = [r[0] for r in store.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        assert "tool_outcomes" in tables
        store.close()
