"""Unit tests for agent memory store."""

import pytest

from src.agent.memory import MemoryStore


@pytest.fixture
def memory(tmp_path):
    store = MemoryStore(db_path=str(tmp_path / "test.db"))
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
