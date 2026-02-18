"""Integration tests for memory system: vector search, cross-task recall, salience tracking.

Uses a fake embedding function to test the full vector + keyword retrieval pipeline
without requiring a real LLM.
"""

import asyncio
import random

import pytest

from src.agent.memory import MemoryStore


def _fake_embed(text: str) -> list[float]:
    """Deterministic fake embedding: hash-based 1536-dim vector."""
    random.seed(hash(text) % (2**31))
    return [random.gauss(0, 1) for _ in range(1536)]


async def _async_fake_embed(text: str) -> list[float]:
    """Async wrapper for fake embedding."""
    return _fake_embed(text)


@pytest.fixture
def memory_with_embeddings(tmp_path):
    store = MemoryStore(db_path=str(tmp_path / "test_vec.db"), embed_fn=_async_fake_embed)
    yield store
    store.close()


def test_vector_search_finds_similar_facts(memory_with_embeddings):
    """Store facts and verify vector search ranks relevant ones higher."""
    mem = memory_with_embeddings
    loop = asyncio.get_event_loop()

    # Store diverse facts
    loop.run_until_complete(mem.store_fact("company_name", "Acme Corporation", category="company"))
    loop.run_until_complete(mem.store_fact("company_size", "500 employees", category="company"))
    loop.run_until_complete(mem.store_fact("company_revenue", "50M annual revenue", category="company"))
    loop.run_until_complete(mem.store_fact("favorite_color", "blue", category="personal"))
    loop.run_until_complete(mem.store_fact("lunch_order", "sandwich", category="personal"))

    results = loop.run_until_complete(mem.search("Acme Corporation company info"))
    assert len(results) >= 1
    # Company-related facts should appear in results
    keys = [f.key for f in results]
    assert "company_name" in keys


def test_cross_task_memory_recall(memory_with_embeddings):
    """Facts stored during task 1 should be retrievable during task 2."""
    mem = memory_with_embeddings
    loop = asyncio.get_event_loop()

    # Task 1: store research results
    loop.run_until_complete(mem.store_fact("prospect_domain", "acme.com", category="research"))
    loop.run_until_complete(mem.store_fact("prospect_tech_stack", "Python, React, AWS", category="research"))
    loop.run_until_complete(mem.log_action("tool:web_search", "searched acme", "found info"))

    # Task 2: search for what we know
    results = loop.run_until_complete(mem.search("acme technology stack"))
    assert len(results) >= 1
    keys = [f.key for f in results]
    assert "prospect_tech_stack" in keys or "prospect_domain" in keys


def test_salience_increases_with_repeated_access(memory_with_embeddings):
    """Decay + selective boosting should differentiate fact salience."""
    mem = memory_with_embeddings
    loop = asyncio.get_event_loop()

    loop.run_until_complete(mem.store_fact("hot_fact", "important info", category="key"))
    loop.run_until_complete(mem.store_fact("cold_fact", "unimportant info", category="misc"))

    # Decay all facts first to establish a baseline
    mem.decay_all()
    mem.decay_all()
    mem.decay_all()

    # Now boost hot_fact by re-storing (update path increases salience)
    for _ in range(3):
        loop.run_until_complete(mem.store_fact("hot_fact", "important info updated"))

    hot = mem._get_fact_by_key("hot_fact")
    cold = mem._get_fact_by_key("cold_fact")
    assert hot is not None and cold is not None
    # hot_fact was re-stored multiple times (boosted), cold was only decayed
    assert hot.decay_score > cold.decay_score


def test_high_salience_auto_surface(memory_with_embeddings):
    """High-salience facts should appear in get_high_salience_facts."""
    mem = memory_with_embeddings
    loop = asyncio.get_event_loop()

    # Store many facts
    for i in range(20):
        loop.run_until_complete(mem.store_fact(f"fact_{i}", f"value_{i}"))

    # Boost a few
    for _ in range(5):
        loop.run_until_complete(mem.store_fact("fact_5", "boosted value"))
        loop.run_until_complete(mem.store_fact("fact_10", "also boosted"))

    high = mem.get_high_salience_facts(top_k=5)
    assert len(high) == 5
    keys = [f.key for f in high]
    assert "fact_5" in keys
    assert "fact_10" in keys


def test_decay_across_tasks(memory_with_embeddings):
    """Decay should reduce salience of unused facts over time."""
    mem = memory_with_embeddings
    loop = asyncio.get_event_loop()

    loop.run_until_complete(mem.store_fact("old_fact", "old value"))
    initial = mem._get_fact_by_key("old_fact")
    assert initial is not None

    # Simulate multiple decay cycles (between tasks)
    for _ in range(10):
        mem.decay_all()

    decayed = mem._get_fact_by_key("old_fact")
    assert decayed is not None
    assert decayed.decay_score < initial.decay_score * 0.7


def test_many_facts_vector_search(memory_with_embeddings):
    """Vector search should work with a larger number of stored facts."""
    mem = memory_with_embeddings
    loop = asyncio.get_event_loop()

    # Store 50 facts
    for i in range(50):
        loop.run_until_complete(
            mem.store_fact(f"topic_{i}", f"information about topic number {i}", category=f"cat_{i % 5}")
        )

    results = loop.run_until_complete(mem.search("topic number 25", top_k=5))
    assert len(results) <= 5
    assert len(results) >= 1
