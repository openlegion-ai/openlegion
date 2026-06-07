"""Tests for graceful embedding failure degradation in MemoryStore.

When the embedding function fails 3+ consecutive times, MemoryStore should
disable vector search entirely (set embed_fn = None) and fall back to
keyword-only search, rather than spamming warnings on every call.
"""

import os
import tempfile

import pytest

from src.agent.memory import MemoryStore


@pytest.fixture
def tmp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    for ext in ["-wal", "-shm"]:
        try:
            os.unlink(path + ext)
        except FileNotFoundError:
            pass


class _CallTracker:
    """Tracks how many times the embed function was called."""

    def __init__(self, fail_until=999):
        self.call_count = 0
        self.fail_until = fail_until

    async def __call__(self, text: str):
        self.call_count += 1
        if self.call_count <= self.fail_until:
            raise RuntimeError(f"Embedding service down (call {self.call_count})")
        return [0.1] * 1536


class TestEmbeddingGracefulFallback:
    """Test that embed_fn is disabled after 3 consecutive failures."""

    def test_init_has_embed_failures_counter(self, tmp_db):
        store = MemoryStore(tmp_db)
        assert store._embed_failures == 0

    @pytest.mark.asyncio
    async def test_single_failure_keeps_embed_fn(self, tmp_db):
        tracker = _CallTracker(fail_until=999)
        store = MemoryStore(tmp_db, embed_fn=tracker)

        await store.store_fact("test_key", "test_value")
        assert store.embed_fn is not None
        assert store._embed_failures == 1

    @pytest.mark.asyncio
    async def test_two_failures_keeps_embed_fn(self, tmp_db):
        tracker = _CallTracker(fail_until=999)
        store = MemoryStore(tmp_db, embed_fn=tracker)

        await store.store_fact("key1", "val1")
        await store.store_fact("key2", "val2")
        assert store.embed_fn is not None
        assert store._embed_failures == 2

    @pytest.mark.asyncio
    async def test_three_failures_disables_embed_fn(self, tmp_db):
        tracker = _CallTracker(fail_until=999)
        store = MemoryStore(tmp_db, embed_fn=tracker)

        await store.store_fact("key1", "val1")
        await store.store_fact("key2", "val2")
        await store.store_fact("key3", "val3")
        assert store.embed_fn is None
        assert store._embed_failures >= 3

    @pytest.mark.asyncio
    async def test_success_resets_failure_counter(self, tmp_db):
        # Fail twice, then succeed, then fail twice more — should NOT disable
        tracker = _CallTracker(fail_until=2)  # fails calls 1-2, succeeds from 3+
        store = MemoryStore(tmp_db, embed_fn=tracker)

        await store.store_fact("key1", "val1")  # fail 1
        assert store._embed_failures == 1
        await store.store_fact("key2", "val2")  # fail 2
        assert store._embed_failures == 2
        await store.store_fact("key3", "val3")  # success -> reset
        assert store._embed_failures == 0
        assert store.embed_fn is not None

    @pytest.mark.asyncio
    async def test_after_disable_saves_still_work(self, tmp_db):
        """After embed_fn is disabled, store_fact should still work (keyword only)."""
        tracker = _CallTracker(fail_until=999)
        store = MemoryStore(tmp_db, embed_fn=tracker)

        # Trigger 3 failures to disable
        for i in range(3):
            await store.store_fact(f"key{i}", f"val{i}")
        assert store.embed_fn is None

        # This should work fine without embedding
        await store.store_fact("key_after", "value_after_disable")
        results = await store.search("value_after_disable")
        assert any("value_after_disable" in r.value for r in results)

    @pytest.mark.asyncio
    async def test_after_disable_search_still_works(self, tmp_db):
        """After embed_fn is disabled, search falls back to keyword-only."""
        tracker = _CallTracker(fail_until=999)
        store = MemoryStore(tmp_db, embed_fn=tracker)

        # Save a fact first (will fail embedding but fact is still saved)
        await store.store_fact("greeting", "hello world")

        # Trigger remaining failures
        await store.store_fact("k2", "v2")
        await store.store_fact("k3", "v3")
        assert store.embed_fn is None

        # Search should still work via keyword/FTS
        results = await store.search("hello world")
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_search_failures_count_toward_disable(self, tmp_db):
        """Embedding failures during search also count toward the 3-failure threshold."""
        # Save without embedding (no embed_fn for save)
        store_no_embed = MemoryStore(tmp_db)
        await store_no_embed.store_fact("data", "important information")
        store_no_embed.close()

        # Now search with failing embed_fn — each search attempt fails
        store2 = MemoryStore(tmp_db, embed_fn=_CallTracker(fail_until=999))
        await store2.search("test1")  # fail 1
        await store2.search("test2")  # fail 2
        await store2.search("test3")  # fail 3
        assert store2.embed_fn is None

    @pytest.mark.asyncio
    async def test_no_embed_fn_no_tracking(self, tmp_db):
        """When there's no embed_fn at all, counter stays at 0."""
        store = MemoryStore(tmp_db, embed_fn=None)
        await store.store_fact("key", "value")
        assert store._embed_failures == 0

    @pytest.mark.asyncio
    async def test_fourth_save_after_disable_no_embed_call(self, tmp_db):
        """After disable, embed_fn is not called on subsequent saves."""
        tracker = _CallTracker(fail_until=999)
        store = MemoryStore(tmp_db, embed_fn=tracker)

        for i in range(3):
            await store.store_fact(f"key{i}", f"val{i}")
        assert tracker.call_count == 3
        assert store.embed_fn is None

        # 4th save should NOT call the embed function
        await store.store_fact("key4", "val4")
        assert tracker.call_count == 3  # still 3, not 4


def _fixed_embed(dim: int):
    """Return an async embed_fn that always yields a ``dim``-length vector."""
    async def _embed(text: str):
        return [0.1] * dim
    return _embed


class TestConfigurableEmbeddingDim:
    """Configurable embedding dimension + safe provider-switch handling.

    Backs the product guarantee that 'no embedding model' and 'switched
    embedding provider' are safe, non-breaking states for every agent.
    """

    @pytest.mark.asyncio
    async def test_keyword_only_store_and_recall(self, tmp_db):
        # The 'no embedding model' (EMBEDDING_MODEL=none → embed_fn=None)
        # contract: an agent must still store and recall facts via keyword.
        store = MemoryStore(tmp_db, embed_fn=None)
        await store.store_fact("deploy_target", "production cluster eu-west")
        results = await store.search("production cluster")
        assert any("production cluster" in r.value for r in results)
        store.close()

    def test_embedding_dim_defaults_to_1536(self, tmp_db):
        store = MemoryStore(tmp_db)
        assert store.embedding_dim == 1536
        store.close()

    def test_non_positive_dim_falls_back_to_default(self, tmp_db):
        store = MemoryStore(tmp_db, embedding_dim=0)
        assert store.embedding_dim == 1536
        store.close()

    @pytest.mark.asyncio
    async def test_custom_dim_stores_and_searches(self, tmp_db):
        # A non-1536 provider (e.g. Voyage 1024) works end to end.
        store = MemoryStore(tmp_db, embed_fn=_fixed_embed(1024), embedding_dim=1024)
        await store.store_fact("k", "voyage embedded value")
        results = await store.search("voyage embedded value")
        assert any("voyage embedded value" in r.value for r in results)
        store.close()

    @pytest.mark.asyncio
    async def test_dim_change_rebuilds_and_preserves_rows(self, tmp_db):
        # Create at 1536 and store a fact (with a 1536 vector).
        store = MemoryStore(tmp_db, embed_fn=_fixed_embed(1536), embedding_dim=1536)
        await store.store_fact("switch_key", "value before provider switch")
        assert store._facts_vec_dim() == 1536
        store.close()

        # Reopen at 1024 (operator switched embedding provider). Must NOT
        # crash; the fact ROW survives and stays keyword-searchable, and the
        # recorded dim is updated.
        store2 = MemoryStore(tmp_db, embed_fn=_fixed_embed(1024), embedding_dim=1024)
        assert store2._facts_vec_dim() == 1024   # vec tables rebuilt to the new dim
        results = await store2.search("value before provider switch")
        assert any("value before provider switch" in r.value for r in results)
        # New facts embed cleanly at the new dimension (no dim error).
        await store2.store_fact("new_key", "value after switch")
        store2.close()

    @pytest.mark.asyncio
    async def test_wrong_length_embedding_skips_vector_no_crash(self, tmp_db):
        # embed_fn SUCCEEDS but returns a vector whose length disagrees with
        # the configured dim (misconfiguration). store_fact + search must
        # NOT raise, the embed call is not counted as a failure, and the
        # fact stays keyword-searchable.
        store = MemoryStore(tmp_db, embed_fn=_fixed_embed(999), embedding_dim=1536)
        await store.store_fact("mismatch", "kept despite bad embedding")
        assert store._embed_failures == 0  # the embed CALL itself succeeded
        results = await store.search("kept despite bad embedding")
        assert any("kept despite bad embedding" in r.value for r in results)
        store.close()
