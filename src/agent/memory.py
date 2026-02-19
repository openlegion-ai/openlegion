"""Per-agent memory with SQLite + sqlite-vec vector search + FTS5 keyword search.

Two-layer hierarchy:
  - facts: structured knowledge with salience tracking
  - logs: action history

Dual retrieval: vector similarity (KNN) + keyword (BM25), merged by weighted score.
Embeddings are generated via mesh proxy (embed_fn).
"""

from __future__ import annotations

import sqlite3
from typing import Any, Callable, Coroutine, Optional

import sqlite_vec

from src.shared.types import MemoryFact, MemoryLog
from src.shared.utils import generate_id, setup_logging

logger = setup_logging("agent.memory")

EMBEDDING_DIM = 1536
SALIENCE_DECAY_RATE = 0.95
SALIENCE_BOOST = 1.5

# Type alias for the async embed function
EmbedFn = Callable[[str], Coroutine[Any, Any, list[float]]]


class MemoryStore:
    """Per-agent memory with vector search and salience tracking."""

    def __init__(self, db_path: str, embed_fn: Optional[EmbedFn] = None):
        self.db = sqlite3.connect(db_path)
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        self.embed_fn = embed_fn
        self._init_schema()

    def _init_schema(self) -> None:
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS facts (
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
            CREATE VIRTUAL TABLE IF NOT EXISTS facts_vec USING vec0(
                id TEXT PRIMARY KEY,
                embedding float[1536]
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
                fact_id,
                key,
                value,
                category
            );
            CREATE TABLE IF NOT EXISTS logs (
                id TEXT PRIMARY KEY,
                action TEXT NOT NULL,
                input_summary TEXT,
                output_summary TEXT,
                task_id TEXT,
                tokens_used INTEGER DEFAULT 0,
                duration_ms INTEGER DEFAULT 0,
                timestamp TEXT DEFAULT (datetime('now'))
            );
        """)
        self.db.commit()

    async def store_fact(
        self,
        key: str,
        value: str,
        category: str = "general",
        source: str = "agent",
        confidence: float = 1.0,
    ) -> str:
        """Store or update a fact. Returns the fact ID."""
        # Compute embedding BEFORE starting DB transaction to avoid
        # yielding control (await) with uncommitted writes.
        embedding = None
        if self.embed_fn:
            try:
                embedding = await self.embed_fn(f"{key}: {value}")
            except Exception as e:
                logger.warning(f"Embedding failed for {key}: {e}")

        existing = self.db.execute("SELECT id, access_count FROM facts WHERE key = ?", (key,)).fetchone()

        if existing:
            fact_id = existing[0]
            self.db.execute(
                "UPDATE facts SET value = ?, confidence = ?, "
                "access_count = access_count + 1, last_accessed = datetime('now'), "
                "decay_score = MIN(decay_score * ?, 10.0) WHERE id = ?",
                (value, confidence, SALIENCE_BOOST, fact_id),
            )
            self.db.execute("DELETE FROM facts_fts WHERE fact_id = ?", (fact_id,))
            self.db.execute(
                "INSERT INTO facts_fts (fact_id, key, value, category) VALUES (?, ?, ?, ?)",
                (fact_id, key, value, category),
            )
        else:
            fact_id = generate_id("fact")
            self.db.execute(
                "INSERT INTO facts (id, key, value, category, source, confidence) VALUES (?, ?, ?, ?, ?, ?)",
                (fact_id, key, value, category, source, confidence),
            )
            self.db.execute(
                "INSERT INTO facts_fts (fact_id, key, value, category) VALUES (?, ?, ?, ?)",
                (fact_id, key, value, category),
            )

        if embedding is not None:
            self._store_embedding(fact_id, embedding)

        self.db.commit()
        return fact_id

    def _store_embedding(self, fact_id: str, embedding: list[float]) -> None:
        from sqlite_vec import serialize_float32

        blob = serialize_float32(embedding)
        self.db.execute("DELETE FROM facts_vec WHERE id = ?", (fact_id,))
        self.db.execute("INSERT INTO facts_vec (id, embedding) VALUES (?, ?)", (fact_id, blob))

    async def search(self, query: str, top_k: int = 10) -> list[MemoryFact]:
        """Search facts using combined vector + keyword retrieval."""
        results: dict[str, dict[str, float]] = {}

        if self.embed_fn:
            try:
                query_embedding = await self.embed_fn(query)
                for fact_id, distance in self._vector_search(query_embedding, top_k * 2):
                    similarity = 1.0 / (1.0 + distance)
                    results[fact_id] = {"vector_score": similarity, "keyword_score": 0.0}
            except Exception:
                logger.warning("Vector search failed, falling back to keyword only")

        for fact_id, rank in self._keyword_search(query, top_k * 2):
            if fact_id in results:
                results[fact_id]["keyword_score"] = rank
            else:
                results[fact_id] = {"vector_score": 0.0, "keyword_score": rank}

        scored_facts = []
        for fact_id, scores in results.items():
            fact = self._get_fact(fact_id)
            if fact:
                combined = 0.7 * scores["vector_score"] + 0.3 * scores["keyword_score"]
                final_score = combined * fact.decay_score
                scored_facts.append((final_score, fact))
                self._touch_fact(fact_id)

        if scored_facts:
            self.db.commit()

        scored_facts.sort(key=lambda x: x[0], reverse=True)
        return [fact for _, fact in scored_facts[:top_k]]

    def _vector_search(self, embedding: list[float], top_k: int) -> list[tuple]:
        from sqlite_vec import serialize_float32

        blob = serialize_float32(embedding)
        return self.db.execute(
            "SELECT id, distance FROM facts_vec WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (blob, top_k),
        ).fetchall()

    def _keyword_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        try:
            safe_query = query.replace('"', "").replace("'", "")
            words = safe_query.split()
            fts_query = " OR ".join(w for w in words if len(w) > 1)
            if not fts_query:
                return []
            rows = self.db.execute(
                "SELECT fact_id, bm25(facts_fts) as rank FROM facts_fts "
                "WHERE facts_fts MATCH ? ORDER BY rank LIMIT ?",
                (fts_query, top_k),
            ).fetchall()
            if rows:
                min_r = min(r[1] for r in rows)
                max_r = max(r[1] for r in rows)
                rng = max_r - min_r if max_r != min_r else 1.0
                return [(r[0], 1.0 - (r[1] - min_r) / rng) for r in rows]
            return []
        except Exception as e:
            logger.warning(f"Keyword search failed: {e}")
            return []

    def _get_fact(self, fact_id: str) -> Optional[MemoryFact]:
        row = self.db.execute(
            "SELECT id, key, value, category, source, confidence, "
            "access_count, last_accessed, created_at, decay_score "
            "FROM facts WHERE id = ?",
            (fact_id,),
        ).fetchone()
        if not row:
            return None
        return MemoryFact(
            id=row[0],
            key=row[1],
            value=row[2],
            category=row[3],
            source=row[4],
            confidence=row[5],
            access_count=row[6],
            last_accessed=row[7],
            created_at=row[8],
            decay_score=row[9],
        )

    def _get_fact_by_key(self, key: str) -> Optional[MemoryFact]:
        """Look up a fact by key (used in tests)."""
        row = self.db.execute("SELECT id FROM facts WHERE key = ?", (key,)).fetchone()
        if not row:
            return None
        return self._get_fact(row[0])

    def _touch_fact(self, fact_id: str) -> None:
        self.db.execute(
            "UPDATE facts SET access_count = access_count + 1, "
            "last_accessed = datetime('now'), "
            "decay_score = MIN(decay_score * ?, 10.0) WHERE id = ?",
            (SALIENCE_BOOST, fact_id),
        )

    def decay_all(self) -> None:
        """Apply decay to all facts. Call periodically between tasks."""
        self.db.execute("UPDATE facts SET decay_score = MAX(decay_score * ?, 0.01)", (SALIENCE_DECAY_RATE,))
        self.db.commit()

    def get_high_salience_facts(self, top_k: int = 20) -> list[MemoryFact]:
        """Return facts with highest salience scores."""
        rows = self.db.execute("SELECT id FROM facts ORDER BY decay_score DESC LIMIT ?", (top_k,)).fetchall()
        facts = []
        for r in rows:
            fact = self._get_fact(r[0])
            if fact:
                facts.append(fact)
        return facts

    async def store_facts_batch(self, facts: list[dict]) -> int:
        """Store multiple structured facts at once.

        Each dict should have 'key' and 'value', with optional 'category'.
        Returns the count of facts successfully stored.
        """
        stored = 0
        for fact in facts:
            key = fact.get("key")
            value = fact.get("value")
            if not key or not value:
                continue
            category = fact.get("category", "general")
            try:
                await self.store_fact(key=key, value=value, category=category, source="context_flush")
                stored += 1
            except Exception as e:
                logger.warning(f"Failed to store fact '{key}': {e}")
        return stored

    async def log_action(
        self,
        action: str,
        input_summary: str,
        output_summary: str,
        task_id: str | None = None,
        tokens_used: int = 0,
        duration_ms: int = 0,
    ) -> None:
        """Log an action to the action history."""
        self.db.execute(
            "INSERT INTO logs (id, action, input_summary, output_summary, task_id, tokens_used, duration_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (generate_id("log"), action, input_summary, output_summary, task_id, tokens_used, duration_ms),
        )
        self.db.commit()

    def get_recent_logs(self, limit: int = 50) -> list[MemoryLog]:
        """Return recent action log entries."""
        rows = self.db.execute(
            "SELECT id, action, input_summary, output_summary, task_id, "
            "tokens_used, duration_ms, timestamp FROM logs ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            MemoryLog(
                id=r[0],
                action=r[1],
                input_summary=r[2],
                output_summary=r[3],
                task_id=r[4],
                tokens_used=r[5],
                duration_ms=r[6],
                timestamp=r[7],
            )
            for r in rows
        ]

    def close(self) -> None:
        """Close the database connection."""
        self.db.close()
