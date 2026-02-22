"""Per-agent memory with SQLite + sqlite-vec vector search + FTS5 keyword search.

Three-layer hierarchy:
  - facts: structured knowledge with salience tracking + category assignment
  - categories: hierarchical groupings for tiered search
  - logs: action history

Dual retrieval: vector similarity (KNN) + keyword (BM25), merged by weighted score.
Hierarchical search: category-tier vector search → scoped fact search → flat fallback.
Embeddings are generated via mesh proxy (embed_fn).
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import struct
from datetime import UTC, datetime
from typing import Any, Callable, Coroutine, Optional

import sqlite_vec

from src.shared.types import MemoryFact, MemoryLog
from src.shared.utils import generate_id, setup_logging

logger = setup_logging("agent.memory")

EMBEDDING_DIM = 1536
SALIENCE_DECAY_RATE = 0.95

# Type aliases for async callbacks
EmbedFn = Callable[[str], Coroutine[Any, Any, list[float]]]
CategorizeFn = Callable[[str, str], Coroutine[Any, Any, str]]

# Category similarity threshold for auto-assignment
_CATEGORY_SIM_THRESHOLD = 0.7
# Recompute category embedding every N new members
_CATEGORY_RECOMPUTE_INTERVAL = 10


class MemoryStore:
    """Per-agent memory with vector search, salience tracking, and hierarchical categories."""

    def __init__(
        self,
        db_path: str,
        embed_fn: Optional[EmbedFn] = None,
        categorize_fn: Optional[CategorizeFn] = None,
    ):
        self.db = sqlite3.connect(db_path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=5000")
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        self.embed_fn = embed_fn
        self.categorize_fn = categorize_fn
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
            CREATE TABLE IF NOT EXISTS categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                summary TEXT NOT NULL DEFAULT '',
                embedding BLOB,
                item_count INTEGER DEFAULT 0,
                updated_at TEXT DEFAULT (datetime('now'))
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS categories_vec USING vec0(
                id INTEGER PRIMARY KEY,
                embedding float[1536]
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
            CREATE TABLE IF NOT EXISTS tool_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_name TEXT NOT NULL,
                params_hash TEXT NOT NULL,
                outcome TEXT NOT NULL,
                success INTEGER NOT NULL DEFAULT 1,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_tool_outcomes_name
                ON tool_outcomes(tool_name, created_at DESC);
        """)
        # Lazy migration: add category_id FK to facts if not present
        try:
            self.db.execute("ALTER TABLE facts ADD COLUMN category_id INTEGER REFERENCES categories(id)")
        except sqlite3.OperationalError:
            pass  # Column already exists
        self.db.commit()

    @staticmethod
    def _compute_boost(access_count: int, days_since_last_access: float = 0.0) -> float:
        """Continuous reinforcement boost: increases with access, decays with time."""
        recency_factor = max(0.1, 1.0 - days_since_last_access * 0.05)
        return 1.0 + math.log(1 + access_count) * recency_factor

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
            new_count = existing[1] + 1
            boost = self._compute_boost(new_count)
            self.db.execute(
                "UPDATE facts SET value = ?, confidence = ?, "
                "access_count = ?, last_accessed = datetime('now'), "
                "decay_score = MIN(?, 10.0) WHERE id = ?",
                (value, confidence, new_count, boost, fact_id),
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
            # Auto-categorize (assigns category_id, recomputes category embeddings)
            cat_id = await self._auto_categorize(key, value, category, embedding)
            if cat_id is not None:
                self.db.execute("UPDATE facts SET category_id = ? WHERE id = ?", (cat_id, fact_id))

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

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Sanitize a query for FTS5 MATCH: strip all special chars."""
        # FTS5 special characters: *, ^, :, (, ), +, -, ", NEAR, AND, OR, NOT
        import re
        # Strip everything except alphanumeric and spaces
        safe = re.sub(r"[^\w\s]", " ", query)
        words = safe.split()
        # Only keep words with 2+ chars, limit to 20 terms
        return " OR ".join(w for w in words[:20] if len(w) > 1)

    def _keyword_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        try:
            fts_query = self._sanitize_fts_query(query)
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
            "SELECT f.id, f.key, f.value, f.category, f.source, f.confidence, "
            "f.access_count, f.last_accessed, f.created_at, f.decay_score, "
            "c.name "
            "FROM facts f LEFT JOIN categories c ON f.category_id = c.id "
            "WHERE f.id = ?",
            (fact_id,),
        ).fetchone()
        if not row:
            return None
        # Use category name from categories table if assigned, else text field
        category = row[10] if row[10] else row[3]
        return MemoryFact(
            id=row[0],
            key=row[1],
            value=row[2],
            category=category,
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
        row = self.db.execute(
            "SELECT access_count, last_accessed FROM facts WHERE id = ?",
            (fact_id,),
        ).fetchone()
        if not row:
            return
        new_count = row[0] + 1
        days_since = 0.0
        if row[1]:
            try:
                last = datetime.fromisoformat(row[1].replace("Z", "+00:00"))
                days_since = (datetime.now(UTC) - last).total_seconds() / 86400
            except (ValueError, TypeError) as e:
                logger.debug("Failed to parse last_accessed timestamp: %s", e)
        boost = self._compute_boost(new_count, days_since)
        self.db.execute(
            "UPDATE facts SET access_count = ?, "
            "last_accessed = datetime('now'), "
            "decay_score = MIN(?, 10.0) WHERE id = ?",
            (new_count, boost, fact_id),
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

    # ── Tool outcome tracking ─────────────────────────────────

    @staticmethod
    def _compute_params_hash(arguments: dict | None) -> str:
        """SHA-256 of sorted JSON arguments for deduplication."""
        raw = json.dumps(arguments or {}, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def store_tool_outcome(
        self,
        tool_name: str,
        arguments: dict | None,
        outcome: str,
        success: bool = True,
    ) -> None:
        """Record a tool execution outcome. Auto-prunes to 50 per tool."""
        params_hash = self._compute_params_hash(arguments)
        self.db.execute(
            "INSERT INTO tool_outcomes (tool_name, params_hash, outcome, success) "
            "VALUES (?, ?, ?, ?)",
            (tool_name, params_hash, outcome, 1 if success else 0),
        )
        # Prune: keep last 50 per tool (order by id for deterministic tie-breaking)
        self.db.execute(
            "DELETE FROM tool_outcomes WHERE tool_name = ? AND id NOT IN "
            "(SELECT id FROM tool_outcomes WHERE tool_name = ? ORDER BY id DESC LIMIT 50)",
            (tool_name, tool_name),
        )
        self.db.commit()

    def get_tool_history(
        self,
        tool_name: str | None = None,
        limit: int = 20,
        params_hash: str | None = None,
    ) -> list[dict]:
        """Query recent tool outcomes, optionally filtered by tool and/or params_hash."""
        query = "SELECT tool_name, params_hash, outcome, success, created_at FROM tool_outcomes"
        conditions: list[str] = []
        params: list[Any] = []
        if tool_name:
            conditions.append("tool_name = ?")
            params.append(tool_name)
        if params_hash:
            conditions.append("params_hash = ?")
            params.append(params_hash)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self.db.execute(query, params).fetchall()
        return [
            {
                "tool_name": r[0],
                "params_hash": r[1],
                "outcome": r[2],
                "success": bool(r[3]),
                "created_at": r[4],
            }
            for r in rows
        ]

    # ── Hierarchical category system ────────────────────────────

    async def _auto_categorize(
        self, key: str, value: str, category: str, embedding: list[float],
    ) -> int | None:
        """Assign a fact to a category based on vector similarity or LLM suggestion.

        Returns the category ID if assigned, None otherwise.
        Gracefully degrades: if no categorize_fn is set, uses the fact's text
        category field (e.g. "preference", "tool:web_search") as the name.
        """
        # Check existing categories by vector similarity
        cat_matches = self._search_categories(embedding, top_k=3)
        if cat_matches:
            best_id, best_sim = cat_matches[0]
            if best_sim > _CATEGORY_SIM_THRESHOLD:
                self._increment_category_count(best_id)
                return best_id

        # Ask LLM for category name if callback is available
        if self.categorize_fn:
            try:
                cat_name = await self.categorize_fn(key, value)
                cat_name = cat_name.strip().lower()[:100]
                if not cat_name:
                    return None
            except Exception as e:
                logger.warning(f"Categorize callback failed: {e}")
                return None
        else:
            # No categorize_fn — use the fact's existing category field
            cat_name = category.strip().lower()[:100] or "general"

        return self._get_or_create_category(cat_name, embedding)

    def _get_or_create_category(self, name: str, fact_embedding: list[float]) -> int:
        """Get existing category by name or create a new one. Returns category ID."""
        row = self.db.execute("SELECT id FROM categories WHERE name = ?", (name,)).fetchone()
        if row:
            cat_id = row[0]
            self._increment_category_count(cat_id)
            return cat_id

        # Create new category with the fact's embedding as initial embedding
        from sqlite_vec import serialize_float32

        blob = serialize_float32(fact_embedding)
        cursor = self.db.execute(
            "INSERT INTO categories (name, embedding, item_count) VALUES (?, ?, 1)",
            (name, blob),
        )
        cat_id = cursor.lastrowid
        self.db.execute(
            "INSERT INTO categories_vec (id, embedding) VALUES (?, ?)",
            (cat_id, blob),
        )
        return cat_id

    def _increment_category_count(self, cat_id: int) -> None:
        """Increment item_count and maybe recompute category embedding."""
        self.db.execute(
            "UPDATE categories SET item_count = item_count + 1, "
            "updated_at = datetime('now') WHERE id = ?",
            (cat_id,),
        )
        row = self.db.execute("SELECT item_count FROM categories WHERE id = ?", (cat_id,)).fetchone()
        if row and row[0] > 0 and row[0] % _CATEGORY_RECOMPUTE_INTERVAL == 0:
            self._recompute_category_embedding(cat_id)

    def _recompute_category_embedding(self, cat_id: int) -> None:
        """Recompute category embedding as average of its facts' embeddings."""
        from sqlite_vec import serialize_float32

        rows = self.db.execute(
            "SELECT fv.embedding FROM facts_vec fv "
            "JOIN facts f ON f.id = fv.id "
            "WHERE f.category_id = ?",
            (cat_id,),
        ).fetchall()
        if not rows:
            return

        dim = EMBEDDING_DIM
        avg = [0.0] * dim
        for (blob,) in rows:
            values = struct.unpack(f"{dim}f", blob)
            for i in range(dim):
                avg[i] += values[i]
        n = len(rows)
        avg = [v / n for v in avg]

        blob = serialize_float32(avg)
        self.db.execute("UPDATE categories SET embedding = ? WHERE id = ?", (blob, cat_id))
        self.db.execute("DELETE FROM categories_vec WHERE id = ?", (cat_id,))
        self.db.execute("INSERT INTO categories_vec (id, embedding) VALUES (?, ?)", (cat_id, blob))

    def _search_categories(self, query_embedding: list[float], top_k: int = 3) -> list[tuple[int, float]]:
        """Vector search on category embeddings. Returns [(cat_id, similarity)]."""
        from sqlite_vec import serialize_float32

        # Check if any categories exist
        count = self.db.execute("SELECT COUNT(*) FROM categories").fetchone()[0]
        if count == 0:
            return []

        blob = serialize_float32(query_embedding)
        rows = self.db.execute(
            "SELECT id, distance FROM categories_vec "
            "WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (blob, top_k),
        ).fetchall()
        return [(row[0], 1.0 / (1.0 + row[1])) for row in rows]

    async def search_hierarchical(self, query: str, top_k: int = 10) -> list[MemoryFact]:
        """Search facts using tiered category-first strategy.

        Tier 1: Vector search on category embeddings → top 3 categories
        Tier 2: Vector+BM25 search scoped to matched categories
        Tier 3 fallback: Full flat search if results insufficient
        """
        results: list[MemoryFact] = []
        seen_ids: set[str] = set()

        if self.embed_fn:
            try:
                query_embedding = await self.embed_fn(query)

                # Tier 1: Find relevant categories
                cat_matches = self._search_categories(query_embedding, top_k=3)
                if cat_matches:
                    cat_ids = [cid for cid, _ in cat_matches]
                    # Tier 2: Scoped search within matched categories
                    scoped = self._search_within_categories(
                        query, query_embedding, cat_ids, top_k,
                    )
                    for fact in scoped:
                        if fact.id not in seen_ids:
                            results.append(fact)
                            seen_ids.add(fact.id)

                    # Sufficiency check
                    if len(results) >= top_k:
                        return results[:top_k]
            except Exception:
                logger.warning("Hierarchical search tier 1/2 failed, falling back to flat")

        # Tier 3: Flat fallback
        flat_results = await self.search(query, top_k=top_k)
        for fact in flat_results:
            if fact.id not in seen_ids:
                results.append(fact)
                seen_ids.add(fact.id)

        return results[:top_k]

    def _search_within_categories(
        self,
        query: str,
        query_embedding: list[float],
        category_ids: list[int],
        top_k: int,
    ) -> list[MemoryFact]:
        """Hybrid vector+BM25 search scoped to specific categories."""
        if not category_ids:
            return []

        placeholders = ",".join("?" * len(category_ids))
        results: dict[str, dict[str, float]] = {}

        # Vector search scoped to category
        from sqlite_vec import serialize_float32

        blob = serialize_float32(query_embedding)
        # Get candidate fact IDs from these categories
        scoped_ids = self.db.execute(
            f"SELECT id FROM facts WHERE category_id IN ({placeholders})",
            category_ids,
        ).fetchall()
        scoped_id_set = {row[0] for row in scoped_ids}

        if scoped_id_set:
            # Run vector search and filter by scoped IDs
            vec_rows = self.db.execute(
                "SELECT id, distance FROM facts_vec WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
                (blob, top_k * 3),
            ).fetchall()
            for fact_id, distance in vec_rows:
                if fact_id in scoped_id_set:
                    similarity = 1.0 / (1.0 + distance)
                    results[fact_id] = {"vector_score": similarity, "keyword_score": 0.0}

        # Keyword search scoped to category
        try:
            fts_query = self._sanitize_fts_query(query)
            if fts_query:
                fts_rows = self.db.execute(
                    "SELECT fact_id, bm25(facts_fts) as rank FROM facts_fts "
                    "WHERE facts_fts MATCH ? ORDER BY rank LIMIT ?",
                    (fts_query, top_k * 3),
                ).fetchall()
                if fts_rows:
                    min_r = min(r[1] for r in fts_rows)
                    max_r = max(r[1] for r in fts_rows)
                    rng = max_r - min_r if max_r != min_r else 1.0
                    for fact_id, rank in fts_rows:
                        if fact_id in scoped_id_set:
                            norm_rank = 1.0 - (rank - min_r) / rng
                            if fact_id in results:
                                results[fact_id]["keyword_score"] = norm_rank
                            else:
                                results[fact_id] = {"vector_score": 0.0, "keyword_score": norm_rank}
        except Exception as e:
            logger.warning(f"Scoped keyword search failed: {e}")

        # Score and rank
        scored_facts = []
        for fact_id, scores in results.items():
            fact = self._get_fact(fact_id)
            if fact:
                combined = 0.7 * scores["vector_score"] + 0.3 * scores["keyword_score"]
                final_score = combined * fact.decay_score
                scored_facts.append((final_score, fact))

        scored_facts.sort(key=lambda x: x[0], reverse=True)
        return [fact for _, fact in scored_facts[:top_k]]

    def get_categories(self) -> list[dict]:
        """Return all categories with their item counts."""
        rows = self.db.execute(
            "SELECT id, name, summary, item_count, updated_at FROM categories ORDER BY item_count DESC",
        ).fetchall()
        return [
            {"id": r[0], "name": r[1], "summary": r[2], "item_count": r[3], "updated_at": r[4]}
            for r in rows
        ]

    def close(self) -> None:
        """Close the database connection."""
        self.db.close()
