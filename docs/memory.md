# Memory System

OpenLegion agents have a four-layer memory architecture that provides persistent recall across sessions, tasks, and context window resets.

## Four Layers

```
Layer 4: Context Manager          <- Manages the LLM's context window
  |  Monitors token usage
  |  Proactive flush at 60% capacity
  |  Auto-compacts at 70% capacity
  |  Extracts facts before discarding messages
  |
Layer 3: Workspace Files          <- Durable, human-readable storage
  |  INSTRUCTIONS.md, SOUL.md, USER.md  (loaded into system prompt)
  |  SYSTEM.md                    (auto-generated architecture guide, static preamble)
  |  MEMORY.md                    (curated long-term facts)
  |  HEARTBEAT.md                 (autonomous monitoring rules)
  |  memory/YYYY-MM-DD.md         (daily session logs)
  |  BM25 search across all files
  |
Layer 2: Structured Memory DB     <- Hierarchical vector database
  |  SQLite + sqlite-vec + FTS5
  |  Facts with embeddings (KNN similarity search)
  |  Hybrid scoring: (0.7 * vector + 0.3 * keyword) * decay_score
  |  Three-tier retrieval: categories -> scoped facts -> flat fallback
  |  Vector search auto-disables after 3 consecutive embedding failures
  |  (falls back to keyword-only)
  |  Reinforcement scoring with access-count boost + recency decay
  |
Layer 1: Salience Tracking        <- Prioritizes important facts
     Access count, decay score, last accessed timestamp
     High-salience facts auto-surface in initial context
```

## Context Manager (`src/agent/context.py`)

Manages the LLM's context window to prevent overflow while preserving important information.

### Token Tracking

- **Model-aware token estimation**: tiktoken for OpenAI models (accurate), 3.5 chars/token for Anthropic, 4 chars/token fallback for unknown providers
- **Model-aware context windows**: uses litellm's `max_input_tokens` with a hardcoded fallback table (16 entries), with explicit `max_tokens` override
- Four compaction thresholds:
  - **60%** -- proactive flush (extract facts to MEMORY.md)
  - **70%** -- auto-compact (summarize + trim, preserving tool-call group boundaries). If the LLM is unavailable or summarization returns an empty summary (`context.py:452-453`), this path falls back to hard prune.
  - **80%** -- warning injected into system prompt telling agents to wrap up or save important facts
  - **90%** -- emergency hard prune (`_hard_prune()`): keep first message + last 4 message groups (tool-call boundaries preserved; bridges same-role sequences). Triggers if 80% warning was not heeded or summarization fails.

### Write-Then-Compact Pattern

Before discarding any conversation context:

1. **Extract** -- Asks the LLM to identify important facts from the conversation
2. **Store** -- Saves facts to both `MEMORY.md` and the structured memory DB
3. **Summarize** -- Creates a concise summary of the conversation so far
4. **Replace** -- Swaps full history with: summary + first message + last 4 message groups (groups = tool-call sequences; preserves the user→assistant(tool_calls)→tool(result)→assistant invariant by bridging same-role sequences)

Nothing is permanently lost during compaction.

### Message Grouping

The context trimmer respects LLM tool-calling message role sequences:
```
user -> assistant(tool_calls) -> tool(result) -> assistant
```

An assistant message with `tool_calls` is never separated from its corresponding `tool` result messages. Breaking this invariant causes LLM API errors.

## Workspace (`src/agent/workspace.py`)

Persistent markdown files stored on the agent's `/data/workspace` volume.

### Core Files

| File | Purpose | Cap | When Loaded |
|------|---------|-----|-------------|
| `INSTRUCTIONS.md` | Operating procedures, workflow rules, domain knowledge | 12K chars | System prompt |
| `SOUL.md` | Agent personality and behavioral guidelines | 4K chars | System prompt |
| `USER.md` | User preferences and working style | 4K chars | System prompt |
| `MEMORY.md` | Curated long-term facts | 16K chars | System prompt |
| `TEAM.md` | Team-wide context (optional, per-team, mounted from host, **read-only** when present). Pre-rename `PROJECT.md` files are migrated to `TEAM.md` once at startup; the bootstrap loader reads only `TEAM.md`/`team.md`. | -- | When present in the team config |
| `SYSTEM.md` | System architecture guide (auto-generated, static preamble, read-only) | capped at 6K chars | System prompt |
| `HEARTBEAT.md` | Autonomous monitoring rules | -- | Heartbeat dispatch (auto-loaded) |
| `INTERFACE.md` | Public collaboration contract (inputs, outputs, subscriptions) | 4K chars | Not auto-loaded into system prompt — read by other agents via `get_agent_profile` |

Total bootstrap injection into the system prompt is capped at 48K characters across all files.

### System File (`SYSTEM.md`)

Generated once at agent startup. Contains a static preamble only:

- **Static preamble** — Explains how the mesh, credential vault, blackboard, pub/sub, context window, and tool costs work. Includes a "common errors and what they mean" section (403, 429, budget exceeded, tool loop detected). This teaches agents *how the system works* so they can self-diagnose issues. Permissions and fleet roster are **not** in SYSTEM.md — they come live via the **Runtime Context** block injected on each turn.

SYSTEM.md is capped at 6,000 characters to limit system prompt bloat. It is read-only — agents cannot modify it via `update_workspace`. The authoritative live data comes from the **Runtime Context** block injected into the system prompt on each turn (see [Agent Tools — System Introspection](agent-tools.md#system-introspection)).

### Daily Logs

Session activity is logged to `memory/YYYY-MM-DD.md`. Each entry is timestamped. Three types of entries are recorded:

**Chat turns** — include user intent summary, tool names used, and a multi-line-aware response summary:
```
[14:30:22] Chat: Research Acme Corp competitors | Tools: web_search, memory_save | Response: Found 3 main competitors (+5 lines)
```

**Task completions** — include task type, iteration count, token usage, duration, and tools used:
```
[14:45:10] Task complete: research_prospect | 8 iterations, 12340 tokens, 45s | Tools: web_search, http_request | Input: {"company": "Acme Corp"}
```

**Task failures** — include failure reason (max iterations or error) and context:
```
[15:00:05] Task FAILED (max iterations): qualify_lead | 8500 tokens | Input: {"lead_id": "abc123"}
```

Daily logs are auto-loaded into heartbeat messages (last 2 days, capped at 4000 chars) so agents have continuity across heartbeat wake-ups.

### BM25 Search

The workspace supports keyword search across all markdown files using a built-in BM25 implementation (separate from the FTS5 BM25 used by structured memory — workspace search runs over markdown files on disk, not the per-agent SQLite database):
- Tokenization with stop-word removal
- BM25 scoring (k1=1.5, b=0.75)
- Returns ranked snippets with file paths
- Files already injected into the system prompt (e.g. MEMORY.md, SOUL.md) are excluded from search results to avoid duplicate context

## Structured Memory (`src/agent/memory.py`)

The per-agent SQLite database (`/data/{agent_id}.db` — absolute path inside the container, on the persistent `/data` volume) stores three types of data:
- **Facts** with embeddings for semantic search (described below)
- **Task checkpoints** — iteration state, message history, token usage, and budget state, used to resume interrupted tasks after a container restart
- **Chat checkpoints** — conversation state for crash recovery across chat sessions

SQLite database with three search capabilities:

### Vector Search (sqlite-vec)

- 1536-dimensional embeddings for each fact
- KNN similarity search for semantic recall
- Embeddings generated via the mesh LLM proxy
- Default embedding model is `text-embedding-3-small` (OpenAI). The provider is picked from the LLM model prefix in `src/cli/runtime.py:_default_embedding_model`; non-OpenAI providers (Anthropic, xAI, Groq, …) default to `"none"` which disables vector search entirely — those agents fall back to keyword-only retrieval.
- After `_EMBED_FAILURE_THRESHOLD = 3` consecutive embedding failures (network errors, missing provider key, etc.) the store sets `embed_fn = None` for the rest of the process lifetime and logs a warning. Vector search silently degrades to keyword-only until the container restarts.

### Full-Text Search (FTS5)

- SQLite FTS5 index for keyword matching (no explicit tokenizer — uses SQLite's implicit `unicode61` default)
- Queries are sanitized to alphanumeric terms (special characters stripped, words 2+ chars, limit 20 terms, joined with `OR`)
- Combined with vector search for hybrid retrieval

### Hybrid Scoring

Each fact returned by `search()` is scored as:

```
final_score = (0.7 * vector_score + 0.3 * keyword_score) * decay_score
```

- `vector_score = 1 / (1 + cosine_distance)` from sqlite-vec KNN
- `keyword_score` is normalized BM25 rank (0..1) from FTS5
- `decay_score` is the per-fact salience multiplier (see Salience Scoring below)

Facts not returned by both retrievals get the missing component as `0.0`, so a pure vector hit and a pure keyword hit are still comparable.

### Hierarchical Retrieval

Three-tier search strategy (`search_hierarchical`):

1. **Category search** -- Vector search on category embeddings → top 3 categories
2. **Scoped facts** -- Search within those categories
3. **Flat fallback** -- If scoped search yields too few results, search all facts

The `memory_tool` builtin wraps this with `_search_with_fallback`: it calls `search_hierarchical` first and silently falls back to flat `search()` if the hierarchical path raises.

### Auto-Categorization

Facts are automatically categorized when saved by `_auto_categorize`:

1. **Vector match** — compare the new fact's embedding against existing category embeddings (cosine similarity, threshold `_CATEGORY_SIM_THRESHOLD = 0.7`). On a match, the fact joins that category.
2. **LLM fallback** — if no category matches and a `categorize_fn` callback is configured, the LLM is asked to name a category.
3. **Text fallback** — if no `categorize_fn` is configured (the default for stock agents — `__main__.py` only wires `embed_fn`), the fact's own `category` field text is used as the category name (defaulting to `"general"`). No keyword inference is performed.

Category embeddings are recomputed as the running average of member embeddings every `_CATEGORY_RECOMPUTE_INTERVAL = 10` new members.

### Dedup-by-Key in `store_fact`

`store_fact(key, value, …)` deduplicates on `key`: if a row with the same key already exists, the row's `value`, `confidence`, and `last_accessed` are updated, `access_count` is incremented, and `decay_score` is boosted by `_compute_boost(new_count)` (capped at 10.0 via `MIN(boost, 10.0)`). The FTS5 row is rebuilt to match. This is the mechanism behind the reinforcement scoring described below — saving the same key repeatedly compounds its salience instead of creating duplicates.

### Salience Scoring

Each fact has a `decay_score` (initialised to `1.0` at insert, decayed by `SALIENCE_DECAY_RATE = 0.95` over time) that combines:
- **Access count** -- Incremented on every `store_fact` and `search` hit; drives the boost capped at 10.0
- **Recency decay** -- `decay_score` decreases over time since `last_accessed`
- **Boost** -- The boosted `decay_score` multiplies the hybrid score (see Hybrid Scoring), so high-salience facts auto-surface in initial context

## Cross-Session Memory

Facts persist across chat resets and container restarts:

```
Session 1: User says "My timezone is PST"
           Agent: memory_save("user_timezone", "PST")
           -> Stored in daily log + structured DB

  === Chat Reset ===

Session 2: User asks "What timezone am I in?"
           Agent: memory_search("user timezone")
           -> Returns "PST" via semantic search
```

