# Memory System

OpenLegion agents have a five-layer memory architecture that provides persistent, self-improving recall across sessions, tasks, and context window resets.

## Five Layers

```
Layer 5: Context Manager          <- Manages the LLM's context window
  |  Monitors token usage
  |  Proactive flush at 60% capacity
  |  Auto-compacts at 70% capacity
  |  Extracts facts before discarding messages
  |
Layer 4: Learnings                <- Self-improvement through failure tracking
  |  learnings/errors.md         (tool failures with context)
  |  learnings/corrections.md   (user corrections and preferences)
  |  Auto-injected into system prompt each session
  |
Layer 3: Workspace Files          <- Durable, human-readable storage
  |  AGENTS.md, SOUL.md, USER.md  (loaded into system prompt)
  |  MEMORY.md                    (curated long-term facts)
  |  HEARTBEAT.md                 (autonomous monitoring rules)
  |  memory/YYYY-MM-DD.md         (daily session logs)
  |  BM25 search across all files
  |
Layer 2: Structured Memory DB     <- Hierarchical vector database
  |  SQLite + sqlite-vec + FTS5
  |  Facts with embeddings (KNN similarity search)
  |  Auto-categorization with category-scoped search
  |  3-tier retrieval: categories -> scoped facts -> flat fallback
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
- **Model-aware context windows**: auto-detects max context from model name (12 models supported), with explicit `max_tokens` override
- **80% context warning**: injected into system prompt telling agents to wrap up or save important facts
- Three compaction thresholds:
  - **60%** -- proactive flush (extract facts to MEMORY.md)
  - **70%** -- auto-compact (summarize + trim to last 4 messages)
  - **90%** -- emergency hard-prune (keep only first + last 4 messages)

### Write-Then-Compact Pattern

Before discarding any conversation context:

1. **Extract** -- Asks the LLM to identify important facts from the conversation
2. **Store** -- Saves facts to both `MEMORY.md` and the structured memory DB
3. **Summarize** -- Creates a concise summary of the conversation so far
4. **Replace** -- Swaps full history with: summary + last 4 messages

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

| File | Purpose | When Loaded |
|------|---------|-------------|
| `AGENTS.md` | Fleet context (other agents, their roles) | System prompt |
| `SOUL.md` | Agent personality and behavioral guidelines | System prompt |
| `USER.md` | User preferences and working style | System prompt |
| `MEMORY.md` | Curated long-term facts | System prompt |
| `PROJECT.md` | Project-wide context (optional, mounted read-only from host) | System prompt |
| `HEARTBEAT.md` | Autonomous monitoring rules | Heartbeat probes |

### Daily Logs

Session activity is logged to `memory/YYYY-MM-DD.md`. Each entry is timestamped and includes facts saved by the agent during the session.

### BM25 Search

The workspace supports keyword search across all markdown files using a built-in BM25 implementation:
- Tokenization with stop-word removal
- TF-IDF scoring
- Returns ranked snippets with file paths

## Structured Memory (`src/agent/memory.py`)

SQLite database with three search capabilities:

### Vector Search (sqlite-vec)

- 1536-dimensional embeddings for each fact
- KNN similarity search for semantic recall
- Embeddings generated via the mesh LLM proxy

### Full-Text Search (FTS5)

- SQLite FTS5 index for keyword matching
- Trigram tokenizer for partial matching
- Combined with vector search for hybrid retrieval

### Hierarchical Retrieval

Three-tier search strategy:

1. **Category search** -- Find relevant categories first
2. **Scoped facts** -- Search within those categories
3. **Flat fallback** -- If scoped search yields too few results, search all facts

### Auto-Categorization

Facts are automatically categorized when saved. Categories are inferred from the key/value content (e.g., "user_preference" -> "preferences", "project_goal" -> "project_info").

### Salience Scoring

Each fact has a salience score that combines:
- **Access count** -- How often the fact has been recalled
- **Recency decay** -- Score decreases over time since last access
- **Boost** -- High-salience facts auto-surface in initial context

## Cross-Session Memory

Facts persist across chat resets and container restarts:

```
Session 1: User says "My timezone is PST"
           Agent: memory_save("user_timezone", "PST")
           -> Stored in daily log + structured DB

  === Chat Reset ===

Session 2: User asks "What timezone am I in?"
           Agent: memory_recall("user timezone")
           -> Returns "PST" via semantic search
```

## Learnings (`src/agent/workspace.py`)

Self-improvement through failure tracking:

### Error Learning

When a tool call fails, the error context is recorded in `learnings/errors.md`:
```markdown
## 2026-02-20T10:30:00
**Tool:** exec_command
**Error:** Permission denied: /etc/passwd
**Context:** Tried to read system file outside /data
**Lesson:** File operations are scoped to /data volume
```

### Correction Learning

When a user corrects the agent, it's recorded in `learnings/corrections.md`:
```markdown
## 2026-02-20T11:00:00
**User said:** "No, use pytest not unittest"
**Context:** Was writing tests with unittest framework
**Lesson:** User prefers pytest for testing
```

Both are injected into the system prompt at the start of each session, so the agent learns from past mistakes.
