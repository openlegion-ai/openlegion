"""Memory tools: search and append to persistent workspace memory.

The workspace_manager and memory_store are injected at call time via the
skills system (same pattern as mesh_client injection).
"""

from __future__ import annotations

from src.agent.skills import skill
from src.shared.utils import setup_logging

logger = setup_logging("agent.builtins.memory_tool")


async def _search_with_fallback(memory_store, query: str, top_k: int):
    """Try hierarchical search, fall back to flat search on failure."""
    try:
        return await memory_store.search_hierarchical(query, top_k=top_k)
    except Exception as e:
        logger.debug("Hierarchical search failed, falling back to flat: %s", e)
    try:
        return await memory_store.search(query, top_k=top_k)
    except Exception as e2:
        logger.warning("Flat memory search also failed: %s", e2)
        return None


def _parse_fact(content: str) -> tuple[str, str]:
    """Parse free-text content into a (key, value) pair for structured storage.

    Heuristic: if content has a colon or dash separator, split there.
    Otherwise use the first ~60 chars as the key and full text as value.
    """
    content = content.strip()
    # Try "key: value" format
    for sep in [":", " - ", " — "]:
        if sep in content:
            parts = content.split(sep, 1)
            if len(parts[0]) <= 80 and parts[1].strip():
                return parts[0].strip(), parts[1].strip()
    # Fallback: truncate for key, full text for value
    key = content[:60].rstrip()
    if len(content) > 60:
        key = key.rsplit(" ", 1)[0] if " " in key else key
    return key, content


@skill(
    name="memory_search",
    description=(
        "Search your long-term memory. By default searches both workspace files "
        "and your structured fact database. Provide a category to search only "
        "the fact database filtered to that category. Use this to recall facts, "
        "preferences, decisions, or past events before answering questions."
    ),
    parameters={
        "query": {"type": "string", "description": "What to search for"},
        "category": {
            "type": "string",
            "description": (
                "Optional: filter to a fact category (e.g. 'user_preferences', "
                "'decisions'). When set, searches only the structured fact database."
            ),
            "default": "",
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum results to return (default 5)",
            "default": 5,
        },
    },
)
async def memory_search(
    query: str, category: str = "", max_results: int = 5,
    *, workspace_manager=None, memory_store=None,
) -> dict:
    """Search workspace memory files and structured fact database."""
    results = []

    # Category-filtered search: only the structured fact DB
    if category:
        if memory_store is None:
            return {"error": "No memory_store available for category search", "results": []}
        # Over-fetch when filtering by category since post-fetch filtering
        # may discard many results
        fetch_k = max_results * 3
        facts = await _search_with_fallback(memory_store, query, fetch_k)
        if facts is None:
            return {"error": "Memory search failed", "results": []}
        for fact in facts:
            if fact.category.lower() != category.lower():
                continue
            results.append({
                "key": fact.key,
                "value": fact.value,
                "category": fact.category,
                "confidence": fact.confidence,
                "access_count": fact.access_count,
                "source": "memory_db",
            })
        return {"results": results, "count": len(results)}

    # Default: search both workspace and DB
    if workspace_manager is not None:
        ws_hits = workspace_manager.search(query, max_results=max_results)
        for hit in ws_hits:
            hit["source"] = "workspace"
            results.append(hit)

    if memory_store is not None:
        db_facts = await _search_with_fallback(memory_store, query, max_results)
        if db_facts:
            for fact in db_facts:
                results.append({
                    "key": fact.key,
                    "value": fact.value,
                    "category": fact.category,
                    "confidence": fact.confidence,
                    "source": "memory_db",
                })

    if not results and workspace_manager is None and memory_store is None:
        return {"error": "No memory backends available", "results": []}

    return {"results": results, "count": len(results)}


@skill(
    name="memory_save",
    description=(
        "Save an important fact or note to long-term memory. "
        "Saved to both the daily session log and the structured fact database, "
        "so it can be recalled later with memory_search. "
        "Examples: user preferences, decisions made, key findings."
    ),
    parameters={
        "content": {
            "type": "string",
            "description": "The fact or note to save",
        },
    },
)
async def memory_save(content: str, *, workspace_manager=None, memory_store=None) -> dict:
    """Save a fact to both the daily log and structured memory DB."""
    saved_workspace = False
    saved_db = False

    # 1. Workspace daily log (human-readable markdown)
    if workspace_manager is not None:
        workspace_manager.append_daily_log(content)
        saved_workspace = True

    # 2. Structured memory DB (searchable via memory_search)
    if memory_store is not None:
        try:
            # Parse content into key/value — use first sentence or clause as key
            key, value = _parse_fact(content)
            await memory_store.store_fact(
                key=key, value=value, source="memory_save", confidence=0.9,
            )
            saved_db = True
        except Exception as e:
            logger.warning(f"Failed to store fact in memory DB: {e}")

    if not saved_workspace and not saved_db:
        return {"error": "No memory backends available"}

    return {"saved": True, "saved_workspace": saved_workspace, "saved_db": saved_db, "content": content}
