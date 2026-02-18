"""Memory tools: search and append to persistent workspace memory.

The workspace_manager is injected at call time via the skills system
(same pattern as mesh_client injection).
"""

from __future__ import annotations

from src.agent.skills import skill


@skill(
    name="memory_search",
    description=(
        "Search your long-term memory for relevant information. "
        "Use this when you need to recall facts, preferences, or past events."
    ),
    parameters={
        "query": {"type": "string", "description": "What to search for"},
        "max_results": {
            "type": "integer",
            "description": "Maximum results to return (default 5)",
            "default": 5,
        },
    },
)
def memory_search(query: str, max_results: int = 5, *, workspace_manager=None) -> dict:
    """Search workspace memory files using BM25."""
    if workspace_manager is None:
        return {"error": "No workspace_manager available", "results": []}
    results = workspace_manager.search(query, max_results=max_results)
    return {"results": results, "count": len(results)}


@skill(
    name="memory_save",
    description=(
        "Save an important fact or note to your daily session log. "
        "Use this to remember things for future sessions. "
        "Examples: user preferences, decisions made, key findings."
    ),
    parameters={
        "content": {
            "type": "string",
            "description": "The fact or note to save",
        },
    },
)
def memory_save(content: str, *, workspace_manager=None) -> dict:
    """Append an entry to today's daily log."""
    if workspace_manager is None:
        return {"error": "No workspace_manager available"}
    workspace_manager.append_daily_log(content)
    return {"saved": True, "content": content}
