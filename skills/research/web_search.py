"""Web search skill stub. Routes through mesh API proxy."""

from __future__ import annotations

from src.agent.skills import skill


@skill(
    name="web_search",
    description="Search the web for information about a company or person",
    parameters={
        "query": {"type": "string", "description": "Search query"},
        "max_results": {"type": "integer", "description": "Max results to return", "default": 5},
    },
)
async def web_search(query: str, max_results: int = 5, *, mesh_client=None):
    """External API calls go through mesh API proxy -- never direct."""
    if mesh_client is None:
        return {"error": "No mesh_client provided", "results": []}
    response = await mesh_client.api_call(
        service="brave_search",
        action="search",
        params={"query": query, "count": max_results},
    )
    return response.data if response.success else {"error": response.error}
