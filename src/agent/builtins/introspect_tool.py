"""Runtime introspection tool — lets agents query their own system state.

Agents call this to check permissions, budget, fleet roster, cron schedule,
or health mid-conversation when they need fresh numbers (not the startup
snapshot cached in SYSTEM.md).
"""

from __future__ import annotations

from src.agent.skills import skill


@skill(
    name="get_system_status",
    description=(
        "Query your live runtime status from the mesh: permissions, budget "
        "(current spend), fleet roster, cron schedule, or health. Use this "
        "for fresh data instead of relying on the startup snapshot in "
        "SYSTEM.md. Call with section='all' for everything, or a specific "
        "section to reduce noise."
    ),
    parameters={
        "section": {
            "type": "string",
            "description": (
                "What to query: permissions, budget, fleet, cron, health, or all"
            ),
            "enum": ["permissions", "budget", "fleet", "cron", "health", "all"],
            "default": "all",
        },
    },
)
async def introspect_tool(section: str = "all", *, mesh_client=None) -> dict:
    if not mesh_client:
        return {"error": "No mesh connection available"}
    try:
        result = await mesh_client.introspect(section)
        # Merge fleet-wide metrics when available (operator heartbeat data)
        try:
            metrics = await mesh_client.get_system_metrics()
            result["metrics"] = metrics
        except Exception:
            pass  # Metrics endpoint may not exist on older hosts
        return result
    except Exception as e:
        return {"error": f"Introspect failed: {e}"}
