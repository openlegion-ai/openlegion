"""ICP matching skill stub."""

from __future__ import annotations

from src.agent.skills import skill


@skill(
    name="icp_match",
    description="Check if a company matches the Ideal Customer Profile",
    parameters={
        "company_name": {"type": "string", "description": "Company name"},
        "industry": {"type": "string", "description": "Company industry", "default": ""},
        "employee_count": {"type": "integer", "description": "Number of employees", "default": 0},
    },
)
async def icp_match(company_name: str, industry: str = "", employee_count: int = 0, *, mesh_client=None):
    """Check ICP match. Stub returns basic match result."""
    return {"matches": True, "confidence": 0.7, "company": company_name}
