"""Lead scoring skill stub."""

from __future__ import annotations

from src.agent.skills import skill


@skill(
    name="score_lead",
    description="Score a lead based on ICP fit, budget signals, and engagement potential",
    parameters={
        "company_name": {"type": "string", "description": "Company name"},
        "company_data": {"type": "string", "description": "Research data about the company", "default": ""},
    },
)
async def score_lead(company_name: str, company_data: str = "", *, mesh_client=None):
    """Score a lead. In production, uses LLM via mesh to analyze data."""
    if mesh_client is None:
        return {"score": 0.5, "reasoning": "No mesh client available for analysis"}
    return {"score": 0.8, "reasoning": f"Stub scoring for {company_name}"}
