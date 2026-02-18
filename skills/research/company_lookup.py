"""Company lookup skill stub."""

from __future__ import annotations

from src.agent.skills import skill


@skill(
    name="company_lookup",
    description="Look up detailed company information from business databases",
    parameters={
        "company_name": {"type": "string", "description": "Company name to look up"},
        "domain": {"type": "string", "description": "Company domain", "default": ""},
    },
)
async def company_lookup(company_name: str, domain: str = "", *, mesh_client=None):
    """Look up company info via API proxy. Stub returns mock data."""
    if mesh_client is None:
        return {"error": "No mesh client"}
    return {
        "name": company_name,
        "domain": domain,
        "employees": 500,
        "industry": "Technology",
        "revenue_estimate": "50M",
    }
