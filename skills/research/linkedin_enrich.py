"""LinkedIn enrichment skill stub."""

from __future__ import annotations

from src.agent.skills import skill


@skill(
    name="linkedin_enrich",
    description="Enrich a contact profile with LinkedIn data",
    parameters={
        "name": {"type": "string", "description": "Person's name"},
        "company": {"type": "string", "description": "Company name", "default": ""},
    },
)
async def linkedin_enrich(name: str, company: str = "", *, mesh_client=None):
    """Enrich via LinkedIn API proxy. Stub returns mock data."""
    return {
        "name": name,
        "title": "VP of Engineering",
        "company": company,
        "location": "San Francisco, CA",
    }
