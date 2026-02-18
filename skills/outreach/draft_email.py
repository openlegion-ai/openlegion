"""Email drafting skill stub."""

from __future__ import annotations

from src.agent.skills import skill


@skill(
    name="draft_email",
    description="Draft a personalized outreach email based on prospect research",
    parameters={
        "prospect_name": {"type": "string", "description": "Prospect name"},
        "company_name": {"type": "string", "description": "Company name"},
        "context": {"type": "string", "description": "Research context for personalization", "default": ""},
    },
)
async def draft_email(prospect_name: str, company_name: str, context: str = "", *, mesh_client=None):
    """Draft an email. In production, uses LLM via mesh."""
    return {
        "subject": f"Quick question about {company_name}",
        "body": f"Hi {prospect_name}, stub email content based on research.",
        "personalization_score": 0.8,
    }
