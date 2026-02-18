"""Email personalization skill stub."""

from __future__ import annotations

from src.agent.skills import skill


@skill(
    name="personalize_email",
    description="Personalize an email draft with specific prospect details",
    parameters={
        "email_body": {"type": "string", "description": "Email body to personalize"},
        "prospect_data": {"type": "string", "description": "Prospect research data", "default": ""},
    },
)
async def personalize_email(email_body: str, prospect_data: str = "", *, mesh_client=None):
    """Personalize an email. Stub returns the original with a note."""
    return {"personalized_body": email_body, "changes_made": 0}
