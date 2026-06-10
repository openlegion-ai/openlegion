"""A1 — close the rating → learning loop.

``rate_delivery`` / the dashboard rating UI store outcomes on the task
row, but until this helper existed NOTHING carried actionable feedback
back into the rated agent's behaviour — the docstrings that claimed
"memory writes on rework/rejected" were aspirational. This helper makes
them true: on ``rework`` / ``rejected`` the feedback is pushed
best-effort to the assignee's ``POST /learnings/feedback`` endpoint,
which appends it to the agent's corrections file — from there it rides
``get_learnings_context`` into every future task / chat / heartbeat
prompt automatically.

``accepted`` / ``acknowledged`` are deliberately NOT pushed: they are
rating signals, not corrections, and stuffing praise into the
corrections file would dilute the signal the LLM is told to follow.

Best-effort by design: the rated agent's container may be stopped or
archived. A failed push is logged and surfaced as
``feedback_push: "failed"`` in the rating response — the rating itself
is already durable on the task row, and a ``rework`` outcome still
carries the feedback as the spawned task's brief, so the loop degrades
gracefully rather than blocking the rating.
"""

from __future__ import annotations

from src.shared.utils import setup_logging

logger = setup_logging("host.feedback_push")

_ACTIONABLE_OUTCOMES = ("rework", "rejected")


async def push_outcome_feedback(
    transport, record: dict | None, outcome: str, feedback: str,
) -> str | None:
    """Push rating feedback to the rated agent's learnings endpoint.

    Returns ``"recorded"`` on success, ``"failed"`` on a push error, or
    ``None`` when the outcome carries nothing actionable (wrong outcome
    kind, empty feedback, no assignee, no transport).
    """
    if outcome not in _ACTIONABLE_OUTCOMES or not feedback:
        return None
    if transport is None:
        return None
    assignee = (record or {}).get("assignee") or ""
    if not assignee:
        return None
    try:
        resp = await transport.request(
            assignee, "POST", "/learnings/feedback",
            json={
                "task_id": (record or {}).get("id", ""),
                "title": (record or {}).get("title", ""),
                "outcome": outcome,
                "feedback": feedback,
            },
            timeout=10,
        )
    except Exception as e:
        logger.warning("feedback push to %s failed: %s", assignee, e)
        return "failed"
    if isinstance(resp, dict) and resp.get("error"):
        logger.warning(
            "feedback push to %s failed: %s", assignee, resp.get("error"),
        )
        return "failed"
    return "recorded"
