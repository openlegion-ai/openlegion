"""Agent self-reflection for continuous improvement.

Adds a reflection step to the heartbeat loop where the agent reviews
recent failures and proposes concrete instruction improvements.

Design principles:
- Reflections are suggestions, never auto-applied
- Append-only log for full audit trail
- Rate-limited (max 1 per heartbeat cycle)
- Size-limited (max 500 chars per suggestion)
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.shared.utils import setup_logging, truncate

if TYPE_CHECKING:
    from src.agent.memory import MemoryStore

logger = setup_logging("agent.reflection")

# Configuration
_MAX_FAILURES_TO_REVIEW = 5
_MAX_SUGGESTION_LENGTH = 500
_REFLECTION_COOLDOWN_SECONDS = 300  # 5 min between reflections


@dataclass
class Reflection:
    """A single reflection entry."""

    timestamp: float
    agent_id: str
    failure_ids: list[str] = field(default_factory=list)
    failure_summary: str = ""
    suggestion: str = ""
    applied: bool = False
    rejected: bool = False
    operator_note: str = ""


class ReflectionStore:
    """Append-only reflection log with operator review support."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, reflection: Reflection) -> None:
        """Append a reflection to the log."""
        with self.log_path.open("a") as f:
            f.write(json.dumps(asdict(reflection)) + "\n")
        logger.info(
            "Reflection recorded for %s: %s",
            reflection.agent_id,
            truncate(reflection.suggestion, 80),
        )

    def get_pending(self, agent_id: str | None = None) -> list[Reflection]:
        """Get reflections awaiting operator review."""
        if not self.log_path.exists():
            return []
        results = []
        for line in self.log_path.read_text().splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            if data.get("applied") or data.get("rejected"):
                continue
            if agent_id and data.get("agent_id") != agent_id:
                continue
            results.append(Reflection(**data))
        return results

    def mark_applied(self, timestamp: float, note: str = "") -> bool:
        """Mark a reflection as applied by operator."""
        return self._update_by_timestamp(timestamp, {"applied": True, "operator_note": note})

    def mark_rejected(self, timestamp: float, note: str = "") -> bool:
        """Mark a reflection as rejected by operator."""
        return self._update_by_timestamp(timestamp, {"rejected": True, "operator_note": note})

    def _update_by_timestamp(self, timestamp: float, updates: dict) -> bool:
        """Update a reflection entry by timestamp."""
        if not self.log_path.exists():
            return False
        lines = self.log_path.read_text().splitlines()
        updated = False
        new_lines = []
        for line in lines:
            if not line.strip():
                new_lines.append(line)
                continue
            data = json.loads(line)
            if abs(data.get("timestamp", 0) - timestamp) < 0.001:
                data.update(updates)
                updated = True
            new_lines.append(json.dumps(data))
        if updated:
            self.log_path.write_text("\n".join(new_lines) + "\n")
        return updated


async def generate_reflection(
    agent_id: str,
    memory: MemoryStore,
    llm_complete: Any,
    current_instructions: str,
    cooldown_seconds: int = _REFLECTION_COOLDOWN_SECONDS,
) -> Reflection | None:
    """Generate a reflection based on recent failures.

    Returns None if no failures to reflect on or cooldown hasn't elapsed.
    """
    # Check cooldown
    recent_failures = memory.get_tool_history(limit=_MAX_FAILURES_TO_REVIEW * 2)
    failures = [f for f in recent_failures if not f.get("success")]

    if not failures:
        logger.debug("No failures to reflect on for %s", agent_id)
        return None

    failures = failures[:_MAX_FAILURES_TO_REVIEW]

    # Build failure summary
    failure_lines = []
    for i, f in enumerate(failures, 1):
        failure_lines.append(
            f"{i}. Tool: {f['tool_name']}\n"
            f"   Error: {truncate(f.get('outcome', 'unknown'), 200)}"
        )
    failure_summary = "\n".join(failure_lines)

    # Prompt for reflection
    prompt = f"""You are reviewing your recent tool execution failures to improve your instructions.

## Recent Failures

{failure_summary}

## Your Current Instructions

{truncate(current_instructions, 2000)}

## Task

Propose ONE specific, actionable change to your instructions that would help you avoid these failures.

Format: "When [situation], instead of [wrong approach], do [correct approach]"

Be concrete and specific. Max 500 characters."""

    try:
        suggestion = await llm_complete(prompt)
        suggestion = truncate(suggestion.strip(), _MAX_SUGGESTION_LENGTH)

        # Skip if suggestion is too short or looks like an error
        if len(suggestion) < 20:
            logger.debug("Reflection suggestion too short, skipping")
            return None

        return Reflection(
            timestamp=time.time(),
            agent_id=agent_id,
            failure_ids=[f.get("id", "") for f in failures],
            failure_summary=truncate(failure_summary, 500),
            suggestion=suggestion,
        )
    except Exception as e:
        logger.warning("Failed to generate reflection: %s", e)
        return None
