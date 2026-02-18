"""Context window manager — virtual memory for cognition.

Monitors token usage across the conversation and auto-compacts when
the context window fills up. Before compacting, important facts are
flushed to MEMORY.md so nothing is lost.

Write-then-compact: always persist before discarding.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from src.shared.utils import setup_logging

if TYPE_CHECKING:
    from src.agent.llm import LLMClient
    from src.agent.workspace import WorkspaceManager

logger = setup_logging("agent.context")

_COMPACT_THRESHOLD = 0.70
_PRUNE_THRESHOLD = 0.90


def estimate_tokens(messages: list[dict]) -> int:
    """Rough token estimate: ~4 chars per token."""
    return sum(len(json.dumps(m)) for m in messages) // 4


class ContextManager:
    """Monitors and manages the LLM context window.

    Modes:
      safeguard (default) — auto-compact at 70%, emergency prune at 90%
    """

    def __init__(
        self,
        max_tokens: int = 128_000,
        llm: LLMClient | None = None,
        workspace: WorkspaceManager | None = None,
    ):
        self.max_tokens = max_tokens
        self.llm = llm
        self.workspace = workspace

    def usage(self, messages: list[dict]) -> float:
        """Return context usage as a fraction (0.0 to 1.0)."""
        return estimate_tokens(messages) / self.max_tokens

    def should_compact(self, messages: list[dict]) -> bool:
        return self.usage(messages) >= _COMPACT_THRESHOLD

    def _should_prune(self, messages: list[dict]) -> bool:
        return self.usage(messages) >= _PRUNE_THRESHOLD

    async def maybe_compact(
        self, system_prompt: str, messages: list[dict],
    ) -> list[dict]:
        """Check context usage and compact if needed.

        1. If below threshold, return messages unchanged.
        2. If above 70%, flush important facts to MEMORY.md, then summarize.
        3. If above 90%, hard-prune oldest messages as emergency fallback.
        """
        if not self.should_compact(messages):
            return messages

        usage_pct = int(self.usage(messages) * 100)
        logger.info(f"Context at {usage_pct}% — compacting")

        # Step 1: Extract important facts and flush to MEMORY.md
        if self.workspace and self.llm:
            await self._flush_to_memory(system_prompt, messages)

        # Step 2: Summarize and compress
        if self.llm:
            return await self._summarize_compact(system_prompt, messages)

        # Fallback: hard prune if no LLM available
        return self._hard_prune(messages)

    async def _flush_to_memory(
        self, system_prompt: str, messages: list[dict],
    ) -> None:
        """Ask the LLM to extract important facts, write them to MEMORY.md."""
        conversation_text = self._messages_to_text(messages)
        if len(conversation_text) < 100:
            return

        extract_prompt = (
            "Extract the most important facts, decisions, and user preferences "
            "from this conversation. Output a concise bullet list in Markdown. "
            "Only include information worth remembering long-term. "
            "If there's nothing important, respond with 'NONE'.\n\n"
            f"Conversation:\n{conversation_text[:20000]}"
        )

        try:
            response = await self.llm.chat(
                system="You extract key facts from conversations. Be concise.",
                messages=[{"role": "user", "content": extract_prompt}],
                max_tokens=1024,
                temperature=0.3,
            )
            facts = response.content.strip()
            if facts and facts.upper() != "NONE":
                self.workspace.append_memory(f"\n## Extracted ({_now_str()})\n\n{facts}")
                self.workspace.append_daily_log(f"Context compacted — {len(facts)} chars flushed to MEMORY.md")
                logger.info(f"Flushed {len(facts)} chars to MEMORY.md")
        except Exception as e:
            logger.warning(f"Failed to flush facts to MEMORY.md: {e}")

    async def _summarize_compact(
        self, system_prompt: str, messages: list[dict],
    ) -> list[dict]:
        """Summarize the conversation and replace history with summary + recent."""
        conversation_text = self._messages_to_text(messages[:-4] if len(messages) > 4 else messages)

        summary_prompt = (
            "Summarize this conversation concisely, preserving key context "
            "the assistant needs to continue helpfully. Include: what was discussed, "
            "what actions were taken, what's pending, and any user preferences revealed.\n\n"
            f"Conversation:\n{conversation_text[:20000]}"
        )

        try:
            response = await self.llm.chat(
                system="You produce concise conversation summaries.",
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=1024,
                temperature=0.3,
            )
            summary = response.content.strip()
        except Exception as e:
            logger.warning(f"Summarization failed, falling back to hard prune: {e}")
            return self._hard_prune(messages)

        summary_msg = {
            "role": "user",
            "content": f"## Conversation Summary (auto-compacted)\n\n{summary}",
        }

        recent = messages[-4:] if len(messages) > 4 else messages
        result = [summary_msg] + recent
        logger.info(
            f"Compacted {len(messages)} messages -> {len(result)} "
            f"(usage: {int(self.usage(result) * 100)}%)"
        )
        return result

    def _hard_prune(self, messages: list[dict]) -> list[dict]:
        """Emergency: keep only the first and last few messages."""
        if len(messages) <= 6:
            return messages
        pruned = messages[:1] + messages[-4:]
        logger.warning(f"Hard-pruned {len(messages)} -> {len(pruned)} messages")
        return pruned

    @staticmethod
    def _messages_to_text(messages: list[dict]) -> str:
        """Convert messages to readable text for summarization."""
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role == "tool":
                content = content[:200]
            if content:
                parts.append(f"[{role}]: {content}")
        return "\n\n".join(parts)


def _now_str() -> str:
    from datetime import UTC, datetime
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
