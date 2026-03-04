"""Credential redaction for browser responses.

Pattern-based and exact-value redaction prevents secrets from leaking
into LLM context through browser tool responses.
"""

from __future__ import annotations

import re
from collections import defaultdict

_REDACT_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),                # OpenAI / Anthropic
    re.compile(r"sk-ant-api[A-Za-z0-9\-]{20,}"),       # Anthropic
    re.compile(r"gho_[A-Za-z0-9]{36,}"),                # GitHub OAuth tokens
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),        # GitHub fine-grained PATs
    re.compile(r"xoxb-[A-Za-z0-9\-]{20,}"),             # Slack bot tokens
    re.compile(r"xoxp-[A-Za-z0-9\-]{20,}"),             # Slack user tokens
    re.compile(r"AKIA[A-Z0-9]{16}"),                     # AWS access key IDs
    re.compile(r"(?<![A-Za-z0-9])[A-Fa-f0-9]{40,}(?![A-Za-z0-9])"),
    re.compile(r"(?<![A-Za-z0-9/+=])[A-Za-z0-9+/]{40,}={0,2}(?![A-Za-z0-9/+=])"),
]


class CredentialRedactor:
    """Per-agent credential redaction state."""

    def __init__(self):
        # agent_id -> set of resolved secret values
        self._resolved_values: dict[str, set[str]] = defaultdict(set)

    def track_resolved_value(self, agent_id: str, value: str) -> None:
        """Track a resolved credential value for exact-match redaction."""
        if value and len(value) >= 4:
            self._resolved_values[agent_id].add(value)

    def clear_agent(self, agent_id: str) -> None:
        """Clear tracked values for an agent (e.g. on browser reset)."""
        self._resolved_values.pop(agent_id, None)

    def redact(self, agent_id: str, text: str) -> str:
        """Apply all redaction layers to text."""
        if not text:
            return text
        # Pattern-based
        for pattern in _REDACT_PATTERNS:
            text = pattern.sub("[REDACTED]", text)
        # Exact-value
        for value in self._resolved_values.get(agent_id, set()):
            text = text.replace(value, "[REDACTED]")
        return text

    def deep_redact(self, agent_id: str, obj):
        """Recursively redact credential values from any JSON-serializable structure."""
        if isinstance(obj, str):
            return self.redact(agent_id, obj)
        if isinstance(obj, dict):
            return {k: self.deep_redact(agent_id, v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.deep_redact(agent_id, item) for item in obj]
        return obj
