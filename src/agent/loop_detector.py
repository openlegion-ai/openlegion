"""Tool loop detection with escalating responses.

Tracks a sliding window of recent (tool_name, params_hash, result_hash) tuples
and detects when an agent is stuck repeating identical failing tool calls.

Three escalation levels:
  - warn:      >= 2 prior identical calls (about to be 3rd)
  - block:     >= 4 prior identical calls (about to be 5th)
  - terminate: >= 9 prior calls with same tool+params regardless of result
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from typing import Any

from src.shared.utils import setup_logging

logger = setup_logging("agent.loop_detector")

# Tools that should never be flagged (idempotent retrieval)
_EXEMPT_TOOLS = frozenset({"memory_search", "memory_recall"})

# Escalation thresholds (checked against prior completed calls in window)
_WARN_THRESHOLD = 2       # >= 2 prior identical → warn
_BLOCK_THRESHOLD = 4      # >= 4 prior identical → block
_TERMINATE_THRESHOLD = 9  # >= 9 prior same tool+params (any result) → terminate


def _hash_json(data: Any) -> str:
    """SHA-256 of canonically-serialised JSON, truncated to 16 hex chars."""
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class ToolLoopDetector:
    """Sliding-window detector for stuck tool-call loops."""

    def __init__(self, window_size: int = 15):
        # Each entry: (tool_name, params_hash, result_hash)
        self._window: deque[tuple[str, str, str]] = deque(maxlen=window_size)

    def would_terminate(self, tool_name: str, arguments: dict) -> bool:
        """Check only the terminate condition (no logging for lower levels).

        Used by the pre-scan to avoid duplicate log lines from check_before.
        """
        if tool_name in _EXEMPT_TOOLS:
            return False
        params_hash = _hash_json(arguments)
        return self._count_any(tool_name, params_hash) >= _TERMINATE_THRESHOLD

    def check_before(self, tool_name: str, arguments: dict) -> str:
        """Pre-check before executing a tool call.

        Returns one of: "ok", "warn", "block", "terminate".
        """
        if tool_name in _EXEMPT_TOOLS:
            return "ok"

        params_hash = _hash_json(arguments)

        # Count ALL prior calls with same (tool, params) regardless of result
        any_count = self._count_any(tool_name, params_hash)
        if any_count >= _TERMINATE_THRESHOLD:
            logger.warning(
                "Tool loop TERMINATE: %s called %d times with same params",
                tool_name, any_count,
            )
            return "terminate"

        # Count prior calls with same (tool, params, most-frequent-result)
        identical_count = self._count_identical(tool_name, params_hash)
        if identical_count >= _BLOCK_THRESHOLD:
            logger.warning(
                "Tool loop BLOCK: %s called %d identical times",
                tool_name, identical_count,
            )
            return "block"

        if identical_count >= _WARN_THRESHOLD:
            logger.info(
                "Tool loop WARN: %s called %d identical times",
                tool_name, identical_count,
            )
            return "warn"

        return "ok"

    def record(self, tool_name: str, arguments: dict, result_str: str) -> None:
        """Record a completed tool call in the sliding window."""
        params_hash = _hash_json(arguments)
        result_hash = _hash_json(result_str)
        self._window.append((tool_name, params_hash, result_hash))

    def reset(self) -> None:
        """Clear the sliding window."""
        self._window.clear()

    def _count_any(self, tool_name: str, params_hash: str) -> int:
        """Count ALL window entries matching (tool, params) regardless of result."""
        return sum(
            1 for t, p, _ in self._window
            if t == tool_name and p == params_hash
        )

    def _count_identical(self, tool_name: str, params_hash: str) -> int:
        """Count window entries matching (tool, params, most-frequent-result).

        Finds the most common result_hash for this (tool, params) pair and
        returns how many entries share that exact triple.
        """
        result_hashes = [
            r for t, p, r in self._window
            if t == tool_name and p == params_hash
        ]
        if not result_hashes:
            return 0
        # Most frequent result hash
        most_common = max(set(result_hashes), key=result_hashes.count)
        return result_hashes.count(most_common)
