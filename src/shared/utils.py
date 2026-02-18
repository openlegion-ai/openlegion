"""Shared utilities: ID generation, structured logging, timing, text helpers."""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import UTC, datetime


def generate_id(prefix: str = "id") -> str:
    """Generate a unique ID with a descriptive prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def timestamp_iso() -> str:
    """Current UTC timestamp in ISO format."""
    return datetime.now(UTC).isoformat()


def truncate(text: str, max_len: int = 200) -> str:
    """Truncate text with ellipsis indicator."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def format_dict(d: dict) -> str:
    """Format a dict for inclusion in LLM prompts."""
    return json.dumps(d, indent=2, default=str)


class StructuredFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_data"):
            log_entry.update(record.extra_data)
        return json.dumps(log_entry)


def setup_logging(name: str, level: str = "INFO") -> logging.Logger:
    """Configure structured JSON logging for a named logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(getattr(logging, level.upper()))
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)
    return logger


class Timer:
    """Context manager for timing operations in milliseconds."""

    def __init__(self) -> None:
        self.start: float = 0
        self.elapsed_ms: int = 0

    def __enter__(self) -> Timer:
        self.start = time.time()
        return self

    def __exit__(self, *args: object) -> None:
        self.elapsed_ms = int((time.time() - self.start) * 1000)
