"""Shared utilities: ID generation, structured logging, timing, text helpers."""

from __future__ import annotations

import json
import logging
import os
import unicodedata
import uuid
from datetime import UTC, datetime


def generate_id(prefix: str = "id") -> str:
    """Generate a unique ID with a descriptive prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def truncate(text: str, max_len: int = 200) -> str:
    """Truncate text with ellipsis indicator."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def format_dict(d: dict) -> str:
    """Format a dict for inclusion in LLM prompts."""
    return json.dumps(d, indent=2, default=str)


# ── Prompt injection sanitization ────────────────────────────

_STRIP_CATEGORIES = frozenset({"Cc", "Cf", "Co", "Cs", "Cn"})
_SAFE_CC = frozenset({0x09, 0x0A, 0x0D})  # TAB, LF, CR
_SAFE_CF = frozenset({0x200C, 0x200D, 0xFE0E, 0xFE0F})  # ZWNJ, ZWJ, VS15, VS16
_STRIP_EXTRA = frozenset({
    *range(0xFE00, 0xFE0E),       # VS1-14
    *range(0xE0100, 0xE01F0),     # VS17-256
    0x034F,                        # Combining Grapheme Joiner
    0x115F, 0x1160, 0x3164, 0xFFA0,  # Hangul fillers
    0xFFFC,                        # Object Replacement Character
})


def sanitize_for_prompt(text: str) -> str:
    """Strip invisible Unicode characters that enable prompt injection."""
    if not isinstance(text, str) or not text:
        return ""
    out = []
    for ch in text:
        cp = ord(ch)
        if cp == 0x2028 or cp == 0x2029:
            out.append("\n")
            continue
        if cp in _STRIP_EXTRA:
            continue
        cat = unicodedata.category(ch)
        if cat in _STRIP_CATEGORIES:
            if cat == "Cc" and cp in _SAFE_CC:
                out.append(ch)
            elif cat == "Cf" and cp in _SAFE_CF:
                out.append(ch)
            continue
        out.append(ch)
    return "".join(out)


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


class TextFormatter(logging.Formatter):
    """Human-readable log formatter for development."""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.now(UTC).strftime("%H:%M:%S")
        msg = record.getMessage()
        return f"{ts} [{record.levelname:<5}] {record.name}: {msg}"


def setup_logging(name: str, level: str = "INFO") -> logging.Logger:
    """Configure logging for a named logger.

    Format controlled by OPENLEGION_LOG_FORMAT env var:
      - "json" (default): structured JSON lines
      - "text": human-readable single-line format
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        if logger.level == logging.NOTSET:
            logger.setLevel(getattr(logging, level.upper()))
        handler = logging.StreamHandler()
        log_format = os.environ.get("OPENLEGION_LOG_FORMAT", "json").lower()
        if log_format == "text":
            handler.setFormatter(TextFormatter())
        else:
            handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)
    return logger
