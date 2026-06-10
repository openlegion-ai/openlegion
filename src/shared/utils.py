"""Shared utilities: ID generation, structured logging, timing, text helpers."""

from __future__ import annotations

import json
import logging
import os
import unicodedata
import uuid
from datetime import datetime, timezone
from typing import Any

UTC = timezone.utc


def set_llm_max_tokens_env(env: dict[str, str], agent_cfg: dict) -> None:
    """Inject ``LLM_MAX_TOKENS`` into a per-agent container env dict from the
    agent's YAML config.

    Called by every restart-from-config path (CLI start, dashboard
    "restart agents", REPL ``/restart``, fleet-template apply) so an
    operator's ``edit_agent`` change to ``max_output_tokens`` survives a
    container restart — the agent reads this env at startup (see
    ``src/agent/__main__.py``); the live ``/config`` hot-reload only covers
    the already-running container.

    No-op when the cap is unset or not a plain int, so the LLMClient default
    (16384) then applies. ``bool`` is rejected explicitly (it is an ``int``
    subclass) so a stray ``True``/``False`` can't become ``1``/``0``. Also a
    no-op for a missing/malformed agent config (a null agents.yaml entry yields
    ``None``) rather than raising AttributeError on the restart path.
    """
    if not isinstance(agent_cfg, dict):
        return
    value = agent_cfg.get("max_output_tokens")
    if isinstance(value, int) and not isinstance(value, bool):
        env["LLM_MAX_TOKENS"] = str(value)


def dumps_safe(obj: Any, **kwargs: Any) -> str:
    """json.dumps with `default=str` — handles datetime, UUID, Decimal, Path.

    Other ``json.dumps`` kwargs (``indent``, ``sort_keys``, ``separators``,
    ``ensure_ascii``, …) pass through unchanged.

    The helper enforces ``default=str``: do NOT pass ``default=`` (raises
    ``TypeError: multiple values for keyword argument 'default'``) or
    ``cls=CustomEncoder`` (custom encoder's default() may conflict with
    the forced ``str`` callable). If you need a different encoder, call
    ``json.dumps`` directly at that site.
    """
    return json.dumps(obj, default=str, **kwargs)


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
    return dumps_safe(d, indent=2)


def replace_markdown_section(text: str, section: str, content: str) -> str:
    """Replace (or append) one level-2 ``## {section}`` block in markdown.

    The block spans from its heading to the next ``## `` heading or EOF.
    When the section is absent, it is appended at the end. Used for
    section-scoped TEAM.md brief updates so an operator-written section
    (e.g. ``## User Preferences``) never clobbers the rest of the
    document — last-writer-wins is limited to one section.
    """
    import re as _re

    block = f"## {section}\n\n{content.strip()}\n"
    pattern = _re.compile(
        rf"^## {_re.escape(section)}[ \t]*\n.*?(?=^## |\Z)",
        _re.DOTALL | _re.MULTILINE,
    )
    if pattern.search(text):
        # Function repl so backslashes in content can't act as group refs.
        updated = pattern.sub(lambda _m: block + "\n", text, count=1)
        return updated.rstrip("\n") + "\n"
    base = text.rstrip("\n")
    if base:
        return f"{base}\n\n{block}"
    return block


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


def friendly_streaming_error(exc: Exception) -> str:
    """Return a user-friendly message for LLM streaming errors.

    Replaces raw protocol-level error strings (e.g. 'incomplete chunked
    read') with an actionable message.  Non-protocol errors pass through
    unchanged.
    """
    raw = str(exc)
    if "incomplete chunked read" in raw or "RemoteProtocolError" in type(exc).__name__:
        return "Connection to the AI provider was interrupted. Retrying may help."
    return raw


class StructuredFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "extra_data"):
            log_entry.update(record.extra_data)
        return json.dumps(log_entry)


class TextFormatter(logging.Formatter):
    """Human-readable log formatter for development."""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.now(UTC).strftime("%H:%M:%S")
        msg = record.getMessage()
        line = f"{ts} [{record.levelname:<5}] {record.name}: {msg}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


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
