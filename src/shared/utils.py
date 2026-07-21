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


def atomic_write_text(path: "os.PathLike[str] | str", content: str) -> None:
    """Write ``content`` to ``path`` atomically (tempfile + ``os.replace``).

    A concurrent reader sees the old or the new complete file, never a
    torn one. The temp file is unlinked on any failure. This is the
    canonical home for the pattern — ``_save_permissions``
    (src/cli/config.py), ``_save_budgets`` (src/host/costs.py),
    ``CronScheduler._save`` and the dashboard ``_save_settings`` carry
    older inline copies; migrate them here when next touched.
    """
    import tempfile
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass  # already closed by fdopen
        Path(tmp_path).unlink(missing_ok=True)
        raise
    try:
        os.replace(tmp_path, target)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


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
    """Replace, append, or remove one level-2 ``## {section}`` block.

    Fence-aware LINE parser: lines inside ``` / ~~~ fenced code blocks are
    NEVER treated as headings, so a ``## Goal`` example inside a fenced block
    (or a multi-line body) can't be mistaken for the real section boundary.
    The prior regex treated ANY ``## ``-prefixed line as a boundary and could
    delete a section's content — and everything after it — to EOF.

    The heading is matched exactly (case-insensitively on the name, trailing
    whitespace tolerated); the FIRST matching section is replaced, spanning to
    the next real (non-fenced) ``## `` heading or EOF. When the section is
    absent it is appended. An empty ``content`` REMOVES the section, so
    clearing a reserved section (Goal/Context) leaves no placeholder.
    Content is inserted literally (no regex substitution), so backslashes in
    the body are preserved verbatim.
    """
    def _is_h2(line: str) -> bool:
        return line.startswith("## ") and not line.startswith("### ")

    lines = text.split("\n")
    target = section.strip().lower()
    fence: str | None = None
    start: int | None = None
    end = len(lines)
    for i, line in enumerate(lines):
        marker = line.lstrip()
        if fence is None:
            if marker.startswith("```") or marker.startswith("~~~"):
                fence = marker[:3]
                continue
        else:
            if marker.startswith(fence):
                fence = None
            continue  # inside a fence — never a heading boundary
        if start is None:
            if _is_h2(line) and line[3:].strip().lower() == target:
                start = i
        elif _is_h2(line):
            end = i
            break

    new_body = content.strip()
    if start is not None:
        before = lines[:start]
        after = lines[end:]
        if not new_body:
            merged = "\n".join(before + after)
            return merged.rstrip("\n") + "\n" if merged.strip() else ""
        rebuilt = before + [f"## {section}", "", new_body, ""] + after
        return "\n".join(rebuilt).rstrip("\n") + "\n"

    if not new_body:
        return text  # nothing to append or remove — unchanged
    base = text.rstrip("\n")
    block = f"## {section}\n\n{new_body}\n"
    return f"{base}\n\n{block}" if base else block


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


def usable_agent_reply(text: Any) -> bool:
    """True when a lane-dispatch return value is genuine agent-authored text.

    ``_direct_dispatch`` (src/cli/runtime.py) has three non-success return
    shapes that host-side consumers must NEVER treat as an agent's words: the
    silent sentinel (:data:`SILENT_REPLY_TOKEN`), the literal ``"(no
    response)"`` unreachable-agent success marker, and a ``"dispatch_error:
    <redacted reason>"`` note written by its except-branch. This is the single
    gate every consumer shares (offboard handover, onboarding intro post,
    standup channel post) so a newly-added non-success shape is rejected at
    every site at once instead of leaking into a committed doc or a channel.

    Rejects: non-str, empty/whitespace-only, the silent token, ``"(no
    response)"``, and any string starting with ``"dispatch_error:"``.
    """
    from src.shared.types import SILENT_REPLY_TOKEN

    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if not stripped:
        return False
    if stripped == SILENT_REPLY_TOKEN:
        return False
    if stripped == "(no response)":
        return False
    if stripped.startswith("dispatch_error:"):
        return False
    return True


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


def _log_trace_context() -> dict:
    """Return active correlation IDs (trace/task/agent) for a log line.

    Lazy-imports ``src.shared.trace`` to avoid the import cycle (trace.py
    imports ``setup_logging`` from this module). Repeated imports are a cheap
    ``sys.modules`` lookup. Never raises — a logging path must never break.
    ``current_agent_id`` falls back to the ``AGENT_ID`` env var so agent
    containers attribute every line even when a per-request context never
    inherited the boot-time ``set()``; the mesh host has no ``AGENT_ID`` so it
    stays absent there.
    """
    try:
        from src.shared.trace import current_agent_id, current_task_id, current_trace_id

        ctx: dict = {}
        tid = current_trace_id.get()
        if tid:
            ctx["trace_id"] = tid
        task = current_task_id.get()
        if task:
            ctx["task_id"] = task
        aid = current_agent_id.get() or os.environ.get("AGENT_ID")
        if aid:
            ctx["agent_id"] = aid
        return ctx
    except Exception:
        return {}


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
        # Correlation IDs first so an explicit ``extra_data`` key still wins.
        log_entry.update(_log_trace_context())
        if hasattr(record, "extra_data"):
            log_entry.update(record.extra_data)
        return json.dumps(log_entry)


class TextFormatter(logging.Formatter):
    """Human-readable log formatter for development."""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.now(UTC).strftime("%H:%M:%S")
        msg = record.getMessage()
        line = f"{ts} [{record.levelname:<5}] {record.name}: {msg}"
        ctx = _log_trace_context()
        if ctx:
            line += " [" + " ".join(f"{k}={v}" for k, v in ctx.items()) + "]"
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
