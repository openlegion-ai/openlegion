"""Attachment enrichment for agent chat messages.

When a user message contains the 📎 annotation emitted by the dashboard
file-attachment UI, this module converts it into multimodal LLM content:

  - Images (jpg/jpeg/png/gif/webp) → base64 ``image_url`` content block
    (OpenAI vision format; LiteLLM transparently converts to Anthropic /
    Gemini format.  The OAuth Anthropic fast-path in credentials.py
    handles the conversion manually via ``convert_openai_image_blocks``.)
  - PDFs → extracted text injected inline as a ``[Contents of …]`` block
  - Other → left as plain-text reference; agent falls back to file_tool

Enrichment is best-effort: any file that cannot be read or decoded leaves
the plain-text annotation in place so the agent can still attempt to access
the file through its regular tool suite.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Annotation pattern emitted by the dashboard when a user attaches a file.
# Format: "📎 File attached: <name> (available at /data/uploads/<name>)"
_ATTACHMENT_RE = re.compile(
    r"📎 File attached: (?P<name>[^\n(]+?) \(available at (?P<path>/data/uploads/[^\)]+)\)"
)

_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp"})

# Hard caps to avoid blowing up the context window or hitting API limits.
_MAX_IMAGE_BYTES = 10 * 1024 * 1024   # 10 MB — most vision APIs reject larger
_MAX_PDF_CHARS = 60_000               # ~40 pages of dense text
_MAX_PDF_PAGES = 100


def _mime_for_image(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "image/jpeg"


def _read_image_block(path: Path) -> dict | None:
    """Read an image file and return an OpenAI ``image_url`` content block.

    Returns ``None`` if the file cannot be read or exceeds the size cap.
    """
    try:
        data = path.read_bytes()
    except OSError as e:
        logger.warning("Cannot read image %s: %s", path.name, e)
        return None

    if len(data) > _MAX_IMAGE_BYTES:
        logger.warning(
            "Image too large for vision injection (%d bytes > %d): %s",
            len(data), _MAX_IMAGE_BYTES, path.name,
        )
        return None

    b64 = base64.standard_b64encode(data).decode()
    mime = _mime_for_image(path)
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def _extract_pdf_text(path: Path) -> str | None:
    """Extract plain text from a PDF using pypdf.

    Returns ``None`` when pypdf is unavailable, the file is unreadable, or
    the PDF contains no extractable text (e.g. scanned images only).
    """
    try:
        from pypdf import PdfReader  # lazy import — pypdf not installed in all envs
    except ImportError:
        logger.warning("pypdf not installed; PDF text extraction unavailable")
        return None

    try:
        reader = PdfReader(str(path))
        parts: list[str] = []
        total_chars = 0
        for page in reader.pages[:_MAX_PDF_PAGES]:
            text = page.extract_text() or ""
            parts.append(text)
            total_chars += len(text)
            if total_chars >= _MAX_PDF_CHARS:
                break
        full = "\n\n".join(p for p in parts if p.strip())
        if not full.strip():
            return None
        if len(full) > _MAX_PDF_CHARS:
            full = full[:_MAX_PDF_CHARS] + "\n\n[...PDF truncated — exceeds limit]"
        return full
    except Exception as e:  # noqa: BLE001 — pypdf raises many different types
        logger.warning("PDF extraction failed for %s: %s", path.name, e)
        return None


def enrich_message_with_attachments(message: str) -> str | list[dict]:
    """Upgrade a user message containing 📎 annotations to multimodal content.

    Scans *message* for attachment annotations.  For each one:
      - Image files are base64-encoded into an OpenAI ``image_url`` block.
      - PDFs are text-extracted and injected as a plain-text block.
      - All other types are left as-is (the agent reads them via file_tool).

    Returns:
      - The original *message* string unchanged when no enrichment succeeds
        (preserves existing behaviour; agent falls back to file_tool).
      - A list of OpenAI-compatible content blocks when at least one
        attachment was successfully enriched.
    """
    matches = list(_ATTACHMENT_RE.finditer(message))
    if not matches:
        return message

    blocks: list[dict] = []
    enriched_count = 0
    prev_end = 0

    for m in matches:
        # Flush text that precedes this annotation
        text_before = message[prev_end:m.start()].strip()
        if text_before:
            blocks.append({"type": "text", "text": text_before})

        name = m.group("name").strip()
        fpath = Path(m.group("path"))
        ext = fpath.suffix.lower()

        if ext in _IMAGE_EXTS:
            block = _read_image_block(fpath)
            if block:
                blocks.append(block)
                enriched_count += 1
            else:
                # Fall back to plain-text reference
                blocks.append({"type": "text", "text": m.group(0)})

        elif ext == ".pdf":
            text = _extract_pdf_text(fpath)
            if text:
                blocks.append({"type": "text", "text": f"[Contents of {name}]\n\n{text}"})
                enriched_count += 1
            else:
                blocks.append({"type": "text", "text": m.group(0)})

        else:
            # Unsupported type — keep plain reference; agent reads via file_tool
            blocks.append({"type": "text", "text": m.group(0)})

        prev_end = m.end()

    # Flush any trailing text after the last annotation
    text_after = message[prev_end:].strip()
    if text_after:
        blocks.append({"type": "text", "text": text_after})

    if not enriched_count:
        # Nothing was actually enriched — return the original string so the
        # message stays as a simple string and the agent uses file_tool.
        return message

    return blocks


def convert_openai_image_blocks(content: str | list) -> str | list:
    """Convert OpenAI ``image_url`` blocks to Anthropic Messages API format.

    Used by the OAuth Anthropic fast-path in credentials.py, which bypasses
    LiteLLM and therefore cannot rely on LiteLLM's automatic conversion.

    OpenAI:   {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,…"}}
    Anthropic: {"type": "image", "source": {"type": "base64",
                "media_type": "image/jpeg", "data": "…"}}
    """
    if not isinstance(content, list):
        return content

    result: list[dict] = []
    for block in content:
        if not isinstance(block, dict):
            result.append(block)
            continue

        if block.get("type") == "image_url":
            url = (block.get("image_url") or {}).get("url", "")
            if url.startswith("data:"):
                # data:<media_type>;base64,<data>
                try:
                    meta, data = url.split(",", 1)
                    media_type = meta.split(":")[1].split(";")[0]
                    result.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": data,
                        },
                    })
                except (ValueError, IndexError):
                    logger.warning("Malformed data-URI image_url — skipping block")
            elif url:
                # Plain URL — Anthropic supports url-type sources
                result.append({
                    "type": "image",
                    "source": {"type": "url", "url": url},
                })
            # else: empty URL — drop block silently
        else:
            result.append(block)

    return result
