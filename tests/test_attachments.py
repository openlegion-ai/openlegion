"""Tests for src/agent/attachments.py — attachment enrichment module."""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agent.attachments import (
    _MAX_IMAGE_BYTES,
    _extract_pdf_text,
    _read_image_block,
    convert_openai_image_blocks,
    enrich_message_with_attachments,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tiny_png() -> bytes:
    """Return the smallest valid 1×1 transparent PNG (67 bytes)."""
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )


# ---------------------------------------------------------------------------
# _read_image_block
# ---------------------------------------------------------------------------

def test_read_image_block_success(tmp_path: Path) -> None:
    img = tmp_path / "photo.jpg"
    img.write_bytes(_make_tiny_png())
    block = _read_image_block(img)
    assert block is not None
    assert block["type"] == "image_url"
    url = block["image_url"]["url"]
    assert url.startswith("data:image/")
    assert ";base64," in url


def test_read_image_block_missing_file(tmp_path: Path) -> None:
    block = _read_image_block(tmp_path / "nonexistent.jpg")
    assert block is None


def test_read_image_block_too_large(tmp_path: Path) -> None:
    img = tmp_path / "big.jpg"
    img.write_bytes(b"x" * (_MAX_IMAGE_BYTES + 1))
    block = _read_image_block(img)
    assert block is None


# ---------------------------------------------------------------------------
# _extract_pdf_text
# ---------------------------------------------------------------------------

def test_extract_pdf_text_no_pypdf(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF fake")
    with patch.dict("sys.modules", {"pypdf": None}):
        result = _extract_pdf_text(pdf)
    assert result is None


def test_extract_pdf_text_success(tmp_path: Path) -> None:
    import sys

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF fake")
    fake_page = MagicMock()
    fake_page.extract_text.return_value = "Hello world"
    fake_reader = MagicMock()
    fake_reader.pages = [fake_page]
    fake_pypdf = MagicMock()
    fake_pypdf.PdfReader.return_value = fake_reader
    sys.modules["pypdf"] = fake_pypdf
    try:
        result = _extract_pdf_text(pdf)
    finally:
        del sys.modules["pypdf"]

    assert result == "Hello world"


def test_extract_pdf_text_exception(tmp_path: Path) -> None:
    pdf = tmp_path / "bad.pdf"
    pdf.write_bytes(b"not a pdf")
    import sys
    fake_pypdf = MagicMock()
    fake_pypdf.PdfReader.side_effect = ValueError("bad pdf")
    sys.modules["pypdf"] = fake_pypdf
    try:
        result = _extract_pdf_text(pdf)
    finally:
        del sys.modules["pypdf"]
    assert result is None


def test_extract_pdf_text_empty(tmp_path: Path) -> None:
    pdf = tmp_path / "empty.pdf"
    pdf.write_bytes(b"%PDF empty")
    import sys
    fake_page = MagicMock()
    fake_page.extract_text.return_value = "   "
    fake_reader = MagicMock()
    fake_reader.pages = [fake_page]
    fake_pypdf = MagicMock()
    fake_pypdf.PdfReader.return_value = fake_reader
    sys.modules["pypdf"] = fake_pypdf
    try:
        result = _extract_pdf_text(pdf)
    finally:
        del sys.modules["pypdf"]
    assert result is None


# ---------------------------------------------------------------------------
# enrich_message_with_attachments
# ---------------------------------------------------------------------------

def test_enrich_no_attachment() -> None:
    msg = "Hello, world!"
    assert enrich_message_with_attachments(msg) is msg


def test_enrich_plain_text_annotation_no_file(tmp_path: Path) -> None:
    """Non-existent file → plain-text fallback, returns original string."""
    msg = "Look at this: 📎 File attached: missing.jpg (available at /data/uploads/missing.jpg)"
    result = enrich_message_with_attachments(msg)
    # No enrichment possible; original string returned
    assert result == msg


def test_enrich_image_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    img_path = tmp_path / "photo.jpg"
    img_path.write_bytes(_make_tiny_png())

    monkeypatch.setattr(
        "src.agent.attachments.Path",
        lambda p: tmp_path / Path(p).name if "/data/uploads/" in p else Path(p),
    )

    msg = "What is this? 📎 File attached: photo.jpg (available at /data/uploads/photo.jpg)"
    result = enrich_message_with_attachments(msg)
    assert isinstance(result, list)
    types = [b["type"] for b in result]
    assert "image_url" in types
    text_blocks = [b for b in result if b["type"] == "text"]
    assert any("What is this?" in b["text"] for b in text_blocks)


def test_enrich_pdf_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    pdf_path = tmp_path / "report.pdf"
    pdf_path.write_bytes(b"%PDF fake")

    fake_page = MagicMock()
    fake_page.extract_text.return_value = "Report content here"
    fake_reader = MagicMock()
    fake_reader.pages = [fake_page]
    fake_pypdf = MagicMock()
    fake_pypdf.PdfReader.return_value = fake_reader
    sys.modules["pypdf"] = fake_pypdf

    monkeypatch.setattr(
        "src.agent.attachments.Path",
        lambda p: tmp_path / Path(p).name if "/data/uploads/" in p else Path(p),
    )

    msg = "Summarize: 📎 File attached: report.pdf (available at /data/uploads/report.pdf)"
    try:
        result = enrich_message_with_attachments(msg)
    finally:
        del sys.modules["pypdf"]

    assert isinstance(result, list)
    all_text = " ".join(b.get("text", "") for b in result if b["type"] == "text")
    assert "Report content here" in all_text
    assert "report.pdf" in all_text


def test_enrich_unsupported_type_returns_string(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unsupported file type (.csv) — no enrichment, returns original string."""
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b,c")

    monkeypatch.setattr(
        "src.agent.attachments.Path",
        lambda p: tmp_path / Path(p).name if "/data/uploads/" in p else Path(p),
    )

    msg = "Analyze: 📎 File attached: data.csv (available at /data/uploads/data.csv)"
    result = enrich_message_with_attachments(msg)
    # No enrichment — returns original string
    assert result == msg


def test_enrich_text_before_and_after_annotation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    img_path = tmp_path / "chart.png"
    img_path.write_bytes(_make_tiny_png())

    monkeypatch.setattr(
        "src.agent.attachments.Path",
        lambda p: tmp_path / Path(p).name if "/data/uploads/" in p else Path(p),
    )

    msg = (
        "Please describe:\n\n"
        "📎 File attached: chart.png (available at /data/uploads/chart.png)\n\n"
        "Thank you."
    )
    result = enrich_message_with_attachments(msg)
    assert isinstance(result, list)
    texts = [b["text"] for b in result if b["type"] == "text"]
    assert any("Please describe" in t for t in texts)
    assert any("Thank you" in t for t in texts)


def test_enrich_multiple_attachments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    img1 = tmp_path / "a.jpg"
    img2 = tmp_path / "b.png"
    img1.write_bytes(_make_tiny_png())
    img2.write_bytes(_make_tiny_png())

    monkeypatch.setattr(
        "src.agent.attachments.Path",
        lambda p: tmp_path / Path(p).name if "/data/uploads/" in p else Path(p),
    )

    msg = (
        "Compare:\n"
        "📎 File attached: a.jpg (available at /data/uploads/a.jpg)\n"
        "📎 File attached: b.png (available at /data/uploads/b.png)"
    )
    result = enrich_message_with_attachments(msg)
    assert isinstance(result, list)
    image_blocks = [b for b in result if b["type"] == "image_url"]
    assert len(image_blocks) == 2


# ---------------------------------------------------------------------------
# convert_openai_image_blocks
# ---------------------------------------------------------------------------

def test_convert_string_passthrough() -> None:
    assert convert_openai_image_blocks("hello") == "hello"


def test_convert_data_uri() -> None:
    content = [
        {"type": "text", "text": "Look"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
    ]
    result = convert_openai_image_blocks(content)
    assert isinstance(result, list)
    assert result[0] == {"type": "text", "text": "Look"}
    img = result[1]
    assert img["type"] == "image"
    assert img["source"]["type"] == "base64"
    assert img["source"]["media_type"] == "image/png"
    assert img["source"]["data"] == "abc123"


def test_convert_url_image() -> None:
    content = [{"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}}]
    result = convert_openai_image_blocks(content)
    assert result[0]["type"] == "image"
    assert result[0]["source"]["type"] == "url"
    assert result[0]["source"]["url"] == "https://example.com/img.jpg"


def test_convert_malformed_data_uri_skipped() -> None:
    content = [{"type": "image_url", "image_url": {"url": "data:bad"}}]
    result = convert_openai_image_blocks(content)
    # Malformed URI — block is dropped
    assert result == []


def test_convert_empty_url_dropped() -> None:
    content = [{"type": "image_url", "image_url": {"url": ""}}]
    result = convert_openai_image_blocks(content)
    assert result == []


def test_convert_non_image_blocks_unchanged() -> None:
    content = [
        {"type": "text", "text": "hello"},
        {"type": "tool_use", "id": "x", "name": "foo", "input": {}},
    ]
    result = convert_openai_image_blocks(content)
    assert result == content


def test_convert_empty_list() -> None:
    assert convert_openai_image_blocks([]) == []
